"""Re-embed episodic records from their verbatim text.

A defect in the capture path embedded the provenance cue label instead of the
message content, so existing episodic records carry vectors of a positional
label string ("session <id> turn <n>") rather than of their actual text. The
ANN/cosine index built from those vectors is semantically collapsed.

This migration rebuilds every episodic record's embedding from its intact
``literal_surface`` (the verbatim text, which was always stored correctly),
then rebuilds the recall index from the corrected vectors.

Constitutional boundary: only the ``embedding`` column is rewritten.
``literal_surface`` is never modified, and the at-rest encryption boundary is
untouched -- decryption happens in-process via the normal record-read path,
exactly as graph build and recall already do.

Idempotent: re-embedding the same text yields the same vector, so a second run
is a no-op in effect. Records whose text is missing or undecryptable are
skipped and counted -- no vector is ever fabricated.
"""
from __future__ import annotations

import logging
from uuid import UUID

from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore, RECORDS_TABLE, _uuid_literal

log = logging.getLogger(__name__)


DEFAULT_BATCH_SIZE = 256


def migrate_reembed_from_text(
    store: "MemoryStore",
    *,
    dry_run: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Re-embed every episodic record from its ``literal_surface`` text.

    Streams record ids in id-ordered windows so the whole corpus is never
    loaded at once, decrypts each record's text via the normal read path,
    re-embeds it, and updates only that record's ``embedding`` column. After
    all updates land, the HNSW recall index is rebuilt from the corrected
    vectors.

    Returns a dict with keys: reembedded, skipped, total, dry_run.

    Safe to call multiple times (idempotent): the same text re-embeds to the
    same vector, so re-running has no net effect. Records whose text is empty
    or undecryptable are skipped and counted, never re-embedded with a
    fabricated vector.
    """
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.hippo import HippoDB, HippoIntegrityError

    db = store.db
    if not isinstance(db, HippoDB):
        return {"reembedded": 0, "skipped": 0, "total": 0, "dry_run": dry_run}

    if batch_size < 1:
        batch_size = DEFAULT_BATCH_SIZE

    embedder = embedder_for_store(store)
    tbl = store.db.open_table(RECORDS_TABLE)

    reembedded = 0
    skipped = 0
    total = 0

    # Stream ids in id-ordered windows. A keyset cursor over the primary key
    # keeps memory bounded regardless of corpus size and is stable under the
    # in-place embedding updates this loop performs (updates never change id).
    last_id = ""
    while True:
        with db._conn_lock:
            rows = db._conn.execute(
                "SELECT id FROM records"
                " WHERE tier = 'episodic'"
                "   AND tombstoned_at IS NULL"
                "   AND id > ?"
                " ORDER BY id"
                " LIMIT ?",
                (last_id, int(batch_size)),
            ).fetchall()
        if not rows:
            break
        last_id = rows[-1][0]

        for row in rows:
            rid_str = row[0]
            total += 1
            try:
                rec = store.get(UUID(rid_str))
            except (HippoIntegrityError, ValueError, TypeError) as exc:
                log.warning(
                    "reembed_from_text: skip id=%s (read failed: %s)",
                    rid_str,
                    type(exc).__name__,
                )
                skipped += 1
                continue
            if rec is None:
                skipped += 1
                continue

            text = (rec.literal_surface or "").strip()
            if not text:
                skipped += 1
                continue

            try:
                vec = list(embedder.embed(text))
            except Exception as exc:  # noqa: BLE001 -- per-record fail-safe
                log.warning(
                    "reembed_from_text: skip id=%s (embed failed: %s)",
                    rid_str,
                    type(exc).__name__,
                )
                skipped += 1
                continue

            if not dry_run:
                tbl.update(
                    where=f"id = '{_uuid_literal(rec.id)}'",
                    values={"embedding": [float(x) for x in vec]},
                )
            reembedded += 1

    if not dry_run and reembedded > 0:
        rebuild = db._rebuild_index_from_sqlite()
        try:
            write_event(
                store,
                "migration_reembed_from_text",
                {
                    "reembedded": reembedded,
                    "skipped": skipped,
                    "total": total,
                    "rebuild": rebuild,
                },
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("migration_reembed_from_text event write failed: %s", exc)

    return {
        "reembedded": reembedded,
        "skipped": skipped,
        "total": total,
        "dry_run": dry_run,
    }
