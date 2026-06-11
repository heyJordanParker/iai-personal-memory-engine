from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from uuid import UUID

import pyarrow as pa

from iai_mcp.crypto import encrypt_field, is_encrypted
from iai_mcp.embed import Embedder
from iai_mcp.events import write_event
from iai_mcp.store import (
    EVENTS_TABLE,
    MemoryStore,
    RECORDS_TABLE,
    _uuid_literal,
)
from iai_mcp.types import (
    SCHEMA_VERSION_CURRENT,
    SCHEMA_VERSION_LEGACY,
    MemoryRecord,
)


log = logging.getLogger(__name__)


STAGING_TABLE = "records_v_new"
OLD_TABLE_PREFIX = "records_old_"
PROGRESS_FILE = "migration_progress.json"
CRYPTO_RECOVER_STAGING = "records_crypto_recover_stage"


def _db_table_names_set(db) -> set[str]:
    res = db.list_tables()
    if hasattr(res, "tables"):
        return set(res.tables)
    return set(res)


def migrate_v1_to_v2(
    store: MemoryStore,
    embedder: Optional[Embedder] = None,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    t0 = time.time()
    if embedder is not None:
        emb = embedder
    else:
        from iai_mcp.embed import embedder_for_store
        emb = embedder_for_store(store)

    all_records = store.all_records()
    v1_records = [r for r in all_records if r.schema_version == SCHEMA_VERSION_LEGACY]
    total = len(v1_records)
    migrated = 0

    for idx, record in enumerate(v1_records):
        if progress is not None:
            try:
                progress(idx, total)
            except (TypeError, ValueError):
                pass

        new_lang = record.language if (record.language and record.language.strip()) else "en"

        if dry_run:
            migrated += 1
            continue

        new_embedding = emb.embed(record.literal_surface)

        updated = MemoryRecord(
            id=record.id,
            tier=record.tier,
            literal_surface=record.literal_surface,
            aaak_index=record.aaak_index,
            embedding=new_embedding,
            structure_hv=record.structure_hv,
            community_id=record.community_id,
            centrality=record.centrality,
            detail_level=record.detail_level,
            pinned=record.pinned,
            stability=record.stability,
            difficulty=record.difficulty,
            last_reviewed=record.last_reviewed,
            never_decay=record.never_decay,
            never_merge=record.never_merge,
            provenance=record.provenance,
            created_at=record.created_at,
            updated_at=record.updated_at,
            tags=record.tags,
            language=new_lang,
            s5_trust_score=0.5,
            profile_modulation_gain={},
            schema_version=SCHEMA_VERSION_CURRENT,
        )
        tbl = store.db.open_table(RECORDS_TABLE)
        tbl.delete(f"id = '{_uuid_literal(record.id)}'")
        store.insert(updated)
        migrated += 1

    duration_sec = time.time() - t0

    if not dry_run and migrated > 0:
        write_event(
            store,
            kind="migration_v1_to_v2",
            data={
                "record_count": migrated,
                "duration_sec": duration_sec,
            },
            severity="info",
        )

    return {
        "records_migrated": migrated,
        "skipped": max(0, len(all_records) - total),
        "duration_sec": duration_sec,
        "previous_model": "bge-small-en-v1.5",
        "new_model": emb.model_key,
    }


def _records_schema_at_dim(dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("id", pa.string()),
            ("tier", pa.string()),
            ("literal_surface", pa.string()),
            ("aaak_index", pa.string()),
            ("embedding", pa.list_(pa.float32(), dim)),
            ("structure_hv", pa.binary()),
            ("community_id", pa.string()),
            ("centrality", pa.float32()),
            ("detail_level", pa.int32()),
            ("pinned", pa.bool_()),
            ("stability", pa.float32()),
            ("difficulty", pa.float32()),
            ("last_reviewed", pa.timestamp("us", tz="UTC")),
            ("never_decay", pa.bool_()),
            ("never_merge", pa.bool_()),
            ("tombstoned_at", pa.timestamp("us", tz="UTC")),
            ("schema_bypass", pa.bool_()),
            ("labile_until", pa.timestamp("us", tz="UTC")),
            ("provenance_json", pa.string()),
            ("created_at", pa.timestamp("us", tz="UTC")),
            ("updated_at", pa.timestamp("us", tz="UTC")),
            ("tags_json", pa.string()),
            ("language", pa.string()),
            ("s5_trust_score", pa.float32()),
            ("profile_modulation_gain_json", pa.string()),
            ("schema_version", pa.int32()),
            ("wing", pa.string()),
            ("room", pa.string()),
            ("drawer", pa.string()),
            ("valence", pa.float32()),
            ("hv_tier", pa.string()),
            ("structure_hv_payload", pa.binary()),
        ]
    )


def _progress_path(store: MemoryStore) -> Path:
    return Path(store.root) / PROGRESS_FILE


def _progress_read(store: MemoryStore) -> dict:
    path = _progress_path(store)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _progress_write(store: MemoryStore, state: dict) -> None:
    target = _progress_path(store)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".migration-progress.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    except (OSError, ValueError) as exc:
        log.error("progress save failed: %s", exc)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _progress_clear(store: MemoryStore) -> None:
    path = _progress_path(store)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _stage_record_to_table(
    store: MemoryStore,
    target_tbl,
    rec: MemoryRecord,
    new_embedding: list[float],
) -> None:
    if not rec.structure_hv:
        from iai_mcp.tem import bind_structure
        rec.structure_hv = bind_structure(rec)
    new_rec = MemoryRecord(
        id=rec.id,
        tier=rec.tier,
        literal_surface=rec.literal_surface,
        aaak_index=rec.aaak_index,
        embedding=new_embedding,
        structure_hv=rec.structure_hv,
        community_id=rec.community_id,
        centrality=rec.centrality,
        detail_level=rec.detail_level,
        pinned=rec.pinned,
        stability=rec.stability,
        difficulty=rec.difficulty,
        last_reviewed=rec.last_reviewed,
        never_decay=rec.never_decay,
        never_merge=rec.never_merge,
        provenance=rec.provenance,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        tags=rec.tags,
        language=rec.language,
        s5_trust_score=rec.s5_trust_score,
        profile_modulation_gain=rec.profile_modulation_gain,
        schema_version=rec.schema_version,
    )
    target_tbl.add([store._to_row(new_rec)])


def _stage_loop(
    store: MemoryStore,
    target_embedder,
    target_dim: int,
    target_tbl,
    source_iter,
    *,
    total: int,
    started_at_iso: str,
    started_idx: int = 0,
    already_staged_ids: Optional[set[str]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[int, list[str]]:
    staged_count = 0
    failures: list[str] = []
    staged_ids: list[str] = list(already_staged_ids or [])
    skipped_set: set[str] = set(staged_ids)

    idx = started_idx
    for rec in source_iter:
        rec_id_str = str(rec.id)
        if rec_id_str in skipped_set:
            continue
        if progress is not None:
            try:
                progress(idx, total)
            except (TypeError, ValueError):
                pass
        try:
            new_embedding = target_embedder.embed(rec.literal_surface)
            _stage_record_to_table(store, target_tbl, rec, new_embedding)
        except (KeyboardInterrupt, SystemExit):
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            log.warning(
                "migrate_reembed_per_row_failed",
                extra={
                    "record_id": rec_id_str,
                    "error": str(exc)[:160],
                },
            )
            failures.append(rec_id_str)
            idx += 1
            continue

        staged_count += 1
        staged_ids.append(rec_id_str)
        _progress_write(
            store,
            {
                "started_at": started_at_iso,
                "ts": int(time.time()),
                "row_index": idx,
                "last_rid": rec_id_str,
                "total": total,
                "target_dim": target_dim,
                "target_model_key": getattr(target_embedder, "model_key", "unknown"),
                "staged_ids": staged_ids,
                "failures": failures,
            },
        )
        idx += 1

    return staged_count, failures


def _lancedb_root(db) -> Path:
    return Path(db.uri)


def _swap_tables_filesystem(db, *, source: str, dest: str) -> None:
    from iai_mcp.hippo import HippoDB

    if isinstance(db, HippoDB):
        db._conn.execute(  # nosemgrep
            f"ALTER TABLE [{source}] RENAME TO [{dest}]"
        )
        return
    root = _lancedb_root(db)
    src_path = root / f"{source}.lance"
    dst_path = root / f"{dest}.lance"
    os.replace(src_path, dst_path)


def _validate_and_swap(
    store: MemoryStore,
    *,
    source_dim: int,
    target_dim: int,
    target_embedder,
    staged_count: int,
    failures: list[str],
    duration_sec: float,
) -> dict:
    orig = store.db.open_table(RECORDS_TABLE).count_rows()
    staged = store.db.open_table(STAGING_TABLE).count_rows()
    if orig > 0 and staged < orig * 0.99:
        log.error(
            "migrate_reembed_validate_failed",
            extra={
                "orig": orig,
                "staged": staged,
                "ratio": staged / max(orig, 1),
                "failures": len(failures),
            },
        )
        raise RuntimeError(
            f"reembed staging produced {staged}/{orig} rows "
            f"({staged/max(orig,1):.3%}); refusing to swap. Inspect tables "
            f"manually or run `iai-mcp migrate --rollback`."
        )

    try:
        write_event(
            store,
            kind="migration_reembed",
            data={
                "source_dim": source_dim,
                "target_dim": target_dim,
                "updated": staged_count,
                "duration_sec": duration_sec,
                "target_model_key": getattr(target_embedder, "model_key", "unknown"),
                "failures": len(failures),
            },
            severity="info",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("migration_reembed event write failed: %s", exc)

    ts = int(time.time())
    old_name = f"{OLD_TABLE_PREFIX}{ts}"
    _swap_tables_filesystem(store.db, source=RECORDS_TABLE, dest=old_name)
    _swap_tables_filesystem(store.db, source=STAGING_TABLE, dest=RECORDS_TABLE)

    store._embed_dim = target_dim

    _progress_clear(store)

    return {
        "source_dim": source_dim,
        "target_dim": target_dim,
        "updated": staged_count,
        "skipped": 0,
        "failures": len(failures),
        "duration_sec": duration_sec,
        "old_table": old_name,
    }


def migrate_reembed_to_current_dim(
    store: MemoryStore,
    target_embedder,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    t0 = time.time()

    source_dim = int(store.embed_dim)
    target_dim = int(target_embedder.DIM)
    started_at_iso = datetime.now(timezone.utc).isoformat()

    if source_dim == target_dim:
        try:
            write_event(
                store,
                kind="migration_reembed",
                data={
                    "source_dim": source_dim,
                    "target_dim": target_dim,
                    "updated": 0,
                    "no_op": True,
                    "duration_sec": time.time() - t0,
                    "target_model_key": getattr(
                        target_embedder, "model_key", "unknown"
                    ),
                },
                severity="info",
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("migration_reembed no-op event write failed: %s", exc)
        return {
            "source_dim": source_dim,
            "target_dim": target_dim,
            "updated": 0,
            "skipped": store.db.open_table(RECORDS_TABLE).count_rows(),
            "no_op": True,
            "duration_sec": time.time() - t0,
        }

    if dry_run:
        return {
            "source_dim": source_dim,
            "target_dim": target_dim,
            "would_update": store.db.open_table(RECORDS_TABLE).count_rows(),
            "duration_sec": time.time() - t0,
        }

    if STAGING_TABLE in set(store.db.table_names()):
        store.db.drop_table(STAGING_TABLE)
    target_tbl = store.db.create_table(
        STAGING_TABLE, schema=_records_schema_at_dim(target_dim)
    )

    total = store.db.open_table(RECORDS_TABLE).count_rows()
    source_iter = store.iter_records()
    staged_count, failures = _stage_loop(
        store,
        target_embedder,
        target_dim,
        target_tbl,
        source_iter,
        total=total,
        started_at_iso=started_at_iso,
        progress=progress,
    )

    duration_sec = time.time() - t0
    return _validate_and_swap(
        store,
        source_dim=source_dim,
        target_dim=target_dim,
        target_embedder=target_embedder,
        staged_count=staged_count,
        failures=failures,
        duration_sec=duration_sec,
    )


def detect_partial_migration(db) -> dict:
    names = set(db.table_names())
    has_records = RECORDS_TABLE in names
    has_staging = STAGING_TABLE in names
    old_tables = sorted(n for n in names if n.startswith(OLD_TABLE_PREFIX))

    if not has_staging and not old_tables:
        return {"state": "clean"}

    if has_staging and not has_records and not old_tables:
        return {
            "state": "partial_swap_inconsistent",
            "staging": STAGING_TABLE,
            "old_tables": old_tables,
            "reason": (
                "records_v_new present but neither records nor records_old_<ts> "
                "exist; manual recovery required."
            ),
        }

    if has_staging and has_records:
        return {
            "state": "needs_rollback",
            "old_tables": old_tables,
            "reason": (
                "records_v_new present alongside records — staging did not "
                "complete; recover by dropping records_v_new (rollback) or "
                "resuming from migration_progress.json."
            ),
        }

    if not has_staging and has_records and old_tables:
        return {
            "state": "needs_cleanup",
            "old_tables": old_tables,
            "reason": "successful swap from prior boot; drop old tables.",
        }

    if has_staging and old_tables and not has_records:
        return {
            "state": "needs_rollback",
            "old_tables": old_tables,
            "reason": (
                "records_v_new + records_old_<ts> present, records absent — "
                "swap interrupted between renames; rollback from records_old_<ts>."
            ),
        }

    return {
        "state": "unknown",
        "has_records": has_records,
        "has_staging": has_staging,
        "old_tables": old_tables,
    }


def _decrypt_field_try_keys(
    ciphertext: str,
    record_id: UUID,
    keys: list[bytes],
) -> str:
    from cryptography.exceptions import InvalidTag

    from iai_mcp.crypto import decrypt_field

    if not is_encrypted(ciphertext):
        return str(ciphertext or "")
    ad = _uuid_literal(record_id).encode("ascii")
    last_exc: Exception | None = None
    for key in keys:
        if key is None or len(key) != 32:
            continue
        try:
            return decrypt_field(ciphertext, key, associated_data=ad)
        except (InvalidTag, ValueError) as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise ValueError("no valid keys supplied for decrypt")


def _memory_record_from_raw_row_multikey(
    store: MemoryStore,
    row: dict,
    keys: list[bytes],
) -> MemoryRecord:
    import pandas as pd

    from uuid import UUID as _UUID

    row_uuid = _UUID(row["id"])
    structure_raw = row.get("structure_hv")
    if structure_raw is None:
        structure_hv = b""
    elif isinstance(structure_raw, (bytes, bytearray)):
        structure_hv = bytes(structure_raw)
    else:
        structure_hv = b""

    community_raw = row.get("community_id") or ""
    community_id = _UUID(community_raw) if community_raw else None

    raw_version = row.get("schema_version")
    try:
        version_int = int(raw_version) if raw_version is not None else SCHEMA_VERSION_CURRENT
    except (TypeError, ValueError):
        version_int = SCHEMA_VERSION_CURRENT
    schema_version = version_int

    lang_raw = row.get("language")
    is_empty_language = lang_raw is None or (isinstance(lang_raw, str) and lang_raw == "")
    if is_empty_language and schema_version == 1:
        language = "__LEGACY_EMPTY__"
    elif is_empty_language:
        language = "en"
    else:
        language = str(lang_raw)

    s5_raw = row.get("s5_trust_score")
    s5_trust_score = float(s5_raw) if s5_raw is not None else 0.5

    gain_raw = row.get("profile_modulation_gain_json") or "{}"
    gain_plain = _decrypt_field_try_keys(str(gain_raw), row_uuid, keys)
    try:
        profile_modulation_gain = json.loads(gain_plain) or {}
    except (TypeError, json.JSONDecodeError):
        profile_modulation_gain = {}

    last_reviewed_raw = row.get("last_reviewed")
    try:
        last_reviewed = None if pd.isna(last_reviewed_raw) else last_reviewed_raw
    except (TypeError, ValueError):
        last_reviewed = last_reviewed_raw

    literal_raw = row.get("literal_surface", "")
    literal_plain = _decrypt_field_try_keys(str(literal_raw), row_uuid, keys)

    provenance_raw = row.get("provenance_json") or "[]"
    provenance_plain = _decrypt_field_try_keys(str(provenance_raw), row_uuid, keys)
    try:
        provenance_list = json.loads(provenance_plain) if provenance_plain else []
    except (TypeError, json.JSONDecodeError):
        provenance_list = []

    rec = MemoryRecord(
        id=row_uuid,
        tier=row.get("tier", "episodic"),
        literal_surface=literal_plain,
        aaak_index=row.get("aaak_index") or "",
        embedding=(
            list(row["embedding"])
            if row.get("embedding") is not None
            else []
        ),
        community_id=community_id,
        centrality=float(row.get("centrality", 0.0) or 0.0),
        detail_level=int(row.get("detail_level", 1)),
        pinned=bool(row.get("pinned", False)),
        stability=float(row.get("stability") or 0.0),
        difficulty=float(row.get("difficulty") or 0.0),
        last_reviewed=last_reviewed,
        never_decay=bool(row.get("never_decay", False)),
        never_merge=bool(row.get("never_merge", False)),
        provenance=provenance_list,
        created_at=row.get("created_at") or datetime.now(timezone.utc),
        updated_at=row.get("updated_at") or datetime.now(timezone.utc),
        tags=json.loads(row.get("tags_json") or "[]"),
        language=language,
        s5_trust_score=s5_trust_score,
        profile_modulation_gain=profile_modulation_gain,
        schema_version=schema_version,
        structure_hv=structure_hv,
    )
    if language == "__LEGACY_EMPTY__":
        rec.language = ""
    return rec


def migrate_crypto_recover_prior_key(
    store: MemoryStore,
    prior_key: bytes,
    *,
    dry_run: bool = False,
) -> dict:
    from cryptography.exceptions import InvalidTag

    from iai_mcp.crypto import KEY_BYTES

    if len(prior_key) != KEY_BYTES:
        raise ValueError(f"prior_key must be {KEY_BYTES} raw bytes")

    mig = detect_partial_migration(store.db)
    if mig["state"] not in ("clean", "needs_cleanup"):
        raise RuntimeError(
            "crypto recover requires a non-partial reembed state "
            f"(got {mig['state']!r}); resolve migrate --rollback/--resume first."
        )

    cur_key = store._key()
    key_chain = [cur_key, prior_key] if prior_key != cur_key else [cur_key]

    names = _db_table_names_set(store.db)
    if CRYPTO_RECOVER_STAGING in names:
        try:
            store.db.drop_table(CRYPTO_RECOVER_STAGING)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"drop stale {CRYPTO_RECOVER_STAGING} failed: {exc}"
            ) from exc

    orig_tbl = store.db.open_table(RECORDS_TABLE)
    orig_count = int(orig_tbl.count_rows())
    if orig_count == 0:
        return {"no_op": True, "reason": "empty_store", "records_staged": 0, "dry_run": dry_run}

    df = orig_tbl.to_pandas()
    needs_prior = 0
    for _, r in df.iterrows():
        rid = UUID(str(r["id"]))
        lit = str(r.get("literal_surface") or "")
        if not is_encrypted(lit):
            continue
        try:
            _decrypt_field_try_keys(lit, rid, [cur_key])
        except (InvalidTag, ValueError):
            try:
                _decrypt_field_try_keys(lit, rid, [prior_key])
                needs_prior += 1
            except (InvalidTag, ValueError):
                raise RuntimeError(
                    f"record {rid}: literal_surface not decryptable with current "
                    "or prior key — run crypto redact-undecryptable or restore backup"
                ) from None

    if needs_prior == 0:
        return {
            "no_op": True,
            "reason": "all_rows_decrypt_with_current_key",
            "records_staged": 0,
            "dry_run": dry_run,
        }

    if dry_run:
        return {
            "no_op": False,
            "dry_run": True,
            "would_stage": orig_count,
            "rows_needing_prior_key": needs_prior,
        }

    schema = orig_tbl.schema
    staging_tbl = store.db.create_table(CRYPTO_RECOVER_STAGING, schema=schema)
    staged = 0
    t0 = time.time()
    for _, r in df.iterrows():
        row_dict = r.to_dict()
        rec = _memory_record_from_raw_row_multikey(store, row_dict, key_chain)
        staging_tbl.add([store._to_row(rec)])
        staged += 1

    if staged != orig_count:
        try:
            store.db.drop_table(CRYPTO_RECOVER_STAGING)
        except (OSError, RuntimeError) as exc:
            log.error("failed to drop staging table after mismatch: %s", exc)
        raise RuntimeError(
            f"staging row count mismatch: staged={staged} orig={orig_count}"
        )

    duration_sec = time.time() - t0
    try:
        write_event(
            store,
            kind="migration_crypto_recover",
            data={
                "records_staged": staged,
                "duration_sec": duration_sec,
                "rows_needed_prior_key": needs_prior,
            },
            severity="info",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("migration_crypto_recover event write failed: %s", exc)

    ts = int(time.time())
    old_name = f"{OLD_TABLE_PREFIX}{ts}"
    _swap_tables_filesystem(store.db, source=RECORDS_TABLE, dest=old_name)
    _swap_tables_filesystem(
        store.db, source=CRYPTO_RECOVER_STAGING, dest=RECORDS_TABLE
    )

    return {
        "no_op": False,
        "records_staged": staged,
        "duration_sec": duration_sec,
        "dry_run": False,
        "old_table": old_name,
        "rows_needed_prior_key": needs_prior,
    }


REDACT_UNDECRYPTABLE_MARKER = "<REDACTED: pre-2026-04-30 key rotation>"


def migrate_redact_undecryptable_records(store: MemoryStore) -> dict:
    from cryptography.exceptions import InvalidTag

    tbl = store.db.open_table(RECORDS_TABLE)
    if tbl.count_rows() == 0:
        return {"redacted": 0, "skipped_ok": 0, "skipped_plain": 0}

    df = tbl.to_pandas()
    redacted = 0
    skipped_ok = 0
    skipped_plain = 0
    for _, r in df.iterrows():
        rid = UUID(str(r["id"]))
        lit = str(r.get("literal_surface") or "")
        if not is_encrypted(lit):
            skipped_plain += 1
            continue
        try:
            plain = store._decrypt_for_record(rid, lit)
        except (InvalidTag, ValueError):
            plain = None
        if plain is not None:
            skipped_ok += 1
            continue
        prov_raw = str(r.get("provenance_json") or "[]")
        try:
            if is_encrypted(prov_raw):
                prov_plain = store._decrypt_for_record(rid, prov_raw)
            else:
                prov_plain = prov_raw
        except (InvalidTag, ValueError):
            prov_plain = "[]"
        gain_raw = str(r.get("profile_modulation_gain_json") or "{}")
        try:
            if is_encrypted(gain_raw):
                gain_plain = store._decrypt_for_record(rid, gain_raw)
            else:
                gain_plain = gain_raw
        except (InvalidTag, ValueError):
            gain_plain = "{}"
        new_lit = store._encrypt_for_record(rid, REDACT_UNDECRYPTABLE_MARKER)
        new_prov = store._encrypt_for_record(rid, prov_plain)
        new_gain = store._encrypt_for_record(rid, gain_plain)
        tbl.update(
            where=f"id = '{_uuid_literal(rid)}'",
            values={
                "literal_surface": new_lit,
                "provenance_json": new_prov,
                "profile_modulation_gain_json": new_gain,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        redacted += 1
        try:
            write_event(
                store,
                kind="crypto_redaction",
                data={"record_id": str(rid), "reason": "undecryptable_literal"},
                severity="warning",
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("crypto_redaction event write failed: %s", exc)

    return {
        "redacted": redacted,
        "skipped_ok": skipped_ok,
        "skipped_plain": skipped_plain,
    }


def _rollback(db, store: MemoryStore) -> int:
    names = set(db.table_names())
    has_records = RECORDS_TABLE in names
    has_staging = STAGING_TABLE in names
    old_tables = sorted(n for n in names if n.startswith(OLD_TABLE_PREFIX))

    try:
        if has_staging and has_records:
            db.drop_table(STAGING_TABLE)
            _progress_clear(store)
            log.info(
                "migrate_reembed_rollback_drop_staging",
                extra={"records_count": db.open_table(RECORDS_TABLE).count_rows()},
            )
            return 0

        if not has_records and old_tables:
            newest_old = old_tables[-1]
            if has_staging:
                db.drop_table(STAGING_TABLE)
            _swap_tables_filesystem(db, source=newest_old, dest=RECORDS_TABLE)
            try:
                tbl = db.open_table(RECORDS_TABLE)
                emb_field = tbl.schema.field("embedding")
                actual_dim = getattr(emb_field.type, "list_size", None)
                if actual_dim and int(actual_dim) > 0:
                    store._embed_dim = int(actual_dim)
            except (OSError, ValueError, KeyError, AttributeError) as exc:
                log.error("rollback embed_dim refresh failed: %s", exc)
            _progress_clear(store)
            log.info(
                "migrate_reembed_rollback_restore_old",
                extra={
                    "restored_from": newest_old,
                    "records_count": db.open_table(RECORDS_TABLE).count_rows(),
                },
            )
            return 0

        if has_records and old_tables and not has_staging:
            for old in old_tables:
                try:
                    db.drop_table(old)
                except (OSError, RuntimeError) as exc:
                    log.warning(
                        "migrate_reembed_rollback_drop_old_failed",
                        extra={"table": old, "error": str(exc)[:160]},
                    )
            _progress_clear(store)
            return 0

        if has_records and not has_staging and not old_tables:
            _progress_clear(store)
            return 0

        log.error(
            "migrate_reembed_rollback_unrecoverable",
            extra={
                "has_records": has_records,
                "has_staging": has_staging,
                "old_tables": old_tables,
            },
        )
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        log.error(
            "migrate_reembed_rollback_failed",
            extra={"error": str(exc)[:200]},
        )
        return 1


def _resume(db, store: MemoryStore, target_embedder) -> int:
    progress_state = _progress_read(store)
    if not progress_state:
        log.error(
            "migrate_reembed_resume_no_progress_file",
            extra={"path": str(_progress_path(store))},
        )
        return 1

    target_dim = int(target_embedder.DIM)
    saved_target_dim = int(progress_state.get("target_dim") or 0)
    if saved_target_dim and saved_target_dim != target_dim:
        log.error(
            "migrate_reembed_resume_dim_mismatch",
            extra={
                "saved_target_dim": saved_target_dim,
                "embedder_dim": target_dim,
            },
        )
        return 1

    names = set(db.table_names())
    if RECORDS_TABLE not in names:
        log.error("migrate_reembed_resume_records_missing")
        return 2

    if STAGING_TABLE not in names:
        target_tbl = db.create_table(
            STAGING_TABLE, schema=_records_schema_at_dim(target_dim)
        )
        already_staged: set[str] = set()
    else:
        target_tbl = db.open_table(STAGING_TABLE)
        already_staged = set(progress_state.get("staged_ids") or [])

    source_dim = int(store.embed_dim)
    started_at_iso = progress_state.get(
        "started_at", datetime.now(timezone.utc).isoformat()
    )
    total = db.open_table(RECORDS_TABLE).count_rows()
    last_idx = int(progress_state.get("row_index") or 0)

    t0 = time.time()
    try:
        staged_count, failures = _stage_loop(
            store,
            target_embedder,
            target_dim,
            target_tbl,
            store.iter_records(),
            total=total,
            started_at_iso=started_at_iso,
            started_idx=last_idx + 1,
            already_staged_ids=already_staged,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        log.error(
            "migrate_reembed_resume_stage_failed",
            extra={"error": str(exc)[:200]},
        )
        return 2

    total_staged = len(already_staged) + staged_count

    duration_sec = time.time() - t0
    try:
        _validate_and_swap(
            store,
            source_dim=source_dim,
            target_dim=target_dim,
            target_embedder=target_embedder,
            staged_count=total_staged,
            failures=failures,
            duration_sec=duration_sec,
        )
    except RuntimeError as exc:
        log.error(
            "migrate_reembed_resume_validate_failed",
            extra={"error": str(exc)[:200]},
        )
        return 2
    return 0


def _encrypt_or_passthrough(
    store: MemoryStore,
    record_id: UUID,
    value: str,
) -> tuple[str, bool]:
    if is_encrypted(value):
        return value, False
    ad = _uuid_literal(record_id).encode("ascii")
    ct = encrypt_field(value or "", store._key(), associated_data=ad)
    return ct, True


def migrate_encryption_v2_to_v3(
    store: MemoryStore,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    t0 = time.time()
    result = {
        "records_migrated": 0,
        "events_migrated": 0,
        "records_scanned": 0,
        "events_scanned": 0,
        "duration_sec": 0.0,
    }

    records_tbl = store.db.open_table(RECORDS_TABLE)
    records_df = records_tbl.to_pandas()
    result["records_scanned"] = int(len(records_df))

    records_updates: list[dict] = []
    record_total = len(records_df)
    for idx, (_, row) in enumerate(records_df.iterrows()):
        if progress is not None:
            try:
                progress(idx, record_total)
            except (TypeError, ValueError):
                pass
        try:
            rid = UUID(str(row["id"]))
        except (ValueError, TypeError):
            continue

        literal_raw = row.get("literal_surface") or ""
        prov_raw = row.get("provenance_json") or "[]"
        gain_raw = row.get("profile_modulation_gain_json") or "{}"

        any_plaintext = any(
            not is_encrypted(v) for v in (literal_raw, prov_raw, gain_raw)
        )
        if not any_plaintext:
            continue

        if dry_run:
            result["records_migrated"] += 1
            continue

        new_literal, _ = _encrypt_or_passthrough(store, rid, literal_raw)
        new_prov, _ = _encrypt_or_passthrough(store, rid, prov_raw)
        new_gain, _ = _encrypt_or_passthrough(store, rid, gain_raw)
        records_updates.append(
            {
                "id": _uuid_literal(rid),
                "literal_surface": new_literal,
                "provenance_json": new_prov,
                "profile_modulation_gain_json": new_gain,
            }
        )
        result["records_migrated"] += 1

    if not dry_run and records_updates:
        now = datetime.now(timezone.utc)
        import pyarrow as pa
        update_tbl = pa.table(
            {
                "id": [u["id"] for u in records_updates],
                "literal_surface": [u["literal_surface"] for u in records_updates],
                "provenance_json": [u["provenance_json"] for u in records_updates],
                "profile_modulation_gain_json": [
                    u["profile_modulation_gain_json"] for u in records_updates
                ],
                "updated_at": [now] * len(records_updates),
            }
        )
        try:
            records_tbl.merge_insert("id").when_matched_update_all().execute(update_tbl)
        except (OSError, ValueError, AttributeError, RuntimeError, sqlite3.IntegrityError) as exc:
            log.error("merge_insert fallback triggered: %s", exc)
            for u in records_updates:
                try:
                    records_tbl.update(
                        where=f"id = '{u['id']}'",
                        values={
                            "literal_surface": u["literal_surface"],
                            "provenance_json": u["provenance_json"],
                            "profile_modulation_gain_json": u[
                                "profile_modulation_gain_json"
                            ],
                            "updated_at": now,
                        },
                    )
                except (OSError, ValueError, RuntimeError):
                    continue

    events_tbl = store.db.open_table(EVENTS_TABLE)
    events_df = events_tbl.to_pandas()
    result["events_scanned"] = int(len(events_df))

    events_updates: list[dict] = []
    for _, row in events_df.iterrows():
        data_raw = row.get("data_json") or "{}"
        if is_encrypted(data_raw):
            continue
        event_id = str(row["id"])
        if dry_run:
            result["events_migrated"] += 1
            continue
        ad = event_id.encode("ascii")
        new_data = encrypt_field(data_raw, store._key(), associated_data=ad)
        events_updates.append({"id": event_id, "data_json": new_data})
        result["events_migrated"] += 1

    if not dry_run and events_updates:
        for u in events_updates:
            try:
                events_tbl.update(
                    where=f"id = '{u['id']}'",
                    values={"data_json": u["data_json"]},
                )
            except (OSError, ValueError, RuntimeError):
                continue

    result["duration_sec"] = time.time() - t0

    if not dry_run and (
        result["records_migrated"] > 0 or result["events_migrated"] > 0
    ):
        write_event(
            store,
            kind="migration_v2_to_v3",
            data={
                "record_count": result["records_migrated"],
                "event_count": result["events_migrated"],
                "duration_sec": result["duration_sec"],
                "columns_encrypted": [
                    "records.literal_surface",
                    "records.provenance_json",
                    "records.profile_modulation_gain_json",
                    "events.data_json",
                ],
                "algorithm": "AES-256-GCM",
                "format": "iai:enc:v1:",
            },
            severity="info",
        )

    return result


def migrate_hd_vector_to_structure_hv_v3_to_v4(
    store: MemoryStore,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    t0 = time.time()
    result: dict = {
        "processed": 0,
        "updated": 0,
        "skipped": 0,
        "duration_ms": 0.0,
        "column_renamed_from": "hd_vector_json",
        "column_renamed_to": "structure_hv",
    }

    all_records = store.all_records()
    total = len(all_records)
    result["processed"] = total

    from iai_mcp.tem import bind_structure
    from iai_mcp.types import (
        SCHEMA_VERSION_V4,
        STRUCTURE_HV_BYTES,
    )

    tbl = store.db.open_table(RECORDS_TABLE)
    for idx, record in enumerate(all_records):
        if progress is not None:
            try:
                progress(idx, total)
            except (TypeError, ValueError):
                pass

        already_v4 = record.schema_version >= SCHEMA_VERSION_V4
        has_full_hv = (
            isinstance(record.structure_hv, (bytes, bytearray))
            and len(record.structure_hv) == STRUCTURE_HV_BYTES
        )
        if already_v4 and has_full_hv:
            result["skipped"] += 1
            continue

        if dry_run:
            result["updated"] += 1
            continue

        if not has_full_hv:
            record.structure_hv = bind_structure(record)
        record.schema_version = SCHEMA_VERSION_V4

        try:
            tbl.delete(f"id = '{_uuid_literal(record.id)}'")
        except (OSError, ValueError, RuntimeError):
            pass
        store.insert(record)
        result["updated"] += 1

    result["duration_ms"] = (time.time() - t0) * 1000.0

    if not dry_run and (result["updated"] > 0 or result["skipped"] > 0):
        write_event(
            store,
            kind="migration_v3_to_v4",
            data={
                "processed": result["processed"],
                "updated": result["updated"],
                "skipped": result["skipped"],
                "duration_ms": result["duration_ms"],
                "column_renamed_from": result["column_renamed_from"],
                "column_renamed_to": result["column_renamed_to"],
            },
            severity="info",
        )

    return result


def _migrate_add_hv_tier_columns(conn: sqlite3.Connection) -> dict:
    result = {"hv_tier_added": False, "structure_hv_payload_added": False}

    try:
        conn.execute(
            "ALTER TABLE records ADD COLUMN hv_tier TEXT NOT NULL DEFAULT 'bsc'"
        )
        result["hv_tier_added"] = True
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise

    try:
        conn.execute(
            "ALTER TABLE records ADD COLUMN structure_hv_payload BLOB NOT NULL DEFAULT x''"
        )
        result["structure_hv_payload_added"] = True
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise

    return result


def migrate_codec_metadata_v4_to_v5(
    store: "MemoryStore",
    dry_run: bool = False,
) -> dict:
    from iai_mcp.hippo import HippoDB

    db = store.db
    if not isinstance(db, HippoDB):
        return {"dry_run": dry_run, "hv_tier_added": False, "structure_hv_payload_added": False, "note": "non-hippo backend; skipped"}

    existing = {
        row["name"]
        for row in db._conn.execute("PRAGMA table_info(records)").fetchall()
    }
    needs_hv_tier = "hv_tier" not in existing
    needs_payload = "structure_hv_payload" not in existing

    if dry_run:
        return {
            "dry_run": True,
            "hv_tier_added": needs_hv_tier,
            "structure_hv_payload_added": needs_payload,
        }

    result = _migrate_add_hv_tier_columns(db._conn)
    result["dry_run"] = False
    return result


def cleanup_schema_duplicates(
    store: MemoryStore,
    *,
    apply: bool = False,
    store_path: "Path | None" = None,
) -> dict:
    import shutil
    from pathlib import Path
    from datetime import datetime, timezone

    from iai_mcp.store import EDGES_TABLE
    from iai_mcp.types import SEMANTIC_PRUNED_TIER

    groups: dict[str, list[MemoryRecord]] = {}
    try:
        all_records = store.all_records()
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("schema cleanup all_records read failed: %s", exc)
        return {
            "mode": "apply" if apply else "dry-run",
            "groups": 0,
            "keepers": 0,
            "pruned": 0,
            "edges_reinforced": 0,
            "snapshot_dir": None,
        }

    for rec in all_records:
        if rec.tier != "semantic":
            continue
        pattern_tag = next(
            (t for t in (rec.tags or []) if t.startswith("pattern:")),
            None,
        )
        if pattern_tag is None or ":" not in pattern_tag:
            continue
        pattern = pattern_tag.split(":", 1)[1]
        groups.setdefault(pattern, []).append(rec)

    dup_groups = {p: recs for p, recs in groups.items() if len(recs) > 1}

    keepers: list[MemoryRecord] = []
    duplicates: list[MemoryRecord] = []
    for pattern, recs in dup_groups.items():
        recs_sorted = sorted(recs, key=lambda r: r.created_at)
        keepers.append(recs_sorted[0])
        duplicates.extend(recs_sorted[1:])

    edges_to_reinforce = 0
    try:
        edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
        dup_id_strs = {str(d.id) for d in duplicates}
        if dup_id_strs and "edge_type" in edges_df.columns:
            mask = (
                (edges_df["edge_type"] == "schema_instance_of")
                & (
                    edges_df["dst"].isin(dup_id_strs)
                    | edges_df["src"].isin(dup_id_strs)
                )
            )
            edges_to_reinforce = int(mask.sum())
    except (OSError, ValueError, KeyError) as exc:
        log.error("schema cleanup edges scan failed: %s", exc)
        edges_to_reinforce = 0

    snapshot_dir: str | None = None

    if apply and (keepers or duplicates):
        iai_root = Path(store_path) if store_path is not None else Path(store.root)
        src_lancedb = iai_root / "lancedb"
        src_hippo = iai_root / "hippo"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap = iai_root / f"lancedb-pre-cleanup-{ts}"
        if src_lancedb.exists():
            snapshot_source = src_lancedb
        elif src_hippo.exists():
            snapshot_source = src_hippo
        else:
            snapshot_source = iai_root
        shutil.copytree(snapshot_source, snap)
        snapshot_dir = str(snap)

        keeper_by_pattern: dict[str, MemoryRecord] = {}
        for k in keepers:
            kp = next(
                (t for t in (k.tags or []) if t.startswith("pattern:")),
                None,
            )
            if kp and ":" in kp:
                keeper_by_pattern[kp.split(":", 1)[1]] = k

        try:
            edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
            for dup in duplicates:
                dp = next(
                    (t for t in (dup.tags or []) if t.startswith("pattern:")),
                    None,
                )
                if dp is None or ":" not in dp:
                    continue
                pattern = dp.split(":", 1)[1]
                keeper = keeper_by_pattern.get(pattern)
                if keeper is None or keeper.id == dup.id:
                    continue
                dup_str = str(dup.id)
                incoming_mask = (
                    (edges_df["edge_type"] == "schema_instance_of")
                    & ((edges_df["dst"] == dup_str) | (edges_df["src"] == dup_str))
                )
                incoming = edges_df[incoming_mask]
                if incoming.empty:
                    continue
                pairs: list[tuple[UUID, UUID]] = []
                for _, row in incoming.iterrows():
                    other_str = (
                        row["src"] if row["dst"] == dup_str else row["dst"]
                    )
                    if other_str == dup_str:
                        continue
                    try:
                        other_id = UUID(str(other_str))
                    except (TypeError, ValueError):
                        continue
                    pairs.append((other_id, keeper.id))
                if pairs:
                    store.boost_edges(
                        pairs,
                        edge_type="schema_instance_of",
                        delta=0.1,
                    )
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("schema cleanup edge reinforce failed: %s", exc)

        for dup in duplicates:
            try:
                store.delete(dup.id)
                pruned_rec = MemoryRecord(
                    id=dup.id,
                    tier=SEMANTIC_PRUNED_TIER,
                    literal_surface=dup.literal_surface,
                    aaak_index=dup.aaak_index,
                    embedding=dup.embedding,
                    community_id=dup.community_id,
                    centrality=dup.centrality,
                    detail_level=dup.detail_level,
                    pinned=False,
                    stability=dup.stability,
                    difficulty=dup.difficulty,
                    last_reviewed=dup.last_reviewed,
                    never_decay=False,
                    never_merge=dup.never_merge,
                    provenance=dup.provenance,
                    created_at=dup.created_at,
                    updated_at=datetime.now(timezone.utc),
                    tags=dup.tags,
                    language=dup.language,
                    s5_trust_score=dup.s5_trust_score,
                    profile_modulation_gain=dup.profile_modulation_gain,
                    schema_version=dup.schema_version,
                    structure_hv=dup.structure_hv,
                )
                store.insert(pruned_rec)
            except (OSError, ValueError, RuntimeError):
                continue

    summary: dict = {
        "mode": "apply" if apply else "dry-run",
        "groups": len(dup_groups),
        "keepers": len(keepers),
        "pruned": len(duplicates),
        "edges_reinforced": int(edges_to_reinforce),
        "snapshot_dir": snapshot_dir,
    }
    try:
        write_event(
            store,
            kind="schema_cleanup_run",
            data=summary,
            severity="info",
            source_ids=[k.id for k in keepers[:5]] if keepers else None,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("schema_cleanup_run event write failed: %s", exc)
    return summary


# ---------------------------------------------------------------------------
# Timestamp re-derivation migration
# ---------------------------------------------------------------------------

def _find_transcript_ts(
    session_id: str,
    source_uuid: str | None,
    literal_surface: str,
    transcript_root: Path,
) -> "datetime | None":
    """Return the parsed transcript timestamp for a record, or None if unresolvable.

    Scans all JSONL files under transcript_root matching */<session_id>.jsonl.
    Fast path: match by source_uuid. Fallback: match by content hash of literal_surface
    against the transcript line text field.
    """
    from iai_mcp.capture import _resolve_ts

    # Validate session_id to prevent path traversal.
    if not session_id or "/" in session_id or ".." in session_id:
        return None

    pattern = f"*/{session_id}.jsonl"
    matches = list(transcript_root.glob(pattern))
    if not matches:
        return None

    import hashlib

    surface_hash = hashlib.sha256(literal_surface.encode("utf-8")).hexdigest()

    for transcript_path in matches:
        try:
            with transcript_path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        obj = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    ts_str = obj.get("timestamp")
                    if not ts_str:
                        continue
                    # Fast path: uuid match.
                    if source_uuid and obj.get("uuid") == source_uuid:
                        return _resolve_ts(ts_str)
                    # Content-hash fallback: compare against message text fields.
                    text_candidate = (
                        obj.get("text")
                        or obj.get("content")
                        or ""
                    )
                    if isinstance(text_candidate, list):
                        # Claude's content is sometimes a list of blocks.
                        parts = []
                        for block in text_candidate:
                            if isinstance(block, dict):
                                parts.append(block.get("text") or "")
                            elif isinstance(block, str):
                                parts.append(block)
                        text_candidate = "".join(parts)
                    if text_candidate:
                        candidate_hash = hashlib.sha256(
                            text_candidate.encode("utf-8")
                        ).hexdigest()
                        if candidate_hash == surface_hash:
                            return _resolve_ts(ts_str)
        except (OSError, UnicodeDecodeError):
            continue

    return None


def migrate_rederive_collapsed_timestamps(
    store: "MemoryStore",
    *,
    dry_run: bool = False,
    transcript_root: "Path | None" = None,
) -> dict:
    """Re-derive per-turn created_at from on-disk transcripts for records
    whose timestamps collapsed to a single shared value.

    Returns a dict with keys: records_updated, skipped_no_transcript,
    skipped_no_match, dry_run.

    Safe to call multiple times (idempotent). Records whose transcripts are
    absent or unmatched are never modified.  Only created_at is updated —
    literal_surface and provenance_json are never touched.
    """
    from iai_mcp.hippo import HippoDB

    if transcript_root is None:
        transcript_root = Path.home() / ".claude" / "projects"
    else:
        transcript_root = Path(transcript_root)

    db = store.db
    if not isinstance(db, HippoDB):
        return {
            "records_updated": 0,
            "skipped_no_transcript": 0,
            "skipped_no_match": 0,
            "dry_run": dry_run,
        }

    # Load collapsed-group candidates: episodic records sharing created_at
    # with at least 2 other records (group size >= 3).
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT id, created_at FROM records"
            " WHERE tier = 'episodic'"
            "   AND tombstoned_at IS NULL"
            " GROUP BY created_at"
            " HAVING COUNT(*) >= 3"
        ).fetchall()

    if not rows:
        return {
            "records_updated": 0,
            "skipped_no_transcript": 0,
            "skipped_no_match": 0,
            "dry_run": dry_run,
        }

    # Collect all record IDs in collapsed groups.
    candidate_created_ats = {row[1] for row in rows}
    with db._conn_lock:
        all_candidates = db._conn.execute(
            "SELECT id FROM records"
            " WHERE tier = 'episodic'"
            "   AND tombstoned_at IS NULL"
            "   AND created_at IN ({})".format(
                ",".join("?" * len(candidate_created_ats))
            ),
            list(candidate_created_ats),
        ).fetchall()

    record_ids = [row[0] for row in (all_candidates or [])]

    progress = _progress_read(store)
    done_ids: set[str] = set(progress.get("done_ids", []))

    records_updated = 0
    skipped_no_transcript = 0
    skipped_no_match = 0

    for rec_id_str in record_ids:
        if rec_id_str in done_ids:
            continue

        try:
            from uuid import UUID
            rec = store.get(UUID(rec_id_str))
        except (ValueError, Exception):
            skipped_no_match += 1
            continue

        if rec is None:
            skipped_no_match += 1
            continue

        prov = (rec.provenance or [{}])[0]
        session_id = prov.get("session_id") or ""
        source_uuid = prov.get("source_uuid") or None

        if not session_id:
            skipped_no_transcript += 1
            done_ids.add(rec_id_str)
            continue

        # Check whether any transcript file exists for this session.
        if not session_id or "/" in session_id or ".." in session_id:
            skipped_no_transcript += 1
            done_ids.add(rec_id_str)
            continue

        transcript_matches = list(transcript_root.glob(f"*/{session_id}.jsonl"))
        if not transcript_matches:
            skipped_no_transcript += 1
            done_ids.add(rec_id_str)
            continue

        ts = _find_transcript_ts(
            session_id=session_id,
            source_uuid=source_uuid,
            literal_surface=rec.literal_surface,
            transcript_root=transcript_root,
        )

        if ts is None:
            skipped_no_match += 1
            done_ids.add(rec_id_str)
            continue

        if not dry_run:
            with db._conn_lock:
                db._conn.execute(
                    "UPDATE records SET created_at = ? WHERE id = ?",
                    (ts, rec_id_str),
                )
            records_updated += 1
        else:
            records_updated += 1

        done_ids.add(rec_id_str)

        if not dry_run:
            _progress_write(
                store,
                {"done_ids": list(done_ids)},
            )

    if not dry_run:
        try:
            write_event(
                store,
                "migration_rederive_timestamps",
                {
                    "records_updated": records_updated,
                    "skipped_no_transcript": skipped_no_transcript,
                    "skipped_no_match": skipped_no_match,
                },
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("migration_rederive_timestamps event write failed: %s", exc)

        _progress_clear(store)

    return {
        "records_updated": records_updated,
        "skipped_no_transcript": skipped_no_transcript,
        "skipped_no_match": skipped_no_match,
        "dry_run": dry_run,
    }
