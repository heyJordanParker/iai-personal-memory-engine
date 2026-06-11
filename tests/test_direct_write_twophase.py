from __future__ import annotations

import sqlite3
import struct
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _make_episodic_record(text: str = "generic user turn"):
    import numpy as np
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    rng = np.random.RandomState(seed=42)
    vec = rng.randn(EMBED_DIM).tolist()
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "test-session", "role": "user"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["role:user"],
        language="en",
    )


def _zero_vector_blob(embed_dim: int) -> bytes:
    return struct.pack(f"<{embed_dim}f", *([0.0] * embed_dim))


def test_direct_write_visible_to_recency_daemon_down(hermetic_store: Path) -> None:
    from iai_mcp.direct_write import write_turn_direct  # type: ignore[import]
    from iai_mcp.direct_recency import read_recent_user_turns_direct  # type: ignore[import]

    t0 = time.monotonic()
    write_turn_direct(
        store_root=hermetic_store,
        text="direct write probe text",
        session_id="test-session",
        role="user",
    )
    elapsed_write = time.monotonic() - t0
    assert elapsed_write <= 1.5, f"direct write took {elapsed_write:.3f} s (SLO ≤1.5 s)"

    t1 = time.monotonic()
    turns = read_recent_user_turns_direct(hermetic_store, n=5)
    elapsed_read = time.monotonic() - t1
    assert elapsed_read <= 1.5, f"recency read after direct write took {elapsed_read:.3f} s"

    surfaces = [t.literal_surface for t in turns]
    assert any("direct write probe text" in s for s in surfaces), (
        "directly written turn not visible via recency immediately after write"
    )


def test_no_duplicate_row_on_redrain(hermetic_store: Path) -> None:
    from iai_mcp.direct_write import write_turn_direct  # type: ignore[import]
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.capture import capture_turn

    ts_iso = datetime.now(timezone.utc).isoformat()
    text = "idem-dedup probe text"
    session_id = "idem-session"

    write_turn_direct(
        store_root=hermetic_store,
        text=text,
        session_id=session_id,
        role="user",
        ts_iso=ts_iso,
    )

    store = MemoryStore(hermetic_store)
    try:
        capture_turn(
            store,
            cue=text,
            text=text,
            tier="episodic",
            session_id=session_id,
            role="user",
            ts=ts_iso,
        )
        flush_record_buffer(store)

        from iai_mcp.capture import _idem_tag
        tag = _idem_tag(session_id, "user", ts_iso, text)
        record_id = store.find_record_by_tag(tag)
        assert record_id is not None, "idem-tagged row should exist after direct write"

        records = store.all_records()
        matching = [r for r in records if text in (r.literal_surface or "")]
        assert len(matching) == 1, (
            f"expected exactly 1 row for idem text, got {len(matching)} (duplicate on re-drain)"
        )
    finally:
        store.close()


def test_integrity_rebuild_triggers_mid_run(hermetic_store: Path) -> None:
    from iai_mcp.hippo import HippoDB
    from iai_mcp.types import EMBED_DIM

    import numpy as np

    hippo = HippoDB(hermetic_store)
    try:
        vec_blob = _zero_vector_blob(EMBED_DIM)
        record_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with hippo._conn_lock:
            hippo._conn.execute(
                "INSERT INTO records "
                "(id, tier, literal_surface, aaak_index, embedding, "
                " created_at, updated_at, hv_tier, structure_hv_payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'bsc', x'')",
                (record_id, "episodic", "mid-run probe", "", vec_blob, now, now),
            )
            hippo._conn.commit()

        active_before = len(hippo._label_map)
        with hippo._conn_lock:
            sqlite_count_row = hippo._conn.execute(
                "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NULL"
            ).fetchone()
        sqlite_count = sqlite_count_row[0]
        assert sqlite_count > active_before, (
            "test setup failed: SQLite count should exceed label-map count"
        )

        from iai_mcp.hippo import reconcile_index_mid_run  # type: ignore[import]
        reconcile_index_mid_run(hippo)

        assert record_id in hippo._label_map, (
            "injected record not in _label_map after mid-run rebuild"
        )
    finally:
        hippo.close()


def test_daemon_down_write_deferred_embedding_slo(hermetic_store: Path) -> None:
    from iai_mcp.direct_write import write_turn_direct  # type: ignore[import]
    from iai_mcp.direct_recency import read_recent_user_turns_direct  # type: ignore[import]
    from iai_mcp.types import EMBED_DIM

    t0 = time.monotonic()
    write_turn_direct(
        store_root=hermetic_store,
        text="deferred embedding probe text",
        session_id="test-session",
        role="user",
        deferred_embedding=True,
    )
    elapsed = time.monotonic() - t0

    assert elapsed <= 1.5, (
        f"deferred-embedding write took {elapsed:.3f} s (SLO ≤1.5 s) — "
        "write must complete fast without calling the embedder"
    )

    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT embedding, embedding_pending FROM records "
            "WHERE literal_surface LIKE '%deferred embedding probe%' "
            "AND tombstoned_at IS NULL"
        ).fetchone()
        assert row is not None, "deferred-embedding row not found in SQLite"
        blob = row["embedding"]
        assert blob is not None and len(blob) > 0, (
            "embedding BLOB must not be NULL/empty (records.embedding is BLOB NOT NULL)"
        )
        floats = struct.unpack(f"<{EMBED_DIM}f", blob)
        assert len(floats) == EMBED_DIM, (
            f"BLOB length mismatch: got {len(floats)} floats, expected {EMBED_DIM}"
        )
        assert all(f == 0.0 for f in floats), (
            "pending row must carry a zero-vector BLOB (not a real embedding)"
        )
        pending_flag = row["embedding_pending"]
        assert pending_flag == 1, (
            f"embedding_pending flag must be 1 for a deferred-embed row, got {pending_flag}"
        )
    finally:
        conn.close()

    turns = read_recent_user_turns_direct(hermetic_store, n=5)
    surfaces = [t.literal_surface for t in turns]
    assert any("deferred embedding probe" in s for s in surfaces), (
        "pending row not recency-recallable immediately (recency is embedding-independent)"
    )

    from iai_mcp.direct_write import simulate_daemon_reembed  # type: ignore[import]
    import numpy as np

    rng = np.random.RandomState(seed=99)
    real_embedding = rng.randn(EMBED_DIM).tolist()
    simulate_daemon_reembed(hermetic_store, text_fragment="deferred embedding probe", embedding=real_embedding)

    conn2 = sqlite3.connect(str(db_path))
    conn2.row_factory = sqlite3.Row
    try:
        row2 = conn2.execute(
            "SELECT embedding, embedding_pending FROM records "
            "WHERE literal_surface LIKE '%deferred embedding probe%' "
            "AND tombstoned_at IS NULL"
        ).fetchone()
        assert row2 is not None
        blob2 = row2["embedding"]
        floats2 = struct.unpack(f"<{EMBED_DIM}f", blob2)
        assert any(f != 0.0 for f in floats2), (
            "after daemon re-embed the BLOB must be non-zero (a real embedding)"
        )
        assert row2["embedding_pending"] == 0, (
            "embedding_pending flag must be cleared after daemon re-embed"
        )
    finally:
        conn2.close()


def test_boot_with_pending_row_no_crash_no_churn(hermetic_store: Path) -> None:
    from iai_mcp.types import EMBED_DIM
    import numpy as np

    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(hermetic_store)
    try:
        rec = _make_episodic_record("normal embedded row")
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    pending_id = str(uuid.uuid4())
    zero_blob = _zero_vector_blob(EMBED_DIM)
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
        if "embedding_pending" not in cols:
            conn.execute(
                "ALTER TABLE records ADD COLUMN embedding_pending INTEGER NOT NULL DEFAULT 0"
            )
            conn.commit()

        conn.execute(
            "INSERT INTO records "
            "(id, tier, literal_surface, aaak_index, embedding, embedding_pending, "
            " created_at, updated_at, hv_tier, structure_hv_payload) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?, 'bsc', x'')",
            (pending_id, "episodic", "pending row text", "", zero_blob, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    from iai_mcp.hippo import HippoDB
    hippo = HippoDB(hermetic_store)
    try:
        active_label_count = len(hippo._label_map)
        with hippo._conn_lock:
            non_pending_count_row = hippo._conn.execute(
                "SELECT COUNT(*) FROM records "
                "WHERE tombstoned_at IS NULL AND (embedding_pending IS NULL OR embedding_pending = 0)"
            ).fetchone()
        non_pending_count = non_pending_count_row[0]

        assert active_label_count == non_pending_count, (
            f"label churn bug: active_label_count={active_label_count} != "
            f"non_pending_count={non_pending_count}; pending rows must be excluded "
            "from the ANN label map to prevent perpetual rebuild"
        )

        from iai_mcp.direct_recency import read_recent_user_turns_direct  # type: ignore[import]
        turns = read_recent_user_turns_direct(hermetic_store, n=10)
        surfaces = [t.literal_surface for t in turns]
        assert any("pending row text" in s for s in surfaces), (
            "pending row must be recency-recallable immediately (embedding-independent)"
        )

        from iai_mcp.direct_write import simulate_daemon_reembed  # type: ignore[import]
        rng = np.random.RandomState(seed=77)
        real_vec = rng.randn(EMBED_DIM).tolist()
        simulate_daemon_reembed(hermetic_store, text_fragment="pending row", embedding=real_vec)

        hippo.close()
        hippo = HippoDB(hermetic_store)
        assert pending_id in hippo._label_map, (
            "pending row should be in the ANN label map after re-embed"
        )
    finally:
        hippo.close()


def test_pre_migration_store_opens_and_reconciles(hermetic_store: Path) -> None:
    from iai_mcp.types import EMBED_DIM
    import numpy as np

    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(hermetic_store)
    try:
        rec = _make_episodic_record("migration test row")
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    _CREATE_BACKUP_SQL = (
        "CREATE TABLE records_v4_backup AS SELECT"
        " vec_label, id, tier, literal_surface, aaak_index, embedding, structure_hv,"
        " community_id, centrality, detail_level, pinned, stability, difficulty,"
        " last_reviewed, never_decay, never_merge, tombstoned_at, schema_bypass,"
        " labile_until, provenance_json, created_at, updated_at, tags_json, language,"
        " s5_trust_score, profile_modulation_gain_json, schema_version, wing, room,"
        " drawer, valence, hv_tier, structure_hv_payload"
        " FROM records"
    )
    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    conn = sqlite3.connect(str(db_path))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
        if "embedding_pending" in cols:
            conn.execute("BEGIN")
            conn.execute(_CREATE_BACKUP_SQL)
            conn.execute("DROP TABLE records")
            conn.execute("ALTER TABLE records_v4_backup RENAME TO records")
            conn.execute("COMMIT")
    finally:
        conn.close()

    from iai_mcp.hippo import HippoDB
    hippo = HippoDB(hermetic_store)
    try:
        with hippo._conn_lock:
            cols_after = {row[1] for row in hippo._conn.execute("PRAGMA table_info(records)")}
        assert "embedding_pending" in cols_after, (
            "_reconcile_columns must add embedding_pending column to pre-migration store"
        )

        hippo.close()

        store2 = MemoryStore(hermetic_store)
        try:
            records = store2.all_records()
            assert len(records) >= 1, "migration test row should be present after reconcile"
        finally:
            store2.close()

    except Exception:
        hippo.close()
        raise
    else:
        hippo.close()
