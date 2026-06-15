from __future__ import annotations

import concurrent.futures
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from iai_mcp.hippo import HippoDB
from iai_mcp.types import EMBED_DIM


def _rng_unit_vec(rng: np.random.Generator) -> list[float]:
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-10
    return v.tolist()


def _record_row(*, rid: str | None = None, embedding: list[float]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": rid or str(uuid4()),
        "tier": "episodic",
        "literal_surface": "test",
        "embedding": embedding,
        "created_at": now,
    }


def test_knn_query_returns_top_k(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    rows = []
    query_idx = 7

    vecs = [_rng_unit_vec(rng) for _ in range(100)]
    ids = [str(uuid4()) for _ in range(100)]

    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        for i, (rid, vec) in enumerate(zip(ids, vecs)):
            tbl.add([_record_row(rid=rid, embedding=vec)])

        query_vec = vecs[query_idx]
        df = tbl.search(query_vec).limit(10).to_pandas()

    assert len(df) == 10, f"Expected 10 results, got {len(df)}"
    assert "_distance" in df.columns, "_distance column missing from ANN result"
    top_hit_id = str(df.iloc[0]["id"])
    assert top_hit_id == ids[query_idx], (
        f"Top-1 should be the query record (id={ids[query_idx]}), got {top_hit_id}"
    )
    assert (df["_distance"] >= 0).all(), "Negative distances in result"


def test_atomic_save_tmp_rename(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        tbl.add([_record_row(embedding=_rng_unit_vec(rng))])

    hnsw_path = tmp_path / "hippo" / "records.hnsw"
    tmp_path_hnsw = tmp_path / "hippo" / "records.hnsw.tmp"

    assert hnsw_path.exists(), "records.hnsw should exist after close()"
    assert not tmp_path_hnsw.exists(), "records.hnsw.tmp should not exist after close()"
    assert hnsw_path.stat().st_size > 0, "records.hnsw should not be empty"


def test_rebuild_from_sqlite_after_hnsw_corruption(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 20
    vecs = [_rng_unit_vec(rng) for _ in range(n)]
    ids = [str(uuid4()) for _ in range(n)]

    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        for rid, vec in zip(ids, vecs):
            tbl.add([_record_row(rid=rid, embedding=vec)])

    hnsw_path = tmp_path / "hippo" / "records.hnsw"
    assert hnsw_path.exists(), "precondition: records.hnsw written on close"

    hnsw_path.write_bytes(b"\x00" * 16)

    with HippoDB(tmp_path) as db2:
        assert db2._hnsw.get_current_count() == n, (
            f"After rebuild, index should have {n} items, "
            f"got {db2._hnsw.get_current_count()}"
        )
        tbl2 = db2.open_table("records")
        query_vec = vecs[0]
        df = tbl2.search(query_vec).limit(5).to_pandas()

    assert len(df) == 5, "Should return 5 results after rebuild"
    assert str(df.iloc[0]["id"]) == ids[0], "Top hit should be the exact-match record"


def test_label_map_repopulates_on_boot(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 15
    ids = [str(uuid4()) for _ in range(n)]
    vecs = [_rng_unit_vec(rng) for _ in range(n)]

    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        for rid, vec in zip(ids, vecs):
            tbl.add([_record_row(rid=rid, embedding=vec)])

    with HippoDB(tmp_path) as db2:
        rows = db2._conn.execute(
            "SELECT id, vec_label FROM records WHERE tombstoned_at IS NULL"
        ).fetchall()
        expected = {str(r["id"]): int(r["vec_label"]) for r in rows}
        actual = dict(db2._label_map)

    assert actual == expected, (
        f"_label_map mismatch after boot.\n"
        f"  expected {len(expected)} entries, got {len(actual)}"
    )


def test_active_count_excludes_tombstoned(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 10
    ids = [str(uuid4()) for _ in range(n)]
    vecs = [_rng_unit_vec(rng) for _ in range(n)]

    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        for rid, vec in zip(ids, vecs):
            tbl.add([_record_row(rid=rid, embedding=vec)])
        initial_label_map_size = len(db._label_map)

        now = datetime.now(timezone.utc).isoformat()
        db._conn.execute("BEGIN")
        db._conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            (now, ids[0]),
        )
        db._conn.execute("COMMIT")

    assert initial_label_map_size == n, f"Expected {n} in _label_map before tombstone"

    with HippoDB(tmp_path) as db2:
        label_map_size_after = len(db2._label_map)

    assert label_map_size_after == n - 1, (
        f"After tombstone + reboot, _label_map should have {n - 1} entries, "
        f"got {label_map_size_after}"
    )
    assert ids[0] not in db2._label_map


def test_save_index_every_n_writes(tmp_path: Path) -> None:
    from iai_mcp.hippo import HNSW_SAVE_INTERVAL

    rng = np.random.default_rng(5)
    hnsw_path = tmp_path / "hippo" / "records.hnsw"

    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")

        for _ in range(HNSW_SAVE_INTERVAL - 1):
            tbl.add([_record_row(embedding=_rng_unit_vec(rng))])

        size_before = hnsw_path.stat().st_size if hnsw_path.exists() else -1

        tbl.add([_record_row(embedding=_rng_unit_vec(rng))])

        assert hnsw_path.exists(), (
            f"records.hnsw should exist after {HNSW_SAVE_INTERVAL} writes"
        )
        size_after = hnsw_path.stat().st_size
        assert size_after > 0, "records.hnsw should not be empty after periodic save"
        if size_before > 0:
            assert size_after >= size_before


def test_concurrent_add_rlock_protected(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    n_threads = 4
    records_per_thread = 25
    total = n_threads * records_per_thread

    all_vecs = [_rng_unit_vec(rng) for _ in range(total)]
    all_ids = [str(uuid4()) for _ in range(total)]

    errors: list[Exception] = []
    lock = threading.Lock()

    def _worker(thread_idx: int) -> None:
        start = thread_idx * records_per_thread
        end = start + records_per_thread
        try:
            with HippoDB(tmp_path) as db:
                tbl = db.open_table("records")
                for rid, vec in zip(all_ids[start:end], all_vecs[start:end]):
                    tbl.add([_record_row(rid=rid, embedding=vec)])
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    errors_shared: list[Exception] = []
    errors_lock = threading.Lock()

    def _shared_worker(thread_idx: int, db: HippoDB) -> None:
        start = thread_idx * records_per_thread
        end = start + records_per_thread
        try:
            tbl = db.open_table("records")
            for rid, vec in zip(all_ids[start:end], all_vecs[start:end]):
                tbl.add([_record_row(rid=rid, embedding=vec)])
        except Exception as exc:  # noqa: BLE001
            with errors_lock:
                errors_shared.append(exc)

    with HippoDB(tmp_path) as db:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(_shared_worker, i, db)
                for i in range(n_threads)
            ]
            concurrent.futures.wait(futures)

        assert not errors_shared, f"Thread errors: {errors_shared}"

        tbl = db.open_table("records")
        count = tbl.count_rows(filter="tombstoned_at IS NULL")
        assert count == total, f"Expected {total} records, got {count}"

        assert len(db._label_map) == total, (
            f"_label_map should have {total} entries, got {len(db._label_map)}"
        )


class _StubHnsw:
    def __init__(self, labels: np.ndarray, distances: np.ndarray) -> None:
        self._labels = labels
        self._distances = distances

    def knn_query(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        return self._labels, self._distances


def test_knn_query_distance_clamped_on_negative(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    vec = _rng_unit_vec(rng)
    rid = str(uuid4())

    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        tbl.add([_record_row(rid=rid, embedding=vec)])

        inserted_label = next(iter(db._label_map.values()))

        synthetic_labels = np.array([[inserted_label]], dtype=np.int32)
        synthetic_distances = np.array([[-1.192e-7]], dtype=np.float32)

        original_hnsw = db._hnsw
        db._hnsw = _StubHnsw(synthetic_labels, synthetic_distances)
        try:
            df = tbl.search(vec).limit(1).to_pandas()
        finally:
            db._hnsw = original_hnsw

    assert len(df) == 1, f"expected 1 row from stubbed knn_query, got {len(df)}"
    assert df["_distance"].iloc[0] == 0.0, (
        "distance clamp must map negative BLAS-rounding values to 0.0; "
        f"got {df['_distance'].iloc[0]!r}"
    )
