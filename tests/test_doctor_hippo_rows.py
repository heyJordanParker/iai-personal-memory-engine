from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def test_row_f_hippo_readable_clean_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import check_f_hippo_readable, run_diagnosis

    from iai_mcp import doctor as _doctor

    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore",
        lambda: None,
    )
    result = check_f_hippo_readable()
    assert result.status == "PASS"
    assert result.passed is True
    assert "hippo storage readable" in result.name
    assert "Hippo storage opens without error" in result.detail


def test_row_f_hippo_readable_missing_file_fail(monkeypatch):
    monkeypatch.setattr(
        "iai_mcp.store.MemoryStore",
        _raise_runtime_error,
    )
    from iai_mcp.doctor import check_f_hippo_readable

    result = check_f_hippo_readable()
    assert result.status == "FAIL"
    assert result.passed is False
    assert "open failed" in result.detail


def _raise_runtime_error():
    raise RuntimeError("simulated open failure")


def test_row_i_hippo_db_size_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    db = hippo / "brain.sqlite3"
    db.write_bytes(b"x" * (10 * 1024 * 1024))

    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "PASS"
    assert "MB" in result.detail
    assert "healthy" in result.detail


class _FakePathWithSize:

    def __init__(self, size_bytes: int, *, exists: bool = True, raise_stat: bool = False):
        self._size = size_bytes
        self._exists = exists
        self._raise_stat = raise_stat

    def exists(self) -> bool:
        return self._exists

    def stat(self):
        if self._raise_stat:
            raise OSError("permission denied")

        class _R:
            pass

        r = _R()
        r.st_size = self._size  # type: ignore[attr-defined]
        return r

    def __truediv__(self, other):
        return self

    def __str__(self):
        return "/fake/brain.sqlite3"


def test_row_i_hippo_db_size_warn_over_500mb(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    import iai_mcp.doctor as _doctor

    monkeypatch.setattr(
        _doctor, "_resolve_hippo_db_path",
        lambda: _FakePathWithSize(600 * 1024 * 1024),
    )
    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "WARN"
    assert result.passed is True
    assert "compact-hippo" in result.detail


def test_row_i_hippo_db_size_fail_at_2048mb(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    import iai_mcp.doctor as _doctor

    monkeypatch.setattr(
        _doctor, "_resolve_hippo_db_path",
        lambda: _FakePathWithSize(2048 * 1024 * 1024),
    )
    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "FAIL"
    assert result.passed is False
    assert "run compaction immediately" in result.detail


def test_row_i_hippo_db_size_warn_on_stat_oserror(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    import iai_mcp.doctor as _doctor

    monkeypatch.setattr(
        _doctor, "_resolve_hippo_db_path",
        lambda: _FakePathWithSize(0, raise_stat=True),
    )
    from iai_mcp.doctor import check_i_hippo_db_size

    result = check_i_hippo_db_size()
    assert result.status == "WARN"
    assert result.passed is True
    assert "stat failed" in result.detail


def test_row_r_hnsw_loadable_absent_warn(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "hippo").mkdir()

    from iai_mcp.doctor import check_r_hippo_hnsw_loadable

    result = check_r_hippo_hnsw_loadable()
    assert result.status == "WARN"
    assert result.passed is True
    assert "absent" in result.detail


def test_row_r_hnsw_zero_bytes_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    hnsw = hippo / "records.hnsw"
    hnsw.write_bytes(b"")

    from iai_mcp.doctor import check_r_hippo_hnsw_loadable

    result = check_r_hippo_hnsw_loadable()
    assert result.status == "FAIL"
    assert result.passed is False
    assert "zero bytes" in result.detail


def test_row_r_hnsw_corrupted_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    hnsw = hippo / "records.hnsw"
    hnsw.write_bytes(b"this is not a valid hnsw index")

    from iai_mcp.doctor import check_r_hippo_hnsw_loadable

    result = check_r_hippo_hnsw_loadable()
    assert result.status == "FAIL"
    assert result.passed is False
    assert "rebuild" in result.detail


def test_row_s_schema_version_match(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    db_path = hippo / "brain.sqlite3"

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE _hippo_meta (key TEXT, value TEXT)")
    conn.execute("INSERT INTO _hippo_meta VALUES ('schema_version', '1')")
    conn.commit()
    conn.close()

    from iai_mcp.doctor import check_s_hippo_schema_version

    result = check_s_hippo_schema_version()
    assert result.status == "PASS"
    assert result.passed is True
    assert "schema_version=1" in result.detail


def test_row_s_schema_drift_warn(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    db_path = hippo / "brain.sqlite3"

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE _hippo_meta (key TEXT, value TEXT)")
    conn.execute("INSERT INTO _hippo_meta VALUES ('schema_version', '99')")
    conn.commit()
    conn.close()

    from iai_mcp.doctor import check_s_hippo_schema_version

    result = check_s_hippo_schema_version()
    assert result.status == "WARN"
    assert result.passed is True
    assert "schema_version=99" in result.detail


def test_row_s_db_absent_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    from iai_mcp.doctor import check_s_hippo_schema_version

    result = check_s_hippo_schema_version()
    assert result.status == "PASS"
    assert "absent" in result.detail


def test_row_t_hippo_compaction_fresh_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from datetime import datetime, timezone

    recent_ts = datetime.now(timezone.utc).isoformat()
    fake_event = {"kind": "hippo_compacted", "ts": recent_ts}

    import iai_mcp.store as _store
    import iai_mcp.events as _events

    class _FakeStore:
        pass

    monkeypatch.setattr(_store, "MemoryStore", _FakeStore)
    monkeypatch.setattr(
        _events, "query_events",
        lambda store, kind=None, limit=1: [fake_event],
    )

    from iai_mcp.doctor import check_t_hippo_compacted_freshness

    result = check_t_hippo_compacted_freshness()
    assert result.status == "PASS"
    assert result.passed is True
    assert "hippo_compacted" in result.name


def test_row_t_hippo_compaction_stale_warn(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    import iai_mcp.store as _store
    import iai_mcp.events as _events

    class _FakeStore:
        pass

    monkeypatch.setattr(_store, "MemoryStore", _FakeStore)
    monkeypatch.setattr(
        _events, "query_events",
        lambda store, kind=None, limit=1: [],
    )

    from iai_mcp.doctor import check_t_hippo_compacted_freshness

    result = check_t_hippo_compacted_freshness()
    assert result.status == "WARN"
    assert result.passed is True
    assert "no hippo_compacted event" in result.detail


def test_doctor_total_count_22(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    assert len(results) == 25, (
        f"expected 25 rows; got {len(results)}: {[r.name for r in results]}"
    )


def _build_records_table(db_path: Path) -> None:
    """Create a minimal records table for timestamp-collapse tests."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS records "
        "(id TEXT PRIMARY KEY, tier TEXT, created_at TEXT)"
    )
    conn.commit()
    conn.close()


def test_check_x_collapsed_timestamps_warns(tmp_path, monkeypatch):
    """A store with >=5 episodic records sharing one created_at yields WARN."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    db_path = hippo / "brain.sqlite3"

    _build_records_table(db_path)
    collapsed_ts = "2024-01-01T00:00:00+00:00"
    conn = sqlite3.connect(str(db_path))
    for i in range(7):
        conn.execute(
            "INSERT INTO records (id, tier, created_at) VALUES (?, ?, ?)",
            (f"r-{i}", "episodic", collapsed_ts),
        )
    conn.commit()
    conn.close()

    from iai_mcp.doctor import check_x_no_collapsed_timestamps

    result = check_x_no_collapsed_timestamps()
    assert result.passed is False
    assert result.status == "WARN"
    assert "7" in result.detail
    assert "iai-mcp migrate --rederive-timestamps" in result.detail


def test_check_x_collapsed_timestamps_pass(tmp_path, monkeypatch):
    """A store with episodic records each having a distinct created_at yields PASS."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    hippo = tmp_path / "hippo"
    hippo.mkdir()
    db_path = hippo / "brain.sqlite3"

    _build_records_table(db_path)
    conn = sqlite3.connect(str(db_path))
    for i in range(6):
        conn.execute(
            "INSERT INTO records (id, tier, created_at) VALUES (?, ?, ?)",
            (f"r-{i}", "episodic", f"2024-01-01T00:00:{i:02d}+00:00"),
        )
    conn.commit()
    conn.close()

    from iai_mcp.doctor import check_x_no_collapsed_timestamps

    result = check_x_no_collapsed_timestamps()
    assert result.passed is True
    assert result.status != "WARN"


def test_no_lance_storage_optimized_in_identity_audit():
    import inspect
    from iai_mcp import identity_audit

    src = inspect.getsource(identity_audit)
    assert "lance_storage_optimized" not in src, (
        "identity_audit.py still contains 'lance_storage_optimized'; "
        "it must use 'hippo_compacted' instead."
    )
    assert "optimize_lance_storage" not in src, (
        "identity_audit.py still imports optimize_lance_storage; "
        "it must use optimize_hippo_storage."
    )
