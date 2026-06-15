from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.perf

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

RSS_MB_THRESHOLD = 1500.0

WALL_TIME_SEC_THRESHOLD = 180.0

@pytest.fixture(scope="module")
def bench_result_and_store_path(tmp_path_factory):
    store_dir = tmp_path_factory.mktemp("perf_regression_bench")
    store_path = store_dir / "lancedb"
    store_path.mkdir(parents=True, exist_ok=True)

    from bench.memory_footprint import run_memory_footprint

    start = time.monotonic()
    result = run_memory_footprint(n=1000, store_path=store_path)
    wall = time.monotonic() - start

    return result, store_path, wall

def _query_lance_buffer_flush_events(store_path: Path, table_name: str) -> list[dict]:
    from iai_mcp.events import query_events
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=store_path)
    try:
        events = query_events(store, kind="lance_buffer_flush", limit=10000)
        return [e for e in events if (e.get("data") or {}).get("table") == table_name]
    finally:
        del store

@pytest.mark.slow
def test_n1000_completes_within_wall_time_threshold(bench_result_and_store_path):
    _result, _store_path, wall = bench_result_and_store_path
    assert wall <= WALL_TIME_SEC_THRESHOLD, (
        f"N=1000 bench wall-time regression: {wall:.1f}s > {WALL_TIME_SEC_THRESHOLD}s; "
        f"buffer wiring may have reverted to per-row writes"
    )

@pytest.mark.slow
def test_n1000_rss_peak_under_threshold(bench_result_and_store_path):
    result, _store_path, _wall = bench_result_and_store_path
    assert isinstance(result, dict)
    rss = float(result.get("rss_mb_peak", 0.0))
    assert rss > 0, "bench did not report rss_mb_peak"
    assert rss <= RSS_MB_THRESHOLD, (
        f"N=1000 bench RSS regression: {rss:.1f} MB > {RSS_MB_THRESHOLD} MB threshold; "
        f"buffered-write fix may have regressed"
    )

@pytest.mark.slow
def test_n1000_emits_records_lance_buffer_flush_events(bench_result_and_store_path):
    _result, store_path, _wall = bench_result_and_store_path
    records_flushes = _query_lance_buffer_flush_events(store_path, "records")
    assert len(records_flushes) >= 5, (
        f"records buffer wiring not exercised: only {len(records_flushes)} "
        f"lance_buffer_flush events for table=records "
        f"(expected >=5 with 1000 records / 100-row default threshold)"
    )

@pytest.mark.slow
@pytest.mark.skip(
    reason=(
        "Bench at N=1000 does not exercise the EDGES write path by default. "
        "Hebbian self-loops are gated by pattern-separation enable + non-dry-run "
        "(store.py line ~875); the bench harness does not enable that path. "
        "EDGES buffer wiring is fully verified by the 13 unit/static tests in "
        "tests/test_edge_write_buffer.py (call-site flips at boost_edges insert + "
        "add_contradicts_edge, daemon flush at 3 hooks, telemetry event payload "
        "schema). Bench-driven EDGES exercising is a phase-verification concern "
        "covered at orchestrator level by N=10000 bench run with explicit edge "
        "writes; see bench/memory_footprint.py docstring."
    )
)
def test_n1000_emits_edges_lance_buffer_flush_events(bench_result_and_store_path):
    _result, store_path, _wall = bench_result_and_store_path
    edges_flushes = _query_lance_buffer_flush_events(store_path, "edges")
    assert len(edges_flushes) >= 1, (
        f"edges buffer wiring not exercised: 0 lance_buffer_flush events for table=edges "
        f"(expected >=1 from hebbian self-loop edge writes at N=1000)"
    )

@pytest.mark.slow
def test_n1000_bench_passes_existing_threshold_mb_check(bench_result_and_store_path):
    result, _store_path, _wall = bench_result_and_store_path
    assert result.get("passed") is True, (
        f"bench failed its internal threshold_mb gate: {result}"
    )
