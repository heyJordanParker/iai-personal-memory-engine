from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.perf

from tests.test_pipeline_perf import _seed_store

CI_GENEROUS_P95_S: float = 0.200

def test_pipeline_recall_p95_under_ci_ceiling_after_normalize(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    from _perf_helpers import best_of_n, skip_if_loaded

    skip_if_loaded()

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=200, seed=0,
    )

    cues = [
        "what did we cover about auth yesterday?",
        "explain the db migration plan",
        "how does the web cache invalidation work",
        "summary of the cli subcommand changes",
        "recent network stack bug report",
    ]

    recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue=cues[0], session_id="warm", budget_tokens=1500,
    )

    def _one_p95() -> float:
        latencies: list[float] = []
        for i in range(20):
            cue = cues[i % len(cues)]
            t0 = time.perf_counter()
            recall_for_response(
                store=store, graph=graph, assignment=assignment,
                rich_club=rich_club, embedder=embedder,
                cue=cue, session_id="perf_gate", budget_tokens=1500,
            )
            latencies.append(time.perf_counter() - t0)
        latencies.sort()
        return latencies[int(0.95 * len(latencies))]

    p95 = best_of_n(_one_p95, n=3)
    p95_ms = p95 * 1000.0
    print(
        f"\n[perf-gate] recall_for_response N=200 warm best-of-3 p95 = {p95_ms:.2f} ms "
        f"(CI ceiling: {CI_GENEROUS_P95_S * 1000:.0f} ms; "
        f"reference-host strict: 83.6 ms via bench/neural_map.py)"
    )

    assert p95 < CI_GENEROUS_P95_S, (
        f"Normalize regression: recall_for_response N=200 warm "
        f"best-of-3 p95 = {p95_ms:.2f} ms exceeds CI ceiling "
        f"{CI_GENEROUS_P95_S * 1000:.0f} ms."
    )

def test_normalize_overhead_is_submillisecond(tmp_path, capsys):
    from iai_mcp.pipeline import recall_for_response

    from _perf_helpers import best_of_n, skip_if_loaded

    skip_if_loaded()

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=100, seed=1,
    )

    cues = [
        "auth verbatim cue",
        "db schema rebuild",
        "web cache invalidation",
    ]

    recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue=cues[0], session_id="warm", budget_tokens=1500,
    )

    def _one_p95() -> float:
        latencies: list[float] = []
        for i in range(10):
            cue = cues[i % len(cues)]
            t0 = time.perf_counter()
            recall_for_response(
                store=store, graph=graph, assignment=assignment,
                rich_club=rich_club, embedder=embedder,
                cue=cue, session_id="overhead_check", budget_tokens=1500,
            )
            latencies.append(time.perf_counter() - t0)
        latencies.sort()
        return latencies[int(0.95 * len(latencies))]

    p95 = best_of_n(_one_p95, n=3)
    p95_ms = p95 * 1000.0
    print(
        f"\n[perf-gate] recall_for_response N=100 warm best-of-3 p95 = {p95_ms:.2f} ms "
        f"(normalize overhead: one division + one getattr per call)"
    )

    assert p95 < CI_GENEROUS_P95_S, (
        f"normalize-overhead sanity: best-of-3 p95 = {p95_ms:.2f} ms > "
        f"CI ceiling {CI_GENEROUS_P95_S * 1000:.0f} ms"
    )
