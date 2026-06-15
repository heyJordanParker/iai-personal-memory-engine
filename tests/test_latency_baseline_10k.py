from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.perf

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM

import numpy as np


LATENCY_CEILING_MS = 200
N_RECORDS_SMALL = 100
N_RECORDS_MEDIUM = 1000
N_QUERIES = 20


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _populate(store: MemoryStore, n: int):
    for i in range(n):
        rec = _make(text=f"Record number {i}: test content for latency baseline", vec=_random_vec(i))
        store.insert(rec)


def _measure_query_latencies(store: MemoryStore, n_queries: int) -> list[float]:
    latencies = []
    for i in range(n_queries):
        q = _random_vec(10000 + i)
        t0 = time.perf_counter()
        store.query_similar(q, k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
    return sorted(latencies)


@pytest.mark.slow
def test_p95_latency_at_100_records(tmp_path):
    store = MemoryStore(str(tmp_path))
    _populate(store, N_RECORDS_SMALL)
    latencies = _measure_query_latencies(store, N_QUERIES)
    p95 = latencies[int(len(latencies) * 0.95)]
    assert p95 < LATENCY_CEILING_MS, (
        f"p95 latency at N={N_RECORDS_SMALL}: {p95:.1f}ms > {LATENCY_CEILING_MS}ms ceiling"
    )


@pytest.mark.slow
def test_p95_latency_at_1000_records(tmp_path):
    store = MemoryStore(str(tmp_path))
    _populate(store, N_RECORDS_MEDIUM)
    latencies = _measure_query_latencies(store, N_QUERIES)
    p95 = latencies[int(len(latencies) * 0.95)]
    assert p95 < LATENCY_CEILING_MS, (
        f"p95 latency at N={N_RECORDS_MEDIUM}: {p95:.1f}ms > {LATENCY_CEILING_MS}ms ceiling"
    )


def test_insert_latency_stable(tmp_path):
    from _perf_helpers import best_of_n, skip_if_loaded

    skip_if_loaded()

    counter = {"i": 0}

    def _one_p95() -> float:
        i = counter["i"]
        counter["i"] += 1
        store = MemoryStore(str(tmp_path / f"run{i}"))
        latencies = []
        for j in range(50):
            rec = _make(text=f"Insert latency test {j}", vec=_random_vec(j))
            t0 = time.perf_counter()
            store.insert(rec)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
        return sorted(latencies)[int(len(latencies) * 0.95)]

    p95 = best_of_n(_one_p95, n=3)
    assert p95 < 500, f"Insert best-of-3 p95: {p95:.1f}ms — too slow for ambient capture"
