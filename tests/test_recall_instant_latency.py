from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

import iai_mcp.pipeline as _pipeline_mod
from iai_mcp.embed import Embedder
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM


RNG_SEED = 20260601
N_SMALL = 1_000
N_LARGE = 10_000
N_TRIALS = 12
LATENCY_CEILING_MS = 1500.0

LEXICAL_GENERIC_CUE = "hello"
LEXICAL_SPECIFIC_CUE = "specialized technical framework review"

RICH_CLUB_CAP = 50


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _p95(samples: list[float]) -> float:
    s = sorted(samples)
    return s[int(len(s) * 0.95)]


def _monkeypatch_env(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    monkeypatch.setenv("IAI_MCP_RECALL_SAMPLE_RATE", "1.0")


def _make_gold_record(i: int, vec: list[float]) -> object:
    from iai_mcp.types import MemoryRecord
    return MemoryRecord(
        id=UUID(int=i),
        tier="episodic",
        literal_surface=f"User reference gold doc {i}",
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def _populate_store(store: MemoryStore, n: int, embed_gold: bool = True) -> None:
    rng = np.random.default_rng(RNG_SEED)
    for i in range(n):
        v = rng.random(EMBED_DIM).astype(np.float32)
        v = (v / np.linalg.norm(v)).tolist()
        rec = _make(text=f"User filler record {i}", vec=v)
        store.insert(rec)

    if not embed_gold:
        return

    embedder = Embedder()
    cue_gen_arr = np.asarray(embedder.embed(LEXICAL_GENERIC_CUE), dtype=np.float32)
    cue_gen_arr /= np.linalg.norm(cue_gen_arr)
    cue_spec_arr = np.asarray(embedder.embed(LEXICAL_SPECIFIC_CUE), dtype=np.float32)
    cue_spec_arr /= np.linalg.norm(cue_spec_arr)

    store.insert(_make_gold_record(1, list(cue_gen_arr)))

    rng4 = np.random.default_rng(44444)
    hub_vec = rng4.random(EMBED_DIM).astype(np.float32)
    hub_vec /= np.linalg.norm(hub_vec)
    store.insert(_make_gold_record(2, hub_vec.tolist()))
    store.boost_edges([(UUID(int=2), UUID(int=1))], edge_type="hebbian", delta=[3.0])
    for extra_i in range(12):
        store.boost_edges([(UUID(int=2), UUID(int=1000 + extra_i))], edge_type="hebbian", delta=[1.0])

    store.insert(_make_gold_record(3, list(cue_spec_arr)))

    rng5 = np.random.default_rng(55555)
    inter_noise = rng5.random(EMBED_DIM).astype(np.float32)
    inter_noise -= np.dot(inter_noise, cue_spec_arr) * cue_spec_arr
    inter_noise /= np.linalg.norm(inter_noise)
    inter_vec = 0.4 * cue_spec_arr + 0.9165 * inter_noise
    inter_vec /= np.linalg.norm(inter_vec)
    store.insert(_make_gold_record(4, inter_vec.tolist()))
    for extra_j in range(10):
        store.boost_edges([(UUID(int=4), UUID(int=2000 + extra_j))], edge_type="hebbian", delta=[1.0])

    rng6 = np.random.default_rng(66666)
    noise = rng6.random(EMBED_DIM).astype(np.float32)
    noise -= np.dot(noise, cue_spec_arr) * cue_spec_arr
    noise /= np.linalg.norm(noise)
    target_cosine = 0.02
    orth_mag = float(np.sqrt(max(0.0, 1.0 - target_cosine**2)))
    two_hop_vec = target_cosine * cue_spec_arr + orth_mag * noise
    two_hop_vec /= np.linalg.norm(two_hop_vec)
    store.insert(_make_gold_record(5, two_hop_vec.tolist()))
    store.boost_edges([(UUID(int=3), UUID(int=4))], edge_type="hebbian", delta=[5.0])
    store.boost_edges([(UUID(int=4), UUID(int=5))], edge_type="hebbian", delta=[5.0])
    for extra_k in range(8):
        store.boost_edges([(UUID(int=5), UUID(int=3000 + extra_k))], edge_type="hebbian", delta=[2.0])

    rng3 = np.random.default_rng(77777)
    ca_vec = rng3.random(EMBED_DIM).astype(np.float32)
    ca_vec = (ca_vec / np.linalg.norm(ca_vec)).tolist()
    cb_vec = rng3.random(EMBED_DIM).astype(np.float32)
    cb_vec = (cb_vec / np.linalg.norm(cb_vec)).tolist()
    store.insert(_make_gold_record(6, ca_vec))
    store.insert(_make_gold_record(7, cb_vec))
    store.boost_edges([(UUID(int=6), UUID(int=7))], edge_type="contradicts", delta=[1.0])


def _prime_cache(store: MemoryStore) -> None:
    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc

    graph, assignment, rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, assignment, rc)


def _dispatch_recall(store: MemoryStore, params: dict) -> dict:
    from iai_mcp import core
    return core.dispatch(store, "memory_recall", params)


def _reset_auto_depth() -> None:
    _pipeline_mod._last_recall_latency_ms = 0.0


def _make_recall_params(cue: str, session_id: str = "test-session") -> dict:
    return {
        "cue": cue,
        "session_id": session_id,
        "budget_tokens": 2000,
    }


class _ScanFired(Exception):
    pass


class ScanCounterContext:

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._fired: list[str] = []
        self._patches: list[tuple] = []

    def _make_raiser(self, name: str):
        fired = self._fired
        def _raiser(*args, **kwargs):
            fired.append(name)
            raise _ScanFired(f"full-table scan fired: {name}")
        return _raiser

    def _patch(self, module, attr: str, replacement) -> None:
        orig = getattr(module, attr)
        self._patches.append((module, attr, orig))
        setattr(module, attr, replacement)

    def __enter__(self):
        import iai_mcp.retrieve as _retrieve
        import iai_mcp.community as _community
        import iai_mcp.richclub as _richclub

        self._patch(_retrieve, "build_runtime_graph", self._make_raiser("build_runtime_graph"))

        self._patch(_retrieve, "build_temporal_validity_maps", self._make_raiser("build_temporal_validity_maps"))

        self._patch(_community, "detect_communities", self._make_raiser("detect_communities"))

        self._patch(_richclub, "rich_club_nodes", self._make_raiser("rich_club_nodes"))

        try:
            edges_tbl = self._store.db.open_table("edges")
            self._edges_tbl = edges_tbl
            self._original_to_pandas = edges_tbl.to_pandas
            fired = self._fired

            def _edges_scan_raiser(*args, **kwargs):
                fired.append("edges.to_pandas")
                raise _ScanFired("full edges.to_pandas scan fired")

            edges_tbl.to_pandas = _edges_scan_raiser
        except Exception:
            self._edges_tbl = None
            self._original_to_pandas = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module, attr, orig in reversed(self._patches):
            setattr(module, attr, orig)
        self._patches.clear()

        if self._edges_tbl is not None and self._original_to_pandas is not None:
            self._edges_tbl.to_pandas = self._original_to_pandas

        if self._fired and exc_type is None:
            raise _ScanFired(f"full-table scans fired: {self._fired}")
        return False


@pytest.mark.slow
def test_gate_a_latency_and_scan_counter(tmp_path, monkeypatch, caplog):
    _monkeypatch_env(monkeypatch, tmp_path)

    embedder = Embedder()
    _ = embedder.embed(LEXICAL_GENERIC_CUE)

    results = {}
    scan_counter_results = {}
    ann_path_results = {}

    for n, n_label in [(N_SMALL, "N=1k"), (N_LARGE, "N=10k")]:
        store_path = tmp_path / f"gate-a-store-{n}"
        store_path.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(str(store_path))
        _populate_store(store, n)

        _prime_cache(store)

        monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

        for cue_text, cue_label in [
            (LEXICAL_GENERIC_CUE, "generic"),
            (LEXICAL_SPECIFIC_CUE, "specific"),
        ]:
            cell_label = f"{cue_label}-{n_label}"
            latency_samples: list[float] = []
            scan_fired_trials: list[str] = []
            ann_path_missing_trials: list[int] = []

            for trial_i in range(N_TRIALS):
                _reset_auto_depth()
                params = _make_recall_params(cue_text)
                params["profile_state"] = {
                    "interest_boost": 0.5,
                }

                with ScanCounterContext(store) as _scanner:
                    t0 = time.perf_counter()
                    try:
                        resp = _dispatch_recall(store, params)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    except _ScanFired as sf:
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        scan_fired_trials.append(f"trial-{trial_i}: {sf}")
                        continue

                latency_samples.append(elapsed_ms)

                if not resp.get("ann_path_used", False):
                    ann_path_missing_trials.append(trial_i)

                for record in caplog.records:
                    if "recall_pipeline_fallback" in record.getMessage():
                        ann_path_missing_trials.append(trial_i)
                        break

            scan_counter_results[cell_label] = scan_fired_trials
            ann_path_results[cell_label] = ann_path_missing_trials

            if latency_samples:
                p95_ms = _p95(latency_samples)
                results[cell_label] = p95_ms
            else:
                results[cell_label] = float("inf")

    print("\n" + "=" * 70)
    print("  Gate A Latency Results (p95 ms, ceiling = 1500ms)")
    print("=" * 70)
    for cell_label, p95_ms in results.items():
        status = "PASS" if p95_ms <= LATENCY_CEILING_MS else "FAIL"
        print(f"  {cell_label:<30} {p95_ms:>8.1f} ms  [{status}]")
    print("=" * 70)

    for cell_label, fired in scan_counter_results.items():
        assert not fired, (
            f"Gate A FAIL: full-table scans fired in cell {cell_label!r}: {fired}"
        )

    for cell_label, missing in ann_path_results.items():
        assert not missing, (
            f"Gate A FAIL: ann_path_used was False/missing in trials {missing} "
            f"for cell {cell_label!r}"
        )

    for cell_label, p95_ms in results.items():
        assert p95_ms <= LATENCY_CEILING_MS, (
            f"Gate A FAIL: p95={p95_ms:.1f}ms > {LATENCY_CEILING_MS}ms "
            f"for cell {cell_label!r}"
        )


@pytest.mark.slow
def test_gate_a_within_window_post_write(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "post-write-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))
    _populate_store(store, N_SMALL)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    _prime_cache(store)

    _reset_auto_depth()
    _dispatch_recall(store, _make_recall_params(LEXICAL_GENERIC_CUE))

    new_rec = _make(text="User post-write test record", vec=_random_vec(98765))
    store.insert(new_rec)

    latency_samples = []
    scan_fired = []

    for _ in range(N_TRIALS):
        _reset_auto_depth()
        with ScanCounterContext(store) as _sc:
            t0 = time.perf_counter()
            try:
                resp = _dispatch_recall(store, _make_recall_params(LEXICAL_GENERIC_CUE))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            except _ScanFired as sf:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                scan_fired.append(str(sf))
                continue
        latency_samples.append(elapsed_ms)

    assert not scan_fired, f"Gate A A6 FAIL: full-table scans fired post-write: {scan_fired}"
    assert latency_samples, "No successful trials in post-write cell"
    p95_ms = _p95(latency_samples)
    print(f"\n  Gate A A6 post-write p95: {p95_ms:.1f}ms")
    assert p95_ms <= LATENCY_CEILING_MS, (
        f"Gate A A6 FAIL: post-write p95={p95_ms:.1f}ms > {LATENCY_CEILING_MS}ms"
    )


@pytest.mark.slow
def test_gate_a_boundary_cross_cc_g(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "boundary-cross-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))
    _populate_store(store, N_SMALL)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    _prime_cache(store)

    _reset_auto_depth()
    _dispatch_recall(store, _make_recall_params(LEXICAL_GENERIC_CUE))

    from iai_mcp.runtime_graph_cache import _STALENESS_WINDOW
    for i in range(_STALENESS_WINDOW + 1):
        rec = _make(
            text=f"User boundary-cross test record {i}",
            vec=_random_vec(50000 + i),
        )
        store.insert(rec)

    import iai_mcp.runtime_graph_cache as _rgc
    last_good_calls = {"n": 0, "non_empty": False}
    original_llg = _rgc.load_last_good_structural

    def _spy_llg(store_arg):
        result = original_llg(store_arg)
        last_good_calls["n"] += 1
        if result is not None:
            _assignment, _rc = result
            last_good_calls["non_empty"] = bool(_rc)
        return result

    monkeypatch.setattr(_rgc, "load_last_good_structural", _spy_llg)

    latency_samples = []
    scan_fired = []

    for _ in range(N_TRIALS):
        _reset_auto_depth()
        with ScanCounterContext(store) as _sc:
            t0 = time.perf_counter()
            try:
                resp = _dispatch_recall(store, _make_recall_params(LEXICAL_GENERIC_CUE))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            except _ScanFired as sf:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                scan_fired.append(str(sf))
                continue
        latency_samples.append(elapsed_ms)

    assert not scan_fired, (
        f"Gate A CC-G FAIL: rebuild called on recall path post-boundary: {scan_fired}"
    )
    assert latency_samples, "No successful trials in boundary-cross cell"
    p95_ms = _p95(latency_samples)
    print(f"\n  Gate A CC-G boundary-cross p95: {p95_ms:.1f}ms")
    assert p95_ms <= LATENCY_CEILING_MS, (
        f"Gate A CC-G FAIL: post-boundary p95={p95_ms:.1f}ms > {LATENCY_CEILING_MS}ms"
    )
    assert last_good_calls["n"] >= 1, (
        "load_last_good_structural not called on post-boundary recall — "
        "the case-2 degraded read did not fire"
    )
    assert last_good_calls["non_empty"], (
        "Gate A CC-G FAIL: load_last_good_structural returned empty rich-club — "
        "the GLOBAL rich-club must be non-empty (off-path prime populated it)"
    )


@pytest.mark.slow
def test_gate_a_pending_heavy_cc2_h4(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "pending-heavy-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))
    _populate_store(store, N_SMALL)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    _prime_cache(store)

    pending_count = 100
    now_ts = datetime.now(timezone.utc)
    for i in range(pending_count):
        store.db.insert_pending_row(
            record_id=str(UUID(int=400_000 + i)),
            tier="episodic",
            literal_surface=f"User pending heavy backlog item {i}",
            provenance_json="[]",
            created_at=now_ts.isoformat(),
            updated_at=now_ts.isoformat(),
            tags_json="[]",
        )

    decrypt_counts = {"max_returned": 0}
    original_rpm = store.recent_pending_markers

    def _spy_rpm(n=50):
        results = original_rpm(n=n)
        decrypt_counts["max_returned"] = max(decrypt_counts["max_returned"], len(results))
        return results

    monkeypatch.setattr(store, "recent_pending_markers", _spy_rpm)

    latency_samples = []
    scan_fired = []

    for _ in range(N_TRIALS):
        _reset_auto_depth()
        with ScanCounterContext(store) as _sc:
            t0 = time.perf_counter()
            try:
                resp = _dispatch_recall(store, _make_recall_params(LEXICAL_GENERIC_CUE))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            except _ScanFired as sf:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                scan_fired.append(str(sf))
                continue
        latency_samples.append(elapsed_ms)

    assert not scan_fired, f"Gate A FAIL: scan fired with pending backlog: {scan_fired}"
    assert latency_samples, "No successful trials in pending-heavy cell"
    p95_ms = _p95(latency_samples)
    print(f"\n  Gate A pending-heavy p95: {p95_ms:.1f}ms")
    assert p95_ms <= LATENCY_CEILING_MS, (
        f"Gate A FAIL: pending-heavy p95={p95_ms:.1f}ms > {LATENCY_CEILING_MS}ms"
    )
    assert decrypt_counts["max_returned"] < pending_count, (
        f"recent_pending_markers returned {decrypt_counts['max_returned']} "
        f"rows — equal to or exceeding the full backlog ({pending_count}). "
        "READ A LIMIT must bound the decrypt."
    )


def test_gate_a_cold_no_prime_cc2_h2(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "cold-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))
    _populate_store(store, N_SMALL)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    import iai_mcp.runtime_graph_cache as _rgc

    _rgc.preload_ready.clear()
    cache_file = Path(store_path) / "runtime_graph_cache.json"
    if cache_file.exists():
        cache_file.unlink()

    _reset_auto_depth()
    resp_cold = _dispatch_recall(store, _make_recall_params(LEXICAL_GENERIC_CUE))
    assert resp_cold.get("_source") == "cold-structural-degrade", (
        f"Gate A FAIL: truly-cold recall did not return cold-structural-degrade; "
        f"_source={resp_cold.get('_source')!r}. "
        "A fresh cold daemon must NEVER silently drop the hub-sensitive gold "
        "by serving an unlabelled empty rich-club."
    )

    _prime_cache(store)
    _rgc.preload_ready.set()

    _reset_auto_depth()
    resp_warm = _dispatch_recall(store, _make_recall_params(LEXICAL_GENERIC_CUE))
    assert resp_warm.get("_source") != "cold-structural-degrade", (
        "post-preload recall still returned cold-structural-degrade; "
        "preload_ready is set and cache file exists."
    )
    hub_gold_str = "00000000-0000-0000-0000-000000000001"
    hit_ids = {h["record_id"] for h in resp_warm.get("hits", [])}
    assert hub_gold_str in hit_ids, (
        f"Gate A FAIL: hub-sensitive gold not in hits after preload; "
        f"hit_ids={hit_ids}"
    )


@pytest.mark.slow
def test_gate_a_single_write_2hop_spread_cc_c(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "two-hop-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    rng = np.random.default_rng(RNG_SEED + 1)
    for i in range(50):
        v = rng.random(EMBED_DIM).astype(np.float32)
        v = (v / np.linalg.norm(v)).tolist()
        rec = _make(text=f"User filler spread test {i}", vec=v)
        store.insert(rec)

    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    embedder = Embedder()
    cue_vec = embedder.embed(LEXICAL_GENERIC_CUE)
    cue_arr = np.asarray(cue_vec, dtype=np.float32)
    cue_arr /= np.linalg.norm(cue_arr)

    seed_id = UUID(int=100_001)
    hop1_id = UUID(int=100_002)
    hop2_gold_id = UUID(int=100_003)

    store.insert(_make_gold_record(100_001, list(cue_arr)))

    rng7 = np.random.default_rng(77777)
    hop1_noise = rng7.random(EMBED_DIM).astype(np.float32)
    hop1_noise /= np.linalg.norm(hop1_noise)
    hop1_vec = 0.3 * cue_arr + 0.954 * hop1_noise
    hop1_vec /= np.linalg.norm(hop1_vec)
    store.insert(_make_gold_record(100_002, hop1_vec.tolist()))
    store.boost_edges([(seed_id, hop1_id)], edge_type="hebbian", delta=[3.0])

    rng8 = np.random.default_rng(88888)
    noise2 = rng8.random(EMBED_DIM).astype(np.float32)
    noise2 -= np.dot(noise2, cue_arr) * cue_arr
    noise2 /= np.linalg.norm(noise2)
    hop2_vec = 0.02 * cue_arr + float(np.sqrt(1.0 - 0.02**2)) * noise2
    hop2_vec /= np.linalg.norm(hop2_vec)
    store.insert(_make_gold_record(100_003, hop2_vec.tolist()))
    store.boost_edges([(hop1_id, hop2_gold_id)], edge_type="hebbian", delta=[5.0])
    for extra_n in range(5):
        store.boost_edges([(hop2_gold_id, UUID(int=3100 + extra_n))], edge_type="hebbian", delta=[1.0])

    _prime_cache(store)

    _reset_auto_depth()
    resp = _dispatch_recall(store, _make_recall_params(LEXICAL_GENERIC_CUE))
    hit_ids = {h["record_id"] for h in resp.get("hits", [])}

    assert str(seed_id) in hit_ids, (
        f"CC-C single-write: seed {seed_id} not in hits; hit_ids={hit_ids}"
    )
    assert str(hop1_id) in hit_ids, (
        f"CC-C single-write: hop-1 (edge DST) {hop1_id} not in hits; hit_ids={hit_ids}"
    )
    assert str(hop2_gold_id) in hit_ids, (
        f"CC-C 2-hop spread: hop-2 gold {hop2_gold_id} not in hits; hit_ids={hit_ids}. "
        "2-hop spread must include records two hops from the seed."
    )


@pytest.mark.slow
def test_gate_a_n10k_latency(tmp_path, monkeypatch, caplog):
    _monkeypatch_env(monkeypatch, tmp_path)

    embedder = Embedder()
    _ = embedder.embed(LEXICAL_GENERIC_CUE)

    store_path = tmp_path / "gate-a-n10k-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))
    _populate_store(store, N_LARGE)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    _prime_cache(store)

    results_10k = {}
    scan_fired_10k = {}
    ann_missing_10k = {}

    for cue_text, cue_label in [
        (LEXICAL_GENERIC_CUE, "generic-N=10k"),
        (LEXICAL_SPECIFIC_CUE, "specific-N=10k"),
    ]:
        latency_samples = []
        scan_fired = []
        ann_missing = []

        for trial_i in range(N_TRIALS):
            _reset_auto_depth()
            params = _make_recall_params(cue_text)
            params["profile_state"] = {"interest_boost": 0.5}

            with ScanCounterContext(store) as _sc:
                t0 = time.perf_counter()
                try:
                    resp = _dispatch_recall(store, params)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                except _ScanFired as sf:
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    scan_fired.append(f"trial-{trial_i}: {sf}")
                    continue

            latency_samples.append(elapsed_ms)
            if not resp.get("ann_path_used", False):
                ann_missing.append(trial_i)

        scan_fired_10k[cue_label] = scan_fired
        ann_missing_10k[cue_label] = ann_missing
        results_10k[cue_label] = _p95(latency_samples) if latency_samples else float("inf")

    print("\n" + "=" * 60)
    print("  Gate A N=10k Results")
    print("=" * 60)
    for lbl, p95_ms in results_10k.items():
        status = "PASS" if p95_ms <= LATENCY_CEILING_MS else "FAIL"
        print(f"  {lbl:<36} {p95_ms:>8.1f} ms  [{status}]")
    print("=" * 60)

    for lbl, fired in scan_fired_10k.items():
        assert not fired, f"Gate A N=10k FAIL: scans fired for {lbl!r}: {fired}"

    for lbl, missing in ann_missing_10k.items():
        assert not missing, f"Gate A N=10k FAIL: ann_path_used missing for {lbl!r} trials={missing}"

    for lbl, p95_ms in results_10k.items():
        assert p95_ms <= LATENCY_CEILING_MS, (
            f"Gate A FAIL: p95={p95_ms:.1f}ms > {LATENCY_CEILING_MS}ms for {lbl!r}"
        )


def test_gate_a_client_recall_semantic_warm(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "client-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))
    _populate_store(store, N_SMALL)

    import iai_mcp.embed as _embed_mod

    def _no_construct_funnel(_store):
        raise RuntimeError("hermetic: no embedder construct in this latency gate")

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _no_construct_funnel)

    from iai_mcp.semantic_recall import recall_semantic_warm

    for cue_text in [LEXICAL_GENERIC_CUE, LEXICAL_SPECIFIC_CUE]:
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = recall_semantic_warm(str(store_path), cue_text, n=10)
            samples.append((time.perf_counter() - t0) * 1000.0)
        p95_ms = _p95(samples)
        print(f"\n  A5 client recall_semantic_warm '{cue_text}' p95={p95_ms:.1f}ms")
        assert p95_ms <= LATENCY_CEILING_MS, (
            f"Gate A A5 FAIL: client p95={p95_ms:.1f}ms > {LATENCY_CEILING_MS}ms"
        )
