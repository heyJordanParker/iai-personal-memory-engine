"""— async provenance write queue.

Moves provenance writes off the recall critical path via a daemon-thread
queue so pipeline_recall returns before append_provenance_batch runs.

All 6 tests below MUST FAIL on first run (RED) — the module
`iai_mcp.provenance_queue` and the `MemoryStore.queue_provenance_batch`
entry point do not exist yet.

Fence:
- preserved (every recall still appends a provenance entry;
  writes are async but not dropped).
- Provenance-write failure never blocks recall.
- C3/C6: no external deps, pure stdlib.
"""
from __future__ import annotations

import time
from uuid import UUID

import pytest

from iai_mcp.store import MemoryStore
from tests.test_store import _make


# --------------------------------------------------------------------- P1, P2, P5

def test_enqueue_fast(tmp_path):
    """P1: ProvenanceWriteQueue.enqueue returns in <= 2ms even when worker is slowed.

    We artificially slow the underlying store.append_provenance_batch so that
    each flush takes 200ms; enqueue must NOT wait for it.
    """
    from iai_mcp.provenance_queue import ProvenanceWriteQueue

    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)

    # Wrap append_provenance_batch to be slow.
    real_batch = store.append_provenance_batch

    def slow_batch(pairs, records_cache=None):
        time.sleep(0.2)
        return real_batch(pairs, records_cache=records_cache)

    store.append_provenance_batch = slow_batch  # type: ignore[method-assign]

    q = ProvenanceWriteQueue(store, coalesce_ms=50)
    q.start()
    try:
        t0 = time.perf_counter()
        q.enqueue([(r.id, {"ts": "x", "cue": "c", "session_id": "s"})])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms <= 20.0, f"enqueue took {elapsed_ms:.1f}ms (target <=2ms, headroom <=20ms)"
    finally:
        q.stop()


def test_flush_drains(tmp_path):
    """P2: worker drains all pending pairs within 500ms after.flush()."""
    from iai_mcp.provenance_queue import ProvenanceWriteQueue

    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)

    q = ProvenanceWriteQueue(store, coalesce_ms=50)
    q.start()
    try:
        for i in range(10):
            q.enqueue([(r.id, {"ts": f"t{i}", "cue": f"c{i}", "session_id": "s"})])
        t0 = time.perf_counter()
        q.flush(timeout=2.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms <= 500.0, f"flush took {elapsed_ms:.1f}ms (target <=500ms)"
    finally:
        q.stop()

    # All 10 entries should now be durable.
    got = store.get(r.id)
    assert got is not None
    assert len(got.provenance) == 10


def test_atexit_flush(tmp_path, monkeypatch):
    """P5: atexit hook flushes the queue on interpreter shutdown.

    We simulate by registering a queue, capturing the atexit handler
    it installs, calling it manually, and verifying the store is
    consistent afterward.
    """
    import atexit as _atexit
    from iai_mcp.provenance_queue import ProvenanceWriteQueue

    captured: list = []

    def _fake_register(fn, *a, **kw):
        captured.append(fn)
        return fn

    monkeypatch.setattr(_atexit, "register", _fake_register)

    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)

    q = ProvenanceWriteQueue(store, coalesce_ms=50)
    q.start()
    q.enqueue([(r.id, {"ts": "t", "cue": "c", "session_id": "s"})])

    # The atexit handler should have been registered during start().
    assert captured, "ProvenanceWriteQueue.start() must register atexit flush"
    # Invoke the registered handler — it must drain + not raise.
    captured[0]()

    # After the handler runs, the provenance entry must be durable.
    got = store.get(r.id)
    assert got is not None
    assert len(got.provenance) == 1
    q.stop()


# ---------------------------------------------------------------------- P3, P4, P6

def test_pipeline_recall_does_not_block_on_merge_insert(tmp_path, monkeypatch):
    """P3: pipeline_recall latency does NOT include merge_insert when queue is enabled.

    Setup: make append_provenance_batch artificially slow (500ms). With the
    queue enabled, pipeline_recall should return well under 400ms (the write
    is handed off). Without the queue it would be >=500ms.
    """
    from iai_mcp.core import dispatch

    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)

    # Warm call first — initialises embedders, opens tables, etc. so the
    # timed call below only measures the hot path. Conftest autoflush is
    # still active here so the record lands in the store before the slow
    # mock is installed.
    dispatch(
        store, "memory_recall",
        {"cue": "warmup", "session_id": "s0", "cue_embedding": r.embedding},
    )

    # Disable conftest defer_provenance autoflush before installing the slow
    # mock. flush_deferred_provenance calls append_provenance_batch directly
    # (bypasses the async queue), so leaving autoflush active would cause the
    # pipeline's deferred write to hit the slow mock during the timed call.
    # Production code has no such autoflush; this restores production semantics
    # for the latency measurement.
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    # Enable the provenance queue.
    store.enable_provenance_queue(coalesce_ms=50)
    try:
        # Slow the actual batch write.
        real_batch = store.append_provenance_batch

        def slow_batch(pairs, records_cache=None):
            time.sleep(0.5)  # 500ms slow write
            return real_batch(pairs, records_cache=records_cache)

        store.append_provenance_batch = slow_batch  # type: ignore[method-assign]

        t0 = time.perf_counter()
        dispatch(
            store,
            "memory_recall",
            {"cue": "q", "session_id": "s1", "cue_embedding": r.embedding},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Target: the 500ms slow write is off-path; the queue hands off so
        # pipeline_recall returns well before the write completes. We give
        # a very generous budget (400ms) to accommodate CI-hardware jitter
        # while still proving the write is NOT inline (inline would be
        # >= 500ms).
        assert elapsed_ms < 400.0, (
            f"pipeline_recall blocked on merge_insert: {elapsed_ms:.1f}ms "
            f"(queue should hand off; target <400ms given 500ms slow write)"
        )
    finally:
        store.disable_provenance_queue()


def test_mem05_preserved_after_drain(tmp_path):
    """P4: after flush, store reflects all enqueued provenance entries in insertion order."""
    from iai_mcp.core import dispatch

    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)

    store.enable_provenance_queue(coalesce_ms=50)
    try:
        dispatch(store, "memory_recall",
                 {"cue": "first", "session_id": "s1", "cue_embedding": r.embedding})
        dispatch(store, "memory_recall",
                 {"cue": "second", "session_id": "s2", "cue_embedding": r.embedding})
        dispatch(store, "memory_recall",
                 {"cue": "third", "session_id": "s3", "cue_embedding": r.embedding})
        # Drain.
        store._provenance_queue.flush(timeout=2.0)  # type: ignore[attr-defined]
    finally:
        store.disable_provenance_queue()

    got = store.get(r.id)
    assert got is not None
    assert len(got.provenance) == 3
    cues = [p["cue"] for p in got.provenance]
    assert cues == ["first", "second", "third"], f"order violated: {cues}"


def test_overflow_spill_round_trip(tmp_path, monkeypatch):
    """W1 /: when _q is full, batches spill to
    ~/.iai-mcp/.provenance-overflow/ instead of dropping. The worker
    re-enqueues spilled batches on idle. holds under overload."""
    import threading
    from iai_mcp.provenance_queue import ProvenanceWriteQueue

    # Init store BEFORE HOME redirect (keyring uses real HOME).
    store = MemoryStore(path=tmp_path / "store")
    r = _make()
    store.insert(r)

    monkeypatch.setenv("HOME", str(tmp_path))

    # Throttle the worker's batch flush so _q fills up.
    flushed_pairs: list = []
    flush_release = threading.Event()
    flush_release.clear()
    real_batch = store.append_provenance_batch

    def slow_batch(pairs, records_cache=None):
        # Block until the test releases; then call the real batch.
        flush_release.wait(timeout=10.0)
        flushed_pairs.extend(pairs)
        return real_batch(pairs, records_cache=records_cache)

    store.append_provenance_batch = slow_batch  # type: ignore[method-assign]

    # Tiny queue so we hit overflow fast.
    q = ProvenanceWriteQueue(store, coalesce_ms=10, max_queue_size=2,
                             max_batch_pairs=1)
    q.start()
    try:
        # Push 5 single-pair batches. The worker will pull the first,
        # block on slow_batch; _q at maxsize=2 fills with two more;
        # the remaining 2 must spill.
        for i in range(5):
            q.enqueue([(r.id, {"ts": f"t{i}", "cue": f"c{i}",
                               "session_id": "sov"})])
        # Give the spill writes a moment to land on disk.
        time.sleep(0.1)
        overflow_dir = tmp_path / ".iai-mcp" / ".provenance-overflow"
        spilled_before_release = list(overflow_dir.glob("*.jsonl"))
        assert len(spilled_before_release) >= 1, (
            f"expected at least 1 spilled file, got {len(spilled_before_release)} "
            f"(overflow dir contents: {list(overflow_dir.iterdir()) if overflow_dir.exists() else 'absent'})"
        )
        # Release the worker — it drains _q first, then on idle ticks
        # picks up the overflow dir and re-enqueues spilled batches.
        flush_release.set()
        # Wait for the queue idle-poll cycle (5s) plus headroom — but
        # the immediate flush() pushes a sentinel that wakes it sooner.
        # We poll until overflow dir is empty OR timeout.
        deadline = time.time() + 12.0
        while time.time() < deadline:
            if not list(overflow_dir.glob("*.jsonl")):
                break
            time.sleep(0.2)
        # Final flush + assertions.
        q.flush(timeout=2.0)
    finally:
        q.stop()

    # All 5 cues reached append_provenance_batch exactly once.
    flushed_cues = [p[1]["cue"] for p in flushed_pairs]
    assert sorted(flushed_cues) == [f"c{i}" for i in range(5)], (
        f"expected all 5 cues flushed exactly once; got {sorted(flushed_cues)}"
    )
    # Spill dir is empty (every file unlinked after re-enqueue + flush).
    overflow_dir = tmp_path / ".iai-mcp" / ".provenance-overflow"
    assert list(overflow_dir.glob("*.jsonl")) == [], (
        f"spill dir should be empty after drain; got {list(overflow_dir.iterdir())}"
    )


def test_overflow_dir_lazy_create(tmp_path, monkeypatch):
    """W1 /: the overflow dir is created only on the first spill.
    Cold start with no overload must NOT create it."""
    from iai_mcp.provenance_queue import ProvenanceWriteQueue

    # Build the store BEFORE redirecting HOME so MemoryStore init
    # uses the real keyring + env, then redirect HOME so the
    # overflow dir under HOME points to tmp.
    store = MemoryStore(path=tmp_path / "store")
    r = _make()
    store.insert(r)

    monkeypatch.setenv("HOME", str(tmp_path))

    q = ProvenanceWriteQueue(store, coalesce_ms=50)
    q.start()
    try:
        q.enqueue([(r.id, {"ts": "t", "cue": "c", "session_id": "s"})])
        q.flush(timeout=2.0)
    finally:
        q.stop()

    overflow_dir = tmp_path / ".iai-mcp" / ".provenance-overflow"
    assert not overflow_dir.exists(), (
        "overflow dir must not be created when no spill happens"
    )


def test_overflow_malformed_spill_file_quarantined(tmp_path, monkeypatch):
    """W1 /: a malformed spill file is renamed.failed-<ts>.jsonl
    and does NOT block the drain loop."""
    from iai_mcp.provenance_queue import ProvenanceWriteQueue

    # Init store BEFORE HOME redirect (keyring uses real HOME).
    store = MemoryStore(path=tmp_path / "store")

    monkeypatch.setenv("HOME", str(tmp_path))
    overflow_dir = tmp_path / ".iai-mcp" / ".provenance-overflow"
    overflow_dir.mkdir(parents=True)
    bad_file = overflow_dir / "bad.jsonl"
    bad_file.write_text("this is not valid json at all\n")

    q = ProvenanceWriteQueue(store, coalesce_ms=50)
    q.start()
    try:
        # Trigger an idle drain by waiting past the idle-poll boundary
        # (5s WORKER_IDLE_POLL_S + headroom).
        time.sleep(6.5)
    finally:
        q.stop()

    # Malformed file moved to.failed-*.jsonl
    assert not bad_file.exists()
    failed_files = list(overflow_dir.glob("*.failed-*.jsonl"))
    assert len(failed_files) == 1, (
        f"expected 1 failed-quarantined file; got {len(failed_files)} "
        f"(overflow dir contents: {list(overflow_dir.iterdir())})"
    )


def test_queue_disabled_falls_back_to_sync(tmp_path):
    """P6: store.enable_provenance_queue() toggles behaviour — when disabled,
    pipeline_recall falls back to the sync append_provenance_batch path.

    Verify by monkey-patching append_provenance_batch to record calls and
    confirming it was called synchronously (on the caller thread) before
    dispatch returns.
    """
    import threading
    from iai_mcp.core import dispatch

    store = MemoryStore(path=tmp_path)
    r = _make()
    store.insert(r)

    # Queue NOT enabled.
    assert getattr(store, "_provenance_queue", None) is None

    call_threads: list[int] = []
    real_batch = store.append_provenance_batch

    def tracking_batch(pairs, records_cache=None):
        call_threads.append(threading.get_ident())
        return real_batch(pairs, records_cache=records_cache)

    store.append_provenance_batch = tracking_batch  # type: ignore[method-assign]

    main_ident = threading.get_ident()
    dispatch(store, "memory_recall",
             {"cue": "q", "session_id": "s1", "cue_embedding": r.embedding})

    # Batch was called on the main thread (sync fallback).
    assert call_threads, "append_provenance_batch not called in sync fallback"
    assert call_threads[0] == main_ident, (
        f"sync fallback ran on thread {call_threads[0]!r}, expected main {main_ident!r}"
    )

    got = store.get(r.id)
    assert got is not None
    assert len(got.provenance) == 1
