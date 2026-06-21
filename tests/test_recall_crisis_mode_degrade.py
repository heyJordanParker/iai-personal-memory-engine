"""Hermetic test for the crisis_mode honest-degrade guard in
``core.dispatch(method="memory_recall", ...)``.

When the lifecycle_state's ``crisis_mode`` flag is True (the scheduler is
looping a deferred step and cannot advance), the daemon's warm recall
path returns stale schema-dominated results that violate the
hippocampus-always-available invariant. The guard short-circuits to a
degraded response with score 0.0 and a ``_degraded: True`` flag so the
wrapper falls back to bank-recall via its existing socket-unreachable
code path.
"""
from __future__ import annotations

import pytest


def _make_store():
    """Open a hermetic in-memory MemoryStore. We only need it as the dispatch
    `store` argument; the crisis_mode short-circuit returns BEFORE any store
    access, so a fully-stocked store is unnecessary."""
    from iai_mcp.store import MemoryStore

    return MemoryStore()


def test_memory_recall_degraded_when_crisis_mode_true(
    monkeypatch: pytest.MonkeyPatch,
):
    """When lifecycle_state.load_state() returns crisis_mode=True, the
    dispatch short-circuits to a degraded response — no embedder, no warm
    path."""
    from iai_mcp.core import dispatch

    monkeypatch.setattr(
        "iai_mcp.lifecycle_state.load_state",
        lambda *_a, **_kw: {"crisis_mode": True, "current_state": "SLEEP"},
    )

    store = _make_store()
    resp = dispatch(store, "memory_recall", {"cue": "Hello there"})

    assert resp == {
        "hits": [],
        "_degraded": True,
        "_reason": "daemon_consolidation_stuck",
    }


def test_memory_recall_warm_path_when_crisis_mode_false(
    monkeypatch: pytest.MonkeyPatch,
):
    """When crisis_mode=False, the guard does not interfere and the dispatch
    proceeds to the normal recall path. We assert on the absence of the
    degraded sentinel rather than the full response shape — that way the
    test stays focused on what changed."""
    from iai_mcp.core import dispatch

    monkeypatch.setattr(
        "iai_mcp.lifecycle_state.load_state",
        lambda *_a, **_kw: {"crisis_mode": False, "current_state": "WAKE"},
    )

    store = _make_store()
    resp = dispatch(store, "memory_recall", {"cue": "Hello there"})

    # The non-degraded path returns a dict that does NOT carry the degrade
    # sentinel. The exact contents depend on the warm path (or fallback), so
    # we only check the absence of the crisis-mode short-circuit.
    assert resp.get("_degraded") is not True or resp.get("_reason") != "daemon_consolidation_stuck"


def test_memory_recall_guards_against_load_state_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """If daemon_state.load_state() itself raises, the guard must NOT crash
    recall — it falls through to the warm path (the constitutional
    always-available invariant: the guard cannot itself break recall)."""
    from iai_mcp.core import dispatch

    def _boom(*_a, **_kw):
        raise RuntimeError("synthetic lifecycle_state read failure")

    monkeypatch.setattr("iai_mcp.lifecycle_state.load_state", _boom)

    store = _make_store()
    resp = dispatch(store, "memory_recall", {"cue": "Hello there"})

    # Did not short-circuit to degraded — fell through to warm path.
    assert resp.get("_reason") != "daemon_consolidation_stuck"
