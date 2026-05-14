"""Tests for the MCP-recent-activity yield branch in _tick_body.

Between REM cycles, _tick_body must:
  - break the loop and emit daemon_yielded(reason=mcp_recent_activity)
    when mcp_socket reports active connections OR last_activity_ts
    within INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC (30 s).
  - NOT take that branch when mcp_socket is idle (no active connections
    AND last_activity_ts older than the window).
  - Be a no-op when mcp_socket is None (legacy callers).

Mirrors the fixture style of tests/test_daemon_tick_flags.py: tmp_path
isolated store + lock + seeded record so the empty-store shortcut does
not fire, plus a fast monkeypatched run_rem_cycle stub.
"""
from __future__ import annotations

import asyncio
import time
import types
from datetime import datetime, timezone
from uuid import uuid4

import pytest


@pytest.fixture
def tick_env(tmp_path, monkeypatch):
    """Isolated store + lock + seeded record (mirrors test_daemon_tick_flags)."""
    from iai_mcp import concurrency, daemon_state
    from iai_mcp.concurrency import ProcessLock
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord

    lock_path = tmp_path / ".lock"
    state_path = tmp_path / ".daemon-state.json"

    monkeypatch.setattr(concurrency, "LOCK_PATH", lock_path)
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")

    store = MemoryStore()

    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="seed record so the store is not empty",
        aaak_index="",
        embedding=[0.0] * store.embed_dim,
        community_id=None,
        centrality=0.0,
        detail_level=1,
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
    store.insert(rec)

    lock = ProcessLock(lock_path)
    yield store, lock, tmp_path
    try:
        lock.release()
    except Exception:
        pass
    lock.close()


def _window_covering_now() -> list[int]:
    """A quiet_window [start_bucket, duration] that contains the current local time."""
    from iai_mcp.tz import load_user_tz
    tz = load_user_tz()
    now_local = datetime.now(timezone.utc).astimezone(tz)
    cur_bucket = (now_local.hour * 60 + now_local.minute) // 30
    start = (cur_bucket - 2) % 48
    return [start, 8]


def _tracking_rem_factory(cycle_calls: list[int]):
    async def _tracking_rem(
        store, cycle_num, total_cycles, session_id, *, is_last, claude_enabled,
    ):
        cycle_calls.append(cycle_num)
        await asyncio.sleep(0.005)
        return {
            "cycle": cycle_num,
            "summaries_created": 0,
            "schemas_induced": 0,
            "schema_candidates": 0,
            "claude_call_used": False,
            "main_insight_text": None,
            "timed_out": False,
        }
    return _tracking_rem


def test_tick_body_yields_when_mcp_recently_active(tick_env, monkeypatch):
    """Recent MCP activity (5 s ago, within 30 s window) must break the REM loop
    after the first cycle and emit daemon_yielded(reason=mcp_recent_activity).
    """
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.events import query_events

    store, lock, _ = tick_env

    state = {
        "fsm_state": "WAKE",
        "quiet_window": _window_covering_now(),
        "rem_cycle_count": 5,
    }

    cycle_calls: list[int] = []
    monkeypatch.setattr(
        daemon_mod, "run_rem_cycle", _tracking_rem_factory(cycle_calls)
    )
    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    mcp_socket = types.SimpleNamespace(
        active_connections=0,
        last_activity_ts=time.monotonic() - 5.0,
    )

    asyncio.run(daemon_mod._tick_body(store, lock, state, mcp_socket=mcp_socket))

    # Loop broke after the first cycle.
    assert cycle_calls == [1], (
        f"loop must break after cycle 1 when MCP recently active, got {cycle_calls}"
    )
    # daemon_yielded event recorded with the new reason.
    yield_events = query_events(store, kind="daemon_yielded", limit=10)
    reasons = [e["data"].get("reason") for e in yield_events]
    assert "mcp_recent_activity" in reasons, (
        f"expected mcp_recent_activity in daemon_yielded reasons, got {reasons}"
    )
    # FSM returned cleanly to WAKE.
    assert state["fsm_state"] == "WAKE"


def test_tick_body_does_not_yield_when_mcp_idle(tick_env, monkeypatch):
    """Idle MCP socket (no active connections, last activity 10 min ago) must
    NOT trigger the new branch. The full REM loop runs to completion.
    """
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.events import query_events

    store, lock, _ = tick_env

    state = {
        "fsm_state": "WAKE",
        "quiet_window": _window_covering_now(),
        "rem_cycle_count": 3,
    }

    cycle_calls: list[int] = []
    monkeypatch.setattr(
        daemon_mod, "run_rem_cycle", _tracking_rem_factory(cycle_calls)
    )
    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    mcp_socket = types.SimpleNamespace(
        active_connections=0,
        last_activity_ts=time.monotonic() - 600.0,
    )

    asyncio.run(daemon_mod._tick_body(store, lock, state, mcp_socket=mcp_socket))

    # All 3 cycles ran (no early break).
    assert cycle_calls == [1, 2, 3], (
        f"loop must run all 3 cycles when MCP idle, got {cycle_calls}"
    )
    # The new branch did NOT emit its reason.
    yield_events = query_events(store, kind="daemon_yielded", limit=10)
    reasons = [e["data"].get("reason") for e in yield_events]
    assert "mcp_recent_activity" not in reasons, (
        f"mcp_recent_activity must NOT appear when idle, got {reasons}"
    )
    # FSM returned cleanly to WAKE.
    assert state["fsm_state"] == "WAKE"
