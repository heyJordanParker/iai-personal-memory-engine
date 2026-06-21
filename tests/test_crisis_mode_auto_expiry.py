"""Hermetic tests for the crisis_mode 72h auto-expiry predicate, its set-site
stamping invariant, and the recall-guard wrong-file-read regression.

Three classes:

- ``TestPredicate`` pins the pure predicate axes (legacy-load backfill,
  just-over-threshold, just-under-threshold).
- ``TestSetSiteStamps`` pins the invariant that every write path which flips
  ``crisis_mode`` also writes ``crisis_mode_since_ts`` atomically in the same
  ``save_state`` call (True stamps now, False clears to None).
- ``TestR1RecallGuardActivation`` pins the recall honest-degrade guard against
  re-introduction of the wrong-file-read regression: the guard must read from
  ``iai_mcp.lifecycle_state.load_state`` (the file that actually carries the
  ``crisis_mode`` flag), not from a stale loose-dict store that never carried
  the key.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from iai_mcp import daemon
from iai_mcp.lifecycle_state import (
    LifecycleState,
    default_state,
    load_state,
    save_state,
)


NOW = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)
THRESHOLD_SEC = 259_200  # 72h


def _state_with_crisis(
    crisis_mode: bool,
    since_ts: str | None,
    current_state: str = LifecycleState.SLEEP.value,
) -> dict:
    rec = default_state()
    rec["current_state"] = current_state
    rec["crisis_mode"] = crisis_mode
    if since_ts is not None:
        rec["crisis_mode_since_ts"] = since_ts
    return rec


class TestPredicate:

    def test_legacy_load_no_since_ts_returns_backfill(self):
        # Areg's live wedge shape: crisis_mode=True, no since_ts. Predicate
        # backfills since_ts to first-observation time and does NOT emit.
        state = _state_with_crisis(True, None)
        expired, ctx = daemon._check_crisis_mode_expiry(state, NOW)
        assert expired is False
        assert "backfilled_since_ts" in ctx
        assert ctx["backfilled_since_ts"] == NOW.isoformat()

    def test_expiry_fires_just_over_threshold(self):
        since = (NOW - timedelta(seconds=THRESHOLD_SEC + 1)).isoformat()
        state = _state_with_crisis(True, since)
        expired, ctx = daemon._check_crisis_mode_expiry(state, NOW)
        assert expired is True
        assert ctx["expired_after_sec"] == THRESHOLD_SEC + 1
        assert ctx["since_ts"] == since
        assert ctx["backfilled"] is False

    def test_expiry_does_not_fire_just_under_threshold(self):
        since = (NOW - timedelta(seconds=THRESHOLD_SEC - 1)).isoformat()
        state = _state_with_crisis(True, since)
        expired, ctx = daemon._check_crisis_mode_expiry(state, NOW)
        assert expired is False
        assert ctx == {}


class TestSetSiteStamps:

    def test_s2_set_crisis_mode_true_stamps_since_ts(self, tmp_path):
        # S2Coordinator.set_crisis_mode(True, reason) must stamp
        # crisis_mode_since_ts in the SAME save_state write that flips
        # crisis_mode=True — no two-write race window.
        from iai_mcp.s2_coordinator import S2Coordinator

        state_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / "legacy_state.json"
        legacy_path.write_text(json.dumps({"state": "WAKE"}))
        save_state(default_state(), state_path)
        coord = S2Coordinator(
            store=None,
            state_path=state_path,
            legacy_path=legacy_path,
        )

        asyncio.run(coord.set_crisis_mode(True, reason="test_breach"))
        rec = load_state(state_path)
        assert rec["crisis_mode"] is True
        stamped = rec.get("crisis_mode_since_ts")
        assert stamped is not None
        parsed = datetime.fromisoformat(stamped)
        assert parsed.tzinfo is not None

    def test_s2_set_crisis_mode_false_clears_since_ts(self, tmp_path):
        from iai_mcp.s2_coordinator import S2Coordinator

        state_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / "legacy_state.json"
        legacy_path.write_text(json.dumps({"state": "WAKE"}))
        seed = default_state()
        seed["crisis_mode"] = True
        seed["crisis_mode_since_ts"] = "2026-06-01T00:00:00+00:00"
        save_state(seed, state_path)
        coord = S2Coordinator(
            store=None,
            state_path=state_path,
            legacy_path=legacy_path,
        )

        asyncio.run(coord.set_crisis_mode(False, reason="test_clear"))
        rec_after = load_state(state_path)
        assert rec_after["crisis_mode"] is False
        assert rec_after.get("crisis_mode_since_ts") is None


class TestR1RecallGuardActivation:

    def test_recall_guard_reads_lifecycle_state_not_daemon_state(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        # The honest-degrade guard reads from
        # iai_mcp.lifecycle_state.load_state (the file that actually carries
        # the crisis_mode flag). If a future refactor re-introduces the
        # wrong-file-read regression, this test fails — the monkeypatch of
        # lifecycle_state.load_state has no effect on the guard's response.
        from iai_mcp.core import dispatch
        from iai_mcp.store import MemoryStore

        monkeypatch.setattr(
            "iai_mcp.lifecycle_state.load_state",
            lambda *_a, **_kw: {
                "crisis_mode": True,
                "current_state": "SLEEP",
            },
        )

        store = MemoryStore()
        resp = dispatch(store, "memory_recall", {"cue": "test"})

        assert resp == {
            "hits": [],
            "_degraded": True,
            "_reason": "daemon_consolidation_stuck",
        }
