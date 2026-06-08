"""Tests for daemon FD-limit hardening and legacy FSM-state mirror.

FD-limit tests:
  - _raise_fd_limit clamps the soft limit correctly (never above hard,
    never below current soft, respects a configured floor).
  - _raise_fd_limit swallows a failing setrlimit call without propagating.
  - The launchd plist template carries SoftResourceLimits/NumberOfFiles
    and the value survives cli._render_launchd_plist (the render does NOT
    strip unknown keys).

FSM-mirror tests:
  - After canonical transitions (WAKE→DROWSY→SLEEP) the legacy mirror is
    written in lock-step, so reconcile_fsm_state reports drift=False.
  - A genuinely-divergent pair (e.g. canonical=WAKE, legacy=SLEEP with no
    write-path that could produce it) is still reported as drift=True.
"""
from __future__ import annotations

import json
import resource
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iai_mcp.daemon import _raise_fd_limit
from iai_mcp.fsm_reconcile import _CANONICAL_TO_LEGACY, reconcile_fsm_state
from iai_mcp.s2_coordinator import S2Coordinator


# ---------------------------------------------------------------------------
# FD-limit tests
# ---------------------------------------------------------------------------


class TestRaiseFdLimitClampsToHard:
    """_raise_fd_limit must clamp target to [current_soft, hard]."""

    def test_raises_low_soft_to_floor(self):
        """When soft < FLOOR, target becomes FLOOR (clamped to hard)."""
        fake_soft = 128
        fake_hard = 65536
        calls = []

        def fake_setrlimit(res, limits):
            calls.append((res, limits))

        with (
            patch("resource.getrlimit", return_value=(fake_soft, fake_hard)),
            patch("resource.setrlimit", side_effect=fake_setrlimit),
        ):
            _raise_fd_limit()

        assert len(calls) == 1
        _res, (new_soft, new_hard) = calls[0]
        assert new_soft >= 8192  # at least the default floor
        assert new_soft <= fake_hard  # never above hard
        assert new_hard == fake_hard  # hard is passed through unchanged

    def test_does_not_lower_already_high_soft(self):
        """When current soft >= FLOOR, setrlimit is not called (no-op)."""
        fake_soft = 32768
        fake_hard = 65536
        calls = []

        def fake_setrlimit(res, limits):
            calls.append((res, limits))

        with (
            patch("resource.getrlimit", return_value=(fake_soft, fake_hard)),
            patch("resource.setrlimit", side_effect=fake_setrlimit),
        ):
            _raise_fd_limit()

        # Already above the floor — no setrlimit call is needed
        for _res, (new_soft, _) in calls:
            assert new_soft >= fake_soft  # must never lower the limit

    def test_clamped_by_hard_limit(self):
        """When hard < FLOOR, target must not exceed hard."""
        fake_soft = 64
        fake_hard = 256  # below the 8192 floor
        calls = []

        def fake_setrlimit(res, limits):
            calls.append((res, limits))

        with (
            patch("resource.getrlimit", return_value=(fake_soft, fake_hard)),
            patch("resource.setrlimit", side_effect=fake_setrlimit),
        ):
            _raise_fd_limit()

        assert len(calls) == 1
        _res, (new_soft, new_hard) = calls[0]
        assert new_soft <= fake_hard  # never above hard

    def test_infinity_hard_does_not_request_huge_value(self):
        """When hard == RLIM_INFINITY, target must be the floor, not infinity."""
        fake_soft = 128
        fake_hard = resource.RLIM_INFINITY
        calls = []

        def fake_setrlimit(res, limits):
            calls.append((res, limits))

        with (
            patch("resource.getrlimit", return_value=(fake_soft, fake_hard)),
            patch("resource.setrlimit", side_effect=fake_setrlimit),
        ):
            _raise_fd_limit()

        assert len(calls) == 1
        _res, (new_soft, new_hard) = calls[0]
        # Target must be the floor (or current soft if higher), NOT infinity
        assert new_soft != resource.RLIM_INFINITY
        assert new_soft >= 8192
        # Hard must be passed through unchanged (infinity is valid as the
        # second element of the pair even on macOS).
        assert new_hard == fake_hard

    def test_setrlimit_failure_is_swallowed(self):
        """A failing setrlimit must not propagate — boot proceeds."""
        with (
            patch("resource.getrlimit", return_value=(64, 65536)),
            patch("resource.setrlimit", side_effect=OSError("permission denied")),
        ):
            # Must not raise
            _raise_fd_limit()

    def test_setrlimit_value_error_is_swallowed(self):
        """A ValueError from setrlimit (invalid params) must not propagate."""
        with (
            patch("resource.getrlimit", return_value=(64, 65536)),
            patch("resource.setrlimit", side_effect=ValueError("bad value")),
        ):
            _raise_fd_limit()

    def test_env_tunable_floor(self, monkeypatch):
        """IAI_MCP_DAEMON_NOFILE_FLOOR overrides the default floor."""
        monkeypatch.setenv("IAI_MCP_DAEMON_NOFILE_FLOOR", "16384")
        fake_soft = 64
        fake_hard = 65536
        calls = []

        def fake_setrlimit(res, limits):
            calls.append((res, limits))

        with (
            patch("resource.getrlimit", return_value=(fake_soft, fake_hard)),
            patch("resource.setrlimit", side_effect=fake_setrlimit),
        ):
            _raise_fd_limit()

        assert len(calls) == 1
        _res, (new_soft, _new_hard) = calls[0]
        assert new_soft >= 16384


class TestPlistRendersFdFloor:
    """The plist template must carry SoftResourceLimits/NumberOfFiles and
    that key must survive cli._render_launchd_plist."""

    def test_plist_template_contains_fd_key(self):
        from iai_mcp.cli import _launchd_template

        text = _launchd_template().read_text()
        assert "SoftResourceLimits" in text
        assert "NumberOfFiles" in text

    def test_rendered_plist_contains_fd_floor(self, tmp_path, monkeypatch):
        """The rendered output must preserve SoftResourceLimits/NumberOfFiles."""
        import importlib
        import os

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USER", "testuser")

        # Re-import cli so HOME/USER are picked up in the render
        from iai_mcp.cli import _render_launchd_plist

        rendered = _render_launchd_plist()
        assert "SoftResourceLimits" in rendered
        assert "NumberOfFiles" in rendered

        # Confirm the integer value in the rendered plist is >= 8192
        import defusedxml.ElementTree as ET

        root = ET.fromstring(rendered)
        top_dict = root.find("dict")
        assert top_dict is not None

        keys = [el.text for el in top_dict.findall("key")]
        assert "SoftResourceLimits" in keys

        # Find the SoftResourceLimits sub-dict
        idx = list(top_dict).index(
            next(el for el in top_dict if el.tag == "key" and el.text == "SoftResourceLimits")
        )
        sub_dict = list(top_dict)[idx + 1]
        assert sub_dict.tag == "dict"

        num_el = sub_dict.find("integer")
        assert num_el is not None
        assert int(num_el.text) >= 8192

    def test_rendered_plist_preserves_python_path(self, tmp_path, monkeypatch):
        """The render substitutes /usr/local/bin/python3 → sys.executable."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USER", "testuser")

        from iai_mcp.cli import _render_launchd_plist

        rendered = _render_launchd_plist()
        assert sys.executable in rendered
        assert "/usr/local/bin/python3" not in rendered

    def test_rendered_plist_preserves_watchdog_key(self, tmp_path, monkeypatch):
        """Watchdog keys must survive the render — the token replace must not
        strip unexpected XML keys."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USER", "testuser")

        from iai_mcp.cli import _render_launchd_plist

        rendered = _render_launchd_plist()
        # One representative existing key that must survive unchanged
        assert "IAI_MCP_WATCHDOG_LIVENESS_POLL_SEC" in rendered


# ---------------------------------------------------------------------------
# FSM-mirror lock-step tests
# ---------------------------------------------------------------------------


def _write_canonical(path: Path, state: str) -> None:
    """Write a minimal canonical lifecycle_state.json to path."""
    path.write_text(
        json.dumps(
            {
                "current_state": state,
                "since_ts": "2026-06-01T00:00:00+00:00",
                "last_activity_ts": "2026-06-01T00:00:00+00:00",
                "wrapper_event_seq": 0,
                "sleep_cycle_progress": None,
                "quarantine": None,
                "shadow_run": False,
                "crisis_mode": False,
            }
        )
    )


def _write_legacy(path: Path, fsm_state: str) -> None:
    """Write a minimal legacy .daemon-state.json to path."""
    path.write_text(json.dumps({"fsm_state": fsm_state}))


class TestNoFsmDriftAfterCanonicalTransitions:
    """After a canonical state change, the legacy mirror must reflect it
    so reconcile_fsm_state reports drift=False."""

    def test_canonical_wake_legacy_mirrors_wake(self, tmp_path):
        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        _write_canonical(canonical_path, "WAKE")
        _write_legacy(legacy_path, "WAKE")

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is False

    def test_canonical_drowsy_legacy_mirrors_transitioning(self, tmp_path):
        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        _write_canonical(canonical_path, "DROWSY")
        _write_legacy(legacy_path, "TRANSITIONING")

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is False

    def test_canonical_sleep_legacy_mirrors_sleep(self, tmp_path):
        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        _write_canonical(canonical_path, "SLEEP")
        _write_legacy(legacy_path, "SLEEP")

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is False

    def test_canonical_to_legacy_mapping_is_complete(self):
        """Every canonical state maps to a legacy value in _CANONICAL_TO_LEGACY."""
        from iai_mcp.lifecycle_state import LifecycleState

        for state in LifecycleState:
            assert state.value in _CANONICAL_TO_LEGACY, (
                f"Missing mapping for canonical state {state.value}"
            )

    def test_s2_coordinator_writes_legacy_mirror_on_transition(self, tmp_path):
        """S2Coordinator.transition must write the legacy mirror file so the
        legacy fsm_state stays in lock-step with canonical current_state."""
        import asyncio
        from iai_mcp.lifecycle_state import LifecycleState, default_state, save_state as ls_save

        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        # Boot state: WAKE in canonical, WAKE in legacy
        initial = default_state()
        ls_save(initial, canonical_path)
        _write_legacy(legacy_path, "WAKE")

        # Build a minimal S2Coordinator wired to tmp paths
        store_mock = MagicMock()
        store_mock.root = tmp_path

        coord = S2Coordinator(
            store=store_mock,
            state_path=canonical_path,
            legacy_path=legacy_path,
        )

        # Transition WAKE → DROWSY via the coordinator
        asyncio.run(
            coord.transition(
                LifecycleState.WAKE,
                LifecycleState.DROWSY,
                "test_idle_5min",
            )
        )

        # Legacy mirror must now reflect TRANSITIONING (≡ DROWSY)
        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is False, (
            f"Expected no drift after WAKE→DROWSY transition, got: {report}"
        )
        assert report["legacy"] == "TRANSITIONING"


class TestReconcileStillFlagsRealDrift:
    """The detector must still report drift=True for a genuinely divergent pair
    that could NOT result from a correct lock-step write."""

    def test_canonical_wake_legacy_sleep_is_drift(self, tmp_path):
        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        _write_canonical(canonical_path, "WAKE")
        _write_legacy(legacy_path, "SLEEP")

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is True

    def test_canonical_sleep_legacy_wake_is_drift(self, tmp_path):
        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        _write_canonical(canonical_path, "SLEEP")
        _write_legacy(legacy_path, "WAKE")

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is True

    def test_canonical_drowsy_legacy_dreaming_is_drift(self, tmp_path):
        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        _write_canonical(canonical_path, "DROWSY")
        _write_legacy(legacy_path, "DREAMING")

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is True
