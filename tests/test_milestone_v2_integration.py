from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from iai_mcp.capture_queue import CaptureQueue
from iai_mcp.heartbeat_scanner import HeartbeatScanner
from iai_mcp.idle_detector import IdleDetector
from iai_mcp.lifecycle import (
    LifecycleEvent,
    LifecycleStateMachine,
)
from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lifecycle_lock import (
    LifecycleLock,
    LifecycleLockConflict,
)
from iai_mcp.lifecycle_state import LifecycleState, load_state
from iai_mcp.s2_coordinator import S2Coordinator


def _dispatch(lsm: LifecycleStateMachine, event: LifecycleEvent, **payload) -> LifecycleState:
    return asyncio.run(lsm.dispatch(event, **payload))


@pytest.fixture
def integration_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "wrappers").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pending").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _make_lsm(integration_root: Path) -> LifecycleStateMachine:
    state_path = integration_root / "lifecycle_state.json"
    coordinator = S2Coordinator(
        store=None,
        state_path=state_path,
        min_interval_sec=5.0,
        dry_run=True,
    )
    return LifecycleStateMachine(
        state_path=state_path,
        event_log=LifecycleEventLog(log_dir=integration_root / "logs"),
        lock_path=integration_root / ".lifecycle.lock",
        shadow_run=False,
        coordinator=coordinator,
    )


def test_wake_to_drowsy_on_idle_5min(integration_root: Path) -> None:
    lsm = _make_lsm(integration_root)
    assert lsm.current_state is LifecycleState.WAKE

    _dispatch(lsm, LifecycleEvent.IDLE_5MIN)
    assert lsm.current_state is LifecycleState.DROWSY

    record = load_state(integration_root / "lifecycle_state.json")
    assert record["current_state"] == "DROWSY"
    assert record["shadow_run"] is False


def test_drowsy_to_sleep_requires_sleep_eligible_payload(
    integration_root: Path,
) -> None:
    lsm = _make_lsm(integration_root)
    _dispatch(lsm, LifecycleEvent.IDLE_5MIN)
    assert lsm.current_state is LifecycleState.DROWSY

    _dispatch(lsm, LifecycleEvent.IDLE_30MIN)
    assert lsm.current_state is LifecycleState.DROWSY

    _dispatch(lsm, LifecycleEvent.IDLE_30MIN, sleep_eligible=True)
    assert lsm.current_state is LifecycleState.SLEEP


def test_sleep_to_hibernation_on_cycle_done_with_still_idle(
    integration_root: Path,
) -> None:
    lsm = _make_lsm(integration_root)
    _dispatch(lsm, LifecycleEvent.IDLE_5MIN)
    _dispatch(lsm, LifecycleEvent.IDLE_30MIN, sleep_eligible=True)
    assert lsm.current_state is LifecycleState.SLEEP

    _dispatch(lsm, LifecycleEvent.SLEEP_CYCLE_DONE)
    assert lsm.current_state is LifecycleState.SLEEP

    _dispatch(lsm, LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=True)
    assert lsm.current_state is LifecycleState.HIBERNATION


def test_hibernation_to_wake_via_wake_signal(integration_root: Path) -> None:
    lsm = _make_lsm(integration_root)
    _dispatch(lsm, LifecycleEvent.IDLE_5MIN)
    _dispatch(lsm, LifecycleEvent.IDLE_30MIN, sleep_eligible=True)
    _dispatch(lsm, LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=True)
    assert lsm.current_state is LifecycleState.HIBERNATION

    _dispatch(lsm, LifecycleEvent.WAKE_SIGNAL)
    assert lsm.current_state is LifecycleState.WAKE


def test_sleep_to_wake_on_request_arrived(integration_root: Path) -> None:
    lsm = _make_lsm(integration_root)
    _dispatch(lsm, LifecycleEvent.IDLE_5MIN)
    _dispatch(lsm, LifecycleEvent.IDLE_30MIN, sleep_eligible=True)
    assert lsm.current_state is LifecycleState.SLEEP

    _dispatch(lsm, LifecycleEvent.REQUEST_ARRIVED)
    assert lsm.current_state is LifecycleState.WAKE


def test_capture_queue_drains_record_across_hibernation(
    integration_root: Path,
) -> None:
    queue = CaptureQueue(queue_dir=integration_root / "pending")

    queue.append({
        "session_id": "test-session",
        "role": "user",
        "cue": "remember this fact",
        "text": "the user prefers Russian for surface but English for storage",
        "tier": "episodic",
    })
    assert queue.pending_count() == 1

    captured: list[dict] = []
    ingested = queue.ingest_pending(handler=lambda rec: captured.append(rec))
    assert ingested == 1
    assert queue.pending_count() == 0
    assert captured[0]["text"].startswith("the user prefers Russian")


def test_lifecycle_lock_blocks_second_daemon(
    integration_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock1 = LifecycleLock(integration_root / ".locked")
    lock1.acquire()

    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_is_pid_alive", lambda pid: True)
    monkeypatch.setattr(
        ll, "_current_hostname",
        lambda: json.loads(
            (integration_root / ".locked").read_text(),
        )["hostname"],
    )

    lock2 = LifecycleLock(integration_root / ".locked")
    with pytest.raises(LifecycleLockConflict):
        lock2.acquire()

    lock1.release()
    assert not (integration_root / ".locked").exists()


def test_lifecycle_lock_release_idempotent(
    integration_root: Path,
) -> None:
    lock = LifecycleLock(integration_root / ".locked")
    lock.acquire()
    lock.release()
    assert not (integration_root / ".locked").exists()
    lock.release()


def test_heartbeat_scanner_active_when_fresh_wrapper_present(
    integration_root: Path,
) -> None:
    from datetime import datetime, timezone

    wrappers_dir = integration_root / "wrappers"
    own_pid = os.getpid()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    (wrappers_dir / f"heartbeat-{own_pid}-uuid-test.json").write_text(
        json.dumps({
            "pid": own_pid,
            "uuid": "uuid-test",
            "started_at": now,
            "last_refresh": now,
            "wrapper_version": "1.0.0",
            "schema_version": 1,
        })
    )

    scanner = HeartbeatScanner(wrappers_dir)
    assert scanner.is_active() is True
    assert scanner.heartbeat_idle_30min() is False


def test_heartbeat_scanner_idle_when_no_wrappers(
    integration_root: Path,
) -> None:
    scanner = HeartbeatScanner(integration_root / "wrappers")
    assert scanner.is_active() is False
    assert scanner.heartbeat_idle_30min() is True


def test_idle_detector_sleep_eligible_short_circuits_on_heartbeat_idle() -> None:
    detector = IdleDetector()
    assert detector.sleep_eligible(heartbeat_idle_30min=True) is True


def test_full_lifecycle_chain_drives_through_all_four_states(
    integration_root: Path,
) -> None:
    lsm = _make_lsm(integration_root)
    log = LifecycleEventLog(log_dir=integration_root / "logs")

    _dispatch(lsm, LifecycleEvent.IDLE_5MIN)
    assert lsm.current_state is LifecycleState.DROWSY

    _dispatch(lsm, LifecycleEvent.IDLE_30MIN, sleep_eligible=True)
    assert lsm.current_state is LifecycleState.SLEEP

    _dispatch(lsm, LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=True)
    assert lsm.current_state is LifecycleState.HIBERNATION

    _dispatch(lsm, LifecycleEvent.WAKE_SIGNAL)
    assert lsm.current_state is LifecycleState.WAKE

    transitions = [
        e for e in log.read_all() if e.get("event") == "state_transition"
    ]
    assert len(transitions) == 4
    expected = [
        ("WAKE", "DROWSY"),
        ("DROWSY", "SLEEP"),
        ("SLEEP", "HIBERNATION"),
        ("HIBERNATION", "WAKE"),
    ]
    actual = [(e["from"], e["to"]) for e in transitions]
    assert actual == expected
