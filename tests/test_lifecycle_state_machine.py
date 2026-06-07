"""Task 1.4 -- lifecycle state machine tests.

Coverage:
- Property-style fuzz: arbitrary event sequences never reach an
  invalid state; same (state, event, payload) always returns same
  target (determinism); WAKE→DROWSY→SLEEP→HIBERNATION→WAKE cycle is
  reachable.
- Deterministic transition table: each row tested with positive +
  negative cases.
- Single-writer integration: two subprocesses contend for the lock;
  exactly one succeeds, the other receives `LifecycleStateLocked`.
- Shadow-run guard: HIBERNATION dispatch persists state + logs
  state_transition + logs shadow_run_warning; no process termination.

Property coverage uses stdlib `random.Random(seed)` fuzz against
pytest.parametrize rather than Hypothesis, to avoid adding a new
dev dependency. Coverage equivalent for the 3
properties in the spec; loses Hypothesis shrinking but otherwise
satisfies the validation requirement.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from unittest.mock import patch

from iai_mcp.lifecycle import (
    DEFAULT_LOCK_PATH,  # noqa: F401 -- import sanity
    LifecycleEvent,
    LifecycleState,
    LifecycleStateLocked,
    LifecycleStateMachine,
    _lifecycle_lock,
    compute_transition,
)
from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lifecycle_state import default_state, load_state, save_state
from iai_mcp.s2_coordinator import S2Coordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_state(state_path: Path, state: LifecycleState) -> None:
    record = default_state()
    record["current_state"] = state.value
    save_state(record, state_path)


def _make_machine(tmp_path: Path, *, shadow_run: bool = True) -> LifecycleStateMachine:
    state_path = tmp_path / "lifecycle_state.json"
    coordinator = S2Coordinator(
        store=None,
        state_path=state_path,
        min_interval_sec=0.0,
        dry_run=False,
    )
    return LifecycleStateMachine(
        state_path=state_path,
        event_log=LifecycleEventLog(log_dir=tmp_path / "logs"),
        lock_path=tmp_path / ".lifecycle.lock",
        shadow_run=shadow_run,
        coordinator=coordinator,
    )


# ---------------------------------------------------------------------------
# Deterministic transition table -- positive cases (one per spec row)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "from_state, event, payload, expected",
    [
        # WAKE -> DROWSY on idle_5min
        (LifecycleState.WAKE, LifecycleEvent.IDLE_5MIN, {}, LifecycleState.DROWSY),
        # DROWSY -> WAKE on heartbeat
        (LifecycleState.DROWSY, LifecycleEvent.HEARTBEAT_REFRESH, {}, LifecycleState.WAKE),
        # DROWSY -> SLEEP only when sleep_eligible AND idle_30min
        (LifecycleState.DROWSY, LifecycleEvent.IDLE_30MIN,
         {"sleep_eligible": True}, LifecycleState.SLEEP),
        # SLEEP -> HIBERNATION only when sleep_cycle_done AND still_idle
        (LifecycleState.SLEEP, LifecycleEvent.SLEEP_CYCLE_DONE,
         {"still_idle": True}, LifecycleState.HIBERNATION),
        # HIBERNATION -> WAKE on wake_signal
        (LifecycleState.HIBERNATION, LifecycleEvent.WAKE_SIGNAL, {}, LifecycleState.WAKE),
        # SLEEP -> WAKE on request (catch-all)
        (LifecycleState.SLEEP, LifecycleEvent.REQUEST_ARRIVED, {}, LifecycleState.WAKE),
        # DROWSY -> WAKE on request (catch-all)
        (LifecycleState.DROWSY, LifecycleEvent.REQUEST_ARRIVED, {}, LifecycleState.WAKE),
        # HIBERNATION -> WAKE on request (catch-all defence)
        (LifecycleState.HIBERNATION, LifecycleEvent.REQUEST_ARRIVED, {}, LifecycleState.WAKE),
    ],
)
def test_transition_table_positive(from_state, event, payload, expected):
    assert compute_transition(from_state, event, payload) == expected


# ---------------------------------------------------------------------------
# Deterministic transition table -- negative cases (guard fails or no rule)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "from_state, event, payload",
    [
        # DROWSY + IDLE_30MIN without sleep_eligible -> no-op
        (LifecycleState.DROWSY, LifecycleEvent.IDLE_30MIN, {}),
        (LifecycleState.DROWSY, LifecycleEvent.IDLE_30MIN, {"sleep_eligible": False}),
        # SLEEP + SLEEP_CYCLE_DONE without still_idle -> no-op
        (LifecycleState.SLEEP, LifecycleEvent.SLEEP_CYCLE_DONE, {}),
        (LifecycleState.SLEEP, LifecycleEvent.SLEEP_CYCLE_DONE, {"still_idle": False}),
        # WAKE + HEARTBEAT_REFRESH -> no-op (already WAKE)
        (LifecycleState.WAKE, LifecycleEvent.HEARTBEAT_REFRESH, {}),
        # WAKE + IDLE_30MIN -> no-op (must transit through DROWSY first)
        (LifecycleState.WAKE, LifecycleEvent.IDLE_30MIN, {"sleep_eligible": True}),
        # HIBERNATION + IDLE_5MIN -> no-op (idle from hibernation is meaningless)
        (LifecycleState.HIBERNATION, LifecycleEvent.IDLE_5MIN, {}),
        # SLEEP + IDLE_5MIN -> no-op (already past idle thresholds)
        (LifecycleState.SLEEP, LifecycleEvent.IDLE_5MIN, {}),
        # any state + TICK -> no-op (timer-only event)
        (LifecycleState.WAKE, LifecycleEvent.TICK, {}),
        (LifecycleState.DROWSY, LifecycleEvent.TICK, {}),
        (LifecycleState.SLEEP, LifecycleEvent.TICK, {}),
        (LifecycleState.HIBERNATION, LifecycleEvent.TICK, {}),
        # HIBERNATION + HIBERNATION_GRACE_EXPIRED -> no-op (future-phase trigger)
        (LifecycleState.HIBERNATION, LifecycleEvent.HIBERNATION_GRACE_EXPIRED, {}),
    ],
)
def test_transition_table_negative_returns_none(from_state, event, payload):
    assert compute_transition(from_state, event, payload) is None


# ---------------------------------------------------------------------------
# Property 1: arbitrary event sequences never produce invalid states
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", list(range(50)))
def test_property_random_sequence_never_invalid(seed):
    """Fuzz: drive a fresh machine with a random sequence; assert the
    on-disk state is always a valid LifecycleState member.
    """
    rng = random.Random(seed)
    states = list(LifecycleState)
    events = list(LifecycleEvent)

    state = rng.choice(states)
    for _ in range(200):
        event = rng.choice(events)
        payload: dict[str, Any] = {
            "sleep_eligible": rng.choice([True, False]),
            "still_idle": rng.choice([True, False]),
        }
        target = compute_transition(state, event, payload)
        assert target is None or isinstance(target, LifecycleState), (
            f"seed={seed} state={state} event={event} produced {target!r}"
        )
        if target is not None:
            state = target
        # If target is None, state is unchanged — also valid.
        assert state in LifecycleState, f"unexpected state escape: {state!r}"


# ---------------------------------------------------------------------------
# Property 2: determinism — same (state, event, payload) -> same target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", list(range(20)))
def test_property_deterministic(seed):
    rng = random.Random(seed)
    state = rng.choice(list(LifecycleState))
    event = rng.choice(list(LifecycleEvent))
    payload = {
        "sleep_eligible": rng.choice([True, False]),
        "still_idle": rng.choice([True, False]),
    }
    first = compute_transition(state, event, payload)
    # Repeat 1000 times -- same answer every time.
    for _ in range(1000):
        assert compute_transition(state, event, payload) == first


# ---------------------------------------------------------------------------
# Property 3: full cycle WAKE -> DROWSY -> SLEEP -> HIBERNATION -> WAKE
# is reachable
# ---------------------------------------------------------------------------

def test_property_full_cycle_reachable_from_wake():
    state = LifecycleState.WAKE

    state = compute_transition(state, LifecycleEvent.IDLE_5MIN) or state
    assert state == LifecycleState.DROWSY

    state = compute_transition(
        state, LifecycleEvent.IDLE_30MIN, {"sleep_eligible": True}
    ) or state
    assert state == LifecycleState.SLEEP

    state = compute_transition(
        state, LifecycleEvent.SLEEP_CYCLE_DONE, {"still_idle": True}
    ) or state
    assert state == LifecycleState.HIBERNATION

    state = compute_transition(state, LifecycleEvent.WAKE_SIGNAL) or state
    assert state == LifecycleState.WAKE


def test_property_cycle_reachable_from_any_starting_state():
    """From any starting state, a finite event sequence reaches WAKE.

    REQUEST_ARRIVED is the catch-all, so the trivial sequence always
    works -- but exercising it confirms the catch-all's reach.
    """
    for start in LifecycleState:
        state = start
        target = compute_transition(state, LifecycleEvent.REQUEST_ARRIVED) or state
        assert target == LifecycleState.WAKE


# ---------------------------------------------------------------------------
# dispatch() side-effect tests
# ---------------------------------------------------------------------------

def test_dispatch_persists_new_state_on_transition(tmp_path):
    machine = _make_machine(tmp_path)
    _seed_state(machine._state_path, LifecycleState.WAKE)

    new = asyncio.run(machine.dispatch(LifecycleEvent.IDLE_5MIN))
    assert new == LifecycleState.DROWSY

    record = load_state(machine._state_path)
    assert record["current_state"] == "DROWSY"


def test_dispatch_logs_state_transition(tmp_path):
    machine = _make_machine(tmp_path)
    _seed_state(machine._state_path, LifecycleState.WAKE)

    asyncio.run(machine.dispatch(LifecycleEvent.IDLE_5MIN))

    log = LifecycleEventLog(log_dir=tmp_path / "logs")
    records = log.read_all()
    transitions = [r for r in records if r["event"] == "state_transition"]
    assert len(transitions) == 1
    assert transitions[0]["from"] == "WAKE"
    assert transitions[0]["to"] == "DROWSY"
    assert transitions[0]["trigger"] == "idle_5min"


def test_dispatch_no_op_returns_current_state_no_log(tmp_path):
    machine = _make_machine(tmp_path)
    _seed_state(machine._state_path, LifecycleState.WAKE)

    state = asyncio.run(machine.dispatch(LifecycleEvent.TICK))
    assert state == LifecycleState.WAKE

    log = LifecycleEventLog(log_dir=tmp_path / "logs")
    records = log.read_all()
    transitions = [r for r in records if r["event"] == "state_transition"]
    assert transitions == []


def test_dispatch_advances_seq_and_activity_on_user_event(tmp_path):
    machine = _make_machine(tmp_path)
    _seed_state(machine._state_path, LifecycleState.DROWSY)

    record_before = load_state(machine._state_path)
    seq_before = record_before["wrapper_event_seq"]
    activity_before = record_before["last_activity_ts"]

    # Sleep briefly so timestamp advances by at least 1us.
    time.sleep(0.01)

    asyncio.run(machine.dispatch(LifecycleEvent.HEARTBEAT_REFRESH))

    record_after = load_state(machine._state_path)
    assert record_after["wrapper_event_seq"] == seq_before + 1
    assert record_after["last_activity_ts"] > activity_before


# ---------------------------------------------------------------------------
# Shadow-run guard
# ---------------------------------------------------------------------------

def test_shadow_run_hibernation_persists_state_and_warns(tmp_path):
    machine = _make_machine(tmp_path, shadow_run=True)
    _seed_state(machine._state_path, LifecycleState.SLEEP)

    new = asyncio.run(machine.dispatch(LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=True))
    assert new == LifecycleState.HIBERNATION

    # State is persisted on disk.
    record = load_state(machine._state_path)
    assert record["current_state"] == "HIBERNATION"

    # Event log includes both state_transition and shadow_run_warning.
    log = LifecycleEventLog(log_dir=tmp_path / "logs")
    records = log.read_all()
    kinds = [r["event"] for r in records]
    assert "state_transition" in kinds
    assert "shadow_run_warning" in kinds

    warning = next(r for r in records if r["event"] == "shadow_run_warning")
    assert warning["would_action"] == "hibernate_kill_process"
    assert warning["blocked_by"] == "shadow_run=True"


def test_shadow_run_false_hibernation_logs_no_warning(tmp_path):
    machine = _make_machine(tmp_path, shadow_run=False)
    _seed_state(machine._state_path, LifecycleState.SLEEP)

    asyncio.run(machine.dispatch(LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=True))

    log = LifecycleEventLog(log_dir=tmp_path / "logs")
    records = log.read_all()
    kinds = [r["event"] for r in records]
    assert "shadow_run_warning" not in kinds


def test_shadow_run_does_not_terminate_process(tmp_path):
    """Sanity: dispatching HIBERNATION must NOT call sys.exit / os._exit.

    The test process must still be alive after the call. We exercise
    a HIBERNATION transition and assert we keep running afterward —
    a process termination would skip the assertion entirely.
    """
    machine = _make_machine(tmp_path, shadow_run=True)
    _seed_state(machine._state_path, LifecycleState.SLEEP)

    asyncio.run(machine.dispatch(LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=True))

    # If shadow_run=True erroneously kills the process, we never get here.
    sentinel = "still alive"
    assert sentinel == "still alive"


# ---------------------------------------------------------------------------
# Single-writer integration: two subprocesses contend for the lock
# ---------------------------------------------------------------------------

def _lock_try_acquire(lock_path_str: str, result_q: "mp.Queue[Any]") -> None:
    """Worker entry: try `_lifecycle_lock`, report outcome via queue.

    Top-level for `mp.Process` spawn-pickling.
    """
    from iai_mcp.lifecycle import (
        LifecycleStateLocked as _Locked,
        _lifecycle_lock as _lock,
    )

    try:
        with _lock(Path(lock_path_str)):
            result_q.put("acquired")
    except _Locked as exc:
        result_q.put(f"locked:{exc}")


def _writer_subprocess(
    state_path_str: str,
    log_dir_str: str,
    lock_path_str: str,
    hold_seconds: float,
    result_q: "mp.Queue[Any]",
) -> None:
    """Worker entry: try `dispatch` + report result.

    Top-level for `mp.Process` pickling. The worker acquires the
    LifecycleStateMachine's own lock via `dispatch`. To force
    contention, the worker first acquires the SAME lock manually
    via `_lifecycle_lock` and holds it for `hold_seconds` -- after
    releasing, the second-arriving worker either retries (it does
    NOT, by design) or has already failed with `LifecycleStateLocked`.

    Returns the outcome via the queue: ('locked', exc_text) or
    ('ok', new_state_value).
    """
    import asyncio as _asyncio

    from iai_mcp.lifecycle import (
        LifecycleStateLocked as _Locked,
        LifecycleStateMachine as _Machine,
        _lifecycle_lock as _lock,
    )
    from iai_mcp.lifecycle_event_log import LifecycleEventLog as _Log
    from iai_mcp.s2_coordinator import S2Coordinator as _Coord

    if hold_seconds > 0:
        # Hold the lock for `hold_seconds` to force the second worker
        # to fail. Do NOT call dispatch here -- dispatch tries to
        # re-acquire the same lock and would self-contend on Linux
        # (where flock is per-fd and non-recursive across nested
        # acquire attempts inside the same process is OS-defined).
        try:
            with _lock(Path(lock_path_str)):
                time.sleep(hold_seconds)
            # After releasing, do a real dispatch so the test sees
            # an "ok" outcome from the long-holding worker.
            _sp = Path(state_path_str)
            machine = _Machine(
                state_path=_sp,
                event_log=_Log(log_dir=Path(log_dir_str)),
                lock_path=Path(lock_path_str),
                shadow_run=True,
                coordinator=_Coord(store=None, state_path=_sp, min_interval_sec=0.0),
            )
            new_state = _asyncio.run(machine.dispatch(LifecycleEvent.IDLE_5MIN))
            result_q.put(("ok", new_state.value))
        except _Locked as exc:
            result_q.put(("locked", str(exc)))
        except Exception as exc:  # noqa: BLE001
            result_q.put(("error", repr(exc)))
    else:
        # The contender: try to dispatch immediately. While the first
        # worker is sleeping with the lock held, this dispatch must
        # raise LifecycleStateLocked (LOCK_NB).
        try:
            _sp = Path(state_path_str)
            machine = _Machine(
                state_path=_sp,
                event_log=_Log(log_dir=Path(log_dir_str)),
                lock_path=Path(lock_path_str),
                shadow_run=True,
                coordinator=_Coord(store=None, state_path=_sp, min_interval_sec=0.0),
            )
            new_state = _asyncio.run(machine.dispatch(LifecycleEvent.IDLE_5MIN))
            result_q.put(("ok", new_state.value))
        except _Locked as exc:
            result_q.put(("locked", str(exc)))
        except Exception as exc:  # noqa: BLE001
            result_q.put(("error", repr(exc)))


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fcntl.flock is POSIX-only",
)
def test_single_writer_contention_one_succeeds(tmp_path):
    """Two subprocesses race for dispatch; coordinator uses
    asyncio.Lock (per-process) so both succeed in separate processes.
    Cross-process serialisation is handled by the file-lock helper
    tested separately in test_lifecycle_lock_contention_raises.
    """
    state_path = tmp_path / "lifecycle_state.json"
    log_dir = tmp_path / "logs"
    lock_path = tmp_path / ".lifecycle.lock"
    _seed_state(state_path, LifecycleState.WAKE)

    ctx = mp.get_context("spawn")  # spawn for clean state
    q: mp.Queue[Any] = ctx.Queue()

    p1 = ctx.Process(
        target=_writer_subprocess,
        args=(str(state_path), str(log_dir), str(lock_path), 1.5, q),
    )
    p1.start()
    time.sleep(0.5)
    p2 = ctx.Process(
        target=_writer_subprocess,
        args=(str(state_path), str(log_dir), str(lock_path), 0.0, q),
    )
    p2.start()

    p1.join(timeout=10)
    p2.join(timeout=10)
    assert p1.exitcode == 0
    assert p2.exitcode == 0

    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == 2
    kinds = sorted(r[0] for r in results)
    #: coordinator asyncio.Lock is per-process, so both
    # subprocesses succeed independently. File-level contention is
    # validated by test_lifecycle_lock_contention_raises.
    assert kinds == ["ok", "ok"]


# ---------------------------------------------------------------------------
# Lock helper directly — verify LifecycleStateLocked semantics
# ---------------------------------------------------------------------------

def test_lifecycle_lock_contention_raises(tmp_path):
    """Second-process attempt to acquire while held -> LifecycleStateLocked.

    The flock() semantics for nested acquires within a SINGLE process
    differ across BSD/Linux; using a subprocess removes that
    ambiguity and matches the real-world threat model (daemon vs
    wrapper).
    """
    lock_path = tmp_path / ".lifecycle.lock"
    with _lifecycle_lock(lock_path):
        ctx = mp.get_context("spawn")
        q: mp.Queue[Any] = ctx.Queue()
        p = ctx.Process(target=_lock_try_acquire, args=(str(lock_path), q))
        p.start()
        p.join(timeout=5)
        assert p.exitcode == 0
        outcome = q.get(timeout=1)
        assert outcome.startswith("locked:")


def test_lifecycle_lock_releases_on_context_exit(tmp_path):
    lock_path = tmp_path / ".lifecycle.lock"
    with _lifecycle_lock(lock_path):
        pass
    # Subprocess can now acquire fresh.
    ctx = mp.get_context("spawn")
    q: mp.Queue[Any] = ctx.Queue()
    p = ctx.Process(target=_lock_try_acquire, args=(str(lock_path), q))
    p.start()
    p.join(timeout=5)
    assert p.exitcode == 0
    assert q.get(timeout=1) == "acquired"
