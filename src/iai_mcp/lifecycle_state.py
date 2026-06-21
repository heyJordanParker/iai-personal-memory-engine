from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import NotRequired, TypedDict

LIFECYCLE_STATE_PATH: Path = Path.home() / ".iai-mcp" / "lifecycle_state.json"


class LifecycleState(str, Enum):

    WAKE = "WAKE"
    DROWSY = "DROWSY"
    SLEEP = "SLEEP"
    HIBERNATION = "HIBERNATION"


class SleepCycleProgress(TypedDict, total=False):

    last_completed_index: int
    last_completed_step: int
    attempt: int
    last_error: str | None
    started_at: str


class Quarantine(TypedDict):

    until_ts: str
    reason: str
    since_ts: str


class LifecycleStateRecord(TypedDict):

    current_state: str
    since_ts: str
    last_activity_ts: str
    wrapper_event_seq: int
    sleep_cycle_progress: SleepCycleProgress | None
    quarantine: Quarantine | None
    shadow_run: bool
    crisis_mode: bool
    crisis_mode_since_ts: NotRequired[str | None]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_state() -> LifecycleStateRecord:
    now = _utc_now_iso()
    return {
        "current_state": LifecycleState.WAKE.value,
        "since_ts": now,
        "last_activity_ts": now,
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": False,
        "crisis_mode": False,
        "crisis_mode_since_ts": None,
    }


def _validate_record(raw: object) -> LifecycleStateRecord:
    if not isinstance(raw, dict):
        raise ValueError(
            f"lifecycle_state record must be a JSON object, got {type(raw).__name__}"
        )

    required_str_keys = ("current_state", "since_ts", "last_activity_ts")
    for k in required_str_keys:
        v = raw.get(k)
        if not isinstance(v, str) or not v:
            raise ValueError(f"lifecycle_state.{k} must be a non-empty string, got {v!r}")

    state_value = raw["current_state"]
    if state_value not in {s.value for s in LifecycleState}:
        raise ValueError(
            f"lifecycle_state.current_state {state_value!r} is not a valid LifecycleState"
        )

    seq = raw.get("wrapper_event_seq")
    if not isinstance(seq, int) or seq < 0:
        raise ValueError(
            f"lifecycle_state.wrapper_event_seq must be a non-negative int, got {seq!r}"
        )

    shadow = raw.get("shadow_run")
    if not isinstance(shadow, bool):
        raise ValueError(
            f"lifecycle_state.shadow_run must be a bool, got {shadow!r}"
        )

    crisis_mode = raw.get("crisis_mode", False)
    if not isinstance(crisis_mode, bool):
        raise ValueError(
            f"lifecycle_state.crisis_mode must be a bool, got {crisis_mode!r}"
        )
    raw["crisis_mode"] = crisis_mode

    since_ts_value = raw.get("crisis_mode_since_ts", None)
    if since_ts_value is not None and not isinstance(since_ts_value, str):
        raise ValueError(
            f"lifecycle_state.crisis_mode_since_ts must be a string or null, got {since_ts_value!r}"
        )
    raw["crisis_mode_since_ts"] = since_ts_value

    progress = raw.get("sleep_cycle_progress")
    if progress is not None and not isinstance(progress, dict):
        raise ValueError(
            f"lifecycle_state.sleep_cycle_progress must be dict or null, got {progress!r}"
        )

    quarantine = raw.get("quarantine")
    if quarantine is not None:
        if not isinstance(quarantine, dict):
            raise ValueError(
                f"lifecycle_state.quarantine must be dict or null, got {quarantine!r}"
            )
        for k in ("until_ts", "reason", "since_ts"):
            if not isinstance(quarantine.get(k), str):
                raise ValueError(
                    f"lifecycle_state.quarantine.{k} must be string"
                )

    return raw  # type: ignore[return-value]


def load_state(path: Path | None = None) -> LifecycleStateRecord:
    target = path if path is not None else LIFECYCLE_STATE_PATH
    if not target.exists():
        return default_state()
    try:
        raw = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return default_state()
    try:
        return _validate_record(raw)
    except ValueError:
        return default_state()


def save_state(record: LifecycleStateRecord, path: Path | None = None) -> None:
    target = path if path is not None else LIFECYCLE_STATE_PATH
    _validate_record(record)

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".lifecycle_state.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    replaced = False
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
        replaced = True
    finally:
        if not replaced:
            try:
                os.unlink(tmp)
            except OSError:
                pass
