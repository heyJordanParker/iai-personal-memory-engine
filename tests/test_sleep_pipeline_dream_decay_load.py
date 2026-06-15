from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import iai_mcp.sleep as sleep_module
import iai_mcp.user_model as user_model_module
from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.user_model import UserModel

@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "lifecycle_state.json"

@pytest.fixture
def event_log(tmp_path: Path) -> LifecycleEventLog:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return LifecycleEventLog(log_dir=log_dir)

@pytest.fixture
def pipeline(state_path: Path, event_log: LifecycleEventLog) -> SleepPipeline:
    return SleepPipeline(
        store=None,
        lifecycle_state_path=state_path,
        event_log=event_log,
        quarantine_ttl_hours=24.0,
    )

def test_dream_decay_step_does_not_raise_attribute_error_on_user_model_load(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_decay(store: Any, plasticity_gain: float = 1.0) -> dict[str, int]:
        return {"decayed": 0, "pruned": 0}

    monkeypatch.setattr(sleep_module, "_decay_edges", _fake_decay)

    completed, payload = pipeline._step_dream_decay(interrupt_check=None)

    assert completed is True
    assert isinstance(payload, dict)
    assert payload.get("decayed") == 0
    assert payload.get("pruned") == 0

def test_dream_decay_step_reads_plasticity_gain_from_user_model(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    def _fake_decay(store: Any, plasticity_gain: float = 1.0) -> dict[str, int]:
        captured["plasticity_gain"] = float(plasticity_gain)
        return {"decayed": 0, "pruned": 0}

    def _fake_load() -> UserModel:
        return UserModel(plasticity_gain=0.5)

    monkeypatch.setattr(sleep_module, "_decay_edges", _fake_decay)
    monkeypatch.setattr(user_model_module, "load", _fake_load)

    completed, _payload = pipeline._step_dream_decay(interrupt_check=None)

    assert completed is True
    assert captured.get("plasticity_gain") == pytest.approx(0.5)
