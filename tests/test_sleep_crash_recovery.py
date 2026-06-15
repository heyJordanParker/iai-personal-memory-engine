from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
except ImportError:
    SleepPipeline = None

@pytest.fixture
def pipeline_dir(tmp_path):
    store_path = tmp_path / "store"
    store_path.mkdir()
    return store_path

def _write_checkpoint(store_path: Path, step_name: str, completed: list[str]):
    cp = store_path / ".sleep-checkpoint.json"
    cp.write_text(json.dumps({
        "current_step": step_name,
        "completed_steps": completed,
        "cycle_id": "test-cycle-001",
    }))
    return cp

class TestSleepCrashRecovery:
    def test_checkpoint_file_created_on_step_start(self, pipeline_dir):
        cp = pipeline_dir / ".sleep-checkpoint.json"
        assert not cp.exists(), "Checkpoint should not exist before pipeline runs"

    def test_checkpoint_survives_step_names(self, pipeline_dir):
        steps = ["SCHEMA_MINE", "KNOB_TUNE", "DREAM_DECAY", "OPTIMIZE_HIPPO", "HIPPO_CLEANUP"]
        for i, step in enumerate(steps):
            _write_checkpoint(pipeline_dir, step, steps[:i])
            cp = json.loads((pipeline_dir / ".sleep-checkpoint.json").read_text())
            assert cp["current_step"] == step
            assert cp["completed_steps"] == steps[:i]

    def test_checkpoint_with_quarantined_step(self, pipeline_dir):
        _write_checkpoint(pipeline_dir, "KNOB_TUNE", ["SCHEMA_MINE"])
        cp = json.loads((pipeline_dir / ".sleep-checkpoint.json").read_text())
        assert cp["completed_steps"] == ["SCHEMA_MINE"]
        assert cp["current_step"] == "KNOB_TUNE"

    def test_empty_checkpoint_means_fresh_start(self, pipeline_dir):
        cp = pipeline_dir / ".sleep-checkpoint.json"
        assert not cp.exists()
