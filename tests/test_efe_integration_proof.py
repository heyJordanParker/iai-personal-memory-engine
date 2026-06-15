from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.store import MemoryStore

class TestEFEIntegration:

    def test_pipeline_scoring_includes_stability_bonus(self, tmp_path):
        import iai_mcp.pipeline as pipeline_mod
        import inspect
        source = inspect.getsource(pipeline_mod)
        assert "_stability" in source, "EFE stability variable not in pipeline.py"
        assert "_ig" in source, "EFE information-gain variable not in pipeline.py"
        assert "1.0 - min(float(_stability)" in source, "EFE formula not in pipeline.py"

class TestGABAIntegration:

    def test_knob_tune_calls_gaba(self, tmp_path):
        import iai_mcp.lilli.cycle.sleep_pipeline._knob_tune as sp_mod
        import inspect
        source = inspect.getsource(sp_mod)
        assert "from iai_mcp.gaba_annealing import compute_annealed_k" in source
        assert "should_normalize" in source

    def test_gaba_module_produces_valid_k(self):
        from iai_mcp.gaba_annealing import compute_annealed_k
        k0 = compute_annealed_k(0)
        assert k0 == 20
        k30 = compute_annealed_k(30)
        assert k30 == 5
        ks = [compute_annealed_k(i) for i in range(31)]
        for i in range(1, len(ks)):
            assert ks[i] <= ks[i - 1]

class TestTimeCellsIntegration:

    def test_store_insert_computes_temporal_hash(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        from datetime import datetime, timezone
        rec = _make(text="time cell test record")
        rec.created_at = datetime.now(timezone.utc)
        store.insert(rec)
        assert hasattr(rec, "_temporal_hash"), "temporal_hash not computed on insert"
        th = rec._temporal_hash
        assert th is not None, "temporal_hash is None"
        assert len(th) == 128, f"temporal_hash dimension wrong: {len(th)}"

    def test_time_cells_source_in_store(self):
        import iai_mcp.store._store as store_mod
        import inspect
        source = inspect.getsource(store_mod)
        assert "from iai_mcp.time_cells import compute_temporal_hash" in source
        assert "_temporal_hash" in source

class TestWALIntegration:

    def test_erasure_agent_imports_wal(self):
        import iai_mcp.lilli.cycle.sleep_pipeline._erasure as sp_mod
        import inspect
        source = inspect.getsource(sp_mod)
        assert "from iai_mcp.sleep_wal import SleepWAL" in source
        assert "_wal = SleepWAL()" in source

    def test_wal_writes_to_file(self, tmp_path):
        from iai_mcp.sleep_wal import SleepWAL
        wal = SleepWAL(path=tmp_path / ".sleep-wal.jsonl")
        entry = wal.begin("tombstone", ["rec-1", "rec-2"])
        assert (tmp_path / ".sleep-wal.jsonl").exists()
        content = (tmp_path / ".sleep-wal.jsonl").read_text()
        data = json.loads(content.strip())
        assert data["operation"] == "tombstone"
        assert data["target_ids"] == ["rec-1", "rec-2"]
        assert data["status"] == "pending"
