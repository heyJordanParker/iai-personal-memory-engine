from __future__ import annotations

from unittest.mock import MagicMock, patch


from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline, SleepStep

class TestSelfHealDeleted:
    def test_maybe_self_heal_version_pileup_does_not_exist(self) -> None:
        assert not hasattr(SleepPipeline, "_maybe_self_heal_version_pileup"), (
            "_maybe_self_heal_version_pileup still present — should have been deleted"
        )

class TestEnumValuesFrozen:
    def test_sleep_step_enum_values_preserved(self) -> None:
        assert SleepStep.OPTIMIZE_HIPPO.value == 4, (
            f"OPTIMIZE_HIPPO value changed: {SleepStep.OPTIMIZE_HIPPO.value}"
        )
        assert SleepStep.HIPPO_CLEANUP.value == 5, (
            f"HIPPO_CLEANUP value changed: {SleepStep.HIPPO_CLEANUP.value}"
        )

class TestMethodRename:
    def test_step_optimize_hippo_delegates_to_step_compact_hippo(self) -> None:
        assert hasattr(SleepPipeline, "_step_compact_hippo"), (
            "_step_compact_hippo is missing"
        )

    def test_step_hippo_cleanup_is_resume_noop(self) -> None:
        pipeline = SleepPipeline.__new__(SleepPipeline)
        with patch(
            "iai_mcp.maintenance.optimize_hippo_storage",
        ) as mock_opt:
            done, payload = pipeline._step_hippo_cleanup_noop(
                interrupt_check=None,
            )

        assert done is True, f"Expected done=True, got {done!r}"
        assert payload == {"action": "hippo_cleanup_noop"}, (
            f"Unexpected noop payload: {payload!r}"
        )
        mock_opt.assert_not_called()

class TestCompactHippoCallsOptimizeHippoStorage:
    def test_compact_hippo_step_calls_optimize_hippo_storage(self) -> None:
        pipeline = SleepPipeline.__new__(SleepPipeline)

        mock_tbl = MagicMock()
        mock_tbl.count_rows.return_value = 0

        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_tbl

        mock_store = MagicMock()
        mock_store.db = mock_db

        pipeline._store = mock_store

        fake_report = {"records": {"rows_before": 10, "rows_after": 10}}

        with (
            patch(
                "iai_mcp.maintenance.optimize_hippo_storage",
                return_value=fake_report,
            ) as mock_opt,
            patch(
                "iai_mcp.daemon_config._load_erasure_config",
            ) as mock_cfg,
            patch("iai_mcp.events.write_event"),
        ):
            from iai_mcp.daemon_config import ErasureConfig
            mock_cfg.return_value = ErasureConfig(
                centrality_threshold=0.5,
                age_days=30,
                retrieval_window_days=7,
                tombstone_ttl_sec=86400,
                dry_run=False,
            )
            done, payload = pipeline._step_compact_hippo(interrupt_check=None)

        assert done is True, f"Expected done=True, got {done!r}"
        mock_opt.assert_called_once_with(mock_store)
