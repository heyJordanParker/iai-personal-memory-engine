from __future__ import annotations

import logging
from typing import Any

from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

log = logging.getLogger(__name__)


def run_rem(brain: Any, store: Any, **kwargs: Any) -> dict:
    pipeline = SleepPipeline(store=store)
    return pipeline.run(**kwargs)


def run_sws(brain: Any, store: Any, **kwargs: Any) -> dict:
    pipeline = SleepPipeline(store=store)
    return pipeline.run(**kwargs)


def run_consolidation(
    brain: Any,
    store: Any,
    hvs: list[bytes],
    tier: str = "bsc",
) -> bytes:
    return brain.ops.consolidation.consolidate(hvs, tier=tier)
