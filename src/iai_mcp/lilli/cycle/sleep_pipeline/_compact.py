from __future__ import annotations

from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep


def step_hippo_cleanup_noop(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if self._check_interrupt(
        SleepStep.HIPPO_CLEANUP, 0, interrupt_check,
    ):
        return False, {}
    return True, {"action": "hippo_cleanup_noop"}


def step_hippo_cleanup(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    return self._step_hippo_cleanup_noop(interrupt_check)
