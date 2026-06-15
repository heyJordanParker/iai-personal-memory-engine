from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any, Callable

from iai_mcp.exceptions import StoreError
from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_compact_hippo(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    from iai_mcp.maintenance import optimize_hippo_storage

    if self._check_interrupt(
        SleepStep.OPTIMIZE_HIPPO, 0, interrupt_check,
    ):
        return False, {}

    compact_t0 = time.monotonic()
    report = optimize_hippo_storage(self._store)
    tables_with_errors = [
        t for t, r in (report or {}).items()
        if isinstance(r, dict) and "error" in r
    ]

    from iai_mcp.daemon_config import _load_erasure_config
    cfg = _load_erasure_config()
    ttl_sec = cfg.tombstone_ttl_sec

    now = self._now()
    drop_cutoff = now - timedelta(seconds=ttl_sec)

    from iai_mcp.store import RECORDS_TABLE
    from iai_mcp.events import write_event

    tbl = self._store.db.open_table(RECORDS_TABLE)
    untomb_where = (
        "tombstoned_at IS NOT NULL "
        "AND (pinned = true OR never_decay = true)"
    )
    try:
        count_untombstoned = int(tbl.count_rows(filter=untomb_where))
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("compact_hippo untombstone count failed: %s", exc)
        count_untombstoned = 0
    if count_untombstoned > 0:
        try:
            tbl.update(
                where=untomb_where,
                values={"tombstoned_at": None},
            )
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.debug("compact_hippo untombstone update failed: %s", exc)
            count_untombstoned = 0

    tbl = self._store.db.open_table(RECORDS_TABLE)
    drop_cutoff_str = drop_cutoff.strftime("%Y-%m-%d %H:%M:%S")
    drop_where = (
        "tombstoned_at IS NOT NULL "
        f"AND tombstoned_at < '{drop_cutoff_str}'"
    )
    try:
        count_dropped = int(tbl.count_rows(filter=drop_where))
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("compact_hippo drop count failed: %s", exc)
        count_dropped = 0
    if count_dropped > 0:
        try:
            tbl.delete(drop_where)
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.debug("compact_hippo drop delete failed: %s", exc)
            count_dropped = 0

    try:
        write_event(
            self._store,
            "erasure_optimize_drops",
            {
                "count_dropped": int(count_dropped),
                "count_untombstoned": int(count_untombstoned),
                "ts": now.isoformat(),
            },
            severity="info",
        )
    except (OSError, ValueError, StoreError) as exc:
        logger.debug("best-effort erasure_optimize_drops event failed: %s", exc)

    elapsed = round(time.monotonic() - compact_t0, 3)
    try:
        write_event(
            self._store,
            "hippo_compacted",
            {
                "phase": "sleep_cycle",
                "per_table": report,
                "total_elapsed_sec": elapsed,
            },
            severity="info",
        )
    except Exception:  # noqa: BLE001
        logger.debug("hippo_compacted event emit failed", exc_info=True)

    return True, {
        "tables_optimized": list((report or {}).keys()),
        "tables_with_errors": tables_with_errors,
        "count_dropped_by_erasure": int(count_dropped),
        "count_untombstoned_by_pin_override": int(count_untombstoned),
    }


def step_optimize_hippo(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    return self._step_compact_hippo(interrupt_check)
