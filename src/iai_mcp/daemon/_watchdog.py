from __future__ import annotations

import asyncio
import concurrent.futures
import faulthandler
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from iai_mcp.events import (
    DAEMON_MEMORY_PRESSURE_KILL,
    DAEMON_SLEEP_CYCLE_STALE,
    DAEMON_WATCHDOG_NEEDS_OPERATOR,
    DAEMON_WEDGE_KILL,
)
from iai_mcp.lifecycle_state import LifecycleState

log = logging.getLogger(__name__)


def _pkg():
    # the package object is the single source of truth so the daemon's own
    # writes and a test monkeypatch hit the SAME slot and stay visible
    return sys.modules["iai_mcp.daemon"]


HIPPEA_CASCADE_POLL_SEC: float = 5.0

HIPPEA_CASCADE_MIN_INTERVAL_SEC: float = float(
    os.environ.get("IAI_MCP_HIPPEA_MIN_INTERVAL_SEC", "60.0"),
)

_last_cascade_completed_at: float = 0.0

_cascade_executor: concurrent.futures.ThreadPoolExecutor | None = None


WATCHDOG_POLL_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_POLL_SEC", "30.0"),
)
WATCHDOG_THRESHOLD_PERCENT: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_THRESHOLD_PERCENT", "50.0"),
)
WATCHDOG_EVENT_COOLDOWN_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_EVENT_COOLDOWN_SEC", "300.0"),
)
WATCHDOG_SAMPLE_WINDOW: int = 4

WATCHDOG_LIVENESS_POLL_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_LIVENESS_POLL_SEC", "30.0"),
)
WATCHDOG_WARN_POLL_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_WARN_POLL_SEC", "7.0"),
)
WATCHDOG_PROBE_TIMEOUT_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_PROBE_TIMEOUT_SEC", "5.0"),
)
WATCHDOG_FAILURE_DEBOUNCE_N: int = int(
    os.environ.get("IAI_MCP_WATCHDOG_FAILURE_DEBOUNCE_N", "3"),
)
# Watchdog ceiling (2.5 GiB). Measured 2026-06-14: warm resident set ~1.72 GiB
# (Rust embedder + corpus + caches), and the per-consolidation-cycle RSS slope is
# now flat — the double-buffered ANN index and the spawn-context graph-rebuild
# worker removed the per-cycle allocation creep, so the daemon plateaus at the
# warm set instead of climbing. A heavy nightly consolidation adds a bounded
# transient on top of the plateau; the cap clears that peak with margin while
# still catching a runaway leak (the daemon no longer drifts toward the ceiling).
# Operator-overridable via the env var.
WATCHDOG_RSS_HARD_CAP_BYTES: int = int(
    os.environ.get("IAI_MCP_WATCHDOG_RSS_HARD_CAP_BYTES", "2684354560"),
)
WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES: int = int(
    os.environ.get("IAI_MCP_WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES", "1610612736"),
)
WATCHDOG_MAX_RECOVERIES: int = int(
    os.environ.get("IAI_MCP_WATCHDOG_MAX_RECOVERIES", "3"),
)
WATCHDOG_RECOVERY_WINDOW_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_RECOVERY_WINDOW_SEC", "600.0"),
)
WATCHDOG_COLD_START_GRACE_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_COLD_START_GRACE_SEC", "600.0"),
)
WATCHDOG_SLEEP_STALE_THRESHOLD_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_SLEEP_STALE_THRESHOLD_SEC", "7200.0"),
)
WATCHDOG_CRISIS_MODE_EXPIRY_SEC: int = int(
    os.environ.get("IAI_MCP_CRISIS_MODE_EXPIRY_SEC", "259200"),
)

_WATCHDOG_LOG_FD: int | None = None

_WATCHDOG_BLACKBOX_FD: int | None = None

_WATCHDOG_BLACKBOX_EPISODE_FIRED: bool = False

_WATCHDOG_BLACKBOX_ENABLED: bool = (
    os.environ.get("IAI_MCP_WATCHDOG_BLACKBOX_ENABLED", "1").lower()
    not in ("0", "false", "no", "off")
)

BOOT_LOCK_RETRY_ATTEMPTS: int = int(
    os.environ.get("IAI_MCP_BOOT_LOCK_RETRY_ATTEMPTS", "5"),
)
BOOT_LOCK_RETRY_BACKOFF_SEC: float = float(
    os.environ.get("IAI_MCP_BOOT_LOCK_RETRY_BACKOFF_SEC", "0.5"),
)

_last_overload_event_at: float = 0.0

_last_sleep_stale_started_at: str = ""

_daemon_started_monotonic: float | None = None


async def _hippea_cascade_loop(
    store, shutdown: asyncio.Event, *, _clock=time.monotonic,
) -> None:
    from iai_mcp import retrieve
    from iai_mcp.daemon_state import load_state, save_state
    from iai_mcp.hippea_cascade import _install_warm, compute_and_fetch_warm
    # late import so the package attribute is re-fetched and a monkeypatch stays visible
    from iai_mcp.daemon import write_event

    while not shutdown.is_set():
        try:
            state = await asyncio.to_thread(load_state)
            req = state.get("hippea_cascade_request") or {}
            if req.get("pending"):
                elapsed = _clock() - _pkg()._last_cascade_completed_at
                if elapsed < HIPPEA_CASCADE_MIN_INTERVAL_SEC:
                    pass
                else:
                    try:
                        assignment = None
                        try:
                            _graph, assignment, _rc = await asyncio.to_thread(
                                retrieve.build_runtime_graph, store,
                            )
                        except (OSError, ValueError, RuntimeError) as exc:
                            log.debug("build_runtime_graph failed in cascade: %s", exc)
                            assignment = None
                        stats: dict = {
                            "communities_selected": 0, "records_warmed": 0,
                            "top_communities": [],
                        }
                        if assignment is not None:
                            try:
                                loop = asyncio.get_event_loop()
                                executor = _pkg()._cascade_executor
                                recs, top = await loop.run_in_executor(
                                    executor,
                                    compute_and_fetch_warm,
                                    store,
                                    assignment,
                                )
                                inserted = await _install_warm(recs)
                                stats = {
                                    "communities_selected": len(top),
                                    "records_warmed": inserted,
                                    "top_communities": [str(c) for c in top],
                                }
                            except (OSError, ValueError, RuntimeError) as exc:
                                log.debug("cascade compute+fetch failed: %s", exc)
                                stats = {
                                    "communities_selected": 0,
                                    "records_warmed": 0,
                                    "top_communities": [],
                                }
                        try:
                            await asyncio.to_thread(
                                write_event,
                                store,
                                "hippea_cascade_completed",
                                {
                                    "session_id": req.get("session_id", ""),
                                    **stats,
                                },
                                severity="info",
                            )
                        except (OSError, RuntimeError) as exc:
                            log.debug("hippea_cascade_completed event write failed: %s", exc)
                        try:
                            state = await asyncio.to_thread(load_state)
                            state["hippea_cascade_request"] = {"pending": False}
                            await asyncio.to_thread(save_state, state)
                        except (OSError, ValueError) as exc:
                            log.debug("cascade state clear failed: %s", exc)
                    finally:
                        setattr(_pkg(), "_last_cascade_completed_at", _clock())
        except Exception:  # noqa: BLE001 -- cascade loop MUST NOT crash
            log.warning("hippea cascade loop iteration failed", exc_info=True)
        try:
            # re-fetch from the package so a test patch on the poll cadence is honored
            from iai_mcp.daemon import HIPPEA_CASCADE_POLL_SEC
            await asyncio.wait_for(
                shutdown.wait(), timeout=HIPPEA_CASCADE_POLL_SEC,
            )
            break
        except asyncio.TimeoutError:
            continue


def _watchdog_active_task_names() -> list[str]:
    out: list[str] = []
    try:
        for t in asyncio.all_tasks():
            if t.done():
                continue
            name = t.get_name() or "?"
            out.append(name)
    except (RuntimeError, AttributeError) as exc:  # noqa: BLE001 -- introspection failure non-fatal
        log.debug("watchdog task introspection failed: %s", exc)
    return out[:5]


async def _cpu_watchdog_loop(store, shutdown: asyncio.Event) -> None:
    from collections import deque

    import psutil

    # late import / re-fetch so the package attributes are resolved at call time
    # and test monkeypatches on these names stay visible
    from iai_mcp.daemon import (
        write_event,
        load_state,
        WATCHDOG_POLL_SEC,
        WATCHDOG_THRESHOLD_PERCENT,
        WATCHDOG_EVENT_COOLDOWN_SEC,
    )

    proc = psutil.Process(os.getpid())
    try:
        proc.cpu_percent(interval=None)
    except (OSError, psutil.Error) as exc:  # noqa: BLE001 -- prime failure non-fatal
        log.debug("psutil cpu_percent prime failed: %s", exc)

    samples: deque[float] = deque(maxlen=WATCHDOG_SAMPLE_WINDOW)

    while not shutdown.is_set():
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=WATCHDOG_POLL_SEC,
            )
            break
        except asyncio.TimeoutError:
            pass

        try:
            cpu_pct = proc.cpu_percent(interval=None)
            samples.append(cpu_pct)
        except (OSError, psutil.Error) as exc:  # noqa: BLE001 -- psutil flakiness must not crash
            log.debug("cpu_percent sample failed: %s", exc)
            continue

        if (
            len(samples) >= 2
            and samples[-1] > WATCHDOG_THRESHOLD_PERCENT
            and samples[-2] > WATCHDOG_THRESHOLD_PERCENT
        ):
            now_mono = time.monotonic()
            if (now_mono - _pkg()._last_overload_event_at) < WATCHDOG_EVENT_COOLDOWN_SEC:
                continue

            fsm_state = "?"
            try:
                state = await asyncio.to_thread(load_state)
                fsm_state = state.get("fsm_state", "?")
            except (OSError, ValueError, json.JSONDecodeError) as exc:  # noqa: BLE001 -- introspection only
                log.debug("watchdog load_state failed: %s", exc)

            uptime_sec: float | None = None
            _started = _pkg()._daemon_started_monotonic
            if _started is not None:
                uptime_sec = round(now_mono - _started, 1)

            payload = {
                "fsm_state": fsm_state,
                "cpu_samples_pct": list(samples),
                "uptime_sec": uptime_sec,
                "active_tasks": _watchdog_active_task_names(),
                "threshold_pct": WATCHDOG_THRESHOLD_PERCENT,
                "sustained_sec": int(WATCHDOG_POLL_SEC * 2),
            }

            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "daemon_cpu_overload",
                    payload,
                    severity="critical",
                )
            except (OSError, RuntimeError) as exc:  # noqa: BLE001 -- ledger emit failure non-fatal
                log.debug("daemon_cpu_overload event write failed: %s", exc)
                continue

            setattr(_pkg(), "_last_overload_event_at", now_mono)


def _next_poll_interval(pressure_level: int | None) -> float:
    if pressure_level is not None and pressure_level >= 2:
        return WATCHDOG_WARN_POLL_SEC
    return WATCHDOG_LIVENESS_POLL_SEC


def _evaluate_watchdog(
    probe_ok: bool,
    rss: int | None,
    pressure_level: int | None,
    uptime_sec: float,
    consecutive_failures: int,
    recovery_timestamps: list[float],
    now_wall: float,
    *,
    hard_cap: int,
    contributor_floor: int,
    debounce_n: int,
    cold_start_grace_sec: float,
    max_recoveries: int,
    recovery_window_sec: float,
) -> tuple[str, str]:
    recent = [t for t in recovery_timestamps if now_wall - t <= recovery_window_sec]
    breaker_tripped = len(recent) >= max_recoveries

    leak = rss is not None and rss > hard_cap
    pressure = pressure_level is not None and pressure_level >= 2
    big = rss is not None and rss > contributor_floor
    in_grace = uptime_sec < cold_start_grace_sec

    mem_trigger = (not in_grace) and (leak or (pressure and big))
    wedge_trigger = not probe_ok

    if not (mem_trigger or wedge_trigger):
        return ("none", "healthy")

    if consecutive_failures < debounce_n:
        return ("none", "debounce")

    if breaker_tripped:
        return ("needs_operator", "circuit_breaker")

    if wedge_trigger:
        return ("kill", "wedge")
    if leak:
        return ("kill", "leak")
    return ("kill", "memory")


def _watchdog_state_dir() -> "Path":
    root = os.environ.get("IAI_MCP_STORE")
    return Path(root) if root else Path.home() / ".iai-mcp"


def _watchdog_log_path() -> "Path":
    return _watchdog_state_dir() / ".daemon-watchdog.log"


def _watchdog_socket_path() -> str:
    return os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(
        _watchdog_state_dir() / ".daemon.sock"
    )


def _vm_pressure_level() -> int | None:
    import ctypes
    import ctypes.util
    import struct

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        size = ctypes.c_size_t(4)
        buf = ctypes.create_string_buffer(4)
        rc = libc.sysctlbyname(
            b"kern.memorystatus_vm_pressure_level",
            buf,
            ctypes.byref(size),
            None,
            0,
        )
        if rc != 0:
            return None
        return struct.unpack("i", buf.raw[:4])[0]
    except Exception:  # noqa: BLE001 -- unreadable pressure must never crash/kill
        return None


def _own_rss_bytes() -> int | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:  # noqa: BLE001 -- psutil flakiness must not crash/kill
        return None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_breadcrumb(line: bytes) -> None:
    fd = _pkg()._WATCHDOG_LOG_FD
    if fd is None:
        raise OSError("watchdog breadcrumb fd not open")
    os.write(fd, line)


def _self_kill(reason: str, kind: str) -> None:
    try:
        line = f"{_iso_now()} {kind} reason={reason} pid={os.getpid()}\n".encode()
        _write_breadcrumb(line)
    except Exception:  # noqa: BLE001 -- breadcrumb is best-effort ONLY
        pass
    os.kill(os.getpid(), signal.SIGKILL)


def _capture_blackbox(
    log_fd: int | None,
    probe_ok: bool,
    consecutive_failures: int,
    debounce_n: int,
) -> None:
    if log_fd is None:
        return
    try:
        try:
            fd_count: int | None = len(os.listdir("/dev/fd"))
        except OSError:
            fd_count = None

        task_names: list[str] = []
        try:
            task_names = _watchdog_active_task_names()
        except Exception:  # noqa: BLE001
            pass

        header = (
            f"{_iso_now()} pre_kill_forensic_dump"
            f" pid={os.getpid()}"
            f" probe_ok={probe_ok}"
            f" consecutive_failures={consecutive_failures}"
            f" debounce_n={debounce_n}"
            f" fd_count={fd_count}"
            f" tasks={task_names}\n"
        ).encode()
        try:
            os.write(log_fd, header)
        except OSError:
            pass

        try:
            faulthandler.dump_traceback(log_fd, all_threads=True)
        except Exception:  # noqa: BLE001 -- faulthandler failure is non-fatal
            pass

        try:
            os.write(log_fd, b"--- end dump ---\n")
        except OSError:
            pass
    except Exception:  # noqa: BLE001 -- capture failure must never crash the watchdog
        pass


async def _open_exclusive_store_with_backoff(
    store_factory,
    *,
    max_attempts: int | None = None,
    backoff_sec: float | None = None,
):
    from iai_mcp.hippo import HippoLockHeldError as _HippoLockHeldError

    _max = max_attempts if max_attempts is not None else BOOT_LOCK_RETRY_ATTEMPTS
    _base = backoff_sec if backoff_sec is not None else BOOT_LOCK_RETRY_BACKOFF_SEC

    last_exc: _HippoLockHeldError | None = None
    for attempt in range(1, _max + 1):
        try:
            return store_factory()
        except _HippoLockHeldError as exc:
            last_exc = exc
            if attempt < _max:
                delay = _base * attempt
                log.warning(
                    "exclusive store open: lock held by predecessor "
                    "(attempt %d/%d) — retrying in %.2f s",
                    attempt,
                    _max,
                    delay,
                )
                await asyncio.sleep(delay)
    assert last_exc is not None
    log.error(
        "exclusive store open: lock still held after %d attempts — giving up",
        _max,
    )
    raise last_exc


def _load_recovery_timestamps(
    log_path: "Path", kinds: tuple[str, ...]
) -> list[float]:
    out: list[float] = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                parts = raw.split(None, 2)
                if len(parts) < 2 or parts[1] not in kinds:
                    continue
                try:
                    dt = datetime.fromisoformat(parts[0])
                    out.append(dt.timestamp())
                except (ValueError, OverflowError):
                    continue
    except FileNotFoundError:
        return []
    except OSError:
        return []
    return out


def _check_sleep_cycle_staleness(
    state: "LifecycleStateRecord | dict",
    now: datetime,
    *,
    threshold_sec: float = WATCHDOG_SLEEP_STALE_THRESHOLD_SEC,
) -> tuple[bool, dict]:
    """Predicate: lifecycle stuck in SLEEP for too long.

    Returns (is_stale, context_dict). context_dict is empty when is_stale=False;
    when is_stale=True it carries the fields the caller copies into the
    daemon_sleep_cycle_stale event payload.

    Never raises. Malformed state, missing keys, or unparseable timestamps all
    return (False, {}) so a watchdog tick is never blocked by state-file decay.
    """
    try:
        if state.get("current_state") != LifecycleState.SLEEP.value:
            return (False, {})
        progress = state.get("sleep_cycle_progress")
        if not isinstance(progress, dict):
            return (False, {})
        attempt = progress.get("attempt")
        if not isinstance(attempt, int) or attempt != 1:
            return (False, {})
        started_at_raw = progress.get("started_at")
        if not isinstance(started_at_raw, str) or not started_at_raw:
            return (False, {})
        try:
            started_dt = datetime.fromisoformat(started_at_raw)
        except (ValueError, TypeError):
            return (False, {})
        if started_dt.tzinfo is None:
            started_dt = started_dt.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        stuck_sec = (now - started_dt).total_seconds()
        if stuck_sec <= threshold_sec:
            return (False, {})
        return (True, {
            "sleep_cycle_started_at": started_at_raw,
            "sleep_stuck_sec": int(stuck_sec),
            "threshold_sec": int(threshold_sec),
            "last_completed_index": progress.get("last_completed_index"),
            "last_error": progress.get("last_error"),
            "attempt": attempt,
            "crisis_mode": bool(state.get("crisis_mode", False)),
        })
    except Exception:  # noqa: BLE001 -- predicate MUST NEVER crash the watchdog
        return (False, {})


def _check_crisis_mode_expiry(
    state: "LifecycleStateRecord | dict",
    now: datetime,
    threshold_sec: int = WATCHDOG_CRISIS_MODE_EXPIRY_SEC,
) -> tuple[bool, dict]:
    """Decide whether crisis_mode has exceeded its self-heal threshold.

    Never raises. Returns (expired, ctx):
      - expired=True  -> caller MUST clear crisis_mode + emit
        crisis_mode_auto_expired.
      - expired=False, ctx={"backfilled_since_ts": iso} -> caller MUST
        backfill since_ts without emitting (legacy or malformed state
        observed for the first time).
      - expired=False, ctx={} -> no-op (not in crisis, or within threshold).
    """
    try:
        if not bool(state.get("crisis_mode", False)):
            return (False, {})
        since_raw = state.get("crisis_mode_since_ts")
        if since_raw is None:
            return (False, {"backfilled_since_ts": now.isoformat()})
        if not isinstance(since_raw, str) or not since_raw:
            return (False, {"backfilled_since_ts": now.isoformat()})
        try:
            since_dt = datetime.fromisoformat(since_raw)
        except (TypeError, ValueError):
            return (False, {"backfilled_since_ts": now.isoformat()})
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        elapsed_sec = (now - since_dt).total_seconds()
        if elapsed_sec <= threshold_sec:
            return (False, {})
        progress = state.get("sleep_cycle_progress") or {}
        if not isinstance(progress, dict):
            progress = {}
        return (True, {
            "since_ts": since_raw,
            "expired_after_sec": int(elapsed_sec),
            "threshold_sec": int(threshold_sec),
            "last_error": progress.get("last_error"),
            "last_completed_index": progress.get("last_completed_index"),
            "attempt": progress.get("attempt"),
            "current_state": state.get("current_state"),
            "backfilled": False,
        })
    except Exception:  # noqa: BLE001 -- predicate MUST NEVER crash the tick
        return (False, {})


async def _probe_status_roundtrip(sock_path: str, read_timeout: float) -> bool:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(sock_path), timeout=5.0
        )
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        return False
    except asyncio.TimeoutError:
        return False
    try:
        writer.write((json.dumps({"type": "status"}) + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=read_timeout)
        return bool(line)
    except (OSError, asyncio.TimeoutError):
        return False
    finally:
        try:
            writer.close()
        except OSError:
            pass


def _watchdog_tick(
    store,
    sock_path: str,
    log_path: "Path",
    consecutive_failures: int,
    *,
    probe_fn=None,
    pressure_fn=None,
    rss_fn=None,
    blackbox_fn=None,
) -> tuple[float, int]:
    # late import so the package attribute is re-fetched and a monkeypatch stays visible
    from iai_mcp.daemon import write_event

    probe_fn = probe_fn or _probe_status_roundtrip
    pressure_fn = pressure_fn or _vm_pressure_level
    rss_fn = rss_fn or _own_rss_bytes

    try:
        probe_ok = asyncio.run(
            probe_fn(sock_path, WATCHDOG_PROBE_TIMEOUT_SEC)
        )
    except Exception:  # noqa: BLE001 -- a probe failure counts as not-ok, never crashes
        probe_ok = False

    pressure_level = pressure_fn()
    rss = rss_fn()

    leak = rss is not None and rss > WATCHDOG_RSS_HARD_CAP_BYTES
    pressure = pressure_level is not None and pressure_level >= 2
    big = rss is not None and rss > WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES
    _started = _pkg()._daemon_started_monotonic
    uptime_sec = (
        time.monotonic() - _started
        if _started is not None
        else 1e9
    )
    in_grace = uptime_sec < WATCHDOG_COLD_START_GRACE_SEC
    mem_trigger = (not in_grace) and (leak or (pressure and big))
    tick_failing = (not probe_ok) or mem_trigger
    consecutive_failures = consecutive_failures + 1 if tick_failing else 0

    if not tick_failing:
        setattr(_pkg(), "_WATCHDOG_BLACKBOX_EPISODE_FIRED", False)
    elif (
        not probe_ok
        and consecutive_failures < WATCHDOG_FAILURE_DEBOUNCE_N
        and not _pkg()._WATCHDOG_BLACKBOX_EPISODE_FIRED
    ):
        setattr(_pkg(), "_WATCHDOG_BLACKBOX_EPISODE_FIRED", True)
        _bb_fn = blackbox_fn
        if _bb_fn is None and _WATCHDOG_BLACKBOX_ENABLED:
            _bb_fn = _capture_blackbox
        if _bb_fn is not None:
            try:
                _bb_fn(
                    _WATCHDOG_BLACKBOX_FD,
                    probe_ok,
                    consecutive_failures,
                    WATCHDOG_FAILURE_DEBOUNCE_N,
                )
            except Exception:  # noqa: BLE001 -- capture failure must never interrupt the watchdog
                pass

    recovery_timestamps = _load_recovery_timestamps(
        log_path, (DAEMON_WEDGE_KILL, DAEMON_MEMORY_PRESSURE_KILL)
    )

    action, reason = _evaluate_watchdog(
        probe_ok,
        rss,
        pressure_level,
        uptime_sec,
        consecutive_failures,
        recovery_timestamps,
        time.time(),
        hard_cap=WATCHDOG_RSS_HARD_CAP_BYTES,
        contributor_floor=WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES,
        debounce_n=WATCHDOG_FAILURE_DEBOUNCE_N,
        cold_start_grace_sec=WATCHDOG_COLD_START_GRACE_SEC,
        max_recoveries=WATCHDOG_MAX_RECOVERIES,
        recovery_window_sec=WATCHDOG_RECOVERY_WINDOW_SEC,
    )

    # --- sleep-cycle-staleness alert (informational, no kill) ---
    try:
        from iai_mcp.lifecycle_state import load_state as _load_lifecycle
        lc_state = _load_lifecycle()
        is_stale, ctx = _check_sleep_cycle_staleness(
            lc_state, datetime.now(timezone.utc)
        )
        if is_stale:
            started_at = ctx["sleep_cycle_started_at"]
            if getattr(_pkg(), "_last_sleep_stale_started_at", "") != started_at:
                try:
                    write_event(
                        store,
                        DAEMON_SLEEP_CYCLE_STALE,
                        ctx,
                        severity="critical",
                    )
                    setattr(_pkg(), "_last_sleep_stale_started_at", started_at)
                except Exception:  # noqa: BLE001 -- ledger emit failure non-fatal
                    log.debug("daemon_sleep_cycle_stale emit failed", exc_info=True)
    except Exception:  # noqa: BLE001 -- staleness check MUST NEVER crash the watchdog
        log.debug("sleep-cycle staleness check failed", exc_info=True)

    if action == "kill":
        kind = (
            DAEMON_WEDGE_KILL if reason == "wedge" else DAEMON_MEMORY_PRESSURE_KILL
        )
        _self_kill(reason, kind)
    elif action == "needs_operator":
        try:
            write_event(
                store,
                DAEMON_WATCHDOG_NEEDS_OPERATOR,
                {
                    "reason": reason,
                    "consecutive_failures": consecutive_failures,
                    "recoveries_in_window": len(recovery_timestamps),
                    "max_recoveries": WATCHDOG_MAX_RECOVERIES,
                },
                severity="critical",
            )
        except Exception:  # noqa: BLE001 -- a loud-event emit failure is non-fatal
            log.debug("watchdog needs_operator emit failed", exc_info=True)

    return (_next_poll_interval(pressure_level), consecutive_failures)


def _liveness_watchdog(store, stop_event, sock_path: str | None = None) -> None:
    global _WATCHDOG_BLACKBOX_FD

    if sock_path is None:
        sock_path = _watchdog_socket_path()
    log_path = _watchdog_log_path()

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        setattr(_pkg(), "_WATCHDOG_LOG_FD", os.open(
            str(log_path),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        ))
    except OSError:
        log.warning("watchdog breadcrumb fd open failed; circuit-breaker degraded")
        setattr(_pkg(), "_WATCHDOG_LOG_FD", None)

    if _WATCHDOG_BLACKBOX_ENABLED:
        try:
            bb_log_path = log_path.with_name(".daemon-blackbox.log")
            _WATCHDOG_BLACKBOX_FD = os.open(
                str(bb_log_path),
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o600,
            )
        except OSError:
            log.debug("watchdog black-box fd open failed; forensic dump disabled")
            _WATCHDOG_BLACKBOX_FD = None

    setattr(_pkg(), "_WATCHDOG_BLACKBOX_EPISODE_FIRED", False)

    consecutive_failures = 0
    while not stop_event.is_set():
        try:
            next_interval, consecutive_failures = _watchdog_tick(
                store, sock_path, log_path, consecutive_failures
            )
        except Exception:  # noqa: BLE001 -- the watchdog must NEVER crash the daemon
            log.debug("watchdog tick failed", exc_info=True)
            next_interval = WATCHDOG_LIVENESS_POLL_SEC
        stop_event.wait(timeout=next_interval)

