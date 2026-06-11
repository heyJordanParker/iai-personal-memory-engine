from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import signal
import sqlite3
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


_LAUNCHD_REACT_DELAY_SEC = 2.0
_RESPAWN_BIND_TIMEOUT_SEC = 8.0
_RESPAWN_POLL_INTERVAL_SEC = 0.1


@dataclass
class CheckResult:

    name: str
    passed: bool
    detail: str
    status: str = ""

    def __post_init__(self) -> None:
        if not self.status:
            self.status = "PASS" if self.passed else "FAIL"


@dataclass
class RepairAction:

    label: str
    description: str
    destructive: bool
    execute: Callable[[], tuple[bool, str, int]]


def _resolve_socket_path() -> Path:
    env_path = os.environ.get("IAI_DAEMON_SOCKET_PATH")
    if env_path:
        return Path(env_path)
    from iai_mcp.cli import SOCKET_PATH

    return Path(SOCKET_PATH)


async def _socket_status_probe(socket_path: Path, timeout: float) -> dict | None:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(path=str(socket_path)),
            timeout=timeout,
        )
    except (FileNotFoundError, ConnectionRefusedError, asyncio.TimeoutError, OSError):
        return None
    try:
        writer.write((json.dumps({"type": "status"}) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not line:
            return None
        return json.loads(line.decode("utf-8"))
    except Exception as exc:
        logger.debug("socket status probe failed: %s", exc)
        return None
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def check_a_daemon_alive() -> CheckResult:
    from iai_mcp.daemon_state import load_state

    try:
        state = load_state() or {}
    except Exception as e:
        logger.debug("check_a: daemon-state.json unreadable: %s", e)
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"daemon-state.json unreadable: {type(e).__name__}: {e}",
        )

    pid = state.get("daemon_pid")
    if pid is None:
        return CheckResult(
            "(a) daemon process alive",
            False,
            "ABSENT (no daemon_pid in state — daemon never booted or already shut down)",
        )

    if not isinstance(pid, int) or pid < 1 or pid > 2**31 - 1:
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"daemon_pid={pid!r} is not a valid PID (corrupt state?)",
        )

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"PID {pid} in state but no process found",
        )
    except PermissionError:
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"PID {pid} exists but is not owned by this user",
        )
    except OSError as e:
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"liveness probe failed: {type(e).__name__}: {e}",
        )

    try:
        import psutil

        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline() or [])
        if "iai_mcp.daemon" not in cmdline:
            return CheckResult(
                "(a) daemon process alive",
                False,
                f"PID {pid} is NOT iai_mcp.daemon (got: {proc.name()!r})",
            )
    except Exception as e:  # noqa: BLE001 — psutil edge cases all roll up here
        logger.debug("check_a: psutil verify PID %d failed: %s", pid, e)
        return CheckResult(
            "(a) daemon process alive",
            False,
            f"could not verify PID {pid}: {type(e).__name__}: {e}",
        )

    return CheckResult(
        "(a) daemon process alive",
        True,
        f"PID {pid} (iai_mcp.daemon)",
    )


async def _socket_connect_probe(socket_path: Path, timeout: float) -> str | None:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(path=str(socket_path)),
            timeout=timeout,
        )
    except FileNotFoundError:
        return "FileNotFoundError"
    except ConnectionRefusedError:
        return "ConnectionRefusedError"
    except asyncio.TimeoutError:
        return f"TimeoutError after {int(timeout * 1000)} ms"
    except OSError as e:
        return f"OSError errno={e.errno}: {e.strerror or e}"
    try:
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass
    return None


def check_b_socket_fresh() -> CheckResult:
    socket_path = _resolve_socket_path()
    if not socket_path.exists():
        return CheckResult(
            "(b) socket file fresh",
            False,
            f"{socket_path} does not exist",
        )

    t0 = time.monotonic()
    try:
        err = asyncio.run(_socket_connect_probe(socket_path, timeout=1.0))
    except Exception as e:  # noqa: BLE001 — surface any unexpected probe failure
        logger.debug("check_b: socket probe failed: %s", e)
        return CheckResult(
            "(b) socket file fresh",
            False,
            f"connect failed: {type(e).__name__}: {e}",
        )
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if err is not None:
        return CheckResult(
            "(b) socket file fresh",
            False,
            f"{socket_path} present but unreachable: {err}",
        )
    return CheckResult(
        "(b) socket file fresh",
        True,
        f"{socket_path} connected in {elapsed_ms} ms",
    )


def check_c_lock_healthy() -> CheckResult:
    import errno as _errno
    import fcntl as _fcntl

    lock_path = _resolve_hippo_db_path().parent / ".lock"
    if not lock_path.exists():
        return CheckResult(
            "(c) lock file healthy",
            True,
            f"{lock_path} absent (store not yet initialized)",
        )
    fd = None
    try:
        fd = os.open(str(lock_path), os.O_RDONLY)
        try:
            _fcntl.flock(fd, _fcntl.LOCK_SH | _fcntl.LOCK_NB)
            _fcntl.flock(fd, _fcntl.LOCK_UN)
            return CheckResult(
                "(c) lock file healthy",
                True,
                f"{lock_path} acquirable (store idle)",
            )
        except OSError as exc:
            if exc.errno in (_errno.EAGAIN, _errno.EWOULDBLOCK):
                return CheckResult(
                    "(c) lock file healthy",
                    True,
                    f"{lock_path} held (consolidating or recall active — normal)",
                )
            raise
    except Exception as e:  # noqa: BLE001 — fcntl/OSError/permission all FAIL
        logger.debug("check_c: store-lock probe failed: %s", e)
        return CheckResult(
            "(c) lock file healthy",
            False,
            f"store-lock probe failed: {type(e).__name__}: {e}",
        )
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def check_d_no_orphan_core() -> CheckResult:
    try:
        import psutil

        orphans: list[int] = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cl = " ".join(p.info.get("cmdline") or [])
                if "iai_mcp.core" in cl:
                    orphans.append(p.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not orphans:
            return CheckResult(
                "(d) no orphan iai_mcp.core procs",
                True,
                "0 found",
            )
        return CheckResult(
            "(d) no orphan iai_mcp.core procs",
            False,
            f"{len(orphans)} found: PIDs {orphans}",
        )
    except Exception as e:  # noqa: BLE001 — psutil edge cases
        logger.debug("check_d: psutil probe failed: %s", e)
        return CheckResult(
            "(d) no orphan iai_mcp.core procs",
            False,
            f"psutil probe failed: {type(e).__name__}: {e}",
        )


def check_e_state_file_valid() -> CheckResult:
    from iai_mcp.daemon_state import load_state

    try:
        state = load_state() or {}
    except Exception as e:  # noqa: BLE001 — corrupt JSON / IO error
        logger.debug("check_e: daemon state unreadable: %s", e)
        return CheckResult(
            "(e) daemon state file valid",
            False,
            f"unreadable: {type(e).__name__}: {e}",
        )

    fsm_state = state.get("fsm_state")
    if fsm_state is None:
        return CheckResult(
            "(e) daemon state file valid",
            True,
            "no state file (daemon never booted — not a bug)",
        )

    valid = {"WAKE", "DROWSY", "SLEEP", "SLEEPING", "DREAMING", "HIBERNATION"}
    if fsm_state in valid:
        return CheckResult(
            "(e) daemon state file valid",
            True,
            f"fsm_state={fsm_state}",
        )
    return CheckResult(
        "(e) daemon state file valid",
        False,
        f"fsm_state={fsm_state!r} not in {sorted(valid)}",
    )


def check_f_hippo_readable() -> CheckResult:
    import sqlite3

    from iai_mcp.hippo import HippoLockHeldError

    try:
        from iai_mcp.store import MemoryStore

        MemoryStore()
        return CheckResult(
            "(f) hippo storage readable",
            True,
            "Hippo storage opens without error",
        )
    except HippoLockHeldError as e:
        logger.debug("check_f: store held by running daemon: %s", e)
        return CheckResult(
            "(f) hippo storage readable",
            True,
            "store held by the live daemon — normal",
        )
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            logger.debug("check_f: store held by running daemon (sqlite): %s", e)
            return CheckResult(
                "(f) hippo storage readable",
                True,
                "store held by the live daemon — normal",
            )
        logger.debug("check_f: hippo storage open failed: %s", e)
        return CheckResult(
            "(f) hippo storage readable",
            False,
            f"open failed: {type(e).__name__}: {e}",
        )
    except Exception as e:  # noqa: BLE001 — surface any open failure
        logger.debug("check_f: hippo storage open failed: %s", e)
        return CheckResult(
            "(f) hippo storage readable",
            False,
            f"open failed: {type(e).__name__}: {e}",
        )


def _extract_binder_pids(lsof_output: str, target_socket: Path) -> set[int]:
    pids: set[int] = set()
    current_pid: int | None = None
    target = str(target_socket)
    for line in lsof_output.splitlines():
        if line.startswith("p"):
            try:
                current_pid = int(line[1:])
            except ValueError:
                current_pid = None
        elif line.startswith("n") and current_pid is not None:
            name = line[1:]
            if name == target:
                pids.add(current_pid)
    return pids


def check_g_no_dup_binders() -> CheckResult:
    socket_path = _resolve_socket_path()
    if not socket_path.exists():
        return CheckResult(
            "(g) no dup binders",
            True,
            "no socket file (skip)",
        )
    try:
        result = subprocess.run(
            ["lsof", "-U", "-F", "pn"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return CheckResult(
            "(g) no dup binders",
            True,
            f"lsof unavailable: {e} (skip)",
        )
    binder_pids = _extract_binder_pids(result.stdout, socket_path)
    if len(binder_pids) <= 1:
        return CheckResult(
            "(g) no dup binders",
            True,
            f"{len(binder_pids)} binder(s)",
        )
    return CheckResult(
        "(g) no dup binders",
        False,
        f"{len(binder_pids)} processes bound to socket: {sorted(binder_pids)}",
    )


def check_h_crypto_file_state() -> CheckResult:
    from iai_mcp.crypto import CryptoKey, CryptoKeyError, SERVICE_NAME_DEFAULT

    ck = CryptoKey(user_id="default")
    path = ck._key_file_path()

    if path.exists():
        try:
            ck._try_file_get()
            return CheckResult(
                "(h) crypto key file state",
                True,
                f"crypto key file present at {path} (mode 0o600, valid)",
                status="PASS",
            )
        except CryptoKeyError as exc:
            return CheckResult(
                "(h) crypto key file state",
                False,
                f"crypto key file is malformed: {exc}",
                status="FAIL",
            )

    keyring_has_key = False
    keyring_probe_failed = False
    try:
        import keyring as _keyring
        import keyring.errors as _keyring_errors
    except ImportError:
        _keyring = None
        _keyring_errors = None  # type: ignore[assignment]

    if _keyring is not None:
        try:
            existing = _keyring.get_password(SERVICE_NAME_DEFAULT, "default")
            keyring_has_key = existing is not None
        except _keyring_errors.NoKeyringError:
            pass
        except _keyring_errors.KeyringError:
            keyring_probe_failed = True
        except Exception as e:  # noqa: BLE001 — defensive against keyring backend quirks
            logger.debug("check_h: keyring probe failed: %s", e)
            keyring_probe_failed = True

    if keyring_has_key:
        return CheckResult(
            "(h) crypto key file state",
            True,
            (
                f"crypto key file missing at {path}, but a Keychain entry was found.\n"
                f"  Run `iai-mcp crypto migrate-to-file` from a Terminal to migrate the key."
            ),
            status="WARN",
        )
    if keyring_probe_failed:
        return CheckResult(
            "(h) crypto key file state",
            True,
            (
                f"crypto key file missing at {path}; Keychain probe could not complete "
                f"(may indicate non-interactive context). If you have an existing Keychain key, "
                f"run `iai-mcp crypto migrate-to-file` from a Terminal."
            ),
            status="WARN",
        )

    return CheckResult(
        "(h) crypto key file state",
        True,
        (
            f"crypto key file absent at {path} and no Keychain entry detected. "
            f"Fresh install — run `iai-mcp crypto init` or set IAI_MCP_CRYPTO_PASSPHRASE."
        ),
        status="PASS",
    )


_HIPPO_EXPECTED_SCHEMA_VERSION = "1"


def _resolve_hippo_db_path() -> Path:
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / "hippo" / "brain.sqlite3"


def check_i_hippo_db_size() -> CheckResult:
    db_path = _resolve_hippo_db_path()
    if not db_path.exists():
        return CheckResult(
            name="(i) hippo db size",
            passed=True,
            detail="brain.sqlite3 not present yet (fresh install or no writes yet)",
            status="PASS",
        )
    try:
        size_bytes = db_path.stat().st_size
    except OSError as exc:
        return CheckResult(
            name="(i) hippo db size",
            passed=True,
            detail=f"stat failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    size_mb = size_bytes / (1024 * 1024)
    if size_mb < 500:
        return CheckResult(
            name="(i) hippo db size",
            passed=True,
            detail=f"{size_mb:.1f} MB — healthy",
            status="PASS",
        )
    if size_mb < 2048:
        return CheckResult(
            name="(i) hippo db size",
            passed=True,
            detail=(
                f"{size_mb:.1f} MB — consider "
                f"`iai-mcp maintenance compact-hippo --apply --yes`"
            ),
            status="WARN",
        )
    return CheckResult(
        name="(i) hippo db size",
        passed=False,
        detail=f"{size_mb:.1f} MB — run compaction immediately",
        status="FAIL",
    )


def _resolve_wrappers_dir() -> Path:
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / "wrappers"


def check_m_heartbeat_scanner() -> CheckResult:
    from iai_mcp.heartbeat_scanner import HeartbeatScanner, HeartbeatStatus

    wrappers_dir = _resolve_wrappers_dir()
    if not wrappers_dir.exists():
        return CheckResult(
            name="(m) heartbeat scanner",
            passed=True,
            detail=(
                f"{wrappers_dir} not present yet (fresh install or no "
                "wrapper has refreshed yet)"
            ),
            status="PASS",
        )

    scanner = HeartbeatScanner(wrappers_dir)
    try:
        entries = scanner.scan()
    except OSError as exc:
        return CheckResult(
            name="(m) heartbeat scanner",
            passed=False,
            detail=(
                f"could not scan {wrappers_dir}: "
                f"{type(exc).__name__}: {exc}"
            ),
            status="FAIL",
        )

    fresh = sum(1 for e in entries if e.status is HeartbeatStatus.FRESH)
    stale = sum(1 for e in entries if e.status is HeartbeatStatus.STALE)
    orphan = sum(1 for e in entries if e.status is HeartbeatStatus.ORPHAN)
    return CheckResult(
        name="(m) heartbeat scanner",
        passed=True,
        detail=f"n={fresh} fresh, {stale} stale, {orphan} orphan",
        status="PASS",
    )


def _resolve_lifecycle_state_path() -> Path:
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / "lifecycle_state.json"


def _resolve_lifecycle_log_dir() -> Path:
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / "logs"


def _format_relative_short(ts_iso: str, *, now: Any = None) -> str:
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    try:
        ts = _dt.fromisoformat(ts_iso)
    except (TypeError, ValueError):
        return "?"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_tz.utc)
    moment = now if now is not None else _dt.now(_tz.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_tz.utc)
    seconds = int((moment - ts).total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} h"
    days = hours // 24
    return f"{days} d"


def check_j_lifecycle_current_state() -> CheckResult:
    from iai_mcp.lifecycle_state import load_state

    state_path = _resolve_lifecycle_state_path()
    record = load_state(state_path)
    current = record.get("current_state", "WAKE")
    since_ts = record.get("since_ts", "?")
    elapsed = _format_relative_short(since_ts)
    shadow_run = record.get("shadow_run", True)

    detail = f"{current} since {elapsed} (shadow_run={'true' if shadow_run else 'false'})"
    return CheckResult(
        name="(j) lifecycle current state",
        passed=True,
        detail=detail,
        status="PASS",
    )


def check_k_lifecycle_history_24h() -> CheckResult:
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_event_log import LifecycleEventLog

    log_dir = _resolve_lifecycle_log_dir()
    if not log_dir.exists():
        return CheckResult(
            name="(k) lifecycle history 24h",
            passed=True,
            detail="no event log yet (fresh install or daemon never run)",
            status="PASS",
        )

    log = LifecycleEventLog(log_dir=log_dir)
    now = _dt.now(_tz.utc)
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - _td(days=1)).strftime("%Y-%m-%d")

    transitions: list[dict[str, Any]] = []
    for date_str in (yesterday, today):
        try:
            events = log.read_all(date_str=date_str)
        except OSError:
            continue
        for ev in events:
            if ev.get("event") == "state_transition":
                transitions.append(ev)

    counts: dict[str, int] = {}
    for ev in transitions:
        to = ev.get("to") or "?"
        counts[to] = counts.get(to, 0) + 1

    if not transitions:
        return CheckResult(
            name="(k) lifecycle history 24h",
            passed=True,
            detail="0 transitions in last 24h",
            status="PASS",
        )

    summary = ", ".join(f"{state}={n}" for state, n in sorted(counts.items()))
    return CheckResult(
        name="(k) lifecycle history 24h",
        passed=True,
        detail=f"{len(transitions)} transitions ({summary})",
        status="PASS",
    )


def check_l_sleep_cycle_status() -> CheckResult:
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_state import load_state

    state_path = _resolve_lifecycle_state_path()
    record = load_state(state_path)
    quarantine = record.get("quarantine")
    if quarantine is None:
        return CheckResult(
            name="(l) sleep cycle quarantine",
            passed=True,
            detail="no quarantine active",
            status="PASS",
        )

    reason = quarantine.get("reason", "?")
    until_ts = quarantine.get("until_ts", "?")
    since_ts = quarantine.get("since_ts", "?")

    now = _dt.now(_tz.utc)
    try:
        since = _dt.fromisoformat(since_ts)
        if since.tzinfo is None:
            since = since.replace(tzinfo=_tz.utc)
        age_hours = (now - since).total_seconds() / 3600.0
    except (TypeError, ValueError):
        age_hours = 0.0

    try:
        until = _dt.fromisoformat(until_ts)
        if until.tzinfo is None:
            until = until.replace(tzinfo=_tz.utc)
        expired = until <= now
    except (TypeError, ValueError):
        expired = False

    if expired:
        return CheckResult(
            name="(l) sleep cycle quarantine",
            passed=True,
            detail=(
                f"quarantine expired (until={until_ts}); will clear on next "
                f"sleep-cycle run; reason={reason}"
            ),
            status="PASS",
        )

    detail = (
        f"quarantined for {age_hours:.1f}h; until={until_ts}; reason={reason}"
    )

    if age_hours >= 12.0:
        return CheckResult(
            name="(l) sleep cycle quarantine",
            passed=False,
            detail=(
                f"{detail}; run `iai-mcp maintenance sleep-cycle "
                "--reset-quarantine` to clear"
            ),
            status="FAIL",
        )
    return CheckResult(
        name="(l) sleep cycle quarantine",
        passed=True,
        detail=detail,
        status="WARN",
    )


def check_n_hid_idle_source() -> CheckResult:
    from iai_mcp.idle_detector import IdleDetector

    detector = IdleDetector()
    status = detector.status()

    hid_str = (
        f"{status.hid_idle_sec}s"
        if status.hid_idle_sec is not None
        else "unavailable"
    )
    pmset_str = "recent-sleep" if status.pmset_recent_sleep else "clean"
    signals_str = (
        ",".join(status.available_signals) if status.available_signals else "none"
    )
    detail = (
        f"HIDIdleTime: {hid_str}, pmset: {pmset_str}, available: {signals_str}"
    )

    if "HIDIdleTime" in status.available_signals:
        return CheckResult(
            name="(n) HID idle source",
            passed=True,
            detail=detail,
            status="PASS",
        )
    return CheckResult(
        name="(n) HID idle source",
        passed=True,
        detail=(
            f"{detail}; L6 will fall back to heartbeat-idle only"
        ),
        status="WARN",
    )


def check_w_no_permanent_failed() -> CheckResult:
    import fnmatch

    env_store = os.environ.get("IAI_MCP_STORE")
    if env_store:
        deferred_dir = Path(env_store).parent / ".deferred-captures"
    else:
        deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"

    if not deferred_dir.exists():
        return CheckResult(
            name="(w) no permanent-failed captures",
            passed=True,
            detail="deferred-captures dir absent — nothing to recover",
        )

    count = 0
    try:
        for entry in os.scandir(deferred_dir):
            if entry.is_file() and fnmatch.fnmatch(entry.name, "*.permanent-failed-*.jsonl"):
                count += 1
    except OSError as exc:
        return CheckResult(
            name="(w) no permanent-failed captures",
            passed=True,
            detail=f"could not scan deferred-captures dir: {exc}",
            status="WARN",
        )

    if count == 0:
        return CheckResult(
            name="(w) no permanent-failed captures",
            passed=True,
            detail="No permanent-failed capture files",
        )
    return CheckResult(
        name="(w) no permanent-failed captures",
        passed=True,
        detail=(
            f"{count} permanent-failed capture file(s) — "
            "run 'iai-mcp drain-permanent-failed' to recover"
        ),
        status="WARN",
    )


def check_x_no_collapsed_timestamps() -> CheckResult:
    """Warn when many episodic records share an identical created_at (time-collapsed session)."""
    db_path = _resolve_hippo_db_path()
    if not db_path.exists():
        return CheckResult(
            name="(x) no collapsed-timestamp groups",
            passed=True,
            detail="db absent (fresh install)",
            status="PASS",
        )
    conn = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=2.0)
        rows = conn.execute(
            "SELECT created_at, COUNT(*) AS n FROM records WHERE tier = 'episodic'"
            " GROUP BY created_at HAVING n >= 5 ORDER BY n DESC LIMIT 20"
        ).fetchall()
    except sqlite3.Error as exc:
        return CheckResult(
            name="(x) no collapsed-timestamp groups",
            passed=True,
            detail=f"check skipped: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    if not rows:
        return CheckResult(
            name="(x) no collapsed-timestamp groups",
            passed=True,
            detail="no collapsed timestamp groups found",
            status="PASS",
        )
    group_count = len(rows)
    total_affected = sum(r[1] for r in rows)
    worst_ts, worst_n = rows[0]
    return CheckResult(
        name="(x) no collapsed-timestamp groups",
        passed=False,
        detail=(
            f"{group_count} group(s) with >= 5 records at one timestamp"
            f" ({total_affected} records total; worst group: {worst_n} records at {worst_ts})"
            " — run 'iai-mcp migrate --rederive-timestamps' to repair"
        ),
        status="WARN",
    )


def check_z_avx2_support() -> CheckResult:
    from iai_mcp.cpu_features import has_avx2

    try:
        avx2_ok = has_avx2()
    except Exception as exc:  # noqa: BLE001 -- defensive against probe quirks
        try:
            from iai_mcp.store import CPU_HAS_AVX2
            avx2_ok = CPU_HAS_AVX2
        except Exception:  # noqa: BLE001 -- store may itself be unimportable
            avx2_ok = True
        logger.debug(
            "check_z: has_avx2() probe failed: %s; fallback=%s",
            exc,
            avx2_ok,
        )

    if avx2_ok:
        return CheckResult(
            name="(z) AVX2 CPU support",
            passed=True,
            detail="AVX2 available (or N/A on this architecture)",
            status="PASS",
        )
    return CheckResult(
        name="(z) AVX2 CPU support",
        passed=False,
        detail=(
            "this host lacks AVX2 -- the vector index cannot load; iai-mcp "
            "memory store is unavailable. Deploy on an AVX2-equipped host (any "
            "Intel CPU 2013+; AMD Excavator 2015+; Mac M-series ARM is "
            "unaffected)."
        ),
        status="FAIL",
    )


def _format_top_of_output_hint(results: list[CheckResult]) -> str | None:
    for r in results:
        if r.name == "(h) crypto key file state" and r.status == "WARN":
            flat = " ".join(line.strip() for line in r.detail.splitlines() if line.strip())
            return f"> hint: {flat}"
    return None


_HEADLESS_DOWNGRADE_ROWS: frozenset[str] = frozenset({
    "(b) socket file fresh",
    "(n) HID idle source",
})


def is_headless(*, force: bool = False) -> bool:
    if force:
        return True
    if platform.system() != "Linux":
        return False
    return (
        os.environ.get("DISPLAY") is None
        and os.environ.get("WAYLAND_DISPLAY") is None
    )


def _apply_headless_downgrade(
    results: list[CheckResult], headless: bool
) -> list[CheckResult]:
    if not headless:
        return results
    for r in results:
        if r.name in _HEADLESS_DOWNGRADE_ROWS and r.status == "FAIL":
            r.passed = True
            r.status = "WARN"
    return results


def check_o_subscription_credentials() -> CheckResult:
    try:
        from iai_mcp.claude_cli import verify_credentials_subscription
    except Exception as exc:  # noqa: BLE001 -- defensive
        return CheckResult(
            name="(o) Claude subscription credentials",
            passed=True,
            detail=f"unable to import claude_cli ({exc}); skipping",
            status="WARN",
        )

    result = verify_credentials_subscription()
    if result.get("ok"):
        sub_type = result.get("subscription_type") or result.get("billing_type") or "unknown"
        return CheckResult(
            name="(o) Claude subscription credentials",
            passed=True,
            detail=f"valid {sub_type} subscription with inference scope",
            status="PASS",
        )

    reason = result.get("reason", "unknown")
    return CheckResult(
        name="(o) Claude subscription credentials",
        passed=True,
        detail=(
            f"reason={reason}; daemon will fall back to local Tier-0 "
            "consolidation (no LLM critic, no nightly insight). Run "
            "`claude /login` to restore subscription path."
        ),
        status="WARN",
    )


def check_q_iai_cli_reachable() -> CheckResult:
    import shutil

    iai_path = shutil.which("iai")
    if iai_path is None:
        return CheckResult(
            name="(q) iai CLI reachable",
            passed=True,
            detail=(
                "iai not in PATH. Re-run `pip install -e .` from the repo "
                "root to register the entry point."
            ),
            status="WARN",
        )

    try:
        completed = subprocess.run(  # noqa: S603 -- argv list, no shell
            [iai_path, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(
            name="(q) iai CLI reachable",
            passed=True,
            detail=f"iai found at {iai_path} but invocation failed: {exc}",
            status="WARN",
        )

    if completed.returncode != 0:
        return CheckResult(
            name="(q) iai CLI reachable",
            passed=True,
            detail=(
                f"iai --version exited {completed.returncode}: "
                f"{completed.stderr.strip()[:120]}"
            ),
            status="WARN",
        )

    version_line = (completed.stdout or completed.stderr).strip().splitlines()[0:1]
    version = version_line[0] if version_line else "?"
    return CheckResult(
        name="(q) iai CLI reachable",
        passed=True,
        detail=f"{iai_path} -> {version}",
        status="PASS",
    )


def check_r_hippo_hnsw_loadable() -> CheckResult:
    hnsw_path = _resolve_hippo_db_path().parent / "records.hnsw"
    if not hnsw_path.exists():
        return CheckResult(
            name="(r) hippo hnsw index",
            passed=True,
            detail="records.hnsw absent (HippoDB rebuilds from SQLite on next boot)",
            status="WARN",
        )
    try:
        size = hnsw_path.stat().st_size
    except OSError as exc:
        return CheckResult(
            name="(r) hippo hnsw index",
            passed=False,
            detail=f"stat failed: {type(exc).__name__}: {exc}",
            status="FAIL",
        )
    if size == 0:
        return CheckResult(
            name="(r) hippo hnsw index",
            passed=False,
            detail=(
                "records.hnsw is zero bytes (corrupt; rebuild needed — "
                "restart the daemon to trigger automatic rebuild)"
            ),
            status="FAIL",
        )
    try:
        import hnswlib as _hnswlib
        from iai_mcp.types import EMBED_DIM

        idx = _hnswlib.Index(space="cosine", dim=EMBED_DIM)
        idx.load_index(str(hnsw_path), max_elements=0)
    except Exception as exc:  # noqa: BLE001 — surface any load failure
        logger.debug("check_r: hnswlib.load_index failed: %s", exc)
        return CheckResult(
            name="(r) hippo hnsw index",
            passed=False,
            detail=(
                f"hnswlib.load_index failed: {type(exc).__name__}: {exc} "
                "(restart the daemon to trigger automatic rebuild)"
            ),
            status="FAIL",
        )
    return CheckResult(
        name="(r) hippo hnsw index",
        passed=True,
        detail=f"{size / (1024 * 1024):.1f} MB",
        status="PASS",
    )


def check_s_hippo_schema_version() -> CheckResult:
    db_path = _resolve_hippo_db_path()
    if not db_path.exists():
        return CheckResult(
            name="(s) hippo schema version",
            passed=True,
            detail="db absent (fresh install)",
            status="PASS",
        )
    conn = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=2.0)
        row = conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.Error as exc:
        return CheckResult(
            name="(s) hippo schema version",
            passed=False,
            detail=f"sqlite3 query failed: {type(exc).__name__}: {exc}",
            status="FAIL",
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    if row is None:
        return CheckResult(
            name="(s) hippo schema version",
            passed=False,
            detail="_hippo_meta missing schema_version row",
            status="FAIL",
        )
    value = str(row[0])
    expected = _HIPPO_EXPECTED_SCHEMA_VERSION
    if value != expected:
        return CheckResult(
            name="(s) hippo schema version",
            passed=True,
            detail=f"schema_version={value} (expected {expected})",
            status="WARN",
        )
    return CheckResult(
        name="(s) hippo schema version",
        passed=True,
        detail=f"schema_version={value}",
        status="PASS",
    )


def check_t_hippo_compacted_freshness() -> CheckResult:
    import sqlite3
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    from iai_mcp.hippo import HippoLockHeldError

    try:
        from iai_mcp.events import query_events
        from iai_mcp.store import MemoryStore

        store = MemoryStore()
        events = query_events(store, kind="hippo_compacted", limit=1)
    except HippoLockHeldError as exc:
        logger.debug("check_t: store held by running daemon: %s", exc)
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail="deferred — daemon holds the store (normal)",
            status="PASS",
        )
    except sqlite3.OperationalError as exc:
        if "database is locked" in str(exc).lower():
            logger.debug("check_t: store held by running daemon (sqlite): %s", exc)
            return CheckResult(
                name="(t) hippo_compacted freshness",
                passed=True,
                detail="deferred — daemon holds the store (normal)",
                status="PASS",
            )
        logger.debug("check_t: events query failed: %s", exc)
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    except Exception as exc:  # noqa: BLE001 — probe failure is advisory
        logger.debug("check_t: events query failed: %s", exc)
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )

    if not events:
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail="no hippo_compacted event found (fresh install or compaction not yet run)",
            status="WARN",
        )

    last_event = events[0]
    ts_str = last_event.get("timestamp") or last_event.get("ts") or ""
    try:
        ts = _dt.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_tz.utc)
        now = _dt.now(_tz.utc)
        age_hours = (now - ts).total_seconds() / 3600.0
    except (TypeError, ValueError):
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail="last hippo_compacted event timestamp unparseable",
            status="WARN",
        )

    if age_hours <= 24.0:
        return CheckResult(
            name="(t) hippo_compacted freshness",
            passed=True,
            detail=f"last hippo_compacted event {age_hours:.1f}h ago",
            status="PASS",
        )
    return CheckResult(
        name="(t) hippo_compacted freshness",
        passed=True,
        detail=(
            f"last hippo_compacted event {age_hours:.1f}h ago "
            f"(consider `iai-mcp maintenance compact-hippo --apply --yes`)"
        ),
        status="WARN",
    )


def check_u_recall_centrality_regression() -> CheckResult:
    import sqlite3
    import statistics
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.hippo import HippoLockHeldError

    try:
        from iai_mcp.events import query_events, write_event
        from iai_mcp.store import MemoryStore

        store = MemoryStore()
        since = _dt.now(_tz.utc) - _td(hours=24)
        events = query_events(
            store, kind="recall_timing", since=since, limit=1000
        )
    except HippoLockHeldError as exc:
        logger.debug("check_u: store held by running daemon: %s", exc)
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,
            detail="deferred — daemon holds the store (normal)",
            status="PASS",
        )
    except sqlite3.OperationalError as exc:
        if "database is locked" in str(exc).lower():
            logger.debug("check_u: store held by running daemon (sqlite): %s", exc)
            return CheckResult(
                name="(u) recall centrality regression",
                passed=True,
                detail="deferred — daemon holds the store (normal)",
                status="PASS",
            )
        logger.debug("check_u: events query failed: %s", exc)
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )
    except Exception as exc:  # noqa: BLE001 — probe failure is advisory
        logger.debug("check_u: events query failed: %s", exc)
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,
            detail=f"events query failed: {type(exc).__name__}: {exc}",
            status="WARN",
        )

    if not events:
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,
            detail="no recall_timing events in last 24h (daemon idle or sampling missed)",
            status="WARN",
        )

    centrality_values: list[float] = []
    for ev in events:
        payload = ev.get("data") or {}
        cv = payload.get("centrality_ms")
        if cv is None:
            continue
        try:
            centrality_values.append(float(cv))
        except (TypeError, ValueError):
            continue
    if not centrality_values:
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,
            detail="recall_timing events present but centrality_ms missing/invalid",
            status="WARN",
        )

    median_ms = statistics.median(centrality_values)
    if median_ms > 30.0:
        try:
            write_event(
                store,
                kind="health_concern",
                data={"centrality_median_ms": float(median_ms)},
                severity="warning",
            )
        except Exception as exc:  # noqa: BLE001 — telemetry best-effort
            logger.debug("check_u: health_concern emit failed: %s", exc)
        return CheckResult(
            name="(u) recall centrality regression",
            passed=True,
            detail=(
                f"centrality_ms median {median_ms:.1f}ms > 30ms threshold "
                f"(n_events={len(centrality_values)})"
            ),
            status="WARN",
        )
    return CheckResult(
        name="(u) recall centrality regression",
        passed=True,
        detail=(
            f"centrality_ms median {median_ms:.1f}ms <= 30ms "
            f"(n_events={len(centrality_values)})"
        ),
        status="PASS",
    )


def check_v_native_embedder() -> CheckResult:
    import math

    try:
        import iai_mcp_native  # noqa: F401
        from iai_mcp.embed import Embedder

        emb = Embedder()
        assert emb._backend == "rust", f"backend={emb._backend!r}"
        vec = emb.embed("smoke")
        assert len(vec) == 384, f"expected 384 dims, got {len(vec)}"
        assert all(math.isfinite(float(x)) for x in vec[:3]), (
            "non-finite values in output"
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="(v) native Rust embedder",
            passed=False,
            detail=(
                f"{type(exc).__name__}: {exc} — rebuild with: "
                "cd rust/iai_mcp_native && maturin develop --release"
            ),
        )
    return CheckResult(
        name="(v) native Rust embedder",
        passed=True,
        detail="encode ok, backend=rust, 384-dim",
    )


def check_p_anthropic_sdk_absent() -> CheckResult:
    try:
        import anthropic  # noqa: F401 -- presence-probe only
        return CheckResult(
            name="(p) anthropic SDK absent",
            passed=True,
            detail=(
                "anthropic SDK is importable in this venv. It was dropped "
                "as a runtime dependency; this is likely leftover site-packages "
                "from an older install. Run `pip uninstall anthropic` "
                "to clean up."
            ),
            status="WARN",
        )
    except ImportError:
        return CheckResult(
            name="(p) anthropic SDK absent",
            passed=True,
            detail="ImportError as expected (subscription-only path)",
            status="PASS",
        )


def run_diagnosis() -> list[CheckResult]:
    return [
        check_a_daemon_alive(),
        check_b_socket_fresh(),
        check_c_lock_healthy(),
        check_d_no_orphan_core(),
        check_e_state_file_valid(),
        check_f_hippo_readable(),
        check_g_no_dup_binders(),
        check_h_crypto_file_state(),
        check_i_hippo_db_size(),
        check_j_lifecycle_current_state(),
        check_k_lifecycle_history_24h(),
        check_l_sleep_cycle_status(),
        check_m_heartbeat_scanner(),
        check_n_hid_idle_source(),
        check_o_subscription_credentials(),
        check_p_anthropic_sdk_absent(),
        check_q_iai_cli_reachable(),
        check_r_hippo_hnsw_loadable(),
        check_s_hippo_schema_version(),
        check_t_hippo_compacted_freshness(),
        check_u_recall_centrality_regression(),
        check_v_native_embedder(),
        check_w_no_permanent_failed(),
        check_x_no_collapsed_timestamps(),
        check_z_avx2_support(),
    ]


def print_checklist(results: list[CheckResult]) -> None:
    print("iai doctor — daemon health check\n")
    for r in results:
        if r.status == "WARN":
            tag = "[WARN]"
        elif r.passed:
            tag = "[PASS]"
        else:
            tag = "[FAIL]"
        print(f"  {tag} {r.name:<40} {r.detail}")


def _kill_orphan_cores() -> tuple[bool, str, int]:
    import psutil

    t0 = time.monotonic()
    killed: list[int] = []
    failed: list[tuple[int, str]] = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if "iai_mcp.core" not in cl:
                continue
            pid = p.info["pid"]
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except OSError as e:
            failed.append((p.info.get("pid", -1), str(e)))
    duration_ms = int((time.monotonic() - t0) * 1000)
    if failed:
        return (
            False,
            f"killed {len(killed)} ({killed}); FAILED on {failed}",
            duration_ms,
        )
    return True, f"killed {len(killed)} orphan(s): {killed}", duration_ms


def _unlink_stale_socket() -> tuple[bool, str, int]:
    socket_path = _resolve_socket_path()
    t0 = time.monotonic()
    if not socket_path.exists():
        return True, "no stale socket to unlink", int((time.monotonic() - t0) * 1000)
    try:
        socket_path.unlink()
        return True, f"unlinked {socket_path}", int((time.monotonic() - t0) * 1000)
    except OSError as e:
        return False, f"unlink failed: {e}", int((time.monotonic() - t0) * 1000)


def _respawn_daemon() -> tuple[bool, str, int]:
    from iai_mcp.cli import LAUNCHD_TARGET

    t0 = time.monotonic()
    socket_path = _resolve_socket_path()

    using_default_socket = os.environ.get("IAI_DAEMON_SOCKET_PATH") is None
    if (
        using_default_socket
        and LAUNCHD_TARGET
        and Path(LAUNCHD_TARGET).expanduser().exists()
    ):
        time.sleep(_LAUNCHD_REACT_DELAY_SEC)
        return (
            True,
            "launchd-managed (KeepAlive will respawn)",
            int((time.monotonic() - t0) * 1000),
        )

    try:
        spawn_env = os.environ.copy()
        spawn_env["IAI_DAEMON_RESPAWN_BY"] = "doctor"
        subprocess.Popen(
            [sys.executable, "-m", "iai_mcp.daemon"],
            env=spawn_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:  # noqa: BLE001 — spawn failure is a recovery error
        logger.debug("respawn daemon failed: %s", e)
        return (
            False,
            f"respawn failed: {type(e).__name__}: {e}",
            int((time.monotonic() - t0) * 1000),
        )

    deadline = time.monotonic() + _RESPAWN_BIND_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if socket_path.exists():
            duration_ms = int((time.monotonic() - t0) * 1000)
            return (
                True,
                f"daemon respawned (socket bound in {duration_ms} ms)",
                duration_ms,
            )
        time.sleep(_RESPAWN_POLL_INTERVAL_SEC)
    duration_ms = int((time.monotonic() - t0) * 1000)
    return (
        False,
        f"daemon respawn timed out (socket not bound after {_RESPAWN_BIND_TIMEOUT_SEC}s)",
        duration_ms,
    )


def _kill_dup_binders() -> tuple[bool, str, int]:
    import psutil

    t0 = time.monotonic()
    socket_path = _resolve_socket_path()
    try:
        result = subprocess.run(
            ["lsof", "-U", "-F", "pn"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return (
            False,
            f"lsof unavailable: {e}",
            int((time.monotonic() - t0) * 1000),
        )
    binder_pids = _extract_binder_pids(result.stdout, socket_path)
    if len(binder_pids) <= 1:
        return (
            True,
            f"{len(binder_pids)} dup binders to kill",
            int((time.monotonic() - t0) * 1000),
        )

    pid_etimes: list[tuple[int, float]] = []
    for pid in binder_pids:
        try:
            p = psutil.Process(pid)
            create_time = p.create_time()
            pid_etimes.append((pid, time.time() - create_time))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not pid_etimes:
        return (
            False,
            "all binders disappeared between lsof and psutil",
            int((time.monotonic() - t0) * 1000),
        )

    pid_etimes.sort(key=lambda x: x[1], reverse=True)
    keep_pid = pid_etimes[0][0]
    kill_candidates = [pid for pid, _ in pid_etimes[1:]]

    killed: list[int] = []
    for pid in kill_candidates:
        try:
            p = psutil.Process(pid)
            cmdline = " ".join(p.cmdline() or [])
            if "iai_mcp.daemon" not in cmdline:
                continue
            p.kill()
            killed.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    time.sleep(_LAUNCHD_REACT_DELAY_SEC)
    return (
        True,
        f"kept PID {keep_pid} (oldest); killed {killed}",
        int((time.monotonic() - t0) * 1000),
    )


def _plan_repair_actions(results: list[CheckResult]) -> list[RepairAction]:
    actions: list[RepairAction] = []
    fail_names = {r.name for r in results if not r.passed}

    if "(b) socket file fresh" in fail_names:
        actions.append(
            RepairAction(
                label="unlink_stale_socket",
                description="unlink stale ~/.iai-mcp/.daemon.sock",
                destructive=True,
                execute=_unlink_stale_socket,
            )
        )

    if "(g) no dup binders" in fail_names:
        actions.append(
            RepairAction(
                label="kill_dup_binders",
                description="keep oldest-etime daemon binder, SIGKILL the rest",
                destructive=True,
                execute=_kill_dup_binders,
            )
        )

    if "(d) no orphan iai_mcp.core procs" in fail_names:
        actions.append(
            RepairAction(
                label="kill_orphan_cores",
                description="SIGTERM every orphan iai_mcp.core process",
                destructive=True,
                execute=_kill_orphan_cores,
            )
        )

    if "(a) daemon process alive" in fail_names:
        actions.append(
            RepairAction(
                label="respawn_daemon",
                description="spawn `python -m iai_mcp.daemon` detached",
                destructive=True,
                execute=_respawn_daemon,
            )
        )

    return actions


def _prompt_action(action: RepairAction) -> bool:
    try:
        response = input(f"  [y/N] {action.description}: ")
    except EOFError:
        response = ""
    return response.strip().lower() == "y"


def cmd_doctor(args: argparse.Namespace) -> int:
    apply = bool(getattr(args, "apply", False))
    yes = bool(getattr(args, "yes", False))
    if yes and not apply:
        print(
            "[warn] --yes without --apply is meaningless; ignoring --yes.",
            file=sys.stderr,
        )

    results = run_diagnosis()
    headless = is_headless(force=bool(getattr(args, "headless", False)))
    results = _apply_headless_downgrade(results, headless)
    total = len(results)
    hint = _format_top_of_output_hint(results)
    if hint is not None:
        print(hint)
        print()
    print_checklist(results)
    fail_count = sum(1 for r in results if not r.passed)

    if fail_count == 0:
        print("\nAll checks passed. Exit 0.")
        return 0

    if not apply:
        print(
            f"\n{fail_count}/{total} FAIL. Run with --apply to attempt recovery. Exit 1."
        )
        return 1

    print(
        f"\n{fail_count}/{total} FAIL. Attempting recovery (--apply{' --yes' if yes else ''}):\n"
    )
    actions = _plan_repair_actions(results)
    if not actions:
        print(
            "(no automated repair actions for the FAILs above; manual intervention required)"
        )
    for action in actions:
        if action.destructive and not yes:
            if not _prompt_action(action):
                print(f"  [skipped] {action.description}")
                continue
        ok, msg, ms = action.execute()
        tag = "[done]" if ok else "[FAIL]"
        print(f"  {tag} {action.label}: {msg} ({ms} ms)")
        try:
            from iai_mcp.events import write_event
            from iai_mcp.store import MemoryStore

            write_event(
                MemoryStore(),
                kind="doctor_action",
                data={
                    "action": action.label,
                    "target": action.description,
                    "success": ok,
                    "duration_ms": ms,
                    "detail": msg,
                },
            )
        except Exception as e:
            logger.debug("doctor audit event write failed: %s", e)

    print("\nRe-running checks ...")
    final_results = run_diagnosis()
    print_checklist(final_results)
    final_fails = [r.name for r in final_results if not r.passed]
    if not final_fails:
        print(f"\nFIXED. All {len(final_results)} checks pass. Exit 0.")
        return 0
    print(f"\nSTILL BROKEN: {final_fails}. Exit 2.")
    return 2
