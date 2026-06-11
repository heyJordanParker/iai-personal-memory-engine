from __future__ import annotations

import argparse
import importlib.resources as _res
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


LOCK_PATH: Path = Path.home() / ".iai-mcp" / ".lock"
SOCKET_PATH: Path = Path.home() / ".iai-mcp" / ".daemon.sock"
STATE_PATH: Path = Path.home() / ".iai-mcp" / ".daemon-state.json"

LAUNCHD_TARGET: Path = Path.home() / "Library" / "LaunchAgents" / "com.iai-mcp.daemon.plist"
SYSTEMD_TARGET: Path = Path.home() / ".config" / "systemd" / "user" / "iai-mcp-daemon.service"

DAEMON_LABEL: str = "com.iai-mcp.daemon"
SERVICE_NAME: str = "iai-mcp-daemon.service"

CONSENT_BANNER: str = """\
==============================================================================
iai Sleep Daemon -- First Install Consent
==============================================================================

The sleep daemon runs in the background between Claude Code sessions to
perform neural consolidation (REM cycles, schema induction, drift detection).

Resource cost:
  - RAM: ~400 MB (bge-small-en-v1.5 embedding model kept warm to avoid cold-start)
  - CPU: brief bursts during REM cycles inside your learned quiet window
  - Disk: ~50MB/week in event logs + schema candidates

Claude subscription impact:
  - Max 1 `claude -p` call per night ("lucid moment" main insight)
  - Hard cap: 1% of daily subscription quota, 7% weekly buffer
  - ZERO API costs (no paid-API key -- uses your subscription only)

Opt out anytime:
  iai-mcp daemon uninstall

Continue? [y/N]: """


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _ensure_crypto_key_present():
    if os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
        return None
    from iai_mcp.crypto import KEY_BYTES, CryptoKey
    ck = CryptoKey(user_id="default")
    path = ck._key_file_path()
    if path.exists():
        return None
    import secrets as _secrets
    fresh = _secrets.token_bytes(KEY_BYTES)
    ck._try_file_set(fresh)
    print(f"crypto: created {path} (mode 0o600, {KEY_BYTES} bytes)")
    return path


def _launchd_template():
    return _res.files("iai_mcp") / "_deploy" / "launchd" / "com.iai-mcp.daemon.plist"


def _render_launchd_plist() -> str:
    text = _launchd_template().read_text()
    username = os.environ.get("USER") or Path.home().name
    text = text.replace("/usr/local/bin/python3", sys.executable)
    text = text.replace("{USERNAME}", username)
    return text


def _render_systemd_unit() -> str:
    tmpl = _res.files("iai_mcp") / "_deploy" / "systemd" / "iai-mcp-daemon.service"
    text = tmpl.read_text()
    text = text.replace("/usr/bin/python3", sys.executable)
    return text


def _try_short_timeout_connect(timeout_ms: int = 250) -> bool:
    import socket as _socket

    sock_path = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(SOCKET_PATH)
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    s.settimeout(timeout_ms / 1000.0)
    try:
        s.connect(sock_path)
        return True
    except (FileNotFoundError, ConnectionRefusedError, OSError, _socket.timeout):
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def _prompt_consent(stream_out=None) -> bool:
    if stream_out is None:
        stream_out = sys.stderr
    print(CONSENT_BANNER, file=stream_out, end="")
    stream_out.flush()
    try:
        response = input("")
    except EOFError:
        return False
    return response.strip().lower() == "y"


def _record_consent_receipt() -> None:
    state_dir = LOCK_PATH.parent
    state_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    payload = {
        "consent": True,
        "ts": ts,
        "executable": sys.executable,
        "platform": platform.system(),
        "user": os.environ.get("USER") or "",
    }
    safe_ts = ts.replace(":", "").replace("-", "").replace(".", "")
    receipt = state_dir / f".consent-{safe_ts}.json"
    try:
        receipt.write_text(json.dumps(payload, indent=2))
        os.chmod(receipt, 0o600)
    except OSError as exc:
        print(f"warning: could not write consent receipt: {exc}", file=sys.stderr)


def _remove_state_files() -> None:
    for p in (LOCK_PATH, SOCKET_PATH, STATE_PATH):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"warning: could not remove {p}: {exc}", file=sys.stderr)


_HOOK_TRUNCATION_TRAILER = "[... payload truncated to fit Claude Code 10000-char limit ...]"


def _truncate_for_claude_code_hook(text: str, cap: int = 10000) -> str:
    if len(text) <= cap:
        return text
    head_len = cap - len(_HOOK_TRUNCATION_TRAILER)
    if head_len <= 0:
        return _HOOK_TRUNCATION_TRAILER[:cap]
    return text[:head_len] + _HOOK_TRUNCATION_TRAILER


def _is_custom_store() -> bool:
    env_store = os.environ.get("IAI_MCP_STORE")
    if not env_store:
        return False
    from iai_mcp.store import DEFAULT_STORAGE_PATH as _DEFAULT

    try:
        custom = Path(env_store).expanduser().resolve()
        default = Path(_DEFAULT).expanduser().resolve()
        return custom != default
    except Exception:
        return False


def _send_jsonrpc_request(
    method: str,
    params: dict,
    *,
    connect_timeout: float = 5.0,
    read_timeout: float = 30.0,
) -> dict | None:
    import asyncio
    if not os.environ.get("IAI_DAEMON_SOCKET_PATH") and _is_custom_store():
        return None

    sock_path = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(SOCKET_PATH)

    async def _runner() -> dict | None:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(sock_path),
                timeout=connect_timeout,
            )
        except (FileNotFoundError, ConnectionRefusedError, OSError, asyncio.TimeoutError):
            return None
        try:
            req = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
            writer.write((json.dumps(req) + "\n").encode("utf-8"))
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=read_timeout)
            if not line:
                return None
            return json.loads(line.decode("utf-8"))
        except (OSError, asyncio.TimeoutError, ValueError) as exc:
            logger.debug("jsonrpc request failed: %s", exc)
            return None
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    try:
        return asyncio.run(_runner())
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("jsonrpc asyncio.run failed: %s", exc)
        return None


def cmd_session_start(args: argparse.Namespace) -> int:
    try:
        from iai_mcp.session import format_payload_as_markdown
        session_id = getattr(args, "session_id", "-") or "-"
        resp = _send_jsonrpc_request(
            "session_start_payload", {"session_id": session_id}
        )
        if not isinstance(resp, dict) or "result" not in resp:
            return 0
        result = resp.get("result")
        if not isinstance(result, dict):
            return 0
        rendered = format_payload_as_markdown(result)
        if not rendered:
            return 0
        sys.stdout.write(_truncate_for_claude_code_hook(rendered, cap=10000))
        return 0
    except Exception as exc:
        logger.error("session-start failed: %s", exc)
        return 0


def get_other_sessions_live_size(session_id: str) -> int:
    try:
        deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
        if not deferred_dir.exists():
            return 0
        own_name = f"{session_id}.live.jsonl"
        total = 0
        for entry in deferred_dir.iterdir():
            if not entry.is_file():
                continue
            if not entry.name.endswith(".live.jsonl"):
                continue
            if entry.name == own_name:
                continue
            try:
                total += entry.stat().st_size
            except OSError:
                pass
        return total
    except Exception:
        return 0


def read_live_fingerprint(session_id: str) -> int | None:
    p = Path.home() / ".iai-mcp" / ".capture-state" / f"{session_id}.live-fingerprint"
    try:
        if not p.exists():
            return None
        raw = p.read_text().strip()
        if not raw:
            return None
        return int(raw)
    except (OSError, ValueError):
        return None


def write_live_fingerprint(session_id: str, total_size: int) -> None:
    d = Path.home() / ".iai-mcp" / ".capture-state"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{session_id}.live-fingerprint.tmp"
    tmp.write_text(str(total_size))
    os.replace(tmp, d / f"{session_id}.live-fingerprint")


def get_max_created_at() -> str | None:
    import sqlite3 as _sqlite3

    db_path = Path.home() / ".iai-mcp" / "hippo" / "brain.sqlite3"
    if not db_path.exists():
        return None
    try:
        conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT MAX(created_at) FROM records WHERE tombstoned_at IS NULL"
            ).fetchone()
            return row[0] if row and row[0] else None
        finally:
            conn.close()
    except Exception:
        return None


def _utc_iso(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()
    except (TypeError, ValueError):
        return ts


def read_watermark(session_id: str) -> str | None:
    p = Path.home() / ".iai-mcp" / ".capture-state" / f"{session_id}.watermark"
    try:
        if not p.exists():
            return None
        return p.read_text().strip() or None
    except OSError:
        return None


def write_watermark(session_id: str, ts: str) -> None:
    d = Path.home() / ".iai-mcp" / ".capture-state"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{session_id}.watermark.tmp"
    tmp.write_text(_utc_iso(ts))
    os.replace(tmp, d / f"{session_id}.watermark")


def cmd_session_refresh_if_stale(args: argparse.Namespace) -> int:
    try:
        session_id: str = (getattr(args, "session_id", None) or "-")

        current = get_max_created_at()
        if current is None:
            return 0

        wm = read_watermark(session_id)
        live_size = get_other_sessions_live_size(session_id)

        if wm is None:
            write_watermark(session_id, current)
            write_live_fingerprint(session_id, live_size)
            return 0

        store_advanced = _utc_iso(current) > _utc_iso(wm)

        fp = read_live_fingerprint(session_id)
        if fp is None:
            write_live_fingerprint(session_id, live_size)
            fp = live_size
        live_grew = live_size > fp

        if not store_advanced and not live_grew:
            return 0

        resp = _send_jsonrpc_request(
            "session_refresh_if_stale",
            {"watermark": wm, "session_id": session_id},
            connect_timeout=5.0,
            read_timeout=30.0,
        )
        if resp is None:
            return 0

        result = resp.get("result") or {}
        rendered: str = result.get("rendered") or ""
        new_max: str = result.get("new_max_ts") or current

        if rendered:
            payload = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": rendered,
                }
            }
            sys.stdout.write(json.dumps(payload, ensure_ascii=False))
            write_watermark(session_id, new_max)
            write_live_fingerprint(session_id, live_size)

        return 0
    except Exception:
        return 0


def _send_socket_request(req: dict, *, timeout: float = 30.0) -> dict | None:
    import asyncio

    async def _runner() -> dict | None:
        _sock = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(SOCKET_PATH)
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(_sock),
                timeout=5.0,
            )
        except (FileNotFoundError, ConnectionRefusedError):
            return None
        except OSError:
            return None
        try:
            writer.write((json.dumps(req) + "\n").encode("utf-8"))
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not line:
                return None
            return json.loads(line.decode("utf-8"))
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    return asyncio.run(_runner())


def cmd_daemon_install(args: argparse.Namespace) -> int:
    dry_run = bool(getattr(args, "dry_run", False))
    yes = bool(getattr(args, "yes", False))

    if not yes and not dry_run:
        if not _prompt_consent():
            print("Install cancelled.", file=sys.stderr)
            return 1
        _record_consent_receipt()

    if _is_macos():
        content = _render_launchd_plist()
        target = LAUNCHD_TARGET
    elif _is_linux():
        content = _render_systemd_unit()
        target = SYSTEMD_TARGET
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1

    if dry_run:
        print(f"# Would install to: {target}")
        print(content)
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    try:
        os.chmod(target, 0o644)
    except OSError:
        pass

    _ensure_crypto_key_present()

    uid = os.getuid()
    if _is_macos():
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(target)],
            check=False, capture_output=True,
        )
        result = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(target)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0 and result.stderr:
            print(
                f"warning: launchctl bootstrap returned {result.returncode}: "
                f"{result.stderr.strip()}",
                file=sys.stderr,
            )
        subprocess.run(
            ["launchctl", "kickstart", f"gui/{uid}/{DAEMON_LABEL}"],
            check=False, capture_output=True,
        )
    else:
        user = os.environ.get("USER") or ""
        linger_probe = subprocess.run(
            ["loginctl", "show-user", user, "--property=Linger"],
            check=False, capture_output=True, text=True,
        )
        if "Linger=yes" not in linger_probe.stdout:
            subprocess.run(
                ["loginctl", "enable-linger", user],
                check=False, capture_output=True,
            )
            linger_recheck = subprocess.run(
                ["loginctl", "show-user", user, "--property=Linger"],
                check=False, capture_output=True, text=True,
            )
            if "Linger=yes" not in linger_recheck.stdout:
                print(
                    "WARNING: loginctl enable-linger did not take effect -- "
                    "daemon may die at logout",
                    file=sys.stderr,
                )
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=False, capture_output=True,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", SERVICE_NAME],
            check=False, capture_output=True,
        )

    print(f"Installed to {target}")
    return 0


def cmd_daemon_uninstall(args: argparse.Namespace) -> int:
    yes = bool(getattr(args, "yes", False))
    if not yes:
        try:
            response = input(
                "Uninstall iai daemon? "
                "(removes plist/unit + state files) [y/N]: "
            )
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            print("Uninstall cancelled.", file=sys.stderr)
            return 1

    uid = os.getuid()
    if _is_macos():
        if LAUNCHD_TARGET.exists():
            subprocess.run(
                ["launchctl", "bootout", f"gui/{uid}", str(LAUNCHD_TARGET)],
                check=False, capture_output=True,
            )
            try:
                LAUNCHD_TARGET.unlink()
            except OSError as exc:
                print(f"warning: could not remove plist: {exc}", file=sys.stderr)
    elif _is_linux():
        if SYSTEMD_TARGET.exists():
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", SERVICE_NAME],
                check=False, capture_output=True,
            )
            try:
                SYSTEMD_TARGET.unlink()
            except OSError as exc:
                print(f"warning: could not remove unit: {exc}", file=sys.stderr)
            subprocess.run(
                ["systemctl", "--user", "daemon-reload"],
                check=False, capture_output=True,
            )

    _remove_state_files()
    print("Daemon uninstalled. State files removed.")
    return 0


def cmd_daemon_start(args: argparse.Namespace) -> int:
    uid = os.getuid()
    if _is_macos():
        target = LAUNCHD_TARGET
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(target)],
            check=False, capture_output=True,
        )
        subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(target)],
            check=False, capture_output=True,
        )
        subprocess.run(
            ["launchctl", "kickstart", f"gui/{uid}/{DAEMON_LABEL}"],
            check=False, capture_output=True,
        )
    elif _is_linux():
        subprocess.run(
            ["systemctl", "--user", "start", SERVICE_NAME],
            check=False,
        )
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1
    return 0


STOP_TERM_TIMEOUT_S: float = 3.0
STOP_POLL_INTERVAL_S: float = 0.1


def _stop_escalation_bound() -> float:
    raw = os.environ.get("IAI_DAEMON_STOP_TIMEOUT_S")
    if raw:
        try:
            val = float(raw)
            if val >= 0:
                return val
        except ValueError:
            pass
    return STOP_TERM_TIMEOUT_S


def _stop_poll_interval() -> float:
    raw = os.environ.get("IAI_DAEMON_STOP_POLL_S")
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return STOP_POLL_INTERVAL_S


def cmd_daemon_stop(args: argparse.Namespace) -> int:
    import signal as _signal
    import time as _time

    try:
        from iai_mcp.daemon_state import load_state, save_state

        state = load_state()
        state["user_requested_shutdown"] = True
        save_state(state)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("sentinel write failed (non-blocking): %s", exc)

    uid = os.getuid()
    if _is_macos():
        from iai_mcp.lifecycle_lock import LifecycleLock, _is_pid_alive

        payload = LifecycleLock().read()
        pid = payload["pid"] if payload else None

        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(LAUNCHD_TARGET)],
            check=False, capture_output=True,
        )

        if pid is None:
            return 0

        if _is_pid_alive(pid):
            try:
                os.kill(pid, _signal.SIGTERM)
            except (ProcessLookupError, PermissionError) as exc:
                logger.debug("SIGTERM to daemon pid=%d failed: %s", pid, exc)
                return 0

            deadline = _time.monotonic() + _stop_escalation_bound()
            interval = _stop_poll_interval()
            while _time.monotonic() < deadline:
                if not _is_pid_alive(pid):
                    return 0
                _time.sleep(interval)

            if _is_pid_alive(pid):
                try:
                    os.kill(pid, _signal.SIGKILL)
                except (ProcessLookupError, PermissionError) as exc:
                    logger.debug("SIGKILL to daemon pid=%d failed: %s", pid, exc)
        return 0
    elif _is_linux():
        subprocess.run(
            ["systemctl", "--user", "stop", SERVICE_NAME],
            check=False,
        )
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1
    return 0


def compute_session_start_tokens_p90(store: "MemoryStore") -> dict[str, int | None]:
    import statistics

    from iai_mcp.events import query_events

    events = query_events(store, kind="session_started", limit=100)
    samples = [
        int(e["data"]["total_cached_tokens"])
        for e in events
        if isinstance(e.get("data"), dict) and "total_cached_tokens" in e["data"]
    ]
    if not samples:
        return {"p90": None, "n_samples": 0}
    if len(samples) == 1:
        return {"p90": samples[0], "n_samples": 1}
    q = statistics.quantiles(samples, n=10, method="inclusive")
    p90 = int(round(q[8]))
    return {"p90": p90, "n_samples": len(samples)}


def _compute_p90_from_events(events: list[dict]) -> dict[str, int | None]:
    import statistics

    samples = [
        int(e["data"]["total_cached_tokens"])
        for e in events
        if isinstance(e.get("data"), dict) and "total_cached_tokens" in e["data"]
    ]
    if not samples:
        return {"p90": None, "n_samples": 0}
    if len(samples) == 1:
        return {"p90": samples[0], "n_samples": 1}
    q = statistics.quantiles(samples, n=10, method="inclusive")
    p90 = int(round(q[8]))
    return {"p90": p90, "n_samples": len(samples)}


def _render_daemon_stats(result: dict[str, int | None]) -> None:
    p90_str = str(result["p90"]) if result["p90"] is not None else "no-data"
    print(f"session_start_tokens_p90: {p90_str}")
    print(f"n_samples: {result['n_samples']}")
    if 0 < (result["n_samples"] or 0) < 100:
        print(f"note: rolling window under-filled (have {result['n_samples']}, need 100)")


def cmd_daemon_stats(args: argparse.Namespace) -> int:
    resp = _send_jsonrpc_request("events_query", {"kind": "session_started", "limit": 100})
    if isinstance(resp, dict) and "result" in resp:
        payload = resp["result"]
        if isinstance(payload, dict) and "events" in payload:
            result = _compute_p90_from_events(payload["events"])
            _render_daemon_stats(result)
            return 0

    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.store import MemoryStore

    try:
        store_dir = Path(os.environ.get("IAI_MCP_STORE", Path.home() / ".iai-mcp"))
        store = MemoryStore(path=store_dir)
        result = compute_session_start_tokens_p90(store)
    except HippoLockHeldError:
        print("daemon holds store lock; retry when daemon is idle")
        return 0

    _render_daemon_stats(result)
    return 0


def cmd_daemon_status(args: argparse.Namespace) -> int:
    import asyncio
    try:
        resp = _send_socket_request({"type": "status"}, timeout=10.0)
    except asyncio.TimeoutError:
        print("daemon not responding", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- surface socket errors cleanly
        logger.error("daemon status failed: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if resp is None:
        print("daemon not running")
        return 1

    try:
        from iai_mcp import __version__ as installed_version
    except (ImportError, AttributeError):
        installed_version = "unknown"
    daemon_version = resp.get("version", "unknown")
    if (
        daemon_version != "unknown"
        and installed_version != "unknown"
        and daemon_version != installed_version
    ):
        print(
            f"WARNING: daemon version {daemon_version} != "
            f"installed {installed_version} -- run iai-mcp daemon "
            f"stop && iai-mcp daemon start to restart",
            file=sys.stderr,
        )

    for k, v in resp.items():
        print(f"{k}: {v}")
    return 0


def cmd_daemon_logs(args: argparse.Namespace) -> int:
    follow = bool(getattr(args, "follow", False))
    lines = int(getattr(args, "lines", 50))
    if _is_macos():
        path = Path.home() / "Library" / "Logs" / "iai-mcp-daemon.stderr.log"
        argv = ["tail"]
        if follow:
            argv.append("-f")
        argv.extend(["-n", str(lines), str(path)])
        subprocess.run(argv, check=False)
    elif _is_linux():
        argv = ["journalctl", "--user", "-u", SERVICE_NAME, "-n", str(lines)]
        if follow:
            argv.append("-f")
        subprocess.run(argv, check=False)
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1
    return 0


def cmd_daemon_force_rem(args: argparse.Namespace) -> int:
    import asyncio
    try:
        resp = _send_socket_request(
            {"type": "force_rem", "ts": datetime.now(timezone.utc).isoformat()},
            timeout=15 * 60,
        )
    except asyncio.TimeoutError:
        print("force_rem timed out after 15 minutes", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("force_rem failed: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if resp is None:
        print("daemon not running")
        return 1
    print(json.dumps(resp))
    return 0


def cmd_daemon_pause(args: argparse.Namespace) -> int:
    seconds = int(args.seconds)
    try:
        resp = _send_socket_request(
            {"type": "pause", "seconds": seconds}, timeout=10.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("pause failed: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if resp is None:
        print("daemon not running")
        return 1
    print(f"paused for {seconds}s")
    return 0


def cmd_daemon_resume(args: argparse.Namespace) -> int:
    try:
        resp = _send_socket_request({"type": "resume"}, timeout=10.0)
    except Exception as exc:  # noqa: BLE001
        logger.error("resume failed: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if resp is None:
        print("daemon not running")
        return 1
    print("resumed")
    return 0


def cmd_daemon_configure(args: argparse.Namespace) -> int:
    from iai_mcp.daemon_state import load_state, save_state

    key = args.key
    value = getattr(args, "value", None)
    state = load_state()

    if key == "set-budget":
        if value is None:
            print("set-budget requires a float value", file=sys.stderr)
            return 2
        state["daily_quota_pct_override"] = float(value)
    elif key == "set-cycle-count":
        if value is None:
            print("set-cycle-count requires an int value", file=sys.stderr)
            return 2
        state["cycle_count_override"] = int(value)
    elif key == "set-quiet-window":
        if value is None or "-" not in value:
            print(
                "set-quiet-window requires HH:MM-HH:MM format",
                file=sys.stderr,
            )
            return 2
        start, end = value.split("-", 1)
        state["quiet_window_manual_override"] = [start.strip(), end.strip()]
    elif key == "disable-claude":
        state["claude_enabled"] = False
    elif key == "enable-claude":
        state["claude_enabled"] = True
    else:
        print(f"unknown configure key: {key}", file=sys.stderr)
        return 2

    save_state(state)
    print(f"{key} -> {value if value is not None else 'toggled'}")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    from datetime import datetime as _dt

    from iai_mcp.tz import load_user_tz, to_local

    tz = load_user_tz()

    def _render_event(e: dict) -> None:
        ts_raw = e.get("ts")
        if isinstance(ts_raw, str):
            ts_raw = _dt.fromisoformat(ts_raw.replace("Z", "+00:00"))
        local = to_local(ts_raw, tz) if ts_raw is not None else None
        severity = e.get("severity") or "?"
        ts_str = local.isoformat() if local is not None else str(ts_raw)
        print(f"llm_health: {severity} at {ts_str} ({tz.key})")
        print(f"  data: {e.get('data', {})}")

    resp = _send_jsonrpc_request("events_query", {"kind": "llm_health", "limit": 1})
    if isinstance(resp, dict) and "result" in resp:
        payload = resp["result"]
        if isinstance(payload, dict) and "events" in payload:
            events = payload["events"]
            if not events:
                print("llm_health: no events recorded")
                return 0
            _render_event(events[0])
            return 0

    from iai_mcp.events import query_events
    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.store import MemoryStore

    try:
        store = MemoryStore()
        events = query_events(store, kind="llm_health", limit=1)
    except HippoLockHeldError:
        print("daemon holds store lock; retry when daemon is idle")
        return 0

    if not events:
        print("llm_health: no events recorded")
        return 0
    _render_event(events[0])
    return 0


def cmd_build_native(args: argparse.Namespace) -> int:
    import shutil

    if shutil.which("cargo") is None:
        print(
            "cargo not found on PATH.\n"
            "Install Rust: https://rustup.rs/",
            file=sys.stderr,
        )
        return 1

    repo_root = Path(__file__).resolve().parents[2]
    native_dir = repo_root / "rust" / "iai_mcp_native"
    if not native_dir.exists():
        print(
            f"Rust source not found at {native_dir}.\n"
            "Are you running from an installed wheel? "
            "build-native requires the repo checkout.",
            file=sys.stderr,
        )
        return 1

    cmd = [
        sys.executable, "-m", "maturin", "develop", "--release",
        "--manifest-path", str(native_dir / "Cargo.toml"),
    ]
    result = subprocess.run(cmd, cwd=str(repo_root))
    if result.returncode != 0:
        print(
            "\nbuild-native failed (see cargo output above).\n"
            "Common fix: rustup update",
            file=sys.stderr,
        )
        return result.returncode
    print("iai_mcp_native built successfully. Restart the daemon or MCP server.")
    return 0


def cmd_capture_transcript(args: argparse.Namespace) -> int:
    import json
    import sys as _sys

    no_spawn = bool(getattr(args, "no_spawn", False))

    if no_spawn:
        from iai_mcp.capture import write_deferred_captures

        try:
            out = write_deferred_captures(
                session_id=args.session_id,
                transcript_path=args.transcript_path,
                cwd=os.getcwd(),
                max_turns=args.max_turns,
            )
            print(json.dumps({"status": "deferred", "path": str(out)}, ensure_ascii=False))
            return 0
        except Exception as e:
            logger.error("capture-transcript --no-spawn failed: %s", e)
            print(
                f"capture-transcript --no-spawn: failed {type(e).__name__}: {e}",
                file=_sys.stderr,
            )
            return 0

    # Default path
    from iai_mcp.capture import capture_transcript
    from iai_mcp.store import MemoryStore

    try:
        store = MemoryStore()
        counts = capture_transcript(
            store,
            args.transcript_path,
            session_id=args.session_id,
            max_turns=args.max_turns,
        )
        print(json.dumps(counts, ensure_ascii=False))
        return 0
    except Exception as e:
        logger.error("capture-transcript inline failed: %s", e)
        print(f"capture-transcript: failed {type(e).__name__}: {e}", file=_sys.stderr)
        return 0


def cmd_capture_turn_deferred(args: argparse.Namespace) -> int:
    import sys as _sys

    try:
        from iai_mcp.capture import _parse_transcript_line, write_deferred_event

        transcript = Path(args.transcript_path).expanduser()
        if not transcript.exists():
            return 0

        state_dir = Path.home() / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        offset_path = state_dir / f"{args.session_id}.offset"

        prev_offset = 0
        if offset_path.exists():
            try:
                prev_offset = int(offset_path.read_text().strip() or "0")
            except ValueError:
                prev_offset = 0

        with transcript.open() as fh:
            all_lines = fh.readlines()
        total = len(all_lines)

        if prev_offset > total:
            prev_offset = 0

        new_lines = all_lines[prev_offset:]
        consumed = 0
        emitted = 0
        max_emit = int(getattr(args, "max_turns_per_call", 200))
        cwd = os.getcwd()
        for line in new_lines:
            if emitted >= max_emit:
                break
            consumed += 1
            parsed = _parse_transcript_line(line)
            if parsed is None:
                continue
            role, text, src_uuid, src_ts = parsed
            write_deferred_event(
                args.session_id, role, text,
                cwd=cwd,
                ts=src_ts,
                source_uuid=src_uuid,
            )
            emitted += 1

        new_offset = prev_offset + consumed
        tmp_path = offset_path.parent / (offset_path.name + ".tmp")
        tmp_path.write_text(str(new_offset))
        os.replace(tmp_path, offset_path)
        return 0
    except Exception as e:
        logger.error("capture-turn-deferred failed: %s", e)
        print(
            f"capture-turn-deferred: failed {type(e).__name__}: {e}",
            file=_sys.stderr,
        )
        return 0


def _capture_hook_paths() -> tuple:
    src = _res.files("iai_mcp") / "_deploy" / "hooks" / "iai-mcp-session-capture.sh"
    dst = Path.home() / ".claude" / "hooks" / "iai-mcp-session-capture.sh"
    settings = Path.home() / ".claude" / "settings.json"
    return src, dst, settings


def _turn_hook_paths() -> tuple:
    src = _res.files("iai_mcp") / "_deploy" / "hooks" / "iai-mcp-turn-capture.sh"
    dst = Path.home() / ".claude" / "hooks" / "iai-mcp-turn-capture.sh"
    return src, dst


def _claude_desktop_config_path() -> Path | None:
    import platform as _plat
    home = Path.home()
    sysname = _plat.system()
    if sysname == "Darwin":
        p = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif sysname == "Windows":
        appdata = os.environ.get("APPDATA") or str(home / "AppData" / "Roaming")
        p = Path(appdata) / "Claude" / "claude_desktop_config.json"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME") or str(home / ".config")
        p = Path(xdg) / "Claude" / "claude_desktop_config.json"
    return p if p.parent.exists() else None


def _resolve_wrapper_path() -> Path:
    import iai_mcp as _pkg

    env_val = os.environ.get("IAI_MCP_WRAPPER_PATH")
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"IAI_MCP_WRAPPER_PATH={env_val!r} is set but the file does not exist."
        )

    try:
        pkg_p = Path(str(_res.files("iai_mcp") / "_wrapper" / "index.js"))
        if pkg_p.exists():
            return pkg_p
    except (TypeError, FileNotFoundError):
        pass

    src_file = Path(_pkg.__file__).resolve()
    repo_root = src_file.parent.parent.parent
    editable_path = repo_root / "mcp-wrapper" / "dist" / "index.js"
    if editable_path.exists():
        return editable_path

    raise FileNotFoundError(
        "MCP wrapper (index.js) not found. Checked locations:\n"
        f"  1. IAI_MCP_WRAPPER_PATH env var (not set)\n"
        f"  2. Package data: {str(_res.files('iai_mcp') / '_wrapper' / 'index.js')}\n"
        f"  3. Editable source: {editable_path}\n"
        "To build: cd mcp-wrapper && npm run build\n"
        "Or run: bash scripts/install.sh\n"
        "For packaged installs: reinstall the wheel (it should include the wrapper)."
    )


def _build_iai_mcp_server_entry() -> dict:
    wrapper = _resolve_wrapper_path()
    return {
        "command": "node",
        "args": [str(wrapper)],
        "env": {
            "IAI_MCP_PYTHON": sys.executable,
            "IAI_MCP_STORE": str(Path.home() / ".iai-mcp"),
            "TRANSFORMERS_VERBOSITY": "error",
            "TOKENIZERS_PARALLELISM": "false",
        },
    }


def _patch_claude_desktop_config(action: str) -> str:
    import json as _json

    cfg_path = _claude_desktop_config_path()
    if cfg_path is None:
        return "Claude Desktop: not installed (no config dir) — skipped"

    if not cfg_path.exists():
        if action == "uninstall":
            return f"Claude Desktop: {cfg_path} absent — skipped"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"mcpServers": {"iai-mcp": _build_iai_mcp_server_entry()}}
        cfg_path.write_text(_json.dumps(data, indent=2))
        return f"Claude Desktop: created {cfg_path} with iai-mcp registered"

    try:
        data = _json.loads(cfg_path.read_text())
    except (OSError, ValueError) as e:
        return f"Claude Desktop: {cfg_path} unreadable ({type(e).__name__}) — skipped"

    servers = data.setdefault("mcpServers", {})

    if action == "uninstall":
        if "iai-mcp" in servers:
            servers.pop("iai-mcp", None)
            cfg_path.write_text(_json.dumps(data, indent=2))
            return f"Claude Desktop: removed iai-mcp from {cfg_path}"
        return f"Claude Desktop: iai-mcp not in config — no change"

    new_entry = _build_iai_mcp_server_entry()
    if servers.get("iai-mcp") == new_entry:
        return f"Claude Desktop: {cfg_path} already has iai-mcp — no change"
    servers["iai-mcp"] = new_entry
    cfg_path.write_text(_json.dumps(data, indent=2))
    return f"Claude Desktop: patched {cfg_path} (iai-mcp registered)"


def _patch_claude_code_config(action: str) -> str:
    import json as _json

    cfg_path = Path.home() / ".claude.json"

    if action == "uninstall":
        if not cfg_path.exists():
            return "Claude Code: ~/.claude.json absent — skipped"
        try:
            data = _json.loads(cfg_path.read_text())
        except (OSError, ValueError) as e:
            return f"Claude Code: ~/.claude.json unreadable ({type(e).__name__}) — skipped"
        servers = data.get("mcpServers", {})
        if "iai-mcp" in servers:
            servers.pop("iai-mcp")
            data["mcpServers"] = servers
            cfg_path.write_text(_json.dumps(data, indent=2))
            return "Claude Code: removed iai-mcp from ~/.claude.json"
        return "Claude Code: iai-mcp not in ~/.claude.json — no change"

    try:
        entry = _build_iai_mcp_server_entry()
    except FileNotFoundError as exc:
        entry = {
            "type": "stdio",
            "command": "node",
            "args": ["<run: cd mcp-wrapper && npm run build>"],
            "env": {
                "IAI_MCP_PYTHON": sys.executable,
                "IAI_MCP_STORE": str(Path.home() / ".iai-mcp"),
                "TRANSFORMERS_VERBOSITY": "error",
                "TOKENIZERS_PARALLELISM": "false",
            },
        }
        print(
            f"WARN: MCP wrapper not found — ~/.claude.json entry written with "
            f"placeholder args. Build it first: cd mcp-wrapper && npm run build. "
            f"({exc})",
            file=sys.stderr,
        )
    else:
        entry.setdefault("type", "stdio")

    if not cfg_path.exists():
        cfg_path.write_text(_json.dumps({"mcpServers": {"iai-mcp": entry}}, indent=2))
        return "Claude Code: created ~/.claude.json with iai-mcp registered"

    try:
        data = _json.loads(cfg_path.read_text())
    except (OSError, ValueError) as e:
        return f"Claude Code: ~/.claude.json unreadable ({type(e).__name__}) — skipped"

    servers = data.setdefault("mcpServers", {})
    if servers.get("iai-mcp") == entry:
        return "Claude Code: ~/.claude.json already has iai-mcp — no change"
    servers["iai-mcp"] = entry
    cfg_path.write_text(_json.dumps(data, indent=2))
    return "Claude Code: patched ~/.claude.json (iai-mcp registered)"


_CAPTURE_HOOK_MARKER = "iai-mcp-session-capture.sh"
_TURN_HOOK_MARKER = "iai-mcp-turn-capture.sh"
_SESSION_RECALL_HOOK_MARKER = "iai-mcp-session-recall.sh"


def _session_recall_hook_paths() -> tuple:
    src = _res.files("iai_mcp") / "_deploy" / "hooks" / "iai-mcp-session-recall.sh"
    dst = Path.home() / ".claude" / "hooks" / "iai-mcp-session-recall.sh"
    settings = Path.home() / ".claude" / "settings.json"
    return src, dst, settings


def _load_settings(path):
    import json as _json
    if not path.exists():
        return {}
    try:
        return _json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def cmd_capture_hooks_install(args: argparse.Namespace) -> int:
    import json as _json
    import stat

    src, dst, settings = _capture_hook_paths()
    turn_src, turn_dst = _turn_hook_paths()

    if not src.exists():
        print(f"ERROR: hook template missing in package data: {src}", file=sys.stderr)
        return 1
    if not turn_src.exists():
        print(f"ERROR: turn-hook template missing in package data: {turn_src}", file=sys.stderr)
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"installed: {dst}")

    turn_dst.parent.mkdir(parents=True, exist_ok=True)
    turn_dst.write_bytes(turn_src.read_bytes())
    turn_dst.chmod(turn_dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"installed: {turn_dst}")

    settings.parent.mkdir(parents=True, exist_ok=True)
    data = _load_settings(settings)
    data.setdefault("hooks", {})
    stop_list = data["hooks"].setdefault("Stop", [])
    submit_list = data["hooks"].setdefault("UserPromptSubmit", [])

    hook_cmd = f"bash {dst}"
    turn_cmd = f"bash {turn_dst}"

    already_stop = any(
        any(_CAPTURE_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in stop_list
    )
    if already_stop:
        print(f"settings.json already has Stop hook — no change")
    else:
        stop_list.append({"hooks": [{"type": "command", "command": hook_cmd, "timeout": 35}]})
        print(f"patched: {settings} (Stop hook registered)")

    already_turn = any(
        any(_TURN_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in submit_list
    )
    if already_turn:
        print(f"settings.json already has UserPromptSubmit hook — no change")
    else:
        submit_list.append({"hooks": [{"type": "command", "command": turn_cmd, "timeout": 5}]})
        print(f"patched: {settings} (UserPromptSubmit hook registered)")

    src_recall, dst_recall, _ = _session_recall_hook_paths()
    if src_recall.exists():
        dst_recall.parent.mkdir(parents=True, exist_ok=True)
        dst_recall.write_bytes(src_recall.read_bytes())
        dst_recall.chmod(dst_recall.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        print(f"installed: {dst_recall}")

        ss_list = data["hooks"].setdefault("SessionStart", [])
        recall_cmd = f"bash {dst_recall}"
        already_recall = any(
            any(_SESSION_RECALL_HOOK_MARKER in (h.get("command") or "")
                for h in (entry.get("hooks") or []))
            for entry in ss_list
        )
        if already_recall:
            print("settings.json already has SessionStart hook — no change")
        else:
            ss_list.append({
                "matcher": "startup|resume|clear|compact",
                "hooks": [{"type": "command", "command": recall_cmd, "timeout": 30}],
            })
            print(f"patched: {settings} (SessionStart hook registered)")
    else:
        print(f"WARN: recall hook template missing in package data: {src_recall}")

    settings.write_text(_json.dumps(data, indent=2))

    code_msg = _patch_claude_code_config("install")
    print(code_msg)
    desktop_msg = _patch_claude_desktop_config("install")
    print(desktop_msg)

    print("\nNext: fully quit + relaunch Claude Code AND Claude Desktop")
    print("      so both pick up the registration (macOS: `killall Claude`).")
    print("Verify: iai-mcp capture-hooks status")
    return 0


def cmd_capture_hooks_uninstall(args: argparse.Namespace) -> int:
    import json as _json

    _, dst, settings = _capture_hook_paths()
    _, turn_dst = _turn_hook_paths()
    _, dst_recall, _ = _session_recall_hook_paths()

    if dst.exists():
        dst.unlink()
        print(f"removed: {dst}")
    else:
        print(f"(not present) {dst}")

    if turn_dst.exists():
        turn_dst.unlink()
        print(f"removed: {turn_dst}")
    else:
        print(f"(not present) {turn_dst}")

    if dst_recall.exists():
        dst_recall.unlink()
        print(f"removed: {dst_recall}")
    else:
        print(f"(not present) {dst_recall}")

    if settings.exists():
        data = _load_settings(settings)
        changed = False
        for key, marker in (
            ("Stop", _CAPTURE_HOOK_MARKER),
            ("UserPromptSubmit", _TURN_HOOK_MARKER),
        ):
            entries = data.get("hooks", {}).get(key, [])
            kept = [
                entry for entry in entries
                if not any(marker in (h.get("command") or "")
                           for h in (entry.get("hooks") or []))
            ]
            if len(kept) != len(entries):
                if kept:
                    data["hooks"][key] = kept
                else:
                    data["hooks"].pop(key, None)
                changed = True
                print(f"patched: {settings} ({key} entry removed)")
        if changed:
            settings.write_text(_json.dumps(data, indent=2))
        else:
            print(f"(no hook entry to remove) {settings}")

        data = _load_settings(settings)
        ss_list = data.get("hooks", {}).get("SessionStart", [])
        kept_ss = [
            entry for entry in ss_list
            if not any(_SESSION_RECALL_HOOK_MARKER in (h.get("command") or "")
                       for h in (entry.get("hooks") or []))
        ]
        if len(kept_ss) != len(ss_list):
            if kept_ss:
                data["hooks"]["SessionStart"] = kept_ss
            else:
                data["hooks"].pop("SessionStart", None)
            settings.write_text(_json.dumps(data, indent=2))
            print(f"patched: {settings} (SessionStart entry removed)")
        else:
            print(f"(no SessionStart entry to remove) {settings}")

    code_msg = _patch_claude_code_config("uninstall")
    print(code_msg)
    desktop_msg = _patch_claude_desktop_config("uninstall")
    print(desktop_msg)

    return 0


def cmd_capture_hooks_status(args: argparse.Namespace) -> int:
    import json as _json
    src, dst, settings = _capture_hook_paths()
    turn_src, turn_dst = _turn_hook_paths()
    src_recall, dst_recall, _ = _session_recall_hook_paths()

    print(f"Stop template:        {src}  {'PRESENT' if src.exists() else 'MISSING'}")
    print(f"Stop installed:       {dst}  {'PRESENT' if dst.exists() else 'MISSING'}")
    print(f"Turn template:        {turn_src}  {'PRESENT' if turn_src.exists() else 'MISSING'}")
    print(f"Turn installed:       {turn_dst}  {'PRESENT' if turn_dst.exists() else 'MISSING'}")
    print(f"Recall template:      {src_recall}  {'PRESENT' if src_recall.exists() else 'MISSING'}")
    print(f"Recall installed:     {dst_recall}  {'PRESENT' if dst_recall.exists() else 'MISSING'}")

    data = _load_settings(settings)
    stop_list = data.get("hooks", {}).get("Stop", [])
    submit_list = data.get("hooks", {}).get("UserPromptSubmit", [])
    wired = any(
        any(_CAPTURE_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in stop_list
    )
    turn_wired = any(
        any(_TURN_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in submit_list
    )
    ss_list = data.get("hooks", {}).get("SessionStart", [])
    recall_wired = any(
        any(_SESSION_RECALL_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in ss_list
    )
    print(f"Claude Code settings.json Stop:             {settings}  {'WIRED' if wired else 'NOT WIRED'}")
    print(f"Claude Code settings.json UserPromptSubmit: {settings}  {'WIRED' if turn_wired else 'NOT WIRED'}")
    print(f"Claude Code settings.json SessionStart:     {settings}  {'WIRED' if recall_wired else 'NOT WIRED'}")

    desktop_cfg = _claude_desktop_config_path()
    if desktop_cfg is None:
        desktop_line = "Claude Desktop: not installed"
        desktop_wired = False
    elif not desktop_cfg.exists():
        desktop_line = f"Claude Desktop: {desktop_cfg} MISSING"
        desktop_wired = False
    else:
        try:
            d = _json.loads(desktop_cfg.read_text())
            desktop_wired = "iai-mcp" in d.get("mcpServers", {})
            desktop_line = f"Claude Desktop: {desktop_cfg}  {'WIRED' if desktop_wired else 'NOT WIRED'}"
        except (OSError, ValueError):
            desktop_line = f"Claude Desktop: {desktop_cfg} (unreadable)"
            desktop_wired = False
    print(desktop_line)

    ok = (
        dst.exists() and wired
        and turn_dst.exists() and turn_wired
        and dst_recall.exists() and recall_wired
    )
    desktop_problem = desktop_cfg is not None and desktop_cfg.exists() and not desktop_wired

    if ok and not desktop_problem:
        print(f"\nstatus: ACTIVE — Stop + UserPromptSubmit + SessionStart hooks wired "
              f"(Claude Code{'; Desktop also wired' if desktop_wired else ''})")
        return 0
    msg = []
    if not ok:
        msg.append("Claude Code not fully wired")
    if desktop_problem:
        msg.append("Claude Desktop present but iai-mcp not registered")
    print(f"\nstatus: INACTIVE — {'; '.join(msg)}. Run: iai-mcp capture-hooks install")
    return 1


def cmd_migrate(args: argparse.Namespace) -> int:
    from iai_mcp.store import MemoryStore
    store = MemoryStore()

    if bool(getattr(args, "rollback", False)):
        from iai_mcp import migrate
        return migrate._rollback(store.db, store)
    if bool(getattr(args, "resume", False)):
        from iai_mcp import migrate
        from iai_mcp.embed import embedder_for_store
        target = embedder_for_store(store)
        return migrate._resume(store.db, store, target)

    if bool(getattr(args, "rederive_timestamps", False)):
        from iai_mcp.migrate import migrate_rederive_collapsed_timestamps
        dry_run = bool(getattr(args, "dry_run", False))
        result = migrate_rederive_collapsed_timestamps(store, dry_run=dry_run)
        prefix = "[dry-run] would update" if dry_run else "updated"
        print(
            f"{prefix} {result['records_updated']} records; "
            f"skipped_no_transcript={result['skipped_no_transcript']} "
            f"skipped_no_match={result['skipped_no_match']}"
        )
        return 0

    from_v = int(getattr(args, "from_", 1))
    to_v = int(getattr(args, "to", 2))
    dry_run = bool(getattr(args, "dry_run", False))
    verbose = bool(getattr(args, "verbose", False))

    def _progress(i: int, n: int) -> None:
        if verbose:
            print(f"[{i + 1}/{n}] migrating...")

    if from_v == 1 and to_v == 2:
        from iai_mcp.migrate import migrate_v1_to_v2
        result = migrate_v1_to_v2(store, dry_run=dry_run, progress=_progress)
        prefix = "would migrate" if dry_run else "migrated"
        print(
            f"{prefix} {result['records_migrated']} records in "
            f"{result['duration_sec']:.2f}s "
            f"({result['previous_model']} -> {result['new_model']})"
        )
        return 0

    if from_v == 2 and to_v == 3:
        from iai_mcp.migrate import migrate_encryption_v2_to_v3
        result = migrate_encryption_v2_to_v3(
            store, dry_run=dry_run, progress=_progress
        )
        prefix = "would encrypt" if dry_run else "encrypted"
        print(
            f"{prefix} {result['records_migrated']} records + "
            f"{result['events_migrated']} events in "
            f"{result['duration_sec']:.2f}s "
            f"(AES-256-GCM, iai:enc:v1:)"
        )
        return 0

    if from_v == 3 and to_v == 4:
        from iai_mcp.migrate import migrate_hd_vector_to_structure_hv_v3_to_v4
        result = migrate_hd_vector_to_structure_hv_v3_to_v4(
            store, dry_run=dry_run, progress=_progress
        )
        prefix = "would rename" if dry_run else "renamed"
        print(
            f"{prefix} {result['updated']} records' "
            f"hd_vector_json->structure_hv column in "
            f"{result['duration_ms'] / 1000:.2f}s "
            f"(schema v3->v4, TEM factorization, D=10000 BSC packed)"
        )
        return 0

    print(
        f"unsupported migration --from={from_v} --to={to_v}; "
        f"supported: 1->2 (schema), 2->3 (encryption), "
        f"3->4 (TEM factorization)",
        file=sys.stderr,
    )
    return 2


def cmd_crypto_status(args: argparse.Namespace) -> int:
    import json as _json
    import os as _os

    from iai_mcp.crypto import CIPHERTEXT_PREFIX, CryptoKey, KEY_BYTES

    user_id = getattr(args, "user_id", None) or "default"
    ck = CryptoKey(user_id=user_id)
    path = ck._key_file_path()

    present = path.exists()
    status: dict[str, object] = {
        "user_id": user_id,
        "backend": "file",
        "path": str(path),
        "present": present,
        "algorithm": "AES-256-GCM",
        "format": CIPHERTEXT_PREFIX,
    }

    if present:
        st = path.stat()
        mode_octal = f"0o{st.st_mode & 0o777:03o}"
        length = st.st_size
        status["mode"] = mode_octal
        status["mode_secure"] = (st.st_mode & 0o077 == 0)
        status["uid"] = st.st_uid
        status["uid_matches_process"] = (st.st_uid == _os.geteuid())
        status["length_bytes"] = length
        status["length_valid"] = (length == KEY_BYTES)
        status["passphrase_fallback_set"] = bool(
            _os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE")
        )
    else:
        status["passphrase_fallback_set"] = bool(
            _os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE")
        )
        status["hint"] = (
            "no key file. Run `iai-mcp crypto migrate-to-file` "
            "(existing Keychain key) or `iai-mcp crypto init` "
            "(fresh install), or set IAI_MCP_CRYPTO_PASSPHRASE."
        )

    print(_json.dumps(status, indent=2))
    return 0


def cmd_crypto_rotate(args: argparse.Namespace) -> int:
    import json as _json

    from iai_mcp.crypto import encrypt_field
    from iai_mcp.store import (
        EVENTS_TABLE,
        MemoryStore,
        RECORDS_TABLE,
        _uuid_literal,
    )

    user_id = getattr(args, "user_id", None) or "default"
    store = MemoryStore(user_id=user_id)

    decrypted_records = store.all_records()

    events_tbl = store.db.open_table(EVENTS_TABLE)
    events_df = events_tbl.to_pandas()
    decrypted_events: list[dict] = []
    from iai_mcp.crypto import decrypt_field, is_encrypted
    for _, row in events_df.iterrows():
        raw = row.get("data_json") or "{}"
        eid = str(row["id"])
        if is_encrypted(raw):
            try:
                raw = decrypt_field(
                    raw, store._key(), associated_data=eid.encode("ascii")
                )
            except (OSError, ValueError, RuntimeError):
                raw = "{}"
        decrypted_events.append({"id": eid, "data_json": raw})

    new_key = store._crypto_key_wrapper.rotate()
    store._crypto_key = new_key
    store._invalidate_aesgcm_cache()

    tbl = store.db.open_table(RECORDS_TABLE)
    record_count = 0
    for rec in decrypted_records:
        try:
            tbl.delete(f"id = '{_uuid_literal(rec.id)}'")
        except (OSError, ValueError, RuntimeError):
            pass
        try:
            store.insert(rec)
            record_count += 1
        except (OSError, ValueError, RuntimeError):
            continue

    event_count = 0
    for ev in decrypted_events:
        ad = ev["id"].encode("ascii")
        new_ct = encrypt_field(ev["data_json"], new_key, associated_data=ad)
        try:
            events_tbl.update(
                where=f"id = '{ev['id']}'",
                values={"data_json": new_ct},
            )
            event_count += 1
        except (OSError, ValueError, RuntimeError):
            continue

    print(
        _json.dumps(
            {
                "status": "rotated",
                "user_id": user_id,
                "records_re_encrypted": record_count,
                "events_re_encrypted": event_count,
                "algorithm": "AES-256-GCM",
                "format": "iai:enc:v1:",
            },
            indent=2,
        )
    )
    try:
        from iai_mcp.crypto_key_watch import sync_crypto_key_watcher_to_disk
        from iai_mcp.events import write_event

        write_event(
            store,
            kind="crypto_key_rotated",
            data={
                "source": "cli_rotate",
                "records_re_encrypted": record_count,
                "events_re_encrypted": event_count,
            },
            severity="info",
        )
        sync_crypto_key_watcher_to_disk(store)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("crypto rotate audit event failed: %s", exc)
    return 0


def cmd_crypto_recover_prior_key(args: argparse.Namespace) -> int:
    import json as _json

    from iai_mcp.crypto import KEY_BYTES
    from iai_mcp.migrate import migrate_crypto_recover_prior_key
    from iai_mcp.store import MemoryStore

    path: Path = args.prior_key_file
    try:
        prior = path.read_bytes()
    except OSError as exc:
        print(f"cannot read prior key file: {exc}", file=sys.stderr)
        return 1
    if len(prior) != KEY_BYTES:
        print(
            f"prior key file must be exactly {KEY_BYTES} bytes, got {len(prior)}",
            file=sys.stderr,
        )
        return 1
    user_id = getattr(args, "user_id", None) or "default"
    store = MemoryStore(user_id=user_id)
    try:
        out = migrate_crypto_recover_prior_key(
            store, prior, dry_run=bool(getattr(args, "dry_run", False)),
        )
    except Exception as exc:
        logger.error("crypto recover-prior-key failed: %s", exc)
        print(str(exc), file=sys.stderr)
        return 1
    print(_json.dumps(out, indent=2, default=str))
    return 0


def cmd_crypto_redact_undecryptable(args: argparse.Namespace) -> int:
    import json as _json

    from iai_mcp.migrate import migrate_redact_undecryptable_records
    from iai_mcp.store import MemoryStore

    user_id = getattr(args, "user_id", None) or "default"
    store = MemoryStore(user_id=user_id)
    try:
        out = migrate_redact_undecryptable_records(store)
    except Exception as exc:
        logger.error("crypto redact-undecryptable failed: %s", exc)
        print(str(exc), file=sys.stderr)
        return 1
    print(_json.dumps(out, indent=2, default=str))
    return 0


def cmd_crypto_migrate_to_file(args: argparse.Namespace) -> int:
    import base64 as _b64
    import keyring as _keyring
    import keyring.errors as _keyring_errors

    from iai_mcp.crypto import (
        CryptoKey,
        CryptoKeyError,
        KEY_BYTES,
        SERVICE_NAME_DEFAULT,
    )

    user_id = getattr(args, "user_id", None) or "default"
    keep_keychain = getattr(args, "keep_keychain", True)

    ck = CryptoKey(user_id=user_id)

    try:
        existing = ck._try_file_get()
    except CryptoKeyError as exc:
        print(
            f"refusing: existing key file is malformed: {exc}",
            file=sys.stderr,
        )
        return 1
    if existing is not None:
        print(f"already migrated: {ck._key_file_path()}")
        return 0

    try:
        encoded = _keyring.get_password(SERVICE_NAME_DEFAULT, user_id)
    except _keyring_errors.NoKeyringError:
        print(
            "no keyring backend available; nothing to migrate. "
            "If this is a fresh install, run `iai-mcp crypto init` instead.",
            file=sys.stderr,
        )
        return 1
    except _keyring_errors.KeyringError as exc:
        print(f"keyring read failed: {exc}", file=sys.stderr)
        return 1
    if encoded is None:
        print(
            f"no key found in keyring for user_id={user_id!r}. "
            f"If this is a fresh install, run `iai-mcp crypto init` instead.",
            file=sys.stderr,
        )
        return 1

    try:
        source = _b64.urlsafe_b64decode(encoded.encode("ascii"))
    except (ValueError, TypeError) as exc:
        print(f"keyring entry is malformed: {exc}", file=sys.stderr)
        return 1
    if len(source) != KEY_BYTES:
        print(
            f"keyring entry has wrong length {len(source)} (expected {KEY_BYTES})",
            file=sys.stderr,
        )
        return 1

    try:
        ck._try_file_set(source)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"failed to write key file: {exc}", file=sys.stderr)
        return 1

    try:
        roundtrip = ck._try_file_get()
    except CryptoKeyError as exc:
        try:
            ck._key_file_path().unlink()
        except OSError:
            pass
        print(f"round-trip verification failed: {exc}", file=sys.stderr)
        return 1
    if roundtrip != source:
        try:
            ck._key_file_path().unlink()
        except OSError:
            pass
        print(
            "round-trip verification failed: bytes differ", file=sys.stderr
        )
        return 1

    path = ck._key_file_path()
    print(f"migrated: {path} (mode 0o600, {KEY_BYTES} bytes)")

    if not keep_keychain:
        try:
            _keyring.delete_password(SERVICE_NAME_DEFAULT, user_id)
            print(f"deleted keyring entry for user_id={user_id!r}")
        except _keyring_errors.PasswordDeleteError:
            pass
        except _keyring_errors.KeyringError as exc:
            print(
                f"warning: failed to delete keyring entry: {exc}",
                file=sys.stderr,
            )
    else:
        print(
            "keyring entry kept (default). "
            "To remove manually, run "
            "`iai-mcp crypto migrate-to-file --delete-keychain` "
            "or use macOS Keychain Access.app."
        )

    return 0


def cmd_crypto_init(args: argparse.Namespace) -> int:
    import secrets as _secrets

    from iai_mcp.crypto import CryptoKey, KEY_BYTES

    user_id = getattr(args, "user_id", None) or "default"
    ck = CryptoKey(user_id=user_id)
    path = ck._key_file_path()
    if path.exists():
        print(
            f"refusing: key file already exists at {path}. "
            f"To rotate, run `iai-mcp crypto rotate`. "
            f"To wipe and start over, remove the file manually first.",
            file=sys.stderr,
        )
        return 1
    fresh = _secrets.token_bytes(KEY_BYTES)
    ck._try_file_set(fresh)
    print(f"created: {path} (mode 0o600, {KEY_BYTES} bytes)")
    return 0


def cmd_bank_recall(args: argparse.Namespace) -> int:
    import json as _json

    from iai_mcp.memory_bank import bank_recall_substring

    include_processed = not getattr(args, "recent_only", False)
    include_recent = not getattr(args, "processed_only", False)

    result = bank_recall_substring(
        args.query,
        limit=args.limit,
        include_processed=include_processed,
        include_recent=include_recent,
    )
    print(_json.dumps(result, ensure_ascii=False))
    return 0


def cmd_topology(args: argparse.Namespace) -> int:  # noqa: ARG001 -- argparse contract

    def _fmt(v) -> str:
        if v is None:
            return "insufficient_data"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def _render(d: dict) -> None:
        print(f"C: {_fmt(d.get('C'))}")
        print(f"L: {_fmt(d.get('L'))}")
        print(f"sigma: {_fmt(d.get('sigma'))}")
        print(f"communities: {_fmt(d.get('community_count'))}")
        print(f"rich_club_ratio: {_fmt(d.get('rich_club_ratio'))}")
        print(f"N: {_fmt(d.get('N'))}")
        print(f"regime: {_fmt(d.get('regime'))}")

    resp = _send_jsonrpc_request("topology", {})
    if isinstance(resp, dict):
        result = resp.get("result")
        if isinstance(result, dict):
            _render(result)
            return 0

    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.sigma import compute_topology_snapshot
    from iai_mcp.store import MemoryStore

    try:
        store = MemoryStore()
        graph, _assignment, _rich_club = build_runtime_graph(store)
        snap = compute_topology_snapshot(graph)
    except HippoLockHeldError:
        _render({})
        return 0

    _render(snap)
    return 0


def cmd_drain_permanent_failed(args: argparse.Namespace) -> int:
    dry_run = bool(getattr(args, "dry_run", False))

    resp = _send_jsonrpc_request("drain_permanent_failed", {"dry_run": dry_run}, read_timeout=120.0)
    if isinstance(resp, dict):
        result = resp.get("result")
        if isinstance(result, dict):
            _print_drain_result(result)
            return 0

    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.store import MemoryStore
    from iai_mcp.capture import drain_permanent_failed_files

    try:
        store = MemoryStore()
        result = drain_permanent_failed_files(store, dry_run=dry_run)
    except HippoLockHeldError:
        print(
            "Daemon holds the store lock — is it running? "
            "Ensure the daemon is reachable or stopped before using the direct-open fallback.",
            file=sys.stderr,
        )
        return 1

    _print_drain_result(result)
    return 0


def _print_drain_result(result: dict) -> None:
    files = result.get("files") or []
    if result.get("dry_run"):
        count = result.get("count", len(files))
        print(f"dry-run: {count} permanent-failed file(s) found")
        for f in files:
            print(f"  {f['name']}  ({f.get('line_count', '?')} lines)")
        return
    inserted = result.get("inserted", 0)
    dropped = result.get("dropped", 0)
    recovered = result.get("files_recovered") or []
    q_dir = result.get("quarantine_dir", "")
    print(f"recovered {len(recovered)} file(s): inserted={inserted} dropped={dropped}")
    for name in recovered:
        print(f"  {name}")
    if q_dir:
        print(f"quarantine copies at: {q_dir}")


def _aggregate_trajectory_from_events(
    events: list[dict],
) -> dict[str, list[tuple]]:
    from iai_mcp.trajectory import METRIC_NAMES

    out: dict[str, list[tuple]] = {m: [] for m in METRIC_NAMES}
    for e in events:
        data = e.get("data") or {}
        m = data.get("metric")
        v = data.get("value")
        if m in METRIC_NAMES and v is not None:
            try:
                out[m].append((e.get("ts"), float(v)))
            except (TypeError, ValueError):
                continue
    return out


def _render_trajectory(data: dict, metric_names: list) -> None:
    if not any(data.get(m) for m in metric_names):
        print("no trajectory data recorded")
        return
    for metric in metric_names:
        points = data.get(metric, [])
        if not points:
            print(f"{metric.upper()}: (no data)")
            continue
        values = [v for _, v in points]
        n = len(values)
        mean = sum(values) / n
        print(
            f"{metric.upper()}: n={n} mean={mean:.3f} "
            f"min={min(values):.3f} max={max(values):.3f}"
        )


def cmd_trajectory(args: argparse.Namespace) -> int:
    from datetime import datetime, timedelta, timezone

    from iai_mcp.trajectory import METRIC_NAMES

    weeks = getattr(args, "since", None)
    since = None
    since_iso = None
    if weeks is not None:
        since = datetime.now(timezone.utc) - timedelta(weeks=int(weeks))
        since_iso = since.isoformat()

    socket_params: dict = {"kind": "trajectory_metric", "limit": 1000}
    if since_iso:
        socket_params["since"] = since_iso
    resp = _send_jsonrpc_request("events_query", socket_params)
    if isinstance(resp, dict) and "result" in resp:
        payload = resp["result"]
        if isinstance(payload, dict) and "events" in payload:
            data = _aggregate_trajectory_from_events(payload["events"])
            _render_trajectory(data, METRIC_NAMES)
            return 0

    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.store import MemoryStore
    from iai_mcp.trajectory import aggregate_trajectory

    try:
        store = MemoryStore()
        data = aggregate_trajectory(store, since=since)
    except HippoLockHeldError:
        print("daemon holds store lock; retry when daemon is idle")
        return 0

    _render_trajectory(data, METRIC_NAMES)
    return 0


def _redact_shield_data(data: dict) -> str:
    matched = data.get("matched") or []
    tier = data.get("tier", "-")
    record_id = data.get("record_id", "-")
    action = data.get("action", "-")
    return (
        f"tier={tier} action={action} "
        f"matched_count={len(matched)} record_id={record_id}"
    )


def _format_audit_event(event: dict, tz) -> str:
    from datetime import datetime as _dt

    from iai_mcp.tz import to_local

    ts = event.get("ts")
    if isinstance(ts, str):
        try:
            ts = _dt.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            ts = None
    try:
        local_ts = to_local(ts, tz) if ts is not None else None
    except (ValueError, TypeError, OSError):
        local_ts = None
    ts_str = local_ts.isoformat() if local_ts is not None else str(event.get("ts"))

    kind = event.get("kind", "?")
    sev = event.get("severity") or "-"
    data = event.get("data") or {}
    if kind in ("shield_rejection", "shield_flag", "shield_log"):
        data_str = _redact_shield_data(data)
    else:
        data_str = str(data)[:200]
    return f"[{ts_str}] {kind:32s} [{sev:8s}] {data_str}"


def cmd_audit(args: argparse.Namespace) -> int:
    from datetime import datetime, timedelta, timezone

    from iai_mcp.tz import load_user_tz

    tz = load_user_tz()

    since_raw = getattr(args, "since", None)
    since = None
    since_iso = None
    if since_raw is not None:
        since = datetime.now(timezone.utc) - timedelta(weeks=int(since_raw))
        since_iso = since.isoformat()

    sub = getattr(args, "audit_sub", None)

    if sub == "drift":
        resp = _send_jsonrpc_request("detect_drift", {})
        if isinstance(resp, dict) and "result" in resp:
            payload = resp["result"]
            if isinstance(payload, dict) and "alerts" in payload:
                alerts = payload["alerts"]
                if not alerts:
                    print("drift: no anomaly detected (variance stable)")
                else:
                    for a in alerts:
                        print(
                            f"drift: variance increasing across "
                            f"{a.get('window_sessions')} sessions; "
                            f"first={a.get('first_value'):.3f} "
                            f"last={a.get('last_value'):.3f}"
                        )
                return 0

        from iai_mcp.hippo import HippoLockHeldError
        from iai_mcp.s5 import detect_drift_anomaly
        from iai_mcp.store import MemoryStore

        try:
            store = MemoryStore()
            alerts = detect_drift_anomaly(store)
        except HippoLockHeldError:
            print("daemon holds store lock; retry when daemon is idle")
            return 0

        if not alerts:
            print("drift: no anomaly detected (variance stable)")
        else:
            for a in alerts:
                print(
                    f"drift: variance increasing across "
                    f"{a.get('window_sessions')} sessions; "
                    f"first={a.get('first_value'):.3f} "
                    f"last={a.get('last_value'):.3f}"
                )
        return 0

    SHIELD_KINDS = ("shield_rejection", "shield_flag", "shield_log")
    IDENTITY_KINDS = (
        "s5_invariant_update",
        "s5_invariant_proposal",
        "s5_cooldown_block",
        "s5_drift_alert",
        "identity_cross_lingual_warning",
    )

    if sub == "shield":
        audit_kinds = list(SHIELD_KINDS)
        empty_msg = "audit shield: no events recorded"
    elif sub == "identity":
        audit_kinds = list(IDENTITY_KINDS)
        empty_msg = "audit identity: no events recorded"
    else:
        from iai_mcp.s5 import AUDIT_EVENT_KINDS
        audit_kinds = list(AUDIT_EVENT_KINDS)
        empty_msg = "No identity events recorded"

    severity = getattr(args, "severity", None)

    socket_params: dict = {"kinds": audit_kinds}
    if since_iso:
        socket_params["since"] = since_iso
    resp = _send_jsonrpc_request("audit_query", socket_params)
    if isinstance(resp, dict) and "result" in resp:
        payload = resp["result"]
        if isinstance(payload, dict) and "events" in payload:
            events = payload["events"]
            if severity:
                events = [e for e in events if e.get("severity") == severity]
            if not events:
                print(empty_msg)
                return 0
            for e in events:
                print(_format_audit_event(e, tz))
            return 0

    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.s5 import audit_identity_events
    from iai_mcp.store import MemoryStore

    try:
        store = MemoryStore()
        events = audit_identity_events(store, since=since, kinds=tuple(audit_kinds))
    except HippoLockHeldError:
        print("daemon holds store lock; retry when daemon is idle")
        return 0

    if severity:
        events = [e for e in events if e.get("severity") == severity]
    if not events:
        print(empty_msg)
        return 0
    for e in events:
        print(_format_audit_event(e, tz))
    return 0


def cmd_schema_cleanup(args: argparse.Namespace) -> int:
    from iai_mcp.migrate import cleanup_schema_duplicates
    from iai_mcp.store import MemoryStore

    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    if not store_path.exists():
        print(
            f"error: store path does not exist: {store_path}",
            file=sys.stderr,
        )
        return 2

    apply = bool(getattr(args, "apply", False))

    store = MemoryStore(path=store_path)
    summary = cleanup_schema_duplicates(
        store, apply=apply, store_path=store_path,
    )

    mode_str = summary.get("mode", "dry-run")
    print(f"iai-mcp schema-cleanup [{mode_str}]")
    print(f"  groups (patterns with N>1 duplicates): {summary.get('groups', 0)}")
    print(f"  keepers (one per group):               {summary.get('keepers', 0)}")
    print(
        f"  pruned (soft-deleted, tier=semantic_pruned): "
        f"{summary.get('pruned', 0)}"
    )
    print(
        f"  edges to reinforce onto keepers:       "
        f"{summary.get('edges_reinforced', 0)}"
    )
    if summary.get("snapshot_dir"):
        print(f"  snapshot directory:                    {summary['snapshot_dir']}")
    if mode_str == "dry-run" and summary.get("groups", 0) > 0:
        print()
        print("  Run with --apply to execute.")
    return 0


def _maintenance_compact_preflight_daemon_alive() -> str | None:
    import json as _json
    import os as _os

    if not STATE_PATH.exists():
        return None
    try:
        state = _json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        return None
    pid = state.get("daemon_pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        _os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return None
    except OSError:
        return None
    try:
        import psutil
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline())
    except Exception as exc:
        logger.debug("psutil inspect pid %d failed: %s", pid, exc)
        return (
            f"daemon running (pid {pid}); run `iai-mcp daemon stop` "
            f"first, then retry"
        )
    if "iai_mcp.daemon" not in cmdline:
        return None
    return (
        f"daemon running (pid {pid}); run `iai-mcp daemon stop` first, "
        f"then retry"
    )


def _maintenance_compact_metrics(
    hippo_dir: Path,
    store: object | None = None,
) -> dict:
    db_path = hippo_dir / "brain.sqlite3"
    size_bytes = 0
    try:
        if db_path.exists():
            size_bytes = db_path.stat().st_size
    except OSError:
        pass
    size_mb = round(size_bytes / (1024 * 1024), 1)
    records_count = 0
    record_id_set: set[str] = set()
    if store is not None:
        try:
            tbl = store.db.open_table("records")
            records_count = int(tbl.count_rows())
            df = tbl.search().select(["id"]).to_pandas()
            record_id_set = {str(x) for x in df["id"].tolist()}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.debug("compact metrics read failed: %s", exc)
    return {
        "db_size_mb": size_mb,
        "records_count": records_count,
        "record_id_set": record_id_set,
    }


def _maintenance_compact_dry_run(
    store_path: Path, hippo_dir: Path,
) -> int:
    import json as _json
    from iai_mcp.store import MemoryStore

    store = None
    try:
        store = MemoryStore(path=store_path)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("compact dry-run MemoryStore open failed: %s", exc)
        print(
            f"warning: could not open MemoryStore (records_count + "
            f"record_id_set will be 0): {exc}",
            file=sys.stderr,
        )
    metrics = _maintenance_compact_metrics(hippo_dir, store=store)
    out = {
        "mode": "dry-run",
        "metrics": {
            "pre": {
                k: v for k, v in metrics.items() if k != "record_id_set"
            },
            "post": None,
        },
        "would_invoke": "optimize_hippo_storage()",
    }
    print(_json.dumps(out, indent=2))
    return 0


def _maintenance_compact_apply(
    store_path: Path, hippo_dir: Path,
) -> int:
    import json as _json
    import time as _time
    from datetime import datetime, timezone
    from iai_mcp.maintenance import optimize_hippo_storage
    from iai_mcp.store import MemoryStore

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_path = (
        Path.home() / ".iai-mcp" / f".maintenance-compact-{ts}.json"
    )

    store = MemoryStore(path=store_path)
    pre_metrics = _maintenance_compact_metrics(hippo_dir, store=store)
    pre_id_set = pre_metrics["record_id_set"]

    t0 = _time.monotonic()
    report = optimize_hippo_storage(store)
    elapsed = round(_time.monotonic() - t0, 3)

    store_after = MemoryStore(path=store_path)
    post_metrics = _maintenance_compact_metrics(hippo_dir, store=store_after)
    post_id_set = post_metrics["record_id_set"]

    if pre_id_set != post_id_set:
        missing = pre_id_set - post_id_set
        extra = post_id_set - pre_id_set
        failed_path = (
            Path.home() / ".iai-mcp"
            / f".maintenance-compact-FAILED-{ts}.json"
        )
        failed_payload = {
            "command": "iai-mcp maintenance compact-hippo --apply",
            "timestamp_utc": ts,
            "status": "aborted",
            "reason": "record_id_set divergence post-optimize",
            "metrics_pre": {
                k: v for k, v in pre_metrics.items()
                if k != "record_id_set"
            },
            "metrics_post": {
                k: v for k, v in post_metrics.items()
                if k != "record_id_set"
            },
            "missing_ids_count": len(missing),
            "extra_ids_count": len(extra),
            "missing_ids_sample": list(sorted(missing))[:10],
            "extra_ids_sample": list(sorted(extra))[:10],
            "optimize_report": report,
            "elapsed_sec": elapsed,
        }
        try:
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            failed_path.write_text(_json.dumps(failed_payload, indent=2))
        except OSError:
            pass
        print(
            f"ABORT: record_id_set divergence — missing={len(missing)} "
            f"extra={len(extra)}; details written to {failed_path}",
            file=sys.stderr,
        )
        return 1

    payload = {
        "command": "iai-mcp maintenance compact-hippo --apply",
        "timestamp_utc": ts,
        "status": "ok",
        "metrics_pre": {
            k: v for k, v in pre_metrics.items() if k != "record_id_set"
        },
        "metrics_post": {
            k: v for k, v in post_metrics.items() if k != "record_id_set"
        },
        "elapsed_sec": elapsed,
        "optimize_report": report,
    }
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(_json.dumps(payload, indent=2))
    except OSError as exc:
        print(
            f"warning: could not write audit file {audit_path}: {exc}",
            file=sys.stderr,
        )
    print(_json.dumps({
        "mode": "apply",
        "metrics": {
            "pre": payload["metrics_pre"],
            "post": payload["metrics_post"],
        },
        "elapsed_sec": elapsed,
        "audit_file": str(audit_path),
        "status": "ok",
    }, indent=2))
    return 0


def cmd_maintenance_compact_hippo(args: argparse.Namespace) -> int:
    if getattr(args, "maintenance_cmd", None) == "compact-records":
        print(
            "warning: compact-records is the deprecated name for "
            "compact-hippo; use compact-hippo going forward",
            file=sys.stderr,
        )

    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    hippo_dir = store_path / "hippo"
    if not hippo_dir.exists():
        print(
            f"error: hippo storage not found at {hippo_dir}",
            file=sys.stderr,
        )
        return 1

    apply = bool(getattr(args, "apply", False))
    yes = bool(getattr(args, "yes", False))
    if not apply:
        return _maintenance_compact_dry_run(store_path, hippo_dir)

    refusal = _maintenance_compact_preflight_daemon_alive()
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 1

    if not yes and not sys.stdin.isatty():
        print(
            "error: --apply on non-tty requires --yes (refusing to proceed "
            "without interactive consent or explicit --yes)",
            file=sys.stderr,
        )
        return 2

    if not yes:
        prompt = (
            "About to compact Hippo storage via wal_checkpoint + VACUUM + "
            "hnswlib rebuild. Daemon must be stopped. Type 'y' to proceed: "
        )
        try:
            response = input(prompt)
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            print("aborted: user did not consent", file=sys.stderr)
            return 1

    return _maintenance_compact_apply(store_path, hippo_dir)


def cmd_maintenance_compact_records(args: argparse.Namespace) -> int:
    args.maintenance_cmd = "compact-records"
    return cmd_maintenance_compact_hippo(args)


def cmd_maintenance_symmetrize_self_loops(args: argparse.Namespace) -> int:
    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    hippo_dir = store_path / "hippo"
    if not hippo_dir.exists():
        print(
            f"error: hippo storage not found at {hippo_dir}",
            file=sys.stderr,
        )
        return 1

    apply = bool(getattr(args, "apply", False))
    yes = bool(getattr(args, "yes", False))

    from iai_mcp.maintenance import symmetrize_self_loops
    from iai_mcp.store import MemoryStore

    if not apply:
        store = MemoryStore(path=store_path)
        result = symmetrize_self_loops(store, dry_run=True)
        print(json.dumps(result, indent=2))
        return 0

    refusal = _maintenance_compact_preflight_daemon_alive()
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 1

    if not yes and not sys.stdin.isatty():
        print(
            "error: --apply on non-tty requires --yes (refusing to "
            "proceed without interactive consent or explicit --yes)",
            file=sys.stderr,
        )
        return 2

    if not yes:
        prompt = (
            "About to backfill missing hebbian self-loops on records. "
            "Daemon must be stopped. Type 'y' to proceed: "
        )
        try:
            response = input(prompt)
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            print("aborted: user did not consent", file=sys.stderr)
            return 1

    store = MemoryStore(path=store_path)
    result = symmetrize_self_loops(store, dry_run=False)
    print(json.dumps(result, indent=2))
    return 0


def _format_relative(ts_iso: str, now: datetime | None = None) -> str:
    try:
        ts = datetime.fromisoformat(ts_iso)
    except (TypeError, ValueError):
        return "unknown"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    delta = moment - ts
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds} seconds"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''}"


def cmd_lifecycle_force_unlock(args: argparse.Namespace) -> int:
    from iai_mcp.lifecycle_lock import DEFAULT_LOCK_PATH, LifecycleLock

    lock_path = getattr(args, "lock_path", None)
    if lock_path is not None:
        lock = LifecycleLock(Path(lock_path))
    else:
        lock = LifecycleLock(DEFAULT_LOCK_PATH)

    existing = lock.read()
    if existing is None:
        print("No lockfile present; nothing to unlock.")
        return 0

    print(
        f"Existing lockfile: pid={existing['pid']} "
        f"hostname={existing['hostname']} "
        f"started_at={existing['started_at']}"
    )

    yes = bool(getattr(args, "yes", False))
    if not yes:
        try:
            response = input(
                "Force unlock and remove the lockfile? [y/N]: "
            )
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            print("Force-unlock cancelled.", file=sys.stderr)
            return 1

    previous = lock.force_unlock()
    if previous is None:
        print("Lockfile already removed by another process.")
        return 0
    print("Lockfile removed.")
    return 0


def cmd_lifecycle_status(args: argparse.Namespace) -> int:
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH, load_state

    record = load_state(LIFECYCLE_STATE_PATH)
    print(f"state: {record['current_state']}")
    print(
        f"since: {record['since_ts']} "
        f"({_format_relative(record['since_ts'])})"
    )
    print(f"last_activity: {record['last_activity_ts']}")
    print(f"wrapper_event_seq: {record['wrapper_event_seq']}")

    progress = record.get("sleep_cycle_progress")
    if progress is None:
        print("sleep_cycle_progress: none")
    else:
        step = progress.get(
            "last_completed_index",
            progress.get("last_completed_step", 0),
        )
        attempt = progress.get("attempt", 0)
        last_error = progress.get("last_error") or "none"
        started_at = progress.get("started_at", "?")
        print(
            f"sleep_cycle_progress: step={step} attempt={attempt} "
            f"last_error={last_error} started_at={started_at}"
        )

    quarantine = record.get("quarantine")
    if quarantine is None:
        print("quarantine: none")
    else:
        print(
            f"quarantine: until={quarantine['until_ts']} "
            f"reason={quarantine['reason']} since={quarantine['since_ts']}"
        )

    shadow = record.get("shadow_run", True)
    if shadow:
        print(
            "shadow_run: true (legacy RSS-watchdog still owns shutdown)"
        )
    else:
        print("shadow_run: false")

    return 0


def cmd_maintenance_sleep_cycle(args: argparse.Namespace) -> int:
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH
    from iai_mcp.sleep_pipeline import SleepPipeline, SleepStep
    from iai_mcp.store import MemoryStore

    if getattr(args, "store_path", None) is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    try:
        store = MemoryStore(path=store_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("sleep-cycle MemoryStore open failed: %s", exc)
        print(
            f"error: could not open MemoryStore at {store_path}: {exc}",
            file=sys.stderr,
        )
        return 2

    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=LIFECYCLE_STATE_PATH,
        event_log=LifecycleEventLog(),
    )

    reset_quarantine = bool(getattr(args, "reset_quarantine", False))
    force = bool(getattr(args, "force", False))

    if reset_quarantine:
        if pipeline.is_quarantined():
            pipeline.reset_quarantine()
            print("Quarantine cleared.")
        else:
            print("Quarantine not active; --reset-quarantine had no effect.")

    if pipeline.is_quarantined() and not force:
        from iai_mcp.lifecycle_state import load_state

        record = load_state(LIFECYCLE_STATE_PATH)
        quarantine = record.get("quarantine") or {}
        until_ts = quarantine.get("until_ts", "?")
        reason = quarantine.get("reason", "unknown")
        print(
            f"Sleep cycle quarantined until {until_ts}.",
            file=sys.stderr,
        )
        print(f"Reason: {reason}", file=sys.stderr)
        print(
            "Use --force to override OR --reset-quarantine to clear.",
            file=sys.stderr,
        )
        return 1

    step_index = {
        step: i + 1 for i, step in enumerate(SleepPipeline._STEP_ORDER)
    }
    total_steps = len(SleepPipeline._STEP_ORDER)

    print("Sleep cycle started.")
    runner = pipeline.force_run if force else pipeline.run
    result = runner()

    for step in result["completed_steps"]:
        idx = step_index.get(step, "?")
        print(f"[{idx}/{total_steps}] {step.name.lower()} ... ok")

    duration = result.get("duration_sec", 0.0)
    failed = result.get("failed_step")
    interrupted = result.get("interrupted", False)
    quarantine_triggered = result.get("quarantine_triggered", False)

    if failed is not None:
        idx = step_index.get(failed, "?")
        err = result.get("error") or "unknown"
        print(
            f"[{idx}/{total_steps}] {failed.name.lower()} ... FAILED: {err}",
            file=sys.stderr,
        )
        if quarantine_triggered:
            print(
                "Sleep cycle quarantined for 24h after 3rd consecutive "
                "failure of this step. Use --reset-quarantine to clear.",
                file=sys.stderr,
            )
        else:
            print(
                "Sleep cycle aborted; rerun to retry from this step.",
                file=sys.stderr,
            )
        return 1

    if interrupted:
        print(
            f"Sleep cycle deferred (bounded interrupt; "
            f"{duration:.1f}s elapsed). Resume on next invocation.",
        )
        return 0

    print(f"Sleep cycle complete ({duration:.1f}s total).")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iai-mcp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("health", help="show LLM health status")
    h.set_defaults(func=cmd_health)

    bn = sub.add_parser(
        "build-native",
        help=(
            "compile the Rust native extension (iai_mcp_native) in-place. "
            "Run after Python upgrade or on fresh clone. Requires cargo."
        ),
    )
    bn.set_defaults(func=cmd_build_native)

    m = sub.add_parser(
        "migrate",
        help=(
            "migrate records: 1->2 or 2->3 (encryption); "
            "OR --resume / --rollback a partial reembed migration"
        ),
    )
    m.add_argument("--from", dest="from_", type=int, default=1)
    m.add_argument("--to", type=int, default=2)
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--verbose", "-v", action="store_true")
    m.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partial reembed migration from migration_progress.json checkpoint.",
    )
    m.add_argument(
        "--rollback",
        action="store_true",
        help=(
            "Roll back a partial reembed migration: drop records_v_new and "
            "(if needed) restore records from records_old_<ts>."
        ),
    )
    m.add_argument(
        "--rederive-timestamps",
        action="store_true",
        help=(
            "Re-derive collapsed created_at timestamps from on-disk transcripts. "
            "One-time operation; idempotent. Records with no recoverable transcript "
            "are left unchanged."
        ),
    )
    m.set_defaults(func=cmd_migrate)

    c = sub.add_parser(
        "crypto",
        help="encryption key management",
    )
    crypto_sub = c.add_subparsers(dest="crypto_cmd", required=True)

    cs = crypto_sub.add_parser(
        "status",
        help=(
            "show file-backend key status: backend, path, "
            "mode, uid, length validation, passphrase-fallback flag"
        ),
    )
    cs.add_argument("--user-id", dest="user_id", default="default")
    cs.set_defaults(func=cmd_crypto_status)

    cr = crypto_sub.add_parser(
        "rotate", help="rotate encryption key + re-encrypt all records"
    )
    cr.add_argument("--user-id", dest="user_id", default="default")
    cr.set_defaults(func=cmd_crypto_rotate)

    mtf = crypto_sub.add_parser(
        "migrate-to-file",
        help=(
            "one-time: read existing key from macOS Keychain "
            "and write to .crypto.key file (interactive Terminal only)"
        ),
    )
    mtf.add_argument("--user-id", dest="user_id", default="default")
    mtf_group = mtf.add_mutually_exclusive_group()
    mtf_group.add_argument(
        "--keep-keychain",
        dest="keep_keychain",
        action="store_true",
        default=True,
        help="leave the existing macOS Keychain entry in place (default)",
    )
    mtf_group.add_argument(
        "--delete-keychain",
        dest="keep_keychain",
        action="store_false",
        help="delete the macOS Keychain entry after successful migration",
    )
    mtf.set_defaults(func=cmd_crypto_migrate_to_file)

    ci = crypto_sub.add_parser(
        "init",
        help=(
            "generate a fresh .crypto.key file "
            "(fresh installs only — refuses if file exists)"
        ),
    )
    ci.add_argument("--user-id", dest="user_id", default="default")
    ci.set_defaults(func=cmd_crypto_init)

    rwpk = crypto_sub.add_parser(
        "recover-with-prior-key",
        help=(
            "stage all records, decrypt literal/provenance/gain with current "
            "then prior key, re-encrypt under current key; atomic store swap"
        ),
    )
    rwpk.add_argument(
        "--prior-key-file",
        type=Path,
        required=True,
        help="path to exactly 32 raw AES key bytes (same format as .crypto.key)",
    )
    rwpk.add_argument("--user-id", dest="user_id", default="default")
    rwpk.add_argument(
        "--dry-run",
        action="store_true",
        help="report rows that need the prior key without mutating tables",
    )
    rwpk.set_defaults(func=cmd_crypto_recover_prior_key)

    cred = crypto_sub.add_parser(
        "redact-undecryptable",
        help=(
            "replace literal_surface that fails AES-GCM decrypt with a redacted "
            "marker (preserves embeddings, edges, metadata)"
        ),
    )
    cred.add_argument("--user-id", dest="user_id", default="default")
    cred.set_defaults(func=cmd_crypto_redact_undecryptable)

    t = sub.add_parser(
        "trajectory",
        help="aggregate trajectory events",
    )
    t.add_argument(
        "--since",
        type=int,
        default=None,
        help="weeks back to include (default: all history)",
    )
    t.set_defaults(func=cmd_trajectory)

    topo = sub.add_parser(
        "topology",
        help="live small-world topology snapshot: C, L, sigma, communities, rich-club ratio, N, regime",
    )
    topo.set_defaults(func=cmd_topology)

    cap = sub.add_parser(
        "capture-transcript",
        help=(
            "batch-capture a Claude Code JSONL transcript into episodic tier. "
            "Used by the Stop hook for ambient WRITE-side observation capture."
        ),
    )
    cap.add_argument("transcript_path", help="path to the Claude Code JSONL transcript file")
    cap.add_argument("--session-id", default="-", help="session id for provenance")
    cap.add_argument("--max-turns", type=int, default=200,
                     help="cap on turns to scan (default 200; older turns skipped)")
    cap.add_argument(
        "--no-spawn",
        action="store_true",
        default=False,
        help=(
            "Hook-only mode: try connect with 250ms timeout. On miss, write "
            "transcript to ~/.iai-mcp/.deferred-captures/ and exit 0 within 2s. "
            "NEVER spawn daemon. Used by ~/.claude/hooks/iai-mcp-session-capture.sh "
            "to eliminate spawn vector."
        ),
    )
    cap.set_defaults(func=cmd_capture_transcript)

    ctd = sub.add_parser(
        "capture-turn-deferred",
        help=(
            "append a single JSONL event per new transcript turn to "
            "{session_id}.live.jsonl. UserPromptSubmit-hook backend."
        ),
    )
    ctd.add_argument("--session-id", required=True)
    ctd.add_argument("--transcript-path", required=True)
    ctd.add_argument(
        "--max-turns-per-call",
        type=int,
        default=200,
        help="max new turns to process per invocation (default 200)",
    )
    ctd.set_defaults(func=cmd_capture_turn_deferred)

    ssp = sub.add_parser(
        "session-start",
        help=(
            "print the session-start recall payload as markdown on stdout. "
            "Hook target for ~/.claude/hooks/iai-mcp-session-recall.sh."
        ),
    )
    ssp.add_argument("--session-id", default="-", help="session id for provenance")
    ssp.set_defaults(func=cmd_session_start)

    sris = sub.add_parser(
        "session-refresh-if-stale",
        help=(
            "UserPromptSubmit hook gate: compare MAX(created_at) against the "
            "per-session watermark sidecar; call session_refresh_if_stale RPC "
            "only when new memory exists; emit additionalContext JSON on trigger."
        ),
    )
    sris.add_argument("--session-id", default="-", help="session id for watermark sidecar")
    sris.set_defaults(func=cmd_session_refresh_if_stale)

    ch = sub.add_parser(
        "capture-hooks",
        help="install/uninstall/status the Claude Code Stop hook for ambient session capture",
    )
    ch_sub = ch.add_subparsers(dest="capture_hooks_cmd", required=True)
    ch_sub.add_parser("install",
                      help="copy Stop hook to ~/.claude/hooks/ and register in settings.json"
                      ).set_defaults(func=cmd_capture_hooks_install)
    ch_sub.add_parser("uninstall",
                      help="remove the Stop hook and its settings.json entry"
                      ).set_defaults(func=cmd_capture_hooks_uninstall)
    ch_sub.add_parser("status",
                      help="show whether the Stop hook is installed and active"
                      ).set_defaults(func=cmd_capture_hooks_status)

    a = sub.add_parser(
        "audit",
        help="identity + shield audit log",
    )
    a.add_argument(
        "--since",
        type=int,
        default=None,
        help="weeks back to include (default: all history)",
    )
    a.add_argument(
        "--severity",
        choices=["info", "warning", "critical"],
        default=None,
        help="filter by severity",
    )
    audit_sub = a.add_subparsers(dest="audit_sub")
    for name, helptext in (
        ("shield", "shield-only audit (match counts redacted)"),
        ("drift", "detect M4 drift anomaly and surface it"),
        ("identity", "s5_* identity events only"),
    ):
        sp = audit_sub.add_parser(name, help=helptext)
        sp.add_argument("--since", type=int, default=None)
        sp.add_argument(
            "--severity",
            choices=["info", "warning", "critical"],
            default=None,
        )
    a.set_defaults(func=cmd_audit)

    d = sub.add_parser(
        "daemon",
        help="sleep daemon: install/uninstall/start/stop/status/logs/...",
    )
    daemon_sub = d.add_subparsers(dest="daemon_cmd", required=True)

    di = daemon_sub.add_parser(
        "install",
        help=(
            "install launchd plist (macOS) / systemd user unit (Linux); "
            "first-run consent banner unless --yes"
        ),
    )
    di.add_argument(
        "--dry-run",
        action="store_true",
        help="print plist/unit contents without writing or invoking launchctl/systemctl",
    )
    di.add_argument(
        "--yes", "-y",
        action="store_true",
        help="skip the consent banner (records --yes audit-trail still)",
    )
    di.set_defaults(func=cmd_daemon_install)

    du = daemon_sub.add_parser(
        "uninstall",
        help="clean uninstall: remove plist/unit + 3 state files",
    )
    du.add_argument("--yes", "-y", action="store_true")
    du.set_defaults(func=cmd_daemon_uninstall)

    daemon_sub.add_parser(
        "start", help="launchctl kickstart / systemctl --user start",
    ).set_defaults(func=cmd_daemon_start)

    daemon_sub.add_parser(
        "stop", help="launchctl kill SIGTERM / systemctl --user stop",
    ).set_defaults(func=cmd_daemon_stop)

    daemon_sub.add_parser(
        "status",
        help=(
            "socket round-trip: print daemon FSM state, uptime, version "
            "(warns on version skew vs installed package)"
        ),
    ).set_defaults(func=cmd_daemon_status)

    dlogs = daemon_sub.add_parser(
        "logs",
        help="tail daemon log file (macOS Library/Logs) or journalctl (Linux)",
    )
    dlogs.add_argument("-f", "--follow", action="store_true")
    dlogs.add_argument("-n", "--lines", type=int, default=50)
    dlogs.set_defaults(func=cmd_daemon_logs)

    daemon_sub.add_parser(
        "force-rem",
        help="cooperative force: trigger one REM cycle out-of-schedule",
    ).set_defaults(func=cmd_daemon_force_rem)

    dpause = daemon_sub.add_parser(
        "pause", help="pause daemon scheduler for N seconds",
    )
    dpause.add_argument("seconds", type=int)
    dpause.set_defaults(func=cmd_daemon_pause)

    daemon_sub.add_parser(
        "resume", help="resume daemon scheduler after a pause",
    ).set_defaults(func=cmd_daemon_resume)

    daemon_sub.add_parser(
        "stats",
        help=(
            "Longitudinal metrics: session_start_tokens_p90 over the "
            "most recent 100 session_started events (persisted in the events table)"
        ),
    ).set_defaults(func=cmd_daemon_stats)

    dconf = daemon_sub.add_parser(
        "configure",
        help=(
            "per-setting override: set-budget / set-cycle-count / "
            "set-quiet-window / disable-claude / enable-claude"
        ),
    )
    dconf.add_argument(
        "key",
        choices=[
            "set-budget",
            "set-cycle-count",
            "set-quiet-window",
            "disable-claude",
            "enable-claude",
        ],
    )
    dconf.add_argument("value", nargs="?", default=None)
    dconf.set_defaults(func=cmd_daemon_configure)

    sc = sub.add_parser(
        "schema-cleanup",
        help=(
            "soft-delete duplicate schema records. Default "
            "mode is --dry-run; --apply snapshots the store dir and "
            "performs the cleanup. Idempotent (re-running is a no-op)."
        ),
    )
    sc_mode = sc.add_mutually_exclusive_group()
    sc_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(default) print the cleanup diff without mutating the store",
    )
    sc_mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="snapshot the store dir + soft-delete duplicates",
    )
    sc.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; Hippo data "
            "lives at <store-path>/hippo)"
        ),
    )
    sc.set_defaults(func=cmd_schema_cleanup)

    mtn = sub.add_parser(
        "maintenance",
        help=(
            "one-shot maintenance ops. Currently: compact-hippo "
            "(PRAGMA wal_checkpoint + VACUUM + hnswlib rebuild)."
        ),
    )
    mtn_sub = mtn.add_subparsers(dest="maintenance_cmd", required=True)
    mtn_compact = mtn_sub.add_parser(
        "compact-hippo",
        help=(
            "compact Hippo storage: wal_checkpoint + VACUUM + hnswlib rebuild. "
            "DAEMON MUST BE STOPPED. Default --dry-run; --apply requires "
            "--yes for non-tty."
        ),
    )
    mtn_compact_mode = mtn_compact.add_mutually_exclusive_group()
    mtn_compact_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(default) print metrics-only JSON; do NOT call optimize",
    )
    mtn_compact_mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="run wal_checkpoint + VACUUM + hnswlib rebuild on Hippo storage",
    )
    mtn_compact.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="(use with --apply) skip the interactive 'y/N' prompt",
    )
    mtn_compact.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; Hippo data "
            "lives at <store-path>/hippo). Mirrors `schema-cleanup` flag."
        ),
    )
    mtn_compact.set_defaults(func=cmd_maintenance_compact_hippo)
    mtn_compact_legacy = mtn_sub.add_parser(
        "compact-records",
        help="Deprecated alias for compact-hippo (kept for one release).",
    )
    mtn_compact_legacy_mode = mtn_compact_legacy.add_mutually_exclusive_group()
    mtn_compact_legacy_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(default) print metrics-only JSON; do NOT call optimize",
    )
    mtn_compact_legacy_mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="run wal_checkpoint + VACUUM + hnswlib rebuild on Hippo storage",
    )
    mtn_compact_legacy.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="(use with --apply) skip the interactive 'y/N' prompt",
    )
    mtn_compact_legacy.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; Hippo data "
            "lives at <store-path>/hippo). Mirrors `schema-cleanup` flag."
        ),
    )
    mtn_compact_legacy.set_defaults(func=cmd_maintenance_compact_records)

    mtn_symmetrize = mtn_sub.add_parser(
        "symmetrize-self-loops",
        help=(
            "backfill missing hebbian self-loops on existing records. "
            "DAEMON MUST BE STOPPED. Default --dry-run; --apply requires "
            "--yes for non-tty."
        ),
    )
    mtn_symmetrize_mode = mtn_symmetrize.add_mutually_exclusive_group()
    mtn_symmetrize_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(default) print counts JSON; do NOT write self-loops",
    )
    mtn_symmetrize_mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="write missing self-loops at delta=0.1 (hebbian edge_type)",
    )
    mtn_symmetrize.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="(use with --apply) skip the interactive 'y/N' prompt",
    )
    mtn_symmetrize.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; Hippo data "
            "lives at <store-path>/hippo). Mirrors compact-hippo flag."
        ),
    )
    mtn_symmetrize.set_defaults(func=cmd_maintenance_symmetrize_self_loops)

    mtn_sleep = mtn_sub.add_parser(
        "sleep-cycle",
        help=(
            "run the 5-step sleep pipeline once: "
            "schema_mine, knob_tune, dream_decay, optimize_hippo, "
            "compact_records. 3-strike auto-quarantine; use --force "
            "to override, --reset-quarantine to clear."
        ),
    )
    mtn_sleep.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="run even if quarantined (operator override)",
    )
    mtn_sleep.add_argument(
        "--reset-quarantine",
        dest="reset_quarantine",
        action="store_true",
        default=False,
        help="clear quarantine state before running",
    )
    mtn_sleep.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; Hippo data "
            "lives at <store-path>/hippo)"
        ),
    )
    mtn_sleep.set_defaults(func=cmd_maintenance_sleep_cycle)

    doc = sub.add_parser(
        "doctor",
        help=(
            "Diagnose daemon health (7 checks; (g) duplicate-binder detection). "
            "With --apply, attempt safe repairs "
            "(unlink stale socket, kill duplicate binders, cleanup orphans, "
            "respawn daemon). With --apply --yes, skip confirmations. "
            "Exit 0=all green, 1=any FAIL, 2=--apply tried but FAIL persists."
        ),
    )
    doc.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="attempt safe repairs after diagnosis; prompts before each destructive action",
    )
    doc.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="(use with --apply) skip confirmation prompts; equivalent to typing 'y' to all",
    )
    doc.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help=(
            "force headless mode (downgrade `(n) HID idle source` and "
            "`(b) socket file fresh` from FAIL to WARN). Auto-detected on "
            "Linux when DISPLAY/WAYLAND_DISPLAY are unset; on macOS use this "
            "flag explicitly."
        ),
    )
    def _cmd_doctor_lazy(args: argparse.Namespace) -> int:
        from iai_mcp.doctor import cmd_doctor
        return cmd_doctor(args)
    doc.set_defaults(func=_cmd_doctor_lazy)

    lc = sub.add_parser(
        "lifecycle",
        help=(
            "inspect lifecycle state machine "
            "(WAKE/DROWSY/SLEEP/HIBERNATION). Currently: status."
        ),
    )
    lc_sub = lc.add_subparsers(dest="lifecycle_cmd", required=True)
    lc_status = lc_sub.add_parser(
        "status",
        help=(
            "print current lifecycle state, since-ts, last activity, "
            "wrapper event seq, sleep-cycle progress, quarantine, and "
            "shadow_run flag"
        ),
    )
    lc_status.set_defaults(func=cmd_lifecycle_status)

    lc_unlock = lc_sub.add_parser(
        "force-unlock",
        help=(
            "clear a stale ~/.iai-mcp/.locked lockfile and "
            "print the prior PID / hostname / started_at"
        ),
    )
    lc_unlock.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive [y/N] prompt",
    )
    lc_unlock.set_defaults(func=cmd_lifecycle_force_unlock)

    br = sub.add_parser(
        "bank-recall",
        help=(
            "substring recall over bank/processed + bank/recent without "
            "booting the daemon. Used by the wrapper as a socket-dead "
            "fallback path."
        ),
    )
    br.add_argument("--query", required=True, help="cue substring to match")
    br.add_argument(
        "--limit", type=int, default=20, help="max hits (default 20)"
    )
    br.add_argument(
        "--processed-only", action="store_true", default=False
    )
    br.add_argument(
        "--recent-only", action="store_true", default=False
    )
    br.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="emit JSON to stdout (current default; --no-json is reserved)",
    )
    br.set_defaults(func=cmd_bank_recall)

    dpf = sub.add_parser(
        "drain-permanent-failed",
        help=(
            "recover terminal .permanent-failed-*.jsonl files from "
            ".deferred-captures/. Routes through daemon socket when daemon "
            "is running; direct-open fallback when daemon is down. "
            "--dry-run lists files without mutating anything."
        ),
    )
    dpf.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="list terminal files + event counts without inserting or renaming",
    )
    dpf.set_defaults(func=cmd_drain_permanent_failed)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
