"""iai-mcp CLI: health + migrate + trajectory + audit + crypto + daemon.

Commands:
- `iai-mcp health`           -- print the most recent llm_health event in user-local TZ
- `iai-mcp migrate`          -- schema migration (chosen by --from / --to)
- `iai-mcp trajectory`       -- aggregate trajectory events
- `iai-mcp audit`            -- identity + shield audit log
- `iai-mcp crypto status`           -- file-backend key status
- `iai-mcp crypto rotate`           -- rotate AES-256-GCM key
- `iai-mcp crypto migrate-to-file`  -- one-time migration from Keychain to file
- `iai-mcp crypto init`             -- fresh-install: generate a new key file
- `iai-mcp crypto recover-with-prior-key` -- re-encrypt records after wrong-key rotation (32-byte prior key file)
- `iai-mcp crypto redact-undecryptable` -- replace surfaces that fail decrypt with a redacted marker
- `iai-mcp daemon install`   -- silent install + first-run consent
- `iai-mcp daemon uninstall` -- clean uninstall (plist/unit + 3 state files)
- `iai-mcp daemon start|stop|status|logs|force-rem|pause|resume|configure`

All timestamps render in the user's IANA timezone via
`iai_mcp.tz.load_user_tz() + to_local()`. Storage remains UTC.

Audit privacy: shield match patterns are REDACTED to the MATCH COUNT
in CLI output (info-disclosure mitigation). Full payload remains
in the events table for forensics.

Guards (daemon group):
- ZERO API costs. The paid-API env-var token is forbidden in
  daemon-side modules; this CLI delegates LLM-aware operations to the
  daemon process which uses `claude -p` subprocess (subscription only).
- `daemon uninstall` MUST remove plist/unit AND ~/.iai-mcp/.lock,
  ~/.iai-mcp/.daemon.sock, ~/.iai-mcp/.daemon-state.json -- verified by
  tests/shell/test_launchd_install.sh and tests/test_cli_daemon.py.
- launchd PATH: install renders the plist with absolute
  `sys.executable` substituted -- launchd has no PATH, relative `python3`
  would resolve to /usr/bin/python3 even if user installed in /opt/python.
- systemd linger: install probes `loginctl show-user --property=Linger`
  on Linux; if Linger=no, runs `loginctl enable-linger $USER` and re-verifies.
  PAM-variant systems may silently refuse, hence the post-enable check + WARN.
- Subprocess invocation: argv-list form ALWAYS, never shell=True. launchctl /
  systemctl / loginctl / tail / journalctl all receive list args.
"""
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

# cmd_doctor is imported lazily in the argparse setup to avoid paying
# asyncio + doctor.py import cost on every CLI invocation. Only the
# `iai-mcp doctor` subcommand needs it.

# ---------------------------------------------------------------------------
# Daemon CLI group constants
# ---------------------------------------------------------------------------

# Re-export the daemon-side state paths so tests + uninstall can clear them
# in lock-step with `iai_mcp.concurrency` / `iai_mcp.daemon_state`. These
# duplicate Path.home() lookups so monkeypatching Path.home in tests works.
LOCK_PATH: Path = Path.home() / ".iai-mcp" / ".lock"
SOCKET_PATH: Path = Path.home() / ".iai-mcp" / ".daemon.sock"
STATE_PATH: Path = Path.home() / ".iai-mcp" / ".daemon-state.json"

# Deployment artefact install targets: per-user system service paths.
LAUNCHD_TARGET: Path = Path.home() / "Library" / "LaunchAgents" / "com.iai-mcp.daemon.plist"
SYSTEMD_TARGET: Path = Path.home() / ".config" / "systemd" / "user" / "iai-mcp-daemon.service"

DAEMON_LABEL: str = "com.iai-mcp.daemon"
SERVICE_NAME: str = "iai-mcp-daemon.service"

# First-run consent banner. Wording cites RAM cost, Claude budget cap,
# opt-out command. Aborts unless user types lowercase 'y' (strict).
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _ensure_crypto_key_present():
    """Idempotent: write a fresh 32-byte 0o600 key file when neither a key
    file nor ``IAI_MCP_CRYPTO_PASSPHRASE`` is present. Returns the new path
    or ``None`` when no work was needed.
    """
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
    """Return the launchd plist package-data resource as a Traversable.

    Callers invoke ``.read_text()`` on the returned object to obtain the raw
    plist content. Backed by importlib.resources package data so it works from
    both an editable install and a built wheel.
    """
    return _res.files("iai_mcp") / "_deploy" / "launchd" / "com.iai-mcp.daemon.plist"


def _render_launchd_plist() -> str:
    """Render the launchd plist template with sys.executable and the current user.

    Substitutes the placeholder interpreter path with sys.executable so launchd
    (which has no PATH) launches the correct interpreter at boot.
    """
    text = _launchd_template().read_text()
    username = os.environ.get("USER") or Path.home().name
    text = text.replace("/usr/local/bin/python3", sys.executable)
    text = text.replace("{USERNAME}", username)
    return text


def _render_systemd_unit() -> str:
    """Render the systemd unit template with sys.executable.

    Substitutes the placeholder interpreter path with sys.executable so systemd
    launches the correct interpreter even when the user's Python lives outside /usr.
    """
    tmpl = _res.files("iai_mcp") / "_deploy" / "systemd" / "iai-mcp-daemon.service"
    text = tmpl.read_text()
    text = text.replace("/usr/bin/python3", sys.executable)
    return text


def _try_short_timeout_connect(timeout_ms: int = 250) -> bool:
    """Probe daemon socket reachability with a hard timeout. Returns True if
    connect succeeded. Used by ``capture-transcript --no-spawn`` to
    decide between inline ingest vs JSONL defer — hook is best-effort and
    must NEVER block session teardown waiting on a 5s cold-start.

    Honors the ``IAI_DAEMON_SOCKET_PATH`` env override (test isolation).
    Closes the probe socket immediately — we never write a request, only
    check that connect(2) returns.
    """
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
    """Print the consent banner, read one line from stdin, return True
    only if the response stripped + lowercased equals exactly 'y'.

    Resolve sys.stderr at call time (NOT at module import) so pytest's capsys
    fixture can intercept the banner -- capsys swaps sys.stderr after our
    module is imported.
    """
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
    """Write a timestamped JSON receipt under ~/.iai-mcp/.consent-<ts>.json
    so a forensic review can verify the user actually consented (not bypassed
    via --yes). Failure to write the receipt is logged to stderr but never
    blocks the install."""
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
    """Clean uninstall removes ALL daemon-created state files."""
    for p in (LOCK_PATH, SOCKET_PATH, STATE_PATH):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"warning: could not remove {p}: {exc}", file=sys.stderr)


_HOOK_TRUNCATION_TRAILER = "[... payload truncated to fit Claude Code 10000-char limit ...]"


def _truncate_for_claude_code_hook(text: str, cap: int = 10000) -> str:
    """Cap `text` at `cap` chars; oversized inputs end with a fixed trailer
    so the consumer sees the truncation explicitly."""
    if len(text) <= cap:
        return text
    head_len = cap - len(_HOOK_TRUNCATION_TRAILER)
    if head_len <= 0:
        return _HOOK_TRUNCATION_TRAILER[:cap]
    return text[:head_len] + _HOOK_TRUNCATION_TRAILER


def _is_custom_store() -> bool:
    """Return True when IAI_MCP_STORE is set to a path other than DEFAULT_STORAGE_PATH.

    Used by _send_jsonrpc_request to decide whether the default-socket probe
    should be skipped.  When a caller points IAI_MCP_STORE at a custom location
    and has not set an explicit IAI_DAEMON_SOCKET_PATH, probing the default
    daemon socket would return data from the daemon's store rather than the
    caller's intended store — an information-disclosure / isolation bug.

    Note: iai last / episodes_recent (iai_cli.py) has no direct-open fallback,
    so under custom-store-no-socket it will print "(daemon unreachable)" and
    return 1.  This is the accepted, intended consequence; adding a direct-open
    fallback for episodes_recent is out of scope.
    """
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
    """One-shot JSON-RPC 2.0 request over the daemon AF_UNIX socket.

    Returns the response dict on success, None on connect refused, missing
    socket, timeout, malformed reply, or any other failure. Honors
    `IAI_DAEMON_SOCKET_PATH` env override.

    Store isolation: when IAI_MCP_STORE points to a non-default location and
    no IAI_DAEMON_SOCKET_PATH override is set, returns None immediately so
    every caller falls through to its own direct-open MemoryStore() path —
    which correctly reads the custom store.  Explicit IAI_DAEMON_SOCKET_PATH
    always wins (override socket routes there even with a custom store).
    """
    import asyncio  # lazy: only paid when an RPC call is actually issued
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
    """Print the session-start recall payload as markdown on stdout.

    Sends a JSON-RPC `session_start_payload` request to the daemon, renders
    the four-segment response via `iai_mcp.session.format_payload_as_markdown`,
    caps the result at 10000 characters, and writes it to stdout.

    Fail-safe: empty store, daemon unreachable, malformed reply, or any
    exception yields empty stdout and exit 0. Never blocks session start.
    """
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


# ---------------------------------------------------------------------------
# Helpers for freshness-on-return: watermark sidecar + read-only MAX gate
# ---------------------------------------------------------------------------

def get_other_sessions_live_size(session_id: str) -> int:
    """Return the total byte-size of all other sessions' ``.live.jsonl`` files.

    Scans ``~/.iai-mcp/.deferred-captures/`` for files whose name ends with
    ``.live.jsonl`` and whose stem does NOT equal ``session_id``.  Uses only
    ``os.stat`` — no file opens, no MemoryStore, no embedder.

    Returns 0 when the directory is absent or no qualifying files exist.
    Fail-safe: returns 0 on any error.
    """
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
    """Return the stored live-fingerprint (total other-sessions size) for ``session_id``.

    Returns None if the sidecar does not exist yet (first prompt).
    Fail-safe: returns None on any parse or IO error.
    """
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
    """Atomically write the live-fingerprint (total other-sessions size) for ``session_id``."""
    d = Path.home() / ".iai-mcp" / ".capture-state"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{session_id}.live-fingerprint.tmp"
    tmp.write_text(str(total_size))
    os.replace(tmp, d / f"{session_id}.live-fingerprint")


def get_max_created_at() -> str | None:
    """Return MAX(created_at) over non-tombstoned records, or None.

    Opens the Hippo SQLite file in read-only mode using stdlib sqlite3.
    WAL mode is active on the daemon side, so a concurrent read gets a
    consistent snapshot without contending on daemon writes.

    Does NOT import or construct a MemoryStore. Does NOT load the embedder.
    Fail-safe: returns None on any error (missing DB, corrupt file, etc.).
    """
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
    """Normalize an ISO-8601 timestamp to UTC isoformat for lexicographic compare.

    Handles both 'Z' suffix and '+00:00' / offset forms. Returns the
    original string unchanged on any parse error so comparisons degrade
    gracefully rather than raising.
    """
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
    """Return the stored ISO watermark for ``session_id``, or None if absent."""
    p = Path.home() / ".iai-mcp" / ".capture-state" / f"{session_id}.watermark"
    try:
        if not p.exists():
            return None
        return p.read_text().strip() or None
    except OSError:
        return None


def write_watermark(session_id: str, ts: str) -> None:
    """Atomically write the UTC-normalized ISO watermark for ``session_id``."""
    d = Path.home() / ".iai-mcp" / ".capture-state"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{session_id}.watermark.tmp"
    tmp.write_text(_utc_iso(ts))
    os.replace(tmp, d / f"{session_id}.watermark")


def cmd_session_refresh_if_stale(args: argparse.Namespace) -> int:
    """Read-only freshness gate + conditional session_refresh_if_stale RPC.

    Called by the UserPromptSubmit hook on each turn.  Does cheap local work
    (watermark sidecar read, read-only SQLite MAX query, os.stat live-file
    scan) and only sends an RPC to the daemon when genuinely new memory exists.

    Exit 0 in all cases.  Emits additionalContext JSON to stdout only when
    the daemon returns a non-empty rendered brief.  Never blocks the prompt
    (daemon-down path is silent and fail-safe).

    Two trigger signals
    -------------------
    Signal A — store advance: ``MAX(created_at)`` in Hippo SQLite advanced
    past the per-session watermark.  Covers the normal case: another session
    ended, daemon drained it, records landed in the store.

    Signal B — live-file growth: the total byte-size of all OTHER sessions'
    ``.live.jsonl`` files (``os.stat`` only, no file opens, no MemoryStore)
    grew since the last look.  Covers the headline gap case: session B is
    still open, its turns have NOT reached the store yet, so Signal A is
    silent — but Signal B detects that B wrote new turns since A last checked.

    The live-fingerprint baseline is stored in
    ``~/.iai-mcp/.capture-state/{session_id}.live-fingerprint`` (a single
    integer, the summed size of all qualifying files at last check).

    Flow
    ----
    1. Read MAX(created_at) from Hippo SQLite (read-only, no MemoryStore).
    2. Read the per-session watermark sidecar.
    3. No watermark (first prompt of session): set both baselines (watermark
       + live-fingerprint), do NOT trigger.
    4. current_max > watermark  OR  live-size > fingerprint_baseline:
       send session_refresh_if_stale RPC.
    5. Otherwise: exit 0 (common path, zero IPC).
    6. On a non-empty rendered result: emit additionalContext JSON and advance
       BOTH the watermark and the live-fingerprint baseline.
       Daemon-down leaves both sidecars unchanged.
    """
    try:
        session_id: str = (getattr(args, "session_id", None) or "-")

        current = get_max_created_at()
        if current is None:
            # Empty store or DB not found — nothing to do.
            return 0

        wm = read_watermark(session_id)
        live_size = get_other_sessions_live_size(session_id)

        if wm is None:
            # First prompt of this session: set both baselines without triggering.
            write_watermark(session_id, current)
            write_live_fingerprint(session_id, live_size)
            return 0

        # Signal A: store advance.
        store_advanced = _utc_iso(current) > _utc_iso(wm)

        # Signal B: other-session live-file growth.
        fp = read_live_fingerprint(session_id)
        # If fingerprint sidecar is absent (session predates this feature):
        # treat current live_size as the new baseline — no trigger on first look.
        if fp is None:
            write_live_fingerprint(session_id, live_size)
            fp = live_size
        live_grew = live_size > fp

        if not store_advanced and not live_grew:
            # Nothing new since last look — common path, no RPC.
            return 0

        # New memory exists (store or live): ask the daemon to drain + recompose.
        resp = _send_jsonrpc_request(
            "session_refresh_if_stale",
            {"watermark": wm, "session_id": session_id},
            connect_timeout=5.0,
            read_timeout=30.0,
        )
        if resp is None:
            # Daemon unreachable: silent fail-safe, leave sidecars unchanged.
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
            # Advance the live-fingerprint baseline so the same growth does
            # not trigger a second redundant refresh on the next prompt.
            write_live_fingerprint(session_id, live_size)

        return 0
    except Exception:
        # Fail-safe: any unhandled exception exits 0 with no output.
        return 0


def _send_socket_request(req: dict, *, timeout: float = 30.0) -> dict | None:
    """One-shot NDJSON request/response over the daemon control socket.

    Returns None when the daemon is unreachable (socket missing, connection
    refused). Raises asyncio.TimeoutError if the daemon accepted the
    connection but never replied within `timeout` seconds.
    """
    import asyncio  # lazy: only paid when a socket request is actually issued

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


# ---------------------------------------------------------------------------
# Daemon subcommand handlers
# ---------------------------------------------------------------------------


def cmd_daemon_install(args: argparse.Namespace) -> int:
    """Render plist/unit, drop into per-user system path,
    enable via launchctl bootstrap or systemctl --user enable --now.

    --dry-run prints the would-be path + rendered contents and exits.
    --yes skips the consent banner.
    """
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

    # Write the rendered file; idempotent re-install is fine (overwrite).
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    try:
        os.chmod(target, 0o644)
    except OSError:
        pass

    _ensure_crypto_key_present()

    uid = os.getuid()
    if _is_macos():
        # Idempotent bootstrap: bootout first if a previous version is loaded.
        # Both calls are best-effort; a fresh system has nothing to bootout.
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
        # Linux: probe loginctl Linger state. If not enabled, try
        # to enable; if still not enabled after that, warn loudly.
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
    """Clean removal of plist/unit + ALL state files."""
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
    """Start the singleton iai-mcp daemon.

    Symmetry contract with ``cmd_daemon_stop``: a clean stop boots the job
    OUT of the supervisor (so the supervisor cannot respawn it). ``start``
    must therefore re-register the job before kicking it, not merely
    ``kickstart`` an already-loaded one. On macOS we mirror the install
    bootstrap idiom (bootout-then-bootstrap is idempotent + tolerates an
    already-loaded job) so a booted-out daemon comes back. On Linux a plain
    ``systemctl --user start`` already (re)activates a stopped unit.
    """
    uid = os.getuid()
    if _is_macos():
        target = LAUNCHD_TARGET
        # Idempotent re-register: bootout clears any already-loaded copy,
        # bootstrap (re)registers the job, kickstart starts it now. Mirrors
        # the install flow; all best-effort so a partial state self-heals.
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


# Bounded-wait escalation parameters for `daemon stop` (macOS). After
# SIGTERM, the daemon's PID is polled up to STOP_TERM_TIMEOUT_S seconds at
# STOP_POLL_INTERVAL_S granularity; if still alive at the deadline, an
# uncatchable SIGKILL is issued. Both are env-tunable (tests set them tiny;
# operators almost never need to). The defaults give a graceful daemon ample
# time to flush and exit 0 before the hard kill.
STOP_TERM_TIMEOUT_S: float = 3.0
STOP_POLL_INTERVAL_S: float = 0.1


def _stop_escalation_bound() -> float:
    """Resolve the SIGTERM->SIGKILL deadline (env override -> default)."""
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
    """Resolve the liveness poll granularity (env override -> default)."""
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
    """Stop the singleton iai-mcp daemon (user-initiated shutdown).

    A wedged asyncio loop cannot service the in-loop signal handler
    (``loop.add_signal_handler(SIGTERM, ...)``), so a stop that relies on
    that handler can hang forever. This stop therefore does NOT depend on
    the in-loop handler. On macOS it:

      1. Writes a best-effort ``user_requested_shutdown`` sentinel (purely
         informational; a state-file failure must NOT block the kill).
      2. Reads the daemon PID from the lifecycle lockfile.
      3. Disables supervisor respawn governance FIRST (``launchctl bootout``
         removes the KeepAlive contract) so a later hard kill cannot be
         undone by a crash-respawn of the daemon the user just stopped.
      4. Self-issues ``SIGTERM`` to the daemon's own PID, polls liveness up
         to a bounded deadline, and escalates to an uncatchable ``SIGKILL``
         if the process is still alive at the deadline. The SIGKILL is the
         guarantee: a wedged loop is terminated within the bound.

    Every signal is gated by ``_is_pid_alive(pid)`` (PID-recycle-safe:
    ``os.kill(pid, 0)`` + a ``psutil`` cmdline cross-check) so only the
    confirmed daemon process is ever signalled — never a recycled PID. When
    the lockfile is absent there is no PID to signal: we still remove the
    KeepAlive governance (bootout) and return.

    The sentinel is cleared by the daemon on a graceful shutdown; it is not
    consumed for any control-flow decision — it only lets a post-mortem of
    the state file distinguish a user-stop from other shutdown paths.

    Linux (``systemctl --user stop``) is unchanged: systemd already escalates
    to SIGKILL after ``TimeoutStopSec``, so it already terminates a wedged
    loop within a bound.
    """
    import signal as _signal
    import time as _time

    # Best-effort sentinel write: we do NOT abort on failure.
    try:
        from iai_mcp.daemon_state import load_state, save_state

        state = load_state()
        state["user_requested_shutdown"] = True
        save_state(state)
    except (OSError, ValueError, RuntimeError) as exc:
        # Persistence failure must not block the kill (user explicitly
        # wants the daemon down). Worst case: one extra respawn cycle.
        logger.debug("sentinel write failed (non-blocking): %s", exc)

    uid = os.getuid()
    if _is_macos():
        from iai_mcp.lifecycle_lock import LifecycleLock, _is_pid_alive

        payload = LifecycleLock().read()
        pid = payload["pid"] if payload else None

        # Disable supervisor respawn governance FIRST so a later SIGKILL
        # cannot trigger a crash-respawn of the daemon we are stopping.
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(LAUNCHD_TARGET)],
            check=False, capture_output=True,
        )

        if pid is None:
            # No PID to signal; KeepAlive is already removed above.
            return 0

        # Cross-process escalation, independent of the in-loop handler.
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
                    return 0  # clean exit within the bound; no SIGKILL
                _time.sleep(interval)

            # Still alive at the deadline -> uncatchable kill (the guarantee).
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
    """Return p90 of `total_cached_tokens` over the most recent 100 session_started events.

    Reuses the existing emit in ``session.py`` — every call
    to `assemble_session_start` already persists `data["total_cached_tokens"]`
    on a `kind="session_started"` event, so no new event kind is needed.

    Returns
    -------
    dict
        ``{"p90": int | None, "n_samples": int}``. ``p90`` is ``None`` when
        ``n_samples == 0`` (no crash on empty store). For under-filled windows
        (``n_samples < 100``) ``p90`` is still computed and ``n_samples`` is
        echoed so the caller can warn. Uses
        ``statistics.quantiles(..., n=10, method="inclusive")[8]`` so a 100-
        uniform input returns the input value exactly.
    """
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
    # n=10 splits into deciles; index 8 is the 90th percentile boundary.
    # `inclusive` makes uniform inputs return the input value exactly.
    q = statistics.quantiles(samples, n=10, method="inclusive")
    p90 = int(round(q[8]))
    return {"p90": p90, "n_samples": len(samples)}


def _compute_p90_from_events(events: list[dict]) -> dict[str, int | None]:
    """Compute session_start_tokens p90 from a pre-fetched event list.

    Mirrors compute_session_start_tokens_p90 but accepts an event list
    directly (used when events are sourced from the daemon socket rather
    than a live MemoryStore).
    """
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
    """Print daemon_stats output given a p90 result dict."""
    p90_str = str(result["p90"]) if result["p90"] is not None else "no-data"
    print(f"session_start_tokens_p90: {p90_str}")
    print(f"n_samples: {result['n_samples']}")
    if 0 < (result["n_samples"] or 0) < 100:
        print(f"note: rolling window under-filled (have {result['n_samples']}, need 100)")


def cmd_daemon_stats(args: argparse.Namespace) -> int:
    """Longitudinal metrics: session_start_tokens_p90 + n_samples.

    Socket-first: queries the daemon via JSON-RPC events_query so the
    running daemon's Hippo lock is not contended. Falls back to a direct
    MemoryStore open when the daemon is down (socket absent or unreachable).
    If the fallback also fails because the daemon still holds the lock
    (e.g. mid-REM socket timeout), the error is caught and a clean message
    is printed instead of crashing the operator terminal.

    Output schema (stdout, one field per line):
        session_start_tokens_p90: <int or "no-data">
        n_samples: <int>          # 0..100; under 100 means window not yet full

    Note: session_started must be present in EVENTS_QUERY_WHITELIST for the
    socket path to succeed. If the live daemon predates that whitelist entry,
    the socket returns an error dict and the fallback activates automatically.
    """
    # --- socket path (daemon up and holding the lock) ---
    resp = _send_jsonrpc_request("events_query", {"kind": "session_started", "limit": 100})
    if isinstance(resp, dict) and "result" in resp:
        payload = resp["result"]
        if isinstance(payload, dict) and "events" in payload:
            result = _compute_p90_from_events(payload["events"])
            _render_daemon_stats(result)
            return 0

    # --- direct-open fallback (daemon down / socket absent / whitelist miss) ---
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
    """Socket round-trip + version-skew detection."""
    import asyncio  # lazy: asyncio.TimeoutError needed for the except clause
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

    # Version skew check: compare daemon's reported version with installed.
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
    """Cooperative force: wait up to 15min for current cycle to finish."""
    import asyncio  # lazy: asyncio.TimeoutError needed for the except clause
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
    """Per-setting overrides written to ~/.iai-mcp/.daemon-state.json.

    Subcommands:
      - set-budget <float>          -- daily_quota_pct_override
      - set-cycle-count <int>       -- cycle_count_override
      - set-quiet-window HH:MM-HH:MM -- quiet_window_manual_override
      - disable-claude              -- claude_enabled = False (force Tier-0)
      - enable-claude               -- claude_enabled = True
    """
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
    """Show the most recent llm_health event in the user's local timezone.

    Socket-first: asks the daemon via JSON-RPC events_query so the running
    daemon's Hippo lock is not contended. Falls back to a direct MemoryStore
    open when the daemon is down (socket absent or unreachable). If the
    fallback also fails because the daemon still holds the lock (e.g. mid-REM
    socket timeout), the error is caught and a clean message is printed.
    """
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

    # --- socket path (daemon up and holding the lock) ---
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

    # --- direct-open fallback (daemon down / socket absent) ---
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
    """Compile iai_mcp_native via maturin develop --release.

    Use after a Python-version upgrade or on a fresh clone where the
    Rust extension is absent. Requires cargo and rustup on PATH.
    """
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
    """Batch-capture a Claude Code JSONL transcript into the store.

    Called by ~/.claude/hooks/iai-mcp-session-capture.sh on Stop event.
    Fail-safe by design: any exception logs and returns 0 so the hook never
    blocks session teardown.

    ``--no-spawn`` ALWAYS writes a deferred-captures JSONL file under
    ``~/.iai-mcp/.deferred-captures/<id>-<ts>.jsonl`` and exits 0 within
    2s — NEVER spawning the daemon, NEVER importing
    ``iai_mcp.capture.capture_transcript`` (which transitively loads
    the native Rust embedder in a brand-new subprocess).
    The daemon's WAKE drain loop consumes the deferred file later with
    the daemon-process embedder that's already loaded.

    Default mode (without ``--no-spawn``) embeds eagerly — user-explicit
    ``iai-mcp capture-transcript`` invocations behave as documented.
    """
    import json
    import sys as _sys

    no_spawn = bool(getattr(args, "no_spawn", False))

    if no_spawn:
        # Hook is best-effort. ALWAYS defer; the daemon's WAKE drain picks up
        # the JSONL file within seconds with its already-loaded embedder, which
        # is dramatically cheaper than cold-loading bge-small-en-v1.5 in
        # short-lived Stop-hook subprocesses per day.
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
            # Fail-safe: hook MUST exit 0. Log to stderr, return 0.
            logger.error("capture-transcript --no-spawn failed: %s", e)
            print(
                f"capture-transcript --no-spawn: failed {type(e).__name__}: {e}",
                file=_sys.stderr,
            )
            return 0

    # Default path (no --no-spawn): inline-ingest behavior, unchanged.
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
    """Append one event per new transcript turn to {sid}.live.jsonl.

    Reads the line-count offset from ~/.iai-mcp/.capture-state/{sid}.offset,
    skips already-seen lines, appends up to ``--max-turns-per-call`` events
    via ``write_deferred_event``, persists the new line count atomically.

    Truncation (transcript shorter than stored offset) resets offset to 0.
    Missing transcript = no-op exit 0. Fail-safe: any exception → exit 0.
    """
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


# ---------------------------------------------------------------------------
# Capture-hooks installer (makes ambient WRITE-capture portable).
# ---------------------------------------------------------------------------

def _capture_hook_paths() -> tuple:
    """Return (hook_src_traversable, hook_dst_in_home, settings_path).

    hook_src is a package-data Traversable; callers must use .read_bytes()
    to copy rather than shutil.copy2 (Traversable may be inside a zip).
    """
    src = _res.files("iai_mcp") / "_deploy" / "hooks" / "iai-mcp-session-capture.sh"
    dst = Path.home() / ".claude" / "hooks" / "iai-mcp-session-capture.sh"
    settings = Path.home() / ".claude" / "settings.json"
    return src, dst, settings


def _turn_hook_paths() -> tuple:
    """Return (turn_hook_src_traversable, turn_hook_dst_in_home)."""
    src = _res.files("iai_mcp") / "_deploy" / "hooks" / "iai-mcp-turn-capture.sh"
    dst = Path.home() / ".claude" / "hooks" / "iai-mcp-turn-capture.sh"
    return src, dst


def _claude_desktop_config_path() -> Path | None:
    """Locate the Claude Desktop app config file, or None if Desktop isn't
    installed. Claude Desktop and Claude Code CLI use SEPARATE config files:

      - Claude Code CLI:  ~/.claude.json (managed by `claude mcp add`)
      - Claude Desktop:   platform-specific path (this function)

    So MCP registered via `claude mcp add` is NOT visible to Desktop, which
    is why iai-mcp has to be registered in both configs independently.
    """
    import platform as _plat
    home = Path.home()
    sysname = _plat.system()
    if sysname == "Darwin":
        p = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif sysname == "Windows":
        appdata = os.environ.get("APPDATA") or str(home / "AppData" / "Roaming")
        p = Path(appdata) / "Claude" / "claude_desktop_config.json"
    else:  # Linux / BSD
        xdg = os.environ.get("XDG_CONFIG_HOME") or str(home / ".config")
        p = Path(xdg) / "Claude" / "claude_desktop_config.json"
    return p if p.parent.exists() else None


def _resolve_wrapper_path() -> Path:
    """Resolve the MCP wrapper index.js path.

    Tries in order:
    1. IAI_MCP_WRAPPER_PATH env var (dev override / emergency escape hatch)
    2. importlib.resources package-data path (wheel install, primary production path)
    3. mcp-wrapper/dist/ relative to the package source root (editable install fallback)

    Raises FileNotFoundError with actionable instructions if none resolve.
    """
    import iai_mcp as _pkg

    # 1. Env override — wins unconditionally; fail-loud if the pointed file is absent.
    env_val = os.environ.get("IAI_MCP_WRAPPER_PATH")
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"IAI_MCP_WRAPPER_PATH={env_val!r} is set but the file does not exist."
        )

    # 2. Package-data path (built wheel, post-build-hook install).
    try:
        pkg_p = Path(str(_res.files("iai_mcp") / "_wrapper" / "index.js"))
        if pkg_p.exists():
            return pkg_p
    except (TypeError, FileNotFoundError):
        pass

    # 3. Editable-install fallback: mcp-wrapper/dist/ in the source tree.
    src_file = Path(_pkg.__file__).resolve()
    repo_root = src_file.parent.parent.parent  # src/iai_mcp/__init__.py -> repo root
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
    """Build the mcpServers entry for iai-mcp.

    Uses sys.executable for IAI_MCP_PYTHON — always the interpreter running
    this CLI, which is the one that has iai_mcp installed. Resolves the
    wrapper path via _resolve_wrapper_path (package-data then editable fallback).
    """
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
    """action: 'install' | 'uninstall'. Returns a status message for logging.

    install: add/overwrite mcpServers.iai-mcp in the Desktop config.
    uninstall: remove mcpServers.iai-mcp; leave other servers + preferences
    untouched. Idempotent. If Desktop isn't installed, return a skip message.
    """
    import json as _json

    cfg_path = _claude_desktop_config_path()
    if cfg_path is None:
        return "Claude Desktop: not installed (no config dir) — skipped"

    if not cfg_path.exists():
        if action == "uninstall":
            return f"Claude Desktop: {cfg_path} absent — skipped"
        # install: create minimal config with just our entry.
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

    # install
    new_entry = _build_iai_mcp_server_entry()
    if servers.get("iai-mcp") == new_entry:
        return f"Claude Desktop: {cfg_path} already has iai-mcp — no change"
    servers["iai-mcp"] = new_entry
    cfg_path.write_text(_json.dumps(data, indent=2))
    return f"Claude Desktop: patched {cfg_path} (iai-mcp registered)"


def _patch_claude_code_config(action: str) -> str:
    """action: 'install' | 'uninstall'. Returns a status message for logging.

    Writes mcpServers.iai-mcp into ~/.claude.json (the Claude Code CLI config).
    Claude Code and Claude Desktop use SEPARATE config files; this function
    owns the Claude Code side. If the MCP wrapper is not yet built, the config
    is still written (with a best-effort wrapper path) so IAI_MCP_PYTHON is
    correct immediately; the user will see a node error at MCP startup until
    the wrapper is built.
    """
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

    # install
    try:
        entry = _build_iai_mcp_server_entry()
    except FileNotFoundError as exc:
        # Wrapper not yet built — write partial entry so IAI_MCP_PYTHON is
        # set correctly; user must build the wrapper before MCP starts.
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
    """Return (hook_src_traversable, hook_dst_in_home, settings_path) for the
    SessionStart recall hook. Mirrors _capture_hook_paths semantics."""
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
    """Copy both hook scripts into ~/.claude/hooks/ and register Stop +
    UserPromptSubmit entries in settings.json. Idempotent on re-run."""
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

    # Register iai-mcp in both Claude Code (~/.claude.json) and Claude Desktop
    # (claude_desktop_config.json) — they are separate apps with separate configs.
    code_msg = _patch_claude_code_config("install")
    print(code_msg)
    desktop_msg = _patch_claude_desktop_config("install")
    print(desktop_msg)

    print("\nNext: fully quit + relaunch Claude Code AND Claude Desktop")
    print("      so both pick up the registration (macOS: `killall Claude`).")
    print("Verify: iai-mcp capture-hooks status")
    return 0


def cmd_capture_hooks_uninstall(args: argparse.Namespace) -> int:
    """Remove all three hook scripts and their settings.json entries
    (idempotent)."""
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

    # Unregister from both Claude Code and Claude Desktop configs.
    code_msg = _patch_claude_code_config("uninstall")
    print(code_msg)
    desktop_msg = _patch_claude_desktop_config("uninstall")
    print(desktop_msg)

    return 0


def cmd_capture_hooks_status(args: argparse.Namespace) -> int:
    """Show whether all three hooks (Stop / UserPromptSubmit / SessionStart)
    are installed and active on both surfaces."""
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

    # Claude Desktop (separate config file, separate app).
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
    # Desktop wiring is a bonus, not a requirement — if Desktop isn't
    # installed there's no surface to wire up. Only flag INACTIVE when
    # Desktop IS installed but not wired.
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
    """Run the appropriate migration based on --from / --to version pair,
    or a crash-safe-reembed action (--resume / --rollback).

    Supported:
      --from=1 --to=2   schema v1 -> v2
      --from=2 --to=3   encryption-at-rest migration
      --from=3 --to=4   TEM factorization
      --rollback        drop records_v_new and (if needed) restore records
                        from records_old_<ts>. Routes to migrate._rollback.
                        Exit codes: 0 success, 1 user-correctable error,
                        2 unrecoverable.
      --resume          continue an interrupted reembed migration from
                        migration_progress.json. Routes to migrate._resume
                        with the live store's embedder. Same exit-code contract.

    Anything else returns exit code 2 with a clear error message.
    """
    from iai_mcp.store import MemoryStore
    store = MemoryStore()

    # Rollback / resume entry points. Mutually exclusive with the --from/--to
    # dispatch below; checked first so they short-circuit.
    if bool(getattr(args, "rollback", False)):
        from iai_mcp import migrate
        return migrate._rollback(store.db, store)
    if bool(getattr(args, "resume", False)):
        # Resume requires the same target embedder the original migration
        # used. The simplest contract: resume to the embedder configured in
        # the running environment (IAI_MCP_EMBED_DIM).
        # The progress-file's saved_target_dim is cross-checked in
        # migrate._resume — a mismatch returns rc=1.
        from iai_mcp import migrate
        from iai_mcp.embed import embedder_for_store
        target = embedder_for_store(store)
        return migrate._resume(store.db, store, target)

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
        # TEM factorization migration: rename the legacy `hd_vector_json`
        # (pa.string()) column to `structure_hv` (pa.binary()) and backfill
        # every row via tem.bind_structure().
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
    """Report file-backend key state.

    Output is a single JSON document with the file-backend invariants:
      - backend = "file"
      - path = absolute key-file path
      - present = file exists
      - mode = "0o600" + mode_secure flag (true iff group/world bits are zero)
      - uid + uid_matches_process flag
      - length_bytes + length_valid (== KEY_BYTES)
      - passphrase_fallback_set (whether IAI_MCP_CRYPTO_PASSPHRASE is set)
      - hint when the file is missing (dual-remediation message)

    Never prints the key bytes (information-disclosure mitigation).
    """
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
    """Rotate the encryption key and re-encrypt every record.

    Flow:
    1. Load current key + decrypt all records into in-memory MemoryRecord list.
    2. Rotate the key file (writes a fresh 32 bytes via _try_file_set, atomic
       temp+rename, mode 0o600). Also invalidates the cached AESGCM bound to
       the old key so subsequent encrypts use the fresh key.
    3. Re-encrypt every record with the new key via a delete+insert cycle.

    Events data_json is also re-encrypted.
    """
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

    # 1) Read everything under the old key (decryption is automatic).
    decrypted_records = store.all_records()

    # Decrypt events payloads up front so we can re-encrypt after rotation.
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

    # 2) Rotate the key (this flips store._crypto_key via wrapper cache).
    new_key = store._crypto_key_wrapper.rotate()
    store._crypto_key = new_key  # Force subsequent encrypts under the fresh key.
    # Invalidate the cached AESGCM bound to the old key. Without this, the
    # next encrypt would use AESGCM(old_key) and produce ciphertext that
    # cannot be decrypted under new_key.
    store._invalidate_aesgcm_cache()

    # 3) Re-encrypt every record via delete + insert (MVCC-safe).
    tbl = store.db.open_table(RECORDS_TABLE)
    record_count = 0
    for rec in decrypted_records:
        try:
            tbl.delete(f"id = '{_uuid_literal(rec.id)}'")
        except (OSError, ValueError, RuntimeError):
            pass
        # store.insert() encrypts using the new cached key.
        try:
            store.insert(rec)
            record_count += 1
        except (OSError, ValueError, RuntimeError):
            continue

    # Re-encrypt events data_json under the new key.
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
    """Re-stage all records and swap after decrypting with a prior AES key."""
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
    """CLI entry for literal_surface redaction when decrypt fails."""
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
    """One-time migration from macOS Keychain to file backend.

    Reads the existing key from the macOS Keychain (the call that hangs in
    launchd context — this command MUST be run from an interactive Terminal
    so the Keychain ACL prompt can appear and the user can click "Always Allow"),
    writes it to ``{store_root}/.crypto.key``, verifies a round-trip read.

    Idempotent: a valid existing file is a no-op success that does NOT touch
    keyring. If the file exists but is malformed, the command refuses with a
    clear error pointing at the file path; user must remove the file manually
    before retrying.

    Default ``--keep-keychain`` leaves the keyring entry in place (lower-risk
    default; user can manually delete via Keychain Access.app).
    ``--delete-keychain`` deletes the entry only AFTER round-trip verification
    succeeds.
    """
    import base64 as _b64
    # LOCAL import: crypto.py + everything else stays keyring-free at module
    # scope. The migration command itself is the ONLY in-process code path that
    # imports keyring.
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

    # Idempotent path: if the file is already valid, exit 0 without touching
    # keyring.
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

    # Read from macOS Keychain (this is THE call that hangs in launchd;
    # interactive Terminal only).
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

    # Write via the atomic helper.
    try:
        ck._try_file_set(source)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"failed to write key file: {exc}", file=sys.stderr)
        return 1

    # Round-trip verification: read what we just wrote, byte-compare.
    try:
        roundtrip = ck._try_file_get()
    except CryptoKeyError as exc:
        # Read-back failed; remove the partial file.
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

    # Success path.
    path = ck._key_file_path()
    print(f"migrated: {path} (mode 0o600, {KEY_BYTES} bytes)")

    if not keep_keychain:
        try:
            _keyring.delete_password(SERVICE_NAME_DEFAULT, user_id)
            print(f"deleted keyring entry for user_id={user_id!r}")
        except _keyring_errors.PasswordDeleteError:
            # Already absent — treat as success.
            pass
        except _keyring_errors.KeyringError as exc:
            # Non-fatal: file is written + verified, keyring delete failed;
            # print warning and continue (exit 0).
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
    """Generate a fresh ``.crypto.key`` (fresh installs only).

    Refuses if the file already exists (any state, valid or malformed). The
    ONLY code path in the project that creates a fresh key — daemon
    refusal-to-start explicitly forbids silent key generation.

    To rotate an existing key, use ``iai-mcp crypto rotate``. To wipe and
    start over, the user must remove the file manually before re-running
    ``crypto init``.
    """
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


# Substring fallback handler for the bank-recall CLI — no daemon, no
# store, no embedder.
def cmd_bank_recall(args: argparse.Namespace) -> int:
    """Print a JSON memory_recall-shape response to stdout, substring-only.

    Reads the disk-side bank/processed + bank/recent artifacts without
    opening the store or loading the embedder. CryptoKey resolution
    happens lazily inside ``read_recent_records`` when ``key`` is None.
    """
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
    """Print live small-world topology snapshot.

    One key:value line per metric:

        C: <average clustering>
        L: <characteristic path length>
        sigma: <fast_sigma() | "insufficient_data">
        communities: <Leiden community count>
        rich_club_ratio: <|rich_club| / N>
        N: <node count>
        regime: <"developmental" | "mid_life_drift" | "healthy" | "insufficient_data">

    sigma is a diagnostic only; never a routing decision. The CLI is a
    print-only command -- no event writes, no state mutation.
    compute_and_emit() runs in S4's offline pass instead (see iai_mcp.s4.run_offline_pass).

    Socket-first: when the daemon is running it holds the HippoDB exclusive lock.
    Routing through the AF_UNIX socket lets the daemon answer while keeping its own
    lock -- no deadlock, no HippoLockHeldError. Direct-open fallback activates when
    the socket is absent or unreachable (daemon down), at which point the lock is free.
    If the socket times out (daemon mid-REM) and the direct open fails because the
    daemon still holds the lock, the exception is caught and topology degrades to
    insufficient_data -- the command never crashes.
    """

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

    # --- socket path (daemon up and holding the lock) ---
    resp = _send_jsonrpc_request("topology", {})
    if isinstance(resp, dict):
        result = resp.get("result")
        if isinstance(result, dict):
            _render(result)
            return 0

    # --- direct-open fallback (daemon down / socket absent) ---
    # Guard: if the daemon is mid-REM the socket may time out yet the lock is
    # still held. Catch HippoLockHeldError and degrade gracefully instead of
    # crashing the operator terminal.
    from iai_mcp.hippo import HippoLockHeldError
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.sigma import compute_topology_snapshot
    from iai_mcp.store import MemoryStore

    try:
        store = MemoryStore()
        graph, _assignment, _rich_club = build_runtime_graph(store)
        snap = compute_topology_snapshot(graph)
    except HippoLockHeldError:
        # Daemon holds the lock (e.g. mid-REM socket timeout). Degrade, don't crash.
        _render({})
        return 0

    _render(snap)
    return 0


def cmd_drain_permanent_failed(args: argparse.Namespace) -> int:
    """Recover terminal .permanent-failed-*.jsonl files in .deferred-captures/.

    Socket-first: when the daemon is running it holds the HippoDB exclusive
    lock. Routing through the AF_UNIX socket lets the daemon perform the drain
    under its own lock — no second writer, no HippoLockHeldError. Direct-open
    fallback activates when the socket is absent or unreachable (daemon down),
    at which point the lock is free.

    --dry-run lists the files + event counts without renaming or inserting
    anything (safe to run at any time).
    """
    dry_run = bool(getattr(args, "dry_run", False))

    # --- socket path (daemon up and holding the lock) ---
    resp = _send_jsonrpc_request("drain_permanent_failed", {"dry_run": dry_run}, read_timeout=120.0)
    if isinstance(resp, dict):
        result = resp.get("result")
        if isinstance(result, dict):
            _print_drain_result(result)
            return 0

    # --- direct-open fallback (daemon down / socket absent) ---
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
    """Pretty-print the drain_permanent_failed_files result dict."""
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
    """Build the same {metric: [(ts, value), ...]} dict as aggregate_trajectory
    but from a pre-fetched socket event list rather than a live MemoryStore.

    ts values are ISO strings from the socket — kept as strings here because
    the render path only uses the float values, not timestamps.
    """
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
    """Print trajectory output — shared by socket and direct-open paths."""
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
    """Aggregate M1..M6 trajectory events.

    Socket-first: asks the daemon via JSON-RPC events_query so the running
    daemon's Hippo lock is not contended. Falls back to a direct MemoryStore
    open when the daemon is down (socket absent or unreachable). If the
    fallback also fails because the daemon still holds the lock (e.g. mid-REM
    socket timeout), the error is caught and a clean message is printed.

    Window note: the socket path is capped at 1000 events by the daemon's
    events_query dispatcher. The direct-open fallback uses limit=10000.
    For large stores with many trajectory events the two paths may show
    slightly different aggregates; the socket path covers ~1000 most-recent
    trajectory_metric events while the fallback covers up to 10000.
    """
    from datetime import datetime, timedelta, timezone

    from iai_mcp.trajectory import METRIC_NAMES

    weeks = getattr(args, "since", None)
    since = None
    since_iso = None
    if weeks is not None:
        since = datetime.now(timezone.utc) - timedelta(weeks=int(weeks))
        since_iso = since.isoformat()

    # --- socket path (daemon up and holding the lock) ---
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

    # --- direct-open fallback (daemon down / socket absent) ---
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
    """Render a shield event's data dict with matched-pattern redaction.

    shield_rejection / shield_flag events store the matched
    patterns. CLI output shows ONLY the count to avoid leaking the shield's
    signal-word dictionary to attackers inspecting logs.
    """
    matched = data.get("matched") or []
    tier = data.get("tier", "-")
    record_id = data.get("record_id", "-")
    action = data.get("action", "-")
    return (
        f"tier={tier} action={action} "
        f"matched_count={len(matched)} record_id={record_id}"
    )


def _format_audit_event(event: dict, tz) -> str:
    """Single-line audit event rendering in the user's local TZ.

    Handles both datetime objects (direct-open path) and ISO-string timestamps
    (socket path, where the daemon serializes ts to an ISO-8601 string).
    """
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
    """Render identity-event audit log.

    Accepts a sub-command via the `audit_sub` attribute:
      - None / 'all'      -- full audit (s5_* + shield_* + drift alerts)
      - 'shield'          -- shield events only
      - 'drift'           -- runs detect_drift_anomaly + prints status
      - 'identity'        -- s5_* events only (no shield)

    Shared flags: --since WEEKS, --severity SEV.

    Socket-first: asks the daemon via JSON-RPC (audit_query / detect_drift)
    so the running daemon's Hippo lock is not contended. Falls back to a
    direct MemoryStore open when the daemon is down (socket absent or
    unreachable). If the fallback also fails because the daemon still holds
    the lock (e.g. mid-REM socket timeout), the error is caught and a clean
    message is printed instead of crashing the operator terminal.
    """
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

    # ------------------------------------------------------------------ drift
    # detect_drift also writes s5_drift_alert events (side-effect preserved).
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

        # Fallback: direct-open.
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

    # ------------------------------------------------------------ event modes
    # All of shield / identity / all use audit_query with a kinds list.
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
        # Default: full audit (all kinds).
        from iai_mcp.s5 import AUDIT_EVENT_KINDS
        audit_kinds = list(AUDIT_EVENT_KINDS)
        empty_msg = "No identity events recorded"

    severity = getattr(args, "severity", None)

    # --- socket path ---
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

    # --- direct-open fallback ---
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
    """Schema-cleanup CLI dispatch.

    Soft-deletes duplicate schema records that accumulated in production
    stores before `persist_schema` was made idempotent.

    Default mode is --dry-run (reversibility).
    --apply requires the explicit flag; no interactive prompts so the
    flow is reproducible and testable.

    `--store-path` targets the IAI root directory (the path passed to
    MemoryStore() — contains the `hippo/` subdir with the actual tables).
    Default is ~/.iai-mcp (matches MemoryStore() no-args default per
    DEFAULT_STORAGE_PATH).
    """
    from iai_mcp.migrate import cleanup_schema_duplicates
    from iai_mcp.store import MemoryStore

    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        # Match MemoryStore() default semantics: store.root = ~/.iai-mcp
        # (the IAI root); Hippo data lives at store.root / "hippo".
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


# ---------------------------------------------------------------------------
# One-shot Hippo storage compaction CLI
# ---------------------------------------------------------------------------
#
# Wraps `optimize_hippo_storage()` from `iai_mcp.maintenance` with:
#   - daemon-stopped pre-flight (psutil cmdline check rules out PID-recycle)
#   - record-id set equality assertion (verbatim-recall invariant)
#   - audit JSON trail (UTC ISO timestamp; mirrors `.consent-{ts}.json` shape)
#
# This CLI runs WITH DAEMON STOPPED. The optimize call is pure storage
# compaction — never reads or paraphrases stored `literal_surface`.
# ---------------------------------------------------------------------------


def _maintenance_compact_preflight_daemon_alive() -> str | None:
    """Return None if the daemon is NOT alive (safe to proceed); return a
    friendly error string if alive (caller prints to stderr + returns 1).

    Defense in depth: read `~/.iai-mcp/.daemon-state.json`, extract
    `daemon_pid`. If absent, daemon is not alive → None. If present, check
    `os.kill(pid, 0)` (does NOT signal — only checks process existence).
    If alive, confirm `psutil.Process(pid).cmdline()` contains
    `iai_mcp.daemon` to rule out PID-recycle false positives.
    """
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
    # Process exists. Confirm it is iai_mcp.daemon (not PID recycle).
    try:
        import psutil
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline())
    except Exception as exc:
        # If psutil cannot inspect, conservatively treat as alive — REFUSE.
        logger.debug("psutil inspect pid %d failed: %s", pid, exc)
        return (
            f"daemon running (pid {pid}); run `iai-mcp daemon stop` "
            f"first, then retry"
        )
    if "iai_mcp.daemon" not in cmdline:
        return None  # PID recycle — not our daemon.
    return (
        f"daemon running (pid {pid}); run `iai-mcp daemon stop` first, "
        f"then retry"
    )


def _maintenance_compact_metrics(
    hippo_dir: Path,
    store: object | None = None,
) -> dict:
    """Capture metrics for the Hippo storage backend.

    Returns dict with keys: db_size_mb, records_count, record_id_set.
    `store` may be None on the dry-run pass when caller only checks the
    directory; on the apply pass it must be a live MemoryStore so we can
    read tbl.count_rows() and the record-id set.
    """
    # Measure the SQLite DB file size directly (single authoritative file).
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
    """--dry-run: open the store, capture pre-metrics, print JSON; do NOT
    call optimize, do NOT write an audit file.
    """
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
    """--apply: open store, capture pre-metrics, call optimize_hippo_storage()
    on records/edges/events, capture post-metrics, assert record-id set
    equality on the records table, write audit file.
    """
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

    # Post: re-open store for fresh metadata view.
    store_after = MemoryStore(path=store_path)
    post_metrics = _maintenance_compact_metrics(hippo_dir, store=store_after)
    post_id_set = post_metrics["record_id_set"]

    # Verbatim-recall invariant — record-id set equality.
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
    """Compact the Hippo storage: wal_checkpoint + VACUUM + hnswlib rebuild and
    atomic save. Requires the daemon to be stopped (pre-flight check refuses
    to proceed when daemon is alive).

    Pre-flight: refuse if the daemon process is alive (PID + cmdline check).
    Mode: `--dry-run` (default) prints metrics-only JSON; `--apply --yes`
    runs `optimize_hippo_storage()` on the storage, asserts record-id set
    equality on the records table, and writes an audit JSON.

    Exit codes: 0 ok, 1 pre-flight refusal or invariant abort, 2 wrong-flag
    combo (apply without yes on a non-tty).

    This CLI runs with the daemon stopped, so `_should_yield_to_mcp` is
    irrelevant. The optimize call never paraphrases or smooths stored
    content — it is pure storage compaction.
    """
    # Emit deprecation warning when invoked via the legacy alias.
    if getattr(args, "maintenance_cmd", None) == "compact-records":
        print(
            "warning: compact-records is the deprecated name for "
            "compact-hippo; use compact-hippo going forward",
            file=sys.stderr,
        )

    # Resolve store path (same convention as cmd_schema_cleanup).
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
    # Default to dry-run when neither flag set.
    if not apply:
        # Treat `--dry-run` and "neither flag" identically.
        return _maintenance_compact_dry_run(store_path, hippo_dir)

    # --apply path: pre-flight + optional consent + optimize + invariant.
    # Pre-flight 1: daemon alive?
    refusal = _maintenance_compact_preflight_daemon_alive()
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 1

    # Pre-flight 2: --apply on non-tty without --yes is refused.
    if not yes and not sys.stdin.isatty():
        print(
            "error: --apply on non-tty requires --yes (refusing to proceed "
            "without interactive consent or explicit --yes)",
            file=sys.stderr,
        )
        return 2

    # Pre-flight 3: interactive consent.
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
    """Deprecated alias for cmd_maintenance_compact_hippo.

    Kept for one release cycle; prints a deprecation warning to stderr
    and delegates to the new function.
    """
    # Ensure the deprecation warning fires via the maintenance_cmd check
    # in cmd_maintenance_compact_hippo.
    args.maintenance_cmd = "compact-records"
    return cmd_maintenance_compact_hippo(args)


def cmd_maintenance_symmetrize_self_loops(args: argparse.Namespace) -> int:
    """Backfill missing hebbian self-loops on existing records.

    Older stores have a per-record asymmetry: dedup-touched records had
    `(rid, rid)` hebbian self-loops; fresh-INSERT records did not. A
    write-path fix closed the source; this CLI backfills existing stores
    so every record has a self-loop (degree-norm symmetry across corpus).

    Pre-flight: refuse if the daemon process is alive (PID + cmdline
    check). Reuses `_maintenance_compact_preflight_daemon_alive`.
    Mode: `--dry-run` (default) prints counts JSON; `--apply --yes`
    writes the missing self-loops via `store.boost_edges` at delta=0.1.

    Exit codes:
      0 ok
      1 pre-flight refusal or user-declined-consent
      2 wrong-flag combo (apply without yes on a non-tty)
    """
    # Resolve store path (same convention as compact-hippo).
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
        # Default to dry-run when neither flag set.
        store = MemoryStore(path=store_path)
        result = symmetrize_self_loops(store, dry_run=True)
        print(json.dumps(result, indent=2))
        return 0

    # --apply path: pre-flight + optional consent + write.
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


# ---------------------------------------------------------------------------
# iai-mcp lifecycle status
# ---------------------------------------------------------------------------

def _format_relative(ts_iso: str, now: datetime | None = None) -> str:
    """Render a friendly elapsed string for an ISO-8601 UTC timestamp.

    Output examples: "12 minutes", "3 hours", "2 days". Used by
    `cmd_lifecycle_status` to mirror the spec's "(12 minutes)" suffix
    next to the `since:` line.
    """
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
    """Clear ``~/.iai-mcp/.locked``.

    Operator-facing recovery path for a stale lockfile that the
    daemon's own dead-PID takeover did not clear (e.g. cross-host
    iCloud/NFS sync where the user wants to wipe the foreign
    hostname BEFORE booting a new daemon, or a corrupt schema
    bump that the operator wants to inspect).

    Output: prints the prior payload (PID + hostname + started_at)
    so the operator can confirm what was cleared. ``--yes`` skips
    the interactive [y/N] prompt; tests pass ``--yes`` to avoid
    blocking on input().

    Exit codes:
      0 -- file cleared (or absent already, which is also "clear")
      1 -- user declined the prompt
    """
    from iai_mcp.lifecycle_lock import DEFAULT_LOCK_PATH, LifecycleLock

    # Resolve the lock-path. Tests inject ``args.lock_path`` to point
    # at a tmp file; production callers fall through to the default.
    lock_path = getattr(args, "lock_path", None)
    if lock_path is not None:
        lock = LifecycleLock(Path(lock_path))
    else:
        lock = LifecycleLock(DEFAULT_LOCK_PATH)

    existing = lock.read()
    if existing is None:
        print("No lockfile present; nothing to unlock.")
        return 0

    # Diagnostic surface so the operator can verify what they are clearing.
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
        # Race: file vanished between our read and unlink. Same exit
        # status -- the desired end state ("no lockfile") is reached.
        print("Lockfile already removed by another process.")
        return 0
    print("Lockfile removed.")
    return 0


def cmd_lifecycle_status(args: argparse.Namespace) -> int:
    """Print formatted snapshot of `lifecycle_state.json`.

    Returns 0 unless the state file is unreadable in a way that bypasses
    the self-heal path (rare; load_state recovers from missing/corrupt
    files by returning a fresh default WAKE record).
    """
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
        # Prefer the canonical key `last_completed_index` (position into
        # `SleepPipeline._STEP_ORDER`); fall back to the legacy
        # `last_completed_step` key (a `SleepStep.<NAME>.value`) so older
        # `lifecycle_state.json` files render without a daemon restart.
        # The display field name `step` stays — operator-facing shorthand.
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


# ---------------------------------------------------------------------------
# iai-mcp maintenance sleep-cycle
# ---------------------------------------------------------------------------
#
# CLI surface for the SleepPipeline. Two flags:
#   --force              Run even when quarantined (operator override).
#   --reset-quarantine   Clear quarantine first; then run normally.
#
# Output: one line per step in `[N/5] step_name ... ok (Ms)` format,
# plus a final summary line. On quarantine without --force, exits non-zero
# with an informational message pointing at --force / --reset-quarantine.
# ---------------------------------------------------------------------------


def cmd_maintenance_sleep_cycle(args: argparse.Namespace) -> int:
    """Run the sleep pipeline once.

    Exit codes:
      0 — success (5/5 steps complete) OR auto-recovery succeeded
      1 — quarantined and --force not specified, OR a step failed
      2 — store could not be opened (rare; same convention as
          other maintenance subcommands)

    The pipeline is invoked synchronously and prints a step-by-step
    progress trail. Output is plain text (NOT JSON) so the operator can
    follow along in a terminal; structured event-log entries cover
    machine-readable telemetry needs.

    No daemon-stopped pre-flight: unlike `compact-hippo`, the sleep
    pipeline calls optimize_hippo_storage inside the running daemon, so
    coexistence with the daemon's own compaction pass is safe.
    Step 5 (compact_records) uses the same helper, run sequentially.
    """
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH
    from iai_mcp.sleep_pipeline import SleepPipeline, SleepStep
    from iai_mcp.store import MemoryStore

    # Resolve store path the same way other maintenance commands do.
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

    # Quarantine gate (when --force is NOT passed).
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

    # Step-name -> 1..N progress index. Derived from `_STEP_ORDER` so
    # future APPEND-without-renumber inserts auto-display in the right
    # slot (the dispatch order is the source of truth, not enum value).
    step_index = {
        step: i + 1 for i, step in enumerate(SleepPipeline._STEP_ORDER)
    }
    total_steps = len(SleepPipeline._STEP_ORDER)

    print("Sleep cycle started.")
    # Run via force_run() if --force was passed, else run().
    runner = pipeline.force_run if force else pipeline.run
    result = runner()

    # Render per-step lines. Note: result["completed_steps"] is the list
    # of steps THIS invocation completed (resumes do NOT replay prior
    # steps), so the prefix is the index of the SleepStep, not its
    # position in completed_steps.
    for step in result["completed_steps"]:
        idx = step_index.get(step, "?")
        # We do not have per-step durations from the result dict (only
        # `duration_sec` for the whole run). Print "ok" without timing
        # to keep the line shape stable; precise per-step timings live
        # in the lifecycle event log under sleep_step_completed.
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
    # Crash-safe-reembed entry points. Additive flags;
    # --from/--to dispatch is unchanged when neither --resume nor --rollback
    # is passed.
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
    m.set_defaults(func=cmd_migrate)

    # Crypto subcommand.
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

    # migrate-to-file + init subcommands.
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

    # live topology snapshot (sigma + C + L + community + rich-club).
    topo = sub.add_parser(
        "topology",
        help="live small-world topology snapshot: C, L, sigma, communities, rich-club ratio, N, regime",
    )
    topo.set_defaults(func=cmd_topology)

    # Ambient capture: capture a Claude Code JSONL transcript into the store
    # (called by ~/.claude/hooks/iai-mcp-session-capture.sh).
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

    # Ambient-capture installer: drops the Stop hook into
    # ~/.claude/hooks/ and patches ~/.claude/settings.json. Makes a fresh
    # install of iai-mcp on another machine a two-step flow:
    #   pip install -e ".[dev,compress]"
    #   iai-mcp capture-hooks install
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

    # Audit subcommand + sub-subcommands.
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

    # Daemon subcommand group.
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

    # schema-cleanup top-level subcommand. NOT under `iai-mcp migrate ...` —
    # the `migrate` namespace is reserved for v-bump schema migrations
    # (v3 -> v4 etc); this is a maintenance op.
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

    # Top-level `maintenance` subcommand for one-shot Hippo compaction.
    # Same placement precedent as `schema-cleanup` and `doctor` — top-level
    # discoverability matters for first-touch ops.
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
    # Deprecation alias: compact-records → compact-hippo (one release cycle).
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

    # maintenance sleep-cycle subcommand: runs the 5-step SleepPipeline
    # (schema_mine -> knob_tune -> dream_decay -> optimize_hippo ->
    # compact_records) once, with quarantine gating + bounded-deferral.
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

    # doctor top-level subcommand. NOT nested under `iai-mcp daemon` —
    # top-level discoverability matters when the user sees
    # `daemon_unreachable`.
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
    # --apply is additive (NOT a mode switch like dry-run/apply on
    # schema-cleanup), so no mutually-exclusive group; --yes is a sub-modifier
    # that cmd_doctor checks for warning-and-ignore semantics if used alone.
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
    # Forces headless mode: downgrades (n) HID idle source + (b) socket
    # file fresh from FAIL to WARN. Auto-detected on Linux when
    # DISPLAY/WAYLAND_DISPLAY are unset; on macOS this flag is the only
    # path (Quartz desktops never set DISPLAY).
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

    # lifecycle status. Top-level placement follows the `doctor` /
    # `maintenance` precedent: first-touch observability matters.
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

    # force-unlock recovery for ~/.iai-mcp/.locked. Operator path;
    # daemon-side dead-PID takeover handles the common case automatically.
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

    # Read-side substring fallback over the memory-bank tier.
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

    # Terminal capture file recovery — socket-first, direct-open fallback.
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
