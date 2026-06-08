"""Bash-hook end-to-end tests for the inlined freshness gate.

These tests drive the REAL bash hook as a subprocess so they cover the
gate code that lives inside the inline PY_SCRIPT — which monkeypatching
cannot reach.

Three cases:
  5a. test_folded_gate_emits_full_oversized_brief
      A triggered gate (Signal A) emits the FULL additionalContext reply via
      a fake AF_UNIX daemon socket returning an oversized (>16 KB) brief.
      Guards the newline-framing fix: a single recv(4096) would truncate the
      reply into invalid JSON and fail to emit additionalContext.

  5b. test_folded_gate_custom_store_does_not_touch_default_socket
      Custom IAI_MCP_STORE + no IAI_DAEMON_SOCKET_PATH: the gate DOES NOT
      contact a fake listener bound at the default socket location.
      accept_count == 0 proves the guard skipped the RPC.

  5c. test_folded_gate_failure_does_not_abort_capture
      A dead gate socket (no server bound): the capture write and offset
      persist still succeed, rc is 0, and stdout contains no additionalContext.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX bash + AF_UNIX",
)

REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-turn-capture.sh"


def _skip_guards():
    """Shared skip logic mirroring the latency-test guards."""
    if not HOOK.exists():
        pytest.skip(f"hook script missing at {HOOK}")
    if not shutil.which("bash"):
        pytest.skip("bash not on PATH")


def _seed_db_and_watermark(home: Path, sid: str, past_ts: str, old_watermark_ts: str) -> None:
    """Create a minimal Hippo SQLite file and an OLD watermark so Signal A fires.

    The DB row has tombstoned_at=NULL so get_max_created_at() sees it.
    The watermark is OLDER than the DB row so store_advanced is True.
    """
    hippo_dir = home / ".iai-mcp" / "hippo"
    hippo_dir.mkdir(parents=True, exist_ok=True)
    db_path = hippo_dir / "brain.sqlite3"
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE records (created_at TEXT, tombstoned_at TEXT)"
    )
    conn.execute(
        "INSERT INTO records (created_at, tombstoned_at) VALUES (?, NULL)",
        (past_ts,),
    )
    conn.commit()
    conn.close()

    state_dir = home / ".iai-mcp" / ".capture-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    wm_path = state_dir / f"{sid}.watermark"
    wm_path.write_text(old_watermark_ts)


def _write_transcript(home: Path, sid: str) -> Path:
    """Write a minimal single-turn transcript so the capture has work to do."""
    transcript = home / f"{sid}.jsonl"
    transcript.write_text(
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "hello gate test"},
        }) + "\n"
    )
    return transcript


def _run_hook(home: Path, sid: str, transcript: Path, extra_env: dict) -> subprocess.CompletedProcess:
    """Run the bash hook as a subprocess with isolated HOME."""
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.update(extra_env)
    # Remove any live daemon socket from the environment unless overridden.
    env.pop("IAI_DAEMON_SOCKET_PATH", None)
    env.update(extra_env)  # apply again so overrides win

    stdin_data = json.dumps({
        "session_id": sid,
        "transcript_path": str(transcript),
        "cwd": str(home),
    })
    return subprocess.run(
        ["bash", str(HOOK)],
        input=stdin_data,
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )


# ---------------------------------------------------------------------------
# 5a: Positive path — oversized brief round-trips intact via newline framing
# ---------------------------------------------------------------------------

def test_folded_gate_emits_full_oversized_brief():
    """Gate emits the FULL (>16 KB) additionalContext via a fake daemon socket.

    A single recv(4096) implementation truncates the reply into invalid JSON
    and the test fails — exactly the latent production bug guarded here.
    The correct newline-accumulating read passes and emits the full surface.
    """
    _skip_guards()

    # Use /tmp directly for short socket paths (darwin AF_UNIX limit ~104 chars).
    home = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        sid = "gate5a-" + uuid.uuid4().hex[:8]
        past_ts = "2026-01-01T00:00:00+00:00"
        old_wm = "2025-12-31T23:59:59+00:00"

        _seed_db_and_watermark(home, sid, past_ts, old_wm)
        transcript = _write_transcript(home, sid)

        # Oversized rendered brief: well above a single recv(4096) chunk.
        big_surface = "## Memory refreshed\n\nALICE-SURFACE-TOKEN " + ("x" * 20000)
        future_ts = "2026-06-01T00:00:00+00:00"
        reply_obj = {"result": {"rendered": big_surface, "new_max_ts": future_ts}}
        reply_frame = (json.dumps(reply_obj) + "\n").encode("utf-8")

        sock_path = str(home / "fake.sock")
        accept_done = threading.Event()

        def _listener():
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(sock_path)
            srv.listen(1)
            srv.settimeout(0.1)
            try:
                while not accept_done.is_set():
                    try:
                        conn, _ = srv.accept()
                    except socket.timeout:
                        continue
                    except Exception:
                        break
                    try:
                        # Read the request line (not strictly needed, but drains the buffer).
                        conn.settimeout(1.0)
                        buf = b""
                        while b"\n" not in buf:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            buf += chunk
                        conn.sendall(reply_frame)
                    except Exception:
                        pass
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    break
            finally:
                try:
                    srv.close()
                except Exception:
                    pass

        t = threading.Thread(target=_listener, daemon=True)
        t.start()
        time.sleep(0.05)  # let the listener bind before the hook fires

        result = _run_hook(home, sid, transcript, {"IAI_DAEMON_SOCKET_PATH": sock_path})
        accept_done.set()

        assert result.returncode == 0, f"Hook rc={result.returncode}\nstderr: {result.stderr}"

        stdout = result.stdout
        assert stdout, "Hook stdout was empty — gate did not emit additionalContext"
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"Hook stdout is not valid JSON (likely truncated reply): {exc}\n"
                f"stdout length={len(stdout)}, first 200 chars: {stdout[:200]}"
            )

        ac = payload.get("hookSpecificOutput", {}).get("additionalContext", "")
        assert "ALICE-SURFACE-TOKEN" in ac, (
            "additionalContext missing the expected surface token"
        )
        assert len(ac) > 16000, (
            f"additionalContext truncated: got {len(ac)} chars, expected > 16000"
        )
    finally:
        shutil.rmtree(str(home), ignore_errors=True)


# ---------------------------------------------------------------------------
# 5b: Custom-store guard — default socket is never contacted
# ---------------------------------------------------------------------------

def test_folded_gate_custom_store_does_not_touch_default_socket():
    """Custom IAI_MCP_STORE + no IAI_DAEMON_SOCKET_PATH: default socket is not contacted.

    A fake listener at the default socket location counts accepts.
    The guard must fire before opening any socket, leaving accept_count == 0.
    """
    _skip_guards()

    home = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        sid = "gate5b-" + uuid.uuid4().hex[:8]
        past_ts = "2026-01-01T00:00:00+00:00"
        old_wm = "2025-12-31T23:59:59+00:00"

        # Seed DB and watermark under HOME so Signal A would otherwise trip.
        _seed_db_and_watermark(home, sid, past_ts, old_wm)
        transcript = _write_transcript(home, sid)

        # Custom store that resolves differently from <home>/.iai-mcp.
        custom_store = home / "custom_store"
        custom_store.mkdir()

        # Bind a fake listener at the default socket path.
        default_sock_dir = home / ".iai-mcp"
        default_sock_dir.mkdir(parents=True, exist_ok=True)
        default_sock_path = str(default_sock_dir / ".daemon.sock")

        accept_count = [0]
        stop_listener = threading.Event()

        def _listener():
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(default_sock_path)
            srv.listen(1)
            srv.settimeout(0.1)
            try:
                while not stop_listener.is_set():
                    try:
                        conn, _ = srv.accept()
                        accept_count[0] += 1
                        conn.close()
                    except socket.timeout:
                        continue
                    except Exception:
                        break
            finally:
                try:
                    srv.close()
                except Exception:
                    pass

        t = threading.Thread(target=_listener, daemon=True)
        t.start()
        time.sleep(0.05)  # let the listener bind

        # Run the hook with custom IAI_MCP_STORE and NO IAI_DAEMON_SOCKET_PATH.
        # The env dict must NOT include IAI_DAEMON_SOCKET_PATH — _run_hook pops it
        # from the environment before applying extra_env, so passing only
        # IAI_MCP_STORE is sufficient.
        env = os.environ.copy()
        env["HOME"] = str(home)
        env["IAI_MCP_STORE"] = str(custom_store)
        env.pop("IAI_DAEMON_SOCKET_PATH", None)

        stdin_data = json.dumps({
            "session_id": sid,
            "transcript_path": str(transcript),
            "cwd": str(home),
        })
        result = subprocess.run(
            ["bash", str(HOOK)],
            input=stdin_data,
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )

        stop_listener.set()

        assert result.returncode == 0, f"Hook rc={result.returncode}"
        assert accept_count[0] == 0, (
            f"Default socket was contacted {accept_count[0]} time(s); "
            "the custom-store guard should have skipped the RPC entirely"
        )
        assert "additionalContext" not in result.stdout, (
            "Hook emitted additionalContext despite custom-store guard"
        )
    finally:
        shutil.rmtree(str(home), ignore_errors=True)


# ---------------------------------------------------------------------------
# 5c: Failure isolation — dead gate socket does not abort capture
# ---------------------------------------------------------------------------

def test_folded_gate_failure_does_not_abort_capture():
    """A dead gate socket leaves capture and offset intact; rc is 0.

    Signal A is tripped so the gate REACHES the socket attempt.
    No server is bound there, so connect fails. The blanket try/except
    swallows the failure; the capture write and offset advance are unaffected.
    """
    _skip_guards()

    home = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        sid = "gate5c-" + uuid.uuid4().hex[:8]
        past_ts = "2026-01-01T00:00:00+00:00"
        old_wm = "2025-12-31T23:59:59+00:00"

        _seed_db_and_watermark(home, sid, past_ts, old_wm)
        transcript = _write_transcript(home, sid)

        # Point at a nonexistent socket so the connect attempt fails.
        dead_sock = str(home / "dead_nonexistent.sock")

        result = _run_hook(
            home, sid, transcript,
            {"IAI_DAEMON_SOCKET_PATH": dead_sock},
        )

        # (i) Hook must exit 0.
        assert result.returncode == 0, (
            f"Hook rc={result.returncode} — gate failure must not make the hook non-zero"
        )

        # (ii) The live.jsonl event must be present despite the dead gate socket.
        live_file = home / ".iai-mcp" / ".deferred-captures" / f"{sid}.live.jsonl"
        assert live_file.exists(), "live.jsonl not created — capture failed"
        lines = [ln for ln in live_file.read_text().splitlines() if ln.strip()]
        events = []
        for ln in lines:
            try:
                obj = json.loads(ln)
                if "text" in obj:
                    events.append(obj)
            except Exception:
                pass
        assert len(events) >= 1, "No capture events in live.jsonl"
        assert any("gate test" in ev.get("text", "") for ev in events), (
            "Expected transcript text not found in captured events"
        )

        # (iii) Offset file must be advanced (> 0).
        offset_file = home / ".iai-mcp" / ".capture-state" / f"{sid}.offset"
        assert offset_file.exists(), "offset file not written"
        offset_val = int(offset_file.read_text().strip())
        assert offset_val > 0, f"offset not advanced: {offset_val}"

        # (iv) No additionalContext in stdout (dead socket emitted nothing).
        assert "additionalContext" not in result.stdout, (
            "Gate emitted additionalContext despite dead socket — unexpected"
        )
    finally:
        shutil.rmtree(str(home), ignore_errors=True)
