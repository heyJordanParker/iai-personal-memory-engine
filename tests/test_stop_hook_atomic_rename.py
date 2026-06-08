"""Stop hook atomically renames live file before safety-net capture.

Contract:
- Stop hook renames `{session_id}.live.jsonl` to
  `{session_id}.live-{epoch}.jsonl` BEFORE invoking
  `capture-transcript --no-spawn`.
- Rename target uses `.live-{epoch}.jsonl` pattern (NOT `-{epoch}.jsonl`)
  to avoid colliding with the safety-net output of `write_deferred_captures`.
- When no live file exists, hook still calls the safety-net and exits 0.
"""
from __future__ import annotations

import json
import os
import platform
import re
import stat
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-session-capture.sh"


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="bash + POSIX mv",
)


def _install_shim(home: Path, log_path: Path) -> Path:
    """Write a fake `iai-mcp` script that logs argv + records ordering."""
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "iai-mcp"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        f"echo \"$@\" >> {log_path}\n"
        f"ls {home}/.iai-mcp/.deferred-captures/ 2>/dev/null >> {log_path}.dirsnap || true\n"
        "exit 0\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # Cache the shim path so the hook short-circuits its probe list.
    cli_cache = home / ".iai-mcp" / ".cli-path"
    cli_cache.parent.mkdir(parents=True, exist_ok=True)
    cli_cache.write_text(str(shim))
    return shim


def _make_transcript(home: Path, session_id: str) -> Path:
    """Create a fake Claude Code transcript at the expected projects path."""
    projects = home / ".claude" / "projects" / "fakeproj"
    projects.mkdir(parents=True, exist_ok=True)
    transcript = projects / f"{session_id}.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}) + "\n"
    )
    return transcript


def _run_hook(home: Path, session_id: str, transcript: Path) -> subprocess.CompletedProcess:
    stdin = json.dumps({
        "session_id": session_id,
        "transcript_path": str(transcript),
        "cwd": str(home),
    })
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = f"{home}/bin:" + env.get("PATH", "")
    return subprocess.run(
        ["bash", str(HOOK)],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )


def test_stop_hook_renames_live_file_before_capture_transcript(tmp_path):
    """Rename happens BEFORE the shim sees the deferred dir."""
    home = tmp_path
    deferred = home / ".iai-mcp" / ".deferred-captures"
    deferred.mkdir(parents=True, exist_ok=True)

    sid = "SESSION-RENAME-1"
    live = deferred / f"{sid}.live.jsonl"
    live.write_text('{"version":1,"session_id":"' + sid + '","deferred_at":"x","cwd":"/tmp"}\n')

    shim_log = home / "shim.log"
    _install_shim(home, shim_log)
    transcript = _make_transcript(home, sid)

    result = _run_hook(home, sid, transcript)
    assert result.returncode == 0, result.stderr

    after = list(deferred.iterdir())
    renamed = [p for p in after if re.match(rf"^{re.escape(sid)}\.live-\d+\.jsonl$", p.name)]
    assert len(renamed) == 1, f"expected one renamed file, got {after}"
    assert not live.exists(), "original .live.jsonl must be gone after rename"

    # Shim was invoked with capture-transcript --no-spawn.
    assert shim_log.exists(), "shim must have been called"
    shim_argv = shim_log.read_text()
    assert "capture-transcript" in shim_argv
    assert "--no-spawn" in shim_argv

    # The rename happened before the shim ran — dir-snapshot at shim
    # invocation must show the renamed file, not the original.
    snap_path = Path(str(shim_log) + ".dirsnap")
    assert snap_path.exists(), "directory snapshot must be recorded by the shim"
    snap = snap_path.read_text()
    assert f"{sid}.live.jsonl" not in snap.splitlines(), snap
    assert renamed[0].name in snap


def test_rename_target_does_not_collide_with_safety_net_output(tmp_path):
    """Rename target is `{sid}.live-{epoch}.jsonl`, never `{sid}-{epoch}.jsonl`."""
    home = tmp_path
    deferred = home / ".iai-mcp" / ".deferred-captures"
    deferred.mkdir(parents=True, exist_ok=True)

    sid = "SESSION-RENAME-2"
    (deferred / f"{sid}.live.jsonl").write_text(
        '{"version":1,"session_id":"' + sid + '","deferred_at":"x","cwd":"/tmp"}\n'
    )

    _install_shim(home, home / "shim.log")
    transcript = _make_transcript(home, sid)

    result = _run_hook(home, sid, transcript)
    assert result.returncode == 0

    names = [p.name for p in deferred.iterdir()]
    collision_pattern = re.compile(rf"^{re.escape(sid)}-\d+\.jsonl$")
    rename_pattern = re.compile(rf"^{re.escape(sid)}\.live-\d+\.jsonl$")
    assert any(rename_pattern.match(n) for n in names), names
    assert not any(collision_pattern.match(n) for n in names), \
        f"safety-net collision shape found in {names}"


def test_stop_hook_no_op_if_no_live_file(tmp_path):
    """No live file: hook still calls safety-net capture and exits 0."""
    home = tmp_path
    deferred = home / ".iai-mcp" / ".deferred-captures"
    deferred.mkdir(parents=True, exist_ok=True)

    shim_log = home / "shim.log"
    _install_shim(home, shim_log)
    sid = "SESSION-NO-LIVE"
    transcript = _make_transcript(home, sid)

    result = _run_hook(home, sid, transcript)
    assert result.returncode == 0

    assert list(deferred.iterdir()) == []
    assert shim_log.exists(), "shim still gets called as the safety net"
    assert "capture-transcript" in shim_log.read_text()
