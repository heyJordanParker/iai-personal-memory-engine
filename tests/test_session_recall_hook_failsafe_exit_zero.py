"""Contract:
- Bash hook exits 0 with empty stdout when the CLI subprocess fails (non-zero
  exit, signal, or timeout).
- IAI_MCP_RECALL_HOOK_TIMEOUT env var caps the CLI call; an over-long stub
  still yields exit 0 / empty stdout within roughly the cap budget.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell hook")


HOOK_PATH = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-session-recall.sh"


def _make_stub_cli(dir_: Path, script: str) -> Path:
    cli = dir_ / "iai-mcp"
    cli.write_text(script)
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def _run_hook(home: Path, *, extra_env: dict[str, str] | None = None,
              stdin_payload: str = '{"session_id":"x","source":"startup","cwd":"/tmp","transcript_path":""}',
              timeout: float = 10.0):
    env = os.environ.copy()
    env["HOME"] = str(home)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=stdin_payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_hook_exits_zero_when_cli_fails(tmp_path):
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\necho boom >&2\nexit 1\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout


def test_hook_exits_zero_under_timeout_against_sleep_stub(tmp_path):
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\nsleep 60\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    t0 = time.monotonic()
    proc = _run_hook(home, extra_env={"IAI_MCP_RECALL_HOOK_TIMEOUT": "2"}, timeout=15.0)
    elapsed = time.monotonic() - t0
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout
    assert elapsed < 8.0, f"hook took {elapsed:.1f}s — timeout cap not honored"
