"""Per-turn capture latency budget: in-process + end-to-end shell hook.

Two distinct p95 budgets:
- write_deferred_event p95 <= 10 ms (in-process, no shell overhead)
- end-to-end shell hook invocation p95 <= 100 ms (bash + CLI startup + write)
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX bash + paths",
)


REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-turn-capture.sh"
VENV_CLI = REPO / ".venv" / "bin" / "iai-mcp"


@pytest.mark.perf
def test_write_deferred_event_p95_under_10ms(tmp_path, monkeypatch):
    """In-process write_deferred_event p95 <= 10 ms over 200 calls."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import write_deferred_event

    session_id = "perf-" + uuid.uuid4().hex[:8]
    samples_ms: list[float] = []

    for _ in range(200):
        t0 = time.perf_counter_ns()
        write_deferred_event(session_id, "user", "hello world payload", cwd=str(tmp_path))
        samples_ms.append((time.perf_counter_ns() - t0) / 1_000_000)

    samples_ms.sort()
    p95 = samples_ms[int(0.95 * len(samples_ms)) - 1]
    print(f"\nwrite_deferred_event p95: {p95:.3f} ms (n=200)")
    assert p95 <= 10.0, f"p95={p95:.3f}ms exceeds 10 ms budget"


@pytest.mark.perf
def test_end_to_end_shell_hook_p95_under_100ms(tmp_path):
    """End-to-end shell hook invocation p95 <= 100 ms over 200 sequential runs."""
    if not VENV_CLI.exists():
        pytest.skip(f"iai-mcp not installed at {VENV_CLI}")
    if not HOOK.exists():
        pytest.skip(f"hook script missing at {HOOK}")
    if not shutil.which("bash"):
        pytest.skip("bash not on PATH")

    home = tmp_path
    transcript = home / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "hi enough text"}})
        + "\n"
    )
    cache = home / ".iai-mcp" / ".cli-path"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(str(VENV_CLI))

    session_id = "perf-shell-" + uuid.uuid4().hex[:8]
    stdin = json.dumps({
        "session_id": session_id,
        "transcript_path": str(transcript),
        "cwd": str(home),
    })
    env = os.environ.copy()
    env["HOME"] = str(home)

    # Warm-up: first invocation pays page-cache + cold-import costs and is
    # not part of the steady-state p95 we care about.
    subprocess.run(
        ["bash", str(HOOK)],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    samples_ms: list[float] = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        subprocess.run(
            ["bash", str(HOOK)],
            input=stdin,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        samples_ms.append((time.perf_counter_ns() - t0) / 1_000_000)

    samples_ms.sort()
    p95 = samples_ms[int(0.95 * len(samples_ms)) - 1]
    print(f"\nshell hook end-to-end p95: {p95:.3f} ms (n=200, warmup excluded)")
    assert p95 <= 100.0, f"p95={p95:.3f}ms exceeds 100 ms budget"


def test_shell_hook_and_cli_produce_equivalent_structure(tmp_path, monkeypatch):
    """The inline shell-hook writer and cmd_capture_turn_deferred MUST agree
    on header keys, event keys, file path, and offset semantics."""
    import json as _json
    import subprocess as _sp

    if not VENV_CLI.exists():
        pytest.skip("iai-mcp CLI not installed")
    if not HOOK.exists():
        pytest.skip("hook script missing")

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        _json.dumps({"type": "user", "message": {"role": "user", "content": "hello A"}}) + "\n"
        + _json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "reply B"}}) + "\n"
    )

    # 1. CLI path -> writes to HOME/A/.iai-mcp/.deferred-captures/{sid}.live.jsonl
    home_a = tmp_path / "HOME_A"
    home_a.mkdir()
    monkeypatch.setenv("HOME", str(home_a))
    from iai_mcp.cli import cmd_capture_turn_deferred
    import argparse
    cmd_capture_turn_deferred(argparse.Namespace(
        session_id="P", transcript_path=str(transcript), max_turns_per_call=200,
    ))
    a_live = home_a / ".iai-mcp" / ".deferred-captures" / "P.live.jsonl"
    a_lines = a_live.read_text().splitlines()

    # 2. Shell-hook path -> writes to HOME/B/.iai-mcp/.deferred-captures/{sid}.live.jsonl
    home_b = tmp_path / "HOME_B"
    home_b.mkdir()
    env = os.environ.copy()
    env["HOME"] = str(home_b)
    stdin = _json.dumps({
        "session_id": "P", "transcript_path": str(transcript), "cwd": str(home_b),
    })
    _sp.run(["bash", str(HOOK)], input=stdin, env=env, capture_output=True, text=True, timeout=10)
    b_live = home_b / ".iai-mcp" / ".deferred-captures" / "P.live.jsonl"
    b_lines = b_live.read_text().splitlines()

    assert len(a_lines) == len(b_lines) == 3, (a_lines, b_lines)
    head_a = _json.loads(a_lines[0])
    head_b = _json.loads(b_lines[0])
    assert head_a.keys() == head_b.keys()
    assert head_a["version"] == head_b["version"] == 1
    assert head_a["session_id"] == head_b["session_id"] == "P"

    for la, lb in zip(a_lines[1:], b_lines[1:]):
        ea = _json.loads(la)
        eb = _json.loads(lb)
        assert ea.keys() == eb.keys()
        assert ea["text"] == eb["text"]
        assert ea["role"] == eb["role"]
        assert ea["tier"] == eb["tier"] == "episodic"
