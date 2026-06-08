"""Regression tests for turn-capture hook PY_SCRIPT offset accounting.

Drives the REAL PY_SCRIPT extracted from the hook file at test time — no
hand-copied snapshot, so the test always validates the live hook code.

Proven failure class (H-d, rotated/shorter transcript):
  When transcript_path has FEWER lines than the stored offset (total < prev),
  the current code resets prev=0 then rewrites lines[0:], clobbering the valid
  large offset AND re-emitting old turns as if they were new. The fix must
  preserve the larger offset and skip silently (nothing new from a shorter
  transcript).

H-a (timeout) was refuted: a 1520-line transcript with offset=1324 completes
in milliseconds; the write block runs normally (total > prev); H-a is not a
freeze mechanism.

Note on total==prev: the write block is intentionally skipped when total==prev
(nothing new) and offset stays unchanged. That is CORRECT behavior — no fix
needed for that case.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Extract real PY_SCRIPT from the hook file
# ---------------------------------------------------------------------------

HOOK_FILE = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-turn-capture.sh"


def _extract_py_script() -> str:
    """Extract the PY_SCRIPT heredoc from the canonical hook file.

    Slices from the line after `PY_SCRIPT='` to the standalone closing `'`
    line. This always validates the live hook code, not a snapshot.
    """
    text = HOOK_FILE.read_text()
    # Match PY_SCRIPT='...' (single-quoted heredoc)
    m = re.search(r"PY_SCRIPT='(.*?)'\s*\n", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not find PY_SCRIPT heredoc in {HOOK_FILE}")
    return m.group(1)


def _run_py_script(
    py_script: str,
    session_id: str,
    transcript_path: Path,
    home_dir: Path,
) -> tuple[int, float]:
    """Run PY_SCRIPT in a subprocess with HOME=tmp. Returns (returncode, elapsed_s)."""
    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", py_script, session_id, str(transcript_path)],
        env=env,
        capture_output=True,
        timeout=15,
    )
    elapsed = time.monotonic() - t0
    return result.returncode, elapsed


def _make_transcript(path: Path, n_lines: int) -> None:
    """Write a synthetic transcript with n_lines JSONL entries to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n_lines):
            role = "user" if i % 2 == 0 else "assistant"
            f.write(json.dumps({
                "type": role,
                "message": {"role": role, "content": f"Turn {i}"},
            }) + "\n")


def _make_transcript_with_nonce(path: Path, n_lines: int, nonce: str) -> None:
    """Write a synthetic transcript with n_lines entries; first user turn contains nonce."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n_lines):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Turn {i} {nonce}" if (i == 0 and role == "user") else f"Turn {i}"
            f.write(json.dumps({
                "type": role,
                "message": {"role": role, "content": content},
            }) + "\n")


def _read_offset(state_dir: Path, session_id: str) -> int:
    offset_file = state_dir / f"{session_id}.offset"
    if not offset_file.exists():
        return -1
    return int(offset_file.read_text().strip() or "0")


def _count_live_turns(deferred_dir: Path, session_id: str) -> int:
    live_file = deferred_dir / f"{session_id}.live.jsonl"
    if not live_file.exists():
        return 0
    count = 0
    with live_file.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "role" in obj:
                    count += 1
            except Exception:
                pass
    return count


def _live_contains_text(deferred_dir: Path, session_id: str, text: str) -> bool:
    """Return True if any turn event in the live file contains the given text."""
    live_file = deferred_dir / f"{session_id}.live.jsonl"
    if not live_file.exists():
        return False
    with live_file.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "role" in obj and text in obj.get("text", ""):
                    return True
            except Exception:
                pass
    return False


# ---------------------------------------------------------------------------
# H-a refutation: 1520-line transcript with offset=1324 completes fast
# ---------------------------------------------------------------------------


def test_ha_refuted_large_transcript_advances_offset():
    """H-a refuted: 1520-line transcript + offset=1324 is fast, offset advances.

    total(1520) > prev(1324) → write block runs normally. This is GREEN on
    current (unmodified) code — it does NOT reproduce a freeze. Included to
    document the refutation explicitly.
    """
    py_script = _extract_py_script()
    sid = "test-ha-refutation"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        transcript = home / "transcript.jsonl"
        _make_transcript(transcript, 1520)

        # Set offset to 1324 (as in the real evidence).
        (state_dir / f"{sid}.offset").write_text("1324")

        rc, elapsed = _run_py_script(py_script, sid, transcript, home)

        assert rc == 0
        new_offset = _read_offset(state_dir, sid)
        # Offset advanced (anything ≥ 1520 is fine — consumed = 1520 - 1324 = 196)
        assert new_offset == 1520, f"expected 1520, got {new_offset}"
        # Should complete well within 5s budget
        assert elapsed < 4.0, f"took {elapsed:.2f}s — unexpected timeout risk"


# ---------------------------------------------------------------------------
# H-d: shorter transcript clobbers valid large offset (RED on current code)
# ---------------------------------------------------------------------------


def test_hd_shorter_transcript_must_not_clobber_offset():
    """H-d (proven freeze class): transcript shorter than stored offset must NOT clobber it.

    Current behavior (bug):
      total=50, prev=1324 → `if prev > total: prev=0` → write block runs with
      prev=0, rewrites lines[0:50] (re-emitting old turns), new_offset = 0+50 = 50
      → clobbers the valid 1324 offset down to 50.

    Expected behavior (after fix):
      A transcript shorter than the stored offset means the transcript was
      rotated or replaced. The hook should preserve the existing large offset
      (or at minimum not shrink it) and not re-emit old turns.

    This test is RED on the unmodified hook and GREEN after the fix.
    """
    py_script = _extract_py_script()
    sid = "test-hd-short-transcript"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        # Shorter transcript: only 50 lines, but offset says we already saw 1324
        transcript = home / "transcript.jsonl"
        _make_transcript(transcript, 50)
        (state_dir / f"{sid}.offset").write_text("1324")

        rc, _ = _run_py_script(py_script, sid, transcript, home)

        assert rc == 0
        final_offset = _read_offset(state_dir, sid)

        # The offset must NOT shrink below the stored value.
        # Accepting = 1324 (preserved, no clobber).
        assert final_offset >= 1324, (
            f"offset was clobbered: stored 1324, final {final_offset}. "
            f"Shorter transcript reset prev=0 and rewrote old turns."
        )

        # Must NOT re-emit old turns as new (live file should be empty / absent,
        # or if it existed before, should not have grown).
        live_turns = _count_live_turns(deferred_dir, sid)
        assert live_turns == 0, (
            f"re-emitted {live_turns} old turns as new events (clobber bug)"
        )


# ---------------------------------------------------------------------------
# Normal growing case still works after the fix
# ---------------------------------------------------------------------------


def test_normal_growing_transcript_advances_and_writes_turns():
    """Sanity: a growing transcript with a smaller offset advances correctly."""
    py_script = _extract_py_script()
    sid = "test-normal-grow"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        transcript = home / "transcript.jsonl"
        _make_transcript(transcript, 20)
        (state_dir / f"{sid}.offset").write_text("10")

        rc, _ = _run_py_script(py_script, sid, transcript, home)

        assert rc == 0
        final_offset = _read_offset(state_dir, sid)
        assert final_offset == 20, f"expected 20, got {final_offset}"

        # Some turns should be written (lines 10-19 → up to 10 events)
        live_turns = _count_live_turns(deferred_dir, sid)
        assert live_turns > 0, "expected at least one turn written for new lines"


# ---------------------------------------------------------------------------
# Fresh session (no offset) — captures from line 0
# ---------------------------------------------------------------------------


def test_fresh_session_no_offset_captures_all_turns():
    """Fresh session with no prior offset file captures all turns from line 0."""
    py_script = _extract_py_script()
    sid = "test-fresh-session"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        # 10-line transcript: 5 user turns, 5 assistant turns
        transcript = home / "transcript.jsonl"
        _make_transcript(transcript, 10)

        # No offset file — simulates first fire for a brand-new session
        rc, _ = _run_py_script(py_script, sid, transcript, home)

        assert rc == 0
        final_offset = _read_offset(state_dir, sid)
        assert final_offset == 10, f"expected offset=10, got {final_offset}"

        live_turns = _count_live_turns(deferred_dir, sid)
        assert live_turns == 10, (
            f"expected 10 turns captured, got {live_turns}"
        )


# ---------------------------------------------------------------------------
# Scan fallback: hook-provided path is wrong, real transcript found via scan
# ---------------------------------------------------------------------------


def test_stale_path_scan_fallback_captures_turns():
    """Canonical-first: wrong/nonexistent stdin path → canonical scanned and used.

    When Claude Code passes a transcript_path that does not exist (stale temp
    path, cwd mismatch, in-flight write), the hook must scan
    ~/.claude/projects/*/{session_id}.jsonl and use the canonical file instead
    of exiting silently. Without this, sessions with stale paths are never
    captured by the UserPromptSubmit hook.
    """
    py_script = _extract_py_script()
    sid = "test-scan-fallback"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        # Place the real transcript under a projects sub-directory, mimicking
        # how Claude Code stores transcripts at ~/.claude/projects/{hash}/{uuid}.jsonl
        project_dir = home / ".claude" / "projects" / "-Users-example-project"
        project_dir.mkdir(parents=True, exist_ok=True)
        real_transcript = project_dir / f"{sid}.jsonl"
        _make_transcript(real_transcript, 12)

        # Pass a nonexistent "stale" path as transcript_path — what the hook
        # receives when the path has rotated or Claude Code sent a temp location.
        stale_path = home / "nonexistent" / f"{sid}.jsonl"

        rc, _ = _run_py_script(py_script, sid, stale_path, home)

        assert rc == 0
        final_offset = _read_offset(state_dir, sid)
        # Scan found the real transcript: offset should be 12
        assert final_offset == 12, (
            f"canonical-first did not activate: offset={final_offset}, "
            f"expected 12 (all lines of real transcript)"
        )
        live_turns = _count_live_turns(deferred_dir, sid)
        assert live_turns == 12, (
            f"expected 12 turns captured via canonical-first, got {live_turns}"
        )


# ---------------------------------------------------------------------------
# Scan fallback: transcript genuinely does not exist anywhere → exit cleanly
# ---------------------------------------------------------------------------


def test_missing_transcript_everywhere_exits_cleanly():
    """When transcript_path is wrong AND scan finds nothing, exit 0 silently.

    Graceful degradation: first-fire H1 timing (transcript not yet written by
    Claude Code) — no file exists anywhere yet. The hook must not crash,
    must not create partial state, and must return 0.
    """
    py_script = _extract_py_script()
    sid = "test-missing-everywhere"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        # Empty projects dir — no transcript exists anywhere
        (home / ".claude" / "projects").mkdir(parents=True, exist_ok=True)

        stale_path = home / "no-such-file.jsonl"
        rc, _ = _run_py_script(py_script, sid, stale_path, home)

        assert rc == 0
        # No offset or live file should be created
        assert _read_offset(state_dir, sid) == -1, "offset must not be created"
        assert _count_live_turns(deferred_dir, sid) == 0, "live file must not be created"


# ---------------------------------------------------------------------------
# DETERMINISTIC REPRO: present-but-empty stdin — the 7173b585 failure mode
# ---------------------------------------------------------------------------


def test_present_but_empty_stdin_uses_canonical_and_writes_nonce():
    """Core regression: stdin path exists but is empty → canonical used → nonce written.

    This is the precise failure mode for session 7173b585: the hook-supplied
    transcript_path existed on disk (exists() True) but was empty, while the
    real 35-line file with the nonce turn sat at the canonical projects path.
    The 815cfbc fix only triggered on not-exists(), so the fallback was skipped.

    After the canonical-first fix: the canonical file is always scanned first.
    If it exists and is non-empty, it wins regardless of the stdin path state.
    """
    py_script = _extract_py_script()
    sid = "test-empty-stdin-canonical"
    nonce = "e7k9p"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        # Real 35-line canonical transcript containing the nonce in first user turn
        project_dir = home / ".claude" / "projects" / "-Users-example-project"
        project_dir.mkdir(parents=True, exist_ok=True)
        canonical_transcript = project_dir / f"{sid}.jsonl"
        _make_transcript_with_nonce(canonical_transcript, 35, nonce)

        # stdin path EXISTS but is EMPTY — the exact 7173b585 failure shape
        empty_stdin = home / "empty-transcript.jsonl"
        empty_stdin.write_text("")

        # No offset file (prev=0) — as in the real failure
        rc, _ = _run_py_script(py_script, sid, empty_stdin, home)

        assert rc == 0, f"hook exited {rc}"

        # The nonce turn must be in the live file
        assert _live_contains_text(deferred_dir, sid, nonce), (
            f"nonce '{nonce}' not found in live file — canonical-first fallback did not fire. "
            f"This is the 7173b585 regression."
        )

        final_offset = _read_offset(state_dir, sid)
        assert final_offset == 35, f"expected offset=35, got {final_offset}"

        live_turns = _count_live_turns(deferred_dir, sid)
        assert live_turns > 0, "no turns written despite 35-line canonical transcript"


# ---------------------------------------------------------------------------
# present-but-wrong-session stdin — discriminates canonical-first vs max-lines
# ---------------------------------------------------------------------------


def test_present_but_wrong_session_stdin_uses_canonical_not_stdin():
    """Canonical-first vs max-lines discriminator: stdin has wrong-session content.

    stdin points to a 50-line file belonging to a different session (no nonce).
    Canonical has 35 lines with the nonce. Canonical-first correctly picks the
    right file. A max-lines strategy would pick the 50-line stdin file and
    would NOT write the nonce — that approach is incorrect for this case.
    """
    py_script = _extract_py_script()
    sid = "test-wrong-session-stdin"
    nonce = "e7k9p"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        state_dir = home / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        deferred_dir = home / ".iai-mcp" / ".deferred-captures"
        deferred_dir.mkdir(parents=True, exist_ok=True)

        # Canonical: 35 lines for this session with nonce
        project_dir = home / ".claude" / "projects" / "-Users-example-project"
        project_dir.mkdir(parents=True, exist_ok=True)
        canonical_transcript = project_dir / f"{sid}.jsonl"
        _make_transcript_with_nonce(canonical_transcript, 35, nonce)

        # stdin: 50-line file for a different session (exists, longer, NO nonce)
        other_sid = "other-session-xyz"
        wrong_stdin = home / "wrong-session.jsonl"
        _make_transcript(wrong_stdin, 50)

        rc, _ = _run_py_script(py_script, sid, wrong_stdin, home)

        assert rc == 0

        # Nonce must be written (from canonical, not from the wrong stdin file)
        assert _live_contains_text(deferred_dir, sid, nonce), (
            f"nonce '{nonce}' not found — canonical-first did not override longer wrong-session stdin. "
            f"A max-lines strategy would fail this test."
        )

        final_offset = _read_offset(state_dir, sid)
        assert final_offset == 35, (
            f"offset should be 35 (canonical line count), got {final_offset}"
        )
