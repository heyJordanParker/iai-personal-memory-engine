"""Hermetic tests for the dead-PID `.processing-<pid>.jsonl` salvage migration.

Covers the unlock-policy three branches (dead, live-foreign, live-current),
dry-run accounting, idempotency, collision-safe naming, and the CLI surface.

All fixtures live in `tmp_path / ".deferred-captures"`; the live
`~/.iai-mcp/.deferred-captures/` tree is NEVER touched. Dead PIDs are obtained
by spawning a Python child that immediately exits, then waiting on it. Live-
foreign and live-current PIDs are simulated via the explicit ``live_daemon_pid=``
kwarg override on the migration function — no daemon construction required.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="os.rename overwrite semantics differ on Windows; salvage targets POSIX only",
)


# ---- helpers ---------------------------------------------------------------


def _spawn_dead_pid() -> int:
    """Spawn a Python child that exits immediately; return its (now-dead) pid."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _make_locked_file(deferred: Path, basename: str, owner_pid: int) -> Path:
    """Create a fake `<basename>.processing-<owner_pid>.jsonl` file with two lines."""
    deferred.mkdir(parents=True, exist_ok=True)
    path = deferred / f"{basename}.processing-{owner_pid}.jsonl"
    path.write_text(
        '{"event": "fake_capture", "n": 1}\n'
        '{"event": "fake_capture", "n": 2}\n'
    )
    return path


# ---- tests -----------------------------------------------------------------


def test_dead_owner_pid_is_unlocked(tmp_path):
    """Dead owner pid → file renamed to bare `<basename>.jsonl`."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    dead_pid = _spawn_dead_pid()
    locked = _make_locked_file(deferred, "sess-dead", dead_pid)

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
    )

    target = deferred / "sess-dead.jsonl"
    assert not locked.exists(), "locked file should be gone after rename"
    assert target.exists(), "bare `.jsonl` target should exist"
    assert result["files_scanned"] == 1
    assert result["files_unlocked"] == 1
    assert result["skipped_live_current_daemon"] == 0
    assert result["collision_safe_renames"] == 0
    assert result["dry_run"] is False


def test_live_foreign_pid_is_unlocked(tmp_path):
    """Alive owner pid that is NOT the live daemon → unlocked anyway."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    # We are alive — simulate the daemon being a DIFFERENT alive pid.
    foreign_pid = os.getpid()
    simulated_daemon_pid = foreign_pid + 1  # explicitly NOT us

    locked = _make_locked_file(deferred, "sess-foreign", foreign_pid)

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=simulated_daemon_pid,
    )

    target = deferred / "sess-foreign.jsonl"
    assert not locked.exists()
    assert target.exists()
    assert result["files_scanned"] == 1
    assert result["files_unlocked"] == 1
    assert result["skipped_live_current_daemon"] == 0


def test_live_current_daemon_pid_is_preserved(tmp_path):
    """Owner pid == live-daemon pid → file is NEVER touched (protective branch)."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    current_pid = os.getpid()
    locked = _make_locked_file(deferred, "sess-current", current_pid)

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=current_pid,
    )

    target = deferred / "sess-current.jsonl"
    assert locked.exists(), "current-daemon-owned file MUST remain locked"
    assert not target.exists(), "no bare target should be created"
    assert result["files_scanned"] == 1
    assert result["files_unlocked"] == 0
    assert result["skipped_live_current_daemon"] == 1


def test_plain_jsonl_files_untouched(tmp_path):
    """Files matching no `_PROCESSING_MARKER_RE` are not even scanned."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    deferred.mkdir(parents=True, exist_ok=True)
    plain = deferred / "sess-X.jsonl"
    live = deferred / "sess-Y.live.jsonl"
    crash = deferred / "sess-Z.crash-1.jsonl"
    for p in (plain, live, crash):
        p.write_text('{"x": 1}\n')

    contents_before = {p.name: p.read_text() for p in (plain, live, crash)}

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
    )

    assert result["files_scanned"] == 0
    assert result["files_unlocked"] == 0
    for p in (plain, live, crash):
        assert p.exists(), f"{p.name} must remain"
        assert p.read_text() == contents_before[p.name]


def test_dry_run_makes_no_changes(tmp_path):
    """`dry_run=True` reports counts but mutates nothing."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    dead_pid = _spawn_dead_pid()
    locked = _make_locked_file(deferred, "sess-dry", dead_pid)
    target = deferred / "sess-dry.jsonl"

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["files_scanned"] == 1
    assert result["files_unlocked"] == 1, "dry-run reports what would happen"
    assert locked.exists(), "FS must be untouched in dry-run"
    assert not target.exists(), "no bare target should be created in dry-run"


def test_idempotent(tmp_path):
    """After a successful rename, the marker is gone — second call finds nothing."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    dead_pid = _spawn_dead_pid()
    _make_locked_file(deferred, "sess-idem", dead_pid)

    first = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
    )
    second = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
    )

    assert first["files_unlocked"] == 1
    assert second["files_scanned"] == 0
    assert second["files_unlocked"] == 0
    assert second["collision_safe_renames"] == 0


def test_collision_safe_naming(tmp_path):
    """When bare `<basename>.jsonl` already exists, recover via
    `<basename>.recovered-<utc_ts>-<unlock_pid>.jsonl` and PRESERVE the
    pre-existing target verbatim."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    deferred.mkdir(parents=True, exist_ok=True)

    pre_existing = deferred / "sess-clash.jsonl"
    pre_existing_content = '{"this": "must-not-be-overwritten"}\n'
    pre_existing.write_text(pre_existing_content)

    dead_pid = _spawn_dead_pid()
    locked = _make_locked_file(deferred, "sess-clash", dead_pid)

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
    )

    assert result["files_unlocked"] == 1
    assert result["collision_safe_renames"] == 1
    assert not locked.exists()
    assert pre_existing.exists(), "pre-existing target must be untouched"
    assert pre_existing.read_text() == pre_existing_content

    siblings = [
        p for p in deferred.iterdir()
        if p.name.startswith("sess-clash.recovered-") and p.name.endswith(".jsonl")
    ]
    assert len(siblings) == 1, (
        f"expected exactly one `.recovered-` sibling, found {[p.name for p in siblings]}"
    )
    sibling_name = siblings[0].name
    # Suffix shape: sess-clash.recovered-<utc_ts>-<unlock_pid>.jsonl
    assert sibling_name.startswith("sess-clash.recovered-")
    assert sibling_name.endswith(".jsonl")


def test_cli_help_lists_subcommand_and_flags():
    """`iai-mcp deferred-unlock-dead-pids --help` exits 0 and lists --dry-run + --json.
    Pins the CLI surface against accidental rename/regression."""
    result = subprocess.run(
        [sys.executable, "-m", "iai_mcp.cli", "deferred-unlock-dead-pids", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"--help exited non-zero. stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    combined = result.stdout + result.stderr
    assert "--dry-run" in combined
    assert "--json" in combined
