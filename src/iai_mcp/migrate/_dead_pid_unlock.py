"""Salvage `.processing-<owner_pid>.jsonl` files whose owner PID is dead OR is alive but
does not match the currently-running daemon.

When the daemon wedges with a `.processing-<pid>` marker still held by an owner PID
(dead from a previous instance, OR alive but belonging to a pre-restart daemon), the
drain pipeline at ``capture.py`` skips the file forever — the drain's predicate trusts
``_pid_is_alive(owner_pid)``. This migration walks the deferred-captures dir and renames
each such file back to its bare ``<basename>.jsonl`` form so the next drain tick claims it
as a normal candidate. Idempotent — after the rename, the marker is gone; a second run
finds nothing to do. Collision-safe — when the bare ``<basename>.jsonl`` already exists,
the unlocked file is renamed to ``<basename>.recovered-<utc_ts>-<unlock_pid>.jsonl`` (also
``.jsonl``-suffixed and drain-eligible) so the pre-existing target is preserved.

The currently-running daemon's owner-PID files are NEVER touched. Documented operator
workflow: stop the wedged daemon → start a fresh daemon (the ``daemon_pid`` in
``.daemon-state.json`` becomes the new pid; the wedge's owner pid is now "foreign") → run
this migration → next drain tick reclaims the files.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from iai_mcp.capture import _PROCESSING_MARKER_RE, _pid_is_alive

log = logging.getLogger(__name__)


class _Sentinel:
    """Distinguishes 'auto-detect live_daemon_pid' from explicit ``None``."""


_NOT_GIVEN = _Sentinel()


def _read_live_daemon_pid() -> int | None:
    """Return the live daemon's PID from ``.daemon-state.json``, or ``None`` if absent,
    corrupt, or pointing at a dead pid. Never raises — fail-open to ``None`` so the
    salvage proceeds to unlock everything when the daemon state is missing."""
    try:
        from iai_mcp.daemon_state import load_state
        state = load_state() or {}
    except Exception:  # noqa: BLE001 -- load_state already swallows; defence-in-depth
        return None
    pid = state.get("daemon_pid")
    if not isinstance(pid, int) or pid < 1 or pid > 2**31 - 1:
        return None
    if not _pid_is_alive(pid):
        return None
    return pid


def _unlock_one(
    src: Path,
    deferred_dir: Path,
    *,
    unlock_pid: int,
    dry_run: bool,
) -> tuple[Path | None, bool]:
    """Rename one locked file back to its bare ``.jsonl`` form.

    Returns ``(target_path, collision_safe)``. ``target_path`` is ``None`` on
    rename failure (raced or OSError) — caller treats as no-op for counter accounting.
    """
    stripped = _PROCESSING_MARKER_RE.sub(".jsonl", src.name)
    if stripped == src.name:
        # Defensive: shouldn't reach here — caller already matched the marker.
        return (None, False)
    target = deferred_dir / stripped
    collision_safe = False
    if target.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_stem = stripped.removesuffix(".jsonl")
        target = deferred_dir / f"{safe_stem}.recovered-{ts}-{unlock_pid}.jsonl"
        collision_safe = True
    if dry_run:
        return (target, collision_safe)
    try:
        os.rename(src, target)
    except FileNotFoundError:
        # Another drain raced and claimed the file. Treat as success.
        return (None, collision_safe)
    except OSError as exc:
        log.warning(
            "dead-pid-unlock: rename %s -> %s failed: %s",
            src.name,
            target.name,
            exc,
        )
        return (None, collision_safe)
    return (target, collision_safe)


def migrate_unlock_dead_pid_processing_files(
    *,
    deferred_dir: Path | None = None,
    live_daemon_pid: int | None | _Sentinel = _NOT_GIVEN,
    dry_run: bool = False,
) -> dict:
    """Rename ``.processing-<owner_pid>.jsonl`` -> ``.jsonl`` for owner PIDs that are
    DEAD, or ALIVE-but-foreign (``owner_pid != live_daemon_pid``).

    Parameters
    ----------
    deferred_dir
        Override target dir (test injection). Default ``~/.iai-mcp/.deferred-captures``.
    live_daemon_pid
        Override the live-daemon PID (test injection). Sentinel ``_NOT_GIVEN`` means
        "load from ``.daemon-state.json``"; an explicit ``None`` means "no live daemon —
        unlock everything".
    dry_run
        If True, report counts only; no renames.

    Returns
    -------
    dict
        ``files_scanned``, ``files_unlocked``, ``skipped_live_current_daemon``,
        ``skipped_unparseable``, ``collision_safe_renames``, ``live_daemon_pid``,
        ``dry_run``.
    """
    if deferred_dir is None:
        deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    deferred_dir = Path(deferred_dir)

    if isinstance(live_daemon_pid, _Sentinel):
        live_daemon_pid = _read_live_daemon_pid()

    result: dict = {
        "files_scanned": 0,
        "files_unlocked": 0,
        "skipped_live_current_daemon": 0,
        "skipped_unparseable": 0,
        "collision_safe_renames": 0,
        "live_daemon_pid": live_daemon_pid,
        "dry_run": dry_run,
    }
    if not deferred_dir.exists():
        return result

    unlock_pid = os.getpid()

    for entry in sorted(deferred_dir.iterdir()):
        if not entry.is_file():
            continue
        m = _PROCESSING_MARKER_RE.search(entry.name)
        if not m:
            continue
        result["files_scanned"] += 1
        try:
            owner_pid = int(m.group(1))
        except (TypeError, ValueError):
            result["skipped_unparseable"] += 1
            continue

        if live_daemon_pid is not None and owner_pid == live_daemon_pid:
            # ALIVE owner == current daemon: never touch (the drain owns the claim).
            result["skipped_live_current_daemon"] += 1
            continue

        # Otherwise: owner is dead OR foreign-alive — both unlocked.
        target, collision_safe = _unlock_one(
            entry,
            deferred_dir,
            unlock_pid=unlock_pid,
            dry_run=dry_run,
        )
        if target is not None:
            result["files_unlocked"] += 1
            if collision_safe:
                result["collision_safe_renames"] += 1
            if not dry_run:
                log.info(
                    "dead-pid-unlock: %s -> %s",
                    entry.name,
                    target.name,
                )

    return result
