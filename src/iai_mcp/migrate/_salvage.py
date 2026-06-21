"""Salvage torn .permanent-failed-*.jsonl files.

When ``write_deferred_captures`` wrote directly to its final ``.jsonl``
path (before the atomic ``.tmp`` + ``os.replace`` fix), a concurrent drain
could claim a half-streamed file, fail to parse mid-record, and ultimately
walk the file through ``.failed-attempt-N`` → ``.permanent-failed-*``,
quarantining real captures that were merely interrupted.

This recovery migration walks each ``.permanent-failed-*.jsonl`` file,
trims it to the last newline-terminated record, re-deferres the
salvageable prefix into ``.deferred-captures/`` for normal drain, and
moves the original to ``.deferred-captures/.quarantine/`` so the operator
can inspect it.

Soft-recovery only: the original file is preserved verbatim under
``.quarantine/``; nothing is hard-deleted. Idempotent — a second run
finds no torn files to salvage and does no work.
"""
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def _salvage_one(
    src: Path,
    deferred_dir: Path,
    quarantine_dir: Path,
    *,
    dry_run: bool,
) -> tuple[int, int]:
    """Return (salvaged_records, dropped_bytes) for a single file."""
    try:
        raw = src.read_bytes()
    except OSError as exc:
        log.warning("salvage: cannot read %s: %s", src.name, exc)
        return (0, 0)

    if not raw:
        return (0, 0)

    last_nl = raw.rfind(b"\n")
    if last_nl < 0:
        return (0, len(raw))
    salvage_bytes = raw[: last_nl + 1]
    dropped = len(raw) - len(salvage_bytes)

    salvage_lines = [
        ln for ln in salvage_bytes.splitlines() if ln.strip()
    ]
    if not salvage_lines:
        return (0, dropped)

    if dry_run:
        return (len(salvage_lines), dropped)

    quarantine_dir.mkdir(parents=True, exist_ok=True)
    ts_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_target = quarantine_dir / f"{ts_stamp}-{src.name}"
    try:
        shutil.copy2(src, quarantine_target)
    except OSError as exc:
        log.warning("salvage: cannot quarantine %s: %s", src.name, exc)
        return (0, 0)

    salvage_name = src.name.replace(".permanent-failed", ".salvaged")
    salvage_path = deferred_dir / f"{ts_stamp}-{salvage_name}"
    tmp_path = salvage_path.with_suffix(salvage_path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as fh:
            fh.write(salvage_bytes)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp_path, salvage_path)
    except OSError as exc:
        log.warning("salvage: cannot write %s: %s", salvage_path.name, exc)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return (0, dropped)

    try:
        src.unlink()
    except OSError as exc:
        log.warning("salvage: cannot unlink %s after recovery: %s",
                    src.name, exc)

    return (len(salvage_lines), dropped)


def migrate_salvage_torn_permanent_failed(
    *,
    deferred_dir: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Salvage torn ``.permanent-failed-*.jsonl`` files.

    Returns a dict with keys: files_salvaged, records_salvaged,
    bytes_dropped, dry_run.
    """
    if deferred_dir is None:
        deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    deferred_dir = Path(deferred_dir)
    quarantine_dir = deferred_dir / ".quarantine"

    if not deferred_dir.exists():
        return {
            "files_salvaged": 0,
            "records_salvaged": 0,
            "bytes_dropped": 0,
            "dry_run": dry_run,
        }

    files_salvaged = 0
    records_salvaged = 0
    bytes_dropped = 0

    for entry in sorted(deferred_dir.iterdir()):
        if not entry.is_file():
            continue
        if ".permanent-failed-" not in entry.name:
            continue
        if entry.suffix != ".jsonl":
            continue
        recs, dropped = _salvage_one(
            entry, deferred_dir, quarantine_dir, dry_run=dry_run
        )
        if recs > 0 or dropped > 0:
            files_salvaged += 1
            records_salvaged += recs
            bytes_dropped += dropped

    return {
        "files_salvaged": files_salvaged,
        "records_salvaged": records_salvaged,
        "bytes_dropped": bytes_dropped,
        "dry_run": dry_run,
    }
