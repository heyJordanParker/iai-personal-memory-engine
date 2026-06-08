"""Memory capture — two write-side entry points:

1. `capture_turn(store, cue, text, tier, session_id)`:
   in-session, explicit. Called via MCP tool `memory_capture` when Claude
   detects a surprising correction, load-bearing decision, or lesson.

2. `capture_transcript(store, transcript_path, session_id)`:
   end-of-session, ambient. Called by `~/.claude/hooks/iai-mcp-session-capture.sh`
   Stop-hook on SessionEnd. Reads Claude Code JSONL transcript, extracts
   user + assistant turns, filters through shield + dedup, inserts records.

Both paths respect:
- Shield: HARD_BLOCK drops the record; FLAG_FOR_REVIEW stores with tag.
- Dedup: if query_similar returns a hit with cos >= DEDUP_THRESHOLD
  (0.95), we reinforce instead of insert (boost Hebbian edge).
- Language: defaults to 'en' (English-only brain invariant; Claude
  translates inbound text to English on the way in).
- Encryption: goes through the standard store.insert() path which handles
  AES-256-GCM column encryption.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from iai_mcp.exceptions import CaptureError, CaptureDeduplicationError, CaptureDrainError, NativeError

# Per-pass event cap for drain_deferred_captures. When a single file would
# push the running total past this threshold, the unprocessed remainder is
# rewritten to {basename}.partial.jsonl (header preserved) for the next pass.
MAX_DRAIN_EVENTS_PER_RUN = 5000

# Matches the active-writer marker name shape exactly: `{anything}.live.jsonl`
# with no hyphen-epoch between `.live` and `.jsonl`. The Stop-hook rename
# target `{id}.live-{epoch}.jsonl` does NOT match (the `-` after `.live`
# breaks the pattern).
_LIVE_ACTIVE_RE = re.compile(r"\.live\.jsonl$")

# `iai_mcp.embed` pulls in transformers + torch (~2.9s cold import). Loading
# capture.py for the `--no-spawn` deferred path (which never embeds anything)
# would exceed the 2s wall-clock budget. Moved to lazy import inside
# `capture_turn` — keeps the
# write_deferred_captures cold path under ~1s. `from __future__ import
# annotations` (line 29) keeps type hints intact without runtime import.
# `MemoryStore` left at module top — its 0.4s import is acceptable.
from iai_mcp.store import MemoryStore
from iai_mcp.types import (
    SCHEMA_VERSION_CURRENT,
    TIER_ENUM,
    MemoryRecord,
)

log = logging.getLogger(__name__)

DEDUP_COS_THRESHOLD = 0.95
MIN_CAPTURE_LEN = 12
MAX_CAPTURE_LEN = 8000

# Bounded retry policy for `.failed-*` deferred-capture evidence files.
# A file is retried up to FAILED_MAX_ATTEMPTS times with exponential
# backoff (60s, 120s, 240s); after that it transitions to
#.permanent-failed-<ts>.jsonl as a terminal evidence state and a
# `permanent_capture_failure` event is emitted at severity=critical.
FAILED_MAX_ATTEMPTS: int = 3
FAILED_BACKOFF_BASE_SEC: float = 60.0

_FAILED_ATTEMPT_RE = re.compile(r"-attempt-(\d+)\.jsonl$")
_FAILED_SHAPE_RE = re.compile(r"^(.+?)\.failed-(\d+)(?:-attempt-\d+)?\.jsonl$")

# Crash-loop quarantine markers and attempt counter. A drain pass that gets
# SIGKILL'd or panics mid-ingest leaves a `.processing-<pid>.jsonl` file in
# the deferred-captures queue. The next pass detects the stale pid and
# increments `.crash-N` counter; past QUARANTINE_MAX_ATTEMPTS the file is
# moved out of the active queue into `.quarantine/` so it never re-touches
# the daemon's startup path.
_PROCESSING_MARKER_RE = re.compile(r"\.processing-(\d+)\.jsonl$")
_CRASH_ATTEMPT_RE = re.compile(r"\.crash-(\d+)\.jsonl$")
QUARANTINE_MAX_ATTEMPTS: int = 2


# Liveness check for a pid found in a `.processing-<pid>.jsonl` marker.
# ProcessLookupError -> reaped; PermissionError -> exists but uid mismatch
# (treat as alive — another user owns it). Any other OSError -> dead.
def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# Strip the `.processing-<pid>` suffix and return (new_path, ok). On
# rename failure return (input_path, False) so the caller can skip
# downstream operations whose regex matchers assume the marker is gone
# (FAILED_SHAPE_RE end-anchored on.jsonl$ — silent failure here would
# create a duplicate attempt-1 file and reset the retry counter).
def _strip_processing_marker(
    path: Path, *, log_path: Path | None = None
) -> tuple[Path, bool]:
    new_name = _PROCESSING_MARKER_RE.sub(".jsonl", path.name)
    if new_name == path.name:
        return path, True  # nothing to strip — vacuous success
    new_path = path.with_name(new_name)
    try:
        path.rename(new_path)
    except OSError as e:
        if log_path is not None:
            try:
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} "
                        f"strip-marker-failed {path.name}: {type(e).__name__}\n"
                    )
            except (OSError, ValueError) as exc:
                log.debug("strip_marker_log_write_failed: %s", exc)
        return path, False
    return new_path, True


# Move a crash-looping file into `.quarantine/` with a UTC timestamp prefix
# so it can be inspected later. Emits `deferred_captures_quarantined` event
# at severity=warning, domain=ops. Fail-safe — quarantine MUST succeed even
# if the event sink raises.
def _quarantine_file(
    fpath: Path,
    store: "MemoryStore",
    *,
    log_path: Path,
    attempts: int,
) -> Path:
    quarantine_dir = fpath.parent / ".quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Strip any.processing-<pid> and.crash-<n> suffixes so the recovered
    # name matches the original deferred-capture basename.
    recovered = _PROCESSING_MARKER_RE.sub(".jsonl", fpath.name)
    recovered = _CRASH_ATTEMPT_RE.sub(".jsonl", recovered)

    ts_prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = quarantine_dir / f"{ts_prefix}-{recovered}"

    shutil.move(str(fpath), str(target))

    try:
        from iai_mcp.events import write_event

        write_event(
            store,
            "deferred_captures_quarantined",
            {
                "file": target.name,
                "reason": "crash_loop",
                "attempts": attempts,
            },
            severity="warning",
            domain="ops",
        )
    except Exception as exc:  # noqa: BLE001 -- fail-safe boundary
        log.debug("quarantine_event_write_failed: %s", exc)
        try:
            with log_path.open("a") as logf:
                logf.write(
                    f"{datetime.now(timezone.utc).isoformat()} "
                    f"quarantined-event-skipped {target.name}\n"
                )
        except (OSError, ValueError) as exc2:
            log.debug("quarantine_event_log_fallback_failed: %s", exc2)

    try:
        with log_path.open("a") as logf:
            logf.write(
                f"{datetime.now(timezone.utc).isoformat()} "
                f"quarantined {target.name}: crash_loop attempts={attempts}\n"
            )
    except (OSError, ValueError) as exc:
        log.debug("quarantine_log_write_failed: %s", exc)

    return target


def _parse_failed_attempt(name: str) -> int:
    """Return the prior-attempt count encoded in a deferred-capture filename.

    Mapping:
      * ``<base>.failed-<ts>-attempt-<n>.jsonl`` -> n
      * ``<base>.failed-<ts>.jsonl`` (legacy shape) -> 1
      * any name without ``.failed-`` (clean file) -> 0
    """
    m = _FAILED_ATTEMPT_RE.search(name)
    if m:
        return int(m.group(1))
    if ".failed-" in name:
        return 1
    return 0


def _advance_failed_path(
    fpath: Path,
    store: "MemoryStore",
    *,
    first_error: str,
    log_path: Path,
) -> Path:
    """Rename ``fpath`` forward to the next-attempt or terminal evidence shape.

    Clean -> attempt-1, attempt-N -> attempt-(N+1), and on the fourth
    failure (would-be-attempt-4) the file transitions to
    ``.permanent-failed-<ts>.jsonl`` and a ``permanent_capture_failure``
    event is emitted at severity=critical.

    The timestamp from the prior filename is preserved when present so the
    permanent-failed shape stays correlatable to the original failure;
    clean files use ``int(time.time())``.
    """
    prior_attempt = _parse_failed_attempt(fpath.name)
    next_attempt = prior_attempt + 1
    m = _FAILED_SHAPE_RE.match(fpath.name)
    if m:
        base = m.group(1)
        ts_str = m.group(2)
    else:
        base = fpath.stem
        ts_str = str(int(time.time()))
    if next_attempt > FAILED_MAX_ATTEMPTS:
        new_name = f"{base}.permanent-failed-{ts_str}.jsonl"
        failed_path = fpath.with_name(new_name)
        fpath.rename(failed_path)
        try:
            # Lazy import: keeps capture.py importable in low-cost paths
            # (mirrors the memory_bank lazy-import precedent inside this
            # module).
            from iai_mcp.events import write_event

            write_event(
                store,
                "permanent_capture_failure",
                {
                    "file": new_name,
                    "first_error": first_error,
                    "attempts": FAILED_MAX_ATTEMPTS,
                },
                severity="critical",
                domain="ops",
            )
        except Exception as exc:  # noqa: BLE001 -- fail-safe boundary
            log.debug("permanent_capture_failure_event_failed: %s", exc)
            try:
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} "
                        f"permanent_capture_failure-event-skipped {new_name}\n"
                    )
            except (OSError, ValueError) as exc2:
                log.debug("permanent_capture_failure_log_failed: %s", exc2)
        return failed_path
    new_name = f"{base}.failed-{ts_str}-attempt-{next_attempt}.jsonl"
    failed_path = fpath.with_name(new_name)
    fpath.rename(failed_path)
    return failed_path


def _run_shield(text: str) -> tuple[str, list[str]]:
    """Run shield; return (verdict, tags) where verdict in HARD_BLOCK|FLAG|OK."""
    try:
        from iai_mcp.shield import evaluate

        result = evaluate(text)
        verdict = getattr(result, "verdict", "OK")
        tags = list(getattr(result, "tags", []) or [])
        return verdict, tags
    except Exception as exc:  # noqa: BLE001 -- capture fail-safe
        log.debug("shield_evaluate_failed: %s", exc)
        return "OK", []


def _resolve_ts(ts: str | None) -> datetime:
    """Parse an ISO-8601 timestamp string to a tz-aware datetime.

    Returns datetime.now(timezone.utc) when ts is absent, empty, or
    malformed — never raises.
    """
    if ts:
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc)


def _idem_tag(
    session_id: str,
    role: str,
    ts_iso: str,
    text: str,
    *,
    source_uuid: str | None = None,
) -> str:
    """Derive a deterministic idempotency tag for an episodic conversational turn.

    When ``source_uuid`` is provided (the transcript line's native ``uuid``
    field), the key is ``session_id|role|source_uuid`` — stable across
    re-emissions regardless of when the hook fires. Fallback (no uuid, e.g.
    test fixtures or older Claude Code versions) uses the original
    ``session_id|role|ts|text`` key unchanged so existing tests remain green.
    """
    if source_uuid:
        key = f"{session_id}|{role}|{source_uuid}"
    else:
        key = f"{session_id}|{role}|{ts_iso}|{text}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"idem:{digest}"


def _is_episodic_conversational(tier: str, role: str) -> bool:
    """Return True for episodic conversational turns (user or assistant role).

    Used identically at gate #1 (capture_turn) and gate #2
    (_pattern_separation_gate_with_hits) to keep both exemption sites in
    lock-step. Intentionally narrow: semantic and consolidated tiers always
    return False so their cos-dedup paths remain unchanged.
    """
    return tier == "episodic" and role in {"user", "assistant"}


def capture_turn(
    store: MemoryStore,
    *,
    cue: str,
    text: str,
    tier: str = "episodic",
    session_id: str = "-",
    role: str = "user",
    ts: str | None = None,
    source_uuid: str | None = None,
) -> dict[str, Any]:
    """Write a single conversation turn to the iai-mcp store.

    Args:
        source_uuid: the transcript line's native ``uuid`` field when
            available. Used as the idempotency key so re-emissions of the
            same transcript line (e.g. after an empty/failing offset resets
            to 0) map to an identical idem tag and are deduplicated without
            creating a duplicate row. Falls back to the (session, role,
            ts, text) key when absent (e.g. older Claude Code, test
            fixtures).

    Returns {"status": "inserted|reinforced|skipped", "record_id": uuid-or-null,
             "reason": short-string}.
    """
    if tier not in TIER_ENUM:
        return {"status": "skipped", "record_id": None, "reason": f"invalid tier {tier!r}"}

    text = (text or "").strip()
    if len(text) < MIN_CAPTURE_LEN:
        return {"status": "skipped", "record_id": None, "reason": "too short"}
    if len(text) > MAX_CAPTURE_LEN:
        text = text[:MAX_CAPTURE_LEN]

    verdict, shield_tags = _run_shield(text)
    if verdict == "HARD_BLOCK":
        return {"status": "skipped", "record_id": None, "reason": "shield HARD_BLOCK"}

    # Resolve the live-event timestamp BEFORE the dedup gate so the idem key
    # is stable across re-drains (same ts => same hash). Fallback to
    # now() when ts is absent, empty, or malformed.
    now = _resolve_ts(ts)

    # Lazy import: keeps the cold module-import cost low for the
    # `--no-spawn` deferred path which never embeds.
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.events import TELEMETRY_EMBED_NATIVE_FAILURE, write_event

    try:
        emb = embedder_for_store(store).embed(cue or text)
    except Exception as exc:
        write_event(
            store,
            TELEMETRY_EMBED_NATIVE_FAILURE,
            {
                "op_type": "capture",
                "backend": "rust",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise NativeError(f"capture encode failed: {exc}") from exc
    embedding = list(emb)

    # Dedup gate: behaviour depends on whether this is an episodic
    # conversational turn (user or assistant role).
    #
    # Episodic conversational turns: bypass the cosine near-dup SKIP.
    # Distinct turns from different sessions (or different points in time)
    # must each produce their own row even when their text is near-identical.
    # An exact re-drain of the SAME turn (same session/role/ts/text) is
    # detected by an exact idempotency key stored as a plaintext tag;
    # only that case returns "reinforced" (no duplicate row).
    #
    # Non-conversational episodic and semantic/consolidated tiers: use the
    # original cos>=0.95 reinforce path unchanged.
    if _is_episodic_conversational(tier, role):
        ts_iso = now.isoformat()
        idem_t = _idem_tag(session_id, role, ts_iso, text, source_uuid=source_uuid)
        existing_id = store.find_record_by_tag(idem_t)
        if existing_id is not None:
            # True exact re-drain: skip without creating a duplicate row.
            try:
                store.reinforce_record(existing_id)
            except (ValueError, IOError) as exc:
                log.warning(
                    "capture_dedup_reinforce_failed",
                    extra={
                        "err_type": type(exc).__name__,
                        "record_id": str(existing_id),
                    },
                )
            return {
                "status": "reinforced",
                "record_id": str(existing_id),
                "reason": "exact-key re-drain",
            }
        # Distinct turn: fall through to build and insert.
    else:
        # Non-conversational path: original cos-dedup (query_similar at the same
        # tier, returns list[tuple[MemoryRecord, float]]).
        try:
            neighbours = store.query_similar(embedding, k=3, tier=tier)
        except (ValueError, IOError) as exc:
            log.warning(
                "capture_dedup_query_failed",
                extra={"err_type": type(exc).__name__, "err": str(exc)[:120]},
            )
            neighbours = []

        for record, score in neighbours:  # tuple-unpack
            if score >= DEDUP_COS_THRESHOLD:
                try:
                    store.reinforce_record(record.id)
                except (ValueError, IOError) as exc:
                    log.warning(
                        "capture_dedup_reinforce_failed",
                        extra={
                            "err_type": type(exc).__name__,
                            "record_id": str(record.id),
                        },
                    )
                return {
                    "status": "reinforced",
                    "record_id": str(record.id),
                    "reason": f"cos={score:.3f} >= {DEDUP_COS_THRESHOLD}",
                }

    tags = ["capture", f"role:{role}"]
    if verdict == "FLAG_FOR_REVIEW":
        tags.append("shield:flagged")
        tags.extend(f"shield:{t}" for t in shield_tags[:3])

    # Stamp the idempotency tag on episodic conversational turns so both
    # this gate and gate #2 can probe for exact re-drains cheaply (plaintext
    # tags_json scan, no provenance decrypt).
    if _is_episodic_conversational(tier, role):
        ts_iso = now.isoformat()
        tags.append(_idem_tag(session_id, role, ts_iso, text, source_uuid=source_uuid))

    rec = MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=embedding,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": now.isoformat(), "cue": cue or "(auto-capture)",
                     "session_id": session_id, "role": role}],
        created_at=now,
        updated_at=now,
        tags=tags,
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )

    try:
        store.insert(rec)
    except Exception as e:
        log.exception("capture_turn insert failed")
        return {"status": "skipped", "record_id": None, "reason": f"insert-failed: {type(e).__name__}"}

    # STC peri-event buffer add. Newly-inserted record joins the
    # ring buffer so a near-future STRONG_EVENT can upgrade it from semantic
    # to episodic. None-guard: CLI / one-shot paths that never register a
    # daemon buffer skip silently. Buffer-add failure must NOT block the
    # capture success path (consistency with the observability-vs-correctness
    # split elsewhere in capture_turn). Only the inserted branch buffers;
    # reinforced (dedup-hit) and skipped paths are not peri-event candidates.
    try:
        from iai_mcp.peri_event_buffer import get_buffer
        buf = get_buffer()
        if buf is not None:
            buf.add(rec.id, rec.created_at, rec.tier)
    except Exception as exc:  # noqa: BLE001 -- capture fail-safe
        log.warning(
            "capture_peri_event_buffer_add_failed",
            extra={
                "record_id": str(rec.id),
                "err_type": type(exc).__name__,
            },
        )

    return {"status": "inserted", "record_id": str(rec.id), "reason": f"tier={tier}"}


def capture_transcript(
    store: MemoryStore,
    transcript_path: Path | str,
    *,
    session_id: str = "-",
    max_turns: int = 200,
) -> dict[str, Any]:
    """Read a Claude Code JSONL transcript, capture user + assistant turns.

    Returns {"inserted": N, "reinforced": M, "skipped": K, "errors": E}.
    """
    path = Path(transcript_path).expanduser()
    if not path.exists():
        return {"inserted": 0, "reinforced": 0, "skipped": 0, "errors": 1,
                "reason": f"transcript not found: {path}"}

    counts = {"inserted": 0, "reinforced": 0, "skipped": 0, "errors": 0}
    seen = 0
    with path.open() as fh:
        for line in fh:
            if seen >= max_turns:
                break
            seen += 1
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                log.debug("capture_transcript_json_parse_failed: %s", exc)
                counts["errors"] += 1
                continue
            msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
            role = obj.get("type") or msg.get("role", "")
            if role not in {"user", "assistant"}:
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                # Claude Code messages use block format; collect text blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                text = "\n".join(text_parts).strip()
            else:
                text = str(content).strip()
            if not text:
                continue
            result = capture_turn(
                store,
                cue=f"session {session_id} turn {seen}",
                text=text,
                tier="episodic",
                session_id=session_id,
                role=role,
                ts=obj.get("timestamp"),
                source_uuid=obj.get("uuid"),
            )
            status = result.get("status", "skipped")
            if status in counts:
                counts[status] += 1
            else:
                counts["skipped"] += 1

    return counts


# Structural markers injected by the harness/hook layer — never genuine user text.
# Each tuple is (match_type, prefix_or_exact_string).
# Match rules: "startswith" checks text.startswith(s); "equals" checks text == s.
# NEVER use substring ("in") matching — a genuine user line quoting a marker
# mid-sentence must survive (lossless invariant).
_NOISE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("startswith", "<command-message>"),
    ("startswith", "<command-name>"),
    ("startswith", "Base directory for this skill:"),
    ("startswith", "<task-notification>"),
    ("equals",     "[Request interrupted by user]"),
)


def _is_noise(text: str) -> bool:
    """Return True when *text* matches one of the structural injection markers.

    Uses only ``startswith`` and ``==`` — never substring containment — so
    a genuine user turn that happens to quote a marker mid-sentence is never
    dropped (lossless invariant).
    """
    for match_type, pattern in _NOISE_PATTERNS:
        if match_type == "startswith":
            if text.startswith(pattern):
                return True
        else:  # "equals"
            if text == pattern:
                return True
    return False


def _parse_transcript_line(
    line: str,
) -> tuple[str, str, str | None, str | None] | None:
    """Parse a Claude Code transcript JSONL line into (role, text, uuid, timestamp).

    Returns None if the line is not a user/assistant turn, has empty text,
    matches a structural injection marker, or cannot be JSON-decoded.
    Used by both ``capture_transcript`` and the deferred-write paths to
    keep parsing rules in one place.

    The returned ``uuid`` and ``timestamp`` are the transcript line's native
    identity fields (present in real Claude Code JSONL; absent in simplified
    test fixtures). Callers that only need (role, text) can ignore the extra
    fields.
    """
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
    role = obj.get("type") or msg.get("role", "")
    if role not in {"user", "assistant"}:
        return None
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = "\n".join(parts).strip()
    else:
        text = str(content).strip()
    if not text:
        return None
    if _is_noise(text):
        return None
    return role, text, obj.get("uuid"), obj.get("timestamp")


def write_deferred_event(
    session_id: str,
    role: str,
    text: str,
    *,
    cwd: str | None = None,
    ts: str | None = None,
    source_uuid: str | None = None,
) -> Path:
    """Append a single JSONL event to `{session_id}.live.jsonl`.

    Creates the file with a header on first call; appends events on
    subsequent calls. Pure file IO — no daemon socket, no embedder,
    no shield, no store.

    The drain function skips files matching the exact `*.live.jsonl`
    suffix, so the writer/drain race is structurally impossible while
    this file is the active marker. The Stop hook renames the file to
    `{session_id}.live-{epoch}.jsonl` at session end; the drain then
    picks it up.

    Format invariants are duplicated by the per-turn shell hook at
    src/iai_mcp/_deploy/hooks/iai-mcp-turn-capture.sh — keep
    header/event keys in sync.
    """
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    path = deferred_dir / f"{session_id}.live.jsonl"
    need_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a") as fh:
        if need_header:
            header = {
                "version": 1,
                "deferred_at": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "cwd": cwd or os.getcwd(),
            }
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        event = {
            "text": text,
            "cue": f"session {session_id} turn",
            "tier": "episodic",
            "role": role,
            "ts": ts if ts else datetime.now(timezone.utc).isoformat(),
        }
        if source_uuid:
            event["source_uuid"] = source_uuid
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Pending live-capture read helper (immediate recall, daemon-drain-independent)
# ---------------------------------------------------------------------------

# Per-file event-line budget for the deque tail read. Keeps only the LAST
# ≤500 complete lines of a live file, bounding memory at any file size without
# skipping large append-only files, and without any byte-seek (UTF-8 safe).
_TAIL_MAX_EVENT_LINES: int = 500

# Maximum live / processing files to open per read_pending_live_events call.
# Applies in both the bare (no session_id) and session-aware paths.
_LIVE_SELECT_MAX_FILES: int = 20


def read_pending_live_events(session_id: str | None = None) -> list[dict]:
    """Return pending (not-yet-drained) live-capture events as a list of dicts.

    Reads ``.live.jsonl`` and ``.processing-*.jsonl`` files from the
    ``.deferred-captures`` directory under the current HOME. Pure file IO —
    no embed, no store write, no daemon RPC.

    Parameters
    ----------
    session_id:
        When given, force-include the requested session's own allowlisted
        files first (session-aware-additive selection), then fill the
        remaining ≤20 slots from other files by mtime descending. Events
        from other sessions are returned too (the final per-file header
        filter keeps only events whose header session_id matches the requested
        session_id, but the selection step ensures the session's own file is
        always in the candidate set even when 20+ newer other-session files
        exist).
        When ``None``, plain mtime-desc top-20 selection (bare / global path).

    Returns
    -------
    list[dict]
        Each dict has keys: ``text``, ``role``, ``tier``, ``session_id``,
        ``ts`` (tz-aware datetime, for sorting), ``ts_iso`` (the
        ``_resolve_ts(...).isoformat()`` string — the contract key for the
        idem-tag in callers), and ``source_uuid`` (str | None).
        Sorted by ``ts`` descending (newest first).
    """
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    if not deferred_dir.exists():
        return []

    # Step 1: enumerate allowlisted entries via os.scandir (no sort/stat of whole dir).
    # ALLOWLIST: _LIVE_ACTIVE_RE matches {anything}.live.jsonl;
    # _PROCESSING_MARKER_RE matches {stem}.processing-{pid}.jsonl
    # Everything else (.failed-,.permanent-failed-,.live-{epoch}.jsonl without
    #.processing,.partial,.crash-N,.quarantine) is excluded structurally.
    allowlisted: list[tuple[Path, float]] = []
    try:
        with os.scandir(deferred_dir) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                name = entry.name
                if _LIVE_ACTIVE_RE.search(name) or _PROCESSING_MARKER_RE.search(name):
                    try:
                        st = entry.stat()
                        allowlisted.append((Path(entry.path), st.st_mtime))
                    except OSError:
                        pass
    except OSError:
        return []

    if not allowlisted:
        return []

    # Step 2: SESSION-AWARE-ADDITIVE selection, capped at _LIVE_SELECT_MAX_FILES.
    if session_id is None:
        # Bare / global path: plain mtime-desc top-20.
        allowlisted.sort(key=lambda t: t[1], reverse=True)
        candidates = allowlisted[:_LIVE_SELECT_MAX_FILES]
    else:
        # Session-aware path: force-include the requested session's own files FIRST.
        # "Own" = filename starts with "{session_id}.live" (matches active live file
        # AND drain-claimed "{session_id}.live-{epoch}.processing-{pid}.jsonl").
        prefix = f"{session_id}.live"
        own = [(p, m) for p, m in allowlisted if p.name.startswith(prefix)]
        other = [(p, m) for p, m in allowlisted if not p.name.startswith(prefix)]

        # Sort each group by mtime desc.
        own.sort(key=lambda t: t[1], reverse=True)
        other.sort(key=lambda t: t[1], reverse=True)

        # Take all own files first (up to the cap), then fill with other.
        cap = _LIVE_SELECT_MAX_FILES
        own_capped = own[:cap]
        remaining = cap - len(own_capped)
        candidates = own_capped + other[:remaining]

    # Steps 3–6: read each candidate file.
    events: list[dict] = []
    for path, _mtime in candidates:
        try:
            with path.open(encoding="utf-8") as fh:
                # Step 3: header read (front).
                first_line = fh.readline()
                if not first_line.endswith("\n"):
                    # Incomplete header — skip file.
                    continue
                try:
                    header = json.loads(first_line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if header.get("version", 0) > 1:
                    # Forward-compat: skip unknown versions.
                    continue
                file_session_id = header.get("session_id", "-")
                # Header is authoritative: apply session filter here.
                if session_id is not None and file_session_id != session_id:
                    continue

                # Step 4: TAIL read via maxlen deque on the SAME file handle.
                # The handle is already positioned after the header line —
                # no byte-seek, no re-read from offset 0, UTF-8 safe.
                tail = deque(fh, maxlen=_TAIL_MAX_EVENT_LINES)

                # Step 5: drop the trailing partial line (no terminating \n).
                complete_lines = [ln for ln in tail if ln.endswith("\n")]

                # Step 6: parse events.
                for line in complete_lines:
                    try:
                        ev = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    # Normalize ts (contract: carry BOTH the datetime and
                    # the isoformat string so callers use ts_iso for idem-tags).
                    ts_raw = ev.get("ts")
                    ts_dt = _resolve_ts(ts_raw)
                    ts_iso = ts_dt.isoformat()
                    events.append({
                        "text": ev.get("text", ""),
                        "role": ev.get("role", "user"),
                        "tier": ev.get("tier", "episodic"),
                        "session_id": file_session_id,
                        "ts": ts_dt,
                        "ts_iso": ts_iso,
                        "source_uuid": ev.get("source_uuid"),
                    })
        except OSError:
            continue

    # Step 7: sort newest-first by resolved ts datetime.
    events.sort(key=lambda e: e["ts"], reverse=True)
    return events


# ---------------------------------------------------------------------------
# Deferred-captures writer for `--no-spawn` hook mode
# ---------------------------------------------------------------------------


def write_deferred_captures(
    session_id: str,
    transcript_path: Path | str,
    *,
    cwd: str | None = None,
    max_turns: int = 200,
) -> Path:
    """Defer transcript capture by writing events to a JSONL file under
    ``~/.iai-mcp/.deferred-captures/``. Returns the path written.

    Used by ``iai-mcp capture-transcript --no-spawn`` when the
    daemon is unreachable. The Stop hook calls this so it never blocks
    session teardown waiting for a daemon spawn.

    The daemon's drain loop consumes these on next WAKE. Format is JSONL v1:

    - Line 1: header ``{"version":1,"deferred_at":<ISO>,"session_id":<id>,"cwd":<path>}``
    - Lines 2..N: one event per user/assistant turn
      ``{"text":<verbatim>,"cue":<short>,"tier":"episodic","role":<u|a>,"ts":<ISO>}``

    Pure-write: no MemoryStore touch, no socket touch, no daemon import.
    Uses ``Path.home()`` at call time so HOME-monkeypatched tests get the
    right tmp dir. Idempotent ``mkdir(parents=True, exist_ok=True)``.

    Args:
        session_id: Claude Code session id (provenance + filename component).
        transcript_path: path to the JSONL transcript file (or non-existent —
            we write the header then return; daemon drain treats as no-op).
        cwd: optional CWD override for the header (defaults to ``os.getcwd()``).
        max_turns: cap on transcript turns to emit (default 200, matches
            ``capture_transcript`` semantics).

    Returns:
        ``Path`` of the written ``.jsonl`` file.

    Notes:
        - Filename pattern ``{session_id}-{int(time.time())}.jsonl`` — the
          unix-ts suffix avoids collisions if the same session captures
          multiple times.
        - Reuses the same parsing logic as ``capture_transcript`` so the
          deferred path and the inline path stay consistent.
        - Returns even on missing transcript (writes header only) — daemon
          drain treats as no-op. Hook MUST never raise here.
        - Stdlib only: ``json``, ``time``, ``pathlib.Path``, ``datetime``, ``os``.
    """
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    out_path = deferred_dir / f"{session_id}-{int(time.time())}.jsonl"
    with out_path.open("w") as fh:
        # Header (line 1, version=1 forward-compat marker).
        header = {
            "version": 1,
            "deferred_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "cwd": cwd or os.getcwd(),
        }
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        # Read transcript and emit one event per user/assistant turn.
        path = Path(transcript_path).expanduser()
        if not path.exists():
            return out_path  # empty body — daemon drain will treat as no-op
        seen = 0
        with path.open() as src:
            for line in src:
                if seen >= max_turns:
                    break
                seen += 1
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
                role = obj.get("type") or msg.get("role", "")
                if role not in {"user", "assistant"}:
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    text = "\n".join(text_parts).strip()
                else:
                    text = str(content).strip()
                if not text:
                    continue
                event = {
                    "text": text,
                    "cue": f"session {session_id} turn {seen}",
                    "tier": "episodic",
                    "role": role,
                    "ts": obj.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                }
                src_uuid = obj.get("uuid")
                if src_uuid:
                    event["source_uuid"] = src_uuid
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Deferred-captures drain (READ side, daemon-resident)
# ---------------------------------------------------------------------------


def drain_deferred_captures(store: MemoryStore) -> dict[str, int]:
    """Consume ``~/.iai-mcp/.deferred-captures/*.jsonl`` produced by
    ``iai-mcp capture-transcript --no-spawn`` (WRITE side).

    For each ``.jsonl`` file in the deferred-captures dir:

    * Read line 1 (header). If ``version > 1`` (forward-compat guard), log a
      "skip" line to ``~/.iai-mcp/logs/deferred-drain-YYYY-MM-DD.log`` and
      leave the file in place — a future daemon version will know how to
      handle it.
    * For each event line (lines 2..N), call ``capture_turn(store,...)``
      and inspect its return-status dict. W2 /:
      - status="inserted" → events_inserted += 1
      - status="reinforced" → events_reinforced += 1
      - status="skipped" with reason matching ^insert-failed:* (capture_turn
        path where store.insert raised) → events_skipped_insert_failed += 1
        and the WHOLE FILE is treated as failed: renamed to
        .failed-<ts>.jsonl, NOT unlinked.
      - status="skipped" with any other reason (shield HARD_BLOCK, too short,
        invalid tier — all *intentional* drops) → events_skipped_intentional
        += 1.
    * On full success (zero insert-failed events): delete the file,
      files_drained += 1.
    * On any insert-failed event: rename the file to
      ``<basename>.failed-<unix_ts>.jsonl`` (preserves evidence for manual
      inspection), log a "insert-failed" line with the first error,
      files_failed += 1.
    * On parser/header exception: same outer rename + log path as before
      (existing behaviour), files_failed += 1.
    * On 0-byte / empty file: delete it (no-op header-only deferral).

    Idempotent: re-running on a directory with no ``.jsonl`` files (or no
    deferred-captures dir at all) returns zero counts without error.

    Returns dict with keys:
        files_drained, files_failed,
        events_inserted, events_reinforced,
        events_skipped_intentional, events_skipped_insert_failed.

    Notes:
        - Uses ``Path.home()`` at call time so HOME-monkeypatched tests get
          the right tmp dir.
        - Stdlib only — no new deps.
        - Caller (daemon.main / _tick_body) MUST wrap in try/except so a
          drain crash never propagates into the asyncio event loop. This
          function itself catches per-file exceptions defensively.
        - The ``store`` argument is the same MemoryStore instance the
          daemon uses for all other writes (so connection/lock semantics
          are consistent). Drain MUST run inside ``asyncio.to_thread`` from
          async callers because ``capture_turn`` does sync store I/O.
    """
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    log_dir = Path.home() / ".iai-mcp" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        log_dir / f"deferred-drain-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
    )
    counts = {
        "files_drained": 0,
        "files_failed": 0,
        "events_inserted": 0,
        "events_reinforced": 0,
        "events_skipped_intentional": 0,
        "events_skipped_insert_failed": 0,
    }
    if not deferred_dir.exists():
        return counts
    total_events_processed = 0
    cap_hit = False

    # Stale-PID rescan. A `.processing-<pid>.jsonl` whose pid is dead
    # signals a crashed drain. Increment the `.crash-N` counter; at
    # N+1 > QUARANTINE_MAX_ATTEMPTS, move the file into `.quarantine/`
    # so subsequent drain passes never re-touch it.
    for fpath in sorted(deferred_dir.iterdir()):
        if not fpath.is_file():
            continue
        m = _PROCESSING_MARKER_RE.search(fpath.name)
        if not m:
            continue
        pid = int(m.group(1))
        if _pid_is_alive(pid):
            continue  # another drain owns it; leave alone
        base_no_marker = _PROCESSING_MARKER_RE.sub(".jsonl", fpath.name)
        crash_m = _CRASH_ATTEMPT_RE.search(base_no_marker)
        if crash_m:
            prior_n = int(crash_m.group(1))
            base_no_crash = _CRASH_ATTEMPT_RE.sub(".jsonl", base_no_marker)
        else:
            prior_n = 0
            base_no_crash = base_no_marker
        next_n = prior_n + 1
        if next_n > QUARANTINE_MAX_ATTEMPTS:
            try:
                _quarantine_file(
                    fpath, store, log_path=log_path, attempts=next_n
                )
            except Exception as exc:  # noqa: BLE001 -- fail-safe boundary
                log.debug("quarantine_file_failed: %s", exc)
        else:
            new_name = base_no_crash.replace(
                ".jsonl", f".crash-{next_n}.jsonl"
            )
            try:
                fpath.rename(fpath.with_name(new_name))
            except Exception as exc:  # noqa: BLE001
                log.debug("crash_rename_failed %s: %s", fpath.name, exc)

    # Iterate every.jsonl entry but skip exact active-writer markers,
    # *.failed-*.jsonl files (preserved evidence from prior drains),
    # *.processing-<pid>.jsonl markers (owned by another drain or just
    # advanced by the stale-PID rescan above), and the.quarantine/
    # subdirectory.
    candidates = []
    for fpath in sorted(deferred_dir.iterdir()):
        if not fpath.is_file():
            continue  # skips.quarantine/ subdir and anything non-regular
        if fpath.suffix != ".jsonl":
            continue
        if _LIVE_ACTIVE_RE.search(fpath.name):
            continue
        # Skip `.processing-<pid>.jsonl` files — either owned by another
        # drain (pid alive) or just renamed forward by the rescan loop
        # (pid dead). Either way they are not standalone candidates.
        if _PROCESSING_MARKER_RE.search(fpath.name):
            continue
        # Terminal evidence — never reprocess.
        if ".permanent-failed-" in fpath.name:
            continue
        # `.failed-<ts>[-attempt-N].jsonl` is a retry candidate when its
        # per-attempt backoff has elapsed. Younger files are left untouched
        # so a flaky downstream can recover before the file ages forward.
        if ".failed-" in fpath.name:
            attempt_n = _parse_failed_attempt(fpath.name)
            backoff_sec = FAILED_BACKOFF_BASE_SEC * (2 ** (attempt_n - 1))
            try:
                file_mtime = fpath.stat().st_mtime
            except OSError:
                continue
            if (time.time() - file_mtime) < backoff_sec:
                continue
        candidates.append(fpath)

    for fpath in candidates:
        if cap_hit:
            break
        # Atomic ownership claim — rename to `.processing-<pid>.jsonl` so a
        # concurrent drain skips it (filter above) AND a SIGKILL'd drain
        # leaves a stale marker the next pass detects as a crash. A FileNot-
        # FoundError means another drain beat us; OSError covers transient
        # filesystem hiccups — both cases skip silently.
        claim_path = fpath.with_name(
            fpath.stem + f".processing-{os.getpid()}.jsonl"
        )
        # Discriminate "lost the race" from "system error". FileNotFoundError
        # stays silent (another drain beat us to the rename). Any other OSError
        # (EACCES after a directory chmod, ENOSPC, EROFS) gets one log line —
        # without this, a chmod'd `.deferred-captures/` loops the same file
        # forever with no observable signal.
        try:
            fpath.rename(claim_path)
        except FileNotFoundError:
            continue
        except OSError as e:
            try:
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} "
                        f"claim-failed {fpath.name}: {type(e).__name__}\n"
                    )
            except (OSError, ValueError) as exc:
                log.debug("claim_failed_log_write_failed: %s", exc)
            continue
        work_path = claim_path

        file_had_insert_failure = False
        file_first_error: str | None = None
        try:
            with work_path.open() as fh:
                lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
            if not lines:
                # Empty file (e.g. partial write that never got header) — drop.
                work_path.unlink()
                continue
            header = json.loads(lines[0])
            if header.get("version", 0) > 1:
                # Forward-compat guard: leave the file in place; a future
                # daemon revision will know the format. Strip the marker so
                # the next drain re-evaluates the clean filename.
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} skip "
                        f"{work_path.name}: version={header.get('version')}\n"
                    )
                # Best-effort strip; even if it fails we still `continue` to
                # skip the file this pass — forward-compat skip doesn't touch
                # the file's data.
                _strip_processing_marker(work_path, log_path=log_path)
                continue
            session_id = header.get("session_id", "-")
            event_lines = lines[1:]
            processed_in_file = 0
            for idx, ln in enumerate(event_lines):
                if total_events_processed >= MAX_DRAIN_EVENTS_PER_RUN:
                    # Cap reached mid-file — write the unprocessed remainder
                    # to {basename}.partial.jsonl atomically and unlink the
                    # original only after the partial is durable on disk.
                    # Strip the.processing-<pid> marker first so partial's
                    # basename matches the original file's shape.
                    remainder = event_lines[idx:]
                    # If strip fails, abort the partial write and leave the
                    # file with the.processing-<pid> marker — the next pass's
                    # stale-PID rescan will crash-bump it via the PATCH A
                    # rescan loop. Logging the strip failure (inside the
                    # helper) is enough to make this observable.
                    work_path, _strip_ok = _strip_processing_marker(
                        work_path, log_path=log_path
                    )
                    if not _strip_ok:
                        cap_hit = True
                        break
                    partial_path = work_path.with_suffix(".partial.jsonl")
                    tmp_path = work_path.with_suffix(".partial.tmp")
                    with tmp_path.open("w") as ph:
                        ph.write(lines[0] + "\n")
                        for r in remainder:
                            ph.write(r + "\n")
                        ph.flush()
                        os.fsync(ph.fileno())
                    os.replace(tmp_path, partial_path)
                    work_path.unlink()
                    counts["files_drained"] += 1
                    cap_hit = True
                    break
                ev = json.loads(ln)
                # Reuse capture_turn so the deferred path lands in the same
                # shield + dedup + encryption pipeline as live captures.
                result = capture_turn(
                    store,
                    cue=ev.get("cue", ""),
                    text=ev.get("text", ""),
                    tier=ev.get("tier", "episodic"),
                    session_id=session_id,
                    role=ev.get("role", "user"),
                    ts=ev.get("ts"),
                    source_uuid=ev.get("source_uuid"),
                )
                status = result.get("status", "skipped")
                reason = result.get("reason", "")
                if status == "inserted":
                    counts["events_inserted"] += 1
                    # Mirror the just-inserted record into the encrypted
                    # bank/recent window file. Fail-safe -- the store is the
                    # single source of truth; bank/recent is a denormalized
                    # read-side cache for the future substring-fallback path.
                    try:
                        from iai_mcp.memory_bank import append_recent_record

                        rid_str = result.get("record_id")
                        if rid_str:
                            rec = store.get(UUID(rid_str))
                            if rec is not None:
                                append_recent_record(store, rec)
                    except Exception:  # noqa: BLE001 -- best-effort fail-safe boundary
                        log.warning(
                            "bank-recent append failed for record %s",
                            result.get("record_id"),
                            exc_info=True,
                        )
                elif status == "reinforced":
                    counts["events_reinforced"] += 1
                elif status == "skipped" and reason.startswith("insert-failed:"):
                    counts["events_skipped_insert_failed"] += 1
                    file_had_insert_failure = True
                    if file_first_error is None:
                        file_first_error = reason
                else:
                    counts["events_skipped_intentional"] += 1
                total_events_processed += 1
                processed_in_file += 1
            if cap_hit:
                break
            if file_had_insert_failure:
                #: preserve the file as evidence — at least one
                # event hit the insert-failed code path inside capture_turn
                # (store.insert raised, capture_turn swallowed and returned
                # status=skipped reason=insert-failed:*). Pre-07.9 the file
                # was unlinked here and the data was silently lost.
                # Skip _advance_failed_path on strip failure — its
                # FAILED_SHAPE_RE is end-anchored on `.jsonl$` and would not
                # match a path that still has a `.processing-<pid>` segment,
                # silently creating attempt-1 and resetting the retry counter.
                work_path, _strip_ok = _strip_processing_marker(
                    work_path, log_path=log_path
                )
                if not _strip_ok:
                    try:
                        with log_path.open("a") as logf:
                            logf.write(
                                f"{datetime.now(timezone.utc).isoformat()} "
                                f"insert-failed-skip {work_path.name}: "
                                f"strip-failed, leaving for next pass\n"
                            )
                    except (OSError, ValueError) as exc:
                        log.debug("insert_failed_skip_log_write_failed: %s", exc)
                    counts["files_failed"] += 1
                    continue
                failed_path = _advance_failed_path(
                    work_path,
                    store,
                    first_error=file_first_error or "unknown",
                    log_path=log_path,
                )
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} insert-failed "
                        f"{work_path.name}: first_error={file_first_error}\n"
                    )
                counts["files_failed"] += 1
            else:
                work_path.unlink()
                counts["files_drained"] += 1
        except Exception as e:  # noqa: BLE001 -- per-file isolation, never raise
            try:
                # Preserve evidence: rename so the next drain pass skips it
                # AND a human can inspect the failure.
                # Same skip-on-strip-failure pattern as the insert-failed
                # path — _advance_failed_path's regex would mis-match a
                # marker'd path and reset the retry counter.
                work_path, _strip_ok = _strip_processing_marker(
                    work_path, log_path=log_path
                )
                if not _strip_ok:
                    try:
                        with log_path.open("a") as logf:
                            logf.write(
                                f"{datetime.now(timezone.utc).isoformat()} "
                                f"exception-skip {work_path.name}: "
                                f"strip-failed, leaving for next pass: {e!r}\n"
                            )
                    except (OSError, ValueError) as exc:
                        log.debug("exception_skip_log_write_failed: %s", exc)
                    counts["files_failed"] += 1
                    continue
                failed_path = _advance_failed_path(
                    work_path,
                    store,
                    first_error=file_first_error or repr(e),
                    log_path=log_path,
                )
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} failed "
                        f"{work_path.name}: {type(e).__name__}: {e}\n"
                    )
            except Exception as exc:  # noqa: BLE001 -- capture fail-safe
                log.debug("drain_exception_handler_failed: %s", exc)
            counts["files_failed"] += 1
    # One retention sweep per drain pass -- cheap O(N) where N <= keep_days +
    # grace in steady state. Fail-safe.
    try:
        from iai_mcp.memory_bank import prune_recent_windows

        prune_recent_windows()
    except Exception:  # noqa: BLE001 -- best-effort fail-safe boundary
        log.warning("bank-recent prune failed", exc_info=True)
    return counts


# ---------------------------------------------------------------------------
# Recovery drain for terminal.permanent-failed-*.jsonl files
# ---------------------------------------------------------------------------

_PERMANENT_FAILED_RE = re.compile(r"^\.permanent-failed-([^.]+)\.jsonl$")
_PERMANENT_FAILED_NAMED_RE = re.compile(r"^(.+)\.permanent-failed-([^.]+)\.jsonl$")


def _count_lines(fpath: Path) -> int:
    """Return the number of non-empty lines in *fpath*. Best-effort; returns 0 on error."""
    try:
        with fpath.open() as fh:
            return sum(1 for ln in fh if ln.strip())
    except OSError:
        return 0


def drain_permanent_failed_files(
    store: MemoryStore,
    *,
    deferred_dir: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Recover terminal ``.permanent-failed-*.jsonl`` files from *deferred_dir*.

    These files accumulated when earlier drain passes failed repeatedly. The
    original failure condition no longer applies once the underlying issue is
    resolved, so this function re-reads each file, applies the noise filter,
    and inserts genuine turns into the store.

    Safety invariants:
    - ``shutil.copy2`` to ``.quarantine/`` is performed BEFORE any rename or
      unlink, so the originals survive if the drain step crashes.
    - ``dry_run=True`` returns a file listing without mutating anything.
    - Only role:user turns reach ``capture_turn``; noise is dropped.

    Args:
        store: open MemoryStore instance (must be the same one the daemon uses
            when called via daemon RPC, or a fresh instance in direct-open mode).
        deferred_dir: path to the deferred-captures directory. Defaults to
            the directory computed from ``IAI_MCP_STORE`` env var or
            ``~/.iai-mcp/.deferred-captures``.
        dry_run: when True, list files + line counts without mutating anything.

    Returns:
        dict with keys:
        - ``dry_run`` (bool)
        - ``files`` (list[dict]) — always present: [{name, line_count}]
        - On real run also: ``inserted``, ``dropped``, ``files_recovered``,
          ``quarantine_dir``.
    """
    if deferred_dir is None:
        store_env = os.environ.get("IAI_MCP_STORE")
        if store_env:
            deferred_dir = Path(store_env).parent / ".deferred-captures"
        else:
            deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"

    if not deferred_dir.exists():
        if dry_run:
            return {"dry_run": True, "files": [], "count": 0}
        return {
            "dry_run": False,
            "files": [],
            "inserted": 0,
            "dropped": 0,
            "files_recovered": [],
            "quarantine_dir": str(deferred_dir / ".quarantine"),
        }

    # Discover all terminal files.
    terminal_files: list[Path] = []
    for entry in sorted(deferred_dir.iterdir()):
        if not entry.is_file():
            continue
        if ".permanent-failed-" in entry.name and entry.suffix == ".jsonl":
            terminal_files.append(entry)

    if dry_run:
        file_list = [
            {"name": f.name, "line_count": _count_lines(f)}
            for f in terminal_files
        ]
        return {"dry_run": True, "files": file_list, "count": len(file_list)}

    # Real run: quarantine copies first, then parse + ingest.
    quarantine_dir = deferred_dir / ".quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    inserted_total = 0
    dropped_total = 0
    files_recovered: list[str] = []
    file_list = []

    for fpath in terminal_files:
        # Preservation copy BEFORE any mutation.
        try:
            shutil.copy2(fpath, quarantine_dir / fpath.name)
        except Exception as exc:  # noqa: BLE001 -- fail-safe; log and continue
            log.warning("drain_permanent_failed_quarantine_failed %s: %s", fpath.name, exc)
            # Quarantine failed — do NOT proceed with this file; skip it to
            # preserve the zero-verbatim-loss guarantee.
            continue

        line_count = 0
        file_inserted = 0
        file_dropped = 0

        try:
            with fpath.open() as fh:
                lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

            if not lines:
                fpath.unlink(missing_ok=True)
                files_recovered.append(fpath.name)
                file_list.append({"name": fpath.name, "line_count": 0})
                continue

            line_count = len(lines)

            # Detect file shape: header-prefixed (deferred write_deferred_captures format)
            # vs. headerless transcript format.
            first_obj: dict | None = None
            try:
                first_obj = json.loads(lines[0])
            except (json.JSONDecodeError, ValueError):
                pass

            has_header = isinstance(first_obj, dict) and "version" in first_obj
            if has_header:
                # Deferred format: line 0 = header, lines 1+ = {text, cue, tier, role, ts}
                session_id = (first_obj or {}).get("session_id", "-")
                event_lines = lines[1:]
                for ln in event_lines:
                    try:
                        ev = json.loads(ln)
                    except (json.JSONDecodeError, ValueError):
                        file_dropped += 1
                        continue
                    text = (ev.get("text") or "").strip()
                    role = ev.get("role", "user")
                    if not text or _is_noise(text):
                        file_dropped += 1
                        continue
                    result = capture_turn(
                        store,
                        cue=ev.get("cue") or "recovered turn",
                        text=text,
                        tier=ev.get("tier", "episodic"),
                        session_id=session_id,
                        role=role,
                        ts=ev.get("ts"),
                        source_uuid=ev.get("source_uuid"),
                    )
                    if result.get("status") in ("inserted", "reinforced"):
                        file_inserted += 1
                    else:
                        file_dropped += 1
            else:
                # Transcript format: every line is a Claude Code transcript event
                # {type, message:{role, content}, session_id}.
                # Use _parse_transcript_line which applies the noise filter.
                raw_session_id = "-"
                for ln in lines:
                    try:
                        obj = json.loads(ln)
                        if isinstance(obj, dict) and "session_id" in obj:
                            raw_session_id = obj.get("session_id") or "-"
                    except (json.JSONDecodeError, ValueError):
                        pass
                    parsed = _parse_transcript_line(ln)
                    if parsed is None:
                        file_dropped += 1
                        continue
                    role, text, src_uuid, src_ts = parsed
                    result = capture_turn(
                        store,
                        cue="recovered turn",
                        text=text,
                        tier="episodic",
                        session_id=raw_session_id,
                        role=role,
                        ts=src_ts,
                        source_uuid=src_uuid,
                    )
                    if result.get("status") in ("inserted", "reinforced"):
                        file_inserted += 1
                    else:
                        file_dropped += 1

            # Drain succeeded — remove the original (quarantine copy is the backup).
            try:
                fpath.unlink()
            except OSError as exc:
                log.warning("drain_permanent_failed_unlink_failed %s: %s", fpath.name, exc)

            inserted_total += file_inserted
            dropped_total += file_dropped
            files_recovered.append(fpath.name)
            file_list.append({"name": fpath.name, "line_count": line_count})

        except Exception as exc:  # noqa: BLE001 -- per-file isolation
            log.warning("drain_permanent_failed_file_error %s: %s", fpath.name, exc)
            dropped_total += 1
            file_list.append({"name": fpath.name, "line_count": line_count})

    return {
        "dry_run": False,
        "files": file_list,
        "inserted": inserted_total,
        "dropped": dropped_total,
        "files_recovered": files_recovered,
        "quarantine_dir": str(quarantine_dir),
    }


# ---------------------------------------------------------------------------
# Offset-tracked partial drain for still-open live capture files
# ---------------------------------------------------------------------------


def drain_active_live_captures(
    store: MemoryStore,
    *,
    exclude_session_id: str,
) -> dict[str, int]:
    """Drain turns from OTHER sessions' still-open ``.live.jsonl`` files.

    Reads up to the current line count for each file whose name matches
    ``_LIVE_ACTIVE_RE`` (the active-writer marker pattern), skipping the file
    whose ``session_id`` matches ``exclude_session_id``. The session id is
    read from the JSONL header (line 1), not inferred from the filename, so
    it is immune to filename collisions.

    An offset sidecar at
    ``~/.iai-mcp/.capture-state/{session_id}.drain-offset`` records the number
    of **event lines** already drained so that repeated calls are idempotent
    and each run processes only genuinely new lines. The offset is written
    atomically via ``os.replace``.

    The live file is NEVER renamed, truncated, or unlinked — the other session
    must continue appending.

    Duplicate prevention
    --------------------
    Lines already processed carry a drain-offset so they are skipped on the
    next call. Lines processed here AND later re-ingested by the normal
    ``drain_deferred_captures`` pass (after the session ends and the file is
    renamed to ``.live-{epoch}.jsonl``) will match the exact idempotency
    tag (``idem:`` keyed on transcript-native uuid or (session, role, ts,
    text)) inside ``capture_turn`` and return ``status="reinforced"`` — no
    duplicate record is created.

    Returns dict with keys:
        files_drained, events_inserted, events_reinforced, events_skipped.
    """
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    state_dir = Path.home() / ".iai-mcp" / ".capture-state"
    counts: dict[str, int] = {
        "files_drained": 0,
        "events_inserted": 0,
        "events_reinforced": 0,
        "events_skipped": 0,
    }
    if not deferred_dir.exists():
        return counts

    for fpath in sorted(deferred_dir.iterdir()):
        if not fpath.is_file():
            continue
        if not _LIVE_ACTIVE_RE.search(fpath.name):
            continue
        try:
            with fpath.open() as fh:
                # Read all *complete* lines (ending in \n). An in-progress
                # write may have left a partial line at EOF; skip it so the
                # offset never points into incomplete JSON.
                raw_lines = fh.readlines()
        except OSError:
            continue
        if not raw_lines:
            continue

        # Strip the trailing partial line if it has no newline terminator.
        complete_lines = [ln for ln in raw_lines if ln.endswith("\n")]
        if not complete_lines:
            continue

        # Parse the header from line 0.
        try:
            header = json.loads(complete_lines[0])
        except (json.JSONDecodeError, ValueError):
            continue
        if header.get("version", 0) > 1:
            # Forward-compat guard: unknown format version, skip.
            continue

        file_session_id: str = header.get("session_id", "-")
        if file_session_id == exclude_session_id:
            # Never drain the refreshing session's own live file.
            continue

        # Load existing drain offset (number of event lines already processed).
        offset_path = state_dir / f"{file_session_id}.drain-offset"
        prev_offset: int = 0
        try:
            if offset_path.exists():
                prev_offset = int(offset_path.read_text().strip() or "0")
        except (ValueError, OSError):
            prev_offset = 0

        event_lines = complete_lines[1:]  # header is line 0
        new_lines = event_lines[prev_offset:]
        if not new_lines:
            # Nothing new beyond what was already drained.
            continue

        new_offset = prev_offset
        file_had_insert = False
        for ln in new_lines:
            try:
                ev = json.loads(ln)
            except (json.JSONDecodeError, ValueError):
                new_offset += 1
                counts["events_skipped"] += 1
                continue
            result = capture_turn(
                store,
                cue=ev.get("cue", ""),
                text=ev.get("text", ""),
                tier=ev.get("tier", "episodic"),
                session_id=file_session_id,
                role=ev.get("role", "user"),
                ts=ev.get("ts"),
                source_uuid=ev.get("source_uuid"),
            )
            status = result.get("status", "skipped")
            if status == "inserted":
                counts["events_inserted"] += 1
                file_had_insert = True
            elif status == "reinforced":
                counts["events_reinforced"] += 1
            else:
                counts["events_skipped"] += 1
            new_offset += 1

        # Flush the record write-buffer to disk BEFORE advancing the
        # offset sidecar. If the process exits between insert() and flush,
        # the next drain re-processes the same lines. For episodic
        # conversational turns, re-processing is idempotent via the exact
        # idempotency key (idem tag) derived from (session_id, role, ts, text);
        # an exact re-drain is detected at both dedup gates without creating
        # a duplicate row. If we advanced the offset first, a crash in that
        # window would leave the records silently lost with no recovery path.
        if file_had_insert:
            try:
                from iai_mcp.store import flush_record_buffer
                flush_record_buffer(store)
            except Exception as _flush_exc:  # noqa: BLE001 -- flush is best-effort
                log.warning("drain_active_flush_failed: %s", _flush_exc)

        # Write offset atomically so a concurrent crash leaves the previous
        # offset intact and the next call re-processes from the last safe point.
        state_dir.mkdir(parents=True, exist_ok=True)
        tmp_offset = offset_path.with_suffix(".drain-offset.tmp")
        try:
            tmp_offset.write_text(str(new_offset))
            os.replace(tmp_offset, offset_path)
        except OSError as exc:
            log.warning("drain_active_offset_write_failed: %s", exc)

        if file_had_insert:
            counts["files_drained"] += 1

    return counts
