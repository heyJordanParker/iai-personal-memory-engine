"""Events table interface.

Single source of runtime state. Every kind of event — S4 contradictions,
trajectory metrics, LLM health probes, schema induction runs, CLS consolidation
runs, migration traces, alerts — goes through write_event.

No .jsonl files. No per-alert or per-trajectory JSON files scattered on disk.
Everything persists in the `events` table.

CLI queries (iai-mcp health, iai-mcp trajectory) read via query_events.

events.data_json is AES-256-GCM encrypted at rest (some event payloads carry
user quotes / cues -- safest default). The event UUID is the associated data
binding. kind / severity / domain / ts / session_id stay plaintext so audit
queries (`iai-mcp health`, `iai-mcp trajectory`) can filter on them without
decrypting.

Event kinds (free-form strings, no taxonomy enum):
- `migration_v3_to_v4`.
- `sigma_observation`, `sigma_drift` (sigma-curve diagnostic).
- `retrieval_used`, `profile_updated`, `session_started`.
- `formality_score_weekly` — per-turn aggregate of user SURFACE formality.
- `camouflaging_detected` — over-formal trajectory detected over 5-point weekly window.
- `register_relaxed` — `camouflaging_relaxation` knob bumped; the system
  relaxes its OWN register (never the user's).
- `schema_reinforced` — emitted when `persist_schema` finds an existing
  schema for the candidate pattern and reinforces incoming
  `schema_instance_of` edges from new evidence onto the existing keeper
  instead of inserting a duplicate row. Payload:
    {schema_id: str, pattern: str, evidence_added: int, total_evidence: int}
  Source IDs: [keeper_schema_id, *new_evidence_ids[:5]] mirroring the
  existing `schema_induction_run` shape.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from iai_mcp.crypto import (
    decrypt_field,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.store import EVENTS_TABLE, MemoryStore

# ---------------------------------------------------------------------------
# Telemetry kind constants for lilli HDC layer risk-register events.
# ---------------------------------------------------------------------------
TELEMETRY_RANK_DEFICIENCY: str = "rank_deficiency_warning"
TELEMETRY_ROLE_SATURATION: str = "role_saturation_warning"
TELEMETRY_CODEC_MARKER_MISSING: str = "codec_marker_missing"
TELEMETRY_EMBED_NATIVE_FAILURE: str = "embed_native_failure"
# Layer-2 overlay freshness-fuse telemetry.  Emitted (buffered) whenever
# the O(1) freshness fuse trips the overlay consult to the Layer-1 bypass,
# making the accept-stable-global-bias-intra-day choice OBSERVABLE.
# Payload: {age_ms: int, pending_rebuild: bool (always False here)}.
TELEMETRY_FRESHNESS_FUSE_TRIPPED: str = "freshness_fuse_tripped"
# Recall-path observability.  Emitted best-effort on every awake recall so the
# real-world recall tails are measurable WITHOUT putting any user cue text on
# the wire.  Payload carries cue-DERIVED metrics only:
#   {source: "semantic-inprocess" | "daemon" | "recency-degrade",
#    construct_ms?: float, encode_ms?: float, reason?: str (scrubbed token)}
# `source` lets a downstream count query DERIVE the fallback_rate (degrade /
# total) at query time — no separate stored metric.  NEVER carries the raw cue
# or any cue-derived substring; error strings are sanitized to a fixed token.
TELEMETRY_RECALL_SOURCE: str = "recall_source"
# Embedder construct timing.  Reserved kind for a future resident embed service
# decision (the deferred upgrade behind the construct seam): if construct cost
# ever needs to be tracked separately from recall_source it lands here.
# Payload: {construct_ms: float, encode_ms?: float, process: str}.
TELEMETRY_EMBED_CONSTRUCT: str = "embed_construct"

# ---------------------------------------------------------------------------
# Daemon self-watchdog event kinds (liveness + memory-pressure recovery).
# ---------------------------------------------------------------------------
# Emitted by the daemon's self-watchdog thread when it performs a controlled
# self-recovery, or when its bounded-attempts circuit-breaker trips.
#
# IMPORTANT — the two *_kill kinds are NOT emitted via write_event on the kill
# path.  The kill path is LOCK-FREE: it writes a raw os.write breadcrumb (no
# HippoTable.add, no connection lock) carrying the kind token, then SIGKILLs the
# process UNCONDITIONALLY.  Routing the kill breadcrumb through write_event would
# acquire the Hippo connection lock, which the consolidation worker can hold for
# long stretches — a self-kill that first blocked on that lock would never run,
# defeating the watchdog in exactly the jetsam-during-consolidation case it
# exists to handle.  The kind constants are defined here for the breadcrumb token
# and for any post-mortem log parser.
#
# The needs-operator kind (no kill) IS emitted via write_event normally — a
# blocked emit there only delays a loud event; it is not a safety hazard.
DAEMON_WEDGE_KILL: str = "daemon_wedge_kill"
DAEMON_MEMORY_PRESSURE_KILL: str = "daemon_memory_pressure_kill"
DAEMON_WATCHDOG_NEEDS_OPERATOR: str = "daemon_watchdog_needs_operator"

_event_buffer: dict[int, list[dict]] = {}

# Time-threshold tracking for periodic flush from the daemon tick. Paired with
# _event_buffer; both keyed by id(store). Updated only on a SUCCESSFUL flush
# (failure path leaves the timestamp stale so the next tick retries sooner).
_last_flush_at: dict[int, datetime] = {}

# Shared reentrant lock that serializes all module-level write-buffer state
# mutations: the buffer/flush-timestamp pops inside the flush helpers, AND the
# drain-purge-release sequence inside MemoryStore.close(). RLock (not plain
# Lock) so MemoryStore.close() can hold the lock across the drain step while
# each flush helper re-acquires the same lock in the same thread without
# deadlocking. Race target: daemon tick `asyncio.to_thread(flush_*, store)`
# concurrent with explicit `store.close()` from another thread.
_BUFFER_LOCK = threading.RLock()


def flush_event_buffer(store: MemoryStore) -> int:
    """Flush buffered events to the store in one batch write.

    Returns the number of events flushed. Safe to call when buffer is empty.

    Runs entirely under the shared `_BUFFER_LOCK` so a concurrent
    `MemoryStore.close()` from another thread cannot interleave a buffer pop
    between this body's own pop and the store write.
    """
    with _BUFFER_LOCK:
        store_id = id(store)
        pending = _event_buffer.pop(store_id, [])
        if not pending:
            return 0
        try:
            store.db.open_table(EVENTS_TABLE).add(pending)
            # Record success timestamp for the periodic-tick gate.
            _last_flush_at[store_id] = datetime.now(timezone.utc)
        except (OSError, RuntimeError, ValueError) as exc:
            logging.getLogger(__name__).warning("flush_event_buffer_failed", extra={"n": len(pending), "err": str(exc)[:120]})
            # NOTE: do NOT update _last_flush_at on failure; next tick will retry sooner.
        return len(pending)


def should_flush(store_id: int, max_size: int | None = None) -> bool:
    """Return True iff the in-memory buffer has reached the size threshold.

    Parameters
    ----------
    store_id:
        The ``id(store)`` of the MemoryStore whose buffer to inspect.
    max_size:
        Override the env-configured default of 100. When None, resolves from
        ``IAI_MCP_EVENT_BUFFER_MAX`` env var with fallback 100.
    """
    if max_size is None:
        try:
            max_size = int(os.environ.get("IAI_MCP_EVENT_BUFFER_MAX", "100"))
        except ValueError:
            max_size = 100
    return len(_event_buffer.get(store_id, [])) >= max_size


def should_flush_by_time(
    store_id: int,
    last_flush_at: datetime | None,
    max_age_sec: float = 5.0,
) -> bool:
    """Return True iff the buffer is non-empty AND aged past ``max_age_sec``.

    None ``last_flush_at`` means "never flushed": treat as immediately due
    iff the buffer is non-empty.
    """
    if not _event_buffer.get(store_id):
        return False
    if last_flush_at is None:
        return True
    return (datetime.now(timezone.utc) - last_flush_at).total_seconds() >= max_age_sec


def write_event(
    store: MemoryStore,
    kind: str,
    data: dict[str, Any],
    *,
    severity: str | None = None,
    domain: str | None = None,
    session_id: str = "-",
    source_ids: list[UUID] | None = None,
    buffered: bool = False,
) -> UUID:
    """Persist a single event to the events table.

    Parameters
    ----------
    store:
        Open MemoryStore instance.
    kind:
        Logical event kind (e.g. "s4_contradiction", "trajectory_metric",
        "llm_health", "migration_v1_to_v2"). Free-form string; downstream
        consumers filter on it.
    data:
        JSON-serialisable kind-specific payload. Encoded to data_json.
    severity:
        Optional alert severity ("info" | "warning" | "critical"). Stored
        as empty string for non-alert events.
    domain:
        Optional monotropic-domain tag. Stored as empty string when absent.
    session_id:
        Session identifier; defaults to "-" when no session is active.
    source_ids:
        Optional list of MemoryRecord UUIDs that triggered this event.

    Returns the newly-minted event UUID.
    """
    event_id = uuid4()
    # Encrypt data_json with AD = event UUID bytes. kind / severity /
    # domain / ts / session_id stay plaintext for filter queries.
    data_plain = json.dumps(data)
    ad = str(event_id).encode("ascii")
    data_ct = encrypt_field(data_plain, store._key(), associated_data=ad)
    row = {
        "id": str(event_id),
        "kind": kind,
        "severity": severity or "",
        "domain": domain or "",
        "ts": datetime.now(timezone.utc),
        "data_json": data_ct,
        "session_id": session_id,
        "source_ids_json": json.dumps([str(x) for x in (source_ids or [])]),
    }
    if buffered:
        _event_buffer.setdefault(id(store), []).append(row)
        return event_id
    store.db.open_table(EVENTS_TABLE).add([row])

    # Post-emit STC trigger: if this event kind is configured as a
    # STRONG_EVENT, fan out to the peri-event buffer so eligible semantic
    # turns in the window get upgraded to episodic. Failure here MUST NOT
    # raise from write_event -- STC is observability / consolidation, not
    # a critical write path.
    try:
        from iai_mcp.daemon_config import _load_stc_config
        from iai_mcp.peri_event_buffer import get_buffer

        cfg = _load_stc_config()
        if kind in cfg.strong_event_types:
            buf = get_buffer()
            if buf is not None:
                buf.trigger_stc(store, kind)
    except Exception as exc:  # noqa: BLE001 -- STC trigger must never crash write_event
        logging.getLogger(__name__).warning(
            "stc_trigger_failed",
            extra={
                "kind": kind,
                "err_type": type(exc).__name__,
                "err": str(exc)[:120],
            },
        )
    return event_id


def emit_best_effort(
    store: "MemoryStore | None",
    kind: str,
    data: dict[str, Any],
    *,
    severity: str | None = None,
    session_id: str = "-",
) -> None:
    """Emit a telemetry event without ever breaking the caller.

    Observability-only wrapper for the recall path: a telemetry failure (or a
    missing store handle on the deepest degrade) must NEVER propagate to the
    caller and must never change the recall return value or timing materially.

    Behaviour:
      - store present  -> buffered write_event (the daemon flush picks it up).
      - store is None  -> a stdlib debug log line carrying the same metrics
                          (the deepest degrade may have no open store).
      - any exception  -> swallowed; the caller proceeds unaffected.

    Callers MUST pass cue-DERIVED metrics only (construct_ms / encode_ms /
    source / a scrubbed reason token) — never the raw cue or a cue-derived
    substring.  This wrapper does not sanitize; scrubbing is the caller's job.
    """
    try:
        if store is None:
            logging.getLogger(__name__).debug("telemetry %s %s", kind, data)
            return
        write_event(store, kind, data, severity=severity,
                    session_id=session_id, buffered=True)
    except Exception:  # noqa: BLE001 -- telemetry must never break the caller
        try:
            logging.getLogger(__name__).debug("telemetry_emit_failed %s", kind)
        except Exception:  # noqa: BLE001
            pass


def query_events(
    store: MemoryStore,
    kind: str | None = None,
    since: datetime | None = None,
    severity: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Query events matching the given filters, newest first.

    Parameters
    ----------
    store:
        Open MemoryStore instance.
    kind:
        Filter by event kind. None returns all kinds.
    since:
        Only return events with ts >= since. Naive datetimes are treated as UTC.
    severity:
        Exact-match filter on severity field.
    limit:
        Maximum rows returned (default 100). Caller can pass e.g. 1 to get
        only the most recent event of a given kind (iai-mcp health).

    Returns a list of dicts with keys: id, kind, severity, domain, ts, data,
    session_id, source_ids. data and source_ids are decoded from JSON.
    """
    tbl = store.db.open_table(EVENTS_TABLE)
    df = tbl.to_pandas()
    if df.empty:
        return []
    # Hippo stores events.ts as ISO TEXT; coerce to tz-aware pandas Timestamps
    # so downstream comparisons and consumer callers (CLI audit formatter,
    # doctor freshness checks) receive datetime objects rather than strings.
    if "ts" in df.columns:
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if kind is not None:
        df = df[df["kind"] == kind]
    if severity is not None:
        df = df[df["severity"] == severity]
    if since is not None:
        # Ensure tz-aware comparison
        since_cmp = since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
        # Pandas Timestamp compares naturally with tz-aware datetimes
        df = df[df["ts"] >= since_cmp]
    if df.empty:
        return []
    df = df.sort_values("ts", ascending=False).head(limit)
    out: list[dict] = []
    for _, row in df.iterrows():
        # Decrypt data_json when it carries the iai:enc:v1: prefix.
        # Legacy plaintext rows stay plaintext; migration rewrites them lazily.
        raw_data = row["data_json"] or "{}"
        if is_encrypted(raw_data):
            ad = str(row["id"]).encode("ascii")
            try:
                raw_data = decrypt_field(raw_data, store._key(), associated_data=ad)
            except (OSError, ValueError, RuntimeError) as exc:
                # Diagnostic semantics: a corrupt event row should not
                # fail the entire query. Return empty payload + mark in meta.
                logging.getLogger(__name__).debug("event_decrypt_failed", extra={"id": row["id"], "err": str(exc)[:80]})
                raw_data = "{}"
        try:
            data = json.loads(raw_data)
        except (TypeError, json.JSONDecodeError):
            data = {}
        try:
            source_ids = json.loads(row["source_ids_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            source_ids = []
        # Convert pandas Timestamp (or NaT) to pydatetime for CLI consumers
        ts_value = row["ts"]
        if hasattr(ts_value, "to_pydatetime"):
            ts_value = ts_value.to_pydatetime()
        out.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "severity": row["severity"] or None,
                "domain": row["domain"] or None,
                "ts": ts_value,
                "data": data,
                "session_id": row["session_id"],
                "source_ids": source_ids,
            }
        )
    return out
