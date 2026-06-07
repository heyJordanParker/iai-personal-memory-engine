"""SQLite+hnswlib-backed persistent memory store.

Tables:
- `records`: MemoryRecord rows (one per memory).
- `edges`: (src, dst, edge_type, weight, updated_at) -- Hebbian + contradicts edges.
- `events`: all runtime state (S4 contradictions, trajectory metrics, alerts,
  llm_health, schema_induction_run, cls_consolidation_run, etc.).
- `budget_ledger`: per-day USD spend by kind (BudgetLedger).
- `ratelimit_ledger`: 429 history for 15-min cooldown (RateLimitLedger).

Every runtime event lives in the database; no scattered .jsonl or .json files.

Embedding dimension defaults to `bge-small-en-v1.5` (384d). The records schema
reads the configured dimension from `iai_mcp.embed.DEFAULT_DIM` at first table
creation. Stores created with 1024d embeddings stay readable via
`embedder_for_store(store)` until the user re-embeds them down to 384d.

Storage root defaults to `~/.iai-mcp`, overridable via IAI_MCP_STORE env var or
the `path` constructor argument.

Encryption-at-rest:
- literal_surface / provenance_json / profile_modulation_gain_json on records
  table are AES-256-GCM encrypted with a key sourced from the OS keychain.
- events.data_json on events table is also encrypted.
- Embeddings / tags / language / schema_version / timestamps stay plaintext.
- Encryption is transparent to callers: store.insert() encrypts and
  store.get() decrypts; no change to the MemoryRecord dataclass.
- AD = record UUID bytes, binding ciphertext to its row so cut-and-paste
  attacks fail on decrypt.
"""
from __future__ import annotations

import asyncio
import base64
import enum
import functools
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable, Union
from uuid import UUID

import logging

from iai_mcp.hippo import _REAL_IAI_ROOT, AccessMode, HippoDB, HippoIntegrityError

# Retained for backward compatibility: doctor.py imports this symbol to check
# CPU AVX2 support. Always True on the Hippo backend (no AVX2 dependency).
CPU_HAS_AVX2: bool = True

import pyarrow as pa

# Cached AESGCM cipher per store; reuse safe per
# https://cryptography.io/en/latest/hazmat/primitives/aead/ — single AESGCM
# can be reused across operations as long as nonces are unique. We use random
# per-record nonces in encrypt_field, so cache reuse is correct.
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from iai_mcp.crypto import (
    CIPHERTEXT_PREFIX,
    NONCE_BYTES,
    CryptoKey,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.exceptions import (
    StoreCorruptionError,
    StoreInsertError,
    StoreQueryError,
    StoreSchemaError,
)
from iai_mcp.types import (
    DEFAULT_EMBED_DIM,
    EMBED_DIM,
    HV_TIER_ENUM,
    SCHEMA_VERSION_CURRENT,
    MemoryRecord,
    TIER_ENUM,
)

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_PATH = Path.home() / ".iai-mcp"

# Core tables
RECORDS_TABLE = "records"
EDGES_TABLE = "edges"

# Runtime-event tables
EVENTS_TABLE = "events"
BUDGET_TABLE = "budget_ledger"
RATELIMIT_TABLE = "ratelimit_ledger"

# STC tier-ordering single source of truth. upgrade_tier compares
# _STC_TIER_ORDER[new] > _STC_TIER_ORDER[current] to refuse any non-upward
# move (semantic -> episodic -> procedural only).
_STC_TIER_ORDER: dict[str, int] = {"semantic": 0, "episodic": 1, "procedural": 2}

# Edge type enum.
# consolidated_from   -- consolidation sleep cycle: semantic <- source episodes
# schema_instance_of  -- schema induction: episode -> schema hub
# temporal_next       -- record insert (same session) episode chain
# invariant_anchor    -- S5 kernel stable-fact permanent hub
# curiosity_bridge    -- question -> triggering records
# profile_modulates   -- profile knob runtime gain
# hebbian_structure   -- TEM factorization LTP on structure edges
# pattern_separation_seed  -- pre-insert link layer; FSRS-decays like hebbian (eligible in sleep._decay_edges)
# hebbian_cluster_replay  -- cluster-replay temporal Hebbian seed; FSRS-decays like hebbian (eligible in sleep._decay_edges)
EDGE_TYPES: frozenset[str] = frozenset({
    "hebbian",
    "contradicts",
    "consolidated_from",
    "schema_instance_of",
    "temporal_next",
    "invariant_anchor",
    "curiosity_bridge",
    "profile_modulates",
    "hebbian_structure",
    "pattern_separation_seed",
    "hebbian_cluster_replay",
})


# GateAction: pattern_separation_gate's return-action enum.
# SKIP   -> caller short-circuits to reinforce_record + record.id mutation
# INSERT -> caller proceeds with the store add + post-add edge seeding
class GateAction(enum.Enum):
    SKIP = "skip"
    INSERT = "insert"


# GatePayload: SKIP carries the existing-record UUID (caller-transparent
# merging); INSERT carries the list of (target_uuid, cosine) pairs
# to seed pattern_separation_seed edges to.
GatePayload = Union[UUID, list[tuple[UUID, float]]]


# RFC-4122 canonical UUID regex. Accept both str and UUID inputs; reject anything
# that could embed a SQL-like escape. Hoisted to module scope so the pattern
# object is compiled once.
_UUID_STR_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _uuid_literal(value: UUID | str) -> str:
    """Return a WHERE-safe UUID literal.

    Normalises any UUID (object or canonical str) into its canonical lowercase
    form and rejects anything that does not match the RFC-4122 shape, so the
    f-string cannot carry injection content.
    """
    s = str(value).lower()
    if not _UUID_STR_RE.match(s):
        raise ValueError(f"not a canonical UUID: {value!r}")
    return s


# Local dim table so store creation does NOT pull in iai_mcp.embed
# (and by extension the Rust native extension boot cost). The
def _resolve_embed_dim() -> int:
    """Pick the embedding dimension for the records table on first creation.

    Priority:
    1. Environment override IAI_MCP_EMBED_DIM (test hermeticity / migration dry-runs)
    2. DEFAULT_EMBED_DIM (384 -- the English bge-small-en-v1.5 default)
    """
    env_dim = os.environ.get("IAI_MCP_EMBED_DIM")
    if env_dim:
        try:
            return int(env_dim)
        except ValueError:
            pass
    return DEFAULT_EMBED_DIM


class _PendingTurn:
    """Lightweight wrapper for a pending (not-yet-drained) live-capture event.

    Mimics the subset of MemoryRecord attributes consumed by recent_user_turns
    callers (episodes_recent handler + _recent_thread_segment) without building
    a full MemoryRecord (which requires an embedding).
    """

    __slots__ = ("_text", "_session_id", "_ts", "_idem_tag", "_source_uuid", "_role")

    def __init__(
        self,
        *,
        text: str,
        session_id: str,
        ts: "datetime",
        idem_tag: str,
        source_uuid: "str | None",
        role: str = "user",
    ) -> None:
        self._text = text
        self._session_id = session_id
        self._ts = ts
        self._idem_tag = idem_tag
        self._source_uuid = source_uuid
        self._role = role

    # MemoryRecord-compatible attributes consumed by callers.
    @property
    def id(self):
        return None  # pending turn has no store UUID

    @property
    def tier(self) -> str:
        return "episodic"

    @property
    def literal_surface(self) -> str:
        return self._text

    @property
    def tags(self) -> list:
        return [f"role:{self._role}", self._idem_tag]

    @property
    def provenance(self) -> list:
        prov: dict = {"session_id": self._session_id, "role": self._role}
        if self._source_uuid is not None:
            prov["source_uuid"] = self._source_uuid
        return [prov]

    @property
    def created_at(self):
        return self._ts

    # Extra: idem_tag and source_uuid for the episodes_recent record_id builder.
    @property
    def _pending_idem_tag(self) -> str:
        return self._idem_tag

    @property
    def _pending_source_uuid(self) -> "str | None":
        return self._source_uuid


class MemoryStore:
    """SQLite+hnswlib-backed persistent memory store.

    Sync writes, single-user, local filesystem. Supports
    records/edges/events/budget_ledger/ratelimit_ledger tables and v2
    MemoryRecord fields (language, s5_trust_score, profile_modulation_gain,
    schema_version). Rows with schema_version=1 remain readable.

    Thread-safety: HippoDB._hnsw_lock (threading.RLock) serialises all mutations
    across the foreground daemon coroutines and the background async-write queue
    thread. The SQLite connection is opened with check_same_thread=False inside
    HippoDB.__init__; the RLock makes concurrent access safe without a per-call
    re-connect.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        user_id: str = "default",
        read_consistency_interval: timedelta | None = None,
        *,
        access_mode: AccessMode = AccessMode.EXCLUSIVE,
        read_only: bool = False,
    ) -> None:
        """Open (or initialise) a SQLite+hnswlib-backed store.

        ``read_consistency_interval`` is accepted for API stability but
        ignored — the SQLite WAL mode provides consistent reads without
        per-call version checks.

        Thread-safety: HippoDB acquires an fcntl LOCK_EX on
        ``<root>/hippo/.lock`` at open so that dual-process open attempts
        raise HippoLockHeldError rather than corrupting the database.
        Within a single process, HippoDB._hnsw_lock (threading.RLock)
        serialises every SQLite + hnswlib mutation so the background
        async-write queue thread and foreground daemon coroutines share
        one sqlite3.Connection safely.
        """
        env_path = os.environ.get("IAI_MCP_STORE")
        if path is not None:
            self.root = Path(path)
        elif env_path:
            self.root = Path(env_path)
        else:
            self.root = Path(DEFAULT_STORAGE_PATH)
        # Test-only backstop: under a test run, refuse a resolution to the real
        # operator store before any filesystem touch. Mirrors the resolver-level
        # guard so the primary store entry point is closed by construction even
        # if a test fixture regresses. Never triggers in normal operation.
        if os.environ.get("PYTEST_CURRENT_TEST") and self.root == _REAL_IAI_ROOT:
            raise RuntimeError(
                "hermeticity guard: store-root resolved to the real home store "
                "during a test run; tests must use a tmp path (autouse redirect "
                "fixture). This guard never fires in normal operation."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self._read_consistency_interval: timedelta | None = read_consistency_interval
        # Wire the encryption key provider so HippoDB encrypts fields on write
        # and decrypts on read.  The lambda is evaluated lazily: _key() only
        # calls CryptoKey.get_or_create() on the first actual encrypt/decrypt,
        # not on store open, so test suites that create many MemoryStore
        # instances don't each hit the (mocked or real) keyring backend.
        self._user_id: str = user_id
        self._crypto_key_wrapper: CryptoKey = CryptoKey(user_id=user_id, store_root=self.root)
        self._crypto_key: bytes | None = None
        self.db: HippoDB = HippoDB(
            self.root,
            crypto_key_provider=self._key,
            access_mode=access_mode,
            read_only=read_only,
        )
        # Resolve the embedding dimension once so insert guard agrees with the
        # actual table schema.  _ensure_tables() may update _embed_dim when
        # the store was created with a non-default embedder.
        self._embed_dim: int = _resolve_embed_dim()
        self._ensure_tables()
        # Optional store -> runtime-graph sync callback. Set by
        # retrieve.build_runtime_graph via register_graph_sync_hook(). Every
        # insert / update / delete fires this hook inside try/except so the
        # store write remains authoritative — a buggy or absent hook can
        # never break the store.
        self._graph_sync_hook: Callable[[str, "MemoryRecord"], None] | None = None
        # Optional async write queue. When live, insert() routes
        # through it; when None, insert() uses the legacy sync path. The
        # event loop runs on a dedicated background thread so sync callers
        # can dispatch via asyncio.run_coroutine_threadsafe.
        self._write_queue = None  # type: ignore[assignment]
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None
        self._async_conn = None  # HippoTable sync adapter (no async connection needed)
        # Optional async provenance queue. When set, writes routed through
        # queue_provenance_batch go off the recall critical path; when None
        # we fall back to the sync append_provenance_batch call (back-compat).
        self._provenance_queue = None  # type: ignore[assignment]

    def close(self) -> None:
        """Drain buffered writes, purge id-keyed buffer state, then release
        the fcntl lock and close SQLite.

        Three-step protocol:

        1. DRAIN  -- best-effort flush_event_buffer / flush_record_buffer /
           flush_edge_buffer so any rows still in the in-memory buffers land
           on disk under THIS store's key+AAD (popping before flush would
           lose them; flushing after another store reused id(self) would
           write under the wrong key).
        2. PURGE  -- pop id(self) from all 6 module-level dicts so the next
           store to reuse the same id() value (after GC) starts with a
           clean buffer view, with no ghost ciphertext from this store.
        3. RELEASE -- close the underlying SQLite/Hippo handle and clear
           self.db so subsequent close() calls are no-ops (idempotent).

        Required by callers that open a MemoryStore for fixture setup and
        then re-open the same path through a different code path (e.g. CLI
        subcommands invoked via `cli_main` in unit tests). The fcntl lock
        is non-deterministic on garbage-collected close so explicit close
        is needed for predictable test cleanup.

        See also: the shared `_BUFFER_LOCK` (defined in events.py, imported
        function-locally here to dodge the events->store module-load cycle)
        serializes the entire drain/purge/release sequence against the
        daemon tick's concurrent `asyncio.to_thread(flush_*, store)` calls.
        """
        if self.db is None:
            return

        # Function-local imports: events.py imports MemoryStore at module
        # top, so store.py cannot import from events.py at module level
        # without breaking the import order.
        from iai_mcp.events import (
            _BUFFER_LOCK,
            _event_buffer,
            _last_flush_at,
            flush_event_buffer,
        )

        with _BUFFER_LOCK:
            # 1. DRAIN -- best-effort. close() must not block process exit
            # on a flush failure; each helper catches OSError/RuntimeError/
            # ValueError internally and logs at WARNING. We still wrap each
            # call in try/except as a belt-and-braces defence against any
            # unexpected exception bubbling out (e.g. an attribute error
            # during shutdown when self.db is already half-torn-down).
            _log = logging.getLogger(__name__)
            try:
                flush_event_buffer(self)
            except Exception as exc:  # noqa: BLE001 -- drain MUST NOT block close()
                _log.warning(
                    "memorystore_close_drain_failed",
                    extra={
                        "flush": "flush_event_buffer",
                        "err_type": type(exc).__name__,
                        "err": str(exc)[:120],
                    },
                )
            try:
                flush_record_buffer(self)
            except Exception as exc:  # noqa: BLE001 -- drain MUST NOT block close()
                _log.warning(
                    "memorystore_close_drain_failed",
                    extra={
                        "flush": "flush_record_buffer",
                        "err_type": type(exc).__name__,
                        "err": str(exc)[:120],
                    },
                )
            try:
                flush_edge_buffer(self)
            except Exception as exc:  # noqa: BLE001 -- drain MUST NOT block close()
                _log.warning(
                    "memorystore_close_drain_failed",
                    extra={
                        "flush": "flush_edge_buffer",
                        "err_type": type(exc).__name__,
                        "err": str(exc)[:120],
                    },
                )

            # 2. PURGE -- pop id(self) from all 8 dicts so GC-recycled
            # id() values do not inherit stale buffer rows. Idempotent
            # via default=None; safe even if drain already popped.
            #
            # Six write-side dicts (events / records / edges, each with
            # its _last_flush_at companion) live in events.py + store.py.
            # Two read-side dicts (_tv_cache + _tv_cache_dirty) live in
            # retrieve.py and cache temporal-validity decisions per store;
            # ghosting them across id() reuse poisons recall ranking for
            # downstream recall_temporal_validity / sleep / pipeline tests.
            # Function-local import dodges the same retrieve->store cycle
            # that the write-side dicts dodge for events.py.
            _event_buffer.pop(id(self), None)
            _last_flush_at.pop(id(self), None)
            _record_buffer.pop(id(self), None)
            _record_last_flush_at.pop(id(self), None)
            _edge_buffer.pop(id(self), None)
            _edge_last_flush_at.pop(id(self), None)
            try:
                from iai_mcp.retrieve import _tv_cache, _tv_cache_dirty
                _tv_cache.pop(id(self), None)
                _tv_cache_dirty.pop(id(self), None)
            except ImportError:
                # retrieve.py optional in some import-pruning scenarios;
                # purge is best-effort under the same drain-MUST-NOT-block
                # rule that governs the flush helpers above.
                pass

            # 3. RELEASE -- close the SQLite handle + clear the attribute
            # so subsequent close() calls early-return at the guard above.
            self.db.close()
            self.db = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ schema

    def _ensure_tables(self) -> None:
        """Verify the store is open and read back the actual embedding dimension.

        HippoDB._ensure_tables() creates all five tables (records, edges, events,
        budget_ledger, ratelimit_ledger) plus _hippo_meta on first open, with the
        full production schema including every column added across all migrations.
        No inline ALTER TABLE work is needed here.

        We do read back _embed_dim from the existing records table so that stores
        with a non-default dimension (e.g. legacy 1024d) keep the right dimension
        through the re-embed migration lifecycle.
        """
        try:
            tbl = self.db.open_table(RECORDS_TABLE)
            arrow_schema = tbl.schema
            emb_field = arrow_schema.field("embedding")
            # pa.list_(..., N) fixed-size list -> .type.list_size
            actual_dim = getattr(emb_field.type, "list_size", None)
            if actual_dim and int(actual_dim) > 0:
                self._embed_dim = int(actual_dim)
        except (OSError, KeyError, ValueError, AttributeError) as exc:
            logger.debug("records table schema introspection skipped: %s", exc)

    def _table_names(self) -> list[str]:
        """Return the list of table names from the underlying store.

        HippoDB.list_tables() returns a HippoTableList whose `.tables` attr
        is a list[str]; the fallback branch handles any future shim change.
        """
        result = self.db.list_tables()
        if hasattr(result, "tables"):
            return list(result.tables)
        return list(result)

    @property
    def embed_dim(self) -> int:
        """Actual embedding dimension in the records table."""
        return self._embed_dim

    @property
    def user_id(self) -> str:
        """User id that scopes the encryption key (multi-tenant ready)."""
        return self._user_id

    # -------------------------------------------------------------- encryption

    def _key(self) -> bytes:
        """Lazy-load the encryption key from keyring."""
        if self._crypto_key is None:
            self._crypto_key = self._crypto_key_wrapper.get_or_create()
        return self._crypto_key

    def _ad(self, record_id: UUID | str) -> bytes:
        """Associated data for a record's encrypted fields: canonical UUID str bytes.

        Binds ciphertext to its row. An attacker who swaps ciphertext between
        rows (copy row A's literal_surface into row B on disk) will fail to
        decrypt because AD(B) != AD(A) -- InvalidTag.
        """
        return _uuid_literal(record_id).encode("ascii")

    def _encrypt_for_record(self, record_id: UUID, value: str) -> str:
        """Encrypt a per-record sensitive field; idempotent on already-encrypted input."""
        if is_encrypted(value):
            return value
        return encrypt_field(value, self._key(), associated_data=self._ad(record_id))

    @functools.cached_property
    def _cached_aesgcm(self) -> AESGCM:
        """One AESGCM cipher per store lifetime.

        Materialised lazily on first access. Reused across all
        :meth:`_decrypt_for_record` calls — saves the per-call ``AESGCM(key)``
        construction cost.

        Cache invalidation: if ``self._key()`` rotates, callers must invoke
        :meth:`_invalidate_aesgcm_cache`.
        """
        return AESGCM(self._key())

    def _invalidate_aesgcm_cache(self) -> None:
        """Drop the cached AESGCM. Next access re-materialises against current key.

        Reserved for key-rotation events.
        """
        self.__dict__.pop("_cached_aesgcm", None)

    def _decrypt_for_record(self, record_id: UUID, value: str) -> str:
        """Decrypt a per-record sensitive field; pass through plaintext unchanged.

        Back-compat: pre-encryption rows are plaintext -- return them as-is so
        readers see the same data until migration re-encrypts them.

        Uses :attr:`_cached_aesgcm` instead of constructing a fresh
        ``AESGCM(key)`` on every call. ``crypto.decrypt_field`` is
        intentionally NOT modified -- keep crypto.py decoupled and stateless
        for callers that pass key bytes directly.
        """
        if not is_encrypted(value):
            return value
        if not value.startswith(CIPHERTEXT_PREFIX):
            # Defensive: is_encrypted() should already have guaranteed this.
            raise ValueError("field is not iai:enc:v1:-prefixed ciphertext")
        payload_b64 = value[len(CIPHERTEXT_PREFIX):]
        payload = base64.b64decode(payload_b64)
        if len(payload) < NONCE_BYTES + 16:  # nonce + minimum GCM tag
            raise ValueError("ciphertext payload too short")
        nonce = payload[:NONCE_BYTES]
        ct_with_tag = payload[NONCE_BYTES:]
        associated_data = self._ad(record_id)
        plaintext_bytes = self._cached_aesgcm.decrypt(
            nonce, ct_with_tag, associated_data or None
        )
        return plaintext_bytes.decode("utf-8")

    # -------------------------------------------------------------------- I/O

    # ------------------------------------------------------- graph sync hook

    def register_graph_sync_hook(
        self, hook: Callable[[str, MemoryRecord], None] | None
    ) -> None:
        """Register a callback that mirrors store writes to the runtime graph.

        The hook is called with ``(op, record)`` after every successful
        store write where ``op`` is one of ``"insert" | "update" |
        "delete"``. Hook exceptions are caught and logged to stderr as
        a structured JSON ``graph_sync_failed`` event; the store write
        is authoritative and never rolled back on hook failure.

        Idempotent — passing a new callable replaces the previous hook;
        passing ``None`` unregisters it.
        """
        self._graph_sync_hook = hook

    def _fire_graph_sync_hook(self, op: str, record: MemoryRecord) -> None:
        """Dispatch the (op, record) event. Failures are swallowed +
        logged. Never raises."""
        hook = self._graph_sync_hook
        if hook is None:
            return
        try:
            hook(op, record)
        except Exception as exc:  # noqa: BLE001 -- hook isolation, daemon stability
            logger.warning("graph_sync_hook failed op=%s: %s", op, exc, exc_info=True)
            try:
                sys.stderr.write(
                    json.dumps({
                        "event": "graph_sync_failed",
                        "op": op,
                        "record_id": str(getattr(record, "id", "")),
                        "error": str(exc),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    })
                    + "\n"
                )
            except Exception:  # noqa: BLE001 -- logger-of-logger recursion guard
                pass

    def insert(self, record: MemoryRecord) -> None:
        """Append a record verbatim with no rewrite at write time.

        Sensitive fields are encrypted in _to_row before the row hits
        the store. Decryption happens in get()/_from_row for callers.

        If record.structure_hv is empty bytes (the pre-migration sentinel),
        computes it via tem.bind_structure(record) before persisting so the
        memory_recall_structural branch can rank it without re-derivation.

        Fires the optional ``_graph_sync_hook`` after the store write
        so the runtime graph stays in sync. Hook failures are logged, never
        raised.

        If ``enable_async_writes()`` has been called, the insert is routed
        through the coalescing AsyncWriteQueue and this call blocks until
        the batch containing ``record`` has flushed to disk.

        pattern_separation_gate runs BEFORE the store write. SKIP
        mutates record.id to the existing-record uuid (caller-transparent
        merging), reinforces the existing record, emits a single
        pattern_separation_pass event, and returns without adding a new
        row. INSERT proceeds with the regular write path; INSERT-non-dry-run
        additionally seeds pattern_separation_seed edges to every
        link-eligible target via boost_edges. Exactly ONE
        pattern_separation_pass event is emitted per insert() call.
        """
        if record.tier not in TIER_ENUM:
            raise ValueError(f"invalid tier {record.tier!r}")
        if len(record.embedding) != self._embed_dim:
            raise ValueError(
                f"embedding must be {self._embed_dim}d, got {len(record.embedding)}"
            )
        # Lazy structure_hv fill via tem.bind_structure.  ImportError is caught so
        # a transient module-unavailability (e.g. during a native-rebuild window)
        # does not abort the insert.  structure_hv stays b"" (the established sentinel);
        # a later consolidation pass backfills it.
        if not record.structure_hv:
            try:
                from iai_mcp.tem import bind_structure
                record.structure_hv = bind_structure(record)
            except ImportError:
                pass  # structure_hv stays b"" sentinel; backfilled on a later consolidation pass

        # Time cells: compute temporal hash for cross-session sequence queries.
        try:
            from iai_mcp.time_cells import compute_temporal_hash
            if record.created_at and not getattr(record, "_temporal_hash", None):
                record._temporal_hash = compute_temporal_hash(
                    session_id=getattr(record, "session_id", "-") or "-",
                    timestamp=record.created_at,
                )
        except (ImportError, TypeError, ValueError):
            pass

        # Pattern-separation gate. Runs after structure_hv fill, before
        # any write. Caller-transparent merging: SKIP path mutates
        # record.id = existing_record_id so callers reading record.id
        # post-insert see the merged-into id. Dry-run honors
        # emit-but-no-mutate.
        from iai_mcp.daemon_config import _load_patsep_config
        from iai_mcp.events import write_event
        _psep_cfg = _load_patsep_config()
        # Use the internal 3-tuple gate so the gate's query_similar hits
        # are reused for event-payload bookkeeping below instead of firing
        # a second tbl.search().to_pandas() scan per insert. MUST run
        # BEFORE any tbl.add so top_k_probed reflects the pre-insert
        # store size.
        (
            _psep_action,
            _psep_payload,
            _psep_hits,
        ) = self._pattern_separation_gate_with_hits(record)
        _psep_near_dup_hit_id: str | None = None
        _psep_near_dup_cos: float | None = None
        _psep_edges_seeded = 0
        _psep_top_k_probed = len(_psep_hits)
        if _psep_action == GateAction.SKIP:
            _psep_near_dup_hit_id = str(_psep_payload)
            # Recover the cosine from the gate's top-1 hit (same query,
            # same store state -- gate just ran).
            if _psep_hits:
                _psep_near_dup_cos = float(_psep_hits[0][1])
            if not _psep_cfg.dry_run:
                # Caller-transparent merge: mutate record.id, reinforce
                # existing. Emit single 'skip' event and return — NO
                # row added.
                existing_id = (
                    _psep_payload if isinstance(_psep_payload, UUID)
                    else UUID(str(_psep_payload))
                )
                self.reinforce_record(existing_id)
                record.id = existing_id
                # Buffered EVENTS-table write — coalesced fragments via the
                # daemon WAKE / periodic-tick / shutdown flush hooks.
                write_event(self, "pattern_separation_pass", {
                    "action": "skip",
                    "near_dup_hit_id": _psep_near_dup_hit_id,
                    "near_dup_cos": _psep_near_dup_cos,
                    "edges_seeded": 0,
                    "top_k_probed": _psep_top_k_probed,
                    "threshold_near_dup": float(_psep_cfg.near_dup_threshold),
                    "threshold_link": float(_psep_cfg.link_threshold),
                    "dry_run_mode": False,
                }, severity="info", buffered=True)
                return
            # Dry-run SKIP: emit 'skip' event with dry_run_mode=True
            # and fall through to the regular insert below.
            write_event(self, "pattern_separation_pass", {
                "action": "skip",
                "near_dup_hit_id": _psep_near_dup_hit_id,
                "near_dup_cos": _psep_near_dup_cos,
                "edges_seeded": 0,
                "top_k_probed": _psep_top_k_probed,
                "threshold_near_dup": float(_psep_cfg.near_dup_threshold),
                "threshold_link": float(_psep_cfg.link_threshold),
                "dry_run_mode": True,
            }, severity="info", buffered=True)

        # Schema-bypass tagging: cosine-probe vs community_centroids,
        # AFTER the pattern-separation gate has fired INSERT and BEFORE
        # the store write commits. Gated on GateAction.INSERT so
        # dry-run-SKIP-fell-through never tags. _maybe_tag_schema_bypass
        # is self-contained — emits the single schema_bypass_pass event
        # with {record_id, max_cos, tagged, dry_run_mode}.
        if _psep_action == GateAction.INSERT:
            self._maybe_tag_schema_bypass(record)
            # Spatial scaffold tagging: derive (wing, room, drawer)
            # from provenance source_path AFTER schema-bypass tagging and
            # BEFORE the store write commits. Same INSERT-gate placement
            # so the SKIP-merged near-dup path never tags.
            self._maybe_spatial_tag(record)

        # Async-mode route. The queue's coalesce window batches
        # concurrent inserts; run_coroutine_threadsafe + fut.result() give
        # us the same "returns after disk flush" contract as the sync path.
        if self._write_queue is not None and self._async_loop is not None:
            coro = self._write_queue.enqueue(record)
            submit = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
            fut = submit.result()
            # fut is an asyncio.Future owned by the background loop; we
            # need to wait on it from this (sync) thread too.
            done_event = threading.Event()
            result_box: dict = {}

            def _watch(_f: asyncio.Future) -> None:
                if _f.cancelled():
                    result_box["exc"] = asyncio.CancelledError()
                elif _f.exception() is not None:
                    result_box["exc"] = _f.exception()
                else:
                    result_box["val"] = _f.result()
                done_event.set()

            self._async_loop.call_soon_threadsafe(fut.add_done_callback, _watch)
            done_event.wait()
            if "exc" in result_box:
                raise result_box["exc"]
            # Async-path post-write: seed pattern_separation_seed edges
            # if INSERT-with-targets and not dry-run; then emit the
            # 'insert' event. Guarded so the dry-run-SKIP-fell-through
            # case does NOT double-emit (it already emitted 'skip' above).
            # Hebbian self-loop ALWAYS writes on fresh-INSERT (not
            # dry-run) regardless of _psep_payload truthiness.
            # delta=link_initial_weight matches the dedup-path delta so
            # the Hebbian weight gradient between fresh and reinforced
            # records is preserved.
            if _psep_action == GateAction.INSERT and not _psep_cfg.dry_run:
                # Symmetric self-loop write — async-path arm.
                self.boost_edges(
                    [(record.id, record.id)],
                    delta=float(_psep_cfg.link_initial_weight),
                    edge_type="hebbian",
                )
                if _psep_payload:
                    edge_targets = _psep_payload  # list[tuple[UUID, float]]
                    pairs = [
                        (record.id, target_uuid) for target_uuid, _cos in edge_targets
                    ]
                    self.boost_edges(
                        pairs,
                        delta=float(_psep_cfg.link_initial_weight),
                        edge_type="pattern_separation_seed",
                    )
                    _psep_edges_seeded = len(edge_targets)
            if not (_psep_action == GateAction.SKIP and _psep_cfg.dry_run):
                write_event(self, "pattern_separation_pass", {
                    "action": "insert",
                    "near_dup_hit_id": _psep_near_dup_hit_id,
                    "near_dup_cos": _psep_near_dup_cos,
                    "edges_seeded": _psep_edges_seeded,
                    "top_k_probed": _psep_top_k_probed,
                    "threshold_near_dup": float(_psep_cfg.near_dup_threshold),
                    "threshold_link": float(_psep_cfg.link_threshold),
                    "dry_run_mode": bool(_psep_cfg.dry_run),
                }, severity="info", buffered=True)
            return

        # Legacy sync path (back-compat for all existing callers).
        # Buffered batch write: row appended to _record_buffer, flushed on threshold + daemon lifecycle hooks.
        row = self._to_row(record)
        _record_buffer.setdefault(id(self), []).append(row)
        if should_flush_record_buffer(id(self)):
            flush_record_buffer(self)
        from iai_mcp.retrieve import invalidate_temporal_validity_cache
        invalidate_temporal_validity_cache(self)
        self._fire_graph_sync_hook("insert", record)
        # Sync-path post-write: seed pattern_separation_seed edges if
        # INSERT-with-targets and not dry-run; then emit the 'insert'
        # event. Same guard as async path to avoid double-emit after
        # dry-run-SKIP-fell-through.
        # Hebbian self-loop ALWAYS writes on fresh-INSERT (not dry-run)
        # regardless of _psep_payload truthiness. delta=link_initial_weight
        # matches dedup delta so the Hebbian weight gradient between fresh
        # and frequently-reinforced records is preserved.
        if _psep_action == GateAction.INSERT and not _psep_cfg.dry_run:
            # Symmetric self-loop write — sync-path arm.
            self.boost_edges(
                [(record.id, record.id)],
                delta=float(_psep_cfg.link_initial_weight),
                edge_type="hebbian",
            )
            if _psep_payload:
                edge_targets = _psep_payload  # list[tuple[UUID, float]]
                pairs = [
                    (record.id, target_uuid) for target_uuid, _cos in edge_targets
                ]
                self.boost_edges(
                    pairs,
                    delta=float(_psep_cfg.link_initial_weight),
                    edge_type="pattern_separation_seed",
                )
                _psep_edges_seeded = len(edge_targets)
        if not (_psep_action == GateAction.SKIP and _psep_cfg.dry_run):
            write_event(self, "pattern_separation_pass", {
                "action": "insert",
                "near_dup_hit_id": _psep_near_dup_hit_id,
                "near_dup_cos": _psep_near_dup_cos,
                "edges_seeded": _psep_edges_seeded,
                "top_k_probed": _psep_top_k_probed,
                "threshold_near_dup": float(_psep_cfg.near_dup_threshold),
                "threshold_link": float(_psep_cfg.link_threshold),
                "dry_run_mode": bool(_psep_cfg.dry_run),
            }, severity="info", buffered=True)

    # -------------------------------------------------------- async writes

    async def enable_async_writes(
        self,
        coalesce_ms: int = 100,
        max_batch: int = 128,
        max_queue_size: int = 4096,
    ) -> None:
        """Switch ``insert()`` onto the coalescing AsyncWriteQueue.

        Runs the queue on a dedicated background event loop so sync
        callers (every existing user of ``store.insert``) can keep
        calling ``insert(record)`` and block on the batch flush via
        ``run_coroutine_threadsafe``. The read path stays synchronous
        and untouched.

        Thread-safety: HippoTable.add acquires HippoDB._hnsw_lock
        (threading.RLock) around every SQLite INSERT + hnswlib add_items
        pair. This lock is shared with all other Hippo mutations from
        the foreground daemon coroutines, so concurrent foreground +
        background writes serialise safely on a single sqlite3.Connection
        (opened with check_same_thread=False in HippoDB.__init__).

        Idempotent: a second call while already enabled is a no-op.
        """
        if self._write_queue is not None:
            return

        from iai_mcp.write_queue import AsyncWriteQueue

        # Spawn a dedicated loop on a daemon thread. The calling
        # coroutine stays on the caller's loop — we do not block it.
        ready = threading.Event()
        loop_holder: dict = {}

        def _run() -> None:
            loop = asyncio.new_event_loop()
            loop_holder["loop"] = loop
            asyncio.set_event_loop(loop)
            ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        thread = threading.Thread(
            target=_run, name="iai-mcp-async-writes", daemon=True,
        )
        thread.start()
        ready.wait()
        bg_loop: asyncio.AbstractEventLoop = loop_holder["loop"]

        # Get a sync HippoTable reference for the records table.
        # No async connection needed: HippoTable.add is synchronous and
        # acquires HippoDB._hnsw_lock internally, so it's safe to call
        # from the background thread via asyncio.to_thread.
        sync_records_tbl = self.db.open_table(RECORDS_TABLE)

        # Adapter: queue enqueues MemoryRecord objects; HippoTable.add
        # expects a list of row dicts. We convert here so the queue's
        # on_flushed callback still sees MemoryRecords.
        to_row = self._to_row

        class _RecordTableAdapter:
            """AsyncWriteQueue adapter for the Hippo records table.

            Thread-safety: HippoTable.add acquires HippoDB._hnsw_lock
            (threading.RLock) around the SQLite INSERT + hnswlib add_items
            pair. This lock is shared with all other Hippo mutations from
            the foreground daemon coroutines, so concurrent foreground +
            background writes serialise safely on a single sqlite3.Connection
            (opened with check_same_thread=False in HippoDB.__init__).
            """

            def __init__(self, real_tbl, to_row_fn) -> None:
                self._real = real_tbl
                self._to_row = to_row_fn

            async def add(self, records: list) -> None:
                rows = [self._to_row(r) for r in records]
                # HippoTable.add is sync (acquires HippoDB._hnsw_lock around
                # SQLite + hnswlib). Run in a worker thread so the event loop
                # is not blocked by SQLite WAL + hnswlib add_items work.
                await asyncio.to_thread(self._real.add, rows)

        adapter = _RecordTableAdapter(sync_records_tbl, to_row)

        # on_flushed: fire the graph-sync hook once per record in
        # batch order. This is synchronous (runs on the background loop)
        # but the hook itself is pure-python — no blocking I/O expected.
        fire_hook = self._fire_graph_sync_hook

        def _on_flushed(batch: list) -> None:
            for rec in batch:
                fire_hook("insert", rec)

        # pre_flush_gate=None — the caller-side gate inside
        # MemoryStore.insert() is authoritative. The queue-side seam
        # exists but is reserved for future S2 coordination of
        # concurrent writes.
        queue = AsyncWriteQueue(
            adapter,
            coalesce_ms=coalesce_ms,
            max_batch=max_batch,
            max_queue_size=max_queue_size,
            on_flushed=_on_flushed,
        )
        asyncio.run_coroutine_threadsafe(queue.start(), bg_loop).result()

        self._async_loop = bg_loop
        self._async_thread = thread
        self._async_conn = None  # no async connection object on Hippo path
        self._write_queue = queue

        # Same opt-in enables the provenance queue too — anyone who wants
        # async record writes also wants async provenance writes (both are
        # off the user-facing critical path).
        self.enable_provenance_queue()

    async def disable_async_writes(self) -> None:
        """Drain the queue, tear down the background loop.

        After this returns, ``insert()`` reverts to the legacy sync
        path. Idempotent.
        """
        if self._write_queue is None:
            # Still tear down the provenance queue if only that half was up.
            self.disable_provenance_queue()
            return
        # Tear down the provenance queue first so in-flight writes drain
        # via the still-live sync append path.
        self.disable_provenance_queue()
        bg_loop = self._async_loop
        queue = self._write_queue
        try:
            asyncio.run_coroutine_threadsafe(queue.stop(), bg_loop).result()
            # No async connection object to close on the Hippo path.
        finally:
            # Stop the background loop + join its thread.
            if bg_loop is not None:
                bg_loop.call_soon_threadsafe(bg_loop.stop)
            if self._async_thread is not None:
                self._async_thread.join(timeout=5.0)
            self._write_queue = None
            self._async_loop = None
            self._async_thread = None
            self._async_conn = None

    # -------------------------------------------------- provenance queue

    def enable_provenance_queue(self, *, coalesce_ms: int = 50) -> None:
        """Route provenance writes through a daemon-thread queue.

        After this call, ``queue_provenance_batch(pairs)`` hands the
        pairs off to a background worker and returns immediately;
        ``pipeline_recall`` no longer blocks on ``append_provenance_batch``.
        Idempotent — a second call with an already-live queue is a
        no-op.

        The queue is purpose-built for provenance (pure side effect,
        never read back). Record inserts still go through the
        ``AsyncWriteQueue`` from ``enable_async_writes()`` because they
        must be durable before return (S4 viability).
        """
        if self._provenance_queue is not None:
            return
        from iai_mcp.provenance_queue import ProvenanceWriteQueue

        q = ProvenanceWriteQueue(self, coalesce_ms=coalesce_ms)
        q.start()
        self._provenance_queue = q

    def disable_provenance_queue(self) -> None:
        """Drain and stop the provenance queue.

        After this returns, ``queue_provenance_batch`` reverts to the
        sync fallback. Idempotent.
        """
        q = self._provenance_queue
        if q is None:
            return
        try:
            q.flush(timeout=2.0)
        except (OSError, RuntimeError, TimeoutError) as exc:
            logger.debug("provenance queue flush during teardown: %s", exc)
        try:
            q.stop()
        except (OSError, RuntimeError) as exc:
            logger.debug("provenance queue stop during teardown: %s", exc)
        self._provenance_queue = None

    def queue_provenance_batch(
        self,
        pairs: "list[tuple[UUID, dict]]",
        records_cache: "dict | None" = None,
    ) -> None:
        """Fire-and-forget provenance write.

        If the async queue is live, enqueue and return (non-blocking).
        Otherwise fall back to the sync ``append_provenance_batch`` call.
        This is what ``pipeline_recall`` calls in place of the direct
        sync write.

        The sync fallback is wrapped in the caller's own try/except; a
        second layer is not added here so failures surface the same way
        they always did.
        """
        if not pairs:
            return
        q = self._provenance_queue
        if q is not None:
            q.enqueue(pairs)
            return
        # Sync fallback (back-compat).
        self.append_provenance_batch(pairs, records_cache=records_cache)

    # ------------------------------------------------------- record writes

    def update(self, record: MemoryRecord) -> None:
        """Full-record update (used by the graph-sync surface).

        Rewrites the core columns we expose on graph node attrs
        (embedding, literal_surface, centrality, tier, pinned) plus
        updated_at. Encrypts literal_surface under the record's AD.
        Missing record id is a silent no-op (matches append_provenance
        semantics). Writes-first, hook-second: store is authoritative.

        Scope note: this is deliberately narrower than _to_row — we only
        touch columns relevant to the runtime recall surface. FSRS-only
        updates should keep using update_record(record). Callers that
        need to rewrite every column (migration path) should delete +
        insert instead.
        """
        if len(record.embedding) != self._embed_dim:
            raise ValueError(
                f"embedding must be {self._embed_dim}d, got {len(record.embedding)}"
            )
        tbl = self.db.open_table(RECORDS_TABLE)
        # Fast existence check before issuing the update.
        df = tbl.to_pandas()
        if df.empty or str(record.id) not in set(df["id"].tolist()):
            return
        literal_ct = self._encrypt_for_record(record.id, record.literal_surface)
        tbl.update(
            where=f"id = '{_uuid_literal(record.id)}'",
            values={
                "literal_surface": literal_ct,
                "embedding": [float(x) for x in record.embedding],
                "centrality": float(record.centrality),
                "tier": record.tier,
                "pinned": bool(record.pinned),
                "updated_at": datetime.now(timezone.utc),
            },
        )
        self._fire_graph_sync_hook("update", record)

    def delete(self, record_id: UUID) -> None:
        """Remove a record by id and mirror to the runtime graph.

        The store's ``tbl.delete(where=...)`` is the authoritative operation.
        Unknown id is a silent no-op. Graph-sync hook fires with a
        minimal shim record carrying only ``id`` so the hook can drop
        the node from the runtime graph without needing the full
        payload.
        """
        tbl = self.db.open_table(RECORDS_TABLE)
        try:
            tbl.delete(where=f"id = '{_uuid_literal(record_id)}'")
        except (OSError, ValueError, RuntimeError) as exc:
            # The store raises on malformed WHERE; normalise to no-op so
            # callers get the same semantics as unknown-id.
            logger.warning("store delete normalised to no-op for %s: %s", record_id, exc)
            return

        # Fire the hook with a minimal shim — the graph sync only needs
        # the id to call G.remove_node.
        class _DeleteShim:
            def __init__(self, rid):
                self.id = rid
        self._fire_graph_sync_hook("delete", _DeleteShim(record_id))

    def get(self, record_id: UUID) -> MemoryRecord | None:
        """Filter-pushdown point read.

        Uses ``tbl.search().where(...).limit(1).to_pandas()`` so only the
        matching row is materialised; cost is O(index-lookup), sub-ms at
        N=1k. Unknown id returns None; existing id returns a ``MemoryRecord``
        via ``_from_row``. ``_uuid_literal`` guards against SQL-injection /
        malformed-UUID inputs.
        """
        tbl = self.db.open_table(RECORDS_TABLE)
        df = (
            tbl.search()
            .where(f"id = '{_uuid_literal(record_id)}'")
            .limit(1)
            .to_pandas()
        )
        if df.empty:
            return None
        return self._from_row(df.iloc[0].to_dict())

    def all_records(self) -> list[MemoryRecord]:
        # STAYS pending-INCLUSIVE — recent_user_turns() calls this method
        # so filtering here would delete pending rows from recency.  The
        # semantic/ANN exclusion lives in query_similar + the retrieve.py
        # graph-build MISS path.
        tbl = self.db.open_table(RECORDS_TABLE)
        df = tbl.to_pandas()
        return [self._from_row(r.to_dict()) for _, r in df.iterrows()]

    def active_records_count(self) -> int:
        """Return the count of non-pending, non-tombstoned records.

        Used by the warm-graph cache gate (retrieve.py) and cache key
        (runtime_graph_cache.py) so pending rows do not force perpetual
        rebuild churn while they exist.
        """
        with self.db._conn_lock:
            row = self.db._conn.execute(
                "SELECT COUNT(*) FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
            ).fetchone()
        return int(row[0]) if row else 0

    def find_record_by_tag(self, tag: str) -> UUID | None:
        """Return the id of the first record carrying the exact tag, or None.

        Uses a targeted SQL query with a LIKE substring pre-filter on the
        plaintext ``tags_json`` column, then JSON-verifies the exact match in
        Python. No Arrow materialization; no full-store scan. Runs under
        ``HippoDB._conn_lock`` so the ``execute().fetchall()`` pair is
        serialized against concurrent writers (shared connection, WAL mode,
        ``check_same_thread=False``).

        Returns the record's UUID, or None when no match exists.
        Used by the idempotency-key probe at both dedup gates.
        """
        # JSON-literal substring pre-filter: LIKE cannot contain wildcards in
        # the user-supplied tag, but JSON.dumps escapes nothing that LIKE
        # treats as special for normal tag strings — and the Python
        # JSON-verify below catches any false positives regardless.
        tag_json_literal = json.dumps(tag)  # e.g. '"idem:abc123"'
        sql = (
            "SELECT id, tags_json FROM records"
            " WHERE tags_json LIKE :pat"
        )
        params = {"pat": f"%{tag_json_literal}%"}
        with self.db._conn_lock:
            rows = self.db._conn.execute(sql, params).fetchall()
        if rows is None:
            raise HippoIntegrityError(
                "find_record_by_tag: fetchall() returned None — connection may be"
                " in an error state"
            )
        for row in rows:
            tags_raw = row["tags_json"] if row["tags_json"] else "[]"
            try:
                tags = json.loads(tags_raw)
            except (ValueError, TypeError):
                continue
            if tag in tags:
                raw_id = row["id"]
                if raw_id is None:
                    continue
                try:
                    return UUID(str(raw_id))
                except (ValueError, AttributeError):
                    continue
        return None

    def centrality_for_ids(self, ids: list[UUID]) -> dict[UUID, float]:
        """Return a {id: centrality} map for the given record ids.

        Projects only the plaintext ``(id, centrality)`` columns in a single
        bounded scan — zero AES-GCM decrypt operations.  The scan flows through
        ``iter_record_columns`` and therefore through the ``_conn_lock``-guarded
        ``to_batches`` path (safe under concurrent writes).

        Behaviour matches the per-member getattr default:
        - NULL / missing centrality maps to 0.0.
        - Record ids absent from the store are omitted from the result.

        Parameters
        ----------
        ids:
            The record ids to look up.  An empty list returns ``{}``.
        """
        if not ids:
            return {}
        target = frozenset(str(i) for i in ids)
        out: dict[UUID, float] = {}
        for row in self.iter_record_columns(["id", "centrality"]):
            raw_id = row.get("id")
            if raw_id is None:
                continue
            id_str = str(raw_id)
            if id_str not in target:
                continue
            try:
                centrality = float(row.get("centrality") or 0.0)
            except (TypeError, ValueError):
                centrality = 0.0
            try:
                out[UUID(id_str)] = centrality
            except (ValueError, AttributeError):
                continue
        return out

    def recent_user_turns(
        self,
        n: int = 10,
        session_id: str | None = None,
        pending_live_events: "list | None" = None,
    ) -> "list":
        """Return the N most-recent role:user episodic records, time-desc.

        Scans all records in Python after decryption (provenance_json is
        AES-GCM encrypted — SQL WHERE on it silently matches ciphertext).
        GLOBAL — no cwd filter. Returns at most n records (clamped to [0, 1000]
        by the caller).

        Parameters
        ----------
        n:
            Maximum number of records to return. Must be ≥ 0.
        session_id:
            If given, only records whose originating provenance entry has this
            session_id are returned. Matched against provenance[0]["session_id"].
        pending_live_events:
            OPT-IN: when provided (a list of dicts from
            ``capture.read_pending_live_events``), merges pending turns with
            the store candidates, deduplicated by the ts-normalized idem-tag
            against BOTH the store's tag set AND a seen-pending set (deduping
            pending-vs-pending re-emissions), filtered to role==user, then
            sorted by ``created_at`` desc.  When ``None`` (the default), the
            method behaves exactly as before (bare callers are unaffected).
        """
        from iai_mcp.capture import _idem_tag as _cap_idem_tag

        records = self.all_records()
        # Pending rows (embedding_pending=1) are recency hits even without the
        # role:user tag — they were just written and are awaiting re-embed.
        # The primary filter stays role:user + episodic; pending episodic rows
        # are also included (pending IS a recency hit).
        cands = [
            r for r in records
            if r.tier == "episodic"
            and (
                "role:user" in (r.tags or [])
                or r.embedding_pending
            )
        ]
        if session_id:
            cands = [
                r for r in cands
                if (r.provenance or [{}])[0].get("session_id") == session_id
            ]

        if pending_live_events is not None:
            # Build the store idem-tag SET in one pass (membership check for dedup).
            store_idem_set: set[str] = set()
            for r in records:
                for tag in (r.tags or []):
                    if tag.startswith("idem:"):
                        store_idem_set.add(tag)

            # Pending-vs-pending dedup set.
            seen_pending_idem: set[str] = set()

            pending_wrappers = []
            for ev in pending_live_events:
                # Only role:user events for this query (conversational filter).
                if ev.get("role") != "user":
                    continue
                ev_session = ev.get("session_id", "-")
                # Apply session filter to pending events the same way as store records.
                if session_id and ev_session != session_id:
                    continue
                src_uuid = ev.get("source_uuid")
                ts_iso = ev["ts_iso"]  # contract key (already normalized by the live reader)
                text = ev.get("text", "")
                idem = _cap_idem_tag(ev_session, "user", ts_iso, text, source_uuid=src_uuid)
                # Skip if already in store.
                if idem in store_idem_set:
                    continue
                # Skip if we already saw this pending idem (pending-vs-pending dedup).
                if idem in seen_pending_idem:
                    continue
                seen_pending_idem.add(idem)
                pending_wrappers.append(_PendingTurn(
                    text=text,
                    session_id=ev_session,
                    ts=ev["ts"],
                    idem_tag=idem,
                    source_uuid=src_uuid,
                ))

            # Merge store candidates with pending wrappers.
            cands = list(cands) + pending_wrappers  # type: ignore[arg-type]

        cands.sort(key=lambda r: r.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return cands[:n]

    def iter_records(
        self,
        *,
        columns: list[str] | None = None,
        batch_size: int = 1024,
        where: str | None = None,
    ):
        """Streaming + projection iterator over records.

        Yields ``MemoryRecord`` instances batch by batch. Unlike
        :meth:`all_records`, nothing is materialised into a single
        in-memory list; downstream consumers can process records lazily
        and keep peak RSS bounded.

        Parameters
        ----------
        columns:
            If given, only these columns are read from disk. Encrypted
            columns NOT in this list are never decrypted. When ``None``,
            all columns are read.
        batch_size:
            Rows per ``RecordBatch``. Default 1024.
        where:
            Optional SQL-style predicate forwarded to the scanner.
            Example: ``"tier = 'episodic'"``. ``None`` = full scan.
        """
        tbl = self.db.open_table(RECORDS_TABLE)
        query = tbl.search()
        if where is not None:
            query = query.where(where)
        if columns is not None:
            query = query.select(columns)
        reader = query.to_batches(batch_size=batch_size)
        for batch in reader:
            for row_dict in batch.to_pylist():
                yield self._from_row(row_dict)

    def iter_record_columns(
        self,
        columns: list[str],
        *,
        batch_size: int = 1024,
        where: str | None = None,
    ):
        """Projection-only iteration; no MemoryRecord construction, no decrypt.

        Yields raw ``dict`` rows containing only the requested columns. Encrypted
        fields (literal_surface, provenance_json, profile_modulation_gain_json),
        if listed in ``columns``, pass through as ciphertext strings -- the caller
        decides whether to decrypt. For tag-only paths, no AES-GCM operations
        happen anywhere on the path.

        Parameters mirror :meth:`iter_records`. ``columns`` is REQUIRED; if you
        want every column, use :meth:`all_records` or :meth:`iter_records`.
        """
        if not columns:
            raise ValueError("iter_record_columns requires a non-empty columns list")
        tbl = self.db.open_table(RECORDS_TABLE)
        query = tbl.search()
        if where is not None:
            query = query.where(where)
        query = query.select(columns)
        reader = query.to_batches(batch_size=batch_size)
        for batch in reader:
            for row_dict in batch.to_pylist():
                yield row_dict

    def query_similar(
        self,
        vec: list[float],
        k: int = 10,
        tier: str | None = None,
        *,
        n: int | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Cosine-distance kNN search. Returns (record, cosine_similarity) pairs.

        The store's default L2 distance is mapped via explicit `.distance_type("cosine")`
        so `_distance` is cosine distance; we return `1.0 - _distance` as similarity.

        Optional ``tier`` kwarg applies a where-clause filter at the
        search layer. Validated against ``TIER_ENUM``; bad tier values raise
        ``ValueError`` before any I/O is attempted (also acts as a
        SQL-injection guard since tier values are alphanumeric ASCII). When
        ``tier=None``, no where-clause is appended.
        """
        # Accept n= as an alias for k=.  When n= is used as a keyword-only
        # argument (not k=), return just the list of MemoryRecord objects
        # without score tuples — this matches the test convention that iterates
        # over results with r.id directly.
        _return_records_only = n is not None
        if n is not None:
            k = n
        # Validate `tier` BEFORE any I/O so a bad value never touches the store.
        # Sentinel raise lets callers catch ValueError on the bad-tier path.
        if tier is not None and tier not in TIER_ENUM:
            raise ValueError(
                f"invalid tier {tier!r}; must be one of {sorted(TIER_ENUM)}"
            )

        tbl = self.db.open_table(RECORDS_TABLE)
        # Fast path for empty store -- tbl.search on empty raises or returns empty;
        # the explicit check also avoids store warnings about missing indices at N=0.
        if tbl.count_rows() == 0:
            return []
        # Build the query chain. Mirrors the predicate-where idiom in
        # `iter_records`.
        q = tbl.search(list(vec)).distance_type("cosine")
        # Exclude pending rows from the cosine/ANN search path
        # (defense-in-depth — a zero-vector is a degenerate cosine neighbor).
        # all_records() is NOT filtered (recency stays pending-inclusive).
        where_clause = "COALESCE(embedding_pending, 0) = 0"
        if tier is not None:
            # Tier validated above against TIER_ENUM (alphanumeric ASCII), so
            # direct string interpolation here is safe.
            where_clause = f"tier = '{tier}' AND " + where_clause
        q = q.where(where_clause)
        results = q.limit(k).to_pandas()
        out: list[tuple[MemoryRecord, float]] = []
        for _, row in results.iterrows():
            record = self._from_row(row.to_dict())
            # The store returns `_distance` as cosine distance in [0, 2]; similarity = 1 - distance.
            distance = float(row.get("_distance", 1.0)) if "_distance" in row else 1.0
            score = 1.0 - distance
            out.append((record, score))
        if _return_records_only:
            return [r for r, _s in out]
        return out

    def pattern_separation_gate(
        self,
        record: MemoryRecord,
    ) -> tuple["GateAction", "GatePayload"]:
        """Pre-insert pattern-separation gate. Runs BEFORE the write.

        Returns either:
          (GateAction.SKIP, existing_record_uuid)
            when top-1 query_similar hit cosine >= near_dup_threshold;
            caller MUST reinforce_record(existing_record_uuid) and mutate
            record.id = existing_record_uuid (caller-transparent merging).
          (GateAction.INSERT, [(target_uuid, cosine), ...])
            in all other cases. Empty list when no hits clear
            link_threshold or the store is empty. Caller MUST proceed with
            the store add and then seed pattern_separation_seed edges for
            each target_uuid via boost_edges(weight=link_initial_weight).

        IDEMPOTENCY: calling this twice on the same record without
        intervening writes returns the same (action, payload) shape —
        query_similar is stateless and PatSepConfig is reloaded fresh
        each call.

        EMBEDDING INVARIANT: this function MUST NOT mutate
        record.embedding. The contract is read-only against record.
        """
        # Public 2-tuple shape preserved for backward compatibility with
        # tests / external callers. The internal-only 3-tuple variant
        # below (with the hits already in hand) is what insert() should
        # call to avoid a redundant query_similar scan per insert.
        action, payload, _hits = self._pattern_separation_gate_with_hits(record)
        return (action, payload)

    def _pattern_separation_gate_with_hits(
        self,
        record: MemoryRecord,
    ) -> tuple["GateAction", "GatePayload", list[tuple[MemoryRecord, float]]]:
        """Internal: like pattern_separation_gate but also returns the
        raw query_similar hits so the caller can read top_k_probed and
        top-1 cos without a second scan.

        Returns (action, payload, hits) where hits is the list from
        ``query_similar(record.embedding, k=cfg.top_k)``. The 2-tuple
        public ``pattern_separation_gate`` wraps this and drops the hits.
        """
        from iai_mcp.daemon_config import _load_patsep_config
        cfg = _load_patsep_config()

        # Empty/small-store: query_similar already short-circuits on
        # tbl.count_rows() == 0; the returned list may be shorter than top_k.
        hits = self.query_similar(list(record.embedding), k=cfg.top_k)

        # Orthogonal-routing variant (default OFF, gated on env flag).
        # The orthogonalized vector is routing-only, never written back to
        # record.embedding. Cosines are recomputed manually against the
        # existing hits (NOT re-querying) so the top-k set is identical to
        # the legacy path; only the comparison values change.
        _ortho_enabled = os.environ.get(
            "IAI_MCP_ORTHO_ENABLED", "",
        ).lower() in {"1", "true"}
        if _ortho_enabled and hits:
            try:
                from iai_mcp.pattern_separation import orthogonalize_for_routing
                import numpy as _np
                neighbor_vecs = [r.embedding for r, _ in hits]
                routing_vec, _ortho_result = orthogonalize_for_routing(
                    list(record.embedding), neighbor_vecs, strength=0.3,
                )
                # Recompute cosines against routing_vec for each hit (the
                # routing-only swap). Don't re-fire query_similar -- that
                # could change the top-k set; we only want the comparison
                # values updated.
                _rv = _np.asarray(routing_vec, dtype=_np.float32)
                _rv_norm = float(_np.linalg.norm(_rv))
                if _rv_norm > 1e-8:
                    _rv = _rv / _rv_norm
                    _new_hits: list[tuple[MemoryRecord, float]] = []
                    for _rec, _ in hits:
                        _ev = _np.asarray(
                            _rec.embedding, dtype=_np.float32,
                        )
                        _en = float(_np.linalg.norm(_ev))
                        if _en > 1e-8:
                            _ev = _ev / _en
                            _new_hits.append(
                                (_rec, float(_np.dot(_rv, _ev))),
                            )
                        else:
                            _new_hits.append((_rec, 0.0))
                    hits = _new_hits
            except Exception as _exc:  # noqa: BLE001 -- routing MUST NOT crash gate
                logger.debug(
                    "pattern_separation orthogonalize skipped: %s",
                    str(_exc)[:120],
                )

        # Episodic conversational turn exemption: distinct turns (different
        # session, role, ts, or text) must each produce their own row.  Only
        # an exact re-drain of the SAME turn (matching idem tag) is a
        # legitimate SKIP.  This check runs BEFORE the cosine near-dup test
        # so a near-identical turn from a different session is never merged.
        #
        # Predicate: tier=="episodic" AND a role:user or role:assistant tag.
        # Non-conversational episodic and semantic/consolidated: fall through
        # to the existing cosine top-1 check (unchanged).
        _record_tags = list(getattr(record, "tags", None) or [])
        _is_conv = (
            record.tier == "episodic"
            and ("role:user" in _record_tags or "role:assistant" in _record_tags)
        )
        if _is_conv:
            # Probe for the idem tag that capture_turn stamped on this record.
            # If found, it is an exact re-drain: return SKIP with the existing
            # id so insert()'s SKIP handler can reinforce and mutate record.id.
            # If not found, this is a new distinct turn: bypass cosine SKIP and
            # fall through to edge collection so INSERT fires correctly.
            _idem_tag_val: str | None = next(
                (t for t in _record_tags if t.startswith("idem:")), None
            )
            if _idem_tag_val is not None:
                _existing_id = self.find_record_by_tag(_idem_tag_val)
                if _existing_id is not None:
                    return (GateAction.SKIP, _existing_id, hits)
            # Distinct turn: skip the cosine near-dup check entirely.
            # Fall through to edge collection below.
        else:
            # Top-1 short-circuit: cosine >= near_dup_threshold => SKIP-and-merge.
            # never_merge bypass: if either party has never_merge=True, the gate
            # must fall through to INSERT regardless of cosine similarity. Defensive
            # getattr guards against shim/fixture variants that may lack the field.
            if hits:
                top_record, top_cos = hits[0]
                if top_cos >= cfg.near_dup_threshold:
                    if not (
                        getattr(record, "never_merge", False)
                        or getattr(top_record, "never_merge", False)
                    ):
                        return (GateAction.SKIP, top_record.id, hits)

        # Otherwise collect link-eligible targets.
        edges: list[tuple[UUID, float]] = []
        for rec, cos in hits:
            if cfg.link_threshold <= cos < cfg.near_dup_threshold:
                edges.append((rec.id, float(cos)))
        return (GateAction.INSERT, edges, hits)

    def update_record(self, record: MemoryRecord) -> None:
        """Persist FSRS-relevant columns back to the records table.

        Scope (deliberately narrow):
            stability, difficulty, last_reviewed, updated_at

        Everything else on the record is LEFT UNTOUCHED so this method cannot
        clobber concurrent writers. The store's tbl.update(values=...) only
        rewrites the listed columns.

        Unknown record id is a silent no-op (no exception, no table growth).
        """
        tbl = self.db.open_table(RECORDS_TABLE)
        predicate = f"id = '{_uuid_literal(record.id)}'"
        tbl.update(
            where=predicate,
            values={
                "stability": float(record.stability),
                "difficulty": float(record.difficulty),
                "last_reviewed": record.last_reviewed,
                "updated_at": datetime.now(timezone.utc),
            },
        )

    # -------------------------------------------------------- reconsolidation

    def append_provenance(self, record_id: UUID, entry: dict) -> None:
        """Append a provenance entry to the record.

        Read-modify-write. Existing provenance is decrypted when encrypted;
        the updated list is re-encrypted before write.
        """
        tbl = self.db.open_table(RECORDS_TABLE)
        predicate = f"id = '{_uuid_literal(record_id)}'"
        df = tbl.search().where(predicate).limit(1).to_pandas()
        if df.empty:
            return
        raw = df.iloc[0].get("provenance_json") or "[]"
        if is_encrypted(raw):
            raw = self._decrypt_for_record(record_id, raw)
        try:
            existing = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            existing = []
        existing.append(entry)
        new_json_plain = json.dumps(existing)
        new_json_ct = self._encrypt_for_record(record_id, new_json_plain)
        tbl.update(
            where=f"id = '{_uuid_literal(record_id)}'",
            values={
                "provenance_json": new_json_ct,
                "updated_at": datetime.now(timezone.utc),
            },
        )

    def append_provenance_batch(
        self, pairs: "list[tuple[UUID, dict]]",
        records_cache: "dict | None" = None,
    ) -> None:
        """Batched provenance append.

        Collapses the per-hit N+1 `to_pandas()` scan pattern into:
          * ONE `tbl.to_pandas()` scan to read current provenance (or ZERO
            if `records_cache` is provided).
          * ONE `tbl.merge_insert(...)` transaction to write back all updates.

        Semantics match append_provenance:
        - Each entry is appended to its record's provenance list.
        - Unknown record_ids are silently skipped.
        - Empty `pairs` -> no-op.
        - Order of entries per record is preserved.
        - `merge_insert` with a subset of columns preserves all other
          columns untouched.

        ``records_cache``: optional dict[UUID | str, MemoryRecord]. When
        provided, existing provenance is read from the cache -- skipping the
        full-table scan entirely. Missing ids are silently skipped.
        """
        if not pairs:
            return
        tbl = self.db.open_table(RECORDS_TABLE)

        # Group entries by record_id, preserving per-record insertion order.
        from collections import defaultdict
        grouped: dict[str, list[dict]] = defaultdict(list)
        for rid, entry in pairs:
            grouped[str(rid)].append(entry)

        # Build the merge-insert payload: one row per unique id with the new
        # provenance_json (existing list + appended entries) and fresh updated_at.
        now = datetime.now(timezone.utc)
        update_ids: list[str] = []
        update_prov: list[str] = []

        if records_cache is not None:
            # Fast path: read existing provenance from the pre-loaded cache.
            # Zero scan. Keyed by UUID object OR str (be permissive).
            for rid_str, entries in grouped.items():
                try:
                    canonical = _uuid_literal(rid_str)
                except ValueError:
                    continue
                # Try UUID object key first, then str fallback.
                try:
                    rec = records_cache.get(UUID(rid_str))
                except (TypeError, ValueError):
                    rec = None
                if rec is None:
                    rec = records_cache.get(rid_str)
                if rec is None:
                    # Not in cache -- silently skip (matches single-call semantics).
                    continue
                existing = list(rec.provenance or [])
                existing.extend(entries)
                # Encrypt the new provenance JSON so the updated row
                # matches the encrypted contract enforced by insert().
                new_plain = json.dumps(existing)
                new_ct = self._encrypt_for_record(UUID(rid_str), new_plain)
                update_ids.append(canonical)
                update_prov.append(new_ct)
        else:
            # Slow path: one full to_pandas() scan for existing provenance.
            df = tbl.to_pandas()
            if df.empty:
                return
            for rid_str, entries in grouped.items():
                idx_list = df.index[df["id"] == rid_str].tolist()
                if not idx_list:
                    continue
                try:
                    canonical = _uuid_literal(rid_str)
                except ValueError:
                    continue
                i = idx_list[0]
                raw_prov = df.at[i, "provenance_json"] or "[]"
                # Decrypt pre-existing ciphertext before merging
                # (fresh entries are plaintext dicts).
                if is_encrypted(raw_prov):
                    try:
                        raw_prov = self._decrypt_for_record(UUID(rid_str), raw_prov)
                    except (ValueError, OSError, TypeError) as exc:
                        logger.warning("provenance decrypt failed for %s: %s", rid_str, exc)
                        raw_prov = "[]"
                try:
                    existing = json.loads(raw_prov)
                except (TypeError, ValueError):
                    existing = []
                existing.extend(entries)
                new_plain = json.dumps(existing)
                new_ct = self._encrypt_for_record(UUID(rid_str), new_plain)
                update_ids.append(canonical)
                update_prov.append(new_ct)

        if not update_ids:
            return

        # Single merge_insert transaction: join on `id`, update matched rows'
        # provenance_json + updated_at columns. All other record columns are
        # preserved untouched (merge_insert with subset columns is surgical).
        import pyarrow as pa
        update_tbl = pa.table({
            "id": update_ids,
            "provenance_json": update_prov,
            "updated_at": [now] * len(update_ids),
        })
        try:
            tbl.merge_insert("id").when_matched_update_all().execute(update_tbl)
        except Exception as exc:  # noqa: BLE001 -- fallback gate, must stay broad
            logger.warning("provenance merge_insert fallback triggered: %s", exc, exc_info=True)
            # Never block recall on a provenance-write failure.
            # Fallback: per-id tbl.update() (slower but correct).
            for rid_str, new_json in zip(update_ids, update_prov):
                try:
                    tbl.update(
                        where=f"id = '{rid_str}'",
                        values={
                            "provenance_json": new_json,
                            "updated_at": now,
                        },
                    )
                except Exception as exc_inner:  # noqa: BLE001 -- per-row fallback continue
                    logger.debug("provenance per-row update failed for %s: %s", rid_str, exc_inner)
                    continue

    # ------------------------------------------------------------------ bounded read primitives (Layer-1)

    # SQL columns projected by get_batch and recent_pending_markers.
    # Must match the _from_row input contract (all columns _from_row reads).
    _RECORD_COLS = (
        "id, tier, literal_surface, aaak_index, embedding,"
        " community_id, centrality, detail_level, pinned,"
        " stability, difficulty, last_reviewed, never_decay, never_merge,"
        " provenance_json, created_at, updated_at, tags_json, language,"
        " s5_trust_score, profile_modulation_gain_json, schema_version,"
        " hv_tier, structure_hv_payload,"
        " COALESCE(embedding_pending, 0) AS embedding_pending"
    )

    # READ A: index-backed pending-marker query (hits idx_records_pending).
    # The ORDER BY rowid DESC LIMIT ? caps the decrypt to n rows regardless
    # of backlog size — never omit the LIMIT.
    _PENDING_READ_SQL = (
        f"SELECT {_RECORD_COLS} FROM records"  # noqa: S608
        " WHERE embedding_pending = 1"
        " ORDER BY rowid DESC LIMIT ?"
    )

    # READ B: role:user recency — index-backed on tier, LIKE is a residual
    # predicate over the tier-matched subset (idx_records_tier).  ORDER BY
    # rowid DESC LIMIT ? bounds the decrypt window.
    _ROLE_USER_READ_SQL = (
        f"SELECT {_RECORD_COLS} FROM records"  # noqa: S608
        " WHERE tier='episodic' AND tags_json LIKE ?"
        " ORDER BY rowid DESC LIMIT ?"
    )

    @staticmethod
    def _decode_raw_row(row: "dict") -> "dict":
        """Convert a raw sqlite3.Row dict to a _from_row-compatible dict.

        sqlite3 returns embedding as raw bytes (BLOB).  _from_row does
        ``list(row["embedding"])`` which yields byte-ints 0-255 on bytes,
        not float32 values.  This helper decodes the BLOB via np.frombuffer
        and converts to a Python float list so _from_row receives the same
        shape as HippoTable.to_pandas + _decode_df_embedding produces.
        """
        import numpy as _np
        emb_raw = row.get("embedding")
        if isinstance(emb_raw, (bytes, bytearray)) and emb_raw:
            row = dict(row)
            row["embedding"] = _np.frombuffer(emb_raw, dtype=_np.float32).tolist()
        return row

    def incident_edges(
        self,
        ids: "list[UUID]",
        edge_types: "list[str] | None" = None,
        top_k: "int | None" = 5,
    ) -> "dict[UUID, list[tuple[UUID, str, float]]]":
        """Return incident edges for the given record ids (ONE batched query).

        The graph is undirected and boost_edges canonicalises direction by
        sorted UUID, so a seed stored as 'dst' would be invisible to a
        src-only read.  This method uses an OR-bind covering both endpoints:
          WHERE (src IN (<placeholders>) OR dst IN (<placeholders>))
        so every edge incident on any of the input ids is returned,
        regardless of which UUID sorted first at write time.

        For each input id the NEIGHBOUR is the OTHER endpoint (if the
        matched row's src == id → neighbour = dst; else → src).

        top_k (default 5): when an int, sort each id's neighbours by weight
        descending and keep at most top_k (mirrors two_hop_neighborhood
        top-5-by-weight semantics).  When None, return
        UNCAPPED — callers that need ALL edges (e.g. contradicts /
        temporal-validity) pass top_k=None to avoid dropping low-weight
        superseding edges.

        The execute()+fetchall() pair runs inside self.db._conn_lock (RLock)
        to guard the shared sqlite3.Connection against cursor-state reset
        under asyncio.to_thread fan-out (Hippo invariant).
        """
        if not ids:
            return {}

        str_ids = [str(i) for i in ids]
        id_set = set(str_ids)

        # Build the parameterized IN-list.  The id list is bound TWICE —
        # once for the src IN clause and once for the dst IN clause.
        # Only the COUNT of placeholders is dynamic; the values are always
        # bound via ? so SQLite never parses them as SQL text.
        ph = ", ".join("?" for _ in str_ids)
        sql = (  # nosemgrep: sql-injection
            f"SELECT src, dst, edge_type, weight FROM edges"  # noqa: S608
            f" WHERE (src IN ({ph}) OR dst IN ({ph}))"
        )
        params: list = str_ids + str_ids  # id list bound twice

        if edge_types is not None:
            et_ph = ", ".join("?" for _ in edge_types)
            sql += f" AND edge_type IN ({et_ph})"  # nosemgrep: sql-injection
            params += list(edge_types)

        with self.db._conn_lock:
            rows = self.db._conn.execute(sql, params).fetchall()  # nosemgrep: sql-injection

        # Group by queried id; neighbour = the OTHER endpoint.
        result: dict[UUID, list[tuple[UUID, str, float]]] = {i: [] for i in ids}
        id_to_uuid: dict[str, UUID] = {str(i): i for i in ids}

        for row in rows:
            src_s = str(row[0] if hasattr(row, "__getitem__") else row["src"])
            dst_s = str(row[1] if hasattr(row, "__getitem__") else row["dst"])
            et = str(row[2] if hasattr(row, "__getitem__") else row["edge_type"])
            wt = float(row[3] if hasattr(row, "__getitem__") else row["weight"])

            # Determine which queried id(s) matched this row and the neighbour.
            if src_s in id_set:
                qid = id_to_uuid[src_s]
                try:
                    neighbour = UUID(dst_s)
                except (ValueError, AttributeError):
                    continue
                result[qid].append((neighbour, et, wt))

            if dst_s in id_set and dst_s != src_s:
                qid = id_to_uuid[dst_s]
                try:
                    neighbour = UUID(src_s)
                except (ValueError, AttributeError):
                    continue
                result[qid].append((neighbour, et, wt))

        # Apply top_k per id (sort by weight descending, keep top_k).
        if top_k is not None:
            for uid, edges in result.items():
                edges.sort(key=lambda t: t[2], reverse=True)
                result[uid] = edges[:top_k]

        return result

    def get_batch(self, ids: "list[UUID]") -> "dict[UUID, MemoryRecord]":
        """Return a dict of MemoryRecord for the given record ids (ONE batched query).

        Issues a single ``WHERE id IN (?, ?, ...)`` against the records table
        (parameterized — no f-string ids) and decrypts each row via _from_row.
        Unknown ids are simply absent from the returned dict.

        Replaces N+1 ``store.get`` calls on the 1-hop spread path (1000+
        dst ids resolved in one query instead of one query per id).

        The execute()+fetchall() pair runs inside self.db._conn_lock (RLock)
        to guard the shared sqlite3.Connection.
        """
        if not ids:
            return {}

        str_ids = [str(i) for i in ids]
        ph = ", ".join("?" for _ in str_ids)
        # Parameterized IN-bind: values are bound via ? placeholders, never
        # interpolated into the SQL string.  Only the count is dynamic.
        sql = (  # nosemgrep: sql-injection
            f"SELECT {self._RECORD_COLS} FROM records"  # noqa: S608
            f" WHERE id IN ({ph})"
        )
        with self.db._conn_lock:
            raw_rows = self.db._conn.execute(sql, str_ids).fetchall()  # nosemgrep: sql-injection

        out: dict[UUID, MemoryRecord] = {}
        for raw in raw_rows:
            row_dict = self._decode_raw_row(dict(raw))
            try:
                rec = self._from_row(row_dict)
                out[rec.id] = rec
            except Exception:  # noqa: BLE001 — skip corrupt rows, never crash
                continue
        return out

    def recent_pending_markers(self, n: int = 50) -> "list[MemoryRecord]":
        """Return recent embedding-pending and role:user episodic records.

        Uses TWO index-backed bounded reads (never calls all_records):

        READ A — pending rows (idx_records_pending partial index):
          SELECT ... FROM records WHERE embedding_pending = 1
          ORDER BY rowid DESC LIMIT n
          EXPLAIN: SEARCH records USING INDEX idx_records_pending

        READ B — role:user recency (idx_records_tier + bounded LIMIT):
          SELECT ... FROM records WHERE tier='episodic' AND tags_json LIKE ?
          ORDER BY rowid DESC LIMIT n*C
          EXPLAIN: SEARCH records USING INDEX idx_records_tier (tier=?)
          (LIKE is a residual predicate; exact Python match follows to drop
          substring false positives.)

        The two reads are deduped by id and the first n are returned.

        Each read's execute()+fetchall() pair runs under self.db._conn_lock.
        The LIMIT is always parameter-bound (? placeholder, never interpolated).
        """
        seen: dict[UUID, "MemoryRecord"] = {}

        # READ A: pending rows — hits idx_records_pending partial index.
        # LIMIT=n caps the decrypt to n rows (a stalled re-embed
        # backlog must not cause a full decrypt of all pending rows).
        with self.db._conn_lock:
            rows_a = self.db._conn.execute(
                self._PENDING_READ_SQL, (n,)
            ).fetchall()

        for raw in rows_a:
            row_dict = self._decode_raw_row(dict(raw))
            try:
                rec = self._from_row(row_dict)
                if rec.id not in seen:
                    seen[rec.id] = rec
            except Exception:  # noqa: BLE001
                continue

        # READ B: role:user episodic rows — idx_records_tier bounds the tier
        # scan; LIMIT=n*4 over-fetches to survive substring false positives.
        # A Python exact-match follows to drop false positives.
        over_fetch = n * 4
        with self.db._conn_lock:
            rows_b = self.db._conn.execute(
                self._ROLE_USER_READ_SQL, ('%"role:user"%', over_fetch)
            ).fetchall()

        for raw in rows_b:
            row_dict = self._decode_raw_row(dict(raw))
            try:
                rec = self._from_row(row_dict)
            except Exception:  # noqa: BLE001
                continue
            # Exact match: LIKE is substring, not exact JSON membership.
            if "role:user" not in (rec.tags or []):
                continue
            if rec.id not in seen:
                seen[rec.id] = rec

        # Return the most-recent n, deduplicated.
        candidates = list(seen.values())
        candidates.sort(
            key=lambda r: r.created_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return candidates[:n]

    # ------------------------------------------------------------------ edges

    def boost_edges(
        self,
        pairs: list[tuple[UUID, UUID]],
        delta: float | Sequence[float] = 0.1,
        edge_type: str = "hebbian",
    ) -> dict[tuple[str, str], float]:
        """Pairwise edge boost.

        Accepts any edge_type from EDGE_TYPES. Edge key is canonicalised to
        sorted (src, dst) so (a, b) and (b, a) collide. Returns the new weight
        for each pair (tuple keys).

        Produces AT MOST 2 store versions per call (one for `merge_insert`
        updating pre-existing rows, one for `tbl.add` of new rows) regardless
        of pair count:

        1. Validate `edge_type` and coerce `delta` to a per-pair list.
        2. Coalesce duplicate canonical (src, dst) keys in-memory by summing
           their deltas.
        3. ONE `tbl.to_pandas()` to load existing edges.
        4. Partition into update_rows (key already present) and insert_rows.
        5. ONE `tbl.merge_insert(["src","dst","edge_type"]).when_matched_update_all().execute(arrow)`
           for updates (composite-key merge_insert).
        6. ONE `tbl.add(insert_rows)` for new rows.
        7. Returns `dict[tuple[str, str], float]` keyed by canonical sorted pair.

        `delta` accepts a scalar (applied to every pair) or a
        `Sequence[float]` of per-pair deltas. Length mismatch raises
        `ValueError`.
        """
        if edge_type not in EDGE_TYPES:
            raise ValueError(
                f"invalid edge_type {edge_type!r}; must be one of {sorted(EDGE_TYPES)}"
            )

        # Coerce delta to per-pair list. Length validation BEFORE any work.
        if isinstance(delta, (int, float)):
            deltas = [float(delta)] * len(pairs)
        else:
            deltas = [float(d) for d in delta]
            if len(deltas) != len(pairs):
                raise ValueError(
                    f"deltas length {len(deltas)} != pairs length {len(pairs)}"
                )

        if not pairs:
            return {}

        # Coalesce duplicate canonical (src, dst) keys IN-MEMORY: SUM their
        # deltas. `[(a,b), (a,b)]` with delta=0.1 -> cur + 0.2,
        # NOT cur + 0.1. The legacy per-pair tbl.to_pandas() refresh existed
        # purely to support this semantic; in-memory coalescing replaces it.
        coalesced: dict[tuple[str, str], float] = {}
        for (a, b), d in zip(pairs, deltas):
            key = (str(a), str(b))
            canonical = tuple(sorted(key))
            coalesced[canonical] = coalesced.get(canonical, 0.0) + d
        if not coalesced:
            return {}

        tbl = self.db.open_table(EDGES_TABLE)

        # Small-batch fast path: when ``coalesced`` has at most
        # ``_SMALL_BATCH`` distinct keys, issue per-key predicate-filtered
        # scans instead of materialising the entire edges table to pandas.
        # At the typical self-loop call site (1 pair, 1 column lookup)
        # this avoids a full-table to_pandas() per insert. The full-scan
        # path is retained for larger calls where one scan is cheaper
        # than many predicate filters.
        _SMALL_BATCH = 4
        update_rows: list[dict] = []
        insert_rows: list[dict] = []
        new_weights: dict[tuple[str, str], float] = {}
        now = datetime.now(timezone.utc)

        if len(coalesced) <= _SMALL_BATCH:
            for (src_str, dst_str), accum_delta in coalesced.items():
                predicate = (
                    f"src = '{_uuid_literal(src_str)}' "
                    f"AND dst = '{_uuid_literal(dst_str)}' "
                    f"AND edge_type = '{edge_type}'"
                )
                # Scoped scan: only the matching row's weight column.
                # Wrapped in try/except so any store predicate change
                # falls back transparently to the legacy full-scan path
                # via the else-branch below.
                try:
                    df = (
                        tbl.search()
                        .where(predicate)
                        .select(["weight"])
                        .limit(1)
                        .to_pandas()
                    )
                except Exception as exc:  # noqa: BLE001 -- fallback gate
                    logger.warning(
                        "edge predicate-scan fallback for (%s,%s,%s): %s",
                        src_str, dst_str, edge_type, exc, exc_info=True,
                    )
                    df = None
                if df is not None and not df.empty:
                    cur = float(df["weight"].iloc[0])
                    nw = cur + accum_delta
                    update_rows.append({
                        "src": src_str, "dst": dst_str,
                        "edge_type": edge_type,
                        "weight": nw, "updated_at": now,
                    })
                elif df is not None:
                    nw = accum_delta
                    insert_rows.append({
                        "src": src_str, "dst": dst_str,
                        "edge_type": edge_type,
                        "weight": nw, "updated_at": now,
                    })
                else:
                    # Fallback: load full table once for THIS pair only.
                    existing = tbl.to_pandas()
                    mask = (
                        (existing["src"] == src_str)
                        & (existing["dst"] == dst_str)
                        & (existing["edge_type"] == edge_type)
                    ) if len(existing) > 0 else None
                    if mask is not None and mask.any():
                        cur = float(existing.loc[mask, "weight"].iloc[0])
                        nw = cur + accum_delta
                        update_rows.append({
                            "src": src_str, "dst": dst_str,
                            "edge_type": edge_type,
                            "weight": nw, "updated_at": now,
                        })
                    else:
                        nw = accum_delta
                        insert_rows.append({
                            "src": src_str, "dst": dst_str,
                            "edge_type": edge_type,
                            "weight": nw, "updated_at": now,
                        })
                new_weights[(src_str, dst_str)] = nw
        else:
            # Large-batch path: ONE full-table scan at entry. Acceptable
            # at the project's edge-count scale (<= ~5K rows). For batches
            # > _SMALL_BATCH a single scan beats many predicate filters.
            existing = tbl.to_pandas()
            for (src_str, dst_str), accum_delta in coalesced.items():
                if len(existing) > 0:
                    mask = (
                        (existing["src"] == src_str)
                        & (existing["dst"] == dst_str)
                        & (existing["edge_type"] == edge_type)
                    )
                else:
                    mask = None
                if mask is not None and mask.any():
                    cur = float(existing.loc[mask, "weight"].iloc[0])
                    nw = cur + accum_delta
                    update_rows.append({
                        "src": src_str, "dst": dst_str,
                        "edge_type": edge_type,
                        "weight": nw, "updated_at": now,
                    })
                else:
                    nw = accum_delta
                    insert_rows.append({
                        "src": src_str, "dst": dst_str,
                        "edge_type": edge_type,
                        "weight": nw, "updated_at": now,
                    })
                new_weights[(src_str, dst_str)] = nw

        # ONE merge_insert for updates. Composite key (src, dst, edge_type).
        # Fallback to per-row tbl.update preserves correctness on any future
        # store regression.
        if update_rows:
            try:
                upd_arrow = pa.Table.from_pylist(
                    update_rows,
                    schema=pa.schema(
                        [
                            ("src", pa.string()),
                            ("dst", pa.string()),
                            ("edge_type", pa.string()),
                            ("weight", pa.float32()),
                            ("updated_at", pa.timestamp("us", tz="UTC")),
                        ]
                    ),
                )
                _WRITE_RETRYABLE_SIGNALS = (
                    "retryable commit conflict",
                    "too many concurrent writers",
                )
                _WRITE_MAX_RETRIES = 2
                for _w_attempt in range(_WRITE_MAX_RETRIES + 1):
                    try:
                        (
                            tbl.merge_insert(["src", "dst", "edge_type"])
                            .when_matched_update_all()
                            .execute(upd_arrow)
                        )
                        break
                    except (RuntimeError, OSError) as _w_exc:
                        _w_msg = str(_w_exc).lower()
                        if (
                            any(sig in _w_msg for sig in _WRITE_RETRYABLE_SIGNALS)
                            and _w_attempt < _WRITE_MAX_RETRIES
                        ):
                            time.sleep(0.050 + random.uniform(0, 0.050))
                            try:
                                tbl = self.db.open_table(EDGES_TABLE)
                            except (OSError, RuntimeError, ValueError):
                                pass
                        else:
                            raise
            except Exception as exc:  # noqa: BLE001 -- fallback gate, must stay broad
                logger.warning("edge merge_insert fallback triggered: %s", exc, exc_info=True)
                # Fallback: per-row tbl.update. Slower (N versions) but
                # correctness-preserving if merge_insert ever misbehaves.
                for r in update_rows:
                    tbl.update(
                        where=(
                            f"src = '{_uuid_literal(r['src'])}' "
                            f"AND dst = '{_uuid_literal(r['dst'])}' "
                            f"AND edge_type = '{edge_type}'"
                        ),
                        values={
                            "weight": r["weight"],
                            "updated_at": r["updated_at"],
                        },
                    )

        # Buffered batch write: new edge rows go into _edge_buffer and flush on
        # threshold / daemon lifecycle hook. merge_insert (above) stays synchronous.
        if insert_rows:
            buf = _edge_buffer.setdefault(id(self), [])
            buf.extend(insert_rows)
            if should_flush_edge_buffer(id(self)):
                flush_edge_buffer(self)

        return new_weights

    def reinforce_record(
        self,
        record_id: UUID,
        anchor_id: UUID | None = None,
        edge_type: str = "hebbian",
        delta: float = 0.1,
        *,
        is_retrieval: bool = False,
    ) -> dict[tuple[str, str], float]:
        """Typed wrapper: single-record Hebbian reinforcement.

        The canonical reinforcement target for ``memory_capture``
        dedup-on-cos>=0.95. Makes the single-record-reinforcement intent
        explicit at the call site.

        When ``anchor_id is None`` (the dedup-call shape), records a
        ``(record_id, record_id)`` self-loop edge — the canonical
        self-loop semantic for ``capture_turn``'s dedup path.

        When ``anchor_id`` is provided, routes to the existing pair-mode
        contract (``anchor_id`` -> ``record_id``) edge.

        Returns the same ``dict[tuple[str, str], float]`` shape as
        :meth:`boost_edges`. ``edge_type`` validation is delegated to
        ``boost_edges``.

        When ``is_retrieval=True`` (memory_recall hit path), additionally
        stamps the record's ``labile_until`` column with
        ``now + LABILE_WINDOW_SEC`` so the next REM
        ``_step_reconsolidation`` pass can scan it. The default
        ``is_retrieval=False`` preserves existing behaviour for all other
        callers. When ``_load_reconsolidation_config().dry_run`` is True,
        the labile-write is suppressed. A missing ``labile_until`` column
        (half-migrated store) is silently swallowed; any other update
        exception is re-raised so legit migration errors surface loud.
        """
        if anchor_id is None:
            pair = (record_id, record_id)
        else:
            pair = (anchor_id, record_id)
        result = self.boost_edges([pair], delta=delta, edge_type=edge_type)
        # Labile-write — gated on is_retrieval AND not dry_run. Lazy
        # import of _load_reconsolidation_config avoids a circular import
        # at module load (daemon.py transitively imports store.py).
        if is_retrieval:
            try:
                from iai_mcp.daemon_config import _load_reconsolidation_config
                cfg = _load_reconsolidation_config()
                if not cfg.dry_run:
                    labile_until = datetime.now(timezone.utc) + timedelta(
                        seconds=cfg.labile_window_sec
                    )
                    tbl = self.db.open_table(RECORDS_TABLE)
                    try:
                        tbl.update(
                            where=f"id = '{_uuid_literal(record_id)}'",
                            values={"labile_until": labile_until},
                        )
                    except (RuntimeError, ValueError, OSError, KeyError) as exc:
                        # Half-migrated store: column absent. Silently
                        # swallow. Any other failure (encryption, IO) is
                        # re-raised so legit migration errors fail loud.
                        msg = str(exc).lower()
                        column_missing = (
                            "labile_until" in msg
                            or "no such column" in msg
                            or ("column" in msg and "not found" in msg)
                        )
                        if not column_missing:
                            raise
                        logger.debug("labile_until column missing, skipped: %s", exc)
            except ImportError:
                # daemon module unavailable (rare bootstrap edge case);
                # treat as labile-write disabled. Edge-update result is
                # already computed above so we return it unchanged.
                pass
        return result

    def upgrade_tier(
        self,
        record_id: UUID,
        new_tier: str,
        *,
        trigger_event_type: str,
        dry_run: bool = False,
    ) -> bool:
        """Monotone-upward tier upgrade (STC pass).

        Upgrades a record from a lower tier to a strictly higher tier. The
        direction is enforced by ``_STC_TIER_ORDER``; any same-tier or
        downward attempt raises ``ValueError`` and never mutates the row.

        When ``dry_run=True``, the store update is skipped; the
        ``stc_upgrade_pass`` event still fires with ``dry_run_mode=True``
        so observability stays intact.

        Pinned / never_decay are deliberately not checked here -- STC
        upgrades override decay protections.

        Returns
        -------
        True
            On successful row update (or successful dry-run-emit).
        False
            When the record id is unknown (silent no-op, matches
            ``reinforce_record`` semantics).

        Raises
        ------
        ValueError
            When ``new_tier`` is not one of the three known tiers, or when
            the requested move is not strictly upward.
        """
        record = self.get(record_id)
        if record is None:
            return False
        current_tier = record.tier
        if new_tier not in _STC_TIER_ORDER:
            raise ValueError(
                f"upgrade_tier: invalid new_tier {new_tier!r}, "
                f"expected one of {set(_STC_TIER_ORDER.keys())}"
            )
        if _STC_TIER_ORDER[new_tier] <= _STC_TIER_ORDER[current_tier]:
            raise ValueError(
                f"upgrade_tier: refusing non-upgrade "
                f"{current_tier!r} -> {new_tier!r}: tier upgrades are one-directional"
            )

        if not dry_run:
            tbl = self.db.open_table(RECORDS_TABLE)
            try:
                tbl.update(
                    where=f"id = '{_uuid_literal(record_id)}'",
                    values={
                        "tier": new_tier,
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
            except (RuntimeError, ValueError, OSError, KeyError) as exc:
                # Half-migrated store: column absent (mirrors the
                # reinforce_record labile_until swallow).
                # Any other failure (encryption, IO) is re-raised so legit
                # migration errors fail loud.
                msg = str(exc).lower()
                column_missing = (
                    "tier" in msg
                    and (
                        "no such column" in msg
                        or ("column" in msg and "not found" in msg)
                    )
                )
                if not column_missing:
                    raise
                logger.debug("tier column missing in upgrade_tier, skipped: %s", exc)
            else:
                # Mutate in-memory then fire the graph-sync hook so the
                # in-memory graph mirrors the new tier.
                record.tier = new_tier
                self._fire_graph_sync_hook("update", record)

        # Emit exactly one stc_upgrade_pass event per call (including
        # dry-run). Lazy import of write_event avoids the circular import
        # (events.py imports MemoryStore from store.py at module load).
        # NOTE: stc_upgrade_pass must NOT appear in strong_event_types --
        # if a user adds it the post-emit STC hook would recurse without
        # bound. Default config excludes it; defense-in-depth invariant.
        from iai_mcp.events import write_event
        write_event(
            self,
            "stc_upgrade_pass",
            {
                "record_id": str(record_id),
                "from_tier": current_tier,
                "to_tier": new_tier,
                "trigger_event_type": trigger_event_type,
                "dry_run_mode": dry_run,
            },
            severity="info",
            source_ids=[record_id],
        )
        return True

    def add_contradicts_edge(self, original: UUID, new_id: UUID) -> None:
        """Edge-based reconsolidation: original record unchanged."""
        # Flush the record buffer first so both endpoints (original + new_id)
        # are durable in SQLite before the edge row references them. A
        # contradicts edge whose dst is still in _record_buffer would make the
        # superseding record invisible to temporal-validity hydration until the
        # next unrelated drain. The flush is O(buffer_size) worst-case and a
        # no-op when empty; contradicts writes are rare (handful/session).
        flush_record_buffer(self)
        # Buffered batch write: edge row appended to _edge_buffer.
        row = {
            "src": str(original),
            "dst": str(new_id),
            "edge_type": "contradicts",
            "weight": 1.0,
            "updated_at": datetime.now(timezone.utc),
        }
        _edge_buffer.setdefault(id(self), []).append(row)
        # Contradicts edges are rare and load-bearing: the superseded-original
        # recall path reads them from the edges TABLE, not the in-memory
        # buffer. The size/time threshold that
        # drains hebbian edges is never reached on small stores, which would
        # leave a contradiction invisible to recall until an unrelated edge
        # write happened to trip the flush. Flush this edge immediately so the
        # reconsolidation link is durable the instant it is recorded; the
        # batch-coalescing win is negligible for the handful of contradicts
        # edges a session produces.
        flush_edge_buffer(self)

    # ---------------------------------------------------------------- helpers

    def _to_row(self, r: MemoryRecord) -> dict:
        # Encrypt sensitive columns with AD = record.id.
        # literal_surface, provenance_json, profile_modulation_gain_json
        # are the three encrypted columns on the records table.
        literal_ct = self._encrypt_for_record(r.id, r.literal_surface)
        provenance_plain = json.dumps(r.provenance)
        provenance_ct = self._encrypt_for_record(r.id, provenance_plain)
        gain_plain = json.dumps(r.profile_modulation_gain or {})
        gain_ct = self._encrypt_for_record(r.id, gain_plain)
        return {
            "id": str(r.id),
            "tier": r.tier,
            "literal_surface": literal_ct,
            "aaak_index": r.aaak_index,
            "embedding": [float(x) for x in r.embedding],
            # structure_hv is raw bytes (D=10000 BSC packed to 1250 bytes).
            # Empty bytes default for pre-migration / lazy bind.
            "structure_hv": bytes(r.structure_hv or b""),
            "community_id": str(r.community_id) if r.community_id else "",
            "centrality": float(r.centrality),
            "detail_level": int(r.detail_level),
            "pinned": bool(r.pinned),
            "stability": float(r.stability),
            "difficulty": float(r.difficulty),
            "last_reviewed": r.last_reviewed,
            "never_decay": bool(r.never_decay),
            "never_merge": bool(r.never_merge),
            "provenance_json": provenance_ct,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "tags_json": json.dumps(r.tags),
            # v2 columns
            "language": str(r.language),
            "s5_trust_score": float(r.s5_trust_score),
            "profile_modulation_gain_json": gain_ct,
            "schema_version": int(r.schema_version),
            # schema_bypass: schema-fit flag stamped by _maybe_tag_schema_bypass
            # JUST BEFORE insert(). Private-attribute escape hatch so the
            # MemoryRecord dataclass and _from_row round-trip stay untouched.
            "schema_bypass": bool(getattr(r, "_schema_bypass", False)),
            # labile_until: reconsolidation labile window. Default null on
            # every insert; memory_recall path uses an UPDATE to set it to
            # (now + LABILE_WINDOW_SEC) on every hit. The _labile_until
            # escape hatch allows seeding at insert time when needed.
            "labile_until": getattr(r, "_labile_until", None),
            # wing/room/drawer: spatial scaffold. Default null on every
            # insert; _maybe_spatial_tag sets the public attributes on the
            # MemoryRecord JUST BEFORE this _to_row runs when auto_tag=True.
            # getattr fallback keeps legacy callers writing NULL into the columns.
            "wing": getattr(r, "wing", None),
            "room": getattr(r, "room", None),
            "drawer": getattr(r, "drawer", None),
            # Lilli HD/HDC codec metadata boundary (V5 schema).
            "hv_tier": r.hv_tier,
            "structure_hv_payload": bytes(r.structure_hv_payload or b""),
        }

    def _maybe_tag_schema_bypass(self, record: MemoryRecord) -> None:
        """Tag a fresh INSERT record's schema-fit flag.

        Called from ``insert()`` AFTER ``pattern_separation_gate`` has fired
        ``GateAction.INSERT`` and BEFORE the row is committed. When the
        cosine of ``record.embedding`` against any persisted community
        centroid is ``>= cfg.schema_bypass_cos_threshold``, the record's
        ``_schema_bypass`` private attribute is set to True so that
        ``_to_row`` writes ``schema_bypass=True`` into the row.

        Centroids accessor: ``runtime_graph_cache.try_load(store)`` — a cheap
        local read of the on-disk Leiden cache. When the cache is cold,
        centroids are treated as the empty dict and ``schema_bypass`` stays
        False. ``build_runtime_graph(store)`` is never called here — the
        heavy rebuild on the insert hot path would dominate write latency.

        When ``cfg.dry_run`` is True, the ``schema_bypass_pass`` event still
        fires with ``dry_run_mode=True`` so operators can observe the
        candidate rate, but ``_schema_bypass`` is never set and no row
        is mutated.

        Defensive: any exception in the probe path leaves
        ``_schema_bypass`` unset (defaults to False via getattr fallback).
        The schema-bypass flag is purely advisory; a probe failure must
        never abort the insert.
        """
        max_cos: float = 0.0
        tagged: bool = False
        dry_run: bool = False
        try:
            from iai_mcp.daemon_config import _load_reconsolidation_config
            cfg = _load_reconsolidation_config()
            dry_run = bool(cfg.dry_run)
            # Cheap centroids lookup. try_load returns
            # (assignment, rich_club, node_payload, max_degree) | None.
            centroids: dict[Any, list[float]] = {}
            try:
                from iai_mcp import runtime_graph_cache
                cached = runtime_graph_cache.try_load(self)
                if cached is not None:
                    assignment = cached[0]
                    centroids = (
                        getattr(assignment, "community_centroids", {}) or {}
                    )
            except (OSError, ValueError, ImportError, RuntimeError) as exc:
                logger.debug("schema-bypass centroid load skipped: %s", exc)
                centroids = {}
            if centroids:
                # Defensive dim-skip for legacy 1024d centroids vs current
                # 384d store: only compare same-dim vectors. Compute max
                # over centroid values; zero-norm centroids contribute 0.
                emb = record.embedding
                emb_dim = len(emb)
                # Single-pass numpy cosine to keep cost flat at small N.
                import numpy as _np
                e_arr = _np.asarray(emb, dtype=_np.float32)
                e_norm = float(_np.linalg.norm(e_arr))
                if e_norm > 0.0:
                    for cent in centroids.values():
                        if cent is None or len(cent) != emb_dim:
                            continue
                        c_arr = _np.asarray(cent, dtype=_np.float32)
                        c_norm = float(_np.linalg.norm(c_arr))
                        if c_norm <= 0.0:
                            continue
                        sim = float(
                            _np.dot(e_arr, c_arr) / (e_norm * c_norm)
                        )
                        if sim > max_cos:
                            max_cos = sim
                if max_cos >= float(cfg.schema_bypass_cos_threshold):
                    tagged = True
            # Mutate the record only on the live path.
            if tagged and not dry_run:
                record._schema_bypass = True
            else:
                # Leave attribute unset so _to_row's getattr fallback
                # writes False (the default). Explicit False also OK and
                # equally compatible with _to_row.
                pass
        except Exception as exc:  # noqa: BLE001 -- advisory tagger, must never abort insert
            logger.warning("schema-bypass tagging failed (advisory): %s", exc, exc_info=True)
            return
        # Event emit — wrapped in try/except so emit failures cannot
        # corrupt the insert path. Imported lazily to avoid circular import
        # with events.py (events.py imports from store).
        try:
            from iai_mcp.events import write_event
            write_event(
                self,
                "schema_bypass_pass",
                {
                    "record_id": str(record.id),
                    "max_cos": float(max_cos),
                    "tagged": bool(tagged and not dry_run),
                    "dry_run_mode": bool(dry_run),
                },
                severity="info",
            )
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.debug("schema_bypass_pass event emit failed: %s", exc)

    def _maybe_spatial_tag(self, record: MemoryRecord) -> None:
        """Tag a fresh INSERT record with wing/room/drawer.

        Called from ``insert()`` AFTER ``_maybe_tag_schema_bypass`` and
        BEFORE the row is committed. When ``IAI_MCP_SPATIAL_AUTO_TAG``
        is true AND the record has no pre-existing wing/room/drawer attributes,
        runs :func:`iai_mcp.spatial_tagger.SpatialTagger.tag` against the
        record's provenance ``source_path`` and stamps the public attributes on
        the MemoryRecord so ``_to_row`` writes them into the row.

        Source-path lookup: ``record.provenance`` is the parsed ``list[dict]``
        field. We walk the list and take the first dict carrying a
        ``source_path`` key. SpatialTagger handles ``None`` natively.

        When ``config.dry_run`` is True, the ``spatial_tag_pass`` event still
        fires with ``dry_run_mode=True`` so operators can observe the inferred
        tuple, but ``record.wing/room/drawer`` are never mutated.

        Pre-existing-field guard: when the caller already set any of
        ``record.wing`` / ``record.room`` / ``record.drawer`` to a non-None
        value, we short-circuit (no inference, no event).

        Defensive: every exception path leaves the record unmutated. The
        spatial scaffold is advisory; a tagger or event-emit failure must
        never abort the insert.
        """
        try:
            from iai_mcp.daemon_config import _load_spatial_config
            config = _load_spatial_config()
        except Exception as exc:  # noqa: BLE001 -- advisory tagger, must never abort insert
            # Config load failure (malformed env var, etc.) MUST NOT
            # abort the insert. Fall through to no-op.
            logger.warning("spatial config load failed (advisory): %s", exc, exc_info=True)
            return
        if not config.auto_tag:
            return

        # Pre-existing-field guard: never overwrite caller-supplied
        # spatial attributes. getattr with default None keeps legacy
        # MemoryRecord instances (no spatial attrs at all) on the
        # inference path.
        existing_wing = getattr(record, "wing", None)
        existing_room = getattr(record, "room", None)
        existing_drawer = getattr(record, "drawer", None)
        if (
            existing_wing is not None
            or existing_room is not None
            or existing_drawer is not None
        ):
            return

        # Source-path extraction. record.provenance is list[dict] at
        # this stage (the JSON-string form only exists post-_to_row).
        # Walk entries; first dict carrying a `source_path` wins.
        # Wrapped in try/except: any malformed provenance entry falls
        # back to source_path=None.
        source_path: str | None = None
        try:
            prov = getattr(record, "provenance", None)
            if isinstance(prov, list):
                for entry in prov:
                    if isinstance(entry, dict) and "source_path" in entry:
                        candidate = entry.get("source_path")
                        if isinstance(candidate, str) and candidate.strip():
                            source_path = candidate
                            break
        except (TypeError, ValueError, AttributeError) as exc:
            logger.debug("spatial source_path extraction skipped: %s", exc)
            source_path = None

        # Inference + (conditional) mutation. SpatialTagger.tag is pure
        # and returns (None, None, None) when source_path is None / empty,
        # so we can always invoke it without a guard. The default_wing
        # kwarg threads through the env-configured fallback.
        wing: str | None = None
        room: str | None = None
        drawer: str | None = None
        try:
            from iai_mcp.spatial_tagger import SpatialTagger
            wing, room, drawer = SpatialTagger.tag(
                record,
                source_path,
                default_wing=config.default_wing,
            )
        except Exception as exc:  # noqa: BLE001 -- advisory tagger, must never abort insert
            # Tagger fault: leave the inferred tuple at None and fall
            # through to the event-emit branch. The record stays
            # unmutated; the event records the no-signal outcome so
            # operators can observe the failure.
            logger.warning("spatial tagger inference failed (advisory): %s", exc, exc_info=True)
            wing, room, drawer = (None, None, None)

        # Dry-run guard: when dry_run=True, NEVER mutate the record.
        # The event still fires below with dry_run_mode=True so the
        # shadow-deploy observability path stays intact.
        if not config.dry_run:
            record.wing = wing
            record.room = room
            record.drawer = drawer

        # Event emit -- wrapped in try/except so emit failures cannot
        # corrupt the insert path. Lazy import of write_event avoids
        # circular import with events.py which imports from store.
        try:
            from iai_mcp.events import write_event
            write_event(
                self,
                "spatial_tag_pass",
                {
                    "record_id": str(record.id),
                    "wing": wing,
                    "room": room,
                    "drawer": drawer,
                    "source_path": source_path,
                    "dry_run_mode": bool(config.dry_run),
                },
                severity="info",
            )
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.debug("spatial_tag_pass event emit failed: %s", exc)

    def _from_row(self, row: dict) -> MemoryRecord:
        from uuid import UUID as _UUID

        import pandas as pd  # local import: only hot on reads

        def _safe_int(val: Any, default: int) -> int:
            """Convert val to int, returning default for None/NaN/unconvertible."""
            if val is None:
                return default
            try:
                fval = float(val)
                if fval != fval:  # NaN check (NaN != NaN)
                    return default
                return int(fval)
            except (TypeError, ValueError):
                return default

        def _parse_ts(val: Any) -> datetime | None:
            """Coerce ISO TEXT / Timestamp / datetime / NaT / None to tz-aware UTC datetime."""
            if val is None:
                return None
            try:
                if pd.isna(val):
                    return None
            except (TypeError, ValueError):
                pass
            if isinstance(val, datetime):
                return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
            if hasattr(val, "to_pydatetime"):
                dt = val.to_pydatetime()
                return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
            try:
                dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                return None
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

        # Partial-row safety: iter_records consumers may project a subset of
        # columns; any non-projected column is absent from the row dict. `id`
        # is the primary key and projection without it is a caller bug -- fail
        # loud.
        if "id" not in row:
            raise KeyError(
                "iter_records consumer must include 'id' in column projection"
            )

        # Prefer the `structure_hv` (pa.binary()) column. Legacy stores may
        # not have it populated; surface b"" so MemoryRecord stays valid.
        structure_raw = row.get("structure_hv")
        if structure_raw is None:
            structure_hv = b""
        elif isinstance(structure_raw, (bytes, bytearray)):
            structure_hv = bytes(structure_raw)
        else:
            structure_hv = b""

        # Lilli HD/HDC codec metadata (V5 schema).
        # Graceful-degradation: both fields default if either is missing or
        # inconsistent.  On inconsistency, emit a telemetry event then fall
        # back to hv_tier='bsc' + structure_hv_payload=b'' — NEVER raise.
        from iai_mcp import events as _ev_mod
        _codec_event_kind = getattr(
            _ev_mod,
            "TELEMETRY_CODEC_MARKER_MISSING",
            "codec_marker_missing",
        )
        hv_tier_raw = row.get("hv_tier")
        structure_hv_payload_raw = row.get("structure_hv_payload")
        _codec_reason: str | None = None

        if hv_tier_raw is None:
            # Column absent (pre-V5 store) — silent back-compat, no telemetry.
            hv_tier = "bsc"
            structure_hv_payload = b""
        elif hv_tier_raw not in HV_TIER_ENUM:
            _codec_reason = f"hv_tier {hv_tier_raw!r} not in HV_TIER_ENUM; reset to bsc"
            hv_tier = "bsc"
            structure_hv_payload = b""
        elif structure_hv_payload_raw is not None and not isinstance(
            structure_hv_payload_raw, (bytes, bytearray)
        ):
            _codec_reason = (
                f"structure_hv_payload expected bytes, "
                f"got {type(structure_hv_payload_raw).__name__}"
            )
            hv_tier = "bsc"
            structure_hv_payload = b""
        else:
            hv_tier = str(hv_tier_raw)
            structure_hv_payload = (
                bytes(structure_hv_payload_raw)
                if isinstance(structure_hv_payload_raw, (bytes, bytearray))
                else b""
            )

        if _codec_reason is not None:
            try:
                _ev_mod.write_event(
                    self,
                    kind=_codec_event_kind,
                    data={
                        "record_id": row.get("id", ""),
                        "reason": _codec_reason,
                    },
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 — telemetry must never crash _from_row
                pass

        _community_val = row.get("community_id")
        # Guard against pandas NaN (truthy but not a string) in addition to
        # empty string / None.
        try:
            import math as _math
            if _community_val is not None and not isinstance(_community_val, str):
                if _math.isnan(float(_community_val)):
                    _community_val = None
        except (TypeError, ValueError):
            pass
        community_raw = (_community_val or "")
        community_id = _UUID(community_raw) if community_raw and isinstance(community_raw, str) else None

        # Back-compat read path: a legacy v1 row (or externally written row)
        # may lack language / s5_trust_score / profile_modulation_gain_json /
        # schema_version. Fill with defaults: language="en", s5=0.5, gain={},
        # version=1.
        #
        # Migration note: for schema_version=1 rows with empty language, we
        # preserve the empty string on the in-memory record so migrate_v1_to_v2
        # can set language="en" through its normal write path. MemoryRecord
        # __post_init__ requires non-empty language, so we route through a
        # placeholder and zero it back out before returning. For v2+ rows
        # missing a schema_version we default to "en" as before -- those paths
        # don't run migration.
        lang_raw = row.get("language")
        raw_version = row.get("schema_version")
        try:
            version_int = int(raw_version) if raw_version is not None else SCHEMA_VERSION_CURRENT
        except (TypeError, ValueError):
            version_int = SCHEMA_VERSION_CURRENT
        schema_version = version_int

        is_empty_language = lang_raw is None or (isinstance(lang_raw, str) and lang_raw == "")
        if is_empty_language and schema_version == 1:
            # v1 legacy row -> preserve empty so migration can re-detect.
            # We use a placeholder to satisfy __post_init__ then reset below.
            language = "__LEGACY_EMPTY__"
        elif is_empty_language:
            language = "en"
        else:
            language = str(lang_raw)

        s5_raw = row.get("s5_trust_score")
        try:
            _s5 = float(s5_raw) if s5_raw is not None else 0.5
            s5_trust_score = _s5 if (_s5 == _s5 and 0.0 <= _s5 <= 1.0) else 0.5  # NaN guard
        except (TypeError, ValueError):
            s5_trust_score = 0.5

        # Decrypt profile_modulation_gain_json if it carries the
        # iai:enc:v1: prefix (mixed plaintext/ciphertext on migrated stores).
        from uuid import UUID as _UUID2
        _row_uuid = _UUID2(row["id"])
        gain_raw = row.get("profile_modulation_gain_json") or "{}"
        if is_encrypted(gain_raw):
            gain_raw = self._decrypt_for_record(_row_uuid, gain_raw)
        try:
            profile_modulation_gain = json.loads(gain_raw) or {}
        except (TypeError, json.JSONDecodeError):
            profile_modulation_gain = {}

        # Coerce last_reviewed from ISO TEXT / NaT / None to tz-aware datetime or None.
        last_reviewed = _parse_ts(row.get("last_reviewed"))

        # Decrypt literal_surface + provenance_json if encrypted.
        # Bracket access hardened to defensive .get() so column-projected
        # reads (where these columns may be absent) do not KeyError.
        # is_encrypted("") and is_encrypted("[]") are both False, so
        # empty defaults flow through as plaintext untouched.
        row_uuid = _UUID(row["id"])
        literal_raw = row.get("literal_surface", "")
        if is_encrypted(literal_raw):
            literal_raw = self._decrypt_for_record(row_uuid, literal_raw)
        provenance_raw = row.get("provenance_json") or "[]"
        if is_encrypted(provenance_raw):
            provenance_raw = self._decrypt_for_record(row_uuid, provenance_raw)
        try:
            provenance_list = json.loads(provenance_raw) if provenance_raw else []
        except (TypeError, json.JSONDecodeError):
            provenance_list = []

        rec = MemoryRecord(
            id=row_uuid,
            tier=row.get("tier", "episodic"),
            literal_surface=literal_raw,
            aaak_index=row.get("aaak_index") or "",
            embedding=(
                list(row["embedding"])
                if row.get("embedding") is not None
                else []
            ),
            community_id=community_id,
            centrality=float(row.get("centrality", 0.0) or 0.0),
            detail_level=_safe_int(row.get("detail_level"), 1),
            pinned=bool(row.get("pinned", False) or False),
            stability=float(row.get("stability") or 0.0),
            difficulty=float(row.get("difficulty") or 0.0),
            last_reviewed=last_reviewed,
            never_decay=bool(row.get("never_decay", False) or False),
            never_merge=bool(row.get("never_merge", False) or False),
            provenance=provenance_list,
            created_at=_parse_ts(row.get("created_at")) or datetime.now(timezone.utc),
            updated_at=_parse_ts(row.get("updated_at")) or datetime.now(timezone.utc),
            tags=json.loads((row.get("tags_json") or "[]") if isinstance(row.get("tags_json"), str) else "[]"),
            language=language,
            s5_trust_score=s5_trust_score,
            profile_modulation_gain=profile_modulation_gain,
            schema_version=schema_version,
            structure_hv=structure_hv,
            hv_tier=hv_tier,
            structure_hv_payload=structure_hv_payload,
            embedding_pending=_safe_int(row.get("embedding_pending"), 0),
        )
        if language == "__LEGACY_EMPTY__":
            rec.language = ""  # post-construction: signal to migration path
        return rec


# ---------------------------------------------------------------------------
# Module-level write buffer for the RECORDS table.
# Keyed by id(store); flushed from daemon lifecycle hooks (WAKE / tick / shutdown).
# Rows enter the buffer ALREADY ENCRYPTED (see _to_row + _encrypt_for_record).
# On hard kill the buffer is lost — matches the documented event-buffer loss
# contract (no WAL backing for in-process buffers).
#
# Hard-kill data-loss window: bounded by min(buffer_size, time_since_flush).
# The buffer-size cap (env IAI_MCP_RECORD_BUFFER_MAX) defaults to 500, so up
# to 500 records can be lost on hard kill between flushes — 5x the original
# 100-default cap.  Throughput-vs-durability trade-off documented in
# CHANGELOG `[Unreleased]` under the buffer-default bump entry.  Operators
# with strict durability needs can pin back to 100 via the env var.
# ---------------------------------------------------------------------------

_record_buffer: dict[int, list[dict]] = {}
_record_last_flush_at: dict[int, datetime] = {}


def flush_record_buffer(store: "MemoryStore") -> int:
    """Flush buffered records to the records table in one batch write.

    Returns the number of rows flushed. Safe to call when the buffer is empty.
    Failure (OSError / RuntimeError / ValueError) is logged at WARNING; the
    rows are dropped (matches the EVENTS buffer fail-loud-but-don't-crash
    contract — re-queue on exception would risk duplicates on retry).

    Runs entirely under the shared `_BUFFER_LOCK` (defined in events.py;
    imported function-locally to avoid an import cycle) so a concurrent
    `MemoryStore.close()` from another thread cannot interleave a buffer
    pop between this body's pop and the store write.
    """
    from iai_mcp.events import _BUFFER_LOCK

    with _BUFFER_LOCK:
        store_id = id(store)
        pending = _record_buffer.pop(store_id, [])
        if not pending:
            return 0
        try:
            store.db.open_table(RECORDS_TABLE).add(pending)
            _record_last_flush_at[store_id] = datetime.now(timezone.utc)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "flush_record_buffer_failed",
                extra={"n": len(pending), "err": str(exc)[:120]},
            )
        # Emit telemetry once per successful flush (buffered=False — never feed
        # the buffer back into itself; recursion risk).
        if pending:
            try:
                from iai_mcp.events import write_event
                write_event(
                    store,
                    "lance_buffer_flush",
                    {"table": "records", "count": len(pending)},
                    severity="info",
                    buffered=False,
                )
            except Exception as exc:  # noqa: BLE001 -- telemetry MUST NOT crash flush
                logger.debug("lance_buffer_flush telemetry failed: %s", str(exc)[:120])
        return len(pending)


def should_flush_record_buffer(store_id: int, max_size: int | None = None) -> bool:
    """Return True iff the records buffer has reached the size threshold.

    Resolves max_size from IAI_MCP_RECORD_BUFFER_MAX env var (default 500).

    The default raised from 100 → 500 after the buffered-writes path
    landed: at high N the marginal cost of a flush is dominated by store
    MVCC overhead, not per-row Python work, so larger batches amortise
    the transaction cost more efficiently.
    Workloads with strict latency or per-tick budget bounds can pin it
    back to 100 via the env var.
    """
    if max_size is None:
        try:
            max_size = int(os.environ.get("IAI_MCP_RECORD_BUFFER_MAX", "500"))
        except ValueError:
            max_size = 500
    return len(_record_buffer.get(store_id, [])) >= max_size


def should_flush_record_buffer_by_time(
    store_id: int,
    last_flush_at: datetime | None,
    max_age_sec: float = 5.0,
) -> bool:
    """Return True iff the records buffer is non-empty and aged past max_age_sec.

    None last_flush_at means "never flushed": immediately due iff non-empty.
    """
    if not _record_buffer.get(store_id):
        return False
    if last_flush_at is None:
        return True
    return (datetime.now(timezone.utc) - last_flush_at).total_seconds() >= max_age_sec


# ---------------------------------------------------------------------------
# Module-level write buffer for the EDGES table.
# Buffers new-row inserts ONLY. The merge_insert update path stays synchronous
# (it has read-before-write conflict semantics that cannot be deferred).
# Same documented-loss-on-hard-kill contract as the RECORDS / EVENTS buffers.
# ---------------------------------------------------------------------------

_edge_buffer: dict[int, list[dict]] = {}
_edge_last_flush_at: dict[int, datetime] = {}


def flush_edge_buffer(store: "MemoryStore") -> int:
    """Flush buffered edges to the edges table in one batch write.

    Returns the number of rows flushed. Safe to call when the buffer is empty.
    Uses merge_insert(key=[src,dst,edge_type]) so duplicate edges are handled
    via ON CONFLICT DO UPDATE SET weight=excluded.weight, updated_at=excluded.updated_at
    (latest-wins upsert).  No IntegrityError; no batch drops.
    Failure (OSError / RuntimeError / ValueError) is logged at WARNING; the
    rows are dropped (matches the EVENTS / RECORDS buffer fail-loud-but-don't-crash
    contract — re-queue on exception would risk duplicates on retry).

    Runs entirely under the shared `_BUFFER_LOCK` (defined in events.py;
    imported function-locally to avoid an import cycle) so a concurrent
    `MemoryStore.close()` from another thread cannot interleave a buffer
    pop between this body's pop and the store write.
    """
    from iai_mcp.events import _BUFFER_LOCK

    with _BUFFER_LOCK:
        store_id = id(store)
        pending = _edge_buffer.pop(store_id, [])
        if not pending:
            return 0
        try:
            store.db.open_table(EDGES_TABLE).merge_insert(["src", "dst", "edge_type"]).execute(pending)
            _edge_last_flush_at[store_id] = datetime.now(timezone.utc)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "flush_edge_buffer_failed",
                extra={"n": len(pending), "err": str(exc)[:120]},
            )
        # Emit telemetry once per successful flush (buffered=False — never feed
        # the buffer back into itself; recursion risk).
        if pending:
            try:
                from iai_mcp.events import write_event
                write_event(
                    store,
                    "lance_buffer_flush",
                    {"table": "edges", "count": len(pending)},
                    severity="info",
                    buffered=False,
                )
            except Exception as exc:  # noqa: BLE001 -- telemetry MUST NOT crash flush
                logger.debug("lance_buffer_flush telemetry failed: %s", str(exc)[:120])
        return len(pending)


def should_flush_edge_buffer(store_id: int, max_size: int | None = None) -> bool:
    """Return True iff the edges buffer has reached the size threshold.

    Resolves max_size from IAI_MCP_EDGE_BUFFER_MAX env var (default 500).

    Same rationale as the records buffer: at high N the marginal cost of
    a flush is dominated by store MVCC overhead, so larger batches
    amortise the transaction cost more efficiently.  Pin back to 100 via
    the env var when latency or per-tick budget constraints require it.
    """
    if max_size is None:
        try:
            max_size = int(os.environ.get("IAI_MCP_EDGE_BUFFER_MAX", "500"))
        except ValueError:
            max_size = 500
    return len(_edge_buffer.get(store_id, [])) >= max_size


def should_flush_edge_buffer_by_time(
    store_id: int,
    last_flush_at: datetime | None,
    max_age_sec: float = 5.0,
) -> bool:
    """Return True iff the edges buffer is non-empty and aged past max_age_sec.

    None last_flush_at means "never flushed": immediately due iff non-empty.
    """
    if not _edge_buffer.get(store_id):
        return False
    if last_flush_at is None:
        return True
    return (datetime.now(timezone.utc) - last_flush_at).total_seconds() >= max_age_sec
