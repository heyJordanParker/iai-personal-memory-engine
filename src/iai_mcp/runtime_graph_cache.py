from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from cryptography.exceptions import InvalidTag

logger = logging.getLogger(__name__)

from iai_mcp.crypto import (
    CryptoKey,
    decrypt_field,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.types import SCHEMA_VERSION_CURRENT

preload_ready: threading.Event = threading.Event()

rebuild_ready: threading.Event = threading.Event()


CACHE_VERSION: str = "62-02-v5"

_STALENESS_WINDOW: int = 10
LEGACY_CACHE_VERSION_PLAINTEXT: str = "06-02-v1"

_CACHE_AAD: bytes = b"runtime-graph-cache:v3"

CACHE_FILENAME: str = "runtime_graph_cache.json"


_FUSE_MAX_AGE_SECONDS: float = 25.0 * 3600.0

_FUSE_DIRTY_THRESHOLD: int = 50

_dirty_counter: int = 0
_DIRTY_COUNTER_LOCK = threading.Lock()


def increment_dirty_counter() -> None:
    global _dirty_counter  # noqa: PLW0603
    with _DIRTY_COUNTER_LOCK:
        _dirty_counter += 1


def reset_dirty_counter() -> None:
    global _dirty_counter  # noqa: PLW0603
    with _DIRTY_COUNTER_LOCK:
        _dirty_counter = 0


def get_dirty_counter() -> int:
    with _DIRTY_COUNTER_LOCK:
        return _dirty_counter


# One shared graph instance reused across refreshes so the allocator footprint
# stays bounded (a fresh instance per cycle fragments the heap arenas). The lock
# serializes concurrent refreshes so adjacency cannot be corrupted mid-rebuild.
_persistent_graph = None
_PERSISTENT_GRAPH_LOCK = threading.Lock()


def _get_persistent_graph():
    """Module-level reusable graph instance.

    No longer fed by the periodic rebuild path — the rebuild runs in a child
    process and reclaims its own address space. Kept callable for
    backward-compatibility with existing fixture-reset tests and any future
    in-parent graph consumers.
    """
    global _persistent_graph  # noqa: PLW0603
    if _persistent_graph is None:
        from iai_mcp.graph import MemoryGraph
        _persistent_graph = MemoryGraph()
    return _persistent_graph


# Worker timeouts. The rebuild itself is a background sleep-time operation
# and recall is served from the last-good snapshot throughout; the watchdog
# exists to catch a hung worker, not a slow-but-progressing one. The
# centrality + rich_club + community-detection compute scales super-linearly
# with graph size, so we use a base allowance plus a per-1k-nodes ramp:
#   timeout = base + per_1k * (active_records_count / 1000)
# capped at WORKER_TIMEOUT_MAX_S. First spawn after daemon boot uses a
# slightly larger base to absorb numba JIT cold-start. All reads and writes
# of `_first_spawn_seen` happen under `_PERSISTENT_GRAPH_LOCK`, which
# already serializes rebuilds — no new lock.
_WORKER_TIMEOUT_BASE_S: float = 60.0
_WORKER_TIMEOUT_FIRST_BASE_S: float = 120.0
# Coefficient calibrated to the measured per-1k-nodes cost of
# `MemoryGraph.centrality()` (Brandes betweenness, O(V*E)) on this hardware
# plus `detect_communities` overhead. Capped at WORKER_TIMEOUT_MAX_S so a
# truly hung worker is still caught in finite time.
_WORKER_TIMEOUT_PER_1K_NODES_S: float = 35.0
_WORKER_TIMEOUT_MAX_S: float = 3600.0
_first_spawn_seen: bool = False


class WorkerCrashedError(RuntimeError):
    """Child worker exited with a non-zero exit code."""


class WorkerTimeoutError(RuntimeError):
    """Child worker did not produce a complete result within the timeout."""


def _worker_entry_indirection(conn) -> None:
    """Picklable spawn target.

    The worker module is imported only inside the child after spawn, so the
    parent process itself never loads the worker module until spawn time.
    """
    from iai_mcp.runtime_graph_cache_worker import _worker_entry
    _worker_entry(conn)


def _resolve_timeout(active_records_count: int = 0) -> float:
    """Size-scaled watchdog timeout.

    Base plus a per-1k-active-records ramp, capped at
    `_WORKER_TIMEOUT_MAX_S`. First spawn after daemon boot uses a larger
    base to absorb numba JIT cold-start.
    """
    base = _WORKER_TIMEOUT_FIRST_BASE_S if not _first_spawn_seen else _WORKER_TIMEOUT_BASE_S
    ramp = _WORKER_TIMEOUT_PER_1K_NODES_S * (max(0, active_records_count) / 1000.0)
    return min(base + ramp, _WORKER_TIMEOUT_MAX_S)


def _terminate_worker(process) -> None:
    """Idempotent terminate-then-kill of the worker process."""
    try:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
    except Exception:  # noqa: BLE001 -- worker cleanup must not raise
        pass


def _drain_worker_result(parent_conn, timeout: float) -> dict:
    """Drain the chunked compact result envelope into a parent-side dict.

    Raises WorkerTimeoutError if the worker has not emitted the `done`
    terminator within `timeout` seconds. Any `error` envelope is converted
    into a RuntimeError so the caller can dispose.
    """
    import time
    from uuid import UUID

    import numpy as np

    deadline = time.perf_counter() + timeout
    community_table_uuids: list = []
    community_centroids: dict = {}
    assignments: dict = {}
    backend: str | None = None
    top_communities: list = []
    mid_regions: dict = {}
    rich_club: list = []
    max_degree: int = 0
    done = False

    while not done:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise WorkerTimeoutError(
                f"worker did not complete within {timeout:.1f}s"
            )
        if not parent_conn.poll(min(remaining, 1.0)):
            continue
        envelope = parent_conn.recv()
        kind, payload = envelope
        if kind == "community_table":
            for comm_bytes, centroid_bytes in payload:
                cu = UUID(bytes=comm_bytes)
                community_table_uuids.append(cu)
                if centroid_bytes is None:
                    community_centroids[cu] = []
                else:
                    community_centroids[cu] = np.frombuffer(
                        centroid_bytes, dtype=np.float32
                    ).tolist()
        elif kind == "assign":
            for node_bytes, comm_idx in payload:
                assignments[UUID(bytes=node_bytes)] = int(comm_idx)
        elif kind == "assign_end":
            continue
        elif kind == "backend":
            backend = str(payload)
        elif kind == "top_communities":
            top_communities = [UUID(bytes=b) for b in payload]
        elif kind == "mid_regions":
            for comm_bytes, member_bytes_list in payload:
                mid_regions[UUID(bytes=comm_bytes)] = [
                    UUID(bytes=mb) for mb in member_bytes_list
                ]
        elif kind == "rich_club":
            rich_club = [UUID(bytes=b) for b in payload]
        elif kind == "max_degree":
            max_degree = int(payload)
        elif kind == "done":
            done = True
        elif kind == "error":
            raise RuntimeError(f"worker reported error: {payload!r}")
        else:
            raise RuntimeError(f"worker emitted unknown envelope kind: {kind!r}")

    node_to_community: dict = {}
    for node_uuid, idx in assignments.items():
        node_to_community[node_uuid] = community_table_uuids[idx]

    return {
        "node_to_community": node_to_community,
        "community_centroids": community_centroids,
        "backend": backend if backend is not None else "flat",
        "top_communities": top_communities,
        "mid_regions": mid_regions,
        "rich_club": rich_club,
        "max_degree": max_degree,
    }


MAX_CACHE_BYTES: int = 10 * 1024 * 1024
_SNAPSHOT_NEAR_LIMIT_FRACTION: float = 0.80
_snapshot_near_limit_last_gen: int = -1


def _cache_path(store: Any) -> Path:
    root = getattr(store, "root", None)
    if root is None:
        root = Path.cwd()
    return Path(root) / CACHE_FILENAME


def _cache_encryption_key(store: Any) -> bytes:
    cached_via_store = getattr(store, "_crypto_key", None)
    if isinstance(cached_via_store, (bytes, bytearray)) and len(cached_via_store) == 32:
        return bytes(cached_via_store)
    if hasattr(store, "_key") and callable(store._key):
        try:
            key = store._key()
            if isinstance(key, (bytes, bytearray)) and len(key) == 32:
                return bytes(key)
        except (OSError, ValueError, RuntimeError):
            pass
    user_id = getattr(store, "user_id", "default") or "default"
    return CryptoKey(user_id=user_id).get_or_create()


def _cache_key(store: Any) -> tuple:
    try:
        records_count = int(store.active_records_count())
    except (OSError, ValueError, KeyError, AttributeError):
        try:
            records_count = int(store.db.open_table("records").count_rows())
        except (OSError, ValueError, KeyError, AttributeError):
            records_count = -1
    try:
        edges_count = int(store.db.open_table("edges").count_rows())
    except (OSError, ValueError, KeyError, AttributeError):
        edges_count = -1
    embed_dim = int(getattr(store, "embed_dim", 0))
    rc_window = records_count // _STALENESS_WINDOW if records_count >= 0 else records_count
    ec_window = edges_count // _STALENESS_WINDOW if edges_count >= 0 else edges_count
    return (
        rc_window,
        ec_window,
        SCHEMA_VERSION_CURRENT,
        embed_dim,
        CACHE_VERSION,
    )


def _parity_components(store: Any) -> tuple:
    embed_dim = int(getattr(store, "embed_dim", 0))
    return (SCHEMA_VERSION_CURRENT, embed_dim, CACHE_VERSION)


class _OverlayBypass:
    __slots__ = ("reason", "age_ms")

    def __init__(self, reason: str, age_ms: int = 0) -> None:
        self.reason = reason
        self.age_ms = age_ms

    def __repr__(self) -> str:  # pragma: no cover
        return f"_OverlayBypass(reason={self.reason!r}, age_ms={self.age_ms})"


def _check_snapshot_invariants(data: dict) -> bool:
    assignment_raw = data.get("assignment")
    if not isinstance(assignment_raw, dict):
        return False
    node_to_community = assignment_raw.get("node_to_community") or {}
    if not isinstance(node_to_community, dict):
        return False
    n_communities = len(set(node_to_community.values()))
    if n_communities == 0 and len(node_to_community) > 0:
        return False
    if n_communities > 100_000:
        return False
    rich_club_raw = data.get("rich_club") or []
    if isinstance(rich_club_raw, list) and rich_club_raw:
        node_ids = set(node_to_community.keys())
        for rc_id in rich_club_raw:
            if rc_id not in node_ids:
                return False
    try:
        modularity = float(assignment_raw.get("modularity", 0.0) or 0.0)
        if not (-1.0 <= modularity <= 1.0):
            return False
    except (TypeError, ValueError):
        return False
    return True


def consult_overlay(store: Any) -> "tuple | _OverlayBypass":
    data = _load_and_decrypt_cache(store)
    if data is None:
        return _OverlayBypass("no_snapshot")

    if data.get("cache_version") != CACHE_VERSION:
        return _OverlayBypass("parity_mismatch")

    saved_key = tuple(data.get("key", []))
    if len(saved_key) < 5:
        return _OverlayBypass("parity_mismatch")
    current_parity = _parity_components(store)
    if saved_key[2] != current_parity[0]:
        return _OverlayBypass("parity_mismatch")
    if saved_key[3] != current_parity[1]:
        return _OverlayBypass("parity_mismatch")
    if saved_key[4] != current_parity[2]:
        return _OverlayBypass("parity_mismatch")

    snapshot_generation = data.get("generation", 0)
    if not isinstance(snapshot_generation, int):
        return _OverlayBypass("epoch_mismatch")
    current_gen = get_current_generation()
    if current_gen == 0 or snapshot_generation == 0 or snapshot_generation != current_gen:
        return _OverlayBypass("epoch_mismatch")

    rebuild_ts_str = data.get("rebuild_timestamp")
    age_ms = 0
    if rebuild_ts_str:
        try:
            rebuild_dt = datetime.fromisoformat(str(rebuild_ts_str))
            if rebuild_dt.tzinfo is None:
                rebuild_dt = rebuild_dt.replace(tzinfo=timezone.utc)
            age_sec = (datetime.now(timezone.utc) - rebuild_dt).total_seconds()
            age_ms = max(0, int(age_sec * 1000))
        except (TypeError, ValueError):
            age_sec = _FUSE_MAX_AGE_SECONDS + 1.0
            age_ms = int(age_sec * 1000)
    else:
        age_sec = 0.0
        age_ms = 0

    dirty = get_dirty_counter()
    if age_sec > _FUSE_MAX_AGE_SECONDS or dirty > _FUSE_DIRTY_THRESHOLD:
        _emit_freshness_fuse_tripped(store, age_ms=age_ms)
        return _OverlayBypass("fuse_tripped", age_ms=age_ms)

    if not _check_snapshot_invariants(data):
        return _OverlayBypass("invariant_failure")

    try:
        assignment = _decode_assignment(data["assignment"])
        rich_club = _decode_rich_club(data.get("rich_club"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("runtime_graph_cache overlay decode failed: %s", exc)
        return _OverlayBypass("invariant_failure")

    return assignment, rich_club


def _emit_freshness_fuse_tripped(store: Any, *, age_ms: int) -> None:
    try:
        from iai_mcp.events import (
            TELEMETRY_FRESHNESS_FUSE_TRIPPED,
            write_event,
        )
        from iai_mcp.store import MemoryStore

        if not isinstance(store, MemoryStore):
            return
        write_event(
            store,
            TELEMETRY_FRESHNESS_FUSE_TRIPPED,
            {"age_ms": int(age_ms), "pending_rebuild": False},
            severity="info",
            buffered=True,
        )
    except Exception:  # noqa: BLE001 -- telemetry must never break recall
        pass


_current_generation: int = 0
_GEN_LOCK = threading.Lock()


def get_current_generation() -> int:
    with _GEN_LOCK:
        return _current_generation


def advance_generation() -> int:
    global _current_generation  # noqa: PLW0603
    with _GEN_LOCK:
        _current_generation += 1
        return _current_generation


def load_current_generation_from_snapshot(store: Any) -> int:
    data = _load_and_decrypt_cache(store)
    if data is None:
        return 0
    if data.get("cache_version") != CACHE_VERSION:
        return 0
    gen = data.get("generation", 0)
    try:
        result = int(gen)
        global _current_generation  # noqa: PLW0603
        with _GEN_LOCK:
            if result > _current_generation:
                _current_generation = result
        return result
    except (TypeError, ValueError):
        return 0


def _encode_assignment(assignment: Any) -> dict:
    return {
        "node_to_community": {
            str(leaf): str(comm)
            for leaf, comm in getattr(assignment, "node_to_community", {}).items()
        },
        "community_centroids": {
            str(comm): list(vec)
            for comm, vec in getattr(assignment, "community_centroids", {}).items()
        },
        "modularity": float(getattr(assignment, "modularity", 0.0)),
        "backend": str(getattr(assignment, "backend", "flat")),
        "top_communities": [str(c) for c in getattr(assignment, "top_communities", [])],
        "mid_regions": {
            str(comm): [str(m) for m in members]
            for comm, members in getattr(assignment, "mid_regions", {}).items()
        },
    }


def _decode_assignment(raw: dict) -> Any:
    from iai_mcp.community import CommunityAssignment

    return CommunityAssignment(
        node_to_community={
            UUID(leaf): UUID(comm)
            for leaf, comm in raw.get("node_to_community", {}).items()
        },
        community_centroids={
            UUID(comm): list(vec)
            for comm, vec in raw.get("community_centroids", {}).items()
        },
        modularity=float(raw.get("modularity", 0.0)),
        backend=str(raw.get("backend", "flat")),
        top_communities=[UUID(c) for c in raw.get("top_communities", [])],
        mid_regions={
            UUID(comm): [UUID(m) for m in members]
            for comm, members in raw.get("mid_regions", {}).items()
        },
    )


def _encode_rich_club(rich_club: Any) -> list[str]:
    return [str(u) for u in (rich_club or [])]


def _decode_rich_club(raw: Any) -> list[UUID]:
    return [UUID(u) for u in (raw or [])]


_JSON_DICT_ENTRY_OVERHEAD: int = 4
# 384-dim float vector dominates: 384*24=9216 + structural ~1024
_NODE_PAYLOAD_BYTES_PER_RECORD: int = 10240
# 384-dim float same calculus as node_payload embedding -> 9216 + UUID
_CENTROID_BYTES_PER_RECORD: int = 9472

_MID_REGION_BYTES_PER_RECORD: int = 1280

_RICH_CLUB_BYTES_PER_ENTRY: int = 38

_BASE_SCAFFOLD_BYTES: int = 4096


def _estimate_serialised_bytes(data: dict) -> int:
    total = _BASE_SCAFFOLD_BYTES

    np_block = data.get("node_payload") or {}
    if isinstance(np_block, dict):
        total += len(np_block) * (
            _NODE_PAYLOAD_BYTES_PER_RECORD + _JSON_DICT_ENTRY_OVERHEAD + 38
        )

    assignment_block = data.get("assignment") or {}
    if isinstance(assignment_block, dict):
        ntc = assignment_block.get("node_to_community") or {}
        if isinstance(ntc, dict):
            total += len(ntc) * 50

        centroids = assignment_block.get("community_centroids") or {}
        if isinstance(centroids, dict):
            total += len(centroids) * (
                _CENTROID_BYTES_PER_RECORD + _JSON_DICT_ENTRY_OVERHEAD
            )

        mid = assignment_block.get("mid_regions") or {}
        if isinstance(mid, dict):
            total += len(mid) * (
                _MID_REGION_BYTES_PER_RECORD + _JSON_DICT_ENTRY_OVERHEAD
            )

        top = assignment_block.get("top_communities") or []
        if isinstance(top, list):
            total += len(top) * 16

    rich_club = data.get("rich_club") or []
    if isinstance(rich_club, list):
        total += len(rich_club) * _RICH_CLUB_BYTES_PER_ENTRY

    return total


def try_load(store: Any) -> tuple | None:
    path = _cache_path(store)
    if not path.exists():
        return None
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.debug("runtime_graph_cache read failed: %s", exc)
        return None

    legacy_v2_plaintext = False
    if is_encrypted(raw_text):
        try:
            key = _cache_encryption_key(store)
            plaintext_json = decrypt_field(raw_text, key, _CACHE_AAD)
            data = json.loads(plaintext_json)
        except (InvalidTag, OSError, ValueError, KeyError, RuntimeError) as exc:
            try:
                sys.stderr.write(
                    '{"event":"runtime_graph_cache_decrypt_failed","error":'
                    + json.dumps(str(exc) or type(exc).__name__)
                    + '}\n'
                )
            except (OSError, ValueError):
                pass
            return None
    else:
        try:
            data = json.loads(raw_text)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("cache_version") == LEGACY_CACHE_VERSION_PLAINTEXT:
            legacy_v2_plaintext = True
        else:
            return None

    if not isinstance(data, dict):
        return None
    if not legacy_v2_plaintext and data.get("cache_version") != CACHE_VERSION:
        return None
    saved_key = tuple(data.get("key", []))
    current_key = _cache_key(store)
    if legacy_v2_plaintext:
        expected_legacy_key = tuple(
            list(current_key)[:-1] + [LEGACY_CACHE_VERSION_PLAINTEXT]
        )
        if saved_key != expected_legacy_key:
            return None
    else:
        if saved_key != current_key:
            return None

    try:
        assignment = _decode_assignment(data["assignment"])
        rich_club = _decode_rich_club(data.get("rich_club"))
        node_payload_raw = data.get("node_payload")
        node_payload: dict[str, dict] | None
        if isinstance(node_payload_raw, dict):
            node_payload = {}
            drop_count = 0
            for k, v in node_payload_raw.items():
                if not isinstance(v, dict):
                    continue
                surface = v.get("surface")
                if surface in (None, "") or v.get("_decrypt_failed"):
                    drop_count += 1
                    continue
                node_payload[str(k)] = dict(v)
            if drop_count > 0:
                try:
                    sys.stderr.write(
                        '{"event":"runtime_graph_cache_drop_poisoned_entry","count":'
                        + str(drop_count)
                        + '}\n'
                    )
                except OSError:
                    pass
        else:
            node_payload = None
        try:
            max_degree = int(data.get("max_degree", 0) or 0)
        except (TypeError, ValueError):
            max_degree = 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("runtime_graph_cache decode failed: %s", exc)
        return None

    if legacy_v2_plaintext:
        try:
            save(
                store, assignment, rich_club,
                node_payload=node_payload, max_degree=max_degree,
            )
        except (OSError, ValueError) as exc:
            logger.debug("runtime_graph_cache legacy re-save failed: %s", exc)

    return assignment, rich_club, node_payload, max_degree


def _load_and_decrypt_cache(store: Any) -> "dict | None":
    path = _cache_path(store)
    if not path.exists():
        return None
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.debug("runtime_graph_cache read failed: %s", exc)
        return None
    if not is_encrypted(raw_text):
        return None
    try:
        key = _cache_encryption_key(store)
        plaintext_json = decrypt_field(raw_text, key, _CACHE_AAD)
        data = json.loads(plaintext_json)
    except (InvalidTag, OSError, ValueError, KeyError, RuntimeError) as exc:
        try:
            sys.stderr.write(
                '{"event":"runtime_graph_cache_decrypt_failed","error":'
                + json.dumps(str(exc) or type(exc).__name__)
                + '}\n'
            )
        except (OSError, ValueError):
            pass
        return None
    if not isinstance(data, dict):
        return None
    return data


def load_last_good_structural(store: Any) -> "tuple | None":
    data = _load_and_decrypt_cache(store)
    if data is None:
        return None
    if data.get("cache_version") != CACHE_VERSION:
        return None
    saved_key = tuple(data.get("key", []))
    if len(saved_key) < 5:
        return None
    current_parity = _parity_components(store)
    if saved_key[2] != current_parity[0]:
        return None
    if saved_key[3] != current_parity[1]:
        return None
    if saved_key[4] != current_parity[2]:
        return None
    try:
        assignment = _decode_assignment(data["assignment"])
        rich_club = _decode_rich_club(data.get("rich_club"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("runtime_graph_cache last_good decode failed: %s", exc)
        return None
    return assignment, rich_club


def load_recall_structural(store: Any) -> "tuple":
    from iai_mcp.community import CommunityAssignment

    if get_current_generation() == 0:
        load_current_generation_from_snapshot(store)
    try:
        overlay_result = consult_overlay(store)
        if not isinstance(overlay_result, _OverlayBypass):
            ov_assignment, ov_rich_club = overlay_result
            data = _load_and_decrypt_cache(store)
            ov_max_degree = 0
            if data is not None:
                try:
                    ov_max_degree = int(data.get("max_degree", 0) or 0)
                except (TypeError, ValueError):
                    ov_max_degree = 0
            return ov_assignment, ov_rich_club, ov_max_degree, "overlay"
    except Exception:  # noqa: BLE001 -- overlay errors must never break recall
        pass

    cached = try_load(store)
    if cached is not None:
        assignment, rich_club, _node_payload, max_degree = cached
        return assignment, rich_club, int(max_degree or 0), "normal"

    last_good = load_last_good_structural(store)
    if last_good is not None:
        assignment, rich_club = last_good
        return assignment, rich_club, 0, "last_good"

    empty_assignment = CommunityAssignment(
        node_to_community={},
        community_centroids={},
        modularity=0.0,
        backend="cold-degrade",
        top_communities=[],
        mid_regions={},
    )
    return empty_assignment, [], 0, "cold_degrade"


_rebuild_timestamp_override: str = ""


def _maybe_emit_snapshot_near_limit(store: Any, estimated_bytes: int) -> None:
    """One-shot per-generation telemetry when the snapshot is about to degrade.

    Rate-limited to one emission per `_GEN_LOCK` window so the event remains
    informative even when many saves fire in quick succession.
    """
    global _snapshot_near_limit_last_gen  # noqa: PLW0603
    threshold = int(MAX_CACHE_BYTES * _SNAPSHOT_NEAR_LIMIT_FRACTION)
    if estimated_bytes < threshold:
        return
    with _GEN_LOCK:
        current_gen = _current_generation
        if _snapshot_near_limit_last_gen == current_gen:
            return
        _snapshot_near_limit_last_gen = current_gen
    try:
        from iai_mcp.events import (
            TELEMETRY_RGC_SNAPSHOT_NEAR_LIMIT,
            emit_best_effort,
        )
        from iai_mcp.store import MemoryStore

        # Guard against test doubles that look like a store but cannot supply
        # the encryption key the events writer needs.
        if not isinstance(store, MemoryStore):
            return
        emit_best_effort(
            store,
            TELEMETRY_RGC_SNAPSHOT_NEAR_LIMIT,
            {
                "estimated_bytes": int(estimated_bytes),
                "max_cache_bytes": int(MAX_CACHE_BYTES),
                "fraction": round(estimated_bytes / max(MAX_CACHE_BYTES, 1), 3),
            },
            severity="info",
        )
    except Exception:  # noqa: BLE001 -- telemetry must never break save
        pass


def save(
    store: Any,
    assignment: Any,
    rich_club: Any,
    node_payload: "dict[str, dict] | None" = None,
    max_degree: int = 0,
) -> bool:
    path = _cache_path(store)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    encoded_node_payload: dict[str, dict] | None = None
    if node_payload:
        encoded_node_payload = {}
        for k, v in node_payload.items():
            if not isinstance(v, dict):
                continue
            raw_emb = v.get("embedding") or []
            raw_tags = v.get("tags") or []
            encoded_node_payload[str(k)] = {
                "embedding": [float(x) for x in raw_emb],
                "surface": str(v.get("surface", "")),
                "centrality": float(v.get("centrality") or 0.0),
                "tier": str(v.get("tier", "episodic")),
                "pinned": bool(v.get("pinned", False)),
                "tags": [str(t) for t in raw_tags if t is not None],
                "language": str(v.get("language", "en") or "en"),
            }

    data = {
        "cache_version": CACHE_VERSION,
        "key": list(_cache_key(store)),
        "assignment": _encode_assignment(assignment),
        "rich_club": _encode_rich_club(rich_club),
        "node_payload": encoded_node_payload or {},
        "max_degree": int(max_degree or 0),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "generation": int(get_current_generation()),
        "rebuild_timestamp": _rebuild_timestamp_override or "",
    }

    estimated_bytes = _estimate_serialised_bytes(data)
    _maybe_emit_snapshot_near_limit(store, estimated_bytes)
    if estimated_bytes > MAX_CACHE_BYTES:
        data["node_payload"] = {}
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        if isinstance(data.get("assignment"), dict):
            data["assignment"]["community_centroids"] = {}
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        if isinstance(data.get("assignment"), dict):
            data["assignment"]["mid_regions"] = {}
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        return False

    serialised = json.dumps(data, ensure_ascii=False)

    try:
        key = _cache_encryption_key(store)
        ciphertext = encrypt_field(serialised, key, _CACHE_AAD)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("runtime_graph_cache encrypt failed: %s", exc)
        try:
            sys.stderr.write(
                '{"event":"runtime_graph_cache_encrypt_failed"}\n'
            )
        except OSError:
            pass
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="ascii") as f:
            f.write(ciphertext)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))
        return True
    except OSError as exc:
        logger.debug("runtime_graph_cache write failed: %s", exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False


def save_with_generation(
    store: Any,
    assignment: Any,
    rich_club: Any,
    node_payload: "dict[str, dict] | None" = None,
    max_degree: int = 0,
) -> bool:
    new_gen = advance_generation()
    reset_dirty_counter()
    ts_iso = datetime.now(timezone.utc).isoformat()
    global _rebuild_timestamp_override  # noqa: PLW0603
    with _GEN_LOCK:
        _rebuild_timestamp_override = ts_iso
    result = save(store, assignment, rich_club, node_payload=node_payload, max_degree=max_degree)
    with _GEN_LOCK:
        _rebuild_timestamp_override = ""
    return result


def invalidate(store: Any) -> None:
    path = _cache_path(store)
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        logger.debug("runtime_graph_cache invalidate failed: %s", exc)


def _rebuild_and_save_rgc(store: Any, *, force: bool = False) -> dict:
    import multiprocessing
    import time

    from iai_mcp.community import CommunityAssignment
    from iai_mcp.events import (
        TELEMETRY_RGC_WORKER_CRASH,
        TELEMETRY_RGC_WORKER_SUCCESS,
        TELEMETRY_RGC_WORKER_TIMEOUT,
        emit_best_effort,
    )
    from iai_mcp.runtime_graph_cache_ro_export import (
        iter_edges_chunks,
        iter_records_chunks,
        open_ro_connection,
        read_transaction,
    )

    global _first_spawn_seen  # noqa: PLW0603

    with _PERSISTENT_GRAPH_LOCK:
        if not force:
            # Skip the rebuild (and its allocation) only when the cached snapshot
            # is still usable for recall. The read path's structural source is
            # the authoritative signal: warm iff overlay/normal, cold otherwise.
            # It already folds in no-snapshot / parity / epoch / generation==0 /
            # age+dirty fuse. The dirty counter is a separate write-volume signal,
            # so a cache can be cold while the counter is zero — gate on both.
            try:
                structural_source = load_recall_structural(store)[3]
            except Exception:  # noqa: BLE001 -- a probe failure must never drop a warm-up
                structural_source = "cold_degrade"  # fail toward rebuilding
            cache_is_warm = structural_source in ("overlay", "normal")
            if cache_is_warm and get_dirty_counter() <= _FUSE_DIRTY_THRESHOLD:
                return {
                    "rebuilt": False,
                    "skipped": "warm_and_below_dirty_threshold",
                    "structural_source": structural_source,
                    "node_count": 0,
                    "generation": get_current_generation(),
                }

        # Estimate the dataset size up-front so the watchdog timeout can be
        # scaled to the workload. `active_records_count` is the cheap
        # COUNT(*) under the same predicate used by the streaming SELECT.
        try:
            est_node_count = int(store.active_records_count())
        except Exception:  # noqa: BLE001
            est_node_count = 0

        # Spawn the worker. Spawn-context (not fork) so the child re-imports
        # cleanly on macOS and Linux; the child closes its end after start so
        # the parent does not hold a half-of-pipe alive on crash detection.
        first_spawn_flag = not _first_spawn_seen
        timeout_s = _resolve_timeout(est_node_count)
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        process = ctx.Process(
            target=_worker_entry_indirection,
            args=(child_conn,),
            daemon=True,
        )
        process.start()
        child_conn.close()

        db_path = store.db._hippo_dir / "brain.sqlite3"
        ro_conn = None
        node_count = 0
        t0 = time.perf_counter()
        try:
            # Stream the projection. The dedicated read-only connection means
            # the shared write lock is never held during streaming.
            ro_conn = open_ro_connection(db_path)
            try:
                with read_transaction(ro_conn):
                    for chunk in iter_records_chunks(ro_conn):
                        parent_conn.send(("nodes", chunk))
                        node_count += len(chunk)
                    parent_conn.send(("nodes_end", None))
                    for chunk in iter_edges_chunks(ro_conn):
                        parent_conn.send(("edges", chunk))
                    parent_conn.send(("edges_end", None))
            finally:
                try:
                    ro_conn.close()
                except Exception:  # noqa: BLE001
                    pass
                ro_conn = None

            # Receive the compact result.
            result = _drain_worker_result(parent_conn, timeout=timeout_s)

            process.join(timeout=5.0)
            if process.exitcode != 0:
                raise WorkerCrashedError(
                    f"worker exited with code {process.exitcode}"
                )

            # Reassemble parent-side.
            assignment = CommunityAssignment(
                node_to_community=result["node_to_community"],
                community_centroids=result["community_centroids"],
                modularity=0.0,
                backend=result["backend"],
                top_communities=result["top_communities"],
                mid_regions=result["mid_regions"],
                lineage_report=None,
            )
            rich_club = result["rich_club"]
            max_degree = int(result["max_degree"])

            saved = save_with_generation(
                store, assignment, rich_club, max_degree=max_degree
            )

            duration_s = time.perf_counter() - t0
            _first_spawn_seen = True

            emit_best_effort(
                store,
                TELEMETRY_RGC_WORKER_SUCCESS,
                {
                    "duration_s": round(duration_s, 3),
                    "node_count": int(node_count),
                    "max_degree": int(max_degree),
                    "first_spawn": first_spawn_flag,
                },
            )
            return {
                "rebuilt": True,
                "saved": saved,
                "node_count": int(node_count),
                "generation": get_current_generation(),
            }

        except WorkerTimeoutError as exc:
            _terminate_worker(process)
            emit_best_effort(
                store,
                TELEMETRY_RGC_WORKER_TIMEOUT,
                {
                    "first_spawn": first_spawn_flag,
                    "timeout_s": timeout_s,
                    "node_count": int(node_count),
                },
                severity="warn",
            )
            return {
                "rebuilt": False,
                "error": str(exc)[:200],
                "node_count": int(node_count),
                "generation": get_current_generation(),
            }
        except WorkerCrashedError as exc:
            _terminate_worker(process)
            emit_best_effort(
                store,
                TELEMETRY_RGC_WORKER_CRASH,
                {
                    "exitcode": getattr(process, "exitcode", None),
                    "reason": "nonzero_exit",
                    "first_spawn": first_spawn_flag,
                },
                severity="warn",
            )
            return {
                "rebuilt": False,
                "error": str(exc)[:200],
                "node_count": int(node_count),
                "generation": get_current_generation(),
            }
        except (BrokenPipeError, EOFError) as exc:
            reason = "broken_pipe" if isinstance(exc, BrokenPipeError) else "pipe_eof"
            _terminate_worker(process)
            emit_best_effort(
                store,
                TELEMETRY_RGC_WORKER_CRASH,
                {
                    "exitcode": getattr(process, "exitcode", None) or "unknown",
                    "reason": reason,
                    "first_spawn": first_spawn_flag,
                },
                severity="warn",
            )
            return {
                "rebuilt": False,
                "error": "worker_disconnected",
                "node_count": int(node_count),
                "generation": get_current_generation(),
            }
        finally:
            try:
                parent_conn.close()
            except Exception:  # noqa: BLE001
                pass
            if ro_conn is not None:
                try:
                    ro_conn.close()
                except Exception:  # noqa: BLE001
                    pass
            if process.is_alive():
                _terminate_worker(process)
