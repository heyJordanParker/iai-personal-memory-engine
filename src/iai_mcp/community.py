"""Hierarchical community detection (bootstrap + stable UUIDs).

Policy:
- N < SMALL_N_FLAT (200): single flat community. Rich-club coefficient is too noisy
  below this per van den Heuvel & Sporns 2011; Leiden output is unstable too.
- SMALL_N_FLAT <= N < MID_N_LEIDEN (500): run Leiden; accept only if CPM-Q >=
  CPM_MODULARITY_FLOOR, else fall back to flat.
  Protects against Leiden producing visible but unjustified communities in
  sparse graphs.
- N >= MID_N_LEIDEN: always run Leiden; accept result regardless of Q
  (graph is big enough that any modular structure is meaningful).

Stable UUIDs:
- Every community gets a persistent UUID at creation.
- continuity is now enforced via the explicit `LineageTracker` event log, which threads prior-UUID birth timestamps through aggregation
  and uses `pick_merge_survivor` to honour the older-survives policy on merges.
- The legacy 0.7-cosine centroid-match heuristic (`_map_to_stable_uuids`) was
  deleted in. The constant
  `UUID_ROTATE_COSINE = 0.7` is retained because `_flat_assignment` still
  uses it for single-community continuity across re-runs (a smaller, simpler
  case that doesn't need full event-driven tracking).

Three-level parcellation (approximation):
- Level 1: top_communities -- top 7 (Yeo-like) by member count.
- Level 2: mid_regions -- community UUID -> member node UUIDs.
- Level 3: node_to_community -- every leaf record's community assignment.

Refresh threshold:
- needs_refresh(prior, current_Q) returns True iff |prior.Q - current_Q| > 0.05.
  The pipeline or session-start assembler decides when to re-run detect_communities
  based on this signal.
"""
from __future__ import annotations

# The CPM-Q mid-N guard imports CPM_MODULARITY_FLOOR from `mosaic_policy`
# -- legacy `MODULARITY_FLOOR=0.2` was calibrated for
# ModularityVertexPartition and is NOT comparable to CPM-Q.
#
# `leidenalg` and `python-igraph` are no longer imported (top-level OR
# lazy) in this module; both packages were removed from pyproject.toml.
#
# Imports from `mosaic*` modules are DEFERRED to inside
# `detect_communities` (lazy import) because those modules import
# `CommunityAssignment` from THIS module -- a top-level import would
# create a circular dependency at module-load time.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

import numpy as np

from iai_mcp.graph import MemoryGraph

if TYPE_CHECKING:
    # Imported only for type checkers -- the runtime field annotation is a
    # forward reference string via `from __future__ import annotations`.
    from iai_mcp.mosaic_lineage import LineageReport

# bootstrap thresholds
SMALL_N_FLAT = 200
MID_N_LEIDEN = 500
MODULARITY_FLOOR = 0.2

# refresh trigger
REFRESH_DELTA = 0.05

# stable-UUID cosine floor
UUID_ROTATE_COSINE = 0.7

# level-1 cap (Yeo-like 7 networks)
MAX_TOP_COMMUNITIES = 7


@dataclass
class CommunityAssignment:
    """Output of detect_communities -- consumed by pipeline.pipeline_recall.

    - node_to_community: leaf UUID -> community UUID
    - community_centroids: community UUID -> mean of member embeddings
    - modularity: CPM-Q from the MOSAIC backend (0.0 for flat).
    - backend: "flat" | "leiden-custom"
    - top_communities: up to MAX_TOP_COMMUNITIES by member count (L1)
    - mid_regions: community UUID -> list of member leaf UUIDs (L2)
    - lineage_report: optional event log from the MOSAIC run.
      Default `None` for backwards compatibility with existing constructors;
      `detect_communities` populates it on every Leiden path. The flat
      fallback path also populates it with an empty `LineageReport`
      (type-stability for downstream consumers).
    """

    node_to_community: dict[UUID, UUID] = field(default_factory=dict)
    community_centroids: dict[UUID, list[float]] = field(default_factory=dict)
    modularity: float = 0.0
    backend: str = "flat"
    top_communities: list[UUID] = field(default_factory=list)
    mid_regions: dict[UUID, list[UUID]] = field(default_factory=dict)
    # Forward reference via `from __future__ import annotations`; the
    # runtime instance is `iai_mcp.mosaic_lineage.LineageReport`.
    lineage_report: "LineageReport | None" = None


# ---------------------------------------------------------------- math helpers


def _cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def _compute_centroid(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    arr = np.asarray(embeddings, dtype=np.float32)
    centroid = arr.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    return centroid.tolist()


# deleted the legacy `_map_to_stable_uuids(raw_partition, graph, prior)`
# helper -- continuity is now event-driven via
# `iai_mcp.mosaic_lineage.LineageTracker` + `pick_merge_survivor`
#. The flat-path UUID continuity in `_flat_assignment` still
# uses the cosine threshold UUID_ROTATE_COSINE = 0.7 for single-community
# matching (simpler case, no aggregation cascade).


# ------------------------------------------------------------- flat assignment


def _flat_assignment(
    graph: MemoryGraph, prior: CommunityAssignment | None
) -> CommunityAssignment:
    """Single flat community covering every node."""
    nodes: list[UUID] = []
    valid_embs: list[list[float]] = []
    for u in graph.iter_nodes():
        nodes.append(u)
        emb = graph.get_embedding(u)
        if emb:
            valid_embs.append(emb)
    if not nodes:
        return CommunityAssignment(backend="flat")

    # Zero-pad any sentinel nodes to the detected store dim so centroid math
    # stays homogeneous post-re-embed (was hardcoded 384d before 1024d support).
    dim = len(valid_embs[0]) if valid_embs else 0
    embs: list[list[float]] = []
    for u in graph.iter_nodes():
        emb = graph.get_embedding(u)
        embs.append(emb if emb else [0.0] * dim)
    centroid = _compute_centroid(embs) if dim else []

    # Stable UUID across flat runs: reuse prior's single UUID if centroid matches.
    flat_uuid: UUID | None = None
    if prior and len(prior.community_centroids) == 1:
        prior_uuid, prior_cent = next(iter(prior.community_centroids.items()))
        if _cosine(centroid, prior_cent) >= UUID_ROTATE_COSINE:
            flat_uuid = prior_uuid
    if flat_uuid is None:
        flat_uuid = uuid4()

    node_to_community = {n: flat_uuid for n in nodes}
    community_centroids = {flat_uuid: centroid}
    return CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids=community_centroids,
        modularity=0.0,
        backend="flat",
        top_communities=[flat_uuid],
        mid_regions={flat_uuid: nodes},
    )


# deleted the legacy `_run_leiden(graph)` shim that wrapped
# leidenalg + igraph. The production path is `run_mosaic` (pure-MIT,
# imported lazily inside `detect_communities`).


# ------------------------------------------------------------------ public API


def detect_communities(
    graph: MemoryGraph,
    prior: CommunityAssignment | None = None,
    prior_mode: Literal["seeded", "cold"] = "seeded",
) -> CommunityAssignment:
    """bootstrap + stable UUIDs + three-level parcellation.

    The backend is now `mosaic.run_mosaic`
    -- the pure-MIT replacement for the `leidenalg` path (renamed
    `custom_leiden*` -> `mosaic*` as part of the MOSAIC branding). The
    `prior_mode` argument routes two invocation flavours:
      - `"seeded"` (default) -- normal recall paths (`retrieve.py`,
        `sigma.py`); reuse the prior assignment for continuity.
      - `"cold"` -- crisis_recluster path intentionally discards the
        prior partition.

    Empty graph -> empty CommunityAssignment(backend="flat").
    """
    # Lazy import to avoid circular dependency at module-load time
    # (mosaic_* modules import CommunityAssignment from here).
    from iai_mcp.mosaic import run_mosaic
    from iai_mcp.mosaic_lineage import LineageReport
    from iai_mcp.mosaic_policy import CPM_MODULARITY_FLOOR

    n = graph.node_count()
    if n == 0:
        return CommunityAssignment(
            backend="flat", lineage_report=LineageReport(events=())
        )
    if n < SMALL_N_FLAT:
        flat = _flat_assignment(graph, prior)
        flat.lineage_report = LineageReport(events=())
        return flat

    try:
        inner_assignment, lineage_report = run_mosaic(
            graph, prior=prior, prior_mode=prior_mode, seed=42
        )
    except (ImportError, RuntimeError, ValueError, TypeError):
        # Leiden unavailable or graph pathological -> degrade gracefully.
        flat = _flat_assignment(graph, prior)
        flat.lineage_report = LineageReport(events=())
        return flat

    # Mid-N modularity guard -- imports CPM_MODULARITY_FLOOR from
    # `mosaic_policy` (calibration: 0.1338). The legacy
    # `MODULARITY_FLOOR=0.2` was sampled against
    # `ModularityVertexPartition`; CPM-Q is gamma-dependent and not
    # comparable to classical-Q at 0.2.
    if n < MID_N_LEIDEN and inner_assignment.modularity < CPM_MODULARITY_FLOOR:
        flat = _flat_assignment(graph, prior)
        flat.lineage_report = LineageReport(events=())
        return flat

    # Auto-fix: a 1-community partition is
    # semantically flat (CPM-Q can be > 0
    # for a fully-connected clique at gamma < 1.0, but a single community
    # covering every node is the same structural outcome as the explicit
    # _flat_assignment path). Route through `_flat_assignment` so the
    # backend label, structure, and downstream consumers match the
    # historical "flat" contract -- preserves
    # `test_mid_n_non_modular_falls_back_to_flat`.
    if len(set(inner_assignment.node_to_community.values())) <= 1:
        flat = _flat_assignment(graph, prior)
        flat.lineage_report = lineage_report
        return flat

    # The inner assignment already carries node_to_community,
    # community_centroids, modularity, and backend="leiden-custom" from
    # `_build_assignment`. We augment with the lineage_report and
    # re-derive top_communities + mid_regions to honour caps.
    inner_assignment.lineage_report = lineage_report

    # level 1: top 7 communities by member count.
    counts: dict[UUID, int] = {}
    for c in inner_assignment.node_to_community.values():
        counts[c] = counts.get(c, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[
        :MAX_TOP_COMMUNITIES
    ]
    inner_assignment.top_communities = [u for u, _ in top]

    # level 2 (mid-regions): community UUID -> member node UUIDs.
    mid_regions: dict[UUID, list[UUID]] = {}
    for node, comm in inner_assignment.node_to_community.items():
        mid_regions.setdefault(comm, []).append(node)
    inner_assignment.mid_regions = mid_regions

    return inner_assignment


def needs_refresh(
    prior: CommunityAssignment, current_modularity: float
) -> bool:
    """Refresh signal when |Δ modularity| > REFRESH_DELTA (0.05).

    Consumer (session-start assembler / maintenance job) calls this on each
    new Leiden run; a True return triggers a re-assignment + cache invalidation.
    """
    return abs(prior.modularity - current_modularity) > REFRESH_DELTA
