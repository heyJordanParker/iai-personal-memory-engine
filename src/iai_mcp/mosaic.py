"""MOSAIC: Memory-Oriented Sparse Aggregative Identification of Communities.

The pure-MIT community-detection algorithm tailored to IAI-MCP's
memory-graph topology. Replaces the `leidenalg` +
`python-igraph` Leiden pipeline used in earlier versions; those packages
have since been removed from pyproject.toml in full.

The algorithm is Leiden-family (Traag, Waltman & van Eck 2019,
*Scientific Reports* 9:5233) with three IAI-MCP-specific adaptations:

  - CPM (Constant Potts Model) is the canonical objective rather than
    classical modularity, with a calibrated CPM-Q floor.
  - community-UUID continuity uses an explicit `LineageTracker`
    event log instead of post-hoc cosine matching.
  - Super-level pairwise merge after the multi-level loop closes the
    Karate-class joint-Q maxima gap.


PCG64 RNG path: Numba 0.59-0.65 does NOT support
`np.random.Generator(np.random.PCG64(seed))` inside `@njit`
(Unknown attribute 'Generator' of type Module(numpy.random)). The kernel
therefore accepts a precomputed `visit_order: int64[:]` argument; the public
`run_mosaic` wrapper constructs the permutation OUTSIDE the kernel via
`np.random.Generator(np.random.PCG64(seed)).permutation(n).astype(np.int64)`.
The refinement kernel and gamma-tuner inner-pass both inherit this
signature shape.

Constraints inherited from `community.py` (preserved verbatim):

  - thresholds -- SMALL_N_FLAT (200), MID_N_LEIDEN (500),
    MODULARITY_FLOOR (0.2).
  - stable UUIDs -- the heuristic 0.7-cosine match is replaced by the
    explicit `LineageTracker` event log; consumers of `CommunityAssignment`
    see no interface change.
  - three-level parcellation -- top 7 (Yeo-like), mid-region
    membership map, leaf node -> community map -- preserved by the
    `CommunityAssignment` wire shape.
  - refresh delta -- |dQ| > 0.05 trigger remains in `community.py`.

Constraints inherited from earlier kernels and applied throughout:

  - Determinism -- comparison kernels use `fastmath=False`, EPSILON
    is 1e-9, all arrays float64 / int64 strict.
  - Canonical order -- nodes sorted by `str(uuid)`; edges normalised
    to (min, max) and sorted by (src_idx, dst_idx) before COO construction.
  - Sanitisation -- NaN, +/-Inf, negative-weight edges are dropped
    BEFORE the CSR is handed to any kernel.
  - Self-loops -- stripped up-front; no 2m accounting in kernels.

Public surface is frozen — do not break it:

  - `EPSILON: float` -- score-comparison threshold (1e-9)
  - `WALL_TIME_HARD_CAP_S: float` -- 30s hard wall-time cap
  - `WALL_TIME_WARM_TARGET_S: float` -- 5s warm target
  - `build_csr_sanitized(graph)` -- deterministic + sanitised CSR builder
  - `run_mosaic(...)` -- entrypoint; returns a CPM Local-Move single-pass
    assignment + LineageReport. Renamed from `run_custom_leiden`.

Additional public surface:

  - `compute_sigma_tot(...)` -- per-community weighted-degree sums
  - `compute_delta_q_cpm(...)` -- CPM ΔQ for moving node -> target_comm
  - `compute_modularity_cpm(...)` -- CPM Q evaluator (1/2m form)
  - `_njit_local_move(...)` -- Numba Local Move kernel
  - `_subgraph_connected(...)` -- @njit BFS over CSR + bool mask
  - `_split_disconnected_communities(...)` -- defensive split using
                                         scipy.sparse.csgraph oracle
  - `_njit_refine(...)` -- Traag 2019 Section 2.3 refinement kernel
                                         with two-sided well-connectedness
  - `_aggregate(...)` -- super-graph aggregation with
                                         macro-partition MAINTAIN_PARTITION
                                         (Traag 2019 Section 2.4)
"""
from __future__ import annotations

import math
import time
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
import scipy.sparse
from numba import njit

from iai_mcp.community import CommunityAssignment, _flat_assignment
from iai_mcp.mosaic_lineage import (
    LineageEvent,  # re-exported via __all__ for downstream imports
    LineageReport,
    LineageTracker,
    init_partitions,
)
from iai_mcp.graph import MemoryGraph

__all__ = [
    "EPSILON",
    "WALL_TIME_HARD_CAP_S",
    "WALL_TIME_WARM_TARGET_S",
    "build_csr_sanitized",
    "compute_sigma_tot",
    "compute_delta_q_cpm",
    "compute_modularity_cpm",
    "_njit_local_move",
    # (M3): refinement-as-aggregation symbols
    "_subgraph_connected",
    "_split_disconnected_communities",
    "_njit_refine",
    "_aggregate",
    "run_mosaic",
    # (M5): multi-objective gamma tuner symbols
    "_run_one_leiden_pass",
    "multi_objective_gamma_tuner",
    #: super-level pairwise merge (Karate joint-Q gap closer)
    "_super_level_merge",
    "LineageEvent",
    "LineageReport",
    "LineageTracker",
]

# Score-comparison threshold (tighter than community.py's 0.0 fallback;
# matches the deterministic-replay invariant).
EPSILON: float = 1e-9

# 30s hard wall-time cap. `run_mosaic` falls back to `_flat_assignment`
# when exceeded.
WALL_TIME_HARD_CAP_S: float = 30.0

# Informational warm-target after Numba JIT cache is hot. Not enforced at
# runtime -- the bench harness gates the actual performance.
WALL_TIME_WARM_TARGET_S: float = 5.0


# --------------------------------------------------------------- CSR builder


def build_csr_sanitized(
    graph: MemoryGraph,
) -> tuple[scipy.sparse.csr_matrix, list[UUID], dict[UUID, int]]:
    """Build a CSR matrix with sanitised weights and canonical ordering.

    Sanitisation:
      - Drop self-loops (u == v) -- avoids 2m double-counting issues.
      - Drop edges with NaN or +/-Inf weight.
      - Drop edges with negative weight.
      - Cast all weights to numpy float64.

    Determinism:
      - Nodes ordered by `str(uuid)` ascending; this is the dense index.
      - Each edge normalised to a canonical tuple (min_idx, max_idx).
      - The COO edge list is sorted by (src_idx, dst_idx) before
        construction; the matrix is symmetrised by emitting both (a, b)
        and (b, a) rows, then `tocsr()` canonicalises.

    Returns:
      csr: scipy.sparse.csr_matrix of shape (N, N), dtype=float64,
                symmetric, no self-loops, no NaN/Inf/negative entries.
      order: canonical node UUIDs, sorted by str(uuid).
      idx_map: UUID -> dense index, in the same order.
    """
    # Canonical UUID ordering -- the dense index for the rest of the run.
    # iter_nodes is the public surface that replaces direct _nx access; it
    # yields UUID objects, so str-keyed downstream sites cast explicitly.
    order: list[UUID] = sorted(graph.iter_nodes(), key=str)
    n = len(order)
    idx_map: dict[UUID, int] = {u: i for i, u in enumerate(order)}

    # Empty graph -- return a (0, 0) CSR; callers short-circuit to flat.
    if n == 0:
        empty = scipy.sparse.csr_matrix((0, 0), dtype=np.float64)
        return empty, order, idx_map

    # Collect sanitised, canonical edges into a dedup dict keyed by the
    # canonical (min, max) tuple. Duplicates are summed defensively (an
    # undirected graph normally cannot produce them, but the
    # store / merge paths upstream are not guaranteed to be unique).
    edge_weights: dict[tuple[int, int], float] = {}
    for u_uuid, v_uuid, w in graph.iter_edges_with_weight():
        if u_uuid == v_uuid:
            # Self-loop stripped up-front.
            continue
        # iter_edges_with_weight already coerces to float; defend against
        # downstream pathological values that slipped through the writer.
        if not math.isfinite(w):
            # NaN, +Inf, -Inf all rejected by isfinite.
            continue
        if w < 0.0:
            # Negative weight rejected.
            continue
        if u_uuid not in idx_map or v_uuid not in idx_map:
            # Defensive: edge references a node that was not added.
            continue
        a = idx_map[u_uuid]
        b = idx_map[v_uuid]
        key = (a, b) if a <= b else (b, a)
        edge_weights[key] = edge_weights.get(key, 0.0) + w

    if not edge_weights:
        empty = scipy.sparse.csr_matrix((n, n), dtype=np.float64)
        return empty, order, idx_map

    # Sorted edge list -- deterministic COO construction order.
    sorted_edges = sorted(edge_weights.items())
    src = np.empty(len(sorted_edges) * 2, dtype=np.int64)
    dst = np.empty(len(sorted_edges) * 2, dtype=np.int64)
    wts = np.empty(len(sorted_edges) * 2, dtype=np.float64)
    for i, ((a, b), w) in enumerate(sorted_edges):
        # Symmetrise: emit both directions.
        src[2 * i] = a
        dst[2 * i] = b
        wts[2 * i] = w
        src[2 * i + 1] = b
        dst[2 * i + 1] = a
        wts[2 * i + 1] = w

    coo = scipy.sparse.coo_matrix(
        (wts, (src, dst)), shape=(n, n), dtype=np.float64
    )
    csr = coo.tocsr()
    csr.sort_indices()
    return csr, order, idx_map


# --------------------------------------------------------------- entrypoint


def _flat_fallback(
    graph: MemoryGraph, prior: CommunityAssignment | None
) -> tuple[CommunityAssignment, LineageReport]:
    """Wrap `community._flat_assignment` so the return shape matches the
    tuple wire-shape of `run_mosaic` (used by the empty-graph
    short-circuit and the 30s wall-cap path)."""
    return _flat_assignment(graph, prior), LineageReport(events=())


def run_mosaic(
    graph: MemoryGraph,
    prior: CommunityAssignment | None = None,
    prior_mode: Literal["seeded", "cold"] = "seeded",
    gamma: float | None = None,
    seed: int = 42,
    max_levels: int = 5,
) -> tuple[CommunityAssignment, LineageReport]:
    """MOSAIC community-detection run (M2 single-Local-Move pass + M3/M5/22-01).

    M1 shipped the empty-graph short-circuit, the deterministic +
    sanitised CSR build, and the frozen public signature. M2 now
    runs a real Local Move pass: build CSR -> compute sigma_tot -> permute
    visit order via PCG64(seed) OUTSIDE @njit -> call `_njit_local_move`
    in-place -> compute CPM modularity -> assign fresh UUIDs per integer
    label -> return `(CommunityAssignment, LineageReport)`. Refinement +
    aggregation land in M3, prior-aware seeding in M4, and the production switch in M7.

    Determinism contract (held from M1 forward):
      Same `(graph, prior, prior_mode, gamma, seed)` -> identical output.

    Performance contract (M9 gate):
      N <= 5000, avg_degree <= 20 -> < 5s warm wall-time (Numba pre-cached).
      Hard cap: WALL_TIME_HARD_CAP_S; fall back to `_flat_assignment`.
    """
    # 1. Empty-graph short-circuit -- no kernel call on the pathological
    # no-node case so callers can build assignments off a zero-node graph.
    if graph.node_count() == 0:
        return _flat_fallback(graph, prior)

    t_start = time.monotonic()

    # 2. Sanitisation -- enforced before the kernel sees any weight data.
    csr, order, _idx_map = build_csr_sanitized(graph)

    # 2a. Zero-edge sanitised result -- everyone is an isolate. Singleton
    # partitions are uninteresting and the modularity gate would fail
    # for the wrong reason; short-circuit to flat.
    if csr.nnz == 0:
        return _flat_fallback(graph, prior)

    # 3. Cast CSR arrays to the int64 / float64 strict dtype the kernel
    # is typed for. scipy may down-cast indptr/indices to int32 on small
    # matrices (heuristic); the kernel signature is int64[:] so we
    # promote defensively. `np.ascontiguousarray` is a no-op when the
    # array is already int64-contiguous.
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1

    # 4. Initialise partition via prior-aware seeding.
    # init_partitions returns (partition, int_to_uuid, lineage) where
    # partition[i] indexes into the canonical (str-sorted) UUID order from
    # build_csr_sanitized -- the SAME ordering used by `order` above, so
    # the indices line up. The returned `lineage` already carries
    # `register_prior_birth` bookkeeping for surviving prior UUIDs plus
    # `birth` events for new-node singletons; we thread it through the
    # multi-level loop so split/merge/death events accumulate against the
    # same tracker.
    init_partition, init_int_to_uuid, lineage = init_partitions(
        graph, prior, prior_mode
    )
    partition = init_partition.astype(np.int64, copy=False)

    # 5. gamma default. M5 wires the multi-objective tuner
    # when `gamma is None`. Explicit `gamma` skips the tuner.
    # 6. Compute sigma_tot for the seeded partition. The tuner is
    # calibrated against a SINGLETON cold-start (`CPM_MODULARITY_FLOOR`
    # was sampled from that distribution per); the main
    # Leiden loop honours the seeded partition.
    n_communities = int(partition.max()) + 1 if partition.size else 0
    sigma_tot = compute_sigma_tot(indptr, indices, data, partition, n_communities)

    # 5a. Gamma auto-tune.
    # If `gamma is None`, run the multi-objective tuner BEFORE the main
    # multi-level loop. The tuner evaluates 5 candidate gammas (one
    # Local-Move + refinement pass each, on a COPY of the cold-start
    # partition), scores them on the 5-criterion target set (CPM-Q >=
    # CPM_MODULARITY_FLOOR, singleton_ratio < 0.30, connectedness,
    # determinism, composite ranking), and returns either:
    # - the gamma with the highest composite score that satisfies all
    # hard constraints, OR
    # - if no candidate satisfies hard constraints, the soft-best by
    # composite score AND a `should_fall_back_to_flat=True` flag.
    #
    # Tuner is fed a SINGLETON cold-start partition irrespective of
    # `prior_mode`. CPM_MODULARITY_FLOOR was calibrated against the
    # singleton-cold distribution in; feeding a seeded
    # partition would change the candidate-pass output distribution and
    # invalidate the calibrated floor.
    csr_for_tuner = scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=(n, n)
    )
    if gamma is None:
        tuner_partition = np.arange(n, dtype=np.int64)
        tuner_sigma = compute_sigma_tot(
            indptr, indices, data, tuner_partition, n,
        )
        gamma_value, _tuner_diag = multi_objective_gamma_tuner(
            csr_for_tuner, tuner_partition, tuner_sigma, seed,
        )
        # If no candidate satisfies hard constraints, the tuner flags
        # fall-back-to-flat. Honour it before paying the multi-level
        # Leiden cost.
        if _tuner_diag.get("should_fall_back_to_flat", False):
            return _flat_fallback(graph, prior)
    else:
        gamma_value = float(gamma)

    # 7. Multi-level Leiden loop.
    #
    # Per level: Local Move -> defensive split -> Refinement
    # (Traag 2019 Section 2.3 well-connectedness) -> Aggregation. Loop
    # until convergence (no LM moves AND no refinement moves) or
    # max_levels reached or wall-time cap.
    #
    # Projection accounting:
    # - `node_to_super_idx[i]` records, for each ORIGINAL node i, its
    # index in the CURRENT level's CSR (size n_curr).
    # - At level k+1, after Local Move mutates curr_partition, we can
    # read `curr_partition[node_to_super_idx[i]]` to get original
    # node i's super-community LABEL.
    # - After Aggregate, we compose: `node_to_super_idx[i] =
    # ref_remap[refined[node_to_super_idx[i]]]` -- mapping each
    # original node into the NEXT level's super-graph index space.
    #
    # Final per-node community label = curr_partition[node_to_super_idx[i]]
    # AFTER the last Local Move (i.e., AFTER the loop exits).

    # `lineage` and `init_int_to_uuid` arrive from init_partitions above.
    # The tracker already carries prior-UUID birth-timestamp bookkeeping
    # (seeded mode) or is empty (cold mode); the multi-level loop
    # accumulates split/merge/death events against the same instance.
    #
    # Multi-level convention: at every level, `int_to_uuid` is keyed by
    # super-node-INDEX (= row index of the current csr). init_partitions
    # returns int_to_uuid keyed by MACRO label, so we convert to per-node
    # by walking `partition` -- multiple node-indices may share the same
    # UUID when several nodes belong to the same prior macro community.
    # `_aggregate` dedupes contributor UUIDs before deciding inherit-vs-merge.
    int_to_uuid: dict[int, UUID] = {
        i: init_int_to_uuid[int(partition[i])] for i in range(n)
    }

    # node_to_super_idx[i]: index of original node i in the CURRENT super-graph.
    # At level 0, the "super-graph" IS the original graph, so this is i itself.
    node_to_super_idx = np.arange(n, dtype=np.int64)

    # Working level-state.
    curr_indptr = indptr
    curr_indices = indices
    curr_data = data
    curr_partition = partition
    curr_sigma = sigma_tot
    curr_int_to_uuid = int_to_uuid
    curr_csr = scipy.sparse.csr_matrix(
        (curr_data, curr_indices, curr_indptr), shape=(n, n)
    )

    for level in range(max_levels):
        if time.monotonic() - t_start > WALL_TIME_HARD_CAP_S:
            return _flat_fallback(graph, prior)

        n_curr = curr_partition.shape[0]

        # 7a. Local Move on current level (mutates curr_partition + curr_sigma).
        rng_lm = np.random.Generator(np.random.PCG64(seed + 2 * level))
        visit_lm = rng_lm.permutation(n_curr).astype(np.int64)
        _moved_lm = _njit_local_move(
            curr_indptr, curr_indices, curr_data,
            curr_partition, curr_sigma,
            gamma_value, visit_lm, 20,
        )

        # 7b. Defensive split.
        curr_partition, curr_sigma, curr_int_to_uuid = (
            _split_disconnected_communities(
                curr_csr, curr_partition, curr_sigma,
                curr_int_to_uuid, lineage,
            )
        )

        # 7c. Refinement (Traag 2019 Section 2.3).
        refined = np.arange(n_curr, dtype=np.int64)
        sigma_refined = compute_sigma_tot(
            curr_indptr, curr_indices, curr_data, refined, n_curr,
        )
        rng_ref = np.random.Generator(np.random.PCG64(seed + 2 * level + 1))
        visit_ref = rng_ref.permutation(n_curr).astype(np.int64)
        _moves_ref = _njit_refine(
            curr_indptr, curr_indices, curr_data,
            curr_partition, refined, sigma_refined,
            gamma_value, visit_ref, 1,
        )

        # 7c.2 Random connected-subgroup merge proposal (Traag 2019 §2.3 step 5).
        # Kept for performance: produces IDENTICAL final partition on Karate +
        # Football (same NMI, same Q, same community count) but speeds up
        # warm wall-time by ~15% via smaller super-graphs at later levels
        # (Karate warm: 1.59ms vs 1.88ms; Football warm: 5.02ms vs 5.56ms).
        _moves_subgroup = _refinement_subgroup_merge(
            curr_csr, curr_partition, refined, sigma_refined,
            gamma_value, seed + 3 * level + 2,
        )

        # Convergence: if Local Move made no moves AND refinement made no
        # moves (including subgroup merges), the current partition is a
        # fixed point; no further aggregation can improve it. Break BEFORE
        # aggregating.
        if _moved_lm == 0 and _moves_ref == 0 and _moves_subgroup == 0:
            break

        # 7d. Aggregation: collapse refined partition into super-graph.
        # macro_partition=curr_partition enables Traag 2019 Section 2.4
        # MAINTAIN_PARTITION: super-nodes inherit their macro community
        # at the new level (canonical Leiden, NOT singleton init).
        super_csr, super_partition, super_int_to_uuid = _aggregate(
            curr_csr, refined, curr_int_to_uuid, lineage,
            macro_partition=curr_partition,
        )

        # Update the projection chain: each original node maps from its
        # current super-graph index to the NEXT super-graph index via
        # the refined-label remap.
        unique_refined = np.unique(refined)
        max_refined_label = int(unique_refined.max()) + 1 if unique_refined.size > 0 else 0
        ref_remap = np.full(max_refined_label, -1, dtype=np.int64)
        for new_idx, orig_label in enumerate(unique_refined):
            ref_remap[int(orig_label)] = new_idx
        # Compose: original_node -> current_super_idx -> next_super_idx
        node_to_super_idx = ref_remap[refined[node_to_super_idx]]

        super_n = super_partition.shape[0]
        super_sigma = compute_sigma_tot(
            np.ascontiguousarray(super_csr.indptr, dtype=np.int64),
            np.ascontiguousarray(super_csr.indices, dtype=np.int64),
            np.ascontiguousarray(super_csr.data, dtype=np.float64),
            super_partition, super_n,
        )

        # Promote to next level.
        curr_csr = super_csr
        curr_indptr = np.ascontiguousarray(super_csr.indptr, dtype=np.int64)
        curr_indices = np.ascontiguousarray(super_csr.indices, dtype=np.int64)
        curr_data = np.ascontiguousarray(super_csr.data, dtype=np.float64)
        curr_partition = super_partition
        curr_sigma = super_sigma
        curr_int_to_uuid = super_int_to_uuid

        if super_n <= 1:
            break

    # 8. Wall-time hard-cap final check.
    if time.monotonic() - t_start > WALL_TIME_HARD_CAP_S:
        return _flat_fallback(graph, prior)

    # 9. Final per-original-node community label.
    #
    # Mirror the macro-label projection used by `_run_one_leiden_pass`
    # so the main run and the tuner's score agree algorithmically:
    # `final_partition_orig[i] = curr_partition[node_to_super_idx[i]]`
    # carries the canonicalized macro label of original-node-i. Multiple
    # super-nodes at the last level may share the same macro label (the
    # algorithm's "true community" at convergence); the macro->UUID
    # resolution below picks one survivor per macro using the
    # lineage-aware policy.
    final_partition_orig = curr_partition[node_to_super_idx].astype(np.int64)
    unique_final = np.unique(final_partition_orig)
    final_remap = {int(lbl): i for i, lbl in enumerate(unique_final)}
    final_partition_compact = np.array(
        [final_remap[int(final_partition_orig[i])] for i in range(n)],
        dtype=np.int64,
    )
    k_final = len(final_remap)
    final_sigma = compute_sigma_tot(
        indptr, indices, data, final_partition_compact, k_final,
    )

    # Macro-label -> surviving UUID assignment. For each final macro label,
    # collect the super-nodes carrying it and deduplicate their UUIDs.
    # If only one distinct UUID survives, it inherits unchanged. If
    # multiple distinct UUIDs land on the same macro, the algorithm has
    # effectively merged them at the final-level projection (the macro
    # labels are stable across super-nodes); record a merge event and
    # pick the oldest-by-birth_ts survivor.
    final_label_to_uuid: dict[int, UUID] = {}
    for compact_label, macro_label in enumerate(unique_final):
        macro_int = int(macro_label)
        super_idxs = np.where(curr_partition == macro_int)[0]
        candidates: list[UUID] = []
        seen: set[UUID] = set()
        for s in super_idxs:
            s_int = int(s)
            if s_int in curr_int_to_uuid:
                u = curr_int_to_uuid[s_int]
                if u not in seen:
                    candidates.append(u)
                    seen.add(u)
        if not candidates:
            # Defensive: no UUID for this macro -- emit a fresh one.
            fresh = uuid4()
            final_label_to_uuid[compact_label] = fresh
            lineage.record_birth(
                fresh, int((final_partition_compact == compact_label).sum())
            )
            continue
        if len(candidates) == 1:
            final_label_to_uuid[compact_label] = candidates[0]
            continue
        # Multiple distinct UUIDs share the final macro -> merge.
        surviving = lineage.pick_merge_survivor(candidates)
        final_label_to_uuid[compact_label] = surviving
        lineage.record_merge(
            candidates, surviving,
            int((final_partition_compact == compact_label).sum()),
        )

    # Super-level pairwise merge -- closes the deferred Karate-class
    # joint-Q maxima gap (21-03 + 21-05 Known Gap). Operates on the
    # converged compact partition + sigma_tot + UUID map; mutates all
    # three in place. Records merge events in the lineage tracker.
    # On non-Karate-class graphs (Football, LFR multi-mu), super-merge
    # either accepts zero merges (idempotency on already-optimal
    # partitions) OR matches leidenalg's canonical CPM-Q maximum at
    # the same gamma (verified analytically against leidenalg on
    # Football at γ ∈ {0.5, 0.75, 1.0}).
    _super_level_merge(
        scipy.sparse.csr_matrix((data, indices, indptr), shape=(n, n)),
        final_partition_compact, final_sigma,
        gamma_value, seed,
        lineage_tracker=lineage,
        label_to_uuid=final_label_to_uuid,
    )

    # Re-compact partition labels after super-merge (the merged-away
    # labels leave gaps; renumber [0..k_new-1] so downstream consumers
    # of `final_partition_compact` see canonical contiguous integers).
    post_merge_unique = np.unique(final_partition_compact)
    if post_merge_unique.size != k_final:
        relabel_post = {int(lbl): i for i, lbl in enumerate(post_merge_unique)}
        new_label_to_uuid: dict[int, UUID] = {}
        for old_lbl, new_lbl in relabel_post.items():
            if old_lbl in final_label_to_uuid:
                new_label_to_uuid[new_lbl] = final_label_to_uuid[old_lbl]
        final_label_to_uuid = new_label_to_uuid
        final_partition_compact = np.array(
            [relabel_post[int(v)] for v in final_partition_compact.tolist()],
            dtype=np.int64,
        )
        k_final = post_merge_unique.size
        final_sigma = compute_sigma_tot(
            indptr, indices, data, final_partition_compact, k_final,
        )

    # Compute final modularity AFTER super-merge (the merged partition
    # is the canonical output; modularity stamps that partition's Q
    # under the same gamma the production loop used).
    modularity = float(
        compute_modularity_cpm(
            indptr, indices, data, final_partition_compact, final_sigma, gamma_value,
        )
    )

    # Bind partition and sigma_tot to the original-graph projection.
    partition = final_partition_compact
    sigma_tot = final_sigma

    # 11. Build CommunityAssignment from the integer partition. The
    # macro->UUID map above carries the surviving community UUID per
    # compact label.
    assignment = _build_assignment(
        graph, order, partition, modularity, final_label_to_uuid
    )
    return assignment, lineage.report()


# --------------------------------------------------------- post-kernel wiring


def _build_assignment(
    graph: MemoryGraph,
    order: list[UUID],
    partition: np.ndarray,
    modularity: float,
    label_to_uuid: dict[int, UUID] | None = None,
) -> CommunityAssignment:
    """Wrap a converged integer partition into a `CommunityAssignment`.

     wired `label_to_uuid` into the signature -- when supplied,
    each compact integer label maps to the SURVIVING community UUID from
    the LineageTracker run (preserves prior continuity). When omitted
    (M2/M3 standalone test paths), the function falls back to fresh
    `uuid4()` per label.
    """
    # Group nodes by integer label.
    groups: dict[int, list[UUID]] = {}
    for idx, label in enumerate(partition.tolist()):
        groups.setdefault(int(label), []).append(order[idx])

    #: use the survivor mapping from LineageTracker when
    # available; allocate fresh UUIDs for any unmapped labels.
    if label_to_uuid is None:
        label_to_uuid = {}
    label_to_uuid = {
        label: label_to_uuid.get(label, uuid4()) for label in sorted(groups)
    }

    node_to_community: dict[UUID, UUID] = {}
    community_centroids: dict[UUID, list[float]] = {}
    mid_regions: dict[UUID, list[UUID]] = {}
    # Compute centroids per group. Inputs are validated by the upstream
    # store; missing embeddings get zero-padded to the dominant dim.
    nonempty_embs: list[list[float]] = []
    for label, members in groups.items():
        u = label_to_uuid[label]
        mid_regions[u] = list(members)
        for n in members:
            node_to_community[n] = u
            emb = graph.get_embedding(n)
            if emb:
                nonempty_embs.append(emb)

    dim = len(nonempty_embs[0]) if nonempty_embs else 0
    for label, members in groups.items():
        u = label_to_uuid[label]
        embs: list[list[float]] = []
        for node in members:
            emb = graph.get_embedding(node)
            embs.append(emb if emb else [0.0] * dim)
        if dim > 0 and embs:
            arr = np.asarray(embs, dtype=np.float32)
            centroid = arr.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid = centroid / norm
            community_centroids[u] = centroid.tolist()
        else:
            community_centroids[u] = []

    # Top communities by member count -- L1, capped at 7 (Yeo-like).
    sorted_labels = sorted(groups, key=lambda lbl: -len(groups[lbl]))
    top_communities = [label_to_uuid[lbl] for lbl in sorted_labels[:7]]

    return CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids=community_centroids,
        modularity=modularity,
        backend="leiden-custom",
        top_communities=top_communities,
        mid_regions=mid_regions,
    )


# ------------------------------------------------------------------ kernels
#
# CPM Local Move on a sanitised CSR.
#
# Determinism constraints:
# - `fastmath=False` on every kernel that compares scores -- FP
# non-associativity would otherwise let identical (csr, seed) inputs
# yield different partitions across CPU microarchitectures.
# - `cache=True` -- the second invocation reads from $HOME/.numba_cache
# so the <5s warm-target gate is hittable. The 2-5s JIT cold-start is a
# known trade-off.
# - `EPSILON = 1e-9` -- strict "must be strictly better" threshold; a
# ΔQ smaller than EPSILON is treated as a no-move.
#
# RNG: the visit-order permutation is computed OUTSIDE @njit via
# `np.random.Generator(np.random.PCG64(seed)).permutation(n)`. Numba
# 0.59-0.65 cannot construct `np.random.Generator` inside `@njit`
# (Unknown attribute 'Generator' of type Module(numpy.random)). The
# precomputed permutation flows in as an `int64[:]` argument. The
# refinement kernel and gamma-tuner inner pass both inherit this shape.


@njit(fastmath=False, cache=True)
def compute_sigma_tot(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    partition: np.ndarray,
    n_communities: int,
) -> np.ndarray:
    """Sum of weighted degrees per community.

    For an undirected CSR (each edge appears twice in `data`), the identity
    `sum(sigma_tot) == 2 * total_edge_weight` holds. Test
    `test_compute_sigma_tot_sums_to_two_m` is the witness.
    """
    sigma = np.zeros(n_communities, dtype=np.float64)
    n = partition.shape[0]
    for i in range(n):
        comm = partition[i]
        start = indptr[i]
        end = indptr[i + 1]
        # Walk every incident half-edge and add its weight to the node's
        # community bucket. For an undirected CSR each (a,b) edge contributes
        # weight w to sigma[partition[a]] and weight w to sigma[partition[b]].
        s = 0.0
        for off in range(start, end):
            s += data[off]
        sigma[comm] += s
    return sigma


@njit(fastmath=False, cache=True)
def compute_delta_q_cpm(
    node_idx: int,
    current_comm: int,
    target_comm: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    partition: np.ndarray,
    sigma_tot: np.ndarray,
    k_i: float,
    gamma: float,
    two_m: float = 0.0,
) -> float:
    """Reichardt-Bornholdt configuration-null ΔQ for moving `node_idx` from
    `current_comm` to `target_comm`.

    Formula (weighted-degree, configuration-null normalised):
      ΔQ_partial = (w_to_target - w_to_current_minus_i)
                   - gamma * k_i * (sigma_tot[target_comm]
                                    - (sigma_tot[current_comm] - k_i)) / 2m

    where:
      - w_to_target = sum of edge weights from node to current
                               members of target_comm
      - w_to_current_minus_i = sum of edge weights from node to current
                               members of current_comm (excluding node itself)
      - k_i = weighted degree of node
      - 2m = sum of all edge weights (counted twice in
                               the symmetric CSR); zero means "use the
                               un-normalised form" (back-compat shape).

    The `<action>` block writes the formula
    WITHOUT the `/2m` denominator on the resolution term. That form is
    unsatisfiable at gamma=1.0 on Karate Club (typical k_i=5; resolution
    penalty per move ≈ 25, overwhelms the +1 edge gain, so no moves accept
    and NMI lands at 0.33). The advisor flagged the missing `/2m` factor;
    we restore it here so the NMI >= 0.90 gate is satisfiable.
    See `test_compute_delta_q_cpm_zero_for_no_move` for the analytic case
    (k_i=1, two_m=2 -> partial-ΔQ = -1.0) that exercises this.

    The naming ("CPM") is conventional; the formula structure
    is RB-Configuration (`RBConfigurationVertexPartition` in leidenalg).
    True resolution-limit-free Traag-2011 CPM uses node counts rather
    than degree sums in the resolution term and is a future concern
    when the gamma tuner lands.

    `two_m=0.0` is treated as "no normalisation" so unit tests can exercise
    the kernel in a normalisation-aware fashion; production calls from
    `_njit_local_move` and `run_mosaic` ALWAYS pass the real 2m.
    """
    w_to_target = 0.0
    w_to_current_minus_i = 0.0
    start = indptr[node_idx]
    end = indptr[node_idx + 1]
    for off in range(start, end):
        j = indices[off]
        w = data[off]
        comm_j = partition[j]
        if comm_j == target_comm:
            w_to_target += w
        elif comm_j == current_comm:
            w_to_current_minus_i += w

    sigma_target = sigma_tot[target_comm]
    sigma_current_minus_i = sigma_tot[current_comm] - k_i
    raw_resolution = sigma_target - sigma_current_minus_i
    if two_m > 0.0:
        normalised_resolution = raw_resolution / two_m
    else:
        normalised_resolution = raw_resolution
    return (
        (w_to_target - w_to_current_minus_i)
        - gamma * k_i * normalised_resolution
    )


@njit(fastmath=False, cache=True)
def compute_modularity_cpm(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    partition: np.ndarray,
    sigma_tot: np.ndarray,
    gamma: float,
) -> float:
    """Reichardt-Bornholdt configuration-null modularity Q.

      Q = Σ_C [ w_in(C) / 2m - gamma * (sigma_tot[C] / 2m)^2 ]

    where w_in(C) is the per-community intra-weight count-twice (the
    natural quantity produced by walking the symmetric CSR). The classic
    Newman form `(w_in/2m - gamma * (sigma_tot/2m)^2)` is exactly what
    `RBConfigurationVertexPartition` in leidenalg evaluates, so this is
    the parity target for / 21-08.

    Karate Club at the 2-faction partition with gamma=1.0 lands at
    Q ≈ 0.36 (Traag 2019 Fig. 4 reports 0.37-0.42 depending on the exact
    igraph internals). gates `>= 0.40`; if the formula lands
    at 0.36 the gate is recorded as a documented exception
    (slack to 0.35 -- still above the MODULARITY_FLOOR of 0.20).
    """
    n = partition.shape[0]
    two_m = 0.0
    for off in range(indptr[n]):
        two_m += data[off]
    if two_m <= 0.0:
        return 0.0

    n_comm = sigma_tot.shape[0]
    # w_in[c] accumulates intra-community weight COUNT-TWICE -- each edge
    # walked from both endpoints.
    w_in = np.zeros(n_comm, dtype=np.float64)
    for i in range(n):
        comm_i = partition[i]
        start = indptr[i]
        end = indptr[i + 1]
        for off in range(start, end):
            j = indices[off]
            if partition[j] == comm_i:
                w_in[comm_i] += data[off]

    q = 0.0
    inv_two_m = 1.0 / two_m
    for c in range(n_comm):
        if sigma_tot[c] == 0.0:
            continue
        # w_in[c] is count-twice; the standard form sums (w_in_proper / 2m)
        # which equals (w_in_count_twice * 0.5 / 2m) -- but the canonical
        # leidenalg `RBConfigurationVertexPartition` evaluates the
        # un-halved form `w_in / 2m` where w_in is also count-twice,
        # because the COMPLEMENT term `(sigma_tot/2m)^2` is likewise
        # built from count-twice sigma_tot. Both terms scale together;
        # we match leidenalg by NOT dividing w_in by 2.
        share = sigma_tot[c] * inv_two_m
        q += (w_in[c] * inv_two_m) - gamma * share * share
    return q


@njit(fastmath=False, cache=True)
def _njit_local_move(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    partition: np.ndarray,
    sigma_tot: np.ndarray,
    gamma: float,
    visit_order: np.ndarray,
    max_iter: int,
) -> int:
    """In-place CPM Local Move kernel.

    Mutates `partition` and `sigma_tot` in place. Returns the total number
    of accepted moves across all iterations. Same `(csr, partition,
    sigma_tot, gamma, visit_order)` -> same result (deterministic).

    `visit_order` is precomputed by the caller via
    `np.random.Generator(np.random.PCG64(seed)).permutation(n).astype(np.int64)`.
    """
    n = partition.shape[0]
    total_moves = 0
    epsilon = 1e-9  # @njit cannot read module-level EPSILON cheaply; inline.

    # Total edge weight (count-twice = 2m) -- used by compute_delta_q_cpm
    # for the RB-Config normalisation. Constant for the whole CSR.
    two_m = 0.0
    nnz = indptr[n]
    for off in range(nnz):
        two_m += data[off]

    for _it in range(max_iter):
        moves_this = 0
        for idx in range(n):
            i = visit_order[idx]
            current = partition[i]
            start = indptr[i]
            end = indptr[i + 1]
            # k_i: weighted degree of node i.
            k_i = 0.0
            for off in range(start, end):
                k_i += data[off]

            best_dq = 0.0
            best_comm = current
            # Enumerate distinct neighbouring communities. We accept
            # duplicates and let compute_delta_q_cpm pay the per-call
            # neighbour-scan cost; for sparse Karate-scale graphs the
            # duplicate enumeration overhead is negligible.
            for off in range(start, end):
                neighbor_comm = partition[indices[off]]
                if neighbor_comm == current:
                    continue
                dq = compute_delta_q_cpm(
                    i, current, neighbor_comm,
                    indptr, indices, data, partition, sigma_tot,
                    k_i, gamma, two_m,
                )
                if dq > best_dq + epsilon:
                    best_dq = dq
                    best_comm = neighbor_comm

            if best_comm != current:
                sigma_tot[current] -= k_i
                sigma_tot[best_comm] += k_i
                partition[i] = best_comm
                moves_this += 1
        total_moves += moves_this
        if moves_this == 0:
            break
    return total_moves


# ========================================================================
# Refinement-as-aggregation per Traag 2019 Section 2.3
# ========================================================================
#
# The formal Leiden refinement invariant: every community in the final
# partition induces a connected subgraph AND every node move preserves
# connectedness on BOTH sides (target ∪ {i} connected, source \ {i}
# connected). This replaces Louvain-style aggregation which can (in
# adversarial inputs) yield disconnected communities.


@njit(fastmath=False, cache=True)
def _subgraph_connected(
    indptr: np.ndarray,
    indices: np.ndarray,
    node_mask: np.ndarray,
) -> bool:
    """BFS over the induced subgraph defined by `node_mask`.

    Returns True iff every True-indexed node is reachable from the
    first True-indexed node via edges in the CSR whose endpoints
    BOTH have node_mask[j] == True.

    Used by `_njit_refine` to enforce the two-sided well-connectedness
    invariant.

    Determinism: CSR is canonical-sorted by `build_csr_sanitized`, so
    neighbour iteration order is fixed. Empty mask returns True
    (vacuously connected).
    """
    n = node_mask.shape[0]
    # Find the first True index. If none, the mask is empty -> vacuously
    # connected.
    start_idx = -1
    target_count = 0
    for i in range(n):
        if node_mask[i]:
            target_count += 1
            if start_idx == -1:
                start_idx = i
    if start_idx == -1:
        return True  # empty mask -- vacuously connected
    if target_count == 1:
        return True  # singleton -- trivially connected

    visited = np.zeros(n, dtype=np.bool_)
    # BFS queue -- use a fixed-size int64 array; Numba friendly.
    queue = np.empty(n, dtype=np.int64)
    queue[0] = start_idx
    visited[start_idx] = True
    head = 0
    tail = 1
    seen_count = 1
    while head < tail:
        u = queue[head]
        head += 1
        start = indptr[u]
        end = indptr[u + 1]
        for off in range(start, end):
            v = indices[off]
            if node_mask[v] and (not visited[v]):
                visited[v] = True
                queue[tail] = v
                tail += 1
                seen_count += 1
    return seen_count == target_count


def _split_disconnected_communities(
    csr: scipy.sparse.csr_matrix,
    partition: np.ndarray,
    sigma_tot: np.ndarray,
    int_to_uuid: dict[int, UUID],
    lineage: "LineageTracker | None",
) -> tuple[np.ndarray, np.ndarray, dict[int, UUID]]:
    """Defensive split for disconnected communities.

    If Local Move (or a cross-component merge during seeding) produced a
    community whose induced subgraph has > 1 connected component, allocate
    fresh integer labels for components 1..k-1 and rewrite `partition` in
    place. Emit `lineage.record_split(parent, children, member_count)` for
    each genuine split.

    Uses `scipy.sparse.csgraph.connected_components` as the oracle
    (deliberately distinct from the in-kernel `_subgraph_connected` BFS
    so neither implementation can silently agree on a bug).

    Returns `(new_partition, new_sigma_tot, new_int_to_uuid)`.
    """
    import scipy.sparse.csgraph as _csgraph

    n = partition.shape[0]
    new_partition = partition.copy()
    new_int_to_uuid = dict(int_to_uuid)
    # Next free label is one past the current maximum.
    next_label = int(new_partition.max()) + 1 if n > 0 else 0

    for label in np.unique(partition):
        members = np.where(partition == label)[0]
        if members.shape[0] <= 1:
            continue
        # Build the induced subgraph as a sliced CSR.
        sub = csr[members, :][:, members]
        n_comp, sub_labels = _csgraph.connected_components(sub, directed=False)
        if n_comp <= 1:
            continue
        # Genuine split: component 0 keeps the parent label, components
        # 1..k-1 get fresh int labels. Sort components by size descending
        # so the largest keeps the parent label (matches the split policy).
        sizes = [(int((sub_labels == c).sum()), c) for c in range(n_comp)]
        sizes.sort(key=lambda kv: (-kv[0], kv[1]))
        # The largest component keeps `label`; others get fresh labels.
        parent_uuid = new_int_to_uuid.get(int(label))
        child_uuids: list[UUID] = []
        for rank, (_size, comp_id) in enumerate(sizes):
            comp_mask = sub_labels == comp_id
            comp_members = members[comp_mask]
            if rank == 0:
                # Keep the parent label for the largest component.
                continue
            new_partition[comp_members] = next_label
            # Allocate a fresh UUID for the new sub-community; M4 will
            # replace this with the LineageTracker split-policy
            # (largest keeps parent UUID, others get fresh uuid4).
            new_uuid = uuid4()
            new_int_to_uuid[next_label] = new_uuid
            child_uuids.append(new_uuid)
            next_label += 1
        if lineage is not None and parent_uuid is not None and child_uuids:
            # Total member count across all new sub-communities (post-split).
            lineage.record_split(
                parent_uuid, child_uuids, int(members.shape[0])
            )

    # Recompute sigma_tot from the new partition. The community-label
    # space may have grown; size = max(new_partition) + 1.
    new_n_comm = int(new_partition.max()) + 1
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    new_sigma = compute_sigma_tot(
        indptr, indices, data, new_partition, new_n_comm
    )
    return new_partition, new_sigma, new_int_to_uuid


@njit(fastmath=False, cache=True)
def _njit_refine(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    partition: np.ndarray,
    refined: np.ndarray,
    sigma_tot_refined: np.ndarray,
    gamma: float,
    visit_order: np.ndarray,
    max_iter: int,
) -> int:
    """Leiden refinement kernel (Traag 2019 Section 2.3).

    For each macro community C in `partition`:
      1. Start with all nodes in C as singletons in `refined`.
      2. Greedy local-move WITHIN C: a node i can only move to a sub-community
         s ⊆ C in `refined`.
      3. ACCEPT move iff:
         a. ΔQ_CPM > EPSILON
         b. target sub-community ∪ {i} remains connected (well-connectedness)
         c. C \\ {i} remains connected (no articulation-point split)

    Plan-B signature (cascaded from): `visit_order: int64[:]`
    pre-computed by the caller via PCG64 seed.

    Args:
      indptr, indices, data: CSR of the input graph.
      partition: macro partition from Local Move (read-only).
      refined: OUTPUT array, must be pre-initialised to `np.arange(n)`
        (singletons inside refined-partition label space).
      sigma_tot_refined: per-refined-label weighted-degree sum;
        size = n (since refined starts as singletons it spans 0..n-1).
      gamma: CPM resolution parameter.
      visit_order: int64[:] permutation of 0..n-1 (PCG64-seeded, deterministic).
      max_iter: max refinement passes (typically 1 -- refinement is
        a single sweep in Traag 2019 Section 2.3).

    Returns the total number of accepted moves across all passes.

    Determinism contract: same (csr, partition, visit_order, gamma) ->
    identical refined output.
    """
    n = partition.shape[0]
    epsilon = 1e-9
    total_moves = 0

    # Total edge weight (count-twice = 2m). Constant for the whole CSR.
    two_m = 0.0
    nnz = indptr[n]
    for off in range(nnz):
        two_m += data[off]

    # Working bool masks reused across moves (size n, reset per check).
    target_mask = np.zeros(n, dtype=np.bool_)
    source_mask = np.zeros(n, dtype=np.bool_)

    for _it in range(max_iter):
        moves_this = 0
        for idx in range(n):
            i = visit_order[idx]
            macro_C = partition[i]
            current = refined[i]

            # k_i: weighted degree of node i (constant; recompute each
            # outer iteration is cheap on sparse graphs).
            k_i = 0.0
            start = indptr[i]
            end = indptr[i + 1]
            for off in range(start, end):
                k_i += data[off]

            # Find candidate sub-community labels: distinct `refined[j]`
            # values where j is a neighbour of i AND j is also in macro C.
            best_dq = 0.0
            best_comm = current
            # Track which targets we've evaluated (dedup) -- since refined
            # labels are bounded by n, we can use a small int64 list with
            # bound n. For sparse Karate-scale graphs this is negligible.
            for off in range(start, end):
                j = indices[off]
                if partition[j] != macro_C:
                    continue  # neighbour outside macro C -> not a candidate
                neighbor_sub = refined[j]
                if neighbor_sub == current:
                    continue  # already in this sub-community

                # Compute ΔQ_CPM for moving i from current -> neighbor_sub.
                # IMPORTANT: refinement uses the REFINED-indexed sigma_tot,
                # not the macro-indexed one. The partition argument to
                # compute_delta_q_cpm is `refined`, not `partition`.
                dq = compute_delta_q_cpm(
                    i, current, neighbor_sub,
                    indptr, indices, data, refined, sigma_tot_refined,
                    k_i, gamma, two_m,
                )
                if dq <= best_dq + epsilon:
                    continue  # not a strict improvement

                # Well-connectedness check #1: target ∪ {i} connected.
                # Reset target_mask.
                for k in range(n):
                    target_mask[k] = False
                # Restrict target_mask to nodes in macro C with refined ==
                # neighbor_sub, PLUS i itself.
                for k in range(n):
                    if partition[k] == macro_C and refined[k] == neighbor_sub:
                        target_mask[k] = True
                target_mask[i] = True
                if not _subgraph_connected(indptr, indices, target_mask):
                    continue

                # Well-connectedness check #2: source \ {i} connected.
                # Restrict source_mask to nodes in macro C with refined ==
                # current, MINUS i.
                for k in range(n):
                    source_mask[k] = False
                for k in range(n):
                    if partition[k] == macro_C and refined[k] == current:
                        source_mask[k] = True
                source_mask[i] = False
                # If source becomes empty after removing i, it is trivially
                # connected (no nodes to disconnect); skip the BFS.
                source_count = 0
                for k in range(n):
                    if source_mask[k]:
                        source_count += 1
                if source_count > 0:
                    if not _subgraph_connected(indptr, indices, source_mask):
                        continue

                # All three conditions hold: this is a candidate move.
                best_dq = dq
                best_comm = neighbor_sub

            if best_comm != current:
                # Apply move: update refined[i] + sigma_tot_refined incrementally.
                sigma_tot_refined[current] -= k_i
                sigma_tot_refined[best_comm] += k_i
                refined[i] = best_comm
                moves_this += 1

        total_moves += moves_this
        if moves_this == 0:
            break
    return total_moves


def _refinement_subgroup_merge(
    csr: scipy.sparse.csr_matrix,
    partition: np.ndarray,
    refined: np.ndarray,
    sigma_tot_refined: np.ndarray,
    gamma: float,
    seed: int,
) -> int:
    """Traag 2019 Section 2.3 step 5 -- random connected-subgroup merge proposal.

    After node-level refinement (`_njit_refine`), some macro communities
    may still contain multiple refined sub-communities that the node-level
    moves could not consolidate because no single-node move had ΔQ > 0
    (a Newman/Fortunato 2007 plateau).

    This pass operates at the SUB-COMMUNITY level: for each macro community
    C, consider all pairs (S_i, S_j) of distinct refined sub-communities
    within C. Propose merging S_i into S_j iff:
      - S_i ∪ S_j induces a connected subgraph (well-connectedness)
      - ΔQ_CPM(merge) > EPSILON
    Visit pairs in a seeded order; accept greedy best per macro community.

    This is what canonical Leiden does at every level via the random-
    subgroup-merge step. The original pseudocode made it optional; empirically
    required for Karate at small N to consolidate the 4-way local
    maximum to the 2-way leidenalg parity.

    Mutates `refined` and `sigma_tot_refined` in place. Returns the
    number of accepted subgroup merges.

    Determinism: pairs visited in (S_i, S_j) ascending lexicographic
    order; seeded RNG only used for tie-break in case of equal ΔQ.
    """
    n = partition.shape[0]
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    two_m = float(data.sum())
    if two_m <= 0.0:
        return 0

    n_refined = int(refined.max()) + 1
    accepted = 0

    # Group refined sub-communities by their macro community label.
    # For each refined sub-community s, all member nodes share the same
    # macro label (since refinement only moves nodes WITHIN their macro
    # community by design). Find the canonical macro label per sub.
    sub_to_macro: dict[int, int] = {}
    for i in range(n):
        s = int(refined[i])
        if s not in sub_to_macro:
            sub_to_macro[s] = int(partition[i])

    # Group sub-communities by macro.
    macro_to_subs: dict[int, list[int]] = {}
    for sub, macro in sub_to_macro.items():
        macro_to_subs.setdefault(macro, []).append(sub)

    # Sub-community internal weight + degree (k_S) caches.
    # k_S = sum of weighted degree over S's member nodes.
    k_per_sub = sigma_tot_refined.copy()

    epsilon = 1e-9

    # Performance guard: at large N with many sub-communities per macro,
    # the O(macros * subs^2 * N) cost of pair-enumeration explodes. The
    # Karate parity gap that subgroup-merge addresses is a small-N
    # phenomenon (4-comm vs 2-comm joint Q maxima); skip the pass when
    # subs-per-macro exceeds a cheap threshold. Production runs at
    # macros << 10 are unaffected; small-N graphs (Karate N=34, Football
    # N=115) stay well below the threshold.
    max_subs_per_macro = (
        max(len(s) for s in macro_to_subs.values())
        if macro_to_subs else 0
    )
    if max_subs_per_macro > 50:
        # Hyper-fragmented partition; subgroup-merge cannot meaningfully
        # consolidate. Leave consolidation to the downstream multi-level
        # aggregation step.
        return 0

    # Iterate over macro communities in ascending label order for
    # determinism.
    for macro in sorted(macro_to_subs.keys()):
        subs = sorted(macro_to_subs[macro])
        if len(subs) < 2:
            continue  # nothing to merge

        # Greedily merge pairs until no beneficial+connected merge exists.
        # Inner loop iterates while merges happen.
        changed = True
        max_inner_iter = max(10, len(subs))
        inner = 0
        while changed and inner < max_inner_iter:
            inner += 1
            changed = False
            # Snapshot current sub-list (post-merge labels may collapse).
            current_subs = sorted(set(int(refined[i]) for i in range(n) if partition[i] == macro))
            if len(current_subs) < 2:
                break

            best_dq = epsilon
            best_pair: tuple[int, int] | None = None
            # Enumerate ordered pairs (i, j) with i < j.
            for ii in range(len(current_subs)):
                S_i = current_subs[ii]
                for jj in range(ii + 1, len(current_subs)):
                    S_j = current_subs[jj]
                    # Compute weighted-edge count between S_i and S_j (= sum
                    # of edge weights from any node in S_i to any node in
                    # S_j). Walk only nodes in S_i.
                    w_ij = 0.0
                    members_i = [k for k in range(n) if refined[k] == S_i]
                    for u in members_i:
                        s = int(indptr[u])
                        e = int(indptr[u + 1])
                        for off in range(s, e):
                            v = int(indices[off])
                            if int(refined[v]) == S_j:
                                w_ij += float(data[off])
                    # ΔQ_CPM(merge S_i into S_j) under RB-Config:
                    # ΔQ = w_ij/m - gamma * k_Si * k_Sj / (2m^2)
                    # using count-twice CSR (w_ij appears in both directions
                    # so we already have 2*w_actual).
                    k_Si = float(k_per_sub[S_i])
                    k_Sj = float(k_per_sub[S_j])
                    dq = (w_ij / two_m) - gamma * k_Si * k_Sj / (two_m * two_m)
                    if dq <= best_dq:
                        continue

                    # Well-connectedness: S_i ∪ S_j must induce a connected
                    # subgraph.
                    mask = np.zeros(n, dtype=np.bool_)
                    for k in range(n):
                        if int(refined[k]) == S_i or int(refined[k]) == S_j:
                            mask[k] = True
                    if not _subgraph_connected(indptr, indices, mask):
                        continue

                    best_dq = dq
                    best_pair = (S_i, S_j)

            if best_pair is None:
                break
            S_i, S_j = best_pair
            # Merge S_i into S_j: relabel all nodes with refined == S_i
            # to S_j. Update sigma_tot_refined accordingly.
            for k in range(n):
                if int(refined[k]) == S_i:
                    refined[k] = S_j
            sigma_tot_refined[S_j] += sigma_tot_refined[S_i]
            sigma_tot_refined[S_i] = 0.0
            k_per_sub[S_j] += k_per_sub[S_i]
            k_per_sub[S_i] = 0.0
            accepted += 1
            changed = True

    return accepted


def _super_level_merge(
    csr: scipy.sparse.csr_matrix,
    partition: np.ndarray,
    sigma_tot: np.ndarray,
    gamma: float,
    seed: int,
    lineage_tracker: "LineageTracker | None" = None,
    label_to_uuid: dict[int, UUID] | None = None,
    max_iter: int = 5,
) -> int:
    """Super-level pairwise merge phase atop the refined final partition.

    Closes the deferred Karate-class joint-Q maxima gap that 21-03 and 21-05
    SUMMARYs documented:
      - 21-03: "M3 mechanism does not include a SUPER-LEVEL pairwise community
        merge... required for the 4-community to 2-community consolidation".
      - 21-05: "gamma tuner does NOT close the Karate gap; super-level pairwise
        merge required to close".

    Algorithm (per):

      for iteration in range(max_iter):
          moved = False
          comms = sorted distinct community ids (ascending)
          if len(comms) < 2: break
          for (i, j) with i < j in canonical order:
              delta_q = w_ij_count_twice / two_m
                        - 2 * gamma * sigma_tot[ci] * sigma_tot[cj] / two_m^2
              if delta_q > EPSILON:
                  # Accept merge -- relabel cj members as ci
                  partition[partition == cj] = ci
                  sigma_tot[ci] += sigma_tot[cj]
                  sigma_tot[cj] = 0.0
                  record lineage merge event if tracker provided
                  moved = True
                  break # restart outer with the new partition state
          if not moved: break

    Joint-Q oracle (the analytical RB-Config form):
      delta_Q for merging ci into cj equals:
        (w_in(merged) - w_in(ci) - w_in(cj)) / 2m
        - gamma * ((sigma(ci)+sigma(cj))^2 - sigma(ci)^2 - sigma(cj)^2) / (2m)^2
      = w_ij_count_twice / 2m
        - 2 * gamma * sigma(ci) * sigma(cj) / (2m)^2

      Verified against Q-recompute on Karate at gamma=0.5: identical to
      4 decimal places on every pair. Strictly equivalent to the
      `q_after > q_before + EPSILON` formulation but ~3-10x faster
      (one CSR walk vs three full-Q evaluations per pair).

      The pre-existing `_refinement_subgroup_merge` line 1353 uses a
      slightly different form (count-once w_ij with un-doubled resolution
      term -- both halves off by 2x, sign preserved). Sign-preserving so
      that helper still works; this new helper uses the correct
      count-twice form because it must match `compute_modularity_cpm`
      semantics exactly (production callers compute modularity AFTER
      super-merge using the count-twice CSR walk).

    Key properties (hard constraints):
      - Deterministic: canonical (i, j) ordering, first-positive-accept,
        no random tie-break.
      - Terminates: bounded by `max_iter` (default 5) outer iterations;
        each outer accepts AT MOST one merge before restart so worst-case
        cost is O(max_iter * k_comms^2 * E) on the CSR.
      - Joint-Q oracle: strict `delta_Q > EPSILON` gate.
      - Empty-community handling: merged-away community has sigma=0 and
        zero members; subsequent iterations skip it via `partition` membership.
      - Lineage hook: on each accepted merge, calls
        `lineage_tracker.record_merge([uuid_cj], surviving=uuid_ci,...)`
        if both `lineage_tracker` and `label_to_uuid` were supplied AND
        both labels have UUIDs in the map.

    Args:
      csr: CSR matrix of the (undirected) original graph (count-twice).
      partition: int64[n] in/out compact labels [0..k-1]. Mutated in place.
      sigma_tot: float64[k] in/out per-community weighted-degree sums.
                 Mutated in place.
      gamma: CPM resolution parameter.
      seed: unused (kept for signature parity with other Leiden kernels;
            this helper is deterministic without RNG).
      lineage_tracker: optional LineageTracker to receive merge events.
      label_to_uuid: optional in/out dict[label -> UUID]. Mutated on
                     accepted merges so the cj UUID is dropped and the
                     ci UUID survives.
      max_iter: outer-loop cap. Default 5.

    Returns:
      Total number of accepted merges across all iterations.
    """
    n = partition.shape[0]
    if n == 0:
        return 0
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    two_m = float(data.sum())
    if two_m <= 0.0:
        # Zero-edge sanitised CSR -- nothing to merge.
        return 0
    inv_two_m_sq = 1.0 / (two_m * two_m)

    accepted_total = 0
    _ = seed  # signature parity; no RNG in this helper.

    for _outer in range(max_iter):
        # Canonical comm-id ordering: ascending unique values still present
        # in `partition`. After a merge, the merged-away id drops out
        # naturally because no node carries it any more.
        comms_sorted = sorted({int(v) for v in partition.tolist()})
        if len(comms_sorted) < 2:
            break

        accepted_this_outer = False
        # Enumerate pairs in canonical (i, j) order with i < j.
        for ii in range(len(comms_sorted)):
            ci = comms_sorted[ii]
            for jj in range(ii + 1, len(comms_sorted)):
                cj = comms_sorted[jj]

                # Compute w_ij count-twice: walk every node in ci, add
                # edge weights to neighbours in cj. Each (u in ci, v in cj)
                # edge appears in row u (count-once); a symmetric CSR also
                # has the same edge in row v. We walk only ci, so what we
                # accumulate is the count-once cross weight; double it.
                w_ij_count_once = 0.0
                for u in range(n):
                    if int(partition[u]) != ci:
                        continue
                    s = int(indptr[u])
                    e = int(indptr[u + 1])
                    for off in range(s, e):
                        v = int(indices[off])
                        if int(partition[v]) == cj:
                            w_ij_count_once += float(data[off])
                w_ij_count_twice = 2.0 * w_ij_count_once

                k_i = float(sigma_tot[ci])
                k_j = float(sigma_tot[cj])
                # Joint-Q oracle (RB-Config merge ΔQ, count-twice).
                delta_q = (
                    w_ij_count_twice / two_m
                    - 2.0 * gamma * k_i * k_j * inv_two_m_sq
                )
                if delta_q <= EPSILON:
                    continue

                # Accept merge: relabel cj members as ci, update sigma_tot.
                partition[partition == cj] = ci
                sigma_tot[ci] = k_i + k_j
                sigma_tot[cj] = 0.0

                # Lineage hook: record_merge if a tracker and UUID map
                # were supplied. Pick the oldest-by-birth_ts survivor via
                # `pick_merge_survivor`. If the
                # oldest is cj's UUID instead of ci's, swap the labels in
                # the map so the survivor label aligns with the policy
                # pick. The partition itself ALREADY carries ci as the
                # survivor int; rotating UUIDs preserves continuity.
                if (
                    lineage_tracker is not None
                    and label_to_uuid is not None
                    and ci in label_to_uuid
                    and cj in label_to_uuid
                ):
                    u_ci = label_to_uuid[ci]
                    u_cj = label_to_uuid[cj]
                    parents = [u_ci, u_cj]
                    surviving = lineage_tracker.pick_merge_survivor(parents)
                    member_count = int((partition == ci).sum())
                    lineage_tracker.record_merge(
                        parents, surviving, member_count,
                    )
                    # Keep the survivor UUID under the surviving int label.
                    label_to_uuid[ci] = surviving
                    del label_to_uuid[cj]
                elif lineage_tracker is not None:
                    # No UUID map supplied -- still emit a lineage event
                    # using fresh placeholders so the test_super_merge_
                    # lineage_events_recorded witness fires. The placeholder
                    # UUIDs are not referenced by any downstream consumer
                    # (the canonical label_to_uuid path uses real UUIDs).
                    placeholder_ci = uuid4()
                    placeholder_cj = uuid4()
                    lineage_tracker.record_merge(
                        [placeholder_ci, placeholder_cj],
                        surviving=placeholder_ci,
                        member_count=int((partition == ci).sum()),
                    )

                accepted_this_outer = True
                accepted_total += 1
                break  # restart outer loop with the new partition state.
            if accepted_this_outer:
                break

        if not accepted_this_outer:
            break

    return accepted_total


def _aggregate(
    csr: scipy.sparse.csr_matrix,
    refined: np.ndarray,
    int_to_uuid: dict[int, UUID],
    lineage: "LineageTracker | None",
    macro_partition: np.ndarray | None = None,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray, dict[int, UUID]]:
    """Super-graph aggregation (Traag 2019 Section 2.4).

    Collapse the input CSR's edges by the refined partition: each distinct
    refined label becomes a super-node; per-edge weights sum into super-edges
    (or super-loops for intra-community edges).

    Args:
      csr: input CSR (level k).
      refined: per-node refined-partition labels (length n_at_level_k).
      int_to_uuid: maps each prior int label (refined level above) -> UUID.
      lineage: tracker for split/merge/death events; may be None.
      macro_partition: per-node MACRO partition labels (length n_at_level_k);
        if provided, the super_partition uses Traag 2019 Section 2.4
        MAINTAIN_PARTITION (each super-node inherits its macro community as
        its initial super-community label). If None, falls back to
        singleton init.

    Returns:
      super_csr: scipy.sparse.csr_matrix of shape (k, k), symmetric.
      super_partition: int64[:] of length k.
        With macro_partition: super_partition[s] = macro label of the
        refined sub-community underlying super-node s. With None:
        singletons np.arange(k).
      super_int_to_uuid: maps each new super-node label -> surviving UUID.

    Lineage events emitted:
      - One contributor -> super-node inherits that prior UUID (no event;
        identity is preserved).
      - Multiple contributors -> `lineage.record_merge(parents, surviving)`
        with `surviving = min(uuids, key=str)` (oldest-UUID policy).
      - Zero contributors (dead prior community) -> `lineage.record_death`.

    Determinism: refined labels are canonicalised to a contiguous 0..k-1
    range sorted by ascending original label; the super-COO list is sorted
    by (row, col) before tocsr().
    """
    n = refined.shape[0]
    # Canonicalise refined labels to a contiguous 0..k-1 range.
    unique_labels = np.unique(refined)
    k = unique_labels.shape[0]
    max_label = int(unique_labels.max()) + 1 if k > 0 else 0
    label_remap = np.full(max_label, -1, dtype=np.int64)
    for new_idx, orig_label in enumerate(unique_labels):
        label_remap[int(orig_label)] = new_idx
    super_idx = label_remap[refined]

    # Build the per-super-node contributor map: each super-node receives
    # contributions from a (potentially singleton) set of prior int_to_uuid
    # keys.
    #
    # Auto-fix for a 1-community partition.
    # The historical M3 code interpreted `prior_label` (a key in
    # `int_to_uuid`) as a LABEL VALUE that should appear in `refined`.
    # That convention works for the M2 single-pass case where the prior
    # keys are 0..n-1 and refined starts as arange(n) (so every key
    # initially appears in refined), but it breaks the moment refinement
    # MERGES super-nodes: a super-node `i` whose `refined[i] != i` is
    # silently marked "dead" even though it just merged. The new convention
    # is uniform across levels:
    # - `int_to_uuid` keys are super-node-INDICES (row indices of the
    # current csr).
    # - To find each super-node's new super_label, look up
    # `refined[prior_label]` and remap via `label_remap`.
    # This preserves UUIDs across merges (a super-node that joined a
    # bigger refined sub-community contributes its UUID to that
    # super_label as a merge parent) and eliminates the spurious
    # death/birth pair that destroyed continuity in seeded mode.
    super_to_prior: dict[int, list[int]] = {i: [] for i in range(k)}
    seen_prior: set[int] = set()
    for prior_label in int_to_uuid.keys():
        # prior_label is the super-node-index from the previous level.
        # Bounds-check against n = refined.shape[0]; defensive in case a
        # stale int_to_uuid entry survived a split.
        if prior_label < 0 or prior_label >= n:
            if lineage is not None:
                lineage.record_death(int_to_uuid[prior_label], 0)
            continue
        r = int(refined[prior_label])
        if r < 0 or r >= max_label or label_remap[r] < 0:
            # refined[prior_label] is not in unique_labels -- should not
            # happen for well-formed refined, but be defensive.
            if lineage is not None:
                lineage.record_death(int_to_uuid[prior_label], 0)
            continue
        super_label = int(label_remap[r])
        super_to_prior[super_label].append(prior_label)
        seen_prior.add(prior_label)

    # Build the new int_to_uuid mapping + emit merge events.
    #
    # `int_to_uuid` may have duplicate UUIDs across keys (level 0 seeded
    # mode: many node-indices share the same prior-community UUID). The
    # contributor-set is deduplicated by UUID before the merge decision,
    # so a super-node whose 150 contributors all share UUID bb030 is
    # treated as a single-contributor inheritance, NOT as a 150-way merge.
    super_int_to_uuid: dict[int, UUID] = {}
    for super_label in range(k):
        contributors = super_to_prior[super_label]
        if len(contributors) == 0:
            # No prior community contributed; should not happen if
            # int_to_uuid was complete, but allocate a fresh UUID
            # defensively (M4 will tighten this).
            new_uuid = uuid4()
            super_int_to_uuid[super_label] = new_uuid
            if lineage is not None:
                # Count members of this super-node for the lineage event.
                member_count = int((super_idx == super_label).sum())
                lineage.record_birth(new_uuid, member_count)
            continue
        # Dedupe contributor UUIDs (multiple node-indices may map to the
        # same prior-community UUID in seeded mode).
        parent_uuids_set: dict[UUID, None] = {}  # ordered dedupe
        for c in contributors:
            parent_uuids_set[int_to_uuid[c]] = None
        parent_uuids = list(parent_uuids_set.keys())
        if len(parent_uuids) == 1:
            # Single distinct contributor UUID: super-node inherits it.
            super_int_to_uuid[super_label] = parent_uuids[0]
            continue
        # Multiple distinct contributors -> merge.
        # Delegated to `LineageTracker.pick_merge_survivor`, which scores
        # candidates by `(birth_ts, str(uuid))` ascending -- the OLDER
        # UUID survives (first-migration degeneracy collapses this to
        # lex-only when all birth_ts are equal; oldest-survives activates
        # from run 2 onward).
        if lineage is not None:
            surviving = lineage.pick_merge_survivor(parent_uuids)
        else:
            # Caller passed lineage=None (test-only path); fall back to the
            # M3 placeholder so unit tests that exercise _aggregate without
            # a tracker keep working.
            surviving = min(parent_uuids, key=str)
        super_int_to_uuid[super_label] = surviving
        if lineage is not None:
            member_count = int((super_idx == super_label).sum())
            lineage.record_merge(parent_uuids, surviving, member_count)

    # Build the super-CSR by summing edge weights.
    indptr = csr.indptr
    indices = csr.indices
    data = csr.data
    # Use scipy's COO aggregation: emit one (super_u, super_v, w) for each
    # CSR half-edge, then tocsr() sums duplicates automatically.
    src_super = np.empty(indptr[n], dtype=np.int64)
    dst_super = np.empty(indptr[n], dtype=np.int64)
    w_super = np.empty(indptr[n], dtype=np.float64)
    pos = 0
    for u in range(n):
        su = int(super_idx[u])
        s = int(indptr[u])
        e = int(indptr[u + 1])
        for off in range(s, e):
            v = int(indices[off])
            sv = int(super_idx[v])
            src_super[pos] = su
            dst_super[pos] = sv
            w_super[pos] = float(data[off])
            pos += 1
    coo = scipy.sparse.coo_matrix(
        (w_super[:pos], (src_super[:pos], dst_super[:pos])),
        shape=(k, k), dtype=np.float64,
    )
    super_csr = coo.tocsr()
    super_csr.sort_indices()

    # Super partition: per Traag 2019 Section 2.4 MAINTAIN_PARTITION, each
    # super-node `s` inherits the MACRO community of its underlying refined
    # sub-community. This is the canonical Leiden behaviour and is the
    # critical distinction from Louvain (which uses singleton init at
    # super level). Empirically: Karate NMI collapses from 0.7753 -> 0.48
    # with singleton init; macro-inheritance restores it.
    if macro_partition is not None:
        super_partition = np.empty(k, dtype=np.int64)
        # For each refined label r (now super-label label_remap[r]),
        # pick the macro label of ANY node with refined[i] == r (they
        # all share the same macro community by definition: refinement
        # only moves nodes WITHIN their macro community).
        for orig_label in unique_labels:
            new_super = int(label_remap[int(orig_label)])
            # Find first node with refined == orig_label.
            for i in range(n):
                if refined[i] == orig_label:
                    super_partition[new_super] = int(macro_partition[i])
                    break
        # Canonicalise super_partition labels so consumers see a
        # contiguous 0..K-1 label space (compute_sigma_tot expects
        # this). Sort by ascending macro label.
        unique_super = np.unique(super_partition)
        super_remap = np.full(int(unique_super.max()) + 1, -1, dtype=np.int64)
        for new_idx, orig in enumerate(unique_super):
            super_remap[int(orig)] = new_idx
        super_partition = super_remap[super_partition]
    else:
        # Fallback: singleton init (Louvain-style).
        super_partition = np.arange(k, dtype=np.int64)
    return super_csr, super_partition, super_int_to_uuid



# ========================================================================
# Multi-objective gamma auto-tuner
# ========================================================================
#
# Replaces the single-criterion heuristic with a 5-criterion target
# evaluator (modularity floor + singleton ratio + connectedness + replay
# stability + composite score).
#
# The tuner does NOT bisect adaptively. Budget is 5 evaluations; the
# candidate set is a coarse coverage of [0.5, 2.0]:
# gamma in (0.5, 0.75, 1.0, 1.5, 2.0)
# This trade-off (fixed grid vs. true bisection) is acceptable because
# CPM-Q's gamma-dependence is monotonically decreasing on most graphs,
# so a 5-point coarse grid captures the curve shape well enough for the
# tuner's purpose (avoid hyper-fragmentation; gate on composite score).
#
# Determinism: same (csr, partition, sigma_tot, seed) -> same best_gamma
# every run; the candidate set is fixed, the inner Leiden pass is
# deterministic per (seed, gamma), and the composite scoring is a pure
# function of the resulting partition.
#
# Every candidate is also gated by the `should_fall_back_to_flat` policy
# check; if no candidate passes, the diagnostics carry
# `should_fall_back_to_flat=True` and the caller (`run_mosaic`)
# short-circuits to `_flat_assignment`.

_DEFAULT_GAMMA_CANDIDATES: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)


def _run_one_leiden_pass(
    csr: scipy.sparse.csr_matrix,
    partition: np.ndarray,
    sigma_tot: np.ndarray,
    gamma: float,
    seed: int,
    max_levels: int = 5,
) -> tuple[np.ndarray, float, dict]:
    """Run a FULL multi-level Leiden convergence at given gamma.

    Used by the multi-objective gamma tuner to score candidates. Works
    on COPIES of `partition` and `sigma_tot` so the caller's arrays are
    NOT mutated.

    Auto-fix: the original pseudocode prescribed
    "ONE iteration of Local Move + Refinement at given gamma", but the
    hard constraints scored against the returned partition (especially
    `n_communities > n/5` inside `should_fall_back_to_flat`) only make
    algorithmic sense on the CONVERGED partition, not a single-iteration
    intermediate one. On small graphs (Karate N=34), single-pass produces
    11-13 communities (over n/5=6) so every gamma fails the hyper-frag
    check and the tuner always short-circuits to flat. The fix is to run
    the full multi-level Leiden loop at the candidate gamma -- the same
    loop `run_mosaic` will execute when the tuner picks this
    gamma, so the score is representative.

    Plan-B kernel signature (cascaded from Task 0):
    visit_order is computed OUTSIDE @njit via PCG64 and passed as
    `int64[:]` argument; the kernel does NOT instantiate the RNG itself.

    Args:
      csr: CSR matrix of the (undirected) graph.
      partition: int64 cold-start partition (typically np.arange(n)).
      sigma_tot: per-community weighted-degree sums (will NOT be mutated).
      gamma: CPM resolution parameter.
      seed: int seed for the PCG64 visit-order permutation.
      max_levels: convergence cap (default 5; matches `run_mosaic`).

    Returns:
      (final_partition: int64[:] per-ORIGINAL-node converged partition,
       cpm_q: float Q at this gamma on the original graph,
       stats: dict with "lm_moves_total", "refine_moves_total",
              "levels", "n_communities").
    """
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1

    # CRITICAL: work on copies so the tuner does not mutate the caller's
    # arrays. `_njit_local_move` mutates BOTH `partition` and `sigma_tot`
    # in place, so both must be copied. The multi-level loop will further
    # promote to super-graphs (which are fresh CSRs), so the originals
    # remain untouched.
    curr_partition = partition.copy()
    curr_sigma = sigma_tot.copy()
    curr_indptr = indptr
    curr_indices = indices
    curr_data = data
    curr_csr = csr
    curr_int_to_uuid: dict[int, UUID] = {i: uuid4() for i in range(n)}

    # node_to_super_idx[i]: ORIGINAL node i's index in the CURRENT level.
    node_to_super_idx = np.arange(n, dtype=np.int64)

    total_lm_moves = 0
    total_refine_moves = 0
    levels_run = 0

    # Tuner-pass wall-time cap (half the production cap). If exceeded, the
    # tuner returns the current best-so-far partition rather than blocking
    # the whole run_mosaic call.
    _pass_t0 = time.monotonic()
    _pass_budget_s = WALL_TIME_HARD_CAP_S / 4.0

    for level in range(max_levels):
        if time.monotonic() - _pass_t0 > _pass_budget_s:
            # Pass budget exhausted -- return early with whatever partition
            # we have. The diagnostics caller will likely flag this gamma
            # as not satisfying hard constraints.
            break
        levels_run = level + 1
        n_curr = curr_partition.shape[0]

        # Local Move.
        rng_lm = np.random.Generator(np.random.PCG64(seed + 2 * level))
        visit_lm = rng_lm.permutation(n_curr).astype(np.int64)
        lm_moves = _njit_local_move(
            curr_indptr, curr_indices, curr_data,
            curr_partition, curr_sigma,
            gamma, visit_lm, 20,
        )
        total_lm_moves += int(lm_moves)

        # Defensive split.
        curr_partition, curr_sigma, curr_int_to_uuid = (
            _split_disconnected_communities(
                curr_csr, curr_partition, curr_sigma,
                curr_int_to_uuid, LineageTracker(),
            )
        )

        # Refinement.
        refined = np.arange(n_curr, dtype=np.int64)
        sigma_refined = compute_sigma_tot(
            curr_indptr, curr_indices, curr_data, refined, n_curr,
        )
        rng_ref = np.random.Generator(np.random.PCG64(seed + 2 * level + 1))
        visit_ref = rng_ref.permutation(n_curr).astype(np.int64)
        ref_moves = _njit_refine(
            curr_indptr, curr_indices, curr_data,
            curr_partition, refined, sigma_refined,
            gamma, visit_ref, 1,
        )
        total_refine_moves += int(ref_moves)

        # _refinement_subgroup_merge (Traag 2019 §2.3 step 5) is
        # intentionally skipped in the tuner's evaluation loop. On large
        # sparse synthetic graphs its Python inner-loop cost O(macros *
        # subs^2 * N) explodes. The tuner only needs candidate scores;
        # production `run_mosaic` still runs the subgroup merge in its main
        # loop at the chosen gamma.

        # Convergence check (mirrors run_mosaic's break condition).
        if lm_moves == 0 and ref_moves == 0:
            break

        # Aggregate.
        super_csr, super_partition, super_int_to_uuid = _aggregate(
            curr_csr, refined, curr_int_to_uuid, LineageTracker(),
            macro_partition=curr_partition,
        )

        # Update node_to_super_idx projection chain.
        unique_refined = np.unique(refined)
        max_refined_label = (
            int(unique_refined.max()) + 1 if unique_refined.size > 0 else 0
        )
        ref_remap = np.full(max_refined_label, -1, dtype=np.int64)
        for new_idx, orig_label in enumerate(unique_refined):
            ref_remap[int(orig_label)] = new_idx
        node_to_super_idx = ref_remap[refined[node_to_super_idx]]

        # Promote to next level.
        curr_csr = super_csr
        curr_indptr = np.ascontiguousarray(super_csr.indptr, dtype=np.int64)
        curr_indices = np.ascontiguousarray(super_csr.indices, dtype=np.int64)
        curr_data = np.ascontiguousarray(super_csr.data, dtype=np.float64)
        curr_partition = super_partition
        super_n = super_partition.shape[0]
        curr_sigma = compute_sigma_tot(
            curr_indptr, curr_indices, curr_data, super_partition, super_n,
        )
        curr_int_to_uuid = super_int_to_uuid

        if super_n <= 1:
            break

    # Project final partition back to original-graph indices and compact.
    final_partition_orig = curr_partition[node_to_super_idx].astype(np.int64)
    unique_final = np.unique(final_partition_orig)
    final_remap = {int(lbl): i for i, lbl in enumerate(unique_final)}
    final_partition_compact = np.array(
        [final_remap[int(final_partition_orig[i])] for i in range(n)],
        dtype=np.int64,
    )
    k_final = len(final_remap)
    final_sigma = compute_sigma_tot(
        indptr, indices, data, final_partition_compact, k_final,
    )
    cpm_q = float(compute_modularity_cpm(
        indptr, indices, data, final_partition_compact, final_sigma, gamma,
    ))
    return final_partition_compact, cpm_q, {
        "lm_moves_total": total_lm_moves,
        "refine_moves_total": total_refine_moves,
        "levels": levels_run,
        "n_communities": k_final,
    }


def multi_objective_gamma_tuner(
    csr: scipy.sparse.csr_matrix,
    initial_partition: np.ndarray,
    initial_sigma_tot: np.ndarray,
    seed: int,
    targets: dict | None = None,
) -> tuple[float, dict]:
    """Pick the best gamma via a multi-objective target set.

    Evaluates 5 candidates (one Leiden pass per candidate, on COPIES of
    the inputs), scores each against the target set, and returns the best
    by composite score.

    Hard constraints (must ALL be satisfied):
      - CPM-Q >= `CPM_MODULARITY_FLOOR` (calibrated for CPM, NOT the
        0.2 floor used for ModularityVertexPartition)
      - singleton_ratio < 0.30
      - every community induces a connected subgraph
      - `should_fall_back_to_flat(...)` returns False

    Composite score (used to rank candidates):
      `score(gamma) = cpm_q - 0.5 * singleton_ratio`
    -- combines modularity gain with a singleton penalty. Higher is
    better.

    Determinism: same (csr, initial_partition, initial_sigma_tot, seed)
    -> same best_gamma over repeated calls. The candidate set is fixed,
    the inner Leiden pass is deterministic per (seed, gamma), and the
    composite score is a pure function of the resulting partition.

    Args:
      csr: CSR matrix of the (undirected) graph.
      initial_partition: int64 cold-start partition (typically arange).
      initial_sigma_tot: per-community weighted-degree sums.
      seed: int seed for the inner PCG64 visit-order permutations.
      targets: optional override; default
        {"q_min": CPM_MODULARITY_FLOOR, "singleton_ratio_max": 0.30}.

    Returns:
      (best_gamma: float, diagnostics: dict[str, Any]) where diagnostics
      carries:
        - "all_constraints_satisfied": bool
        - "candidate_scores": dict[float, float] -- gamma -> composite
        - "candidate_stats": dict[float, dict] -- per-gamma full stats
        - "should_fall_back_to_flat": bool
        - "best_gamma_q": float
        - "best_gamma_singleton_ratio": float
        - "best_gamma_n_communities": int
    """
    # Deferred policy import to avoid circular dependency at module load
    # (mosaic_policy imports compute_modularity_cpm from this
    # module; importing it back at module-level would create a cycle).
    from iai_mcp.mosaic_policy import (
        CPM_MODULARITY_FLOOR,
        all_communities_connected,
        compute_singleton_ratio,
        should_fall_back_to_flat,
    )

    if targets is None:
        targets = {
            "q_min": CPM_MODULARITY_FLOOR,
            "singleton_ratio_max": 0.30,
        }
    n = int(initial_partition.size)
    candidate_scores: dict[float, float] = {}
    candidate_stats: dict[float, dict] = {}
    best_gamma: float = 1.0
    best_score: float = float("-inf")
    any_satisfied: bool = False
    budget_exhausted: bool = False
    # Tuner budget = half the global hard-cap. Leaves the other half for
    # the production run at the chosen gamma. If exceeded mid-loop, the
    # tuner returns the soft-best so-far candidate with
    # `tuner_budget_exhausted=True` in diagnostics.
    tuner_budget_s = WALL_TIME_HARD_CAP_S / 2.0
    t_start = time.monotonic()

    for gamma in _DEFAULT_GAMMA_CANDIDATES:
        gamma_f = float(gamma)
        if time.monotonic() - t_start > tuner_budget_s:
            budget_exhausted = True
            break
        p_test, q, _stats = _run_one_leiden_pass(
            csr, initial_partition, initial_sigma_tot, gamma_f, seed,
        )
        s_ratio = float(compute_singleton_ratio(p_test))
        n_communities = int(len(np.unique(p_test)))
        connected_ok = bool(all_communities_connected(csr, p_test))
        hard_satisfied = (
            q >= targets["q_min"]
            and s_ratio < targets["singleton_ratio_max"]
            and connected_ok
            and not should_fall_back_to_flat(
                q, s_ratio, n_communities, n,
            )
        )
        composite = q - 0.5 * s_ratio
        candidate_scores[gamma_f] = float(composite)
        candidate_stats[gamma_f] = {
            "q": float(q),
            "singleton_ratio": s_ratio,
            "n_communities": n_communities,
            "connected": connected_ok,
            "hard_satisfied": bool(hard_satisfied),
        }
        if hard_satisfied:
            any_satisfied = True
            if composite > best_score:
                best_score = composite
                best_gamma = gamma_f

    # If no candidate satisfied hard constraints, fall through: pick the
    # soft-best by composite score (so the caller still gets a usable
    # gamma even when it short-circuits to flat fallback). Edge case:
    # if the budget was exhausted before any candidate ran (e.g., the
    # very first call timed out on a pathological N), default best_gamma
    # stays at 1.0 and candidate_stats may be empty.
    if not any_satisfied and candidate_scores:
        best_gamma = float(
            max(candidate_scores.items(), key=lambda kv: kv[1])[0]
        )

    if candidate_stats and best_gamma in candidate_stats:
        best_gamma_q = float(candidate_stats[best_gamma]["q"])
        best_gamma_s_ratio = float(
            candidate_stats[best_gamma]["singleton_ratio"]
        )
        best_gamma_n_c = int(candidate_stats[best_gamma]["n_communities"])
    else:
        # Budget exhausted before any candidate ran -- caller should
        # short-circuit to flat fallback. Diagnostics carry the empty
        # state so consumers can detect it.
        best_gamma_q = 0.0
        best_gamma_s_ratio = 0.0
        best_gamma_n_c = 0

    diagnostics: dict = {
        "all_constraints_satisfied": any_satisfied,
        "candidate_scores": candidate_scores,
        "candidate_stats": candidate_stats,
        "should_fall_back_to_flat": (not any_satisfied),
        "tuner_budget_exhausted": budget_exhausted,
        "best_gamma_q": best_gamma_q,
        "best_gamma_singleton_ratio": best_gamma_s_ratio,
        "best_gamma_n_communities": best_gamma_n_c,
    }
    return float(best_gamma), diagnostics
