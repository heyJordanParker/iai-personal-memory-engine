"""Embedding ↔ hypervector bridge.

Public API
----------
from_embedding(emb) -> bytes
    Project a 384-dim float embedding to a 10000-bit hypervector via
    sign(emb @ P). Returns 1250 packed bytes.

from_embedding_batch(embs, *, store=None, deviation_threshold=0.2) -> list[bytes]
    Batch wrapper. When ``store`` is provided and ``len(embs) >=
    RANK_DEFICIENCY_MIN_BATCH_SIZE``, computes per-bit frequency deviation
    from 0.5 across the batch. If ``deviation > deviation_threshold``,
    emits a "rank_deficiency_warning" telemetry event.

to_embedding_neighbors(hv, store, k=5) -> list
    Approximate embedding reconstruction via P^T @ hv_signed, then
    store.query_similar nearest-neighbor lookup. Returns the same shape as
    MemoryStore.query_similar: list of (MemoryRecord, float) pairs.

Constants
---------
RANK_DEFICIENCY_MIN_BATCH_SIZE = 8
    Minimum batch size before rank-deficiency monitoring activates.

Notes
-----
The projection matrix P (shape 384 × 10000, float32) lives in
``iai_mcp.lilli.core.projection``. It is loaded once at module import via
the seed-locked generation procedure and verified against a SHA256 hash.
Never regenerate P — stored hypervectors would become incompatible.

The telemetry kind string "_TELEMETRY_RANK_DEFICIENCY_KIND" is a local
constant that MUST match ``events.TELEMETRY_RANK_DEFICIENCY`` once that
constant lands in the events module. Same pattern as bsc.py lines 83-86.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from iai_mcp.lilli.core.projection import EMBED_DIM, HV_DIM, P

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum batch size before rank-deficiency monitoring activates.
RANK_DEFICIENCY_MIN_BATCH_SIZE: int = 8

#: Default deviation threshold for rank-deficiency detection.
#: Chosen to sit between healthy-batch mean (~0.13) and clustered-batch
#: deviation (~0.5). A test uses 0.01 to guarantee
#: a trip on nearly-identical vectors.
RANK_DEFICIENCY_DEFAULT_THRESHOLD: float = 0.2

# Telemetry kind string. MUST match events.TELEMETRY_RANK_DEFICIENCY once
# that constant lands. Same "forward-declared local" pattern as bsc.py.
_TELEMETRY_RANK_DEFICIENCY_KIND: str = "rank_deficiency_warning"

# Packed bytes per hypervector: HV_DIM=10000 bits → 1250 bytes.
_HV_BYTES: int = HV_DIM // 8  # == 1250


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------


def from_embedding(emb: list[float]) -> bytes:
    """Project a 384-dim embedding to a 10000-bit binary hypervector.

    Computes ``sign(emb @ P)``, maps {negative → 0, non-negative → 1}, and
    packs the result into 1250 bytes with ``numpy.packbits`` (big-endian bit
    order, matching the rest of the lilli tier suite).

    Args:
        emb: Length-384 list of floats (normalised embedding from Embedder).

    Returns:
        1250 bytes representing the binary hypervector.

    Raises:
        ValueError: If ``len(emb) != 384`` or if any element is non-finite.
    """
    arr = np.asarray(emb, dtype=np.float32)
    if arr.shape != (EMBED_DIM,):
        raise ValueError(
            f"from_embedding expects a length-{EMBED_DIM} embedding, "
            f"got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("from_embedding: embedding contains non-finite values")
    projected = arr @ P  # (10000) float32
    # sign: non-negative → 1, negative → 0.
    # np.float32(0.0) → 1 (ties go to 1, matching BSC bundle tiebreak convention).
    bits = (projected >= 0).astype(np.uint8)  # (10000) uint8 {0,1}
    return np.packbits(bits).tobytes()  # 1250 bytes


# ---------------------------------------------------------------------------
# Batch wrapper
# ---------------------------------------------------------------------------


def from_embedding_batch(
    embs: list[list[float]],
    *,
    store: "MemoryStore | None" = None,
    deviation_threshold: float = RANK_DEFICIENCY_DEFAULT_THRESHOLD,
) -> list[bytes]:
    """Project a batch of embeddings to hypervectors.

    For each embedding, calls ``from_embedding`` individually. When
    ``store`` is provided AND ``len(embs) >= RANK_DEFICIENCY_MIN_BATCH_SIZE``,
    computes the per-bit frequency across the batch and measures the maximum
    absolute deviation from 0.5. If that deviation exceeds
    ``deviation_threshold``, a "rank_deficiency_warning" event is emitted via
    ``events.write_event``.

    Telemetry emission is wrapped in try/except — it NEVER raises or disrupts
    the return path (same contract as bsc.bundle saturation guard).

    Args:
        embs: List of length-384 embedding vectors.
        store: Optional MemoryStore for telemetry. If None, monitoring is
               silently skipped regardless of batch size.
        deviation_threshold: Maximum allowed per-bit frequency deviation from
               0.5 before emitting a rank-deficiency warning. Default 0.2.

    Returns:
        List of 1250-byte hypervectors, one per input embedding.
    """
    hvs = [from_embedding(e) for e in embs]

    # Rank-deficiency monitoring: requires store + minimum batch size.
    if store is None or len(embs) < RANK_DEFICIENCY_MIN_BATCH_SIZE:
        return hvs

    try:
        # Stack packed bytes and unpack to (N, HV_DIM) uint8 {0,1}.
        packed = np.frombuffer(b"".join(hvs), dtype=np.uint8).reshape(
            len(hvs), _HV_BYTES
        )
        bit_matrix = np.unpackbits(packed, axis=1)  # (N, HV_DIM)
        # Per-bit frequency (fraction of 1s across the batch).
        freq = bit_matrix.mean(axis=0)  # (HV_DIM) float64
        # Mean absolute deviation from 0.5 (per-bit average, not max).
        # A healthy diverse batch sits around 0.13; clustered/identical
        # batches approach 0.5. The REVIEWS.md spec cites these landmarks.
        deviation = float(np.abs(freq - 0.5).mean())

        if deviation > deviation_threshold:
            from iai_mcp import events  # deferred — avoid hard import loop

            events.write_event(
                store,
                _TELEMETRY_RANK_DEFICIENCY_KIND,
                {
                    "batch_size": len(embs),
                    "deviation": deviation,
                    "threshold": deviation_threshold,
                    "hv_dim": HV_DIM,
                },
                severity="warning",
                domain="lilli.crossmodal.embed_to_hv",
            )
    except Exception:  # noqa: BLE001 — telemetry must never crash batch path
        log.warning(
            "rank_deficiency telemetry emit failed (non-fatal)", exc_info=True
        )

    return hvs


# ---------------------------------------------------------------------------
# Reverse lookup
# ---------------------------------------------------------------------------


def to_embedding_neighbors(
    hv: bytes,
    store: "MemoryStore",
    k: int = 5,
) -> list:
    """Approximate nearest-neighbor lookup grounded in the Hippo store.

    Reconstructs an approximate 384-dim embedding from a packed binary HV via
    the pseudo-inverse relationship ``P^T @ hv_signed``, then delegates to
    ``store.query_similar(emb, k=k)`` for ANN search.

    The reconstruction is approximate: the sign projection is non-invertible,
    so the reconstructed embedding lives in the same direction as the original
    but is not identical. For retrieval purposes this is sufficient.

    Args:
        hv: 1250-byte packed binary hypervector (HV_DIM=10000 bits).
        store: Open MemoryStore instance.
        k: Number of neighbors to return.

    Returns:
        List of (MemoryRecord, float) pairs from ``store.query_similar``.
        Empty list when the store is empty or ``hv`` has wrong length.
    """
    if len(hv) != _HV_BYTES:
        log.warning(
            "to_embedding_neighbors: expected %d bytes, got %d — returning empty",
            _HV_BYTES,
            len(hv),
        )
        return []

    packed = np.frombuffer(hv, dtype=np.uint8)
    bits = np.unpackbits(packed)  # (10000) uint8 {0,1}
    # Map {0→-1, 1→+1} for a signed representation compatible with cosine search.
    hv_signed = bits.astype(np.float32) * 2.0 - 1.0  # (10000) float32
    # Approximate inverse projection: hv_signed @ P.T → (384) float32.
    # P is (384, 10000); hv_signed is (10000); hv_signed @ P.T gives (384).
    approx_emb = hv_signed @ P.T  # (384) float32
    # L2-normalise so cosine similarity in hnswlib space is meaningful.
    norm = float(np.linalg.norm(approx_emb))
    if norm == 0.0:
        log.warning("to_embedding_neighbors: zero-norm reconstructed embedding")
        return []
    approx_emb = approx_emb / norm  # (384) float32
    return store.query_similar(approx_emb.tolist(), k=k)
