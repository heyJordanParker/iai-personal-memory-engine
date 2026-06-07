"""Binary Spatter Code tier — episodic memory backend.

Dense binary hypervectors packed 8 bits per byte. bind = XOR (self-inverse).
bundle = per-bit majority vote (capacity ~ D/2 vectors before noise floor;
enforced hard cap at D // 400 to stay well below saturation). Default D=4096
(episodic capacity ~10 pairs by guard). At D=10000, output is byte-identical to
the legacy tem.py module — the back-compat shim depends on this fidelity.

bundle() emits a TELEMETRY_ROLE_SATURATION event when usage crosses 80% of the cap
(only if a store kwarg is supplied — pure-function callers see no I/O), then raises
BundleCapacityError on over-capacity input. EMIT-THEN-RAISE ordering is load-bearing
for downstream telemetry verification.
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import numpy as np

from iai_mcp.lilli.core.seed import hv_from_seed, seed_from_str
from iai_mcp.lilli.core.similarity import hamming

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimensional constant
# ---------------------------------------------------------------------------

LILLI_BSC_DEFAULT_DIM: int = 4096
"""Default dimensionality for the EPISODIC tier. 4096 bits = 512 bytes per HV."""

# ---------------------------------------------------------------------------
# Seed prefixes (LOAD-BEARING: these must match tem.py exactly for bit fidelity)
# ---------------------------------------------------------------------------

BSC_ROLE_SEED_PREFIX: str = "tem-role-v1"
"""Seed prefix for role codebook vectors. Matches tem.py for byte-identical output at D=10000."""

BSC_FILLER_SEED_PREFIX: str = "tem-filler-v1"
"""Seed prefix for filler hypervectors. Matches tem.py for byte-identical output at D=10000."""

# ---------------------------------------------------------------------------
# Role vocabulary (18 fixed symbols — ORDER IS PART OF THE CONTRACT)
# ---------------------------------------------------------------------------

BSC_ROLE_VOCABULARY: tuple[str, ...] = (
    "WHEN",
    "WHERE",
    "ROLE",
    "PROJECT",
    "COMMUNITY_ID",
    "TEMPORAL_POSITION",
    "ACTOR",
    "OBJECT",
    "INTENT",
    "MODALITY",
    "LANG",
    "SESSION_ID",
    "TIER",
    "VALENCE",
    "CERTAINTY",
    "SOURCE",
    "TOPIC",
    "PARENT_ID",
)

# ---------------------------------------------------------------------------
# Saturation guard constants
# ---------------------------------------------------------------------------

BSC_CAPACITY_DIVISOR: int = 400
"""Divisor for computing the max bundle pair count per dimension D."""

BSC_SATURATION_WARN_RATIO: float = 0.8
"""Fraction of max bundle pairs at which to emit a saturation warning event."""

# This string literal MUST match events.TELEMETRY_ROLE_SATURATION once that
# constant lands. Same value by spec contract (verified at test time by
# test_telemetry_kind_string_matches_events_module when both modules are loaded).
_TELEMETRY_ROLE_SATURATION_KIND: str = "role_saturation_warning"


def _max_bundle_pairs(D: int) -> int:
    """Return the maximum number of pairs allowed for a bundle of dimension D.

    Hard cap at D // 400 (at least 1). Keeps the bundle well below the BSC
    noise floor of ~D/2 vectors.
    """
    return max(1, D // BSC_CAPACITY_DIVISOR)


BSC_MAX_BUNDLE_PAIRS: int = _max_bundle_pairs(LILLI_BSC_DEFAULT_DIM)
"""Default-D hard cap: _max_bundle_pairs(4096) == 10."""

# ---------------------------------------------------------------------------
# Tier metadata dict (consumed by lilli.tier_info() in a future plan)
# ---------------------------------------------------------------------------

TIER_INFO: dict = {
    "backend": "bsc",
    "D": LILLI_BSC_DEFAULT_DIM,
    "bytes_per_hv": LILLI_BSC_DEFAULT_DIM // 8,
    "use_case": "episodic",
    "max_bundle_pairs": BSC_MAX_BUNDLE_PAIRS,
}

# ---------------------------------------------------------------------------
# Codebook functions
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def role_hv(role: str, *, D: int = LILLI_BSC_DEFAULT_DIM) -> bytes:
    """Return the deterministic binary codebook vector for a role symbol.

    At D=10000, output is byte-identical to tem.role_hv(role). Parameterised
    over D so the same function serves all dimensionalities without a fixed
    global constant.

    Args:
        role: Role symbol (e.g. "WHEN"). Any string is accepted; callers
              should use BSC_ROLE_VOCABULARY for the canonical 18 roles.
        D: Hypervector dimensionality. Default is LILLI_BSC_DEFAULT_DIM.

    Returns:
        Packed bytes of length D // 8.
    """
    seed = seed_from_str(BSC_ROLE_SEED_PREFIX, role)
    return hv_from_seed(seed, D)


@lru_cache(maxsize=256)
def filler_hv(value: str, *, D: int = LILLI_BSC_DEFAULT_DIM) -> bytes:
    """Return the deterministic binary hypervector for a filler string value.

    At D=10000, output is byte-identical to tem.filler_hv(value).

    Args:
        value: Filler value (e.g. "today", "iai-mcp").
        D: Hypervector dimensionality. Default is LILLI_BSC_DEFAULT_DIM.

    Returns:
        Packed bytes of length D // 8.
    """
    seed = seed_from_str(BSC_FILLER_SEED_PREFIX, value)
    return hv_from_seed(seed, D)


# ---------------------------------------------------------------------------
# Core BSC operations
# ---------------------------------------------------------------------------


def bind(a: bytes, b: bytes) -> bytes:
    """BSC tensor-product binding: bytewise XOR. Self-inverse semantics.

    Args:
        a: First hypervector (packed bytes).
        b: Second hypervector (packed bytes, must be same length as a).

    Returns:
        XOR of a and b as packed bytes.

    Raises:
        ValueError: If a and b have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(
            f"bind requires equal-length hypervectors, got {len(a)} and {len(b)}"
        )
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    return np.bitwise_xor(aa, bb).tobytes()


def unbind(bound: bytes, key: bytes) -> bytes:
    """XOR inverse of bind. Identical to bind() because XOR is self-inverse.

    Args:
        bound: Bound hypervector (packed bytes).
        key: Key hypervector to unbind with.

    Returns:
        Result of XOR, recovering the original vector bound under key.
    """
    return bind(bound, key)


def bundle(
    pairs: list[tuple[str, bytes]],
    *,
    D: int = LILLI_BSC_DEFAULT_DIM,
    store: "Optional[MemoryStore]" = None,
) -> bytes:
    """Bundle role-filler pairs via per-bit majority vote.

    Empty pair list returns bytes(D // 8) — matching tem.pack_pairs behaviour.
    Deterministic tiebreak: bit=1 on even ties (sums * 2 >= n).

    SATURATION GUARD — EMIT-THEN-RAISE ordering (load-bearing contract):

    1. Compute n = len(pairs) and max_pairs = _max_bundle_pairs(D).
    2. If n >= 80% of max_pairs AND store is provided: emit TELEMETRY_ROLE_SATURATION.
       Telemetry NEVER crashes the bundle path (wrapped in try/except).
    3. If n > max_pairs: raise BundleCapacityError.

    The emit-before-raise ordering means callers that catch BundleCapacityError
    can observe the telemetry event in the store after the exception is handled.

    Args:
        pairs: List of (role, filler_bytes) pairs. Roles in BSC_ROLE_VOCABULARY
               are canonical; arbitrary role strings are also supported.
        D: Hypervector dimensionality. Default is LILLI_BSC_DEFAULT_DIM.
        store: Optional MemoryStore for telemetry emission. When None (default),
               saturation warning telemetry is silently skipped — the function
               stays callable without I/O binding.

    Returns:
        Packed bytes of length D // 8.

    Raises:
        BundleCapacityError: If len(pairs) > _max_bundle_pairs(D).
    """
    if not pairs:
        return bytes(D // 8)

    max_pairs = _max_bundle_pairs(D)
    n = len(pairs)
    warn_threshold = math.ceil(BSC_SATURATION_WARN_RATIO * max_pairs)

    # STEP 1: EMIT TELEMETRY FIRST (before any raise).
    # Emit at warn_threshold so the saturation test observes the event
    # even when the call is immediately followed by a BundleCapacityError raise.
    if n >= warn_threshold and store is not None:
        try:
            from iai_mcp import events  # deferred import — avoid hard dependency loop

            events.write_event(
                store,
                _TELEMETRY_ROLE_SATURATION_KIND,
                {"D": D, "n_pairs": n, "max_pairs": max_pairs, "ratio": n / max_pairs},
                severity="warning",
                domain="lilli.tiers.bsc",
            )
        except Exception:  # noqa: BLE001 — telemetry must never crash bundle
            log.warning("role_saturation telemetry emit failed (non-fatal)", exc_info=True)

    # STEP 2: RAISE on over-capacity (AFTER emit). Order is load-bearing.
    if n > max_pairs:
        from iai_mcp.lilli.errors import BundleCapacityError

        raise BundleCapacityError(
            f"BSC bundle at D={D} accepts at most {max_pairs} pairs "
            f"({BSC_CAPACITY_DIVISOR}:1 capacity ratio); got {n}. "
            f"Reduce role count or migrate this pair set to the FHRR tier "
            f"(semantic, higher capacity)."
        )

    # STEP 3: Per-bit majority vote.
    # For each (role, filler_bytes), bind role_hv with filler_bytes.
    bound: list[np.ndarray] = []
    for role, filler in pairs:
        bound.append(np.frombuffer(bind(role_hv(role, D=D), filler), dtype=np.uint8))

    stacked_bytes = np.stack(bound)  # shape (N, D//8)
    bits = np.unpackbits(stacked_bytes, axis=1).astype(np.int32)  # (N, D)
    sums = bits.sum(axis=0)
    # majority: bit=1 when more than half of inputs are 1; ties -> 1 (>=).
    voted = (sums * 2 >= n).astype(np.uint8)
    return np.packbits(voted).tobytes()


def permute(hv: bytes, shift: int) -> bytes:
    """Cyclic bit-permutation by shift bits.

    Positive shift = right shift; negative = left shift. Applying permute(hv, k)
    followed by permute(result, -k) recovers the original hv exactly.

    Args:
        hv: Packed binary hypervector.
        shift: Number of bit positions to rotate.

    Returns:
        Permuted packed bytes of the same length.
    """
    bits = np.unpackbits(np.frombuffer(hv, dtype=np.uint8))
    shifted = np.roll(bits, shift)
    return np.packbits(shifted).tobytes()


def similarity(a: bytes, b: bytes) -> float:
    """Return BSC similarity as 1.0 - normalized Hamming distance.

    Identical vectors return 1.0; length-mismatch returns 0.0 (maximally
    dissimilar) to degrade gracefully on cross-tier comparisons.

    Args:
        a: First packed binary hypervector.
        b: Second packed binary hypervector.

    Returns:
        Float in [0.0, 1.0].
    """
    if len(a) != len(b):
        return 0.0
    return 1.0 - hamming(a, b)


def unpack_role(hv: bytes, role: str, *, D: int = LILLI_BSC_DEFAULT_DIM) -> bytes:
    """Unbind hv by role's codebook vector.

    Returns a noisy filler hypervector; caller nearest-neighbour decodes
    against a known filler codebook.

    Args:
        hv: Bundle hypervector to unpack.
        role: Role symbol to unbind (uses role_hv internally).
        D: Hypervector dimensionality.

    Returns:
        Noisy filler bytes of length D // 8.
    """
    return unbind(hv, role_hv(role, D=D))
