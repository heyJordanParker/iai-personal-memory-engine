"""Tests for lilli.ops.{continual, consolidation, decay}.

18 tests total:
- 6 continual tests (add_pair, update_role, empty_hv, determinism, XOR-merge trade-off)
- 6 consolidation tests (bsc empty/identical/D10000, fhrr, sparse_vsa, unknown tier)
- 6 decay tests (grace, 91-day, tem-equivalence, no-decay-in-grace, determinism, more-flips)
"""
from __future__ import annotations

import numpy as np
import pytest

from iai_mcp.lilli.ops.continual import add_pair, empty_hv, update_role
from iai_mcp.lilli.ops.consolidation import consolidate
from iai_mcp.lilli.ops.decay import DECAY_GRACE_DAYS, decay_structure_edge, temporal_decay
from iai_mcp.lilli.tiers import bsc, fhrr, sparse_vsa
from iai_mcp.lilli.core.similarity import hamming


# ============================================================================
# Continual tests (6)
# ============================================================================


def test_continual_empty_hv_default_D():
    """empty_hv() at default D=4096 returns 512 zero bytes."""
    hv = empty_hv()
    assert hv == bytes(512)
    assert len(hv) == 512


def test_continual_add_pair_recovers_filler():
    """Single-pair recovery: unbind(add_pair(empty, role, value), role_hv) == filler_hv."""
    hv = add_pair(empty_hv(), "WHEN", "today")
    recovered = bsc.unbind(hv, bsc.role_hv("WHEN"))
    expected = bsc.filler_hv("today")
    assert recovered == expected


def test_continual_add_pair_at_D_10000():
    """add_pair works at D=10000 with correct output length."""
    hv = add_pair(empty_hv(D=10000), "WHERE", "mars", D=10000)
    assert len(hv) == 10000 // 8  # 1250 bytes
    recovered = bsc.unbind(hv, bsc.role_hv("WHERE", D=10000))
    expected = bsc.filler_hv("mars", D=10000)
    assert recovered == expected


def test_continual_update_role_replaces_filler():
    """update_role swaps an existing role binding; new filler is recoverable via unbind."""
    hv0 = empty_hv()
    hv1 = add_pair(hv0, "ROLE", "user")
    hv2 = update_role(hv1, "user", "ROLE", "admin")
    recovered = bsc.unbind(hv2, bsc.role_hv("ROLE"))
    assert recovered == bsc.filler_hv("admin")


def test_continual_add_two_pairs_xor_merge():
    """XOR-merge of two pairs: documents the XOR-overlay approximation trade-off.

    add_pair is an APPROXIMATE XOR-overlay -- multi-pair vote-margin information is lost.
    After XOR-merging two distinct pairs, unbinding either role yields a noisy vector
    (contaminated by the other bound pair), NOT the clean filler. This is the fundamental
    limitation: single-pair recovery via unbind is only exact when ONE pair is in the hv.

    The XOR-overlay result is also NOT equal to bsc.bundle(pairs) which uses majority vote,
    confirming add_pair is an approximation distinct from the canonical bundle path.
    """
    hv0 = empty_hv()
    hv1 = add_pair(hv0, "WHEN", "today")
    hv2 = add_pair(hv1, "WHERE", "home")

    # After two pairs, unbinding WHEN gives noise, NOT clean filler_hv("today").
    # This is the XOR-overlay trade-off: once >1 pair is overlaid, single-pair
    # recovery via unbind is contaminated by the other bound pair.
    recovered_when = bsc.unbind(hv2, bsc.role_hv("WHEN"))
    assert recovered_when != bsc.filler_hv("today"), (
        "With two XOR-overlaid pairs, unbind(WHEN) must be contaminated by the WHERE pair "
        "-- clean recovery only works when exactly one pair is in the hv"
    )

    # XOR-overlay result differs from bsc.bundle(pairs) majority-vote outcome.
    bundle_result = bsc.bundle([("WHEN", bsc.filler_hv("today")), ("WHERE", bsc.filler_hv("home"))])
    assert hv2 != bundle_result, (
        "XOR-merge of two pairs must differ from majority-vote bundle "
        "-- this confirms add_pair is an approximation, not a drop-in for bundle"
    )


def test_continual_add_pair_deterministic():
    """add_pair is deterministic: same inputs always produce the same output."""
    hv_base = empty_hv()
    out1 = add_pair(hv_base, "TOPIC", "cognition")
    out2 = add_pair(hv_base, "TOPIC", "cognition")
    assert out1 == out2


# ============================================================================
# Consolidation tests (6)
# ============================================================================


def test_consolidate_bsc_empty():
    """consolidate([], 'bsc') returns 512-byte zero bytes (D=4096 default)."""
    result = consolidate([], "bsc")
    assert result == bytes(512)
    assert len(result) == 512


def test_consolidate_bsc_two_identical():
    """consolidate([hv, hv], 'bsc') returns hv (majority vote of two identical hvs)."""
    hv = bsc.bundle([("WHEN", bsc.filler_hv("yesterday"))])
    result = consolidate([hv, hv], "bsc")
    assert result == hv


def test_consolidate_bsc_at_D_10000():
    """consolidate works at D=10000; returns hv of correct length (1250 bytes)."""
    from iai_mcp.lilli.core.seed import hv_from_seed
    hv1 = hv_from_seed(1, 10000)
    hv2 = hv_from_seed(2, 10000)
    result = consolidate([hv1, hv2], "bsc")
    assert len(result) == 1250


def test_consolidate_fhrr_empty_returns_10000_zeros():
    """consolidate([], 'fhrr') returns 10000-byte zero buffer."""
    result = consolidate([], "fhrr")
    assert result == bytes(10000)
    assert len(result) == 10000


def test_consolidate_sparse_vsa_returns_list():
    """consolidate with sparse_vsa returns a list of ints."""
    a = sparse_vsa.role_hv("WHEN")
    b = sparse_vsa.role_hv("WHERE")
    result = consolidate([a, b], "sparse_vsa")
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


def test_consolidate_unknown_tier_raises():
    """consolidate raises ValueError for an unrecognised tier string."""
    with pytest.raises(ValueError, match="unknown tier"):
        consolidate([bytes(512)], "garbage")


# ============================================================================
# Decay tests (6)
# ============================================================================


def test_decay_structure_edge_no_decay_in_grace():
    """decay_structure_edge returns 1.0 for any dt_days within DECAY_GRACE_DAYS."""
    assert decay_structure_edge(0, 0, 0) == 1.0
    assert decay_structure_edge(0, 0, 50) == 1.0
    assert decay_structure_edge(0, 0, 89) == 1.0
    assert decay_structure_edge(0, 0, DECAY_GRACE_DAYS) == 1.0


def test_decay_structure_edge_91_days():
    """decay_structure_edge(0, 0, 91) returns exactly 0.9 (one step past grace)."""
    result = decay_structure_edge(0, 0, 91)
    assert result == pytest.approx(0.9)


def test_decay_structure_edge_bit_equiv_with_tem():
    """decay_structure_edge is behaviourally identical to tem.decay_structure_edge.

    Verified across 7 representative dt values spanning the grace window,
    immediate post-grace, and long-horizon decay.
    """
    from iai_mcp.tem import decay_structure_edge as tem_decay

    dt_values = [0, 50, 90, 91, 180, 365, 730]
    for dt in dt_values:
        lilli_result = decay_structure_edge(0, 0, dt)
        tem_result = tem_decay(0, 0, dt)
        assert lilli_result == pytest.approx(tem_result), (
            f"Mismatch at dt={dt}: lilli={lilli_result}, tem={tem_result}"
        )


def test_temporal_decay_no_decay_in_grace_window():
    """temporal_decay(hv, dt) returns hv unchanged for dt <= DECAY_GRACE_DAYS."""
    hv = bsc.filler_hv("test-vector")
    assert temporal_decay(hv, 0) == hv
    assert temporal_decay(hv, 30) == hv
    assert temporal_decay(hv, DECAY_GRACE_DAYS) == hv


def test_temporal_decay_with_seed_deterministic():
    """temporal_decay with seed=42 is fully deterministic across calls."""
    hv = bytes([0xFF] * 512)
    a = temporal_decay(hv, 365, seed=42)
    b = temporal_decay(hv, 365, seed=42)
    assert a == b
    assert a != hv  # some bits must have flipped


def test_temporal_decay_flips_more_at_higher_age():
    """Higher dt_days produces more bit flips.

    Uses dt=91 (flip_prob ~0.10) vs dt=100 (flip_prob ~0.65) to ensure a
    reliable ordering in the transition zone well away from saturation.
    At dt=200+ both saturate near 100% flips making comparison unreliable.

    Note: the original acceptance spec used dt=1000 vs dt=200 which both saturate --
    amended here (test instability) to use dt=91 vs dt=100 which spans
    a reliable 10% vs 65% flip range with seed=42.
    """
    hv = bytes([0xFF] * 512)
    out_low = temporal_decay(hv, 91, seed=42)   # ~10% flips
    out_high = temporal_decay(hv, 100, seed=42)  # ~65% flips

    # Both must differ from original
    assert out_low != hv
    assert out_high != hv

    # Higher age must produce more hamming distance
    d_low = hamming(hv, out_low)
    d_high = hamming(hv, out_high)
    assert d_high > d_low, (
        f"Expected more flips at dt=100 ({d_high:.4f}) than dt=91 ({d_low:.4f})"
    )
