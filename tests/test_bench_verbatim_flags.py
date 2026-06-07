"""Tests for diagnostic flags on bench/verbatim.py.

Covers the 5 behaviors from the plan:
 1. `python -m bench.verbatim --help` lists --skip-l0-seed, --storage-direct,
    --n, --gap, --noise-per-session, --k.
 2. `run_verbatim_bench(skip_l0_seed=True,...)` does NOT seed L0 identity.
 3. `run_verbatim_bench(storage_direct=True,...)` writes zero provenance
    entries on pinned records across the query loop.
 4. Default invocation (no new flags set) is byte-identical to pre-plan
    behavior on the public dict keys.
 5. `--k` override propagates to `recall(k_hits=K)` (or `query_similar(k=K)`
    in storage-direct mode).

All tests use tmp_path for hermeticity; N kept tiny for CI speed.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cli_help_lists_all_new_flags():
    """Behavior 1: --help must list all 6 diagnostic/config flags."""
    out = subprocess.run(
        [sys.executable, "-m", "bench.verbatim", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert out.returncode == 0, f"--help exited {out.returncode}: {out.stderr}"
    text = out.stdout
    for flag in (
        "--skip-l0-seed",
        "--storage-direct",
        "--n",
        "--gap",
        "--noise-per-session",
        "--k",
    ):
        assert flag in text, f"--help missing flag {flag}\n\n{text}"


def test_skip_l0_seed_does_not_seed_l0(tmp_path):
    """Behavior 2: with skip_l0_seed=True no L0 record exists in the store."""
    from bench.verbatim import run_verbatim_bench
    from iai_mcp.core import L0_ID
    from iai_mcp.store import MemoryStore

    s = MemoryStore(path=tmp_path)
    result = run_verbatim_bench(
        store=s,
        n_records=5,
        session_gap=2,
        noise_per_session=3,
        skip_l0_seed=True,
    )
    assert "accuracy" in result
    assert result["skip_l0_seed"] is True
    assert s.get(L0_ID) is None, (
        "skip_l0_seed=True must not seed L0 identity record"
    )


def test_storage_direct_writes_zero_provenance_to_pinned(tmp_path):
    """Behavior 3: storage_direct bypasses recall() so no provenance writes."""
    from bench.verbatim import run_verbatim_bench
    from iai_mcp.store import MemoryStore

    s = MemoryStore(path=tmp_path)
    result = run_verbatim_bench(
        store=s,
        n_records=5,
        session_gap=2,
        noise_per_session=3,
        storage_direct=True,
    )
    assert "accuracy" in result
    assert result["storage_direct"] is True

    # Every pinned record must have an empty provenance list after the run
    # (storage_direct bypass -> no append_provenance calls).
    pinned_offenders: list[tuple[str, int]] = []
    for rec in s.all_records():
        if rec.pinned and "benchmark" in (rec.tags or []):
            if len(rec.provenance or []) != 0:
                pinned_offenders.append(
                    (rec.literal_surface[:40], len(rec.provenance or []))
                )
    assert not pinned_offenders, (
        f"storage_direct must leave pinned provenance empty, got: {pinned_offenders}"
    )


def test_default_invocation_keys_preserved(tmp_path):
    """Behavior 4: default invocation returns legacy keys unchanged."""
    from bench.verbatim import run_verbatim_bench
    from iai_mcp.store import MemoryStore

    s = MemoryStore(path=tmp_path)
    result = run_verbatim_bench(
        store=s,
        n_records=5,
        session_gap=2,
        noise_per_session=3,
    )
    # Legacy keys still present.
    for key in (
        "accuracy",
        "n_records",
        "session_gap",
        "noise_per_session",
        "hits_exact",
        "passed",
        "floor",
        "noise_mode",
    ):
        assert key in result, f"legacy key {key} missing"
    # New diagnostic traceability keys added.
    for key in ("skip_l0_seed", "storage_direct", "k"):
        assert key in result, f"diagnostic key {key} missing"
    assert result["skip_l0_seed"] is False
    assert result["storage_direct"] is False


def test_k_override_propagates_in_storage_direct(tmp_path):
    """Behavior 5: --k override in storage_direct mode propagates to query_similar.

    With n_records=5 and k=3, storage-direct can only return 3 rows per query;
    the pinned-text hit count is therefore capped at a function of k rather
    than the default max(n_records+10, 20). We assert that a deliberately
    tiny k drives accuracy strictly below 1.0 on a harness where the default
    k would return all pinned records.
    """
    from bench.verbatim import run_verbatim_bench
    from iai_mcp.store import MemoryStore

    s = MemoryStore(path=tmp_path)
    result = run_verbatim_bench(
        store=s,
        n_records=5,
        session_gap=2,
        noise_per_session=3,
        storage_direct=True,
        k=3,
    )
    assert result["k"] == 3, f"k should be echoed back, got {result.get('k')!r}"
    # With k < n_records, at least some pinned cues will not find their exact
    # literal in the top-k -> accuracy strictly below 1.0. This would not
    # happen with the default k (max(n+10, 20) = 20 for n=5).
    assert result["accuracy"] < 1.0, (
        f"k=3 with n=5 must cap accuracy below 1.0, got {result['accuracy']}"
    )
