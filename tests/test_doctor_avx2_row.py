"""/ D- tests for the new (z) AVX2 CPU support doctor row.

The row consults `iai_mcp.cpu_features.has_avx2()` directly (not via lancedb
import) so the diagnostic is correct even on a host where `import lancedb`
would SIGILL. Three behavioral cases:

  1. PASS when has_avx2() returns True.
  2. FAIL with the actionable message when has_avx2() returns False.
  3. run_diagnosis() returns 15 rows (the 14 existing + new (z) AVX2 row),
     and the last row's name starts with "(z)".
"""
from __future__ import annotations

import pytest


def test_check_z_pass_when_avx2_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """has_avx2()=True -> row PASS, status='PASS', name '(z) AVX2 CPU support'."""
    import iai_mcp.cpu_features as cf

    monkeypatch.setattr(cf, "has_avx2", lambda: True)

    from iai_mcp.doctor import check_z_avx2_support

    result = check_z_avx2_support()

    assert result.passed is True, f"expected passed=True; got {result.passed}"
    assert result.status == "PASS", f"expected status='PASS'; got {result.status!r}"
    assert result.name == "(z) AVX2 CPU support", (
        f"expected name='(z) AVX2 CPU support'; got {result.name!r}"
    )


def test_check_z_fail_when_avx2_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """has_avx2()=False -> row FAIL with actionable message naming AVX2."""
    import iai_mcp.cpu_features as cf

    monkeypatch.setattr(cf, "has_avx2", lambda: False)

    from iai_mcp.doctor import check_z_avx2_support

    result = check_z_avx2_support()

    assert result.passed is False, f"expected passed=False; got {result.passed}"
    assert result.status == "FAIL", f"expected status='FAIL'; got {result.status!r}"
    assert "AVX2" in result.detail, (
        f"detail must name AVX2; got {result.detail!r}"
    )
    assert "LanceDB cannot load" in result.detail, (
        f"detail must explain LanceDB cannot load; got {result.detail!r}"
    )
    assert "iai-mcp memory store is unavailable" in result.detail, (
        f"detail must say store is unavailable; got {result.detail!r}"
    )


def test_run_diagnosis_includes_z_row() -> None:
    """run_diagnosis() returns 15 rows; the last is the new (z) AVX2 row.

    The row order MUST place (z) last, after the existing a..n block.
    """
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    names = [r.name for r in results]

    assert len(results) == 24, (
        f"expected 24 rows after hippo rows (r/s/t) + (u) centrality + (v) native embedder + (w) permanent-failed added; got {len(results)}: {names}"
    )
    assert results[-1].name.startswith("(z)"), (
        f"the new (z) row must be last; got last name {results[-1].name!r}"
    )
