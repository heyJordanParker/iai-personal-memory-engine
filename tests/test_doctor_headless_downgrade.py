from __future__ import annotations

import pytest


def test_is_headless_linux_no_display_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iai_mcp.doctor as doc_mod

    monkeypatch.setattr(doc_mod.platform, "system", lambda: "Linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    from iai_mcp.doctor import is_headless

    assert is_headless(force=False) is True


def test_is_headless_linux_with_display_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iai_mcp.doctor as doc_mod

    monkeypatch.setattr(doc_mod.platform, "system", lambda: "Linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    from iai_mcp.doctor import is_headless

    assert is_headless(force=False) is False


def test_is_headless_macos_no_display_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iai_mcp.doctor as doc_mod

    monkeypatch.setattr(doc_mod.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    from iai_mcp.doctor import is_headless

    assert is_headless(force=False) is False


def test_is_headless_macos_with_force_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iai_mcp.doctor as doc_mod

    monkeypatch.setattr(doc_mod.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    from iai_mcp.doctor import is_headless

    assert is_headless(force=True) is True


def test_apply_headless_downgrade_mutates_b_and_n() -> None:
    from iai_mcp.doctor import CheckResult, _apply_headless_downgrade

    results = [
        CheckResult(
            name="(a) daemon process alive",
            passed=True,
            detail="PID 1 (iai_mcp.daemon)",
            status="PASS",
        ),
        CheckResult(
            name="(b) socket file fresh",
            passed=False,
            detail="present but unreachable (timeout/refused)",
            status="FAIL",
        ),
        CheckResult(
            name="(n) HID idle source",
            passed=False,
            detail="HIDIdleTime: unavailable; no idle source",
            status="FAIL",
        ),
        CheckResult(
            name="(z) AVX2 CPU support",
            passed=False,
            detail="this host lacks AVX2 -- the native memory store cannot load",
            status="FAIL",
        ),
    ]

    out = _apply_headless_downgrade(results, headless=True)

    assert out is results

    by_name = {r.name: r for r in out}

    b = by_name["(b) socket file fresh"]
    assert b.passed is True, f"(b) should now pass (WARN); got {b.passed}"
    assert b.status == "WARN", f"(b) status should be WARN; got {b.status!r}"
    assert "unreachable" in b.detail, (
        f"(b) detail must survive the downgrade; got {b.detail!r}"
    )

    n = by_name["(n) HID idle source"]
    assert n.passed is True, f"(n) should now pass (WARN); got {n.passed}"
    assert n.status == "WARN", f"(n) status should be WARN; got {n.status!r}"

    a = by_name["(a) daemon process alive"]
    assert a.passed is True and a.status == "PASS"

    z = by_name["(z) AVX2 CPU support"]
    assert z.passed is False, f"(z) must stay FAIL; got passed={z.passed}"
    assert z.status == "FAIL", f"(z) must stay FAIL; got status={z.status!r}"


def test_apply_headless_downgrade_noop_when_not_headless() -> None:
    from iai_mcp.doctor import CheckResult, _apply_headless_downgrade

    results = [
        CheckResult(
            name="(b) socket file fresh",
            passed=False,
            detail="present but unreachable",
            status="FAIL",
        ),
    ]

    out = _apply_headless_downgrade(results, headless=False)

    assert out is results
    b = out[0]
    assert b.passed is False, f"(b) must stay FAIL; got passed={b.passed}"
    assert b.status == "FAIL", f"(b) must stay FAIL; got status={b.status!r}"


def test_cli_doctor_accepts_headless_flag() -> None:
    from iai_mcp.cli import _build_parser

    parser = _build_parser()

    ns_with = parser.parse_args(["doctor", "--headless"])
    assert ns_with.headless is True, (
        f"expected headless=True with --headless; got {ns_with.headless!r}"
    )

    ns_default = parser.parse_args(["doctor"])
    assert ns_default.headless is False, (
        f"expected headless=False by default; got {ns_default.headless!r}"
    )
