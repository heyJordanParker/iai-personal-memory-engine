from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path

import pytest


def _rust_available() -> bool:
    try:
        from iai_mcp_native import embed  # noqa: F401
        return True
    except ImportError:
        return False


def _rust_lib_path() -> Path:
    import iai_mcp_native
    candidate = Path(iai_mcp_native.__file__)
    if candidate.suffix in (".so", ".dylib"):
        return candidate
    for sibling in candidate.parent.glob("iai_mcp_native*"):
        if sibling.suffix in (".so", ".dylib"):
            return sibling
    pytest.skip(f"could not locate native lib next to {candidate}")
    raise RuntimeError("unreachable")


@pytest.mark.skipif(not _rust_available(), reason="iai_mcp_native wheel not installed")
@pytest.mark.skipif(platform.system() != "Darwin", reason="otool is macOS-only")
def test_accelerate_src_not_in_default_features():
    """Verify accelerate-src crate is NOT compiled into the default build.

    Note: Accelerate.framework may still appear via gemm's macOS BLAS
    auto-detection — that's expected and harmless (it's a system framework,
    not the opt-in accelerate-src crate). The important invariant is that
    the accelerate-src build.rs (which hard-fails on Linux) is not in the
    default dependency tree.
    """
    lib = _rust_lib_path()
    result = subprocess.run(["nm", str(lib)], capture_output=True, text=True)
    assert result.stdout, f"nm produced no output for {lib}"
    accelerate_src_markers = [
        line for line in result.stdout.splitlines()
        if "accelerate_src" in line.lower()
    ]
    assert not accelerate_src_markers, (
        f"accelerate-src symbols found in {lib} (should be opt-in only):\n"
        + "\n".join(accelerate_src_markers[:10])
    )


@pytest.mark.skipif(not _rust_available(), reason="iai_mcp_native wheel not installed")
def test_no_metal_symbols():
    lib = _rust_lib_path()
    result = subprocess.run(["nm", str(lib)], capture_output=True, text=True)
    assert result.stdout, f"nm produced no output for {lib}"
    offenders = [
        line for line in result.stdout.splitlines()
        if re.search(r"metal|MTL", line, flags=re.IGNORECASE)
        and "dummy_metal_backend" not in line
        and "CustomOp" not in line
        # candle Device::is_metal is an API method (returns whether device is Metal),
        # not a backend symbol — present in any candle build regardless of the
        # (disabled) metal feature; the real Metal-linkage guard is
        # test_no_metal_framework_linked (otool Metal.framework check)
        and "is_metal" not in line
    ]
    assert not offenders, (
        "Metal symbols leaked into Rust embedder binary:\n"
        + "\n".join(offenders[:10])
    )


@pytest.mark.skipif(not _rust_available(), reason="iai_mcp_native wheel not installed")
@pytest.mark.skipif(platform.system() != "Darwin", reason="otool is macOS-only")
def test_no_metal_framework_linked():
    lib = _rust_lib_path()
    result = subprocess.run(["otool", "-L", str(lib)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    bad = [
        line for line in result.stdout.splitlines()
        if "Metal.framework" in line or "libmetal" in line.lower()
    ]
    assert not bad, (
        "Metal framework linked into Rust embedder binary:\n"
        + "\n".join(bad[:10])
    )


@pytest.mark.skipif(not _rust_available(), reason="iai_mcp_native wheel not installed")
def test_offline_mode_works_with_warm_cache(monkeypatch):
    monkeypatch.setenv("IAI_MCP_EMBED_OFFLINE", "1")
    from iai_mcp.embed import Embedder
    e = Embedder()
    v = e.embed("hello")
    assert len(v) == 384
