"""Fail-loud unit tests for the MCP wrapper resolver.

Fast, in-process tests (no subprocess, no wheel build) that verify:
- A missing wrapper (no env override, no package-data, no editable dist)
  raises FileNotFoundError with an actionable message naming the fix commands.
- An IAI_MCP_WRAPPER_PATH env override wins and returns the overridden path.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_missing_wrapper_raises_actionable(tmp_path, monkeypatch):
    """When no wrapper is locatable, FileNotFoundError is raised with an
    actionable message that tells the user how to fix the problem.

    Monkeypatching strategy:
    1. Unset IAI_MCP_WRAPPER_PATH so the env-override path is not taken.
    2. Redirect importlib.resources lookup to a non-existent _wrapper/index.js.
    3. Point iai_mcp.__file__ to tmp_path so the editable fallback also misses.
    """
    import iai_mcp.cli as _cli_module

    # 1. No env override.
    monkeypatch.delenv("IAI_MCP_WRAPPER_PATH", raising=False)

    # 2. Patch importlib.resources.files to return a traversal rooted at a
    #    directory that does NOT contain _wrapper/index.js.
    import importlib.resources as _res

    original_files = _res.files

    def _fake_files(package):
        if package == "iai_mcp":
            # Return a traversal from tmp_path — _wrapper/ does not exist there.
            return original_files.__class__  # fallback: just use tmp_path anchor
        return original_files(package)

    # Simpler: patch iai_mcp.__file__ to a place without _wrapper/ AND patch
    # the importlib.resources path to the same non-existent location.
    fake_pkg_init = tmp_path / "iai_mcp" / "__init__.py"
    fake_pkg_init.parent.mkdir(parents=True, exist_ok=True)
    fake_pkg_init.write_text("")

    import iai_mcp as _pkg
    monkeypatch.setattr(_pkg, "__file__", str(fake_pkg_init))

    # Reload the attribute reference inside cli so `_pkg.__file__` resolves
    # to the patched value (some resolvers cache the module reference).
    # We patch via the module attribute directly.
    monkeypatch.setattr(_cli_module, "iai_mcp", _pkg, raising=False)  # type: ignore[attr-defined]

    # 3. Patch importlib.resources.files for iai_mcp to land in tmp_path.
    from importlib.resources.abc import Traversable  # type: ignore[import]

    class _FakeTraversable:
        """Minimal Traversable that returns non-existent paths for __truediv__."""

        def __init__(self, base: Path):
            self._base = base

        def __truediv__(self, child: str) -> "_FakeTraversable":
            return _FakeTraversable(self._base / child)

        def __str__(self) -> str:
            return str(self._base)

        def exists(self) -> bool:
            return self._base.exists()

    import iai_mcp.cli as _cli

    original_res_files = None
    try:
        import importlib.resources as _ir

        original_res_files = _ir.files

        def _fake_ir_files(pkg):
            if pkg == "iai_mcp":
                return _FakeTraversable(tmp_path)
            return original_res_files(pkg)  # type: ignore[misc]

        monkeypatch.setattr(_ir, "files", _fake_ir_files)
    except Exception:
        pass  # If patching fails, the test may still reach FileNotFoundError

    # The resolver must raise FileNotFoundError with actionable instructions.
    from iai_mcp.cli import _resolve_wrapper_path  # type: ignore[attr-defined]

    with pytest.raises(FileNotFoundError) as exc_info:
        _resolve_wrapper_path()

    msg = str(exc_info.value)
    assert "npm run build" in msg, (
        f"Error message lacks 'npm run build' instruction:\n{msg}"
    )
    assert "scripts/install.sh" in msg, (
        f"Error message lacks 'scripts/install.sh' instruction:\n{msg}"
    )


def test_env_override_wins(tmp_path, monkeypatch):
    """When IAI_MCP_WRAPPER_PATH is set to a real file, the resolver returns
    that path without checking package-data or the editable fallback.
    """
    fake_wrapper = tmp_path / "index.js"
    fake_wrapper.write_text("// stub\n")

    monkeypatch.setenv("IAI_MCP_WRAPPER_PATH", str(fake_wrapper))

    from iai_mcp.cli import _resolve_wrapper_path  # type: ignore[attr-defined]

    result = _resolve_wrapper_path()
    assert result == fake_wrapper, (
        f"Env override not respected: got {result!r}, expected {fake_wrapper!r}"
    )
