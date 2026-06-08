"""Clean-environment wheel-install gate.

Builds a wheel from the repo root, installs it into a fresh throwaway
venv (no .venv, scrubbed PYTHONPATH/VIRTUAL_ENV), and asserts that all
runtime-located files (deploy hooks, daemon plist, MCP wrapper) resolve
under the installed package rather than the source tree.

ALL tests are @pytest.mark.slow — gated by --runslow; skipped by default
so the standard suite stays green.  Run with --runslow to exercise the gate.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Env vars that re-mask packaging bugs when inherited from a dev shell.
# A subprocess that inherits these will load the editable install from the
# dev source tree instead of the clean venv's site-packages.
_MASKING_VARS = {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME", "PYTHONSTARTUP"}

# Repo root: tests/ -> repo_root
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def clean_install_whl(tmp_path_factory):
    """Build a wheel and return its path.  Session-scoped: one build per run.

    Uses pip wheel . --no-deps so setuptools-rust compiles the Rust extension
    and packages it alongside the Python source.  Cached objects in
    rust/target/ are reused on incremental builds (much faster than cold).
    """
    whl_dir = tmp_path_factory.mktemp("whl", numbered=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "-w",
            str(whl_dir),
        ],
        cwd=str(_REPO_ROOT),
        check=True,
    )
    wheels = list(whl_dir.glob("iai_mcp-*.whl"))
    assert len(wheels) == 1, f"Expected exactly 1 wheel, got: {wheels}"
    return wheels[0]


@pytest.fixture
def clean_install_env(tmp_path, clean_install_whl):
    """Install the wheel into a fresh venv in tmp_path.

    Returns (venv_bin: Path, clean_env: dict) — both are needed by every
    clean-env test.

    CRITICAL: the subprocess environment is built by FILTERING OUT _MASKING_VARS
    rather than by inheriting **os.environ.  Inheriting PYTHONPATH/VIRTUAL_ENV
    from the dev shell re-masks the packaging bug by loading the editable install
    over the clean venv's site-packages.
    """
    venv_dir = tmp_path / "clean-env"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    venv_bin = venv_dir / "bin"

    # Build a clean subprocess env: start from the current env, drop masking vars.
    clean_env = {k: v for k, v in os.environ.items() if k not in _MASKING_VARS}
    clean_env["HOME"] = str(tmp_path)
    clean_env["IAI_MCP_STORE"] = str(tmp_path / ".iai-mcp")

    subprocess.run(
        [str(venv_bin / "pip"), "install", str(clean_install_whl), "--quiet"],
        env=clean_env,
        check=True,
    )
    return venv_bin, clean_env


# ---------------------------------------------------------------------------
# Masking-guard helper
# ---------------------------------------------------------------------------


def _assert_not_masked(venv_bin: Path, clean_env: dict) -> None:
    """Assert iai_mcp.__file__ resolves under the clean venv, not the source tree.

    Call at the top of every clean-env test to catch env-masking regressions
    before reaching the substantive assertion.
    """
    result = subprocess.run(
        [
            str(venv_bin / "python"),
            "-c",
            "import iai_mcp; print(iai_mcp.__file__)",
        ],
        capture_output=True,
        text=True,
        env=clean_env,
        check=True,
    )
    pkg_file = result.stdout.strip()
    assert "site-packages" in pkg_file, (
        f"iai_mcp loaded from wrong location — masking still active: {pkg_file}"
    )
    assert str(venv_bin.parent) in pkg_file, (
        f"iai_mcp not loaded from the clean venv: {pkg_file}"
    )


# ---------------------------------------------------------------------------
# Smoke test — the wheel always builds and installs cleanly
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_wheel_builds_and_installs_clean(clean_install_env):
    """Smoke-test: the wheel installs into a fresh venv and iai_mcp resolves
    under site-packages rather than the source tree.
    """
    venv_bin, clean_env = clean_install_env
    _assert_not_masked(venv_bin, clean_env)


# ---------------------------------------------------------------------------
# Packaging gate assertions
#
# These assertions verify the wheel packages _deploy/ and _wrapper/ contents
# and that the daemon/hook installers resolve those files from the installed
# package rather than the source tree.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_daemon_install_finds_plist(clean_install_env, tmp_path):
    """After wheel install, daemon install --dry-run renders the plist using the
    venv's own python interpreter, not a hard-coded system path.
    """
    venv_bin, clean_env = clean_install_env
    _assert_not_masked(venv_bin, clean_env)
    env = {
        **clean_env,
        "IAI_DAEMON_SOCKET_PATH": str(tmp_path / "no.sock"),
        "IAI_MCP_CRYPTO_PASSPHRASE": "test-passphrase",
    }
    result = subprocess.run(
        [str(venv_bin / "iai-mcp"), "daemon", "install", "--dry-run", "--yes"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"daemon install --dry-run --yes failed:\n{result.stderr}"
    )
    assert "com.iai-mcp.daemon" in result.stdout, (
        "Plist content not printed by --dry-run"
    )
    # The installed plist must embed the venv python, never a system-wide path.
    assert str(venv_bin / "python") in result.stdout, (
        "Venv python not found in rendered plist"
    )
    assert "/usr/local/bin/python3" not in result.stdout, (
        "Hard-coded system python3 still present in rendered plist"
    )


@pytest.mark.slow
def test_capture_hooks_install_finds_hooks(clean_install_env, tmp_path):
    """After wheel install, capture-hooks install copies hook sources from
    within the installed package, not from the source repo deploy/ directory.
    """
    venv_bin, clean_env = clean_install_env
    _assert_not_masked(venv_bin, clean_env)
    result = subprocess.run(
        [str(venv_bin / "iai-mcp"), "capture-hooks", "install"],
        capture_output=True,
        text=True,
        env=clean_env,
    )
    assert result.returncode == 0, (
        f"capture-hooks install failed:\n{result.stderr}"
    )
    hooks_dir = tmp_path / ".claude" / "hooks"
    assert (hooks_dir / "iai-mcp-session-capture.sh").exists(), (
        "session-capture hook not installed"
    )
    assert (hooks_dir / "iai-mcp-turn-capture.sh").exists(), (
        "turn-capture hook not installed"
    )
    assert (hooks_dir / "iai-mcp-session-recall.sh").exists(), (
        "session-recall hook not installed"
    )

    # The MCP server entry written to .claude.json must record the clean venv's
    # python interpreter path, not a source-tree .venv or repo path.
    # NOTE: sys.executable inside a console-script subprocess is the versioned
    # binary (e.g. python3.12), so assert on the venv ROOT prefix rather than
    # the exact basename to avoid a false failure on the versioned name.
    import json

    claude_json = tmp_path / ".claude.json"
    assert claude_json.exists(), ".claude.json not written by capture-hooks install"
    data = json.loads(claude_json.read_text())
    iai_mcp_python = (
        data.get("mcpServers", {}).get("iai-mcp", {}).get("env", {}).get("IAI_MCP_PYTHON", "")
    )
    venv_root = str(venv_bin.parent)
    assert iai_mcp_python.startswith(venv_root), (
        f"IAI_MCP_PYTHON={iai_mcp_python!r} does not start with venv root {venv_root!r}"
    )
    # Must not point into the source tree's .venv or the repo root.
    assert str(_REPO_ROOT) not in iai_mcp_python, (
        f"IAI_MCP_PYTHON points into the source repo: {iai_mcp_python!r}"
    )


@pytest.mark.slow
def test_session_capture_hook_candidates(clean_install_env, tmp_path):
    """After capture-hooks install, the installed session-capture hook shell
    script contains the expected CLI resolution candidates.
    """
    venv_bin, clean_env = clean_install_env
    _assert_not_masked(venv_bin, clean_env)
    # Install the hooks first (same as the previous test, but function-scoped
    # so we get a fresh venv).
    result = subprocess.run(
        [str(venv_bin / "iai-mcp"), "capture-hooks", "install"],
        capture_output=True,
        text=True,
        env=clean_env,
    )
    assert result.returncode == 0, (
        f"capture-hooks install failed:\n{result.stderr}"
    )
    hook_path = tmp_path / ".claude" / "hooks" / "iai-mcp-session-capture.sh"
    assert hook_path.exists(), "session-capture hook file not created"
    hook_text = hook_path.read_text()
    # The hook must attempt to locate the CLI via PATH and via the well-known
    # pyenv shim path so it works outside an active venv.
    assert "command -v iai-mcp" in hook_text, (
        "Hook does not probe PATH for iai-mcp CLI"
    )
    assert ".pyenv/shims/iai-mcp" in hook_text, (
        "Hook does not fall back to pyenv shim path"
    )


@pytest.mark.slow
def test_wheel_contains_wrapper_and_rust_ext(clean_install_env):
    """The installed wheel contains all compiled wrapper JS files and the Rust
    extension.
    """
    venv_bin, clean_env = clean_install_env
    _assert_not_masked(venv_bin, clean_env)
    probe = (
        "import importlib.resources as r, pathlib\n"
        "wrapper = pathlib.Path(str(r.files('iai_mcp') / '_wrapper' / 'index.js'))\n"
        "print('wrapper_exists:', wrapper.exists())\n"
        "print('bridge_exists:', (wrapper.parent / 'bridge.js').exists())\n"
        "siblings = list(wrapper.parent.glob('*.js'))\n"
        "print('js_count:', len(siblings))\n"
        "import iai_mcp_native\n"
        "print('rust_ext_ok:', True)\n"
    )
    result = subprocess.run(
        [str(venv_bin / "python"), "-c", probe],
        capture_output=True,
        text=True,
        env=clean_env,
    )
    # Allow the probe to fail with an import error so we get the real assertion
    # messages rather than a check=True exception.
    out = result.stdout
    assert "wrapper_exists: True" in out, (
        f"_wrapper/index.js absent from wheel:\n{result.stderr}"
    )
    assert "bridge_exists: True" in out, (
        "_wrapper/bridge.js absent from wheel"
    )
    assert "js_count: 7" in out, (
        f"Expected 7 JS files in _wrapper/, got:\n{out}"
    )
    assert "rust_ext_ok: True" in out, (
        "Rust extension import failed inside clean venv"
    )


@pytest.mark.slow
def test_fresh_editable_install_resolver(tmp_path_factory, clean_install_whl):
    """A fresh pip install -e . (editable) resolves the wrapper via the
    editable fallback path (mcp-wrapper/dist/index.js), not via package-data.

    This guards the requirement that the editable install + scripts/install.sh
    path continues to work after the packaging changes land.
    """
    # Create a second independent tmp dir so the venv does not share HOME
    # with clean_install_env.
    editable_tmp = tmp_path_factory.mktemp("editable", numbered=True)

    venv_dir = editable_tmp / "editable-env"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    venv_bin = venv_dir / "bin"

    clean_env = {k: v for k, v in os.environ.items() if k not in _MASKING_VARS}
    clean_env["HOME"] = str(editable_tmp)
    clean_env["IAI_MCP_STORE"] = str(editable_tmp / ".iai-mcp")
    # Rustup resolves toolchains relative to RUSTUP_HOME (defaults to $HOME/.rustup).
    # The conftest autouse fixture redirects $HOME in os.environ to a tmp dir, so
    # os.path.expanduser("~") would resolve to that tmp dir rather than the real
    # user home.  Use the passwd-database lookup to get the invariant real home
    # and pin RUSTUP_HOME / CARGO_HOME to the actual toolchain directories so the
    # Rust extension build can find the stable toolchain.
    import pwd as _pwd
    _real_home = _pwd.getpwuid(os.getuid()).pw_dir
    clean_env.setdefault("RUSTUP_HOME", os.path.join(_real_home, ".rustup"))
    clean_env.setdefault("CARGO_HOME", os.path.join(_real_home, ".cargo"))

    # Install in editable mode from the repo root.  The mcp-wrapper/dist/ tree
    # is pre-built in the source tree (scripts/install.sh has been run).
    result = subprocess.run(
        [str(venv_bin / "pip"), "install", "-e", str(_REPO_ROOT), "--quiet"],
        env=clean_env,
        check=False,
    )
    assert result.returncode == 0, (
        f"pip install -e failed:\n{getattr(result, 'stderr', '')}"
    )

    # _resolve_wrapper_path() must return a path ending mcp-wrapper/dist/index.js
    # (editable fallback / path-3 in the resolver).
    result2 = subprocess.run(
        [
            str(venv_bin / "python"),
            "-c",
            "from iai_mcp.cli import _resolve_wrapper_path; print(_resolve_wrapper_path())",
        ],
        capture_output=True,
        text=True,
        env=clean_env,
    )
    assert result2.returncode == 0, (
        f"_resolve_wrapper_path() failed:\n{result2.stderr}"
    )
    resolved = result2.stdout.strip()
    assert resolved.endswith("mcp-wrapper/dist/index.js"), (
        f"Editable resolver returned unexpected path: {resolved!r}"
    )
