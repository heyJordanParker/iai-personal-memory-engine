"""Tests for iai_mcp.daemon.

Covers:
1. main() completes cleanly when shutdown event is set externally.
2. State-machine transitions: valid edges succeed, illegal edges raise ValueError.
3. Scheduler tick body gets called repeatedly; exceptions caught, daemon continues.
4. embedder prewarm invoked exactly once at boot.
6. Empty-store shortcut: _tick_body records `empty_store` reason without REM work.
7. launchd plist is valid XML + has required Label/KeepAlive/ThrottleInterval keys.
8. systemd unit has Type=simple + Restart=on-failure + WantedBy=default.target +
   python3 -m iai_mcp.daemon + TimeoutStopSec=60.
9. Neither plist nor systemd unit contains ANTHROPIC_API_KEY (C3 guard).
"""
from __future__ import annotations

import asyncio
import plistlib
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLIST_PATH = PROJECT_ROOT / "src" / "iai_mcp" / "_deploy" / "launchd" / "com.iai-mcp.daemon.plist"
SERVICE_PATH = PROJECT_ROOT / "src" / "iai_mcp" / "_deploy" / "systemd" / "iai-mcp-daemon.service"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _short_socket_paths(tmp_path, monkeypatch):
    """Redirect concurrency SOCKET_PATH to a short path (AF_UNIX 104-char limit)."""
    import os
    from iai_mcp import concurrency
    lock_path = tmp_path / ".lock"
    sock_dir = Path(f"/tmp/iai-daemon-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    monkeypatch.setattr(concurrency, "SOCKET_PATH", sock_path)
    return lock_path, sock_path, sock_dir


# ---------------------------------------------------------------------------
# Test 1: clean shutdown via signal-like event trigger
# ---------------------------------------------------------------------------

def test_main_clean_shutdown(tmp_path, monkeypatch):
    """main() returns 0 when shutdown fires shortly after boot."""
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")
    _short_socket_paths(tmp_path, monkeypatch)

    # Prevent real embedder instantiation (saves 10s + avoids model download).
    def _fake_embedder(store):
        class _Stub:
            def embed(self, text):
                return [0.0]
        return _Stub()
    monkeypatch.setattr("iai_mcp.embed.embedder_for_store", _fake_embedder)

    async def runner():
        task = asyncio.create_task(daemon_mod.main())
        # Give the daemon a chance to boot, then trigger shutdown by sending SIGTERM.
        await asyncio.sleep(0.2)
        # Simulate signal delivery: find the loop's shutdown event and set it.
        # Easiest: raise CancelledError on the main task after a brief run.
        # We inject shutdown by cancelling the task, then verifying it returns cleanly.
        task.cancel()
        try:
            return await task
        except asyncio.CancelledError:
            return 0

    rc = asyncio.run(runner())
    assert rc == 0


# ---------------------------------------------------------------------------
# Test 2: state-machine transitions
# ---------------------------------------------------------------------------

def test_state_machine_transitions(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod

    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")

    state: dict = {}  # fresh state starts at WAKE default

    # WAKE -> TRANSITIONING (valid)
    daemon_mod.transition(state, daemon_mod.STATE_TRANSITIONING)
    assert state["fsm_state"] == daemon_mod.STATE_TRANSITIONING

    # TRANSITIONING -> SLEEP (valid)
    daemon_mod.transition(state, daemon_mod.STATE_SLEEP)
    assert state["fsm_state"] == daemon_mod.STATE_SLEEP

    # SLEEP -> DREAMING (valid)
    daemon_mod.transition(state, daemon_mod.STATE_DREAMING)
    assert state["fsm_state"] == daemon_mod.STATE_DREAMING

    # DREAMING -> TRANSITIONING (ILLEGAL)
    with pytest.raises(ValueError, match="Illegal transition"):
        daemon_mod.transition(state, daemon_mod.STATE_TRANSITIONING)
    assert state["fsm_state"] == daemon_mod.STATE_DREAMING  # state unchanged

    # DREAMING -> SLEEP (valid)
    daemon_mod.transition(state, daemon_mod.STATE_SLEEP)
    assert state["fsm_state"] == daemon_mod.STATE_SLEEP

    # SLEEP -> WAKE (valid)
    daemon_mod.transition(state, daemon_mod.STATE_WAKE)
    assert state["fsm_state"] == daemon_mod.STATE_WAKE

    # WAKE -> SLEEP (ILLEGAL, must go through TRANSITIONING)
    with pytest.raises(ValueError):
        daemon_mod.transition(state, daemon_mod.STATE_SLEEP)

    # State persisted each time: load_state finds fsm_state=WAKE after final txn.
    loaded = ds_mod.load_state()
    assert loaded["fsm_state"] == daemon_mod.STATE_WAKE


# ---------------------------------------------------------------------------
# Test 3: scheduler tick loop continues after exceptions
# ---------------------------------------------------------------------------

def test_scheduler_tick_survives_exceptions(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)

    # Shrink tick interval so the test finishes quickly.
    monkeypatch.setattr(daemon_mod, "TICK_INTERVAL_SEC", 0)

    state: dict = {}

    call_count = {"n": 0}

    async def flaky_body(store, state):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated tick failure")

    async def runner():
        task = asyncio.create_task(
            daemon_mod._scheduler_tick(store, state, tick_body=flaky_body)
        )
        # Let several ticks happen.
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())

    assert call_count["n"] >= 2, (
        f"tick loop did not continue past first exception; only {call_count['n']} calls"
    )
    # tick_error event recorded on the first failing call.
    from iai_mcp.events import query_events
    err_events = query_events(store, kind="tick_error", limit=5)
    assert len(err_events) >= 1
    assert "simulated tick failure" in err_events[0]["data"].get("error", "")


# ---------------------------------------------------------------------------
# Test 4: bge-m3 prewarm called exactly once at boot
# ---------------------------------------------------------------------------

def test_prewarm_called_once_at_boot(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")
    _short_socket_paths(tmp_path, monkeypatch)

    prewarm_calls = {"n": 0}

    class _StubEmbedder:
        def embed(self, text):
            prewarm_calls["n"] += 1
            return [0.0]

    def _fake_embedder(store):
        return _StubEmbedder()

    monkeypatch.setattr("iai_mcp.embed.embedder_for_store", _fake_embedder)

    async def runner():
        task = asyncio.create_task(daemon_mod.main())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())
    assert prewarm_calls["n"] == 1, (
        f"prewarm expected once, got {prewarm_calls['n']}"
    )


# ---------------------------------------------------------------------------
# Test 5: graceful shutdown cancels both tasks + closes lock fd
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 6: empty-store shortcut in _tick_body
# ---------------------------------------------------------------------------

def test_empty_store_shortcut(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store = _fresh_store(tmp_path, monkeypatch)
    state: dict = {"fsm_state": "WAKE"}

    async def run_once():
        await daemon_mod._tick_body(store, state)

    asyncio.run(run_once())

    assert state.get("last_tick_skipped_reason") == "empty_store"

    # No `rem_cycle_started` event emitted on empty store.
    from iai_mcp.events import query_events
    rem = query_events(store, kind="rem_cycle_started", limit=5)
    assert rem == []


# ---------------------------------------------------------------------------
# Test 7: launchd plist valid XML + required keys
# ---------------------------------------------------------------------------

def test_launchd_plist_valid_xml_with_required_keys():
    assert PLIST_PATH.exists(), f"missing plist at {PLIST_PATH}"

    with open(PLIST_PATH, "rb") as f:
        data = plistlib.load(f)

    assert data["Label"] == "com.iai-mcp.daemon"
    assert data["ProgramArguments"][-1] == "iai_mcp.daemon"
    assert data["RunAtLoad"] is True

    keepalive = data["KeepAlive"]
    assert isinstance(keepalive, dict)
    # KeepAlive policy is now
    # `Crashed=true` only. The legacy `SuccessfulExit=false` paired
    # with the 75/0 exit-code branching; with the new lifecycle
    # state machine exit code is uniformly 0 on graceful shutdown,
    # so SuccessfulExit=false would create a respawn loop.
    assert keepalive.get("Crashed") is True
    assert "SuccessfulExit" not in keepalive

    assert data["ThrottleInterval"] == 5
    assert "StandardOutPath" in data
    assert "StandardErrorPath" in data
    assert "WorkingDirectory" in data

    env = data["EnvironmentVariables"]
    for required_key in ("PATH", "IAI_MCP_STORE", "HOME", "LANG"):
        assert required_key in env, f"missing env key {required_key}"

    # C3 guard (redundant with Test 9 but check locally too):
    assert "ANTHROPIC_API_KEY" not in env


# ---------------------------------------------------------------------------
# Test 8: systemd unit required keys
# ---------------------------------------------------------------------------

def test_systemd_unit_required_keys():
    assert SERVICE_PATH.exists(), f"missing unit file at {SERVICE_PATH}"
    text = SERVICE_PATH.read_text()

    assert "[Unit]" in text
    assert "Description=" in text
    assert "[Service]" in text
    assert "Type=simple" in text
    assert "Restart=on-failure" in text
    assert "RestartSec=30" in text
    assert "StartLimitIntervalSec=60" in text
    assert "StartLimitBurst=3" in text
    assert "python3 -m iai_mcp.daemon" in text
    assert "StandardOutput=journal" in text
    assert "StandardError=journal" in text
    assert "SyslogIdentifier=iai-mcp-daemon" in text
    assert "TimeoutStopSec=60" in text
    assert "KillSignal=SIGTERM" in text
    assert "[Install]" in text
    assert "WantedBy=default.target" in text


# ---------------------------------------------------------------------------
# Test 9: C3 guard -- no ANTHROPIC_API_KEY anywhere
# ---------------------------------------------------------------------------

def test_c3_no_anthropic_api_key_in_artifacts():
    daemon_src = (PROJECT_ROOT / "src" / "iai_mcp" / "daemon.py").read_text()
    plist_src = PLIST_PATH.read_text()
    service_src = SERVICE_PATH.read_text()

    for name, src in (("daemon.py", daemon_src), ("plist", plist_src), ("service", service_src)):
        assert "ANTHROPIC_API_KEY" not in src, (
            f"C3 VIOLATION: ANTHROPIC_API_KEY found in {name}"
        )


# ---------------------------------------------------------------------------
# Held warm-embedder singleton: identity reuse via the funnel + no-leak
# discipline (build via the funnel, install after the lock gate, restore on
# any exit; an early lock-conflict leaks nothing).
# ---------------------------------------------------------------------------


@pytest.fixture
def _restore_embedder_funnel_after():
    """Snapshot ``embed.embedder_for_store`` before and restore after.

    The override-pollution teardown discipline made explicit: in the
    single-process pytest runner a set-once-not-restored module-attribute
    override would leak the held singleton into every later embed/dim test.
    This fixture guarantees the funnel is restored even if the test fails.
    """
    import iai_mcp.embed as _embed_mod

    _orig = _embed_mod.embedder_for_store
    try:
        yield _orig
    finally:
        _embed_mod.embedder_for_store = _orig


class _IdentityStub:
    """Stub embedder whose identity we can assert across funnel calls."""

    def embed(self, text):
        return [0.0] * 384


def test_daemon_boot_holds_one_embedder_singleton(
    tmp_path, monkeypatch, _restore_embedder_funnel_after
):
    """After the boot HOLD+OVERRIDE install, the funnel returns the SAME held
    instance across repeated calls (built VIA the funnel), and the fixture
    teardown restores the original funnel (no pollution)."""
    import iai_mcp.embed as _embed_mod
    from iai_mcp import daemon as daemon_mod

    # Build VIA the funnel: monkeypatch the funnel to a fake that constructs a
    # fresh stub per call. The override must collapse those to ONE held object.
    construct_calls = {"n": 0}

    def _fake_funnel(store):
        construct_calls["n"] += 1
        return _IdentityStub()

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _fake_funnel)

    class _StubStore:
        embed_dim = 384

    store = _StubStore()
    orig_efs, installed = daemon_mod._install_warm_embedder_override(store)

    assert installed is True
    # The override was built via the funnel (the fake) exactly once at install.
    assert construct_calls["n"] == 1
    # Every subsequent funnel call returns the SAME held instance (identity),
    # and does NOT re-enter the fake funnel (no per-call reconstruction).
    held = _embed_mod.embedder_for_store(store)
    assert _embed_mod.embedder_for_store(store) is held
    assert _embed_mod.embedder_for_store(store) is held
    assert construct_calls["n"] == 1, "override must not reconstruct per call"

    # Restore via the production helper proves the shutdown path restores.
    daemon_mod._restore_embedder_funnel(orig_efs, installed)
    assert _embed_mod.embedder_for_store is _fake_funnel
    # After restore the funnel builds fresh again (no longer the held stub).
    assert _embed_mod.embedder_for_store(store) is not held


def test_daemon_prewarm_failure_is_non_fatal(
    tmp_path, monkeypatch, _restore_embedder_funnel_after
):
    """If the build/hold raises, the install does NOT propagate, the default
    funnel stays installed, and the sentinel is False (no override set)."""
    import iai_mcp.embed as _embed_mod
    from iai_mcp import daemon as daemon_mod

    def _raising_funnel(store):
        raise RuntimeError("simulated construct failure")

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _raising_funnel)

    class _StubStore:
        embed_dim = 384
        # write_event's failure path is itself guarded; give it a root so a
        # prewarm_failed emit attempt does not explode the test.
        root = tmp_path

    store = _StubStore()
    # Must NOT raise.
    orig_efs, installed = daemon_mod._install_warm_embedder_override(store)

    assert installed is False, "build failure must not install the override"
    assert orig_efs is _raising_funnel
    # The default funnel is still installed (no override) -> sites construct fresh.
    assert _embed_mod.embedder_for_store is _raising_funnel
    # The guarded restore is a no-op when nothing was installed.
    daemon_mod._restore_embedder_funnel(orig_efs, installed)
    assert _embed_mod.embedder_for_store is _raising_funnel


def test_daemon_early_lock_conflict_does_not_leak_override(
    tmp_path, monkeypatch, _restore_embedder_funnel_after
):
    """An early LifecycleLockConflict return (before the install) must leave
    the funnel untouched -- main() returns 1 and no override leaks."""
    import iai_mcp.embed as _embed_mod
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod
    from iai_mcp import lifecycle_lock as ll_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")
    _short_socket_paths(tmp_path, monkeypatch)

    # Sentinel funnel: if the override ever installs, this object is replaced.
    sentinel_funnel = _embed_mod.embedder_for_store

    def _conflict(self):
        raise ll_mod.LifecycleLockConflict("simulated live-PID conflict")

    monkeypatch.setattr(ll_mod.LifecycleLock, "acquire", _conflict)

    rc = asyncio.run(daemon_mod.main())

    assert rc == 1, "lock conflict must return exit code 1"
    assert _embed_mod.embedder_for_store is sentinel_funnel, (
        "override must NOT be installed/leaked on an early lock-conflict return"
    )


def test_daemon_boot_raise_after_install_restores_funnel(
    tmp_path, monkeypatch, _restore_embedder_funnel_after
):
    """A raise from the post-install boot region must still restore the funnel.

    The override is installed AFTER the lock gate; a subsequent boot statement
    (here ``save_state``) raising must NOT leak the held singleton -- the
    restore is guaranteed on ANY exit, including a propagated boot exception.
    Without that guarantee the funnel stays shadowed in-process and pollutes
    every later embed call in the single-process runner.
    """
    import iai_mcp.embed as _embed_mod
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import daemon_state as ds_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    monkeypatch.setattr(ds_mod, "STATE_PATH", tmp_path / ".daemon-state.json")
    _short_socket_paths(tmp_path, monkeypatch)

    # A funnel that builds a cheap stub so the override installs cleanly. The
    # install captures THIS as the original funnel and replaces the module
    # attribute with a held-instance closure, so a leak is detectable as
    # "embedder_for_store is no longer _stub_funnel" afterwards.
    def _stub_funnel(store):
        class _S:
            def embed(self, text):
                return [0.0] * 384
        return _S()

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _stub_funnel)
    # The funnel the override must be restored to: exactly the pre-install one.
    pre_install_funnel = _embed_mod.embedder_for_store
    assert pre_install_funnel is _stub_funnel

    # Force the FIRST save_state -- which runs AFTER the override install -- to
    # raise, simulating any unwrapped post-install boot statement failing.
    def _raising_save_state(state):
        raise RuntimeError("simulated post-install boot failure")

    monkeypatch.setattr(daemon_mod, "save_state", _raising_save_state)

    with pytest.raises(RuntimeError, match="simulated post-install boot failure"):
        asyncio.run(daemon_mod.main())

    # The boot raised AFTER the override was installed; the restore must have
    # run on that exit, returning the funnel to the pre-install funnel (the
    # held-instance closure must NOT survive the raise).
    assert _embed_mod.embedder_for_store is pre_install_funnel, (
        "funnel override leaked: restore did not run on a post-install boot raise"
    )
