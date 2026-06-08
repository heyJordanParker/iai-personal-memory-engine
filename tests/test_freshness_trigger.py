"""Tests for freshness-on-return trigger: watermark helpers, gate logic,
and SC3 end-to-end verification.

Tests
-----
test_watermark_round_trip -- write then read returns the same ISO string
test_baseline_on_first -- no watermark -> sets baseline, no additionalContext
test_no_trigger_when_not_newer -- watermark == MAX -> no RPC, no additionalContext
test_trigger_when_newer -- watermark < MAX -> RPC called, additionalContext emitted,
                                      watermark advanced to new_max_ts
test_daemon_down -- RPC returns None -> no additionalContext, wm unchanged
test_utc_normalization -- mixed-offset timestamps compare correctly (no spurious trigger)
test_sc3_end_to_end -- gate -> core.dispatch in-process -> drain -> compose ->
                                      inject: emitted additionalContext contains OOB record text
test_hook_shape_regression -- src/iai_mcp/_deploy/hooks/iai-mcp-turn-capture.sh still contains
                                      per-turn capture, invokes session-refresh-if-stale,
                                      does NOT reference MemoryStore(or drain_deferred_captures
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixture: isolated HOME
# ---------------------------------------------------------------------------


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    """Redirect HOME to tmp_path so tests never touch ~/.iai-mcp."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-freshness-trigger-passphrase")
    # Force MemoryStore to land in tmp_path so it shares the same root that
    # get_max_created_at() will read (both resolve via Path.home()).
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))

    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_store(home: Path):
    """Open a MemoryStore under the isolated HOME."""
    from iai_mcp.store import MemoryStore

    return MemoryStore(path=home / ".iai-mcp")


def _insert_record(store, text: str):
    """Insert a record directly and flush the write buffer to SQLite.

    The MemoryStore uses a module-level write buffer (flushed at 500 rows or
    on daemon lifecycle hooks). For tests that call MAX(created_at) or
    count_rows() we must flush after each insert so SQLite reflects the write
    immediately — otherwise the module-level buffer's id(store) key can be
    reused by a subsequent test's store object, causing spurious dedup hits.
    """
    from iai_mcp.capture import capture_turn
    from iai_mcp.store import flush_record_buffer

    result = capture_turn(store, text=text, cue="", tier="episodic", role="user")
    # Flush so SQLite (and therefore MAX(created_at) / count_rows) is up-to-date.
    flush_record_buffer(store)
    return result


def _write_drainable_deferred(home: Path, session_id: str, text: str) -> Path:
    """Write a non-active (drainable) deferred JSONL file."""
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    suffix = int(time.time())
    out = deferred_dir / f"{session_id}-{suffix}.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "cwd": "/tmp",
    }
    event = {
        "text": text,
        "cue": f"session {session_id} deferred cue",
        "tier": "episodic",
        "role": "user",
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(
        json.dumps(header, ensure_ascii=False) + "\n"
        + json.dumps(event, ensure_ascii=False) + "\n"
    )
    return out


# ---------------------------------------------------------------------------
# Watermark round-trip
# ---------------------------------------------------------------------------


def test_watermark_round_trip(iai_home):
    """write_watermark then read_watermark returns the same UTC-normalized ISO string."""
    from iai_mcp.cli import read_watermark, write_watermark

    ts = "2026-05-29T10:00:00+00:00"
    write_watermark("test-session", ts)
    result = read_watermark("test-session")
    assert result is not None
    # Stored value should be UTC-normalized (Z suffix is fine either way; just
    # must be non-empty and represent the same instant).
    dt_stored = datetime.fromisoformat(result.replace("Z", "+00:00"))
    dt_orig = datetime.fromisoformat(ts)
    assert abs((dt_stored - dt_orig).total_seconds()) < 1


def test_read_watermark_absent(iai_home):
    """read_watermark returns None when no sidecar file exists."""
    from iai_mcp.cli import read_watermark

    assert read_watermark("nonexistent-session-xyz") is None


# ---------------------------------------------------------------------------
# Baseline-on-first: no watermark -> set baseline, no additionalContext
# ---------------------------------------------------------------------------


def test_baseline_on_first(iai_home, monkeypatch):
    """First prompt (no watermark): sets baseline watermark, emits no additionalContext."""
    from iai_mcp import cli

    store = _open_store(iai_home)
    _insert_record(store, "alice wrote the tokenizer module")

    rpc_calls: list = []

    def fake_rpc(method, params, **_kw):
        rpc_calls.append((method, params))
        return None

    monkeypatch.setattr(cli, "_send_jsonrpc_request", fake_rpc)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    import argparse

    args = argparse.Namespace(session_id="baseline-session")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert rpc_calls == [], "RPC must not be called on the first (baseline) prompt"
    assert captured.getvalue() == "", "No additionalContext on first prompt"

    # Watermark should now be set.
    from iai_mcp.cli import read_watermark

    wm = read_watermark("baseline-session")
    assert wm is not None


# ---------------------------------------------------------------------------
# No trigger when not newer
# ---------------------------------------------------------------------------


def test_no_trigger_when_not_newer(iai_home, monkeypatch):
    """watermark == current MAX -> no RPC call, no additionalContext (common path)."""
    from iai_mcp import cli
    from iai_mcp.cli import read_watermark, write_watermark

    store = _open_store(iai_home)
    _insert_record(store, "alice refactored the parser")

    # Seed the watermark at the current MAX.
    from iai_mcp.session import max_record_created_at

    current_max = max_record_created_at(store)
    assert current_max is not None
    write_watermark("same-session", current_max)

    rpc_calls: list = []

    def fake_rpc(method, params, **_kw):
        rpc_calls.append((method, params))
        return None

    monkeypatch.setattr(cli, "_send_jsonrpc_request", fake_rpc)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    import argparse

    args = argparse.Namespace(session_id="same-session")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert rpc_calls == [], "RPC must not fire when nothing new exists"
    assert captured.getvalue() == ""


# ---------------------------------------------------------------------------
# Trigger when newer
# ---------------------------------------------------------------------------


def test_trigger_when_newer(iai_home, monkeypatch):
    """watermark < current MAX -> RPC called once; additionalContext emitted;
    watermark advances to new_max_ts."""
    from iai_mcp import cli
    from iai_mcp.cli import read_watermark, write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)

    # Insert a record and capture its MAX as the "old" watermark.
    _insert_record(store, "alice shipped the tokenizer module for the compiler pipeline")
    old_max = max_record_created_at(store)
    assert old_max is not None
    write_watermark("trigger-session", old_max)

    # Insert a second (newer) record so MAX(created_at) advances.
    # The text must be distinct enough to avoid the cos>=0.95 dedup path;
    # a completely different domain prevents any accidental reinforcement.
    time.sleep(0.2)
    r2 = _insert_record(store, "chlorophyll absorbs red and blue light to drive the light reactions")
    # Verify the second insert was a genuine new record (not reinforced/skipped).
    assert r2.get("status") == "inserted", f"Second insert status: {r2}"
    new_max = max_record_created_at(store)
    assert new_max is not None
    assert new_max > old_max  # sanity: created_at is monotone

    new_max_ts_returned = new_max

    rpc_calls: list = []

    def fake_rpc(method, params, **_kw):
        rpc_calls.append((method, params))
        return {"result": {"rendered": "## Memory refreshed\n\nalice shipped the parser refactor", "new_max_ts": new_max_ts_returned}}

    monkeypatch.setattr(cli, "_send_jsonrpc_request", fake_rpc)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    import argparse

    args = argparse.Namespace(session_id="trigger-session")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert len(rpc_calls) == 1, "RPC must be called exactly once"
    method, params = rpc_calls[0]
    assert method == "session_refresh_if_stale"
    from iai_mcp.cli import _utc_iso
    assert _utc_iso(params["watermark"]) == _utc_iso(old_max)
    assert params["session_id"] == "trigger-session"

    out = captured.getvalue()
    assert out != "", "additionalContext JSON must be emitted"
    payload = json.loads(out)
    assert "hookSpecificOutput" in payload
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "alice shipped the parser refactor" in payload["hookSpecificOutput"]["additionalContext"]

    # Watermark must advance.
    wm_after = read_watermark("trigger-session")
    assert wm_after is not None
    # The new watermark should equal new_max_ts (UTC-normalized).
    from iai_mcp.cli import _utc_iso
    assert _utc_iso(wm_after) == _utc_iso(new_max_ts_returned)


# ---------------------------------------------------------------------------
# Daemon-down path
# ---------------------------------------------------------------------------


def test_daemon_down(iai_home, monkeypatch):
    """RPC returns None (daemon unreachable): no additionalContext, watermark unchanged."""
    from iai_mcp import cli
    from iai_mcp.cli import read_watermark, write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)
    _insert_record(store, "alice added the event bus")
    old_max = max_record_created_at(store)
    assert old_max is not None

    # Seed watermark OLDER than current MAX so the gate would fire.
    old_wm = "2020-01-01T00:00:00+00:00"
    write_watermark("daemon-down-session", old_wm)

    monkeypatch.setattr(cli, "_send_jsonrpc_request", lambda *_a, **_kw: None)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    import argparse

    args = argparse.Namespace(session_id="daemon-down-session")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert captured.getvalue() == "", "No output when daemon is down"

    # Watermark must be unchanged.
    wm_after = read_watermark("daemon-down-session")
    assert wm_after is not None
    from iai_mcp.cli import _utc_iso
    assert _utc_iso(wm_after) == _utc_iso(old_wm)


# ---------------------------------------------------------------------------
# UTC normalization
# ---------------------------------------------------------------------------


def test_utc_normalization(iai_home, monkeypatch):
    """Mixed-offset timestamps compare correctly: same instant, no spurious trigger."""
    from iai_mcp import cli
    from iai_mcp.cli import _utc_iso, write_watermark

    # 'Z' and '+00:00' are the same instant.
    ts_z = "2026-05-29T12:00:00Z"
    ts_offset = "2026-05-29T12:00:00+00:00"
    assert _utc_iso(ts_z) == _utc_iso(ts_offset), "Z and +00:00 must normalize identically"

    # A positive-offset form that is in the future in UTC when compared naively.
    ts_positive = "2026-05-29T14:00:00+02:00"  # same instant as 12:00:00Z
    assert _utc_iso(ts_positive) == _utc_iso(ts_z)

    # If watermark == current MAX (both normalized), no trigger.
    # Seed watermark with the '+02:00' form, seed db with 'Z' form.
    # The comparison must see them as equal and NOT call the RPC.
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)
    _insert_record(store, "alice checked in the event log")
    current_max = max_record_created_at(store)
    assert current_max is not None

    # Write the watermark as an equivalent but differently-formatted string.
    # We manufacture the equivalence by normalizing current_max and then
    # reformatting it with a +00:00 offset instead of Z (if it ends in Z).
    if current_max.endswith("+00:00"):
        wm_alt = current_max.replace("+00:00", "Z")
    else:
        wm_alt = current_max.replace("Z", "+00:00") if current_max.endswith("Z") else current_max

    write_watermark("tz-session", wm_alt)

    rpc_calls: list = []
    monkeypatch.setattr(cli, "_send_jsonrpc_request", lambda *a, **kw: rpc_calls.append(a) or None)

    import argparse

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    args = argparse.Namespace(session_id="tz-session")
    cli.cmd_session_refresh_if_stale(args)

    assert rpc_calls == [], "Same instant in different TZ format must NOT trigger"


# ---------------------------------------------------------------------------
# SC3 end-to-end (gate -> core.dispatch in-process -> drain -> compose -> inject)
# ---------------------------------------------------------------------------


def test_sc3_end_to_end(iai_home, monkeypatch):
    """SC3: out-of-band record inserted after baseline surfaces in additionalContext.

    Wires _send_jsonrpc_request in-process to core.dispatch so the full
    gate -> drain -> compose -> inject path is exercised without a socket.
    """
    from iai_mcp import cli
    from iai_mcp.cli import write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)

    # Insert a seed record and set the baseline watermark.
    _insert_record(store, "alice set up the project scaffolding")
    baseline_max = max_record_created_at(store)
    assert baseline_max is not None
    write_watermark("sc3-session", baseline_max)

    # Wait long enough to ensure the OOB record gets a strictly newer created_at.
    # SQLite TEXT timestamps have microsecond resolution; a 200ms sleep is
    # sufficient on all supported platforms.
    time.sleep(0.2)

    # Insert an out-of-band record (newer than the watermark).
    # Use text distinct enough to avoid the cos>=0.95 dedup path.
    r2 = _insert_record(store, "mitochondria produce ATP through oxidative phosphorylation in the inner membrane")
    assert r2.get("status") == "inserted", f"OOB insert must be a new record: {r2}"

    new_max = max_record_created_at(store)
    assert new_max is not None
    assert new_max > baseline_max, "OOB record must have a newer created_at"

    # Wire _send_jsonrpc_request to call core.dispatch in-process.
    def in_process_rpc(method: str, params: dict, **_kw) -> dict:
        from iai_mcp.core import dispatch as core_dispatch
        result = core_dispatch(store, method, params)
        return {"result": result}

    monkeypatch.setattr(cli, "_send_jsonrpc_request", in_process_rpc)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    import argparse

    args = argparse.Namespace(session_id="sc3-session")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    out = captured.getvalue()
    assert out != "", "additionalContext must be emitted on SC3 trigger"

    payload = json.loads(out)
    assert "hookSpecificOutput" in payload
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert ctx, "additionalContext must not be empty"
    # The OOB record text must appear in the rendered brief.
    assert "mitochondria" in ctx or "oxidative phosphorylation" in ctx, (
        f"OOB record text not found in additionalContext:\n{ctx[:500]}"
    )


# ---------------------------------------------------------------------------
# Hook-shape regression: src/iai_mcp/_deploy/hooks/iai-mcp-turn-capture.sh stays thin-client
# ---------------------------------------------------------------------------


def test_hook_shape_regression():
    """src/iai_mcp/_deploy/hooks/iai-mcp-turn-capture.sh must:
    (a) still reference.live.jsonl (per-turn deferred capture),
    (b) contain the inlined freshness gate wired via the RPC method string
        (session_refresh_if_stale, underscore) and emit additionalContext,
    (c) NOT contain MemoryStore(or drain_deferred_captures (thin-client invariant).
    """
    # Resolve the hook from the repo root (not the installed ~/.claude/hooks copy).
    hook_path = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-turn-capture.sh"
    assert hook_path.exists(), f"Hook not found at {hook_path}"

    content = hook_path.read_text()

    # (a) Per-turn deferred capture still present.
    assert ".live.jsonl" in content, (
        "Hook must still write to .live.jsonl (per-turn capture must not be removed)"
    )

    # (b) Freshness gate is inlined in the hook: the gate contacts the daemon via the
    # session_refresh_if_stale RPC method (underscore form, not the deleted CLI
    # subcommand) and emits additionalContext JSON to stdout on a triggered response.
    assert "session_refresh_if_stale" in content, (
        "Hook must contain the inlined gate using the session_refresh_if_stale RPC method"
    )
    assert "additionalContext" in content, (
        "Hook must emit additionalContext JSON to stdout when the gate triggers"
    )

    # (c) Thin-client invariant: hook must not open a MemoryStore or call drain directly.
    assert "MemoryStore(" not in content, (
        "Hook must NOT open a MemoryStore — all store mutations are daemon-owned"
    )
    assert "drain_deferred_captures" not in content, (
        "Hook must NOT call drain_deferred_captures directly — call via daemon RPC"
    )


# ---------------------------------------------------------------------------
# drain_active_live_captures — unit-level tests
# ---------------------------------------------------------------------------


def _write_live_file(home: Path, session_id: str, texts: list) -> Path:
    """Write a still-open.live.jsonl file for session_id with the given turns."""
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    live = deferred_dir / f"{session_id}.live.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "cwd": "/tmp",
    }
    with live.open("w") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for text in texts:
            ev = {
                "text": text,
                "cue": f"session {session_id} live turn",
                "tier": "episodic",
                "role": "user",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return live


def test_drain_active_live_b_still_open(iai_home):
    """SC3 headline: B's live file is drained; file still exists; drain-offset recorded."""
    from iai_mcp.capture import drain_active_live_captures

    store = _open_store(iai_home)

    # Write a still-open.live.jsonl for session B with one turn.
    b_session = "session-b-live"
    live_file = _write_live_file(
        iai_home,
        b_session,
        ["alice completed the live-file parser feature"],
    )

    # Drain from session A (exclude_session_id != b_session).
    counts = drain_active_live_captures(store, exclude_session_id="session-a-refresh")

    # B's turn must have been inserted.
    assert counts["events_inserted"] >= 1, f"Expected at least 1 insert, got {counts}"

    # The.live.jsonl file must still exist — B must keep appending.
    assert live_file.exists(), ".live.jsonl must NOT be deleted during active drain"

    # drain-offset sidecar must be recorded.
    offset_path = iai_home / ".iai-mcp" / ".capture-state" / f"{b_session}.drain-offset"
    assert offset_path.exists(), ".drain-offset sidecar must be written"
    offset_val = int(offset_path.read_text().strip())
    assert offset_val >= 1, "drain-offset must reflect the number of events drained"


def test_drain_active_live_idempotency(iai_home):
    """A second drain with no new B lines processes nothing (offset honored)."""
    from iai_mcp.capture import drain_active_live_captures
    from iai_mcp.session import max_record_created_at
    from iai_mcp.store import flush_record_buffer

    store = _open_store(iai_home)
    b_session = "session-b-idem"
    _write_live_file(iai_home, b_session, ["alice drafted the idempotency contract"])

    # First drain.
    c1 = drain_active_live_captures(store, exclude_session_id="session-a")
    # Flush so SQLite reflects the write (count_rows, max_record_created_at read from SQLite).
    flush_record_buffer(store)
    assert c1["events_inserted"] >= 1

    # Guard against vacuity: something must be in SQLite.
    record_count_after_first = store.db.open_table("records").count_rows()
    assert record_count_after_first >= 1, "First drain must land rows in SQLite"
    max_after_first = max_record_created_at(store)

    # Second drain — no new lines.
    c2 = drain_active_live_captures(store, exclude_session_id="session-a")
    flush_record_buffer(store)
    assert c2["events_inserted"] == 0, "Second drain must insert nothing (offset honored)"
    assert c2["events_reinforced"] == 0, "Second drain must reinforce nothing either"

    # Store state must be identical.
    assert store.db.open_table("records").count_rows() == record_count_after_first
    assert max_record_created_at(store) == max_after_first


def test_drain_active_live_no_self_drain(iai_home):
    """The refreshing session's own.live.jsonl is excluded."""
    from iai_mcp.capture import drain_active_live_captures

    store = _open_store(iai_home)
    a_session = "session-a-self"
    _write_live_file(iai_home, a_session, ["alice wrote a self-referential turn"])

    # Drain with exclude_session_id == a_session (own session).
    counts = drain_active_live_captures(store, exclude_session_id=a_session)

    # Nothing must be inserted — own file is excluded.
    assert counts["events_inserted"] == 0, "Own .live.jsonl must NOT be drained"

    # The.live.jsonl must still exist.
    live_file = iai_home / ".iai-mcp" / ".deferred-captures" / f"{a_session}.live.jsonl"
    assert live_file.exists(), ".live.jsonl must not be deleted"


def test_drain_active_live_no_double_insert(iai_home):
    """After partial live-drain, renaming B's file and running normal drain
    does NOT duplicate already-drained turns (cos>=0.95 dedup backstop).

    Note on duplicate prevention: drain_active_live_captures does NOT
    carry the.drain-offset across the Stop-hook rename to
    .live-{epoch}.jsonl. The duplicate backstop is the cos>=0.95 dedup
    inside capture_turn — identical text embeds to cos=1.0, returning
    status="reinforced" without a new row. This test verifies that backstop
    fires and no new record is created by the normal drain pass.
    """
    from iai_mcp.capture import drain_active_live_captures, drain_deferred_captures
    from iai_mcp.store import flush_record_buffer

    store = _open_store(iai_home)
    b_session = "session-b-nodup"

    # Create the live file and drain it via drain_active_live_captures.
    live_file = _write_live_file(
        iai_home,
        b_session,
        ["alice finalized the dedup contract logic"],
    )
    counts_live = drain_active_live_captures(store, exclude_session_id="session-a")
    # Flush so SQLite reflects the insert and the dedup query in the next
    # drain can find it via count_rows() > 0 + hnswlib knn_query.
    flush_record_buffer(store)
    assert counts_live["events_inserted"] >= 1

    # Guard against vacuity: count_rows must be > 0 (the insert actually landed).
    record_count_after_live = store.db.open_table("records").count_rows()
    assert record_count_after_live >= 1, (
        "Live drain did not land any rows in SQLite — test would be vacuous"
    )

    # Simulate end-of-session: rename.live.jsonl ->.live-{epoch}.jsonl.
    epoch = int(time.time())
    ended_file = live_file.parent / f"{b_session}.live-{epoch}.jsonl"
    live_file.rename(ended_file)

    # Run normal drain (which processes the now-renamed file).
    # The duplicate backstop is cos>=0.95: identical text → cos=1.0 → reinforced.
    counts_norm = drain_deferred_captures(store)
    flush_record_buffer(store)

    # Dedup must have fired (reinforced >= 1) and no new insert must have happened.
    assert counts_norm.get("events_reinforced", 0) >= 1, (
        f"Expected at least one reinforcement from cos>=0.95 dedup, got: {counts_norm}"
    )
    assert counts_norm.get("events_inserted", 0) == 0, (
        f"Normal drain must not insert duplicate records: {counts_norm}"
    )

    record_count_after_normal = store.db.open_table("records").count_rows()
    assert record_count_after_normal == record_count_after_live, (
        f"Normal drain after live-drain must not add records: "
        f"{record_count_after_live} -> {record_count_after_normal}"
    )


# ---------------------------------------------------------------------------
# SC3 headline: full path with B still open (gate -> drain_active -> compose -> inject)
# ---------------------------------------------------------------------------


def test_sc3_b_still_open_surfaces_via_refresh(iai_home, monkeypatch):
    """SC3 with B still open: A's refresh trigger drains B's live file and
    the composed brief contains B's turn text."""
    from iai_mcp import cli
    from iai_mcp.cli import write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)

    # Insert a seed record and set A's baseline watermark.
    _insert_record(store, "alice seeded the store for cross-session test")
    baseline_max = max_record_created_at(store)
    assert baseline_max is not None
    write_watermark("session-a-trigger", baseline_max)

    # Create B's still-open.live.jsonl with a turn that is NOT yet in the store.
    b_session = "session-b-open"
    _write_live_file(
        iai_home,
        b_session,
        ["alice shipped the live-file cross-session feature in session B"],
    )

    # A's gate reads MAX(created_at) from the store — B's turn is not there yet,
    # so the store MAX is still at baseline. To trigger the RPC we need the
    # store MAX to exceed the watermark. Insert a distinct-domain record
    # (simulating any concurrent store advance) so the gate fires.
    # Use text unrelated to the seed to avoid the cos>=0.95 dedup path.
    time.sleep(0.2)
    r_trigger = _insert_record(store, "photosynthesis converts carbon dioxide and water into glucose using sunlight")
    assert r_trigger.get("status") == "inserted", f"Trigger insert must be a new record: {r_trigger}"
    new_store_max = max_record_created_at(store)
    assert new_store_max is not None
    assert new_store_max > baseline_max

    # Wire _send_jsonrpc_request in-process to core.dispatch (same pattern as
    # test_sc3_end_to_end).
    def in_process_rpc(method: str, params: dict, **_kw) -> dict:
        from iai_mcp.core import dispatch as core_dispatch
        result = core_dispatch(store, method, params)
        return {"result": result}

    monkeypatch.setattr(cli, "_send_jsonrpc_request", in_process_rpc)

    captured = StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    import argparse
    args = argparse.Namespace(session_id="session-a-trigger")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    out = captured.getvalue()
    assert out != "", "additionalContext must be emitted when B's live turn is drained"

    payload = json.loads(out)
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert ctx, "additionalContext must not be empty"

    # B's live-file turn must surface in the composed brief.
    assert "alice shipped the live-file cross-session feature in session B" in ctx or \
           "live-file cross-session" in ctx or \
           "session B" in ctx, (
        f"B's live-file turn not found in brief:\n{ctx[:600]}"
    )

    # B's.live.jsonl must still exist (B has not ended).
    live_b = iai_home / ".iai-mcp" / ".deferred-captures" / f"{b_session}.live.jsonl"
    assert live_b.exists(), "B's .live.jsonl must survive the active drain"

    # drain-offset for B must be set.
    offset_b = iai_home / ".iai-mcp" / ".capture-state" / f"{b_session}.drain-offset"
    assert offset_b.exists(), "drain-offset for B must be recorded after live drain"


# ---------------------------------------------------------------------------
# Live-fingerprint gate: B-still-open case (the trigger gap fix)
# ---------------------------------------------------------------------------


def test_live_growth_only_trips_gate(iai_home, monkeypatch):
    """B-still-open via LIVE-GROWTH ONLY.

    Store MAX(created_at) is UNCHANGED after baseline, but another session's
    .live.jsonl grows with a new turn. The gate must detect the live growth
    and trip → refresh is invoked → fingerprint baseline advances.
    """
    from iai_mcp import cli
    from iai_mcp.cli import (
        read_live_fingerprint,
        read_watermark,
        write_live_fingerprint,
        write_watermark,
    )
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)
    _insert_record(store, "alice seeded the store for live-growth gate test")
    baseline_max = max_record_created_at(store)
    assert baseline_max is not None

    # Set baseline watermark at current store MAX — Signal A is silent.
    write_watermark("session-a-lg", baseline_max)

    # No other live files exist yet — seed the fingerprint baseline at 0.
    write_live_fingerprint("session-a-lg", 0)

    # Now write B's live file (store MAX does NOT advance — no drain ran).
    b_session = "session-b-lg"
    _write_live_file(
        iai_home,
        b_session,
        ["alice completed the live-growth gate feature for cross-session continuity"],
    )

    # The new live file makes live_size > 0 (the stored fingerprint baseline).
    rpc_calls: list = []

    def fake_rpc(method, params, **_kw):
        rpc_calls.append((method, params))
        return {
            "result": {
                "rendered": "## Memory refreshed\n\nalice live-growth gate fired",
                "new_max_ts": baseline_max,
            }
        }

    monkeypatch.setattr(cli, "_send_jsonrpc_request", fake_rpc)

    captured = __import__("io").StringIO()
    monkeypatch.setattr(__import__("sys"), "stdout", captured)

    import argparse

    args = argparse.Namespace(session_id="session-a-lg")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert len(rpc_calls) == 1, "Gate must trip (live growth) and send RPC"
    assert captured.getvalue() != "", "additionalContext must be emitted on live-growth trigger"

    # Fingerprint baseline must advance to current live size.
    fp_after = read_live_fingerprint("session-a-lg")
    assert fp_after is not None
    live_size_now = cli.get_other_sessions_live_size("session-a-lg")
    assert fp_after == live_size_now, "Fingerprint must advance to current live size after refresh"


def test_live_growth_idempotent(iai_home, monkeypatch):
    """Idempotency: second gate check with no further live growth and no store
    advance does NOT trip."""
    from iai_mcp import cli
    from iai_mcp.cli import write_live_fingerprint, write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)
    _insert_record(store, "alice seeded the store for live-growth idempotency test")
    baseline_max = max_record_created_at(store)
    assert baseline_max is not None

    # Write B's live file FIRST so we can capture its size.
    b_session = "session-b-idem-lg"
    _write_live_file(
        iai_home,
        b_session,
        ["alice wrote the idempotency contract for live-growth gating"],
    )

    # Seed watermark and fingerprint at current state (simulates: just refreshed).
    write_watermark("session-a-idem-lg", baseline_max)
    current_live_size = cli.get_other_sessions_live_size("session-a-idem-lg")
    write_live_fingerprint("session-a-idem-lg", current_live_size)

    rpc_calls: list = []
    monkeypatch.setattr(
        cli,
        "_send_jsonrpc_request",
        lambda *a, **kw: rpc_calls.append(a) or {"result": {"rendered": "x", "new_max_ts": baseline_max}},
    )

    import argparse
    captured = __import__("io").StringIO()
    monkeypatch.setattr(__import__("sys"), "stdout", captured)

    args = argparse.Namespace(session_id="session-a-idem-lg")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert rpc_calls == [], (
        "Gate must NOT trip when live size is unchanged and store MAX is unchanged"
    )
    assert captured.getvalue() == "", "No additionalContext on idempotent check"


def test_no_self_trigger_own_live(iai_home, monkeypatch):
    """Growth in the current session's own.live.jsonl must NOT trip the gate."""
    from iai_mcp import cli
    from iai_mcp.cli import write_live_fingerprint, write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)
    _insert_record(store, "alice seeded the store for self-trigger exclusion test")
    baseline_max = max_record_created_at(store)
    assert baseline_max is not None

    # Seed watermark and fingerprint at current state.
    session_a = "session-a-self-lg"
    write_watermark(session_a, baseline_max)
    write_live_fingerprint(session_a, 0)

    # Write the CURRENT session's own live file — must NOT count toward live_size.
    own_live = (
        iai_home / ".iai-mcp" / ".deferred-captures" / f"{session_a}.live.jsonl"
    )
    own_live.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "version": 1,
        "deferred_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "session_id": session_a,
        "cwd": "/tmp",
    }
    import json as _json
    with own_live.open("w") as fh:
        fh.write(_json.dumps(header) + "\n")
        ev = {"text": "alice wrote her own turn", "cue": "", "tier": "episodic", "role": "user", "ts": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
        fh.write(_json.dumps(ev) + "\n")

    rpc_calls: list = []
    monkeypatch.setattr(
        cli,
        "_send_jsonrpc_request",
        lambda *a, **kw: rpc_calls.append(a) or None,
    )

    import argparse
    captured = __import__("io").StringIO()
    monkeypatch.setattr(__import__("sys"), "stdout", captured)

    args = argparse.Namespace(session_id=session_a)
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert rpc_calls == [], (
        "Own session's .live.jsonl growth must NOT trip the gate (no self-trigger)"
    )
    assert captured.getvalue() == ""


def test_first_prompt_live_fingerprint_baseline(iai_home, monkeypatch):
    """First prompt with no fingerprint sidecar must set baseline, NOT trigger.

    Covers the case where a session was started before the live-fingerprint
    feature was deployed: the sidecar is absent but B's live file already
    exists. The gate must treat the current live_size as the new baseline
    without firing the RPC.
    """
    from iai_mcp import cli
    from iai_mcp.cli import read_live_fingerprint, write_watermark
    from iai_mcp.session import max_record_created_at

    store = _open_store(iai_home)
    _insert_record(store, "alice seeded the store for first-prompt fingerprint test")
    baseline_max = max_record_created_at(store)
    assert baseline_max is not None

    # Set the watermark (simulates: session already had a first prompt with old code).
    # Do NOT set the fingerprint sidecar — it is absent.
    write_watermark("session-a-fp-first", baseline_max)

    # B's live file exists before A's first fingerprint check.
    b_session = "session-b-fp-first"
    _write_live_file(
        iai_home,
        b_session,
        ["alice wrote a pre-existing B turn before A's first fingerprint check"],
    )

    rpc_calls: list = []
    monkeypatch.setattr(
        cli,
        "_send_jsonrpc_request",
        lambda *a, **kw: rpc_calls.append(a) or None,
    )

    import argparse
    captured = __import__("io").StringIO()
    monkeypatch.setattr(__import__("sys"), "stdout", captured)

    args = argparse.Namespace(session_id="session-a-fp-first")
    rc = cli.cmd_session_refresh_if_stale(args)

    assert rc == 0
    assert rpc_calls == [], (
        "First look at live files (no fingerprint sidecar) must NOT trigger — "
        "it sets the baseline instead"
    )
    assert captured.getvalue() == ""

    # Fingerprint must now be set to the current live size.
    fp = read_live_fingerprint("session-a-fp-first")
    assert fp is not None, "Fingerprint sidecar must be written on first look"
    expected = cli.get_other_sessions_live_size("session-a-fp-first")
    assert fp == expected, "Fingerprint must equal current live size"
