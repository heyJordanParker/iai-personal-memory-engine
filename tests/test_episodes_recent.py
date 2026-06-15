from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.core import dispatch
from tests._helpers import make_tmp_store


def _write_live_event(
    home: Path,
    session_id: str,
    text: str,
    *,
    role: str = "user",
    ts: str | None = None,
    source_uuid: str | None = None,
) -> None:
    from iai_mcp.capture import write_deferred_event
    write_deferred_event(session_id, role, text, ts=ts, source_uuid=source_uuid)


def test_returns_n_most_recent_user_turns_time_desc(tmp_path):
    store = make_tmp_store(tmp_path)

    for i in range(5):
        res = capture_turn(
            store,
            cue=f"user turn alpha{i}",
            text=f"user turn alpha{i} phase59 episodic content for recency test",
            tier="episodic",
            session_id="sess-recency-test",
            role="user",
        )
        assert res["status"] == "inserted", f"insert {i} failed: {res}"

    result = dispatch(store, "episodes_recent", {"n": 3})
    assert "turns" in result, (
        f"episodes_recent response missing 'turns' key; "
        f"dispatch returned: {result!r}"
    )
    turns = result["turns"]
    assert len(turns) == 3, f"expected 3 turns, got {len(turns)}"

    timestamps = [t.get("captured_at") for t in turns]
    assert timestamps == sorted(timestamps, reverse=True), (
        f"turns must be newest-first; got {timestamps}"
    )


def test_session_id_filter(tmp_path):
    store = make_tmp_store(tmp_path)

    for i in range(3):
        capture_turn(
            store,
            cue=f"x turn {i}",
            text=f"session x turn {i} distinctive content phase59 xfilter test",
            tier="episodic",
            session_id="session-X-filter",
            role="user",
        )

    for i in range(2):
        capture_turn(
            store,
            cue=f"y turn {i}",
            text=f"session y turn {i} should not appear in x filter result",
            tier="episodic",
            session_id="session-Y-filter",
            role="user",
        )

    result = dispatch(
        store,
        "episodes_recent",
        {"n": 5, "session_id": "session-X-filter"},
    )
    assert "turns" in result, f"episodes_recent response missing 'turns': {result!r}"
    turns = result["turns"]
    assert len(turns) == 3, f"expected 3 turns for session X, got {len(turns)}"
    for t in turns:
        assert t.get("session_id") == "session-X-filter", (
            f"turn {t.get('record_id')} belongs to wrong session: {t.get('session_id')!r}"
        )
    assert turns[0]["literal_surface"].startswith("session x turn 2"), (
        f"most-recent X turn unexpected: {turns[0]['literal_surface']!r}"
    )


def test_no_filter_returns_global_most_recent(tmp_path):
    store = make_tmp_store(tmp_path)

    for i in range(3):
        capture_turn(
            store,
            cue=f"global turn {i}",
            text=f"global turn {i} earlier content for global recency check",
            tier="episodic",
            session_id="sess-global-A",
            role="user",
        )

    capture_turn(
        store,
        cue="final distinctive turn",
        text="final distinctive turn bxyz9999 globally newest in store",
        tier="episodic",
        session_id="sess-global-B",
        role="user",
    )

    result = dispatch(store, "episodes_recent", {"n": 5})
    assert "turns" in result, f"episodes_recent response missing 'turns': {result!r}"
    turns = result["turns"]
    assert turns, "expected at least one turn"
    assert "bxyz9999" in turns[0]["literal_surface"], (
        f"globally newest turn not first (got {turns[0]['literal_surface']!r})"
    )


def test_negative_n_clamp_returns_empty(tmp_path):
    store = make_tmp_store(tmp_path)
    capture_turn(
        store,
        cue="clamp test",
        text="clamp test content phase59 negative n guard",
        tier="episodic",
        session_id="sess-clamp",
        role="user",
    )

    result = dispatch(store, "episodes_recent", {"n": -5})
    assert "turns" in result, f"episodes_recent missing 'turns': {result!r}"
    turns = result["turns"]
    assert turns == [], f"n=-5 must return empty list; got {turns!r}"
    assert result.get("count") == 0


pytestmark_posix = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths / live-capture helper",
)


@pytest.fixture
def iai_home_60(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    yield tmp_path


def test_no_double_count_after_drain(iai_home_60):
    store = make_tmp_store(iai_home_60)

    session = "dedup-session-60a"
    text = "dedup test turn content that is long enough for capture phase60 test"
    src_uuid = "uuid-dedup-a001"
    ts_str = "2026-05-31T10:00:00.000000+00:00"

    capture_turn(
        store,
        cue="dedup drain cue",
        text=text,
        tier="episodic",
        session_id=session,
        role="user",
        ts=ts_str,
        source_uuid=src_uuid,
    )

    _write_live_event(iai_home_60, session, text, ts=ts_str, source_uuid=src_uuid)

    result = dispatch(store, "episodes_recent", {"n": 10, "session_id": session})
    assert "turns" in result, f"episodes_recent response missing 'turns': {result!r}"
    turns = result["turns"]
    assert len(turns) == 1, (
        f"expected 1 turn after dedup; got {len(turns)}: {[t.get('literal_surface') for t in turns]!r}"
    )


def test_dedup_pending_vs_pending(iai_home_60):
    session = "dedup-pend-session-60b"
    text = "pending dedup test turn text long enough for phase60 pending dedup"
    src_uuid = "uuid-pend-dedup-b002"
    ts_str = "2026-05-31T11:00:00.000000+00:00"

    store = make_tmp_store(iai_home_60)

    _write_live_event(iai_home_60, session, text, ts=ts_str, source_uuid=src_uuid)
    _write_live_event(iai_home_60, session, text, ts=ts_str, source_uuid=src_uuid)

    result = dispatch(store, "episodes_recent", {"n": 10, "session_id": session})
    assert "turns" in result
    turns = result["turns"]
    assert len(turns) == 1, (
        f"re-emitted pending line must appear once (seen_pending_idem dedup); "
        f"got {len(turns)}: {[t.get('literal_surface') for t in turns]!r}"
    )


def test_live_turn_sorts_before_older_stored(iai_home_60):
    session = "sort-live-session-60c"
    text_old = "older stored turn for recency sort test phase60 content here"
    text_new = "newer pending live turn sort test phase60 content here brand new"

    ts_old = "2026-05-31T08:00:00.000000+00:00"
    ts_new = "2026-05-31T09:00:00.000000+00:00"

    store = make_tmp_store(iai_home_60)
    capture_turn(
        store,
        cue="old stored turn",
        text=text_old,
        tier="episodic",
        session_id=session,
        role="user",
        ts=ts_old,
        source_uuid="uuid-old-stored-60c",
    )

    _write_live_event(
        iai_home_60, session, text_new,
        ts=ts_new, source_uuid="uuid-new-live-60c",
    )

    result = dispatch(store, "episodes_recent", {"n": 10, "session_id": session})
    turns = result["turns"]
    assert len(turns) >= 2, f"expected >= 2 turns; got {len(turns)}"
    assert text_new in turns[0]["literal_surface"], (
        f"newer live turn must be first; got {turns[0]['literal_surface']!r}"
    )


def test_distinct_uuid_same_text_both_appear(iai_home_60):
    session = "distinct-uuid-session-60d"
    text = "identical text turn for distinct uuid test content phase60 here"
    ts1 = "2026-05-31T12:00:00.000000+00:00"
    ts2 = "2026-05-31T12:01:00.000000+00:00"

    store = make_tmp_store(iai_home_60)

    _write_live_event(
        iai_home_60, session, text, ts=ts1, source_uuid="uuid-distinct-1"
    )
    _write_live_event(
        iai_home_60, session, text, ts=ts2, source_uuid="uuid-distinct-2"
    )

    result = dispatch(store, "episodes_recent", {"n": 10, "session_id": session})
    turns = result["turns"]
    matching = [t for t in turns if t.get("literal_surface") == text]
    assert len(matching) == 2, (
        f"two distinct-uuid events with same text must both appear; "
        f"got {len(matching)}: {[t.get('literal_surface') for t in turns]!r}"
    )


def test_pending_assistant_excluded_from_user_turns(iai_home_60):
    session = "asst-role-session-60e"
    text = "assistant response text that should not appear in user turns query here"

    store = make_tmp_store(iai_home_60)
    _write_live_event(iai_home_60, session, text, role="assistant")

    result = dispatch(store, "episodes_recent", {"n": 10, "session_id": session})
    turns = result["turns"]
    assert len(turns) == 0, (
        f"assistant-role pending turn must NOT appear in episodes_recent; "
        f"got {len(turns)}: {[t.get('literal_surface') for t in turns]!r}"
    )


def test_pending_malformed_role_dropped(iai_home_60):
    session = "system-role-session-60f"

    store = make_tmp_store(iai_home_60)

    deferred = iai_home_60 / ".iai-mcp" / ".deferred-captures"
    deferred.mkdir(parents=True, exist_ok=True)
    path = deferred / f"{session}.live.jsonl"
    header = json.dumps({
        "version": 1, "deferred_at": "2026-05-31T12:00:00+00:00",
        "session_id": session, "cwd": "/tmp",
    })
    ev = json.dumps({
        "text": "system message that must be dropped from user turns query here",
        "role": "system",
        "tier": "episodic",
        "ts": "2026-05-31T12:00:00.000000+00:00",
    })
    path.write_text(header + "\n" + ev + "\n", encoding="utf-8")

    result = dispatch(store, "episodes_recent", {"n": 10, "session_id": session})
    turns = result["turns"]
    assert len(turns) == 0, (
        f"system-role pending event must be dropped by recent_user_turns; got {turns!r}"
    )


def test_pending_record_id_not_literal_none(iai_home_60):
    session_with_uuid = "pending-rid-session-60g1"
    session_no_uuid = "pending-rid-session-60g2"
    text = "pending record id test content long enough for phase60 test here"
    src_uuid = "uuid-rid-test-g001"

    store = make_tmp_store(iai_home_60)

    _write_live_event(iai_home_60, session_with_uuid, text, source_uuid=src_uuid)
    result1 = dispatch(store, "episodes_recent", {"n": 10, "session_id": session_with_uuid})
    turns1 = result1["turns"]
    assert len(turns1) == 1, f"expected 1 turn; got {len(turns1)}"
    rid1 = turns1[0]["record_id"]
    assert rid1 != "None", f"record_id must not be literal 'None'; got {rid1!r}"
    assert rid1.startswith("pending:"), f"pending turn record_id must start with 'pending:'; got {rid1!r}"
    assert src_uuid in rid1, f"source_uuid must be in record_id; got {rid1!r}"

    _write_live_event(iai_home_60, session_no_uuid, text)
    result2 = dispatch(store, "episodes_recent", {"n": 10, "session_id": session_no_uuid})
    turns2 = result2["turns"]
    assert len(turns2) == 1, f"expected 1 turn; got {len(turns2)}"
    rid2 = turns2[0]["record_id"]
    assert rid2 != "None", f"record_id must not be literal 'None'; got {rid2!r}"
    assert rid2.startswith("pending:"), f"must start with 'pending:'; got {rid2!r}"
    suffix = rid2[len("pending:"):]
    assert suffix, f"record_id must not be bare 'pending:'; got {rid2!r}"
