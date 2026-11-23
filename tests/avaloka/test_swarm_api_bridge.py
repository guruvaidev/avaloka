"""Prove upload/database connection events reach and leave the PR 157 graph."""

from app.api.swarm_bridge import build_swarm_state, consume_connection_event


def test_bridge_keeps_thread_and_session_identifiers_distinct():
    event = {"kind": "database", "tables_available": 4}
    state = build_swarm_state(
        thread_id="thread-real",
        session_id="session-real",
        session={"connection_event": event},
    )
    assert state == {
        "thread_id": "thread-real",
        "session_id": "session-real",
        "connection_event": event,
    }


def test_consumed_connection_event_is_cleared_once():
    session = {"connection_event": {"kind": "upload"}, "dataset_id": "d1"}
    assert consume_connection_event(session, {"connection_event": None}) is True
    assert "connection_event" not in session
    assert consume_connection_event(session, {"connection_event": None}) is False


def test_missing_graph_key_does_not_drop_unconsumed_event():
    event = {"kind": "upload"}
    session = {"connection_event": event}
    assert consume_connection_event(session, {}) is False
    assert session["connection_event"] == event
