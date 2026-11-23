"""J1: a deferred turn whose result is stored as done is always returned.

In the rc5 web re-check one of 19 deferred turns (a taxi question) came back
``superseded`` and empty, although the server then stored it as done. The poll
landed between the end of the graph and the done write. In that window the
turn merged its own session copy back (``cur.update(primary_sess)``); that copy
was read at the start of the turn, before the turn's "running" record existed,
so it put back the PREVIOUS turn's record, and the poll compared ids and said
"superseded". The UI poller stops on "superseded", so a browser user saw nothing.

A thread also kept one slot only, so a turn finishing after a newer one started
overwrote the newer one's record (and the newer turn's poll said "superseded").
"""
import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app.api.server as server

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
        SESSIONS, THREAD_TO_SESSION, make_auth_headers, client, shared_loop_client, _do_upload,
        update_session_fake,
    )
except ImportError:
    from server_harness import (  # noqa: F401
        SESSIONS, THREAD_TO_SESSION, make_auth_headers, client, shared_loop_client, _do_upload,
        update_session_fake,
    )

THREAD = "thread-race"
USER = "user-race"


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# The observed race, end to end through send_message
# ---------------------------------------------------------------------------

def test_turn_merge_never_puts_back_the_previous_turns_record(shared_loop_client: TestClient, monkeypatch):
    client = shared_loop_client
    monkeypatch.setattr(server, "TURN_SYNC_DEADLINE_S", 0.05, raising=False)
    monkeypatch.setattr(server, "_PENDING_TURN_MEM", {})
    for name in ("_parse_fidelity_from_text", "_parse_selected_sample_from_text"):
        monkeypatch.setattr(server, name, lambda t: None, raising=False)
    monkeypatch.setattr(server, "_is_mode_switch_message", lambda t: False, raising=False)

    class SlowGraph:
        checkpointer = object()

        def invoke(self, state_in, config=None):
            time.sleep(0.4)
            return {
                "messages": state_in["messages"] + [AIMessage(content="the taxi answer")],
                "planner_definition": {}, "ready_to_summarize": False, "ready_to_code": False,
                "coder_definition": {}, "visualization_config": {}, "visualization_status": "",
                "execution_result": {},
                # Any session change makes the turn merge its copy back.
                "pending_clarification": None,
            }

    monkeypatch.setattr(server, "GRAPH", SlowGraph(), raising=False)
    upload = _do_upload(client, user_id="race-user")
    thread_id, dsid = upload["thread_id"], upload["dataset_id"]
    sid = THREAD_TO_SESSION[thread_id]
    # The previous deferred turn on this thread, already finished.
    SESSIONS[sid]["pending_turn"] = {"id": "previous", "status": "done", "kind": "analysis",
                                     "result": {"messages": [{"role": "assistant", "content": "old"}]}}

    # Every poll the user could have made: one after each session write.
    polls = []

    async def spy(session_id, mutator):
        merged = await update_session_fake(session_id, mutator)
        if session_id == sid and deferred.get("id"):
            polls.append(await server.get_pending_turn(None, thread_id, deferred_id=deferred["id"]))
        return merged

    deferred = {}
    monkeypatch.setattr(server, "update_session", spy, raising=False)
    monkeypatch.setattr(server, "_resolve_user_id", lambda req: "race-user")

    resp = client.post(f"/threads/{thread_id}/messages",
                       json={"content": "what drives the fare?", "metadata": {"dataset_id": dsid}},
                       headers=make_auth_headers("race-user"))
    assert resp.status_code == 200, resp.text
    deferred["id"] = resp.json()["analysis_task_id"]
    assert deferred["id"]

    deadline = time.monotonic() + 10
    final = None
    while time.monotonic() < deadline:
        final = client.get(f"/threads/{thread_id}/pending-turn?deferred_id={deferred['id']}",
                           headers=make_auth_headers("race-user")).json()
        if final["status"] == "done":
            break
        time.sleep(0.05)

    assert polls, "the turn wrote nothing to the session after it was deferred"
    assert all(p["status"] in ("running", "done") for p in polls), polls
    assert final["status"] == "done", final
    assert "the taxi answer" in str(final["result"])
    # The turn before it is still answerable by its own id.
    prev = client.get(f"/threads/{thread_id}/pending-turn?deferred_id=previous",
                      headers=make_auth_headers("race-user")).json()
    assert prev["status"] == "done" and "old" in str(prev["result"])


# ---------------------------------------------------------------------------
# The poll, against stored state
# ---------------------------------------------------------------------------

@pytest.fixture
def stored(monkeypatch):
    sess = {"user_id": USER}
    monkeypatch.setattr(server, "_PENDING_TURN_MEM", {})
    monkeypatch.setattr(server, "_resolve_user_id", lambda req: USER)

    async def thread_session(thread_id):
        return "sid-race"

    async def get(sid):
        return sess

    monkeypatch.setattr(server, "get_thread_session", thread_session)
    monkeypatch.setattr(server, "get_session", get)
    return sess


def poll(deferred_id=None):
    return run(server.get_pending_turn(None, THREAD, deferred_id=deferred_id))


def test_older_turn_finishing_after_a_newer_one_started_is_returned(stored):
    server._pending_turn_running("A")(stored)
    server._pending_turn_running("B")(stored)          # a newer turn starts
    server._pending_turn_done("A", {"answer": "A"})(stored)
    assert poll("A") == {"status": "done", "kind": "analysis", "result": {"answer": "A"}}
    # ...and the newer turn is still running, not "superseded" by A's write.
    assert poll("B") == {"status": "running", "kind": "analysis", "queued": False}
    assert poll()["status"] == "running"               # newest turn is still B
    server._pending_turn_done("B", {"answer": "B"})(stored)
    assert poll("B")["result"] == {"answer": "B"}
    assert poll("A")["result"] == {"answer": "A"}


def test_a_stale_session_copy_cannot_hide_a_running_turn(stored):
    server._pending_turn_done("previous", {"answer": "old"})(stored)
    stale = dict(stored)                                # the turn's copy, read at its start
    baseline = dict(stored)
    server._pending_turn_running("current")(stored)
    stale["latest_output_preview"] = [{"x": 1}]          # the turn changed something else
    server._merge_session_with_drops(stored, stale, baseline)
    assert poll("current") == {"status": "running", "kind": "analysis", "queued": False}
    assert stored["latest_output_preview"] == [{"x": 1}]


def test_memory_copy_answers_when_the_session_lost_the_record(stored):
    # Another writer put back an older record; this process still knows the turn.
    server._remember_pending_turn(THREAD, USER, server._pending_turn_running("current"))
    stored["pending_turn"] = {"id": "previous", "status": "done", "result": {}}
    assert poll("current") == {"status": "running", "kind": "analysis", "queued": False}
    server._remember_pending_turn(THREAD, USER, server._pending_turn_done("current", {"answer": 1}))
    assert poll("current")["status"] == "done"


def test_superseded_only_for_an_unknown_turn_and_names_the_newer_one(stored):
    server._pending_turn_running("newest")(stored)
    assert poll("never-seen") == {"status": "superseded", "superseded_by": "newest"}


def test_done_is_never_downgraded_by_a_late_running_write(stored):
    server._pending_turn_running("A")(stored)
    server._pending_turn_running("B")(stored)
    server._pending_turn_done("A", {"answer": "A"})(stored)
    server._pending_turn_running("A")(stored)           # a late duplicate
    assert poll("A")["status"] == "done"
    assert poll("B")["status"] == "running"


def test_kept_turns_are_bounded(stored):
    for i in range(10):
        server._pending_turn_running(f"t{i}")(stored)
        server._pending_turn_done(f"t{i}", {"i": i})(stored)
    assert len(stored.get("pending_turns") or {}) <= server._PENDING_TURNS_KEEP
    assert poll("t9")["result"] == {"i": 9}
    assert poll("t8")["result"] == {"i": 8}
