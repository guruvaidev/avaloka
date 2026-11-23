"""End to end through POST /threads/{id}/messages: a background job that
finishes before or during a turn is told to the thread AFTER that turn's reply,
once, in the response (and in a deferred turn's stashed result), with no model
call. Uses tests/server_harness (FakeGraph, in-memory sessions) plus the job
fakes; no Redis, Supabase or provider.

    pytest tests/test_send_message_job_notes.py -q
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

try:
    from app.api import server
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"app.api.server could not be imported here: {exc}", allow_module_level=True)

from app.core import job_ledger as jl
from app.core import job_notes as jn
from app.services import notification_service as ns
from tests.job_fakes import FakeDB

try:
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
        SESSIONS, THREAD_TO_SESSION, make_auth_headers, client, shared_loop_client, _do_upload,
    )
except ImportError:
    from server_harness import (  # noqa: F401
        SESSIONS, THREAD_TO_SESSION, make_auth_headers, client, shared_loop_client, _do_upload,
    )

POLL_TIMEOUT_S = 10.0


@pytest.fixture
def db(monkeypatch):
    fake = FakeDB()
    monkeypatch.setattr(ns, "_client", lambda: fake)
    ns._profile_cache.clear()
    jl.set_store(jl.SupabaseJobs(client=fake))
    jn.set_store(jn.SupabaseNotes(client=fake))
    jn._TURNS_IN_FLIGHT.clear()
    yield fake
    jl.set_store(None)
    jn.set_store(None)
    jn._TURNS_IN_FLIGHT.clear()


def _profile(db, user_id):
    db.tables["profiles"].append({"id": f"profile-{user_id}", "user_id": user_id})


def _finish_job(user_id, thread_id, label="orders ➡️ warehouse"):
    jid = jl.start(user_id, "transfer", thread_id=thread_id, label=label)
    jl.finish(jid, "succeeded")
    return jid


class _GraphThatFinishesAJob:
    """The graph stands in for the whole turn; a Celery worker finishes a job
    while it runs (from a plain thread, as the worker would)."""
    checkpointer = object()

    def __init__(self, user_id, thread_id, delay=0.0):
        self.user_id, self.thread_id, self.delay, self.finished = user_id, thread_id, delay, []

    def invoke(self, state_in, config=None):
        if self.delay:
            time.sleep(self.delay)
        self.finished.append(_finish_job(self.user_id, self.thread_id))
        assert jn.turn_in_flight(self.thread_id)                    # the outbox knows a turn is running
        last = state_in["messages"][-1]
        return {
            "messages": state_in["messages"] + [AIMessage(content=f"Echo: {last.content}")],
            "planner_definition": {}, "ready_to_summarize": False, "ready_to_code": False,
            "coder_definition": {}, "visualization_config": {}, "visualization_status": "",
            "execution_result": {},
        }


def _post(client, thread_id, dsid, user_id, text="hello"):
    return client.post(f"/threads/{thread_id}/messages",
                       json={"content": text, "metadata": {"dataset_id": dsid}},
                       headers=make_auth_headers(user_id))


def test_a_job_finished_before_the_turn_is_told_after_the_reply(client: TestClient, db):
    user = "notes-user-1"
    _profile(db, user)
    up = _do_upload(client, user_id=user)
    thread_id, dsid = up["thread_id"], up["dataset_id"]
    jid = _finish_job(user, thread_id)
    assert db.notes[0]["delivered_at"] is None

    resp = _post(client, thread_id, dsid, user)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["job_notes"] and body["job_notes"][0]["job_ids"] == [jid]
    assert body["job_notes"][0]["text"].endswith("✓ Data transfer (orders ➡️ warehouse) finished.")
    msgs = server.THREAD_META[thread_id]["lc_msgs"]
    assert msgs[-2].content.startswith("Echo:") and jn.is_job_note(msgs[-1]) and not jn.is_job_note(msgs[-2])
    assert db.notes[0]["delivered_at"] and len(db.notifications) == 1

    resp2 = _post(client, thread_id, dsid, user, "again")
    assert resp2.status_code == 200 and resp2.json()["job_notes"] is None       # never twice


def test_a_job_finishing_during_the_turn_is_queued_then_told_after_the_reply(client: TestClient, db, monkeypatch):
    user = "notes-user-2"
    _profile(db, user)
    up = _do_upload(client, user_id=user)
    thread_id, dsid = up["thread_id"], up["dataset_id"]
    graph = _GraphThatFinishesAJob(user, thread_id)
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)

    resp = _post(client, thread_id, dsid, user, "start the transfer")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # "start the transfer" classifies the turn as long, so the API opened a
    # row for it; the fake graph claimed nothing, so that row ended quietly:
    # the only note is the worker's job.
    assert body["job_notes"][0]["job_ids"] == graph.finished
    turn_rows = [r for r in db.jobs if r["id"] not in graph.finished]
    assert len(turn_rows) == 1 and turn_rows[0]["kind"] == "analysis" and turn_rows[0]["status"] == "succeeded"
    assert len(db.notifications) == 1 and db.notifications[0]["metadata"]["job_id"] == graph.finished[0]
    kinds = [("note" if jn.is_job_note(m) else type(m).__name__) for m in server.THREAD_META[thread_id]["lc_msgs"]]
    assert kinds[-3:] == ["HumanMessage", "AIMessage", "note"]
    assert not jn.turn_in_flight(thread_id)


def test_three_turns_with_jobs_finishing_throughout_deliver_each_once(client: TestClient, db, monkeypatch):
    user = "notes-user-3"
    _profile(db, user)
    up = _do_upload(client, user_id=user)
    thread_id, dsid = up["thread_id"], up["dataset_id"]
    graph = _GraphThatFinishesAJob(user, thread_id)
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)
    db.tables["analyses"].append({"id": "9a0b1c2d-0000-4000-8000-00000000aaaa", "thread_id": thread_id})
    told = []
    for i in range(3):
        _finish_job(user, thread_id, label=f"batch {i}")             # one before the turn
        body = _post(client, thread_id, dsid, user, f"q{i}").json()  # and one during it
        assert body["job_notes"] and body["job_notes"][0]["count"] == 2
        told += body["job_notes"][0]["job_ids"]
    assert sorted(told) == sorted(r["id"] for r in db.jobs) and len(told) == 6
    kinds = [("note" if jn.is_job_note(m) else type(m).__name__) for m in server.THREAD_META[thread_id]["lc_msgs"]]
    assert kinds[-9:] == ["HumanMessage", "AIMessage", "note"] * 3
    assert all(n["delivered_at"] for n in db.notes) and len(db.messages) == 3


def test_a_deferred_turn_carries_the_note_in_its_stashed_result(shared_loop_client: TestClient, db, monkeypatch):
    client = shared_loop_client
    monkeypatch.setattr(server, "TURN_SYNC_DEADLINE_S", 0.05, raising=False)
    monkeypatch.setattr(server, "_parse_fidelity_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_parse_selected_sample_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_is_mode_switch_message", lambda t: False, raising=False)
    user = "notes-user-4"
    _profile(db, user)
    up = _do_upload(client, user_id=user)
    thread_id, dsid = up["thread_id"], up["dataset_id"]
    graph = _GraphThatFinishesAJob(user, thread_id, delay=0.4)
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)

    resp = _post(client, thread_id, dsid, user, "run the big one")
    body = resp.json()
    assert body["pending_status"] == "running" and body["job_notes"] is None
    deferred_id = body["analysis_task_id"]
    deadline = time.monotonic() + POLL_TIMEOUT_S
    result = None
    while time.monotonic() < deadline:
        pr = client.get(f"/threads/{thread_id}/pending-turn?deferred_id={deferred_id}", headers=make_auth_headers(user))
        if pr.json()["status"] == "done":
            result = pr.json()["result"]
            break
        time.sleep(0.05)
    assert result is not None
    # Two notes, one batch: the worker's job and the deferred turn itself.
    assert sorted(result["job_notes"][0]["job_ids"]) == sorted(graph.finished + [deferred_id])
    assert not jn.turn_in_flight(thread_id)


def test_an_in_turn_error_still_closes_the_turn_and_tells_the_note(client: TestClient, db, monkeypatch):
    user = "notes-user-5"
    _profile(db, user)
    up = _do_upload(client, user_id=user)
    thread_id, dsid = up["thread_id"], up["dataset_id"]
    jid = _finish_job(user, thread_id)

    class Boom:
        checkpointer = object()

        def invoke(self, state_in, config=None):
            raise RuntimeError("provider retired")

    monkeypatch.setattr(server, "GRAPH", Boom(), raising=False)
    resp = _post(client, thread_id, dsid, user, "anything")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["job_notes"][0]["job_ids"] == [jid]
    msgs = server.THREAD_META[thread_id]["lc_msgs"]
    assert not jn.is_job_note(msgs[-2]) and jn.is_job_note(msgs[-1])      # friendly error, then the note
    assert not jn.turn_in_flight(thread_id)


def test_no_model_is_called_to_deliver(client: TestClient, db, monkeypatch):
    import langchain_core.language_models.chat_models as cm
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called"))  # noqa: E731
    monkeypatch.setattr(cm.BaseChatModel, "invoke", boom)
    monkeypatch.setattr(cm.BaseChatModel, "ainvoke", boom)
    user = "notes-user-6"
    _profile(db, user)
    up = _do_upload(client, user_id=user)
    thread_id, dsid = up["thread_id"], up["dataset_id"]
    _finish_job(user, thread_id)
    body = _post(client, thread_id, dsid, user).json()
    assert body["job_notes"]


# ---- the long-jobs pool and the deferred turn's own ledger row -------------------------

from app.core import long_jobs  # noqa: E402


@pytest.fixture
def long_pool():
    pool = long_jobs.LongJobPool(max_workers=1)
    long_jobs.set_pool(pool)
    yield pool
    pool.shutdown()
    long_jobs.set_pool(None)


class _RecordingGraph:
    """Records the thread it ran on and, like the MTA does, claims the turn's
    ledger row when ``claim_kind`` is set (a graph that claims nothing leaves
    an ordinary turn behind, which the API settles quietly)."""
    checkpointer = object()

    def __init__(self, claim_kind=None):
        self.threads = []
        self.claim_kind = claim_kind

    def invoke(self, state_in, config=None):
        import threading
        self.threads.append(threading.current_thread().name)
        if self.claim_kind and state_in.get("ledger_job_id"):
            jl.claim(state_in["ledger_job_id"], kind=self.claim_kind, resource_type="model", resource_id="run-1")
        last = state_in["messages"][-1]
        return {
            "messages": state_in["messages"] + [AIMessage(content=f"Echo: {last.content}")],
            "planner_definition": {}, "ready_to_summarize": False, "ready_to_code": False,
            "coder_definition": {}, "visualization_config": {}, "visualization_status": "",
            "execution_result": {}, "ledger_seen": state_in.get("ledger_job_id"),
        }


def test_a_training_turn_runs_on_the_long_pool_and_is_a_ledger_job(client: TestClient, db, monkeypatch, long_pool):
    user = "long-user-1"
    _profile(db, user)
    up = _do_upload(client, user_id=user)
    thread_id, dsid = up["thread_id"], up["dataset_id"]
    graph = _RecordingGraph(claim_kind="training")
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)

    body = _post(client, thread_id, dsid, user, "please train a model to predict churn").json()
    assert graph.threads and graph.threads[0].startswith("avaloka-long-job")
    row = db.jobs[0]
    assert (row["kind"], row["status"], row["thread_id"], row["origin"]) == ("training", "succeeded", thread_id, "api")
    assert row["resource_type"] == "model" and row["resource_id"] == "run-1"
    assert "train a model" in row["label"] and row["notified_at"]
    n = db.notifications[0]
    assert (n["type"], n["title"]) == ("training.completed", "Model training completed")
    assert "train a model to predict churn" in n["body"] and n["metadata"]["ref"] == f"job:{row['id']}"
    assert body["job_notes"][0]["job_ids"] == [row["id"]]          # its own note lands after its reply


def test_a_training_turn_that_trained_nothing_ends_quietly(client: TestClient, db, monkeypatch, long_pool):
    user = "long-user-1b"
    _profile(db, user)
    up = _do_upload(client, user_id=user)
    thread_id, dsid = up["thread_id"], up["dataset_id"]
    monkeypatch.setattr(server, "GRAPH", _RecordingGraph(), raising=False)       # claims nothing
    body = _post(client, thread_id, dsid, user, "please train a model to predict churn").json()
    row = db.jobs[0]
    assert (row["kind"], row["status"]) == ("analysis", "succeeded") and row["notified_at"]
    assert db.notifications == [] and body["job_notes"] is None                  # no false "training completed"


def test_an_ordinary_turn_stays_on_the_default_pool_with_no_ledger_row(client: TestClient, db, monkeypatch, long_pool):
    user = "long-user-2"
    _profile(db, user)
    up = _do_upload(client, user_id=user)
    thread_id, dsid = up["thread_id"], up["dataset_id"]
    graph = _RecordingGraph()
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)
    body = _post(client, thread_id, dsid, user, "sum sales by region").json()
    assert graph.threads and not graph.threads[0].startswith("avaloka-long-job")
    assert db.jobs == [] and db.notifications == [] and body["job_notes"] is None


def test_a_full_long_pool_shows_the_turn_as_queued_then_running(shared_loop_client: TestClient, db, monkeypatch, long_pool):
    import threading
    client = shared_loop_client
    monkeypatch.setattr(server, "TURN_SYNC_DEADLINE_S", 0.05, raising=False)
    monkeypatch.setattr(server, "_parse_fidelity_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_parse_selected_sample_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_is_mode_switch_message", lambda t: False, raising=False)
    user = "long-user-3"
    _profile(db, user)
    up = _do_upload(client, user_id=user)
    thread_id, dsid = up["thread_id"], up["dataset_id"]
    graph = _RecordingGraph()
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)

    # Occupy the single long worker from outside, as another user's job would.
    release = threading.Event()
    long_pool.submit_sync(lambda: release.wait(10))
    time.sleep(0.05)
    assert long_pool.would_queue()

    body = _post(client, thread_id, dsid, user, "train a model on churn").json()
    assert body["pending_status"] == "running"
    deferred_id = body["analysis_task_id"]
    pr = client.get(f"/threads/{thread_id}/pending-turn?deferred_id={deferred_id}", headers=make_auth_headers(user)).json()
    assert pr["status"] == "running" and pr["queued"] is True
    assert db.jobs[0]["status"] == "queued" and db.jobs[0]["id"] == deferred_id

    release.set()
    deadline = time.monotonic() + POLL_TIMEOUT_S
    seen_running_unqueued = False
    final = None
    while time.monotonic() < deadline:
        pr = client.get(f"/threads/{thread_id}/pending-turn?deferred_id={deferred_id}", headers=make_auth_headers(user)).json()
        if pr["status"] == "running" and pr.get("queued") is False:
            seen_running_unqueued = True
        if pr["status"] == "done":
            final = pr
            break
        time.sleep(0.02)
    assert final is not None
    assert db.jobs[0]["status"] == "succeeded" and db.jobs[0]["started_at"]
    assert len(db.notifications) == 1
    # queued -> running was observable unless the job ran faster than one poll
    assert seen_running_unqueued or final["result"]["job_notes"]


def test_a_deferred_analysis_turn_is_a_ledger_job_that_rings_once(shared_loop_client: TestClient, db, monkeypatch):
    client = shared_loop_client
    monkeypatch.setattr(server, "TURN_SYNC_DEADLINE_S", 0.05, raising=False)
    monkeypatch.setattr(server, "_parse_fidelity_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_parse_selected_sample_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_is_mode_switch_message", lambda t: False, raising=False)
    user = "deferred-user-1"
    _profile(db, user)
    up = _do_upload(client, user_id=user)
    thread_id, dsid = up["thread_id"], up["dataset_id"]

    class SlowGraph(_RecordingGraph):
        def invoke(self, state_in, config=None):
            time.sleep(0.4)
            return super().invoke(state_in, config)

    monkeypatch.setattr(server, "GRAPH", SlowGraph(), raising=False)
    body = _post(client, thread_id, dsid, user, "count the rows per region").json()
    assert body["pending_status"] == "running"
    deferred_id = body["analysis_task_id"]
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        pr = client.get(f"/threads/{thread_id}/pending-turn?deferred_id={deferred_id}", headers=make_auth_headers(user)).json()
        if pr["status"] == "done":
            break
        time.sleep(0.05)
    assert pr["status"] == "done"
    row = db.jobs[0]
    assert (row["id"], row["kind"], row["status"]) == (deferred_id, "analysis", "succeeded")
    assert len(db.notifications) == 1
    n = db.notifications[0]
    assert (n["type"], n["title"]) == ("analysis.completed", "Analysis completed")
    assert n["body"].startswith("“count the rows per region”") and n["metadata"]["ref"] == f"job:{deferred_id}"
    assert pr["result"]["job_notes"][0]["job_ids"] == [deferred_id]
