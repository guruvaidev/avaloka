"""Deferred (slow) chat turns: the pending-turn endpoint, fast vs slow turns,
the training/analysis wait message, and the training-turn classifier.

Moved out of test_server_integration.py, which the CI gate deselects
(`-m "not integration"`), so these run on every pipeline.
"""
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app.api.server as server

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
    SESSIONS, THREAD_TO_SESSION, FakeGraph, make_auth_headers, client, shared_loop_client, _do_upload, _quiet_send,
    )
except ImportError:
    from server_harness import (  # noqa: F401
    SESSIONS, THREAD_TO_SESSION, FakeGraph, make_auth_headers, client, shared_loop_client, _do_upload, _quiet_send,
    )


# Slow-turn polling budget. Generous on purpose: the test only needs ~0.4 s,
# but must not flake on a heavily loaded CI machine.
POLL_TIMEOUT_S = 10.0


# ======================================================================================
# Deferred turn: /threads/{thread_id}/pending-turn
# ======================================================================================


def test_pending_turn_requires_auth(client: TestClient):
    resp = client.get("/threads/some-thread/pending-turn")
    assert resp.status_code == 401


def test_pending_turn_unknown_thread_404(client: TestClient):
    resp = client.get(
        "/threads/no-such-thread/pending-turn",
        headers=make_auth_headers("pt-user"),
    )
    assert resp.status_code == 404


def test_pending_turn_foreign_user_404(client: TestClient):
    upload = _do_upload(client, user_id="pt-owner")
    thread_id = upload["thread_id"]
    resp = client.get(
        f"/threads/{thread_id}/pending-turn",
        headers=make_auth_headers("pt-other"),
    )
    assert resp.status_code == 404


def test_pending_turn_none_when_absent(client: TestClient):
    upload = _do_upload(client, user_id="pt-none")
    thread_id = upload["thread_id"]
    resp = client.get(
        f"/threads/{thread_id}/pending-turn",
        headers=make_auth_headers("pt-none"),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "none"


def test_pending_turn_done_returns_result(client: TestClient):
    upload = _do_upload(client, user_id="pt-done")
    thread_id = upload["thread_id"]
    sid = THREAD_TO_SESSION[thread_id]
    SESSIONS[sid]["pending_turn"] = {
        "id": "d1", "status": "done", "result": {"foo": "bar"},
    }
    resp = client.get(
        f"/threads/{thread_id}/pending-turn",
        headers=make_auth_headers("pt-done"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["result"] == {"foo": "bar"}


def test_pending_turn_error_status(client: TestClient):
    upload = _do_upload(client, user_id="pt-err")
    thread_id = upload["thread_id"]
    sid = THREAD_TO_SESSION[thread_id]
    SESSIONS[sid]["pending_turn"] = {
        "id": "e1", "status": "error", "message": "Training failed.",
    }
    resp = client.get(
        f"/threads/{thread_id}/pending-turn",
        headers=make_auth_headers("pt-err"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["message"] == "Training failed."


def test_pending_turn_superseded_when_id_mismatch(client: TestClient):
    upload = _do_upload(client, user_id="pt-sup")
    thread_id = upload["thread_id"]
    sid = THREAD_TO_SESSION[thread_id]
    SESSIONS[sid]["pending_turn"] = {"id": "current", "status": "running"}
    resp = client.get(
        f"/threads/{thread_id}/pending-turn?deferred_id=old",
        headers=make_auth_headers("pt-sup"),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "superseded"


# ======================================================================================
# send_message: fast turn vs slow (deferred) turn
# ======================================================================================


def test_send_message_fast_turn_leaves_no_pending_handle(client: TestClient):
    """A turn that finishes within the deadline must not write a running handle."""
    upload = _do_upload(client, user_id="fast-user")
    thread_id = upload["thread_id"]
    dsid = upload["dataset_id"]
    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"content": "hello", "metadata": {"dataset_id": dsid}},
        headers=make_auth_headers("fast-user"),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("analysis_fidelity")
    assert body.get("execution_context")
    sid = THREAD_TO_SESSION[thread_id]
    assert "pending_turn" not in (SESSIONS.get(sid) or {})


def test_send_message_slow_turn_returns_running_then_done(shared_loop_client: TestClient, monkeypatch):
    """A turn that overruns the deadline returns a running handle and the
    background task later stashes the finished result for polling."""
    client = shared_loop_client
    monkeypatch.setattr(server, "TURN_SYNC_DEADLINE_S", 0.05, raising=False)
    monkeypatch.setattr(server, "_parse_fidelity_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_parse_selected_sample_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_is_mode_switch_message", lambda t: False, raising=False)

    class SlowGraph:
        checkpointer = object()

        def invoke(self, state_in, config=None):
            time.sleep(0.4)  # overrun the 0.05s deadline reliably
            last = state_in["messages"][-1]
            content = getattr(last, "content", "")
            return {
                "messages": state_in["messages"] + [AIMessage(content=f"slow: {content}")],
                "planner_definition": {},
                "ready_to_summarize": False,
                "ready_to_code": False,
                "coder_definition": {},
                "visualization_config": {},
                "visualization_status": "",
                "execution_result": {},
            }

    monkeypatch.setattr(server, "GRAPH", SlowGraph(), raising=False)

    upload = _do_upload(client, user_id="slow-user")
    thread_id = upload["thread_id"]
    dsid = upload["dataset_id"]

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"content": "train a model", "metadata": {"dataset_id": dsid}},
        headers=make_auth_headers("slow-user"),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("training_status") == "running"
    deferred_id = body.get("analysis_task_id")
    assert deferred_id

    # The background task finishes ~0.4s later and stashes the result.
    # Poll for up to POLL_TIMEOUT_S: the turn needs ~0.4 s, but a loaded CI
    # machine can be much slower (2.5 s once failed under load average 111).
    final_status = None
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        pr = client.get(
            f"/threads/{thread_id}/pending-turn?deferred_id={deferred_id}",
            headers=make_auth_headers("slow-user"),
        )
        assert pr.status_code == 200
        final_status = pr.json()["status"]
        if final_status == "done":
            break
        time.sleep(0.05)
    assert final_status == "done"


# ======================================================================================
# Deferred-turn wait message: analysis vs training
# ======================================================================================


def test_a_slow_analysis_turn_says_analysis_not_training(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "TURN_SYNC_DEADLINE_S", 0.05, raising=False)
    _quiet_send(monkeypatch)

    class SlowGraph(FakeGraph):
        def invoke(self, state_in, config=None):
            time.sleep(0.3)
            return super().invoke(state_in, config)

    monkeypatch.setattr(server, "GRAPH", SlowGraph(), raising=False)
    upload = _do_upload(client, user_id="slow-analysis")
    resp = client.post(
        f"/threads/{upload['thread_id']}/messages",
        json={"content": "average of b by a", "metadata": {"dataset_id": upload["dataset_id"]}},
        headers=make_auth_headers("slow-analysis"),
    )
    assert resp.status_code == 200, resp.text
    text = [m for m in resp.json()["messages"] if m["role"] == "assistant"][-1]["content"]
    assert "taking longer than usual" in text
    assert "training" not in text.lower()
    assert resp.json()["analysis_task_id"]  # still pollable


def test_a_slow_training_turn_keeps_the_training_message(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "TURN_SYNC_DEADLINE_S", 0.05, raising=False)
    _quiet_send(monkeypatch)

    class SlowGraph(FakeGraph):
        def invoke(self, state_in, config=None):
            time.sleep(0.3)
            return super().invoke(state_in, config)

    monkeypatch.setattr(server, "GRAPH", SlowGraph(), raising=False)
    upload = _do_upload(client, user_id="slow-train")
    resp = client.post(
        f"/threads/{upload['thread_id']}/messages",
        json={"content": "train a model to predict b", "metadata": {"dataset_id": upload["dataset_id"]}},
        headers=make_auth_headers("slow-train"),
    )
    text = [m for m in resp.json()["messages"] if m["role"] == "assistant"][-1]["content"]
    assert "Model training is running" in text


# ======================================================================================
# Deferred-turn pure helpers
# ======================================================================================


def test_pending_turn_state_helpers():
    running = {}
    server._pending_turn_running("d1")(running)
    assert running["pending_turn"]["status"] == "running"
    assert running["pending_turn"]["id"] == "d1"

    # A finished turn must never be downgraded back to running.
    done = {}
    server._pending_turn_done("d1", {"foo": "bar"})(done)
    server._pending_turn_running("d1")(done)
    assert done["pending_turn"]["status"] == "done"
    assert done["pending_turn"]["result"] == {"foo": "bar"}

    err = {}
    server._pending_turn_error("d2", "Training failed.")(err)
    assert err["pending_turn"]["status"] == "error"
    assert err["pending_turn"]["message"] == "Training failed."


# ======================================================================================
# Training-turn classifier (server._TRAINING_TURN_RE)
# ======================================================================================


@pytest.mark.parametrize("text", [
    "train a model on churn",
    "Train a model to predict b",
    "retrain the classification model",
    "re-train my model",
    "training a model on last year's data",
    "train the random forest model",
    "fit a model",
    "build the model",
    "start training",
    "Start Training",
    "how long does model training take",
])
def test_training_requests_are_detected(text):
    assert server._TRAINING_TURN_RE.search(text), text


@pytest.mark.parametrize("text", [
    "which train line is busiest",
    "average delay by train",
    "count trains per station",
    "show the training data split",
    "use the trained model to score this file",
    "average of b by a",
    "",
])
def test_ordinary_questions_are_not_training(text):
    assert not server._TRAINING_TURN_RE.search(text), text