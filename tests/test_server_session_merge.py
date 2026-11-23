"""Keys a turn clears must stay cleared in the stored session.

send_message saves the session by merging: ``cur.update(primary_sess)``.
``update()`` adds and overwrites but never deletes, so a key the turn removed
from ``primary_sess`` survives in the stored copy. connection_event got an
explicit drop when its wiring was ported; pending_clarification did not, so a
transfer slot left open by a failed transfer read every later question on the
thread as a table name ("⚠️ Table What doesn't exist", rc3 web re-check).
"""
from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

import app.api.server as server

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
        SESSIONS, FakeGraph, client, make_auth_headers, _do_upload, _quiet_send,
    )
except ImportError:
    from server_harness import (  # noqa: F401
        SESSIONS, FakeGraph, client, make_auth_headers, _do_upload, _quiet_send,
    )


class _ResolvesClarification(FakeGraph):
    """Plays a planner that has just had its clarification answered."""

    def invoke(self, state_in, config=None):
        final = super().invoke(state_in, config)
        final["pending_clarification"] = None
        return final


def test_a_resolved_clarification_is_cleared_from_the_stored_session(
    client: TestClient, monkeypatch
):
    _quiet_send(monkeypatch)
    user = "merge-clar"
    upload = _do_upload(client, user_id=user)
    sid = upload["session_id"]
    SESSIONS[sid]["pending_clarification"] = '{"field": "region", "question": "Which region?"}'

    monkeypatch.setattr(server, "GRAPH", _ResolvesClarification(), raising=False)
    resp = client.post(
        f"/threads/{upload['thread_id']}/messages",
        json={"content": "the west region", "metadata": {"dataset_id": upload["dataset_id"]}},
        headers=make_auth_headers(user),
    )
    assert resp.status_code == 200, resp.text
    assert "pending_clarification" not in SESSIONS[sid]


class _FailedTransferThenQuestion(FakeGraph):
    """Turn 1 plays a transfer that failed and cleared its slot; turn 2 records
    the clarification the planner was handed with the next question."""

    def __init__(self):
        self.seen = []

    def invoke(self, state_in, config=None):
        self.seen.append(state_in.get("pending_clarification"))
        final = super().invoke(state_in, config)
        final["pending_clarification"] = None
        return final


def test_a_failed_transfer_does_not_hand_its_slot_to_the_next_question(
    client: TestClient, monkeypatch
):
    _quiet_send(monkeypatch)
    user = "merge-dta"
    upload = _do_upload(client, user_id=user)
    sid = upload["session_id"]
    SESSIONS[sid]["pending_clarification"] = {
        "type": "dta_transfer", "awaiting": "table_choice", "source_agent": "dta",
        "collected": {"destination_alias": "e2ewhgemma", "user_prompt": "Transfer the patients table"},
    }
    graph = _FailedTransferThenQuestion()
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)
    for content in ("new table patients_copy",
                    "What is the stroke rate for patients with hypertension versus without?"):
        resp = client.post(
            f"/threads/{upload['thread_id']}/messages",
            json={"content": content, "metadata": {"dataset_id": upload["dataset_id"]}},
            headers=make_auth_headers(user),
        )
        assert resp.status_code == 200, resp.text
    assert graph.seen[0] and graph.seen[0].get("awaiting") == "table_choice"
    assert graph.seen[1] is None, "the next question was handed the failed transfer's slot"
    assert "pending_clarification" not in SESSIONS[sid]


# ------------------------------------------------- cleared training state
_TRAINED = {
    "training_plan": '{"algorithm": "gradient_boosting", "model_type": "classification"}',
    "training_result": '{"status": "success", "mlflow_run_id": "run-1"}',
    "training_task": '{"task_id": "t-1"}',
    "ready_to_train": True,
    "training_completed": True,
    "mlflow_run_id": "run-1",
    "enable_training": True,
}


class _ClearsTraining(FakeGraph):
    """Turn 1 clears the training state; later turns record what they were handed."""

    def __init__(self):
        self.seen = []

    def invoke(self, state_in, config=None):
        self.seen.append({k: state_in.get(k) for k in ("training_plan", "training_result", "enable_training")})
        final = super().invoke(state_in, config)
        if len(self.seen) == 1:
            final["clear_training_state"] = True
        return final


def test_cleared_training_state_stays_cleared_after_a_save_and_load(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    user = "merge-train"
    upload = _do_upload(client, user_id=user)
    sid = upload["session_id"]
    SESSIONS[sid].update(_TRAINED)
    graph = _ClearsTraining()
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)
    for content in ("start over without the model", "what is the average units by region?"):
        resp = client.post(
            f"/threads/{upload['thread_id']}/messages",
            json={"content": content, "metadata": {"dataset_id": upload["dataset_id"]}},
            headers=make_auth_headers(user),
        )
        assert resp.status_code == 200, resp.text
    stored = SESSIONS[sid]
    for key in ("training_plan", "training_result", "training_task", "ready_to_train",
                "training_completed", "mlflow_run_id"):
        assert key not in stored, f"{key} survived clear_training_state"
    assert stored.get("enable_training") is False
    assert graph.seen[0]["training_plan"], "turn 1 must start with a plan, or the test proves nothing"
    assert graph.seen[1]["training_plan"] is None, "the next turn was handed the cleared plan"


def test_reset_dataset_drops_the_transformed_source_keys(client: TestClient, monkeypatch):
    """The reset fast path restores the snapshot and pops keys it did not have;
    those pops reach the stored session too."""
    _quiet_send(monkeypatch)
    user = "merge-reset"
    upload = _do_upload(client, user_id=user)
    sid = upload["session_id"]
    original = SESSIONS[sid].get("active_data_source_location")
    SESSIONS[sid]["original_dataset_snapshot"] = json.dumps(
        {"active_data_source_location": original or "orig.csv"})
    SESSIONS[sid]["active_data_source_location_cloud"] = "gs://bucket/transformed.parquet"
    SESSIONS[sid]["data_source_was_modified"] = True
    resp = client.post(
        f"/threads/{upload['thread_id']}/messages",
        json={"content": "reset the dataset", "metadata": {"dataset_id": upload["dataset_id"]}},
        headers=make_auth_headers(user),
    )
    assert resp.status_code == 200, resp.text
    assert "Restored the original" in resp.text
    assert "active_data_source_location_cloud" not in SESSIONS[sid]
    assert "original_dataset_snapshot" not in SESSIONS[sid]


def test_the_merge_keeps_a_value_written_meanwhile():
    baseline = {"pending_clarification": "old", "a": 1}
    sess = {"a": 2}                                    # the turn dropped the clarification
    cur = {"pending_clarification": "new", "a": 1}     # another writer set a new one
    server._merge_session_with_drops(cur, sess, baseline)
    assert cur == {"pending_clarification": "new", "a": 2}
    cur = {"pending_clarification": "old", "a": 1}
    server._merge_session_with_drops(cur, sess, baseline)
    assert cur == {"a": 2}
