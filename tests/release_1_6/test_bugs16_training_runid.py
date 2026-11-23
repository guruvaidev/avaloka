"""BG11: the run ID is announced on the turn training finishes, not on every later turn.

Bugs-1.6 BG11 ("RunID was attached to all messages after model training
completes") is recorded Closed. The UI adds a "Training completed / Run ID"
message whenever a reply carries ``training_completed`` (ui/src/routes/analysis.tsx),
and send_message builds that field from ``final.get("training_completed")``
(app/api/server.py, step 19). The session keeps ``training_completed`` and
``mlflow_run_id`` across turns on purpose, so a later inference request can find
the model, and hydrates them into the next turn's graph state (step 10). LangGraph
returns every input channel in the final state unless a node overwrites it, and
only the planner's cancel/modify branches clear them. So every unrelated turn
after a training run comes back with ``training_completed=True`` and the old run
ID, and the UI announces the run again.

These tests drive the real handler through TestClient with fake graphs (fixtures
from tests/test_server_unit.py). No LLM, no store, no network.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

import app.api.server as server
from tests.test_server_unit import (  # noqa: F401  (client is a fixture)
    FakeGraph,
    _do_upload,
    _quiet_send,
    client,
    make_auth_headers,
)

RUN_ID = "run-0000-bg11"

#: A complete app.agents.mta_v2.schema.TrainingResult (the response model
#: validates it; a partial dict fails validation and the turn errors out).
TRAINING_RESULT = {
    "task_id": "task-bg11",
    "mlflow_run_id": RUN_ID,
    "dashboard_url": None,
    "user_id": "bg11-user",
    "session_id": "session-bg11",
    "model_type": "classification",
    "model_name": "fare_model",
    "model_description": "synthetic",
    "model_version": "1",
    "runtime_s": 1.0,
    "rows_processed": 10,
    "num_epochs_trained": 1,
    "final_train_loss": 0.5,
    "final_val_loss": 0.6,
    "num_features": 3,
    "num_classes": 2,
    "created_at": "2026-10-31T00:00:00Z",
}


class _TrainingTurn(FakeGraph):
    """The turn on which training finishes: the MTA node reports a run."""

    def invoke(self, state_in: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = super().invoke(state_in, config)
        out.update(
            training_completed=True,
            mlflow_run_id=RUN_ID,
            training_result=dict(TRAINING_RESULT),
        )
        return out


class _PassThroughTurn(FakeGraph):
    """An unrelated later turn. Like LangGraph, every input channel the nodes do
    not overwrite comes back unchanged in the final state."""

    def invoke(self, state_in: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = dict(state_in)
        out.update(super().invoke(state_in, config))
        return out


@pytest.fixture()
def turns(client, monkeypatch):
    _quiet_send(monkeypatch)
    user = "bg11-user"
    upload = _do_upload(client, user_id=user)

    def send(graph, content: str) -> Dict[str, Any]:
        monkeypatch.setattr(server, "GRAPH", graph, raising=False)
        resp = client.post(
            f"/threads/{upload['thread_id']}/messages",
            json={"content": content, "metadata": {"dataset_id": upload["dataset_id"]}},
            headers=make_auth_headers(user),
        )
        assert resp.status_code == 200, resp.text
        return resp.json()

    return send


def test_bg11_training_turn_reports_its_run_id(turns):
    """BG11: the turn on which training finishes carries training_completed and its run ID.

    The positive half: without it, the xfail below would also "pass" if the API
    never reported training at all.
    """
    body = turns(_TrainingTurn(), "train a model to predict fare")

    assert body["training_completed"] is True
    assert body["mlflow_run_id"] == RUN_ID


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "BG11: send_message reports final['training_completed'] / mlflow_run_id, which the "
        "session re-hydrates into every later turn, so the UI re-announces the run each turn"
    ),
)
def test_bg11_later_unrelated_turn_does_not_reannounce_the_run(turns):
    """BG11: after training, an unrelated turn's reply carries no training_completed and no run ID."""
    first = turns(_TrainingTurn(), "train a model to predict fare")
    assert first["mlflow_run_id"] == RUN_ID, "the training turn under test did not register"

    later = turns(_PassThroughTurn(), "what is the average tip by hour")

    assert not later["training_completed"]
    assert later["mlflow_run_id"] is None
