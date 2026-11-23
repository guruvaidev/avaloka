"""1.6 release suite: model-training agent (plan cases AG-11, AG-12).

AG-11 drives ModelTrainingAgent.execute with the tool-selecting model stubbed
and both trainers replaced by recorders. AG-12 trains for real with the local
trainer into an isolated MLflow store and lists the run through GET /api/models;
the Ray path needs a cluster and stays manual.
See docs/test-reports/1.6-traceability.csv.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

# The API client fixture (fake cache/blob store, test JWT secret).
from tests.server_harness import client, make_auth_headers  # noqa: F401


def _plan(dataset: str) -> dict:
    return {
        "model_type": "classification",
        "model_name": "release16-churn",
        "model_description": "Synthetic churn classifier",
        "model_version": "v1.0",
        "hyperparameter_config": {
            "learning_rate": 0.01, "epochs": 3, "batch_size": 16,
            "optimizer_name": "adam", "random_seed": 7, "early_stopping_patience": 2,
            "hidden_layer_sizes": [8], "activation": "relu", "dropout_rate": 0.0,
            "batch_norm": False,
        },
        "ray_config": None,
        "data_config": {
            "dataset_uri": dataset,
            "feature_columns": ["tenure", "spend"],
            "target_column": "churned",
        },
    }


@pytest.mark.parametrize("selected_tool", ["generate_training_plan", "execute_training"])
def test_ag11_first_training_request_returns_a_plan_and_starts_no_training(monkeypatch, selected_tool):
    """AG-11: the first training request returns the plan (task, target, features,
    epochs) and does NOT start training, even if the model asks to execute."""
    import app.agents.mta_v2.agent as mta

    tool_llm = MagicMock()
    tool_llm.invoke.return_value = MagicMock(tool_calls=[{"name": selected_tool}], content="")
    monkeypatch.setattr(mta, "llm", tool_llm)
    local_trainer, ray_trainer = MagicMock(name="LocalTrainer"), MagicMock(name="RayTrainer")
    monkeypatch.setattr(mta, "LocalTrainer", local_trainer)
    monkeypatch.setattr(mta, "RayTrainer", ray_trainer)

    agent = mta.ModelTrainingAgent()
    plan = _plan("/tmp/release16-churn.csv")
    monkeypatch.setattr(agent, "_extract_training_plan", lambda _state: plan)

    state = {
        "user_id": "user-1", "session_id": "session-1",
        "messages": [HumanMessage(content="Train a model to predict churned")],
        "training_plan": None, "training_result": None, "training_scheduled": False,
        "enable_training": True, "skip_to_training": False,
        "data_source_location": "/tmp/release16-churn.csv", "dataset_size_bytes": 1024,
    }
    result = agent.execute(state)

    assert result.get("training_plan") == plan
    assert not result.get("training_result")
    assert not result.get("training_scheduled")
    local_trainer.assert_not_called()
    ray_trainer.assert_not_called()
    reply = [m for m in result["messages"] if isinstance(m, AIMessage)][-1].content
    assert "Classification" in reply
    assert "`churned`" in reply
    assert "2 selected columns" in reply
    assert "3 epochs" in reply
    assert "Training has not started." in reply


@pytest.fixture()
def isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import mlflow

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_BACKEND_STORE_URI", "")
    monkeypatch.setenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", artifacts.as_uri())
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", f"release16-{tmp_path.name}")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.end_run()
    yield artifacts
    mlflow.end_run()


def test_ag12_local_training_run_is_listed_by_the_models_api(isolated_mlflow, client, tmp_path):
    """AG-12: a completed training run is listed by GET /api/models for its owner,
    with its metrics, and is not listed for another user."""
    from app.agents.mta_v2.local_trainer import LocalTrainer

    rng = np.random.default_rng(7)
    tenure = rng.integers(1, 60, size=120)
    spend = rng.normal(50, 15, size=120).round(2)
    data = pd.DataFrame({"tenure": tenure, "spend": spend, "churned": (tenure < 20).astype(int)})
    dataset = tmp_path / "churn.csv"
    data.to_csv(dataset, index=False)

    result = LocalTrainer(user_id="user-1", session_id="session-1").train(_plan(str(dataset)))
    assert "error" not in result, result
    run_id = result["mlflow_run_id"]

    resp = client.get("/api/models", headers=make_auth_headers("user-1"))
    assert resp.status_code == 200, resp.text
    listed = {m["mlflow_run_id"]: m for m in resp.json()["models"]}
    assert run_id in listed, sorted(listed)
    model = listed[run_id]
    assert model["model_name"] == "release16-churn"
    assert model["status"] == "FINISHED"
    assert model["final_accuracy"] is not None and 0.0 <= model["final_accuracy"] <= 1.0
    assert model["final_val_loss"] is not None

    other = client.get("/api/models", headers=make_auth_headers("user-2"))
    assert other.status_code == 200
    assert run_id not in {m["mlflow_run_id"] for m in other.json()["models"]}
