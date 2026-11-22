"""The chat turn never trains: the MTA node enqueues and returns.

On develop-1.6 @ 9cd9e122 the local backend trained inside the graph turn
(agent.py `_execute_training`, the `if use_local:` branch), so these fail
there by assertion (A). The worker path (training_scheduled=True) is what
actually trains, and is covered too.
"""
from __future__ import annotations

import os
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import HumanMessage

from app.agents.model_training_agent import model_training_agent_node

# Imported lazily so that on a tree without the feature (the base) the
# chat-turn tests still run and fail by assertion instead of erroring at
# import; the store-dependent assertions then fail on `store is None`.
try:
    from app.core import ml_store
except ImportError:  # pragma: no cover - base tree only
    ml_store = None


@pytest.fixture
def store():
    if ml_store is None:
        yield None
        return
    s = ml_store.InMemoryMLStore()
    ml_store.set_ml_store(s)
    yield s
    ml_store.set_ml_store(None)


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
    df["target"] = (X[:, 0] + X[:, 1] > 0).astype(int)
    path = os.path.join(str(tmp_path), "train.csv")
    df.to_csv(path, index=False)
    return path


def _plan(csv_path):
    return {
        "model_type": "classification",
        "model_name": "enqueue_clf",
        "model_description": "",
        "model_version": "1",
        "hyperparameter_config": {
            "learning_rate": 0.01, "epochs": 2, "batch_size": 16, "optimizer_name": "adam",
            "random_seed": 42, "early_stopping_patience": 2, "hidden_layer_sizes": [8],
            "activation": "relu", "dropout_rate": 0.0, "batch_norm": False,
        },
        "ray_config": None,
        "data_config": {
            "dataset_uri": csv_path,
            "feature_columns": ["feature1", "feature2", "feature3"],
            "target_column": "target",
        },
    }


def _state(csv_path, **overrides):
    state = {
        "user_id": "user-1",
        "session_id": "session-1",
        "messages": [HumanMessage(content="yes, train it")],
        "training_plan": _plan(csv_path),
        "training_result": None,
        "training_scheduled": False,
        "enable_training": True,
        "skip_to_training": True,
        "training_task": None,
        "data_source_location_local": csv_path,
    }
    state.update(overrides)
    return state


class _MustNotTrain:
    """Stands in for LocalTrainer: training inside the chat turn is the bug."""

    def __init__(self, *a, **k):
        pass

    def train(self, plan):
        raise AssertionError("LocalTrainer.train was called inside the chat turn")


# T-02 -----------------------------------------------------------------------

def test_local_plan_is_enqueued_not_trained_in_the_turn(csv_path, store):
    state = _state(csv_path)
    with patch("app.agents.mta_v2.agent.llm", None), \
         patch("app.agents.mta_v2.agent.LocalTrainer", _MustNotTrain):
        out = model_training_agent_node(state)

    assert out.get("training_result") is None
    assert out["training_scheduled"] is True
    assert (out.get("task_schedule") or {}).get("task_type") == "training"
    assert out["training_plan"]["execution_backend"] == "local"


def test_turn_records_a_queued_training_job(csv_path, store):
    state = _state(csv_path)
    with patch("app.agents.mta_v2.agent.llm", None), \
         patch("app.agents.mta_v2.agent.LocalTrainer", _MustNotTrain):
        out = model_training_agent_node(state)

    job_id = out.get("ml_job_id")
    assert job_id, "the turn must hand back a job handle"
    assert store is not None, "app.core.ml_store is missing"
    job = store.get_job(job_id)
    assert job is not None
    assert (job.kind, job.status, job.user_id) == ("training", "queued", "user-1")


# T-03 -----------------------------------------------------------------------

def test_turn_returns_fast_even_when_training_is_slow(csv_path, store):
    class _Slow:
        def __init__(self, *a, **k):
            pass

        def train(self, plan):
            time.sleep(3)
            return {"status": "ok", "mlflow_run_id": "r"}

    state = _state(csv_path)
    t0 = time.monotonic()
    with patch("app.agents.mta_v2.agent.llm", None), \
         patch("app.agents.mta_v2.agent.LocalTrainer", _Slow):
        model_training_agent_node(state)
    assert time.monotonic() - t0 < 0.5, "the chat turn waited for training"


# The worker path -------------------------------------------------------------

def test_worker_pass_trains_and_closes_the_job(csv_path, store, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", os.path.join(os.path.dirname(csv_path), "mlruns"))
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-trainfer-worker-pass")

    assert store is not None, "app.core.ml_store is missing"
    job = store.create_job(ml_store.MLJob(user_id="user-1", kind="training"))
    state = _state(csv_path, training_scheduled=True, ml_job_id=job.id)
    with patch("app.agents.mta_v2.agent.llm", None):
        out = model_training_agent_node(state)

    assert out["training_result"].get("status") != "error", out["training_result"].get("error")
    assert out["training_completed"] is True
    closed = store.get_job(job.id)
    assert closed.status == "succeeded"
    assert closed.mlflow_run_id == out["training_result"]["mlflow_run_id"]
    assert closed.started_at is not None and closed.finished_at is not None


def test_worker_pass_records_a_failed_job_with_a_one_line_error(csv_path, store):
    class _Fails:
        def __init__(self, *a, **k):
            pass

        def train(self, plan):
            return {"status": "error", "error": "boom\nstack line 2 with column names",
                    "failure": {"code": "x", "title": "Training failed: data problem"}}

    assert store is not None, "app.core.ml_store is missing"
    job = store.create_job(ml_store.MLJob(user_id="user-1", kind="training"))
    state = _state(csv_path, training_scheduled=True, ml_job_id=job.id)
    with patch("app.agents.mta_v2.agent.llm", None), \
         patch("app.agents.mta_v2.agent.LocalTrainer", _Fails):
        out = model_training_agent_node(state)

    assert out["training_completed"] is False
    closed = store.get_job(job.id)
    assert closed.status == "failed"
    assert closed.error == "Training failed: data problem"
    assert "\n" not in closed.error


def test_training_proceeds_when_the_store_is_down(csv_path, monkeypatch):
    class _Down:
        def create_job(self, job):
            raise RuntimeError("postgres down")

    assert ml_store is not None, "app.core.ml_store is missing"
    ml_store.set_ml_store(_Down())
    try:
        state = _state(csv_path)
        with patch("app.agents.mta_v2.agent.llm", None), \
             patch("app.agents.mta_v2.agent.LocalTrainer", _MustNotTrain):
            out = model_training_agent_node(state)
        assert out["training_scheduled"] is True
        assert out.get("ml_job_id") is None
    finally:
        ml_store.set_ml_store(None)


# Dev mode: no worker, inline flag on -----------------------------------------

def test_dev_mode_runs_training_on_a_background_thread_not_in_the_turn(monkeypatch):
    from app.agents import scheduler

    monkeypatch.setattr(scheduler, "is_celery_worker_running", lambda: False)
    monkeypatch.setattr(scheduler, "_RUN_INLINE_WITHOUT_WORKER", True)
    started = {}

    def fake_bg(state):
        started["state"] = state

    monkeypatch.setattr(scheduler, "_run_training_in_background", fake_bg)
    state = {
        "messages": [HumanMessage(content="train")],
        "task_schedule": {"task_type": "training", "schedule_type": "relative", "second": 0, "max_runs": 1},
        "task_operation": None,
    }
    out = scheduler.task_scheduler_node(state)
    assert "state" in started, "training must be handed to the background runner"
    assert "background" in out["messages"][-1].content.lower()
