"""With no MLflow configuration, training and the model endpoints still work.

They defaulted to ``mlruns`` -- MLflow's file store, which the pinned MLflow
refuses ("the filesystem tracking backend is in maintenance mode"). A local
API run without MLFLOW_TRACKING_URI failed every training turn and returned
500 from GET /api/models.
"""
from __future__ import annotations

import mlflow
import pytest

from app.agents.mta_v2.tracking import resolve_tracking_uri

_VARS = ("MLFLOW_TRACKING_URI", "MLFLOW_BACKEND_STORE_URI")


@pytest.fixture
def no_mlflow_env(monkeypatch, tmp_path):
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    mlflow.set_tracking_uri(None)


def test_the_default_is_a_local_sqlite_store(no_mlflow_env) -> None:
    assert resolve_tracking_uri() == f"sqlite:///{(no_mlflow_env / 'mlflow.db').resolve()}"


def test_configuration_still_wins(monkeypatch, no_mlflow_env) -> None:
    monkeypatch.setenv("MLFLOW_BACKEND_STORE_URI", "postgresql://db/mlflow")
    assert resolve_tracking_uri() == "postgresql://db/mlflow"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert resolve_tracking_uri() == "http://mlflow:5000"
    assert resolve_tracking_uri("http://explicit") == "http://explicit"


def test_the_model_endpoints_can_read_with_no_configuration(no_mlflow_env) -> None:
    from app.agents.mta_v2.mlflow_manager import MLflowManager

    manager = MLflowManager()          # raised MlflowException on the file store
    assert mlflow.get_tracking_uri().startswith("sqlite:///")
    assert not (no_mlflow_env / "mlruns").exists()
    assert manager is not None


def test_the_trainer_writes_where_the_endpoints_read(no_mlflow_env) -> None:
    from app.agents.mta_v2.local_trainer import setup_mlflow

    setup_mlflow()
    with mlflow.start_run() as run:   # the write the file store refused
        mlflow.log_metric("accuracy", 0.5)
    assert (no_mlflow_env / "mlflow.db").exists()
    assert mlflow.get_run(run.info.run_id).data.metrics["accuracy"] == 0.5
