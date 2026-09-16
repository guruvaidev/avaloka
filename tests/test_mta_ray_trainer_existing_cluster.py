"""Focused tests for MTA training on an existing Kubernetes RayCluster."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agents.mta_v2.gcp.submit_ray_job import submit_ray_job
from app.agents.mta_v2.ray_trainer import RayTrainer
from app.core.scheduled_run_context import scheduled_run_context


def _training_plan() -> dict:
    return {
        "model_type": "classification",
        "model_name": "shared_ray_test",
        "model_description": "Existing RayCluster test",
        "model_version": "v1.0",
        "data_config": {
            "dataset_uri": "gs://example-bucket/train.csv",
            "feature_columns": ["age", "income"],
            "target_column": "churn",
        },
        "hyperparameter_config": {
            "learning_rate": 0.001,
            "epochs": 3,
            "batch_size": 16,
            "optimizer_name": "adam",
            "random_seed": 42,
            "early_stopping_patience": 2,
            "hidden_layer_sizes": [32, 16],
            "activation": "relu",
            "dropout_rate": 0.1,
            "batch_norm": True,
        },
        "ray_config": {
            "num_workers": 2,
            "cpu_per_worker": 1,
            "memory_per_worker": "2Gi",
            "use_gpu": False,
        },
    }


def _successful_result() -> dict:
    return {
        "task_id": "ray-job-task",
        "mlflow_run_id": "mlflow-run",
        "user_id": "user-1",
        "session_id": "session-1",
        "model_type": "classification",
        "model_name": "shared_ray_test",
        "model_description": "Existing RayCluster test",
        "model_version": "v1.0",
        "runtime_s": 1.5,
        "rows_processed": 20,
        "num_epochs_trained": 3,
        "final_train_loss": 0.2,
        "final_val_loss": 0.3,
        "final_accuracy": 0.9,
        "final_f1_score": 0.9,
        "num_features": 2,
        "num_classes": 2,
        "created_at": "2026-09-05T00:00:00Z",
        "status": "success",
    }


def test_existing_cluster_submits_without_provisioning_or_gcp_key(monkeypatch):
    monkeypatch.setenv(
        "MTA_RAY_DASHBOARD_URL",
        "http://avaloka-raycluster-head-svc:8265/",
    )
    monkeypatch.setenv("MTA_RAY_NAMESPACE", "avaloka-test")
    monkeypatch.setenv("MTA_RAY_TRAINING_ENTRYPOINT", "python /opt/mta/ray_job.py")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://avaloka-mlflow:5000")

    trainer = RayTrainer(user_id="user-1", session_id="session-1")
    submitted = []

    with (
        patch.object(trainer, "_read_gcp_sa_json") as read_gcp_key,
        patch("app.agents.mta_v2.ray_trainer.get_k8s_api") as get_k8s_api,
        patch("app.agents.mta_v2.ray_trainer.create_namespace") as create_namespace,
        patch("app.agents.mta_v2.ray_trainer.create_ray_cluster") as create_cluster,
        patch("app.agents.mta_v2.ray_trainer.submit_ray_job", return_value="job-123") as submit,
        patch(
            "app.agents.mta_v2.ray_trainer.wait_for_ray_job_completion",
            return_value="SUCCEEDED",
        ),
        patch(
            "app.agents.mta_v2.ray_trainer.get_job_result",
            return_value=_successful_result(),
        ),
        patch.object(trainer, "_schedule_cluster_deletion") as delete_cluster,
    ):
        with scheduled_run_context(
            "schedule-1",
            "execution-1",
            ray_job_callback=lambda schedule_id, execution_id, metadata: submitted.append(
                (schedule_id, execution_id, dict(metadata))
            ),
        ):
            result = trainer.train(_training_plan())

    read_gcp_key.assert_not_called()
    get_k8s_api.assert_not_called()
    create_namespace.assert_not_called()
    create_cluster.assert_not_called()
    delete_cluster.assert_not_called()

    dashboard_url, job_env = submit.call_args.args
    assert dashboard_url == "http://avaloka-raycluster-head-svc:8265"
    assert submit.call_args.kwargs["entry_point"] == "python /opt/mta/ray_job.py"
    assert job_env["DATA_SOURCE_URI"] == "gs://example-bucket/train.csv"
    assert job_env["MLFLOW_TRACKING_URI"] == "http://avaloka-mlflow:5000"
    assert "GCP_SERVICE_ACCOUNT_JSON" not in job_env
    assert result["dashboard_url"] == dashboard_url
    assert result["ray_job_id"] == "job-123"
    assert result["ray_job_status"] == "SUCCEEDED"
    assert result["ray_namespace"] == "avaloka-test"
    assert submitted == [(
        "schedule-1",
        "execution-1",
        {
            "job_id": "job-123",
            "status": "SUBMITTED",
            "dashboard_url": "http://avaloka-raycluster-head-svc:8265",
            "namespace": "avaloka-test",
        },
    )]
    assert trainer.namespace == "avaloka-test"


def test_legacy_job_environment_still_forwards_explicit_gcp_key(monkeypatch):
    monkeypatch.delenv("MTA_RAY_DASHBOARD_URL", raising=False)
    trainer = RayTrainer(user_id="user-1", session_id="session-1")

    job_env = trainer._build_job_env(_training_plan(), gcp_sa_json='{"type":"service_account"}')

    assert job_env["GCP_SERVICE_ACCOUNT_JSON"] == '{"type":"service_account"}'
    assert job_env["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/gcp_sa.json"
    assert trainer._existing_dashboard_url() == ""


@pytest.mark.parametrize(
    "response_payload, expected",
    [
        ({"submission_id": "ray-253-job"}, "ray-253-job"),
        ({"job_id": "legacy-ray-job"}, "legacy-ray-job"),
    ],
)
def test_submit_ray_job_accepts_current_and_legacy_identifiers(
    response_payload,
    expected,
):
    with patch("app.agents.mta_v2.gcp.submit_ray_job.requests.post") as post:
        post.return_value.json.return_value = response_payload
        post.return_value.raise_for_status.return_value = None

        result = submit_ray_job(
            "http://avaloka-raycluster-head-svc:8265",
            {"TASK_ID": "task-1"},
        )

    assert result == expected
