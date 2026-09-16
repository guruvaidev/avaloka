"""Scheduled MTA failures retain diagnostics without exposing them to users."""

from app.agents.mta_v2.failure_diagnostics import build_training_failure
from app.core import celery_app as celery_module


class _UpdateQuery:
    def __init__(self, captured):
        self.captured = captured

    def update(self, payload):
        self.captured.update(payload)
        return self

    def eq(self, *_args):
        return self

    def execute(self):
        return None


class _Supabase:
    def __init__(self, captured):
        self.captured = captured

    def table(self, _name):
        return _UpdateQuery(self.captured)


def test_scheduled_training_persists_safe_actionable_message(monkeypatch):
    failure = build_training_failure(
        "Ray job failed",
        kubernetes={
            "reason": "OOMKilled",
            "pod_name": "ray-worker-1",
            "exit_code": 137,
        },
    )
    final = {
        "training_completed": False,
        "training_result": {
            "status": "error",
            "error": "Ray job failed",
            "failure": failure,
        },
    }
    captured = {}
    monkeypatch.setattr(celery_module, "_supabase", lambda: _Supabase(captured))

    celery_module._record_run_finish(
        "schedule-1",
        "execution-1",
        final,
        0.0,
        scheduled_state={"task_schedule": {"task_type": "training"}},
    )

    assert captured["status"] == "failure"
    assert final["training_result"]["failure"]["code"] == "OOM_KILLED"
    assert "failure" not in captured["result"]
    assert captured["result"]["logs"] is None
    assert "Model training could not be completed." in captured["result"]["message"]
    assert "OOM_KILLED" in captured["result"]["message"]
    assert "increase the training worker memory" in captured["result"]["message"].lower()
    assert "ray-worker-1" not in captured["result"]["message"]
    assert captured["error"] == captured["result"]["message"]


def test_ray_submission_is_attached_to_running_schedule(monkeypatch):
    captured = {}
    monkeypatch.setattr(celery_module, "_supabase", lambda: _Supabase(captured))

    celery_module._record_ray_job_submission(
        "schedule-1",
        "execution-1",
        {
            "job_id": "ray-job-123",
            "status": "SUBMITTED",
            "dashboard_url": "http://ray-head:8265",
            "namespace": "default",
        },
    )

    assert captured["status"] == "running"
    assert captured["result"]["ray_job"] == {
        "job_id": "ray-job-123",
        "status": "SUBMITTED",
        "dashboard_url": "http://ray-head:8265",
        "namespace": "default",
    }


def test_completed_scheduled_training_retains_ray_identity(monkeypatch):
    captured = {}
    monkeypatch.setattr(celery_module, "_supabase", lambda: _Supabase(captured))
    final = {
        "training_completed": True,
        "training_result": {
            "status": "success",
            "mlflow_run_id": "mlflow-1",
            "ray_job_id": "ray-job-123",
            "ray_job_status": "SUCCEEDED",
            "ray_namespace": "default",
            "dashboard_url": "http://ray-head:8265",
        },
    }

    celery_module._record_run_finish(
        "schedule-1",
        "execution-1",
        final,
        0.0,
        scheduled_state={"task_schedule": {"task_type": "training"}},
    )

    assert captured["status"] == "success"
    assert captured["result"]["ray_job"] == {
        "job_id": "ray-job-123",
        "status": "SUCCEEDED",
        "dashboard_url": "http://ray-head:8265",
        "namespace": "default",
    }


def test_celery_success_does_not_hide_nested_training_failure():
    final = {
        "training_result": {
            "status": "error",
            "execution_error": "ImagePullBackOff: manifest unknown",
        }
    }

    status, message = celery_module._run_status_from_final(final, task_type="training")

    assert status == "failure"
    assert "Model training could not be completed." in message
    assert "IMAGE_PULL_FAILED" in message


def test_uncaught_scheduled_training_failure_hides_exception(monkeypatch):
    captured = {}
    monkeypatch.setattr(celery_module, "_supabase", lambda: _Supabase(captured))

    celery_module._record_run_failure(
        "schedule-1",
        "execution-1",
        RuntimeError("secret Ray traceback"),
        0.0,
        scheduled_state={"task_schedule": {"task_type": "training"}},
    )

    assert "Model training could not be completed." in captured["error"]
    assert "TRAINING_FAILED" in captured["error"]
    assert captured["result"]["message"] == captured["error"]
    assert captured["result"]["logs"] is None
    assert "secret Ray traceback" not in str(captured)
