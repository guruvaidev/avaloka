"""PR-1c: training as a RayJob on the shared cluster, submit-only, polled by the worker.

Base classes (develop-1.6 @ 9cd9e122): RayTrainer.train blocks in
wait_for_ray_job_completion and needs a dashboard URL (A for the submit-only
test); everything else errors because app.infra.ray_training_job and the
poll/cancel functions do not exist. TI-12/13/15/37/44/62 and the k rule.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import yaml

try:
    from app.core import ml_store
except ImportError:  # pragma: no cover - base tree only
    ml_store = None

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store():
    if ml_store is None:
        yield None
        return
    s = ml_store.InMemoryMLStore()
    ml_store.set_ml_store(s)
    yield s
    ml_store.set_ml_store(None)


from contextlib import nullcontext


def _maybe_patch(target, *args, **kwargs):
    """patch() when the module exists; a no-op on a tree without the feature, so the
    behavioural assertions are reached on the base instead of erroring on the patch."""
    try:
        __import__(target.rsplit(".", 1)[0])
    except ImportError:
        return nullcontext(MagicMock())
    return patch(target, *args, **kwargs)


def _plan(**over):
    plan = {"model_type": "classification", "model_name": "m", "model_description": "", "model_version": "1",
            "hyperparameter_config": {}, "ray_config": {"num_workers": 1, "cpu_per_worker": 1, "memory_per_worker": "1Gi"},
            "data_config": {"dataset_uri": "s3://b/x.csv", "feature_columns": ["a"], "target_column": "t"}}
    plan.update(over); return plan


# T-37 / TI-62: the manifest ---------------------------------------------------------

def test_training_rayjob_targets_the_shared_cluster_and_never_its_own():
    from app.infra import ray_training_job as rtj
    m = rtj.training_rayjob_manifest("avaloka-train-abc", entrypoint="python /app/ray_job.py",
                                     env={"EPOCHS": "3", "MLFLOW_TRACKING_PASSWORD": "x", "AWS_SECRET_ACCESS_KEY": "y"},
                                     job_id="job-1", namespace="default")
    spec = m["spec"]
    assert spec["clusterSelector"] == {"ray.io/cluster": "avaloka-raycluster"}
    assert "rayClusterSpec" not in spec, "no cluster per job"
    assert spec["shutdownAfterJobFinishes"] is False and spec["ttlSecondsAfterFinished"] == 600
    assert "entrypointNumGpus" not in spec, "CPU unless the plan asks for a GPU"
    env = yaml.safe_load(spec["runtimeEnvYAML"])["env_vars"]
    assert env == {"EPOCHS": "3"}, "secrets never travel in the CR"
    assert m["metadata"]["labels"]["avaloka.ai/ml-job"] == "job-1"
    gpu = rtj.training_rayjob_manifest("n", entrypoint="e", env={}, use_gpu=True)
    assert gpu["spec"]["entrypointNumGpus"] == 1


def test_result_block_and_backoff():
    from app.infra import ray_training_job as rtj
    logs = "noise\n```json\n{\"a\": 1}\n```\nmore\n```json\n{\"mlflow_run_id\": \"r1\", \"final_accuracy\": 0.9}\n```\n"
    assert rtj.result_from_logs(logs) == {"mlflow_run_id": "r1", "final_accuracy": 0.9}, "the last block wins"
    assert rtj.result_from_logs("no block") is None
    assert [rtj.backoff_s(i) for i in range(5)] == [15, 30, 60, 60, 60]
    assert rtj.is_terminal("SUCCEEDED") and rtj.is_terminal("failed") and not rtj.is_terminal("RUNNING") and not rtj.is_terminal(None)


# T-16 / TI-62: submit-only ----------------------------------------------------------

def test_raytrainer_submits_and_returns_without_waiting(monkeypatch):
    from app.agents.mta_v2 import ray_trainer as rt
    monkeypatch.delenv("MTA_RAY_SUBMISSION_MODE", raising=False)
    assert getattr(rt, "submission_mode", lambda: "direct")() == "kuberay", "the default must be the RayJob CR path"
    submitted = {}
    with _maybe_patch("app.infra.ray_reaper.ensure_shared_cluster") as ensure, \
         _maybe_patch("app.infra.ray_training_job.submit", side_effect=lambda m: submitted.setdefault("m", m) and m["metadata"]["name"]), \
         patch.object(rt, "wait_for_ray_job_completion", side_effect=AssertionError("train() waited for the job")), \
         patch.object(rt, "notify_ray_job_submitted"), \
         patch.object(rt, "create_ray_cluster", side_effect=AssertionError("a per-job cluster was provisioned")), \
         patch.object(rt, "get_k8s_api", return_value=MagicMock()), \
         patch.object(rt.RayTrainer, "_read_gcp_sa_json", return_value="{}", create=True):
        result = rt.RayTrainer(user_id="u", session_id="s").train(_plan())
    ensure.assert_called_once()
    assert ensure.call_args.kwargs["wait"] is True
    assert result["status"] == "submitted"
    assert result["ray_job_name"] == submitted["m"]["metadata"]["name"]
    assert submitted["m"]["spec"]["clusterSelector"] == {"ray.io/cluster": "avaloka-raycluster"}
    assert yaml.safe_load(submitted["m"]["spec"]["runtimeEnvYAML"])["env_vars"]["CV_FOLDS"] == "5"


def test_unavailable_shared_cluster_fails_plainly_and_creates_nothing(monkeypatch):
    from app.agents.mta_v2 import ray_trainer as rt
    from app.infra.ray_reaper import SharedClusterUnavailable
    monkeypatch.delenv("MTA_RAY_SUBMISSION_MODE", raising=False)
    with patch("app.infra.ray_reaper.ensure_shared_cluster", side_effect=SharedClusterUnavailable("The shared Ray cluster 'avaloka-raycluster' in namespace 'default' did not become ready within 600 seconds; training was not started.")), \
         patch("app.infra.ray_training_job.submit", side_effect=AssertionError("submitted despite no cluster")), \
         patch.object(rt, "create_ray_cluster", side_effect=AssertionError("provisioned a per-job cluster")):
        result = rt.RayTrainer(user_id="u", session_id="s").train(_plan())
    assert result["status"] == "error"
    assert "avaloka-raycluster" in result["error"]


# TI-44: ensure with a readiness timeout -----------------------------------------------

class _K8s:
    def __init__(self, states):
        self.states = list(states); self.applied = 0
    def list_ray(self, plural, namespace=None):
        st = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return [] if st is None else [{"metadata": {"name": "avaloka-raycluster"}, "status": {"state": st}}]


def test_ensure_waits_for_ready_and_times_out_naming_the_cluster():
    from app.infra import ray_reaper as rr
    clock = iter(range(0, 10_000, 5))
    k8s = _K8s([None, "", "", "ready"])
    apply = MagicMock()
    assert rr.ensure_shared_cluster(k8s=k8s, apply=apply, wait=True, timeout_s=600, sleep=lambda s: None, clock=lambda: next(clock)) is True
    apply.assert_called_once()
    slow = _K8s(["", ""])
    t = iter([0, 100, 700, 700])
    with pytest.raises(rr.SharedClusterUnavailable, match="avaloka-raycluster.*600"):
        rr.ensure_shared_cluster(k8s=slow, apply=apply, wait=True, timeout_s=600, sleep=lambda s: None, clock=lambda: next(t))


# TI-12 / TI-15: the poll chain -----------------------------------------------------------

class _Rtj:
    def __init__(self, statuses, logs=""):
        self.statuses = list(statuses); self.logs = logs; self.deleted = []
    def status(self, name, ns):
        st = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return st, None, None, None
    def is_terminal(self, s):
        from app.infra.ray_training_job import is_terminal
        return is_terminal(s)
    def fetch_logs(self, name, ns):
        return self.logs
    def result_from_logs(self, logs):
        from app.infra.ray_training_job import result_from_logs
        return result_from_logs(logs)
    def delete(self, name, ns="default"):
        self.deleted.append(name); return True


def test_poll_reenqueues_itself_with_a_bounded_countdown_while_running(store):
    from app.infra import training_tasks as t
    job = store.create_job(ml_store.MLJob(user_id="u", kind="training", status="running", ray_job_name="avaloka-train-1"))
    dispatched = []
    def dispatch(args, kwargs, countdown):
        dispatched.append((args, kwargs, countdown))
    for attempt in (0, 1, 2, 5):
        out = t.poll_training_job_once(job.id, {"ray_namespace": "default"}, attempt, rtj=_Rtj(["RUNNING"]), dispatch=dispatch)
        assert out["state"] == "running"
    countdowns = [c for _, _, c in dispatched]
    assert countdowns == [30, 60, 60, 60]
    assert all(15 <= c <= 60 for c in countdowns)
    assert all(k["attempt"] == a + 1 for (_, k, _), a in zip(dispatched, (0, 1, 2, 5)))
    assert store.get_job(job.id).status == "running"


def test_poll_finishes_a_succeeded_job_registers_and_lists_the_endpoint(store):
    from app.infra import training_tasks as t
    job = store.create_job(ml_store.MLJob(user_id="u", kind="training", status="running", ray_job_name="avaloka-train-1", session_id="s"))
    logs = "```json\n{\"mlflow_run_id\": \"run-9\", \"model_name\": \"m\", \"model_type\": \"classification\", \"final_accuracy\": 0.9}\n```"
    mgr = MagicMock(); mgr.prepare_inference_service.return_value = {"model_uri": "models:/x/1"}
    bells = []
    with patch("app.infra.training_tasks._manager", return_value=mgr), \
         patch("app.core.ml_registry.register_run", return_value={"registered_name": "avaloka.u.m", "version": 1}) as reg, \
         patch("app.infra.training_tasks._finalize_run_record") as fin, \
         patch("app.infra.training_tasks._notify", side_effect=lambda fs, eid=None: bells.append((fs, eid))):
        out = t.poll_training_job_once(job.id, {"execution_id": "celery-1", "thread_id": "th"}, 3, rtj=_Rtj(["SUCCEEDED"], logs), dispatch=lambda *a, **k: pytest.fail("re-enqueued a finished job"))
    assert out["state"] == "succeeded" and out["mlflow_run_id"] == "run-9"
    closed = store.get_job(job.id)
    assert (closed.status, closed.mlflow_run_id) == ("succeeded", "run-9")
    reg.assert_called_once(); assert reg.call_args.args[:2] == ("u", "run-9")
    mgr.configure_inference_service.assert_not_called(); mgr.prepare_inference_service.assert_called_once_with("run-9")
    assert store.get_endpoint("run-9").status == "ready"
    fin.assert_called_once()
    assert bells and bells[0][0]["training_completed"] is True and bells[0][0]["thread_id"] == "th"
    # The submitting run's Celery task id travels with the bell, so the ledger
    # row that run started is the one finished (app/core/celery_app._ledger_job_id).
    assert bells[0][1] == "celery-1"


def test_poll_finishes_a_failed_job_with_a_named_failure_and_no_registration(store):
    from app.infra import training_tasks as t
    job = store.create_job(ml_store.MLJob(user_id="u", kind="training", status="running", ray_job_name="avaloka-train-1"))
    with patch("app.core.ml_registry.register_run") as reg, patch("app.infra.training_tasks._finalize_run_record"), \
         patch("app.infra.training_tasks._notify"):
        out = t.poll_training_job_once(job.id, {}, 2, rtj=_Rtj(["FAILED"], "exit code 137 OOMKilled"))
    assert out["state"] == "failed"
    closed = store.get_job(job.id)
    assert closed.status == "failed" and closed.error and "\n" not in closed.error
    reg.assert_not_called()
    assert store.get_endpoint("run-9") is None


def test_cancelled_job_is_not_polled_further(store):
    from app.infra import training_tasks as t
    job = store.create_job(ml_store.MLJob(user_id="u", kind="training", status="cancelled", ray_job_name="x"))
    assert t.poll_training_job_once(job.id, {}, 0, rtj=_Rtj(["RUNNING"]), dispatch=lambda *a, **k: pytest.fail("polled"))["state"] == "cancelled"


def test_resume_rearms_only_stale_running_jobs(store):
    from app.infra import training_tasks as t
    stale = store.create_job(ml_store.MLJob(user_id="u", kind="training", status="running", ray_job_name="a"))
    stale.updated_at = NOW - timedelta(seconds=1200)
    fresh = store.create_job(ml_store.MLJob(user_id="u", kind="training", status="running", ray_job_name="b"))
    fresh.updated_at = NOW - timedelta(seconds=30)
    store.create_job(ml_store.MLJob(user_id="u", kind="training", status="running"))   # no RayJob: local training
    dispatched = []
    resumed = t.resume_stale_training_polls(store=store, now=NOW, dispatch=lambda args, kwargs, countdown: dispatched.append(args[0]))
    assert resumed == [stale.id] and dispatched == [stale.id]


# TI-13: cancel ----------------------------------------------------------------------------

def test_cancel_deletes_the_rayjob_and_marks_the_row_cancelled(store):
    from app.infra import training_tasks as t
    job = store.create_job(ml_store.MLJob(user_id="u", kind="training", status="running", ray_job_name="avaloka-train-1"))
    rtj = _Rtj(["RUNNING"])
    out = t.cancel_training_job(job.id, "u", store=store, rtj=rtj)
    assert out.status == "cancelled" and out.finished_at is not None
    assert rtj.deleted == ["avaloka-train-1"]
    assert t.cancel_training_job(job.id, "someone-else", store=store, rtj=rtj) is None, "another user cannot see it"
    done = store.create_job(ml_store.MLJob(user_id="u", kind="training", status="succeeded", ray_job_name="z"))
    assert t.cancel_training_job(done.id, "u", store=store, rtj=rtj).status == "succeeded", "a finished job is left alone"
    assert rtj.deleted == ["avaloka-train-1"]


def test_cancel_route_is_owner_scoped(store):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.models_routes import build_router
    async def premium(u, f): return None
    app = FastAPI(); app.include_router(build_router(lambda r: r.headers.get("X-User"), premium))
    c = TestClient(app)
    job = store.create_job(ml_store.MLJob(user_id="u1", kind="training", status="running", ray_job_name="j"))
    with patch("app.infra.ray_training_job.delete", return_value=True):
        assert c.post(f"/api/ml-jobs/{job.id}/cancel", headers={"X-User": "u2"}).status_code == 404
        r = c.post(f"/api/ml-jobs/{job.id}/cancel", headers={"X-User": "u1"})
    assert r.status_code == 200 and r.json()["status"] == "cancelled"


# The worker's second pass and the run record ----------------------------------------------

def test_mta_records_a_submitted_run_as_running_and_starts_the_chain(store):
    from app.agents.mta_v2.agent import ModelTrainingAgent
    from langchain_core.messages import HumanMessage
    job = store.create_job(ml_store.MLJob(user_id="u", kind="training", status="queued"))
    state = {"user_id": "u", "session_id": "s", "messages": [HumanMessage(content="train")], "training_plan": _plan(),
             "training_result": None, "training_scheduled": True, "ml_job_id": job.id, "enable_training": True,
             "skip_to_training": True, "training_task": None, "data_source_location_cloud": "s3://b/x.csv",
             "dataset_size_bytes": 10 ** 10}
    trainer = MagicMock(); trainer.train.return_value = {"status": "submitted", "ray_job_name": "avaloka-train-1", "ray_namespace": "default", "model_name": "m", "task_id": "t"}
    with patch("app.agents.mta_v2.agent.llm", None), patch("app.agents.mta_v2.agent.RayTrainer", return_value=trainer):
        out = ModelTrainingAgent()._execute_training(state)
    assert out["training_completed"] is False and out["ray_job_name"] == "avaloka-train-1"
    assert "Training submitted" in out["messages"][-1].content
    assert store.get_job(job.id).status == "running" and store.get_job(job.id).ray_job_name == "avaloka-train-1"


def test_run_record_stays_running_and_bell_is_silent_for_a_submitted_training():
    from app.core import celery_app as ca
    client = MagicMock()
    final = {"training_result": {"status": "submitted", "ray_job_name": "avaloka-train-1"}, "user_id": "u"}
    with patch.object(ca, "_supabase", return_value=client):
        ca._record_run_finish("sched", "exec-1", final, 0.0, {"task_schedule": {"task_type": "training"}})
    payload = client.table.return_value.update.call_args.args[0]
    assert payload["status"] == "running" and "finished_at" not in payload
    sent = []
    import app.services.notification_service as ns
    async def fake(*a, **k): sent.append(a)
    with patch.object(ns, "create_notification", fake):
        ca._notify_job_finished({"task_schedule": {"task_type": "training", "schedule_type": "relative", "max_runs": 1}, "user_id": "u"}, final)
    assert sent == []


# k from the plan ------------------------------------------------------------------------------

def test_cv_folds_are_parsed_from_the_plan_text_and_default_to_five():
    from app.agents.mta_v2.agent import DEFAULT_CV_FOLDS, parse_cv_folds
    assert DEFAULT_CV_FOLDS == 5
    assert parse_cv_folds("use 3-fold cross validation") == 3
    assert parse_cv_folds("k = 7") == 7
    assert parse_cv_folds("10 folds please") == 10
    assert parse_cv_folds("folds: 1") == 2 and parse_cv_folds("25 folds") == 10, "clamped to [2, 10]"
    assert parse_cv_folds("just train it") is None


def test_cv_folds_reach_the_job_env_and_the_local_result():
    from app.agents.mta_v2 import ray_trainer as rt
    env = rt.RayTrainer(user_id="u", session_id="s")._build_job_env(_plan(cross_validation={"k": 3}))
    assert env["CV_FOLDS"] == "3"
    import inspect
    from app.agents.mta_v2 import local_trainer
    assert "cv_folds" in inspect.signature(local_trainer.training_main).parameters
