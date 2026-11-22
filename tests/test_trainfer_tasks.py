"""The infra layer's background tasks: deploy/stop endpoints, post-training hook.

Base classes (develop-1.6 @ 9cd9e122): the worker-branch test (T-05) fails by
assertion (the MTA node is re-entered); the chat-node tests (T-04) fail by
assertion (no job handle / task_schedule set); the rest error because
app.infra.training_tasks does not exist.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

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


def _tasks():
    from app.infra import training_tasks
    return training_tasks


from contextlib import contextmanager, nullcontext


def _maybe_patch(target: str, *args, **kwargs):
    """patch() when the target module exists; a no-op on a tree without it.

    Lets the chat-node and worker-branch tests reach their assertions on the
    base (where they fail by assertion) instead of erroring on the patch.
    """
    module = target.rsplit(".", 1)[0]
    try:
        __import__(module)
    except ImportError:
        return nullcontext(MagicMock())
    return patch(target, *args, **kwargs)


# T-04: the chat node enqueues --------------------------------------------------

def _service_state(action: str):
    return {
        "user_id": "user-1", "session_id": "s-1",
        "messages": [HumanMessage(content=f"{action} the inference service")],
        "training_result": {"mlflow_run_id": "run-1", "model_name": "m"},
        "mlflow_run_id": "run-1",
        "training_plan": {"model_name": "m"}, "training_completed": True,
        "enable_training": True, "skip_to_training": False, "training_task": None,
        "training_scheduled": False,
        "configure_inference_service_scheduled": False, "stop_inference_service_scheduled": False,
    }


def test_chat_start_service_enqueues_and_returns_a_job_handle(store):
    from app.agents.mta_v2.agent import ModelTrainingAgent

    class _NeverInTurn:
        def configure_inference_service(self, run_id):
            raise AssertionError("configure_inference_service ran inside the chat turn")

    with patch("app.agents.mta_v2.agent.InferenceServiceManager", _NeverInTurn), \
         _maybe_patch("app.infra.training_tasks._worker_available", return_value=True), \
         _maybe_patch("app.infra.training_tasks.tasks") as fake_tasks:
        deploy = MagicMock(); deploy.apply_async.return_value = MagicMock(id="celery-1")
        fake_tasks.return_value = (deploy, MagicMock())
        out = ModelTrainingAgent()._configure_inference_service(_service_state("start"))

    assert not out.get("task_schedule"), "the node must not schedule a RedBeat entry any more"
    assert out.get("ml_job_id"), "the turn must hand back a job handle"
    assert store is not None
    job = store.get_job(out["ml_job_id"])
    assert (job.kind, job.status, job.mlflow_run_id, job.celery_task_id) == ("deploy_endpoint", "queued", "run-1", "celery-1")
    assert store.get_endpoint("run-1").status == "pending"
    assert deploy.apply_async.call_args.kwargs["kwargs"] == {"warm": True}


def test_chat_stop_service_writes_the_intent_before_the_worker_runs(store):
    from app.agents.mta_v2.agent import ModelTrainingAgent
    if store is not None:
        store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="user-1", status="ready"))
    with _maybe_patch("app.infra.training_tasks._worker_available", return_value=True), \
         _maybe_patch("app.infra.training_tasks.tasks") as fake_tasks:
        stop = MagicMock(); stop.apply_async.return_value = MagicMock(id="celery-2")
        fake_tasks.return_value = (MagicMock(), stop)
        out = ModelTrainingAgent()._stop_inference_service(_service_state("stop"))
    assert not out.get("task_schedule"), "the node must not schedule a RedBeat entry any more"
    assert store is not None
    assert store.get_endpoint("run-1").status == "stopped"
    assert store.get_job(out["ml_job_id"]).kind == "stop_endpoint"


def test_chat_start_service_without_a_worker_fails_the_job_and_says_so(store):
    from app.agents.mta_v2.agent import ModelTrainingAgent
    with _maybe_patch("app.infra.training_tasks._worker_available", return_value=False):
        out = ModelTrainingAgent()._configure_inference_service(_service_state("start"))
    assert "no background worker" in out["messages"][-1].content.lower()
    assert store is not None
    jobs = store.list_jobs("user-1", kind="deploy_endpoint")
    assert jobs and jobs[0].status == "failed"


# T-05: the worker branch no longer re-enters the MTA ---------------------------

def test_worker_start_inference_branch_does_not_reenter_the_mta(store, monkeypatch):
    from app.core import celery_app as ca

    def _boom(state):
        raise AssertionError("model_training_agent_node was re-entered for start_inference")

    import app.agents.model_training_agent as mta_mod
    monkeypatch.setattr(mta_mod, "model_training_agent_node", _boom)

    mgr = MagicMock()
    mgr.configure_inference_service.return_value = {"model_uri": "runs:/run-1/model_files/model.onnx"}
    with _maybe_patch("app.infra.training_tasks._manager", return_value=mgr):
        final = ca._run_agent_task({
            "task_schedule": {"task_type": "start_inference", "schedule_type": "relative", "second": 0, "max_runs": 1},
            "user_id": "user-1", "session_id": "s-1",
            "training_result": {"mlflow_run_id": "run-1"},
            "messages": [{"role": "user", "content": "start the service"}],
        })
    assert final["messages"][-1]["content"].startswith("Inference service setup is complete")
    assert store is not None
    assert store.get_endpoint("run-1").status == "ready"
    mgr.configure_inference_service.assert_called_once_with("run-1")


# run_deploy_endpoint / run_stop_endpoint -----------------------------------------

def test_deploy_warm_false_prepares_without_loading(store):
    t = _tasks()
    mgr = MagicMock()
    mgr.prepare_inference_service.return_value = {"model_uri": "runs:/run-1/model_files/model.onnx"}
    job = t.enqueue_deploy_endpoint("user-1", "run-1", warm=False, dispatch=False)
    job = t.run_deploy_endpoint(job.id, warm=False, manager=mgr)
    mgr.configure_inference_service.assert_not_called()
    mgr.prepare_inference_service.assert_called_once_with("run-1")
    ep = store.get_endpoint("run-1")
    assert (ep.status, ep.model_uri, job.status) == ("ready", "runs:/run-1/model_files/model.onnx", "succeeded")


def test_deploy_failure_marks_endpoint_and_job_failed_with_one_line(store):
    t = _tasks()
    mgr = MagicMock()
    mgr.configure_inference_service.side_effect = RuntimeError("Ray Serve could not load it\nTraceback: columns a,b,c")
    job = t.enqueue_deploy_endpoint("user-1", "run-1", warm=True, dispatch=False)
    job = t.run_deploy_endpoint(job.id, warm=True, manager=mgr)
    ep = store.get_endpoint("run-1")
    assert ep.status == "failed"
    assert ep.error == "Ray Serve could not load it"
    assert job.status == "failed" and "\n" not in job.error


def test_redeploy_after_failure_goes_through_pending_again(store):
    t = _tasks()
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="user-1", status="failed", error="x"))
    job = t.enqueue_deploy_endpoint("user-1", "run-1", warm=True, dispatch=False)
    ep = store.get_endpoint("run-1")
    assert (ep.status, ep.error) == ("pending", None)
    mgr = MagicMock(); mgr.configure_inference_service.return_value = {}
    t.run_deploy_endpoint(job.id, warm=True, manager=mgr)
    assert store.get_endpoint("run-1").status == "ready"


def test_enqueue_refuses_a_second_deploy_while_one_is_deploying(store):
    t = _tasks()
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="user-1", status="deploying"))
    with pytest.raises(t.EndpointBusy):
        t.enqueue_deploy_endpoint("user-1", "run-1", dispatch=False)


def test_stop_unloads_and_records(store):
    t = _tasks()
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="run-1", user_id="user-1", status="ready"))
    job = t.enqueue_stop_endpoint("user-1", "run-1", dispatch=False)
    assert store.get_endpoint("run-1").status == "stopped", "the intent is written synchronously"
    mgr = MagicMock()
    job = t.run_stop_endpoint(job.id, manager=mgr)
    mgr.stop_inference_service.assert_called_once_with("run-1")
    assert job.status == "succeeded"


# on_training_finished -------------------------------------------------------------

def test_training_success_registers_then_lists_the_endpoint_ready_on_demand(store):
    t = _tasks()
    mgr = MagicMock()
    mgr.prepare_inference_service.return_value = {"model_uri": "runs:/run-9/model_files/model.onnx"}
    state = {"user_id": "user-1", "session_id": "s", "training_result": {"mlflow_run_id": "run-9", "model_name": "m"}}
    # Patch by attribute: a sys.modules fake is bypassed once the real module
    # exists (the package attribute wins), and the real registry would then run
    # against the local-file MLflow fallback.
    with patch("app.infra.training_tasks._manager", return_value=mgr), \
         patch("app.core.ml_registry.register_run", return_value={"registered_name": "avaloka.user-1.m", "version": 1}) as register_run:
        out = t.on_training_finished(state)
    register_run.assert_called_once()
    assert register_run.call_args.args[:2] == ("user-1", "run-9")
    mgr.configure_inference_service.assert_not_called(), "auto-deploy must not warm a pod"
    mgr.prepare_inference_service.assert_called_once_with("run-9")
    assert out["ml_endpoint_status"] == "ready"
    assert out["ml_registration"]["version"] == 1
    assert store.get_endpoint("run-9").status == "ready"
    assert [j.kind for j in store.list_jobs("user-1")] == ["deploy_endpoint"]


def test_training_failure_registers_nothing(store):
    t = _tasks()
    state = {"user_id": "user-1", "training_result": {"mlflow_run_id": "run-9", "status": "error", "error": "x"}}
    with patch("app.core.ml_registry.register_run") as register_run:
        out = t.on_training_finished(state)
    register_run.assert_not_called()
    assert store.get_endpoint("run-9") is None
    assert "ml_endpoint_status" not in out


def test_registry_error_does_not_lose_the_training_result(store):
    t = _tasks()
    mgr = MagicMock(); mgr.prepare_inference_service.return_value = {}
    state = {"user_id": "user-1", "training_result": {"mlflow_run_id": "run-9"}}
    with patch("app.infra.training_tasks._manager", return_value=mgr), \
         patch("app.core.ml_registry.register_run", side_effect=RuntimeError("registry 500\nmore")):
        out = t.on_training_finished(state)
    assert out["training_result"]["mlflow_run_id"] == "run-9"
    assert out["ml_registration"] == {"error": "registry 500"}
    assert out["ml_endpoint_status"] == "ready"


# T-23: the bell fires for training ------------------------------------------------

def test_notifier_fires_for_a_training_run(monkeypatch):
    from app.core import celery_app as ca
    sent = {}

    async def fake_create(user_id, kind, title, body, **kw):
        sent.update(user_id=user_id, kind=kind, title=title, body=body)

    import app.services.notification_service as ns
    monkeypatch.setattr(ns, "create_notification", fake_create)
    monkeypatch.setattr(ca, "_run_status_from_final", lambda final, task_type=None: ("success", None))
    ca._notify_job_finished(
        {"task_schedule": {"task_type": "training", "schedule_type": "relative", "max_runs": 1}, "user_id": "user-1"},
        {"training_result": {"mlflow_run_id": "r"}},
    )
    assert sent.get("title") == "Model training completed"
    # The job ledger rings with the training family (app/core/job_ledger.py);
    # before it, the Celery notifier reused analysis.completed.
    assert sent.get("kind") == "training.completed"
