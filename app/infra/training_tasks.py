"""Background operations of the training/inference pipeline (the infra layer).

The chat graph and the API only *enqueue*; everything that can take more than
a moment runs here, in the Celery worker, with its state in Postgres
(``app/core/ml_store``). Nothing in this module is awaited by a request.

Operations
----------
``on_training_finished(final_state)``   worker hook after the MTA's training pass:
                                        register the model in the MLflow registry,
                                        then auto-deploy the endpoint *ready-on-demand*
                                        (listed, pod starts on first request -- nothing
                                        stays warm; owner ruling O-3).
``enqueue_deploy_endpoint(...)``        API/chat: create the job + endpoint rows and
                                        dispatch ``deploy_endpoint``.
``enqueue_stop_endpoint(...)``          same for stop; the stop *intent* is written
                                        synchronously so the badge flips at once.
``run_deploy_endpoint(job_id, warm)``   the work: lifecycle pending -> deploying ->
                                        ready|failed; ``warm=True`` also loads the model
                                        into the serving backend.
``run_stop_endpoint(job_id)``           unload and clear the alias.
``run_scheduled_service_action(state)`` the legacy RedBeat ``start_inference`` /
                                        ``stop_inference`` entries land here instead of
                                        re-entering the MTA graph node.

Errors recorded in ``ml_jobs.error`` / ``ml_endpoints.error`` are one line and
user-safe; stack traces go to the log only.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core import ml_endpoints as lifecycle
from app.core.ml_store import MLEndpoint, MLJob, get_ml_store

logger = logging.getLogger(__name__)

DEPLOY_SOFT_LIMIT_S = int(os.getenv("AVALOKA_DEPLOY_SOFT_LIMIT_S", "600"))
STOP_SOFT_LIMIT_S = int(os.getenv("AVALOKA_STOP_SOFT_LIMIT_S", "120"))


class EndpointBusy(RuntimeError):
    """A deploy is already in flight for this run."""


class NoWorker(RuntimeError):
    """No Celery worker is running to take the job."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _error_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    first = text[0] if text else type(exc).__name__
    return first[:300]


def _manager():
    from app.agents.mta_v2.inference_service_manager import InferenceServiceManager
    return InferenceServiceManager()


def _worker_available() -> bool:
    try:
        from app.agents.scheduler import is_celery_worker_running
        return bool(is_celery_worker_running())
    except Exception:  # noqa: BLE001 - treat an unreachable broker as "no worker"
        logger.warning("[training_tasks] could not probe the Celery worker", exc_info=True)
        return False


# --------------------------------------------------------------------------- #
# Celery tasks
# --------------------------------------------------------------------------- #

def _celery_app():
    from app.core.celery_app import celery_app
    return celery_app


def _register_tasks():
    """Bind the task functions to the Celery app (idempotent)."""
    app = _celery_app()

    @app.task(name="app.infra.training_tasks.deploy_endpoint", bind=True, acks_late=True,
              soft_time_limit=DEPLOY_SOFT_LIMIT_S, time_limit=DEPLOY_SOFT_LIMIT_S + 60)
    def deploy_endpoint(self, job_id: str, warm: bool = True) -> Dict[str, Any]:
        job = run_deploy_endpoint(job_id, warm=warm)
        return {"job_id": job_id, "status": job.status if job else "unknown"}

    @app.task(name="app.infra.training_tasks.stop_endpoint", bind=True, acks_late=True,
              soft_time_limit=STOP_SOFT_LIMIT_S, time_limit=STOP_SOFT_LIMIT_S + 60)
    def stop_endpoint(self, job_id: str) -> Dict[str, Any]:
        job = run_stop_endpoint(job_id)
        return {"job_id": job_id, "status": job.status if job else "unknown"}

    return deploy_endpoint, stop_endpoint


_TASKS: Optional[tuple] = None


def tasks():
    global _TASKS
    if _TASKS is None:
        _TASKS = _register_tasks()
    return _TASKS


# --------------------------------------------------------------------------- #
# Training on the shared Ray cluster: submit-only + a countdown poll chain
# --------------------------------------------------------------------------- #

POLL_SOFT_LIMIT_S = int(os.getenv("AVALOKA_POLL_SOFT_LIMIT_S", "120"))
STALE_POLL_S = int(os.getenv("AVALOKA_STALE_POLL_S", "600"))

_POLL_TASK: Optional[Any] = None


def poll_task():
    """The Celery task that follows a RayJob; registered on first use."""
    global _POLL_TASK
    if _POLL_TASK is None:
        app = _celery_app()

        @app.task(name="app.infra.training_tasks.poll_training_job", bind=True, acks_late=True,
                  soft_time_limit=POLL_SOFT_LIMIT_S, time_limit=POLL_SOFT_LIMIT_S + 30)
        def poll_training_job(self, job_id: str, ctx: Optional[Dict[str, Any]] = None, attempt: int = 0) -> Dict[str, Any]:
            return poll_training_job_once(job_id, ctx or {}, attempt)

        _POLL_TASK = poll_training_job
    return _POLL_TASK


def schedule_poll(job_id: str, ctx: Dict[str, Any], attempt: int = 0, *, dispatch: Any = None) -> int:
    """Enqueue the next poll with a 15 -> 60 s countdown. Returns the countdown used."""
    from app.infra.ray_training_job import backoff_s

    countdown = backoff_s(attempt)
    (dispatch or poll_task().apply_async)(args=[job_id], kwargs={"ctx": ctx, "attempt": attempt}, countdown=countdown)
    return countdown


def _result_from_rayjob(job: MLJob, ctx: Dict[str, Any], job_status: str, *, rtj: Any) -> Dict[str, Any]:
    """Turn a terminal RayJob into the TrainingResult dict the chat and panel expect."""
    name, ns = job.ray_job_name or "", ctx.get("ray_namespace") or os.getenv("MTA_RAY_NAMESPACE", "default")
    logs = rtj.fetch_logs(name, ns)
    base = {"task_id": ctx.get("task_id") or "", "user_id": job.user_id, "session_id": job.session_id or "",
            "ray_job_id": name, "ray_job_name": name, "ray_job_status": job_status, "ray_namespace": ns,
            "dashboard_url": None}
    if str(job_status).upper() == "SUCCEEDED":
        payload = rtj.result_from_logs(logs)
        if not payload:
            from app.agents.mta_v2.failure_diagnostics import build_training_failure
            failure = build_training_failure("Ray training succeeded but returned no parseable result payload.",
                                             job_id=name, namespace=ns)
            return {**base, "status": "error", "error": failure["technical_details"], "failure": failure}
        return {**payload, **base}
    from app.agents.mta_v2.failure_diagnostics import build_training_failure
    failure = build_training_failure(f"Ray job finished with status {job_status}.", logs=logs, job_id=name, namespace=ns)
    return {**base, "status": "error", "error": failure["technical_details"], "failure": failure}


def poll_training_job_once(job_id: str, ctx: Dict[str, Any], attempt: int = 0, *, rtj: Any = None,
                           dispatch: Any = None, store: Any = None) -> Dict[str, Any]:
    """One step of the chain: read the CR; re-enqueue while it runs; finish the run when terminal.

    Finishing = TrainingResult from the submitter logs, ml_jobs succeeded/failed,
    registry + ready-on-demand endpoint (on_training_finished), the scheduled_runs
    row and the bell -- what the in-worker wait used to do in one blocking call.
    """
    from app.infra import ray_training_job as default_rtj

    rtj = rtj or default_rtj
    store = store or get_ml_store()
    job = store.get_job(job_id)
    if job is None:
        logger.error("[poll_training_job] job %s not found", job_id)
        return {"job_id": job_id, "state": "missing"}
    if job.status == "cancelled":
        return {"job_id": job_id, "state": "cancelled"}
    if not job.ray_job_name:
        store.update_job(job.id, status="failed", finished_at=_now(), error="Training job has no RayJob name to follow.")
        return {"job_id": job_id, "state": "failed"}
    ns = ctx.get("ray_namespace") or os.getenv("MTA_RAY_NAMESPACE", "default")
    try:
        job_status, deployment_status, _, _ = rtj.status(job.ray_job_name, ns)
    except Exception as exc:  # noqa: BLE001 - a transient read error is retried on the next poll
        logger.warning("[poll_training_job] status read failed for %s: %s", job.ray_job_name, exc)
        job_status, deployment_status = None, None
    if str(deployment_status or "") in {"Failed"} and not rtj.is_terminal(job_status):
        job_status = "FAILED"   # KubeRay could not even start it (image pull, scheduling)
    if not rtj.is_terminal(job_status):
        store.update_job(job.id, status="running", attempts=attempt + 1)
        countdown = schedule_poll(job_id, ctx, attempt + 1, dispatch=dispatch)
        return {"job_id": job_id, "state": "running", "ray_status": job_status, "next_poll_s": countdown}

    result = _result_from_rayjob(job, ctx, str(job_status), rtj=rtj)
    failed = result.get("status") == "error" or bool(result.get("error"))
    final_state: Dict[str, Any] = {
        "user_id": job.user_id, "session_id": job.session_id, "thread_id": ctx.get("thread_id"),
        "dataset_id": ctx.get("dataset_id"), "schedule_id": ctx.get("schedule_id"),
        "task_schedule": {"task_type": "training", "schedule_type": "relative", "second": 0, "max_runs": 1},
        "training_result": result, "training_completed": not failed, "training_scheduled": False,
        "mlflow_run_id": result.get("mlflow_run_id"), "ml_job_id": job.id, "messages": [],
        "ledger_job_id": ctx.get("ledger_job_id"),
    }
    if failed:
        line = str((result.get("failure") or {}).get("title") or "Training failed.")[:300]
        store.update_job(job.id, status="failed", finished_at=_now(), error=line)
    else:
        store.update_job(job.id, status="succeeded", finished_at=_now(), mlflow_run_id=result.get("mlflow_run_id"), error=None)
        final_state = on_training_finished(final_state)
    _finalize_run_record(ctx, final_state)
    _notify(final_state, ctx.get("execution_id"))
    return {"job_id": job_id, "state": "failed" if failed else "succeeded", "mlflow_run_id": result.get("mlflow_run_id")}


def _finalize_run_record(ctx: Dict[str, Any], final_state: Dict[str, Any]) -> None:
    """Close the scheduled_runs row the submitting Celery run left 'running'."""
    execution_id, schedule_id = ctx.get("execution_id"), ctx.get("schedule_id")
    if not execution_id:
        return
    try:
        from app.core import celery_app as ca
        ca._record_run_finish(schedule_id or "", execution_id, final_state, ctx.get("t0") or 0.0,
                              {"task_schedule": final_state.get("task_schedule")})
    except Exception:  # noqa: BLE001
        logger.warning("[poll_training_job] could not close the run record %s", execution_id, exc_info=True)


def _notify(final_state: Dict[str, Any], execution_id: Optional[str] = None) -> None:
    """Finish the job-ledger row the submitting run started (its ledger_job_id,
    else that run's Celery task id, which is the row id it used) and ring the bell."""
    try:
        from app.core import celery_app as ca
        ca._notify_job_finished({"task_schedule": final_state.get("task_schedule"), "user_id": final_state.get("user_id"),
                                 "thread_id": final_state.get("thread_id"), "dataset_id": final_state.get("dataset_id"),
                                 "schedule_id": final_state.get("schedule_id"), "messages": [],
                                 "ledger_job_id": final_state.get("ledger_job_id")}, final_state,
                                execution_id=execution_id)
    except Exception:  # noqa: BLE001
        logger.warning("[poll_training_job] notification failed", exc_info=True)


def start_poll_chain(final_state: Dict[str, Any], *, dispatch: Any = None) -> Optional[str]:
    """Called by the worker after a submit-only training pass: record the RayJob and arm the chain."""
    result = final_state.get("training_result") or {}
    job_id = final_state.get("ml_job_id")
    store = get_ml_store()
    if not job_id:
        job = store.create_job(MLJob(user_id=str(final_state.get("user_id") or ""), session_id=final_state.get("session_id"),
                                     kind="training", status="running"))
        job_id = job.id
    try:
        from celery import current_task
        execution_id = getattr(getattr(current_task, "request", None), "id", None)
    except Exception:  # noqa: BLE001
        execution_id = None
    store.update_job(job_id, status="running", ray_job_name=result.get("ray_job_name"), celery_task_id=execution_id,
                     started_at=_now())
    ctx = {"task_id": result.get("task_id"), "ray_namespace": result.get("ray_namespace"),
           "thread_id": final_state.get("thread_id"), "dataset_id": final_state.get("dataset_id"),
           "schedule_id": final_state.get("schedule_id"), "execution_id": execution_id,
           "ledger_job_id": final_state.get("ledger_job_id")}
    schedule_poll(job_id, ctx, 0, dispatch=dispatch)
    return job_id


def resume_stale_training_polls(*, store: Any = None, now: Optional[datetime] = None, dispatch: Any = None,
                                stale_s: int = STALE_POLL_S) -> List[str]:
    """Beat safety net: a running training whose poll chain went quiet (broker loss,
    worker restart mid-countdown) gets its chain re-armed from the ml_jobs row."""
    store = store or get_ml_store()
    clock = now or _now()
    resumed: List[str] = []
    for job in store.active_jobs(kind="training"):
        if job.status != "running" or not job.ray_job_name:
            continue
        if (clock - job.updated_at).total_seconds() < stale_s:
            continue
        schedule_poll(job.id, {"execution_id": job.celery_task_id}, 0, dispatch=dispatch)
        store.update_job(job.id, attempts=job.attempts + 1)
        resumed.append(job.id)
    return resumed


def cancel_training_job(job_id: str, user_id: str, *, store: Any = None, rtj: Any = None) -> Optional[MLJob]:
    """Cancel a queued/running training: delete its RayJob CR, mark the row cancelled (TI-13).
    Returns None when the job does not exist or belongs to someone else."""
    from app.infra import ray_training_job as default_rtj

    store = store or get_ml_store()
    job = store.get_job(job_id)
    if job is None or str(job.user_id) != str(user_id) or job.kind != "training":
        return None
    if job.terminal:
        return job
    if job.ray_job_name:
        try:
            (rtj or default_rtj).delete(job.ray_job_name)
        except Exception:  # noqa: BLE001 - the CR's TTL reaps it anyway
            logger.warning("[cancel] could not delete RayJob %s", job.ray_job_name, exc_info=True)
    return store.update_job(job.id, status="cancelled", finished_at=_now(), error=None)


# --------------------------------------------------------------------------- #
# Enqueue (API / chat side)
# --------------------------------------------------------------------------- #

def enqueue_deploy_endpoint(user_id: str, mlflow_run_id: str, *, session_id: Optional[str] = None,
                            warm: bool = True, dispatch: bool = True) -> MLJob:
    """Create the endpoint (pending) and job rows and dispatch the deploy.

    Raises :class:`EndpointBusy` when a deploy is already in flight and
    :class:`NoWorker` when nothing can take the job (the rows are then marked
    failed so the UI does not show a job that will never run).
    """
    store = get_ml_store()
    ep = store.get_endpoint(mlflow_run_id)
    if ep is None:
        ep = MLEndpoint(mlflow_run_id=mlflow_run_id, user_id=str(user_id), status=lifecycle.PENDING)
    elif ep.status == lifecycle.DEPLOYING:
        raise EndpointBusy(f"an endpoint deploy is already running for {mlflow_run_id}")
    elif ep.status in (lifecycle.FAILED, lifecycle.STOPPED):
        ep.status = lifecycle.transition(ep.status, "redeploy")
        ep.error = None
    elif ep.status == lifecycle.READY and not warm:
        # Already listed and ready-on-demand; nothing to do.
        pass
    store.upsert_endpoint(ep)

    job = store.create_job(MLJob(user_id=str(user_id), session_id=session_id, kind="deploy_endpoint",
                                 status="queued", mlflow_run_id=mlflow_run_id))
    if not dispatch:
        return job
    if not _worker_available():
        store.update_job(job.id, status="failed", error="No background worker is running.", finished_at=_now())
        raise NoWorker("no Celery worker is running")
    deploy, _ = tasks()
    result = deploy.apply_async(args=[job.id], kwargs={"warm": bool(warm)})
    store.update_job(job.id, celery_task_id=result.id)
    return job


def enqueue_stop_endpoint(user_id: str, mlflow_run_id: str, *, session_id: Optional[str] = None,
                          dispatch: bool = True) -> MLJob:
    """Write the stop intent now (badge flips) and dispatch the unload."""
    store = get_ml_store()
    ep = store.get_endpoint(mlflow_run_id)
    if ep is None:
        ep = MLEndpoint(mlflow_run_id=mlflow_run_id, user_id=str(user_id), status=lifecycle.STOPPED)
    elif lifecycle.can(ep.status, "stop"):
        ep.status = lifecycle.transition(ep.status, "stop")
    store.upsert_endpoint(ep)

    job = store.create_job(MLJob(user_id=str(user_id), session_id=session_id, kind="stop_endpoint",
                                 status="queued", mlflow_run_id=mlflow_run_id))
    if not dispatch:
        return job
    if not _worker_available():
        store.update_job(job.id, status="failed", error="No background worker is running.", finished_at=_now())
        raise NoWorker("no Celery worker is running")
    _, stop = tasks()
    result = stop.apply_async(args=[job.id])
    store.update_job(job.id, celery_task_id=result.id)
    return job


# --------------------------------------------------------------------------- #
# The work (worker side)
# --------------------------------------------------------------------------- #

def run_deploy_endpoint(job_id: str, *, warm: bool = True, manager: Any = None) -> Optional[MLJob]:
    store = get_ml_store()
    job = store.get_job(job_id)
    if job is None:
        logger.error("[deploy_endpoint] job %s not found", job_id)
        return None
    run_id = job.mlflow_run_id or ""
    ep = store.get_endpoint(run_id) or MLEndpoint(mlflow_run_id=run_id, user_id=job.user_id)
    store.update_job(job.id, status="running", started_at=_now(), attempts=job.attempts + 1)

    try:
        if ep.status in (lifecycle.FAILED, lifecycle.STOPPED):
            ep.status = lifecycle.transition(ep.status, "redeploy")
        if ep.status == lifecycle.PENDING:
            ep.status = lifecycle.transition(ep.status, "start_deploy")
        elif ep.status == lifecycle.READY and warm:
            # Warming an endpoint that is already listed: stay ready throughout.
            pass
        store.upsert_endpoint(ep)

        record = store.get_model(run_id)
        ep.server = "gpu" if (record is not None and record.requires_gpu) else "cpu"   # TI-58: CPU unless tagged
        store.upsert_endpoint(ep)
        mgr = manager or _manager()
        if warm:
            details = mgr.configure_inference_service(run_id)
        else:
            details = mgr.prepare_inference_service(run_id)
        _set_serving_alias(run_id)

        if ep.status == lifecycle.DEPLOYING:
            ep.status = lifecycle.transition(ep.status, "deployed")
        ep.model_uri = (details or {}).get("model_uri") if isinstance(details, dict) else None
        ep.error = None
        store.upsert_endpoint(ep)
        store.update_job(job.id, status="succeeded", finished_at=_now(), error=None)
    except Exception as exc:  # noqa: BLE001 - recorded, never retried blindly
        logger.error("[deploy_endpoint] %s failed: %s", run_id, exc, exc_info=True)
        line = _error_line(exc)
        if ep.status == lifecycle.DEPLOYING:
            ep.status = lifecycle.transition(ep.status, "deploy_failed")
        ep.error = line
        store.upsert_endpoint(ep)
        store.update_job(job.id, status="failed", finished_at=_now(), error=line)
    return store.get_job(job.id)


def run_stop_endpoint(job_id: str, *, manager: Any = None) -> Optional[MLJob]:
    store = get_ml_store()
    job = store.get_job(job_id)
    if job is None:
        logger.error("[stop_endpoint] job %s not found", job_id)
        return None
    run_id = job.mlflow_run_id or ""
    ep = store.get_endpoint(run_id)
    store.update_job(job.id, status="running", started_at=_now(), attempts=job.attempts + 1)
    try:
        mgr = manager or _manager()
        mgr.stop_inference_service(run_id)
        _clear_serving_alias(run_id)
        if ep is not None:
            if lifecycle.can(ep.status, "stop"):
                ep.status = lifecycle.transition(ep.status, "stop")
            ep.error = None
            store.upsert_endpoint(ep)
        store.update_job(job.id, status="succeeded", finished_at=_now(), error=None)
    except Exception as exc:  # noqa: BLE001
        logger.error("[stop_endpoint] %s failed: %s", run_id, exc, exc_info=True)
        store.update_job(job.id, status="failed", finished_at=_now(), error=_error_line(exc))
    return store.get_job(job.id)


def _set_serving_alias(run_id: str) -> None:
    """Point the registry alias at this version; absent registry support is fine."""
    try:
        from app.core import ml_registry
    except ImportError:
        return
    try:
        ml_registry.set_serving_alias(run_id)
    except Exception:  # noqa: BLE001 - the alias is a convenience, the endpoint is the truth
        logger.warning("[deploy_endpoint] could not set the serving alias for %s", run_id, exc_info=True)


def _clear_serving_alias(run_id: str) -> None:
    try:
        from app.core import ml_registry
    except ImportError:
        return
    try:
        ml_registry.clear_serving_alias(run_id)
    except Exception:  # noqa: BLE001
        logger.warning("[stop_endpoint] could not clear the serving alias for %s", run_id, exc_info=True)


# --------------------------------------------------------------------------- #
# Worker hooks called from app.core.celery_app
# --------------------------------------------------------------------------- #

def on_training_finished(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """After the MTA's worker pass: register the model, then auto-deploy ready-on-demand.

    Returns the state with ``ml_registration`` / ``ml_endpoint_status`` added.
    Never raises: a registry or endpoint problem is recorded on the rows and
    in the state; the training result itself stands.
    """
    result = final_state.get("training_result")
    if not isinstance(result, dict):
        return final_state
    if result.get("status") == "submitted":
        # The RayJob is running; the poll chain calls back into this function when it ends.
        final_state["ml_job_id"] = start_poll_chain(final_state)
        return final_state
    run_id = result.get("mlflow_run_id")
    failed = bool(result.get("status") == "error" or result.get("error") or result.get("execution_error"))
    if failed or not run_id:
        return final_state
    user_id = final_state.get("user_id")
    if not user_id:
        logger.warning("[on_training_finished] no user_id in state; skipping registry/endpoint")
        return final_state
    session_id = final_state.get("session_id")

    # 1. registry
    registration: Optional[Dict[str, Any]] = None
    try:
        from app.core import ml_registry
        registration = ml_registry.register_run(str(user_id), run_id, result)
    except ImportError:
        logger.info("[on_training_finished] registry module absent; skipping registration")
    except Exception as exc:  # noqa: BLE001
        logger.error("[on_training_finished] registration failed for %s: %s", run_id, exc, exc_info=True)
        registration = {"error": _error_line(exc)}
    if registration is not None:
        final_state["ml_registration"] = registration

    # 1b. cross-validation (1.7/1.8): k-fold on the training split, confusion matrix per fold
    try:
        final_state["ml_cross_validation"] = run_cross_validation(str(user_id), run_id, final_state)
    except Exception as exc:  # noqa: BLE001 - CV never fails the training it describes
        logger.error("[on_training_finished] cross-validation failed for %s: %s", run_id, exc, exc_info=True)
        final_state["ml_cross_validation"] = {"status": "failed", "reason": _error_line(exc)}

    # 2. endpoint, ready-on-demand (no warm pod)
    try:
        job = enqueue_deploy_endpoint(str(user_id), run_id, session_id=session_id, warm=False, dispatch=False)
        job = run_deploy_endpoint(job.id, warm=False)
        ep = get_ml_store().get_endpoint(run_id)
        final_state["ml_endpoint_status"] = ep.status if ep else None
        if job is not None and job.status != "succeeded":
            final_state["configure_inference_error"] = job.error
    except EndpointBusy:
        final_state["ml_endpoint_status"] = lifecycle.DEPLOYING
    except Exception as exc:  # noqa: BLE001
        logger.error("[on_training_finished] auto-deploy failed for %s: %s", run_id, exc, exc_info=True)
        final_state["configure_inference_error"] = _error_line(exc)
    return final_state


def run_scheduled_service_action(state: Dict[str, Any], schedule_data: Dict[str, Any]) -> Dict[str, Any]:
    """Serve a legacy RedBeat ``start_inference`` / ``stop_inference`` entry in the worker.

    Replaces re-entering the MTA graph node for these; appends the assistant
    message the pending-turn poll shows.
    """
    from langchain_core.messages import AIMessage

    task_type = schedule_data.get("task_type")
    run_id = (state.get("training_result") or {}).get("mlflow_run_id") or state.get("mlflow_run_id")
    messages = list(state.get("messages") or [])
    user_id = state.get("user_id") or ""
    if not run_id:
        messages.append(AIMessage(content="A trained model is required before managing an inference service."))
        state["messages"] = messages
        return state
    try:
        if task_type == "start_inference":
            job = state.get("ml_job_id") and get_ml_store().get_job(state["ml_job_id"])
            if not job:
                job = enqueue_deploy_endpoint(str(user_id), run_id, session_id=state.get("session_id"),
                                              warm=True, dispatch=False)
            job = run_deploy_endpoint(job.id, warm=True)
            ok = job is not None and job.status == "succeeded"
            state["inference_service_details"] = {"run_id": run_id, "status": "ready"} if ok else None
            messages.append(AIMessage(content=(
                "Inference service setup is complete. The model is available in the Inference panel."
                if ok else f"Sorry, there was an error configuring the inference service: {job.error if job else 'unknown'}")))
            if not ok and job is not None:
                state["configure_inference_error"] = job.error
        else:
            job = state.get("ml_job_id") and get_ml_store().get_job(state["ml_job_id"])
            if not job:
                job = enqueue_stop_endpoint(str(user_id), run_id, session_id=state.get("session_id"), dispatch=False)
            job = run_stop_endpoint(job.id)
            ok = job is not None and job.status == "succeeded"
            state["inference_service_details"] = None
            messages.append(AIMessage(content=(
                "The inference service has been stopped." if ok
                else f"Sorry, there was an error stopping the inference service: {job.error if job else 'unknown'}")))
            if not ok and job is not None:
                state["stop_inference_error"] = job.error
    except Exception as exc:  # noqa: BLE001
        logger.error("[service_action] %s failed: %s", task_type, exc, exc_info=True)
        messages.append(AIMessage(content="Sorry, there was an error with the inference service. Please try again."))
        state["configure_inference_error" if task_type == "start_inference" else "stop_inference_error"] = _error_line(exc)
    state["messages"] = messages
    state["configure_inference_service_scheduled"] = False
    state["stop_inference_service_scheduled"] = False
    state["task_schedule"] = None
    return state


# --------------------------------------------------------------------------- #
# Cross-validation (1.7/1.8)
# --------------------------------------------------------------------------- #

def run_cross_validation(user_id: str, run_id: str, final_state: Dict[str, Any], *, store: Any = None,
                         cv_module: Any = None, mlflow_log: Any = None) -> Dict[str, Any]:
    """k-fold the run's recipe on its training split, store the per-fold and summed
    confusion matrices (regression: fold metrics, no matrix) in ml_cross_validation
    and on the MLflow run. Returns the row's dict.

    Runs where training ran: a local training is cross-validated in this worker; a
    run trained on the shared Ray cluster is recorded as skipped (its dataset did not
    fit the worker) -- the RayJob variant is the owed follow-up (TI-39).
    """
    from app.core.ml_store import MLCrossValidation
    from app.agents.mta_v2 import cross_validation as default_cv

    cv = cv_module or default_cv
    store = store or get_ml_store()
    plan = final_state.get("training_plan") or {}
    result = final_state.get("training_result") or {}
    row = MLCrossValidation(mlflow_run_id=run_id, user_id=str(user_id), status="running")
    store.upsert_cross_validation(row)
    backend = plan.get("execution_backend") or ("ray" if result.get("ray_job_name") or result.get("ray_job_id") else "local")
    if not plan.get("data_config"):
        row.status, row.reason = "skipped", "no training plan with a data_config was available to cross-validate"
        store.upsert_cross_validation(row); return _cv_dict(row)
    if backend == "ray":
        row.status, row.reason = "skipped", "trained on the shared Ray cluster; cross-validation as a RayJob is a follow-up"
        store.upsert_cross_validation(row); return _cv_dict(row)
    k = int((plan.get("cross_validation") or {}).get("k") or result.get("cv_folds") or cv.DEFAULT_K)
    report = cv.cross_validate_plan(plan, k=k)
    row.status = report["status"]; row.k = report.get("k"); row.splitter = report.get("splitter"); row.reason = report.get("reason")
    row.aggregate = report.get("aggregate"); row.folds = [{kk: v for kk, v in f.items() if kk != "val_index"} for f in (report.get("folds") or [])]
    store.upsert_cross_validation(row)
    if report["status"] == "succeeded":
        _log_cv_to_mlflow(run_id, report, cv, mlflow_log)
    try:
        from app.core import ml_telemetry
        ml_telemetry.emit("cross_validation.finished", run_id=run_id, status=report["status"], k=report.get("k"),
                          n_rows=report.get("n_rows"), duration_s=report.get("duration_s"), task_type=report.get("task"))
    except Exception:  # noqa: BLE001
        pass
    return _cv_dict(row)


def _cv_dict(row: Any) -> Dict[str, Any]:
    return {"status": row.status, "k": row.k, "splitter": row.splitter, "reason": row.reason,
            "aggregate": row.aggregate, "folds": row.folds}


def _log_cv_to_mlflow(run_id: str, report: Dict[str, Any], cv: Any, mlflow_log: Any = None) -> None:
    """Per-fold + aggregate JSON artifacts and cv_* run metrics; best-effort."""
    try:
        import json
        import tempfile
        from pathlib import Path

        if mlflow_log is not None:
            mlflow_log(run_id, cv.fold_artifacts(report), cv.run_metrics(report)); return
        import mlflow
        from mlflow.tracking import MlflowClient
        from app.agents.mta_v2.tracking import resolve_tracking_uri

        mlflow.set_tracking_uri(resolve_tracking_uri(None))
        client = MlflowClient()
        for key, value in cv.run_metrics(report).items():
            client.log_metric(run_id, key, value)
        with tempfile.TemporaryDirectory() as tmp:
            for path, payload in cv.fold_artifacts(report).items():
                local = Path(tmp) / Path(path).name
                local.write_text(json.dumps(payload, indent=2))
                client.log_artifact(run_id, str(local), artifact_path=str(Path(path).parent))
    except Exception:  # noqa: BLE001
        logger.warning("[cross_validation] MLflow logging skipped for %s", run_id, exc_info=True)
