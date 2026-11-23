"""Trained models, their registry record and their serving endpoint.

Moved out of ``server.py`` so the surface the inference panel calls is one
module with one contract:

* every route is owner-scoped (the MLflow run's ``user_id`` tag and the
  Postgres rows must match the caller);
* slow operations answer **202 with a job handle** and run in the Celery
  worker (``app/infra/training_tasks``); nothing here waits for a model load;
* endpoint truth is ``ml_endpoints`` in Postgres, never an MLflow artifact or
  Redis -- these routes no longer require Redis (the old precondition guarded
  nothing they do);
* predictions go through ``POST /api/models/{run_id}/predict`` (canonical;
  ``/inference`` is the panel's alias) and are proxied to the serving backend.
  The endpoint itself is never exposed.

Payload additions over the old routes: ``endpoint`` (status, url, server,
registered name/version/alias, error, updated_at) and ``registry`` on every
model; ``inference_service_details`` stays and is set only when ``ready``.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from app.core import ml_endpoints as lifecycle
from app.core.ml_store import MLEndpoint, MLJob, MLModel, get_ml_store

logger = logging.getLogger(__name__)

ResolveUser = Callable[[Request], Optional[str]]
RequirePlan = Callable[[str, str], Awaitable[None]]


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def endpoint_payload(run_id: str, ep: Optional[MLEndpoint], model: Optional[MLModel]) -> Dict[str, Any]:
    """What the panel reads for one model's endpoint. Stable keys, no customer data."""
    status_ = ep.status if ep else None
    return {
        "run_id": run_id,
        "status": status_,
        "servable": bool(ep and lifecycle.is_servable(ep.status)),
        "url": f"/api/models/{run_id}/predict" if ep and lifecycle.is_servable(ep.status) else None,
        "server": ep.server if ep else None,
        "model_uri": ep.model_uri if ep else None,
        "error": ep.error if ep else None,
        "last_request_at": _iso(ep.last_request_at) if ep else None,
        "updated_at": _iso(ep.updated_at) if ep else None,
        "registered_name": model.registered_name if model else None,
        "version": model.version if model else None,
        "alias": model.alias if model else None,
    }


def endpoint_asleep(run_id: str, ep: Optional[MLEndpoint]) -> bool:
    """Whether a ready endpoint's serving pod is scaled to zero. The serving
    backend answers this (PR-1b, model_server_scaler); without one, never asleep."""
    return False


def registry_payload(model: Optional[MLModel]) -> Optional[Dict[str, Any]]:
    if model is None:
        return None
    return {
        "registered_name": model.registered_name,
        "version": model.version,
        "alias": model.alias,
        "task_type": model.task_type,
        "requires_gpu": model.requires_gpu,
        "signature_logged": model.signature_logged,
        "created_at": _iso(model.created_at),
    }


def job_payload(job: MLJob) -> Dict[str, Any]:
    return {
        "id": job.id, "kind": job.kind, "status": job.status, "terminal": job.terminal,
        "run_id": job.mlflow_run_id, "error": job.error, "attempts": job.attempts,
        "created_at": _iso(job.created_at), "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at), "updated_at": _iso(job.updated_at),
    }


def decorate_model(item: Dict[str, Any], ep: Optional[MLEndpoint], model: Optional[MLModel]) -> Dict[str, Any]:
    """Attach endpoint/registry state to an MLflow summary/detail dict."""
    run_id = str(item.get("mlflow_run_id") or item.get("run_id") or "")
    out = dict(item)
    out["endpoint"] = endpoint_payload(run_id, ep, model)
    out["registry"] = registry_payload(model)
    # Legacy field the panel keys "running" on: present only when servable.
    if ep is not None and lifecycle.is_servable(ep.status):
        out["inference_service_details"] = out.get("inference_service_details") or {
            "run_id": run_id, "status": ep.status, "server": ep.server, "model_uri": ep.model_uri,
        }
    else:
        out["inference_service_details"] = None
    return out


def build_router(resolve_user_id: ResolveUser, require_premium_plan: RequirePlan) -> APIRouter:
    router = APIRouter(tags=["models"])

    def _caller(request: Request) -> str:
        user_id = resolve_user_id(request)
        if not user_id:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
        return str(user_id)

    def _mlflow():
        from app.agents.mta_v2.mlflow_manager import MLflowManager
        return MLflowManager()

    # Another user's run answers 404, exactly like a run that does not exist
    # (TI-24, as the trace routes do): a 403 would confirm the model exists.
    NOT_FOUND = "Model not found."

    async def _owned_run(user_id: str, run_id: str, *, forbidden: str = NOT_FOUND) -> Dict[str, Any]:
        details = await asyncio.to_thread(_mlflow().get_model_details, run_id)
        if details is None or details.get("user_id") != user_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_FOUND)
        return details

    def _owned_row(user_id: str, row: Any) -> None:
        if row is not None and str(getattr(row, "user_id", "")) != user_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_FOUND)

    # ------------------------------------------------------------------ list
    @router.get("/api/models")
    async def list_models(request: Request):
        user_id = _caller(request)
        models = await asyncio.to_thread(_mlflow().list_models)
        store = get_ml_store()
        endpoints = {e.mlflow_run_id: e for e in store.list_endpoints(user_id)}
        records = {m.mlflow_run_id: m for m in store.list_models(user_id)}
        items = []
        for m in models:
            if m.get("user_id") != user_id:
                continue
            run_id = str(m.get("mlflow_run_id") or "")
            items.append(decorate_model(m, endpoints.get(run_id), records.get(run_id)))
        return {"models": items}

    # ---------------------------------------------------------------- detail
    @router.get("/api/models/{run_id}")
    async def get_model_details(request: Request, run_id: str):
        user_id = _caller(request)
        details = await _owned_run(user_id, run_id, forbidden="You do not have permission to access this model.")
        store = get_ml_store()
        return decorate_model(details, store.get_endpoint(run_id), store.get_model(run_id))

    # -------------------------------------------------------------- endpoint
    @router.get("/api/models/{run_id}/endpoint")
    async def get_endpoint(request: Request, run_id: str):
        """Postgres only: cheap enough for the panel to poll every few seconds."""
        user_id = _caller(request)
        store = get_ml_store()
        ep = store.get_endpoint(run_id)
        model = store.get_model(run_id)
        _owned_row(user_id, ep)
        _owned_row(user_id, model)
        if ep is None and model is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No endpoint for this model.")
        return endpoint_payload(run_id, ep, model)

    # ------------------------------------------------------- cross-validation
    @router.get("/api/models/{run_id}/cross-validation")
    async def get_cross_validation(request: Request, run_id: str):
        """Per-fold and summed confusion matrices (classification) or fold metrics (regression)."""
        user_id = _caller(request)
        store = get_ml_store()
        cv = store.get_cross_validation(run_id)
        if cv is None or str(cv.user_id) != user_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, NOT_FOUND)
        return {"run_id": run_id, "status": cv.status, "k": cv.k, "splitter": cv.splitter, "reason": cv.reason,
                "aggregate": cv.aggregate, "folds": cv.folds or [], "updated_at": _iso(cv.updated_at)}

    # ------------------------------------------------------------------ jobs
    @router.get("/api/ml-jobs/{job_id}")
    async def get_job(request: Request, job_id: str):
        user_id = _caller(request)
        job = get_ml_store().get_job(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found.")
        _owned_row(user_id, job)
        return job_payload(job)

    @router.post("/api/ml-jobs/{job_id}/cancel")
    async def cancel_job(request: Request, job_id: str):
        """TI-13: cancel a queued/running training; the RayJob CR is deleted, the row is cancelled."""
        user_id = _caller(request)
        from app.infra import training_tasks

        job = await asyncio.to_thread(training_tasks.cancel_training_job, job_id, user_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found.")
        return job_payload(job)

    @router.get("/api/ml-jobs")
    async def list_jobs(request: Request, run_id: Optional[str] = Query(None),
                        kind: Optional[str] = Query(None), limit: int = Query(50, ge=1, le=200)):
        user_id = _caller(request)
        jobs = get_ml_store().list_jobs(user_id, mlflow_run_id=run_id, kind=kind, limit=limit)
        return {"jobs": [job_payload(j) for j in jobs]}

    # ---------------------------------------------------------------- delete
    @router.delete("/api/models/{run_id}")
    async def delete_model(request: Request, run_id: str):
        user_id = _caller(request)
        await _owned_run(user_id, run_id, forbidden="You do not have permission to delete this model.")
        store = get_ml_store()
        ep = store.get_endpoint(run_id)
        if ep is not None and ep.status not in (lifecycle.STOPPED, lifecycle.FAILED):
            from app.infra import training_tasks
            try:
                job = training_tasks.enqueue_stop_endpoint(user_id, run_id, dispatch=False)
                await asyncio.to_thread(training_tasks.run_stop_endpoint, job.id)
            except Exception:  # noqa: BLE001 - deleting must not be blocked by a dead backend
                logger.warning("[models] stop before delete failed for %s", run_id, exc_info=True)
        try:
            from app.core import ml_registry
            await asyncio.to_thread(ml_registry.delete_run_version, run_id)
        except Exception:  # noqa: BLE001
            logger.warning("[models] registry delete failed for %s", run_id, exc_info=True)
        store.delete_endpoint(run_id)
        store.delete_model(run_id)
        await asyncio.to_thread(_mlflow().delete_model, run_id)
        return {"status": "deleted"}

    # --------------------------------------------------------------- predict
    async def _predict(request: Request, run_id: str,
                       input_data: Union[Dict[str, Any], List[Dict[str, Any]]], response: Response):
        user_id = _caller(request)
        await require_premium_plan(user_id, "model_inference")
        store = get_ml_store()
        ep = store.get_endpoint(run_id)
        _owned_row(user_id, ep)
        if ep is None:
            # A pre-registry run: fall back to the MLflow owner check and its artifact flag.
            details = await _owned_run(user_id, run_id, forbidden="You do not have permission to use this model.")
            if not details.get("inference_service_details"):
                raise HTTPException(status.HTTP_409_CONFLICT, "Inference service is not configured for this model.")
        elif not lifecycle.is_servable(ep.status):
            raise HTTPException(status.HTTP_409_CONFLICT,
                                f"Inference endpoint is {ep.status}; it must be ready to serve predictions.")
        if isinstance(input_data, list) and (not input_data or not all(isinstance(r, dict) for r in input_data)):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Batch inference requires a non-empty JSON array of row objects.")
        try:
            from app.agents.mta_v2.inference_service_manager import InferenceServiceManager, ServingAsleep

            result = await asyncio.to_thread(InferenceServiceManager().inference, run_id, input_data)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
        except ServingAsleep as exc:
            # The pod is scaled to zero: say so and let the caller retry; never hold the connection.
            retry_after = int(getattr(exc, "retry_after", 15))
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                {"detail": "warming", "retry_after": retry_after},
                                headers={"Retry-After": str(retry_after)})
        except Exception as exc:  # noqa: BLE001
            logger.error("Inference gateway failed for run '%s': %s", run_id, exc, exc_info=True)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Inference gateway failed: {exc}")
        store.touch_endpoint(run_id)
        return {"result": result}

    @router.post("/api/models/{run_id}/predict")
    async def predict(request: Request, run_id: str, input_data: Union[Dict[str, Any], List[Dict[str, Any]]],
                      response: Response):
        return await _predict(request, run_id, input_data, response)

    @router.post("/api/models/{run_id}/inference")
    async def inference(request: Request, run_id: str, input_data: Union[Dict[str, Any], List[Dict[str, Any]]],
                        response: Response):
        """Alias of /predict kept for the inference panel."""
        return await _predict(request, run_id, input_data, response)

    # ------------------------------------------------------- configure / stop
    @router.post("/api/models/{run_id}/configure-inference-service", status_code=status.HTTP_202_ACCEPTED)
    async def configure_inference_service(request: Request, run_id: str, response: Response):
        user_id = _caller(request)
        # Starting a service spends hosted compute; stopping is never gated.
        await require_premium_plan(user_id, "model_inference")
        await _owned_run(user_id, run_id, forbidden="You do not have permission to use this model.")
        from app.infra import training_tasks

        store = get_ml_store()
        ep = store.get_endpoint(run_id)
        if ep is not None and ep.status == lifecycle.DEPLOYING:
            raise HTTPException(status.HTTP_409_CONFLICT, "The inference service is already starting.")
        if ep is not None and ep.status == lifecycle.READY and not endpoint_asleep(run_id, ep):
            # TI-09: already serving -> 200 with the current state, no job started.
            response.status_code = status.HTTP_200_OK
            return {"status": "ready", "job_id": None, "endpoint": endpoint_payload(run_id, ep, store.get_model(run_id))}
        try:
            job = await asyncio.to_thread(training_tasks.enqueue_deploy_endpoint, user_id, run_id, warm=True)
        except training_tasks.EndpointBusy:
            raise HTTPException(status.HTTP_409_CONFLICT, "The inference service is already starting.")
        except training_tasks.NoWorker:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                "No background worker is running to start the inference service.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Error enqueuing inference service deploy: %s", exc, exc_info=True)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Failed to configure inference service: {exc}")
        ep = store.get_endpoint(run_id)
        return {"status": "deploying", "job_id": job.id, "endpoint": endpoint_payload(run_id, ep, store.get_model(run_id))}

    @router.post("/api/models/{run_id}/stop-inference-service", status_code=status.HTTP_202_ACCEPTED)
    async def stop_inference_service(request: Request, run_id: str):
        user_id = _caller(request)
        await _owned_run(user_id, run_id, forbidden="You do not have permission to use this model.")
        from app.infra import training_tasks

        store = get_ml_store()
        try:
            job = await asyncio.to_thread(training_tasks.enqueue_stop_endpoint, user_id, run_id)
        except training_tasks.NoWorker:
            # The intent is already recorded (stopped); say the unload could not be dispatched.
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                "The endpoint is marked stopped, but no background worker is running to unload it.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Error enqueuing inference service stop: %s", exc, exc_info=True)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Failed to stop inference service: {exc}")
        ep = store.get_endpoint(run_id)
        return {"status": "stopping", "job_id": job.id, "endpoint": endpoint_payload(run_id, ep, store.get_model(run_id))}

    return router
