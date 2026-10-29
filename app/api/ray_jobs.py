"""/api/ray/*: the caller's own Ray jobs, for the Ray jobs panel in the UI.

The browser never talks to Ray: the Ray dashboard has no login and its Jobs API
runs any code it is sent. These routes check the login and the plan, then
app/infra/ray_jobs_service.py checks ownership and talks to Ray.

Wired in app/api/server.py the same way as the analytics router:
    app.include_router(build_router(_resolve_user_id, _require_premium_plan))
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.infra import ray_jobs_service as svc

# Same plan gate as starting a transfer from chat (app/core/plan_gate.py).
FEATURE = "data_transfer"


def build_router(
    resolve_user_id: Callable[[Request], Optional[str]],
    require_premium: Callable[[str, str], Awaitable[None]],
) -> APIRouter:
    router = APIRouter(prefix="/api/ray", tags=["ray"])

    async def _caller(request: Request) -> str:
        user_id = resolve_user_id(request)
        if not user_id:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
        await require_premium(user_id, FEATURE)  # 403 with an upgrade payload
        return user_id

    async def _run(fn, *args, **kwargs):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except svc.RayJobsError as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc

    @router.get("/status")
    async def ray_status(request: Request):
        await _caller(request)
        return await _run(svc.cluster_status)

    @router.get("/jobs")
    async def ray_jobs(
        request: Request,
        kind: Optional[str] = Query(None, max_length=32),
        status_: Optional[str] = Query(None, alias="status", max_length=16),
        limit: int = Query(50, ge=1, le=200),
    ):
        user_id = await _caller(request)
        return {"jobs": await _run(svc.list_jobs, user_id, kind=kind, status=status_, limit=limit)}

    @router.get("/jobs/{submission_id}")
    async def ray_job(submission_id: str, request: Request):
        user_id = await _caller(request)
        return await _run(svc.get_job, user_id, submission_id)

    @router.get("/jobs/{submission_id}/logs")
    async def ray_job_logs(submission_id: str, request: Request, tail: int = Query(500, ge=1, le=5000)):
        user_id = await _caller(request)
        return await _run(svc.get_logs, user_id, submission_id, tail=tail)

    @router.post("/jobs/{submission_id}/stop")
    async def ray_job_stop(submission_id: str, request: Request):
        user_id = await _caller(request)
        return await _run(svc.stop_job, user_id, submission_id)

    return router
