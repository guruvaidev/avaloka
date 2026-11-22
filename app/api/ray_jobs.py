"""/api/ray/*: Ray for the signed-in user.

Two parts:

1. Ray's own dashboard at /ray/ (the open-source UI, nothing rebuilt here).
   The browser opens /ray/ in a new tab, so it cannot send the app's bearer
   token. Instead:
     POST /api/ray/session  (bearer token)  -> sets a short-lived, signed
                                               cookie scoped to /ray/
     GET  /api/ray/authz    (that cookie)   -> called by the nginx proxy for
                                               EVERY /ray/ request; answers
                                               200 + X-Ray-Upstream, or 401/403
   Ray itself has no login and its Jobs API runs any code it is sent, so the
   proxy forwards nothing that authz has not approved.

2. The older panel endpoints (status, jobs, job, logs, stop), kept for now so
   nothing that still calls them breaks; app/infra/ray_jobs_service.py checks
   ownership and talks to Ray.

Wired in app/api/server.py, unchanged:
    app.include_router(build_router(_resolve_user_id, _require_premium_plan))
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import time
from typing import Awaitable, Callable, Iterable, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from app.infra import ray_jobs_service as svc

# ---------------------------------------------------------------- plan gate --
# Ray runs data transfers, large scheduled jobs and model training, so access is
# not tied to "data_transfer" alone. The caller passes if their plan allows ANY
# of these features (app/core/plan_gate.py). Add "ray_dashboard" to the
# Professional and Enterprise plans there to make the rule explicit; until then
# the other three cover the same plans. Override with AVALOKA_RAY_FEATURES.
RAY_FEATURES: tuple[str, ...] = tuple(
    f.strip()
    for f in os.getenv(
        "AVALOKA_RAY_FEATURES",
        "ray_dashboard,data_transfer,scheduled_large_jobs,model_training",
    ).split(",")
    if f.strip()
)

# ------------------------------------------------------------ /ray/ session --
COOKIE_NAME = "avaloka_ray"
COOKIE_PATH = "/ray/"
SESSION_TTL_S = int(os.getenv("RAY_SESSION_TTL_S", "1800"))  # ~30 minutes

# An open /ray/ tab keeps the user's cluster alive (POST /heartbeat) only while
# it is in use: after this many minutes without activity in the tab, the UI
# stops sending heartbeats and the cluster's own idle shutdown applies.
# 0 = an open tab keeps the cluster alive however long it sits unused.
IDLE_MINUTES = max(0, int(os.getenv("RAY_DASHBOARD_IDLE_MINUTES", "15")))

# Shown while a stopped per-user cluster is being started.
STARTING_MESSAGE = os.getenv(
    "RAY_DASHBOARD_STARTING_MESSAGE",
    "Starting your Ray cluster. This usually takes a few minutes; this page "
    "opens the dashboard as soon as it is ready.",
)

# Where /ray/ goes when no per-user cluster is known: the deployment's own Ray
# head. Correct for local kind and for an Enterprise (self-hosted) deployment.
DEFAULT_UPSTREAM = os.getenv(
    "RAY_DASHBOARD_UPSTREAM",
    "avaloka-raycluster-head-svc.default.svc.cluster.local:8265",
)

# host:port only; anything else in a cookie or hook result is refused.
_UPSTREAM_RE = re.compile(r"^[A-Za-z0-9.\-]{1,253}:\d{2,5}$")

# Inside /ray/: read anything; the only write allowed is stopping a job.
# Submitting, deleting jobs and Serve changes are refused.
_READ_METHODS = {"GET", "HEAD"}
_STOP_RE = re.compile(r"^/ray/api/jobs/(?!\.+/)[A-Za-z0-9_.\-]{1,128}/stop/?(\?.*)?$")


def _session_secret() -> bytes:
    secret = os.getenv("RAY_SESSION_SECRET") or os.getenv("SUPABASE_JWT_SECRET") or ""
    if not secret:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Ray dashboard sessions are not configured")
    return secret.encode()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_session(payload: dict, secret: bytes) -> str:
    body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    sig = _b64(hmac.new(secret, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_session(token: str, secret: bytes, now: Optional[float] = None) -> Optional[dict]:
    """The payload of a valid, unexpired cookie, else None."""
    try:
        body, sig = (token or "").split(".", 1)
        expected = _b64(hmac.new(secret, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_unb64(body))
    except Exception:
        return None
    if not isinstance(payload, dict) or float(payload.get("exp", 0)) < (now or time.time()):
        return None
    if not _UPSTREAM_RE.match(str(payload.get("upstream", ""))):
        return None
    return payload


def is_allowed_action(method: str, uri: str) -> bool:
    """View anything; stop a job; nothing else."""
    method = (method or "").upper()
    if method in _READ_METHODS:
        return True
    return method == "POST" and bool(_STOP_RE.match(uri or ""))


def _resolve_upstream(user_id: str) -> dict:
    """The user's Ray cluster.

    Professional users get their own Infra-agent cluster. When the Infra agent
    exposes ``user_ray_cluster(user_id)`` in app/infra/ray_jobs_service.py it is
    used here, returning {"state": "running"|"starting"|"stopped",
    "upstream": "host:port"}, or None for a user without a cluster of their
    own, who goes to the deployment's shared cluster (local kind, Enterprise).
    """
    hook = getattr(svc, "user_ray_cluster", None)
    info = hook(user_id) if callable(hook) else None
    if info is not None:  # None: no cluster of their own, use the shared one
        state = str(info.get("state") or "stopped")
        upstream = str(info.get("upstream") or "")
        if state == "running" and not _UPSTREAM_RE.match(upstream):
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Ray cluster address is invalid")
        return {"state": state, "upstream": upstream}
    return {"state": "running", "upstream": DEFAULT_UPSTREAM}


def _cookie_secure() -> bool:
    return os.getenv("COOKIE_SECURE", "").lower() in ("1", "true", "yes")


def _cookie_samesite() -> str:
    """lax (default) when /ray/ is on the app's own site; none when the UI and
    /ray/ are on different sites (needs HTTPS, so Secure is forced)."""
    value = os.getenv("RAY_COOKIE_SAMESITE", "lax").strip().lower()
    return value if value in ("lax", "strict", "none") else "lax"


def build_router(
    resolve_user_id: Callable[[Request], Optional[str]],
    require_premium: Callable[[str, str], Awaitable[None]],
    features: Iterable[str] = RAY_FEATURES,
) -> APIRouter:
    router = APIRouter(prefix="/api/ray", tags=["ray"])
    features = tuple(features)

    async def _require_any_feature(user_id: str) -> None:
        last: Optional[HTTPException] = None
        for feature in features:
            try:
                await require_premium(user_id, feature)
                return
            except HTTPException as exc:  # 403 with an upgrade payload
                last = exc
            except Exception:              # feature unknown to plan_gate
                continue
        raise last or HTTPException(status.HTTP_403_FORBIDDEN, "Your plan does not include Ray")

    async def _caller(request: Request) -> str:
        user_id = resolve_user_id(request)
        if not user_id:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
        await _require_any_feature(user_id)
        return user_id

    async def _run(fn, *args, **kwargs):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except svc.RayJobsError as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc

    # ------------------------------------------------ Ray's own dashboard --
    @router.post("/session")
    async def ray_session(request: Request, response: Response):
        """Bearer-authenticated: set the /ray/ cookie and say where to go."""
        user_id = await _caller(request)
        cluster = await asyncio.to_thread(_resolve_upstream, user_id)
        if cluster["state"] != "running":
            starter = getattr(svc, "start_user_ray_cluster", None)
            if cluster["state"] == "stopped" and callable(starter):
                await asyncio.to_thread(starter, user_id)
                cluster["state"] = "starting"
            message = STARTING_MESSAGE if cluster["state"] == "starting" else "Your Ray cluster is not running."
            return {"state": cluster["state"], "url": None, "message": message,
                    "idle_minutes": IDLE_MINUTES}

        token = sign_session(
            {"sub": user_id, "upstream": cluster["upstream"], "exp": int(time.time()) + SESSION_TTL_S},
            _session_secret(),
        )
        samesite = _cookie_samesite()
        response.set_cookie(
            COOKIE_NAME, token, max_age=SESSION_TTL_S, path=COOKIE_PATH,
            httponly=True, samesite=samesite,
            secure=_cookie_secure() or samesite == "none",
        )
        return {"state": "running", "url": COOKIE_PATH, "idle_minutes": IDLE_MINUTES}

    @router.get("/authz", include_in_schema=False)
    async def ray_authz(request: Request):
        """For the nginx proxy only (auth_request): one yes/no per /ray/ request."""
        payload = verify_session(request.cookies.get(COOKIE_NAME, ""), _session_secret())
        if not payload:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to Avaloka again")
        method = request.headers.get("X-Original-Method", "GET")
        uri = request.headers.get("X-Original-URI", "/ray/")
        if not is_allowed_action(method, uri):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "This action is not available in the dashboard")
        return Response(status_code=200, headers={"X-Ray-Upstream": payload["upstream"]})

    @router.post("/heartbeat")
    async def ray_heartbeat(request: Request):
        """While /ray/ is open and in use: keep the user's cluster from idling out."""
        user_id = await _caller(request)
        touch = getattr(svc, "touch_user_ray_cluster", None)
        if callable(touch):
            await asyncio.to_thread(touch, user_id)
            return {"ok": True, "idle_minutes": IDLE_MINUTES}
        return {"ok": True, "idle_minutes": IDLE_MINUTES, "note": "no per-user cluster to keep alive"}

    # ------------------------------------- older panel endpoints (kept) --
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
