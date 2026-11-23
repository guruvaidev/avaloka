"""The GoTrue subset supabase-js needs, served by the API in local auth mode.

See :mod:`app.core.local_auth` for the design and the refusal rules. The UI
points its Supabase client at ``<ui origin>/api/public/local-auth``, which the
UI server forwards to ``<API>/local-auth``; supabase-js then appends
``/auth/v1/...`` exactly as it would for a real project.

Every route answers 404 unless local auth is active, so mounting this router
in a Supabase deployment adds nothing anyone can use.

``/local-auth/rest/v1/*`` is the UI's PostgREST traffic (profiles, plans,
saved reports). Local mode has no Supabase database, so reads answer an empty
list, which the UI already renders as "nothing yet", and writes answer 501
with a plain reason rather than pretending to have stored anything.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from app.core import local_auth

router = APIRouter(prefix="/local-auth", tags=["local-auth"], include_in_schema=False)

_NOT_ACTIVE = {"code": 404, "error_code": "not_found", "msg": "Local auth mode is not enabled."}


def _err(status: int, code: str, msg: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": status, "error_code": code, "msg": msg})


def _bearer(request: Request) -> str:
    h = request.headers.get("Authorization", "")
    return h.split(" ", 1)[1].strip() if h.startswith("Bearer ") else ""


@router.post("/auth/v1/token")
async def token(request: Request):
    cfg = local_auth.active_config()
    if cfg is None:
        return JSONResponse(status_code=404, content=_NOT_ACTIVE)
    grant = request.query_params.get("grant_type", "")
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    if grant == "password":
        if local_auth.locked_out():
            return _err(429, "over_request_rate_limit", "Too many failed sign-ins. Wait five minutes.")
        if not local_auth.check_password(cfg, str(body.get("email") or ""), str(body.get("password") or "")):
            return _err(400, "invalid_credentials", "Invalid login credentials")
        return local_auth.issue_session(cfg)
    if grant == "refresh_token":
        if local_auth.verify(str(body.get("refresh_token") or ""), cfg, typ="refresh") is None:
            return _err(400, "refresh_token_not_found", "Invalid Refresh Token")
        return local_auth.issue_session(cfg)
    return _err(400, "unsupported_grant_type", "Local auth supports only password sign-in.")


@router.get("/auth/v1/user")
async def user(request: Request):
    cfg = local_auth.active_config()
    if cfg is None:
        return JSONResponse(status_code=404, content=_NOT_ACTIVE)
    if local_auth.verify(_bearer(request), cfg) is None:
        return _err(401, "bad_jwt", "invalid JWT")
    return local_auth.user_object(cfg)


@router.post("/auth/v1/logout")
async def logout():
    # Stateless tokens: the browser drops its copy. Nothing to revoke here.
    if local_auth.active_config() is None:
        return JSONResponse(status_code=404, content=_NOT_ACTIVE)
    return Response(status_code=204)


@router.api_route("/auth/v1/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def other_auth(rest: str):
    if local_auth.active_config() is None:
        return JSONResponse(status_code=404, content=_NOT_ACTIVE)
    return _err(501, "not_implemented", "Not available in local auth mode (one local account, password sign-in only).")


@router.api_route("/rest/v1/{rest:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"])
async def rest(rest: str, request: Request):
    cfg = local_auth.active_config()
    if cfg is None:
        return JSONResponse(status_code=404, content=_NOT_ACTIVE)
    if local_auth.verify(_bearer(request), cfg) is None:
        return JSONResponse(status_code=401, content={"message": "invalid JWT"})
    if request.method in ("GET", "HEAD"):
        return JSONResponse(status_code=200, content=[], headers={"Content-Range": "*/0"})
    return JSONResponse(
        status_code=501,
        content={"code": "PGRST501", "message": "Local auth mode has no Supabase database; this was not saved."},
    )
