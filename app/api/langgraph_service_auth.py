"""Service-token gate for the LangGraph server (``langgraph dev``).

The LangGraph server is internal: the API is its only caller, and uses it as a
thread registry (``POST /threads``, ``POST /threads/search``,
``DELETE /threads/{id}``, ``GET /ok``; app/api/server.py ``lg_request``). The
server itself has no authentication. This module adds it.

How it is wired: ``langgraph.json`` names this module's ``app`` under
``http.app``. langgraph-api merges a custom Starlette app into its own and puts
that app's middleware in front of its global middleware
(langgraph_api/server.py, ``app.user_middleware = custom_middleware +
global_middleware``), so :class:`ServiceTokenMiddleware` runs before every route
the server has, its meta routes included.

Why not langgraph's own auth hook (``langgraph.json`` ``auth.path``):

* it guards only the "protected" routes. The meta routes stay open, and under
  ``langgraph dev`` those include ``POST /internal/truncate``, which deletes
  every thread and checkpoint (langgraph_api/api/__init__.py);
* under ``langgraph dev`` a request carrying ``x-auth-scheme: langsmith`` skips
  the hook altogether unless ``disable_studio_auth`` is set
  (langgraph_api/auth/custom.py, ``StudioNoopAuthBackend``);
* it answers 403, not 401.

A reverse-proxy sidecar would also work, but it adds a container, an image to
mirror and pin, and a second port to keep off the network. This needs none.

The rule:

* every request needs ``Authorization: Bearer <LANGGRAPH_SERVICE_TOKEN>``,
  compared in constant time; anything else gets 401 with no detail about why;
* ``GET /ok`` is the one exception. It answers ``{"ok": true}`` and nothing
  else; the CI health check (bitbucket-pipelines.yml) and the probes use it;
* no usable token configured (unset, empty, or a value published in this
  repository) means every request is refused. It fails closed;
* requests that langgraph-api makes to itself through its in-process loopback
  transport (ASGI scope ``root_path`` ``/noauth``, which no network client can
  set) pass, as they do through langgraph's own auth.

The token is self-provisioned: the Helm charts generate it into a Secret and
keep it across upgrades. A local ``langgraph dev`` generates it into ``.env``
(app/core/provisioning.py) at import, as the API does at its own start, so the
two processes agree whichever starts first.
"""

from __future__ import annotations

import hmac
import json
import logging
import os

from fastapi import FastAPI
from starlette.middleware import Middleware

logger = logging.getLogger(__name__)

TOKEN_ENV = "LANGGRAPH_SERVICE_TOKEN"

#: The only route answered without the token: method -> paths.
OPEN_ROUTES = {"GET": frozenset({"/ok"}), "HEAD": frozenset({"/ok"})}

_UNAUTHORIZED = json.dumps({"detail": "Unauthorized"}).encode()


def _usable(value: str | None) -> bool:
    try:
        from app.core.provisioning import is_usable
    except Exception:  # pragma: no cover - the repo is always importable here
        return bool((value or "").strip())
    return is_usable(value)


def expected_token() -> str:
    """The configured token, or "" when none is usable (which refuses every call)."""
    value = os.environ.get(TOKEN_ENV, "")
    return value.strip() if _usable(value) else ""


def presented_token(headers) -> str:
    """The bearer token in an ASGI header list, or ""."""
    for name, value in headers or ():
        if name.lower() == b"authorization":
            scheme, _, token = value.decode("latin-1").strip().partition(" ")
            if scheme.lower() == "bearer":
                return token.strip()
            return ""
    return ""


def is_authorized(headers) -> bool:
    expected = expected_token()
    presented = presented_token(headers)
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


class ServiceTokenMiddleware:
    """Pure ASGI middleware: 401 unless the request carries the service token."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        kind = scope.get("type")
        if kind not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        if (scope.get("root_path") or "").startswith("/noauth"):
            return await self.app(scope, receive, send)
        if kind == "http" and scope.get("path") in OPEN_ROUTES.get(scope.get("method", ""), ()):
            return await self.app(scope, receive, send)
        if is_authorized(scope.get("headers")):
            return await self.app(scope, receive, send)
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_UNAUTHORIZED)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        })
        await send({"type": "http.response.body", "body": _UNAUTHORIZED})


def _provision_locally() -> None:
    """Outside Kubernetes, generate the token into .env when none is set (never printed)."""
    if _usable(os.environ.get(TOKEN_ENV)):
        return
    try:
        from app.core.provisioning import in_kubernetes, provision, default_env_file
    except Exception:  # pragma: no cover
        return
    if in_kubernetes() or os.getenv("AVALOKA_AUTO_PROVISION", "1").strip() == "0":
        logger.error("%s is not set: the LangGraph server refuses every call except GET /ok", TOKEN_ENV)
        return
    try:
        provision(default_env_file(), [TOKEN_ENV])
    except OSError as exc:
        logger.error("could not provision %s: %s; the LangGraph server refuses every call",
                     TOKEN_ENV, type(exc).__name__)


_provision_locally()

#: langgraph.json ``http.app``. No routes of its own: only the middleware.
#: FastAPI, not a bare Starlette: for a Starlette app langgraph-api 0.5 builds
#: its OpenAPI spec by parsing every route's docstring as YAML, and its own MCP
#: route's docstring does not parse, so the server fails to start. Its default
#: /docs, /redoc and /openapi.json are off so they cannot shadow langgraph's.
app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None,
              middleware=[Middleware(ServiceTokenMiddleware)])
