"""Authenticate state-changing requests before FastAPI reads the body.

Every handler resolves its caller by hand (``_resolve_user_id`` in the handler
body), but FastAPI parses and validates the request BEFORE it calls the handler:
``get_request_handler`` reads the body (``request.json()`` / ``request.form()``)
and ``solve_dependencies`` validates it against the Pydantic model, and a
failure there is a 422 that names the missing fields. So an anonymous caller who
posts ``{}`` to a route with a required body field learns its schema, and the
in-handler 401 is never reached. A ``Depends`` guard does not close this either:
the JSON decode (a ``json_invalid`` 422) and the multipart parse both happen
before any dependency runs.

This middleware sits in front of routing. For POST/PUT/PATCH/DELETE it resolves
the caller from the headers alone and answers 401 before the body is read, so no
route -- including one added later, or one inside an included router -- can be
reached by an anonymous caller unless it is listed in ``exempt``. A caller who IS
authenticated passes straight through, and FastAPI still answers a bad body 422.

The handlers keep their own ``_resolve_user_id`` call; this is the outer gate,
not a replacement for the per-route identity and ownership checks.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Pattern, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import compile_path
from starlette.types import ASGIApp, Receive, Scope, Send

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
UNAUTH_DETAIL = "Missing or invalid auth token"


class AuthBeforeBodyMiddleware:
    """401 for an anonymous POST/PUT/PATCH/DELETE, before the body is touched.

    ``resolve_user_id`` must read only headers (it is given a ``Request`` with no
    receive channel). ``exempt`` is ``(METHOD, "/path/{template}")`` pairs that
    stay reachable without a token; keep it short and say why for each entry.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        resolve_user_id: Callable[[Request], Optional[str]],
        exempt: Iterable[Tuple[str, str]] = (),
    ) -> None:
        self.app = app
        self._resolve_user_id = resolve_user_id
        self._exempt: List[Tuple[str, Pattern[str]]] = [
            (method.upper(), compile_path(path)[0]) for method, path in exempt
        ]

    def _is_exempt(self, method: str, path: str) -> bool:
        return any(m == method and rx.match(path) for m, rx in self._exempt)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in MUTATING_METHODS:
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path):
            path = path[len(root_path):] or "/"

        if self._is_exempt(scope["method"], path) or self._resolve_user_id(Request(scope)):
            await self.app(scope, receive, send)
            return

        response = JSONResponse({"detail": UNAUTH_DETAIL}, status_code=401)
        await response(scope, receive, send)
