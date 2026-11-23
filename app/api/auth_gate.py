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

Query parameters are validated the same way, so a GET with a required query
field (``GET /catalog/resolve`` without ``ref``) answers an anonymous caller 422
too. ``methods`` widens the gate to cover reads: the 1.7 API passes
``GATED_METHODS`` and lists its public reads in ``exempt``. HEAD is gated with
GET and matched against a GET exemption, so a HEAD of a public read is answered
by routing exactly as before (FastAPI routes do not take HEAD: 405).
OPTIONS is never gated: a CORS preflight carries no credentials.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Pattern, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import compile_path
from starlette.types import ASGIApp, Receive, Scope, Send

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
#: Every method a route can validate input on: the mutating ones plus reads.
GATED_METHODS = MUTATING_METHODS | {"GET", "HEAD"}
UNAUTH_DETAIL = "Missing or invalid auth token"


class AuthBeforeBodyMiddleware:
    """401 for an anonymous POST/PUT/PATCH/DELETE (or any of ``methods``), before
    the body or query is touched.

    ``resolve_user_id`` must read only headers (it is given a ``Request`` with no
    receive channel). ``exempt`` is ``(METHOD, "/path/{template}")`` pairs that
    stay reachable without a token; keep it short and say why for each entry.
    ``conditional_exempt`` pairs are reachable without a token only while
    ``conditional_applies()`` is true; it is asked per request, so it reads the
    deployment as it is now, not as it was at import.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        resolve_user_id: Callable[[Request], Optional[str]],
        exempt: Iterable[Tuple[str, str]] = (),
        methods: Iterable[str] = MUTATING_METHODS,
        conditional_exempt: Iterable[Tuple[str, str]] = (),
        conditional_applies: Callable[[], bool] = lambda: False,
    ) -> None:
        self.app = app
        self._resolve_user_id = resolve_user_id
        self._methods = frozenset(m.upper() for m in methods) - {"OPTIONS"}
        self._exempt = self._compile(exempt)
        self._conditional = self._compile(conditional_exempt)
        self._conditional_applies = conditional_applies

    @staticmethod
    def _compile(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, Pattern[str]]]:
        return [(method.upper(), compile_path(path)[0]) for method, path in pairs]

    def _is_exempt(self, method: str, path: str) -> bool:
        if method == "HEAD":
            method = "GET"
        if any(m == method and rx.match(path) for m, rx in self._exempt):
            return True
        return (any(m == method and rx.match(path) for m, rx in self._conditional)
                and self._conditional_applies())

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in self._methods:
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
