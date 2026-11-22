"""
Guard: an anonymous POST/PUT/PATCH/DELETE gets 401 before the body is validated.

FastAPI parses and validates the body before it calls a handler, and the handlers
resolve their caller by hand inside the body of the function. So an anonymous
``{}`` to a route with a required body field used to answer 422 and list the
missing fields -- a free schema probe. ``app/api/auth_gate.py`` now answers 401
in front of routing.

The sweep is GENERATED from the app's route table, including routes inside
included routers (FastAPI keeps those as ``_IncludedRouter`` entries with no
``path`` of their own, which a flat walk of ``app.routes`` silently skips). A
route added later is swept automatically; the only way out is ALLOW_LIST below,
which must match ``server.AUTH_GATE_EXEMPT`` exactly.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Set, Tuple

import pytest
from fastapi import params as fastapi_params

import app.api.server as server

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import client, make_auth_headers  # noqa: F401  (fixtures are used by name)
except ImportError:
    from server_harness import client, make_auth_headers  # noqa: F401

UNAUTH_DETAIL = "Missing or invalid auth token"
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}

# Reviewed allow-list: (method, path) -> why it may be reached without a token.
# Must equal server.AUTH_GATE_EXEMPT, so the gate cannot be widened in one place.
ALLOW_LIST = {
    ("POST", "/threads"): "known fail-open E2.05a (not public by design); pinned by "
    "strict xfails in tests/e2e/test_e2_auth.py and fixed separately",
}

# The routes the hermetic e2e sweep (E2.05) flagged: an anonymous {} answered 422
# because each body model has required fields.
SCHEMA_PROBED = [
    ("POST", "/analysis/{analysis_id}/feedback"),
    ("POST", "/analysis/{analysis_id}/insights/explain"),
    ("POST", "/analysis/{analysis_id}/restore"),
    ("POST", "/analysis/{analysis_id}/save-and-execute"),
    ("POST", "/api/database/tables-to-analysis"),
    ("POST", "/api/integrations/github/connect"),
    ("POST", "/api/register-existing-folder"),
]


def _walk(routes: Iterable[Any], found: Set[Tuple[str, str]], unknown: List[str]) -> None:
    for route in routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path and methods:
            for method in set(methods) & MUTATING:
                found.add((method, path))
        elif hasattr(route, "effective_candidates"):  # FastAPI's _IncludedRouter
            _walk(route.effective_candidates(), found, unknown)
        elif getattr(route, "routes", None):  # a Mount of a sub-router
            prefix = getattr(route, "path", "") or ""
            sub: Set[Tuple[str, str]] = set()
            _walk(route.routes, sub, unknown)
            found.update((m, prefix + p) for m, p in sub)
        elif type(route).__name__ not in {"APIWebSocketRoute", "WebSocketRoute"}:
            unknown.append(f"{type(route).__module__}.{type(route).__name__}")


def _mutating_routes() -> List[Tuple[str, str]]:
    found: Set[Tuple[str, str]] = set()
    unknown: List[str] = []
    _walk(server.app.routes, found, unknown)
    # Fail loudly rather than sweep a partial table.
    assert not unknown, f"route types the guard cannot enumerate: {sorted(set(unknown))}"
    return sorted(found)


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "guard-probe", path)


def _route_object(method: str, path: str) -> Any:
    stack = list(server.app.routes)
    while stack:
        route = stack.pop()
        if hasattr(route, "effective_candidates"):
            stack.extend(route.effective_candidates())
        elif getattr(route, "path", None) == path and method in (getattr(route, "methods", None) or ()):
            return route
    return None


def _takes_json_body(method: str, path: str) -> bool:
    field = getattr(_route_object(method, path), "body_field", None)
    return field is not None and not isinstance(getattr(field, "field_info", None), fastapi_params.Form)


SWEPT = [r for r in _mutating_routes() if r not in ALLOW_LIST]
JSON_BODY_ROUTES = [r for r in SWEPT if _takes_json_body(*r)]

INVALID_BODIES = {
    "schema_invalid": {"json": {}},
    "malformed_json": {"content": b'{"unterminated": ', "headers": {"Content-Type": "application/json"}},
}
ANON_HEADERS = {
    "no_header": {},
    "garbage_bearer": {"Authorization": "Bearer aaa.bbb.ccc"},
}


def _rid(route: Tuple[str, str]) -> str:
    return f"{route[0]} {route[1]}"


# --------------------------------------------------------------------------- inventory


def test_inventory_covers_every_mutating_operation_in_openapi() -> None:
    """The walk must see at least what the OpenAPI schema publishes, so a FastAPI
    internals change cannot shrink the sweep without failing here."""
    published = {
        (m.upper(), p)
        for p, ops in server.app.openapi()["paths"].items()
        for m in ops
        if m.upper() in MUTATING
    }
    missing = published - set(_mutating_routes())
    assert not missing, f"guard inventory misses published routes: {sorted(missing)}"
    assert len(SWEPT) >= 25, f"sweep collapsed to {len(SWEPT)} routes"


def test_inventory_reaches_included_routers() -> None:
    """The analytics and Ray routers are mounted with include_router; a flat walk
    of app.routes never sees them."""
    paths = {p for _, p in _mutating_routes()}
    assert {"/analytics/events", "/api/ray/jobs/{submission_id}/stop"} <= paths


def test_allow_list_matches_the_gate_and_is_live() -> None:
    gate = set(getattr(server, "AUTH_GATE_EXEMPT", ()))
    assert set(ALLOW_LIST) == gate, (
        "server.AUTH_GATE_EXEMPT and this test's reviewed ALLOW_LIST disagree: "
        f"gate-only={sorted(gate - set(ALLOW_LIST))} test-only={sorted(set(ALLOW_LIST) - gate)}"
    )
    stale = set(ALLOW_LIST) - set(_mutating_routes())
    assert not stale, f"allow-list names routes that no longer exist: {sorted(stale)}"


def test_schema_probed_routes_are_swept() -> None:
    assert set(SCHEMA_PROBED) <= set(SWEPT)


# --------------------------------------------------------------------------- the sweep


@pytest.mark.parametrize("route", SWEPT, ids=[_rid(r) for r in SWEPT])
@pytest.mark.parametrize("body", sorted(INVALID_BODIES), ids=sorted(INVALID_BODIES))
@pytest.mark.parametrize("anon", sorted(ANON_HEADERS), ids=sorted(ANON_HEADERS))
def test_anonymous_invalid_body_is_401(client, route, body, anon) -> None:
    method, path = route
    kw = dict(INVALID_BODIES[body])
    kw["headers"] = {**kw.get("headers", {}), **ANON_HEADERS[anon]}
    resp = client.request(method, _concrete(path), **kw)
    assert resp.status_code == 401, (
        f"{method} {path} anon={anon} body={body} -> {resp.status_code} {resp.text[:200]}"
    )
    assert resp.json().get("detail") == UNAUTH_DETAIL


# --------------------------------------------------------------------------- still 422 when signed in


@pytest.mark.parametrize("route", SCHEMA_PROBED, ids=[_rid(r) for r in SCHEMA_PROBED])
def test_authenticated_schema_invalid_body_is_still_422(client, route) -> None:
    method, path = route
    resp = client.request(method, _concrete(path), json={}, headers=make_auth_headers("user-1"))
    assert resp.status_code == 422, f"{method} {path} -> {resp.status_code} {resp.text[:200]}"


@pytest.mark.parametrize("route", JSON_BODY_ROUTES, ids=[_rid(r) for r in JSON_BODY_ROUTES])
def test_authenticated_malformed_json_is_still_422(client, route) -> None:
    method, path = route
    resp = client.request(
        method,
        _concrete(path),
        content=b'{"unterminated": ',
        headers={**make_auth_headers("user-1"), "Content-Type": "application/json"},
    )
    assert resp.status_code == 422, f"{method} {path} -> {resp.status_code} {resp.text[:200]}"


# --------------------------------------------------------------------------- what the gate leaves alone


def test_public_reads_and_preflight_are_untouched(client) -> None:
    assert client.get("/health").status_code != 401
    preflight = client.options(
        "/api/register-existing-folder",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert preflight.status_code != 401
