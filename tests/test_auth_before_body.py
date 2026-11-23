"""
Guard: an anonymous request gets 401 before its body or query is validated.

FastAPI parses and validates the body and the query before it calls a handler,
and the handlers resolve their caller by hand inside the body of the function.
So an anonymous ``{}`` to a route with a required body field used to answer 422
and list the missing fields -- a free schema probe -- and so did a GET missing a
required query field (``GET /catalog/resolve``). Router handlers that check their
store or body before the caller answered 200 or 400 the same way.
``app/api/auth_gate.py`` now answers 401 in front of routing.

The sweep is GENERATED from the app's route table, including routes inside
included routers (FastAPI keeps those as ``_IncludedRouter`` entries with no
``path`` of their own, which a flat walk of ``app.routes`` silently skips) and
routes hidden from the schema. A route added later is swept automatically; the
only way out is ALLOW_LIST below, which must match ``server.AUTH_GATE_EXEMPT``
exactly.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set, Tuple

import pytest
from fastapi import params as fastapi_params
from starlette.routing import compile_path

import app.api.server as server

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import client, make_auth_headers  # noqa: F401  (fixtures are used by name)
except ImportError:
    from server_harness import client, make_auth_headers  # noqa: F401

UNAUTH_DETAIL = "Missing or invalid auth token"
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
GATED = MUTATING | {"GET"}  # HEAD is gated with GET; no FastAPI route takes HEAD itself

_LOCAL_AUTH = ("the GoTrue subset for AVALOKA_AUTH_MODE=local: how a caller gets a token, "
               "404 unless local mode is on, and its session routes check the bearer themselves")

# Reviewed allow-list: (method, path) -> why it may be reached without a token.
# Must equal server.AUTH_GATE_EXEMPT, so the gate cannot be widened in one place.
ALLOW_LIST: Dict[Tuple[str, str], str] = {
    ("GET", "/health"): "liveness/readiness probe (Helm, compose)",
    ("GET", "/version"): "the release string",
    ("GET", "/api/capabilities"): "public by owner ruling: feature discovery read before sign-in; "
    "anonymous callers get the least-privileged plan's limits",
    ("GET", "/analytics/config"): "tells a signed-out browser enabled=false so the UI stops",
    **{(m, "/local-auth/{rest:path}"): _LOCAL_AUTH for m in ("GET", "POST", "PUT", "PATCH", "DELETE")},
}

# Public except on the hosted deployment, where they need a login (owner ruling).
# Must equal server.AUTH_GATE_EXEMPT_UNLESS_HOSTED.
UNLESS_HOSTED: Dict[Tuple[str, str], str] = {
    ("GET", "/openapi.json"): "FastAPI schema page",
    ("GET", "/docs"): "FastAPI docs page",
    ("GET", "/docs/oauth2-redirect"): "FastAPI docs page",
    ("GET", "/redoc"): "FastAPI docs page",
    ("GET", "/debug/whoami"): "reports who the token resolves to; null when it does not",
}
_ALLOW_RX = [(m, compile_path(p)[0]) for m, p in {**ALLOW_LIST, **UNLESS_HOSTED}]

# The routes the hermetic e2e sweep (E2.05) flagged on develop-1.7 once it walked
# into included routers: each answered an anonymous caller before looking at it.
# These eight are FastAPI body/query validation (422); the analytics two are the
# collector's own store and body checks (200 / 400).
SCHEMA_PROBED = [
    ("GET", "/catalog/resolve"),
    ("POST", "/analysis/{analysis_id}/feedback"),
    ("POST", "/analysis/{analysis_id}/insights/explain"),
    ("POST", "/analysis/{analysis_id}/restore"),
    ("POST", "/analysis/{analysis_id}/save-and-execute"),
    ("POST", "/api/database/tables-to-analysis"),
    ("POST", "/api/integrations/github/connect"),
    ("POST", "/api/register-existing-folder"),
]
HANDLER_CHECKED_FIRST = [("POST", "/analytics/events"), ("PUT", "/analytics/opt-out")]


def _walk(routes: Iterable[Any], found: Set[Tuple[str, str]], unknown: List[str]) -> None:
    for route in routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path and methods:
            for method in set(methods) & GATED:
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


def _gated_routes() -> List[Tuple[str, str]]:
    found: Set[Tuple[str, str]] = set()
    unknown: List[str] = []
    _walk(server.app.routes, found, unknown)
    # Fail loudly rather than sweep a partial table.
    assert not unknown, f"route types the guard cannot enumerate: {sorted(set(unknown))}"
    return sorted(found)


def _allowed(method: str, path: str) -> bool:
    return any(m == method and rx.match(path) for m, rx in _ALLOW_RX)


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


SWEPT = [r for r in _gated_routes() if not _allowed(*r)]
SWEPT_WRITES = [r for r in SWEPT if r[0] in MUTATING]
SWEPT_READS = [r for r in SWEPT if r[0] not in MUTATING]
JSON_BODY_ROUTES = [r for r in SWEPT_WRITES if _takes_json_body(*r)]

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


def test_inventory_covers_every_operation_in_openapi() -> None:
    """The walk must see at least what the OpenAPI schema publishes, so a FastAPI
    internals change cannot shrink the sweep without failing here."""
    published = {
        (m.upper(), p)
        for p, ops in server.app.openapi()["paths"].items()
        for m in ops
        if m.upper() in GATED
    }
    missing = published - set(_gated_routes())
    assert not missing, f"guard inventory misses published routes: {sorted(missing)}"
    assert len(SWEPT_WRITES) >= 25, f"write sweep collapsed to {len(SWEPT_WRITES)} routes"
    assert len(SWEPT_READS) >= 25, f"read sweep collapsed to {len(SWEPT_READS)} routes"


def test_inventory_reaches_included_routers() -> None:
    """The analytics, connections and catalog routers are mounted with include_router;
    a flat walk of app.routes never sees them."""
    routes = set(_gated_routes())
    assert {("POST", "/analytics/events"), ("GET", "/catalog/resolve"),
            ("GET", "/api/connections")} <= routes


def test_allow_list_matches_the_gate_and_is_live() -> None:
    gate = set(getattr(server, "AUTH_GATE_EXEMPT", ()))
    assert set(ALLOW_LIST) == gate, (
        "server.AUTH_GATE_EXEMPT and this test's reviewed ALLOW_LIST disagree: "
        f"gate-only={sorted(gate - set(ALLOW_LIST))} test-only={sorted(set(ALLOW_LIST) - gate)}"
    )
    live = _gated_routes()
    stale = [(m, p) for m, p in ALLOW_LIST
             if not any(lm == m and compile_path(p)[0].match(lp) for lm, lp in live)]
    assert not stale, f"allow-list names routes that no longer exist: {sorted(stale)}"


def test_unless_hosted_list_matches_the_gate() -> None:
    gate = set(getattr(server, "AUTH_GATE_EXEMPT_UNLESS_HOSTED", ()))
    assert set(UNLESS_HOSTED) == gate, (
        "server.AUTH_GATE_EXEMPT_UNLESS_HOSTED and this test's UNLESS_HOSTED disagree: "
        f"gate-only={sorted(gate - set(UNLESS_HOSTED))} test-only={sorted(set(UNLESS_HOSTED) - gate)}"
    )
    assert not set(UNLESS_HOSTED) & set(getattr(server, "AUTH_GATE_EXEMPT", ())), (
        "a route on both lists would stay public on the hosted deployment"
    )


def test_schema_probed_routes_are_swept() -> None:
    assert set(SCHEMA_PROBED) | set(HANDLER_CHECKED_FIRST) <= set(SWEPT)


# --------------------------------------------------------------------------- the sweep


@pytest.mark.parametrize("route", SWEPT_WRITES, ids=[_rid(r) for r in SWEPT_WRITES])
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


@pytest.mark.parametrize("route", SWEPT_READS, ids=[_rid(r) for r in SWEPT_READS])
@pytest.mark.parametrize("anon", sorted(ANON_HEADERS), ids=sorted(ANON_HEADERS))
def test_anonymous_bare_read_is_401(client, route, anon) -> None:
    """The bare path omits every required query field."""
    method, path = route
    resp = client.request(method, _concrete(path), headers=ANON_HEADERS[anon])
    assert resp.status_code == 401, f"{method} {path} anon={anon} -> {resp.status_code} {resp.text[:200]}"
    assert resp.json().get("detail") == UNAUTH_DETAIL


def test_anonymous_head_of_a_gated_read_is_401(client) -> None:
    # Before the gate this was routing's 405; the gate answers before routing.
    assert client.head("/catalog/resolve").status_code == 401


# --------------------------------------------------------------------------- still validated when signed in


@pytest.mark.parametrize("route", SCHEMA_PROBED, ids=[_rid(r) for r in SCHEMA_PROBED])
def test_authenticated_schema_invalid_request_is_still_422(client, route) -> None:
    method, path = route
    kw = {"json": {}} if method in MUTATING else {}
    resp = client.request(method, _concrete(path), headers=make_auth_headers("user-1"), **kw)
    assert resp.status_code == 422, f"{method} {path} -> {resp.status_code} {resp.text[:200]}"


def test_authenticated_opt_out_body_check_is_still_400(client) -> None:
    resp = client.put("/analytics/opt-out", json={}, headers=make_auth_headers("user-1"))
    assert resp.status_code == 400, resp.text


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

# A concrete request for each allow-list entry; the local-auth prefix is probed
# on the routes supabase-js calls.
_PUBLIC_PROBES = sorted(
    {(m, p) for m, p in ALLOW_LIST if "{" not in p}
    | {("POST", "/local-auth/auth/v1/token"), ("GET", "/local-auth/auth/v1/user"),
       ("POST", "/local-auth/auth/v1/logout"), ("GET", "/local-auth/rest/v1/profiles"),
       ("PUT", "/local-auth/auth/v1/user"), ("PATCH", "/local-auth/rest/v1/profiles"),
       ("DELETE", "/local-auth/rest/v1/profiles")}
)


def test_public_probes_cover_the_allow_list() -> None:
    uncovered = [(m, p) for m, p in ALLOW_LIST
                 if not any(pm == m and compile_path(p)[0].match(pp) for pm, pp in _PUBLIC_PROBES)]
    assert not uncovered, uncovered


@pytest.mark.parametrize("route", _PUBLIC_PROBES, ids=[_rid(r) for r in _PUBLIC_PROBES])
def test_allow_listed_route_answers_an_anonymous_caller(client, route) -> None:
    method, path = route
    resp = client.request(method, path, json={} if method in MUTATING else None)
    assert resp.status_code != 401, f"{method} {path} -> 401 {resp.text[:200]}"
    assert resp.status_code < 500, f"{method} {path} -> {resp.status_code} {resp.text[:200]}"


def test_local_auth_answers_in_its_own_shape_not_the_gates(client, monkeypatch) -> None:
    """With local mode on, a missing session is the route's GoTrue-shaped 401,
    which supabase-js parses, not the gate's {"detail": ...}."""
    from app.core import local_auth

    monkeypatch.setattr(local_auth, "active_config", lambda: object())
    monkeypatch.setattr(local_auth, "verify", lambda *a, **k: None)
    resp = client.get("/local-auth/auth/v1/user")
    assert resp.status_code == 401
    assert resp.json().get("error_code") == "bad_jwt"


def test_head_of_a_public_read_and_preflight_are_untouched(client) -> None:
    assert client.head("/health").status_code != 401
    preflight = client.options(
        "/api/register-existing-folder",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert preflight.status_code != 401


# --------------------------------------------------------------------------- hosted vs self-hosted

# AVALOKA_DEPLOYMENT as each state is declared (None: unset).
DEPLOYMENT_VALUES = {"hosted": "avaloka_hosted", "self_hosted": "self_hosted", "undeclared": None}


def _declare(monkeypatch, state: str) -> None:
    value = DEPLOYMENT_VALUES[state]
    if value is None:
        monkeypatch.delenv("AVALOKA_DEPLOYMENT", raising=False)
    else:
        monkeypatch.setenv("AVALOKA_DEPLOYMENT", value)


@pytest.mark.parametrize("route", sorted(UNLESS_HOSTED), ids=[_rid(r) for r in sorted(UNLESS_HOSTED)])
@pytest.mark.parametrize("state", sorted(DEPLOYMENT_VALUES))
@pytest.mark.parametrize("caller", ["anonymous", "signed_in"])
def test_docs_and_whoami_need_a_login_only_on_hosted(client, monkeypatch, route, state, caller) -> None:
    method, path = route
    _declare(monkeypatch, state)
    headers = make_auth_headers("user-1") if caller == "signed_in" else {}
    resp = client.request(method, path, headers=headers)
    if state == "hosted" and caller == "anonymous":
        assert resp.status_code == 401, f"hosted: anonymous {method} {path} -> {resp.status_code}"
        assert resp.json().get("detail") == UNAUTH_DETAIL
    else:
        assert resp.status_code == 200, f"{state}/{caller}: {method} {path} -> {resp.status_code} {resp.text[:200]}"


def test_signed_in_whoami_on_hosted_names_the_caller(client, monkeypatch) -> None:
    _declare(monkeypatch, "hosted")
    assert client.get("/debug/whoami", headers=make_auth_headers("user-1")).json() == {"user_id": "user-1"}


@pytest.mark.parametrize("state", sorted(DEPLOYMENT_VALUES))
def test_capabilities_stay_public_in_every_deployment(client, monkeypatch, state) -> None:
    _declare(monkeypatch, state)
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200, f"{state}: /api/capabilities -> {resp.status_code}"


def _deployment_state():
    from app.core import editions

    reader = getattr(editions, "deployment_state", None)
    assert callable(reader), "app.core.editions has no deployment_state(): no single deployment signal"
    return editions, reader


@pytest.mark.parametrize("value,state", [
    ("avaloka_hosted", "hosted"), ("avaloka-hosted", "hosted"), (" Self_Hosted ", "self_hosted"),
    ("self-hosted", "self_hosted"), (None, "undeclared"), ("", "undeclared"), ("hosted", "undeclared"),
])
def test_deployment_state_reads_the_declared_value(value, state) -> None:
    _, reader = _deployment_state()
    env = {} if value is None else {"AVALOKA_DEPLOYMENT": value}
    assert reader(env) == state


def test_an_undeclared_deployment_warns_once_and_names_no_value(monkeypatch, caplog) -> None:
    editions, reader = _deployment_state()
    monkeypatch.setattr(editions, "_undeclared_warned", False)
    with caplog.at_level("WARNING", logger=editions.__name__):
        for _ in range(3):
            assert reader({"AVALOKA_DEPLOYMENT": "customer-acme-prod"}) == "undeclared"
        assert reader({}) == "undeclared"
    warnings = [r for r in caplog.records if r.name == editions.__name__ and "AVALOKA_DEPLOYMENT" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    message = warnings[0].getMessage()
    assert "customer-acme-prod" not in message, "the warning repeats the configured value"
    assert "self_hosted" in message and "avaloka_hosted" in message


def test_a_declared_deployment_does_not_warn(monkeypatch, caplog) -> None:
    editions, reader = _deployment_state()
    monkeypatch.setattr(editions, "_undeclared_warned", False)
    with caplog.at_level("WARNING", logger=editions.__name__):
        reader({"AVALOKA_DEPLOYMENT": "self_hosted"})
        reader({"AVALOKA_DEPLOYMENT": "avaloka_hosted"})
    assert not [r for r in caplog.records if r.name == editions.__name__]


@pytest.mark.parametrize("value,billed", [("avaloka_hosted", True), ("self_hosted", False), (None, True), ("hosted", True)])
def test_billing_reads_the_same_signal(monkeypatch, value, billed) -> None:
    """Undeclared stays billed exactly as before the hosted value existed."""
    from app.core import plan_limits

    for name in ("AVALOKA_PLAN_LIMITS", "AVALOKA_EDITION", "AVALOKA_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv("AVALOKA_DEPLOYMENT", value)
    assert plan_limits._billing_disabled() is (not billed)
