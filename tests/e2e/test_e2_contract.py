"""
Plan suite E2 -- API contract: health, version, whoami, OpenAPI surface freeze,
pagination reality, rate-limiting reality.

Hermetic (default): everything except the notes below. The app runs in-process
over the fakes from tests/test_server_integration.py, so the degraded-dependency
leg of /health, the /threads limit-clamp leg and the middleware introspection
leg are all observable.

Deployed-only: nothing in this suite *requires* a deployment; the legs that need
in-process monkeypatching or app-object introspection (E2.02 degraded, E2.03
default pin, E2.13 clamp, E2.14 middleware) skip when AVALOKA_API_URL is set.
"""

from __future__ import annotations

import os
from typing import Any, Dict, FrozenSet, List, Optional

import pytest

from tests.e2e.conftest import MODE, ApiClient, requires_deployed

# --------------------------------------------------------------------------- E2.01
# The frozen /health body. A new key here is a contract break for every k8s probe
# and every UI that destructures this object -- it must fail loudly, not silently.
HEALTH_KEYS: FrozenSet[str] = frozenset(
    {
        "status",
        "graph_ready",
        "langgraph_url",
        "assistant_id",
        "upstream_reachable",
        "upstream_timeout_s",
        "redis_connected",
        "redis_mode",
    }
)

# --------------------------------------------------------------------------- E2.09
# Committed OpenAPI surface snapshot: the 32 unique paths served by the 36 route
# decorators in app/api/server.py (app/api/cloud_connections.py declares none).
# Regenerate with:
#   python -c "import app.api.server as s; print(sorted(s.app.openapi()['paths']))"
# and update this literal *in the same commit* that adds or removes a route.
OPENAPI_PATH_SNAPSHOT: FrozenSet[str] = frozenset(
    {
        "/api/assets/{session_id}",
        "/api/assets/{session_id}/code",
        "/api/assets/{session_id}/job",
        "/api/assets/{session_id}/output",
        "/api/database/connect",
        "/api/datasets/{dataset_id}/background-task-status",
        "/api/missions/plan",
        "/api/models",
        "/api/models/{run_id}",
        "/api/models/{run_id}/configure-inference-service",
        "/api/models/{run_id}/inference",
        "/api/models/{run_id}/stop-inference-service",
        "/api/register-existing-storage",
        "/api/upload",
        "/api/v1/database/query",
        "/buckets/list",
        "/datasets",
        "/datasets/{dataset_id}",
        "/datasets/{dataset_id}/preview",
        "/debug/whoami",
        "/health",
        "/tasks",
        "/tasks/{task_id}",
        "/tasks/{task_id}/info",
        "/tasks/{task_id}/result/{index}",
        "/tasks/{task_id}/status",
        "/threads",
        "/threads/{thread_id}",
        "/threads/{thread_id}/code",
        "/threads/{thread_id}/messages",
        "/threads/{thread_id}/planner-graph",
        "/version",
    }
)

# Matches tests/k8s/test_t2_cluster_deployment.py:100 -- "is this the real API?".
MIN_OPENAPI_PATHS = 30

# --------------------------------------------------------------------------- E2.14
_RATE_LIMIT_MARKERS = ("ratelimit", "rate_limit", "slowapi", "limiter", "throttle")


def _requires_hermetic(reason: str) -> None:
    if MODE != "hermetic":
        pytest.skip(f"{reason} (needs the in-process app; unset AVALOKA_API_URL)")


def _openapi_spec(api: ApiClient) -> Dict[str, Any]:
    res = api.get("/openapi.json")
    assert res.status_code == 200, f"/openapi.json not served: HTTP {res.status_code}"
    return res.json()


def _openapi_paths(api: ApiClient) -> FrozenSet[str]:
    return frozenset(_openapi_spec(api).get("paths", {}))


def _query_param_names(spec: Dict[str, Any], path: str, method: str = "get") -> List[str]:
    op = spec["paths"][path][method]
    return [p["name"] for p in (op.get("parameters") or []) if p.get("in") == "query"]


# =========================================================================== E2.01


def test_e2_01_health_contract_freeze(api: ApiClient) -> None:
    """GET /health returns exactly the 8 frozen keys -- no more, no fewer."""
    res = api.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, dict)
    assert frozenset(body) == HEALTH_KEYS, (
        "GET /health key-set drifted. added="
        f"{sorted(frozenset(body) - HEALTH_KEYS)} removed={sorted(HEALTH_KEYS - frozenset(body))}. "
        "Update HEALTH_KEYS only together with every probe/UI consumer of this body."
    )


@pytest.mark.parametrize(
    "key, kind",
    [
        ("status", str),
        ("graph_ready", bool),
        ("langgraph_url", str),
        ("assistant_id", str),
        ("upstream_reachable", bool),
        ("upstream_timeout_s", (int, float)),
        ("redis_connected", bool),
        ("redis_mode", str),
    ],
)
def test_e2_01_health_field_types_frozen(api: ApiClient, key: str, kind: Any) -> None:
    """Each frozen /health field keeps its declared JSON type."""
    body = api.get("/health").json()
    assert isinstance(body[key], kind), f"/health[{key!r}] is {type(body[key]).__name__}"


# =========================================================================== E2.02


def test_e2_02_pins_status_is_hardcoded_ok(api: ApiClient) -> None:
    """/health "status" is the literal "ok" (server.py:764); it never reflects health."""
    assert api.get("/health").json()["status"] == "ok"


def test_e2_02_pins_status_stays_ok_when_dependencies_are_down(
    api: ApiClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With upstream + redis both failing, status is still "ok" and only the booleans flip."""
    _requires_hermetic("forcing dependency failure needs in-process monkeypatching")

    import app.api.server as server
    from app.services import session_service
    from tests import test_server_integration as si
    from fastapi import HTTPException, status as http_status

    # The app lifespan replaces session_service.cache with a real RedisCache, so a
    # healthy baseline has to be re-installed before it can be broken on purpose.
    monkeypatch.setattr(session_service, "cache", si.FakeCache())
    monkeypatch.setattr(session_service, "CACHE_DEGRADED_UNTIL", 0.0)

    baseline = api.get("/health").json()
    assert baseline["status"] == "ok"
    assert baseline["upstream_reachable"] is True
    assert baseline["redis_connected"] is True

    async def _upstream_down(*_a: Any, **_kw: Any) -> Any:
        raise HTTPException(http_status.HTTP_502_BAD_GATEWAY, "upstream down")

    async def _redis_down(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("redis down")

    monkeypatch.setattr(server, "lg_request", _upstream_down)
    monkeypatch.setattr(session_service.cache, "ping", _redis_down)

    res = api.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok", (
        "This pins the CURRENT contract: /health returns a hardcoded status. "
        "Probes and the UI must read the booleans, never `status`."
    )
    assert body["upstream_reachable"] is False
    assert body["redis_connected"] is False


def test_e2_02_health_stays_200_when_dependencies_are_down(api: ApiClient) -> None:
    """Degraded /health still answers HTTP 200 -- pinned deployed-side too."""
    requires_deployed("degraded-dependency /health against a real cluster")
    res = api.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# =========================================================================== E2.03


def test_e2_03_version_contract(api: ApiClient) -> None:
    """GET /version returns exactly {"version": <non-empty string>}."""
    res = api.get("/version")
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"version"}
    assert isinstance(body["version"], str) and body["version"]


def test_e2_03_pins_version_is_app_version_env_or_1_2_default(api: ApiClient) -> None:
    """/version echoes APP_VERSION (server.py:296), defaulting to the literal "1.2"."""
    _requires_hermetic("reading the server's APP_VERSION constant needs the in-process app")

    import app.api.server as server

    expected = os.getenv("APP_VERSION", "1.2")
    assert server.APP_VERSION == expected
    assert api.get("/version").json()["version"] == expected


# =========================================================================== E2.04


def test_e2_04_whoami_returns_subject_for_a_valid_token(api: ApiClient) -> None:
    """Valid bearer token -> {"user_id": <sub>}."""
    res = api.get("/debug/whoami", user="user-1")
    assert res.status_code == 200
    assert res.json() == {"user_id": "user-1"}


@pytest.mark.parametrize(
    "case, headers",
    [
        ("no_header", None),
        ("garbage_bearer", {"Authorization": "Bearer not-a-jwt"}),
        ("wrong_scheme", {"Authorization": "Basic dXNlcjpwYXNz"}),
        ("bearer_empty", {"Authorization": "Bearer "}),
        ("three_dot_garbage", {"Authorization": "Bearer aaa.bbb.ccc"}),
    ],
)
def test_e2_04_whoami_returns_null_user_never_500(
    api: ApiClient, case: str, headers: Optional[Dict[str, str]]
) -> None:
    """Missing/garbage credentials -> 200 {"user_id": null}, never a 500."""
    res = api.get("/debug/whoami", headers=headers)
    assert res.status_code == 200, f"{case}: HTTP {res.status_code}"
    assert res.json() == {"user_id": None}, case


# =========================================================================== E2.09


def test_e2_09_openapi_surface_no_path_removed_or_renamed(api: ApiClient) -> None:
    """No path in the committed OpenAPI snapshot may disappear or be renamed."""
    live = _openapi_paths(api)
    missing = sorted(OPENAPI_PATH_SNAPSHOT - live)
    assert not missing, (
        f"OpenAPI paths removed or renamed: {missing}. Removing/renaming a served path "
        "breaks existing clients -- restore it, or delete it from OPENAPI_PATH_SNAPSHOT "
        "in the same commit with a migration note."
    )


def test_e2_09_openapi_surface_no_undeclared_path_added(api: ApiClient) -> None:
    """Any newly served path must be added to OPENAPI_PATH_SNAPSHOT deliberately."""
    live = _openapi_paths(api)
    added = sorted(live - OPENAPI_PATH_SNAPSHOT)
    assert not added, (
        f"New API paths are being served that the surface snapshot does not know about: {added}. "
        "If this is intentional, add them to OPENAPI_PATH_SNAPSHOT in tests/e2e/test_e2_contract.py "
        "and confirm each one authenticates (see the E7 auth sweep)."
    )


def test_e2_09_openapi_path_count_floor(api: ApiClient) -> None:
    """The served OpenAPI advertises >= 30 paths -- proof this is the API, not a stub."""
    live = _openapi_paths(api)
    assert len(live) >= MIN_OPENAPI_PATHS, (
        f"expected the real API (>={MIN_OPENAPI_PATHS} paths), got {len(live)}"
    )


@pytest.mark.parametrize("path", ["/health", "/version", "/debug/whoami", "/api/upload", "/threads"])
def test_e2_09_openapi_advertises_load_bearing_paths(api: ApiClient, path: str) -> None:
    """The paths probes, the UI and the k8s suite depend on are advertised."""
    assert path in _openapi_paths(api)


# =========================================================================== E2.13


@pytest.mark.parametrize(
    "requested, clamped",
    [(-10, 1), (0, 1), (1, 1), (50, 50), (200, 200), (201, 200), (100000, 200)],
)
def test_e2_13_threads_clamps_limit_into_1_200(
    api: ApiClient, monkeypatch: pytest.MonkeyPatch, requested: int, clamped: int
) -> None:
    """GET /threads clamps ?limit into [1, 200] before forwarding to /threads/search."""
    _requires_hermetic("observing the upstream search payload needs the in-process app")

    import app.api.server as server

    seen: Dict[str, Any] = {}

    async def _capture(method: str, path: str, **kw: Any) -> Dict[str, Any]:
        if path == "/threads/search":
            seen["payload"] = kw.get("json") or {}
        return {"items": []}

    monkeypatch.setattr(server, "lg_json", _capture)

    res = api.get("/threads", user="user-1", params={"limit": requested})
    assert res.status_code == 200
    assert seen["payload"]["limit"] == clamped


def test_e2_13_threads_default_limit_is_50(
    api: ApiClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /threads with no ?limit forwards limit=50."""
    _requires_hermetic("observing the upstream search payload needs the in-process app")

    import app.api.server as server

    seen: Dict[str, Any] = {}

    async def _capture(method: str, path: str, **kw: Any) -> Dict[str, Any]:
        if path == "/threads/search":
            seen["payload"] = kw.get("json") or {}
        return {"items": []}

    monkeypatch.setattr(server, "lg_json", _capture)

    assert api.get("/threads", user="user-1").status_code == 200
    assert seen["payload"] == {"limit": 50}


def test_e2_13_threads_forwards_q_and_page_token(
    api: ApiClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /threads forwards ?q as `query` and ?page_token verbatim."""
    _requires_hermetic("observing the upstream search payload needs the in-process app")

    import app.api.server as server

    seen: Dict[str, Any] = {}

    async def _capture(method: str, path: str, **kw: Any) -> Dict[str, Any]:
        if path == "/threads/search":
            seen["payload"] = kw.get("json") or {}
        return {"items": []}

    monkeypatch.setattr(server, "lg_json", _capture)

    res = api.get(
        "/threads", user="user-1", params={"q": "sales", "page_token": "tok-2", "limit": 7}
    )
    assert res.status_code == 200
    assert seen["payload"] == {"limit": 7, "query": "sales", "page_token": "tok-2"}


def test_e2_13_pins_threads_is_the_only_paged_collection(api: ApiClient) -> None:
    """Only GET /threads declares paging params; /tasks and /datasets declare none."""
    spec = _openapi_spec(api)
    assert sorted(_query_param_names(spec, "/threads")) == ["limit", "page_token", "q"]


@pytest.mark.parametrize("path", ["/tasks", "/datasets"])
def test_e2_13_pins_no_pagination_on_tasks_and_datasets(api: ApiClient, path: str) -> None:
    """GET /tasks (4419) and GET /datasets (4837) declare zero query params -- unbounded by design-so-far."""
    spec = _openapi_spec(api)
    params = _query_param_names(spec, path)
    assert params == [], (
        f"GET {path} now declares query params {params}. This test pinned the decision that "
        f"{path} returns an unbounded collection; if paging was added, move it into the paged set."
    )


def test_e2_13_pins_datasets_returns_unbounded_collection(
    api: ApiClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /datasets?limit=1 still returns every dataset -- ?limit is silently ignored."""
    _requires_hermetic("seeding the session store needs the in-process fakes")

    from tests import test_server_integration as si

    seeded = 5
    for i in range(seeded):
        si.SESSIONS[f"sess-e2-13-{i}"] = {
            "user_id": "user-1",
            "dataset_id": f"ds-e2-13-{i}",
            "data_source_location": f"gs://fake-bucket/test/ds-e2-13-{i}.csv",
            "object_name": f"ds-e2-13-{i}.csv",
            "schema": {"a": "int"},
        }

    unpaged = api.get("/datasets", user="user-1")
    limited = api.get("/datasets", user="user-1", params={"limit": 1})
    assert unpaged.status_code == 200 and limited.status_code == 200
    assert len(unpaged.json()) == seeded
    assert len(limited.json()) == seeded, (
        "?limit was honoured by /datasets -- pagination has been added; this pin must be revisited."
    )


# =========================================================================== E2.14


def test_e2_14_pins_no_rate_limiter_installed(api: ApiClient) -> None:
    """app.user_middleware holds exactly CORS + the PNA preflight handler -- no rate limiter."""
    _requires_hermetic("middleware introspection needs the in-process app object")

    import app.api.server as server

    installed = [mw.cls.__name__ for mw in server.app.user_middleware]
    assert installed == ["BaseHTTPMiddleware", "CORSMiddleware"], (
        f"middleware stack changed: {installed}. This test pins the decision that Avaloka ships "
        "with NO rate limiting; update it deliberately when one is added."
    )

    dispatch = server.app.user_middleware[0].kwargs.get("dispatch")
    assert getattr(dispatch, "__name__", "") == "handle_private_network_preflight"

    blob = " ".join(
        f"{mw.cls.__module__}.{mw.cls.__name__}" for mw in server.app.user_middleware
    ).lower()
    hits = [marker for marker in _RATE_LIMIT_MARKERS if marker in blob]
    assert not hits, f"a rate limiter appears to be installed ({hits}); update this pin"


def test_e2_14_pins_repeated_requests_are_never_throttled(api: ApiClient) -> None:
    """30 back-to-back /health calls all return 200 -- nothing 429s."""
    codes = {api.get("/health").status_code for _ in range(30)}
    assert codes == {200}, f"unexpected status codes under a burst: {sorted(codes)}"
