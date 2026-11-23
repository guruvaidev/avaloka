"""
Shared harness for the L5 API pack (plan suites E2, E3, E7).

Every test in this package runs unchanged in two modes:

  hermetic  (default)  in-process FastAPI TestClient over the in-memory fake
                       stack from tests/test_server_unit.py. No Docker,
                       no cluster, no keys -- safe as a PR gate.
  deployed             black-box HTTP against a real deployment, selected by
                       exporting AVALOKA_API_URL. Tokens are minted with
                       SUPABASE_JWT_SECRET so they verify like portal tokens.

    export AVALOKA_API_URL=http://localhost:9000
    export SUPABASE_JWT_SECRET=<the cluster's secret>
    pytest tests/e2e -m "not cluster"     # hermetic legs only
    pytest tests/e2e                      # everything the mode supports

Cases that can only be observed against a real deployment are marked
`cluster`; they skip in hermetic mode instead of failing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import jwt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DEPLOYED_URL = (os.getenv("AVALOKA_API_URL") or "").rstrip("/")
MODE = "deployed" if DEPLOYED_URL else "hermetic"

# Not set here: MEMORY_WARMUP. app.services.memory_plane reads it once, at import
# (memory_plane.py, `if os.environ.get("MEMORY_WARMUP", "1") != "0"`), to decide
# whether to spawn a warm-up thread that loads the embedding model. Setting it
# from this conftest leaked into the whole session -- pytest imports every
# conftest during collection -- and a fixture would run after that import, so
# would change nothing. Whether the test session warms up is a gate-wide choice
# (scripts/ci.sh or the root conftest), not this pack's.
# tests/test_conftest_env_hygiene.py fails if a conftest writes os.environ.

# Hermetic mode signs with the same constant tests/test_server_unit.py
# patches into the server; deployed mode must use the cluster's real secret.
HERMETIC_SECRET = "test-jwt-secret"
JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET") if MODE == "deployed" else HERMETIC_SECRET

UNAUTH_MESSAGE = "Missing or invalid auth token"

# Hermetic mode answers every upload-limit plan lookup (server.resolve_plan) with
# this plan, instead of a Supabase lookup that fails and falls back to Free
# with an error per request. Tests that depend on a plan's limits say so.
# Feature gates (check_feature / try_resolve_plan) are not pinned: they fail open.
HERMETIC_PLAN = "free"

# Fake mcp_connections table, keyed by connection id; reset per hermetic client.
MCP_CONNECTIONS: Dict[str, Dict[str, Any]] = {}

# mcp_connections.id is a uuid column, and the server answers 404 for any
# connection id that is not a uuid before it looks anything up. Fixed uuids, so
# a probe reaches the lookup and the ownership check rather than that pre-check.
# OWNED is user A's row (foreign from user B's side); MISSING matches no row.
MCP_CONNECTION_OWNED = "5e3c1f2a-7b4d-4e6f-9a1b-0c2d3e4f5a01"
MCP_CONNECTION_MISSING = "5e3c1f2a-7b4d-4e6f-9a1b-0c2d3e4f5a02"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "defect: pins a confirmed defect; the assertion states intended behaviour",
    )


def mint_token(
    user_id: str,
    *,
    secret: Optional[str] = None,
    expires_in: int = 3600,
    claims: Optional[Dict[str, Any]] = None,
    algorithm: str = "HS256",
) -> str:
    """Mint a Supabase-shaped HS256 token. Used for both happy and attack paths."""
    now = int(time.time())
    payload: Dict[str, Any] = {"sub": user_id, "iat": now, "exp": now + expires_in}
    if claims:
        payload.update(claims)
        for key, value in list(claims.items()):
            if value is None:
                payload.pop(key, None)
    return jwt.encode(payload, secret or JWT_SECRET or "", algorithm=algorithm)


def auth_headers(user_id: str, **kw: Any) -> Dict[str, str]:
    return {"Authorization": f"Bearer {mint_token(user_id, **kw)}"}


# The identity every schema read uses. Nothing in the pack reads /openapi.json
# without a token, so the sweep and the surface checks keep working wherever the
# schema sits behind authentication.
SCHEMA_READER = "e2-schema-reader"


def schema_headers() -> Dict[str, str]:
    return auth_headers(SCHEMA_READER)


class ApiClient:
    """
    Thin uniform wrapper over TestClient / httpx so a test body reads the same
    in both modes. `user=` mints and attaches a bearer token; omit it to probe
    the unauthenticated path.
    """

    def __init__(self, transport: Any, mode: str):
        self._t = transport
        self.mode = mode

    def request(
        self,
        method: str,
        path: str,
        *,
        user: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        session: Optional[str] = None,
        **kw: Any,
    ):
        merged: Dict[str, str] = {}
        if user is not None:
            merged.update(auth_headers(user))
        if session is not None:
            merged["X-Avaloka-Session"] = session
        if headers:
            merged.update(headers)
        kw.setdefault("timeout", 30)
        return self._t.request(method, path, headers=merged or None, **kw)

    def get(self, path: str, **kw: Any):
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw: Any):
        return self.request("POST", path, **kw)

    def delete(self, path: str, **kw: Any):
        return self.request("DELETE", path, **kw)

    def options(self, path: str, **kw: Any):
        return self.request("OPTIONS", path, **kw)

    def stream(self, method: str, path: str, *, user: Optional[str] = None, **kw: Any):
        headers = auth_headers(user) if user is not None else None
        return self._t.stream(method, path, headers=headers, **kw)


def _hermetic_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Rebuild the fake stack from tests/test_server_unit.py so the L5 pack
    shares one definition of the fakes rather than forking a second one.
    """
    from fastapi.testclient import TestClient

    import app.api.server as server
    from app.services import session_service, storage_service
    from tests import test_server_unit as si

    # Hermetic means hermetic. settings.storage_backend defaults to "gcs" and is
    # evaluated at import (app/core/settings.py:9), so storage_service builds a
    # real GCSBlobStore (storage_service.py:248) and any machine without a
    # service-account key dies with DefaultCredentialsError. Patch the resolved
    # setting - not the env var, which is read too early to matter - before the
    # app boots, and drop ambient cloud credentials.
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    for module in (server, storage_service):
        if hasattr(module, "settings"):
            monkeypatch.setattr(module.settings, "storage_backend", "local", raising=False)

    si.SESSIONS.clear()
    si.THREAD_TO_SESSION.clear()
    server.THREAD_META.clear()

    monkeypatch.setattr(server, "JWT_SECRET", HERMETIC_SECRET, raising=False)

    tmp_root = tmp_path / "avaloka_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(server, "TMP_ROOT", tmp_root, raising=False)
    monkeypatch.setenv("TMP_ROOT", str(tmp_root))

    fake_cache = si.FakeCache()
    session_service.cache = fake_cache  # type: ignore[attr-defined]
    fake_store = si.FakeBlobStore(bucket="fake-bucket", prefix="test")
    storage_service.blob_store = fake_store  # type: ignore[attr-defined]

    # The fixture enters the client with `with`, which runs the app's lifespan,
    # and app/api/config.py's lifespan reassigns both of the fakes above:
    # `session_service.cache = await _init_cache()` (None when no Redis URL is
    # set) and `blob_store = storage_service.init_blob_store()`. Every handler
    # that checks the cache then answered 503 "Session storage (Redis) is not
    # available" -- the fake stack this pack claims to run over was never the
    # one it ran over. Hand the lifespan the same fakes instead.
    async def _init_cache_fake():
        return fake_cache

    monkeypatch.setattr(session_service, "_init_cache", _init_cache_fake)
    monkeypatch.setattr(storage_service, "init_blob_store", lambda: fake_store)

    # The lifespan also warms the local embedding model, which downloads it from
    # huggingface.co on a cold cache. Hermetic means no network.
    monkeypatch.setenv("AVALOKA_EMBED_WARMUP", "0")

    def _store_and_key_from_uri_fake(uri: str, object_name: Optional[str]):
        return fake_store, object_name or Path(uri).name

    async def _store_from_connection_uri_fake(storage_uri: str, conn: Dict[str, Any]):
        store = si.FakeBlobStore()
        store.objects["foo.csv"] = b"a,b\n1,2\n"
        return store, ""

    async def get_cloud_connection_fake(connection_id: str) -> Dict[str, Any]:
        return {}

    def _resolve_plan_fake(user_id: Optional[str]) -> str:
        return HERMETIC_PLAN

    def sample_data_from_source_fake(path, source_type, stratify_by=None, sample_size: float = 1.0):
        return {
            "schema": {"a": "int", "b": "int"},
            "rows": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            "ddl_schema": "CREATE TABLE t(a int, b int);",
        }

    MCP_CONNECTIONS.clear()

    def _owned_row_fake(connection_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        # Stands in for the saved-connections lookup the MCP credential routes
        # make on 1.7 (app.api.db_connections.owned_row: the caller's own row,
        # else None, so another user's row and a missing one are the same 404).
        # Without it every MCP route answered 500 "Connection lookup failed" (no
        # store) and the ownership check behind it was never reached.
        row = MCP_CONNECTIONS.get(connection_id)
        return dict(row) if row and str(row.get("user_id")) == str(user_id) else None

    monkeypatch.setattr(server._db_connections, "owned_row", _owned_row_fake)

    def _resolve_plan_fake(user_id: Optional[str]) -> str:
        return HERMETIC_PLAN

    for name, value in [
        ("save_session", si.save_session_fake),
        # Must pair with save_session: the upload rollback calls delete_session,
        # and against the real one it deleted from the cache while the session
        # lived in the fake store -- a rolled-back upload looked half-kept.
        ("delete_session", si.delete_session_fake),
        ("resolve_plan", _resolve_plan_fake),
        # Would post upload profiles to a real Supabase project if real
        # credentials were loaded (e.g. a developer's .env); in CI none are set
        # and the call ends at "Supabase env vars not set". Stubbed in
        # test_e3_ingestion.py before; every module here uploads, so it
        # belongs in the shared harness.
        ("_schedule_small_file_persistence", lambda **kw: None),
        ("update_session", si.update_session_fake),
        ("get_session", si.get_session_fake),
        ("refresh_session_ttl", si.refresh_session_ttl_fake),
        ("find_session_by_dataset_for_user", si.find_session_by_dataset_for_user_fake),
        ("bind_thread_session", si.bind_thread_session_fake),
        ("get_thread_session", si.get_thread_session_fake),
        ("_user_datasets", si._user_datasets_fake),
        ("_jsonify", si._jsonify_fake),
        ("_store_and_key_from_uri", _store_and_key_from_uri_fake),
        ("_store_from_connection_uri", _store_from_connection_uri_fake),
        ("get_cloud_connection", get_cloud_connection_fake),
        ("sample_data_from_source", sample_data_from_source_fake),
        ("lg_request", si.lg_request_fake),
        ("lg_json", si.lg_json_fake),
        ("GRAPH", si.FakeGraph()),
        ("GRAPH_READY", True),
        ("read_thread_msgs", si.read_thread_msgs_fake),
        ("resolve_plan", _resolve_plan_fake),
    ]:
        monkeypatch.setattr(server, name, value, raising=False)

    monkeypatch.setattr(
        storage_service, "_store_and_key_from_uri", _store_and_key_from_uri_fake, raising=False
    )

    if hasattr(server, "settings"):
        server.settings.storage_backend = "gcs"
        server.settings.gcs_bucket = "fake-bucket"
        server.settings.gcs_prefix = "test/"

    return TestClient(server.app)


@pytest.fixture()
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ApiClient]:
    if MODE == "deployed":
        import httpx

        if not JWT_SECRET:
            pytest.skip("deployed mode needs SUPABASE_JWT_SECRET to mint tokens")
        with httpx.Client(base_url=DEPLOYED_URL, follow_redirects=False) as http:
            yield ApiClient(http, "deployed")
    else:
        client = _hermetic_transport(tmp_path, monkeypatch)
        with client:
            yield ApiClient(client, "hermetic")


@pytest.fixture(scope="session")
def api_mode() -> str:
    return MODE


def requires_deployed(reason: str = "needs a running deployment") -> None:
    if MODE != "deployed":
        pytest.skip(f"{reason} (export AVALOKA_API_URL to run)")


# ---------------------------------------------------------------------------
# Route inventory -- the auth sweep is generated from the app, never hand-listed,
# so a route added tomorrow is swept tomorrow (plan risk: "no-Depends auth model").
# ---------------------------------------------------------------------------

# Public by design; everything else must reject an anonymous caller.
PUBLIC_ROUTES = {
    ("GET", "/health"),
    ("GET", "/version"),
    ("GET", "/debug/whoami"),
    # The analytics collector answers an anonymous browser 200 with
    # enabled=false, reason="unauthenticated" by design (app/analytics/collector.py).
    ("GET", "/analytics/config"),
}

# Placeholder values used when a swept path has parameters. They must not exist,
# so an authenticated probe lands on the ownership check rather than real data.
PATH_PARAM_SAMPLE = {
    "thread_id": "thread-does-not-exist",
    "task_id": "task-does-not-exist",
    "dataset_id": "dataset-does-not-exist",
    "run_id": "run-does-not-exist",
    "job_id": "job-does-not-exist",
    "submission_id": "submission-does-not-exist",
    "session_id": "session-does-not-exist",
    "connection_id": MCP_CONNECTION_OWNED,
    "index": "0",
}

# Feature discovery answers an anonymous caller with the least-privileged plan's
# limits by design (server._caller_edition). Kept apart from the set above so the
# included-router inventory hunk stays identical to the trainfer branch's.
PUBLIC_ROUTES |= {("GET", "/api/capabilities")}

# On the hosted deployment /debug/whoami needs a login (server.AUTH_GATE_EXEMPT_UNLESS_HOSTED),
# so it is swept like any other route. Declare the target with AVALOKA_DEPLOYMENT, as the
# deployment itself does; hermetically that is this process's own value.
from app.core.editions import normalize_deployment as _normalize_deployment  # noqa: E402

if _normalize_deployment(os.getenv("AVALOKA_DEPLOYMENT")) == "avaloka_hosted":
    PUBLIC_ROUTES -= {("GET", "/debug/whoami")}


def _fill(path: str) -> str:
    out = path
    for name, value in PATH_PARAM_SAMPLE.items():
        out = out.replace("{" + name + "}", value)
    return out


def route_inventory() -> List[Tuple[str, str, str]]:
    """
    Returns (method, template_path, concrete_path) for every non-public route.

    In deployed mode the inventory comes from the live /openapi.json so the sweep
    reflects what is actually serving; hermetically it comes from the app object.
    """
    paths: Dict[str, List[str]] = {}

    if MODE == "deployed":
        import httpx

        resp = httpx.get(f"{DEPLOYED_URL}/openapi.json", headers=schema_headers(), timeout=30)
        if resp.status_code != 200:
            # An empty inventory would make the sweep pass vacuously; say why instead.
            raise RuntimeError(
                f"deployed /openapi.json answered {resp.status_code} to a signed request; "
                "is SUPABASE_JWT_SECRET the cluster's secret?"
            )
        spec = resp.json()
        for path, ops in spec.get("paths", {}).items():
            methods = [m.upper() for m in ops if m.lower() in
                       {"get", "post", "put", "patch", "delete"}]
            if methods:
                paths[path] = methods
    else:
        import app.api.server as server

        # Included routers (analytics, Ray jobs, models) sit in app.routes as
        # ``_IncludedRouter`` entries with no path of their own; a flat walk
        # skipped every route inside them, so the sweep never saw them. Walk
        # into them (tests/test_auth_before_body.py does the same), and skip
        # ``include_in_schema=False`` routes exactly as the deployed branch
        # does by reading /openapi.json.
        def _walk(routes: Iterable[Any]) -> None:
            for route in routes:
                path = getattr(route, "path", None)
                methods = getattr(route, "methods", None)
                if hasattr(route, "effective_candidates"):  # FastAPI's _IncludedRouter
                    _walk(route.effective_candidates())
                    continue
                if not path or not methods or not path.startswith("/"):
                    continue
                if getattr(route, "include_in_schema", True) is False:
                    continue
                usable = sorted(m for m in methods if m in
                                {"GET", "POST", "PUT", "PATCH", "DELETE"})
                if usable:
                    paths.setdefault(path, []).extend(usable)

        _walk(server.app.routes)

    inventory: List[Tuple[str, str, str]] = []
    for path, methods in sorted(paths.items()):
        for method in sorted(set(methods)):
            if (method, path) in PUBLIC_ROUTES:
                continue
            if path.startswith(("/openapi", "/docs", "/redoc")):
                continue
            inventory.append((method, path, _fill(path)))
    return inventory


def route_id(method: str, path: str) -> str:
    return f"{method} {path}"
