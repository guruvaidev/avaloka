"""
Shared harness for the L5 API pack (plan suites E2, E3, E7).

Every test in this package runs unchanged in two modes:

  hermetic  (default)  in-process FastAPI TestClient over the in-memory fake
                       stack from tests/test_server_integration.py. No Docker,
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
from typing import Any, Dict, Iterator, List, Optional, Tuple

import jwt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DEPLOYED_URL = (os.getenv("AVALOKA_API_URL") or "").rstrip("/")
MODE = "deployed" if DEPLOYED_URL else "hermetic"

# Hermetic mode signs with the same constant tests/test_server_integration.py
# patches into the server; deployed mode must use the cluster's real secret.
HERMETIC_SECRET = "test-jwt-secret"
JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET") if MODE == "deployed" else HERMETIC_SECRET

UNAUTH_MESSAGE = "Missing or invalid auth token"


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
    Rebuild the fake stack from tests/test_server_integration.py so the L5 pack
    shares one definition of the fakes rather than forking a second one.
    """
    from fastapi.testclient import TestClient

    import app.api.server as server
    from app.services import session_service, storage_service
    from tests import test_server_integration as si

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

    session_service.cache = si.FakeCache()  # type: ignore[attr-defined]
    fake_store = si.FakeBlobStore(bucket="fake-bucket", prefix="test")
    storage_service.blob_store = fake_store  # type: ignore[attr-defined]

    def _store_and_key_from_uri_fake(uri: str, object_name: Optional[str]):
        return fake_store, object_name or Path(uri).name

    async def _store_from_connection_uri_fake(storage_uri: str, conn: Dict[str, Any]):
        store = si.FakeBlobStore()
        store.objects["foo.csv"] = b"a,b\n1,2\n"
        return store, ""

    async def get_cloud_connection_fake(connection_id: str) -> Dict[str, Any]:
        return {}

    def sample_data_from_source_fake(path, source_type, stratify_by=None, sample_size: float = 1.0):
        return {
            "schema": {"a": "int", "b": "int"},
            "rows": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            "ddl_schema": "CREATE TABLE t(a int, b int);",
        }

    for name, value in [
        ("save_session", si.save_session_fake),
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
PUBLIC_ROUTES = {("GET", "/health"), ("GET", "/version"), ("GET", "/debug/whoami")}

# Placeholder values used when a swept path has parameters. They must not exist,
# so an authenticated probe lands on the ownership check rather than real data.
PATH_PARAM_SAMPLE = {
    "thread_id": "thread-does-not-exist",
    "task_id": "task-does-not-exist",
    "dataset_id": "dataset-does-not-exist",
    "run_id": "run-does-not-exist",
    "session_id": "session-does-not-exist",
    "index": "0",
}


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

        spec = httpx.get(f"{DEPLOYED_URL}/openapi.json", timeout=30).json()
        for path, ops in spec.get("paths", {}).items():
            methods = [m.upper() for m in ops if m.lower() in
                       {"get", "post", "put", "patch", "delete"}]
            if methods:
                paths[path] = methods
    else:
        import app.api.server as server

        for route in server.app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if not path or not methods or not path.startswith("/"):
                continue
            usable = sorted(m for m in methods if m in
                            {"GET", "POST", "PUT", "PATCH", "DELETE"})
            if usable:
                paths.setdefault(path, []).extend(usable)

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
