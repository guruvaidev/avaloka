"""Shared fakes, fixtures and helpers for the API server tests.

Not a test module (pytest does not collect it). Used by:
  - tests/test_server_integration.py  (deselected by the CI gate: `not integration`)
  - tests/test_slow_turns.py           (runs in the CI gate)
  - tests/test_plan_limits_api.py      (runs in the CI gate)

Moved here unchanged from test_server_integration.py so the deferred-turn and
plan-limit endpoint tests can live in files the gate runs (review item 10).
"""
import os
import io
import csv
import json
import time
import base64
import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

import app.api.server as server
import app.api.integrations as integrations
import app.services.persistence_service as persistence
from app.services import storage_service, session_service
from app.core.storage import ResourceNotFoundError
from langchain_core.messages import AIMessage, HumanMessage



# ======================================================================================
# Fakes and in-memory stores
# ======================================================================================


class FakeBlobStore:
    """
    Minimal in-memory blob store implementing the methods used by the API,
    including list_hierarchy() which /buckets/list depends on.
    """

    def __init__(self, scheme: str = "gs", bucket: str = "fake-bucket", prefix: str = "test"):
        self.scheme = scheme
        self.bucket = bucket
        self.prefix = prefix.strip("/") if prefix else ""
        self.objects: Dict[str, bytes] = {}

    def _full_uri(self, key: str) -> str:
        base = f"{self.scheme}://{self.bucket}"
        if self.prefix:
            return f"{base}/{self.prefix}/{key.lstrip('/')}"
        return f"{base}/{key.lstrip('/')}"

    def put_file(self, path: Union[Path, str], key: str) -> str:
        path = Path(path)
        data = path.read_bytes()
        self.objects[key] = data
        return self._full_uri(key)

    def get_file(self, key: str, dest_path: Union[Path, str]) -> None:
        if key not in self.objects:
            raise ResourceNotFoundError(f"{key} not found")
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(self.objects[key])

    def stat(self, key: str) -> Tuple[int, str]:
        if key not in self.objects:
            raise ResourceNotFoundError(key)
        data = self.objects[key]
        return len(data), "2024-01-01T00:00:00Z"

    def list(self, prefix: str):
        prefix = prefix.lstrip("/")
        for key, data in self.objects.items():
            if not prefix or key.startswith(prefix):
                yield key, len(data), "2024-01-01T00:00:00Z"

    def list_hierarchy(self, prefix: str):
        """Flat fake: no folders, every object is a file at the root level."""
        prefix = (prefix or "").lstrip("/")
        files: List[Tuple[str, int, str]] = []
        for key, data in self.objects.items():
            if not prefix or key.startswith(prefix):
                files.append((key, len(data), "2024-01-01T00:00:00Z"))
        folders: List[str] = []
        return folders, files

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)


class FakeCache:
    """Very small async cache used in place of Redis."""

    mode = "single"

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._sets: Dict[str, Set[str]] = {}

    async def ping(self) -> bool:
        return True

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self._data[key] = value

    async def smembers(self, key: str) -> List[str]:
        return list(self._sets.get(key, set()))

    async def sadd(self, key: str, member: str) -> None:
        self._sets.setdefault(key, set()).add(member)

    async def srem(self, key: str, member: str) -> None:
        self._sets.get(key, set()).discard(member)


SESSIONS: Dict[str, Dict[str, Any]] = {}
THREAD_TO_SESSION: Dict[str, str] = {}
PERSIST_CALLS: List[Dict[str, Any]] = []
PROFILE_RESULT = {"domain": {"category": "Test/Domain"}, "quick_insights": {"data_readiness_score": 90}}


async def save_session_fake(session_id: str, data: Dict[str, Any]) -> bool:
    SESSIONS[session_id] = dict(data)
    return True


async def update_session_fake(session_id: str, mutator):
    cur = dict(SESSIONS.get(session_id) or {})
    result = mutator(cur)
    merged = result if isinstance(result, dict) else cur
    SESSIONS[session_id] = dict(merged)
    return merged


async def delete_session_fake(session_id: str) -> bool:
    SESSIONS.pop(session_id, None)
    return True


async def get_session_fake(session_id: Optional[str], bypass_circuit: bool = False):
    if not session_id:
        return None
    return SESSIONS.get(session_id)


async def refresh_session_ttl_fake(session_id: str) -> None:
    return None


async def find_session_by_dataset_for_user_fake(dataset_id: str, user_id: str) -> Optional[str]:
    for sid, sess in SESSIONS.items():
        if sess.get("dataset_id") == dataset_id and sess.get("user_id") == user_id:
            return sid
    return None


async def find_active_db_customer_for_user_fake(user_id: str):
    return None


async def bind_thread_session_fake(thread_id: str, session_id: str) -> None:
    THREAD_TO_SESSION[thread_id] = session_id


async def get_thread_session_fake(thread_id: str) -> Optional[str]:
    return THREAD_TO_SESSION.get(thread_id)


async def _user_datasets_fake(user_id: str) -> List[str]:
    dsids: List[str] = []
    for sess in SESSIONS.values():
        if sess.get("user_id") == user_id and sess.get("dataset_id"):
            dsids.append(sess["dataset_id"])
    return dsids


def _jsonify_fake(x: Any) -> Any:
    return x


async def hydrate_thread_history_fake(thread_id: str) -> None:
    return None


async def persist_thread_history_fake(thread_id: str) -> None:
    return None


async def delete_thread_history_fake(thread_id: str) -> None:
    return None


async def persist_session_snapshot_fake(thread_id: str, sess: Dict[str, Any]) -> int:
    return 0


async def resolve_shared_session_fake(thread_id: str, user_id: str, analysis_id=None):
    return None


class DummyResponse:
    def __init__(self, status_code: int = 200, text: str = "ok"):
        self.status_code = status_code
        self.text = text
        self.headers = {"content-type": "application/json"}
        self.is_success = status_code < 400

    def json(self) -> Dict[str, Any]:
        return {"ok": True}


async def lg_request_fake(method: str, path: str, **kw) -> DummyResponse:
    return DummyResponse(200, "ok")


_THREAD_COUNTER = 0


async def lg_json_fake(method: str, path: str, **kw) -> Dict[str, Any]:
    global _THREAD_COUNTER
    if path == "/threads" and method.upper() == "POST":
        _THREAD_COUNTER += 1
        return {"thread_id": f"thread-{_THREAD_COUNTER}"}
    if path == "/threads/search":
        return {"items": []}
    if path == "/ok":
        return {"ok": True}
    return {}


class FakeGraph:
    # test_runtime_graph_compiled_with_checkpointer asserts a non-None checkpointer.
    checkpointer = object()

    def invoke(self, state_in: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        last = state_in["messages"][-1]
        content = last.content if isinstance(last, (AIMessage, HumanMessage)) else str(last)
        reply = AIMessage(content=f"Echo: {content}")
        return {
            "messages": state_in["messages"] + [reply],
            "planner_definition": {},
            "ready_to_summarize": False,
            "ready_to_code": True,
            "coder_definition": {"code": "print('hello world')"},
            "planner_graph_path": None,
            "planner_graph_status": None,
            "visualization_config": {},
            "visualization_status": "",
            "output_file_data": {},
            "execution_result": {},
        }


async def read_thread_msgs_fake(thread_id: str) -> List[Dict[str, Any]]:
    meta = server.THREAD_META.get(thread_id) or {}
    lc_msgs = meta.get("lc_msgs") or []
    out: List[Dict[str, Any]] = []
    for m in lc_msgs:
        if isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, HumanMessage):
            role = "user"
        else:
            role = "user"
        out.append({"role": role, "content": getattr(m, "content", str(m))})
    return out


# ======================================================================================
# Auth helpers (matches server._resolve_user_id HS256 path)
# ======================================================================================

TEST_JWT_SECRET = "test-jwt-secret"


def make_auth_headers(user_id: str) -> Dict[str, str]:
    token = jwt.encode(
        {"sub": user_id, "iat": int(time.time())},
        TEST_JWT_SECRET,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


# ======================================================================================
# Pytest fixture
# ======================================================================================


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    # Fresh per-test in-memory state
    SESSIONS.clear()
    THREAD_TO_SESSION.clear()
    PERSIST_CALLS.clear()
    server.THREAD_META.clear()
    server._profile_meta.clear()

    # Ensure server uses our JWT secret for decode()
    monkeypatch.setattr(server, "JWT_SECRET", TEST_JWT_SECRET, raising=False)

    # TMP_ROOT in per-test dir
    tmp_root = tmp_path / "avaloka_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(server, "TMP_ROOT", tmp_root, raising=False)
    os.environ["TMP_ROOT"] = str(tmp_root)

    # Fake cache
    fake_cache = FakeCache()
    session_service.cache = fake_cache  # type: ignore[attr-defined]

    # Fake blob store
    fake_store = FakeBlobStore(bucket="fake-bucket", prefix="test")
    storage_service.blob_store = fake_store  # type: ignore[attr-defined]

    def _store_and_key_from_uri_fake(uri: str, object_name: Optional[str]):
        key = object_name or Path(uri).name
        return fake_store, key

    async def _store_from_connection_uri_fake(storage_uri: str, conn: Dict[str, Any]):
        store = FakeBlobStore()
        store.objects["foo.csv"] = b"a,b\n1,2\n"
        store.objects["bar.csv"] = b"a,b\n3,4\n"
        return store, ""

    async def normalize_storage_uri_fake(uri: str, conn: Dict[str, Any]) -> str:
        return (uri or "").rstrip("/")

    # Session helpers
    monkeypatch.setattr(server, "save_session", save_session_fake, raising=False)
    monkeypatch.setattr(server, "update_session", update_session_fake, raising=False)
    monkeypatch.setattr(server, "delete_session", delete_session_fake, raising=False)
    monkeypatch.setattr(server, "get_session", get_session_fake, raising=False)
    monkeypatch.setattr(server, "refresh_session_ttl", refresh_session_ttl_fake, raising=False)
    monkeypatch.setattr(server, "find_session_by_dataset_for_user", find_session_by_dataset_for_user_fake, raising=False)
    monkeypatch.setattr(server, "find_active_db_customer_for_user", find_active_db_customer_for_user_fake, raising=False)
    monkeypatch.setattr(server, "bind_thread_session", bind_thread_session_fake, raising=False)
    monkeypatch.setattr(server, "get_thread_session", get_thread_session_fake, raising=False)
    monkeypatch.setattr(server, "_user_datasets", _user_datasets_fake, raising=False)
    monkeypatch.setattr(server, "_jsonify", _jsonify_fake, raising=False)

    # Thread history persistence helpers (no-op; keep everything in THREAD_META)
    monkeypatch.setattr(server, "hydrate_thread_history", hydrate_thread_history_fake, raising=False)
    monkeypatch.setattr(server, "persist_thread_history", persist_thread_history_fake, raising=False)
    monkeypatch.setattr(server, "delete_thread_history", delete_thread_history_fake, raising=False)

    # Supabase-backed paths: never hit a real project from tests.
    monkeypatch.setattr(server, "persist_session_snapshot", persist_session_snapshot_fake, raising=False)
    monkeypatch.setattr(server, "_resolve_shared_session_for_thread", resolve_shared_session_fake, raising=False)
    monkeypatch.setattr(server, "load_portfolio", lambda dataset_id: None, raising=False)
    monkeypatch.setattr(server, "load_profile", lambda dataset_id: None, raising=False)

    def _schedule_persistence_fake(**kw):
        PERSIST_CALLS.append(kw)

    monkeypatch.setattr(server, "_schedule_small_file_persistence", _schedule_persistence_fake, raising=False)

    # LLM-backed upload steps: deterministic stubs, no network.
    def _viz_stub(**kw):
        return {"visualization_status": "ready", "charts": []}

    def _profile_full_stub(**kw):
        return {"full_profiling_result": dict(PROFILE_RESULT), "profiling_status": "full_profile"}

    monkeypatch.setattr(server, "build_visualization_config_from_sample", _viz_stub, raising=False)
    monkeypatch.setattr(server, "profile_full", _profile_full_stub, raising=False)

    # Upload finalizer: no re-assert sleeps in tests.
    monkeypatch.setattr(server, "_UPLOAD_REASSERT_DELAYS_S", (), raising=False)

    # Storage helpers
    monkeypatch.setattr(server, "_store_and_key_from_uri", _store_and_key_from_uri_fake, raising=False)
    monkeypatch.setattr(storage_service, "_store_and_key_from_uri", _store_and_key_from_uri_fake, raising=False)
    monkeypatch.setattr(server, "_store_from_connection_uri", _store_from_connection_uri_fake, raising=False)
    monkeypatch.setattr(server, "normalize_storage_uri", normalize_storage_uri_fake, raising=False)

    async def get_cloud_connection_fake(connection_id: str) -> Dict[str, Any]:
        return {}
    monkeypatch.setattr(server, "get_cloud_connection", get_cloud_connection_fake, raising=False)

    def sample_data_from_source_fake(path: str, source_type: str, stratify_by=None, sample_size: float = 1.0):
        return {
            "schema": {"a": "int", "b": "int"},
            "rows": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            "ddl_schema": "CREATE TABLE t(a int, b int);",
        }
    monkeypatch.setattr(server, "sample_data_from_source", sample_data_from_source_fake, raising=False)

    # Default profiling sampler stub (individual tests may override).
    def sample_with_profiling_default(path, source_type, sample_size, use_ray=False, **kw):
        return {
            "schema": {"a": "int", "b": "int"},
            "ddl_schema": "CREATE TABLE t(a int, b int);",
            "portfolio_samples": {"random_baseline": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]},
            "sample_statistics": None,
        }
    monkeypatch.setattr(server, "sample_with_profiling", sample_with_profiling_default, raising=False)

    # LangGraph stubs
    monkeypatch.setattr(server, "lg_request", lg_request_fake, raising=False)
    monkeypatch.setattr(server, "lg_json", lg_json_fake, raising=False)
    monkeypatch.setattr(server, "GRAPH", FakeGraph(), raising=False)
    monkeypatch.setattr(server, "resolve_plan", lambda user_id: "professional", raising=False)
    monkeypatch.setattr(server, "GRAPH_READY", True, raising=False)

    # Thread history helper
    monkeypatch.setattr(server, "read_thread_msgs", read_thread_msgs_fake, raising=False)

    # Settings (minimal)
    if hasattr(server, "settings"):
        server.settings.storage_backend = "gcs"
        server.settings.gcs_bucket = "fake-bucket"
        server.settings.gcs_prefix = "test/"

    return TestClient(server.app)


# ======================================================================================
# Helpers
# ======================================================================================


def _run(coro):
    return asyncio.run(coro)


def _do_upload(client: TestClient, user_id: str = "user-1") -> Dict[str, Any]:
    csv_bytes = b"a,b\n1,2\n3,4\n"
    files = {"file": ("test.csv", csv_bytes, "text/csv")}
    resp = client.post(
        "/api/upload",
        files=files,
        data={},
        headers=make_auth_headers(user_id),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _rows(n: int) -> List[Dict[str, Any]]:
    return [{"a": i, "b": i * 2} for i in range(n)]


def _install_sampler(
    monkeypatch,
    portfolio: Dict[str, List[Dict[str, Any]]],
    *,
    sample_statistics: Optional[Dict[str, Any]] = None,
    delay_s: float = 0.0,
):
    """Replace the sampler with one that returns `portfolio` (optionally slowly)."""

    def _sampler(path, source_type, sample_size, use_ray=False, **kw):
        if delay_s:
            time.sleep(delay_s)
        return {
            "schema": {"a": "int", "b": "int"},
            "ddl_schema": "CREATE TABLE t(a int, b int);",
            "portfolio_samples": portfolio,
            "sample_statistics": sample_statistics,
            "profiling_result": {"data_shape": {"rows": 10, "columns": 2}},
        }

    monkeypatch.setattr(server, "sample_with_profiling", _sampler, raising=False)


def _wait_for(predicate, timeout_s: float = 3.0, interval_s: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


# ======================================================================================
# Shared-loop client (deferred turns) and send_message helper
# ======================================================================================


@pytest.fixture
def shared_loop_client(client):
    """TestClient used without `with` starts a fresh event loop for every
    request and closes it when the request ends, which kills any task the
    request spawned. A deferred turn keeps running after its response (that
    is the whole point), so requests here must share one loop, as they do
    under uvicorn. Setting `portal` gives them one without starting the
    app's lifespan (Redis, Supabase)."""
    from anyio.from_thread import start_blocking_portal
    with start_blocking_portal() as portal:
        client.portal = portal
        try:
            yield client
        finally:
            client.portal = None


def _quiet_send(monkeypatch):
    """Neutralise planner text parsers and background asset persistence so
    send_message tests only exercise server.py routing."""
    monkeypatch.setattr(server, "_parse_fidelity_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_parse_selected_sample_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_is_mode_switch_message", lambda t: False, raising=False)

    async def _noop_persist(**kw):
        return None

    monkeypatch.setattr(server, "_persist_assets_background", _noop_persist, raising=False)