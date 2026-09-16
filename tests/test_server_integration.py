import os
import io
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

import app.api.server as server
from app.services import storage_service, session_service
from app.core.storage import ResourceNotFoundError
from langchain_core.messages import AIMessage, HumanMessage


# ======================================================================================
# Fakes and in-memory stores
# ======================================================================================


class FakeBlobStore:
    """
    Minimal in-memory blob store implementing the methods used by the API,
    including list_hierarchy() which /buckets/list now depends on.
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
        """
        Return (folder_prefixes, file_tuples) for the current level. This fake keeps
        it flat: no folders, every object is a file at the root level.
        file_tuples: List[(key, size, updated)].
        """
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
    server.THREAD_META.clear()

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

    # Patch session helpers used by server module
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


# ======================================================================================
# Health / version / whoami
# ======================================================================================


def test_health_ok(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["redis_connected"] is True
    assert data["graph_ready"] is True


def test_version_returns_app_version(client: TestClient):
    resp = client.get("/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_whoami_returns_user_for_valid_token(client: TestClient):
    resp = client.get("/debug/whoami", headers=make_auth_headers("who-user"))
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "who-user"


def test_whoami_returns_none_without_token(client: TestClient):
    resp = client.get("/debug/whoami")
    assert resp.status_code == 200
    assert resp.json()["user_id"] is None


# ======================================================================================
# Missions plan
# ======================================================================================


def test_missions_plan_requires_auth(client: TestClient):
    resp = client.post("/api/missions/plan", json={})
    assert resp.status_code == 401


# ======================================================================================
# Upload flow
# ======================================================================================


def test_upload_flow_creates_session_and_thread(client: TestClient):
    out = _do_upload(client, user_id="u1")
    assert out["dataset_id"]
    assert out["session_id"]
    assert out["thread_id"]

    sid = out["session_id"]
    assert sid in SESSIONS
    sess = SESSIONS[sid]
    assert sess["dataset_id"] == out["dataset_id"]
    assert sess["user_id"] == "u1"
    # Uploaded objects are Avaloka-owned so they can be deleted later.
    assert sess.get("source_kind") == "uploaded"


def test_upload_requires_auth(client: TestClient):
    files = {"file": ("test.csv", b"a,b\n1,2\n", "text/csv")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 401


def test_upload_no_files_422(client: TestClient):
    resp = client.post("/api/upload", data={}, headers=make_auth_headers("no-files"))
    assert resp.status_code == 422


def test_upload_rejects_unsupported_extension(client: TestClient):
    files = {"file": ("bad.xyz", b"hello", "application/octet-stream")}
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("user-unsupported"))
    assert resp.status_code == 415
    assert ".xyz is not supported yet" in resp.json()["detail"]


# ======================================================================================
# Datasets: list / get / preview
# ======================================================================================


# def test_list_datasets_and_preview(client: TestClient):
#     upload = _do_upload(client, user_id="u2")
#     dsid = upload["dataset_id"]

#     resp = client.get("/datasets", headers=make_auth_headers("u2"))
#     assert resp.status_code == 200, resp.text
#     items = resp.json()
#     assert len(items) == 1
#     assert items[0]["dataset_id"] == dsid

#     resp2 = client.get(f"/datasets/{dsid}/preview", headers=make_auth_headers("u2"))
#     assert resp2.status_code == 200, resp2.text
#     prev = resp2.json()
#     assert prev["dataset_id"] == dsid
#     assert prev["rows_sampled"] > 0
#     assert prev["schema"]


def test_list_datasets_requires_auth(client: TestClient):
    resp = client.get("/datasets")
    assert resp.status_code == 401
    assert "Missing or invalid auth token" in resp.json()["detail"]


def test_list_datasets_empty_for_new_user(client: TestClient):
    resp = client.get("/datasets", headers=make_auth_headers("brand-new-user"))
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_dataset_404_for_other_user(client: TestClient):
    upload_out = _do_upload(client, user_id="owner-user")
    dataset_id = upload_out["dataset_id"]

    resp = client.get(f"/datasets/{dataset_id}", headers=make_auth_headers("other-user"))
    assert resp.status_code == 404
    assert "Dataset not found for this user" in resp.json()["detail"]


def test_get_dataset_returns_item(client: TestClient):
    upload_out = _do_upload(client, user_id="own-get")
    dataset_id = upload_out["dataset_id"]
    resp = client.get(f"/datasets/{dataset_id}", headers=make_auth_headers("own-get"))
    assert resp.status_code == 200, resp.text
    assert resp.json()["dataset_id"] == dataset_id


def test_preview_rebuilds_missing_local_file_from_blobstore(client: TestClient):
    upload_out = _do_upload(client, user_id="u-preview-rebuild")
    dataset_id = upload_out["dataset_id"]
    session_id = upload_out["session_id"]

    assert session_id in SESSIONS
    sess = SESSIONS[session_id]
    local_input_path = Path(sess["work_local_input"])
    assert local_input_path.exists()

    local_input_path.unlink()
    assert not local_input_path.exists()

    resp = client.get(f"/datasets/{dataset_id}/preview", headers=make_auth_headers("u-preview-rebuild"))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["dataset_id"] == dataset_id
    assert data["rows_sampled"] > 0

    sid = data.get("session_id") or session_id
    assert sid in SESSIONS
    restored = Path(SESSIONS[sid]["work_local_input"])
    assert restored.exists()


# ======================================================================================
# register-existing-storage
# ======================================================================================


def test_register_existing_storage_creates_session_and_preview(client: TestClient):
    body = {
        "storage_uri": "gs://external-bucket/some-prefix",
        "key": "foo.csv",
        "connection_id": "conn-1",
        "schema_json": None,
    }
    resp = client.post(
        "/api/register-existing-storage",
        json=body,
        headers=make_auth_headers("user-reg"),
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["dataset_id"]
    assert out["session_id"]
    assert out["rows_sampled"] > 0

    # register-existing must tag the source as customer-owned (never deleted).
    sid = out["session_id"]
    assert SESSIONS[sid].get("source_kind") == "registered"


def test_register_existing_requires_auth(client: TestClient):
    body = {"storage_uri": "gs://b/p", "key": "foo.csv", "connection_id": "c"}
    resp = client.post("/api/register-existing-storage", json=body)
    assert resp.status_code == 401


def test_register_existing_rejects_unsupported_extension(client: TestClient):
    body = {"storage_uri": "gs://b/p", "key": "foo.weird", "connection_id": "c"}
    resp = client.post(
        "/api/register-existing-storage",
        json=body,
        headers=make_auth_headers("reg-bad-ext"),
    )
    assert resp.status_code == 415


# ======================================================================================
# Buckets listing (now via list_hierarchy)
# ======================================================================================


def test_buckets_list_gcs_returns_objects(client: TestClient):
    resp = client.get(
        "/buckets/list",
        params={"backend": "gcs", "bucket": "any-bucket", "prefix": ""},
        headers=make_auth_headers("bucket-user"),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    keys = {obj["key"] for obj in data["objects"]}
    assert "foo.csv" in keys
    assert "bar.csv" in keys


def test_buckets_list_requires_auth(client: TestClient):
    resp = client.get(
        "/buckets/list", params={"backend": "gcs", "bucket": "any-bucket"}
    )
    assert resp.status_code == 401


def test_buckets_list_rejects_invalid_token(client: TestClient):
    resp = client.get(
        "/buckets/list",
        params={"backend": "gcs", "bucket": "any-bucket"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert resp.status_code == 401


def test_buckets_list_s3_requires_connection_id(client: TestClient):
    resp = client.get(
        "/buckets/list",
        params={"backend": "s3", "bucket": "some-bucket", "prefix": ""},
        headers=make_auth_headers("me"),
    )
    assert resp.status_code == 400
    assert "connection_id is required" in resp.json()["detail"]


def test_buckets_list_rejects_foreign_connection(client: TestClient, monkeypatch):
    async def foreign_conn(connection_id: str) -> Dict[str, Any]:
        return {"id": connection_id, "user_id": "someone-else", "provider": "aws"}

    monkeypatch.setattr(server, "get_cloud_connection", foreign_conn, raising=False)

    resp = client.get(
        "/buckets/list",
        params={"backend": "s3", "bucket": "victim-bucket", "connection_id": "conn-1"},
        headers=make_auth_headers("me"),
    )
    assert resp.status_code == 404


def test_buckets_list_rejects_connection_without_owner_column(client: TestClient, monkeypatch):
    async def ownerless_conn(connection_id: str) -> Dict[str, Any]:
        return {"id": connection_id, "provider": "aws"}

    monkeypatch.setattr(server, "get_cloud_connection", ownerless_conn, raising=False)

    resp = client.get(
        "/buckets/list",
        params={"backend": "s3", "bucket": "b", "connection_id": "conn-1"},
        headers=make_auth_headers("me"),
    )
    assert resp.status_code == 404


def test_buckets_list_allows_owned_connection(client: TestClient, monkeypatch):
    async def owned_conn(connection_id: str) -> Dict[str, Any]:
        return {"id": connection_id, "user_id": "me", "provider": "aws"}

    monkeypatch.setattr(server, "get_cloud_connection", owned_conn, raising=False)

    resp = client.get(
        "/buckets/list",
        params={"backend": "s3", "bucket": "my-bucket", "connection_id": "conn-1"},
        headers=make_auth_headers("me"),
    )
    assert resp.status_code == 200, resp.text
    keys = {obj["key"] for obj in resp.json()["objects"]}
    assert keys == {"foo.csv", "bar.csv"}


def test_buckets_list_azure_missing_account_name(client: TestClient, monkeypatch):
    async def owned_conn(connection_id: str) -> Dict[str, Any]:
        return {"id": connection_id, "user_id": "me", "provider": "azure"}

    monkeypatch.setattr(server, "get_cloud_connection", owned_conn, raising=False)

    resp = client.get(
        "/buckets/list",
        params={"backend": "azure", "bucket": "container-name", "prefix": "", "connection_id": "conn-1"},
        headers=make_auth_headers("me"),
    )
    assert resp.status_code == 500
    assert "Azure connection is missing the storage account name" in resp.json()["detail"]


# ======================================================================================
# connection_belongs_to_user matrix
# ======================================================================================


def test_connection_belongs_to_user_matrix():
    from app.api.cloud_connections import connection_belongs_to_user

    assert connection_belongs_to_user({"user_id": "u1"}, "u1")
    assert connection_belongs_to_user({"owner_id": "u1"}, "u1")
    assert connection_belongs_to_user({"user_id": "", "created_by": "u1"}, "u1")
    assert not connection_belongs_to_user({"user_id": "u2"}, "u1")
    assert not connection_belongs_to_user({}, "u1")
    assert not connection_belongs_to_user({"user_id": "u1"}, "")
    assert not connection_belongs_to_user({"user_id": "u1"}, None)


# ======================================================================================
# Threads: create / message / code / history / delete / list
# ======================================================================================


def test_threads_create_and_message_roundtrip(client: TestClient):
    upload = _do_upload(client, user_id="u-thread")
    thread_id = upload["thread_id"]
    dsid = upload["dataset_id"]

    msg_body = {"content": "hello world", "metadata": {"dataset_id": dsid}}
    resp = client.post(
        f"/threads/{thread_id}/messages",
        json=msg_body,
        headers=make_auth_headers("u-thread"),
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["ready_to_code"] is True
    roles = [m["role"] for m in out["messages"]]
    assert "assistant" in roles

    resp2 = client.get(
        f"/threads/{thread_id}/code",
        headers=make_auth_headers("u-thread"),
    )
    assert resp2.status_code == 200
    assert "hello world" in resp2.text


def test_thread_message_without_session_returns_400(client: TestClient):
    resp = client.post("/threads", json={"metadata": {"note": "no-session"}})
    assert resp.status_code == 200
    thread_id = resp.json()["thread_id"]

    msg_body = {"content": "hello without session"}
    msg_resp = client.post(
        f"/threads/{thread_id}/messages",
        json=msg_body,
        headers=make_auth_headers("user-no-session"),
    )
    assert msg_resp.status_code == 400
    detail = msg_resp.json().get("detail", "")
    assert "session" in detail.lower() or "dataset" in detail.lower()


def test_fetch_history_requires_auth(client: TestClient):
    resp = client.get("/threads/non-existent-thread/messages")
    assert resp.status_code == 401


def test_fetch_history_empty(client: TestClient):
    resp = client.get(
        "/threads/non-existent-thread/messages",
        headers=make_auth_headers("u-history-empty"),
    )
    assert resp.status_code == 200
    assert resp.json() == {"messages": []}


def test_fetch_history_foreign_user_403(client: TestClient):
    upload = _do_upload(client, user_id="hist-owner")
    thread_id = upload["thread_id"]
    dsid = upload["dataset_id"]
    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"content": "hello", "metadata": {"dataset_id": dsid}},
        headers=make_auth_headers("hist-owner"),
    )
    assert resp.status_code == 200

    resp = client.get(
        f"/threads/{thread_id}/messages",
        headers=make_auth_headers("hist-intruder"),
    )
    assert resp.status_code == 403


def test_fetch_history_orphaned_thread_not_served(client: TestClient):
    server.THREAD_META["orphan-thread"] = {
        "lc_msgs": [HumanMessage(content="secret chat")],
    }
    resp = client.get(
        "/threads/orphan-thread/messages",
        headers=make_auth_headers("someone-else"),
    )
    assert resp.status_code == 404


def test_get_thread_messages_after_chat(client: TestClient):
    upload = _do_upload(client, user_id="u-history")
    thread_id = upload["thread_id"]
    dsid = upload["dataset_id"]

    msg_body = {"content": "hello history", "metadata": {"dataset_id": dsid}}
    resp = client.post(
        f"/threads/{thread_id}/messages",
        json=msg_body,
        headers=make_auth_headers("u-history"),
    )
    assert resp.status_code == 200

    hist_resp = client.get(
        f"/threads/{thread_id}/messages",
        headers=make_auth_headers("u-history"),
    )
    assert hist_resp.status_code == 200
    data = hist_resp.json()
    assert "messages" in data
    assert len(data["messages"]) >= 2
    roles = {m["role"] for m in data["messages"]}
    assert "user" in roles
    assert "assistant" in roles


def test_thread_code_requires_auth(client: TestClient):
    resp = client.get("/threads/some-thread/code")
    assert resp.status_code == 401


def test_thread_code_hidden_from_other_users(client: TestClient):
    upload = _do_upload(client, user_id="code-owner")
    thread_id = upload["thread_id"]
    dsid = upload["dataset_id"]
    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"content": "generate code", "metadata": {"dataset_id": dsid}},
        headers=make_auth_headers("code-owner"),
    )
    assert resp.status_code == 200

    resp = client.get(
        f"/threads/{thread_id}/code",
        headers=make_auth_headers("code-intruder"),
    )
    assert resp.status_code == 404


def test_list_threads_requires_auth(client: TestClient):
    resp = client.get("/threads")
    assert resp.status_code == 401


def test_list_threads_scoped_to_caller(client: TestClient, monkeypatch):
    up_a = _do_upload(client, user_id="list-user-a")
    up_b = _do_upload(client, user_id="list-user-b")

    async def search_fake(method: str, path: str, **kw) -> Dict[str, Any]:
        if path == "/threads/search":
            return {
                "items": [
                    {"thread_id": up_a["thread_id"], "metadata": {}},
                    {"thread_id": up_b["thread_id"], "metadata": {}},
                    {"thread_id": "unbound-thread", "metadata": {}},
                ]
            }
        return {}

    monkeypatch.setattr(server, "lg_json", search_fake, raising=False)

    resp = client.get("/threads", headers=make_auth_headers("list-user-a"))
    assert resp.status_code == 200, resp.text
    tids = {t["thread_id"] for t in resp.json()}
    assert tids == {up_a["thread_id"]}


def test_delete_thread_soft_success(client: TestClient):
    resp = client.post("/threads", json={"metadata": {"foo": "bar"}})
    assert resp.status_code == 200
    thread_id = resp.json()["thread_id"]

    del_resp = client.delete(f"/threads/{thread_id}", headers=make_auth_headers("user-del-thread"))
    assert del_resp.status_code == 204


def test_delete_thread_requires_auth(client: TestClient):
    resp = client.delete("/threads/some-thread")
    assert resp.status_code == 401


def test_delete_thread_foreign_user_403(client: TestClient):
    upload = _do_upload(client, user_id="del-owner")
    thread_id = upload["thread_id"]
    dsid = upload["dataset_id"]
    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"content": "keep me", "metadata": {"dataset_id": dsid}},
        headers=make_auth_headers("del-owner"),
    )
    assert resp.status_code == 200

    del_resp = client.delete(
        f"/threads/{thread_id}", headers=make_auth_headers("del-intruder")
    )
    assert del_resp.status_code == 403

    hist = client.get(
        f"/threads/{thread_id}/messages", headers=make_auth_headers("del-owner")
    )
    assert hist.status_code == 200
    assert len(hist.json()["messages"]) >= 2


def test_delete_thread_owner_succeeds(client: TestClient):
    upload = _do_upload(client, user_id="del-owner-2")
    thread_id = upload["thread_id"]

    del_resp = client.delete(
        f"/threads/{thread_id}", headers=make_auth_headers("del-owner-2")
    )
    assert del_resp.status_code == 204


def test_delete_thread_orphaned_history_404(client: TestClient):
    server.THREAD_META["orphan-del-thread"] = {
        "lc_msgs": [HumanMessage(content="someone's chat")],
    }
    del_resp = client.delete(
        "/threads/orphan-del-thread", headers=make_auth_headers("someone-else")
    )
    assert del_resp.status_code == 404
    assert "orphan-del-thread" in server.THREAD_META


# ======================================================================================
# Planner graph image endpoint
# ======================================================================================


def test_planner_graph_requires_auth(client: TestClient):
    resp = client.get("/threads/some-thread/planner-graph")
    assert resp.status_code == 401


def test_planner_graph_unknown_thread_404(client: TestClient):
    resp = client.get(
        "/threads/no-such-thread/planner-graph",
        headers=make_auth_headers("pg-user"),
    )
    assert resp.status_code == 404


# ======================================================================================
# Database connect / query
# ======================================================================================


def test_database_connect_creates_session(client: TestClient, monkeypatch):
    async def list_tables_ok(customer_id: str, api_key: str) -> Dict[str, Any]:
        return {"tables": ["t1", "t2"], "meta": {"database_type": "postgres"}}

    monkeypatch.setattr(server, "list_database_tables", list_tables_ok, raising=False)

    resp = client.post(
        "/api/database/connect",
        json={"customer_id": "cust-1", "api_key": "secret"},
        headers=make_auth_headers("db-user"),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "connected"
    assert data["customer_id"] == "cust-1"
    assert data["tables_available"] == 2
    assert data["database_type"] == "postgres"
    # The raw api_key must not be persisted (nothing reads it back).
    stored = [s for s in SESSIONS.values() if s.get("type") == "database"]
    assert stored and all("api_key" not in s for s in stored)


def test_database_connect_rejects_query_param_credentials(client: TestClient, monkeypatch):
    async def list_tables_ok(customer_id: str, api_key: str) -> Dict[str, Any]:
        return {"tables": ["t1"], "meta": {"database_type": "postgres"}}

    monkeypatch.setattr(server, "list_database_tables", list_tables_ok, raising=False)

    resp = client.post(
        "/api/database/connect",
        params={"customer_id": "cust-1", "api_key": "secret"},
        headers=make_auth_headers("db-user"),
    )
    assert resp.status_code == 422


def test_database_connect_propagates_error(client: TestClient, monkeypatch):
    async def list_tables_error(customer_id: str, api_key: str) -> Dict[str, Any]:
        return {"error": "boom"}

    monkeypatch.setattr(server, "list_database_tables", list_tables_error, raising=False)

    resp = client.post(
        "/api/database/connect",
        json={"customer_id": "cust-err", "api_key": "bad"},
        headers=make_auth_headers("db-user"),
    )
    assert resp.status_code == 400
    assert "Database connection failed" in resp.json()["detail"]


def test_database_query_missing_credentials_400(client: TestClient):
    body = {"content": "SELECT 1", "metadata": {}}
    resp = client.post(
        "/api/v1/database/query",
        json=body,
        headers=make_auth_headers("db-user"),
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "customer_id" in detail and "api_key" in detail


def _force_encryption_key(monkeypatch):
    import hashlib
    from app.api import cloud_connections as cc

    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"test-key").digest(), raising=False)


def _mock_mcp(monkeypatch, payload=None):
    payload = payload or {"tables": ["t1", "t2"]}

    class DummyResp:
        status_code = 200

        def json(self):
            return payload

        def raise_for_status(self):
            return None

    class DummyClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json, headers, timeout):
            return DummyResp()

    monkeypatch.setattr(server.httpx, "AsyncClient", DummyClient, raising=False)


def test_database_connect_then_query_reuses_session(client: TestClient, monkeypatch):
    _force_encryption_key(monkeypatch)

    async def list_tables_ok(customer_id: str, api_key: str) -> Dict[str, Any]:
        return {"tables": ["t1", "t2"], "meta": {"database_type": "postgres"}}

    monkeypatch.setattr(server, "list_database_tables", list_tables_ok, raising=False)

    connect = client.post(
        "/api/database/connect",
        json={"customer_id": "cust-1", "api_key": "super-secret-key"},
        headers=make_auth_headers("db-user"),
    )
    assert connect.status_code == 200, connect.text
    conn_data = connect.json()
    assert conn_data["credentials_saved"] is True
    session_id = conn_data["session_id"]

    # The stored key must be encrypted, never plaintext.
    stored = SESSIONS[session_id]
    assert "api_key" not in stored
    assert "super-secret-key" not in json.dumps(stored)

    _mock_mcp(monkeypatch)

    resp = client.post(
        "/api/v1/database/query",
        json={"content": "list tables", "session_id": session_id},
        headers=make_auth_headers("db-user"),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["database_info"]["customer_id"] == "cust-1"


def test_database_query_foreign_session_not_used(client: TestClient, monkeypatch):
    _force_encryption_key(monkeypatch)

    async def list_tables_ok(customer_id: str, api_key: str) -> Dict[str, Any]:
        return {"tables": ["t1"], "meta": {"database_type": "postgres"}}

    monkeypatch.setattr(server, "list_database_tables", list_tables_ok, raising=False)

    connect = client.post(
        "/api/database/connect",
        json={"customer_id": "cust-1", "api_key": "owner-secret"},
        headers=make_auth_headers("owner"),
    )
    session_id = connect.json()["session_id"]

    _mock_mcp(monkeypatch)

    resp = client.post(
        "/api/v1/database/query",
        json={"content": "list tables", "session_id": session_id},
        headers=make_auth_headers("intruder"),
    )
    assert resp.status_code == 400


def test_database_query_creates_dataset_and_session(client: TestClient, monkeypatch):
    class DummyResp:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers, timeout):
            return DummyResp({"columns": ["id", "name"], "rows": [[1, "Alice"], [2, "Bob"]]})

    monkeypatch.setattr(server.httpx, "AsyncClient", DummyClient, raising=False)

    body = {
        "content": "SELECT id, name FROM users",
        "customer_id": "cust-1",
        "metadata": {"api_key": "secret"},
    }
    resp = client.post(
        "/api/v1/database/query",
        json=body,
        headers=make_auth_headers("db-user"),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["query_result"]["columns"] == ["id", "name"]
    assert data["query_result"]["rows"][0] == [1, "Alice"]
    assert data["dataset_id"]
    assert data["session_id"]


def test_database_query_http_error_returns_error_message(client: TestClient, monkeypatch):
    class FailingResp:
        def __init__(self, status_code=500, text="boom"):
            self.status_code = status_code
            self.text = text

        def json(self):
            return {}

        def raise_for_status(self):
            request = httpx.Request("POST", "http://example.com")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("boom", request=request, response=response)

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers, timeout):
            return FailingResp()

    monkeypatch.setattr(server.httpx, "AsyncClient", DummyClient, raising=False)

    body = {"content": "SELECT 1", "customer_id": "cust-err", "metadata": {"api_key": "secret"}}
    resp = client.post(
        "/api/v1/database/query",
        json=body,
        headers=make_auth_headers("db-user"),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data["query_result"]
    assert "Database server error" in data["query_result"]["error"]


# ======================================================================================
# database/tables-to-analysis
# ======================================================================================


def test_tables_to_analysis_requires_auth(client: TestClient):
    resp = client.post(
        "/api/database/tables-to-analysis",
        json={"tables": ["t1"], "customer_id": "c", "api_key": "k"},
    )
    assert resp.status_code == 401


def test_tables_to_analysis_empty_tables_422(client: TestClient):
    resp = client.post(
        "/api/database/tables-to-analysis",
        json={"tables": [], "customer_id": "c", "api_key": "k"},
        headers=make_auth_headers("tta-user"),
    )
    assert resp.status_code == 422


def test_tables_to_analysis_missing_credentials_400(client: TestClient):
    resp = client.post(
        "/api/database/tables-to-analysis",
        json={"tables": ["t1"]},
        headers=make_auth_headers("tta-user"),
    )
    assert resp.status_code == 400


def test_tables_to_analysis_creates_datasets(client: TestClient, monkeypatch):
    async def run_mcp_fake(customer_id, api_key, sql):
        return {"columns": ["id", "name"], "rows": [[1, "A"], [2, "B"]]}

    monkeypatch.setattr(server, "_run_mcp_db_query", run_mcp_fake, raising=False)

    resp = client.post(
        "/api/database/tables-to-analysis",
        json={"tables": ["users", "orders"], "customer_id": "c1", "api_key": "k1"},
        headers=make_auth_headers("tta-user"),
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["session_id"]
    assert len(out["datasets"]) == 2
    aliases = {d["alias"] for d in out["datasets"]}
    assert "users" in aliases and "orders" in aliases


def test_tables_to_analysis_rejects_invalid_table_name(client: TestClient):
    resp = client.post(
        "/api/database/tables-to-analysis",
        json={"tables": ["bad; DROP TABLE x"], "customer_id": "c", "api_key": "k"},
        headers=make_auth_headers("tta-user"),
    )
    assert resp.status_code == 400


# ======================================================================================
# Offloading heavy work off the event loop
# ======================================================================================


def test_heavy_sampling_runs_off_event_loop(client: TestClient, monkeypatch):
    import threading

    seen = {}

    def recording_sample_with_profiling(path, source_type, sample_size, use_ray=False, **kw):
        seen["thread"] = threading.current_thread().name
        return {
            "schema": {"a": "int", "b": "int"},
            "ddl_schema": "CREATE TABLE t(a int, b int);",
            "portfolio_samples": {"random_baseline": [{"a": 1, "b": 2}]},
            "sample_statistics": None,
        }

    monkeypatch.setattr(
        server, "sample_with_profiling", recording_sample_with_profiling, raising=False
    )

    out = _do_upload(client, user_id="offload-user")
    assert out["dataset_id"]
    assert seen.get("thread") is not None
    assert seen["thread"] != "MainThread"


def test_blob_io_runs_off_event_loop(client: TestClient, monkeypatch):
    import threading
    from app.services import storage_service

    seen = {}
    orig_put = storage_service.blob_store.put_file
    orig_get = storage_service.blob_store.get_file

    def rec_put(path, key):
        seen["put"] = threading.current_thread().name
        return orig_put(path, key)

    def rec_get(key, dest):
        seen["get"] = threading.current_thread().name
        return orig_get(key, dest)

    monkeypatch.setattr(storage_service.blob_store, "put_file", rec_put)
    monkeypatch.setattr(storage_service.blob_store, "get_file", rec_get)

    _do_upload(client, user_id="blobio-user")

    assert seen.get("put") and seen["put"] != "MainThread"
    assert seen.get("get") and seen["get"] != "MainThread"


def test_register_existing_sampler_runs_off_event_loop(client: TestClient, monkeypatch):
    import threading

    seen = {}

    def rec_sample_with_profiling(path, source_type, sample_size, use_ray=True, **kw):
        seen["thread"] = threading.current_thread().name
        return {
            "schema": {"a": "int", "b": "int"},
            "ddl_schema": "CREATE TABLE t(a int, b int);",
            "portfolio_samples": {"random_baseline": [{"a": 1, "b": 2}]},
            "sample_statistics": None,
        }

    monkeypatch.setattr(
        server, "sample_with_profiling", rec_sample_with_profiling, raising=False
    )

    # s3:// avoids the gcsfs streaming path -> falls back to download + sampler.
    resp = client.post(
        "/api/register-existing-storage",
        json={
            "storage_uri": "s3://external-bucket/some-prefix",
            "key": "foo.csv",
            "connection_id": "conn-1",
            "schema_json": None,
        },
        headers=make_auth_headers("reg-offload-user"),
    )
    assert resp.status_code == 200, resp.text
    assert seen.get("thread") is not None
    assert seen["thread"] != "MainThread"


# ======================================================================================
# Auth-config / lifespan / JWT
# ======================================================================================


def test_lifespan_refuses_empty_jwt_secret(monkeypatch):
    monkeypatch.setattr(server, "JWT_SECRET", "", raising=False)
    monkeypatch.setattr(server, "ALLOW_INSECURE_AUTH", False, raising=False)
    with pytest.raises(RuntimeError):
        with TestClient(server.app):
            pass


def test_lifespan_allows_startup_with_secret(monkeypatch):
    monkeypatch.setattr(server, "JWT_SECRET", "a-real-secret", raising=False)
    monkeypatch.setattr(server, "ALLOW_INSECURE_AUTH", False, raising=False)
    server._validate_auth_config()


def test_validate_auth_config_raises_without_secret(monkeypatch):
    monkeypatch.setattr(server, "JWT_SECRET", "", raising=False)
    monkeypatch.setattr(server, "ALLOW_INSECURE_AUTH", False, raising=False)
    with pytest.raises(RuntimeError):
        server._validate_auth_config()


def test_validate_auth_config_optout_allows_startup(monkeypatch):
    monkeypatch.setattr(server, "JWT_SECRET", "", raising=False)
    monkeypatch.setattr(server, "ALLOW_INSECURE_AUTH", True, raising=False)
    server._validate_auth_config()  # must not raise


def test_validate_auth_config_ok_with_secret(monkeypatch):
    monkeypatch.setattr(server, "JWT_SECRET", "a-real-secret", raising=False)
    monkeypatch.setattr(server, "ALLOW_INSECURE_AUTH", False, raising=False)
    server._validate_auth_config()


def _forge_empty_key_jwt(payload: Dict[str, Any]) -> str:
    import base64
    import hashlib
    import hmac

    def _b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64(json.dumps(payload).encode())
    signing_input = f"{header}.{body}".encode()
    sig = _b64(hmac.new(b"", signing_input, hashlib.sha256).digest())
    return f"{header}.{body}.{sig}"


def test_empty_jwt_secret_rejects_forged_token(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "JWT_SECRET", "", raising=False)
    forged = _forge_empty_key_jwt({"sub": "victim"})

    resp = client.get("/datasets", headers={"Authorization": f"Bearer {forged}"})
    assert resp.status_code == 401


def test_resolve_user_id_returns_none_without_secret(monkeypatch):
    monkeypatch.setattr(server, "JWT_SECRET", "", raising=False)
    forged = _forge_empty_key_jwt({"sub": "victim"})

    class _Req:
        headers = {"Authorization": f"Bearer {forged}"}
        cookies: Dict[str, str] = {}

    assert server._resolve_user_id(_Req()) is None


def test_resolve_user_id_valid_token(monkeypatch):
    monkeypatch.setattr(server, "JWT_SECRET", TEST_JWT_SECRET, raising=False)
    token = jwt.encode({"sub": "abc"}, TEST_JWT_SECRET, algorithm="HS256")

    class _Req:
        headers = {"Authorization": f"Bearer {token}"}
        cookies: Dict[str, str] = {}

    assert server._resolve_user_id(_Req()) == "abc"


def test_resolve_user_id_no_bearer(monkeypatch):
    monkeypatch.setattr(server, "JWT_SECRET", TEST_JWT_SECRET, raising=False)

    class _Req:
        headers: Dict[str, str] = {}
        cookies: Dict[str, str] = {}

    assert server._resolve_user_id(_Req()) is None


# ======================================================================================
# Delete dataset (source_kind aware)
# ======================================================================================


def _seed_dataset_session(user_id: str, dataset_id: str, source_kind, storage_uri, object_name):
    from app.services import session_service

    sid = f"sess-{dataset_id}"
    session_data = {
        "user_id": user_id,
        "dataset_id": dataset_id,
        "data_source_location": storage_uri,
        "object_name": object_name,
        "work_dir": None,
    }
    if source_kind is not None:
        session_data["source_kind"] = source_kind
    SESSIONS[sid] = session_data
    cache = session_service.cache
    cache._data[server._k_session(sid)] = json.dumps(session_data)
    cache._sets.setdefault(server._k_user_sessions(user_id), set()).add(sid)
    return sid


def _record_store_deletes(monkeypatch):
    from app.services import storage_service

    deleted = []
    monkeypatch.setattr(
        storage_service.blob_store, "delete", lambda key: deleted.append(key)
    )
    return deleted


def test_delete_dataset_removes_uploaded_object(client: TestClient, monkeypatch):
    deleted = _record_store_deletes(monkeypatch)
    _seed_dataset_session(
        "own-user", "ds-up", "uploaded", "gs://avaloka-bucket/uploads/f.csv", "uploads/f.csv"
    )

    resp = client.delete("/datasets/ds-up", headers=make_auth_headers("own-user"))
    assert resp.status_code == 204, resp.text
    assert deleted == ["uploads/f.csv"]


def test_delete_dataset_preserves_registered_customer_source(client: TestClient, monkeypatch):
    deleted = _record_store_deletes(monkeypatch)
    _seed_dataset_session(
        "own-user",
        "ds-reg",
        "registered",
        "gs://customer-bucket/their-data.csv",
        "their-data.csv",
    )

    resp = client.delete("/datasets/ds-reg", headers=make_auth_headers("own-user"))
    assert resp.status_code == 204, resp.text
    assert deleted == []
    from app.services import session_service

    assert server._k_session("sess-ds-reg") not in session_service.cache._data


def test_delete_dataset_legacy_untagged_is_not_deleted(client: TestClient, monkeypatch):
    deleted = _record_store_deletes(monkeypatch)
    _seed_dataset_session(
        "own-user", "ds-legacy", None, "gs://some-bucket/legacy.csv", "legacy.csv"
    )

    resp = client.delete("/datasets/ds-legacy", headers=make_auth_headers("own-user"))
    assert resp.status_code == 204, resp.text
    assert deleted == []


def test_delete_dataset_requires_auth(client: TestClient):
    resp = client.delete("/datasets/some-ds")
    assert resp.status_code == 401


def test_delete_dataset_unknown_404(client: TestClient):
    resp = client.delete("/datasets/nonexistent", headers=make_auth_headers("nobody"))
    assert resp.status_code == 404


# ======================================================================================
# Runtime graph checkpointer
# ======================================================================================


def test_runtime_graph_compiled_with_checkpointer(client: TestClient):
    assert server.GRAPH_READY is True, "runtime graph fell back to echo handler"
    assert getattr(server.GRAPH, "checkpointer", None) is not None, (
        "runtime GRAPH compiled without a checkpointer (thread_id would be a no-op)"
    )


# ======================================================================================
# Session tasks helper + task endpoints
# ======================================================================================


def test_session_tasks_helper_tolerates_missing_and_malformed():
    assert server._session_tasks(None) == []
    assert server._session_tasks({}) == []
    assert server._session_tasks({"tasks": ["t1", "t2"]}) == ["t1", "t2"]
    assert server._session_tasks({"tasks": '["t3"]'}) == ["t3"]
    assert server._session_tasks({"tasks": "not-json"}) == []
    assert server._session_tasks({"tasks": 42}) == []


def test_get_tasks_empty_for_new_session(client: TestClient):
    upload = _do_upload(client, user_id="tasks-user")
    session_id = upload["session_id"]

    headers = make_auth_headers("tasks-user")
    headers["X-Avaloka-Session"] = session_id
    resp = client.get("/tasks", headers=headers)

    assert resp.status_code == 200, resp.text
    assert resp.json() == []


def test_get_tasks_requires_auth(client: TestClient):
    resp = client.get("/tasks")
    assert resp.status_code == 401


def test_get_tasks_requires_session(client: TestClient):
    resp = client.get("/tasks", headers=make_auth_headers("tasks-no-sess"))
    assert resp.status_code == 400


def test_discover_session_tasks_filters_by_session_and_user(monkeypatch):
    class FakeRedis:
        def zrange(self, *_args):
            return [b"redbeat:owned", b"redbeat:other-user", b"redbeat:other-session"]

    states = {
        "redbeat:owned": {"session_id": "session-1", "user_id": "user-1"},
        "redbeat:other-user": {"session_id": "session-1", "user_id": "user-2"},
        "redbeat:other-session": {"session_id": "session-2", "user_id": "user-1"},
    }

    class FakeEntry:
        def __init__(self, key):
            self.name = key.removeprefix("redbeat:")
            self.args = [states[key]]

        @staticmethod
        def from_key(key, _app):
            return FakeEntry(key)

    monkeypatch.setattr(server, "get_redis", lambda _app: FakeRedis())
    monkeypatch.setattr(server, "AvalokaEntry", FakeEntry)

    assert server._discover_session_tasks("session-1", "user-1") == ["owned"]


def test_task_info_404_for_new_session(client: TestClient):
    upload = _do_upload(client, user_id="tasks-user-2")
    session_id = upload["session_id"]

    headers = make_auth_headers("tasks-user-2")
    headers["X-Avaloka-Session"] = session_id
    resp = client.get("/tasks/nonexistent-task/info", headers=headers)

    assert resp.status_code == 404


def test_delete_task_404_for_new_session(client: TestClient):
    upload = _do_upload(client, user_id="tasks-user-3")
    session_id = upload["session_id"]

    headers = make_auth_headers("tasks-user-3")
    headers["X-Avaloka-Session"] = session_id
    resp = client.delete("/tasks/nonexistent-task", headers=headers)

    assert resp.status_code == 404


def test_delete_task_preserves_concurrent_persist_key(client: TestClient, monkeypatch):
    class FakeEntry:
        @staticmethod
        def generate_key(app, tid):
            return f"k:{tid}"

        @staticmethod
        def from_key(key, app):
            return FakeEntry()

        def delete(self):
            return None

    monkeypatch.setattr(server, "AvalokaEntry", FakeEntry, raising=False)

    sid = "sess-task-merge"
    SESSIONS[sid] = {
        "user_id": "merge-user",
        "dataset_id": "ds-tm",
        "tasks": ["t1", "t2"],
        "gcs_code_object_key": "code-registry/keep.py",
    }

    headers = make_auth_headers("merge-user")
    headers["X-Avaloka-Session"] = sid
    resp = client.delete("/tasks/t1", headers=headers)
    assert resp.status_code == 200, resp.text

    sess = SESSIONS[sid]
    assert sess["tasks"] == ["t2"]
    assert sess["gcs_code_object_key"] == "code-registry/keep.py"


# ======================================================================================
# Background task status
# ======================================================================================


def test_background_task_status_requires_auth(client: TestClient):
    resp = client.get("/api/datasets/some-ds/background-task-status")
    assert resp.status_code == 401


def test_background_task_status_none_when_no_task(client: TestClient):
    upload = _do_upload(client, user_id="bgstatus-user")
    dsid = upload["dataset_id"]
    session_id = upload["session_id"]

    headers = make_auth_headers("bgstatus-user")
    headers["X-Avaloka-Session"] = session_id
    resp = client.get(f"/api/datasets/{dsid}/background-task-status", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "NONE"


# ======================================================================================
# Pure helpers
# ======================================================================================


def test_get_output_from_state_sanitizes_nan_values(tmp_path):
    output_path = tmp_path / "output.csv"
    output_path.write_text("Age,Name\n,Missing Age\n22,Known Age\n", encoding="utf-8")

    _, output_json = server.get_output_from_state(
        {"messages": [], "output_location": str(output_path)}
    )

    assert output_json[0]["Age"] is None
    json.dumps({"output_json": output_json}, allow_nan=False)


def test_get_output_from_state_parses_markdown_table():
    md = "| a | b |\n| - | - |\n| 1 | 2 |\n| 3 | 4 |"
    ai = AIMessage(content=md)
    _, output_json = server.get_output_from_state({"messages": [ai]})
    assert output_json == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_assistant_messages_from_task_state_supports_chat_results():
    messages = [
        {"role": "human", "content": "run inference"},
        {"role": "ai", "content": "Prediction: survived"},
    ]

    assert server._assistant_messages_from_state({"messages": messages}) == [
        {"role": "assistant", "content": "Prediction: survived"}
    ]


def test_mta_task_completion_message_uses_assistant_result():
    final = {}
    assistant_messages = [
        {
            "role": "assistant",
            "content": "Inference service setup is complete. The service is live at: https://gateway.example.dev",
        }
    ]
    metadata = {
        "task_type": "start_inference",
        "success_message": "Inference service setup is complete.",
    }

    assert server._scheduled_task_completion_message(final, assistant_messages, metadata) == (
        "Inference service setup is complete. The service is live at: https://gateway.example.dev"
    )


def test_scheduled_task_completion_message_falls_back_to_success_message():
    metadata = {"task_type": "execute", "success_message": "Done."}
    assert server._scheduled_task_completion_message({}, [], metadata) == "Done."


def test_scheduled_training_completion_message_hides_details_and_shows_actions():
    final = {
        "training_completed": False,
        "training_result": {
            "status": "error",
            "error": "OOMKilled in ray-worker-secret",
        },
    }
    metadata = {"task_type": "training"}

    message = server._scheduled_task_completion_message(final, [], metadata)

    assert "Model training could not be completed." in message
    assert "OOM_KILLED" in message
    assert "increase the training worker memory" in message.lower()
    assert "OOMKilled" not in message
    assert "ray-worker-secret" not in message


def test_json_safe_payload_handles_specials():
    import math

    assert server._json_safe_payload(float("nan")) is None
    assert server._json_safe_payload(float("inf")) is None
    assert server._json_safe_payload(3.5) == 3.5
    assert server._json_safe_payload({"x": [1, float("nan")]}) == {"x": [1, None]}
    assert server._json_safe_payload("s") == "s"
    assert server._json_safe_payload(True) is True


def test_sanitize_training_plan_defaults():
    out = server._sanitize_training_plan_for_response({})
    assert out["model_type"] == "classification"
    assert out["model_name"]
    assert out["model_version"]
    # Non-dict passes through unchanged.
    assert server._sanitize_training_plan_for_response(None) is None


def test_full_cloud_uri_prefers_folder_read_path():
    sess = {
        "folder_read_path": "gs://b/table/**/*.parquet",
        "data_source_location": "gs://b/table",
        "object_name": None,
    }
    assert server._full_cloud_uri(sess) == "gs://b/table/**/*.parquet"


def test_full_cloud_uri_joins_bucket_and_key():
    sess = {"data_source_location": "gs://b/prefix", "object_name": "obj.csv"}
    assert server._full_cloud_uri(sess) == "gs://b/prefix/obj.csv"


def test_full_cloud_uri_none_when_no_base():
    assert server._full_cloud_uri({"data_source_location": "", "object_name": "x"}) is None


def test_rows_list_to_dicts():
    out = server._rows_list_to_dicts(["a", "b"], [[1, 2], [3]])
    assert out == [{"a": 1, "b": 2}, {"a": 3, "b": None}]


def test_source_table_from_sql():
    assert server._source_table_from_sql("SELECT * FROM users WHERE 1=1") == "users"
    assert server._source_table_from_sql("select a from schema.tbl") == "schema.tbl"
    assert server._source_table_from_sql("not a query") is None


def test_first_usable_path(tmp_path):
    real = tmp_path / "real.csv"
    real.write_text("x")
    # cloud URI is always considered usable
    assert server._first_usable_path("gs://b/o", None) == "gs://b/o"
    # existing local file wins over a non-existent one
    assert server._first_usable_path("/nope/missing.csv", str(real)) == str(real)
    # nothing usable -> first non-empty
    assert server._first_usable_path("/a/missing", "/b/missing") == "/a/missing"
    assert server._first_usable_path(None, None) is None


def test_build_execution_context():
    ctx = server._build_execution_context(server.FIDELITY_PORTFOLIO, "random_baseline")
    assert ctx["mode"] == server.FIDELITY_PORTFOLIO
    assert ctx["sample"] == "random_baseline"
    assert ctx["mode_label"]


def test_human_size():
    assert server._human_size(0) == "0.00 B"
    assert server._human_size(1024).endswith("KB")
    assert server._human_size(1024 * 1024).endswith("MB")


def test_prompt_ts_to_iso_roundtrip():
    iso = server._prompt_ts_to_iso("20260407_143022")
    assert iso is not None and iso.startswith("2026-04-07T14:30:22")
    assert server._prompt_ts_to_iso("garbage") is None


def test_truthy_session_value():
    assert server._truthy_session_value(True) is True
    assert server._truthy_session_value("yes") is True
    assert server._truthy_session_value("1") is True
    assert server._truthy_session_value(None) is False
    assert server._truthy_session_value("no") is False


def test_merge_version_checkpoints_marks_current():
    sess = {
        "code_assets": [
            {"object_key": "code/v1.py", "prompt_ts": "20260101_000000"},
            {"object_key": "code/v2.py", "prompt_ts": "20260102_000000"},
        ],
        "output_assets": [
            {"object_key": "out/v2.csv", "prompt_ts": "20260102_000000"},
        ],
        "gcs_code_object_key": "code/v2.py",
    }
    checkpoints = server._merge_version_checkpoints(sess)
    # newest first
    assert checkpoints[0]["prompt_ts"] == "20260102_000000"
    assert checkpoints[0]["is_current"] is True
    assert checkpoints[0]["has_output"] is True
    assert checkpoints[1]["is_current"] is False


def test_source_kind_defaults_and_reset_regex():
    assert server._RESET_DATASET_RE.search("please reset the dataset")
    assert server._RESET_DATASET_RE.search("use the original dataset")
    assert server._RESET_DATASET_RE.search("go back to the original data")
    assert not server._RESET_DATASET_RE.search("show me the average")


# ======================================================================================
# Models endpoints (auth guards)
# ======================================================================================


def test_list_models_requires_auth(client: TestClient):
    resp = client.get("/api/models")
    assert resp.status_code == 401


def test_get_model_details_requires_auth(client: TestClient):
    resp = client.get("/api/models/run-1")
    assert resp.status_code == 401


def test_delete_model_requires_auth(client: TestClient):
    resp = client.delete("/api/models/run-1")
    assert resp.status_code == 401


def test_inference_requires_auth(client: TestClient):
    resp = client.post("/api/models/run-1/inference", json={"x": 1})
    assert resp.status_code == 401


# ======================================================================================
# Assets / insights / versions endpoints (auth guards)
# ======================================================================================


def test_get_session_assets_requires_auth(client: TestClient):
    resp = client.get("/api/assets/some-session")
    assert resp.status_code == 401


def test_get_session_assets_not_found(client: TestClient):
    resp = client.get("/api/assets/no-session", headers=make_auth_headers("assets-user"))
    assert resp.status_code == 404


def test_get_session_assets_returns_manifest(client: TestClient):
    upload = _do_upload(client, user_id="assets-owner")
    session_id = upload["session_id"]
    resp = client.get(f"/api/assets/{session_id}", headers=make_auth_headers("assets-owner"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"] == session_id
    assert "code_asset" in body and "output_asset" in body


def test_asset_code_url_requires_auth(client: TestClient):
    resp = client.get("/api/assets/some-session/code")
    assert resp.status_code == 401


def test_analysis_code_requires_auth(client: TestClient):
    resp = client.get("/analysis/aid-1/code")
    assert resp.status_code == 401


def test_analysis_versions_requires_auth(client: TestClient):
    resp = client.get("/analysis/aid-1/versions")
    assert resp.status_code == 401


def test_analysis_restore_requires_auth(client: TestClient):
    resp = client.post("/analysis/aid-1/restore", json={"prompt_ts": "20260101_000000"})
    assert resp.status_code == 401


def test_analysis_refresh_requires_auth(client: TestClient):
    resp = client.post("/analysis/aid-1/refresh")
    assert resp.status_code == 401


def test_save_and_execute_requires_auth(client: TestClient):
    resp = client.post("/analysis/aid-1/save-and-execute", json={"code": "print(1)"})
    assert resp.status_code == 401


def test_analysis_feedback_requires_auth(client: TestClient):
    resp = client.post(
        "/analysis/aid-1/feedback",
        json={"feedback_type": "positive"},
    )
    assert resp.status_code == 401


# ======================================================================================
# MCP credentials endpoints (auth guards)
# ======================================================================================


def test_encrypt_mcp_credentials_requires_auth(client: TestClient):
    resp = client.post("/api/mcp-connections/c1/encrypt", json={"apiKey": "k"})
    assert resp.status_code == 401


def test_decrypt_mcp_credentials_requires_auth(client: TestClient):
    resp = client.post("/api/mcp-connections/c1/decrypt")
    assert resp.status_code == 401


# ======================================================================================
# Deferred turn: /threads/{thread_id}/pending-turn
# ======================================================================================


def test_pending_turn_requires_auth(client: TestClient):
    resp = client.get("/threads/some-thread/pending-turn")
    assert resp.status_code == 401


def test_pending_turn_unknown_thread_404(client: TestClient):
    resp = client.get(
        "/threads/no-such-thread/pending-turn",
        headers=make_auth_headers("pt-user"),
    )
    assert resp.status_code == 404


def test_pending_turn_foreign_user_404(client: TestClient):
    upload = _do_upload(client, user_id="pt-owner")
    thread_id = upload["thread_id"]
    resp = client.get(
        f"/threads/{thread_id}/pending-turn",
        headers=make_auth_headers("pt-other"),
    )
    assert resp.status_code == 404


def test_pending_turn_none_when_absent(client: TestClient):
    upload = _do_upload(client, user_id="pt-none")
    thread_id = upload["thread_id"]
    resp = client.get(
        f"/threads/{thread_id}/pending-turn",
        headers=make_auth_headers("pt-none"),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "none"


def test_pending_turn_done_returns_result(client: TestClient):
    upload = _do_upload(client, user_id="pt-done")
    thread_id = upload["thread_id"]
    sid = THREAD_TO_SESSION[thread_id]
    SESSIONS[sid]["pending_turn"] = {
        "id": "d1", "status": "done", "result": {"foo": "bar"},
    }
    resp = client.get(
        f"/threads/{thread_id}/pending-turn",
        headers=make_auth_headers("pt-done"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["result"] == {"foo": "bar"}


def test_pending_turn_error_status(client: TestClient):
    upload = _do_upload(client, user_id="pt-err")
    thread_id = upload["thread_id"]
    sid = THREAD_TO_SESSION[thread_id]
    SESSIONS[sid]["pending_turn"] = {
        "id": "e1", "status": "error", "message": "Training failed.",
    }
    resp = client.get(
        f"/threads/{thread_id}/pending-turn",
        headers=make_auth_headers("pt-err"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["message"] == "Training failed."


def test_pending_turn_superseded_when_id_mismatch(client: TestClient):
    upload = _do_upload(client, user_id="pt-sup")
    thread_id = upload["thread_id"]
    sid = THREAD_TO_SESSION[thread_id]
    SESSIONS[sid]["pending_turn"] = {"id": "current", "status": "running"}
    resp = client.get(
        f"/threads/{thread_id}/pending-turn?deferred_id=old",
        headers=make_auth_headers("pt-sup"),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "superseded"


# ======================================================================================
# send_message: fast turn vs slow (deferred) turn
# ======================================================================================


def test_send_message_fast_turn_leaves_no_pending_handle(client: TestClient):
    """A turn that finishes within the deadline must not write a running handle."""
    upload = _do_upload(client, user_id="fast-user")
    thread_id = upload["thread_id"]
    dsid = upload["dataset_id"]
    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"content": "hello", "metadata": {"dataset_id": dsid}},
        headers=make_auth_headers("fast-user"),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("analysis_fidelity")
    assert body.get("execution_context")
    sid = THREAD_TO_SESSION[thread_id]
    assert "pending_turn" not in (SESSIONS.get(sid) or {})


def test_send_message_slow_turn_returns_running_then_done(client: TestClient, monkeypatch):
    """A turn that overruns the deadline returns a running handle and the
    background task later stashes the finished result for polling."""
    import time as _time

    monkeypatch.setattr(server, "TURN_SYNC_DEADLINE_S", 0.05, raising=False)
    monkeypatch.setattr(server, "_parse_fidelity_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_parse_selected_sample_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_is_mode_switch_message", lambda t: False, raising=False)

    class SlowGraph:
        checkpointer = object()

        def invoke(self, state_in, config=None):
            _time.sleep(0.4)  # overrun the 0.05s deadline reliably
            last = state_in["messages"][-1]
            content = getattr(last, "content", "")
            return {
                "messages": state_in["messages"] + [AIMessage(content=f"slow: {content}")],
                "planner_definition": {},
                "ready_to_summarize": False,
                "ready_to_code": False,
                "coder_definition": {},
                "visualization_config": {},
                "visualization_status": "",
                "execution_result": {},
            }

    monkeypatch.setattr(server, "GRAPH", SlowGraph(), raising=False)

    upload = _do_upload(client, user_id="slow-user")
    thread_id = upload["thread_id"]
    dsid = upload["dataset_id"]

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"content": "train a model", "metadata": {"dataset_id": dsid}},
        headers=make_auth_headers("slow-user"),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("training_status") == "running"
    deferred_id = body.get("analysis_task_id")
    assert deferred_id

    # The background task finishes ~0.4s later and stashes the result.
    final_status = None
    for _ in range(50):
        pr = client.get(
            f"/threads/{thread_id}/pending-turn?deferred_id={deferred_id}",
            headers=make_auth_headers("slow-user"),
        )
        assert pr.status_code == 200
        final_status = pr.json()["status"]
        if final_status == "done":
            break
        _time.sleep(0.05)
    assert final_status == "done"


# ======================================================================================
# send_message: reset-dataset fast path
# ======================================================================================


def test_send_message_reset_dataset_restores_snapshot(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "_parse_fidelity_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_parse_selected_sample_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_is_mode_switch_message", lambda t: False, raising=False)

    upload = _do_upload(client, user_id="reset-user")
    thread_id = upload["thread_id"]
    dsid = upload["dataset_id"]
    sid = upload["session_id"]

    # Simulate a prior transform that swapped in a new active source.
    sess = SESSIONS[sid]
    sess["data_source_was_modified"] = True
    sess["active_data_source_location"] = "/tmp/transformed.csv"
    sess["active_data_source_location_local"] = "/tmp/transformed.csv"
    sess["original_dataset_snapshot"] = {"work_local_input": sess.get("work_local_input")}

    resp = client.post(
        f"/threads/{thread_id}/messages",
        json={"content": "reset the dataset", "metadata": {"dataset_id": dsid}},
        headers=make_auth_headers("reset-user"),
    )
    assert resp.status_code == 200, resp.text
    assistant = [m for m in resp.json()["messages"] if m["role"] == "assistant"]
    assert assistant
    assert "Restored the original" in assistant[-1]["content"]
    assert "original_dataset_snapshot" not in SESSIONS[sid]


# ======================================================================================
# send_message: multi-dataset join routing (deterministic, pre-planner)
# ======================================================================================


def test_join_without_key_prompts_for_columns(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "_parse_fidelity_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_parse_selected_sample_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_is_mode_switch_message", lambda t: False, raising=False)
    monkeypatch.setattr(server, "suggest_join_keys", lambda state: [], raising=False)

    files = [
        ("files", ("a.csv", b"customer_id,name\n1,A\n", "text/csv")),
        ("files", ("b.csv", b"customer_id,amount\n1,10\n", "text/csv")),
    ]
    up = client.post("/api/upload", files=files, headers=make_auth_headers("join-user"))
    assert up.status_code == 200, up.text
    out = up.json()

    headers = make_auth_headers("join-user")
    headers["X-Avaloka-Session"] = out["session_id"]
    resp = client.post(
        f"/threads/{out['thread_id']}/messages",
        json={"content": "join the datasets"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assistant = [m for m in resp.json()["messages"] if m["role"] == "assistant"]
    assert assistant
    assert "join" in assistant[-1]["content"].lower()


# ======================================================================================
# Deferred-turn + dataset-snapshot pure helpers
# ======================================================================================


def test_pending_turn_state_helpers():
    running = {}
    server._pending_turn_running("d1")(running)
    assert running["pending_turn"]["status"] == "running"
    assert running["pending_turn"]["id"] == "d1"

    # A finished turn must never be downgraded back to running.
    done = {}
    server._pending_turn_done("d1", {"foo": "bar"})(done)
    server._pending_turn_running("d1")(done)
    assert done["pending_turn"]["status"] == "done"
    assert done["pending_turn"]["result"] == {"foo": "bar"}

    err = {}
    server._pending_turn_error("d2", "Training failed.")(err)
    assert err["pending_turn"]["status"] == "error"
    assert err["pending_turn"]["message"] == "Training failed."


def test_snapshot_and_restore_original_dataset_session():
    sess = {
        "work_local_input": "/data/orig.csv",
        "uploaded_csv_columns": ["a", "b"],
        "data_source_was_modified": True,
    }
    assert server._snapshot_original_dataset_session(sess) is True
    # second call is a no-op (snapshot already exists)
    assert server._snapshot_original_dataset_session(sess) is False
    assert "original_dataset_snapshot" in sess

    sess["active_data_source_location"] = "/data/transformed.csv"
    sess["work_local_input"] = "/data/transformed.csv"

    assert server._restore_original_dataset_session(sess) is True
    assert sess["work_local_input"] == "/data/orig.csv"
    assert "active_data_source_location" not in sess
    assert server._truthy_session_value(sess.get("data_source_was_modified")) is False
    assert "original_dataset_snapshot" not in sess


def test_restore_original_dataset_session_no_snapshot_is_noop():
    assert server._restore_original_dataset_session({"work_local_input": "/x.csv"}) is False


def test_training_finished_successfully():
    assert server._training_finished_successfully({}) is False
    assert server._training_finished_successfully({"training_completed": True}) is False
    assert server._training_finished_successfully(
        {"training_completed": True, "training_result": {"error": "boom"}}
    ) is False
    assert server._training_finished_successfully(
        {"training_completed": True, "training_result": {"mlflow_run_id": "run-1"}}
    ) is True


def test_sample_fidelity_note():
    portfolio = server._sample_fidelity_note(
        server.FIDELITY_PORTFOLIO, [{"a": 1}], "random_baseline"
    )
    assert "portfolio sample" in portfolio
    assert "random_baseline" in portfolio

    quick = server._sample_fidelity_note(server.FIDELITY_QUICK, [{"a": 1}] * 5, None)
    assert "sample" in quick
    assert "entire dataset" in quick.lower()


def test_record_latest_tabular_output(tmp_path):
    out = tmp_path / "output.csv"
    out.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    sess = {}
    changed = server._record_latest_tabular_output(
        sess, str(out), [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    )
    assert changed is True
    assert sess["latest_output_columns"]
    assert sess["latest_output_row_count"] == 2
    assert sess["latest_output_is_trainable"] is True

    # No output path -> no change.
    assert server._record_latest_tabular_output({}, None, None) is False


def test_resolve_selected_sample_rows_prefers_selected():
    sess = {
        "portfolio_samples": {
            "random_baseline": [{"a": 1}],
            "strat_col": [{"a": 2}],
        },
        "available_samples": ["random_baseline", "strat_col"],
        "selected_sample_name": "strat_col",
    }
    rows, chosen = server._resolve_selected_sample_rows(sess)
    assert chosen == "strat_col"
    assert rows == [{"a": 2}]


def test_resolve_selected_sample_rows_falls_back_to_default():
    sess = {
        "portfolio_samples": {"random_baseline": [{"a": 1}]},
        "available_samples": ["random_baseline"],
        "selected_sample_name": "does_not_exist",
    }
    rows, chosen = server._resolve_selected_sample_rows(sess)
    assert chosen == "random_baseline"
    assert rows == [{"a": 1}]


def test_resolve_selected_sample_rows_uses_fallback_rows():
    fallback = [{"z": 9}]
    rows, chosen = server._resolve_selected_sample_rows(
        {"portfolio_samples": {}, "available_samples": []}, fallback_rows=fallback
    )
    assert rows == fallback
    assert chosen is None


def test_build_multi_dataset_context_includes_paths_and_keys():
    parts = server._build_multi_dataset_context(
        [{"alias": "a", "data_source_location": "/data/a.csv", "columns": ["x", "y"]}],
        join_key_suggestions=[
            {"left_dataset": "a", "right_dataset": "b", "left_key": "id", "right_key": "id"}
        ],
    )
    joined = "\n".join(parts)
    assert "/data/a.csv" in joined
    assert "a.id" in joined


def test_format_join_suggestions():
    out = server._format_join_suggestions([
        {"left_dataset": "orders", "right_dataset": "users",
         "left_key": "user_id", "right_key": "id", "confidence": 0.9},
    ])
    assert "orders.user_id" in out
    assert "users.id" in out
    assert "confidence=0.9" in out





# ======================================================================================
# GitHub Integration feature — integration + end-to-end tests
# ======================================================================================
#
# These tests EXTEND test_server_integration.py. They reuse its `client` fixture,
# `make_auth_headers`, and the in-memory SESSIONS store, and cover:
#
#   * server.py  — the four /api/integrations/* endpoints (list/connect/patch/delete)
#   * integrations.py — resolve_github_config / _sync precedence + validate_github_repo_access
#   * persistence_service.py — persist_job_definition_to_git (.py commit + JSON toggle,
#                              per-user resolver, exec-status gating) and the
#                              _persist_assets_background git wiring (git_job_repo)
#
# The suite patches at the exact seams the code uses:
#   - server.py endpoints call the CRUD/validate helpers imported INTO the server
#     namespace (server.list_integration_connections, server.validate_github_repo_access,
#     server.encrypt_secret, server.decrypt_secret, ...).
#   - persistence_service imports resolve_github_config at CALL TIME from
#     app.api.integrations, and calls _do_git_put in its own module namespace.
#   - integrations.resolve_github_config calls get_integration_connection + decrypt_secret
#     in the integrations module namespace.
#
# Paste the contents below at the end of test_server_integration.py, OR keep this as a
# sibling file — it re-imports the shared fixture from that module so pytest collects it.
# ======================================================================================

import asyncio
import base64
import json

import pytest

import app.api.server as server
import app.api.integrations as integrations
import app.services.persistence_service as persistence

# Reuse the fixtures/helpers from the existing suite. If you paste this INTO
# test_server_integration.py, delete this import line (they're already in scope).
#from test_server_integration import client, make_auth_headers, SESSIONS  # noqa: F401


# ======================================================================================
# Helpers / fakes local to the integration tests
# ======================================================================================


def _github_ok(full_name="guruvaidev/avaloka-jobs-intenal", push=True):
    """A validate_github_repo_access() success payload."""
    return {"ok": True, "push": push, "error": None, "full_name": full_name}


def _github_fail(status_error="Invalid or expired token."):
    return {"ok": False, "push": False, "error": status_error, "full_name": None}


def _blob(ct="ct-abc", iv="iv-xyz"):
    """An encrypt_secret() return blob."""
    return {"ciphertext": ct, "iv": iv}


def _patch_encryption(monkeypatch):
    """encrypt/decrypt are imported into the server namespace by the endpoints."""
    monkeypatch.setattr(server, "encrypt_secret", lambda v: _blob(), raising=False)
    monkeypatch.setattr(server, "decrypt_secret", lambda blob: "ghp_decrypted_token", raising=False)


class _FakeSupabaseTable:
    """Chainable stand-in for supabase-py's table query builder.

    Records the last operation so tests can make CRUD deterministic without a DB.
    Backed by a shared dict keyed by (user_id, provider).
    """

    def __init__(self, store):
        self._store = store
        self._filters = {}
        self._op = None
        self._payload = None
        self._conflict = None

    def select(self, *_a, **_k):
        self._op = "select"
        return self

    def upsert(self, payload, on_conflict=None):
        self._op = "upsert"
        self._payload = payload
        self._conflict = on_conflict
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def limit(self, _n):
        return self

    def execute(self):
        key = (self._filters.get("user_id"), self._filters.get("provider"))
        if self._op == "select":
            row = self._store.get(key)
            # list_integration_connections selects by user_id only (no provider)
            if self._filters.get("provider") is None:
                rows = [v for k, v in self._store.items() if k[0] == self._filters.get("user_id")]
                return _Res(rows)
            return _Res([row] if row else [])
        if self._op == "upsert":
            # supabase-py upsert carries no .eq() filters — the key is IN the payload.
            pk = (self._payload.get("user_id"), self._payload.get("provider"))
            existing = self._store.get(pk) or {}
            merged = {**existing, **self._payload}
            self._store[pk] = merged
            return _Res([merged])
        if self._op == "delete":
            existed = key in self._store
            self._store.pop(key, None)
            return _Res([{"id": "deleted"}] if existed else [])
        return _Res([])


class _Res:
    def __init__(self, data):
        self.data = data


class _FakeSupabaseClient:
    def __init__(self, store):
        self._store = store

    def table(self, _name):
        return _FakeSupabaseTable(self._store)


# ======================================================================================
# server.py — GET /api/integrations
# ======================================================================================


def test_integrations_list_requires_auth(client):
    resp = client.get("/api/integrations")
    assert resp.status_code == 401


def test_integrations_list_all_providers_when_none_connected(client, monkeypatch):
    monkeypatch.setattr(server, "list_integration_connections", lambda uid: [], raising=False)
    monkeypatch.delenv("GITHUB_SYSTEM_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_JOB_REGISTRY_REPO", raising=False)

    resp = client.get("/api/integrations", headers=make_auth_headers("int-user"))
    assert resp.status_code == 200, resp.text
    providers = {p["provider"]: p for p in resp.json()["integrations"]}

    # All four supported providers are surfaced, all not-connected.
    assert set(providers) == {"github", "outlook", "slack", "jira"}
    for p in providers.values():
        assert p["enabled"] is False
        assert p["has_token"] is False
        assert p["using_system_default"] is False


def test_integrations_list_reflects_connected_github(client, monkeypatch):
    def _rows(uid):
        return [{
            "provider": "github",
            "enabled": True,
            "config": {"repo": "guruvaidev/avaloka-jobs-intenal"},
            "token_ciphertext": "ct-abc",
        }]
    monkeypatch.setattr(server, "list_integration_connections", _rows, raising=False)

    resp = client.get("/api/integrations", headers=make_auth_headers("int-user"))
    assert resp.status_code == 200, resp.text
    gh = {p["provider"]: p for p in resp.json()["integrations"]}["github"]
    assert gh["enabled"] is True
    assert gh["has_token"] is True
    assert gh["config"]["repo"] == "guruvaidev/avaloka-jobs-intenal"


def test_integrations_list_never_leaks_token(client, monkeypatch):
    def _rows(uid):
        return [{
            "provider": "github", "enabled": True,
            "config": {"repo": "o/r", "token": "SHOULD_NOT_APPEAR"},
            "token_ciphertext": "ct-abc",
        }]
    monkeypatch.setattr(server, "list_integration_connections", _rows, raising=False)

    resp = client.get("/api/integrations", headers=make_auth_headers("int-user"))
    assert resp.status_code == 200
    body_text = resp.text
    assert "SHOULD_NOT_APPEAR" not in body_text
    assert "token_ciphertext" not in body_text  # the raw column is never serialized


def test_integrations_list_flags_system_default_for_github(client, monkeypatch):
    monkeypatch.setattr(server, "list_integration_connections", lambda uid: [], raising=False)
    monkeypatch.setenv("GITHUB_SYSTEM_TOKEN", "ghp_env_token")
    monkeypatch.setenv("GITHUB_JOB_REGISTRY_REPO", "avaloka/env-repo")

    resp = client.get("/api/integrations", headers=make_auth_headers("int-user"))
    assert resp.status_code == 200
    providers = {p["provider"]: p for p in resp.json()["integrations"]}
    assert providers["github"]["using_system_default"] is True
    # Non-github providers never claim the system default.
    assert providers["slack"]["using_system_default"] is False


def test_integrations_list_surfaces_backend_error_as_500(client, monkeypatch):
    def _boom(uid):
        raise RuntimeError("supabase down")
    monkeypatch.setattr(server, "list_integration_connections", _boom, raising=False)

    resp = client.get("/api/integrations", headers=make_auth_headers("int-user"))
    assert resp.status_code == 500
    assert "Could not load integrations" in resp.json()["detail"]


# ======================================================================================
# server.py — POST /api/integrations/github/connect
# ======================================================================================


def test_github_connect_requires_auth(client):
    resp = client.post("/api/integrations/github/connect",
                       json={"token": "t", "repo": "o/r"})
    assert resp.status_code == 401


def test_github_connect_success_stores_encrypted_token(client, monkeypatch):
    _patch_encryption(monkeypatch)
    saved = {}

    async def _validate(token, repo):
        return _github_ok()

    def _save(uid, provider, *, enabled=None, config_updates=None,
              token_ciphertext=None, token_iv=None):
        saved.update({
            "uid": uid, "provider": provider, "enabled": enabled,
            "config_updates": config_updates,
            "token_ciphertext": token_ciphertext, "token_iv": token_iv,
        })
        return {"provider": provider, "enabled": enabled, "config": config_updates}

    monkeypatch.setattr(server, "validate_github_repo_access", _validate, raising=False)
    monkeypatch.setattr(server, "save_integration_connection", _save, raising=False)

    resp = client.post(
        "/api/integrations/github/connect",
        json={"token": "ghp_realtoken", "repo": "guruvaidev/avaloka-jobs-intenal"},
        headers=make_auth_headers("gh-connect-user"),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["enabled"] is True
    assert body["push_access"] is True
    assert body["repo"] == "guruvaidev/avaloka-jobs-intenal"

    # The token was encrypted before storage; only ciphertext/iv were saved.
    assert saved["enabled"] is True
    assert saved["token_ciphertext"] == "ct-abc"
    assert saved["token_iv"] == "iv-xyz"
    assert saved["config_updates"] == {"repo": "guruvaidev/avaloka-jobs-intenal"}
    # The raw PAT never reaches the storage layer.
    assert "ghp_realtoken" not in json.dumps(saved)


def test_github_connect_rejects_bad_repo_format(client, monkeypatch):
    _patch_encryption(monkeypatch)
    resp = client.post(
        "/api/integrations/github/connect",
        json={"token": "t", "repo": "not-a-valid-repo"},
        headers=make_auth_headers("gh-bad-repo"),
    )
    assert resp.status_code == 422
    assert "owner/repo" in resp.json()["detail"]


def test_github_connect_rejects_missing_token(client, monkeypatch):
    _patch_encryption(monkeypatch)
    # Empty token: Pydantic min_length=1 rejects it before the handler.
    resp = client.post(
        "/api/integrations/github/connect",
        json={"token": "", "repo": "o/r"},
        headers=make_auth_headers("gh-no-token"),
    )
    assert resp.status_code == 422


def test_github_connect_invalid_token_400(client, monkeypatch):
    _patch_encryption(monkeypatch)

    async def _validate(token, repo):
        return _github_fail("Invalid or expired token.")
    monkeypatch.setattr(server, "validate_github_repo_access", _validate, raising=False)

    resp = client.post(
        "/api/integrations/github/connect",
        json={"token": "ghp_bad", "repo": "o/r"},
        headers=make_auth_headers("gh-invalid"),
    )
    assert resp.status_code == 400
    assert "Invalid or expired token" in resp.json()["detail"]


def test_github_connect_no_push_access_403(client, monkeypatch):
    _patch_encryption(monkeypatch)

    async def _validate(token, repo):
        return _github_ok(push=False)   # readable but not writable
    monkeypatch.setattr(server, "validate_github_repo_access", _validate, raising=False)

    resp = client.post(
        "/api/integrations/github/connect",
        json={"token": "ghp_readonly", "repo": "o/r"},
        headers=make_auth_headers("gh-nopush"),
    )
    assert resp.status_code == 403
    assert "push" in resp.json()["detail"].lower()


def test_github_connect_missing_encryption_key_500(client, monkeypatch):
    async def _validate(token, repo):
        return _github_ok()
    monkeypatch.setattr(server, "validate_github_repo_access", _validate, raising=False)
    # encrypt_secret returns None when DB_ENCRYPTION_KEY isn't configured.
    monkeypatch.setattr(server, "encrypt_secret", lambda v: None, raising=False)

    resp = client.post(
        "/api/integrations/github/connect",
        json={"token": "ghp_x", "repo": "o/r"},
        headers=make_auth_headers("gh-nokey"),
    )
    assert resp.status_code == 500
    assert "DB_ENCRYPTION_KEY" in resp.json()["detail"]


def test_github_connect_uses_canonical_full_name_from_github(client, monkeypatch):
    """GitHub normalizes casing; the stored repo should be the API's full_name."""
    _patch_encryption(monkeypatch)
    saved = {}

    async def _validate(token, repo):
        return _github_ok(full_name="GuruvaiDev/Avaloka-Jobs-Intenal")

    def _save(uid, provider, *, enabled=None, config_updates=None, **kw):
        saved.update(config_updates or {})
        return {"config": config_updates}

    monkeypatch.setattr(server, "validate_github_repo_access", _validate, raising=False)
    monkeypatch.setattr(server, "save_integration_connection", _save, raising=False)

    resp = client.post(
        "/api/integrations/github/connect",
        json={"token": "t", "repo": "guruvaidev/avaloka-jobs-intenal"},
        headers=make_auth_headers("gh-canon"),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["repo"] == "GuruvaiDev/Avaloka-Jobs-Intenal"
    assert saved["repo"] == "GuruvaiDev/Avaloka-Jobs-Intenal"


# ======================================================================================
# server.py — PATCH /api/integrations/github
# ======================================================================================


def test_github_patch_requires_auth(client):
    resp = client.patch("/api/integrations/github", json={"enabled": False})
    assert resp.status_code == 401


def test_github_patch_enable_before_connect_409(client, monkeypatch):
    monkeypatch.setattr(server, "get_integration_connection", lambda uid, prov: None, raising=False)
    resp = client.patch(
        "/api/integrations/github",
        json={"enabled": True},
        headers=make_auth_headers("gh-patch-noconn"),
    )
    assert resp.status_code == 409
    assert "Configure GitHub first" in resp.json()["detail"]


def test_github_patch_toggle_off_keeps_token(client, monkeypatch):
    monkeypatch.setattr(
        server, "get_integration_connection",
        lambda uid, prov: {"token_ciphertext": "ct", "token_iv": "iv",
                           "config": {"repo": "o/r"}, "enabled": True},
        raising=False,
    )
    captured = {}

    def _save(uid, provider, *, enabled=None, config_updates=None, **kw):
        captured.update({"enabled": enabled, "config_updates": config_updates,
                         "token_touched": "token_ciphertext" in kw})
        return {"enabled": enabled, "config": {"repo": "o/r"}}

    monkeypatch.setattr(server, "save_integration_connection", _save, raising=False)

    resp = client.patch(
        "/api/integrations/github",
        json={"enabled": False},
        headers=make_auth_headers("gh-toggle"),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["enabled"] is False
    # A plain toggle must NOT resupply the token (it stays intact in storage).
    assert captured["enabled"] is False
    assert captured["config_updates"] is None


def test_github_patch_repo_change_revalidates_with_stored_token(client, monkeypatch):
    monkeypatch.setattr(
        server, "get_integration_connection",
        lambda uid, prov: {"token_ciphertext": "ct", "token_iv": "iv",
                           "config": {"repo": "o/old"}},
        raising=False,
    )
    monkeypatch.setattr(server, "decrypt_secret", lambda blob: "ghp_stored", raising=False)

    validated = {}

    async def _validate(token, repo):
        validated["token"] = token
        validated["repo"] = repo
        return _github_ok(full_name="o/new")

    def _save(uid, provider, *, enabled=None, config_updates=None, **kw):
        return {"enabled": True, "config": config_updates}

    monkeypatch.setattr(server, "validate_github_repo_access", _validate, raising=False)
    monkeypatch.setattr(server, "save_integration_connection", _save, raising=False)

    resp = client.patch(
        "/api/integrations/github",
        json={"repo": "o/new"},
        headers=make_auth_headers("gh-repo-change"),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["repo"] == "o/new"
    # The stored (decrypted) token was used to re-check the new repo.
    assert validated["token"] == "ghp_stored"
    assert validated["repo"] == "o/new"


def test_github_patch_repo_change_no_push_403(client, monkeypatch):
    monkeypatch.setattr(
        server, "get_integration_connection",
        lambda uid, prov: {"token_ciphertext": "ct", "token_iv": "iv"},
        raising=False,
    )
    monkeypatch.setattr(server, "decrypt_secret", lambda blob: "ghp_stored", raising=False)

    async def _validate(token, repo):
        return _github_ok(push=False)
    monkeypatch.setattr(server, "validate_github_repo_access", _validate, raising=False)

    resp = client.patch(
        "/api/integrations/github",
        json={"repo": "o/forbidden"},
        headers=make_auth_headers("gh-repo-nopush"),
    )
    assert resp.status_code == 403


def test_github_patch_bad_repo_format_422(client, monkeypatch):
    monkeypatch.setattr(
        server, "get_integration_connection",
        lambda uid, prov: {"token_ciphertext": "ct", "token_iv": "iv"},
        raising=False,
    )
    resp = client.patch(
        "/api/integrations/github",
        json={"repo": "nope"},
        headers=make_auth_headers("gh-repo-bad"),
    )
    assert resp.status_code == 422


# ======================================================================================
# server.py — DELETE /api/integrations/github
# ======================================================================================


def test_github_disconnect_requires_auth(client):
    resp = client.delete("/api/integrations/github")
    assert resp.status_code == 401


def test_github_disconnect_success_204(client, monkeypatch):
    called = {}

    def _delete(uid, provider):
        called.update({"uid": uid, "provider": provider})
        return True

    monkeypatch.setattr(server, "delete_integration_connection", _delete, raising=False)

    resp = client.delete("/api/integrations/github", headers=make_auth_headers("gh-del"))
    assert resp.status_code == 204
    assert called["provider"] == "github"


def test_github_disconnect_backend_error_500(client, monkeypatch):
    def _boom(uid, provider):
        raise RuntimeError("supabase down")
    monkeypatch.setattr(server, "delete_integration_connection", _boom, raising=False)

    resp = client.delete("/api/integrations/github", headers=make_auth_headers("gh-del-err"))
    assert resp.status_code == 500
    assert "Failed to remove the GitHub connection" in resp.json()["detail"]


# ======================================================================================
# integrations.py — resolve_github_config precedence (async + sync)
# ======================================================================================


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_resolve_github_config_prefers_connection(monkeypatch):
    monkeypatch.setattr(
        integrations, "get_integration_connection",
        lambda uid, prov: {"enabled": True, "token_ciphertext": "ct", "token_iv": "iv",
                           "config": {"repo": "user/repo"}},
        raising=False,
    )
    monkeypatch.setattr(integrations, "decrypt_secret", lambda blob: "ghp_conn", raising=False)
    # Env is set too, but the connection must win.
    monkeypatch.setenv("GITHUB_SYSTEM_TOKEN", "ghp_env")
    monkeypatch.setenv("GITHUB_JOB_REGISTRY_REPO", "env/repo")

    cfg = _run(integrations.resolve_github_config("u1"))
    assert cfg is not None
    assert cfg.source == "connection"
    assert cfg.token == "ghp_conn"
    assert cfg.repo == "user/repo"


def test_resolve_github_config_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(integrations, "get_integration_connection",
                        lambda uid, prov: None, raising=False)
    monkeypatch.setenv("GITHUB_SYSTEM_TOKEN", "ghp_env")
    monkeypatch.setenv("GITHUB_JOB_REGISTRY_REPO", "env/repo")

    cfg = _run(integrations.resolve_github_config("u1"))
    assert cfg is not None
    assert cfg.source == "env"
    assert cfg.token == "ghp_env"
    assert cfg.repo == "env/repo"


def test_resolve_github_config_none_when_nothing_available(monkeypatch):
    monkeypatch.setattr(integrations, "get_integration_connection",
                        lambda uid, prov: None, raising=False)
    monkeypatch.delenv("GITHUB_SYSTEM_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_JOB_REGISTRY_REPO", raising=False)

    assert _run(integrations.resolve_github_config("u1")) is None


def test_resolve_github_config_disabled_connection_falls_back(monkeypatch):
    # Connection exists but is disabled -> ignore it, use env.
    monkeypatch.setattr(
        integrations, "get_integration_connection",
        lambda uid, prov: {"enabled": False, "token_ciphertext": "ct", "token_iv": "iv",
                           "config": {"repo": "user/repo"}},
        raising=False,
    )
    monkeypatch.setenv("GITHUB_SYSTEM_TOKEN", "ghp_env")
    monkeypatch.setenv("GITHUB_JOB_REGISTRY_REPO", "env/repo")

    cfg = _run(integrations.resolve_github_config("u1"))
    assert cfg.source == "env"


def test_resolve_github_config_enabled_but_missing_token_falls_back(monkeypatch):
    # Enabled row but decrypt yields nothing -> fall back to env.
    monkeypatch.setattr(
        integrations, "get_integration_connection",
        lambda uid, prov: {"enabled": True, "token_ciphertext": "ct", "token_iv": "iv",
                           "config": {"repo": "user/repo"}},
        raising=False,
    )
    monkeypatch.setattr(integrations, "decrypt_secret", lambda blob: None, raising=False)
    monkeypatch.setenv("GITHUB_SYSTEM_TOKEN", "ghp_env")
    monkeypatch.setenv("GITHUB_JOB_REGISTRY_REPO", "env/repo")

    cfg = _run(integrations.resolve_github_config("u1"))
    assert cfg.source == "env"


def test_resolve_github_config_lookup_error_falls_back(monkeypatch):
    def _boom(uid, prov):
        raise RuntimeError("supabase down")
    monkeypatch.setattr(integrations, "get_integration_connection", _boom, raising=False)
    monkeypatch.setenv("GITHUB_SYSTEM_TOKEN", "ghp_env")
    monkeypatch.setenv("GITHUB_JOB_REGISTRY_REPO", "env/repo")

    cfg = _run(integrations.resolve_github_config("u1"))
    assert cfg is not None and cfg.source == "env"


def test_resolve_github_config_sync_matches_async(monkeypatch):
    monkeypatch.setattr(
        integrations, "get_integration_connection",
        lambda uid, prov: {"enabled": True, "token_ciphertext": "ct", "token_iv": "iv",
                           "config": {"repo": "user/repo"}},
        raising=False,
    )
    monkeypatch.setattr(integrations, "decrypt_secret", lambda blob: "ghp_conn", raising=False)

    cfg = integrations.resolve_github_config_sync("u1")
    assert cfg.source == "connection"
    assert cfg.token == "ghp_conn"
    assert cfg.repo == "user/repo"


# ======================================================================================
# integrations.py — validate_github_repo_access
# ======================================================================================


class _FakeGHResp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeGHClient:
    """AsyncClient stand-in that returns a preset response for GET /repos/...."""
    def __init__(self, resp=None, raise_exc=None):
        self._resp = resp
        self._raise = raise_exc

    def __init_subclass__(cls, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
        if self._raise:
            raise self._raise
        return self._resp


def _patch_gh_client(monkeypatch, resp=None, raise_exc=None):
    def _factory(*a, **k):
        return _FakeGHClient(resp=resp, raise_exc=raise_exc)
    monkeypatch.setattr(integrations.httpx, "AsyncClient", _factory, raising=False)


def test_validate_github_repo_access_ok_with_push(monkeypatch):
    _patch_gh_client(monkeypatch, resp=_FakeGHResp(
        200, {"full_name": "o/r", "permissions": {"push": True}}))
    out = _run(integrations.validate_github_repo_access("t", "o/r"))
    assert out == {"ok": True, "push": True, "error": None, "full_name": "o/r"}


def test_validate_github_repo_access_ok_without_push(monkeypatch):
    _patch_gh_client(monkeypatch, resp=_FakeGHResp(
        200, {"full_name": "o/r", "permissions": {"push": False}}))
    out = _run(integrations.validate_github_repo_access("t", "o/r"))
    assert out["ok"] is True and out["push"] is False


def test_validate_github_repo_access_401(monkeypatch):
    _patch_gh_client(monkeypatch, resp=_FakeGHResp(401))
    out = _run(integrations.validate_github_repo_access("t", "o/r"))
    assert out["ok"] is False and "Invalid or expired token" in out["error"]


def test_validate_github_repo_access_403(monkeypatch):
    _patch_gh_client(monkeypatch, resp=_FakeGHResp(403))
    out = _run(integrations.validate_github_repo_access("t", "o/r"))
    assert out["ok"] is False and "forbidden" in out["error"].lower()


def test_validate_github_repo_access_404(monkeypatch):
    _patch_gh_client(monkeypatch, resp=_FakeGHResp(404))
    out = _run(integrations.validate_github_repo_access("t", "o/r"))
    assert out["ok"] is False and "not found" in out["error"].lower()


def test_validate_github_repo_access_network_error(monkeypatch):
    _patch_gh_client(monkeypatch, raise_exc=RuntimeError("boom"))
    out = _run(integrations.validate_github_repo_access("t", "o/r"))
    assert out["ok"] is False and "Could not reach GitHub" in out["error"]


# ======================================================================================
# integrations.py — CRUD against a fake Supabase client
# ======================================================================================


def test_save_and_get_integration_connection_roundtrip(monkeypatch):
    store = {}
    monkeypatch.setattr(integrations, "get_supabase_client",
                        lambda: _FakeSupabaseClient(store), raising=False)

    # create
    integrations.save_integration_connection(
        "u1", "github", enabled=True,
        config_updates={"repo": "o/r"},
        token_ciphertext="ct", token_iv="iv",
    )
    row = integrations.get_integration_connection("u1", "github")
    assert row["enabled"] is True
    assert row["config"] == {"repo": "o/r"}
    assert row["token_ciphertext"] == "ct"


def test_save_integration_connection_merges_config(monkeypatch):
    store = {}
    monkeypatch.setattr(integrations, "get_supabase_client",
                        lambda: _FakeSupabaseClient(store), raising=False)

    integrations.save_integration_connection(
        "u1", "github", enabled=True,
        config_updates={"repo": "o/r", "extra": "keep"},
        token_ciphertext="ct", token_iv="iv",
    )
    # A repo-only update must preserve the other config key.
    integrations.save_integration_connection(
        "u1", "github", config_updates={"repo": "o/r2"})
    row = integrations.get_integration_connection("u1", "github")
    assert row["config"]["repo"] == "o/r2"
    assert row["config"]["extra"] == "keep"


def test_save_integration_connection_toggle_keeps_token(monkeypatch):
    store = {}
    monkeypatch.setattr(integrations, "get_supabase_client",
                        lambda: _FakeSupabaseClient(store), raising=False)

    integrations.save_integration_connection(
        "u1", "github", enabled=True, config_updates={"repo": "o/r"},
        token_ciphertext="ct", token_iv="iv")
    # Toggle without a token: ciphertext must survive.
    integrations.save_integration_connection("u1", "github", enabled=False)
    row = integrations.get_integration_connection("u1", "github")
    assert row["enabled"] is False
    assert row["token_ciphertext"] == "ct"


def test_list_integration_connections_scoped_to_user(monkeypatch):
    store = {}
    monkeypatch.setattr(integrations, "get_supabase_client",
                        lambda: _FakeSupabaseClient(store), raising=False)
    integrations.save_integration_connection("u1", "github", enabled=True,
                                             config_updates={"repo": "a/b"})
    integrations.save_integration_connection("u2", "github", enabled=True,
                                             config_updates={"repo": "c/d"})

    rows = integrations.list_integration_connections("u1")
    assert len(rows) == 1
    assert rows[0]["config"]["repo"] == "a/b"


def test_delete_integration_connection(monkeypatch):
    store = {}
    monkeypatch.setattr(integrations, "get_supabase_client",
                        lambda: _FakeSupabaseClient(store), raising=False)
    integrations.save_integration_connection("u1", "github", enabled=True,
                                             config_updates={"repo": "o/r"})
    assert integrations.delete_integration_connection("u1", "github") is True
    assert integrations.get_integration_connection("u1", "github") is None
    # deleting again reports nothing removed
    assert integrations.delete_integration_connection("u1", "github") is False


# ======================================================================================
# persistence_service.py — persist_job_definition_to_git
# ======================================================================================


def _mk_gh_config(source="connection", repo="guruvaidev/avaloka-jobs-intenal"):
    return integrations.GitHubConfig(token="ghp_x", repo=repo, source=source)


def _patch_git_put(monkeypatch):
    """Record every _do_git_put call (file_path, decoded content, commit_msg)."""
    calls = []

    async def _fake_put(repo, token, branch, file_path, content_b64, commit_msg):
        calls.append({
            "repo": repo, "token": token, "branch": branch,
            "file_path": file_path,
            "content": base64.b64decode(content_b64).decode("utf-8"),
            "commit_msg": commit_msg,
        })

    monkeypatch.setattr(persistence, "_do_git_put", _fake_put, raising=False)
    return calls


def test_persist_job_definition_commits_py_and_json(monkeypatch):
    calls = _patch_git_put(monkeypatch)

    async def _resolve(uid):
        return _mk_gh_config(source="connection")
    monkeypatch.setattr(integrations, "resolve_github_config", _resolve, raising=False)
    monkeypatch.setattr(persistence, "WRITE_JOB_DEFINITION_JSON", True, raising=False)

    result = _run(persistence.persist_job_definition_to_git(
        user_id="u1", session_id="s1", dataset_id="d1",
        planner_definition={"job_name": "x"},
        coder_definition={"code": "import pandas as pd\nprint('hi')"},
        execution_result={"status": "success"},
        code_object_key="code-registry/u1/s1/d1_ts_transform.py",
        prompt_ts="20260909_130305",
    ))

    assert result["status"] == "success"
    assert result["source"] == "connection"
    assert result["repo"] == "guruvaidev/avaloka-jobs-intenal"

    paths = [c["file_path"] for c in calls]
    # Exactly two commits: the .py FIRST (Leela's requirement), then the JSON.
    assert len(calls) == 2
    assert paths[0].endswith("_transform.py")
    assert paths[1].endswith("_job_definition.json")

    # The .py holds the real code, not a JSON blob.
    py = next(c for c in calls if c["file_path"].endswith(".py"))
    assert "import pandas as pd" in py["content"]
    assert "[Avaloka] Code persisted" in py["commit_msg"]

    # The JSON links back to the .py and carries the machine record.
    job_json = json.loads(next(c for c in calls if c["file_path"].endswith(".json"))["content"])
    assert job_json["dataset_id"] == "d1"
    assert job_json["code_git_path"].endswith("_transform.py")


def test_persist_job_definition_json_toggle_off_commits_only_py(monkeypatch):
    calls = _patch_git_put(monkeypatch)

    async def _resolve(uid):
        return _mk_gh_config()
    monkeypatch.setattr(integrations, "resolve_github_config", _resolve, raising=False)
    monkeypatch.setattr(persistence, "WRITE_JOB_DEFINITION_JSON", False, raising=False)

    result = _run(persistence.persist_job_definition_to_git(
        user_id="u1", session_id="s1", dataset_id="d1",
        planner_definition={"job_name": "x"},
        coder_definition={"code": "print('only py')"},
        execution_result={"status": "success"},
        code_object_key="k", prompt_ts="ts",
    ))
    assert result["status"] == "success"
    assert len(calls) == 1
    assert calls[0]["file_path"].endswith("_transform.py")


def test_persist_job_definition_skips_when_no_config(monkeypatch):
    _patch_git_put(monkeypatch)

    async def _resolve(uid):
        return None   # no connection AND no env fallback
    monkeypatch.setattr(integrations, "resolve_github_config", _resolve, raising=False)

    result = _run(persistence.persist_job_definition_to_git(
        user_id="u1", session_id="s1", dataset_id="d1",
        planner_definition={"job_name": "x"},
        coder_definition={"code": "print(1)"},
        execution_result={"status": "success"},
        code_object_key="k", prompt_ts="ts",
    ))
    assert result["status"] == "skipped"
    assert "no GitHub connection" in result["reason"]


def test_persist_job_definition_skips_on_non_success(monkeypatch):
    calls = _patch_git_put(monkeypatch)

    async def _resolve(uid):
        return _mk_gh_config()
    monkeypatch.setattr(integrations, "resolve_github_config", _resolve, raising=False)

    result = _run(persistence.persist_job_definition_to_git(
        user_id="u1", session_id="s1", dataset_id="d1",
        planner_definition={"job_name": "x"},
        coder_definition={"code": "print(1)"},
        execution_result={"status": "error"},   # not a success status
        code_object_key="k", prompt_ts="ts",
    ))
    assert result["status"] == "skipped"
    assert "not successful" in result["reason"]
    assert calls == []   # nothing committed


def test_persist_job_definition_no_code_still_commits_json(monkeypatch):
    calls = _patch_git_put(monkeypatch)

    async def _resolve(uid):
        return _mk_gh_config()
    monkeypatch.setattr(integrations, "resolve_github_config", _resolve, raising=False)
    monkeypatch.setattr(persistence, "WRITE_JOB_DEFINITION_JSON", True, raising=False)

    result = _run(persistence.persist_job_definition_to_git(
        user_id="u1", session_id="s1", dataset_id="d1",
        planner_definition={"job_name": "x"},
        coder_definition={},           # no code
        execution_result={"status": "success"},
        code_object_key="k", prompt_ts="ts",
    ))
    assert result["status"] == "success"
    # Only the JSON was written; no .py commit when there is no code.
    assert len(calls) == 1
    assert calls[0]["file_path"].endswith("_job_definition.json")


def test_persist_job_definition_reports_env_source(monkeypatch):
    _patch_git_put(monkeypatch)

    async def _resolve(uid):
        return _mk_gh_config(source="env", repo="avaloka/env-repo")
    monkeypatch.setattr(integrations, "resolve_github_config", _resolve, raising=False)
    monkeypatch.setattr(persistence, "WRITE_JOB_DEFINITION_JSON", False, raising=False)

    result = _run(persistence.persist_job_definition_to_git(
        user_id="u1", session_id="s1", dataset_id="d1",
        planner_definition={"job_name": "x"},
        coder_definition={"code": "print(1)"},
        execution_result={"status": "success"},
        code_object_key="k", prompt_ts="ts",
    ))
    assert result["source"] == "env"
    assert result["repo"] == "avaloka/env-repo"


# ======================================================================================
# persistence_service.py — _persist_assets_background wires git_job_repo
# ======================================================================================


def test_persist_assets_background_stores_git_repo(monkeypatch):
    """The background orchestrator must stash BOTH git_job_branch and git_job_repo
    onto the session so the asset endpoints can link to the right repo."""
    captured = {}

    # Task A (code -> store): success with an object key.
    async def _code(*a, **k):
        return {"status": "success", "object_key": "code-registry/x_transform.py"}
    monkeypatch.setattr(persistence, "persist_generated_code_to_store", _code, raising=False)

    # Task B (job -> git): success returning branch + repo.
    async def _git(*a, **k):
        return {"status": "success", "branch": "main",
                "repo": "guruvaidev/avaloka-jobs-intenal", "written": ["p.py"]}
    monkeypatch.setattr(persistence, "persist_job_definition_to_git", _git, raising=False)

    # Task C / D: skip (no output/viz) so we isolate the git wiring.
    async def _out(*a, **k):
        return {"status": "skipped"}
    async def _viz(*a, **k):
        return {"status": "skipped"}
    monkeypatch.setattr(persistence, "persist_execution_output_to_store", _out, raising=False)
    monkeypatch.setattr(persistence, "persist_visualization_to_store", _viz, raising=False)

    # Capture the session update the orchestrator performs.
    async def _update_session(session_id, mutator):
        sess = {}
        mutator(sess)
        captured.update(sess)
        return sess

    import app.services.session_service as ss
    monkeypatch.setattr(ss, "update_session", _update_session, raising=False)

    _run(persistence._persist_assets_background(
        user_id="u1", session_id="s1", dataset_id="d1",
        generated_code="print(1)",
        planner_definition={"job_name": "x"},
        coder_definition={"code": "print(1)"},
        execution_result={"status": "success"},
        exec_succeeded=True,
        output_file_data=None, output_json=None, output_location=None,
        visualization_config=None,
        connection_id=None, storage_uri=None,
    ))

    assert captured.get("git_job_branch") == "main"
    assert captured.get("git_job_repo") == "guruvaidev/avaloka-jobs-intenal"
    assert captured.get("gcs_code_object_key") == "code-registry/x_transform.py"


def test_persist_assets_background_skips_git_when_no_planner(monkeypatch):
    """Task B (git) only runs when exec_succeeded AND planner_definition are present."""
    git_called = {"n": 0}

    async def _code(*a, **k):
        return {"status": "success", "object_key": "code/x.py"}
    async def _git(*a, **k):
        git_called["n"] += 1
        return {"status": "success", "branch": "main", "repo": "o/r"}
    async def _skip(*a, **k):
        return {"status": "skipped"}

    monkeypatch.setattr(persistence, "persist_generated_code_to_store", _code, raising=False)
    monkeypatch.setattr(persistence, "persist_job_definition_to_git", _git, raising=False)
    monkeypatch.setattr(persistence, "persist_execution_output_to_store", _skip, raising=False)
    monkeypatch.setattr(persistence, "persist_visualization_to_store", _skip, raising=False)

    import app.services.session_service as ss
    async def _update_session(session_id, mutator):
        mutator({})
        return {}
    monkeypatch.setattr(ss, "update_session", _update_session, raising=False)

    _run(persistence._persist_assets_background(
        user_id="u1", session_id="s1", dataset_id="d1",
        generated_code="print(1)",
        planner_definition=None,          # <- no planner def
        coder_definition={"code": "print(1)"},
        execution_result={"status": "success"},
        exec_succeeded=True,
        output_file_data=None, output_json=None, output_location=None,
        visualization_config=None,
        connection_id=None, storage_uri=None,
    ))
    assert git_called["n"] == 0


# ======================================================================================
# End-to-end: connect via API, then a job push uses that connection
# ======================================================================================


def test_e2e_connect_then_job_push_uses_connection(client, monkeypatch):
    """
    Full green path in one test:
      1) POST /connect stores an (encrypted) github connection in a fake Supabase.
      2) persist_job_definition_to_git resolves THAT connection (source=connection)
         and commits transform.py to the connected repo.
    """
    store = {}
    # Both the server endpoints and the integrations resolver share the same fake DB.
    monkeypatch.setattr(integrations, "get_supabase_client",
                        lambda: _FakeSupabaseClient(store), raising=False)

    # --- 1) connect through the API ---
    # server endpoints use the server-namespace CRUD + validate + encrypt.
    monkeypatch.setattr(server, "save_integration_connection",
                        integrations.save_integration_connection, raising=False)
    monkeypatch.setattr(server, "encrypt_secret", lambda v: _blob("CT", "IV"), raising=False)

    async def _validate(token, repo):
        return _github_ok(full_name="guruvaidev/avaloka-jobs-intenal")
    monkeypatch.setattr(server, "validate_github_repo_access", _validate, raising=False)

    connect = client.post(
        "/api/integrations/github/connect",
        json={"token": "ghp_realtoken", "repo": "guruvaidev/avaloka-jobs-intenal"},
        headers=make_auth_headers("e2e-user"),
    )
    assert connect.status_code == 200, connect.text
    assert connect.json()["repo"] == "guruvaidev/avaloka-jobs-intenal"
    # the row landed in the fake DB, keyed by the JWT sub ("e2e-user")
    assert ("e2e-user", "github") in store
    assert store[("e2e-user", "github")]["token_ciphertext"] == "CT"

    # --- 2) job push resolves that connection ---
    # resolver decrypts the stored token; make decrypt deterministic.
    monkeypatch.setattr(integrations, "decrypt_secret", lambda blob: "ghp_realtoken", raising=False)
    calls = _patch_git_put(monkeypatch)
    monkeypatch.setattr(persistence, "WRITE_JOB_DEFINITION_JSON", False, raising=False)

    result = _run(persistence.persist_job_definition_to_git(
        user_id="e2e-user", session_id="s1", dataset_id="d1",
        planner_definition={"job_name": "clean"},
        coder_definition={"code": "print('e2e')"},
        execution_result={"status": "success"},
        code_object_key="k", prompt_ts="ts",
    ))

    assert result["status"] == "success"
    assert result["source"] == "connection"          # used the UI connection, not env
    assert result["repo"] == "guruvaidev/avaloka-jobs-intenal"
    assert len(calls) == 1
    assert calls[0]["repo"] == "guruvaidev/avaloka-jobs-intenal"
    assert calls[0]["token"] == "ghp_realtoken"       # the decrypted, stored token
    assert "print('e2e')" in calls[0]["content"]
