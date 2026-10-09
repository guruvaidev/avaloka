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




try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
    FakeBlobStore, FakeCache, FakeGraph, DummyResponse,
    SESSIONS, THREAD_TO_SESSION, PERSIST_CALLS, PROFILE_RESULT, TEST_JWT_SECRET,
    make_auth_headers, client, shared_loop_client,
    _run, _do_upload, _rows, _install_sampler, _wait_for, _quiet_send,
    save_session_fake, update_session_fake, delete_session_fake, get_session_fake,
    refresh_session_ttl_fake, find_session_by_dataset_for_user_fake,
    find_active_db_customer_for_user_fake, bind_thread_session_fake,
    get_thread_session_fake, _user_datasets_fake, _jsonify_fake,
    hydrate_thread_history_fake, persist_thread_history_fake,
    delete_thread_history_fake, persist_session_snapshot_fake,
    resolve_shared_session_fake, lg_request_fake, lg_json_fake,
    read_thread_msgs_fake,
    )
except ImportError:
    from server_harness import (  # noqa: F401
    FakeBlobStore, FakeCache, FakeGraph, DummyResponse,
    SESSIONS, THREAD_TO_SESSION, PERSIST_CALLS, PROFILE_RESULT, TEST_JWT_SECRET,
    make_auth_headers, client, shared_loop_client,
    _run, _do_upload, _rows, _install_sampler, _wait_for, _quiet_send,
    save_session_fake, update_session_fake, delete_session_fake, get_session_fake,
    refresh_session_ttl_fake, find_session_by_dataset_for_user_fake,
    find_active_db_customer_for_user_fake, bind_thread_session_fake,
    get_thread_session_fake, _user_datasets_fake, _jsonify_fake,
    hydrate_thread_history_fake, persist_thread_history_fake,
    delete_thread_history_fake, persist_session_snapshot_fake,
    resolve_shared_session_fake, lg_request_fake, lg_json_fake,
    read_thread_msgs_fake,
    )


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
# Upload performance changes: local copy, background push, capping, concurrency
# ======================================================================================


def test_upload_keeps_local_input_and_writes_sample_csv(client: TestClient):
    out = _do_upload(client, user_id="local-keep")
    sess = SESSIONS[out["session_id"]]

    work_input = Path(sess["work_local_input"])
    assert work_input.exists(), "local input must be kept (no GCS pull-back)"
    assert work_input.name == f"input_{out['dataset_id']}.csv"
    assert work_input.read_bytes() == b"a,b\n1,2\n3,4\n"

    # sample_input.csv is written alongside the LLM calls, before the response.
    sample_csv = Path(sess["sample_local_input"])
    assert sample_csv.exists()
    with sample_csv.open() as fh:
        assert len(list(csv.DictReader(fh))) == 2


def test_upload_pushes_object_to_store_in_background(client: TestClient):
    out = _do_upload(client, user_id="bg-push")
    object_name = f"{out['dataset_id']}.csv"
    store = storage_service.blob_store

    assert _wait_for(lambda: object_name in store.objects), "background push never landed"
    assert store.objects[object_name] == b"a,b\n1,2\n3,4\n"
    assert SESSIONS[out["session_id"]]["object_name"] == object_name


def test_upload_records_uri_when_push_finishes_first(client: TestClient, monkeypatch):
    # Slow sampler: the (fast) push completes while sampling runs.
    _install_sampler(monkeypatch, {"random_baseline": _rows(2)}, delay_s=0.3)

    out = _do_upload(client, user_id="push-first")
    sess = SESSIONS[out["session_id"]]
    assert sess.get("data_source_location") == (
        f"gs://fake-bucket/test/{out['dataset_id']}.csv"
    )


def test_upload_does_not_wait_for_slow_push(client: TestClient, monkeypatch):
    store = storage_service.blob_store
    orig_put = store.put_file

    def slow_put(path, key):
        time.sleep(1.0)
        return orig_put(path, key)

    monkeypatch.setattr(store, "put_file", slow_put)

    out = _do_upload(client, user_id="slow-push")
    sess = SESSIONS[out["session_id"]]
    # Key must be ABSENT (not None) while the push is running, so a stale
    # session copy merged back later can't overwrite the real URI.
    assert "data_source_location" not in sess
    # The push still completes in the background.
    assert _wait_for(lambda: f"{out['dataset_id']}.csv" in store.objects, timeout_s=5.0)


def test_upload_fails_when_push_already_failed(client: TestClient, monkeypatch):
    _install_sampler(monkeypatch, {"random_baseline": _rows(2)}, delay_s=0.3)

    def failing_put(path, key):
        raise RuntimeError("bucket unavailable")

    monkeypatch.setattr(storage_service.blob_store, "put_file", failing_put)

    files = {"file": ("test.csv", b"a,b\n1,2\n", "text/csv")}
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("push-fail"))
    assert resp.status_code == 500
    assert "Failed to upload to storage" in resp.json()["detail"]
    assert SESSIONS == {}, "failed upload must not leave a session behind"


def test_upload_empty_dataset_rolls_back(client: TestClient, monkeypatch):
    _install_sampler(monkeypatch, {"random_baseline": []})

    files = {"file": ("empty.csv", b"a,b\n", "text/csv")}
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("empty-user"))
    assert resp.status_code == 400
    assert "Empty Dataset" in resp.json()["detail"]
    assert SESSIONS == {}


def test_upload_caps_response_and_session_rows(client: TestClient, monkeypatch):
    cap = server.PREVIEW_RESPONSE_ROWS
    n = cap + 200
    _install_sampler(
        monkeypatch,
        {"random_baseline": _rows(n), "quantile_b": _rows(n)},
    )

    out = _do_upload(client, user_id="cap-user")
    expected_samples = min(n, server.DEFAULT_SAMPLE_MAX_ROWS, cap)

    # Response to the browser is capped.
    assert len(out["samples"]) == expected_samples
    assert out["rows_sampled"] == min(n, server.DEFAULT_SAMPLE_MAX_ROWS)
    for name, rows in out["portfolio_samples"].items():
        assert len(rows) <= cap, f"{name} not capped in response"
    assert set(out["available_samples"]) == {"random_baseline", "quantile_b"}

    # Redis session is capped and no longer carries the portfolio.
    sess = SESSIONS[out["session_id"]]
    assert len(sess["uploaded_csv_preview"]) <= cap
    assert sess["portfolio_samples"] is None

    # The full samples stay in-process for analysis.
    meta = server._profile_meta[out["dataset_id"]]
    assert len(meta["portfolio_samples"]["random_baseline"]) == n
    assert len(meta["portfolio_samples"]["quantile_b"]) == n


def test_upload_runs_profile_and_viz_concurrently(client: TestClient, monkeypatch):
    """Both stubs wait on the same barrier: if the handler ran them one after
    another, the first would time out and the upload would report a failure."""
    barrier = threading.Barrier(2, timeout=3.0)

    def profile_stub(**kw):
        barrier.wait()
        return {"full_profiling_result": dict(PROFILE_RESULT)}

    def viz_stub(**kw):
        barrier.wait()
        return {"visualization_status": "ready"}

    monkeypatch.setattr(server, "profile_full", profile_stub, raising=False)
    monkeypatch.setattr(server, "build_visualization_config_from_sample", viz_stub, raising=False)
    _install_sampler(
        monkeypatch,
        {"random_baseline": _rows(3)},
        sample_statistics={"column_statistics": {}, "data_quality": {}},
    )

    out = _do_upload(client, user_id="concurrent-user")
    assert out["profiling_result"] == PROFILE_RESULT
    assert out["visualization_status"] == "ready"


def test_upload_profile_failure_is_non_fatal(client: TestClient, monkeypatch):
    def profile_boom(**kw):
        raise RuntimeError("groq down")

    monkeypatch.setattr(server, "profile_full", profile_boom, raising=False)
    _install_sampler(
        monkeypatch,
        {"random_baseline": _rows(3)},
        sample_statistics={"column_statistics": {}},
    )

    out = _do_upload(client, user_id="profile-fail")
    assert out["profiling_result"] is None
    assert out["visualization_status"] == "ready"


def test_upload_viz_failure_marks_status_error(client: TestClient, monkeypatch):
    def viz_boom(**kw):
        raise RuntimeError("viz llm down")

    monkeypatch.setattr(server, "build_visualization_config_from_sample", viz_boom, raising=False)

    out = _do_upload(client, user_id="viz-fail")
    assert out["visualization_status"] == "error"
    assert SESSIONS[out["session_id"]]["visualization_status"] == "error"


def test_upload_schedules_persistence_with_profile(client: TestClient, monkeypatch):
    _install_sampler(
        monkeypatch,
        {"random_baseline": _rows(3)},
        sample_statistics={"column_statistics": {}},
    )

    out = _do_upload(client, user_id="persist-user")
    assert len(PERSIST_CALLS) == 1
    call = PERSIST_CALLS[0]
    assert call["dataset_id"] == out["dataset_id"]
    assert call["full_profiling_result"] == PROFILE_RESULT
    assert call["source_type"] == "csv"
    assert Path(call["source_path"]).name == f"input_{out['dataset_id']}.csv"


def test_preview_caps_rows(client: TestClient, monkeypatch):
    cap = server.PREVIEW_RESPONSE_ROWS
    n = cap + 200
    _install_sampler(monkeypatch, {"random_baseline": _rows(n), "quantile_b": _rows(n)})

    out = _do_upload(client, user_id="preview-cap")
    resp = client.get(
        f"/datasets/{out['dataset_id']}/preview",
        headers=make_auth_headers("preview-cap"),
    )
    assert resp.status_code == 200, resp.text
    prev = resp.json()
    assert 0 < len(prev["samples"]) <= cap
    for name, rows in (prev.get("portfolio_samples") or {}).items():
        assert len(rows) <= cap, f"{name} not capped in preview"


def test_send_message_uses_full_portfolio_from_profile_meta(client: TestClient, monkeypatch):
    """The session no longer stores the portfolio; chat must still analyse the
    FULL in-process sample, not the capped preview."""
    cap = server.PREVIEW_RESPONSE_ROWS
    n = cap + 200
    _install_sampler(monkeypatch, {"random_baseline": _rows(n)})

    captured: Dict[str, Any] = {}

    class CapturingGraph(FakeGraph):
        def invoke(self, state_in, config=None):
            captured["sample_data"] = state_in.get("sample_data")
            return super().invoke(state_in, config)

    monkeypatch.setattr(server, "GRAPH", CapturingGraph(), raising=False)

    load_calls: List[str] = []

    def load_portfolio_spy(dataset_id):
        load_calls.append(dataset_id)
        return None

    monkeypatch.setattr(server, "load_portfolio", load_portfolio_spy, raising=False)

    out = _do_upload(client, user_id="chat-full")
    resp = client.post(
        f"/threads/{out['thread_id']}/messages",
        json={"content": "average of b", "metadata": {"dataset_id": out["dataset_id"]}},
        headers=make_auth_headers("chat-full"),
    )
    assert resp.status_code == 200, resp.text

    assert len(captured["sample_data"]) == n
    assert load_calls == [], "portfolio should come from _profile_meta, not Supabase"
    # Writing state back must keep the session preview capped.
    assert len(SESSIONS[out["session_id"]]["uploaded_csv_preview"]) <= cap


def test_resolve_selected_sample_rows_caches_supabase_portfolio(client: TestClient, monkeypatch):
    calls: List[str] = []

    def load_portfolio_spy(dataset_id):
        calls.append(dataset_id)
        return {"random_baseline": [{"a": 1}]}

    monkeypatch.setattr(server, "load_portfolio", load_portfolio_spy, raising=False)

    sess = {"selected_sample_name": "random_baseline"}
    rows1, name1 = server._resolve_selected_sample_rows(sess, "ds-cache")
    rows2, name2 = server._resolve_selected_sample_rows(sess, "ds-cache")

    assert rows1 == rows2 == [{"a": 1}]
    assert name1 == name2 == "random_baseline"
    assert calls == ["ds-cache"], "second call must hit the _profile_meta cache"


# ======================================================================================
# Upload storage finalizer + helpers
# ======================================================================================


def test_finalize_upload_storage_records_uri(client: TestClient):
    SESSIONS["s-fin"] = {"user_id": "u", "dataset_id": "ds-fin"}

    async def main():
        async def put():
            return "gs://fake-bucket/test/ds-fin.csv"

        task = asyncio.create_task(put())
        await server._finalize_upload_storage(task, "s-fin", "ds-fin", "ds-fin.csv")

    _run(main())
    sess = SESSIONS["s-fin"]
    assert sess["data_source_location"] == "gs://fake-bucket/test/ds-fin.csv"
    # send_message must re-save the durable snapshot with the cloud URI.
    assert sess["snapshot_persisted"] is False


def test_finalize_upload_storage_keeps_existing_uri(client: TestClient):
    SESSIONS["s-keep"] = {
        "user_id": "u", "dataset_id": "ds-keep",
        "data_source_location": "gs://already/recorded.csv",
    }

    async def main():
        async def put():
            return "gs://fake-bucket/test/ds-keep.csv"

        task = asyncio.create_task(put())
        await server._finalize_upload_storage(task, "s-keep", "ds-keep", "ds-keep.csv")

    _run(main())
    assert SESSIONS["s-keep"]["data_source_location"] == "gs://already/recorded.csv"


def test_finalize_upload_storage_deletes_orphan_when_session_gone(client: TestClient, monkeypatch):
    deleted: List[Tuple[str, str]] = []
    monkeypatch.setattr(
        server, "_delete_orphan_upload",
        lambda uri, object_name: deleted.append((uri, object_name)),
        raising=False,
    )

    async def main():
        async def put():
            return "gs://fake-bucket/test/ds-orphan.csv"

        task = asyncio.create_task(put())
        await server._finalize_upload_storage(task, "no-such-session", "ds-orphan", "ds-orphan.csv")

    _run(main())
    assert deleted == [("gs://fake-bucket/test/ds-orphan.csv", "ds-orphan.csv")]


def test_finalize_upload_storage_deletes_orphan_on_dataset_mismatch(client: TestClient, monkeypatch):
    SESSIONS["s-other"] = {"user_id": "u", "dataset_id": "a-different-dataset"}
    deleted: List[str] = []
    monkeypatch.setattr(
        server, "_delete_orphan_upload",
        lambda uri, object_name: deleted.append(object_name),
        raising=False,
    )

    async def main():
        async def put():
            return "gs://fake-bucket/test/ds-mismatch.csv"

        task = asyncio.create_task(put())
        await server._finalize_upload_storage(task, "s-other", "ds-mismatch", "ds-mismatch.csv")

    _run(main())
    assert deleted == ["ds-mismatch.csv"]
    assert "data_source_location" not in SESSIONS["s-other"]


def test_finalize_upload_storage_records_push_error(client: TestClient):
    SESSIONS["s-err"] = {"user_id": "u", "dataset_id": "ds-err"}

    async def main():
        async def put():
            raise RuntimeError("403 forbidden")

        task = asyncio.create_task(put())
        await server._finalize_upload_storage(task, "s-err", "ds-err", "ds-err.csv")

    _run(main())
    sess = SESSIONS["s-err"]
    assert "403 forbidden" in sess["storage_error"]
    assert "data_source_location" not in sess


def test_delete_orphan_upload_removes_object(client: TestClient):
    store = storage_service.blob_store
    store.objects["orphan.csv"] = b"x"
    server._delete_orphan_upload("gs://fake-bucket/test/orphan.csv", "orphan.csv")
    assert "orphan.csv" not in store.objects


def test_delete_orphan_upload_never_raises(client: TestClient, monkeypatch):
    def boom(uri, object_name):
        raise RuntimeError("no store")

    monkeypatch.setattr(server, "_store_and_key_from_uri", boom, raising=False)
    server._delete_orphan_upload("gs://x/y.csv", "y.csv")  # must not raise


def test_put_upload_object_default_store(client: TestClient, tmp_path):
    src = tmp_path / "src.csv"
    src.write_bytes(b"a\n1\n")

    uri = _run(server._put_upload_object(src, "obj.csv", None, {}))
    assert uri == "gs://fake-bucket/test/obj.csv"
    assert storage_service.blob_store.objects["obj.csv"] == b"a\n1\n"


def test_put_upload_object_with_dest_uri_prefix(client: TestClient, tmp_path):
    src = tmp_path / "src.csv"
    src.write_bytes(b"a\n1\n")

    _run(server._put_upload_object(src, "obj.csv", "gs://fake-bucket/uploads", {}))
    # The fake resolver maps the URI's last segment to the base key.
    assert "uploads/obj.csv" in storage_service.blob_store.objects


def test_cap_rows_and_portfolio():
    cap = server.PREVIEW_RESPONSE_ROWS
    rows = _rows(cap + 10)
    assert len(server._cap_rows(rows)) == cap
    assert server._cap_rows(rows, 3) == rows[:3]
    assert server._cap_rows(None) is None
    assert server._cap_rows("not-a-list") == "not-a-list"

    portfolio = {"a": _rows(cap + 10), "b": _rows(2), "meta": {"x": 1}}
    capped = server._cap_portfolio(portfolio)
    assert len(capped["a"]) == cap
    assert len(capped["b"]) == 2
    assert capped["meta"] == {"x": 1}
    assert server._cap_portfolio(None) is None
    # Original is not mutated.
    assert len(portfolio["a"]) == cap + 10


def test_stage_propagates_exceptions():
    with pytest.raises(ValueError):
        with server._stage("boom", "ds-x"):
            raise ValueError("inside stage")

    with server._stage("ok", "ds-x"):
        pass


# ======================================================================================
# Datasets: list / get / preview
# ======================================================================================


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


def test_register_existing_caps_response_rows(client: TestClient, monkeypatch):
    cap = server.PREVIEW_RESPONSE_ROWS
    n = cap + 200
    _install_sampler(monkeypatch, {"random_baseline": _rows(n)})

    resp = client.post(
        "/api/register-existing-storage",
        json={"storage_uri": "s3://ext/prefix", "key": "foo.csv", "connection_id": "c1"},
        headers=make_auth_headers("reg-cap"),
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert len(out["samples"]) <= cap
    for rows in (out.get("portfolio_samples") or {}).values():
        assert len(rows) <= cap


# ======================================================================================
# Buckets listing (via list_hierarchy)
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
    """The upload push runs in a worker thread, and the upload no longer pulls
    the object back from storage (sampling reads the local copy)."""
    seen = {}
    store = storage_service.blob_store
    orig_put = store.put_file

    def rec_put(path, key):
        seen["put"] = threading.current_thread().name
        return orig_put(path, key)

    def rec_get(key, dest):
        seen["get"] = threading.current_thread().name
        raise AssertionError("upload must not download the object it just pushed")

    monkeypatch.setattr(store, "put_file", rec_put)
    monkeypatch.setattr(store, "get_file", rec_get)

    _do_upload(client, user_id="blobio-user")

    assert _wait_for(lambda: "put" in seen)
    assert seen["put"] != "MainThread"
    assert "get" not in seen


def test_register_existing_sampler_runs_off_event_loop(client: TestClient, monkeypatch):
    seen = {}

    def rec_sample_with_profiling(path, source_type, sample_size, use_ray=None, **kw):
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


def test_get_tasks_empty_for_new_session(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "_discover_session_tasks", lambda sid, uid: [], raising=False)
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


def test_multi_file_upload_creates_one_push_per_file(client: TestClient):
    files = [
        ("files", ("a.csv", b"x,y\n1,2\n", "text/csv")),
        ("files", ("b.csv", b"x,y\n3,4\n", "text/csv")),
    ]
    up = client.post("/api/upload", files=files, headers=make_auth_headers("multi-push"))
    assert up.status_code == 200, up.text
    dsids = [d["dataset_id"] for d in up.json()["datasets"]]
    assert len(dsids) == 2

    store = storage_service.blob_store
    assert _wait_for(lambda: all(f"{d}.csv" in store.objects for d in dsids))


# ======================================================================================
# Deferred-turn + dataset-snapshot pure helpers
# ======================================================================================


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
# Covers:
#   * server.py — the four /api/integrations/* endpoints (list/connect/patch/delete)
#   * integrations.py — resolve_github_config / _sync precedence + validate_github_repo_access
#   * persistence_service.py — persist_job_definition_to_git (.py commit + JSON toggle,
#                              per-user resolver, exec-status gating) and the
#                              _persist_assets_background git wiring (git_job_repo)
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
    """Chainable stand-in for supabase-py's table query builder, backed by a
    shared dict keyed by (user_id, provider)."""

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


# ---- server.py — GET /api/integrations ----


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
    assert "token_ciphertext" not in body_text


def test_integrations_list_flags_system_default_for_github(client, monkeypatch):
    monkeypatch.setattr(server, "list_integration_connections", lambda uid: [], raising=False)
    monkeypatch.setenv("GITHUB_SYSTEM_TOKEN", "ghp_env_token")
    monkeypatch.setenv("GITHUB_JOB_REGISTRY_REPO", "avaloka/env-repo")

    resp = client.get("/api/integrations", headers=make_auth_headers("int-user"))
    assert resp.status_code == 200
    providers = {p["provider"]: p for p in resp.json()["integrations"]}
    assert providers["github"]["using_system_default"] is True
    assert providers["slack"]["using_system_default"] is False


def test_integrations_list_surfaces_backend_error_as_500(client, monkeypatch):
    def _boom(uid):
        raise RuntimeError("supabase down")
    monkeypatch.setattr(server, "list_integration_connections", _boom, raising=False)

    resp = client.get("/api/integrations", headers=make_auth_headers("int-user"))
    assert resp.status_code == 500
    assert "Could not load integrations" in resp.json()["detail"]


# ---- server.py — POST /api/integrations/github/connect ----


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

    assert saved["enabled"] is True
    assert saved["token_ciphertext"] == "ct-abc"
    assert saved["token_iv"] == "iv-xyz"
    assert saved["config_updates"] == {"repo": "guruvaidev/avaloka-jobs-intenal"}
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
        return _github_ok(push=False)
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
    monkeypatch.setattr(server, "encrypt_secret", lambda v: None, raising=False)

    resp = client.post(
        "/api/integrations/github/connect",
        json={"token": "ghp_x", "repo": "o/r"},
        headers=make_auth_headers("gh-nokey"),
    )
    assert resp.status_code == 500
    assert "DB_ENCRYPTION_KEY" in resp.json()["detail"]


def test_github_connect_uses_canonical_full_name_from_github(client, monkeypatch):
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


# ---- server.py — PATCH /api/integrations/github ----


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


# ---- server.py — DELETE /api/integrations/github ----


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


# ---- integrations.py — resolve_github_config precedence (async + sync) ----


def test_resolve_github_config_prefers_connection(monkeypatch):
    monkeypatch.setattr(
        integrations, "get_integration_connection",
        lambda uid, prov: {"enabled": True, "token_ciphertext": "ct", "token_iv": "iv",
                           "config": {"repo": "user/repo"}},
        raising=False,
    )
    monkeypatch.setattr(integrations, "decrypt_secret", lambda blob: "ghp_conn", raising=False)
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


# ---- integrations.py — validate_github_repo_access ----


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


# ---- integrations.py — CRUD against a fake Supabase client ----


def test_save_and_get_integration_connection_roundtrip(monkeypatch):
    store = {}
    monkeypatch.setattr(integrations, "get_supabase_client",
                        lambda: _FakeSupabaseClient(store), raising=False)

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
    assert integrations.delete_integration_connection("u1", "github") is False


# ---- persistence_service.py — persist_job_definition_to_git ----


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
    assert len(calls) == 2
    assert paths[0].endswith("_transform.py")
    assert paths[1].endswith("_job_definition.json")

    py = next(c for c in calls if c["file_path"].endswith(".py"))
    assert "import pandas as pd" in py["content"]
    assert "[Avaloka] Code persisted" in py["commit_msg"]

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
        return None
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
        execution_result={"status": "error"},
        code_object_key="k", prompt_ts="ts",
    ))
    assert result["status"] == "skipped"
    assert "not successful" in result["reason"]
    assert calls == []


def test_persist_job_definition_no_code_still_commits_json(monkeypatch):
    calls = _patch_git_put(monkeypatch)

    async def _resolve(uid):
        return _mk_gh_config()
    monkeypatch.setattr(integrations, "resolve_github_config", _resolve, raising=False)
    monkeypatch.setattr(persistence, "WRITE_JOB_DEFINITION_JSON", True, raising=False)

    result = _run(persistence.persist_job_definition_to_git(
        user_id="u1", session_id="s1", dataset_id="d1",
        planner_definition={"job_name": "x"},
        coder_definition={},
        execution_result={"status": "success"},
        code_object_key="k", prompt_ts="ts",
    ))
    assert result["status"] == "success"
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


# ---- persistence_service.py — _persist_assets_background wires git_job_repo ----


def test_persist_assets_background_stores_git_repo(monkeypatch):
    captured = {}

    async def _code(*a, **k):
        return {"status": "success", "object_key": "code-registry/x_transform.py"}
    monkeypatch.setattr(persistence, "persist_generated_code_to_store", _code, raising=False)

    async def _git(*a, **k):
        return {"status": "success", "branch": "main",
                "repo": "guruvaidev/avaloka-jobs-intenal", "written": ["p.py"]}
    monkeypatch.setattr(persistence, "persist_job_definition_to_git", _git, raising=False)

    async def _skip(*a, **k):
        return {"status": "skipped"}
    monkeypatch.setattr(persistence, "persist_execution_output_to_store", _skip, raising=False)
    monkeypatch.setattr(persistence, "persist_visualization_to_store", _skip, raising=False)

    async def _update_session(session_id, mutator):
        sess = {}
        mutator(sess)
        captured.update(sess)
        return sess

    monkeypatch.setattr(session_service, "update_session", _update_session, raising=False)

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

    async def _update_session(session_id, mutator):
        mutator({})
        return {}
    monkeypatch.setattr(session_service, "update_session", _update_session, raising=False)

    _run(persistence._persist_assets_background(
        user_id="u1", session_id="s1", dataset_id="d1",
        generated_code="print(1)",
        planner_definition=None,
        coder_definition={"code": "print(1)"},
        execution_result={"status": "success"},
        exec_succeeded=True,
        output_file_data=None, output_json=None, output_location=None,
        visualization_config=None,
        connection_id=None, storage_uri=None,
    ))
    assert git_called["n"] == 0


# ---- End-to-end: connect via API, then a job push uses that connection ----


def test_e2e_connect_then_job_push_uses_connection(client, monkeypatch):
    store = {}
    monkeypatch.setattr(integrations, "get_supabase_client",
                        lambda: _FakeSupabaseClient(store), raising=False)

    # 1) connect through the API
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
    assert ("e2e-user", "github") in store
    assert store[("e2e-user", "github")]["token_ciphertext"] == "CT"

    # 2) job push resolves that connection
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
    assert result["source"] == "connection"
    assert result["repo"] == "guruvaidev/avaloka-jobs-intenal"
    assert len(calls) == 1
    assert calls[0]["repo"] == "guruvaidev/avaloka-jobs-intenal"
    assert calls[0]["token"] == "ghp_realtoken"
    assert "print('e2e')" in calls[0]["content"]













# ======================================================================================
# ADDITIONAL TESTS — append below the existing contents of test_server_integration.py
#
# These reuse the fixtures/helpers already defined in that file:
#   client, SESSIONS, THREAD_TO_SESSION, PERSIST_CALLS, FakeGraph, FakeBlobStore,
#   make_auth_headers, _do_upload, _install_sampler, _rows, _wait_for, _run, DummyResponse
# ======================================================================================

import hashlib

from fastapi import HTTPException


# --------------------------------------------------------------------------------------
# Shared helpers for the new tests
# --------------------------------------------------------------------------------------


def _send(client, upload, content, user, **extra):
    body = {"content": content, "metadata": {"dataset_id": upload["dataset_id"]}}
    body.update(extra)
    return client.post(
        f"/threads/{upload['thread_id']}/messages",
        json=body,
        headers=make_auth_headers(user),
    )


def _last_assistant(resp_json) -> str:
    msgs = [m for m in resp_json.get("messages", []) if m["role"] == "assistant"]
    assert msgs, f"no assistant message in {resp_json}"
    return msgs[-1]["content"]


def _patch_aid(monkeypatch, row):
    """Make _resolve_session_from_aid return `row` (or None)."""
    async def _fake(analysis_id, user_id):
        return row

    monkeypatch.setattr(server, "_resolve_session_from_aid", _fake, raising=False)


def _patch_signed_url(monkeypatch):
    async def _fake(key, conn_id=None, storage_uri=None):
        return f"https://signed.example/{key}" if key else None

    monkeypatch.setattr(server, "_generate_asset_signed_url", _fake, raising=False)


class _FakeReq:
    def __init__(self, headers=None, cookies=None):
        self.headers = headers or {}
        self.cookies = cookies or {}


# ======================================================================================
# _strip_blank_header_columns
# ======================================================================================


def test_strip_blank_header_columns_removes_trailing_comma(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b,\n1,2,\n3,4,\n", encoding="utf-8")

    dropped = server._strip_blank_header_columns(p)

    assert dropped == [2]
    assert p.read_text(encoding="utf-8") == "a,b\n1,2\n3,4\n"
    assert not (tmp_path / "t.csv.clean").exists()


def test_strip_blank_header_columns_keeps_blank_header_with_data(tmp_path):
    p = tmp_path / "t.csv"
    original = "a,b,\n1,2,x\n3,4,\n"
    p.write_text(original, encoding="utf-8")

    assert server._strip_blank_header_columns(p) == []
    assert p.read_text(encoding="utf-8") == original
    assert not (tmp_path / "t.csv.clean").exists()


def test_strip_blank_header_columns_noop_when_all_headers_named(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    assert server._strip_blank_header_columns(p) == []
    assert p.read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_strip_blank_header_columns_non_utf8_is_ignored(tmp_path):
    p = tmp_path / "t.csv"
    p.write_bytes(b"\xff\xfe\x00a,\x00b,\n")
    assert server._strip_blank_header_columns(p) == []


def test_strip_blank_header_columns_missing_file(tmp_path):
    assert server._strip_blank_header_columns(tmp_path / "missing.csv") == []


# ======================================================================================
# Upload: limits, overrides, multi-file grouping, header cleanup
# ======================================================================================


def test_upload_strips_trailing_comma_column_and_records_note(client: TestClient):
    files = {"file": ("trailing.csv", b"a,b,\n1,2,\n3,4,\n", "text/csv")}
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("strip-user"))
    assert resp.status_code == 200, resp.text
    out = resp.json()

    sess = SESSIONS[out["session_id"]]
    assert Path(sess["work_local_input"]).read_bytes() == b"a,b\n1,2\n3,4\n"
    notes = sess.get("ingest_notes") or []
    assert any("empty column" in n for n in notes)

    # The cleaned file is what gets pushed to storage.
    store = storage_service.blob_store
    key = f"{out['dataset_id']}.csv"
    assert _wait_for(lambda: key in store.objects)
    assert store.objects[key] == b"a,b\n1,2\n3,4\n"


def test_upload_too_many_files_413(client: TestClient, monkeypatch):
    monkeypatch.setenv("AVALOKA_MAX_UPLOAD_FILES", "1")
    files = [
        ("files", ("a.csv", b"x\n1\n", "text/csv")),
        ("files", ("b.csv", b"x\n2\n", "text/csv")),
    ]
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("too-many"))
    assert resp.status_code == 413
    assert resp.json()["detail"]["reason"] == "file_count"
    assert SESSIONS == {}


def test_upload_per_file_size_limit_413(client: TestClient, monkeypatch):
    monkeypatch.setenv("AVALOKA_MAX_UPLOAD_FILE_BYTES", "5")
    files = {"file": ("big.csv", b"a,b\n1,2\n3,4\n", "text/csv")}
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("big-file"))
    assert resp.status_code == 413
    assert resp.json()["detail"]["reason"] == "file_size"
    assert SESSIONS == {}
    # Temp upload files were cleaned up on rollback.
    assert not list(server.TMP_ROOT.glob("upload_*"))


def test_upload_total_size_limit_413(client: TestClient, monkeypatch):
    monkeypatch.setenv("AVALOKA_MAX_UPLOAD_TOTAL_BYTES", "20")
    files = [
        ("files", ("a.csv", b"a,b\n1,2\n3,4\n", "text/csv")),
        ("files", ("b.csv", b"a,b\n5,6\n7,8\n", "text/csv")),
    ]
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("total-big"))
    assert resp.status_code == 413
    assert resp.json()["detail"]["reason"] == "total_size"
    assert SESSIONS == {}


def test_upload_infers_extension_from_content_type(client: TestClient):
    files = {"file": ("noext", b"a,b\n1,2\n", "text/csv")}
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("ct-user"))
    assert resp.status_code == 200, resp.text
    sess = SESSIONS[resp.json()["session_id"]]
    assert sess["input_data_type"] == "csv"
    assert sess["object_name"].endswith(".csv")


def test_upload_schema_json_override_applied(client: TestClient):
    override = {"a": "string", "b": "string"}
    files = {"file": ("s.csv", b"a,b\n1,2\n", "text/csv")}
    resp = client.post(
        "/api/upload",
        files=files,
        data={"schema_json": json.dumps(override)},
        headers=make_auth_headers("schema-user"),
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["schema"] == override
    assert SESSIONS[out["session_id"]]["schema"] == override


def test_upload_invalid_schema_json_is_ignored(client: TestClient):
    files = {"file": ("s.csv", b"a,b\n1,2\n", "text/csv")}
    resp = client.post(
        "/api/upload",
        files=files,
        data={"schema_json": "{not json"},
        headers=make_auth_headers("schema-bad"),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["schema"] == {"a": "int", "b": "int"}


def test_upload_files_field_single_file_returns_multi_response(client: TestClient):
    files = [("files", ("only.csv", b"a,b\n1,2\n", "text/csv"))]
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("multi-one"))
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert "datasets" in out and len(out["datasets"]) == 1
    assert out["session_id"]


def test_multi_upload_builds_group_index(client: TestClient):
    files = [
        ("files", ("a.csv", b"x,y\n1,2\n", "text/csv")),
        ("files", ("b.csv", b"x,y\n3,4\n", "text/csv")),
    ]
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("group-user"))
    assert resp.status_code == 200, resp.text
    out = resp.json()

    group = SESSIONS[out["session_id"]]
    dsids = [d["dataset_id"] for d in out["datasets"]]
    assert group["dataset_ids"] == dsids
    assert set(group["dataset_session_map"]) == set(dsids)

    # Every member points back at the group and shares the thread.
    for dsid, sid in group["dataset_session_map"].items():
        member = SESSIONS[sid]
        assert member["dataset_id"] == dsid
        assert member["group_session_id"] == out["session_id"]
        assert member["thread_id"] == out["thread_id"]


def test_multi_upload_deduplicates_aliases(client: TestClient):
    files = [
        ("files", ("dup.csv", b"x\n1\n", "text/csv")),
        ("files", ("dup.csv", b"x\n2\n", "text/csv")),
    ]
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("dup-user"))
    assert resp.status_code == 200, resp.text
    aliases = [d["alias"] for d in resp.json()["datasets"]]
    assert len(aliases) == 2
    assert len(set(aliases)) == 2


def test_upload_sets_secure_cookie(client: TestClient, monkeypatch):
    monkeypatch.setenv("COOKIE_SECURE", "true")
    files = {"file": ("c.csv", b"a\n1\n", "text/csv")}
    resp = client.post("/api/upload", files=files, headers=make_auth_headers("cookie-user"))
    assert resp.status_code == 200, resp.text
    set_cookie = resp.headers.get("set-cookie", "")
    assert server.COOKIE_NAME in set_cookie
    assert "secure" in set_cookie.lower()
    assert "httponly" in set_cookie.lower()


# ======================================================================================
# register-existing-folder
# ======================================================================================


def _owned_conn(monkeypatch, user_id: str):
    async def _conn(connection_id: str) -> Dict[str, Any]:
        return {"id": connection_id, "user_id": user_id, "provider": "gcp"}

    monkeypatch.setattr(server, "get_cloud_connection", _conn, raising=False)


def _folder_body(**over):
    body = {"storage_uri": "gs://bkt", "folder": "tables/sales/", "connection_id": "conn-f"}
    body.update(over)
    return body


def test_register_folder_requires_auth(client: TestClient):
    resp = client.post("/api/register-existing-folder", json=_folder_body())
    assert resp.status_code == 401


def test_register_folder_foreign_connection_404(client: TestClient):
    # Fixture's get_cloud_connection returns {} -> no owner -> rejected.
    resp = client.post(
        "/api/register-existing-folder",
        json=_folder_body(),
        headers=make_auth_headers("folder-user"),
    )
    assert resp.status_code == 404


def test_register_folder_unrecognized_table_422(client: TestClient, monkeypatch):
    _owned_conn(monkeypatch, "folder-user")
    monkeypatch.setattr(
        server, "detect_folder_table_type", lambda keys: {"table_type": "unknown"}, raising=False
    )
    resp = client.post(
        "/api/register-existing-folder",
        json=_folder_body(),
        headers=make_auth_headers("folder-user"),
    )
    assert resp.status_code == 422
    assert "Not a recognized" in resp.json()["detail"]


def test_register_folder_iceberg_without_metadata_422(client: TestClient, monkeypatch):
    _owned_conn(monkeypatch, "folder-user")
    monkeypatch.setattr(
        server, "detect_folder_table_type", lambda keys: {"table_type": "iceberg"}, raising=False
    )
    resp = client.post(
        "/api/register-existing-folder",
        json=_folder_body(),
        headers=make_auth_headers("folder-user"),
    )
    assert resp.status_code == 422
    assert "metadata.json" in resp.json()["detail"]


def test_register_folder_parquet_success(client: TestClient, monkeypatch):
    _owned_conn(monkeypatch, "folder-user")
    monkeypatch.setattr(
        server,
        "detect_folder_table_type",
        lambda keys: {"table_type": "parquet_dir", "glob": "**/*.parquet", "hive_partitioning": False},
        raising=False,
    )
    seen: Dict[str, Any] = {}

    def _sampler(path, source_type, sample_size, use_ray=False, **kw):
        seen.update(path=path, source_type=source_type, use_ray=use_ray, kw=kw)
        return {
            "schema": {"a": "int", "b": "int"},
            "ddl_schema": "CREATE TABLE t(a int, b int);",
            "portfolio_samples": {"random_baseline": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]},
            "sample_statistics": None,
        }

    monkeypatch.setattr(server, "sample_with_profiling", _sampler, raising=False)

    resp = client.post(
        "/api/register-existing-folder",
        json=_folder_body(),
        headers=make_auth_headers("folder-user"),
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["rows_sampled"] == 2

    expected_glob = "gs://bkt/tables/sales/**/*.parquet"
    assert seen["path"] == expected_glob
    assert seen["source_type"] == "parquet"
    assert seen["use_ray"] is False  # tiny folder stays off the cluster
    assert seen["kw"]["hive_partitioning"] is False
    assert "cloud_credentials" in seen["kw"]

    sess = SESSIONS[out["session_id"]]
    assert sess["folder_read_path"] == expected_glob
    assert sess["folder_table_type"] == "parquet_dir"
    assert sess["input_data_type"] == "parquet"
    assert sess["source_kind"] == "registered"
    assert sess["object_name"] is None
    # The executable cloud source is the glob, not the bare folder.
    assert server._full_cloud_uri(sess) == expected_glob

    assert len(PERSIST_CALLS) == 1
    assert PERSIST_CALLS[0]["source_type"] == "parquet"
    assert out["dataset_id"] in server._profile_meta


def test_register_folder_sampler_error_500_cleans_workdir(client: TestClient, monkeypatch):
    _owned_conn(monkeypatch, "folder-user")
    monkeypatch.setattr(
        server,
        "detect_folder_table_type",
        lambda keys: {"table_type": "parquet_dir", "glob": "**/*.parquet"},
        raising=False,
    )
    monkeypatch.setattr(
        server, "sample_with_profiling",
        lambda **kw: {"error": "unreadable parquet"},
        raising=False,
    )
    before = set(server.TMP_ROOT.iterdir())

    resp = client.post(
        "/api/register-existing-folder",
        json=_folder_body(),
        headers=make_auth_headers("folder-user"),
    )
    assert resp.status_code == 500
    assert "unreadable parquet" in resp.json()["detail"]
    assert SESSIONS == {}
    assert set(server.TMP_ROOT.iterdir()) == before


# ======================================================================================
# /datasets listing
# ======================================================================================


def _seed_listed_session(user_id, sid, dataset_id, created_at, owner=None, **extra):
    data = {
        "user_id": owner or user_id,
        "dataset_id": dataset_id,
        "created_at": created_at,
        "file_size_bytes": 10,
        **extra,
    }
    cache = session_service.cache
    cache._data[server._k_session(sid)] = json.dumps(data)
    cache._sets.setdefault(server._k_user_sessions(user_id), set()).add(sid)


def test_list_datasets_sorted_deduped_and_scoped(client: TestClient):
    _seed_listed_session("lister", "s-old", "ds-old", "2026-01-01T00:00:00Z", filename="old.csv")
    _seed_listed_session("lister", "s-new", "ds-new", "2026-03-01T00:00:00Z", filename="new.csv",
                         alias="new", source_kind="uploaded", group_session_id="grp-1")
    _seed_listed_session("lister", "s-new-dup", "ds-new", "2026-03-01T00:00:00Z")
    # A blob that claims another owner must be filtered even if indexed here.
    _seed_listed_session("lister", "s-foreign", "ds-foreign", "2026-05-01T00:00:00Z", owner="intruder")
    # Index entry whose blob has expired.
    session_service.cache._sets[server._k_user_sessions("lister")].add("s-expired")

    resp = client.get("/datasets", headers=make_auth_headers("lister"))
    assert resp.status_code == 200, resp.text
    items = resp.json()

    assert [it["dataset_id"] for it in items] == ["ds-new", "ds-old"]
    new_item = items[0]
    assert new_item["size_bytes"] == 10
    assert new_item["group_session_id"] in ("grp-1",) or new_item["session_id"] == "s-new-dup"
    old_item = items[1]
    # Datasets uploaded alone fall back to their own session as the group.
    assert old_item["group_session_id"] == old_item["session_id"] == "s-old"
    assert old_item["filename"] == "old.csv"


def test_preview_foreign_user_404(client: TestClient):
    upload = _do_upload(client, user_id="prev-owner")
    resp = client.get(
        f"/datasets/{upload['dataset_id']}/preview",
        headers=make_auth_headers("prev-intruder"),
    )
    assert resp.status_code == 404


def test_preview_ignores_stale_session_id_of_other_user(client: TestClient):
    owner = _do_upload(client, user_id="stale-owner")
    mine = _do_upload(client, user_id="stale-me")
    # Pass the OTHER user's session id; it must be ignored, not trusted.
    resp = client.get(
        f"/datasets/{mine['dataset_id']}/preview",
        params={"session_id": owner["session_id"]},
        headers=make_auth_headers("stale-me"),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["session_id"] == mine["session_id"]


# ======================================================================================
# send_message routing
# ======================================================================================


class _MustNotRunGraph(FakeGraph):
    def invoke(self, state_in, config=None):
        raise AssertionError("graph must not be invoked for this turn")


class _RaisingGraph(FakeGraph):
    def __init__(self, exc):
        self.exc = exc

    def invoke(self, state_in, config=None):
        raise self.exc


class _OutputGraph(FakeGraph):
    """Writes a 2-row CSV to output_location and optionally emits a scalar sentinel."""

    def __init__(self, sentinel=None):
        self.sentinel = sentinel

    def invoke(self, state_in, config=None):
        Path(state_in["output_location"]).write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        stdout = "done"
        if self.sentinel is not None:
            stdout += (
                "\n<<<AVALOKA_RESULT>>>" + json.dumps(self.sentinel) + "<<<END_AVALOKA_RESULT>>>"
            )
        return {
            "messages": state_in["messages"] + [AIMessage(content="Result ready")],
            "execution_result": {"status": "success", "stdout": stdout},
            "ready_to_code": True,
            "coder_definition": {"code": "print(1)"},
            "visualization_config": {},
            "visualization_status": "",
        }


def test_send_message_metadata_fast_path_skips_graph(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)

    upload = _do_upload(client, user_id="meta-user")
    resp = _send(client, upload, "how many columns does this have?", "meta-user")
    assert resp.status_code == 200, resp.text
    text = _last_assistant(resp.json())
    assert "2 columns" in text
    assert "`a`" in text and "`b`" in text


def test_send_message_metadata_fast_path_disabled_after_modification(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    upload = _do_upload(client, user_id="meta-mod")
    SESSIONS[upload["session_id"]]["data_source_was_modified"] = True

    resp = _send(client, upload, "how many columns does this have?", "meta-mod")
    assert resp.status_code == 200, resp.text
    # Stored shape is stale once the data was transformed, so the graph answers.
    assert _last_assistant(resp.json()).startswith("Echo:")


def test_send_message_graph_failure_returns_friendly_message(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(
        server, "GRAPH", _RaisingGraph(RuntimeError("Rate limit reached for model")), raising=False
    )
    upload = _do_upload(client, user_id="err-user")

    resp = _send(client, upload, "average of b", "err-user")
    assert resp.status_code == 200, resp.text
    text = _last_assistant(resp.json())
    assert "busy" in text
    assert "Traceback" not in text
    # The friendly message is also recorded in thread history.
    hist = server.THREAD_META[upload["thread_id"]]["lc_msgs"]
    assert any("busy" in getattr(m, "content", "") for m in hist)


def test_send_message_returns_tabular_output_and_records_latest(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _OutputGraph(), raising=False)
    upload = _do_upload(client, user_id="tab-user")

    resp = _send(client, upload, "show rows grouped by a", "tab-user")
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["output_json"] == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

    sess = SESSIONS[upload["session_id"]]
    assert sess.get("latest_output_location")
    assert sess.get("latest_output_row_count") == 2
    assert sess.get("latest_output_is_trainable") is True


def test_send_message_scalar_answer_suppresses_table(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(
        server, "GRAPH", _OutputGraph(sentinel={"kind": "scalar", "value": 42}), raising=False
    )
    upload = _do_upload(client, user_id="scalar-user")

    resp = _send(client, upload, "what is the max of b", "scalar-user")
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["output_json"] is None
    assert out["output_file_data"] is None
    assert "latest_output_location" not in SESSIONS[upload["session_id"]]


def test_send_message_unknown_sample_rejected(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    upload = _do_upload(client, user_id="sample-user")

    resp = _send(client, upload, "switch sample", "sample-user", selected_sample_name="nope")
    assert resp.status_code == 200, resp.text
    text = _last_assistant(resp.json())
    assert "'nope' is not available" in text
    assert "random_baseline" in text


def test_send_message_stream_returns_sse(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    upload = _do_upload(client, user_id="sse-user")

    resp = _send(client, upload, "hello", "sse-user", stream=True)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "event: message" in resp.text
    assert "event: done" in resp.text


def test_send_message_compare_returns_join_suggestions(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    monkeypatch.setattr(
        server, "suggest_join_keys",
        lambda state: [{"left_dataset": "a", "right_dataset": "b",
                        "left_key": "customer_id", "right_key": "customer_id",
                        "confidence": 0.95}],
        raising=False,
    )
    files = [
        ("files", ("a.csv", b"customer_id,name\n1,A\n", "text/csv")),
        ("files", ("b.csv", b"customer_id,amount\n1,10\n", "text/csv")),
    ]
    up = client.post("/api/upload", files=files, headers=make_auth_headers("cmp-user")).json()

    headers = make_auth_headers("cmp-user")
    headers["X-Avaloka-Session"] = up["session_id"]
    resp = client.post(
        f"/threads/{up['thread_id']}/messages",
        json={"content": "compare these datasets"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    text = _last_assistant(resp.json())
    assert "a.customer_id" in text
    assert "Join on customer_id" in text


def test_send_message_join_with_explicit_key_reaches_graph_with_context(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "suggest_join_keys", lambda state: [], raising=False)
    files = [
        ("files", ("a.csv", b"customer_id,name\n1,A\n", "text/csv")),
        ("files", ("b.csv", b"customer_id,amount\n1,10\n", "text/csv")),
    ]
    up = client.post("/api/upload", files=files, headers=make_auth_headers("join-key")).json()

    headers = make_auth_headers("join-key")
    headers["X-Avaloka-Session"] = up["session_id"]
    resp = client.post(
        f"/threads/{up['thread_id']}/messages",
        json={"content": "join the two tables on customer_id"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    text = _last_assistant(resp.json())
    # FakeGraph echoes the augmented prompt: both datasets must be described.
    assert text.startswith("Echo:")
    assert text.count("[Dataset:") == 2
    assert "csv_path:" in text


def test_send_message_foreign_session_header_is_ignored(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    victim = _do_upload(client, user_id="victim")

    headers = make_auth_headers("attacker")
    headers["X-Avaloka-Session"] = victim["session_id"]
    resp = client.post(
        f"/threads/{victim['thread_id']}/messages",
        json={"content": "show me everything"},
        headers=headers,
    )
    assert resp.status_code == 400


# ======================================================================================
# Pure helpers: metadata questions / answers / job card
# ======================================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("how many columns does this have?", True),
        ("list the columns", True),
        ("show the schema", True),
        ("how many rows", True),
        ("what is the date range", True),
        ("what are the data types", True),
        ("what is the average of b", False),
        ("how many rows have nulls", False),
        ("plot the columns", False),
        ("", False),
        ("how many columns " + "x" * 200, False),
    ],
)
def test_is_metadata_only_question(text, expected):
    assert server._is_metadata_only_question(text) is expected


def test_build_metadata_answer_columns_with_dtypes():
    sess = {"schema": {"a": "int", "b": "string"}}
    ans = server._build_metadata_answer(sess, "how many columns")
    assert "**2 columns**" in ans
    assert "`a` — int" in ans and "`b` — string" in ans


def test_build_metadata_answer_columns_without_dtypes():
    sess = {"schema": ["a", "b", "c"]}
    ans = server._build_metadata_answer(sess, "list the columns")
    assert "**3 columns**" in ans
    assert "`a`, `b`, `c`" in ans


def test_build_metadata_answer_exact_rows():
    sess = {"sample_statistics": {"data_shape": {"rows": 1234}}, "sample_status": "full_sample"}
    ans = server._build_metadata_answer(sess, "how many rows")
    assert "1,234 rows" in ans and "(exact)" in ans


def test_build_metadata_answer_estimated_rows():
    sess = {"sample_statistics": {"data_shape": {"rows": 500}}, "sample_status": "quick_sample"}
    ans = server._build_metadata_answer(sess, "how many rows")
    assert "approximately 500 rows" in ans


def test_build_metadata_answer_partial_measurement_never_quotes_head_count():
    sess = {
        "file_size_bytes": 10_000_000,
        "sample_status": "full_sample",
        "sample_statistics": {
            "data_shape": {"rows": 50},
            "estimated_full_bytes": 1000,
            "bytes_per_row": 100,
        },
    }
    ans = server._build_metadata_answer(sess, "how many rows")
    assert "100,000+ rows" in ans
    assert "50 rows" not in ans


def test_build_metadata_answer_date_range_from_stats():
    sess = {
        "sample_statistics": {
            "column_statistics": {
                "order_date": {"min": "2024-01-01", "max": "2024-12-31"},
                "amount": {"min": 1, "max": 9},
            }
        }
    }
    ans = server._build_metadata_answer(sess, "what is the date range")
    assert "`order_date`: 2024-01-01 → 2024-12-31" in ans
    assert "amount" not in ans


def test_build_metadata_answer_date_range_unknown_returns_none():
    assert server._build_metadata_answer({}, "what is the date range") is None


def test_build_deferred_job_card_has_no_numbers_and_truncates():
    long_q = "q" * 400
    card = server._build_deferred_job_card("task-123", "sales", long_q, "about 1 hour")
    assert "`task-123`" in card
    assert "sales" in card
    assert "about 1 hour" in card
    assert "q" * 297 + "..." in card
    assert "q" * 301 not in card


# ======================================================================================
# Pure helpers: size estimation, row totals, partial-sample detection
# ======================================================================================


def test_estimated_inmemory_bytes_measured(monkeypatch):
    monkeypatch.setenv("ANALYSIS_MEM_EXPANSION", "4.0")
    sess = {"file_size_bytes": 1000,
            "sample_statistics": {"estimated_full_bytes": 5000, "bytes_per_row": 50}}
    assert server._estimated_inmemory_bytes(sess) == (5000, 50.0, "measured")


def test_estimated_inmemory_bytes_partial_measurement_floored(monkeypatch):
    monkeypatch.setenv("ANALYSIS_MEM_EXPANSION", "4.0")
    sess = {"file_size_bytes": 1000,
            "sample_statistics": {"estimated_full_bytes": 100, "bytes_per_row": 10}}
    est, bpr, src = server._estimated_inmemory_bytes(sess)
    assert (est, src) == (4000, "expansion_floor")


def test_estimated_inmemory_bytes_from_rows_times_bpr(monkeypatch):
    sess = {"sample_statistics": {"bytes_per_row": 10, "data_shape": {"rows": 20}}}
    assert server._estimated_inmemory_bytes(sess) == (200, 10.0, "measured")


def test_estimated_inmemory_bytes_expansion_fallback(monkeypatch):
    monkeypatch.setenv("ANALYSIS_MEM_EXPANSION", "3.0")
    assert server._estimated_inmemory_bytes({"file_size_bytes": 100}) == (300, 0.0, "expansion")


def test_estimated_inmemory_bytes_bad_env_uses_default(monkeypatch):
    monkeypatch.setenv("ANALYSIS_MEM_EXPANSION", "not-a-number")
    assert server._estimated_inmemory_bytes({"file_size_bytes": 100})[0] == 400


def test_estimated_inmemory_bytes_none():
    assert server._estimated_inmemory_bytes({}) == (0, 0.0, "none")


def test_session_total_rows_variants():
    assert server._session_total_rows({"sample_statistics": {"data_shape": {"rows": 10}}}) == 10
    assert server._session_total_rows(
        {"sample_statistics": json.dumps({"data_shape": {"rows": 7}})}
    ) == 7
    assert server._session_total_rows({"total_rows_exact": 5}) == 5
    assert server._session_total_rows({"total_rows_exact": True}) is None
    assert server._session_total_rows({"sample_statistics": {"data_shape": {"rows": 0}}}) is None
    assert server._session_total_rows({}) is None


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(effective_fidelity="entire_dataset", sample_n=10, total_rows=10,
              is_large_dataset=False, entire_on_head=False), False),
        (dict(effective_fidelity="entire_dataset", sample_n=10, total_rows=100,
              is_large_dataset=True, entire_on_head=True), True),
        (dict(effective_fidelity="quick_sample", sample_n=10, total_rows=10,
              is_large_dataset=True, entire_on_head=False), True),
        (dict(effective_fidelity="portfolio_samples", sample_n=10, total_rows=1000,
              is_large_dataset=False, entire_on_head=False), False),
        (dict(effective_fidelity="quick_sample", sample_n=50, total_rows=100,
              is_large_dataset=False, entire_on_head=False), True),
        (dict(effective_fidelity="quick_sample", sample_n=100, total_rows=100,
              is_large_dataset=False, entire_on_head=False), False),
        (dict(effective_fidelity="quick_sample", sample_n=5, total_rows=None,
              is_large_dataset=False, entire_on_head=False), False),
    ],
)
def test_analysis_ran_on_partial_sample(kwargs, expected):
    assert server._analysis_ran_on_partial_sample(**kwargs) is expected


def test_analysis_ran_on_partial_sample_unknown_total_at_cap():
    assert server._analysis_ran_on_partial_sample(
        effective_fidelity="quick_sample",
        sample_n=server.DEFAULT_SAMPLE_MAX_ROWS,
        total_rows=None,
        is_large_dataset=False,
        entire_on_head=False,
    ) is True


def test_sample_fidelity_note_reports_row_counts():
    note = server._sample_fidelity_note("quick_sample", [{}] * 1000, None, total_rows=5000)
    assert "1,000-row sample of the 5,000-row file" in note


def test_build_execution_context_flags_sampling():
    ctx = server._build_execution_context("quick_sample", None, rows_analyzed=10, total_rows=100)
    assert ctx["is_sampled"] is True
    full = server._build_execution_context("entire_dataset", None, rows_analyzed=100, total_rows=100)
    assert full["is_sampled"] is False
    unknown = server._build_execution_context(None, None)
    assert unknown["mode"] == server.FIDELITY_QUICK
    assert unknown["is_sampled"] is False


# ======================================================================================
# Pure helpers: scalar results, checkpoints, csv helpers
# ======================================================================================


def test_scalar_result_from_sentinel_nested():
    final = {"execution_result": {"logs": [
        "noise",
        {"stdout": 'x <<<AVALOKA_RESULT>>>{"kind":"scalar","value":7}<<<END_AVALOKA_RESULT>>> y'},
    ]}}
    assert server._scalar_result_from_state(final, None) == {"kind": "scalar", "value": 7}


def test_scalar_result_from_one_by_one_table():
    out = server._scalar_result_from_state({}, [{"count": 3}])
    assert out == {"kind": "scalar", "value": 3, "columns": ["count"]}


def test_scalar_result_malformed_sentinel_falls_back():
    final = {"execution_result": {"stdout": "<<<AVALOKA_RESULT>>>not json<<<END_AVALOKA_RESULT>>>"}}
    assert server._scalar_result_from_state(final, [{"a": 1, "b": 2}]) is None
    assert server._scalar_result_from_state(final, [{"n": 9}])["value"] == 9


def test_scalar_result_none_for_tables():
    assert server._scalar_result_from_state({}, [{"a": 1}, {"a": 2}]) is None
    assert server._scalar_result_from_state({}, None) is None


def test_iter_strings_walks_nested():
    assert sorted(server._iter_strings({"a": ["x", {"b": "y"}], "c": 1, "d": ("z",)})) == ["x", "y", "z"]


def test_is_new_checkpoint_worthy():
    assert server._is_new_checkpoint_worthy({}, [{"a": 1}], False) is False
    assert server._is_new_checkpoint_worthy({}, [{"a": 1}], True) is True

    out = [{"a": 1}]
    fp = hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()
    sess = {"gcs_output_object_key": "k", "last_output_fingerprint": fp}
    assert server._is_new_checkpoint_worthy(sess, out, True) is False
    assert server._is_new_checkpoint_worthy(sess, [{"a": 2}], True) is True


def test_count_csv_rows_and_rows_to_columns(tmp_path):
    p = tmp_path / "o.csv"
    p.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    assert server._count_csv_rows(p) == 3
    assert server._count_csv_rows(tmp_path / "missing.csv") is None

    assert server._rows_to_columns([{"x": 1, "y": 2}]) == ["x", "y"]
    assert server._rows_to_columns([]) == []
    assert server._rows_to_columns(None) == []
    assert server._rows_to_columns([[1, 2]]) == []


def test_record_latest_tabular_output_reads_file_when_no_preview(tmp_path):
    p = tmp_path / "o.csv"
    p.write_text("a\n1\n", encoding="utf-8")
    sess: Dict[str, Any] = {}
    assert server._record_latest_tabular_output(sess, str(p)) is True
    assert sess["latest_output_columns"] == ["a"]
    assert sess["latest_output_row_count"] == 1
    # Single column is not trainable.
    assert sess["latest_output_is_trainable"] is False


def test_record_latest_tabular_output_missing_or_empty(tmp_path):
    assert server._record_latest_tabular_output({}, str(tmp_path / "nope.csv")) is False
    empty = tmp_path / "e.csv"
    empty.write_text("a,b\n", encoding="utf-8")
    assert server._record_latest_tabular_output({}, str(empty)) is False


# ======================================================================================
# Pure helpers: auth / config / misc
# ======================================================================================


def test_friendly_turn_error_messages():
    assert "busy" in server._friendly_turn_error(RuntimeError("rate limit exceeded"))
    assert "temporarily unavailable" in server._friendly_turn_error(
        RuntimeError("error code: model_not_found")
    )
    assert "credentials" in server._friendly_turn_error(RuntimeError("Invalid API Key provided"))
    generic = server._friendly_turn_error(ValueError("kaboom"))
    assert "Something went wrong" in generic
    assert "kaboom" not in generic


def test_is_supabase_auth_failure():
    class _CodeErr(Exception):
        def __init__(self, code):
            super().__init__("err")
            self.code = code

    assert server._is_supabase_auth_failure(_CodeErr(401)) is True
    assert server._is_supabase_auth_failure(_CodeErr("403")) is True
    assert server._is_supabase_auth_failure(_CodeErr("PGRST116")) is False
    assert server._is_supabase_auth_failure(RuntimeError("Invalid API key")) is True
    assert server._is_supabase_auth_failure(RuntimeError("Supabase env vars not set")) is True
    assert server._is_supabase_auth_failure(RuntimeError("connection reset")) is False


def test_read_version_env_override(monkeypatch):
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    assert server._read_version() == "9.9.9"


def test_read_version_oss_edition(monkeypatch):
    monkeypatch.delenv("APP_VERSION", raising=False)
    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    monkeypatch.delenv("AVALOKA_OSS_VERSION", raising=False)
    assert server._read_version() == "1.0.0"
    monkeypatch.setenv("AVALOKA_OSS_VERSION", "1.0.3")
    assert server._read_version() == "1.0.3"


def test_resolve_session_id_precedence():
    req = _FakeReq(headers={"X-Avaloka-Session": "hdr"}, cookies={server.COOKIE_NAME: "cookie"})
    assert server._resolve_session_id(req, "body") == "body"
    assert server._resolve_session_id(req, None) == "hdr"
    assert server._resolve_session_id(_FakeReq(cookies={server.COOKIE_NAME: "cookie"}), None) == "cookie"
    assert server._resolve_session_id(_FakeReq(), None) is None


def test_build_asset_history_parses_and_filters():
    sess = {"code_assets": json.dumps([
        {"object_key": "k1", "prompt_ts": "20260101_000000"},
        {"prompt_ts": "missing-key"},
        "garbage",
    ])}
    hist = server._build_asset_history(sess, "code_assets")
    assert [(h.object_key, h.prompt_ts) for h in hist] == [("k1", "20260101_000000")]
    assert server._build_asset_history({"code_assets": "not json"}, "code_assets") == []
    assert server._build_asset_history({}, "code_assets") == []


def test_group_members_snapshot_drops_none_values():
    out = server._group_members_snapshot([
        ("ds1", "s1", {"filename": "a.csv", "alias": None, "schema": {"x": "int"}, "secret": "nope"}),
    ])
    assert out == [{"filename": "a.csv", "schema": {"x": "int"},
                    "dataset_id": "ds1", "session_id": "s1"}]


def test_scheduled_task_has_chat_result_only():
    assert server._scheduled_task_has_chat_result_only({"task_type": "start_inference"}) is True
    assert server._scheduled_task_has_chat_result_only({"task_type": "stop_inference"}) is True
    assert server._scheduled_task_has_chat_result_only({"task_type": "execute"}) is False


def test_log_routing_shadow_bands(monkeypatch):
    lines: List[str] = []

    class _Rec:
        def info(self, fmt, *args):
            lines.append(args[0] if args else fmt)

        def warning(self, *a, **k):
            raise AssertionError("shadow logging must not fail")

    monkeypatch.setattr(server, "logger", _Rec(), raising=False)
    gb = server.LARGE_DATASET_THRESHOLD_BYTES

    server._log_routing_shadow(analysis_max_inmemory_bytes=100, estimated_full_bytes=50, file_size_bytes=10)
    server._log_routing_shadow(analysis_max_inmemory_bytes=100, estimated_full_bytes=200, file_size_bytes=10)
    server._log_routing_shadow(analysis_max_inmemory_bytes=100, estimated_full_bytes=200, file_size_bytes=gb)

    bands = [json.loads(l)["band"] for l in lines]
    assert bands == ["below_cap", "cap_to_1gb", "above_1gb"]
    assert json.loads(lines[1])["would_defer"] is True


# ======================================================================================
# Health / middleware
# ======================================================================================


def test_health_reports_redis_down_when_ping_fails(client: TestClient, monkeypatch):
    async def _boom():
        raise RuntimeError("redis down")

    monkeypatch.setattr(session_service.cache, "ping", _boom)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["redis_connected"] is False


def test_health_reports_upstream_unreachable(client: TestClient, monkeypatch):
    async def _lg_down(method, path, **kw):
        raise HTTPException(502, "down")

    monkeypatch.setattr(server, "lg_request", _lg_down, raising=False)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["upstream_reachable"] is False


def test_private_network_preflight_allowed_origin(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "allow_origins", ["https://app.example"], raising=False)
    resp = client.options(
        "/datasets",
        headers={
            "Origin": "https://app.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert resp.status_code == 204
    assert resp.headers["Access-Control-Allow-Private-Network"] == "true"
    assert resp.headers["Access-Control-Allow-Origin"] == "https://app.example"
    assert resp.headers["Access-Control-Allow-Headers"] == "authorization"


def test_private_network_preflight_rejects_unknown_origin(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "allow_origins", ["https://app.example"], raising=False)
    resp = client.options(
        "/datasets",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert "Access-Control-Allow-Private-Network" not in resp.headers


# ======================================================================================
# Threads: upstream delete failure
# ======================================================================================


def test_delete_thread_upstream_500_propagates(client: TestClient, monkeypatch):
    upload = _do_upload(client, user_id="del-up")

    async def _lg_500(method, path, **kw):
        return DummyResponse(500, "upstream boom")

    monkeypatch.setattr(server, "lg_request", _lg_500, raising=False)
    resp = client.delete(f"/threads/{upload['thread_id']}", headers=make_auth_headers("del-up"))
    assert resp.status_code == 500


def test_delete_thread_upstream_404_is_tolerated(client: TestClient, monkeypatch):
    upload = _do_upload(client, user_id="del-404")

    async def _lg_404(method, path, **kw):
        return DummyResponse(404, "gone")

    monkeypatch.setattr(server, "lg_request", _lg_404, raising=False)
    resp = client.delete(f"/threads/{upload['thread_id']}", headers=make_auth_headers("del-404"))
    assert resp.status_code == 204


# ======================================================================================
# Tasks endpoints
# ======================================================================================


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/tasks/t1/info"),
        ("GET", "/tasks/t1/result/0"),
        ("GET", "/tasks/t1/runs"),
        ("GET", "/tasks/t1/status"),
        ("DELETE", "/tasks/t1"),
    ],
)
def test_task_endpoints_require_auth(client: TestClient, method, path):
    resp = client.request(method, path)
    assert resp.status_code == 401


@pytest.mark.parametrize("path", ["/tasks/t1/runs", "/tasks/t1/result/0", "/tasks/t1/status"])
def test_task_endpoints_404_when_task_not_in_session(client: TestClient, path):
    upload = _do_upload(client, user_id="task-scope")
    headers = make_auth_headers("task-scope")
    headers["X-Avaloka-Session"] = upload["session_id"]
    resp = client.get(path, headers=headers)
    assert resp.status_code == 404


def test_task_endpoint_rejects_foreign_session(client: TestClient):
    upload = _do_upload(client, user_id="task-owner")
    SESSIONS[upload["session_id"]]["tasks"] = ["t1"]
    headers = make_auth_headers("task-intruder")
    headers["X-Avaloka-Session"] = upload["session_id"]
    resp = client.get("/tasks/t1/runs", headers=headers)
    assert resp.status_code == 400


# ======================================================================================
# Assets endpoints
# ======================================================================================


def _seed_asset_session(sid, user_id, **extra):
    SESSIONS[sid] = {"user_id": user_id, "dataset_id": "ds-a", **extra}


def test_asset_code_404_without_key(client: TestClient):
    _seed_asset_session("sa1", "asset-user")
    resp = client.get("/api/assets/sa1/code", headers=make_auth_headers("asset-user"))
    assert resp.status_code == 404


def test_asset_code_returns_signed_url(client: TestClient, monkeypatch):
    _patch_signed_url(monkeypatch)
    _seed_asset_session("sa2", "asset-user", gcs_code_object_key="code/latest.py")
    resp = client.get("/api/assets/sa2/code", headers=make_auth_headers("asset-user"))
    assert resp.status_code == 200, resp.text
    assert resp.json()["signed_url"] == "https://signed.example/code/latest.py"


def test_asset_code_specific_version(client: TestClient, monkeypatch):
    _patch_signed_url(monkeypatch)
    _seed_asset_session(
        "sa3", "asset-user",
        gcs_code_object_key="code/v2.py",
        code_assets=[{"object_key": "code/v1.py", "prompt_ts": "20260101_000000"}],
    )
    headers = make_auth_headers("asset-user")
    ok = client.get("/api/assets/sa3/code", params={"prompt_ts": "20260101_000000"}, headers=headers)
    assert ok.status_code == 200
    assert ok.json()["object_key"] == "code/v1.py"

    missing = client.get("/api/assets/sa3/code", params={"prompt_ts": "19990101_000000"}, headers=headers)
    assert missing.status_code == 404


def test_asset_code_signed_url_failure_500(client: TestClient, monkeypatch):
    async def _none(*a, **k):
        return None

    monkeypatch.setattr(server, "_generate_asset_signed_url", _none, raising=False)
    _seed_asset_session("sa4", "asset-user", gcs_code_object_key="code/x.py")
    resp = client.get("/api/assets/sa4/code", headers=make_auth_headers("asset-user"))
    assert resp.status_code == 500


def test_asset_output_returns_signed_url(client: TestClient, monkeypatch):
    _patch_signed_url(monkeypatch)
    _seed_asset_session("sa5", "asset-user", gcs_output_object_key="out/latest.csv")
    resp = client.get("/api/assets/sa5/output", headers=make_auth_headers("asset-user"))
    assert resp.status_code == 200
    assert resp.json()["object_key"] == "out/latest.csv"


def test_asset_job_404_without_branch(client: TestClient):
    _seed_asset_session("sa6", "asset-user")
    resp = client.get("/api/assets/sa6/job", headers=make_auth_headers("asset-user"))
    assert resp.status_code == 404


def test_asset_job_returns_github_url(client: TestClient, monkeypatch):
    monkeypatch.setenv("GITHUB_JOB_REGISTRY_REPO", "org/jobs")
    _seed_asset_session("sa7", "asset-user", git_job_branch="main")
    resp = client.get("/api/assets/sa7/job", headers=make_auth_headers("asset-user"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["git_repo"] == "org/jobs"
    assert body["github_url"] == "https://github.com/org/jobs/tree/main/jobs/asset-user/sa7"


def test_assets_manifest_hidden_from_other_user(client: TestClient):
    _seed_asset_session("sa8", "asset-owner")
    resp = client.get("/api/assets/sa8", headers=make_auth_headers("asset-intruder"))
    assert resp.status_code == 404


# ======================================================================================
# Analysis (insights) endpoints
# ======================================================================================


def test_resolve_aid_supabase_credential_failure_is_503(client: TestClient, monkeypatch):
    def _bad_client():
        raise RuntimeError("Supabase env vars not set")

    monkeypatch.setattr(server, "get_supabase_client", _bad_client, raising=False)
    resp = client.get("/analysis/aid-1/code", headers=make_auth_headers("aid-user"))
    assert resp.status_code == 503
    assert "credentials were rejected" in resp.json()["detail"]


def test_resolve_aid_generic_failure_is_503(client: TestClient, monkeypatch):
    def _bad_client():
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(server, "get_supabase_client", _bad_client, raising=False)
    resp = client.get("/analysis/aid-1/versions", headers=make_auth_headers("aid-user"))
    assert resp.status_code == 503
    assert "temporarily unavailable" in resp.json()["detail"]


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("GET", "/analysis/aid-x/code", None),
        ("GET", "/analysis/aid-x/versions", None),
        ("POST", "/analysis/aid-x/restore", {"prompt_ts": "20260101_000000"}),
        ("POST", "/analysis/aid-x/refresh", None),
    ],
)
def test_analysis_endpoints_404_when_aid_unknown(client: TestClient, monkeypatch, method, path, body):
    _patch_aid(monkeypatch, None)
    resp = client.request(method, path, json=body, headers=make_auth_headers("aid-user"))
    assert resp.status_code == 404


def test_analysis_code_returns_inline_text(client: TestClient, monkeypatch):
    _patch_aid(monkeypatch, {"session_id": "s-code"})
    _patch_signed_url(monkeypatch)
    SESSIONS["s-code"] = {"user_id": "code-user", "gcs_code_object_key": "code/v1.py"}

    class _Resp:
        status_code = 200
        text = "print('from gcs')"

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            assert url == "https://signed.example/code/v1.py"
            return _Resp()

    monkeypatch.setattr(server.httpx, "AsyncClient", _Client, raising=False)

    resp = client.get("/analysis/aid-c/code", headers=make_auth_headers("code-user"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["code"] == "print('from gcs')"
    assert body["object_key"] == "code/v1.py"


def test_analysis_code_404_when_session_owned_by_other(client: TestClient, monkeypatch):
    _patch_aid(monkeypatch, {"session_id": "s-other"})
    SESSIONS["s-other"] = {"user_id": "someone-else", "gcs_code_object_key": "code/v1.py"}
    resp = client.get("/analysis/aid-o/code", headers=make_auth_headers("code-user"))
    assert resp.status_code == 404


def _seed_versioned_session(sid, user_id):
    SESSIONS[sid] = {
        "user_id": user_id,
        "code_assets": [
            {"object_key": "code/v1.py", "prompt_ts": "20260101_000000"},
            {"object_key": "code/v2.py", "prompt_ts": "20260102_000000"},
        ],
        "output_assets": [
            {"object_key": "out/v2.csv", "prompt_ts": "20260102_000000"},
        ],
        "gcs_code_object_key": "code/v2.py",
        "gcs_output_object_key": "out/v2.csv",
        "version_prompts": {"20260102_000000": "average of b"},
    }


def test_analysis_versions_lists_checkpoints(client: TestClient, monkeypatch):
    _patch_aid(monkeypatch, {"session_id": "s-ver"})
    _seed_versioned_session("s-ver", "ver-user")

    resp = client.get("/analysis/aid-v/versions", headers=make_auth_headers("ver-user"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["current_prompt_ts"] == "20260102_000000"
    latest = body["versions"][0]
    assert latest["prompt"] == "average of b"
    assert latest["created_at"].startswith("2026-01-02T00:00:00")
    assert latest["has_output"] is True
    assert body["versions"][1]["has_output"] is False


def test_analysis_restore_repoints_only_present_artifacts(client: TestClient, monkeypatch):
    _patch_aid(monkeypatch, {"session_id": "s-res"})
    _patch_signed_url(monkeypatch)

    async def _code_text(key, sess):
        return f"# code for {key}"

    monkeypatch.setattr(server, "_fetch_code_text", _code_text, raising=False)
    _seed_versioned_session("s-res", "res-user")

    resp = client.post(
        "/analysis/aid-r/restore",
        json={"prompt_ts": "20260101_000000"},
        headers=make_auth_headers("res-user"),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["code"] == "# code for code/v1.py"
    assert body["output_signed_url"] is None

    sess = SESSIONS["s-res"]
    assert sess["gcs_code_object_key"] == "code/v1.py"
    # v1 has no output -> current output pointer is left alone (non-destructive).
    assert sess["gcs_output_object_key"] == "out/v2.csv"
    assert sess["restored_from_prompt_ts"] == "20260101_000000"
    # History is intact.
    assert len(sess["code_assets"]) == 2


def test_analysis_restore_unknown_version_404(client: TestClient, monkeypatch):
    _patch_aid(monkeypatch, {"session_id": "s-res2"})
    _seed_versioned_session("s-res2", "res-user")
    resp = client.post(
        "/analysis/aid-r/restore",
        json={"prompt_ts": "19990101_000000"},
        headers=make_auth_headers("res-user"),
    )
    assert resp.status_code == 404


def test_analysis_refresh_404_without_code(client: TestClient, monkeypatch):
    _patch_aid(monkeypatch, {"session_id": "s-ref"})
    SESSIONS["s-ref"] = {"user_id": "ref-user"}
    resp = client.post("/analysis/aid-f/refresh", headers=make_auth_headers("ref-user"))
    assert resp.status_code == 404
    assert "No code to refresh" in resp.json()["detail"]


def test_analysis_refresh_500_when_code_unreadable(client: TestClient, monkeypatch):
    _patch_aid(monkeypatch, {"session_id": "s-ref2"})

    async def _none(key, sess):
        return None

    monkeypatch.setattr(server, "_fetch_code_text", _none, raising=False)
    SESSIONS["s-ref2"] = {"user_id": "ref-user", "gcs_code_object_key": "code/x.py"}
    resp = client.post("/analysis/aid-f/refresh", headers=make_auth_headers("ref-user"))
    assert resp.status_code == 500


def test_save_and_execute_rejects_empty_code(client: TestClient):
    resp = client.post(
        "/analysis/aid-1/save-and-execute",
        json={"code": "   "},
        headers=make_auth_headers("sae-user"),
    )
    assert resp.status_code == 422


def test_analysis_feedback_invalid_type_422(client: TestClient):
    resp = client.post(
        "/analysis/aid-1/feedback",
        json={"feedback_type": "meh"},
        headers=make_auth_headers("fb-user"),
    )
    assert resp.status_code == 422


# ======================================================================================
# MCP credentials endpoints
# ======================================================================================


def _patch_mcp_row(monkeypatch, row):
    monkeypatch.setattr(server, "_mcp_conn_select", lambda cid: row, raising=False)


def test_mcp_encrypt_unknown_connection_404(client: TestClient, monkeypatch):
    _patch_mcp_row(monkeypatch, None)
    resp = client.post("/api/mcp-connections/c1/encrypt", json={"apiKey": "k"},
                       headers=make_auth_headers("mcp-user"))
    assert resp.status_code == 404


def test_mcp_encrypt_foreign_connection_403(client: TestClient, monkeypatch):
    _patch_mcp_row(monkeypatch, {"id": "c1", "user_id": "other"})
    resp = client.post("/api/mcp-connections/c1/encrypt", json={"apiKey": "k"},
                       headers=make_auth_headers("mcp-user"))
    assert resp.status_code == 403


def test_mcp_encrypt_requires_a_field(client: TestClient, monkeypatch):
    _patch_mcp_row(monkeypatch, {"id": "c1", "user_id": "mcp-user"})
    resp = client.post("/api/mcp-connections/c1/encrypt", json={},
                       headers=make_auth_headers("mcp-user"))
    assert resp.status_code == 422


def test_mcp_encrypt_stores_ciphertext_and_blanks_plaintext(client: TestClient, monkeypatch):
    _patch_mcp_row(monkeypatch, {"id": "c1", "user_id": "mcp-user"})
    monkeypatch.setattr(server, "encrypt_secret",
                        lambda v: {"ciphertext": f"ct({len(v)})", "iv": "iv"}, raising=False)
    updates: List[Dict[str, Any]] = []
    monkeypatch.setattr(server, "_mcp_conn_update", lambda cid, patch: updates.append(patch),
                        raising=False)

    resp = client.post("/api/mcp-connections/c1/encrypt", json={"apiKey": "secret-key"},
                       headers=make_auth_headers("mcp-user"))
    assert resp.status_code == 200, resp.text
    assert resp.json()["encrypted"] == ["api_key"]
    assert updates == [{"api_key_ciphertext": "ct(10)", "api_key_iv": "iv", "api_key": ""}]
    assert "secret-key" not in json.dumps(updates)


def test_mcp_encrypt_without_key_configured_500(client: TestClient, monkeypatch):
    _patch_mcp_row(monkeypatch, {"id": "c1", "user_id": "mcp-user"})
    monkeypatch.setattr(server, "encrypt_secret", lambda v: None, raising=False)
    resp = client.post("/api/mcp-connections/c1/encrypt", json={"username": "bob"},
                       headers=make_auth_headers("mcp-user"))
    assert resp.status_code == 500
    assert "DB_ENCRYPTION_KEY" in resp.json()["detail"]


def test_mcp_decrypt_success(client: TestClient, monkeypatch):
    _patch_mcp_row(monkeypatch, {"id": "c1", "user_id": "mcp-user",
                                 "api_key_ciphertext": "CT", "api_key_iv": "IV"})
    monkeypatch.setattr(
        server, "decrypt_secret",
        lambda blob: f"plain:{blob['ciphertext']}" if blob.get("ciphertext") else None,
        raising=False,
    )
    resp = client.post("/api/mcp-connections/c1/decrypt", headers=make_auth_headers("mcp-user"))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"apiKey": "plain:CT", "username": None}


def test_mcp_decrypt_key_mismatch_409(client: TestClient, monkeypatch):
    _patch_mcp_row(monkeypatch, {"id": "c1", "user_id": "mcp-user",
                                 "api_key_ciphertext": "CT", "api_key_iv": "IV"})
    monkeypatch.setattr(server, "decrypt_secret", lambda blob: None, raising=False)
    resp = client.post("/api/mcp-connections/c1/decrypt", headers=make_auth_headers("mcp-user"))
    assert resp.status_code == 409


def test_mcp_lookup_failure_500(client: TestClient, monkeypatch):
    def _boom(cid):
        raise RuntimeError("supabase down")

    monkeypatch.setattr(server, "_mcp_conn_select", _boom, raising=False)
    resp = client.post("/api/mcp-connections/c1/decrypt", headers=make_auth_headers("mcp-user"))
    assert resp.status_code == 500


# ======================================================================================
# Database query tool routing + tables-to-analysis edge cases
# ======================================================================================


def _capture_mcp(monkeypatch, payload):
    sent: List[Dict[str, Any]] = []

    class _Resp:
        status_code = 200

        def json(self):
            return payload

        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json, headers, timeout):
            sent.append(json)
            return _Resp()

    monkeypatch.setattr(server.httpx, "AsyncClient", _Client, raising=False)
    return sent


@pytest.mark.parametrize(
    "content,tool,args",
    [
        ("show tables", "list_tables", {}),
        ("LIST TABLES", "list_tables", {}),
        ("describe table users", "describe_table", {"table_name": "users"}),
        ("SELECT 1", "query", {"sql": "SELECT 1"}),
    ],
)
def test_database_query_tool_routing(client: TestClient, monkeypatch, content, tool, args):
    sent = _capture_mcp(monkeypatch, {"ok": True})
    resp = client.post(
        "/api/v1/database/query",
        json={"content": content, "customer_id": "c1", "metadata": {"api_key": "k"}},
        headers=make_auth_headers("db-route"),
    )
    assert resp.status_code == 200, resp.text
    assert sent[0]["params"]["name"] == tool
    assert sent[0]["params"]["arguments"] == args


def test_database_query_no_rows_creates_no_dataset(client: TestClient, monkeypatch):
    _capture_mcp(monkeypatch, {"columns": ["id"], "rows": []})
    resp = client.post(
        "/api/v1/database/query",
        json={"content": "SELECT id FROM t WHERE 1=0", "customer_id": "c1",
              "metadata": {"api_key": "k"}},
        headers=make_auth_headers("db-empty"),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert not data.get("dataset_id")
    assert "no rows" in _last_assistant(data)
    assert SESSIONS == {}


def test_database_query_records_source_table(client: TestClient, monkeypatch):
    _capture_mcp(monkeypatch, {"columns": ["id"], "rows": [[1]]})
    resp = client.post(
        "/api/v1/database/query",
        json={"content": "SELECT id FROM sales.orders", "customer_id": "c1",
              "metadata": {"api_key": "k"}},
        headers=make_auth_headers("db-src"),
    )
    assert resp.status_code == 200, resp.text
    sess = SESSIONS[resp.json()["session_id"]]
    assert sess["source_table"] == "sales.orders"
    assert sess["customer_id"] == "c1"
    assert sess["input_data_type"] == "db"


def test_tables_to_analysis_duplicate_tables_get_unique_aliases(client: TestClient, monkeypatch):
    async def _mcp(customer_id, api_key, sql):
        return {"columns": ["id"], "rows": [[1], [2]]}

    monkeypatch.setattr(server, "_run_mcp_db_query", _mcp, raising=False)
    resp = client.post(
        "/api/database/tables-to-analysis",
        json={"tables": ["users", "users"], "customer_id": "c", "api_key": "k"},
        headers=make_auth_headers("tta-dup"),
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert [d["alias"] for d in out["datasets"]] == ["users", "users_2"]

    group = SESSIONS[out["session_id"]]
    assert group["dataset_ids"] == [d["dataset_id"] for d in out["datasets"]]


def test_tables_to_analysis_mcp_error_rolls_back(client: TestClient, monkeypatch):
    calls = {"n": 0}

    async def _mcp(customer_id, api_key, sql):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"columns": ["id"], "rows": [[1]]}
        return {"error": "permission denied"}

    monkeypatch.setattr(server, "_run_mcp_db_query", _mcp, raising=False)
    resp = client.post(
        "/api/database/tables-to-analysis",
        json={"tables": ["ok_table", "locked_table"], "customer_id": "c", "api_key": "k"},
        headers=make_auth_headers("tta-err"),
    )
    assert resp.status_code == 400
    assert "permission denied" in resp.json()["detail"]
    assert SESSIONS == {}, "first table's session must be rolled back"


def test_tables_to_analysis_empty_table_422(client: TestClient, monkeypatch):
    async def _mcp(customer_id, api_key, sql):
        return {"columns": ["id"], "rows": []}

    monkeypatch.setattr(server, "_run_mcp_db_query", _mcp, raising=False)
    resp = client.post(
        "/api/database/tables-to-analysis",
        json={"tables": ["empty"], "customer_id": "c", "api_key": "k"},
        headers=make_auth_headers("tta-empty"),
    )
    assert resp.status_code == 422


def test_tables_to_analysis_too_many_tables_413(client: TestClient, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_FILES", 1, raising=False)
    resp = client.post(
        "/api/database/tables-to-analysis",
        json={"tables": ["a", "b"], "customer_id": "c", "api_key": "k"},
        headers=make_auth_headers("tta-many"),
    )
    assert resp.status_code == 413


def test_tables_to_analysis_caps_limit(client: TestClient, monkeypatch):
    seen_sql: List[str] = []

    async def _mcp(customer_id, api_key, sql):
        seen_sql.append(sql)
        return {"columns": ["id"], "rows": [[1]]}

    monkeypatch.setattr(server, "_run_mcp_db_query", _mcp, raising=False)
    resp = client.post(
        "/api/database/tables-to-analysis",
        json={"tables": ["t"], "customer_id": "c", "api_key": "k", "limit": 10_000_000},
        headers=make_auth_headers("tta-limit"),
    )
    assert resp.status_code == 200, resp.text
    assert seen_sql == [f"SELECT * FROM t LIMIT {server.MAX_DB_SAMPLE_ROWS}"]