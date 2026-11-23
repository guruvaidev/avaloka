"""Saved database connections through the API (/api/connections), in every auth mode.

What has to hold (app/api/db_connections.py):

* in local auth mode, where there is no Supabase database, connecting a
  database from the UI saves the connection and lists it -- the step the
  quickstart README documents, which used to end at "Saving connection failed";
* one user can neither list nor delete nor rename another user's connections;
* register-then-save is atomic: a failed save deactivates the registration
  through the signed user path, and a retry then succeeds instead of
  answering "already registered";
* nothing a client can read carries a secret.

The API is the real server app. The onboarding API is the real onboarding
router on a fresh registry (the fixture from test_onboarding_owner_scope:
``create_engine`` stubbed, the MCP admin forward answered in process), called
with the real signed user assertion.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from unittest import mock

import jwt
import pytest
from fastapi.testclient import TestClient

from app.core import local_auth

SECRET = "s" * 48
LOCAL_PASSWORD = "correct-horse-battery"
JWT_SECRET = "connections-test-jwt-secret-" + "x" * 24

DB_PASSWORD = "Sample-Db-Pw-" + "q7"
DB_USER = "avaloka_reader"
REG = {
    "connection_name": "Sample orders",
    "customer_id": "sample_db", "customer_name": "Sample", "contact_email": "me@example.com",
    "environment": "development", "database_type": "postgresql",
    "host": "postgres", "port": 5432, "database_name": "sample",
    "username": DB_USER, "password": DB_PASSWORD, "query_timeout": 30, "max_rows": 1000,
}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def onboarding(tmp_path, monkeypatch):
    """The onboarding router on a fresh registry; returns (cdb module, client)."""
    import app.api.cloud_connections as cc
    import app.mcp_server.customer_dbs as cdb
    import app.mcp_server.multi_tenant_mcp_server as srv

    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", "unit-test-key", raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"unit-test-key").digest(), raising=False)

    path = tmp_path / "customers.json"
    monkeypatch.setenv("CUSTOMERS_FILE", str(path))
    monkeypatch.setattr(srv, "CUSTOMERS_FILE", str(path))
    monkeypatch.setattr(cdb.customer_manager, "customers_file", str(path))
    cdb.customer_manager.customers = {}

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            return mock.MagicMock()

    engine = mock.MagicMock()
    engine.connect.return_value = _Conn()
    monkeypatch.setattr(cdb, "create_engine", lambda *a, **k: engine)

    class _Resp:
        status_code = 200
        text = "ok"

    class _AsyncClient:
        def __init__(self, *a, **k): ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _Resp()

    # Only the onboarding module's forward to the MCP admin route is faked.
    monkeypatch.setattr(cdb, "httpx", mock.MagicMock(AsyncClient=_AsyncClient))
    monkeypatch.setattr(cdb.customer_manager.db_manager, "test_database_connection",
                        lambda reg: {"success": True, "database": "sample", "table_count": 1})
    return cdb, TestClient(srv.app)


@pytest.fixture
def api(tmp_path, monkeypatch, onboarding):
    """The real API app, its connections in a temp file, its onboarding calls in process."""
    import app.api.db_connections as dbc
    import app.api.server as server
    from app.mcp_server.client import user_headers

    _cdb, onboarding_client = onboarding
    for k in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "AVALOKA_AUTH_MODE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(dbc.FILE_ENV, str(tmp_path / "db_connections.json"))
    monkeypatch.setattr(server, "JWT_SECRET", JWT_SECRET, raising=False)

    calls = []

    async def _onboarding(method, path, user_id, body=None):
        headers = user_headers(user_id)
        assert headers, "no signed user assertion: the deployment key is missing"
        if body is not None:
            headers["Content-Type"] = "application/json"
        calls.append((method, path, user_id))
        resp = await asyncio.to_thread(onboarding_client.request, method, path,
                                       content=body, headers=headers)
        payload = resp.json() if resp.content else {}
        return resp.status_code, payload if isinstance(payload, dict) else {}

    monkeypatch.setattr(dbc, "_onboarding", _onboarding)
    client = TestClient(server.app)
    client.onboarding_calls = calls
    return client


@pytest.fixture
def local_mode(monkeypatch):
    for k, v in {
        "AVALOKA_AUTH_MODE": "local", "AVALOKA_DEPLOYMENT": "self_hosted",
        "AVALOKA_EDITION": "oss", "AVALOKA_LOCAL_AUTH_SECRET": SECRET,
        "AVALOKA_LOCAL_AUTH_PASSWORD": LOCAL_PASSWORD,
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AVALOKA_LOCAL_AUTH_EMAIL", raising=False)
    local_auth._failures.clear()
    yield
    local_auth._failures.clear()


def _as(user_id):
    token = jwt.encode({"sub": user_id, "iat": int(time.time())}, JWT_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def _registry(onboarding):
    cdb, _client = onboarding
    return cdb.customer_manager._read_customers_file()


# --------------------------------------------------------------------------
# (a) local auth mode: the README's "Connect the sample database"
# --------------------------------------------------------------------------

def test_local_mode_connect_saves_lists_and_opens(api, local_mode, onboarding):
    import app.api.db_connections as dbc

    assert not dbc.uses_supabase()
    signin = api.post("/local-auth/auth/v1/token?grant_type=password",
                      json={"email": local_auth.DEFAULT_EMAIL, "password": LOCAL_PASSWORD})
    assert signin.status_code == 200, signin.text
    h = {"Authorization": f"Bearer {signin.json()['access_token']}"}

    # What the modal used to do, and why it could not work here.
    old = api.post("/local-auth/rest/v1/mcp_connections", headers=h, json={})
    assert old.status_code == 501

    created = api.post("/api/connections", headers=h, json=REG)
    assert created.status_code == 201, created.text
    conn = created.json()
    assert conn["connection_name"] == "Sample orders" and conn["customer_id"] == "sample_db"

    listed = api.get("/api/connections", headers=h)
    assert listed.status_code == 200, listed.text
    assert [c["id"] for c in listed.json()["connections"]] == [conn["id"]]

    # The query panel's next call (decrypt the api_key for list_tables) works
    # against the same store, and hands back the key the registry holds.
    dec = api.post(f"/api/mcp-connections/{conn['id']}/decrypt", headers=h)
    assert dec.status_code == 200, dec.text
    record = _registry(onboarding)["sample_db"]
    assert dec.json()["apiKey"] == record["api_key"]
    assert dec.json()["username"] == DB_USER

    # And the saved-connection resolver (/api/database/connect by connection_id).
    from app.api.legacy_database_connections import _resolve_sync
    resolved = _resolve_sync(conn["id"], local_auth.load().user_id)
    assert resolved.customer_id == "sample_db" and resolved.api_key == record["api_key"]


# --------------------------------------------------------------------------
# (b) ownership
# --------------------------------------------------------------------------

def test_other_user_cannot_list_rename_or_delete(api, onboarding):
    created = api.post("/api/connections", headers=_as("user-a"), json=REG)
    assert created.status_code == 201, created.text
    cid = created.json()["id"]

    assert api.get("/api/connections", headers=_as("user-b")).json() == {"connections": []}
    assert api.delete(f"/api/connections/{cid}", headers=_as("user-b")).status_code == 404
    assert api.put(f"/api/connections/{cid}", headers=_as("user-b"),
                   json={"connection_name": "mine now"}).status_code == 404
    # The crypto routes follow the same rule: 404, the same as a missing id.
    foreign = api.post(f"/api/mcp-connections/{cid}/decrypt", headers=_as("user-b"))
    assert foreign.status_code == 404
    assert foreign.json() == {"detail": "Connection not found"}

    # Untouched: A still has it, and the registration is still active.
    rows = api.get("/api/connections", headers=_as("user-a")).json()["connections"]
    assert [(r["id"], r["connection_name"]) for r in rows] == [(cid, "Sample orders")]
    assert _registry(onboarding)["sample_db"]["status"] == "active"
    assert not any(m == "DELETE" for m, _p, _u in api.onboarding_calls)

    assert api.get("/api/connections").status_code == 401


def test_owner_delete_deactivates_the_registration(api, onboarding):
    cid = api.post("/api/connections", headers=_as("user-a"), json=REG).json()["id"]
    assert api.delete(f"/api/connections/{cid}", headers=_as("user-a")).status_code == 200
    assert api.get("/api/connections", headers=_as("user-a")).json() == {"connections": []}
    assert _registry(onboarding)["sample_db"]["status"] == "inactive"
    # ...so connecting it again is not "already registered".
    assert api.post("/api/connections", headers=_as("user-a"), json=REG).status_code == 201


# --------------------------------------------------------------------------
# (c) a failed save leaves no orphan, and a retry succeeds
# --------------------------------------------------------------------------

def test_save_failure_rolls_back_registration_and_retry_succeeds(api, onboarding, monkeypatch):
    import app.api.db_connections as dbc

    real_insert = dbc.FileConnectionStore.insert
    failures = {"left": 1}

    def flaky_insert(self, row):
        if failures["left"]:
            failures["left"] -= 1
            raise OSError("disk full")
        return real_insert(self, row)

    monkeypatch.setattr(dbc.FileConnectionStore, "insert", flaky_insert)

    first = api.post("/api/connections", headers=_as("user-a"), json=REG)
    assert first.status_code == 500
    assert first.json()["detail"].startswith("Saving connection failed")
    assert "rolled back" in first.json()["detail"]
    assert api.get("/api/connections", headers=_as("user-a")).json() == {"connections": []}
    assert _registry(onboarding)["sample_db"]["status"] == "inactive"
    assert ("DELETE", "/customers/sample_db", "user-a") in api.onboarding_calls

    retry = api.post("/api/connections", headers=_as("user-a"), json=REG)
    assert retry.status_code == 201, retry.text
    rows = api.get("/api/connections", headers=_as("user-a")).json()["connections"]
    assert [r["id"] for r in rows] == [retry.json()["id"]]
    record = _registry(onboarding)["sample_db"]
    assert record["status"] == "active" and record["owner_user_id"] == "user-a"


def test_rolled_back_id_is_not_open_to_another_user(api, onboarding, monkeypatch):
    import app.api.db_connections as dbc

    def broken_insert(self, row):
        raise OSError("disk full")

    monkeypatch.setattr(dbc.FileConnectionStore, "insert", broken_insert)
    assert api.post("/api/connections", headers=_as("user-a"), json=REG).status_code == 500
    # The inactive record is still A's: B taking the id over is "already registered".
    cdb, client = onboarding
    from app.mcp_server.client import user_headers
    body = {k: v for k, v in REG.items() if k != "connection_name"}
    assert client.post("/customers/register", json=body, headers=user_headers("user-b")).status_code == 409
    assert client.post("/customers/register", json=body, headers=user_headers("user-a")).status_code == 200


def test_rollback_revokes_the_key_on_the_mcp_server(api, onboarding, monkeypatch):
    """The rollback's DELETE must evict: the key it minted never answers /call_tool."""
    import app.api.db_connections as dbc
    import app.mcp_server.customer_dbs as cdb
    import app.mcp_server.multi_tenant_mcp_server as srv

    _cdb, mcp_client = onboarding
    srv.db_manager.engines.clear()
    srv.db_manager.connection_configs.clear()
    srv.CUSTOMER_API_KEYS.clear()
    srv._invalidate_registry_cache()

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            return mock.MagicMock()

    engines = []

    def _engine(*a, **k):
        engine = mock.MagicMock()
        engine.connect.return_value = _Conn()
        engines.append(engine)
        return engine

    monkeypatch.setattr(srv, "create_engine", _engine)
    monkeypatch.setattr(srv, "list_tables", lambda: {"tables": []})
    minted = []
    base = cdb.customer_manager.mcp_server_url

    class _AsyncClient:
        """The onboarding forward, answered by the real MCP admin routes, signature checked."""

        def __init__(self, *a, **k): ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, content=None, headers=None, **k):
            # The api_key travels sealed (connection_string_enc / api_key_enc).
            from app.mcp_server.credentials import open_sealed_secret
            minted.append(open_sealed_secret(json.loads(content)["api_key_enc"]))
            return await asyncio.to_thread(mcp_client.post, url[len(base):], content=content, headers=headers)

        async def delete(self, url, headers=None, **k):
            return await asyncio.to_thread(mcp_client.delete, url[len(base):], headers=headers)

    monkeypatch.setattr(cdb, "httpx", mock.MagicMock(AsyncClient=_AsyncClient))

    def tool(key):
        return mcp_client.post("/call_tool", headers={"Authorization": f"Bearer {key}"},
                               json={"method": "call_tool", "params": {"name": "list_tables", "arguments": {}}})

    real_insert = dbc.FileConnectionStore.insert
    failures = {"left": 1}

    def flaky_insert(self, row):
        if failures["left"]:
            failures["left"] -= 1
            raise OSError("disk full")
        return real_insert(self, row)

    monkeypatch.setattr(dbc.FileConnectionStore, "insert", flaky_insert)

    first = api.post("/api/connections", headers=_as("user-a"), json=REG)
    assert first.status_code == 500 and "rolled back" in first.json()["detail"]
    # Two engines: the pre-registration test (disposed at once), then the pooled one.
    assert len(minted) == 1 and len(engines) == 2
    # Evicted by the rollback itself, before any /call_tool could notice.
    assert "sample_db" not in srv.CUSTOMER_API_KEYS and "sample_db" not in srv.db_manager.engines
    engines[1].dispose.assert_called_once()
    assert tool(minted[0]).status_code == 401

    retry = api.post("/api/connections", headers=_as("user-a"), json=REG)
    assert retry.status_code == 201, retry.text
    assert len(minted) == 2
    assert tool(minted[1]).status_code == 200
    assert tool(minted[0]).status_code == 401

    # And the owner's DELETE /api/connections/{id} revokes the new key the same way.
    assert api.delete(f"/api/connections/{retry.json()['id']}", headers=_as("user-a")).status_code == 200
    assert "sample_db" not in srv.CUSTOMER_API_KEYS
    assert tool(minted[1]).status_code == 401


# --------------------------------------------------------------------------
# (d) no secrets out
# --------------------------------------------------------------------------

def test_responses_and_store_carry_no_secrets(api, onboarding, tmp_path):
    import app.api.db_connections as dbc

    created = api.post("/api/connections", headers=_as("user-a"), json=REG)
    assert created.status_code == 201, created.text
    listed = api.get("/api/connections", headers=_as("user-a"))
    api_key = _registry(onboarding)["sample_db"]["api_key"]
    assert api_key

    for resp in (created, listed):
        text = resp.text
        for secret in (DB_PASSWORD, api_key, DB_USER, "me@example.com"):
            assert secret not in text
        for field in ("password", "api_key", "username", "ciphertext", "_iv", "connection_string", "user_id"):
            assert field not in text, field
    assert set(created.json()) == set(dbc.PUBLIC_FIELDS)
    assert all(set(r) == set(dbc.PUBLIC_FIELDS) for r in listed.json()["connections"])

    at_rest = dbc.connections_file().read_text()
    for secret in (DB_PASSWORD, api_key, DB_USER):
        assert secret not in at_rest
    assert json.loads(at_rest)["connections"]
