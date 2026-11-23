"""The onboarding service hands a registration to the MCP server sealed, never in plaintext.

2.x signed the /admin/customers body (#599: a MAC over method, path, time and
body) but still sent the connection string, password included, and the api_key
in the clear inside it. This is the 1.6 port (hotfix/1.6-chat-connection-sealing
88120343): the body carries ``connection_string_enc`` and ``api_key_enc``,
sealed under DB_ENCRYPTION_KEY, and the MCP server opens them with its own copy
of the key. The MAC stays. A plaintext body no longer validates, signed or not.

Every value is generated per test; none is real. Each test fails on c3b27651.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from unittest import mock
from urllib.parse import quote_plus

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    import app.api.cloud_connections as cc
    import app.mcp_server.customer_dbs as cdb
    import app.mcp_server.multi_tenant_mcp_server as srv

    key = "unit-" + secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)
    path = tmp_path / "customers.json"
    monkeypatch.setenv("CUSTOMERS_FILE", str(path))
    monkeypatch.setattr(srv, "CUSTOMERS_FILE", str(path))
    monkeypatch.setattr(cdb.customer_manager, "customers_file", str(path))
    monkeypatch.setattr(cdb.customer_manager, "customers", {})
    monkeypatch.setattr(srv.db_manager, "engines", {})
    monkeypatch.setattr(srv.db_manager, "connection_configs", {})
    monkeypatch.setattr(srv, "CUSTOMER_API_KEYS", {})

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            return mock.MagicMock()

    dialed = []

    def _engine(url, *a, **k):
        dialed.append(str(url))
        engine = mock.MagicMock()
        engine.connect.return_value = _Conn()
        return engine

    monkeypatch.setattr(srv, "create_engine", _engine)
    monkeypatch.setattr(cdb, "create_engine", _engine)
    client = TestClient(srv.app)
    bodies = []

    class _AsyncClient:
        def __init__(self, *a, **k): ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, content=None, headers=None, json=None, **k):
            assert url.endswith("/admin/customers")
            bodies.append((content or b"").decode("utf-8", "replace"))
            return client.post("/admin/customers", content=content, headers=headers)

    monkeypatch.setattr(cdb.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(cdb.customer_manager.db_manager, "test_database_connection",
                        lambda reg: {"success": True, "database": "shop", "table_count": 1})
    return srv, client, dialed, bodies


def test_the_registration_reaches_mcp_with_no_password_in_the_body(mcp):
    srv, client, dialed, bodies = mcp
    from app.mcp_server.client import user_headers
    password = "PROBE-handoff-" + secrets.token_hex(8)
    resp = client.post("/customers/register", headers=user_headers("user-a"), json={
        "customer_id": "onb", "customer_name": "Onboarded", "database_type": "postgresql",
        "host": "db.internal", "port": 5432, "database_name": "shop",
        "username": "alice", "password": password, "contact_email": "a@example.com",
    })
    assert resp.status_code == 200, resp.status_code
    api_key = resp.json()["api_key"]
    assert bodies, "the onboarding service never called the MCP server"
    leaked = [b for b in bodies if password in b or quote_plus(password) in b]
    assert not leaked, "the password travels to the MCP server in plaintext"
    assert not any(api_key in b for b in bodies), "the api_key travels to the MCP server in plaintext"
    sent = json.loads(bodies[-1])
    assert "connection_string_enc" in sent and "api_key_enc" in sent, "the body carries no sealed handles"
    # The MCP server opened them and registered the customer for real.
    assert srv.extract_customer_from_api_key(api_key) == "onb"
    assert any(password in d or quote_plus(password) in d for d in dialed), "MCP did not open the sealed DSN"


def test_a_signed_plaintext_body_is_refused(mcp):
    srv, client, dialed, _ = mcp
    from app.mcp_server.credentials import service_headers
    dsn = "postgresql://alice:" + quote_plus("pw-" + secrets.token_hex(6)) + "@db.internal:5432/shop"
    body = json.dumps({
        "customer_id": "plain", "connection_string": dsn, "database_type": "postgresql",
        "api_key": "ak_" + secrets.token_hex(16), "created_at": "2026-11-01T00:00:00",
        "status": "active", "owner_user_id": "user-a",
    }).encode("utf-8")
    resp = client.post("/admin/customers", content=body,
                       headers={"Content-Type": "application/json",
                                **service_headers("POST", "/admin/customers", body)})
    assert resp.status_code == 422, f"a signed plaintext body was accepted ({resp.status_code})"
    assert dialed == [], "the MCP server dialled a DSN sent in the clear"


def test_a_handle_sealed_under_another_key_is_refused(mcp, monkeypatch):
    srv, client, dialed, _ = mcp
    import app.api.cloud_connections as cc
    from app.mcp_server.credentials import credential_handle, service_headers
    dsn = "postgresql://alice:" + quote_plus("pw-" + secrets.token_hex(6)) + "@db.internal:5432/shop"
    payload = {"customer_id": "foreign", "database_type": "postgresql", "created_at": "2026-11-01T00:00:00",
               "status": "active", "owner_user_id": "user-a"}
    real_key, real_cache = cc.DB_ENCRYPTION_KEY, cc._AESGCM_KEY
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", "other-" + secrets.token_hex(8))
    monkeypatch.setattr(cc, "_AESGCM_KEY", None)
    payload["connection_string_enc"] = credential_handle(dsn)
    payload["api_key_enc"] = credential_handle("ak_" + secrets.token_hex(16))
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", real_key)
    monkeypatch.setattr(cc, "_AESGCM_KEY", real_cache)
    body = json.dumps(payload).encode("utf-8")
    resp = client.post("/admin/customers", content=body,
                       headers={"Content-Type": "application/json",
                                **service_headers("POST", "/admin/customers", body)})
    assert resp.status_code == 400, f"a handle this key cannot open was not refused as unopenable ({resp.status_code})"
    assert dialed == []
