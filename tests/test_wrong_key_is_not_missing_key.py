"""A connection saved under a key the deployment no longer uses is reported truthfully.

When DB_ENCRYPTION_KEY changes (a published key replaced by provisioning, a
rotation), every connection sealed under the old key stops opening. Before this
fix the user saw it vanish: GET /customers left it out, /status and
/credentials said "Customer not found", and the log said "DB_ENCRYPTION_KEY is
not configured" while a key was configured. Now a decryption failure under a
configured key is its own error and log line; the connection is listed with
status needs_reregistering and a message saying what happened and what to do;
/status and /credentials answer 409 with the same; its owner can delete it or
register it again; nobody else can take its id.

End to end through the real onboarding router and the real MCP app (the probe
from port (a), ~/avaloka-work/x-enckey/probe/rotate_probe.py, adapted to 1.6).
Every key, token and password is generated per test; no assertion prints one.
"""
from __future__ import annotations

import json
import logging
import secrets
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import app.api.cloud_connections as cc

TOKEN_HEADER = "X-Avaloka-Service-Token"
USER_HEADER = "X-Avaloka-User"
PUBLISHED = "avaloka-local-dev-encryption-key"


def _use_key(monkeypatch, value):
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", value, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)


@pytest.fixture
def stack(tmp_path, monkeypatch, caplog):
    import app.mcp_server.customer_dbs as cdb
    import app.mcp_server.multi_tenant_mcp_server as srv

    caplog.set_level(logging.WARNING)
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("MCP_SERVICE_TOKEN", token)
    path = tmp_path / "customers.json"
    path.write_text("{}")
    monkeypatch.setattr(srv, "CUSTOMERS_FILE", str(path))
    monkeypatch.setattr(srv, "_registry_cache", {"stat": None, "data": {}}, raising=False)
    monkeypatch.setattr(cdb.customer_manager, "customers_file", str(path))
    monkeypatch.setattr(cdb.customer_manager, "customers", {})
    monkeypatch.setattr(srv, "CUSTOMER_API_KEYS", {})
    monkeypatch.setattr(srv.db_manager, "engines", {})
    monkeypatch.setattr(srv.db_manager, "connection_configs", {})

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            return mock.MagicMock()

    def _engine(*a, **k):
        engine = mock.MagicMock()
        engine.connect.return_value = _Conn()
        return engine

    monkeypatch.setattr(srv, "create_engine", _engine)
    monkeypatch.setattr(cdb, "create_engine", _engine)
    client = TestClient(srv.app)

    class _AsyncClient:  # the onboarding service's forward, answered by the real MCP app
        def __init__(self, *a, **k): ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None, **k):
            return client.post("/admin/customers", json=json, headers=headers)

        async def delete(self, url, headers=None, **k):
            return client.delete(url[url.index("/admin"):], headers=headers)

    monkeypatch.setattr(cdb.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(cdb.DatabaseConnectionManager, "test_database_connection",
                        staticmethod(lambda conn: {"success": True, "database": "shop", "table_count": 1}))

    def as_user(user):
        return {TOKEN_HEADER: token, USER_HEADER: user}

    def register(user, password):
        return client.post("/customers/register", headers=as_user(user), json={
            "customer_id": "onb", "customer_name": "Onboarded", "database_type": "postgresql",
            "host": "db.internal", "port": 5432, "database_name": "shop",
            "username": "alice", "password": password, "contact_email": "a@example.com"})

    def restart_with_key(value):
        """A key change is a restart: both services reload the registry under the new key."""
        _use_key(monkeypatch, value)
        srv.CUSTOMER_API_KEYS.clear()
        srv.db_manager.engines.clear()
        srv.db_manager.connection_configs.clear()
        srv._registry_cache.update({"stat": None, "data": {}})
        caplog.clear()
        cdb.customer_manager.load_customers()
        srv.load_registered_customers()

    return SimpleNamespace(srv=srv, cdb=cdb, client=client, as_user=as_user, register=register,
                           path=path, caplog=caplog, restart_with_key=restart_with_key)


def _secrets_in(text, *values):
    return [v for v in values if v and v in text]


def test_after_a_key_change_the_connection_is_listed_and_explained_not_lost(stack, monkeypatch):
    password = "PROBE-wrongkey-" + secrets.token_hex(8)
    _use_key(monkeypatch, PUBLISHED)
    r = stack.register("user-a", password)
    assert r.status_code == 200, r.status_code
    api_key = r.json()["api_key"]

    stack.restart_with_key("replaced-" + secrets.token_hex(24))  # what provisioning does
    listed = stack.client.get("/customers", headers=stack.as_user("user-a"))
    status = stack.client.get("/customers/onb/status", headers=stack.as_user("user-a"))
    creds = stack.client.get("/customers/onb/credentials", headers=stack.as_user("user-a"))

    rows = {c["customer_id"]: c for c in listed.json()["customers"]}
    assert "onb" in rows, "after the key changed, the saved connection vanished from the list"
    assert rows["onb"]["status"] == "needs_reregistering"
    assert "delete this connection and register it again" in rows["onb"]["message"]
    for name, resp in (("status", status), ("credentials", creds)):
        assert resp.status_code == 409, f"/{name} answered {resp.status_code}, not a true 409"
        detail = resp.json()["detail"]
        assert "not found" not in detail.lower(), f"/{name} says the connection was not found"
        assert "DB_ENCRYPTION_KEY" in detail and "delete this connection and register it again" in detail
    log = stack.caplog.text
    assert "not configured" not in log, "the log says no key is configured while one is"
    assert "cannot be opened" in log and "delete this connection and register it again" in log
    leaked = _secrets_in(listed.text + status.text + creds.text + log, password, api_key, PUBLISHED)
    assert not leaked, "a secret appears in a reply or the log"


def test_its_owner_can_delete_and_register_it_again_and_nobody_else_can_take_the_id(stack, monkeypatch):
    _use_key(monkeypatch, PUBLISHED)
    assert stack.register("user-a", "pw-" + secrets.token_hex(6)).status_code == 200
    stack.restart_with_key("replaced-" + secrets.token_hex(24))
    assert stack.register("user-b", "pw-" + secrets.token_hex(6)).status_code == 409, \
        "another user took the id of a connection that cannot be opened"
    resend = stack.register("user-a", "pw-" + secrets.token_hex(6))
    assert resend.status_code == 409 and "delete this connection and register it again" in resend.json()["detail"], \
        "the owner re-sending it is not told what to do"
    assert stack.client.delete("/customers/onb", headers=stack.as_user("user-a")).status_code == 200
    assert stack.register("user-b", "pw-" + secrets.token_hex(6)).status_code == 409, \
        "another user took the id once its owner deleted it"
    again = stack.register("user-a", "pw-" + secrets.token_hex(6))
    assert again.status_code == 200, again.status_code
    rows = {c["customer_id"]: c for c in
            stack.client.get("/customers", headers=stack.as_user("user-a")).json()["customers"]}
    assert rows["onb"]["status"] == "active", "the re-registered connection is not active"
    assert stack.client.get("/customers/onb/credentials",
                            headers=stack.as_user("user-a")).status_code == 200


def test_another_user_cannot_destroy_a_connection_this_key_cannot_open(stack, monkeypatch):
    """Data loss on 1.6 before this fix: after a key change, another user registering the
    same id overwrote the owner's record on disk -- gone for good, even if the old key
    came back. Its bytes must survive the attempt."""
    _use_key(monkeypatch, PUBLISHED)
    assert stack.register("user-a", "pw-" + secrets.token_hex(6)).status_code == 200
    before = json.dumps(json.loads(stack.path.read_text())["onb"], sort_keys=True)
    stack.restart_with_key("replaced-" + secrets.token_hex(24))
    attempt = stack.register("user-b", "pw-" + secrets.token_hex(6))
    after = json.dumps(json.loads(stack.path.read_text()).get("onb"), sort_keys=True)
    assert after == before, "another user's registration overwrote a connection this key cannot open"
    assert attempt.status_code == 409


def test_its_owner_can_delete_it(stack, monkeypatch):
    _use_key(monkeypatch, PUBLISHED)
    assert stack.register("user-a", "pw-" + secrets.token_hex(6)).status_code == 200
    stack.restart_with_key("replaced-" + secrets.token_hex(24))
    r = stack.client.delete("/customers/onb", headers=stack.as_user("user-a"))
    assert r.status_code == 200, r.status_code
    rows = [c["customer_id"] for c in
            stack.client.get("/customers", headers=stack.as_user("user-a")).json()["customers"]]
    assert "onb" not in rows, "a deleted connection is still listed"


def test_with_no_key_at_all_the_message_still_says_so(stack, monkeypatch):
    """The other fault keeps its own, true message."""
    _use_key(monkeypatch, "first-" + secrets.token_hex(24))
    assert stack.register("user-a", "pw-" + secrets.token_hex(6)).status_code == 200
    stack.restart_with_key(None)
    assert "not configured" in stack.caplog.text, "a missing key is no longer reported as missing"
    assert "different key" not in stack.caplog.text
