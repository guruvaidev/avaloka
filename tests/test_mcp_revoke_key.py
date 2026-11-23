"""A deactivated or deleted customer's api_key stops working on the MCP server at once.

DELETE /customers/{id} only marked the record inactive in the registry. The
MCP server kept the api_key, the config and the pooled engine in memory until
the id was registered again, so a revoked key kept answering /call_tool.

Two layers now (app/mcp_server/multi_tenant_mcp_server.py):

* eviction: the onboarding service calls the signed DELETE /admin/customers/{id},
  which drops the key and config and disposes of the engine;
* /call_tool asks the registry, on every call, whether the customer is active
  with this key -- so a missed eviction cannot keep a key alive.

No database and no network: ``create_engine`` is a recorder, and the
onboarding service's calls to the MCP admin routes reach the real app, signed
and verified for real.
"""

from __future__ import annotations

import hashlib
import json
from unittest import mock

import pytest
from fastapi.testclient import TestClient

PASSWORD = "Rev0ke-Pw-" + "k4"
REG = {
    "customer_id": "acme", "customer_name": "Acme", "database_type": "postgresql",
    "host": "db.internal", "port": 5432, "database_name": "shop",
    "username": "alice", "password": PASSWORD, "contact_email": "a@example.com",
}
TOOL = {"method": "call_tool", "params": {"name": "list_tables", "arguments": {}}}


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    """The MCP server app (onboarding router mounted) with its admin calls routed in process."""
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

    def _engine(url, *a, **k):
        engine = mock.MagicMock(name=f"engine{len(engines)}")
        engine.connect.return_value = _Conn()
        engines.append(engine)
        return engine

    monkeypatch.setattr(srv, "create_engine", _engine)
    monkeypatch.setattr(cdb, "create_engine", _engine)
    monkeypatch.setattr(cdb.customer_manager.db_manager, "test_database_connection",
                        lambda reg: {"success": True, "database": "shop", "table_count": 1})
    # The tool itself is not under test; who may reach it is.
    monkeypatch.setattr(srv, "list_tables", lambda: {"tables": [], "customer": srv.current_customer_id.get()})

    client = TestClient(srv.app)

    class _AsyncClient:
        """The onboarding service's httpx, answered by the real MCP app."""

        def __init__(self, *a, **k): ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, content=None, headers=None, **k):
            return client.post(url[len(cdb.customer_manager.mcp_server_url):], content=content, headers=headers)

        async def delete(self, url, headers=None, **k):
            return client.delete(url[len(cdb.customer_manager.mcp_server_url):], headers=headers)

    monkeypatch.setattr(cdb.httpx, "AsyncClient", _AsyncClient)
    return srv, cdb, client, engines


def _as(user_id):
    from app.mcp_server.client import user_headers
    headers = user_headers(user_id)
    assert headers, "no assertion: the deployment key is missing in the fixture"
    return headers


def _key(api_key):
    return {"Authorization": f"Bearer {api_key}"}


def _register(client, user="user-a"):
    resp = client.post("/customers/register", json=REG, headers=_as(user))
    assert resp.status_code == 200, resp.text
    return resp.json()["api_key"]


def _tool(client, api_key):
    return client.post("/call_tool", json=TOOL, headers=_key(api_key))


def _held(srv, cid="acme"):
    return (cid in srv.CUSTOMER_API_KEYS, cid in srv.db_manager.engines,
            cid in srv.db_manager.connection_configs)


# ---------------------------------------------------------------------------
# (a) DELETE evicts at once
# ---------------------------------------------------------------------------

def test_delete_evicts_key_and_engine_then_call_tool_is_401(mcp):
    srv, _, client, engines = mcp
    api_key = _register(client)
    ok = _tool(client, api_key)
    assert ok.status_code == 200 and ok.json()["customer"] == "acme", ok.text
    engine = srv.db_manager.engines["acme"]

    resp = client.delete("/customers/acme", headers=_as("user-a"))
    assert resp.status_code == 200 and resp.json()["mcp_evicted"] is True, resp.text

    # Before any /call_tool: the eviction itself, not the per-call check, did this.
    assert _held(srv) == (False, False, False)
    engine.dispose.assert_called_once()
    assert _tool(client, api_key).status_code == 401


def test_a_deactivated_key_is_refused_by_the_onboarding_routes_too(mcp):
    _, _, client, _ = mcp
    api_key = _register(client)
    assert client.get("/customers/acme/status", headers=_key(api_key)).status_code == 200
    assert client.delete("/customers/acme", headers=_as("user-a")).status_code == 200
    for method, route in (("GET", "/customers/acme/status"), ("GET", "/customers/acme/credentials"),
                          ("GET", "/customers")):
        assert client.request(method, route, headers=_key(api_key)).status_code == 401, route


# ---------------------------------------------------------------------------
# (c) the eviction route takes only the onboarding service's signed request
# ---------------------------------------------------------------------------

def _signed(method, path, body=b""):
    from app.mcp_server.credentials import service_headers
    return service_headers(method, path, body)


@pytest.mark.parametrize("headers", [
    lambda key: {},
    lambda key: _key(key),
    lambda key: _as("user-a"),
    lambda key: {"Authorization": "Bearer avs1.1700000000.deadbeef"},
    lambda key: _signed("DELETE", "/admin/customers/other"),
    lambda key: _signed("POST", "/admin/customers/acme"),
    lambda key: _signed("DELETE", "/admin/customers/acme", b"x"),
], ids=["none", "api-key", "user-assertion", "garbage-mac", "other-id", "other-method", "other-body"])
def test_unsigned_or_forged_evict_is_401_and_evicts_nothing(mcp, headers):
    srv, _, client, _ = mcp
    api_key = _register(client)
    resp = client.delete("/admin/customers/acme", headers=headers(api_key))
    assert resp.status_code == 401
    assert _held(srv) == (True, True, True)
    assert _tool(client, api_key).status_code == 200


def test_evict_signed_under_another_key_or_with_no_key_is_401(mcp, monkeypatch):
    srv, _, client, _ = mcp
    import app.api.cloud_connections as cc
    _register(client)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"elsewhere").digest(), raising=False)
    foreign = _signed("DELETE", "/admin/customers/acme")
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"unit-test-key").digest(), raising=False)
    assert client.delete("/admin/customers/acme", headers=foreign).status_code == 401
    good = _signed("DELETE", "/admin/customers/acme")
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", None, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    assert client.delete("/admin/customers/acme", headers=good).status_code == 401
    assert _held(srv) == (True, True, True)


def test_signed_evict_is_idempotent(mcp):
    srv, _, client, _ = mcp
    _register(client)
    for _ in range(2):
        resp = client.delete("/admin/customers/acme", headers=_signed("DELETE", "/admin/customers/acme"))
        assert resp.status_code == 200
    assert _held(srv) == (False, False, False)


# ---------------------------------------------------------------------------
# (d) re-registering issues a new key; the old one stays dead
# ---------------------------------------------------------------------------

def test_reregistering_works_and_the_old_key_stays_dead(mcp):
    srv, _, client, engines = mcp
    old_key = _register(client)
    assert _tool(client, old_key).status_code == 200
    assert client.delete("/customers/acme", headers=_as("user-a")).status_code == 200
    assert _tool(client, old_key).status_code == 401

    new_key = _register(client)
    assert new_key != old_key
    assert _tool(client, new_key).status_code == 200
    assert _tool(client, old_key).status_code == 401
    # Another owner may not take the inactive id over.
    assert client.post("/customers/register", json=REG, headers=_as("user-b")).status_code == 409


def _seal_admin_body(payload):
    """The admin body as the onboarding service sends it: connection_string and
    api_key as sealed handles (connection_string_enc, api_key_enc). Without a key
    nothing can be sealed and the body is left as is (the signed route refuses it)."""
    from app.mcp_server.credentials import CredentialKeyMissing, credential_handle
    out = dict(payload)
    try:
        for field in ("connection_string", "api_key"):
            if field in out:
                value = out.pop(field)
                # One handle per value, so a body built twice (once to send, once
                # to sign) is the same bytes.
                import app.api.cloud_connections as _cc
                cache_key = (_cc.DB_ENCRYPTION_KEY, value)
                if cache_key not in _SEALED_CACHE:
                    _SEALED_CACHE[cache_key] = credential_handle(value)
                out[field + "_enc"] = _SEALED_CACHE[cache_key]
    except CredentialKeyMissing:
        return dict(payload)
    return out


_SEALED_CACHE = {}


def test_an_update_disposes_of_the_replaced_engine(mcp):
    srv, _, client, _ = mcp
    _register(client)
    first = srv.db_manager.engines["acme"]
    config = srv.db_manager.get_config("acme")
    body = json.dumps(_seal_admin_body({**config.model_dump(), "owner_user_id": "user-a"}
                      if hasattr(config, "model_dump") else {**config.dict(), "owner_user_id": "user-a"})).encode()
    resp = client.post("/admin/customers", content=body,
                       headers={"Content-Type": "application/json", **_signed("POST", "/admin/customers", body)})
    assert resp.status_code == 200, resp.text
    assert srv.db_manager.engines["acme"] is not first
    first.dispose.assert_called_once()


# ---------------------------------------------------------------------------
# (e) the registry check alone, with eviction skipped
# ---------------------------------------------------------------------------

def test_registry_check_alone_blocks_a_key_whose_eviction_was_missed(mcp, monkeypatch):
    srv, cdb, client, _ = mcp
    api_key = _register(client)
    assert _tool(client, api_key).status_code == 200

    async def _lost(customer_id):
        return False  # the eviction call never reached the MCP server
    monkeypatch.setattr(cdb.customer_manager, "evict_from_mcp", _lost)

    resp = client.delete("/customers/acme", headers=_as("user-a"))
    assert resp.status_code == 200 and resp.json()["mcp_evicted"] is False
    assert _held(srv) == (True, True, True), "eviction ran; this test must exercise the check alone"

    assert _tool(client, api_key).status_code == 401
    # Refused, and the stale customer is dropped on the way out.
    assert _held(srv) == (False, False, False)


def test_registry_check_refuses_a_customer_the_registry_does_not_hold(mcp):
    """In memory but not in the registry (removed by hand, or never written): fail closed."""
    srv, _, client, _ = mcp
    config = srv.DatabaseConfig(
        customer_id="ghost", connection_string="sqlite:///:memory:", database_type="sqlite",
        created_at="2026-10-19T00:00:00", api_key="ak_ghost_" + "0" * 24,
    )
    srv._register_in_memory(config)
    assert _tool(client, config.api_key).status_code == 401
    assert "ghost" not in srv.CUSTOMER_API_KEYS


def test_registry_check_sees_a_change_made_by_another_process(mcp):
    """customer_dbs as its own process: it writes the shared file and nothing else."""
    srv, cdb, client, _ = mcp
    api_key = _register(client)
    assert _tool(client, api_key).status_code == 200  # the active flag is now cached
    registry = cdb.customer_manager._read_customers_file()
    registry["acme"]["status"] = "inactive"
    cdb.customer_manager.customers = registry
    cdb.customer_manager.save_customers()
    assert _tool(client, api_key).status_code == 401


def test_registry_check_refuses_a_key_the_registry_has_replaced(mcp):
    """Active, but under a new key written elsewhere: the key held in memory is dead."""
    srv, cdb, client, _ = mcp
    api_key = _register(client)
    assert _tool(client, api_key).status_code == 200
    registry = cdb.customer_manager._read_customers_file()
    registry["acme"]["api_key"] = "ak_rotated_" + "1" * 22
    cdb.customer_manager.customers = registry
    cdb.customer_manager.save_customers()
    assert registry["acme"]["status"] == "active"
    assert _tool(client, api_key).status_code == 401


# ---------------------------------------------------------------------------
# a restart / reload drops what the registry no longer allows
# ---------------------------------------------------------------------------

def test_reload_drops_a_stale_in_memory_customer(mcp, monkeypatch):
    srv, cdb, client, _ = mcp
    _register(client)
    engine = srv.db_manager.engines["acme"]

    async def _lost(customer_id):
        return False
    monkeypatch.setattr(cdb.customer_manager, "evict_from_mcp", _lost)
    assert client.delete("/customers/acme", headers=_as("user-a")).status_code == 200
    assert _held(srv) == (True, True, True)

    assert srv.load_registered_customers() == 0
    assert _held(srv) == (False, False, False)
    engine.dispose.assert_called_once()
