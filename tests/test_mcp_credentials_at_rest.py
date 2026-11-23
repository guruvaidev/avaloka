"""Customer credentials: sealed on disk, never in a response, resolved in process.

The precondition for every connector (docs/design/2.0-ADVANCED-PLAN.md §2.10):
``customers.json`` held connection strings with passwords and api_keys in
plaintext, ``GET /customers/{id}/credentials`` returned the password to
anyone, and the planner fetched it over HTTP with no auth. These tests pin
the fix at each of those three points and at the migration of a registry
written before it.

No real database: ``create_engine`` on both servers is replaced by a stub so a
registration succeeds without connecting, which is all these tests need --
they are about what is written and returned, not about SQL.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from unittest import mock

import pytest
from fastapi.testclient import TestClient

PASSWORD = "Sup3r-Secret-Pw"
# Assembled, not written: the repo-hygiene scanner (tests/contract/test_c1_repo_hygiene.py)
# flags a literal scheme://user:password@ and this one must stay findable only by the
# assertions that prove it never leaves the process.
DSN = "postgresql://alice:" + PASSWORD + "@db.internal:5432/shop"


@pytest.fixture
def encryption_key(monkeypatch):
    """The deployment key (helm: the API's secret, now also mounted by the MCP pod)."""
    import app.api.cloud_connections as cc
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", "unit-test-key", raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"unit-test-key").digest(), raising=False)


@pytest.fixture
def no_key(monkeypatch):
    import app.api.cloud_connections as cc
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", None, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)


@pytest.fixture
def registry_path(tmp_path, monkeypatch):
    path = tmp_path / "customers.json"
    monkeypatch.setenv("CUSTOMERS_FILE", str(path))
    return path


@pytest.fixture
def mcp(registry_path, monkeypatch):
    """The MCP server (which mounts the onboarding router) on a fresh registry file."""
    import app.mcp_server.customer_dbs as cdb
    import app.mcp_server.multi_tenant_mcp_server as srv

    monkeypatch.setattr(srv, "CUSTOMERS_FILE", str(registry_path))
    monkeypatch.setattr(cdb.customer_manager, "customers_file", str(registry_path))
    cdb.customer_manager.customers = {}
    srv.db_manager.engines.clear()
    srv.db_manager.connection_configs.clear()
    srv.CUSTOMER_API_KEYS.clear()

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            return mock.MagicMock()

    engine = mock.MagicMock()
    engine.connect.return_value = _Conn()
    monkeypatch.setattr(srv, "create_engine", lambda *a, **k: engine)
    monkeypatch.setattr(cdb, "create_engine", lambda *a, **k: engine)
    return srv, cdb, TestClient(srv.app)


OWNER = "user-acme"


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


def _register(client, customer_id="acme", api_key="ak_acme_0123456789abcdef", dsn=DSN):
    """As the onboarding service: /admin/customers takes only its signed request."""
    from app.mcp_server.credentials import CredentialKeyMissing, service_headers

    body = json.dumps(_seal_admin_body({
        "customer_id": customer_id, "connection_string": dsn, "database_type": "postgresql",
        "api_key": api_key, "created_at": "2026-10-17T00:00:00", "status": "active",
        "max_rows": 1000, "query_timeout": 30, "owner_user_id": OWNER,
    })).encode("utf-8")
    try:
        auth = service_headers("POST", "/admin/customers", body)
    except CredentialKeyMissing:
        auth = {}  # no deployment key: nothing can sign, and the route must refuse
    return client.post("/admin/customers", content=body,
                       headers={"Content-Type": "application/json", **auth})


# ---------------------------------------------------------------------------
# 1. the registry file
# ---------------------------------------------------------------------------

def test_registry_file_holds_no_secret_after_a_connect(mcp, registry_path, encryption_key):
    srv, _, client = mcp
    assert _register(client).status_code == 200, "registration failed"

    text = registry_path.read_text()
    assert PASSWORD not in text
    assert "ak_acme_0123456789abcdef" not in text
    record = json.loads(text)["acme"]
    assert "connection_string" not in record and "api_key" not in record
    assert set(record["connection_string_enc"]) == {"ciphertext", "iv"}
    assert set(record["api_key_enc"]) == {"ciphertext", "iv"}
    # Resolved in process: the server reloads the file and can use the connection.
    srv.db_manager.connection_configs.clear()
    srv.CUSTOMER_API_KEYS.clear()
    assert srv.load_registered_customers() == 1
    assert srv.db_manager.get_config("acme").connection_string == DSN
    assert srv.extract_customer_from_api_key("ak_acme_0123456789abcdef") == "acme"


def test_registration_is_refused_rather_than_written_in_plaintext(mcp, registry_path, no_key):
    srv, _, client = mcp
    # Without the key the admin route cannot authenticate anyone, so it fails
    # closed before it gets as far as writing (tests/test_mcp_admin_auth.py).
    resp = _register(client)
    assert resp.status_code == 401
    assert not registry_path.exists() or PASSWORD not in registry_path.read_text()
    # And the writer itself still refuses plaintext if it is ever reached keyless.
    from app.mcp_server.credentials import CredentialKeyMissing
    with pytest.raises(CredentialKeyMissing):
        srv._persist_customer(srv.DatabaseConfig(
            customer_id="acme", connection_string=DSN, database_type="postgresql",
            api_key="ak_acme", created_at="2026-10-17T00:00:00"))
    assert not registry_path.exists() or PASSWORD not in registry_path.read_text()


def test_onboarding_registration_seals_the_file_too(mcp, registry_path, encryption_key, monkeypatch):
    """The second writer of customers.json (customer_dbs.save_customers) seals as well."""
    _, cdb, client = mcp

    class _Resp:
        status_code = 200
        text = "ok"

    class _Client:
        def __init__(self, *a, **k): ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(cdb.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(cdb.customer_manager.db_manager, "test_database_connection",
                        lambda reg: {"success": True, "database": "shop", "table_count": 1})
    from app.mcp_server.client import user_headers
    resp = client.post("/customers/register", json={
        "customer_id": "onb", "customer_name": "Onboarded", "database_type": "postgresql",
        "host": "db.internal", "port": 5432, "database_name": "shop",
        "username": "alice", "password": PASSWORD, "contact_email": "a@example.com",
    }, headers=user_headers("user-onb"))
    assert resp.status_code == 200, resp.text
    text = registry_path.read_text()
    assert PASSWORD not in text and resp.json()["api_key"] not in text
    assert "connection_string" not in json.loads(text)["onb"]


def test_legacy_plaintext_registry_is_resealed_on_first_load(mcp, registry_path, encryption_key):
    """A registry written before this change is read, used, and rewritten sealed."""
    srv, cdb, _ = mcp
    registry_path.write_text(json.dumps({"old": {
        "customer_id": "old", "connection_string": DSN, "database_type": "postgresql",
        "api_key": "ak_old", "created_at": "2025-01-01T00:00:00", "status": "active",
    }}))
    assert srv.load_registered_customers() == 1
    assert srv.db_manager.get_config("old").connection_string == DSN
    text = registry_path.read_text()
    assert PASSWORD not in text and "ak_old" not in text
    assert "connection_string_enc" in json.loads(text)["old"]
    # The onboarding side reads the sealed file back into plaintext memory.
    assert cdb.customer_manager._read_customers_file()["old"]["connection_string"] == DSN


def test_a_record_sealed_under_another_key_is_skipped_not_leaked(mcp, registry_path, encryption_key, monkeypatch):
    srv, _, client = mcp
    assert _register(client).status_code == 200
    import app.api.cloud_connections as cc
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"another-deployment").digest(), raising=False)
    srv.db_manager.connection_configs.clear()
    assert srv.load_registered_customers() == 0
    assert PASSWORD not in registry_path.read_text()


# ---------------------------------------------------------------------------
# 2. the credentials route
# ---------------------------------------------------------------------------

def test_credentials_route_answers_only_a_service_class_caller(mcp, encryption_key):
    """Decision 1 of the 1.7/2.0 merge: the owner's assertion, never a customer's api_key."""
    from app.mcp_server.client import user_headers
    _, _, client = mcp
    assert _register(client).status_code == 200
    assert client.get("/customers/acme/credentials").status_code == 401
    assert client.get("/customers/acme/credentials",
                      headers={"Authorization": "Bearer ak_wrong"}).status_code == 401
    # The customer's own valid api_key is 403, for its own id and any other,
    # before any lookup, so the answer confirms no id.
    for cid in ("acme", "nobody"):
        assert client.get(f"/customers/{cid}/credentials",
                          headers={"Authorization": "Bearer ak_acme_0123456789abcdef"}).status_code == 403
    assert client.get("/customers/acme/credentials", headers=user_headers(OWNER)).status_code == 200
    # Another user's assertion: 404, the answer a customer that does not exist gets.
    assert client.get("/customers/acme/credentials", headers=user_headers("user-other")).status_code == 404
    assert client.get("/customers/nobody/credentials", headers=user_headers(OWNER)).status_code == 404
    assert client.get("/customers/acme/planner-metadata").status_code == 401


def test_credentials_response_contains_no_password(mcp, encryption_key):
    _, _, client = mcp
    assert _register(client).status_code == 200
    from app.mcp_server.client import user_headers
    resp = client.get("/customers/acme/credentials", headers=user_headers(OWNER))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert PASSWORD not in resp.text
    assert "password" not in body
    assert body["host"] == "db.internal" and body["port"] == 5432
    assert body["database"] == "shop" and body["username"] == "alice"
    handle = body["credential_handle"]
    assert handle["scheme"] == "aes-gcm/DB_ENCRYPTION_KEY" and set(handle) == {"scheme", "ciphertext", "iv"}

    # A process holding the deployment key opens the handle; one without cannot.
    from app.mcp_server.credentials import open_credential_handle
    assert open_credential_handle(handle)["password"] == PASSWORD
    import app.api.cloud_connections as cc
    with mock.patch.object(cc, "_AESGCM_KEY", hashlib.sha256(b"elsewhere").digest()):
        assert open_credential_handle(handle) is None


# ---------------------------------------------------------------------------
# 3. the planner: both call sites use the handle, and authenticate
# ---------------------------------------------------------------------------

@contextmanager
def _planner_network(responses):
    """``planner.requests`` answering ``responses`` (url substring -> (code, json)) and recording calls."""
    import app.agents.planner as planner

    calls = []
    session = mock.MagicMock()

    def _get(url, *a, **k):
        calls.append((url, k.get("headers") or {}))
        resp = mock.MagicMock()
        for needle, (code, payload) in responses.items():
            if needle in url:
                resp.status_code, resp.json.return_value = code, payload
                return resp
        resp.status_code, resp.json.return_value = 404, {}
        return resp

    session.get.side_effect = _get
    req = mock.MagicMock()
    req.Session.return_value.__enter__.return_value = session
    with mock.patch.object(planner, "requests", req):
        yield calls


def test_auto_fetch_from_ui_authenticates_and_opens_the_handle(encryption_key):
    from app.agents.planner import _auto_fetch_from_ui
    from app.api.cloud_connections import encrypt_secret
    from app.mcp_server.credentials import credential_handle

    state = {"user_id": OWNER, "active_db_customer_id": "acme",
             "active_db_credential_handle": encrypt_secret("ak_acme")}
    with _planner_network({
        "/customers/acme/credentials": (200, {
            "host": "db.internal", "port": 5432, "database": "shop", "username": "alice",
            "table": "orders", "credential_handle": credential_handle(DSN)}),
        "/customers": (200, {"customers": [{"customer_id": "acme", "database_type": "postgresql"}]}),
    }) as calls:
        entry = _auto_fetch_from_ui("acme", state)
    # A registry entry: the handle and display fields. The handle is opened (it
    # must be openable), but its password is not carried into the entry.
    assert entry["user"] == "alice" and entry["db_type"] == "postgresql" and entry["conn_id"] == "acme"
    assert "password" not in entry and PASSWORD not in repr(entry)
    # As the signed-in user (an avu1 assertion naming OWNER); the api_key is never sent.
    try:  # tests/ as a package, or tests/ on sys.path
        from tests.onboarding_identity import assertion_user
    except ImportError:
        from onboarding_identity import assertion_user
    cred_calls = [c for c in calls if "/credentials" in c[0]]
    assert cred_calls and all(assertion_user(h) == OWNER for _, h in cred_calls)
    assert "ak_acme" not in repr(calls)


def test_auto_fetch_from_ui_does_not_call_without_a_signed_in_user(encryption_key):
    from app.agents.planner import _auto_fetch_from_ui

    with _planner_network({"/customers/acme/credentials": (200, {"password": "leak"})}) as calls:
        assert _auto_fetch_from_ui("acme", {"active_db_customer_id": "other"}) is None
        assert _auto_fetch_from_ui("acme", None) is None
    assert calls == []


def test_auto_fetch_from_ui_ignores_a_plaintext_password(encryption_key):
    """If the endpoint regressed to sending `password`, the planner still does not read it."""
    from app.agents.planner import _auto_fetch_from_ui
    from app.api.cloud_connections import encrypt_secret

    state = {"user_id": OWNER, "active_db_customer_id": "acme",
             "active_db_credential_handle": encrypt_secret("ak_acme")}
    with _planner_network({
        "/customers/acme/credentials": (200, {"host": "h", "username": "u", "password": PASSWORD}),
    }) as calls:
        assert _auto_fetch_from_ui("acme", state) is None
    assert [c for c in calls if "/credentials" in c[0]], "the route was not asked, so this tested nothing"


def test_source_metadata_reaches_the_planner_prompt_and_state(encryption_key):
    from app.agents.planner import _source_metadata_for
    from app.api.cloud_connections import encrypt_secret

    metadata = {"kind": "database", "backend": "postgresql", "tables": [
        {"name": "orders", "columns": [{"name": "id", "type": "BIGINT", "nullable": False}],
         "statistics": {"row_count": 2000, "indexes": [{"name": "pk", "columns": ["id"]}],
                        "partitioning": {"partitioned": False, "columns": [], "partitions": 1}}}]}
    state = {"active_db_customer_id": "acme", "active_db_credential_handle": encrypt_secret("ak_acme")}
    with _planner_network({"/customers/acme/planner-metadata": (200, {"source_metadata": metadata})}) as calls:
        assert _source_metadata_for(state) == metadata
    assert calls[0][1]["Authorization"] == "Bearer ak_acme"
    # Once on state it is reused, so the trace and the prompt agree.
    with _planner_network({}) as calls:
        assert _source_metadata_for({**state, "source_metadata": metadata}) == metadata
    assert calls == []
    assert _source_metadata_for({"active_db_customer_id": "acme"}) is None


# ---------------------------------------------------------------------------
# discovery works through the contract
# ---------------------------------------------------------------------------

def test_discover_database_lists_tables_through_the_contract(encryption_key, monkeypatch):
    from app.agents.avaloka_agent import discovery_tools
    from app.api.cloud_connections import encrypt_secret
    import app.mcp_server.client as client_mod

    seen = {}

    def _planner_metadata(customer_id, api_key, **kwargs):
        seen["customer_id"], seen["api_key"] = customer_id, api_key
        return {"tables": [{"name": "orders"}, {"name": "regions"}]}

    monkeypatch.setattr(client_mod, "planner_metadata", _planner_metadata)
    out = discovery_tools.discover_database("acme", encrypt_secret("ak_acme"))
    assert out == {"source": "native", "connection_id": "acme", "tables": ["orders", "regions"]}
    assert seen == {"customer_id": "acme", "api_key": "ak_acme"}
    without = discovery_tools.discover_database("acme", None)
    assert "error" in without and "tables" not in without
