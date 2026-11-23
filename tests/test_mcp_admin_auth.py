"""The MCP server's admin routes take only the onboarding service's signed request.

``POST /admin/customers`` created or replaced any customer's connection
string and api_key with no auth, and connected to whatever DSN it was given
before registering: account takeover plus SSRF. ``/connect`` minted an
api_key for any DSN, and ``/connections`` listed every customer.

Now each needs ``credentials.service_headers``: a MAC over method, path,
timestamp and body under a key derived from DB_ENCRYPTION_KEY with its own
label. A user assertion -- the other credential built on that key -- does not
verify here. An update to an existing customer must name its owner.

No database and no network: ``create_engine`` is a recorder.
"""

from __future__ import annotations

import hashlib
import json
from unittest import mock

import pytest
from fastapi.testclient import TestClient

PASSWORD = "Adm1n-Route-Pw-" + "q7"
DSN = "postgresql://alice:" + PASSWORD + "@db.internal:5432/shop"
ATTACKER_DSN = "postgresql://x:" + "y" + "@169.254.169.254:80/latest"


@pytest.fixture
def encryption_key(monkeypatch):
    import app.api.cloud_connections as cc
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", "unit-test-key", raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"unit-test-key").digest(), raising=False)


@pytest.fixture
def mcp(tmp_path, monkeypatch, encryption_key):
    import app.mcp_server.customer_dbs as cdb
    import app.mcp_server.multi_tenant_mcp_server as srv

    path = tmp_path / "customers.json"
    monkeypatch.setenv("CUSTOMERS_FILE", str(path))
    monkeypatch.setattr(srv, "CUSTOMERS_FILE", str(path))
    monkeypatch.setattr(cdb.customer_manager, "customers_file", str(path))
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

    dialed = []

    def _engine(url, *a, **k):
        dialed.append(str(url))
        engine = mock.MagicMock()
        engine.connect.return_value = _Conn()
        return engine

    monkeypatch.setattr(srv, "create_engine", _engine)
    monkeypatch.setattr(cdb, "create_engine", _engine)
    return srv, cdb, TestClient(srv.app), dialed


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


def _payload(customer_id="acme", api_key="ak_acme_0123456789abcdef", dsn=DSN, owner="user-a"):
    body = {
        "customer_id": customer_id, "connection_string": dsn, "database_type": "postgresql",
        "api_key": api_key, "created_at": "2026-10-18T00:00:00", "status": "active",
        "max_rows": 1000, "query_timeout": 30,
    }
    if owner is not None:
        body["owner_user_id"] = owner
    return json.dumps(_seal_admin_body(body)).encode("utf-8")


def _signed(body, path="/admin/customers", method="POST"):
    from app.mcp_server.credentials import service_headers
    return {"Content-Type": "application/json", **service_headers(method, path, body)}


def _post(client, body, headers):
    return client.post("/admin/customers", content=body, headers=headers)


# ---------------------------------------------------------------------------
# authentication
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer ak_acme_0123456789abcdef"},
    {"Authorization": "Bearer avs1.1700000000.deadbeef"},
], ids=["none", "api-key", "garbage-mac"])
def test_admin_customers_unauthenticated_is_401_and_dials_nothing(mcp, headers):
    srv, _, client, dialed = mcp
    resp = _post(client, _payload(dsn=ATTACKER_DSN), {"Content-Type": "application/json", **headers})
    assert resp.status_code == 401
    assert dialed == [], "the server connected to a caller's DSN before authenticating it"
    assert "acme" not in srv.CUSTOMER_API_KEYS


def test_a_user_assertion_replayed_at_admin_is_401(mcp):
    """Domain separation: the user credential, from the same deployment key, is not a service one."""
    _, _, client, dialed = mcp
    from app.mcp_server.client import user_headers
    resp = _post(client, _payload(), {"Content-Type": "application/json", **user_headers("user-a")})
    assert resp.status_code == 401 and dialed == []


def test_a_service_mac_is_not_a_user_assertion(mcp):
    """And the other way: a service MAC presented to the onboarding routes is no user."""
    _, _, client, _ = mcp
    headers = _signed(b"", path="/customers", method="GET")
    assert client.get("/customers", headers=headers).status_code == 401


def test_mac_is_bound_to_body_key_and_time(mcp, monkeypatch):
    _, _, client, dialed = mcp
    from app.mcp_server import credentials

    # Signed for one body, sent with another (a different DSN).
    assert _post(client, _payload(dsn=ATTACKER_DSN), _signed(_payload())).status_code == 401
    # Signed for another path.
    assert _post(client, _payload(), _signed(_payload(), path="/connect")).status_code == 401
    # Signed long ago.
    real_time = credentials.time.time
    monkeypatch.setattr(credentials.time, "time", lambda: real_time() - 3600)
    stale = _signed(_payload())
    monkeypatch.setattr(credentials.time, "time", real_time)
    assert _post(client, _payload(), stale).status_code == 401
    # Signed under another deployment's key.
    import app.api.cloud_connections as cc
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"elsewhere").digest(), raising=False)
    foreign = _signed(_payload())
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"unit-test-key").digest(), raising=False)
    assert _post(client, _payload(), foreign).status_code == 401
    assert dialed == []


def test_no_key_means_no_admin(mcp, monkeypatch):
    _, _, client, _ = mcp
    headers = _signed(_payload())
    import app.api.cloud_connections as cc
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", None, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    assert _post(client, _payload(), headers).status_code == 401


@pytest.mark.parametrize("method,path", [("POST", "/connect"), ("GET", "/connections")])
def test_other_admin_routes_need_the_service_credential(mcp, method, path):
    _, _, client, dialed = mcp
    body = json.dumps({"connection_name": "evil", "database_type": "postgresql",
                       "host": "169.254.169.254", "port": 80, "database_name": "x",
                       "username": "x", "password": "y"}).encode()
    resp = client.request(method, path, content=body if method == "POST" else None,
                          headers={"Content-Type": "application/json"})
    assert resp.status_code == 401 and dialed == []


# ---------------------------------------------------------------------------
# the legitimate onboarding call
# ---------------------------------------------------------------------------

def test_signed_registration_works(mcp):
    srv, _, client, dialed = mcp
    resp = _post(client, _payload(), _signed(_payload()))
    assert resp.status_code == 200, resp.text
    assert srv.extract_customer_from_api_key("ak_acme_0123456789abcdef") == "acme"
    assert srv.db_manager.get_config("acme").owner_user_id == "user-a"
    assert dialed == [DSN, DSN]  # the pre-registration test, then the pooled engine


def test_onboarding_register_reaches_mcp_through_the_real_check(mcp, monkeypatch):
    """customer_dbs signs its /admin/customers call; the MCP server verifies it for real."""
    srv, cdb, client, _ = mcp

    class _AsyncClient:
        def __init__(self, *a, **k): ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, content=None, headers=None, json=None, **k):
            assert url.endswith("/admin/customers")
            return client.post("/admin/customers", content=content, headers=headers)

    monkeypatch.setattr(cdb.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(cdb.customer_manager.db_manager, "test_database_connection",
                        lambda reg: {"success": True, "database": "shop", "table_count": 1})
    from app.mcp_server.client import user_headers
    resp = client.post("/customers/register", headers=user_headers("user-a"), json={
        "customer_id": "onb", "customer_name": "Onboarded", "database_type": "postgresql",
        "host": "db.internal", "port": 5432, "database_name": "shop",
        "username": "alice", "password": PASSWORD, "contact_email": "a@example.com",
    })
    assert resp.status_code == 200, resp.text
    api_key = resp.json()["api_key"]
    assert srv.extract_customer_from_api_key(api_key) == "onb"
    assert srv.db_manager.get_config("onb").owner_user_id == "user-a"


# ---------------------------------------------------------------------------
# no silent takeover of an existing customer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("owner", ["user-b", None], ids=["other-owner", "no-owner"])
def test_overwriting_another_owners_customer_is_refused(mcp, owner):
    srv, _, client, dialed = mcp
    assert _post(client, _payload(), _signed(_payload())).status_code == 200
    dialed.clear()
    hijack = _payload(api_key="ak_attacker_000000000000000000", dsn=ATTACKER_DSN, owner=owner)
    resp = _post(client, hijack, _signed(hijack))
    assert resp.status_code == 409
    assert dialed == [], "the server dialled the replacement DSN"
    assert srv.extract_customer_from_api_key("ak_acme_0123456789abcdef") == "acme"
    assert srv.extract_customer_from_api_key("ak_attacker_000000000000000000") is None
    assert srv.db_manager.get_config("acme").connection_string == DSN


def test_the_owner_may_update_their_own_customer(mcp):
    srv, _, client, _ = mcp
    assert _post(client, _payload(), _signed(_payload())).status_code == 200
    update = _payload(api_key="ak_acme_rotated_000000000000000")
    assert _post(client, update, _signed(update)).status_code == 200
    assert srv.extract_customer_from_api_key("ak_acme_rotated_000000000000000") == "acme"


def test_an_unowned_customer_is_re_pushed_only_with_its_own_key(mcp):
    """scripts/resync_mcp_customers.py re-pushes legacy records unchanged; nothing else may."""
    _, _, client, _ = mcp
    legacy = _payload(owner=None)
    assert _post(client, legacy, _signed(legacy)).status_code == 200
    assert _post(client, legacy, _signed(legacy)).status_code == 200
    swapped = _payload(api_key="ak_attacker_000000000000000000", owner=None)
    assert _post(client, swapped, _signed(swapped)).status_code == 409
    claimed = _payload(owner="user-b")
    assert _post(client, claimed, _signed(claimed)).status_code == 409


def test_connect_never_replaces_an_existing_customer(mcp):
    srv, _, client, dialed = mcp
    assert _post(client, _payload(), _signed(_payload())).status_code == 200
    dialed.clear()
    body = json.dumps({"connection_name": "acme", "database_type": "postgresql",
                       "host": "169.254.169.254", "port": 80, "database_name": "x",
                       "username": "x", "password": "y"}).encode()
    resp = client.post("/connect", content=body, headers=_signed(body, path="/connect"))
    assert resp.status_code == 409 and dialed == []
    assert srv.extract_customer_from_api_key("ak_acme_0123456789abcdef") == "acme"
