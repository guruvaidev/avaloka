"""Every onboarding /customers route needs an identity and answers only for its owner.

Before this, register, status, list, delete and test-connection had no auth at
all: anyone who could reach the onboarding API could list every customer,
probe their databases and deactivate them. Two identities are accepted, both
as Bearer:

* a customer's own api_key (#526) -- that one customer;
* a user assertion the API server signs for the user it authenticated
  (``app.mcp_server.client.user_headers``) -- the customers that user registered.

No identity, or a forged one, is 401. Somebody else's customer is 404, the
same as one that does not exist.

No real database and no network: ``create_engine`` is a stub and the MCP
server's /admin/customers forward is answered in process.
"""

from __future__ import annotations

import hashlib
import time
from unittest import mock

import pytest
from fastapi.testclient import TestClient

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import client, make_auth_headers  # noqa: F401  (fixture used by name)
except ImportError:
    from server_harness import client, make_auth_headers  # noqa: F401

PASSWORD = "Onb0arding-Pw-" + "x9"
# Never assembled into a literal DSN: the repo-hygiene scanner flags those.
REG_A = {
    "customer_id": "acme", "customer_name": "Acme", "database_type": "postgresql",
    "host": "db.internal", "port": 5432, "database_name": "shop",
    "username": "alice", "password": PASSWORD, "contact_email": "a@example.com",
}
REG_B = {**REG_A, "customer_id": "bcorp", "customer_name": "B Corp", "username": "bob",
         "contact_email": "b@example.com"}

PER_CUSTOMER_ROUTES = [
    ("GET", "/customers/{cid}/status"),
    ("POST", "/customers/{cid}/test-connection"),
    ("DELETE", "/customers/{cid}"),
    ("GET", "/customers/{cid}/credentials"),
    ("GET", "/customers/{cid}/planner-metadata"),
]


@pytest.fixture
def encryption_key(monkeypatch):
    import app.api.cloud_connections as cc
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", "unit-test-key", raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"unit-test-key").digest(), raising=False)


@pytest.fixture
def onboarding(tmp_path, monkeypatch, encryption_key):
    """The MCP server app (which mounts the onboarding router) on a fresh registry."""
    import app.mcp_server.customer_dbs as cdb
    import app.mcp_server.multi_tenant_mcp_server as srv

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

    monkeypatch.setattr(cdb.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(cdb.customer_manager.db_manager, "test_database_connection",
                        lambda reg: {"success": True, "database": "shop", "table_count": 1})
    return cdb, TestClient(srv.app)


def _as(user_id):
    from app.mcp_server.client import user_headers
    headers = user_headers(user_id)
    assert headers, "no assertion: the deployment key is missing in the fixture"
    return headers


def _key(api_key):
    return {"Authorization": f"Bearer {api_key}"}


@pytest.fixture
def two_users(onboarding):
    """User A owns acme, user B owns bcorp. Returns (client, {cid: api_key})."""
    cdb, client = onboarding
    keys = {}
    for user, reg in (("user-a", REG_A), ("user-b", REG_B)):
        resp = client.post("/customers/register", json=reg, headers=_as(user))
        assert resp.status_code == 200, resp.text
        keys[reg["customer_id"]] = resp.json()["api_key"]
    return client, keys


def _call(client, method, path, headers=None):
    return client.request(method, path, headers=headers or {})


# ---------------------------------------------------------------------------
# unauthenticated: 401 everywhere
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,route", PER_CUSTOMER_ROUTES + [("GET", "/customers")])
@pytest.mark.parametrize("headers", [
    None,
    {"Authorization": "Bearer ak_not_a_key"},
    {"Authorization": "Basic dXNlcjpwdw=="},
], ids=["none", "unknown-key", "not-bearer"])
def test_unauthenticated_is_401(two_users, method, route, headers):
    client, _ = two_users
    resp = _call(client, method, route.format(cid="acme"), headers)
    assert resp.status_code == 401, (method, route, resp.status_code, resp.text)


def test_register_without_a_user_is_401(onboarding):
    _, client = onboarding
    assert client.post("/customers/register", json=REG_A).status_code == 401
    assert client.post("/customers/register", json=REG_A,
                       headers={"Authorization": "Bearer ak_whatever"}).status_code == 401


def test_register_with_a_customers_api_key_is_401(two_users):
    """An api_key names a database, not a person: it cannot create customers."""
    client, keys = two_users
    resp = client.post("/customers/register", json={**REG_A, "customer_id": "acme2"},
                       headers=_key(keys["acme"]))
    assert resp.status_code == 401


def test_forged_and_expired_assertions_are_401(two_users, monkeypatch):
    client, _ = two_users
    from app.mcp_server import credentials

    good = _as("user-a")["Authorization"].split(" ", 1)[1]
    prefix, user, expires, mac = good.split(".")
    forged = f"{prefix}.user-b.{expires}.{mac}"
    assert client.get("/customers", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    expired = credentials.user_assertion("user-a", ttl_s=-1)
    assert client.get("/customers", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    # Signed under another deployment's key.
    import app.api.cloud_connections as cc
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"elsewhere").digest(), raising=False)
    other = credentials.user_assertion("user-a")
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"unit-test-key").digest(), raising=False)
    assert client.get("/customers", headers={"Authorization": f"Bearer {other}"}).status_code == 401


def test_no_assertion_is_made_without_the_deployment_key(monkeypatch):
    import app.api.cloud_connections as cc
    from app.mcp_server.client import user_headers
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", None, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    assert user_headers("user-a") == {}
    assert user_headers(None) == {} and user_headers("") == {}


# ---------------------------------------------------------------------------
# cross-user: B never sees, reads, tests or deletes A's customer
# ---------------------------------------------------------------------------

def test_list_shows_only_the_callers_own(two_users):
    client, keys = two_users
    a = client.get("/customers", headers=_as("user-a")).json()
    b = client.get("/customers", headers=_as("user-b")).json()
    assert [c["customer_id"] for c in a["customers"]] == ["acme"] and a["total_count"] == 1
    assert [c["customer_id"] for c in b["customers"]] == ["bcorp"] and b["total_count"] == 1
    # A user who registered nothing sees nothing.
    assert client.get("/customers", headers=_as("user-c")).json()["customers"] == []
    # An api_key sees its own customer only.
    by_key = client.get("/customers", headers=_key(keys["bcorp"])).json()
    assert [c["customer_id"] for c in by_key["customers"]] == ["bcorp"]


@pytest.mark.parametrize("method,route", PER_CUSTOMER_ROUTES)
@pytest.mark.parametrize("who", ["user-b", "bcorp-key"])
def test_cross_user_is_404_like_a_missing_customer(two_users, method, route, who):
    client, keys = two_users
    headers = _as("user-b") if who == "user-b" else _key(keys["bcorp"])
    theirs = _call(client, method, route.format(cid="acme"), headers)
    missing = _call(client, method, route.format(cid="no-such-customer"), headers)
    # /credentials refuses every api_key before any lookup (merge decision 1): 403, for any id.
    expected = 403 if who == "bcorp-key" and route.endswith("/credentials") else 404
    assert theirs.status_code == expected, (method, route, theirs.status_code, theirs.text)
    assert (missing.status_code, missing.json()) == (theirs.status_code, theirs.json())
    assert PASSWORD not in theirs.text and keys["acme"] not in theirs.text


def test_b_cannot_deactivate_as_customer(two_users):
    client, keys = two_users
    assert client.delete("/customers/acme", headers=_as("user-b")).status_code == 404
    assert client.delete("/customers/acme", headers=_key(keys["bcorp"])).status_code == 404
    status = client.get("/customers/acme/status", headers=_as("user-a"))
    assert status.status_code == 200 and status.json()["status"] == "active"


def test_b_cannot_take_over_a_registered_id(two_users):
    client, keys = two_users
    resp = client.post("/customers/register", json=REG_A, headers=_as("user-b"))
    assert resp.status_code == 409
    status = client.get("/customers/acme/status", headers=_key(keys["acme"]))
    assert status.status_code == 200, "A's key stopped working after B's attempt"
    creds = client.get("/customers/acme/credentials", headers=_as("user-a"))
    assert creds.status_code == 200, "A lost its credentials after B's attempt"


def test_one_users_assertion_never_gets_another_users_handle(two_users):
    """Ownership is at least as strict as possession (merge decision 1, requirement 1).

    User A's signed assertion on user B's record is refused, on the credentials
    route and every other per-customer route, and carries no part of B's secret.
    """
    client, keys = two_users
    for cid, owner, other in (("bcorp", "user-b", "user-a"), ("acme", "user-a", "user-b")):
        r = client.get(f"/customers/{cid}/credentials", headers=_as(other))
        assert r.status_code in (403, 404), (cid, r.status_code, r.text)
        assert "credential_handle" not in r.text and PASSWORD not in r.text
        assert client.get(f"/customers/{cid}/credentials", headers=_as(owner)).status_code == 200


def test_a_deleted_customers_handle_fetch_fails(two_users):
    """Requirement 2: once deleted, the owner's assertion gets no handle and the key is dead."""
    client, keys = two_users
    assert client.get("/customers/acme/credentials", headers=_as("user-a")).status_code == 200
    assert client.delete("/customers/acme", headers=_as("user-a")).status_code == 200
    r = client.get("/customers/acme/credentials", headers=_as("user-a"))
    assert r.status_code == 404 and "credential_handle" not in r.text
    assert client.get("/customers/acme/credentials", headers=_key(keys["acme"])).status_code == 401


def test_a_record_with_no_owner_is_reachable_only_by_its_key(two_users, onboarding):
    """Registered before owners were recorded: nobody's by user, still its key's."""
    cdb, _ = onboarding
    client, keys = two_users
    registry = cdb.customer_manager._read_customers_file()
    registry["acme"].pop("owner_user_id")
    cdb.customer_manager.customers = registry
    cdb.customer_manager.save_customers()
    assert client.get("/customers", headers=_as("user-a")).json()["customers"] == []
    assert client.get("/customers/acme/status", headers=_as("user-a")).status_code == 404
    assert client.get("/customers/acme/status", headers=_key(keys["acme"])).status_code == 200


# ---------------------------------------------------------------------------
# the owner's flow still works
# ---------------------------------------------------------------------------

def test_owner_flow(two_users, onboarding):
    cdb, _ = onboarding
    client, keys = two_users
    record = cdb.customer_manager._read_customers_file()["acme"]
    assert record["owner_user_id"] == "user-a"

    for headers in (_as("user-a"), _key(keys["acme"])):
        assert client.get("/customers/acme/status", headers=headers).status_code == 200
        tc = client.post("/customers/acme/test-connection", headers=headers)
        assert tc.status_code == 200 and tc.json()["customer_id"] == "acme"
    # The handle goes to the owner's assertion only; the customer's own api_key
    # is 403 (merge decision 1). Neither ever carries the password.
    creds = client.get("/customers/acme/credentials", headers=_as("user-a"))
    assert creds.status_code == 200 and "password" not in creds.json()
    assert PASSWORD not in creds.text
    by_key = client.get("/customers/acme/credentials", headers=_key(keys["acme"]))
    assert by_key.status_code == 403 and "password" not in by_key.json()
    assert PASSWORD not in by_key.text

    resp = client.delete("/customers/acme", headers=_as("user-a"))
    assert resp.status_code == 200
    listed = client.get("/customers", headers=_as("user-a")).json()["customers"]
    assert [(c["customer_id"], c["status"]) for c in listed] == [("acme", "inactive")]
    # B's customer is untouched by A's delete.
    b = client.get("/customers", headers=_as("user-b")).json()["customers"]
    assert [(c["customer_id"], c["status"]) for c in b] == [("bcorp", "active")]


def test_list_never_carries_a_secret(two_users):
    client, keys = two_users
    for headers in (_as("user-a"), _key(keys["acme"])):
        resp = client.get("/customers", headers=headers)
        assert resp.status_code == 200
        text = resp.text
        assert PASSWORD not in text
        assert keys["acme"] not in text and keys["bcorp"] not in text
        for field in ("password", "api_key", "connection_string", "owner_user_id"):
            assert f'"{field}"' not in text, field


# ---------------------------------------------------------------------------
# callers: the API server relays registration as the user; the planner lists as the user
# ---------------------------------------------------------------------------

def test_api_server_register_relays_as_the_authenticated_user(client, monkeypatch, onboarding, encryption_key):
    """The UI's register goes through the API server, which signs for the JWT's user."""
    import app.api.server as server
    from app.mcp_server.credentials import verify_user_assertion

    _, onboarding_client = onboarding
    seen = {}

    class _Upstream:
        def __init__(self, resp):
            self.status_code, self._resp = resp.status_code, resp

        def json(self):
            return self._resp.json()

    class _AsyncClient:
        def __init__(self, *a, **k): ...

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, content=None, headers=None, **k):
            # httpx is one module: this stub also answers customer_dbs's own
            # forward to the MCP server's /admin/customers.
            if url.endswith("/admin/customers"):
                return mock.MagicMock(status_code=200, text="ok")
            seen["url"], seen["headers"] = url, dict(headers or {})
            # Hand the relayed request to the real onboarding app.
            return _Upstream(onboarding_client.post(
                "/customers/register", content=content, headers=headers))

    monkeypatch.setattr(server.httpx, "AsyncClient", _AsyncClient)

    assert client.post("/customers/register", json=REG_A).status_code == 401
    assert "url" not in seen, "an unauthenticated register was forwarded"

    resp = client.post("/customers/register", json=REG_A, headers=make_auth_headers("user-a"))
    assert resp.status_code == 200, resp.text
    assert seen["url"].endswith("/customers/register")
    token = seen["headers"]["Authorization"].split(" ", 1)[1]
    assert verify_user_assertion(token) == "user-a"
    listed = onboarding_client.get("/customers", headers=_as("user-a")).json()["customers"]
    assert [c["customer_id"] for c in listed] == ["acme"]


def test_planner_lists_databases_as_the_requester(encryption_key):
    import app.agents.planner as planner
    import app.api.cloud_connections as cc
    from app.mcp_server.credentials import verify_user_assertion

    async def _no_cloud(user_id):
        return []

    calls = []
    session = mock.MagicMock()

    def _get(url, *a, **k):
        calls.append((url, k.get("headers") or {}))
        resp = mock.MagicMock()
        resp.status_code, resp.json.return_value = 200, {"customers": [
            {"customer_id": "acme", "database_type": "postgresql"}]}
        return resp

    session.get.side_effect = _get
    req = mock.MagicMock()
    req.Session.return_value.__enter__.return_value = session
    with mock.patch.object(planner, "requests", req), \
            mock.patch.object(planner, "_get_db_registry", lambda state: {}), \
            mock.patch.object(cc, "list_cloud_connections", _no_cloud):
        out = planner._handle_list_databases({"user_id": "user-a"})
        planner._list_available_destinations({"user_id": "user-a"})
    assert "acme" in out
    listing = [h for url, h in calls if url.endswith("/customers")]
    assert len(listing) == 2
    for headers in listing:
        assert verify_user_assertion(headers["Authorization"].split(" ", 1)[1]) == "user-a"
