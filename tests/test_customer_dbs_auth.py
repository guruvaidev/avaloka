"""Security hotfix: the onboarding /customers routes need an identity.

Before this fix every route in app/mcp_server/customer_dbs.py was open, and
GET /customers/{id}/credentials handed anyone the plaintext database password.
These tests pin the fix:

  * no identity                -> 401 on every route;
  * someone else's customer    -> 404 (not 403, so ids do not leak);
  * GET /customers             -> the caller's own customers, no secrets;
  * /credentials               -> a password ONLY for the API service holding
                                  MCP_SERVICE_TOKEN, and never when it is unset;
  * /admin/customers           -> service-only, since it overwrites api_keys;
  * the planner's DTA auto-fetch still gets the password, end to end, by
    sending the token and the user it acts for.

Every key, token and password here is generated per test. None is real.
"""
import json
import secrets
import sys
import types
from types import SimpleNamespace
from urllib.parse import quote_plus

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.mcp_server.customer_dbs as cdb

TOKEN_HEADER = "X-Avaloka-Service-Token"
USER_HEADER = "X-Avaloka-User"


def _record(cid, api_key, password, owner_user_id=None):
    rec = {
        "customer_id": cid,
        "customer_name": f"{cid} name",
        "database_type": "postgresql",
        "connection_string": f"postgresql://alice:{quote_plus(password)}@db.test.invalid:5432/sales",
        "contact_email": f"{cid}@example.test",
        "environment": "production",
        "api_key": api_key,
        "created_at": "2026-10-18T00:00:00",
        "status": "active",
        "query_timeout": 30,
        "max_rows": 1000,
        "last_health_check": "2026-10-18T00:00:00",
        "database_info": {"database": "sales", "table_count": 1, "version": "16"},
    }
    if owner_user_id:
        rec["owner_user_id"] = owner_user_id
    return rec


class _FakeResult:
    def fetchone(self):
        # (table,) for /credentials' auto-detect; (version, database) for test-connection.
        return ("orders", "sales")


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        return _FakeResult()


class _FakeEngine:
    def connect(self):
        return _FakeConn()

    def dispose(self):
        pass


@pytest.fixture
def reg(tmp_path, monkeypatch):
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("MCP_SERVICE_TOKEN", token)

    keys = {cid: "ak_" + secrets.token_hex(16) for cid in ("cust_a", "cust_a2", "cust_b", "cust_legacy")}
    pws = {cid: secrets.token_urlsafe(18) for cid in keys}
    owners = {"cust_a": "user_a", "cust_a2": "user_a", "cust_b": "user_b", "cust_legacy": None}
    records = {cid: _record(cid, keys[cid], pws[cid], owners[cid]) for cid in keys}

    path = tmp_path / "customers.json"
    path.write_text(json.dumps(records))
    mgr = cdb.customer_manager
    monkeypatch.setattr(mgr, "customers_file", str(path))
    monkeypatch.setattr(mgr, "customers", {})
    mgr.load_customers()
    monkeypatch.setattr(cdb, "create_engine", lambda *a, **k: _FakeEngine())

    app = FastAPI()
    app.include_router(cdb.router)
    return SimpleNamespace(
        client=TestClient(app), token=token, keys=keys, pws=pws, path=path,
        records=records,
    )


def _bearer(key):
    return {"Authorization": f"Bearer {key}"}


def _service(token, user):
    return {TOKEN_HEADER: token, USER_HEADER: user}


def _on_disk(reg):
    return json.loads(reg.path.read_text())


ROUTES = [
    ("GET", "/customers"),
    ("GET", "/customers/cust_a/status"),
    ("DELETE", "/customers/cust_a"),
    ("POST", "/customers/cust_a/test-connection"),
    ("GET", "/customers/cust_a/credentials"),
    ("POST", "/customers/cust_a/owner"),
]
PER_CUSTOMER = [r for r in ROUTES if r[1] != "/customers" and not r[1].endswith("/owner")]


# ---- 401 without an identity ----------------------------------------------

@pytest.mark.parametrize("method,path", ROUTES)
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer ak_not_a_real_key"}, {"Authorization": "Basic x"}])
def test_no_identity_is_401_on_every_route(reg, method, path, headers):
    r = reg.client.request(method, path, headers=headers)
    assert r.status_code == 401
    assert reg.pws["cust_a"] not in r.text
    assert _on_disk(reg)["cust_a"]["status"] == "active"


def test_deactivated_customer_key_no_longer_authenticates(reg):
    assert reg.client.delete("/customers/cust_a2", headers=_bearer(reg.keys["cust_a2"])).status_code == 200
    assert reg.client.get("/customers", headers=_bearer(reg.keys["cust_a2"])).status_code == 401


def test_service_caller_must_name_the_user(reg):
    r = reg.client.get("/customers/cust_a/credentials", headers={TOKEN_HEADER: reg.token})
    assert r.status_code == 401
    assert reg.pws["cust_a"] not in r.text


# ---- 404 for someone else's customer ---------------------------------------

@pytest.mark.parametrize("method,path", PER_CUSTOMER)
@pytest.mark.parametrize("who", ["b_key", "b_service", "legacy_key"])
def test_cross_user_is_404(reg, method, path, who):
    headers = {
        "b_key": _bearer(reg.keys["cust_b"]),
        "b_service": _service(reg.token, "user_b"),
        "legacy_key": _bearer(reg.keys["cust_legacy"]),
    }[who]
    r = reg.client.request(method, path, headers=headers)
    assert r.status_code == 404
    assert r.json()["detail"] == "Customer not found"
    assert reg.pws["cust_a"] not in r.text
    assert _on_disk(reg)["cust_a"]["status"] == "active"


def test_unknown_and_foreign_ids_are_indistinguishable(reg):
    h = _bearer(reg.keys["cust_b"])
    foreign = reg.client.get("/customers/cust_a/status", headers=h)
    missing = reg.client.get("/customers/no_such_customer/status", headers=h)
    assert (foreign.status_code, foreign.json()) == (missing.status_code, missing.json())


def test_owner_can_use_its_own_routes(reg):
    h = _bearer(reg.keys["cust_a"])
    assert reg.client.get("/customers/cust_a2/status", headers=h).status_code == 200
    assert reg.client.post("/customers/cust_a/test-connection", headers=h).status_code == 200
    assert reg.client.delete("/customers/cust_a2", headers=h).status_code == 200
    assert _on_disk(reg)["cust_a2"]["status"] == "inactive"


# ---- GET /customers: own only, never secrets --------------------------------

@pytest.mark.parametrize("who,expected", [
    ("a_key", {"cust_a", "cust_a2"}),
    ("a_service", {"cust_a", "cust_a2"}),
    ("b_service", {"cust_b"}),
    ("legacy_key", {"cust_legacy"}),
    ("nobody_service", set()),
])
def test_list_is_owner_scoped_and_secret_free(reg, who, expected):
    headers = {
        "a_key": _bearer(reg.keys["cust_a"]),
        "a_service": _service(reg.token, "user_a"),
        "b_service": _service(reg.token, "user_b"),
        "legacy_key": _bearer(reg.keys["cust_legacy"]),
        "nobody_service": _service(reg.token, "user_nobody"),
    }[who]
    r = reg.client.get("/customers", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert {c["customer_id"] for c in body["customers"]} == expected
    assert body["total_count"] == len(expected)
    for c in body["customers"]:
        assert not {"connection_string", "password", "api_key"} & set(c)
    for cid in reg.keys:
        assert reg.keys[cid] not in r.text
        assert reg.pws[cid] not in r.text
        assert reg.records[cid]["connection_string"] not in r.text


# ---- /credentials: password only for the service ----------------------------

def test_service_with_token_gets_the_password_for_its_users_customer(reg):
    r = reg.client.get("/customers/cust_a/credentials", headers=_service(reg.token, "user_a"))
    assert r.status_code == 200
    assert r.json()["password"] == reg.pws["cust_a"]
    assert r.json()["table"] == "orders"


@pytest.mark.parametrize("headers_for", [
    lambda reg: _bearer(reg.keys["cust_a"]),                                   # owner, no token
    lambda reg: {**_bearer(reg.keys["cust_a"]), TOKEN_HEADER: "wrong-" + reg.token},
    lambda reg: {TOKEN_HEADER: "wrong-" + reg.token, USER_HEADER: "user_a"},
    lambda reg: {TOKEN_HEADER: reg.token[:-1], USER_HEADER: "user_a"},
])
def test_credentials_without_the_service_token_carry_no_password(reg, headers_for):
    r = reg.client.get("/customers/cust_a/credentials", headers=headers_for(reg))
    assert r.status_code in (401, 403)
    assert "password" not in r.json()
    assert reg.pws["cust_a"] not in r.text


@pytest.mark.parametrize("env", ["unset", "empty"])
@pytest.mark.parametrize("presented", ["", "anything"])
def test_credentials_fail_closed_when_token_is_not_configured(reg, monkeypatch, env, presented):
    if env == "unset":
        monkeypatch.delenv("MCP_SERVICE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("MCP_SERVICE_TOKEN", "")
    for headers in (
        {TOKEN_HEADER: presented, USER_HEADER: "user_a"},
        {**_bearer(reg.keys["cust_a"]), TOKEN_HEADER: presented, USER_HEADER: "user_a"},
    ):
        r = reg.client.get("/customers/cust_a/credentials", headers=headers)
        assert r.status_code in (401, 403)
        assert reg.pws["cust_a"] not in r.text


# ---- register: open, but it can only create ---------------------------------

class _FakeAsyncClient:
    calls = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None, **k):
        _FakeAsyncClient.calls.append({"url": url, "json": json, "headers": headers or {}})
        return SimpleNamespace(status_code=200, text="ok")


@pytest.fixture
def registering(reg, monkeypatch):
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(cdb.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(
        cdb.DatabaseConnectionManager, "test_database_connection",
        staticmethod(lambda conn: {"success": True, "database": "sales", "table_count": 1, "version": "16"}),
    )
    return reg


def _registration(cid):
    return {
        "customer_id": cid, "customer_name": cid, "database_type": "postgresql",
        "host": "db.test.invalid", "port": 5432, "database_name": "sales",
        "username": "alice", "password": secrets.token_urlsafe(12),
        "contact_email": f"{cid}@example.test",
    }


def test_register_never_overwrites_an_existing_customer(registering):
    reg = registering
    # Only on disk (e.g. registered through the MCP server's process), not in memory.
    cdb.customer_manager.customers.pop("cust_b")
    r = reg.client.post("/customers/register", json=_registration("cust_b"))
    assert r.status_code == 409
    assert _FakeAsyncClient.calls == []
    assert _on_disk(reg)["cust_b"]["api_key"] == reg.keys["cust_b"]
    assert _on_disk(reg)["cust_b"]["connection_string"] == reg.records["cust_b"]["connection_string"]


def test_anonymous_register_creates_an_unbound_customer_only_its_key_reaches(registering):
    reg = registering
    r = reg.client.post("/customers/register", json=_registration("cust_new"))
    assert r.status_code == 200
    new_key = r.json()["api_key"]
    assert "owner_user_id" not in _on_disk(reg)["cust_new"]
    # The internal /admin/customers call carries the service token.
    assert _FakeAsyncClient.calls[0]["headers"].get(TOKEN_HEADER) == reg.token

    listed = reg.client.get("/customers", headers=_bearer(new_key)).json()["customers"]
    assert [c["customer_id"] for c in listed] == ["cust_new"]
    assert reg.client.get("/customers/cust_a/status", headers=_bearer(new_key)).status_code == 404


def test_service_register_binds_the_customer_to_the_named_user(registering):
    reg = registering
    r = reg.client.post("/customers/register", json=_registration("cust_a3"),
                        headers=_service(reg.token, "user_a"))
    assert r.status_code == 200
    assert _on_disk(reg)["cust_a3"]["owner_user_id"] == "user_a"
    listed = reg.client.get("/customers", headers=_service(reg.token, "user_a")).json()["customers"]
    assert {c["customer_id"] for c in listed} == {"cust_a", "cust_a2", "cust_a3"}


# ---- binding an unbound customer to its user --------------------------------

def test_service_binds_an_unbound_customer_with_its_own_key(reg):
    h = {**_service(reg.token, "user_c"), **_bearer(reg.keys["cust_legacy"])}
    assert reg.client.post("/customers/cust_legacy/owner", headers=h).status_code == 200
    assert _on_disk(reg)["cust_legacy"]["owner_user_id"] == "user_c"
    r = reg.client.get("/customers/cust_legacy/credentials", headers=_service(reg.token, "user_c"))
    assert r.json()["password"] == reg.pws["cust_legacy"]


@pytest.mark.parametrize("cid,user,key_of", [
    ("cust_a", "user_b", "cust_a"),           # already bound to someone else: never re-bound
    ("cust_legacy", "user_c", "cust_b"),      # wrong customer's key
])
def test_bind_refuses_without_proof_or_over_another_owner(reg, cid, user, key_of):
    before = _on_disk(reg)[cid].get("owner_user_id")
    h = {**_service(reg.token, user), **_bearer(reg.keys[key_of])}
    assert reg.client.post(f"/customers/{cid}/owner", headers=h).status_code == 404
    assert _on_disk(reg)[cid].get("owner_user_id") == before


def test_bind_needs_the_service_token(reg):
    r = reg.client.post("/customers/cust_legacy/owner",
                        headers={**_bearer(reg.keys["cust_legacy"]), USER_HEADER: "user_c"})
    assert r.status_code == 401
    assert "owner_user_id" not in _on_disk(reg)["cust_legacy"]


# ---- /admin/customers on the MCP server -------------------------------------

def test_admin_customers_refuses_without_the_service_token(reg, monkeypatch):
    import app.mcp_server.multi_tenant_mcp_server as mts

    monkeypatch.setattr(mts, "CUSTOMERS_FILE", str(reg.path))
    client = TestClient(mts.app)
    takeover = {
        "customer_id": "cust_a",
        "connection_string": "sqlite:///" + str(reg.path.parent / "attacker.db"),
        "database_type": "sqlite",
        "created_at": "2026-10-18T00:00:00",
        "api_key": "ak_" + secrets.token_hex(16),
    }
    for headers in ({}, {TOKEN_HEADER: "wrong-" + reg.token}):
        r = client.post("/admin/customers", json=takeover, headers=headers)
        assert r.status_code == 401
    assert _on_disk(reg)["cust_a"]["api_key"] == reg.keys["cust_a"]
    assert _on_disk(reg)["cust_a"]["connection_string"] == reg.records["cust_a"]["connection_string"]


@pytest.fixture
def mcp(reg, monkeypatch):
    import app.mcp_server.multi_tenant_mcp_server as mts

    monkeypatch.setattr(mts, "CUSTOMERS_FILE", str(reg.path))
    monkeypatch.setattr(mts, "CUSTOMER_API_KEYS", {})
    monkeypatch.setattr(mts.db_manager, "engines", {})
    monkeypatch.setattr(mts.db_manager, "connection_configs", {})
    dialed = []
    real_create_engine = mts.create_engine

    def _recording_create_engine(url, *a, **k):
        dialed.append(str(url))
        return real_create_engine(url, *a, **k)

    monkeypatch.setattr(mts, "create_engine", _recording_create_engine)
    return SimpleNamespace(mts=mts, client=TestClient(mts.app), dialed=dialed, reg=reg)


def _admin_payload(reg, cid, api_key):
    return {
        "customer_id": cid,
        "connection_string": "sqlite:///" + str(reg.path.parent / f"{cid}.db"),
        "database_type": "sqlite",
        "created_at": "2026-10-18T00:00:00",
        "api_key": api_key,
    }


@pytest.mark.parametrize("env", ["set", "unset", "empty"])
def test_admin_customers_unauthenticated_is_401_and_dials_nothing(mcp, monkeypatch, env):
    if env == "unset":
        monkeypatch.delenv("MCP_SERVICE_TOKEN", raising=False)
    elif env == "empty":
        monkeypatch.setenv("MCP_SERVICE_TOKEN", "")
    for headers in ({}, {TOKEN_HEADER: ""}, {TOKEN_HEADER: "wrong-" + mcp.reg.token}):
        r = mcp.client.post("/admin/customers", headers=headers,
                            json=_admin_payload(mcp.reg, "cust_new", "ak_" + secrets.token_hex(16)))
        assert r.status_code == 401
    assert mcp.dialed == []
    assert "cust_new" not in _on_disk(mcp.reg)


def test_admin_customers_legitimate_onboarding_call_registers(mcp):
    key = "ak_" + secrets.token_hex(16)
    r = mcp.client.post("/admin/customers", headers={TOKEN_HEADER: mcp.reg.token},
                        json=_admin_payload(mcp.reg, "cust_new", key))
    assert r.status_code == 200, r.text
    assert _on_disk(mcp.reg)["cust_new"]["api_key"] == key
    assert mcp.mts.CUSTOMER_API_KEYS["cust_new"].api_key == key


def test_admin_customers_resync_with_the_same_key_is_allowed(mcp):
    payload = _admin_payload(mcp.reg, "cust_a", mcp.reg.keys["cust_a"])
    r = mcp.client.post("/admin/customers", headers={TOKEN_HEADER: mcp.reg.token}, json=payload)
    assert r.status_code == 200, r.text


def test_admin_customers_refuses_an_overwrite_even_with_the_token(mcp):
    attacker_key = "ak_" + secrets.token_hex(16)
    r = mcp.client.post("/admin/customers", headers={TOKEN_HEADER: mcp.reg.token},
                        json=_admin_payload(mcp.reg, "cust_a", attacker_key))
    assert r.status_code == 409
    assert mcp.dialed == []                       # refused before any connection attempt
    assert _on_disk(mcp.reg)["cust_a"]["api_key"] == mcp.reg.keys["cust_a"]
    assert _on_disk(mcp.reg)["cust_a"]["connection_string"] == mcp.reg.records["cust_a"]["connection_string"]
    assert "cust_a" not in mcp.mts.CUSTOMER_API_KEYS


def _connect_payload(reg, name):
    return {"connection_name": name, "database_type": "sqlite",
            "database_name": str(reg.path.parent / f"{name}.db")}


def test_connect_and_connections_need_the_service_token(mcp):
    r = mcp.client.post("/connect", json=_connect_payload(mcp.reg, "conn_x"))
    assert r.status_code == 401
    assert mcp.client.get("/connections").status_code == 401
    assert mcp.dialed == []
    assert "conn_x" not in mcp.mts.CUSTOMER_API_KEYS


def test_connect_and_connections_work_for_the_service(mcp):
    r = mcp.client.post("/connect", headers={TOKEN_HEADER: mcp.reg.token},
                        json=_connect_payload(mcp.reg, "conn_x"))
    assert r.status_code == 200, r.text
    assert r.json()["api_key"] == mcp.mts.CUSTOMER_API_KEYS["conn_x"].api_key
    listed = mcp.client.get("/connections", headers={TOKEN_HEADER: mcp.reg.token})
    assert listed.status_code == 200
    assert [c["customer_id"] for c in listed.json()["connections"]] == ["conn_x"]


@pytest.mark.parametrize("env", ["unset", "empty"])
def test_connect_and_connections_fail_closed_without_a_configured_token(mcp, monkeypatch, env):
    if env == "unset":
        monkeypatch.delenv("MCP_SERVICE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("MCP_SERVICE_TOKEN", "")
    r = mcp.client.post("/connect", headers={TOKEN_HEADER: ""}, json=_connect_payload(mcp.reg, "conn_x"))
    assert r.status_code == 401
    assert mcp.client.get("/connections", headers={TOKEN_HEADER: ""}).status_code == 401
    assert mcp.dialed == []


def test_connect_refuses_to_replace_an_existing_customer(mcp):
    mcp.mts.CUSTOMER_API_KEYS["cust_a"] = mcp.mts.CustomerAPI(
        customer_id="cust_a", api_key=mcp.reg.keys["cust_a"], created_at="2026-10-18T00:00:00")
    r = mcp.client.post("/connect", headers={TOKEN_HEADER: mcp.reg.token},
                        json=_connect_payload(mcp.reg, "cust_a"))
    assert r.status_code == 409
    assert mcp.dialed == []
    assert mcp.mts.CUSTOMER_API_KEYS["cust_a"].api_key == mcp.reg.keys["cust_a"]


# ---- planner DTA auto-fetch, end to end against the onboarding router -------

@pytest.fixture
def planner_against_onboarding(reg, monkeypatch):
    import app.agents.planner as planner_mod

    base = planner_mod.ONBOARDING_API_URL
    seen = []

    class _RoutedSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None, **_kw):
            assert url.startswith(base)
            seen.append(dict(headers or {}))
            return reg.client.get(url[len(base):], headers=headers or {})

    monkeypatch.setattr(planner_mod.requests, "Session", _RoutedSession)

    captured = {}

    def _pipeline(*a, **k):
        captured.update(k)
        raise RuntimeError("pipeline stubbed out in tests")

    fake_dta = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")
    fake_dta.data_transfer_pipeline = _pipeline
    monkeypatch.setitem(sys.modules, "app.agents.data_transfer_agent.data_transfer_agent", fake_dta)
    return SimpleNamespace(planner=planner_mod, captured=captured, seen=seen, reg=reg)


def _transfer(p, user_id, source="cust_a"):
    state = {
        "user_id": user_id,
        "dta_database_registry": {
            "Warehouse": {"db_type": "mysql", "host": "h", "port": 3306, "database": "d",
                          "user": "u", "password": "p", "table": "t"},
        },
    }
    msg = p.planner._handle_initiate_transfer(
        state,
        p.planner.InitiateTransferParams(
            source_alias=source, destination_alias="Warehouse",
            user_prompt=f"transfer from {source} to Warehouse", write_mode="append",
            dest_table="orders_copy",
        ),
    )
    return state, msg


def test_planner_autofetch_gets_the_password_with_the_token(planner_against_onboarding):
    p = planner_against_onboarding
    state, _msg = _transfer(p, "user_a")
    assert state["dta_database_registry"]["cust_a"]["db_type"] == "postgresql"
    assert p.captured["source_credentials"]["password"] == p.reg.pws["cust_a"]
    assert all(h.get(TOKEN_HEADER) == p.reg.token and h.get(USER_HEADER) == "user_a" for h in p.seen)


def test_planner_cannot_fetch_another_users_customer(planner_against_onboarding):
    p = planner_against_onboarding
    state, msg = _transfer(p, "user_b")
    assert "not registered" in msg or "not found" in msg
    assert "cust_a" not in state["dta_database_registry"]
    assert p.captured == {}


def test_planner_without_the_token_gets_no_password(planner_against_onboarding, monkeypatch):
    p = planner_against_onboarding
    monkeypatch.delenv("MCP_SERVICE_TOKEN", raising=False)
    state, msg = _transfer(p, "user_a")
    assert "not registered" in msg or "not found" in msg
    assert "cust_a" not in state["dta_database_registry"]
    assert p.captured == {}
    assert all(TOKEN_HEADER not in h for h in p.seen)


def test_planner_cannot_resolve_a_deleted_customer(planner_against_onboarding):
    p = planner_against_onboarding
    a = _service(p.reg.token, "user_a")
    assert p.reg.client.delete("/customers/cust_a", headers=a).status_code == 200
    assert p.reg.client.get("/customers/cust_a/credentials", headers=a).status_code == 404
    state, msg = _transfer(p, "user_a")
    assert "not registered" in msg or "not found" in msg
    assert "cust_a" not in state["dta_database_registry"]
    assert p.captured == {}


def test_planner_list_databases_shows_only_the_users_customers(planner_against_onboarding):
    p = planner_against_onboarding
    out = p.planner._handle_list_databases({"user_id": "user_a", "dta_database_registry": {}})
    assert "cust_a" in out and "cust_a2" in out
    assert "cust_b" not in out and "cust_legacy" not in out


# ---- a deleted customer's key dies everywhere, at once ----------------------

class _RoutedAsyncClient:
    """httpx.AsyncClient stand-in that sends customer_dbs' MCP calls to the MCP app."""
    mcp_client = None
    base = ""

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None, **k):
        return self.mcp_client.post(url[len(self.base):], json=json, headers=headers or {})

    async def delete(self, url, headers=None, **k):
        return self.mcp_client.delete(url[len(self.base):], headers=headers or {})


@pytest.fixture
def chain(mcp, monkeypatch):
    """Onboarding router + MCP server sharing one registry, as in the chart."""
    _RoutedAsyncClient.mcp_client = mcp.client
    _RoutedAsyncClient.base = cdb.customer_manager.mcp_server_url
    monkeypatch.setattr(cdb.httpx, "AsyncClient", _RoutedAsyncClient)
    monkeypatch.setattr(
        cdb.DatabaseConnectionManager, "test_database_connection",
        staticmethod(lambda conn: {"success": True, "database": "main", "table_count": 0, "version": "3"}),
    )
    return mcp


def _register_sqlite(chain, cid, headers=None):
    body = {
        "customer_id": cid, "customer_name": cid, "database_type": "sqlite",
        "database_name": str(chain.reg.path.parent / f"{cid}.db"),
        "contact_email": f"{cid}@example.test",
    }
    return chain.reg.client.post("/customers/register", json=body, headers=headers or {})


def _call_tool(chain, key):
    return chain.client.post("/call_tool", headers=_bearer(key),
                             json={"params": {"name": "list_tables", "arguments": {}}})


def test_deleted_key_is_401_on_call_tool_and_evicted(chain):
    key = _register_sqlite(chain, "cust_z").json()["api_key"]
    assert _call_tool(chain, key).status_code == 200
    assert "cust_z" in chain.mts.db_manager.engines

    assert chain.reg.client.delete("/customers/cust_z", headers=_bearer(key)).status_code == 200
    assert "cust_z" not in chain.mts.CUSTOMER_API_KEYS
    assert "cust_z" not in chain.mts.db_manager.engines
    assert _call_tool(chain, key).status_code == 401


def test_per_call_check_blocks_the_key_when_eviction_is_skipped(chain, monkeypatch):
    async def _no_eviction(customer_id):
        return None

    monkeypatch.setattr(cdb, "_evict_from_mcp", _no_eviction)
    key = _register_sqlite(chain, "cust_z").json()["api_key"]
    assert chain.reg.client.delete("/customers/cust_z", headers=_bearer(key)).status_code == 200
    assert chain.mts.CUSTOMER_API_KEYS["cust_z"].api_key == key     # still in memory
    assert _call_tool(chain, key).status_code == 401


def test_reregistering_issues_a_new_key_and_the_old_key_stays_dead(chain):
    a = _service(chain.reg.token, "user_a")
    old = _register_sqlite(chain, "cust_z", headers=a).json()["api_key"]
    assert chain.reg.client.delete("/customers/cust_z", headers=a).status_code == 200

    r = _register_sqlite(chain, "cust_z", headers=a)
    assert r.status_code == 200, r.text
    new = r.json()["api_key"]
    assert new != old
    assert _call_tool(chain, new).status_code == 200
    assert _call_tool(chain, old).status_code == 401
    assert chain.reg.client.get("/customers", headers=_bearer(old)).status_code == 401
    listed = chain.reg.client.get("/customers", headers=_bearer(new)).json()["customers"]
    assert {c["customer_id"] for c in listed} == {"cust_a", "cust_a2", "cust_z"}      # user_a's group


def _registered_then_deleted(chain, headers):
    r = _register_sqlite(chain, "cust_z", headers=headers)
    assert r.status_code == 200, r.text
    delete_as = headers or _bearer(r.json()["api_key"])
    assert chain.reg.client.delete("/customers/cust_z", headers=delete_as).status_code == 200


def test_reregistering_a_deleted_unbound_id_is_409_retired(chain):
    _registered_then_deleted(chain, None)
    for headers in (None, _service(chain.reg.token, "user_a")):
        r = _register_sqlite(chain, "cust_z", headers=headers)
        assert r.status_code == 409
        assert "was deleted; choose a new id" in r.json()["detail"]
    assert _on_disk(chain.reg)["cust_z"]["status"] == "inactive"


def test_the_owner_can_reregister_their_own_deleted_bound_id(chain):
    a = _service(chain.reg.token, "user_a")
    _registered_then_deleted(chain, a)
    r = _register_sqlite(chain, "cust_z", headers=a)
    assert r.status_code == 200, r.text
    assert _on_disk(chain.reg)["cust_z"]["owner_user_id"] == "user_a"


@pytest.mark.parametrize("who", ["anonymous", "user_b"])
def test_nobody_else_can_reregister_a_deleted_bound_id(chain, who):
    _registered_then_deleted(chain, _service(chain.reg.token, "user_a"))
    headers = None if who == "anonymous" else _service(chain.reg.token, "user_b")
    assert _register_sqlite(chain, "cust_z", headers=headers).status_code == 409
    assert _on_disk(chain.reg)["cust_z"]["status"] == "inactive"
    assert _on_disk(chain.reg)["cust_z"]["owner_user_id"] == "user_a"


def test_admin_customers_refuses_a_retired_unbound_id_even_with_the_token(chain):
    _registered_then_deleted(chain, None)
    r = chain.client.post("/admin/customers", headers={TOKEN_HEADER: chain.reg.token},
                          json=_admin_payload(chain.reg, "cust_z", "ak_" + secrets.token_hex(16)))
    assert r.status_code == 409
    assert _on_disk(chain.reg)["cust_z"]["status"] == "inactive"


def test_bind_refuses_a_deleted_customer_even_with_its_old_key(reg):
    old = reg.keys["cust_legacy"]
    assert reg.client.delete("/customers/cust_legacy", headers=_bearer(old)).status_code == 200
    h = {**_service(reg.token, "user_c"), **_bearer(old)}
    assert reg.client.post("/customers/cust_legacy/owner", headers=h).status_code == 404
    assert "owner_user_id" not in _on_disk(reg)["cust_legacy"]


def test_an_active_id_is_still_never_replaced(chain):
    assert _register_sqlite(chain, "cust_z").status_code == 200
    assert _register_sqlite(chain, "cust_z").status_code == 409


def test_reload_drops_customers_that_are_no_longer_active(mcp):
    records = _on_disk(mcp.reg)
    records["cust_a"]["status"] = "inactive"
    mcp.reg.path.write_text(json.dumps(records))
    for cid in ("cust_a", "cust_gone"):
        mcp.mts.CUSTOMER_API_KEYS[cid] = mcp.mts.CustomerAPI(
            customer_id=cid, api_key="ak_" + secrets.token_hex(16), created_at="2026-10-18T00:00:00")
    mcp.mts.load_registered_customers()
    assert "cust_a" not in mcp.mts.CUSTOMER_API_KEYS
    assert "cust_gone" not in mcp.mts.CUSTOMER_API_KEYS


def test_evict_route_is_service_only(mcp):
    mcp.mts.CUSTOMER_API_KEYS["cust_a"] = mcp.mts.CustomerAPI(
        customer_id="cust_a", api_key=mcp.reg.keys["cust_a"], created_at="2026-10-18T00:00:00")
    assert mcp.client.delete("/admin/customers/cust_a").status_code == 401
    assert "cust_a" in mcp.mts.CUSTOMER_API_KEYS
    r = mcp.client.delete("/admin/customers/cust_a", headers={TOKEN_HEADER: mcp.reg.token})
    assert r.status_code == 200 and r.json()["evicted"] is True
    assert "cust_a" not in mcp.mts.CUSTOMER_API_KEYS


def test_delete_from_a_separate_onboarding_process_calls_the_evict_route(reg, monkeypatch):
    monkeypatch.delitem(sys.modules, "app.mcp_server.multi_tenant_mcp_server", raising=False)
    calls = []

    class _Recorder(_FakeAsyncClient):
        async def delete(self, url, headers=None, **k):
            calls.append({"url": url, "headers": headers or {}})
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(cdb.httpx, "AsyncClient", _Recorder)
    assert reg.client.delete("/customers/cust_a2", headers=_bearer(reg.keys["cust_a"])).status_code == 200
    assert len(calls) == 1
    assert calls[0]["url"] == f"{cdb.customer_manager.mcp_server_url}/admin/customers/cust_a2"
    assert calls[0]["headers"].get(TOKEN_HEADER) == reg.token


# ---- option (i): a customer bound to nobody is reachable through the service --
#
# Customers registered before owners existed are bound to nobody. The API
# service reaches them for any signed-in user it names, on the routes a
# transfer needs, until the first /api/database/connect that proves the key
# binds them. Nothing else about them opens up: no other key, no delete, no
# listing, no password to anyone but the service.

UNBOUND_ROUTES = [
    ("GET", "/customers/cust_legacy/status"),
    ("POST", "/customers/cust_legacy/test-connection"),
    ("GET", "/customers/cust_legacy/credentials"),
]


@pytest.mark.parametrize("method,path", UNBOUND_ROUTES)
@pytest.mark.parametrize("user", ["user_a", "user_b", "user_new"])
def test_service_reaches_an_unbound_customer_for_any_signed_in_user(reg, method, path, user):
    r = reg.client.request(method, path, headers=_service(reg.token, user))
    assert r.status_code == 200, r.text
    if path.endswith("/credentials"):
        assert r.json()["password"] == reg.pws["cust_legacy"]
        assert r.json()["database_type"] == "postgresql"


@pytest.mark.parametrize("method,path", UNBOUND_ROUTES)
@pytest.mark.parametrize("who", ["a_key", "b_key"])
def test_an_unbound_customer_stays_closed_to_other_customers_keys(reg, method, path, who):
    key = {"a_key": reg.keys["cust_a"], "b_key": reg.keys["cust_b"]}[who]
    r = reg.client.request(method, path, headers=_bearer(key))
    assert r.status_code == 404
    assert reg.pws["cust_legacy"] not in r.text


def test_the_service_cannot_delete_or_list_an_unbound_customer(reg):
    h = _service(reg.token, "user_b")
    assert reg.client.delete("/customers/cust_legacy", headers=h).status_code == 404
    assert _on_disk(reg)["cust_legacy"]["status"] == "active"
    listed = reg.client.get("/customers", headers=h).json()["customers"]
    assert "cust_legacy" not in {c["customer_id"] for c in listed}


def test_an_unbound_customer_without_the_token_is_still_401(reg, monkeypatch):
    r = reg.client.get("/customers/cust_legacy/credentials", headers={USER_HEADER: "user_b"})
    assert r.status_code == 401
    monkeypatch.delenv("MCP_SERVICE_TOKEN")
    r = reg.client.get("/customers/cust_legacy/credentials", headers=_service(reg.token, "user_b"))
    assert r.status_code == 401
    assert reg.pws["cust_legacy"] not in r.text


def test_binding_closes_an_unbound_customer_to_everyone_else(reg):
    b = _service(reg.token, "user_b")
    assert reg.client.get("/customers/cust_legacy/credentials", headers=b).status_code == 200
    bind = reg.client.post(
        "/customers/cust_legacy/owner",
        headers={**_service(reg.token, "user_a"), **_bearer(reg.keys["cust_legacy"])},
    )
    assert bind.status_code == 200
    assert _on_disk(reg)["cust_legacy"]["owner_user_id"] == "user_a"
    r = reg.client.get("/customers/cust_legacy/credentials", headers=b)
    assert r.status_code == 404
    assert reg.pws["cust_legacy"] not in r.text
    r = reg.client.get("/customers/cust_legacy/credentials", headers=_service(reg.token, "user_a"))
    assert r.json()["password"] == reg.pws["cust_legacy"]


def _transfer_between(p, user_id, source, destination):
    state = {"user_id": user_id, "dta_database_registry": {}}
    msg = p.planner._handle_initiate_transfer(
        state,
        p.planner.InitiateTransferParams(
            source_alias=source, destination_alias=destination,
            user_prompt=f"transfer from {source} to {destination}", write_mode="append",
            dest_table="orders_copy", create_if_missing=True,
        ),
    )
    return state, msg


@pytest.mark.parametrize("user", ["user_a", "user_b"])
def test_planner_transfers_from_an_unbound_customer_for_any_user(planner_against_onboarding, user):
    p = planner_against_onboarding
    state, _msg = _transfer(p, user, source="cust_legacy")
    entry = state["dta_database_registry"]["cust_legacy"]
    # Typed from /credentials: /customers does not list a customer bound to nobody.
    assert entry["db_type"] == "postgresql"
    assert p.captured["source_type"] == "postgresql"
    assert p.captured["source_credentials"]["password"] == p.reg.pws["cust_legacy"]


@pytest.mark.parametrize("source", ["cust_a", "cust_legacy"])
def test_planner_transfers_into_an_unbound_destination(planner_against_onboarding, monkeypatch, source):
    p = planner_against_onboarding
    monkeypatch.setattr(p.planner, "_list_dest_tables", lambda creds, db_type: None)
    state, _msg = _transfer_between(p, "user_a", source, "cust_legacy")
    assert state["dta_database_registry"]["cust_legacy"]["db_type"] == "postgresql"
    assert p.captured["destination_type"] == "postgresql"
    assert p.captured["destination_credentials"]["password"] == p.reg.pws["cust_legacy"]
    assert p.captured["source_credentials"]["password"] == p.reg.pws[source]


def test_planner_still_cannot_use_another_users_bound_customer_as_destination(planner_against_onboarding):
    p = planner_against_onboarding
    state, msg = _transfer_between(p, "user_b", "cust_legacy", "cust_a")
    assert "cust_a" not in state["dta_database_registry"]
    assert p.captured == {}
