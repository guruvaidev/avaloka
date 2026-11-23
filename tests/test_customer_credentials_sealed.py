"""Customer database credentials: sealed at rest in customers.json, sealed in transit to MCP.

1.6's tests for its port of app/mcp_server/credentials.py (88120343), adapted
to the 1.7 line they now run on: /credentials answers a service-class caller
with a sealed ``credential_handle`` opened in process, never a password; the
MCP server's admin routes take the onboarding service's signed request, not
the static token; a plaintext registry is re-sealed when the service loads it
at start-up. Before the fix a registration wrote the full connection string --
password included -- and the api_key into customers.json in plaintext, and
POSTed both to the MCP server's /admin/customers in plaintext.

Every key, token and password here is generated per test; none is real.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote_plus

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.cloud_connections as cc
import app.mcp_server.customer_dbs as cdb
from app.mcp_server.credentials import credential_handle, open_credential_handle, service_headers

TOKEN_HEADER = "X-Avaloka-Service-Token"
USER_HEADER = "X-Avaloka-User"
ROOT = Path(__file__).resolve().parents[1]


def _probe(tag):
    return f"PROBE-{tag}-" + secrets.token_hex(8)


@pytest.fixture
def key(monkeypatch):
    k = "test-only-" + secrets.token_hex(24)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", k, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    return k


class _FakeResult:
    def fetchone(self):
        return ("orders", "sales")


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        return _FakeResult()


class _FakeEngine:
    def __init__(self, url=None):
        self.url = url

    def connect(self):
        return _FakeConn()

    def dispose(self):
        pass


class _RecordingAsyncClient:
    """httpx.AsyncClient for the onboarding service's call to MCP: records every body."""

    bodies = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, content=None, **k):
        import json as _json
        type(self).bodies.append(_json.dumps(json) if json is not None else (content or b"").decode())
        return SimpleNamespace(status_code=200, text="ok")


def _fake_connection_test(monkeypatch):
    """Patch the connection test on the CLASS, as tests/test_customer_dbs_auth.py does.

    An instance-level monkeypatch is undone by writing the bound method back onto
    the instance, which leaves an instance attribute that shadows every later
    class-level patch (the registration then runs the real connection test and
    fails). Such a leftover is pollution, not state: it is dropped, not restored.
    """
    vars(cdb.customer_manager.db_manager).pop("test_database_connection", None)
    monkeypatch.setattr(
        cdb.DatabaseConnectionManager, "test_database_connection",
        staticmethod(lambda conn: {"success": True, "database": "sales", "table_count": 1, "version": "16"}),
    )


@pytest.fixture
def onboarding(tmp_path, monkeypatch):
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("MCP_SERVICE_TOKEN", token)
    path = tmp_path / "customers.json"
    path.write_text("{}")
    mgr = cdb.customer_manager
    monkeypatch.setattr(mgr, "customers_file", str(path))
    monkeypatch.setattr(mgr, "customers", {})
    _RecordingAsyncClient.bodies = []
    monkeypatch.setattr(cdb.httpx, "AsyncClient", _RecordingAsyncClient)
    monkeypatch.setattr(cdb, "create_engine", lambda *a, **k: _FakeEngine())
    _fake_connection_test(monkeypatch)
    app = FastAPI()
    app.include_router(cdb.router)
    return SimpleNamespace(client=TestClient(app), token=token, path=path)


def _register(onb, password, cid="sealed_cust", user="user_a"):
    return onb.client.post("/customers/register", headers={TOKEN_HEADER: onb.token, USER_HEADER: user}, json={
        "customer_id": cid, "customer_name": cid, "database_type": "postgresql",
        "host": "db.test.invalid", "port": 5432, "database_name": "sales",
        "username": "alice", "password": password, "contact_email": "a@example.test",
    })


def _credentials(onb, cid="sealed_cust", user="user_a"):
    return onb.client.get(f"/customers/{cid}/credentials", headers={TOKEN_HEADER: onb.token, USER_HEADER: user})


def _opened(resp):
    """What a /credentials handle opens to, in process; the body itself carries no password."""
    body = resp.json()
    assert "password" not in body, "the credentials route handed out a password"
    opened = open_credential_handle(body.get("credential_handle"))
    assert opened is not None, "the credentials route returned no handle this key can open"
    return opened


# ── at rest ─────────────────────────────────────────────────────────────────

def test_a_registration_stores_no_password_or_api_key_on_disk(onboarding, key):
    pw = _probe("rest")
    r = _register(onboarding, pw)
    assert r.status_code == 200, r.status_code
    api_key = r.json()["api_key"]
    on_disk = onboarding.path.read_text()
    assert pw not in on_disk and quote_plus(pw) not in on_disk, "the password is in customers.json in plaintext"
    assert api_key not in on_disk, "the api_key is in customers.json in plaintext"
    record = json.loads(on_disk)["sealed_cust"]
    assert "connection_string" not in record and "api_key" not in record
    assert record.get("connection_string_enc", {}).get("ciphertext"), "no sealed connection string on disk"
    assert record.get("api_key_enc", {}).get("ciphertext"), "no sealed api_key on disk"
    # The service still gets a working credential back: a handle it opens in process.
    creds = _credentials(onboarding)
    assert creds.status_code == 200 and pw not in creds.text
    assert _opened(creds)["password"] == pw


def test_a_plaintext_registry_is_resealed_when_the_service_loads_it(onboarding, key):
    """A registry written before sealing is re-sealed by the service's start-up load
    (customer_manager.load_customers -> credentials.migrate_registry), and keeps working."""
    pw, api_key = _probe("legacy"), "ak_" + secrets.token_hex(16)
    legacy = {"legacy_cust": {
        "customer_id": "legacy_cust", "customer_name": "legacy", "database_type": "postgresql",
        "connection_string": f"postgresql://alice:{quote_plus(pw)}@db.test.invalid:5432/sales",
        "api_key": api_key, "status": "active", "owner_user_id": "user_a",
        "created_at": "2026-10-01T00:00:00",
    }}
    onboarding.path.write_text(json.dumps(legacy))
    cdb.customer_manager.load_customers()          # what the onboarding service does at start-up
    creds = _credentials(onboarding, "legacy_cust")
    assert creds.status_code == 200, "the legacy record stopped working"
    assert _opened(creds)["password"] == pw
    on_disk = onboarding.path.read_text()
    assert pw not in on_disk and quote_plus(pw) not in on_disk and api_key not in on_disk, \
        "a plaintext record read with the key configured was not re-sealed"
    assert json.loads(on_disk)["legacy_cust"].get("connection_string_enc")


def test_without_a_key_a_registration_is_refused_and_nothing_is_written(onboarding, monkeypatch):
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", None, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None, raising=False)
    pw = _probe("nokey")
    r = _register(onboarding, pw)
    assert r.status_code >= 500, "a registration was accepted with no key to seal it"
    assert pw not in onboarding.path.read_text(), "the password reached disk without a key"
    assert not any(pw in b for b in _RecordingAsyncClient.bodies), "the password went to MCP without a key"


# ── in transit to MCP: tests/test_mcp_handoff_sealed.py (the body carries
# connection_string_enc / api_key_enc and never the password or api_key) ─────

# ── the MCP server side ─────────────────────────────────────────────────────

@pytest.fixture
def mcp(tmp_path, monkeypatch, key):
    import app.mcp_server.multi_tenant_mcp_server as mts
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("MCP_SERVICE_TOKEN", token)
    path = tmp_path / "mcp-customers.json"
    path.write_text("{}")
    monkeypatch.setattr(mts, "CUSTOMERS_FILE", str(path))
    mts._invalidate_registry_cache()
    engines = []
    monkeypatch.setattr(mts, "create_engine", lambda url, *a, **k: engines.append(str(url)) or _FakeEngine(url))
    return SimpleNamespace(mts=mts, client=TestClient(mts.app), token=token, path=path, engines=engines)


def _admin_post(client, payload):
    """As the onboarding service calls /admin/customers: the body signed with the deployment key."""
    body = json.dumps(payload).encode("utf-8")
    return client.post("/admin/customers", content=body,
                       headers={"Content-Type": "application/json", **service_headers("POST", "/admin/customers", body)})


def test_the_mcp_server_opens_a_sealed_registration_and_keeps_it_sealed(mcp):
    pw, api_key = _probe("mcp"), "ak_" + secrets.token_hex(16)
    dsn = f"postgresql://alice:{quote_plus(pw)}@db.test.invalid:5432/sales"

    r = _admin_post(mcp.client, {
        "customer_id": "mcp_cust", "connection_string_enc": credential_handle(dsn), "database_type": "postgresql",
        "api_key_enc": credential_handle(api_key), "created_at": "2026-10-31T00:00:00"})
    try:
        assert r.status_code == 200, r.status_code
        assert dsn in mcp.engines, "the MCP server did not open the sealed connection string"
        on_disk = mcp.path.read_text()
        assert pw not in on_disk and quote_plus(pw) not in on_disk and api_key not in on_disk, \
            "the MCP server wrote a secret to disk in plaintext"
        # A body with the secrets in the clear is no longer accepted, signed or not.
        plain = _admin_post(mcp.client, {
            "customer_id": "mcp_cust2", "connection_string": dsn, "database_type": "postgresql",
            "api_key": api_key, "created_at": "2026-10-31T00:00:00"})
        assert plain.status_code == 422, "the MCP admin route still takes a plaintext connection string"
        assert "mcp_cust2" not in mcp.path.read_text()
    finally:
        mcp.mts._evict_customer("mcp_cust")


def test_the_mcp_server_reseals_a_plaintext_registry_on_load(mcp):
    pw, api_key = _probe("mcpload"), "ak_" + secrets.token_hex(16)
    mcp.path.write_text(json.dumps({"old_cust": {
        "customer_id": "old_cust", "database_type": "postgresql", "api_key": api_key, "status": "active",
        "connection_string": f"postgresql://alice:{quote_plus(pw)}@db.test.invalid:5432/sales",
        "created_at": "2026-10-01T00:00:00"}}))
    try:
        assert mcp.mts.load_registered_customers() == 1, "the legacy record was not loaded"
        on_disk = mcp.path.read_text()
        assert pw not in on_disk and api_key not in on_disk, "a plaintext registry was not re-sealed on load"
    finally:
        mcp.mts._evict_customer("old_cust")


# ── UI-B12 (release suite, test/1.6-release-suite): the copy and the storage agree ─
#
# tests/release_1_6/test_uib12_privacy_claim.py pins the mismatch with a strict
# xfail: the UI says "Credentials are encrypted server-side" while the password
# sat in customers.json in plaintext. Here is the same check, as a passing test.

class _FakeMCPPost:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        return SimpleNamespace(status_code=200, text="ok")


def test_uib12_encrypted_credentials_claim_matches_storage(tmp_path, monkeypatch, key):
    """The same registration the UI-B12 suite makes (the suite's fakes; the API service
    registering for its user, as 1.7 requires a user), with the suite's assertion: the
    claim is shipped, the password is not on disk."""
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("MCP_SERVICE_TOKEN", token)
    path = tmp_path / "customers.json"
    path.write_text("{}")
    mgr = cdb.customer_manager
    monkeypatch.setattr(mgr, "customers_file", str(path))
    monkeypatch.setattr(mgr, "customers", {})
    monkeypatch.setattr(cdb.httpx, "AsyncClient", _FakeMCPPost)
    _fake_connection_test(monkeypatch)
    app = FastAPI()
    app.include_router(cdb.router)
    password = "pw-" + secrets.token_hex(12)
    resp = TestClient(app).post("/customers/register", headers={TOKEN_HEADER: token, USER_HEADER: "user-1"}, json={
        "customer_id": "uib12_cust", "customer_name": "uib12", "database_type": "postgresql",
        "host": "db.test.invalid", "port": 5432, "database_name": "sales",
        "username": "user-1", "password": password, "contact_email": "user-1@example.com",
    })
    assert resp.status_code == 200, resp.status_code
    on_disk = path.read_text()
    assert "uib12_cust" in json.loads(on_disk)
    db_modal = (ROOT / "ui/src/components/database/DatabaseConnectModal.tsx").read_text()
    footer = (ROOT / "ui/src/components/dashboard/SecurityFooter.tsx").read_text()
    claim_shipped = "Credentials are encrypted server-side" in db_modal or "Your data is encrypted" in footer
    assert not (claim_shipped and password in on_disk), (
        "copy claims credentials are encrypted while the password sits in plaintext on disk"
    )
    # And whatever the copy says, the password is not on disk.
    assert password not in on_disk, "the password sits in plaintext in customers.json"
