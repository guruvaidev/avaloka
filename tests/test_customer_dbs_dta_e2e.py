"""Security hotfix, end to end: a DB-to-DB transfer against a real Postgres.

The /customers routes need an identity, and /credentials releases only a
handle sealed under DB_ENCRYPTION_KEY, never a password, and only to a
service-class caller: the signed-in user's avu1 assertion for a database that
user owns, or (#618, for a legacy database bound to nobody) the service token.
A customer's api_key is 403. (Merge of develop-1.7 into integration/2.0,
decisions 1 and 3a.) These tests drive the planner's transfer path through the
real onboarding router to real databases and count the rows that land:

  * owned source  -> owned destination    copies with no service token at all
                                          (both ends through the assertion);
  * owned source  -> unbound destination  copies (assertion, then the token);
  * unbound       -> unbound              copies (customers registered before
                                          owners existed keep working);
  * user_b naming user_a's owned source   is refused, and 0 rows are written;
  * no service token, unbound source      is refused, and 0 rows are written;
  * a deleted customer                    is refused, and 0 rows are written;
  * a customer's api_key on /credentials  is 403 and carries no secret.

Every database, role and password here is created per run and dropped after.
Only the code generator and the runner are replaced: by a copier that reads the
source with exactly the credentials the planner handed the pipeline and writes
the destination with exactly the destination's. Those credentials are opened
from the handles customer_dbs released, so a database the onboarding API did
not release to this user cannot be used.

Needs ``AVALOKA_TEST_PG_ADMIN_URL``: a Postgres role that may CREATE DATABASE
and CREATE ROLE, e.g. ``postgresql://avaloka:<password>@localhost:5432/postgres``.
Without it the test skips on a laptop, and FAILS when CI is set: a skipped
security test is a green line that tested nothing.
"""
import hashlib
import json
import os
import secrets
import sys
import types
from types import SimpleNamespace
from urllib.parse import quote_plus, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Real Postgres: CI runs this in its integration line (bitbucket-pipelines.yml),
# next to the other Postgres suites. A hermetic run deselects it, so a gate with
# no database does not collect it and error.
pytestmark = pytest.mark.integration

ADMIN_URL = os.environ.get("AVALOKA_TEST_PG_ADMIN_URL", "").strip()
ROWS = 37


def _admin_url() -> str:
    if ADMIN_URL:
        return ADMIN_URL
    message = "AVALOKA_TEST_PG_ADMIN_URL is not set; the transfer was not exercised against Postgres"
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
        pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
    pytest.skip(message)


def _url(user, password, host, port, database):
    return f"postgresql://{user}:{quote_plus(password)}@{host}:{port}/{database}"


def _engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, isolation_level="AUTOCOMMIT", pool_pre_ping=True)


def _count(url, table):
    from sqlalchemy import text
    eng = _engine(url)
    try:
        with eng.connect() as conn:
            return conn.execute(text(f'SELECT count(*) FROM "{table}"')).scalar_one()
    finally:
        eng.dispose()


@pytest.fixture(scope="module")
def pg():
    """Four throwaway databases, each owned by its own login role."""
    from sqlalchemy import text

    admin = urlsplit(_admin_url())
    host, port = admin.hostname, admin.port or 5432
    run = secrets.token_hex(4)
    names = ("src_a", "src_legacy", "dst_a", "dst_legacy", "dst_legacy2")
    dbs = {}
    eng = _engine(_admin_url())
    try:
        with eng.connect() as conn:
            for n in names:
                db, role, pw = f"e2e_{n}_{run}", f"e2e_{n}_{run}", secrets.token_urlsafe(18)
                # token_urlsafe is [A-Za-z0-9_-] only, so it is safe as a literal;
                # CREATE ROLE takes no bind parameters.
                conn.execute(text(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{pw}'"))
                conn.execute(text(f'CREATE DATABASE "{db}" OWNER "{role}"'))
                dbs[n] = SimpleNamespace(db=db, role=role, pw=pw, url=_url(role, pw, host, port, db))
        for n, d in dbs.items():
            e = _engine(d.url)
            with e.connect() as conn:
                if n.startswith("src"):
                    conn.execute(text("CREATE TABLE orders (id integer PRIMARY KEY, amount numeric(10,2))"))
                    conn.execute(text(
                        f"INSERT INTO orders SELECT g, g * 1.5 FROM generate_series(1, {ROWS}) g"))
                else:
                    conn.execute(text("CREATE TABLE orders_copy (id integer, amount numeric(10,2))"))
            e.dispose()
        yield SimpleNamespace(host=host, port=port, dbs=dbs)
    finally:
        with eng.connect() as conn:
            for d in dbs.values():
                conn.execute(text(f'DROP DATABASE IF EXISTS "{d.db}" WITH (FORCE)'))
                conn.execute(text(f'DROP ROLE IF EXISTS "{d.role}"'))
        eng.dispose()


@pytest.fixture
def e2e(pg, tmp_path, monkeypatch):
    import app.api.cloud_connections as cc
    import app.mcp_server.customer_dbs as cdb
    import app.agents.planner as planner_mod
    from sqlalchemy import text

    # The registry and every handle are sealed under DB_ENCRYPTION_KEY; a throwaway here.
    sealing = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", sealing, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(sealing.encode()).digest(), raising=False)
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("MCP_SERVICE_TOKEN", token)
    # The Ray path, where the copier below stands in for the runner.
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://127.0.0.1:8265")
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setenv("DTA_GKE_REACHES_PRIVATE", "1")

    # cust_a and cust_dst_a are user_a's; the other three were registered before owners existed.
    customers = {
        "cust_a": ("src_a", "user_a"),
        "cust_dst_a": ("dst_a", "user_a"),
        "cust_src_legacy": ("src_legacy", None),
        "cust_dst_legacy": ("dst_legacy", None),
        "cust_dst_legacy2": ("dst_legacy2", None),
    }
    records = {}
    for cid, (dbname, owner) in customers.items():
        rec = {
            "customer_id": cid, "customer_name": cid, "database_type": "postgresql",
            "connection_string": pg.dbs[dbname].url, "environment": "production",
            "api_key": "ak_" + secrets.token_hex(16), "status": "active",
            "created_at": "2026-10-18T00:00:00",
        }
        if owner:
            rec["owner_user_id"] = owner
        records[cid] = rec
    path = tmp_path / "customers.json"
    path.write_text(json.dumps(records))
    mgr = cdb.customer_manager
    monkeypatch.setattr(mgr, "customers_file", str(path))
    monkeypatch.setattr(mgr, "customers", {})
    mgr.load_customers()

    onboarding = FastAPI()
    onboarding.include_router(cdb.router)
    client = TestClient(onboarding)
    base = planner_mod.ONBOARDING_API_URL

    class _RoutedSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None, **_kw):
            assert url.startswith(base)
            path = url[len(base):]
            resp = client.get(path, headers=headers or {})
            if path.endswith("/credentials"):
                who = ("service" if (headers or {}).get("X-Avaloka-Service-Token")
                       else "assertion" if str((headers or {}).get("Authorization", "")).startswith("Bearer avu1.")
                       else "other")
                fetches.append((path.split("/")[2], who, resp.status_code))
            return resp

    fetches = []
    monkeypatch.setattr(planner_mod.requests, "Session", _RoutedSession)

    handed = {}

    def _pipeline(**kw):
        handed.clear()
        handed.update(kw)
        return {"error_message": None, "warnings": None}, "# copy orders"

    def _copy(script, **_kw):
        src, dst = handed["source_credentials"], handed["destination_credentials"]
        src_eng = _engine(_url(src["user"], src["password"], src["host"], src["port"], src["database"]))
        dst_eng = _engine(_url(dst["user"], dst["password"], dst["host"], dst["port"], dst["database"]))
        try:
            with src_eng.connect() as s, dst_eng.connect() as d:
                rows = s.execute(text(f'SELECT id, amount FROM "{src["table"]}" ORDER BY id')).all()
                d.execute(text(f'INSERT INTO "{dst["table"]}" (id, amount) VALUES (:id, :amount)'),
                          [{"id": r[0], "amount": r[1]} for r in rows])
        finally:
            src_eng.dispose()
            dst_eng.dispose()
        return SimpleNamespace(rows_inserted=len(rows), table=dst["table"])

    fake_dta = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")
    fake_dta.data_transfer_pipeline = _pipeline
    fake_docker = types.ModuleType("app.data_transfer_docker.docker_run")
    fake_docker.launch_docker_pipeline = _copy
    fake_gke = types.ModuleType("app.data_transfer_docker.gke_run")
    fake_gke.launch_gke_pipeline = _copy
    monkeypatch.setitem(sys.modules, fake_dta.__name__, fake_dta)
    monkeypatch.setitem(sys.modules, fake_docker.__name__, fake_docker)
    monkeypatch.setitem(sys.modules, fake_gke.__name__, fake_gke)

    for n in ("dst_a", "dst_legacy", "dst_legacy2"):
        eng = _engine(pg.dbs[n].url)
        with eng.connect() as conn:
            conn.execute(text("TRUNCATE orders_copy"))
        eng.dispose()

    try:  # tests/ as a package, or tests/ on sys.path
        from tests.dta_confirm import run_confirmed
    except ImportError:
        from dta_confirm import run_confirmed

    def transfer(user, source, destination):
        handed.clear()
        fetches.clear()
        state = {"user_id": user, "dta_database_registry": {}, "messages": []}
        msg = run_confirmed(
            state,
            planner_mod.InitiateTransferParams(
                source_alias=source, destination_alias=destination, source_table="orders",
                user_prompt=f"copy orders from {source} to {destination}",
                write_mode="append", dest_table="orders_copy",
            ),
        )
        return state, msg

    def copied(dbname):
        return _count(pg.dbs[dbname].url, "orders_copy")

    return SimpleNamespace(transfer=transfer, copied=copied, handed=handed, token=token,
                           client=client, pg=pg, fetches=fetches, records=records)


def _identities(e2e, cid):
    """Which identity got a 200 from /credentials for ``cid`` in the last transfer."""
    return {who for c, who, status in e2e.fetches if c == cid and status == 200}


def test_owned_to_owned_copies_through_the_assertion_with_no_token(e2e, monkeypatch):
    monkeypatch.delenv("MCP_SERVICE_TOKEN")
    state, msg = e2e.transfer("user_a", "cust_a", "cust_dst_a")
    assert e2e.copied("dst_a") == ROWS, msg
    assert state["dta_last_transfer"]["success"] is True
    assert _identities(e2e, "cust_a") == _identities(e2e, "cust_dst_a") == {"assertion"}
    assert e2e.handed["source_credentials"]["password"] == e2e.pg.dbs["src_a"].pw
    assert e2e.pg.dbs["src_a"].pw not in json.dumps(state, default=str)


def test_bound_source_to_unbound_destination_copies(e2e):
    state, msg = e2e.transfer("user_a", "cust_a", "cust_dst_legacy")
    assert e2e.copied("dst_legacy") == ROWS, msg
    assert state["dta_last_transfer"]["success"] is True
    # The owned end through the user's assertion; the legacy end through the token (3a).
    assert _identities(e2e, "cust_a") == {"assertion"}
    assert _identities(e2e, "cust_dst_legacy") == {"service"}
    assert e2e.handed["source_credentials"]["password"] == e2e.pg.dbs["src_a"].pw
    assert e2e.handed["destination_credentials"]["password"] == e2e.pg.dbs["dst_legacy"].pw


def test_unbound_to_unbound_copies_for_any_signed_in_user(e2e):
    state, msg = e2e.transfer("user_b", "cust_src_legacy", "cust_dst_legacy2")
    assert e2e.copied("dst_legacy2") == ROWS, msg
    assert state["dta_last_transfer"]["success"] is True
    assert _identities(e2e, "cust_src_legacy") == _identities(e2e, "cust_dst_legacy2") == {"service"}
    assert e2e.handed["source_type"] == e2e.handed["destination_type"] == "postgresql"


def test_another_users_bound_source_is_refused_and_nothing_is_written(e2e):
    state, msg = e2e.transfer("user_b", "cust_a", "cust_dst_legacy")
    assert "cust_a" not in state["dta_database_registry"]
    assert e2e.handed == {}
    assert e2e.copied("dst_legacy") == 0
    assert e2e.pg.dbs["src_a"].pw not in msg
    r = e2e.client.get("/customers/cust_a/credentials",
                       headers={"X-Avaloka-Service-Token": e2e.token, "X-Avaloka-User": "user_b"})
    assert r.status_code == 404
    assert e2e.pg.dbs["src_a"].pw not in r.text
    from app.mcp_server.client import user_headers
    r = e2e.client.get("/customers/cust_a/credentials", headers=user_headers("user_b"))
    assert r.status_code == 404 and "credential_handle" not in r.text


def test_no_service_token_is_refused_and_nothing_is_written(e2e, monkeypatch):
    monkeypatch.delenv("MCP_SERVICE_TOKEN")
    state, msg = e2e.transfer("user_a", "cust_src_legacy", "cust_dst_legacy")
    assert state["dta_database_registry"] == {}
    assert e2e.handed == {}
    assert e2e.copied("dst_legacy") == 0
    for cid, db in (("cust_src_legacy", "src_legacy"), ("cust_a", "src_a")):
        r = e2e.client.get(f"/customers/{cid}/credentials", headers={"X-Avaloka-User": "user_a"})
        assert r.status_code == 401
        assert e2e.pg.dbs[db].pw not in r.text


def test_a_deleted_customer_is_refused_and_nothing_is_written(e2e):
    from app.mcp_server.client import user_headers
    assert e2e.client.delete("/customers/cust_a", headers=user_headers("user_a")).status_code == 200
    state, msg = e2e.transfer("user_a", "cust_a", "cust_dst_legacy")
    assert "cust_a" not in state["dta_database_registry"]
    assert e2e.handed == {}
    assert e2e.copied("dst_legacy") == 0
    assert {status for c, _w, status in e2e.fetches if c == "cust_a"} <= {404}


def test_a_customers_api_key_gets_403_and_no_secret(e2e):
    key = e2e.records["cust_a"]["api_key"]
    r = e2e.client.get("/customers/cust_a/credentials", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 403
    assert "credential_handle" not in r.text and e2e.pg.dbs["src_a"].pw not in r.text
