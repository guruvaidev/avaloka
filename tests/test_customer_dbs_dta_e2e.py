"""Security hotfix, end to end: a DB-to-DB transfer against a real Postgres.

The /customers routes now need an identity, and /credentials releases a
password only to the API service. These tests drive the planner's transfer path
through the real onboarding router to real databases and count the rows that
land, so they show both halves at once:

  * bound source  -> unbound destination  copies;
  * unbound       -> unbound              copies (customers registered before
                                          owners existed keep working);
  * user_b naming user_a's bound source   is refused, and 0 rows are written;
  * no service token                      is refused, and 0 rows are written.

Every database, role and password here is created per run and dropped after.
Only the code generator and the runner are replaced: by a copier that reads the
source with exactly the credentials the planner handed the pipeline and writes
the destination with exactly the destination's. Those credentials come from
customer_dbs, so a password the onboarding API did not release cannot be used.

Needs ``AVALOKA_TEST_PG_ADMIN_URL``: a Postgres role that may CREATE DATABASE
and CREATE ROLE, e.g. ``postgresql://avaloka:<password>@localhost:5432/postgres``.
Without it the test skips on a laptop, and FAILS when CI is set: a skipped
security test is a green line that tested nothing.
"""
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
    names = ("src_a", "src_legacy", "dst_legacy", "dst_legacy2")
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
    import app.mcp_server.customer_dbs as cdb
    import app.agents.planner as planner_mod
    from sqlalchemy import text

    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("MCP_SERVICE_TOKEN", token)

    # cust_a is user_a's; the other three were registered before owners existed.
    customers = {
        "cust_a": ("src_a", "user_a"),
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
            return client.get(url[len(base):], headers=headers or {})

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

    for n in ("dst_legacy", "dst_legacy2"):
        eng = _engine(pg.dbs[n].url)
        with eng.connect() as conn:
            conn.execute(text("TRUNCATE orders_copy"))
        eng.dispose()

    def transfer(user, source, destination):
        state = {"user_id": user, "dta_database_registry": {}}
        msg = planner_mod._handle_initiate_transfer(
            state,
            planner_mod.InitiateTransferParams(
                source_alias=source, destination_alias=destination,
                user_prompt=f"copy orders from {source} to {destination}",
                write_mode="append", dest_table="orders_copy",
            ),
        )
        return state, msg

    def copied(dbname):
        return _count(pg.dbs[dbname].url, "orders_copy")

    return SimpleNamespace(transfer=transfer, copied=copied, handed=handed, token=token,
                           client=client, pg=pg)


def test_bound_source_to_unbound_destination_copies(e2e):
    state, msg = e2e.transfer("user_a", "cust_a", "cust_dst_legacy")
    assert e2e.copied("dst_legacy") == ROWS, msg
    assert state["dta_last_transfer"]["success"] is True
    assert state["dta_last_transfer"]["rows_inserted"] == ROWS
    assert e2e.handed["source_credentials"]["password"] == e2e.pg.dbs["src_a"].pw
    assert e2e.handed["destination_credentials"]["password"] == e2e.pg.dbs["dst_legacy"].pw


def test_unbound_to_unbound_copies_for_any_signed_in_user(e2e):
    state, msg = e2e.transfer("user_b", "cust_src_legacy", "cust_dst_legacy2")
    assert e2e.copied("dst_legacy2") == ROWS, msg
    assert state["dta_last_transfer"]["success"] is True
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
