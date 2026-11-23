"""A chat-driven transfer from a database that is not the session's own, against real Postgres.

The companion of tests/test_dta_named_source_parse.py, with nothing faked but
the network hop to the onboarding API: the planner's chat turns run through the
real fast-path parsers, the real onboarding router (#623: credentials only to a
service-class caller -- the user's assertion for their own records, the service
token for one bound to nobody), the real confirmation gate and the real local
chunked copy. The rows that land are counted, and every call is recorded.

  * "Transfer the orders table from cust_other to cust_dst as orders_copy",
    with cust_sess the session's database, copies cust_other's 37 rows, asked
    for as the user only;
  * the legacy database (bound to nobody), named in chat, is reached through
    the service token after a /status probe, and its 23 rows copy -- also
    through the "Which table in cust_legacy …?" answer;
  * another user's database gets only the service's /status probe, which the
    router refuses; its credentials are never requested, nothing is created,
    and its password is in no reply, log or state;
  * a missing database and another user's look identical to the user.

Every database, role and password here is created per run and dropped after.

Needs ``AVALOKA_TEST_PG_ADMIN_URL``: a Postgres role that may CREATE DATABASE
and CREATE ROLE. Without it the test skips on a laptop, and FAILS when CI is
set: a skipped transfer test is a green line that tested nothing.
"""
import hashlib
import json
import os
import re
import secrets
from types import SimpleNamespace
from urllib.parse import quote_plus, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

# A real database server: the hermetic gate deselects it, and CI runs it in the
# step that has the Postgres service (bitbucket-pipelines.yml).
pytestmark = pytest.mark.integration

ADMIN_URL = os.environ.get("AVALOKA_TEST_PG_ADMIN_URL", "").strip()
TABLES = {  # database -> (table, rows)
    "sess": ("items", 11),
    "other": ("orders", 37),
    "legacy": ("legacy_orders", 23),
    "foreign": ("payroll", 5),
    "dst": (None, 0),
}


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


def _scalar(url, sql):
    from sqlalchemy import text
    eng = _engine(url)
    try:
        with eng.connect() as conn:
            return conn.execute(text(sql)).scalar_one()
    finally:
        eng.dispose()


@pytest.fixture(scope="module")
def pg():
    """One throwaway database per customer, each owned by its own login role."""
    from sqlalchemy import text

    admin = urlsplit(_admin_url())
    host, port = admin.hostname, admin.port or 5432
    run = secrets.token_hex(4)
    dbs = {}
    eng = _engine(_admin_url())
    try:
        with eng.connect() as conn:
            for n in TABLES:
                db, role, pw = f"nse_{n}_{run}", f"nse_{n}_{run}", secrets.token_urlsafe(18)
                # token_urlsafe is [A-Za-z0-9_-] only, so it is safe as a literal;
                # CREATE ROLE takes no bind parameters.
                conn.execute(text(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{pw}'"))
                conn.execute(text(f'CREATE DATABASE "{db}" OWNER "{role}"'))
                dbs[n] = SimpleNamespace(db=db, role=role, pw=pw, url=_url(role, pw, host, port, db))
        for n, (table, rows) in TABLES.items():
            if not table:
                continue
            e = _engine(dbs[n].url)
            with e.connect() as conn:
                conn.execute(text(f'CREATE TABLE "{table}" (id integer PRIMARY KEY, amount numeric(10,2))'))
                conn.execute(text(
                    f'INSERT INTO "{table}" SELECT g, g * 1.5 FROM generate_series(1, {rows}) g'))
            e.dispose()
        yield SimpleNamespace(host=host, port=port, dbs=dbs)
    finally:
        with eng.connect() as conn:
            for d in dbs.values():
                conn.execute(text(f'DROP DATABASE IF EXISTS "{d.db}" WITH (FORCE)'))
                conn.execute(text(f'DROP ROLE IF EXISTS "{d.role}"'))
        eng.dispose()


@pytest.fixture
def chat(pg, tmp_path, monkeypatch):
    import app.agents.planner as P
    import app.api.cloud_connections as cc
    import app.mcp_server.customer_dbs as cdb

    key = secrets.token_hex(16)
    monkeypatch.setenv("DB_ENCRYPTION_KEY", key)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)
    # The API service's token: the one identity that reaches a record bound to nobody (#618).
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("MCP_SERVICE_TOKEN", token)

    # cust_legacy was registered before owners were recorded; cust_foreign is user_b's.
    owners = {"sess": "user_a", "other": "user_a", "dst": "user_a", "legacy": None, "foreign": "user_b"}
    records, api_keys = {}, {}
    for n, owner in owners.items():
        cid = f"cust_{n}"
        api_keys[cid] = "ak_" + secrets.token_hex(16)
        rec = {"customer_id": cid, "customer_name": cid, "database_type": "postgresql",
               "connection_string": pg.dbs[n].url, "environment": "production",
               "api_key": api_keys[cid], "status": "active", "created_at": "2026-10-31T00:00:00"}
        if owner:
            rec["owner_user_id"] = owner
        records[cid] = rec
    path = tmp_path / "customers.json"
    path.write_text(json.dumps(records))
    mgr = cdb.customer_manager
    monkeypatch.setattr(mgr, "customers_file", str(path))
    monkeypatch.setattr(mgr, "customers", {})
    mgr.load_customers()   # seals the secrets on disk, as a real first load does

    client = TestClient(_app(cdb))
    base = P.ONBOARDING_API_URL
    asked = []

    class _RoutedSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None, **_kw):
            assert url.startswith(base)
            r = client.get(url[len(base):], headers=headers or {})
            ident = "service" if (headers or {}).get("X-Avaloka-Service-Token") else (
                "user" if (headers or {}).get("Authorization") else "none")
            asked.append((url[len(base):], ident, r.status_code))
            return r

    monkeypatch.setattr(P.requests, "Session", _RoutedSession)
    # No Ray here: the planner's own local chunked copy runs the transfer.
    monkeypatch.setattr(P, "_ray_executor_configured", lambda *hosts: False)
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setenv("DTA_GKE_REACHES_PRIVATE", "1")
    monkeypatch.setattr(P, "_seed_ray_fields", lambda state: None)
    monkeypatch.setattr(P, "_seed_infra_request_if_needed", lambda state: None)

    # Signed in as user_a, analysing cust_sess. Their own databases come from the
    # onboarding API's listing; the legacy one is reached by name.
    state = {"messages": [], "user_prompt": "", "user_id": "user_a", "dta_database_registry": {},
             "active_db_customer_id": "cust_sess"}

    def say(text):
        nonlocal state
        state = dict(state)
        state["messages"] = list(state["messages"]) + [HumanMessage(content=text)]
        state["user_prompt"] = ""
        state = P.plan_etl_job(state)
        return str(state["messages"][-1].content)

    def rows(n, table):
        exists = _scalar(pg.dbs[n].url, f"SELECT to_regclass('public.\"{table}\"') IS NOT NULL")
        return _scalar(pg.dbs[n].url, f'SELECT count(*) FROM "{table}"') if exists else None

    def calls_for(cid):
        return [c for c in asked if c[0].startswith(f"/customers/{cid}/")]

    return SimpleNamespace(say=say, rows=rows, asked=asked, calls_for=calls_for, pg=pg,
                           state=lambda: state)


def _app(cdb):
    app = FastAPI()
    app.include_router(cdb.router)
    return app


def test_a_non_session_source_named_in_chat_copies_its_rows(chat):
    q = chat.say("Transfer the orders table from cust_other to cust_dst as orders_copy")
    assert "Copy cust_other.orders → cust_dst.orders_copy (37 rows, new table)" in q, q
    done = chat.say("yes")
    assert chat.rows("dst", "orders_copy") == 37, done
    last = chat.state()["dta_last_transfer"]
    assert last["success"] is True and last["rows_inserted"] == 37
    assert last["source"] == "cust_other"
    assert {ident for _, ident, _ in chat.calls_for("cust_other")} == {"user"}
    # The session's own database was never what moved.
    assert chat.rows("sess", "items") == 11


def test_the_table_after_both_connections_copies_the_legacy_database(chat):
    q = chat.say("Transfer from cust_legacy to cust_dst, table legacy_orders as legacy_copy")
    assert "Copy cust_legacy.legacy_orders → cust_dst.legacy_copy (23 rows, new table)" in q, q
    done = chat.say("yes")
    assert chat.rows("dst", "legacy_copy") == 23, done
    calls = chat.calls_for("cust_legacy")
    assert {ident for _, ident, _ in calls} == {"service"}, calls
    assert calls[0] == ("/customers/cust_legacy/status", "service", 200), calls


def test_answering_which_table_copies_it(chat):
    q = chat.say("Transfer from cust_legacy to cust_dst as legacy_copy_b")
    assert q.startswith("❓ Which table in cust_legacy should I transfer (legacy_orders)?"), q
    q = chat.say("legacy_orders")
    assert "Copy cust_legacy.legacy_orders → cust_dst.legacy_copy_b (23 rows, new table)" in q, q
    done = chat.say("yes")
    assert chat.rows("dst", "legacy_copy_b") == 23, done
    assert chat.state()["dta_last_transfer"]["rows_inserted"] == 23
    assert {ident for _, ident, _ in chat.calls_for("cust_legacy")} == {"service"}


def test_another_users_database_is_refused_and_nothing_is_created(chat, caplog):
    import logging
    with caplog.at_level(logging.DEBUG):
        msg = chat.say("Transfer the payroll table from cust_foreign to cust_dst as payroll_copy")
    # Not a database of theirs, and (the last rung) not one of their cloud connections.
    assert re.search(r"Source cust_foreign (not registered|not found)", msg), msg
    # Only the service's status probe names it, and the real router refuses it;
    # its credentials are never requested.
    assert chat.calls_for("cust_foreign") == [("/customers/cust_foreign/status", "service", 404)]
    assert chat.rows("dst", "payroll_copy") is None
    assert "cust_foreign" not in chat.state()["dta_database_registry"]
    planted = chat.pg.dbs["foreign"].pw
    assert planted not in msg and planted not in caplog.text
    assert planted not in json.dumps(chat.state(), default=str)


def test_a_missing_database_and_a_foreign_one_look_the_same(chat):
    foreign = chat.say("Transfer the payroll table from cust_foreign to cust_dst as payroll_copy")
    foreign_calls = [(u.replace("cust_foreign", "<ID>"), i, st) for u, i, st in chat.asked]
    chat.asked.clear()
    missing = chat.say("Transfer the payroll table from cust_nosuch to cust_dst as payroll_copy")
    missing_calls = [(u.replace("cust_nosuch", "<ID>"), i, st) for u, i, st in chat.asked]
    assert foreign.replace("cust_foreign", "<ID>") == missing.replace("cust_nosuch", "<ID>")
    assert foreign_calls == missing_calls
