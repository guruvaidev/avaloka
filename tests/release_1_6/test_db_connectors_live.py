"""1.6 release suite: DB-01 (PostgreSQL) and DB-02 (MySQL), registered and
queried through the MCP server against a real server.

Marker ``integration``. Each test needs an admin URL for its server
(``AVALOKA_TEST_POSTGRES_URL`` / ``AVALOKA_TEST_MYSQL_URL``), creates a
throwaway database with one small table, registers it exactly as the
onboarding flow does (``DatabaseConnectionManager.register_connection``),
then lists tables and runs a query through the MCP ``query`` tool. Skips,
naming the variable, when the URL is unset.
"""
from __future__ import annotations

import os
import secrets

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import make_url

import app.mcp_server.multi_tenant_mcp_server as mts

pytestmark = pytest.mark.integration

ROWS = [("north", 10.5), ("south", 4.0), ("north", 1.5)]


@pytest.fixture
def fresh_manager(monkeypatch):
    monkeypatch.setattr(mts.db_manager, "engines", {})
    monkeypatch.setattr(mts.db_manager, "connection_configs", {})
    return mts.db_manager


def _as_customer(cid, fn, *args):
    token = mts.current_customer_id.set(cid)
    try:
        return fn(*args)
    finally:
        mts.current_customer_id.reset(token)


def _throwaway_db(admin_url: str, driver: str):
    """Create a database with an orders table; return (url, cleanup)."""
    name = "db16_" + secrets.token_hex(4)
    admin = sa.create_engine(make_url(admin_url).set(drivername=driver), isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.exec_driver_sql(f"CREATE DATABASE {name}")
    url = make_url(admin_url).set(drivername=driver, database=name)
    eng = sa.create_engine(url)
    with eng.begin() as c:
        c.exec_driver_sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, region VARCHAR(20), amount NUMERIC(8,2))")
        for i, (region, amount) in enumerate(ROWS, start=1):
            c.execute(sa.text("INSERT INTO orders VALUES (:i, :r, :a)"), {"i": i, "r": region, "a": amount})
    eng.dispose()

    def cleanup():
        with admin.connect() as c:
            if driver.startswith("postgresql"):
                c.exec_driver_sql(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
            else:
                c.exec_driver_sql(f"DROP DATABASE IF EXISTS {name}")
        admin.dispose()

    return url, cleanup


def _register_and_query(manager, url, database_type: str, name: str):
    conn = mts.DatabaseConnection(
        connection_name=name, database_type=database_type, host=url.host, port=url.port,
        database_name=url.database, username=url.username, password=url.password,
    )
    cid = manager.register_connection(conn, api_key="ak_" + secrets.token_hex(8))
    tables = _as_customer(cid, mts.list_tables)
    assert "error" not in tables, tables
    names = {t["name"] if isinstance(t, dict) else t for t in tables["tables"]}
    assert "orders" in names
    out = _as_customer(
        cid, mts.query,
        "SELECT region, SUM(amount) AS total FROM orders GROUP BY region ORDER BY region",
    )
    assert "error" not in out, out
    assert out["columns"] == ["region", "total"]
    assert [[r, float(t)] for r, t in out["rows"]] == [["north", 12.0], ["south", 4.0]]


def test_db01_postgres_register_list_and_query(fresh_manager):
    """DB-01: a PostgreSQL database registers, its tables are listed, and a
    query returns the rows in it."""
    admin = os.environ.get("AVALOKA_TEST_POSTGRES_URL")
    if not admin:
        pytest.skip("AVALOKA_TEST_POSTGRES_URL is not set (admin URL of a Postgres server)")
    url, cleanup = _throwaway_db(admin, "postgresql+psycopg2")
    try:
        _register_and_query(fresh_manager, url, "postgresql", "db01_postgres")
    finally:
        for eng in list(fresh_manager.engines.values()):
            eng.dispose()
        cleanup()


def test_db02_mysql_register_list_and_query(fresh_manager):
    """DB-02: a MySQL database registers, its tables are listed, and a query
    returns the rows in it."""
    admin = os.environ.get("AVALOKA_TEST_MYSQL_URL")
    if not admin:
        pytest.skip("AVALOKA_TEST_MYSQL_URL is not set (admin URL of a MySQL server)")
    url, cleanup = _throwaway_db(admin, "mysql+pymysql")
    try:
        _register_and_query(fresh_manager, url, "mysql", "db02_mysql")
    finally:
        for eng in list(fresh_manager.engines.values()):
            eng.dispose()
        cleanup()
