"""1.6 release suite: database dialects the MCP query path claims to support.

DB-04 (SQL Server), DB-05 (Oracle) and DB-06 (MariaDB) are recorded BLOCKED
for want of a live server. The parts that decide whether a query or a
registration can work at all are pure SQL building and driver arguments, so
they are tested here against fakes. Each is an open defect today, recorded as
a strict xfail that asserts the intended behaviour.
"""
from __future__ import annotations

from typing import Any, List

import pytest

import app.mcp_server.customer_dbs as cdb
import app.mcp_server.multi_tenant_mcp_server as mts


# ---------------------------------------------------------------------------
# DB-04: SQL Server row limit
# ---------------------------------------------------------------------------

@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "DB-04 (MAJ-102): _wrap_with_limit upper-cases only to test for SELECT, then "
    "replaces the literal 'SELECT', so a lowercase query gets no TOP clause"))
def test_db04_mssql_limit_applies_to_lowercase_select():
    """DB-04: a SQL Server query is row-limited however SELECT is spelled."""
    wrapped = mts._wrap_with_limit("select id, amount from orders", 25, "mssql")
    assert "TOP 25" in wrapped.upper()


# ---------------------------------------------------------------------------
# DB-05: Oracle registration
# ---------------------------------------------------------------------------

class _Cursor:
    """A DB-API result: each fetchone() consumes one row, then returns None."""

    def __init__(self, rows: List[tuple]):
        self._rows = list(rows)

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None


class _OracleConn:
    def __init__(self):
        self.sql: List[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, *a, **k):
        s = str(stmt)
        self.sql.append(s)
        if "v$version" in s:
            return _Cursor([("Oracle Database 19c Enterprise Edition",)])
        if "SYS_CONTEXT" in s:
            return _Cursor([("ORCLPDB1", "APP_READER")])
        if "user_tables" in s:
            return _Cursor([(3,)])
        return _Cursor([])


class _Engine:
    def __init__(self, conn):
        self._conn = conn

    def connect(self):
        return self._conn

    def dispose(self):
        pass


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "DB-05 (MAJ-104): the Oracle version check calls fetchone() twice on a one-row "
    "result, and connect_timeout (not a cx_Oracle argument) is passed to every "
    "non-SQLite driver"))
def test_db05_oracle_registration_reads_version_once(monkeypatch):
    """DB-05: an Oracle connection test succeeds against a server that returns
    one version row, and passes no driver argument cx_Oracle rejects."""
    conn = _OracleConn()
    seen: dict = {}

    def fake_create_engine(url, **kwargs):
        seen["url"] = str(url)
        seen["connect_args"] = dict(kwargs.get("connect_args") or {})
        return _Engine(conn)

    monkeypatch.setattr(cdb, "create_engine", fake_create_engine)
    req = cdb.DatabaseConnectionRequest(
        customer_id="cust_oracle", customer_name="Example Oracle",
        database_type="oracle", host="db.example.invalid", port=1521,
        database_name="ORCLPDB1", username="app_reader", password="pw",
        contact_email="ops@example.com",
    )

    result = cdb.DatabaseConnectionManager.test_database_connection(req)

    assert seen["url"].startswith("oracle+cx_oracle://")
    assert result["success"] is True, result
    assert result["database"] == "ORCLPDB1"
    assert result["table_count"] == 3
    assert "connect_timeout" not in seen["connect_args"]


# ---------------------------------------------------------------------------
# DB-06: MariaDB query timeout
# ---------------------------------------------------------------------------

class _RecordingResult:
    def fetchall(self):
        return [(1,)]

    def keys(self):
        return ["n"]


class _RecordingConn:
    def __init__(self):
        self.sql: List[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, *a, **k):
        self.sql.append(str(stmt))
        return _RecordingResult()


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "DB-06 (MAJ-103): MariaDB gets MySQL's SET SESSION max_execution_time, which "
    "MariaDB does not have (its variable is max_statement_time, in seconds)"))
def test_db06_mariadb_uses_its_own_timeout_variable(monkeypatch):
    """DB-06: a MariaDB query sets MariaDB's own statement timeout, not MySQL's."""
    conn = _RecordingConn()
    cfg = mts.DatabaseConfig(
        customer_id="cust_maria", connection_string="mysql+pymysql://u:p@db.example.invalid/x",
        database_type="mariadb", max_rows=10, query_timeout=7,
        created_at="2026-10-31T00:00:00", api_key="ak_test",
    )
    monkeypatch.setattr(mts.db_manager, "connection_configs", {"cust_maria": cfg})
    monkeypatch.setattr(mts.db_manager, "engines", {"cust_maria": _Engine(conn)})
    token = mts.current_customer_id.set("cust_maria")
    try:
        out = mts.query("select count(*) as n from orders")
    finally:
        mts.current_customer_id.reset(token)

    assert "error" not in out, out
    assert out["rows"] == [[1]]
    timeouts = [s for s in conn.sql if s.upper().startswith("SET ")]
    assert timeouts, conn.sql
    assert not any("max_execution_time" in s for s in timeouts), timeouts
    assert any("max_statement_time" in s and "7" in s for s in timeouts), timeouts
