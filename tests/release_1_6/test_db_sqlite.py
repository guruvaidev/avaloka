"""1.6 release suite: DB-03, SQLite as a registered datasource.

The case has two halves: a SQLite database can be registered and queried
through the MCP server, and a transfer INTO SQLite (a destination the
transfer agent does not support; docs/TECHNICAL_USER_GUIDE.md section 8.1)
is refused with a clear message. The first half works; the second is an open
defect: an unsupported destination falls through to a script that prints the
frame and reports success.
"""
from __future__ import annotations

import sqlite3

import pytest

import app.mcp_server.multi_tenant_mcp_server as mts
from app.agents.data_transfer_agent.data_transfer_agent import make_injection_script


@pytest.fixture
def sqlite_file(tmp_path):
    path = tmp_path / "shop.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, amount REAL);"
        "INSERT INTO orders (region, amount) VALUES ('north', 10.5), ('south', 4.0), ('north', 1.5);"
    )
    con.commit()
    con.close()
    return path


@pytest.fixture
def fresh_manager(monkeypatch):
    monkeypatch.setattr(mts.db_manager, "engines", {})
    monkeypatch.setattr(mts.db_manager, "connection_configs", {})
    return mts.db_manager


def _as_customer(customer_id, fn, *args):
    token = mts.current_customer_id.set(customer_id)
    try:
        return fn(*args)
    finally:
        mts.current_customer_id.reset(token)


def test_db03_sqlite_register_list_and_query(sqlite_file, fresh_manager):
    """DB-03: a SQLite database registers, its tables are listed, and a query
    returns the rows that are actually in the file."""
    conn = mts.DatabaseConnection(
        connection_name="shop_sqlite", database_type="sqlite", database_name=str(sqlite_file),
    )
    cid = fresh_manager.register_connection(conn, api_key="ak_test_sqlite")
    assert cid == "shop_sqlite"

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
    assert out["rows"] == [["north", 12.0], ["south", 4.0]]


def test_db03_sqlite_query_path_stays_read_only(sqlite_file, fresh_manager):
    """DB-03: the registered SQLite source refuses writes through the query tool."""
    conn = mts.DatabaseConnection(
        connection_name="shop_sqlite_ro", database_type="sqlite", database_name=str(sqlite_file),
    )
    cid = fresh_manager.register_connection(conn, api_key="ak_test_sqlite_ro")
    out = _as_customer(cid, mts.query, "DELETE FROM orders")
    assert "error" in out and "read-only" in out["error"].lower()
    con = sqlite3.connect(sqlite_file)
    try:
        assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3
    finally:
        con.close()


def _transfer_state(dest_type: str) -> dict:
    return {
        "source_type": "csv",
        "destination_type": dest_type,
        "source_file": "in.csv",
        "destination_file": "out.db",
        "destination_table": "orders_copy",
        "write_mode": "overwrite",
        "coder_definition": {
            "generated_code": "def transform_data(df):\n    return df",
            "generated_output_schema": "",
        },
    }


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "DB-03 (new finding): an unsupported destination such as sqlite falls through to "
    "print(df.show()) and returns status 'success' "
    "(app/agents/data_transfer_agent/data_transfer_agent.py, final else of the write body)"))
def test_db03_transfer_into_sqlite_is_refused():
    """DB-03: a transfer into SQLite is refused with a message naming the
    destination type; it never produces a script that reports success."""
    try:
        script = make_injection_script(_transfer_state("sqlite"))
    except ValueError as exc:  # refusing up front is acceptable
        assert "sqlite" in str(exc).lower()
        return
    assert '"type": "stdout"' not in script
    assert 'raise ValueError("Unsupported destination type: sqlite' in script
