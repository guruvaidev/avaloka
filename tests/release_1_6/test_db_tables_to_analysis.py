"""1.6 release suite: DB-07, pulling database tables into an analysis thread.

The MCP query call is stubbed (the live database half is DB-01/DB-02); the
endpoint's own work -- one dataset per selected table, the rows it read, the
sample written for the coder, one shared thread -- is what is certified.
"""
from __future__ import annotations

import csv

from fastapi.testclient import TestClient

import app.api.server as server

try:  # tests/ as a package, or tests/ on sys.path
    from tests.server_harness import SESSIONS, client, make_auth_headers  # noqa: F401
except ImportError:  # pragma: no cover
    from server_harness import SESSIONS, client, make_auth_headers  # noqa: F401

TABLES = {
    "customers": {"columns": ["id", "region"], "rows": [[1, "north"], [2, "south"], [3, "east"]]},
    "orders": {"columns": ["order_id", "customer_id", "amount"], "rows": [[10, 1, 9.5], [11, 3, 2.0]]},
}


def test_db07_selected_tables_become_thread_datasets(client: TestClient, monkeypatch):
    """DB-07: each selected table becomes a dataset holding that table's rows,
    and all of them land in one analysis thread."""
    sent_sql = []

    async def fake_mcp(customer_id, api_key, sql):
        sent_sql.append((customer_id, sql))
        table = sql.split(" FROM ", 1)[1].split()[0]
        return TABLES[table]

    monkeypatch.setattr(server, "_run_mcp_db_query", fake_mcp, raising=False)

    resp = client.post(
        "/api/database/tables-to-analysis",
        json={"tables": ["customers", "orders"], "customer_id": "cust_db07", "api_key": "ak_db07"},
        headers=make_auth_headers("db07-user"),
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()

    # One query per table, against the caller's connection, row-limited.
    assert [c for c, _ in sent_sql] == ["cust_db07", "cust_db07"]
    assert all(s.startswith("SELECT * FROM ") and " LIMIT " in s for _, s in sent_sql)

    by_alias = {d["alias"]: d for d in out["datasets"]}
    assert set(by_alias) == {"customers", "orders"}
    for name, table in TABLES.items():
        ds = by_alias[name]
        assert ds["columns"] == table["columns"]
        assert [[r[c] for c in table["columns"]] for r in ds["rows"]] == table["rows"]

    # Each dataset is a session bound to its source table, with the rows the
    # coder will sample from written to disk; all share one thread.
    sessions = [s for s in SESSIONS.values() if s.get("customer_id") == "cust_db07"]
    assert {s["source_table"] for s in sessions} == {"customers", "orders"}
    threads = {s.get("thread_id") for s in sessions}
    assert len(threads) == 1 and None not in threads and "" not in threads
    for s in sessions:
        with open(s["sample_local_input"], newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(TABLES[s["source_table"]]["rows"])
