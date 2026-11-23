"""A database saved in the UI is connected: exact answers and transfers work.

The web UI saves a connection with one ``POST /api/connections`` and never
calls ``/api/database/connect``. The handles map (#535) used to be built from
connect sessions only, so a UI-saved database had no handle: the stroke-rate
question fell back to "a random sample of 1,000 of 5,110 rows" (truth 0.1325,
answered 0.19) and a transfer between two saved databases was refused with
"Both databases must be ones you connected in this session". A manual connect
call fixed both, which is the bug: there were two meanings of "connected".

Now there is one. A connection its owner saved through /api/connections is a
connected database for that owner (and only that owner), in every chat turn,
with no extra step from the UI. These tests drive the real routes the UI
calls -- POST /api/connections, /api/database/tables-to-analysis, the chat
route -- with only the onboarding API and the MCP query tool stood in (the MCP
stand-in runs the SQL on a real SQLite table and applies the saved
connection's own ``max_rows`` cap, as the MCP server does).

Every key and password here is generated in the test.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.api.server as server
import app.agents.planner as planner
from app.agents.planner import InitiateTransferParams, _handle_initiate_transfer
from app.services import session_service

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import FakeGraph, SESSIONS, _quiet_send, client, make_auth_headers  # noqa: F401
    from tests.test_db_sampling_honesty import QUESTION, N_ROWS, make_sorted_patients
except ImportError:
    from server_harness import FakeGraph, SESSIONS, _quiet_send, client, make_auth_headers  # noqa: F401
    from test_db_sampling_honesty import QUESTION, N_ROWS, make_sorted_patients

OWNER, OTHER = "saved-owner", "saved-other"
ORDERS_ROWS = 2000
RETURN_QUESTION = "What is the return rate by channel?"


try:  # tests/ as a package, or tests/ on sys.path
    from tests.dta_confirm import run_confirmed
except ImportError:  # pragma: no cover
    from dta_confirm import run_confirmed


class _MustNotRunGraph(FakeGraph):
    def invoke(self, state_in, config=None):
        raise AssertionError("an exact aggregate must be answered by the database, not the sampled analysis")


class _SampledGraph(FakeGraph):
    """The sampled analysis: writes an output so the turn completes."""

    def invoke(self, state_in, config=None):
        from langchain_core.messages import AIMessage
        Path(state_in["output_location"]).write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        return {
            "messages": state_in["messages"] + [AIMessage(content="Result ready")],
            "execution_result": {"status": "success", "stdout": "done"},
            "ready_to_code": True, "coder_definition": {"code": "print(1)"},
            "visualization_config": {}, "visualization_status": "",
        }


@pytest.fixture
def sealing_key(monkeypatch):
    import app.api.cloud_connections as cc
    key = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)


@pytest.fixture
def world(client, monkeypatch, tmp_path, sealing_key):
    """The API with its connections in a temp file and a stand-in onboarding API.

    ``world.registered[customer_id]`` is the registration body the onboarding
    API received (max_rows included); ``world.keys[customer_id]`` the api_key
    it issued. ``world.dbs[customer_id]`` is the SQLite file behind it.
    """
    import app.api.db_connections as dbc

    for k in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "AVALOKA_AUTH_MODE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(dbc.FILE_ENV, str(tmp_path / "db_connections.json"))

    registered: Dict[str, Dict[str, Any]] = {}
    keys: Dict[str, str] = {}

    async def _onboarding(method, path, user_id, body=None):
        if method == "POST" and path == "/customers/register":
            reg = json.loads(body)
            cid = reg["customer_id"]
            registered[cid] = {**reg, "owner": user_id}
            keys[cid] = "ak_" + secrets.token_hex(8)
            return 201, {"customer_id": cid, "api_key": keys[cid], "database_type": reg["database_type"],
                         "status": "active", "mcp_endpoint": "http://mcp:8010/sse"}
        return 200, {}

    monkeypatch.setattr(dbc, "_onboarding", _onboarding)

    dbs: Dict[str, Path] = {}
    seen: List[tuple] = []

    async def _mcp(customer_id, api_key, sql):
        # The MCP server: the api_key must be the one issued for this customer,
        # and every query is capped at the registration's max_rows.
        assert keys.get(customer_id) == api_key, "MCP called with the wrong api_key"
        seen.append((customer_id, sql))
        cap = int(registered[customer_id].get("max_rows") or 1000)
        con = sqlite3.connect(dbs[customer_id])
        try:
            cur = con.execute(f"SELECT * FROM ({sql}) AS _mcp_sub LIMIT {cap}")
            return {"columns": [d[0] for d in cur.description], "rows": [list(r) for r in cur.fetchall()],
                    "meta": {"database_type": "sqlite", "row_limit": cap}}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "meta": {}}
        finally:
            con.close()

    monkeypatch.setattr(server, "_run_mcp_db_query", _mcp, raising=False)
    import types
    return types.SimpleNamespace(client=client, registered=registered, keys=keys, dbs=dbs, seen=seen,
                                 tmp=tmp_path)


def _save(world, user: str, customer_id: str, db_path: Path, **extra) -> Dict[str, Any]:
    """The connect modal: one POST /api/connections (form fields as the modal sends them)."""
    world.dbs[customer_id] = db_path
    body = {"connection_name": customer_id, "customer_id": customer_id, "customer_name": customer_id,
            "contact_email": "me@example.com", "environment": "development", "database_type": "sqlite",
            "host": "localhost", "port": 0, "database_name": customer_id, "username": "reader",
            "password": "pw_" + secrets.token_hex(6), "query_timeout": 30, **extra}
    resp = world.client.post("/api/connections", json=body, headers=make_auth_headers(user))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _analyse(world, user: str, customer_id: str, table: str) -> Dict[str, Any]:
    """Analyze selected tables: the modal decrypts the saved key and calls tables-to-analysis."""
    resp = world.client.post(
        "/api/database/tables-to-analysis",
        json={"tables": [table], "customer_id": customer_id, "api_key": world.keys[customer_id]},
        headers=make_auth_headers(user),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _ask(world, out, user: str, question: str) -> str:
    resp = world.client.post(
        f"/threads/{out['thread_id']}/messages",
        json={"content": question, "metadata": {"dataset_id": out["datasets"][0]["dataset_id"]}},
        headers=make_auth_headers(user),
    )
    assert resp.status_code == 200, resp.text
    msgs = [m for m in resp.json().get("messages", []) if m["role"] == "assistant"]
    assert msgs, resp.json()
    return msgs[-1]["content"]


def make_orders(path: Path) -> Dict[str, float]:
    """The quickstart's ``orders`` shape: 2,000 rows, a boolean ``returned``, stored returns-first."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, region TEXT, channel TEXT, "
                "units INTEGER, returned INTEGER)")
    channels = ("Partner", "Store", "Web")
    rows = [(i, ("North", "South")[i % 2], channels[i % 3], 1 + i % 5, 1 if i < 500 else 0)
            for i in range(ORDERS_ROWS)]
    con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    by: Dict[str, List[int]] = {}
    for _i, _r, ch, _u, ret in rows:
        by.setdefault(ch, []).append(ret)
    return {ch: sum(v) / len(v) for ch, v in by.items()}


# ---------------------------------------------------------------------------
# (a) saved in the UI -> exact pushdown, no "sample of"
# ---------------------------------------------------------------------------

def test_saved_connection_gives_an_exact_pushdown_answer(world, monkeypatch):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    # Above the analysis cap, so without the database the answer would be a sample.
    monkeypatch.setattr(server, "MAX_DB_SAMPLE_ROWS", 1000, raising=False)
    truth = make_sorted_patients(world.tmp / "clinic.db")
    _save(world, OWNER, "clinic", world.tmp / "clinic.db")

    # No /api/database/connect anywhere: the saved connection is the connection.
    assert not [s for s in SESSIONS.values() if s.get("type") == "database"]
    out = _analyse(world, OWNER, "clinic", "patients")
    assert SESSIONS[out["session_id"]]["db_sample"]["method"] == "random"

    text = _ask(world, out, OWNER, QUESTION)
    assert f"{truth['rate'][1]:.4f}" in text and f"{truth['rate'][0]:.4f}" in text, text
    assert f"Computed by the database over all {N_ROWS:,} rows" in text, text
    assert "sample of" not in text, text
    # The aggregate ran on the database (a GROUP BY), not on the sample.
    assert any("GROUP BY" in sql for _c, sql in world.seen)


# ---------------------------------------------------------------------------
# (b) two saved connections -> the transfer has both handles and copies
# ---------------------------------------------------------------------------

@pytest.fixture
def no_ray(monkeypatch):
    for var in ("RAY_DASHBOARD_URL", "RAY_DASHBOARD_URL_INCLUSTER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setenv("DTA_GKE_REACHES_PRIVATE", "1")


def _transfer_state(world, monkeypatch, user: str) -> Dict[str, Any]:
    """The planner state the chat route builds for ``user``: the handles map from the server."""
    import app.agents.data_transfer_agent.data_transfer_agent as dta
    files = {"clinic": world.dbs["clinic"], "warehouse": world.dbs["warehouse"]}
    monkeypatch.setattr(dta, "build_connection_string", lambda creds, t: (f"sqlite:///{files[creds['database']]}", {}))
    monkeypatch.setattr(planner, "_list_dest_tables", lambda creds, t: [])
    handles = asyncio.run(session_service.db_credential_handles_for_user(user))
    return {
        "dta_database_registry": {
            # Handles and display fields: the password is opened from the
            # credentials route's sealed handle, after the confirmation.
            "clinic": {"db_type": "postgresql", "conn_id": "clinic", "host": "db", "port": 5432,
                       "database": "clinic", "user": "u", "table": "patients"},
            "warehouse": {"db_type": "postgresql", "conn_id": "warehouse", "host": "db", "port": 5432,
                          "database": "warehouse", "user": "u", "table": "other"},
        },
        "db_credential_handles": handles,
        "messages": [],
        "user_id": user,
    }


def _transfer(state):
    params = InitiateTransferParams(
        source_alias="clinic", destination_alias="warehouse",
        user_prompt="Transfer the patients table from clinic to warehouse as patients_copy.",
        write_mode="append", dest_table="patients_copy", create_if_missing=True,
    )
    from app.mcp_server.credentials import credential_handle

    try:  # tests/ as a package, or tests/ on sys.path
        from tests.onboarding_identity import credentials_status
    except ImportError:
        from onboarding_identity import credentials_status

    def credentials_route(url, headers=None, **_kw):
        # /customers/{id}/credentials: the connection string, sealed -- to the
        # owner's signed assertion only, as the real route answers since the
        # 1.7 merge (decision 1). Both databases here are OWNER's.
        resp = MagicMock()
        db = url.rstrip("/").split("/")[-2]
        resp.status_code = credentials_status(headers, OWNER)
        resp.json.return_value = (
            {"credential_handle": credential_handle(f"postgresql://u:p@db:5432/{db}")}
            if resp.status_code == 200 else {})
        return resp

    with patch("app.agents.planner.requests.Session") as sess, \
            patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline") as pipe:
        sess.return_value.__enter__.return_value.get.side_effect = credentials_route
        result = run_confirmed(state, params)
        pipe.assert_not_called()
    return result


def _count(path: Path, table: str):
    con = sqlite3.connect(path)
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
            return None
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def test_transfer_between_two_saved_connections_copies_every_row(world, monkeypatch, no_ray):
    make_sorted_patients(world.tmp / "clinic.db")
    sqlite3.connect(world.tmp / "warehouse.db").close()
    _save(world, OWNER, "clinic", world.tmp / "clinic.db")
    _save(world, OWNER, "warehouse", world.tmp / "warehouse.db")

    state = _transfer_state(world, monkeypatch, OWNER)
    assert set(state["db_credential_handles"]) == {"clinic", "warehouse"}
    from app.mcp_server.client import api_key_from_handle
    assert {c: api_key_from_handle(h) for c, h in state["db_credential_handles"].items()} == world.keys

    result = _transfer(state)
    assert "ones you connected" not in result, result
    assert _count(world.tmp / "warehouse.db", "patients_copy") == N_ROWS, result
    assert f"{N_ROWS:,}" in result


# ---------------------------------------------------------------------------
# (c) another user's saved connection is not usable
# ---------------------------------------------------------------------------

def test_another_users_saved_connection_gives_no_handle_pushdown_or_transfer(world, monkeypatch, no_ray):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "MAX_DB_SAMPLE_ROWS", 1000, raising=False)
    make_sorted_patients(world.tmp / "clinic.db")
    sqlite3.connect(world.tmp / "warehouse.db").close()
    _save(world, OWNER, "clinic", world.tmp / "clinic.db")
    _save(world, OWNER, "warehouse", world.tmp / "warehouse.db")

    assert asyncio.run(session_service.db_credential_handles_for_user(OTHER)) == {}
    assert asyncio.run(session_service.db_credential_handle_for_user(OTHER, "clinic")) is None

    # Even holding the owner's api_key, OTHER's own analysis of that database
    # gets no exact pushdown: the key for it comes only from OTHER's handles.
    out = _analyse(world, OTHER, "clinic", "patients")
    sess = SESSIONS[out["session_id"]]
    before = len(world.seen)
    assert asyncio.run(server._db_pushdown_answer(OTHER, sess, QUESTION)) is None
    assert len(world.seen) == before, "no query may run on another user's saved connection"
    monkeypatch.setattr(server, "GRAPH", _SampledGraph(), raising=False)
    text = _ask(world, out, OTHER, QUESTION)
    assert "Computed by the database" not in text, text
    assert f"a random sample of 1,000 of {N_ROWS:,} rows" in text, text

    result = _transfer(_transfer_state(world, monkeypatch, OTHER))
    # Refused: the credentials route answers OTHER's assertion with 404 (ownership).
    assert "you connected or saved yourself" in result and "Nothing was written" in result, result
    assert _count(world.tmp / "warehouse.db", "patients_copy") is None

    # The owner's own rows, marked inactive, stop counting for the owner too.
    import app.api.db_connections as dbc
    store = dbc.connection_store()
    for row in store.list_for(OWNER):
        store.update(row["id"], {"status": "inactive"})
    assert asyncio.run(session_service.db_credential_handles_for_user(OWNER)) == {}


# ---------------------------------------------------------------------------
# (e) the quickstart's orders table is analysed whole, and its return rate is exact
# ---------------------------------------------------------------------------

def test_orders_saved_with_the_form_defaults_is_not_sampled_at_1000(world, monkeypatch):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    truth = make_orders(world.tmp / "sample.db")
    # The modal sends its max_rows default; the API also defaults one it is not sent.
    from app.core.db_sampling import ANALYSIS_MAX_ROWS
    saved = _save(world, OWNER, "sampledb", world.tmp / "sample.db")
    assert world.registered["sampledb"]["max_rows"] == ANALYSIS_MAX_ROWS >= ORDERS_ROWS
    assert saved["max_rows"] == ANALYSIS_MAX_ROWS

    out = _analyse(world, OWNER, "sampledb", "orders")
    meta = SESSIONS[out["session_id"]]["db_sample"]
    assert (meta["method"], meta["n"], meta["total"]) == ("full", ORDERS_ROWS, ORDERS_ROWS), meta

    text = _ask(world, out, OWNER, RETURN_QUESTION)
    for ch, rate in truth.items():
        assert f"| {ch} | {rate:.4f} |" in text, text
    assert f"over all {ORDERS_ROWS:,} rows" in text and "sample of" not in text, text


def test_a_connection_saved_with_a_low_row_cap_says_n_of_m(world, monkeypatch):
    """The cap is the user's to lower; the analysis then says what it is."""
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _SampledGraph(), raising=False)
    make_orders(world.tmp / "sample.db")
    _save(world, OWNER, "sampledb", world.tmp / "sample.db", max_rows=1000)
    out = _analyse(world, OWNER, "sampledb", "orders")
    meta = SESSIONS[out["session_id"]]["db_sample"]
    assert (meta["method"], meta["n"], meta["total"]) == ("random", 1000, ORDERS_ROWS), meta
    text = _ask(world, out, OWNER, "describe how units relate to region")
    assert f"a random sample of 1,000 of {ORDERS_ROWS:,} rows" in text, text
