"""The DB sampling, pushdown and no-Ray copy against a real PostgreSQL 16.

Run with a throwaway server and two databases, e.g.::

    docker run -d --name w3-pg -e POSTGRES_USER=w3 -e POSTGRES_PASSWORD=w3pass \
        -p 127.0.0.1:55433:5432 postgres:16
    psql ... -c "CREATE DATABASE clinic" -c "CREATE DATABASE warehouse"
    AVALOKA_TEST_PG="w3:w3pass@host.docker.internal:55433" pytest -m integration tests/test_db_honesty_postgres.py

Skipped when AVALOKA_TEST_PG is unset. The connection strings come from the
DTA's own ``build_connection_string``, so the driver and URL shape are the ones
a transfer uses.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration

PG = os.getenv("AVALOKA_TEST_PG", "")
if not PG:
    pytest.skip("AVALOKA_TEST_PG not set", allow_module_level=True)

from sqlalchemy import create_engine, inspect, text  # noqa: E402

import app.agents.planner as planner  # noqa: E402
from app.agents.data_transfer_agent.data_transfer_agent import build_connection_string  # noqa: E402
from app.agents.planner import InitiateTransferParams, _handle_initiate_transfer  # noqa: E402
from app.core.db_pushdown import parse_aggregate_intent, run_pushdown  # noqa: E402
from app.core.db_sampling import draw_table_sample  # noqa: E402

N, POS = 5110, 249
_user_pw, _hostport = PG.rsplit("@", 1)
USER, PASSWORD = _user_pw.split(":", 1)
HOST, PORT = _hostport.rsplit(":", 1)


try:  # tests/ as a package, or tests/ on sys.path
    from tests.dta_confirm import run_confirmed
except ImportError:  # pragma: no cover
    from dta_confirm import run_confirmed


def _creds(database):
    return {"host": HOST, "port": int(PORT), "user": USER, "password": PASSWORD, "database": database}


def _url(database):
    return build_connection_string(_creds(database), "postgresql")[0]


@pytest.fixture
def clinic():
    eng = create_engine(_url("clinic"))
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS patients"))
        c.execute(text("CREATE TABLE patients (id integer PRIMARY KEY, age double precision, "
                       "hypertension integer, stroke integer, smoker boolean, seen_at timestamp, note text)"))
        # Stored with every stroke=1 row first, as in the e2e table.
        c.execute(text(
            "INSERT INTO patients SELECT g, 20 + (g * 7) % 60, CASE WHEN g < 150 OR g % 9 = 0 THEN 1 ELSE 0 END, "
            f"CASE WHEN g < {POS} THEN 1 ELSE 0 END, g % 3 = 0, timestamp '2024-01-01' + g * interval '1 hour', "
            f"CASE WHEN g % 5 = 0 THEN NULL ELSE 'n' || g END FROM generate_series(0, {N - 1}) g"))
        truth = {r[0]: float(r[1]) for r in c.execute(text(
            "SELECT hypertension, AVG(stroke::float) FROM patients GROUP BY hypertension"))}
    eng.dispose()
    return truth


@pytest.fixture
def warehouse():
    eng = create_engine(_url("warehouse"))
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS patients_copy"))
    yield eng
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS patients_copy"))
    eng.dispose()


def _mcp(database):
    eng = create_engine(_url(database))

    async def run(sql):
        try:
            with eng.connect() as c:
                res = c.execute(text(f"SELECT * FROM ({sql}) AS _mcp_sub LIMIT 10000"))
                return {"columns": list(res.keys()), "rows": [list(r) for r in res.fetchall()],
                        "meta": {"database_type": "postgresql"}}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    return run


def test_pg_pushdown_is_exact_and_random_sample_is_labelled(clinic, monkeypatch):
    run = _mcp("clinic")
    intent = parse_aggregate_intent("What is the stroke rate for patients with hypertension versus without?",
                                    ["id", "age", "hypertension", "stroke", "smoker", "seen_at", "note"])
    res = asyncio.run(run_pushdown(run, "patients", intent, "postgresql"))
    assert {g: f for g, f, _n in res.groups} == pytest.approx(clinic, abs=1e-12)
    assert res.total_rows == N

    _c, rows, sample = asyncio.run(draw_table_sample(run, "patients", 1000, table="patients"))
    assert (sample.method, sample.n, sample.total) == ("random", 1000, N)
    assert abs(sum(r[3] for r in rows) / 1000 - POS / N) < 0.035

    import app.core.db_sampling as ds
    monkeypatch.setattr(ds, "PREFILTER_MIN_ROWS", 1000)      # exercise TABLESAMPLE BERNOULLI
    _c, rows, sample = asyncio.run(draw_table_sample(run, "patients", 500, table="patients"))
    assert sample.method == "random" and 0 < sample.n <= 500


@pytest.fixture
def state(monkeypatch):
    import app.api.cloud_connections as cc
    key = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)
    for var in ("RAY_DASHBOARD_URL", "RAY_DASHBOARD_URL_INCLUSTER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setenv("DTA_GKE_REACHES_PRIVATE", "1")
    monkeypatch.setattr(planner, "_list_dest_tables", lambda creds, db_type: None)
    from app.api.cloud_connections import encrypt_secret
    # Handles and display fields only: the password is opened from the sealed
    # handle the credentials route returns (_http_credentials), after the gate.
    display = lambda db: {k: v for k, v in _creds(db).items() if k != "password"}  # noqa: E731
    return {
        "dta_database_registry": {
            "e2eclinic": {"db_type": "postgresql", "conn_id": "e2eclinic", **display("clinic"), "table": "patients"},
            "e2ewarehouse": {"db_type": "postgresql", "conn_id": "e2ewarehouse", **display("warehouse"), "table": "x"},
        },
        "db_credential_handles": {"e2eclinic": encrypt_secret("a"), "e2ewarehouse": encrypt_secret("b")},
        "messages": [],
    }


def _params():
    return InitiateTransferParams(
        source_alias="e2eclinic", destination_alias="e2ewarehouse",
        user_prompt="transfer from e2eclinic to e2ewarehouse, from patients to patients_copy",
        write_mode="append", dest_table="patients_copy", create_if_missing=True,
        source_table="patients",
    )


def _http_credentials():
    """The onboarding credentials route: each database's DSN, sealed, for its own api_key."""
    from app.mcp_server.credentials import credential_handle
    keys = {"e2eclinic": ("a", "clinic"), "e2ewarehouse": ("b", "warehouse")}

    def get(url, headers=None, **_kw):
        cid = url.split("/customers/")[-1].split("/")[0]
        api_key, db = keys.get(cid, (None, None))
        authorised = api_key and (headers or {}).get("Authorization") == f"Bearer {api_key}"
        if url.endswith("/credentials") and authorised:
            c = _creds(db)
            dsn = f"postgresql://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{db}"
            return MagicMock(status_code=200, json=MagicMock(return_value={
                "host": c["host"], "port": c["port"], "database": db, "username": c["user"],
                "credential_handle": credential_handle(dsn)}))
        token = (headers or {}).get("Authorization", "")
        owner = next((c for c, (k, _db) in keys.items() if token == f"Bearer {k}"), None)
        if url.endswith("/customers") and owner:   # as the customer: its own database only
            return MagicMock(status_code=200, json=MagicMock(return_value=[
                {"customer_id": owner, "database_type": "postgresql"}]))
        return MagicMock(status_code=404, json=MagicMock(return_value={}))

    s = MagicMock()
    s.return_value.__enter__.return_value.get.side_effect = get
    return patch("app.agents.planner.requests.Session", s)


_http_404 = _http_credentials  # the tests below predate the handles


def test_pg_copy_opens_credentials_after_the_gate_and_keeps_them_out_of_state(clinic, warehouse, state):
    """The DTA end to end on PostgreSQL: the registry holds handles, the confirm gate
    opens them, the copy runs, and the password is nowhere in state afterwards."""
    # Nothing registered yet: both connections resolve by name through their handles,
    # the way a user's saved connections do.
    state["dta_database_registry"] = {}
    with _http_credentials():
        first = planner._handle_initiate_transfer(state, _params())
        assert "Reply **yes**" in first and PASSWORD not in first, first
        assert not inspect(warehouse).has_table("patients_copy")   # nothing before the yes
        result = run_confirmed(state, _params())
    assert "Transfer Complete" in result, result
    with warehouse.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM patients_copy")).scalar() == N
    assert PASSWORD not in repr(state)
    for entry in state["dta_database_registry"].values():
        assert "password" not in entry


def test_pg_no_ray_copy_completes_with_equal_counts(clinic, warehouse, state):
    with _http_404():
        result = run_confirmed(state, _params())
    assert "Transfer Complete" in result, result
    with warehouse.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM patients_copy")).scalar() == N
        assert c.execute(text("SELECT COUNT(*) FROM patients_copy WHERE note IS NULL")).scalar() == N // 5
        assert c.execute(text("SELECT SUM(stroke) FROM patients_copy")).scalar() == POS


def test_pg_no_ray_oversized_leaves_no_table(clinic, warehouse, state, monkeypatch):
    import app.agents.data_transfer_agent.local_copy as lc
    monkeypatch.setattr(lc, "LOCAL_COPY_MAX_ROWS", 1000)
    with _http_404():
        result = run_confirmed(state, _params())
    assert "Nothing was written" in result, result
    assert not inspect(warehouse).has_table("patients_copy")


def test_pg_no_ray_mid_copy_failure_leaves_no_table(clinic, warehouse, state, monkeypatch):
    import pandas as pd
    import app.agents.data_transfer_agent.local_copy as lc
    monkeypatch.setattr(lc, "LOCAL_COPY_CHUNK_ROWS", 1000)
    real = pd.read_sql

    def _dies(*a, **kw):
        for i, chunk in enumerate(real(*a, **kw)):
            if i == 3:
                raise ConnectionError("connection reset")
            yield chunk

    monkeypatch.setattr(pd, "read_sql", _dies)
    with _http_404():
        result = run_confirmed(state, _params())
    assert "dropped" in result, result
    assert not inspect(warehouse).has_table("patients_copy")
