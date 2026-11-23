"""The 1.6 demo transfer against a real PostgreSQL: the right table, the source's types, no leftovers.

The demo: churn_src.customers (1,200 rows, ``monthly_charges numeric``), a turn
"Transfer to churn_dst table customers_copy", then "create". It answered
"Table churn_dst doesn't exist in churn_dst", then "RAY_DASHBOARD_URL is not
set", and left an empty table named ``churn_dst`` in the destination with
``monthly_charges text``.

Same harness as tests/test_dta_credentials_out_of_rayjob.py: two throwaway
databases on the server named by AVALOKA_TEST_PG_ADMIN_URL, the connections in
the onboarding store reached through its real router, the planner's own
parsers and handler. Only the model is replaced: code generation returns a
fixed transform, which the real sample executor then runs.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import quote_plus, urlsplit

import pytest

try:  # tests/ as a package, or tests/ on sys.path
    from tests.dta_confirm import confirmed, run_confirmed
except ImportError:  # pragma: no cover
    from dta_confirm import confirmed, run_confirmed

# Every case here needs the real server, so the whole module runs on the CI
# Postgres line (bitbucket-pipelines.yml), never in the hermetic suite.
pytestmark = pytest.mark.integration

ADMIN_URL = os.environ.get("AVALOKA_TEST_PG_ADMIN_URL", "").strip()
ROWS = 60
PROMPT = "Transfer to churn_dst table customers_copy"
SOURCE_DDL = ("CREATE TABLE customers (customer_id text, tenure_months integer, monthly_charges numeric, "
              "support_tickets integer, contract text, region text, churned integer)")


def _admin_url():
    if ADMIN_URL:
        return ADMIN_URL
    message = "AVALOKA_TEST_PG_ADMIN_URL is not set; the transfer was not exercised against Postgres"
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
        pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
    pytest.skip(message)


def _engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, isolation_level="AUTOCOMMIT", pool_pre_ping=True)


@pytest.fixture
def pg():
    """churn_src (with customers) and an empty churn_dst, each owned by its own login role."""
    from sqlalchemy import text

    admin = urlsplit(_admin_url())
    host, port = admin.hostname, admin.port or 5432
    run = secrets.token_hex(4)
    dbs = {}
    eng = _engine(_admin_url())
    try:
        with eng.connect() as conn:
            for n in ("src", "dst"):
                db = role = f"e2e_xfer_{n}_{run}"
                pw = "pw-" + secrets.token_hex(12)
                conn.execute(text(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{pw}'"))
                conn.execute(text(f'CREATE DATABASE "{db}" OWNER "{role}"'))
                dbs[n] = SimpleNamespace(db=db, role=role, pw=pw,
                                         url=f"postgresql://{role}:{quote_plus(pw)}@{host}:{port}/{db}")
        e = _engine(dbs["src"].url)
        with e.connect() as conn:
            conn.execute(text(SOURCE_DDL))
            conn.execute(text(
                "INSERT INTO customers SELECT 'c' || g, g % 72, round((18 + (g * 7.3) % 100)::numeric, 2), "
                "g % 5, CASE WHEN g % 3 = 0 THEN 'month-to-month' ELSE 'one-year' END, "
                f"CASE WHEN g % 2 = 0 THEN 'north' ELSE 'south' END, g % 2 FROM generate_series(1, {ROWS}) g"))
        e.dispose()
        yield SimpleNamespace(host=host, port=port, dbs=dbs)
    finally:
        with eng.connect() as conn:
            for d in dbs.values():
                conn.execute(text(f'DROP DATABASE IF EXISTS "{d.db}" WITH (FORCE)'))
                conn.execute(text(f'DROP ROLE IF EXISTS "{d.role}"'))
        eng.dispose()


def _dst_tables(pg):
    from sqlalchemy import inspect
    e = _engine(pg.dbs["dst"].url)
    try:
        return set(inspect(e).get_table_names())
    finally:
        e.dispose()


def _column_types(url, table):
    from sqlalchemy import text
    e = _engine(url)
    try:
        with e.connect() as conn:
            return {r[0]: r[1] for r in conn.execute(text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = :t"), {"t": table})}
    finally:
        e.dispose()


@pytest.fixture
def onboarding(pg, monkeypatch, tmp_path):
    """The two connections in the onboarding store, reached through its real router."""
    import app.agents.planner as P
    import app.mcp_server.customer_dbs as cdb
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MCP_SERVICE_TOKEN", secrets.token_urlsafe(32))
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setenv("DTA_GKE_REACHES_PRIVATE", "1")
    monkeypatch.delenv("RAY_DASHBOARD_URL", raising=False)
    monkeypatch.delenv("RAY_DASHBOARD_URL_INCLUSTER", raising=False)
    records = {}
    for cid, n in (("churn_src", "src"), ("churn_dst", "dst")):
        records[cid] = {
            "customer_id": cid, "customer_name": cid, "database_type": "postgresql",
            "connection_string": pg.dbs[n].url, "environment": "production",
            "api_key": "ak_" + secrets.token_hex(16), "status": "active",
            "created_at": "2026-11-04T00:00:00", "owner_user_id": "user_a",
        }
    store = tmp_path / "customers.json"
    store.write_text(json.dumps(records))
    mgr = cdb.customer_manager
    monkeypatch.setattr(mgr, "customers_file", str(store))
    monkeypatch.setattr(mgr, "customers", {})
    mgr.load_customers()
    app_ = FastAPI()
    app_.include_router(cdb.router)
    client = TestClient(app_)
    base = P.ONBOARDING_API_URL

    class _RoutedSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None, **_kw):
            assert url.startswith(base)
            return client.get(url[len(base):], headers=headers or {})

    monkeypatch.setattr(P.requests, "Session", _RoutedSession)
    yield P
    store.unlink(missing_ok=True)


def _demo_state():
    # What the API's source bridge sets for a db:// analysis of churn_src.customers.
    return {"user_id": "user_a", "messages": [], "dta_database_registry": {},
            "active_db_customer_id": "churn_src", "active_db_source_table": "customers"}


def _create_request(P, state):
    """The "create" turn with its parameters spelled out, independent of the destination parser.

    On 2.x every chat transfer stops at one confirmation question first
    (tests/dta_confirm.py); "yes" answers it, as the user does."""
    return run_confirmed(state, P.InitiateTransferParams(
        destination_alias="churn_dst", dest_table="customers_copy", create_if_missing=True,
        user_prompt=PROMPT, write_mode="append",
    ))


def _demo_turn(P, state):
    """The demo's two turns: the transfer request, then "create" to the offer."""
    parsed = P._extract_transfer_destination(PROMPT)
    params = P.InitiateTransferParams(
        destination_alias=parsed[0] if parsed else None,
        dest_table=P._extract_dest_table_from_prompt(PROMPT),
        user_prompt=PROMPT, write_mode="append",
    )
    first = P._handle_initiate_transfer(state, params)
    pending = state.get("pending_clarification")
    second = P._resolve_pending_dta(state, pending, "create") if isinstance(pending, dict) else None
    # 2.x then asks "Copy A -> B (N rows, new table)? Reply yes": the user says yes.
    second = confirmed(state, second) if second is not None else None
    return first, second


# ---- A + B + C, no Ray: the demo, end to end --------------------------------------

def test_demo_transfer_without_ray_creates_the_named_table_with_the_source_types(pg, onboarding):
    P = onboarding
    state = _demo_state()
    first, second = _demo_turn(P, state)
    assert "`customers_copy` doesn't exist in `churn_dst`" in first, first
    assert second is not None, first
    assert "is not set" not in second, second
    tables = _dst_tables(pg)
    assert "churn_dst" not in tables, "the connection's name must never become a table"
    assert "customers_copy" in tables, second
    src_types = _column_types(pg.dbs["src"].url, "customers")
    dst_types = _column_types(pg.dbs["dst"].url, "customers_copy")
    assert dst_types == src_types, f"created {dst_types}, source has {src_types}"
    assert dst_types["monthly_charges"] == "numeric"
    from sqlalchemy import text
    e = _engine(pg.dbs["dst"].url)
    with e.connect() as conn:
        copied = conn.execute(text("SELECT count(*), sum(monthly_charges) FROM customers_copy")).one()
    e.dispose()
    e = _engine(pg.dbs["src"].url)
    with e.connect() as conn:
        source = conn.execute(text("SELECT count(*), sum(monthly_charges) FROM customers")).one()
    e.dispose()
    assert tuple(copied) == tuple(source) and copied[0] == ROWS


def test_local_copy_failing_part_way_leaves_no_table_in_postgres(pg, onboarding, monkeypatch):
    """PostgreSQL DDL is transactional: the CREATE goes with the rolled-back rows."""
    import pandas as pd
    real_read_sql = pd.read_sql

    written = []

    def _dies_after_one_chunk(*a, **kw):
        kw["chunksize"] = 10
        for i, chunk in enumerate(real_read_sql(*a, **kw)):
            if i == 1:
                raise ConnectionError("source connection reset")
            written.append(len(chunk))
            yield chunk

    monkeypatch.setattr(pd, "read_sql", _dies_after_one_chunk)
    P = onboarding
    _first, second = _demo_turn(P, _demo_state())
    assert second is not None and "local copy failed" in second, second
    assert written, "the copy never started writing (vacuous)"
    assert _dst_tables(pg) == set(), f"left behind: {_dst_tables(pg)}"


# ---- C, with Ray: the pipeline's own create step ----------------------------------

# A real transform (not the no-op stub), keeping every source column.
_CODE = "def transform_data(df):\n    return df.where(df['tenure_months'] >= 0)\n"


@pytest.fixture
def no_model(monkeypatch):
    """Code generation with the model replaced: the real sample executor runs a fixed transform.

    The pipeline's pre-flight wants both models built; they are never called,
    because the code-generation graph they serve is replaced.
    """
    import app.agents.data_transfer_agent.data_transfer_agent as dta
    from app.agents.data_transfer_agent import daft_coder, daft_validator
    monkeypatch.setattr(daft_coder, "coder_llm", object())
    monkeypatch.setattr(daft_validator, "validator_llm", object())
    monkeypatch.setattr(dta, "daft_code_generation_pipeline", _codegen_without_a_model())
    monkeypatch.setattr(dta, "_surface_logic_review_warning", lambda s: None, raising=False)


def _codegen_without_a_model():
    from app.agents.data_transfer_agent import daft_execution

    def _gen(state):
        cdef = state["coder_definition"]
        cdef["generated_code"] = _CODE
        out = daft_execution.execute_code_on_local(_CODE, cdef["input_sample_data"])
        assert out.get("status") == "success", out.get("execution_error")
        cdef["generated_output_schema"] = out["generated_output_schema"]
        return state
    return _gen


@pytest.mark.parametrize("runner", ["raises", "fails"])
def test_ray_run_failing_after_the_pipeline_created_the_table_leaves_none(pg, onboarding, no_model, monkeypatch, runner):
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://ray.invalid:8265")
    P = onboarding
    gke = MagicMock(side_effect=RuntimeError("RayJob submission refused")) if runner == "raises" \
        else MagicMock(return_value=False)
    seen = {}

    def _launch(*a, **kw):
        seen["tables"] = _dst_tables(pg)        # the pipeline really created it, before the job
        return gke(*a, **kw)

    with patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", _launch):
        second = _create_request(P, _demo_state())
    assert gke.called, second
    assert "customers_copy" in seen["tables"], "the pipeline did not create the table (vacuous)"
    assert _dst_tables(pg) == set(), f"left behind after a failed run: {_dst_tables(pg)}\n{second}"


def test_pipeline_created_table_has_the_source_column_types(pg, onboarding, no_model, monkeypatch):
    """The transformed schema comes from a pandas/Parquet round trip of a sample; the types come from the source."""
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://ray.invalid:8265")
    P = onboarding
    seen = {}

    def _launch(*a, **kw):
        seen["types"] = _column_types(pg.dbs["dst"].url, "customers_copy")
        return False   # stop here; the created table is what is under test

    with patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", _launch):
        reply = _create_request(P, _demo_state())
    src_types = _column_types(pg.dbs["src"].url, "customers")
    assert seen.get("types"), f"the pipeline did not create the table (vacuous)\n{reply}"
    assert seen["types"] == src_types, f"created {seen['types']}, source has {src_types}"
