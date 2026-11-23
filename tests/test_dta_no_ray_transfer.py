"""A DB-to-DB transfer completes without Ray, or fails before writing anything.

The web e2e run created an empty ``patients_copy`` (0 rows) and then failed with
"RAY_DASHBOARD_URL is not set": the pipeline creates a missing destination table
at setup, and the runner was found missing only afterwards. These tests drive
``_handle_initiate_transfer`` against real SQLite databases (the connection-string
builder is pointed at them) with no Ray configured, plus the Ray path's
after-creation failure.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import app.agents.planner as planner
from app.agents.planner import InitiateTransferParams, _handle_initiate_transfer

ROWS = 120
PLAIN = "Transfer the patients table from my clinic database to my warehouse database as patients_copy."


try:  # tests/ as a package, or tests/ on sys.path
    from tests.dta_confirm import run_confirmed
except ImportError:  # pragma: no cover
    from dta_confirm import run_confirmed


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    for var in ("RAY_DASHBOARD_URL", "RAY_DASHBOARD_URL_INCLUSTER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setenv("DTA_GKE_REACHES_PRIVATE", "1")


@pytest.fixture
def sealing_key(monkeypatch):
    import app.api.cloud_connections as cc
    key = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)


@pytest.fixture
def dbs(tmp_path, monkeypatch, sealing_key):
    """Two SQLite databases behind the planner's postgresql registry entries."""
    src, dst = tmp_path / "clinic.db", tmp_path / "warehouse.db"
    con = sqlite3.connect(src)
    con.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, age REAL, stroke INTEGER, note TEXT)")
    con.executemany("INSERT INTO patients VALUES (?, ?, ?, ?)",
                    [(i, 30 + i % 50, 1 if i < 10 else 0, None if i % 7 else f"n{i}") for i in range(ROWS)])
    con.commit()
    con.close()
    sqlite3.connect(dst).close()
    files = {"clinic_db": src, "warehouse_db": dst}

    import app.agents.data_transfer_agent.data_transfer_agent as dta

    def _sqlite_url(creds, db_type):
        return f"sqlite:///{files[creds['database']]}", {}

    monkeypatch.setattr(dta, "build_connection_string", _sqlite_url)
    monkeypatch.setattr(planner, "_list_dest_tables", lambda creds, db_type, *a: _tables(dst))

    from app.api.cloud_connections import encrypt_secret
    state = {
        "dta_database_registry": {
            "clinic": {"db_type": "postgresql", "host": "db", "port": 5432, "database": "clinic_db",
                       "user": "u", "password": "p", "table": "patients"},
            "warehouse": {"db_type": "postgresql", "host": "db", "port": 5432, "database": "warehouse_db",
                          "user": "u", "password": "p", "table": "other"},
        },
        "db_credential_handles": {"clinic": encrypt_secret("ak_c"), "warehouse": encrypt_secret("ak_w")},
        "messages": [],
    }
    return state, src, dst


def _tables(path: Path):
    con = sqlite3.connect(path)
    try:
        return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    finally:
        con.close()


def _count(path: Path, table: str) -> int:
    con = sqlite3.connect(path)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def _params(prompt=PLAIN, create=True, table="patients_copy", mode="append"):
    return InitiateTransferParams(
        source_alias="clinic", destination_alias="warehouse", user_prompt=prompt,
        write_mode=mode, dest_table=table, create_if_missing=create,
    )


def _http_404(mock_session):
    resp = MagicMock()
    resp.status_code = 404
    mock_session.return_value.__enter__.return_value.get.return_value = resp


# ---- (d) no Ray, small table: copied locally, counts equal ---------------------

@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_no_ray_small_plain_copy_succeeds_locally(mock_pipeline, mock_session, dbs):
    _http_404(mock_session)
    state, src, dst = dbs
    result = run_confirmed(state, _params())
    assert "Transfer Successful" in result or "✅" in result, result
    assert "patients_copy" in _tables(dst)
    assert _count(dst, "patients_copy") == _count(src, "patients") == ROWS
    assert f"{ROWS:,}" in result
    mock_pipeline.assert_not_called()          # no Ray job was set up at all
    assert state["dta_last_transfer"]["execution_env"] == "local"
    assert state["dta_last_transfer"]["rows_inserted"] == ROWS


@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_no_ray_copy_into_an_existing_table_appends_every_row(mock_pipeline, mock_session, dbs):
    _http_404(mock_session)
    state, src, dst = dbs
    con = sqlite3.connect(dst)
    con.execute("CREATE TABLE patients_copy (id INTEGER, age REAL, stroke INTEGER, note TEXT)")
    con.execute("INSERT INTO patients_copy VALUES (-1, 1, 0, 'pre')")
    con.commit()
    con.close()
    result = run_confirmed(state, _params(create=False))
    assert "✅" in result or "Successful" in result, result
    assert _count(dst, "patients_copy") == ROWS + 1


# ---- (e) unavailable executor + oversized / unsupported: refused before creating ---

@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_no_ray_oversized_table_fails_before_creating_the_destination(mock_pipeline, mock_session, dbs, monkeypatch):
    import app.agents.data_transfer_agent.local_copy as lc
    monkeypatch.setattr(lc, "LOCAL_COPY_MAX_ROWS", 50)
    _http_404(mock_session)
    state, _src, dst = dbs
    result = run_confirmed(state, _params())
    assert "isn't configured" in result and "Nothing was written" in result, result
    assert f"{ROWS:,} rows" in result and "50-row limit" in result
    assert "patients_copy" not in _tables(dst), "no empty destination may be left behind"
    mock_pipeline.assert_not_called()


@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_no_ray_transforming_request_is_refused_before_the_pipeline_can_create(mock_pipeline, mock_session, dbs):
    _http_404(mock_session)
    state, _src, dst = dbs

    def _pipeline_that_creates(**kw):   # what the real pipeline does at setup
        con = sqlite3.connect(dst)
        con.execute("CREATE TABLE patients_copy (id INTEGER)")
        con.commit()
        con.close()
        return {"error_message": None}, "script"

    mock_pipeline.side_effect = _pipeline_that_creates
    result = run_confirmed(
        state, _params(prompt="Copy only patients where stroke is 1 from clinic to warehouse as patients_copy"))
    assert "Nothing was written" in result and "Ray" in result, result
    assert "patients_copy" not in _tables(dst)
    mock_pipeline.assert_not_called()


@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_no_ray_copy_requires_both_ends_to_be_the_callers(mock_pipeline, mock_session, dbs, _sealed_store):
    _http_404(mock_session)
    state, _src, dst = dbs

    class _NotTheirs(dict):
        # The onboarding store keeps no credential of warehouse's for this caller:
        # since the 1.7 merge (decision 1) /credentials decides by ownership, so
        # the planner falls through to the real route, which refuses (404).
        def __setitem__(self, k, v):
            if k != "warehouse":
                super().__setitem__(k, v)

    _sealed_store.db = _NotTheirs()
    result = run_confirmed(state, _params())
    assert "you connected or saved yourself" in result and "Nothing was written" in result, result
    assert "patients_copy" not in _tables(dst)


# ---- (f) mid-transfer failure: nothing this run created is left ---------------------

@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_local_copy_failing_mid_way_drops_the_table_it_created(mock_pipeline, mock_session, dbs, monkeypatch):
    import pandas as pd
    import app.agents.data_transfer_agent.local_copy as lc
    monkeypatch.setattr(lc, "LOCAL_COPY_CHUNK_ROWS", 30)
    real_read_sql = pd.read_sql

    def _dies_after_two_chunks(*a, **kw):
        for i, chunk in enumerate(real_read_sql(*a, **kw)):
            if i == 2:
                raise ConnectionError("source connection reset")
            yield chunk

    monkeypatch.setattr(pd, "read_sql", _dies_after_two_chunks)
    _http_404(mock_session)
    state, _src, dst = dbs
    result = run_confirmed(state, _params())
    assert "local copy failed" in result and "dropped" in result, result
    assert "patients_copy" not in _tables(dst)


@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_local_copy_failing_mid_way_into_an_existing_table_rolls_back_its_rows(mock_pipeline, mock_session, dbs, monkeypatch):
    import pandas as pd
    import app.agents.data_transfer_agent.local_copy as lc
    monkeypatch.setattr(lc, "LOCAL_COPY_CHUNK_ROWS", 30)
    real_read_sql = pd.read_sql

    def _dies_after_two_chunks(*a, **kw):
        for i, chunk in enumerate(real_read_sql(*a, **kw)):
            if i == 2:
                raise ConnectionError("source connection reset")
            yield chunk

    monkeypatch.setattr(pd, "read_sql", _dies_after_two_chunks)
    _http_404(mock_session)
    state, _src, dst = dbs
    con = sqlite3.connect(dst)
    con.execute("CREATE TABLE patients_copy (id INTEGER, age REAL, stroke INTEGER, note TEXT)")
    con.execute("INSERT INTO patients_copy VALUES (-1, 1, 0, 'pre')")
    con.commit()
    con.close()
    result = run_confirmed(state, _params(create=False))
    assert "rolled back" in result, result
    assert _count(dst, "patients_copy") == 1   # only the row that was there before


@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_local_copy_that_silently_loses_rows_fails_and_drops_its_table(mock_pipeline, mock_session, dbs, monkeypatch):
    """The count check is what turns a short copy into a failure, not a "success" with fewer rows."""
    import pandas as pd
    import app.agents.data_transfer_agent.local_copy as lc
    monkeypatch.setattr(lc, "LOCAL_COPY_CHUNK_ROWS", 30)
    real_read_sql = pd.read_sql

    def _skips_a_chunk(*a, **kw):
        for i, chunk in enumerate(real_read_sql(*a, **kw)):
            if i != 1:
                yield chunk

    monkeypatch.setattr(pd, "read_sql", _skips_a_chunk)
    _http_404(mock_session)
    state, _src, dst = dbs
    result = run_confirmed(state, _params())
    assert "row count check failed" in result and "dropped" in result, result
    assert "patients_copy" not in _tables(dst)


@pytest.mark.parametrize("runner", ["raises", "fails"])
@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_ray_run_failing_after_setup_drops_the_table_setup_created(mock_pipeline, mock_session, dbs, monkeypatch, runner):
    """With Ray configured, the pipeline creates the table at setup; a failed run must not leave it."""
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://127.0.0.1:8265")
    _http_404(mock_session)
    state, _src, dst = dbs

    def _pipeline_that_creates(**kw):
        con = sqlite3.connect(dst)
        con.execute("CREATE TABLE patients_copy (id INTEGER, age REAL, stroke INTEGER, note TEXT)")
        con.commit()
        con.close()
        return ({"error_message": None, "warnings": None,
                 "created_destination_table": {"table": "patients_copy", "schema": {"id": "Int64"}},
                 "destination_connection_string": f"sqlite:///{dst}", "destination_connect_args": {}},
                "script")

    mock_pipeline.side_effect = _pipeline_that_creates
    gke = MagicMock(side_effect=RuntimeError("RayJob submission refused")) if runner == "raises" \
        else MagicMock(return_value=False)
    with patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", gke):
        result = run_confirmed(state, _params())
    gke.assert_called_once()
    assert "has been dropped" in result, result
    assert "patients_copy" not in _tables(dst)


@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_ray_run_failing_never_drops_a_table_that_already_existed(mock_pipeline, mock_session, dbs, monkeypatch):
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://127.0.0.1:8265")
    _http_404(mock_session)
    state, _src, dst = dbs
    con = sqlite3.connect(dst)
    con.execute("CREATE TABLE patients_copy (id INTEGER)")
    con.commit()
    con.close()
    mock_pipeline.return_value = ({"error_message": None, "warnings": None,
                                   "destination_connection_string": f"sqlite:///{dst}"}, "script")
    with patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", MagicMock(return_value=False)):
        result = run_confirmed(state, _params(create=False))
    assert "Transfer failed during execution" in result
    assert "patients_copy" in _tables(dst)


# ---- the local copy on its own ------------------------------------------------------

def test_local_table_copy_refuses_oversized_before_any_write(tmp_path):
    from app.agents.data_transfer_agent.local_copy import LocalCopyRefused, local_table_copy
    src, dst = tmp_path / "s.db", tmp_path / "d.db"
    con = sqlite3.connect(src)
    con.execute("CREATE TABLE t (id INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(20)])
    con.commit()
    con.close()
    with pytest.raises(LocalCopyRefused):
        local_table_copy(f"sqlite:///{src}", "t", f"sqlite:///{dst}", "t_copy", create_if_missing=True, max_rows=10)
    assert not dst.exists() or "t_copy" not in _tables(dst)


# Registry credentials these tests seed live in a stand-in sealed store, not in
# state: the planner opens them by handle after the confirm gate (tests/dta_sealed.py).
try:  # tests/ as a package, or tests/ on sys.path
    from tests import dta_sealed as _dta_sealed
except ImportError:  # pragma: no cover
    import dta_sealed as _dta_sealed


@pytest.fixture(autouse=True)
def _sealed_store(monkeypatch):
    return _dta_sealed.install(monkeypatch)
