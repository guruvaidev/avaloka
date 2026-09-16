"""Tests for the DTA conversational destination wizard.

The source is always implicit. When the destination / table / filename / format is
missing, the handler asks ONE guided question, stores what it's waiting for on the
`pending_clarification` channel (type ``dta_transfer``), and resumes on the next reply
until everything is collected — then it runs the transfer. New DB tables are created
from the TRANSFORMED output schema, and cloud files support the full format set.

The pipeline is stubbed to stop right after it's called (so we assert on what it was
handed without a cluster). Network + table reflection are patched.
"""
import sys
import types

import pytest


_CALLS = {"pipeline": []}


@pytest.fixture(autouse=True)
def _wizard_env(monkeypatch):
    _CALLS["pipeline"].clear()

    dta_mod = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")

    def _fake_pipeline(**kwargs):
        _CALLS["pipeline"].append(kwargs)
        return ({"error_message": "__stopped_after_pipeline__", "warnings": None}, None)

    dta_mod.data_transfer_pipeline = _fake_pipeline
    monkeypatch.setitem(sys.modules, "app.agents.data_transfer_agent.data_transfer_agent", dta_mod)

    import app.agents.planner as planner

    class _NoNet:
        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, *a, **k):
                raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(planner, "requests", _NoNet)
    # Reflection would try a real DB connection; return a fixed list so the
    # existing-table branch is deterministic and fast.
    monkeypatch.setattr(planner, "_list_dest_tables", lambda creds, db_type: ["orders", "customers"])
    yield


# ── Helpers ───────────────────────────────────────────────────────────────────

_DB_SRC = {"db_type": "postgresql", "host": "h", "port": 5432, "database": "d",
           "user": "u", "password": "p", "table": "src"}
_DB_DST = {"db_type": "postgresql", "host": "h2", "port": 5432, "database": "d2",
           "user": "u", "password": "p"}
_CLOUD_DST = {"db_type": "gcs", "bucket_name": "wh-bucket", "name": "Warehouse"}


def _state(dst_alias=None, dst_entry=None):
    reg = {"src_db": dict(_DB_SRC)}
    if dst_alias:
        reg[dst_alias] = dict(dst_entry)
    return {
        "dta_database_registry": reg,
        "active_db_customer_id": "src_db",   # implicit DB source
        "user_id": "u1",
        "messages": [],
    }


def _transfer(state, **kw):
    import app.agents.planner as planner
    return planner._handle_initiate_transfer(state, planner.InitiateTransferParams(**kw))


def _resume(state, reply):
    import app.agents.planner as planner
    pending = state.get("pending_clarification")
    assert isinstance(pending, dict) and pending.get("type") == "dta_transfer", pending
    return planner._resolve_pending_dta(state, pending, reply)


def _pending(state):
    return state.get("pending_clarification") or {}


def _last_call():
    assert _CALLS["pipeline"], "data_transfer_pipeline was not called"
    return _CALLS["pipeline"][-1]


# ── Missing destination → grouped list, stored as pending ─────────────────────

def test_missing_destination_stores_pending_and_never_guesses(monkeypatch):
    import app.agents.planner as planner
    async def _fake_list(uid):
        return [{"id": "c1", "name": "gcs_backup", "provider": "gcs", "bucket_name": "b"}]
    monkeypatch.setattr("app.api.cloud_connections.list_cloud_connections", _fake_list)

    state = _state()
    msg = _transfer(state, destination_alias="", user_prompt="transfer keeping age == 29")

    assert "couldn't determine the destination" in msg.lower()
    assert "**Cloud Storage**" in msg and "gcs_backup" in msg
    assert _pending(state).get("awaiting") == "destination"
    assert not _CALLS["pipeline"], "must not run without a destination"


def test_resume_destination_then_new_table_sets_create_flag():
    state = _state("warehouse_db", _DB_DST)
    # Missing destination, but the prompt already asks for a NEW named table.
    _transfer(state, destination_alias="",
              user_prompt="transfer to somewhere and create a new table patient_x")
    assert _pending(state).get("awaiting") == "destination"

    msg = _resume(state, "warehouse_db")
    call = _last_call()
    assert call["destination_type"] == "postgresql"
    assert call["create_if_missing"] is True
    assert call["destination_credentials"]["table"] == "patient_x"
    assert "__stopped_after_pipeline__" in msg


# ── DB destination: new table (ask name) vs existing (list) ───────────────────

def test_db_new_table_without_name_asks_for_name_then_creates():
    state = _state("warehouse_db", _DB_DST)
    msg = _transfer(state, destination_alias="warehouse_db",
                    user_prompt="transfer to warehouse_db and create a new table")
    assert "name the new table" in msg.lower()
    assert _pending(state).get("awaiting") == "new_table_name"
    assert not _CALLS["pipeline"]

    _resume(state, "patient_data_filtered")
    call = _last_call()
    assert call["create_if_missing"] is True
    assert call["destination_credentials"]["table"] == "patient_data_filtered"


def test_db_unstated_table_lists_existing_and_resumes_without_create():
    state = _state("warehouse_db", _DB_DST)
    msg = _transfer(state, destination_alias="warehouse_db",
                    user_prompt="transfer to warehouse_db")
    assert "which table" in msg.lower()
    assert "- orders" in msg and "- customers" in msg      # reflected list shown
    assert _pending(state).get("awaiting") == "table_choice"

    _resume(state, "orders")
    call = _last_call()
    assert call["destination_credentials"]["table"] == "orders"
    assert call["create_if_missing"] is False


def test_db_table_choice_reply_new_creates():
    state = _state("warehouse_db", _DB_DST)
    _transfer(state, destination_alias="warehouse_db", user_prompt="transfer to warehouse_db")
    assert _pending(state).get("awaiting") == "table_choice"
    _resume(state, "new table results_2024")
    call = _last_call()
    assert call["destination_credentials"]["table"] == "results_2024"
    assert call["create_if_missing"] is True


def test_named_table_plus_create_in_same_prompt_sets_create_flag():
    """Regression: 'into table X … create a new table X' must set create_if_missing.
    Previously the create intent was only checked when NO table was named, so a
    named table skipped it and the pipeline rejected the (absent) table."""
    state = _state("warehouse_db", _DB_DST)
    _transfer(
        state,
        destination_alias="warehouse_db",
        dest_table="cloud_import_test",
        user_prompt=("export to warehouse_db, into table cloud_import_test. "
                     "Keep rows where age > 30 and create a new table cloud_import_test."),
    )
    call = _last_call()
    assert call["create_if_missing"] is True
    assert call["destination_credentials"]["table"] == "cloud_import_test"


def test_named_table_that_does_not_exist_offers_to_create():
    """A named table absent from the destination becomes a create-offer (pending),
    not a hard failure — and 'create' resumes into an actual create."""
    state = _state("warehouse_db", _DB_DST)   # _list_dest_tables → ['orders','customers']
    msg = _transfer(state, destination_alias="warehouse_db", dest_table="cloud_import_test",
                    user_prompt="export to warehouse_db, into table cloud_import_test")
    assert "doesn't exist" in msg.lower()
    assert _pending(state).get("awaiting") == "confirm_create_table"
    assert not _CALLS["pipeline"], "must not run until the user decides"

    _resume(state, "create")
    call = _last_call()
    assert call["create_if_missing"] is True
    assert call["destination_credentials"]["table"] == "cloud_import_test"


def test_missing_table_offer_can_redirect_to_existing_table():
    state = _state("warehouse_db", _DB_DST)
    _transfer(state, destination_alias="warehouse_db", dest_table="typoo",
              user_prompt="export to warehouse_db, into table typoo")
    assert _pending(state).get("awaiting") == "confirm_create_table"
    _resume(state, "orders")                    # pick a real one instead
    call = _last_call()
    assert call["destination_credentials"]["table"] == "orders"
    assert call["create_if_missing"] is False


def test_named_existing_table_runs_directly():
    """A named table that DOES exist proceeds straight to the transfer."""
    state = _state("warehouse_db", _DB_DST)
    _transfer(state, destination_alias="warehouse_db", dest_table="orders",
              user_prompt="export to warehouse_db, into table orders")
    call = _last_call()
    assert call["destination_credentials"]["table"] == "orders"
    assert call["create_if_missing"] is False


# ── Cloud destination: filename + format picker ───────────────────────────────

def test_cloud_no_extension_asks_format_then_appends_extension():
    state = _state("Warehouse", _CLOUD_DST)
    msg = _transfer(state, destination_alias="Warehouse",
                    user_prompt="transfer to Warehouse", dest_object="patient_data")
    assert "which file format" in msg.lower()
    assert _pending(state).get("awaiting") == "file_format"
    assert not _CALLS["pipeline"]

    _resume(state, "CSV")
    call = _last_call()
    assert call["destination_type"] == "csv"
    assert call["destination_file"] == "gs://wh-bucket/patient_data.csv"


def test_cloud_with_extension_runs_without_asking_format():
    state = _state("Warehouse", _CLOUD_DST)
    msg = _transfer(state, destination_alias="Warehouse",
                    user_prompt="transfer to Warehouse", dest_object="report.parquet")
    call = _last_call()
    assert call["destination_type"] == "parquet"
    assert "format" not in msg.lower()


def test_cloud_missing_filename_asks_for_name():
    state = _state("Warehouse", _CLOUD_DST)
    msg = _transfer(state, destination_alias="Warehouse", user_prompt="transfer to Warehouse")
    assert "name the file" in msg.lower()
    assert _pending(state).get("awaiting") == "filename"
    assert not _CALLS["pipeline"]


# ── Complete request runs immediately (no wizard turns) ───────────────────────

def test_complete_db_request_runs_immediately():
    state = _state("warehouse_db", _DB_DST)
    _transfer(state, destination_alias="warehouse_db", dest_table="orders",
              user_prompt="transfer to warehouse_db, table orders")
    assert _CALLS["pipeline"], "a fully-specified request should run without questions"
    assert not state.get("pending_clarification")
