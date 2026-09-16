"""Tests for the IMPLICIT-SOURCE + NAMED-DESTINATION transfer grammar.

The customer-facing DTA no longer asks the user to name the source: the source is
the dataset/connection they're already analyzing (surfaced via session state as
``active_db_customer_id`` for a database, or ``connection_id`` +
``data_source_location_cloud`` for a cloud dataset). The prompt only names the
DESTINATION — by its connection NAME (databases also need a table; cloud buckets
also need an output object).

These tests drive ``_handle_initiate_transfer`` with NO ``source_alias`` and assert:
  * the source is resolved from session state (DB and cloud),
  * a missing active source is reported (never silently guessed),
  * the destination resolves by connection NAME, including cloud name lookup with
    case-insensitive match, ambiguity disambiguation, and a not-found listing,
  * the new prompt grammar extracts only the destination (source is ignored).

The transfer pipeline is stubbed to stop right after it's called, so we assert on
the credentials/types it was handed without needing a live cluster / daft / a Groq
key. Network (customer_dbs / Supabase) is disabled; cloud-name lookups are patched.
"""
import sys
import types

import pytest


_CALLS = {"pipeline": []}


@pytest.fixture(autouse=True)
def _stub_pipeline_and_network(monkeypatch):
    _CALLS["pipeline"].clear()

    # Stub the execution pipeline: record what it was handed, then stop the flow
    # BEFORE the runner (error_message set) so no GKE/Docker launch is attempted.
    dta_mod = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")

    def _fake_pipeline(**kwargs):
        _CALLS["pipeline"].append(kwargs)
        return ({"error_message": "__stopped_after_pipeline__", "warnings": None}, None)

    dta_mod.data_transfer_pipeline = _fake_pipeline
    monkeypatch.setitem(
        sys.modules, "app.agents.data_transfer_agent.data_transfer_agent", dta_mod
    )

    # Disable the network so DB credential fetches fall back to the registry entry
    # and unknown DB aliases resolve to None (never a real customer_dbs call).
    import app.agents.planner as planner

    class _NoNetwork:
        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, *a, **k):
                raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(planner, "requests", _NoNetwork)
    # A named DB destination now verifies the table exists before running. With no
    # real DB here, return None (== "couldn't reflect") so the check is skipped and
    # these tests exercise the resolution/extraction logic, not table existence.
    monkeypatch.setattr(planner, "_list_dest_tables", lambda creds, db_type: None)
    yield


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fresh_state():
    return {"dta_database_registry": {}, "messages": []}


def _transfer(state, **kwargs):
    import app.agents.planner as planner
    # No source_alias on purpose — the source is implicit.
    params = planner.InitiateTransferParams(**kwargs)
    return planner._handle_initiate_transfer(state, params)


def _last_pipeline_call():
    assert _CALLS["pipeline"], "data_transfer_pipeline was not called"
    return _CALLS["pipeline"][-1]


_DB_SRC = {"db_type": "mysql", "host": "h", "port": 3306, "database": "d",
           "user": "u", "password": "p", "table": "adult_income"}
_DB_DST = {"db_type": "postgresql", "host": "h2", "port": 5432, "database": "d2",
           "user": "u", "password": "p"}


# ── Implicit SOURCE resolution ────────────────────────────────────────────────

def test_implicit_db_source_from_active_customer_id():
    """The database the user is analyzing (active_db_customer_id) becomes the source."""
    state = _fresh_state()
    state["dta_database_registry"]["pranav_test_2"] = dict(_DB_SRC)
    state["dta_database_registry"]["pranav_test"] = dict(_DB_DST)
    state["active_db_customer_id"] = "pranav_test_2"

    msg = _transfer(
        state,
        destination_alias="pranav_test",
        dest_table="adult_income_dest",
        user_prompt="transfer to pranav_test, table adult_income_dest, keep age == 52",
    )

    call = _last_pipeline_call()
    assert call["source_type"] == "mysql"
    assert call["destination_type"] == "postgresql"
    # Source table comes from the active source; dest table from the prompt.
    assert call["source_credentials"]["table"] == "adult_income"
    assert call["destination_credentials"]["table"] == "adult_income_dest"
    # The flow reached the pipeline (then our stub stopped it).
    assert "__stopped_after_pipeline__" in msg


def test_implicit_db_source_uses_analyzed_table():
    """The transfer reads the table the user ANALYZED (active_db_source_table),
    not the connection's auto-detected first table."""
    state = _fresh_state()
    state["dta_database_registry"]["pranav_test_2"] = dict(_DB_SRC)   # auto-detected table: adult_income
    state["dta_database_registry"]["pranav_test"] = dict(_DB_DST)
    state["active_db_customer_id"] = "pranav_test_2"
    state["active_db_source_table"] = "housing"                       # the table the user analyzed

    _transfer(state, destination_alias="pranav_test", dest_table="housing_dest",
              user_prompt="transfer to pranav_test, table housing_dest")

    call = _last_pipeline_call()
    assert call["source_type"] == "mysql"
    assert call["source_credentials"]["table"] == "housing"          # NOT adult_income


def test_implicit_cloud_source_from_connection_id():
    """The cloud dataset the user is analyzing (connection_id + location) is the source."""
    from app.agents.data_transfer_agent.dta_state import cloud_storage_credentials

    uid = "8a1bdac2-c5ba-4a76-9192-5891d5fead16"
    state = _fresh_state()
    state["dta_database_registry"][uid] = {
        "db_type": "gcs", "bucket_name": "avaloka-src", "name": "C2C_Source_GCP"
    }
    state["dta_database_registry"]["Warehouse"] = {
        "db_type": "gcs", "bucket_name": "avaloka-dst", "name": "Warehouse"
    }
    state["connection_id"] = uid
    state["data_source_location_cloud"] = "gs://avaloka-src/patient_data.csv"

    msg = _transfer(
        state,
        destination_alias="Warehouse",
        dest_object="out/result.parquet",
        user_prompt="transfer to Warehouse, to file out/result.parquet",
    )

    call = _last_pipeline_call()
    # Source object + type derived from the active dataset's cloud location.
    assert call["source_type"] == "csv"
    assert isinstance(call["source_credentials"], cloud_storage_credentials)
    assert call["source_credentials"].file_path == "patient_data.csv"
    assert call["source_file"] == "gs://avaloka-src/patient_data.csv"
    # Destination named + its object from the prompt.
    assert call["destination_type"] == "parquet"
    assert call["destination_file"] == "gs://avaloka-dst/out/result.parquet"


def test_no_active_source_is_reported_not_guessed():
    state = _fresh_state()
    state["dta_database_registry"]["Warehouse"] = {
        "db_type": "gcs", "bucket_name": "b", "name": "Warehouse"
    }
    msg = _transfer(
        state, destination_alias="Warehouse", dest_object="out.csv",
        user_prompt="transfer to Warehouse, to file out.csv",
    )
    assert "active source" in msg.lower()
    assert not _CALLS["pipeline"], "must not run a transfer without a source"


# ── Named DESTINATION resolution (cloud by name) ──────────────────────────────

def _patch_cloud_list(monkeypatch, conns):
    async def _fake_list(user_id):
        return list(conns)
    import app.api.cloud_connections as CC
    monkeypatch.setattr(CC, "list_cloud_connections", _fake_list)


def test_destination_cloud_by_name_resolves_case_insensitively(monkeypatch):
    state = _fresh_state()
    state["dta_database_registry"]["src_db"] = dict(_DB_SRC)
    state["active_db_customer_id"] = "src_db"
    state["user_id"] = "user-1"

    _patch_cloud_list(monkeypatch, [
        {"id": "cid-1", "name": "My Warehouse", "provider": "gcs", "bucket_name": "wh-bucket"},
        {"id": "cid-2", "name": "Other", "provider": "gcs", "bucket_name": "b2"},
    ])

    msg = _transfer(
        state,
        destination_alias="my warehouse",           # lower-case, still resolves
        dest_object="out/result.json",
        user_prompt="transfer to my warehouse, to file out/result.json",
    )

    call = _last_pipeline_call()
    assert call["destination_type"] == "json"
    assert call["destination_file"] == "gs://wh-bucket/out/result.json"


def test_destination_cloud_name_ambiguous_asks_to_disambiguate(monkeypatch):
    state = _fresh_state()
    state["dta_database_registry"]["src_db"] = dict(_DB_SRC)
    state["active_db_customer_id"] = "src_db"
    state["user_id"] = "user-1"

    _patch_cloud_list(monkeypatch, [
        {"id": "a", "name": "Warehouse", "provider": "gcs", "bucket_name": "b1"},
        {"id": "b", "name": "warehouse", "provider": "gcs", "bucket_name": "b2"},
    ])

    msg = _transfer(
        state, destination_alias="Warehouse", dest_object="o.csv",
        user_prompt="transfer to Warehouse, to file o.csv",
    )
    assert "several cloud connections" in msg.lower()
    assert "`a`" in msg and "`b`" in msg          # both ids offered for disambiguation
    assert not _CALLS["pipeline"]


def test_destination_cloud_name_not_found_lists_available(monkeypatch):
    state = _fresh_state()
    state["dta_database_registry"]["src_db"] = dict(_DB_SRC)
    state["active_db_customer_id"] = "src_db"
    state["user_id"] = "user-1"

    _patch_cloud_list(monkeypatch, [
        {"id": "a", "name": "Warehouse", "provider": "gcs", "bucket_name": "b1"},
    ])

    msg = _transfer(
        state, destination_alias="Nope", dest_object="o.csv",
        user_prompt="transfer to Nope, to file o.csv",
    )
    assert "not found" in msg.lower()
    assert "Warehouse" in msg                       # lists the available names
    assert not _CALLS["pipeline"]


# ── New prompt grammar: only the destination is extracted ─────────────────────

def test_new_grammar_extraction_only_captures_destination():
    import app.agents.planner as P

    d = P._extract_transfer_destination(
        "transfer to pranav_test, table adult_income_dest, keeping only rows where age is 52"
    )
    assert d and d[0] == "pranav_test"
    assert P._extract_dest_table_from_prompt(
        "transfer to pranav_test, table adult_income_dest"
    ) == "adult_income_dest"

    d2 = P._extract_transfer_destination("transfer to My Warehouse, to file out.parquet")
    assert d2 and d2[0] == "My Warehouse"           # multi-word connection name
    assert P._extract_object_path_from_prompt(
        "transfer to My Warehouse, to file out.parquet", "dest"
    ) == "out.parquet"

    # Legacy "from <src>" is consumed and ignored; the destination is still captured.
    d3 = P._extract_transfer_destination("transfer from src to dst into table t")
    assert d3 and d3[0] == "dst"

    # A bare object path is never mistaken for a destination connection.
    assert P._extract_transfer_destination("transfer to out.parquet") is None


def test_layman_verbs_extract_destination():
    """Customers phrase transfers in plain English — every synonym must take the
    same deterministic path as 'transfer to' (bare 'move to' used to be missed)."""
    import app.agents.planner as P

    for phrasing in (
        "move to d2c_test, to file housing_d2c_export.parquet",
        "move data to d2c_test, to file housing.parquet",
        "send it to d2c_test, to file housing.parquet",
        "send to d2c_test, table housing_dest",
        "export to d2c_test, to file housing.parquet",
        "export the data to d2c_test, to file housing.parquet",
        "push to d2c_test, table housing_dest",
        "upload to d2c_test, to file housing.parquet",
        "copy to d2c_test, table housing_dest",
        "move this dataset to d2c_test, table housing_dest",
    ):
        d = P._extract_transfer_destination(phrasing)
        assert d and d[0] == "d2c_test", f"failed for: {phrasing!r}"
        # The fast-path trigger must fire for the same phrasings.
        assert P._TRANSFER_TRIGGER_RE.search(phrasing), f"trigger missed: {phrasing!r}"


def test_conversational_phrases_are_not_transfers():
    """Navigation/idioms must not hijack the chat into the transfer flow."""
    import app.agents.planner as P

    # 'move to the next …' is navigation — extractor rejects via the stopword guard.
    assert P._extract_transfer_destination("let's move to the next question") is None
    assert P._extract_transfer_destination("move to the previous step") is None
    # 'a copy of the data compared to…' has no verb→to/from shape — trigger stays quiet.
    assert P._TRANSFER_TRIGGER_RE.search("a copy of the data was shown yesterday") is None
    # Embedded verbs must not match: 'reMOVE', 'photoCOPY' (word-boundary guard).
    assert P._extract_transfer_destination("remove the data to clean it") is None
    assert P._TRANSFER_TRIGGER_RE.search("remove duplicates from the dataset") is None
    assert P._TRANSFER_TRIGGER_RE.search("photocopy to the folder") is None
    # Pronoun destinations are conversation, not connections.
    assert P._extract_transfer_destination("send to me the filtered rows") is None


# ── Two doors, one room: LLM tool calls land on the fast-path's rails ─────────

def test_llm_transfer_params_are_normalized_to_user_words():
    """The LLM path must not invent write_mode or rewrite the user's prompt —
    the observed 'move to …' failure came from the LLM choosing overwrite while
    the fast-path derives append from the same words."""
    import app.agents.planner as P

    raw = "move to d2c_test, to file housing_d2c_export.parquet. Output longitude, latitude."
    params = P.InitiateTransferParams(
        destination_alias="d2c_test",
        user_prompt="Select columns longitude, latitude and transfer",  # LLM's summary
        write_mode="overwrite",                                          # LLM's guess
    )
    P._normalize_llm_transfer_params(params, raw)
    assert params.user_prompt == raw                       # user's words win
    assert params.write_mode == "append"                   # derived, not guessed
    assert params.dest_object == "housing_d2c_export.parquet"  # filled by extractor


def test_write_mode_derivation_is_shared_and_narrow():
    import app.agents.planner as P

    assert P._write_mode_from_prompt("transfer to x, overwrite the file") == "overwrite"
    assert P._write_mode_from_prompt("move to x and replace the existing data") == "overwrite"
    assert P._write_mode_from_prompt("truncate and reload table y") == "overwrite"
    # A transformation instruction must NOT flip the mode.
    assert P._write_mode_from_prompt("transfer to x, replace nulls with 0") == "append"
    assert P._write_mode_from_prompt("move to d2c_test, to file out.parquet") == "append"


def test_customer_text_strips_markdown_noise():
    """The portal shows chat content verbatim — customers must never see
    **bold**/`backtick` tokens, and errors must stay short and readable."""
    import app.agents.planner as P

    raw = (
        "❌ **Transfer Setup Failed:**\n\n"
        "* **Generated Columns:** `['a', 'b']`\n"
        "- Time Taken: `4s`\n\n"
        "*Suggestion: rename them in your prompt.*"
    )
    out = P._dta_customer_text(raw)
    assert "**" not in out and "`" not in out
    assert "• Generated Columns: ['a', 'b']" in out
    assert "• Time Taken: 4s" in out
    assert "Suggestion: rename them in your prompt." in out
    assert "❌" in out                                   # structure/emoji preserved


def test_object_path_recognizes_all_destination_formats():
    """'to file report.xlsx' must be captured — the token regex used to stop at
    csv/json/parquet, so the wizard re-asked for a filename the user gave."""
    import app.agents.planner as P

    for ext in ("csv", "json", "parquet", "tsv", "xlsx", "xml", "avro", "orc"):
        got = P._extract_object_path_from_prompt(f"move to gcs_backup, to file report.{ext}", "dest")
        assert got == f"report.{ext}", f"missed extension: {ext}"
