import os
import pytest
from unittest.mock import patch, MagicMock

from app.agents.planner import (
    _handle_initiate_transfer,
    InitiateTransferParams,
    _extract_dest_table_from_prompt,
)
from app.agents.data_transfer_agent.data_transfer_agent import data_transfer_pipeline

# ==============================================================================
# FIXTURES + HELPERS
# ==============================================================================

@pytest.fixture(autouse=True)
def _force_gke_runner_path(monkeypatch):
    """Route these tests through the (mocked) GKE runner they patch.

    The fixtures register DBs on 127.0.0.1. The planner treats VPC-private hosts
    as unreachable from the GKE Ray cluster and auto-forces the Local Docker
    runner, which imports the `docker` SDK. Under these tests' global
    ``requests.Session`` mock, that import raises a metaclass conflict
    (``APIClient`` subclasses ``requests.Session``, now a MagicMock). Every
    runner-hitting test here patches ``launch_gke_pipeline`` and expects the GKE
    path, so pin execution to GKE and opt out of the private-host auto-forcing.
    """
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setenv("DTA_GKE_REACHES_PRIVATE", "1")
    # A named DB destination now reflects the table list to offer a create when it's
    # absent. These tests point at 127.0.0.1 with no live DB; return None ("couldn't
    # reflect") so the existence pre-check is skipped and they reach the pipeline as
    # before, without a real (slow/hanging) connection attempt.
    import app.agents.planner as _planner
    monkeypatch.setattr(_planner, "_list_dest_tables", lambda creds, db_type: None)


@pytest.fixture
def mock_state():
    return {
        "dta_database_registry": {
            "source_db": {
                "db_type": "mysql", "host": "127.0.0.1", "port": 3306,
                "database": "src", "user": "root", "password": "pwd", "table": "users"
            },
            "dest_db": {
                "db_type": "postgresql", "host": "127.0.0.1", "port": 5432,
                "database": "dest", "user": "admin", "password": "pwd", "table": "target_users"
            }
        },
        "messages": []
    }

@pytest.fixture
def transfer_params():
    # NOTE: dest_table is set so we exercise the Case 2 path and reach the
    # pipeline. Without it, a DB destination short-circuits at Case 1
    # ("Which table should I transfer into?").
    return InitiateTransferParams(
        source_alias="source_db", destination_alias="dest_db",
        user_prompt="Copy the users table into table target_users.",
        write_mode="append", dest_table="target_users",
    )


def _config_session(mock_session, status=404, payload=None):
    """Configure a patched ``app.agents.planner.requests.Session`` mock.

    status=404 → _fetch_full_creds / _auto_fetch fall back to the registry entry.
    status=200 + payload → the onboarding API 'returns' those credentials.
    """
    inst = mock_session.return_value.__enter__.return_value
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload if payload is not None else {}
    inst.get.return_value = resp
    return inst

# ==============================================================================
# SECTION 1: THE MESSENGER TESTS (planner._handle_initiate_transfer)
# ==============================================================================

@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_1_planner_critical_error(mock_pipeline, mock_session, mock_state, transfer_params):
    _config_session(mock_session)
    mock_pipeline.return_value = ({"error_message": "Connection Error: Failed", "warnings": None}, None)
    result = _handle_initiate_transfer(mock_state, transfer_params)
    assert "❌ **Transfer Setup Failed:**" in result
    assert "Connection Error: Failed" in result

@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_2_planner_schema_mismatch(mock_pipeline, mock_session, mock_state, transfer_params):
    _config_session(mock_session)
    mock_pipeline.return_value = ({"error_message": "Schema Mismatch Detected!", "warnings": None}, None)
    result = _handle_initiate_transfer(mock_state, transfer_params)
    assert "Schema Mismatch Detected!" in result

@patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", return_value=True)
@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_3_planner_with_warnings(mock_pipeline, mock_session, mock_gke, mock_state, transfer_params):
    _config_session(mock_session)
    mock_pipeline.return_value = ({"error_message": None, "warnings": ["Table not found"]}, "print('code')")
    result = _handle_initiate_transfer(mock_state, transfer_params)
    assert "✅ **Transfer complete:" in result
    assert "⚠️ **Warnings:**" in result

@patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", return_value=True)
@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_4_planner_success(mock_pipeline, mock_session, mock_gke, mock_state, transfer_params):
    _config_session(mock_session)
    mock_pipeline.return_value = ({"error_message": None, "warnings": None}, "print('code')")
    result = _handle_initiate_transfer(mock_state, transfer_params)
    assert "✅ **Transfer complete:" in result
    assert "❌" not in result

@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_5_planner_system_crash(mock_pipeline, mock_session, mock_state, transfer_params):
    _config_session(mock_session)
    mock_pipeline.side_effect = Exception("Out of Memory")
    result = _handle_initiate_transfer(mock_state, transfer_params)
    assert "❌ **Transfer failed (System Error):**" in result

@patch("app.agents.planner.requests.Session")
def test_6_planner_missing_source(mock_session, mock_state, transfer_params):
    _config_session(mock_session, status=404)  # auto-fetch fails
    transfer_params.source_alias = "ghost_db"
    result = _handle_initiate_transfer(mock_state, transfer_params)
    assert "Source **`ghost_db`** not registered" in result

@patch("app.agents.planner.requests.Session")
def test_7_planner_missing_dest(mock_session, mock_state, transfer_params):
    _config_session(mock_session, status=404)
    transfer_params.destination_alias = "ghost_db"
    result = _handle_initiate_transfer(mock_state, transfer_params)
    assert "Destination **`ghost_db`** not registered" in result

@patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", return_value=True)
@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_8_planner_auto_fetch_source_api_success(mock_pipeline, mock_session, mock_gke, mock_state, transfer_params):
    # The onboarding API 'has' this source; it should be auto-registered.
    _config_session(mock_session, status=200, payload={
        "host": "1.2.3.4", "port": 3306, "database": "api_db",
        "username": "u", "password": "p", "table": "t",
        "customers": [{"customer_id": "new_api_db", "database_type": "mysql"}],
    })
    transfer_params.source_alias = "new_api_db"
    mock_pipeline.return_value = ({"error_message": None}, "print('code')")
    result = _handle_initiate_transfer(mock_state, transfer_params)
    assert "✅ **Transfer complete:" in result
    assert "new_api_db" in mock_state["dta_database_registry"]

@patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", return_value=True)
@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_9_planner_auto_fetch_dest_api_success(mock_pipeline, mock_session, mock_gke, mock_state, transfer_params):
    _config_session(mock_session, status=200, payload={
        "host": "1.2.3.4", "port": 5432, "database": "api_db",
        "username": "u", "password": "p", "table": "t",
        "customers": [{"customer_id": "new_api_db", "database_type": "postgresql"}],
    })
    transfer_params.destination_alias = "new_api_db"
    mock_pipeline.return_value = ({"error_message": None}, "print('code')")
    result = _handle_initiate_transfer(mock_state, transfer_params)
    assert "✅ **Transfer complete:" in result
    assert "new_api_db" in mock_state["dta_database_registry"]

@patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", return_value=True)
@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_10_planner_script_not_dumped_in_response(mock_pipeline, mock_session, mock_gke, mock_state, transfer_params):
    # The success message must NOT echo the (potentially huge) generated script.
    massive_script = "x" * 5000
    mock_pipeline.return_value = ({"error_message": None}, massive_script)
    result = _handle_initiate_transfer(mock_state, transfer_params)
    assert massive_script not in result
    assert len(result) < 500
    assert "✅ **Transfer complete:" in result

@patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", return_value=True)
@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_11_planner_write_mode_overwrite(mock_pipeline, mock_session, mock_gke, mock_state, transfer_params):
    _config_session(mock_session)
    transfer_params.write_mode = "overwrite"
    mock_pipeline.return_value = ({"error_message": None}, "code")
    result = _handle_initiate_transfer(mock_state, transfer_params)
    assert "- Write mode: `overwrite`" in result

@patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", return_value=False)
@patch("app.agents.planner.requests.Session")
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
def test_11b_planner_execution_failure(mock_pipeline, mock_session, mock_gke, mock_state, transfer_params):
    # Setup succeeds (script produced) but the runner reports failure.
    _config_session(mock_session)
    mock_pipeline.return_value = ({"error_message": None, "warnings": None}, "code")
    result = _handle_initiate_transfer(mock_state, transfer_params)
    assert "❌ **Transfer failed during execution**" in result

# ==============================================================================
# SECTION 2: THE ENGINE TESTS (data_transfer_agent.data_transfer_pipeline)
#   Uses the CURRENT internals: probe_db_connection, deduce_schema,
#   _read_daft_df, daft_code_generation_pipeline, make_injection_script.
#   coder_llm is patched so the Step-0 preflight passes without an API key.
# ==============================================================================

_DB_SRC = {"host": "h", "port": 3306, "database": "d", "user": "u", "password": "p", "table": "src"}
_DB_DST = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p", "table": "dst"}

@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
@patch("app.agents.data_transfer_agent.data_transfer_agent.probe_db_connection")
def test_12_engine_source_schema_crash_fatal(mock_probe, mock_deduce):
    mock_deduce.side_effect = Exception("DB is down")
    state, script = data_transfer_pipeline(
        user_prompt="test", source_type="mysql", destination_type="postgresql",
        source_credentials=_DB_SRC, destination_credentials=_DB_DST,
    )
    assert "Could not read source schema" in state["error_message"]
    assert script is None

@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.make_injection_script", return_value="SCRIPT")
@patch("app.agents.data_transfer_agent.data_transfer_agent.daft_code_generation_pipeline")
@patch("app.agents.data_transfer_agent.data_transfer_agent._read_daft_df")
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
def test_13_engine_dest_schema_unverifiable_is_fatal_with_retry_hint(mock_deduce, mock_read, mock_codegen, mock_inject):
    # A dest-schema READ failure (not an absent table — deduce returns empty for
    # that) means compatibility can't be verified. The pipeline deliberately aborts
    # with a retry hint instead of silently skipping the schema check; this test
    # previously pinned the older warn-and-continue contract.
    import pandas as pd

    def _deduce(data_type, conn_str, table=None, io_config=None):
        if conn_str == "b":       # destination path
            raise Exception("Dest unreadable")
        return {"id": "Int64"}
    mock_deduce.side_effect = _deduce
    mock_read.return_value.to_pandas.return_value = pd.DataFrame({"id": [1]})
    mock_codegen.side_effect = lambda s: {
        **s, "code_generated_successfully": True,
        "coder_definition": {"generated_output_schema": {"id": "Int64"}},
    }
    state, script = data_transfer_pipeline(
        user_prompt="test", source_type="csv", destination_type="csv",
        source_credentials={"file_path": "a"}, destination_credentials={"file_path": "b"},
    )
    assert "Could not read the destination schema" in (state.get("error_message") or "")
    assert script is None

@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent._read_daft_df")
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
@patch("app.agents.data_transfer_agent.data_transfer_agent.probe_db_connection")
def test_14_engine_sample_data_db_crash(mock_probe, mock_deduce, mock_read):
    mock_deduce.return_value = {"id": "Int64"}       # both source + dest schemas OK
    mock_read.side_effect = Exception("SQL timeout")
    state, script = data_transfer_pipeline(
        user_prompt="test", source_type="mysql", destination_type="postgresql",
        source_credentials=_DB_SRC, destination_credentials=_DB_DST,
    )
    assert "Could not fetch sample data" in state["error_message"]
    assert script is None

@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent._read_daft_df")
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
def test_15_engine_sample_data_csv_crash(mock_deduce, mock_read):
    mock_deduce.return_value = {"id": "Int64"}
    mock_read.side_effect = Exception("File not found")
    state, script = data_transfer_pipeline(
        user_prompt="test", source_type="csv", destination_type="csv",
        source_credentials={"file_path": "bad.csv"}, destination_credentials={"file_path": "out.csv"},
    )
    assert "Could not fetch sample data" in state["error_message"]

@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.daft_code_generation_pipeline")
@patch("app.agents.data_transfer_agent.data_transfer_agent._read_daft_df")
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
def test_16_engine_code_gen_validation_failure(mock_deduce, mock_read, mock_codegen):
    import pandas as pd
    mock_deduce.return_value = {"a": "Int64"}
    mock_read.return_value.to_pandas.return_value = pd.DataFrame({"a": [1]})
    mock_codegen.side_effect = lambda s: {
        **s, "coder_definition": {"syntax_error": True, "code_validation_feedback": "Missing colon"}
    }
    state, script = data_transfer_pipeline(
        user_prompt="test", source_type="csv", destination_type="csv",
        source_credentials={"file_path": "a"}, destination_credentials={"file_path": "b"},
    )
    # Customer-facing: a short reason line, not the raw validator feedback dump.
    assert "could not be generated" in state["error_message"]
    assert "Reason: Missing colon" in state["error_message"]

@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.daft_code_generation_pipeline")
@patch("app.agents.data_transfer_agent.data_transfer_agent._read_daft_df")
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
def test_17_engine_schema_match_failure(mock_deduce, mock_read, mock_codegen):
    # Destination schema exists but the generated columns differ → mismatch.
    import pandas as pd
    mock_deduce.return_value = {"a": "Int64"}                       # dest has column 'a'
    mock_read.return_value.to_pandas.return_value = pd.DataFrame({"a": [1]})
    mock_codegen.side_effect = lambda s: {
        **s, "code_generated_successfully": True,
        "coder_definition": {
            "generated_output_schema": {"b": "Int64"},              # generated column 'b'
            # Non-stub code: the pipeline's no-op-stub gate runs before the schema
            # check, and a mocked codegen with no generated_code reads as a stub.
            "generated_code": "def transform_data(df):\n    return df.select('b')\n",
        },
    }
    state, script = data_transfer_pipeline(
        user_prompt="test", source_type="csv", destination_type="csv",
        source_credentials={"file_path": "a"}, destination_credentials={"file_path": "b"},
    )
    assert "Schema Mismatch Detected" in state["error_message"]

@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.daft_code_generation_pipeline")
@patch("app.agents.data_transfer_agent.data_transfer_agent._read_daft_df")
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
def test_18_engine_cloud_source_parquet(mock_deduce, mock_read, mock_codegen):
    import pandas as pd
    mock_deduce.return_value = {"a": "Int64"}
    mock_read.return_value.to_pandas.return_value = pd.DataFrame({"a": [1]})
    mock_codegen.side_effect = lambda s: {**s, "code_generated_successfully": False, "coder_definition": {}}
    state, script = data_transfer_pipeline(
        user_prompt="test", source_type="parquet", destination_type="csv",
        source_credentials={"file_path": "s3://bucket/file.parquet"}, destination_credentials={"file_path": "b"},
    )
    assert state is not None

@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.daft_code_generation_pipeline")
@patch("app.agents.data_transfer_agent.data_transfer_agent._read_daft_df")
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
def test_19_engine_cloud_dest_parquet(mock_deduce, mock_read, mock_codegen):
    import pandas as pd
    mock_deduce.return_value = {"a": "Int64"}
    mock_read.return_value.to_pandas.return_value = pd.DataFrame({"a": [1]})
    mock_codegen.side_effect = lambda s: {**s, "code_generated_successfully": False, "coder_definition": {}}
    state, script = data_transfer_pipeline(
        user_prompt="test", source_type="csv", destination_type="parquet",
        source_credentials={"file_path": "a"}, destination_credentials={"file_path": "gs://bucket/out.parquet"},
    )
    assert state is not None

# ==============================================================================
# SECTION 3: REAL CONNECTION-ERROR TEST (no mocks on the engine)
# ==============================================================================

@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.planner.requests.Session")
def test_20_real_connection_error(mock_session):
    _config_session(mock_session)  # onboarding API unavailable -> use registry creds
    real_state = {
        "dta_database_registry": {
            "real_source": {
                "db_type": "postgresql", "host": "127.0.0.1", "port": 59999,
                "database": "nodb", "user": "u", "password": "p", "table": "t"
            },
            "real_dest": {
                "db_type": "postgresql", "host": "127.0.0.1", "port": 59999,
                "database": "nodb", "user": "u", "password": "p", "table": "t"
            }
        },
        "messages": []
    }
    params = InitiateTransferParams(
        source_alias="real_source", destination_alias="real_dest",
        user_prompt="Copy everything into table t.", write_mode="append", dest_table="t",
    )
    result = _handle_initiate_transfer(real_state, params)
    assert "❌ **Transfer Setup Failed:**" in result
    assert "Cannot connect to source database" in result

# ==============================================================================
# SECTION 4: DESTINATION-TABLE HANDLING  (the three cases + default-DB fix)
#   Case 1  — no table named in the prompt  -> ask the user which table
#   Case 2  — table named                   -> use ONLY that table, never the
#             registered default (this is the "default database picking" fix)
#   Case 3  — table named but doesn't exist  -> ask the user to create it
#   Parsing — many prompt phrasings ("to this table X", "into table X", ...)
# ==============================================================================


def _force_registry_fallback(mock_session):
    """Make the patched requests.Session return a non-200 so _fetch_full_creds
    falls back to the in-memory registry entry (no real network in tests)."""
    inst = mock_session.return_value.__enter__.return_value
    inst.get.return_value.status_code = 404
    return inst


# ---- Prompt-condition parsing: "to this table", "onto this table", and friends ----

@pytest.mark.parametrize("prompt,expected", [
    # the exact failing UI case: "in this table X"
    ("transfer from a to b in this table housing_high_income , output cols", "housing_high_income"),
    ("transfer from a to b into this table sales_2024", "sales_2024"),
    ("transfer from a to b onto table warehouse_x", "warehouse_x"),
    ("transfer from a to b into table orders", "orders"),
    ("transfer from a to b to the table revenue in mydb", "revenue"),
    ("transfer from a to b on table analytics", "analytics"),
    ("transfer from a to b save it into results_final table", "results_final"),
    ("transfer from a to b into the table `quoted_name`", "quoted_name"),
    # no destination table named -> None (this drives Case 1)
    ("transfer from a to b", None),
    ("transfer from a to b output longitude latitude columns", None),
    # stopword guard: must never return 'this'/'the' as a table name
    ("dump it in this table", None),
])
def test_21_dest_table_extraction_various_phrasings(prompt, expected):
    assert _extract_dest_table_from_prompt(prompt) == expected


# ---- Case 1: no table named -> ask the user (do not guess / do not default) ----

@patch("app.agents.planner.requests.Session")
def test_22_case1_no_table_asks_user(mock_session, mock_state):
    _force_registry_fallback(mock_session)
    params = InitiateTransferParams(
        source_alias="source_db", destination_alias="dest_db",
        user_prompt="transfer from source_db to dest_db", write_mode="append",
        dest_table=None,
    )
    result = _handle_initiate_transfer(mock_state, params)
    # The wizard now asks which table (listing existing ones, or offering to create a new one).
    assert "Which table" in result
    assert "new table" in result.lower()


# ---- Case 2 + default-DB fix: named table wins over the registered default ----

@patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", return_value=True)
@patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline")
@patch("app.agents.planner.requests.Session")
def test_23_case2_named_table_overrides_registered_default(
    mock_session, mock_pipeline, mock_gke, mock_state
):
    _force_registry_fallback(mock_session)
    mock_pipeline.return_value = ({"error_message": None, "warnings": None}, "print('code')")
    params = InitiateTransferParams(
        source_alias="source_db", destination_alias="dest_db",
        user_prompt="transfer from source_db to dest_db into table custom_orders",
        write_mode="append", dest_table="custom_orders",
    )
    _handle_initiate_transfer(mock_state, params)

    # The pipeline must be handed the NAMED table, not the registry default 'target_users'.
    _, kwargs = mock_pipeline.call_args
    dest_creds = kwargs["destination_credentials"]
    assert dest_creds["table"] == "custom_orders"
    assert dest_creds["table"] != "target_users"


# ---- Case 3: named table doesn't exist in the destination -> ask to create it ----

@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.daft_code_generation_pipeline")
@patch("app.agents.data_transfer_agent.data_transfer_agent._read_daft_df")
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
@patch("app.agents.data_transfer_agent.data_transfer_agent.probe_db_connection")
def test_24_case3_missing_dest_table_asks_to_create(
    mock_probe, mock_deduce, mock_read, mock_codegen
):
    import pandas as pd

    def _deduce(data_type, conn_str, table=None, io_config=None):
        # destination table does not exist -> deduce_schema returns None for it
        if table == "ghost_dest":
            return None
        return {"longitude": "Float64", "median_house_value": "Int64"}
    mock_deduce.side_effect = _deduce

    mock_read.return_value.to_pandas.return_value = pd.DataFrame(
        {"longitude": [-122.2], "median_house_value": [100000]}
    )
    mock_codegen.side_effect = lambda s: {
        **s,
        "code_generated_successfully": True,
        "coder_definition": {
            "generated_output_schema": {"longitude": "Float64", "median_house_value": "Int64"},
            # Non-stub code: the no-op-stub gate runs before the missing-table check.
            "generated_code": "def transform_data(df):\n    return df.where(df['longitude'] < 0)\n",
        },
    }

    src = {"host": "127.0.0.1", "port": 3306, "database": "src",
           "user": "u", "password": "p", "table": "src_tbl"}
    dst = {"host": "127.0.0.1", "port": 5432, "database": "dst",
           "user": "u", "password": "p", "table": "ghost_dest"}

    state, script = data_transfer_pipeline(
        user_prompt="filter and transfer",
        source_type="mysql", destination_type="postgresql",
        source_credentials=src, destination_credentials=dst,
    )

    assert script is None
    assert "does not exist" in (state.get("error_message") or "")


# ==============================================================================
# SECTION 5: GAP FIXES
#   #2 schema check -> subset + case-insensitive (allow extra dest columns)
#   #3 source-table existence guard (mirror of Case 3)
#   #1/#5 SQL sink -> write_mode honored + staging/atomic-swap (with fallback)
# ==============================================================================

from app.agents.data_transfer_agent.data_transfer_agent import (
    check_schema_match,
    columns_missing_from_destination,
)
from app.agents.data_transfer_agent.datasink.sql_sink import (
    PostgresDataSink,
    MySQLDataSink,
)


# ---- #2 schema compatibility ----

@pytest.mark.parametrize("gen,dest,ok", [
    (["longitude", "latitude"], ["id", "longitude", "latitude", "created_at"], True),  # extra dest cols OK
    (["Longitude", "Latitude"], ["longitude", "latitude"], True),                       # case-insensitive
    (["longitude"], ["longitude", "latitude"], True),                                   # subset
    (["ghost"], ["longitude", "latitude"], False),                                      # genuine missing
    (["a", "b", "c"], ["a", "b"], False),                                               # one missing
])
def test_25_schema_check_subset_and_case(gen, dest, ok):
    assert check_schema_match(gen, dest) is ok


def test_26_columns_missing_reports_offenders():
    assert columns_missing_from_destination(["Longitude", "ghost"], ["longitude", "latitude"]) == ["ghost"]


# ---- #3 source-table existence guard ----

@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
@patch("app.agents.data_transfer_agent.data_transfer_agent.probe_db_connection")
def test_27_source_table_missing_guard(mock_probe, mock_deduce):
    def _deduce(data_type, conn_str, table=None, io_config=None):
        return None if table == "ghost_src" else {"a": "Int64"}
    mock_deduce.side_effect = _deduce
    src = {"host": "h", "port": 3306, "database": "d", "user": "u", "password": "p", "table": "ghost_src"}
    dst = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p", "table": "dst"}
    state, script = data_transfer_pipeline(
        user_prompt="x", source_type="mysql", destination_type="postgresql",
        source_credentials=src, destination_credentials=dst,
    )
    assert script is None
    assert "Source table `ghost_src` does not exist" in state["error_message"]


# ---- #1/#5 SQL sink ----

def test_28_sink_no_columns_aborts_instead_of_direct_append():
    # The direct-append fallback was deliberately removed: it was non-atomic,
    # duplicated rows on retry, and ignored write_mode. With no generated columns
    # and no reachable destination to reflect them from, the sink must refuse
    # loudly rather than fall back. (This test previously pinned the fallback.)
    with pytest.raises(RuntimeError, match="insertable columns"):
        PostgresDataSink("postgresql://u:p@h/db", "target", columns=None)


def test_29_sink_identifier_quoting():
    pg = PostgresDataSink.__new__(PostgresDataSink)   # bypass __init__/DB
    assert pg._qi('my"tbl') == '"my""tbl"'
    my = MySQLDataSink.__new__(MySQLDataSink)
    assert my._qi("t") == "`t`"


class _FakeConn:
    def __init__(self, log): self._log = log
    def exec_driver_sql(self, sql): self._log.append(sql)

class _FakeCtx:
    def __init__(self, log): self._log = log
    def __enter__(self): return _FakeConn(self._log)
    def __exit__(self, *a): return False

def test_30_sink_overwrite_builds_atomic_swap_sql():
    from daft.io.sink import WriteResult
    executed = []
    fake_engine = MagicMock()
    fake_engine.begin.side_effect = lambda: _FakeCtx(executed)
    with patch.object(PostgresDataSink, "_engine", return_value=fake_engine):
        sink = PostgresDataSink(
            "postgresql://u:p@h/db", "orders",
            write_mode="overwrite", columns=["longitude", "latitude"],
        )
        assert sink._use_staging is True
        # At least one staged row: overwrite with 0 rows now (deliberately) refuses
        # rather than clearing the target — that guard has its own tests.
        sink.finalize([WriteResult(result={"status": "success"}, rows_written=1, bytes_written=0)])
    joined = " ".join(executed)
    assert 'CREATE TABLE "_dta_stg' in joined                  # staging cloned from target
    assert 'DELETE FROM "orders"' in joined                    # overwrite clears target
    assert 'INSERT INTO "orders" ("longitude", "latitude") SELECT "longitude", "latitude" FROM "_dta_stg' in joined

def test_31_sink_append_does_not_delete():
    executed = []
    fake_engine = MagicMock()
    fake_engine.begin.side_effect = lambda: _FakeCtx(executed)
    with patch.object(PostgresDataSink, "_engine", return_value=fake_engine):
        sink = PostgresDataSink(
            "postgresql://u:p@h/db", "orders",
            write_mode="append", columns=["longitude"],
        )
        sink.finalize([])
    joined = " ".join(executed)
    assert "DELETE FROM" not in joined                         # append never clears the target
    assert 'INSERT INTO "orders"' in joined
