"""Tests for the DTA output-format writers and auto-create-table (engine level).

Covers the Daft-dtype → SQL DDL mapping, the generated write body for every
destination format (csv/tsv/json/parquet/xlsx/xml/avro/orc), and the Step-9
"create the table from the transformed schema" path. These use the REAL
data_transfer_agent module (unlike test_dta_wizard, which stubs the pipeline).
"""
from unittest.mock import patch, MagicMock

import pytest

from app.agents.data_transfer_agent.data_transfer_agent import (
    _daft_schema_to_sql_ddl,
    _sql_type_for_daft_dtype,
    make_injection_script,
    data_transfer_pipeline,
)
from app.agents.data_transfer_agent.dta_state import cloud_storage_credentials


# ── Daft-dtype → SQL DDL ──────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype,pg,my", [
    ("Int64", "BIGINT", "BIGINT"),
    ("Float64", "DOUBLE PRECISION", "DOUBLE"),
    ("Utf8", "TEXT", "TEXT"),
    ("Boolean", "BOOLEAN", "TINYINT(1)"),
    ("Date", "DATE", "DATE"),
    ("Timestamp(us)", "TIMESTAMP", "DATETIME"),   # must NOT be mistaken for TIME
    ("Time(us)", "TIME", "TIME"),
])
def test_sql_type_for_daft_dtype(dtype, pg, my):
    assert _sql_type_for_daft_dtype(dtype, "postgresql") == pg
    assert _sql_type_for_daft_dtype(dtype, "mysql") == my


def test_daft_schema_to_sql_ddl_quotes_and_types():
    sch = {"id": "Int64", "price": "Float64", "name": "Utf8"}
    pg = _daft_schema_to_sql_ddl("t", sch, "postgresql")
    my = _daft_schema_to_sql_ddl("t", sch, "mysql")
    assert pg == 'CREATE TABLE "t" ("id" BIGINT, "price" DOUBLE PRECISION, "name" TEXT)'
    assert my == 'CREATE TABLE `t` (`id` BIGINT, `price` DOUBLE, `name` TEXT)'


def test_daft_schema_to_sql_ddl_rejects_empty():
    with pytest.raises(ValueError):
        _daft_schema_to_sql_ddl("t", {}, "postgresql")


# ── Generated write body per format ───────────────────────────────────────────

def _script(fmt, ext, cloud=False):
    fs = {
        "source_type": "csv", "destination_type": fmt,
        "source_file": "in.csv", "destination_file": f"out.{ext}",
        "coder_definition": {"generated_code": "def transform_data(df):\n    return df",
                             "generated_output_schema": "{}"},
        "write_mode": "overwrite",
    }
    if cloud:
        fs["destination_cloud_credentials"] = cloud_storage_credentials(
            provider="gcp", bucket_name="b", file_path=f"out.{ext}")
    return make_injection_script(fs)


@pytest.mark.parametrize("fmt,needle", [
    ("csv", 'df.write_csv("out.csv"'),
    ("parquet", 'df.write_parquet("out.parquet"'),
    ("tsv", '.to_csv("out.tsv", sep="\\t", index=False)'),
    ("xlsx", '.to_excel("out.xlsx", index=False)'),
    ("xml", '.to_xml("out.xml", index=False)'),
    ("orc", '.to_orc("out.orc", index=False)'),
    ("avro", '_write_avro(_pdf, "out.avro")'),
])
def test_local_write_body(fmt, needle):
    s = _script(fmt, fmt)
    assert needle in s
    import ast
    ast.parse(s)                       # generated script must be valid Python


def test_avro_helper_only_emitted_for_avro():
    assert "def _write_avro(pdf, path):" in _script("avro", "avro")
    assert "def _write_avro" not in _script("csv", "csv")


@pytest.mark.parametrize("fmt", ["tsv", "xlsx", "xml", "orc", "avro"])
def test_cloud_write_body_materializes_and_uploads(fmt):
    s = _script(fmt, fmt, cloud=True)
    assert "_pdf = df.to_pandas()" in s
    assert "_upload_file_to_cloud(combined_path" in s
    assert f"output.{fmt}" in s
    import ast
    ast.parse(s)


# ── Step-9 auto-create the destination table from the transformed schema ──────

@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent._create_destination_table")
@patch("app.agents.data_transfer_agent.data_transfer_agent.make_injection_script", return_value="SCRIPT")
@patch("app.agents.data_transfer_agent.data_transfer_agent.daft_code_generation_pipeline")
@patch("app.agents.data_transfer_agent.data_transfer_agent._read_daft_df")
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
@patch("app.agents.data_transfer_agent.data_transfer_agent.probe_db_connection")
def test_create_if_missing_creates_table_from_transformed_schema(
    mock_probe, mock_deduce, mock_read, mock_codegen, mock_inject, mock_create
):
    import pandas as pd

    # Source table exists; destination table 'new_tbl' does not (deduce → None).
    def _deduce(data_type, conn_str, table=None, io_config=None):
        return None if table == "new_tbl" else {"a": "Int64"}
    mock_deduce.side_effect = _deduce
    mock_read.return_value.to_pandas.return_value = pd.DataFrame({"a": [1], "b": [2.0]})
    mock_codegen.side_effect = lambda s: {
        **s, "code_generated_successfully": True,
        "coder_definition": {
            "generated_code": "def transform_data(df):\n    return df.select('a', 'b')",
            "generated_output_schema": {"a": "Int64", "b": "Float64"},
        },
    }
    mock_create.return_value = 'CREATE TABLE "new_tbl" ("a" BIGINT, "b" DOUBLE PRECISION)'

    src = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p", "table": "src"}
    dst = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p", "table": "new_tbl"}

    state, script = data_transfer_pipeline(
        user_prompt="filter and transfer",
        source_type="postgresql", destination_type="postgresql",
        source_credentials=src, destination_credentials=dst,
        create_if_missing=True,
    )

    mock_create.assert_called_once()
    # The transformed schema (a+b) — NOT the source — is what gets created.
    _, kwargs = mock_create.call_args
    passed_schema = mock_create.call_args[0][2] if len(mock_create.call_args[0]) > 2 else kwargs.get("schema")
    assert set(passed_schema.keys()) == {"a", "b"}
    assert state.get("created_destination_table", {}).get("table") == "new_tbl"
    assert state.get("destination_schema_compatible") is True
    assert script == "SCRIPT"


@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.daft_code_generation_pipeline")
@patch("app.agents.data_transfer_agent.data_transfer_agent._read_daft_df")
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
@patch("app.agents.data_transfer_agent.data_transfer_agent.probe_db_connection")
def test_missing_table_without_create_flag_still_asks(
    mock_probe, mock_deduce, mock_read, mock_codegen
):
    import pandas as pd
    def _deduce(data_type, conn_str, table=None, io_config=None):
        return None if table == "ghost" else {"a": "Int64"}
    mock_deduce.side_effect = _deduce
    mock_read.return_value.to_pandas.return_value = pd.DataFrame({"a": [1]})
    mock_codegen.side_effect = lambda s: {
        **s, "code_generated_successfully": True,
        "coder_definition": {
            "generated_code": "def transform_data(df):\n    return df.where(df['a'] > 0)",
            "generated_output_schema": {"a": "Int64"},
        },
    }
    src = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p", "table": "src"}
    dst = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p", "table": "ghost"}
    state, script = data_transfer_pipeline(
        user_prompt="x", source_type="postgresql", destination_type="postgresql",
        source_credentials=src, destination_credentials=dst,
        create_if_missing=False,
    )
    assert script is None
    assert "does not exist" in (state.get("error_message") or "")


@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.daft_code_generation_pipeline")
@patch("app.agents.data_transfer_agent.data_transfer_agent._read_daft_df")
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
@patch("app.agents.data_transfer_agent.data_transfer_agent.probe_db_connection")
def test_requested_columns_missing_from_source_aborts(mock_probe, mock_deduce, mock_read, mock_codegen):
    """A filter/select on columns absent from the source must FAIL, not silently
    ship the whole untransformed table (the wrong-source-table trap)."""
    import pandas as pd
    mock_deduce.return_value = {"age": "Int64", "workclass": "String"}      # source columns
    mock_read.return_value.to_pandas.return_value = pd.DataFrame({"age": [1], "workclass": ["x"]})
    # Coder no-ops because the requested columns don't exist; pseudocode marks them.
    mock_codegen.side_effect = lambda s: {
        **s, "code_generated_successfully": True,
        "coder_definition": {
            "generated_code": "def transform_data(df):\n    return df",
            "generated_output_schema": {"age": "Int64", "workclass": "String"},
            "coder_pseudocode": (
                "1. Filter rows where median_income > 8.0 [MISSING: median_income]\n"
                "2. Select columns longitude, latitude [MISSING: longitude, latitude]"
            ),
        },
    }
    src = {"host": "h", "port": 3306, "database": "d", "user": "u", "password": "p", "table": "adult_income"}
    state, script = data_transfer_pipeline(
        user_prompt="filter median_income > 8, output longitude, latitude",
        source_type="mysql", destination_type="parquet",
        source_credentials=src, destination_credentials={"file_path": "gs://b/out.parquet"},
    )
    assert script is None
    msg = state.get("error_message") or ""
    assert "don't exist in the source" in msg
    assert "median_income" in msg and "longitude" in msg and "latitude" in msg
    assert "adult_income" in msg                    # names the wrong source table
    assert "age" in msg and "workclass" in msg       # shows the real columns


def test_brief_error_surfaces_the_real_line_not_the_traceback():
    """Customers get one readable line; the full traceback stays in the logs."""
    from app.agents.data_transfer_agent.data_transfer_agent import _brief_error

    tb = (
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in run\n'
        "    do()\n"
        "daft.exceptions.DaftCoreException: DaftError::External Method not implemented"
    )
    assert _brief_error(tb) == (
        "daft.exceptions.DaftCoreException: DaftError::External Method not implemented"
    )
    assert _brief_error("") == "an unknown error occurred"
    assert _brief_error("x" * 500).endswith("…")         # capped
    assert len(_brief_error("x" * 500)) <= 221
