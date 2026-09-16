"""Regression tests for the DTA correctness/robustness fixes.

These are self-contained unit tests — no real database, LLM, GKE cluster,
Docker daemon, or network access. External runtimes are mocked or replaced
with SQLite / in-memory objects.
"""

from pathlib import Path

import pytest
import sqlalchemy as sa
import yaml as pyyaml

from app.agents.data_transfer_agent.dta_state import cloud_storage_credentials, db_credentials
from app.agents.data_transfer_agent.data_transfer_agent import (
    build_connection_string,
    incompatible_column_types,
    make_injection_script,
    static_semantic_validator_node,
    _dtype_category,
)
from app.agents.data_transfer_agent.datasink.sql_sink import BaseSQLDataSink
from app.agents.data_transfer_agent.daft_coder import (
    is_stub_code,
    sanitize_user_prompt_for_transformations,
    _generate_daft_stub_code,
)
from app.agents.data_transfer_agent.daft_execution import execution_agent_node_local

REPO_ROOT = Path(__file__).resolve().parents[1]
RAYJOB_TMPL = REPO_ROOT / "app" / "infra" / "rayjob_data_transfer.yaml"


# ── helpers ──────────────────────────────────────────────────────────────────

def _sqlite_target(tmp_path, ddl):
    """Create a SQLite DB with `ddl` and return its SQLAlchemy URL."""
    url = f"sqlite:///{tmp_path / 't.db'}"
    eng = sa.create_engine(url)
    try:
        with eng.begin() as c:
            c.exec_driver_sql(ddl)
    finally:
        eng.dispose()
    return url


def _rowcount(url, table):
    eng = sa.create_engine(url)
    try:
        with eng.connect() as c:
            return c.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar()
    finally:
        eng.dispose()


def _base_write_state(dest_type, **overrides):
    state = {
        "source_type": "csv",
        "destination_type": dest_type,
        "source_file": "in.csv",
        "destination_file": "out." + dest_type,
        "write_mode": "overwrite",
        "coder_definition": {
            "generated_code": "def transform_data(df):\n    return df",
            "generated_output_schema": "",
        },
    }
    state.update(overrides)
    return state


# ── validators read the DTA schema key (source_schema) ───────────────────────

def test_static_validator_uses_source_schema():
    # Schema supplied only under the DTA key. A bad column must still be caught.
    bad = static_semantic_validator_node({
        "generated_code": "df = df.where(df['nope'] > 1)",
        "source_schema": {"real_col": "Int64"},
    })
    assert bad.get("static_semantic_error") is True

    good = static_semantic_validator_node({
        "generated_code": "df = df.where(df['real_col'] > 1)",
        "source_schema": {"real_col": "Int64"},
    })
    assert not good.get("static_semantic_error")


# ── write_mode threaded into the Daft file writers ───────────────────────────

def test_local_csv_parquet_honor_write_mode():
    csv_script = make_injection_script(_base_write_state("csv"))
    assert 'df.write_csv("out.csv", write_mode="overwrite")' in csv_script

    pq_script = make_injection_script(_base_write_state("parquet"))
    assert 'df.write_parquet("out.parquet", write_mode="overwrite")' in pq_script


# ── cloud CSV/JSON writes are provider-agnostic (not GCS-only) ───────────────

@pytest.mark.parametrize("provider,dest_type,sdk", [
    ("aws", "csv", "boto3"),
    ("azure", "json", "azure.storage.blob"),
    ("gcp", "csv", "google.cloud"),
])
def test_cloud_writes_are_provider_agnostic(provider, dest_type, sdk):
    # Cloud CSV/JSON writes dispatch per provider (not GCS-only): each provider's
    # own SDK is used via the generated _upload_file_to_cloud helper.
    creds = cloud_storage_credentials(provider=provider, bucket_name="b", file_path=f"o.{dest_type}")
    script = make_injection_script(_base_write_state(
        dest_type, destination_cloud_credentials=creds, destination_file=None,
    ))
    assert "_upload_file_to_cloud(" in script
    assert sdk in script


# ── SQL sink is all-or-nothing / retry-safe ──────────────────────────────────

def test_sql_sink_staging_used_and_reflects_columns(tmp_path):
    url = _sqlite_target(tmp_path, "CREATE TABLE tgt (id INTEGER PRIMARY KEY, name TEXT, qty INTEGER)")

    # explicit columns -> staging table, target not written directly
    s = BaseSQLDataSink(url, "tgt", connect_args={}, write_mode="overwrite", columns=["name", "qty"])
    assert s._use_staging is True
    assert s._write_table.startswith("_dta_stg_")


def test_sql_sink_missing_target_raises_instead_of_appending(tmp_path):
    url = _sqlite_target(tmp_path, "CREATE TABLE other (a INTEGER)")
    with pytest.raises(RuntimeError):
        BaseSQLDataSink(url, "does_not_exist", connect_args={}, write_mode="append", columns=["a"])


def test_sql_sink_retry_after_partial_write_does_not_double_insert(tmp_path):
    from daft.recordbatch import MicroPartition
    url = _sqlite_target(tmp_path, "CREATE TABLE tgt (name TEXT, qty INTEGER)")
    p0 = MicroPartition.from_pydict({"name": ["a", "b"], "qty": [1, 2]})
    p1 = MicroPartition.from_pydict({"name": ["c", "d"], "qty": [3, 4]})

    # Attempt 1: write partition 0, then "crash" before finalize.
    sink_a = BaseSQLDataSink(url, "tgt", connect_args={}, write_mode="append", columns=["name", "qty"])
    gen = sink_a.write(iter([p0, p1]))
    next(gen)  # partition 0 -> staging only
    assert _rowcount(url, "tgt") == 0  # nothing committed to target

    # Attempt 2 (retry): fresh sink, full run + finalize.
    sink_b = BaseSQLDataSink(url, "tgt", connect_args={}, write_mode="append", columns=["name", "qty"])
    results = list(sink_b.write(iter([p0, p1])))
    sink_b.finalize(results)
    assert _rowcount(url, "tgt") == 4  # written once, no duplicates


# ── type-aware schema check + stub fallback treated as failure ───────────────

def test_type_conflicts_flagged_conservatively():
    # cross-category conflict caught
    assert incompatible_column_types({"amt": "String"}, {"amt": "Int64"}) == [("amt", "String", "Int64")]
    assert incompatible_column_types({"d": "Date"}, {"d": "String"}) == [("d", "Date", "String")]
    # within-category / lenient cases NOT flagged
    assert incompatible_column_types({"x": "Int32"}, {"x": "Int64"}) == []
    assert incompatible_column_types({"x": "List[Int64]"}, {"x": "String"}) == []
    assert incompatible_column_types({"x": "Boolean"}, {"x": "Int64"}) == []


def test_dtype_category_handles_nested_and_scalars():
    assert _dtype_category("List[Int64]") == "other"      # not "numeric"
    assert _dtype_category("Timestamp(us, None)") == "temporal"
    assert _dtype_category("Decimal128(10,2)") == "numeric"
    assert _dtype_category("String") == "string"


def test_stub_fallback_flags_failure(monkeypatch):
    import app.agents.data_transfer_agent.daft_coder as dc
    monkeypatch.setattr(dc, "coder_llm", None)  # force the stub path
    res = dc.daft_coder_node({"user_prompt": "x", "retry_count": 0, "messages": []})
    assert res.get("stub_fallback_used") is True
    assert res.get("generated_code") == dc._generate_daft_stub_code({})


def test_is_stub_code_detects_noop_structurally():
    # A no-op stub must be recognised even if the stub_fallback_used flag is
    # lost, so it can never ship untransformed data as a success.
    assert is_stub_code(_generate_daft_stub_code({})) is True   # canonical stub
    assert is_stub_code("def main(df):\n    return df") is True  # comment stripped
    assert is_stub_code("") is True and is_stub_code("  \n ") is True and is_stub_code(None) is True
    # A legitimate straight-copy via transform_data is NOT treated as a stub.
    assert is_stub_code("def transform_data(df):\n    return df") is False
    # Real transforming code is never a stub.
    assert is_stub_code("def main(df):\n    return df.with_column('x', df['a'] * 2)") is False


# ── sanitize_user_prompt_for_transformations None-guard ──────────────────────

def test_sanitize_none_llm_passthrough():
    assert sanitize_user_prompt_for_transformations("keep me unchanged", None) == "keep me unchanged"


def test_no_stale_coder_llm_snapshot():
    # The module must not hold a by-value coder_llm; it reads the live
    # daft_coder.coder_llm so an LLM configured after import is picked up.
    import app.agents.data_transfer_agent.data_transfer_agent as dta
    assert not hasattr(dta, "coder_llm")


def test_sanitize_uses_live_coder_llm_after_rebind(monkeypatch):
    # An LLM rebound after import (late-loaded GROQ key) must reach the sanitise
    # step; a stale snapshot would pass the old None instead.
    import app.agents.data_transfer_agent.data_transfer_agent as dta
    import app.agents.data_transfer_agent.daft_coder as dc

    sentinel = object()
    monkeypatch.setattr(dc, "coder_llm", sentinel)

    captured = {}

    class _Stop(Exception):
        pass

    def _fake_sanitize(prompt, llm):
        captured["llm"] = llm
        raise _Stop

    monkeypatch.setattr(dta, "sanitize_user_prompt_for_transformations", _fake_sanitize)

    with pytest.raises(_Stop):
        dta.daft_code_generation_pipeline({"coder_definition": {"user_prompt": "x"}})

    assert captured["llm"] is sentinel


# ── proper credential objects (cloud format inference + DB SSL) ──────────────

def test_cloud_file_type_inference():
    # Cloud file format is derived from the object-path extension (was hardcoded
    # to parquet, which broke CSV/JSON). Unknown → None so callers prompt instead
    # of silently defaulting.
    from app.agents.planner import _file_type_from_path
    assert _file_type_from_path("d/f.csv") == "csv"
    assert _file_type_from_path("d/f.parquet") == "parquet"
    assert _file_type_from_path("x.pq") == "parquet"
    assert _file_type_from_path("noext") is None
    assert _file_type_from_path(None) is None


def test_cloud_uri_and_db_sslmode():
    creds = cloud_storage_credentials(provider="aws", bucket_name="b", file_path="d/f.parquet")
    uri, _ = build_connection_string(creds, "parquet")
    assert uri == "s3://b/d/f.parquet"

    # A plain dict carrying sslmode must produce psycopg2-valid connect_args
    # ({'sslmode': ...}); an ssl.SSLContext is rejected by psycopg2.
    _, connect_args = build_connection_string(
        {"user": "u", "password": "p", "host": "h", "port": 5432, "database": "d", "sslmode": "require"},
        "postgresql",
    )
    assert connect_args == {"sslmode": "require"}


def test_db_credentials_object_forwards_sslmode():
    # A db_credentials object must also carry sslmode into the connect_args,
    # not just plain dicts.
    creds = db_credentials(
        db_type="postgresql", host="h", port=5432, database="d",
        user="u", password="p", sslmode="require",
    )
    _, connect_args = build_connection_string(creds, "postgresql")
    assert connect_args == {"sslmode": "require"}

    # No sslmode → no SSL connect_args (unchanged behaviour).
    plain = db_credentials(db_type="postgresql", host="h", port=5432, database="d", user="u", password="p")
    _, no_ssl = build_connection_string(plain, "postgresql")
    assert no_ssl == {}


def test_verify_full_forwards_sslrootcert():
    # verify-full also forwards sslrootcert (dict path, e.g. from the planner).
    _, connect_args = build_connection_string(
        {"user": "u", "password": "p", "host": "h", "port": 5432,
         "database": "d", "sslmode": "verify-full", "sslrootcert": "/tmp/ca.pem"},
        "postgresql",
    )
    assert connect_args == {"sslmode": "verify-full", "sslrootcert": "/tmp/ca.pem"}


def test_probe_forwards_connect_args(monkeypatch):
    # probe_db_connection must forward connect_args to create_engine, else the
    # SSL settings are silently dropped during the connection probe.
    import sqlalchemy
    from app.agents.data_transfer_agent.data_transfer_agent import probe_db_connection

    captured = {}

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a, **k): return None

    class _Engine:
        def connect(self): return _Conn()

    def _fake_create_engine(url, **kwargs):
        captured["kwargs"] = kwargs
        return _Engine()

    monkeypatch.setattr(sqlalchemy, "create_engine", _fake_create_engine)
    probe_db_connection("postgresql+psycopg2://u:p@h:5432/d", "source", {"sslmode": "require"})
    assert captured["kwargs"].get("connect_args") == {"sslmode": "require"}


def test_url_with_ssl_params_folds_into_query():
    from app.agents.data_transfer_agent.data_transfer_agent import _url_with_ssl_params
    u = "postgresql+psycopg2://u:p@h:5432/d"
    assert _url_with_ssl_params(u, {"sslmode": "require"}) == u + "?sslmode=require"
    assert _url_with_ssl_params(u, None) == u and _url_with_ssl_params(u, {}) == u
    assert _url_with_ssl_params(u + "?x=1", {"sslmode": "require"}) == u + "?x=1&sslmode=require"


def test_register_database_params_carry_sslmode():
    # sslmode must survive registration so a later transfer can build the URL
    # with SSL; previously RegisterDatabaseParams had no such field.
    from app.agents.planner import RegisterDatabaseParams
    params = RegisterDatabaseParams(
        alias="pg", db_type="postgresql", host="h", port=5432,
        database="d", user="u", password="p", sslmode="verify-full",
    )
    assert params.model_dump(exclude_none=True)["sslmode"] == "verify-full"


# ── transfer fast-path alias parsing (case + punctuation) ────────────────────

def test_extract_transfer_aliases_preserves_case_and_strips_punctuation():
    from app.agents.planner import _extract_transfer_aliases
    # mixed-case aliases preserved; trailing '.' stripped
    src, dst, _ = _extract_transfer_aliases("Transfer from SalesDB to Warehouse.")
    assert (src, dst) == ("SalesDB", "Warehouse")
    # connection ids keep exact casing; trailing ',' stripped
    src, dst, _ = _extract_transfer_aliases("copy from C2C_Source_GCP to C2C_Dest_AWS,")
    assert (src, dst) == ("C2C_Source_GCP", "C2C_Dest_AWS")
    # all trigger verbs
    assert _extract_transfer_aliases("move data from A to B")[:2] == ("A", "B")
    assert _extract_transfer_aliases("migrate from A to B")[:2] == ("A", "B")
    # end offset lands right after the "from X to Y" clause (A→B→C table shape)
    _, _, end = _extract_transfer_aliases("transfer from A to B to C")
    assert end == len("transfer from A to B")
    # non-matching input
    assert _extract_transfer_aliases("show me the databases") is None
    assert _extract_transfer_aliases("") is None


# ── DB credentials delivered via env, never embedded in the script ───────────

def _db_to_db_state(src_conn, dst_conn):
    return {
        "source_type": "postgresql",
        "destination_type": "postgresql",
        "source_connection_string": src_conn,
        "destination_connection_string": dst_conn,
        "source_table": "src_tbl",
        "destination_table": "dst_tbl",
        "write_mode": "append",
        "coder_definition": {
            "generated_code": "def transform_data(df):\n    return df",
            "generated_output_schema": "",
        },
    }


def test_injection_script_delivers_db_creds_via_env_not_plaintext():
    src = "postgresql+psycopg2://u:S0urceSecret@dbhost:5432/srcdb"
    dst = "postgresql+psycopg2://u:D3stSecret@dbhost:5432/dstdb"
    state = _db_to_db_state(src, dst)
    script = make_injection_script(state)

    # The script is written to disk / a ConfigMap, so it must contain no
    # plaintext credentials; the conn strings are handed over as env vars.
    assert "S0urceSecret" not in script and "D3stSecret" not in script
    assert src not in script and dst not in script
    assert 'os.environ["DTA_SOURCE_CONN_STR"]' in script
    assert 'os.environ["DTA_DEST_CONN_STR"]' in script
    assert state["runtime_secret_env"] == {
        "DTA_SOURCE_CONN_STR": src,
        "DTA_DEST_CONN_STR": dst,
    }


def test_cloud_uri_source_carries_no_password_into_script():
    # A cloud file location IS interpolated into the script, but it's a bucket
    # URI (no secret) — provider creds arrive via separate env vars.
    state = _base_write_state("csv", source_type="parquet", source_file="s3://bucket/in.parquet")
    script = make_injection_script(state)
    assert "s3://bucket/in.parquet" in script
    assert "runtime_secret_env" in state and "DTA_SOURCE_CONN_STR" not in state["runtime_secret_env"]


def test_cloud_json_source_rejected_before_llm_work():
    # A cloud JSON source is unsupported and must be rejected up front — not
    # after schema deduction, sample fetch, and the whole LLM pipeline. The
    # guard runs before the LLM pre-flight, so it needs no coder LLM configured.
    from app.agents.data_transfer_agent.data_transfer_agent import data_transfer_pipeline
    creds = cloud_storage_credentials(provider="aws", bucket_name="b", file_path="data/in.json")
    final_state, script = data_transfer_pipeline(
        user_prompt="copy it",
        source_type="json",
        destination_type="csv",
        source_credentials=creds,
        destination_file="out.csv",
    )
    assert script is None
    assert "JSON is not supported as a cloud source" in (final_state.get("error_message") or "")


# ── no heavy/unused torch import on the DTA launch path ──────────────────────

def test_dta_launch_path_does_not_import_torch():
    # torch was imported (unused) on the planner-triggered DTA path and crashed
    # the VM's CXXABI environment when launching a transfer.
    import app.rag.daft_retrieval as daft_retrieval
    import app.data_transfer_docker.gke_run as gke_run
    assert not hasattr(daft_retrieval, "torch")
    assert not hasattr(gke_run, "torch")


# ── state schema key matches the runtime key ─────────────────────────────────

def test_dta_state_declares_error_message_not_plural():
    # The runtime sets/reads error_message; the schema must declare that exact
    # key, not the dead error_messages, or a reader of the schema sees nothing.
    from app.agents.data_transfer_agent.dta_state import DaftETLState
    ann = DaftETLState.__annotations__
    assert "error_message" in ann
    assert "error_messages" not in ann


# ── RAG: real embeddings existence check, no dead single-line helper ─────────

def test_load_chroma_collection_raises_on_missing_db(tmp_path):
    # A missing embeddings store must raise instead of silently creating an
    # empty DB that returns nothing from retrieval.
    from app.rag.daft_retrieval import load_chroma_collection
    missing = tmp_path / "no_such_db"
    with pytest.raises(FileNotFoundError):
        load_chroma_collection(db_path=missing)
    # And no empty DB is created as a side effect.
    assert not (missing / "chroma.sqlite3").exists()


def test_retrieve_docs_for_single_line_helper_removed():
    import app.rag.daft_retrieval as daft_retrieval
    assert not hasattr(daft_retrieval, "retrieve_docs_for_single_line")


# ── deduce_schema: absent object vs transient read error ─────────────────────

def test_is_missing_object_error_classifier():
    from app.agents.data_transfer_agent.data_transfer_agent import _is_missing_object_error
    # DB: genuinely absent
    assert _is_missing_object_error(Exception('relation "t" does not exist'), "postgresql") is True
    assert _is_missing_object_error(Exception("Table 'x' doesn't exist"), "mysql") is True
    # DB: transient / permission / host errors are NOT "absent"
    assert _is_missing_object_error(Exception("name or service not found"), "postgresql") is False
    assert _is_missing_object_error(Exception("connection timed out"), "postgresql") is False
    assert _is_missing_object_error(Exception("permission denied for table t"), "postgresql") is False
    # File: absent (including the FileNotFoundError type)
    assert _is_missing_object_error(FileNotFoundError("x"), "csv") is True
    assert _is_missing_object_error(Exception("NoSuchKey: does not exist"), "parquet") is True
    assert _is_missing_object_error(Exception("AccessDenied"), "parquet") is False


def test_deduce_schema_reraises_read_error_but_empty_on_missing(monkeypatch):
    import app.agents.data_transfer_agent.data_transfer_agent as dta

    def _missing(*a, **k):
        raise Exception('relation "t" does not exist')

    def _read_error(*a, **k):
        raise RuntimeError("connection reset by peer")

    # Absent object → empty schema (a legitimate "new table/file"), not an error.
    monkeypatch.setattr(dta, "_read_daft_df", _missing)
    assert dta.deduce_schema("postgresql", "conn", table="t") is None
    assert dta.deduce_schema("csv", "path") == {}

    # Transient read error → re-raised so the caller aborts instead of silently
    # treating it as a new table and skipping the schema-compatibility check.
    monkeypatch.setattr(dta, "_read_daft_df", _read_error)
    with pytest.raises(RuntimeError):
        dta.deduce_schema("postgresql", "conn", table="t")


# ── logic-review findings are surfaced (not silently dropped) ────────────────

def test_logic_review_non_strict_records_finding_without_blocking():
    from app.agents.data_transfer_agent.daft_validator import _parse_review_response

    class _Resp:
        def __init__(self, content): self.content = content

    flagged = _Resp('{"is_logically_correct": false, "rationale": "drops the Price column"}')
    approved = _Resp('{"is_logically_correct": true}')

    # Non-strict (default): does NOT block, but records the finding for surfacing.
    r = _parse_review_response(flagged, strict_mode=False)
    assert r["logical_semantic_error"] is False
    assert r.get("logical_review_feedback") == "drops the Price column"
    # Strict: blocks so the repair loop fires.
    assert _parse_review_response(flagged, strict_mode=True)["logical_semantic_error"] is True
    # Approved: no finding recorded.
    ok = _parse_review_response(approved, strict_mode=False)
    assert ok["logical_semantic_error"] is False and "logical_review_feedback" not in ok


def test_surface_logic_review_warning_reaches_user():
    from app.agents.data_transfer_agent.data_transfer_agent import _surface_logic_review_warning

    # An advisory finding must be appended to warnings (previously dropped).
    fs = {"coder_definition": {"logical_review_feedback": "drops the Price column"}}
    _surface_logic_review_warning(fs)
    assert any("drops the Price column" in w for w in fs["warnings"])

    # No finding → no warning added; existing warnings preserved.
    fs2 = {"coder_definition": {}}
    _surface_logic_review_warning(fs2)
    assert not fs2.get("warnings")
    fs3 = {"coder_definition": {"logical_review_feedback": "x"}, "warnings": ["pre"]}
    _surface_logic_review_warning(fs3)
    assert fs3["warnings"][0] == "pre" and len(fs3["warnings"]) == 2


# ── empty-source / no-sample execution path flags failure ────────────────────

def test_no_sample_data_sets_execution_error():
    out = execution_agent_node_local({
        "coder_definition": {
            "generated_code": "def main(df):\n    return df",
            "input_sample_data": None,
        },
        "messages": [],
    })
    assert out["coder_definition"].get("execution_error")
    assert out.get("code_executed") is False


def test_execution_error_surfaces_real_cause_not_generic(monkeypatch):
    import app.agents.data_transfer_agent.daft_execution as ex

    def _state():
        return {
            "coder_definition": {
                "generated_code": "def main(df):\n    return df",
                "input_sample_data": [{"a": 1}],  # non-None so execution runs
            },
            "messages": [],
        }

    # Timeout: execution_error set, stderr empty → real cause must surface.
    monkeypatch.setattr(ex, "execute_code_on_local", lambda code, sample_data: {
        "status": "error",
        "execution_error": "Timeout Error: took too long to generate a preview.",
        "execution_stderr": "",
    })
    assert "Timeout Error" in ex.execution_agent_node_local(_state())["coder_definition"]["execution_error"]

    # Serialization failure: only execution_error set (no stderr key at all).
    monkeypatch.setattr(ex, "execute_code_on_local", lambda code, sample_data: {
        "status": "error",
        "execution_error": "System Error: Failed to serialize sample data: boom",
    })
    got = ex.execution_agent_node_local(_state())["coder_definition"]["execution_error"]
    assert "Failed to serialize sample data" in got

    # No detail available → generic fallback preserved.
    monkeypatch.setattr(ex, "execute_code_on_local", lambda code, sample_data: {"status": "error"})
    assert ex.execution_agent_node_local(_state())["coder_definition"]["execution_error"] == \
        "Unknown execution error occurred."


# ── AWS/Azure creds mapped to pod env + valid RayJob render ───────────────────

def test_render_cloud_env_vars():
    from app.data_transfer_docker.gke_run import _render_cloud_env_vars
    assert _render_cloud_env_vars({}) == ""
    assert _render_cloud_env_vars(None) == ""
    out = _render_cloud_env_vars({"AWS_ACCESS_KEY_ID": "ak", "AWS_REGION": "us-east-1"})
    assert 'AWS_ACCESS_KEY_ID: "ak"' in out
    assert 'AWS_REGION: "us-east-1"' in out
    assert out.startswith("      ")  # 6-space indent: nests under env_vars:


def test_rayjob_injects_cloud_env_vars_and_renders_valid_yaml():
    from app.data_transfer_docker.gke_run import _render_cloud_env_vars
    tmpl = RAYJOB_TMPL.read_text()
    rendered = (
        tmpl.replace("__RAYJOB_NAME__", "dta-x")
        .replace("__JOB_ID__", "x")
        .replace("__GCS_SECRET_NAME__", "gcs-sa-key")
        .replace("    __INJECTION_SCRIPT__", "    pass")
        .replace("      __CLOUD_ENV_VARS__", _render_cloud_env_vars({"AWS_ACCESS_KEY_ID": "ak"}))
    )
    assert "__" not in rendered  # no leftover placeholders
    docs = list(pyyaml.safe_load_all(rendered))
    # runtimeEnvYAML is a nested YAML string; the cloud var must land under env_vars.
    inner = pyyaml.safe_load(docs[0]["spec"]["runtimeEnvYAML"])
    assert inner["env_vars"]["AWS_ACCESS_KEY_ID"] == "ak"


def test_runtime_requirements_has_no_pathlib_backport():
    # The PyPI 'pathlib' is a py2-era backport that shadows/breaks the py3.11
    # stdlib module; it must never be pinned in the runtime deps.
    reqs = (REPO_ROOT / "app" / "data_transfer_docker" / "runtime-requirements.txt").read_text()
    pkgs = [
        line.split("#")[0].strip().split("==")[0].split(">")[0].split("<")[0].strip()
        for line in reqs.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "pathlib" not in pkgs


# ── Docker runtime receives cloud credentials ─────────────────────────────────

def _fake_docker_client(captured):
    class FakeContainer:
        name = "c"
        def wait(self):
            return {"StatusCode": 0}
        def logs(self):
            return b"AVALOKA_RESULT rows_inserted=1 table=t"
        def remove(self, force=False):
            pass

    class FakeContainers:
        def run(self, **kw):
            captured.update(kw)
            return FakeContainer()

    class FakeClient:
        containers = FakeContainers()

    return FakeClient()


def test_docker_passes_aws_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # keep pipeline_runs/ out of the repo
    import app.data_transfer_docker.docker_run as dr
    captured = {}
    monkeypatch.setattr(dr.docker, "from_env", lambda: _fake_docker_client(captured))
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    cloud_env = {"AWS_ACCESS_KEY_ID": "AK", "AWS_SECRET_ACCESS_KEY": "SK", "AWS_REGION": "us-east-1"}
    dr.launch_docker_pipeline("print(1)", job_id="j1", cloud_env=cloud_env)

    env = captured["environment"]
    assert all(env[k] == v for k, v in cloud_env.items())
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env


def test_docker_writes_gcp_sa_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # keep pipeline_runs/ out of the repo
    import app.data_transfer_docker.docker_run as dr
    captured = {}
    monkeypatch.setattr(dr.docker, "from_env", lambda: _fake_docker_client(captured))

    dr.launch_docker_pipeline(
        "print(1)", job_id="j2", cloud_env={}, gcp_sa_json={"type": "service_account"},
    )

    # The SA key is written into the mounted job dir and pointed at via the env var.
    env = captured["environment"]
    assert env["GOOGLE_APPLICATION_CREDENTIALS"] == "/mnt/gcs/transfers/j2/gcs_sa.json"
    assert (tmp_path / "pipeline_runs" / "j2" / "gcs_sa.json").is_file()


# ── cloud->cloud destination write uses the destination's own SA ─────────────

def _exec_gcp_upload_helper(monkeypatch, dest_sa_env):
    """Exec the generated GCP _upload_file_to_cloud helper against a fake
    google.cloud.storage, returning what it did."""
    import sys, types
    from app.agents.data_transfer_agent.data_transfer_agent import _make_cloud_upload_helper_code

    calls = {"ambient": 0, "from_info": None, "bucket": None, "blob": None, "uploaded": None}

    class _FakeClient:
        def __init__(self):
            calls["ambient"] += 1
        @classmethod
        def from_service_account_info(cls, info):
            calls["from_info"] = info
            return cls.__new__(cls)  # don't count as an ambient Client()
        def bucket(self, name):
            calls["bucket"] = name
            return self
        def blob(self, path):
            calls["blob"] = path
            return self
        def upload_from_filename(self, local_path):
            calls["uploaded"] = local_path

    storage_mod = types.ModuleType("google.cloud.storage")
    storage_mod.Client = _FakeClient
    cloud_mod = types.ModuleType("google.cloud")
    cloud_mod.storage = storage_mod
    google_mod = types.ModuleType("google")
    google_mod.cloud = cloud_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_mod)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_mod)

    if dest_sa_env is None:
        monkeypatch.delenv("DEST_GCP_SA_JSON", raising=False)
    else:
        monkeypatch.setenv("DEST_GCP_SA_JSON", dest_sa_env)

    ns = {}
    exec(_make_cloud_upload_helper_code("gcp"), ns)
    ns["_upload_file_to_cloud"]("/tmp/output.json", "dest-bucket", "transfers/out.json")
    return calls


def test_gcp_upload_uses_dest_sa_when_env_set(monkeypatch):
    import json
    info = {"client_email": "dst@proj.iam.gserviceaccount.com", "type": "service_account"}
    calls = _exec_gcp_upload_helper(monkeypatch, json.dumps(info))
    # Wrote as the destination SA, never the ambient (source) identity.
    assert calls["from_info"] == info
    assert calls["ambient"] == 0
    assert calls["bucket"] == "dest-bucket"
    assert calls["blob"] == "transfers/out.json"
    assert calls["uploaded"] == "/tmp/output.json"


def test_gcp_upload_ambient_fallback_without_env(monkeypatch):
    calls = _exec_gcp_upload_helper(monkeypatch, None)
    # No destination SA -> fall back to ambient GOOGLE_APPLICATION_CREDENTIALS.
    assert calls["from_info"] is None
    assert calls["ambient"] == 1
    assert calls["uploaded"] == "/tmp/output.json"


def test_gke_run_forwards_dest_sa(monkeypatch):
    import app.data_transfer_docker.gke_run as gk
    captured = {}

    class _Outcome:
        status = "SUCCEEDED"
        logs = "AVALOKA_RESULT rows_inserted=1 table=transfers/out.json"
        details = {}

    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://fake:8265")
    monkeypatch.setattr(gk, "run_rayjob_from_yaml", lambda **kw: (captured.update(kw), _Outcome())[1])

    gk.launch_gke_pipeline("print(1)", job_id="jb", gcp_sa_json="SRC", dest_gcp_sa_json="DST")

    cc = captured["cloud_creds"]
    assert cc["gcp_service_account_json"] == "SRC"
    assert cc["dest_gcp_service_account_json"] == "DST"


def test_docker_sets_dest_sa_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # keep pipeline_runs/ out of the repo
    import json
    import app.data_transfer_docker.docker_run as dr
    captured = {}
    monkeypatch.setattr(dr.docker, "from_env", lambda: _fake_docker_client(captured))

    dr.launch_docker_pipeline(
        "print(1)", job_id="jb", cloud_env={},
        gcp_sa_json={"type": "service_account"},
        dest_gcp_sa_json={"client_email": "dst@proj.iam.gserviceaccount.com"},
    )

    env = captured["environment"]
    assert json.loads(env["DEST_GCP_SA_JSON"])["client_email"] == "dst@proj.iam.gserviceaccount.com"
