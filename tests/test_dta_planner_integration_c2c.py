"""Integration tests for cloud transfers driven through the PLANNER / chat flow.

The existing ``test_dta_c2c/c2d/d2c`` tests call ``data_transfer_pipeline`` directly
with hand-built ``cloud_storage_credentials`` — they bypass the planner and so never
exercised the application integration that was actually broken ("Source not
registered", DB-only credentials, forced-parquet type).

These tests drive ``_handle_register_database`` + ``_handle_initiate_transfer`` and
assert the planner now:
  * resolves session-registered cloud buckets (no "not registered"),
  * builds a real ``cloud_storage_credentials`` for cloud endpoints,
  * derives the file type from the object path (csv/json/parquet, not parquet),
  * passes the cloud URIs as source_file/destination_file,
  * injects S3/Azure runtime creds into the runner via ``cloud_env``,
  * prompts for an object path when none is known and rejects JSON cloud sources.

The pipeline and the GKE/Docker runners are stubbed so the tests run without a
Groq key, torch, Docker or a live cluster.
"""
import sys
import types

import pytest


# ── Stub the execution layer BEFORE planner resolves its lazy imports ─────────
# ``_handle_initiate_transfer`` does `from ...data_transfer_agent import
# data_transfer_pipeline` and `from ...gke_run import launch_gke_pipeline` at call
# time; installing stubs in sys.modules makes those resolve to our spies. The real
# modules pull in torch/daft/docker, which aren't importable in every environment.
_CALLS = {"pipeline": [], "gke": [], "docker": []}
# Fake Supabase cloud_datasets rows, keyed by connection id (populated per-test).
_CLOUD_CONNS = {}


@pytest.fixture(autouse=True)
def _stub_execution_layer(monkeypatch):
    _CALLS["pipeline"].clear()
    _CALLS["gke"].clear()
    _CALLS["docker"].clear()
    _CLOUD_CONNS.clear()

    # Stub the Supabase-backed cloud connection lookup (UI 'Cloud Dataset' store).
    cc_mod = types.ModuleType("app.api.cloud_connections")

    async def _fake_get_cloud_connection(connection_id):
        if connection_id not in _CLOUD_CONNS:
            raise RuntimeError(f"cloud connection not found: {connection_id}")
        return dict(_CLOUD_CONNS[connection_id])

    async def _fake_list_cloud_connections(user_id):
        # Name-based resolution lists the user's own connections; the id is the dict
        # key, folded back into each row so name→id matching can report it.
        return [dict(id=cid, **row) for cid, row in _CLOUD_CONNS.items()]

    cc_mod.get_cloud_connection = _fake_get_cloud_connection
    cc_mod.list_cloud_connections = _fake_list_cloud_connections
    monkeypatch.setitem(sys.modules, "app.api.cloud_connections", cc_mod)

    dta_mod = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")

    def _fake_pipeline(**kwargs):
        _CALLS["pipeline"].append(kwargs)
        # Non-empty script + no error → planner proceeds to execution.
        return ({"error_message": None, "warnings": None}, "print('generated script')")

    dta_mod.data_transfer_pipeline = _fake_pipeline
    monkeypatch.setitem(
        sys.modules, "app.agents.data_transfer_agent.data_transfer_agent", dta_mod
    )

    # A unit test can't launch a real GKE Ray cluster / Docker container, so the
    # launchers are intercepted purely to RECORD what they were called with (to
    # verify credential/env injection). The spy fabricates no data — it reports a
    # bare successful launch and no row count. Row-count/message FORMATTING is
    # covered separately by the pure test of `_format_transfer_success`.
    from app.data_transfer_docker.run_result import TransferResult

    gke_mod = types.ModuleType("app.data_transfer_docker.gke_run")

    def _gke_launch_spy(injection_script, job_id=None, cloud_env=None, gcp_sa_json=None, dest_gcp_sa_json=None):
        _CALLS["gke"].append({
            "job_id": job_id, "cloud_env": cloud_env,
            "gcp_sa_json": gcp_sa_json, "dest_gcp_sa_json": dest_gcp_sa_json,
        })
        return TransferResult(success=True)

    gke_mod.launch_gke_pipeline = _gke_launch_spy
    monkeypatch.setitem(sys.modules, "app.data_transfer_docker.gke_run", gke_mod)

    docker_mod = types.ModuleType("app.data_transfer_docker.docker_run")

    def _docker_launch_spy(injection_script, job_id=None, extra_volumes=None, cloud_env=None,
                           gcp_sa_json=None, dest_gcp_sa_json=None):
        _CALLS["docker"].append({"job_id": job_id, "cloud_env": cloud_env})
        return TransferResult(success=True)

    docker_mod.launch_docker_pipeline = _docker_launch_spy
    monkeypatch.setitem(sys.modules, "app.data_transfer_docker.docker_run", docker_mod)

    # Always run the GKE path deterministically (skip the VPC-private → docker guard)
    # and avoid any real onboarding-API network calls in credential fallback.
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setenv("DTA_GKE_REACHES_PRIVATE", "1")

    import app.agents.planner as planner

    class _NoNetwork:
        """requests replacement whose Session().get always raises → forces the
        registry-entry fallback in _fetch_full_creds without touching the network."""

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, *a, **k):
                raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(planner, "requests", _NoNetwork)
    # A named DB destination now reflects the table list to offer a create when
    # absent. With no live DB here, return None so the existence pre-check is
    # skipped (tests reach the pipeline as before, with no slow connection attempt).
    monkeypatch.setattr(planner, "_list_dest_tables", lambda creds, db_type: None)
    yield


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fresh_state():
    return {"dta_database_registry": {}, "messages": []}


def _register(state, **kwargs):
    import app.agents.planner as planner
    params = planner.RegisterDatabaseParams(**kwargs)
    return planner._handle_register_database(state, params)


def _transfer(state, **kwargs):
    import app.agents.planner as planner
    params = planner.InitiateTransferParams(**kwargs)
    return planner._handle_initiate_transfer(state, params)


def _last_pipeline_call():
    assert _CALLS["pipeline"], "data_transfer_pipeline was not called"
    return _CALLS["pipeline"][-1]


# ── Unit-level helpers ────────────────────────────────────────────────────────

def test_file_type_from_path():
    import app.agents.planner as planner
    assert planner._file_type_from_path("a/b.csv") == "csv"
    assert planner._file_type_from_path("out.JSON") == "json"
    assert planner._file_type_from_path("data.parquet") == "parquet"
    assert planner._file_type_from_path("data.pq") == "parquet"
    assert planner._file_type_from_path("noext") is None
    assert planner._file_type_from_path(None) is None


def test_dest_table_extraction_with_and_without_table_keyword():
    import re
    import app.agents.planner as planner

    def resolve(prompt):
        m = re.search(
            r"(?:transfer|move\s+data|copy|migrate)\s+from\s+(\S+)\s+to\s+(\S+)",
            prompt, re.IGNORECASE,
        )
        assert m, prompt
        t = planner._extract_dest_table_from_prompt(prompt)
        if not t:
            t = planner._extract_dest_table_after_aliases(prompt, m.end())
        return t

    # "table X" keyword forms still work.
    assert resolve("transfer from A to B to table housing_high_income, Output x") == "housing_high_income"
    # "A to B to C" (no "table" keyword) now resolves to C.
    assert resolve("transfer from A to B to housing_high_income, Output x") == "housing_high_income"
    assert resolve("transfer from A to B into results_2024") == "results_2024"
    # The dest alias itself is never mistaken for a table, and filler is ignored.
    assert resolve("transfer from A to B, output the columns") is None
    assert resolve("transfer from A to B output the columns") is None


def test_format_transfer_success_is_fully_dynamic():
    import app.agents.planner as planner

    # Cloud→cloud: row count, source object and destination bucket are whatever we pass.
    msg = planner._format_transfer_success(
        src_label="C2C_Source_GCP → my-src-bucket/prefix",
        dst_label="C2C_Destination_GCP → my-dst-bucket/transfers",
        kind="Cloud to Cloud", rows_inserted=1234567,
        src_object="prefix/patient_data.csv", dest_ref="transfers/out.json",
        dst_is_cloud=True, write_mode="append", runner_label="GKE Ray cluster", job_id="dta-abc",
    )
    assert "🎉 **SUCCESS! Cloud to Cloud Transfer Complete!**" in msg
    assert "`C2C_Source_GCP → my-src-bucket/prefix`" in msg
    assert "`C2C_Destination_GCP → my-dst-bucket/transfers`" in msg
    # 1,234,567 is thousands-formatted and NOT hardcoded anywhere.
    assert "Successfully inserted 1,234,567 rows from source `prefix/patient_data.csv` into destination bucket `transfers/out.json`" in msg

    # Database dest → "table", singular noun for 1 row.
    db = planner._format_transfer_success(
        src_label="mysql_src", dst_label="pg_dst", kind="Database to Database",
        rows_inserted=1, src_object=None, dest_ref="housing_high_income",
        dst_is_cloud=False, write_mode="overwrite", runner_label="Local Docker", job_id="dta-xyz",
    )
    assert "Successfully inserted 1 row into destination table `housing_high_income`" in db
    assert "into destination bucket" not in db

    # No count surfaced → no row line at all (never a fabricated number).
    none = planner._format_transfer_success(
        src_label="a", dst_label="b", kind="Cloud to Cloud", rows_inserted=None,
        src_object=None, dest_ref="x", dst_is_cloud=True, write_mode="append",
        runner_label="GKE Ray cluster", job_id="dta-0",
    )
    assert "Successfully inserted" not in none


def test_object_path_extraction_and_uri_stripping():
    import app.agents.planner as planner
    src = planner._extract_object_path_from_prompt(
        "transfer from data/in.csv to out/result.json", "source"
    )
    dst = planner._extract_object_path_from_prompt(
        "transfer from data/in.csv to out/result.json", "dest"
    )
    assert src == "data/in.csv"
    assert dst == "out/result.json"
    # Full cloud URIs are reduced to the in-bucket key.
    assert planner._strip_cloud_uri("gs://bucket/a/b.csv") == "a/b.csv"
    assert (
        planner._extract_object_path_from_prompt("read gs://b/a/b.csv please", "source")
        == "a/b.csv"
    )


def test_destination_extraction_stops_before_to_table():
    """"transfer from A to B to table C" — the destination is B, NOT "B to". The
    trailing " to" used to leak into the alias and break the DB destination lookup
    (which then fell through to the cloud-name resolver and listed cloud connections
    for a database destination)."""
    import app.agents.planner as planner

    d = planner._extract_transfer_destination(
        "transfer from localmysqldestinationdemotest to localmypostgressqldestinationdemobug "
        "to table housing_high_income , Output the 'longitude' columns."
    )
    assert d is not None
    assert d[0] == "localmypostgressqldestinationdemobug"     # no trailing " to"
    # The table is still recovered separately.
    assert planner._extract_dest_table_from_prompt(
        "transfer from a to b to table housing_high_income"
    ) == "housing_high_income"

    # Implicit-source and other phrasings are unaffected.
    assert planner._extract_transfer_destination(
        "transfer to My_Warehouse, to file exports/result.parquet"
    )[0] == "My_Warehouse"
    assert planner._extract_transfer_destination(
        "transfer to warehouse_db to table results_2024"
    )[0] == "warehouse_db"


def test_named_source_transfer_parses_human_c2c_prompt():
    """The human c2c phrasing — connection NAMES for both endpoints plus explicit
    source/destination objects — is parsed into a fully-populated transfer."""
    import app.agents.planner as planner

    p = planner._extract_named_source_transfer(
        "transfer from C2C_Source_GCP to C2C_Destination_GCP, "
        "from patient_data.csv to transfers/2019_nov_data_transformed.json. "
        "Keep all rows. Add a column 'billing_category'."
    )
    assert p is not None
    assert p.source_alias == "C2C_Source_GCP"
    assert p.destination_alias == "C2C_Destination_GCP"
    assert p.source_object == "patient_data.csv"
    assert p.dest_object == "transfers/2019_nov_data_transformed.json"


def test_named_source_parser_ignores_object_only_and_implicit_forms():
    """Only the two-named-connections shape takes the explicit path; object-only
    ("from a.csv to b.json") and implicit ("transfer to <dest>") stay on the
    implicit-source path (parser returns None)."""
    import app.agents.planner as planner

    # Object-only: both tokens are files → not a named-connection transfer.
    assert planner._extract_named_source_transfer(
        "transfer from data/in.csv to out/result.json"
    ) is None
    # Named connections but NO source object → implicit-source grammar.
    assert planner._extract_named_source_transfer(
        "transfer from C2C_Source_GCP to C2C_Destination_GCP"
    ) is None
    # Pure implicit destination-only form.
    assert planner._extract_named_source_transfer(
        "transfer to My_Warehouse, to file exports/result.parquet"
    ) is None


# ── C2C / C2D / D2C through the planner ───────────────────────────────────────

def test_c2c_gcs_builds_cloud_credentials_and_derives_types():
    from app.agents.data_transfer_agent.dta_state import cloud_storage_credentials

    state = _fresh_state()
    _register(state, alias="src", db_type="gcs", bucket_name="in-bucket",
              file_path="patient_data.csv")
    _register(state, alias="dst", db_type="gcs", bucket_name="out-bucket",
              file_path="transfers/result.json")

    msg = _transfer(state, source_alias="src", destination_alias="dst",
                    user_prompt="move it")

    assert "not registered" not in msg.lower()
    assert "Transfer complete" in msg
    assert "Cloud to Cloud Transfer Complete" in msg  # header (row-count formatting tested separately)

    call = _last_pipeline_call()
    # Types derived from the object extensions, NOT forced to parquet.
    assert call["source_type"] == "csv"
    assert call["destination_type"] == "json"
    assert isinstance(call["source_credentials"], cloud_storage_credentials)
    assert isinstance(call["destination_credentials"], cloud_storage_credentials)
    assert call["source_file"] == "gs://in-bucket/patient_data.csv"
    assert call["destination_file"] == "gs://out-bucket/transfers/result.json"
    # GCS auth is via the mounted secret → no cloud_env injected.
    assert _CALLS["gke"][-1]["cloud_env"] == {}


def test_d2d_success_message_has_kind_header_and_table_row():
    state = _fresh_state()
    _register(state, alias="src_db", db_type="mysql", host="db1.example.com",
              port=3306, database="src", user="u", password="p", table="orders")
    _register(state, alias="dst_db", db_type="postgresql", host="db2.example.com",
              port=5432, database="wh", user="u", password="p")

    msg = _transfer(state, source_alias="src_db", destination_alias="dst_db",
                    user_prompt="go", dest_table="housing_high_income")

    assert "SUCCESS! Database to Database Transfer Complete!" in msg
    # DB destination → never phrased as a cloud "bucket".
    assert "into destination bucket" not in msg
    assert "Runner:" in msg


def test_tabled_transfer_parser_reads_both_connections_and_tables():
    """"transfer from <src> table <t> to <dst> table <t>" — the inline "table <t>"
    clauses used to break the alias regex and fool the dest-table extractor."""
    import app.agents.planner as planner

    p = planner._extract_tabled_transfer(
        "transfer from sunny_test2 table adult_income to sunny_test7 table adult_income_dest"
    )
    assert p is not None
    assert p.source_alias == "sunny_test2"
    assert p.source_table == "adult_income"
    assert p.destination_alias == "sunny_test7"
    assert p.dest_table == "adult_income_dest"

    # Destination table optional; source table still captured.
    p2 = planner._extract_tabled_transfer("move data from db1 table orders to db2")
    assert p2 and p2.source_alias == "db1" and p2.source_table == "orders"
    assert p2.destination_alias == "db2" and p2.dest_table is None

    # Must NOT hijack the cloud/plain shapes (no inline "table <t>").
    assert planner._extract_tabled_transfer(
        "transfer from C2C_Source_GCP to C2C_Destination_GCP, from a.csv to b.json"
    ) is None
    assert planner._extract_tabled_transfer("transfer from sunny_test2 to sunny_test7") is None


def test_tabled_d2d_transfer_threads_source_and_dest_tables_to_pipeline():
    """The explicit source/dest tables must reach the pipeline creds, not the
    connection defaults."""
    state = _fresh_state()
    _register(state, alias="sunny_test2", db_type="mysql", host="db1.example.com",
              port=3306, database="src", user="u", password="p", table="default_ignored")
    _register(state, alias="sunny_test7", db_type="postgresql", host="db2.example.com",
              port=5432, database="wh", user="u", password="p")

    msg = _transfer(state, source_alias="sunny_test2", source_table="adult_income",
                    destination_alias="sunny_test7", dest_table="adult_income_dest",
                    user_prompt="go")

    assert "Transfer complete" in msg
    call = _last_pipeline_call()
    # Source reads the NAMED table, not the connection's registered default.
    assert call["source_credentials"]["table"] == "adult_income"
    assert call["destination_credentials"]["table"] == "adult_income_dest"


def test_c2d_s3_source_to_db_injects_aws_env():
    from app.agents.data_transfer_agent.dta_state import cloud_storage_credentials

    state = _fresh_state()
    _register(state, alias="lake", db_type="aws_s3", bucket_name="s3-bucket",
              file_path="events.csv", access_key="AKIA", secret_key="sk", region="us-west-2")
    _register(state, alias="warehouse", db_type="postgresql", host="db.example.com",
              port=5432, database="wh", user="u", password="p")

    msg = _transfer(state, source_alias="lake", destination_alias="warehouse",
                    user_prompt="load it", dest_table="target")

    assert "Transfer complete" in msg
    call = _last_pipeline_call()
    assert call["source_type"] == "csv"
    assert call["destination_type"] == "postgresql"
    assert isinstance(call["source_credentials"], cloud_storage_credentials)
    assert call["destination_credentials"]["table"] == "target"
    # S3 creds must reach the pod as env vars.
    env = _CALLS["gke"][-1]["cloud_env"]
    assert env["AWS_ACCESS_KEY_ID"] == "AKIA"
    assert env["AWS_SECRET_ACCESS_KEY"] == "sk"
    assert env["AWS_REGION"] == "us-west-2"


def test_d2c_db_to_azure_parquet_injects_azure_env():
    from app.agents.data_transfer_agent.dta_state import cloud_storage_credentials

    state = _fresh_state()
    _register(state, alias="warehouse", db_type="postgresql", host="db.example.com",
              port=5432, database="wh", user="u", password="p", table="orders")
    _register(state, alias="archive", db_type="azure_blob", bucket_name="container",
              file_path="dumps/orders.parquet", az_storage_account="acct",
              az_sas_token="sv=2021&sig=abc")

    msg = _transfer(state, source_alias="warehouse", destination_alias="archive",
                    user_prompt="archive it")

    assert "Transfer complete" in msg
    call = _last_pipeline_call()
    assert call["source_type"] == "postgresql"
    assert call["destination_type"] == "parquet"
    assert isinstance(call["destination_credentials"], cloud_storage_credentials)
    assert call["destination_file"] == "az://container/dumps/orders.parquet"
    env = _CALLS["gke"][-1]["cloud_env"]
    # Destination creds are DEST_-prefixed so they can coexist with a same-provider
    # source's creds in the one merged env namespace.
    assert env["DEST_AZURE_STORAGE_ACCOUNT"] == "acct"
    assert env["DEST_AZURE_SAS_TOKEN"] == "sv=2021&sig=abc"


def test_prompt_object_overrides_registered_default():
    state = _fresh_state()
    _register(state, alias="src", db_type="gcs", bucket_name="b", file_path="default.csv")
    _register(state, alias="dst", db_type="gcs", bucket_name="b2", file_path="out.csv")

    _transfer(state, source_alias="src", destination_alias="dst",
              user_prompt="go", source_object="override/special.parquet")

    call = _last_pipeline_call()
    assert call["source_type"] == "parquet"
    assert call["source_credentials"].file_path == "override/special.parquet"


# ── Guard rails ───────────────────────────────────────────────────────────────

def test_cloud_endpoint_without_object_path_prompts_user():
    state = _fresh_state()
    _register(state, alias="src", db_type="gcs", bucket_name="b")  # no file_path
    _register(state, alias="dst", db_type="gcs", bucket_name="b2", file_path="out.csv")

    msg = _transfer(state, source_alias="src", destination_alias="dst", user_prompt="go")

    assert "Which file in" in msg
    assert not _CALLS["pipeline"], "pipeline must not run when the object path is unknown"


def test_json_cloud_source_is_rejected():
    state = _fresh_state()
    _register(state, alias="src", db_type="gcs", bucket_name="b", file_path="data.json")
    _register(state, alias="dst", db_type="postgresql", host="db.example.com",
              port=5432, database="wh", user="u", password="p")

    msg = _transfer(state, source_alias="src", destination_alias="dst",
                    user_prompt="go", dest_table="t")

    assert "JSON is not supported as a cloud source" in msg
    assert not _CALLS["pipeline"]


def test_unregistered_source_still_reports_not_registered():
    state = _fresh_state()
    _register(state, alias="dst", db_type="gcs", bucket_name="b2", file_path="out.csv")
    msg = _transfer(state, source_alias="ghost", destination_alias="dst", user_prompt="go")
    assert "not registered" in msg.lower()
    assert not _CALLS["pipeline"]


def test_register_cloud_bucket_confirmation_mentions_default_object():
    state = _fresh_state()
    msg = _register(state, alias="src", db_type="gcs", bucket_name="b",
                    file_path="patient_data.csv")
    assert "patient_data.csv" in msg
    # And without a default it tells the user to name the file at transfer time.
    msg2 = _register(state, alias="src2", db_type="gcs", bucket_name="b")
    assert "No default object" in msg2


# ── UI Cloud Dataset connections referenced by their cloud_datasets id ─────────

def test_c2c_via_ui_connection_ids():
    from app.agents.data_transfer_agent.dta_state import cloud_storage_credentials

    _CLOUD_CONNS["aaaaaaaa-0000-0000-0000-000000000001"] = {
        "provider": "gcp", "bucket_name": "in-bucket",
        "secret_key": '{"type": "service_account", "project_id": "p"}',
    }
    _CLOUD_CONNS["aaaaaaaa-0000-0000-0000-000000000002"] = {
        "provider": "aws_s3", "bucket_name": "out-bucket",
        "access_key": "AK", "secret_key": "sk", "region": "us-east-1",
    }
    state = _fresh_state()  # nothing registered in chat; resolve purely from UI store

    msg = _transfer(state, source_alias="aaaaaaaa-0000-0000-0000-000000000001", destination_alias="aaaaaaaa-0000-0000-0000-000000000002",
                    user_prompt="move it", source_object="in.csv",
                    dest_object="out/result.csv")

    assert "Transfer complete" in msg
    call = _last_pipeline_call()
    assert call["source_type"] == "csv" and call["destination_type"] == "csv"
    assert isinstance(call["source_credentials"], cloud_storage_credentials)
    assert call["source_credentials"].provider == "gcp"
    assert call["source_credentials"].gcp_service_account_info["type"] == "service_account"
    assert call["destination_credentials"].provider == "aws"
    assert call["destination_file"] == "s3://out-bucket/out/result.csv"
    # S3 destination creds injected as pod env (DEST_-prefixed); GCS source needs none.
    env = _CALLS["gke"][-1]["cloud_env"]
    assert env == {"DEST_AWS_ACCESS_KEY_ID": "AK", "DEST_AWS_SECRET_ACCESS_KEY": "sk",
                   "DEST_AWS_REGION": "us-east-1"}


def test_ui_connection_bucket_prefix_is_folded_into_path():
    _CLOUD_CONNS["aaaaaaaa-0000-0000-0000-000000000003"] = {"provider": "gcp", "bucket_name": "bucket/nested/prefix"}
    _CLOUD_CONNS["aaaaaaaa-0000-0000-0000-000000000004"] = {"provider": "gcp", "bucket_name": "b2"}
    state = _fresh_state()

    _transfer(state, source_alias="aaaaaaaa-0000-0000-0000-000000000003", destination_alias="aaaaaaaa-0000-0000-0000-000000000004",
              user_prompt="go", source_object="in.csv", dest_object="out.csv")

    src = _last_pipeline_call()["source_credentials"]
    assert src.bucket_name == "bucket"
    assert src.file_path == "nested/prefix/in.csv"
    assert _last_pipeline_call()["source_file"] == "gs://bucket/nested/prefix/in.csv"


def test_ui_connection_without_object_path_prompts_user():
    _CLOUD_CONNS["aaaaaaaa-0000-0000-0000-000000000001"] = {"provider": "gcp", "bucket_name": "b"}
    _CLOUD_CONNS["aaaaaaaa-0000-0000-0000-000000000002"] = {"provider": "gcp", "bucket_name": "b2"}
    state = _fresh_state()
    # No source_object and the UI connection stores no path → must ask.
    msg = _transfer(state, source_alias="aaaaaaaa-0000-0000-0000-000000000001", destination_alias="aaaaaaaa-0000-0000-0000-000000000002",
                    user_prompt="go", dest_object="out.csv")
    assert "Which file in" in msg
    assert not _CALLS["pipeline"]


def test_unknown_id_reports_not_registered():
    state = _fresh_state()
    _CLOUD_CONNS["aaaaaaaa-0000-0000-0000-000000000002"] = {"provider": "gcp", "bucket_name": "b2"}
    msg = _transfer(state, source_alias="does-not-exist", destination_alias="aaaaaaaa-0000-0000-0000-000000000002",
                    user_prompt="go", source_object="a.csv", dest_object="b.csv")
    assert "not registered" in msg.lower()
    assert not _CALLS["pipeline"]


def test_cloud_connection_referenced_by_name_resolves_for_both_endpoints():
    """A c2c transfer may name BOTH cloud connections by their human display name
    (``C2C_Source_GCP`` / ``C2C_Destination_GCP``) instead of their opaque UUIDs.
    Name resolution is user-scoped (via ``list_cloud_connections(user_id)``)."""
    state = _fresh_state()
    state["user_id"] = "user-1"  # name lookup is scoped to the current user
    _CLOUD_CONNS["aaaaaaaa-0000-0000-0000-0000000000b1"] = {
        "name": "C2C_Source_GCP", "provider": "gcp", "bucket_name": "b",
    }
    _CLOUD_CONNS["aaaaaaaa-0000-0000-0000-0000000000b2"] = {
        "name": "C2C_Destination_GCP", "provider": "gcp", "bucket_name": "b2",
    }
    msg = _transfer(state, source_alias="C2C_Source_GCP", destination_alias="C2C_Destination_GCP",
                    user_prompt="go", source_object="a.csv", dest_object="b.csv")
    assert "not registered" not in msg.lower()
    assert "Transfer complete" in msg
    call = _last_pipeline_call()
    assert call["source_file"] == "gs://b/a.csv"
    assert call["destination_file"] == "gs://b2/b.csv"


def test_cloud_connection_by_name_requires_a_signed_in_user():
    """Name resolution is user-scoped; without a user_id it can't list the user's
    connections, so a name that isn't a UUID reports 'not registered' (no cross-user
    guessing)."""
    state = _fresh_state()  # no user_id
    _CLOUD_CONNS["aaaaaaaa-0000-0000-0000-0000000000b1"] = {
        "name": "C2C_Source_GCP", "provider": "gcp", "bucket_name": "b",
    }
    msg = _transfer(state, source_alias="C2C_Source_GCP", destination_alias="C2C_Source_GCP",
                    user_prompt="go", source_object="a.csv", dest_object="b.csv")
    assert "not registered" in msg.lower()
    assert not _CALLS["pipeline"]


def test_ui_connection_resolved_by_id_shows_name_and_embedded_object_path():
    """Mirrors the real C2C_Source_GCP / C2C_Destination_GCP connections: referenced
    by their UUID id, GCS provider, SA JSON in secret_key, and the destination's object
    path baked into bucket_name. The name is used only for the display label."""
    from app.agents.data_transfer_agent.dta_state import cloud_storage_credentials

    sa_json = '{"type": "service_account", "project_id": "p"}'
    src_id = "aaaaaaaa-0000-0000-0000-0000000000a1"
    dst_id = "aaaaaaaa-0000-0000-0000-0000000000a2"
    _CLOUD_CONNS[src_id] = {
        "name": "C2C_Source_GCP", "provider": "google_cloud_platform",
        "bucket_name": "avaloka-test-user-filestore", "secret_key": sa_json,
    }
    _CLOUD_CONNS[dst_id] = {
        "name": "C2C_Destination_GCP", "provider": "google_cloud_platform",
        "bucket_name": "avaloka-dta-destination/transfers/patient_data_transformed.json",
        "secret_key": sa_json,
    }
    state = _fresh_state()

    # Referenced by id; source object from the prompt; dest object defaulted from bucket_name.
    msg = _transfer(state, source_alias=src_id, destination_alias=dst_id,
                    user_prompt="go", source_object="patient_data.csv")

    assert "not registered" not in msg.lower()
    assert "Transfer complete" in msg
    call = _last_pipeline_call()
    assert call["source_type"] == "csv"
    assert call["source_file"] == "gs://avaloka-test-user-filestore/patient_data.csv"
    # Destination object was peeled off bucket_name → json, bucket cleaned.
    assert call["destination_type"] == "json"
    assert call["destination_credentials"].bucket_name == "avaloka-dta-destination"
    assert call["destination_file"] == "gs://avaloka-dta-destination/transfers/patient_data_transformed.json"
    # GCS SA JSON carried through from secret_key.
    assert isinstance(call["source_credentials"], cloud_storage_credentials)
    assert call["source_credentials"].gcp_service_account_info["project_id"] == "p"
    # And the raw SA JSON is forwarded to the GKE runner (for GOOGLE_APPLICATION_CREDENTIALS).
    assert _CALLS["gke"][-1]["gcp_sa_json"] == sa_json
    # Success message shows "name → bucket" (not the opaque id).
    assert "C2C_Source_GCP → avaloka-test-user-filestore" in msg
    assert "C2C_Destination_GCP → avaloka-dta-destination" in msg


def test_object_path_not_doubled_when_it_includes_the_connection_prefix():
    """Regression: connection bucket_name 'b/transfers' + prompt object
    'transfers/out.json' must not become 'transfers/transfers/out.json'."""
    _CLOUD_CONNS["aaaaaaaa-0000-0000-0000-000000000001"] = {"provider": "gcp", "bucket_name": "src-bucket"}
    _CLOUD_CONNS["aaaaaaaa-0000-0000-0000-000000000002"] = {"provider": "gcp", "bucket_name": "dest-bucket/transfers"}
    state = _fresh_state()

    _transfer(state, source_alias="aaaaaaaa-0000-0000-0000-000000000001", destination_alias="aaaaaaaa-0000-0000-0000-000000000002",
              user_prompt="go", source_object="in.csv",
              dest_object="transfers/out.json")

    dst = _last_pipeline_call()["destination_credentials"]
    assert dst.bucket_name == "dest-bucket"
    assert dst.file_path == "transfers/out.json"  # not transfers/transfers/out.json
    assert _last_pipeline_call()["destination_file"] == "gs://dest-bucket/transfers/out.json"
