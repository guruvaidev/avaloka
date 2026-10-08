"""Lineage: where data came from, and what breaks if it changes.

Every store test runs twice: against a SQLite file, and against a real Postgres
when ``AVALOKA_TEST_POSTGRES_URL`` names a scratch database. The Postgres half
carries the ``integration`` marker because it needs a service, so the hermetic
gate deselects it; run it with::

    AVALOKA_TEST_POSTGRES_URL=postgresql://... pytest tests/test_lineage.py -m "integration or not integration"

The lineage tables in that database are dropped before each test.
"""

import logging
import os
import sys

import pandas as pd
import pytest
from sqlalchemy import create_engine, event, text

from app.agents import lineage_hooks
from app.core import lineage
from app.core.lineage import Edge, EdgeKind, LineageStore

PG_URL = os.environ.get("AVALOKA_TEST_POSTGRES_URL", "").strip()
TABLES = ("lineage_node", "lineage_edge", "lineage_name_vault", "lineage_meta")

A, B = "tenant-a", "tenant-b"

PASSWORD = "hunter2"


def _url(scheme, rest, user="u", password=PASSWORD):
    # Assembled at run time, so no tracked file holds a string shaped like a
    # live database URI: tests/contract/test_c1_repo_hygiene.py scans for
    # exactly that and cannot tell a fixture from a leak. Same fix as the
    # telemetry and analytics redaction fixtures; do not exempt this file.
    return scheme + "://" + user + ":" + password + "@" + rest


def _postgres_url_or_skip():
    """A skip reads as a pass in a summary line. On a laptop that is fine. On
    the run that gates a merge it would mean the Postgres path -- the one that
    ships -- was never exercised, so there it is a failure."""
    if PG_URL:
        return PG_URL
    message = "AVALOKA_TEST_POSTGRES_URL is not set; the Postgres path was not exercised"
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
        pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
    pytest.skip(message)


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.integration)])
def db_url(request, tmp_path, monkeypatch):
    lineage.reset_for_tests()
    if request.param == "sqlite":
        url = "sqlite:///" + (tmp_path / "lineage.db").as_posix()
    else:
        url = _postgres_url_or_skip()
        engine = create_engine(url)
        with engine.begin() as conn:
            for table in TABLES:
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        engine.dispose()
    # The hooks read the location from the environment, as the pods do.
    monkeypatch.setenv(lineage.ENV_DB_URL, url)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv(lineage.ENV_ENABLED, raising=False)
    yield url
    lineage.reset_for_tests()


@pytest.fixture
def store(db_url):
    s = LineageStore(A)
    yield s
    s.close()


def _raw(url, table):
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return conn.execute(text(f"SELECT * FROM {table}")).fetchall()
    finally:
        engine.dispose()


def _chain(store):
    """raw -> cleaned -> features, with a model on the end."""
    store.record_dataset("ds:raw", label="sales.csv",
                         columns=[("customer_email", "object"), ("amount", "float64")],
                         connection_id="conn:s3", pii={"customer_email": "email"})
    store.record_dataset("ds:cleaned", derived_from="ds:raw", analysis_id="an:1",
                         columns=[("customer_email", "object"), ("amount", "float64")],
                         pii={"customer_email": "email"})
    store.record_dataset("ds:features", derived_from="ds:cleaned", analysis_id="an:2",
                         columns=[("amount", "float64"), ("amount_log", "float64")])
    store.record_model("model:churn", trained_on="ds:features",
                       feature_columns=["amount", "amount_log"])
    return store


@pytest.fixture
def chain(store):
    return _chain(store)


# --------------------------------------------------------------------------- #
# Column names are data
# --------------------------------------------------------------------------- #

def test_column_names_are_not_stored_in_the_graph(chain, db_url):
    """patient_hiv_status is a diagnosis; a schema is data."""
    graph_text = " ".join(str(r) for r in _raw(db_url, "lineage_node"))
    graph_text += " ".join(str(r) for r in _raw(db_url, "lineage_edge"))
    assert "ds:raw" in graph_text, "the graph was not written at all"
    assert "customer_email" not in graph_text
    assert "amount_log" not in graph_text


def test_names_resolve_locally(chain):
    cid = chain.column_id("ds:raw", "customer_email")
    assert chain.resolve_column(cid) == "customer_email"


def test_the_same_name_in_two_datasets_gets_different_ids(chain):
    a = chain.column_id("ds:raw", "amount")
    b = chain.column_id("ds:features", "amount")
    assert a != b, "otherwise the graph asserts two unrelated columns are the same"


def test_ids_are_salted_per_tenant(db_url):
    """Column names are low entropy; without a salt the digests are a rainbow
    table away from plaintext. One salt for the whole shared database would let
    one tenant's known schema confirm another's."""
    assert LineageStore(A).column_id("ds:x", "email") != LineageStore(B).column_id("ds:x", "email")
    salts = _raw(db_url, "lineage_meta")
    assert {row[0] for row in salts} == {A, B}
    assert len({row[2] for row in salts}) == 2


def test_ids_are_stable_across_processes(db_url):
    """Two pods must derive the same id for the same column, or the vault one
    pod wrote is unreadable from the other."""
    first = LineageStore(A).column_id("ds:x", "email")
    lineage.reset_for_tests()   # a second process: no cached engine, no cached salt
    assert LineageStore(A).column_id("ds:x", "email") == first
    assert len(_raw(db_url, "lineage_meta")) == 1


# --------------------------------------------------------------------------- #
# The questions lineage exists to answer
# --------------------------------------------------------------------------- #

def test_where_did_this_come_from(chain):
    assert set(chain.ancestors("ds:features")) == {"ds:cleaned", "ds:raw"}


def test_dataset_lineage_resolves_labels_without_column_names(store):
    store.record_dataset("ds:raw", label="source.csv", columns=[("private_col", "str")])
    store.record_dataset("ds:clean", label="cleaned.csv", derived_from="ds:raw")

    result = store.dataset_lineage("ds:clean")

    assert result["dataset"].label == "cleaned.csv"
    assert [(node.id, node.label) for node in result["parents"]] == [
        ("ds:raw", "source.csv")
    ]
    assert [(node.id, node.label) for node in result["ancestors"]] == [
        ("ds:raw", "source.csv")
    ]
    assert "private_col" not in str(result)


def test_what_depends_on_this(chain):
    assert set(chain.descendants("ds:raw")) == {"ds:cleaned", "ds:features"}


def test_impact_of_renaming_a_column(chain):
    """The 2 a.m. question: upstream renamed a column, what breaks?"""
    impact = chain.impact_of_column_change("ds:raw", "amount")
    assert set(impact["downstream_datasets"]) == {"ds:cleaned", "ds:features"}
    assert "model:churn" in impact["models_affected"]
    assert impact["total_affected"] >= 3


def test_a_model_records_the_columns_it_uses(store):
    """A model on a dataset with no descendants is only reachable from a column
    through its feature edges."""
    store.record_dataset("ds:a", columns=[("used", "int64"), ("unused", "int64")])
    store.record_dataset("ds:b", columns=[("used", "int64")])
    store.record_model("model:m", trained_on="ds:b", feature_columns=["used"])
    assert store.stats()["edge_uses_feature"] == 1
    used = store.column_id("ds:b", "used")
    edges = [r for r in _raw(store.url, "lineage_edge") if r[2] == EdgeKind.USES_FEATURE.value]
    assert [(r[1], r[3]) for r in edges] == [("model:m", used)]


def test_which_models_touched_personal_data(chain):
    hits = chain.models_touching_pii()
    assert len(hits) == 1
    assert hits[0]["model"] == "model:churn"
    assert hits[0]["pii_kinds"] == ["email"]


def test_a_model_on_clean_data_is_not_flagged(store):
    store.record_dataset("ds:clean", columns=[("units", "int64")])
    store.record_model("model:safe", trained_on="ds:clean", feature_columns=["units"])
    assert store.models_touching_pii() == []


def test_orphans_are_datasets_nothing_uses(chain):
    chain.record_dataset("ds:unused", label="scratch.csv")
    assert "ds:unused" in chain.orphans()
    assert "ds:features" not in chain.orphans(), "it has a model trained on it"


def test_pii_is_found_through_lineage_not_just_the_training_frame(chain):
    """The naive query gives the wrong answer reassuringly: PII is usually
    dropped during cleaning, so the trained-on frame looks clean while the
    lineage runs straight through an email address."""
    hits = chain.models_touching_pii()
    assert hits[0]["model"] == "model:churn"
    # ds:features itself has no PII column — it was found upstream.
    assert set(hits[0]["via"]) == {"ds:raw", "ds:cleaned"}


def test_stats_summarise_the_graph(chain):
    s = chain.stats()
    assert s["node_dataset"] == 3 and s["node_model"] == 1
    assert s["edge_derived_from"] == 2
    assert s["edge_classified_as"] == 2


def test_connection_provenance_is_recorded(chain, db_url):
    rows = [r for r in _raw(db_url, "lineage_edge") if r[2] == EdgeKind.LOADED_FROM.value]
    assert [(r[0], r[1], r[3]) for r in rows] == [(A, "ds:raw", "conn:s3")]


# --------------------------------------------------------------------------- #
# One shared database, many customers
# --------------------------------------------------------------------------- #

@pytest.fixture
def two_tenants(db_url):
    """Both tenants use the SAME dataset, model and connection ids, which is
    what uploads named sales.csv and ids minted client-side will do."""
    a = _chain(LineageStore(A))
    b = LineageStore(B)
    b.record_dataset("ds:raw", label="payroll.csv",
                     columns=[("salary", "float64"), ("ssn", "object")], pii={"ssn": "ssn"})
    # B's graph continues past where A's stops. A walk that ignored the tenant
    # would carry A from ds:raw into B's upstream.
    b.record_dataset("ds:secret-source", label="acquisition-target.csv")
    b.add_edge(Edge("ds:raw", EdgeKind.DERIVED_FROM, "ds:secret-source"))
    b.record_dataset("ds:b-child", label="b-report.csv", derived_from="ds:raw")
    b.record_model("model:b", trained_on="ds:b-child", feature_columns=[])
    return a, b


def test_a_tenant_cannot_read_another_tenants_file_names(two_tenants):
    a, b = two_tenants
    assert a.get_node("ds:raw").label == "sales.csv"
    assert b.get_node("ds:raw").label == "payroll.csv"
    assert a.get_node("ds:secret-source") is None
    assert a.get_nodes(["ds:raw", "ds:secret-source", "ds:b-child"])[0].label == "sales.csv"
    assert len(a.get_nodes(["ds:raw", "ds:secret-source", "ds:b-child"])) == 1
    assert "payroll" not in str(a.dataset_lineage("ds:cleaned"))
    assert a.dataset_lineage("ds:b-child") is None


def test_a_walk_never_crosses_into_another_tenants_graph(two_tenants):
    a, b = two_tenants
    assert set(a.ancestors("ds:features")) == {"ds:cleaned", "ds:raw"}
    assert set(a.descendants("ds:raw")) == {"ds:cleaned", "ds:features"}
    assert set(b.ancestors("ds:b-child")) == {"ds:raw", "ds:secret-source"}
    assert set(b.descendants("ds:raw")) == {"ds:b-child"}
    # A has a dataset with the id of B's upstream, unrelated to anything. If
    # either the parent lookup or the walk read B's edges, it would show up
    # here as a parent A never had.
    a.record_dataset("ds:secret-source", label="a-unrelated.csv")
    parents = a.dataset_lineage("ds:raw")
    assert parents["parents"] == [] and parents["ancestors"] == []


def test_a_tenant_cannot_resolve_another_tenants_column_names(two_tenants, db_url):
    a, b = two_tenants
    theirs = b.column_id("ds:raw", "salary")
    assert b.resolve_column(theirs) == "salary"
    assert a.resolve_column(theirs) is None
    # ...including by asking for the name outright.
    assert a.resolve_column(a.column_id("ds:raw", "salary")) is None
    owners = {row[2]: row[0] for row in _raw(db_url, "lineage_name_vault")}
    assert owners["salary"] == B and owners["customer_email"] == A


def test_compliance_and_impact_answers_are_per_tenant(two_tenants):
    a, b = two_tenants
    assert [hit["model"] for hit in a.models_touching_pii()] == ["model:churn"]
    assert a.models_touching_pii()[0]["pii_kinds"] == ["email"]
    assert b.models_touching_pii() == [
        {"model": "model:b", "pii_kinds": ["ssn"], "via": ["ds:raw"]}]
    assert a.impact_of_column_change("ds:raw", "amount")["models_affected"] == ["model:churn"]
    assert b.impact_of_column_change("ds:raw", "salary")["models_affected"] == ["model:b"]
    # The same id is unused by one tenant and built upon by the other.
    a.record_dataset("ds:shared-id", label="a-scratch.csv")
    b.record_dataset("ds:shared-id", label="b-base.csv")
    b.record_dataset("ds:b-unused", label="b-scratch.csv", derived_from="ds:shared-id")
    assert a.orphans() == ["ds:shared-id"]
    assert b.orphans() == ["ds:b-unused"]


def test_reads_stay_scoped_when_another_tenant_holds_rows_under_the_same_ids(db_url):
    """Scoping the writes is the easy half. Each read below is one a tenant
    filter could be dropped from without any single-tenant test noticing, so
    B's rows are planted on exactly the ids A's queries look up: A's dataset
    ids, A's model's training set, and A's own (salted) column ids."""
    a, b = _chain(LineageStore(A)), LineageStore(B)
    a_amount = a.column_id("ds:raw", "amount")
    a_amount_features = a.column_id("ds:features", "amount")

    # impact: a B model that claims A's column as a feature, and one trained on A's dataset id.
    b.add_edge(Edge("model:b-uses-a-column", EdgeKind.USES_FEATURE, a_amount))
    b.record_model("model:b-on-a-dataset", trained_on="ds:features", feature_columns=[])
    # pii, classification side: B marks A's clean column as personal data.
    b.add_edge(Edge(a_amount_features, EdgeKind.CLASSIFIED_AS, "pii:planted-by-b"))
    # pii, has-column side: B hangs a column off A's dataset id, and A's graph
    # happens to hold a classification for that column id.
    b.add_edge(Edge("ds:features", EdgeKind.HAS_COLUMN, "col:b-only"))
    a.add_edge(Edge("col:b-only", EdgeKind.CLASSIFIED_AS, "pii:never-attached-in-a"))
    # parents: B derives A's root from something.
    b.add_edge(Edge("ds:raw", EdgeKind.DERIVED_FROM, "ds:b-upstream"))
    a.record_dataset("ds:b-upstream", label="a-unrelated.csv")

    impact = a.impact_of_column_change("ds:raw", "amount")
    assert impact["models_affected"] == ["model:churn"]
    assert a.models_touching_pii() == [
        {"model": "model:churn", "pii_kinds": ["email"], "via": ["ds:cleaned", "ds:raw"]}]
    lineage_of_root = a.dataset_lineage("ds:raw")
    assert lineage_of_root["parents"] == [] and lineage_of_root["ancestors"] == []
    assert a.ancestors("ds:raw") == []
    assert sorted(a.descendants("ds:raw")) == ["ds:cleaned", "ds:features"]

    # And B sees its own rows, not A's.
    assert b.models_touching_pii() == [], "B's model is on a dataset with no B column classified"
    assert b.impact_of_column_change("ds:features", "x")["models_affected"] == [
        "model:b-on-a-dataset"]


def test_stats_count_only_the_tenants_own_rows(two_tenants):
    a, b = two_tenants
    assert a.stats()["node_dataset"] == 3 and a.stats()["node_model"] == 1
    assert b.stats()["node_dataset"] == 3 and b.stats()["edge_derived_from"] == 2
    assert LineageStore("tenant-nobody").stats() == {"nodes": 0, "edges": 0}


def test_one_tenants_write_does_not_overwrite_anothers(two_tenants):
    a, b = two_tenants
    b.record_dataset("ds:raw", label="renamed-by-b.csv")
    assert a.get_node("ds:raw").label == "sales.csv"


def test_a_store_without_a_tenant_records_and_reads_nothing(db_url, two_tenants, caplog):
    before = {table: len(_raw(db_url, table)) for table in TABLES}
    with caplog.at_level(logging.WARNING, logger="app.core.lineage"):
        for tenant in (None, "", "   "):
            nobody = LineageStore(tenant)
            nobody.record_dataset("ds:raw", label="x.csv", columns=[("a", "int64")])
            assert nobody.available is False
            assert nobody.get_node("ds:raw") is None
            assert nobody.ancestors("ds:features") == []
            assert nobody.stats() == {}
    assert {table: len(_raw(db_url, table)) for table in TABLES} == before
    assert len([r for r in caplog.records if "without a tenant" in r.getMessage()]) == 1


def test_hooks_scope_by_the_requesting_user(db_url):
    state = {"user_id": A, "active_dataset_id": "ds:up",
             "datasets_context": [{"dataset_id": "ds:up", "filename": "a-upload.csv"}],
             "output_location": "a-out.csv"}
    out_a = lineage_hooks.record_analysis_lineage(state, pd.DataFrame({"x": [1]}))
    assert out_a

    asked_by_b = {"user_id": B, "latest_output_dataset_id": out_a}
    assert lineage_hooks.format_lineage_reply(asked_by_b) == (
        "I don't have recorded lineage for the current dataset yet.")
    assert "a-upload.csv" in lineage_hooks.format_lineage_reply(
        {"user_id": A, "latest_output_dataset_id": out_a})
    assert {row[0] for row in _raw(db_url, "lineage_node")} == {A}


@pytest.mark.parametrize("user_id", [None, "", "  "])
def test_a_provenance_question_without_a_user_reads_nobodys_lineage(db_url, user_id):
    """Recording refuses a missing user; the read must too. A fallback bucket
    on the read side would answer an unauthenticated question from whatever
    happened to be stored under that name."""
    for bucket in ("default", "None", "anonymous", ""):
        LineageStore(bucket).record_dataset("ds:x", label="someone-elses.csv",
                                            derived_from="ds:parent")
    reply = lineage_hooks.format_lineage_reply({"user_id": user_id, "active_dataset_id": "ds:x"})
    assert "someone-elses.csv" not in reply and "ds:parent" not in reply
    assert "unavailable" in reply


@pytest.mark.parametrize("user_id", [None, "", "  "])
def test_hooks_record_nothing_without_a_user(db_url, user_id):
    state = {"user_id": user_id, "active_dataset_id": "ds:up", "output_location": "out.csv",
             "training_plan": {"data_config": {"feature_columns": ["x"]}}}
    frame = pd.DataFrame({"x": [1]})
    assert lineage_hooks.record_analysis_lineage(state, frame) is None
    assert lineage_hooks.record_model_lineage(state, {"mlflow_run_id": "r"}) is None
    lineage_hooks.record_dataset_pii(state, frame, {"findings": [{"column": "x", "kind": "email"}]})
    LineageStore(A).stats()   # make sure the tables exist before reading them raw
    assert all(_raw(db_url, table) == [] for table in ("lineage_node", "lineage_edge",
                                                       "lineage_name_vault"))


# --------------------------------------------------------------------------- #
# One store for every pod
# --------------------------------------------------------------------------- #

def test_what_one_process_records_another_can_read(db_url):
    """The bug this store was moved for: the API and the LangGraph pod each
    compile the graph, and each used to write its own file."""
    state = {"user_id": A, "active_dataset_id": "ds:up",
             "datasets_context": [{"dataset_id": "ds:up", "filename": "upload.csv"}],
             "output_location": "result.csv"}
    out = lineage_hooks.record_analysis_lineage(state, pd.DataFrame({"x": [1]}))

    lineage.reset_for_tests()   # the other pod

    reply = lineage_hooks.format_lineage_reply({"user_id": A, "latest_output_dataset_id": out})
    assert "result.csv" in reply and "upload.csv" in reply


# --------------------------------------------------------------------------- #
# Write cost
# --------------------------------------------------------------------------- #

class _Counter:
    def __init__(self, engine):
        self.inserts, self.commits = [], 0
        event.listen(engine, "before_cursor_execute", self._statement)
        event.listen(engine, "commit", self._commit)

    def _statement(self, conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("INSERT"):
            self.inserts.append(statement.split("(")[0].strip())

    def _commit(self, conn):
        self.commits += 1


WIDE = [(f"col_{i}", "float64") for i in range(60)]


def test_a_dataset_is_one_transaction_however_wide_it_is(store):
    store.column_id("ds:warm", "x")          # mint the salt outside the measurement
    counter = _Counter(store.connect())

    store.record_dataset("ds:wide", label="wide.csv", columns=WIDE,
                         pii={"col_3": "email"}, derived_from="ds:p", analysis_id="an:1")

    assert counter.commits == 1
    assert counter.inserts == ["INSERT INTO lineage_node", "INSERT INTO lineage_name_vault",
                               "INSERT INTO lineage_edge"]
    assert store.stats()["edge_has_column"] == 60


def test_a_lineage_event_from_the_hook_is_one_transaction(db_url):
    LineageStore(A).column_id("ds:warm", "x")
    counter = _Counter(LineageStore(A).connect())
    state = {
        "user_id": A,
        "active_dataset_id": "ds:one",
        "active_dataset_ids": ["ds:one", "ds:two"],
        "datasets_context": [
            {"dataset_id": "ds:one", "filename": "one.csv", "columns": dict(WIDE)},
            {"dataset_id": "ds:two", "filename": "two.csv", "columns": dict(WIDE)},
        ],
        "output_location": "joined.csv",
    }
    out = lineage_hooks.record_analysis_lineage(state, pd.DataFrame({name: [1.0] for name, _ in WIDE}))

    assert counter.commits == 1, "two parents, an output and 180 columns are one event"
    assert len(counter.inserts) == 3
    assert set(LineageStore(A).ancestors(out)) == {"ds:one", "ds:two"}
    assert LineageStore(A).stats()["node_column"] == 180


def test_a_batch_that_raises_writes_nothing(store):
    with pytest.raises(RuntimeError):
        with store.batch():
            store.record_dataset("ds:half", label="half.csv", columns=[("a", "int64")])
            raise RuntimeError("the analysis blew up between two writes")
    assert store.get_node("ds:half") is None
    store.record_dataset("ds:after", label="after.csv")
    assert store.get_node("ds:after").label == "after.csv", "the store is still usable"


def test_more_columns_than_one_statement_holds_are_all_recorded(store):
    columns = [(f"c{i}", "int64") for i in range(lineage._CHUNK * 2 + 7)]
    store.record_dataset("ds:huge", columns=columns)
    assert store.stats()["edge_has_column"] == len(columns)
    assert store.resolve_column(store.column_id("ds:huge", columns[-1][0])) == columns[-1][0]


# --------------------------------------------------------------------------- #
# Safety: recording lineage must never fail an analysis
# --------------------------------------------------------------------------- #

def test_queries_on_an_empty_graph_return_empty_not_errors(store):
    assert store.ancestors("ds:nothing") == []
    assert store.descendants("ds:nothing") == []
    assert store.models_touching_pii() == []
    assert store.impact_of_column_change("ds:nothing", "col")["total_affected"] == 0


def test_a_cycle_terminates(store):
    """A bad write must not turn a question into an unbounded query."""
    store.record_dataset("ds:a")
    store.record_dataset("ds:b", derived_from="ds:a")
    store.add_edge(Edge("ds:a", EdgeKind.DERIVED_FROM, "ds:b"))
    assert set(store.ancestors("ds:a")) == {"ds:b"}


def test_recording_the_same_dataset_twice_is_idempotent(store):
    for _ in range(3):
        store.record_dataset("ds:x", label="x.csv", columns=[("a", "int64")])
    assert store.stats()["node_dataset"] == 1
    assert store.stats()["edge_has_column"] == 1


def test_recording_again_updates_the_label(store):
    store.record_dataset("ds:x", label="old.csv")
    store.record_dataset("ds:x", label="new.csv")
    assert store.get_node("ds:x").label == "new.csv"


def test_a_failed_write_is_logged_once_and_never_with_the_names_it_carried(store, db_url, caplog):
    """SQLAlchemy prints a failed statement's parameters by default. Here those
    are column names and file names."""
    store.record_dataset("ds:ok", columns=[("a", "int64")])
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE lineage_name_vault"))
    engine.dispose()

    caplog.clear()   # the store announcing where it records is not a failure
    with caplog.at_level(logging.DEBUG, logger="app.core.lineage"):
        for _ in range(5):
            store.record_dataset("ds:x", label="q3_layoffs_final.csv",
                                 columns=[("patient_hiv_status", "object")])   # must not raise

    logged = "\n".join(r.getMessage() + (str(r.exc_info[1]) if r.exc_info else "")
                       for r in caplog.records)
    assert "patient_hiv_status" not in logged
    assert "q3_layoffs_final" not in logged
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert store.get_node("ds:x") is None, "the event is all or nothing"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    lineage.reset_for_tests()
    for name in (lineage.ENV_DB_URL, "POSTGRES_URL", lineage.ENV_ENABLED, "AVALOKA_LINEAGE_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    yield monkeypatch
    lineage.reset_for_tests()


def test_the_dedicated_url_wins_over_postgres_url(clean_env):
    clean_env.setenv("POSTGRES_URL", _url("postgresql", "shared/app"))
    assert lineage.database_url() == _url("postgresql", "shared/app")
    clean_env.setenv(lineage.ENV_DB_URL, _url("postgresql", "lineage/db"))
    assert lineage.database_url() == _url("postgresql", "lineage/db")


def test_a_libpq_style_url_is_accepted(clean_env):
    clean_env.setenv("POSTGRES_URL", _url("postgres", "host:5432/app"))
    assert lineage.database_url() == _url("postgresql", "host:5432/app")


@pytest.mark.parametrize("value", ["off", "0", "false", "Disabled"])
def test_the_switch_turns_lineage_off_even_with_a_database(clean_env, value):
    clean_env.setenv("POSTGRES_URL", _url("postgresql", "shared/app"))
    clean_env.setenv(lineage.ENV_ENABLED, value)
    assert lineage.database_url() is None
    assert LineageStore(A).available is False


def test_with_no_database_lineage_is_off_says_so_once_and_writes_no_file(clean_env, tmp_path, caplog):
    """No implicit SQLite. A store that quietly appears per pod is the
    split-store bug: each pod answers from its own file."""
    clean_env.setenv("AVALOKA_LINEAGE_PATH", str(tmp_path / "legacy.db"))
    clean_env.chdir(tmp_path)
    state = {"user_id": A, "active_dataset_id": "ds:up", "output_location": "out.csv",
             "training_plan": {"data_config": {"feature_columns": ["x"]}}}
    with caplog.at_level(logging.DEBUG, logger="app"):
        for _ in range(4):
            assert lineage_hooks.record_analysis_lineage(state, pd.DataFrame({"x": [1]})) is None
            assert lineage_hooks.record_model_lineage(state, {"mlflow_run_id": "r"}) is None
            lineage_hooks.record_dataset_pii(state, pd.DataFrame({"x": [1]}), {})
            assert "unavailable" in lineage_hooks.format_lineage_reply(
                {"user_id": A, "active_dataset_id": "ds:up"})
            assert LineageStore(A).stats() == {}

    said = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(said) == 1, [r.getMessage() for r in said]
    assert "AVALOKA_LINEAGE_DB_URL" in said[0].getMessage()
    assert "POSTGRES_URL" in said[0].getMessage()
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_sqlite_in_use_is_announced_once_as_a_choice_naming_who_chose_it(clean_env, tmp_path, caplog):
    url = "sqlite:///" + (tmp_path / "chosen.db").as_posix()
    clean_env.setenv(lineage.ENV_DB_URL, url)
    with caplog.at_level(logging.INFO, logger="app.core.lineage"):
        for _ in range(3):
            LineageStore(A).record_dataset("ds:x", label="x.csv")
            LineageStore(B).stats()
    said = [r for r in caplog.records if "SQLite" in r.getMessage()]
    assert len(said) == 1 and said[0].levelno == logging.WARNING
    message = said[0].getMessage()
    assert "AVALOKA_LINEAGE_DB_URL names it" in message
    assert "chosen, not fallen back" in message
    assert "chosen.db" in message


def test_a_postgres_target_is_not_described_as_sqlite(clean_env, caplog):
    class _Engine:
        def dispose(self):
            pass

    clean_env.setattr(lineage._Backend, "_open", lambda self: _Engine())
    with caplog.at_level(logging.INFO, logger="app.core.lineage"):
        assert LineageStore(A, _url("postgresql", "db:5432/app")).connect() is not None
    assert "SQLite" not in caplog.text and PASSWORD not in caplog.text
    assert "recording to " + _url("postgresql", "db:5432/app", password="***") in caplog.text


def test_an_unsupported_database_disables_lineage_rather_than_raising(clean_env, caplog):
    with caplog.at_level(logging.WARNING, logger="app.core.lineage"):
        store = LineageStore(A, _url("mysql", "host/db"))
        store.record_dataset("ds:x", columns=[("a", "int64")])
        assert store.stats() == {}
    assert PASSWORD not in caplog.text


def test_an_unparseable_url_does_not_put_its_password_in_the_log(clean_env, caplog):
    """SQLAlchemy's parse error has quoted the offending string in some
    versions and not in others. Simulate the one that does, so this holds
    whichever is installed."""
    import sqlalchemy.engine
    from sqlalchemy.exc import ArgumentError

    url = "not a url, but it has " + PASSWORD + " in it"

    def echoing_make_url(value):
        raise ArgumentError(f"Could not parse rfc1738 URL from string '{value}'")

    clean_env.setattr(sqlalchemy.engine, "make_url", echoing_make_url)
    with caplog.at_level(logging.DEBUG, logger="app.core.lineage"):
        store = LineageStore(A, url)
        store.record_dataset("ds:x", columns=[("a", "int64")])
        assert store.ancestors("ds:x") == []
    assert [r for r in caplog.records if r.levelno >= logging.WARNING], "it must still say so"
    assert PASSWORD not in caplog.text


# --------------------------------------------------------------------------- #
# An unreachable database
# --------------------------------------------------------------------------- #

UNREACHABLE = _url("postgresql", "127.0.0.1:1/avaloka", user="avaloka")


def test_an_unreachable_database_is_tried_once_and_reported_once(clean_env, caplog):
    """Several store calls per column, every column, every turn. The old store
    logged a traceback for each."""
    clean_env.setenv(lineage.ENV_DB_URL, UNREACHABLE)
    opens = []
    real_open = lineage._Backend._open
    clean_env.setattr(lineage._Backend, "_open",
                      lambda self: (opens.append(1), real_open(self))[1])
    state = {"user_id": A, "active_dataset_id": "ds:up", "output_location": "out.csv",
             "uploaded_csv_columns": [f"c{i}" for i in range(40)]}

    with caplog.at_level(logging.DEBUG, logger="app"):
        for _ in range(5):
            assert lineage_hooks.record_analysis_lineage(
                state, pd.DataFrame({f"c{i}": [1] for i in range(40)})) is None
            assert "unavailable" in lineage_hooks.format_lineage_reply(
                {"user_id": A, "active_dataset_id": "ds:up"})
        other_tenant = LineageStore(B)
        other_tenant.record_dataset("ds:x", columns=[("a", "int64")])
        assert other_tenant.ancestors("ds:x") == []

    assert len(opens) == 1
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert warnings[0].exc_info is None, "one line, not a traceback"
    assert "127.0.0.1:1" in warnings[0].getMessage()
    assert PASSWORD not in caplog.text


def test_an_unreachable_database_is_retried_after_the_cooldown_and_recovery_is_logged(
        clean_env, tmp_path, caplog):
    """The API pod is routinely up before Postgres. Off-until-restart would
    turn a normal rollout into permanently missing lineage."""
    url = "sqlite:///" + (tmp_path / "late.db").as_posix()
    clock = [1000.0]
    clean_env.setattr(lineage.time, "monotonic", lambda: clock[0])
    real_open = lineage._Backend._open
    up, attempts = [], []

    def flaky_open(self):
        attempts.append(clock[0])
        if not up:
            raise ConnectionError("database is starting up")
        return real_open(self)

    clean_env.setattr(lineage._Backend, "_open", flaky_open)

    with caplog.at_level(logging.INFO, logger="app.core.lineage"):
        LineageStore(A, url).record_dataset("ds:early", label="early.csv")
        # Still down when the cooldown expires: tried again, not reported again.
        for _ in range(3):
            clock[0] += lineage.RETRY_AFTER_S + 1
            LineageStore(A, url).record_dataset("ds:still-down", label="still-down.csv")
        up.append(True)
        clock[0] += lineage.RETRY_AFTER_S - 1
        LineageStore(A, url).record_dataset("ds:too-soon", label="too-soon.csv")
        assert LineageStore(A, url).available is False, "still inside the cooldown"
        clock[0] += 2
        LineageStore(A, url).record_dataset("ds:later", label="later.csv")

    store = LineageStore(A, url)
    assert store.get_node("ds:later").label == "later.csv"
    assert store.get_node("ds:early") is None and store.get_node("ds:too-soon") is None
    assert len(attempts) == 5, "one try per expired cooldown, none inside one"
    messages = [r.getMessage() for r in caplog.records]
    assert len([m for m in messages if "unavailable" in m]) == 1, "once per outage"
    assert len([m for m in messages if "reachable again" in m]) == 1


# --------------------------------------------------------------------------- #
# The suite itself
# --------------------------------------------------------------------------- #

def test_the_postgres_half_fails_rather_than_skips_on_ci_without_a_database(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "PG_URL", "")
    # Not pytest.raises: a Skipped raised inside it escapes and marks THIS test
    # skipped, which is the outcome under test.
    def outcome():
        try:
            _postgres_url_or_skip()
        except pytest.fail.Exception:
            return "failed"
        except pytest.skip.Exception:
            return "skipped"
        return "ran"

    monkeypatch.setenv("CI", "true")
    assert outcome() == "failed"
    monkeypatch.delenv("CI")
    assert outcome() == "skipped"


def test_the_suite_does_not_write_lineage_to_a_real_store():
    """tests/conftest.py owns this. Every test that runs an execution node
    reaches the hooks; none of them may reach a developer's database."""
    import tempfile
    from pathlib import Path

    url = os.environ[lineage.ENV_DB_URL]
    assert url.startswith("sqlite:///")
    assert Path(tempfile.gettempdir()).resolve() in Path(url[len("sqlite:///"):]).resolve().parents
    assert lineage.ENV_ENABLED not in os.environ


def test_the_suite_redirect_overrides_a_lineage_url_already_in_the_environment():
    """setdefault would not do: a developer's exported value would win."""
    import subprocess
    import sys

    env = {**os.environ, lineage.ENV_DB_URL: _url("postgresql", "prod/lineage", user="real"),
           lineage.ENV_ENABLED: "off"}
    out = subprocess.run(
        [sys.executable, "-c",
         "import os, tests.conftest; "
         f"print(os.environ['{lineage.ENV_DB_URL}'], os.environ.get('{lineage.ENV_ENABLED}'))"],
        env=env, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    assert out.returncode == 0, out.stderr
    url, enabled = out.stdout.split()
    assert url.startswith("sqlite:///") and "prod" not in url
    assert enabled == "None"


def test_the_suite_redirect_outranks_an_exported_postgres_url(monkeypatch):
    monkeypatch.setenv("POSTGRES_URL", _url("postgresql", "prod/app", user="real"))
    assert lineage.database_url() == os.environ[lineage.ENV_DB_URL]


# --------------------------------------------------------------------------- #
# The hooks
# --------------------------------------------------------------------------- #

def test_agent_capture_preserves_multi_hop_pii_ancestry(db_url):
    state = {
        "user_id": A,
        "active_dataset_id": "ds:raw",
        "active_dataset_ids": ["ds:raw", "ds:lookup"],
        "datasets_context": [
            {"dataset_id": "ds:raw", "filename": "source.csv"},
            {"dataset_id": "ds:lookup", "filename": "lookup.csv"},
        ],
        "pii_report": {"findings": [{"column": "email", "kind": "email"}]},
        "output_location": "cleaned.csv",
    }
    clean_id = lineage_hooks.record_analysis_lineage(
        state, pd.DataFrame({"amount": [12.0]})
    )
    assert clean_id

    next_state = {
        **state,
        "latest_output_dataset_id": clean_id,
        "latest_output_location": "cleaned.csv",
        "data_source_location": "cleaned.csv",
        "output_location": "features.csv",
    }
    features_id = lineage_hooks.record_analysis_lineage(
        next_state, pd.DataFrame({"amount_log": [2.48]})
    )
    assert features_id

    model_id = lineage_hooks.record_model_lineage(
        {
            **next_state,
            "latest_output_dataset_id": features_id,
            "latest_output_location": "features.csv",
            "training_plan": {"data_config": {"feature_columns": ["amount_log"]}},
        },
        {"mlflow_run_id": "run-1", "model_name": "churn"},
    )

    store = LineageStore(A)
    assert set(store.ancestors(features_id)) == {clean_id, "ds:raw", "ds:lookup"}
    assert model_id == "model:run-1"
    assert store.models_touching_pii() == [
        {"model": model_id, "pii_kinds": ["email"], "via": ["ds:raw"]}
    ]
    reply = lineage_hooks.format_lineage_reply(
        {"user_id": A, "latest_output_dataset_id": features_id})
    assert "features.csv" in reply
    assert "source.csv" in reply


def test_a_recorded_model_keeps_its_feature_columns(db_url):
    model_id = lineage_hooks.record_model_lineage(
        {"user_id": A, "active_dataset_id": "ds:train",
         "training_plan": {"data_config": {"feature_columns": ["age", "income"],
                                           "target_column": "churned"}}},
        {"mlflow_run_id": "run-9", "model_name": "churn", "model_type": "mlp"},
    )
    store = LineageStore(A)
    assert store.get_node(model_id).label == "churn"
    assert store.stats()["edge_uses_feature"] == 2
    assert store.stats()["edge_has_column"] == 3, "two features and the target"
    for name in ("age", "income"):
        impact = store.impact_of_column_change("ds:train", name)
        assert store.resolve_column(impact["column_id"]) == name


def test_a_pii_scan_is_recorded_against_the_scanned_dataset(db_url):
    """Preparation finds the personal data; cleaning then drops the column. If
    the scan is not recorded, the model trained downstream looks clean."""
    state = {"user_id": A, "active_dataset_id": "ds:customers"}
    LineageStore(A).record_dataset("ds:customers", label="customers.csv", row_count=3)
    frame = pd.DataFrame({"email": ["a@b.c"], "spend": [1.0]})

    lineage_hooks.record_dataset_pii(state, frame, {"findings": [
        {"column": "email", "kind": "email"}, {"column": "spend", "kind": "not_pii"}]})

    store = LineageStore(A)
    node = store.get_node("ds:customers")
    assert node.label == "customers.csv" and node.attrs == {"row_count": 3}, \
        "the scan must not wipe what was already known"
    assert store.stats()["edge_has_column"] == 2
    assert store.stats()["edge_classified_as"] == 1
    store.record_dataset("ds:clean", derived_from="ds:customers", columns=[("spend", "float64")])
    store.record_model("model:m", trained_on="ds:clean", feature_columns=["spend"])
    assert store.models_touching_pii() == [
        {"model": "model:m", "pii_kinds": ["email"], "via": ["ds:customers"]}]
