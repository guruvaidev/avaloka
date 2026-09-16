"""Lineage: where data came from, and what breaks if it changes."""

import sqlite3
import pytest

from app.core.lineage import (Edge, EdgeKind, LineageStore, Node, NodeKind)


@pytest.fixture
def store(tmp_path):
    s = LineageStore(tmp_path / "lineage.db")
    yield s
    s.close()


@pytest.fixture
def chain(store):
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


# --------------------------------------------------------------------------- #
# Column names are data
# --------------------------------------------------------------------------- #

def test_column_names_are_not_stored_in_the_graph(chain, tmp_path):
    """patient_hiv_status is a diagnosis; a schema is data."""
    raw = sqlite3.connect(str(tmp_path / "lineage.db"))
    graph_text = " ".join(str(r) for r in raw.execute("SELECT * FROM node").fetchall())
    graph_text += " ".join(str(r) for r in raw.execute("SELECT * FROM edge").fetchall())
    assert "customer_email" not in graph_text
    assert "amount_log" not in graph_text


def test_names_resolve_locally(chain):
    cid = chain.column_id("ds:raw", "customer_email")
    assert chain.resolve_column(cid) == "customer_email"


def test_the_same_name_in_two_datasets_gets_different_ids(chain):
    a = chain.column_id("ds:raw", "amount")
    b = chain.column_id("ds:features", "amount")
    assert a != b, "otherwise the graph asserts two unrelated columns are the same"


def test_ids_are_salted_per_install(tmp_path):
    """Column names are low entropy; without a salt the digests are a rainbow
    table away from plaintext."""
    a = LineageStore(tmp_path / "a.db")
    b = LineageStore(tmp_path / "b.db")
    try:
        assert a.column_id("ds:x", "email") != b.column_id("ds:x", "email")
    finally:
        a.close(); b.close()


def test_ids_are_stable_within_an_install(store):
    first = store.column_id("ds:x", "email")
    store.close()
    reopened = LineageStore(store.path)
    try:
        assert reopened.column_id("ds:x", "email") == first
    finally:
        reopened.close()


# --------------------------------------------------------------------------- #
# The questions lineage exists to answer
# --------------------------------------------------------------------------- #

def test_where_did_this_come_from(chain):
    assert set(chain.ancestors("ds:features")) == {"ds:cleaned", "ds:raw"}


def test_what_depends_on_this(chain):
    assert set(chain.descendants("ds:raw")) == {"ds:cleaned", "ds:features"}


def test_impact_of_renaming_a_column(chain):
    """The 2 a.m. question: upstream renamed a column, what breaks?"""
    impact = chain.impact_of_column_change("ds:raw", "amount")
    assert set(impact["downstream_datasets"]) == {"ds:cleaned", "ds:features"}
    assert "model:churn" in impact["models_affected"]
    assert impact["total_affected"] >= 3


def test_which_models_touched_personal_data(chain):
    """Currently unanswerable; one join once the edges exist."""
    hits = chain.models_touching_pii()
    assert len(hits) == 1
    assert hits[0]["model"] == "model:churn"
    assert hits[0]["pii_kinds"] == ["email"]


def test_a_model_on_clean_data_is_not_flagged(store):
    store.record_dataset("ds:clean", columns=[("units", "int64")])
    store.record_model("model:safe", trained_on="ds:clean", feature_columns=["units"])
    assert store.models_touching_pii() == []


def test_orphans_are_datasets_nothing_uses(chain, store):
    chain.record_dataset("ds:unused", label="scratch.csv")
    assert "ds:unused" in chain.orphans()
    assert "ds:features" not in chain.orphans(), "it has a model trained on it"


# --------------------------------------------------------------------------- #
# Safety: recording lineage must never fail an analysis
# --------------------------------------------------------------------------- #

def test_an_unwritable_path_disables_lineage_rather_than_raising(tmp_path):
    bad = LineageStore(tmp_path / "nope" / "\0" / "lineage.db")
    bad.record_dataset("ds:x", columns=[("a", "int64")])   # must not raise
    assert bad.ancestors("ds:x") == []
    assert bad.stats() == {}


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


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #

def test_stats_summarise_the_graph(chain):
    s = chain.stats()
    assert s["node_dataset"] == 3 and s["node_model"] == 1
    assert s["edge_derived_from"] == 2
    assert s["edge_classified_as"] == 2


def test_connection_provenance_is_recorded(chain):
    conn = chain.connect()
    row = conn.execute("SELECT dst FROM edge WHERE src='ds:raw' AND kind=?",
                       (EdgeKind.LOADED_FROM.value,)).fetchone()
    assert row[0] == "conn:s3"


def test_pii_is_found_through_lineage_not_just_the_training_frame(chain):
    """The naive query gives the wrong answer reassuringly: PII is usually
    dropped during cleaning, so the trained-on frame looks clean while the
    lineage runs straight through an email address."""
    hits = chain.models_touching_pii()
    assert len(hits) == 1
    assert hits[0]["model"] == "model:churn"
    assert hits[0]["pii_kinds"] == ["email"]
    # ds:features itself has no PII column — it was found upstream.
    assert set(hits[0]["via"]) == {"ds:raw", "ds:cleaned"}
