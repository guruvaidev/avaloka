"""CatalogStore: versions on change, owner isolation, reuse, delete.

Design ids are from docs/design/1.7-DATA-PLATFORM.md §F.C. Each test names
its bite in the docstring: the mutation that would turn it red.
"""

from __future__ import annotations

import json
import pickle

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from app.catalog.store import (Binding, CatalogError, CatalogStore, CatalogUnavailable, KINDS,
                               catalog_event, check_locator, normalise_columns, schema_diff,
                               schema_fingerprint, size_bucket, slug)
from app.core.lineage import LineageStore

from tests.catalog.conftest import ORDERS

LOCATOR = {"dataset_id": "u1", "filename": "orders_2026.csv", "format": "csv"}


def _uri_with_password(scheme: str = "postgresql") -> str:
    """Assembled at run time: no tracked file carries a credential-shaped literal."""
    return scheme + "://" + "user" + ":" + "pw-" + "planted" + "@" + "host/db"


def _rows(url, table):
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return conn.execute(text(f"SELECT * FROM {table}")).mappings().all()
    finally:
        engine.dispose()


# --------------------------------------------------------------------------- #
# T16.1 versions are written on change only
# --------------------------------------------------------------------------- #

def test_reading_the_same_schema_twice_writes_one_version(store, db_url):
    """Bite: version on every read."""
    first = store.register("file", LOCATOR, ORDERS, dataset_id="u1", display_name="orders_2026.csv")
    again = store.register("file", LOCATOR, list(reversed(ORDERS)), dataset_id="u1")
    assert first.outcome == "new_dataset" and again.outcome == "unchanged"
    assert again.version_id == first.version_id and again.version_no == 1
    assert len(_rows(db_url, "catalog_dataset_version")) == 1
    assert len(_rows(db_url, "source_schema_snapshots")) == 1


def test_a_retyped_column_writes_a_second_version_and_the_diff_names_it(store):
    """Bite: compare names only, not types; or drop the diff."""
    store.register("file", LOCATOR, ORDERS, dataset_id="u1", display_name="orders_2026.csv")
    retyped = [("order_id", "int64"), ("customer_email", "object"), ("amount", "object")]
    second = store.register("file", LOCATOR, retyped, dataset_id="u1")
    assert second.outcome == "new_version" and second.version_no == 2
    detail = store.get("u1")
    assert [v["version_no"] for v in detail["versions"]] == [1, 2]
    assert detail["schema_change"] == {"added": [], "removed": [],
                                       "retyped": [("amount", "float64", "object")]}


def test_a_changed_source_version_is_a_new_version_even_with_the_same_schema(store):
    """Bite: compare the fingerprint only. An ETag that moved means new data."""
    store.register("object", {"connection_id": "c1", "bucket": "b", "key": "k.parquet"}, ORDERS,
                   source_version={"etag": "aaa"})
    same = store.register("object", {"connection_id": "c1", "bucket": "b", "key": "k.parquet"}, ORDERS,
                          source_version={"etag": "aaa"})
    moved = store.register("object", {"connection_id": "c1", "bucket": "b", "key": "k.parquet"}, ORDERS,
                           source_version={"etag": "bbb"})
    assert same.outcome == "unchanged" and moved.outcome == "new_version" and moved.version_no == 2
    assert same.dataset_id == moved.dataset_id


def test_fingerprint_ignores_column_order_and_type_case_but_not_names_or_types():
    assert schema_fingerprint(ORDERS) == schema_fingerprint(list(reversed(ORDERS)))
    assert schema_fingerprint([("a", "Int64")]) == schema_fingerprint([("a", "int64")])
    assert schema_fingerprint([("a", "int64")]) != schema_fingerprint([("b", "int64")])
    assert schema_fingerprint([("a", "int64")]) != schema_fingerprint([("a", "float64")])
    assert normalise_columns(pd.DataFrame({"x": [1], "y": ["a"]}).dtypes) == [("x", "int64"), ("y", "object")]
    assert schema_diff([("a", "i"), ("b", "i")], [("b", "f"), ("c", "i")]) == {
        "added": ["c"], "removed": ["a"], "retyped": [("b", "i", "f")]}


# --------------------------------------------------------------------------- #
# T16.2 (unit shape): the owner is the only predicate
# --------------------------------------------------------------------------- #

def test_another_user_sees_nothing_of_mine(store, other):
    """Bite: drop the owner predicate from any read. On Postgres the RLS
    policy in the migration is the second lock; this checks the store's own."""
    reg = store.register("file", LOCATOR, ORDERS, dataset_id="u1", display_name="orders_2026.csv",
                         analysis_id="an:1")
    store.add_terms("u1", ["revenue"])
    assert other.list() == [] and other.list(query="orders") == [] and other.get("u1") is None
    with pytest.raises(CatalogError):
        other.resolve("my.orders_2026")
    with pytest.raises(CatalogError):
        other.resolve("analysis:an:1")
    with pytest.raises(CatalogError):
        other.forget("u1")
    with pytest.raises(CatalogError):
        other.add_terms("u1", ["x"])
    # and mine is still there, untouched
    assert store.get("u1")["version_id"] == reg.version_id and store.get("u1")["terms"] == ["revenue"]


def test_the_same_locator_under_two_owners_is_two_datasets(store, other):
    a = store.register("db_table", {"connection_id": "c1", "table": "orders"}, ORDERS)
    b = other.register("db_table", {"connection_id": "c1", "table": "orders"}, ORDERS)
    assert a.dataset_id != b.dataset_id   # derived per owner: no shared id to collide on
    assert [d["dataset_id"] for d in store.list()] == [a.dataset_id]
    assert [d["dataset_id"] for d in other.list()] == [b.dataset_id]
    assert store.get(a.dataset_id)["version_no"] == 1


def test_a_store_needs_an_owner_and_a_database(db_url, monkeypatch):
    with pytest.raises(CatalogError):
        CatalogStore("")
    monkeypatch.delenv("AVALOKA_LINEAGE_DB_URL")
    with pytest.raises(CatalogUnavailable):
        CatalogStore("user-a")


# --------------------------------------------------------------------------- #
# T16.7 reuse: find, my.<name>, the version an analysis used
# --------------------------------------------------------------------------- #

def test_find_matches_name_tag_term_and_analysis(store):
    store.register("file", LOCATOR, ORDERS, dataset_id="u1", display_name="orders_2026.csv",
                   analysis_id="an:churn", tags=["finance"])
    store.register("file", {"dataset_id": "u2", "filename": "hr_roster.xlsx"}, [("name", "object")],
                   dataset_id="u2", display_name="hr_roster.xlsx")
    store.add_terms("u1", ["revenue"])
    for needle in ("orders", "my.orders_2026", "finance", "revenue", "an:churn", "ORDERS"):
        assert [d["dataset_id"] for d in store.find(query=needle)] == ["u1"], needle
    assert [d["dataset_id"] for d in store.find(kind="file")] == ["u2", "u1"]   # newest first
    assert store.find(query="payroll") == []
    hit = store.find(query="orders")[0]
    assert hit["name"] == "my.orders_2026" and hit["used_by"] == 1
    assert hit["summary"].startswith("orders_2026.csv: file, 3 columns")


def test_the_one_from_analysis_x_pins_that_version_not_the_latest(store):
    """Bite: return the latest version for an analysis reference."""
    v1 = store.register("file", LOCATOR, ORDERS, dataset_id="u1", display_name="orders_2026.csv",
                        analysis_id="an:churn")
    v2 = store.register("file", LOCATOR, ORDERS + [("region", "object")], dataset_id="u1",
                        analysis_id="an:forecast")
    assert v2.version_no == 2
    latest = store.resolve("my.orders_2026")
    pinned = store.resolve("analysis:an:churn")
    by_words = store.resolve("from analysis an:churn")
    assert isinstance(latest, Binding) and latest.version_id == v2.version_id and latest.pinned_by is None
    assert pinned.version_id == v1.version_id and pinned.version_no == 1 and pinned.pinned_by == "an:churn"
    assert by_words == pinned
    assert store.resolve("my.orders_2026@1").version_id == v1.version_id
    assert store.resolve("u1").version_id == v2.version_id
    assert [c[0] for c in pinned.columns] == ["order_id", "customer_email", "amount"]
    assert pinned.as_dict()["locator"] == LOCATOR


def test_resolve_refuses_ambiguity_and_unknowns_in_words(store):
    store.register("file", {"dataset_id": "u1", "filename": "orders.csv"}, ORDERS, dataset_id="u1",
                   display_name="orders.csv", analysis_id="an:1")
    store.register("file", {"dataset_id": "u2", "filename": "orders.parquet"}, ORDERS, dataset_id="u2",
                   display_name="orders.parquet", analysis_id="an:1")
    with pytest.raises(CatalogError, match="names 2 of your datasets"):
        store.resolve("my.orders")
    with pytest.raises(CatalogError, match="used 2 datasets"):
        store.resolve("analysis:an:1")
    assert store.resolve("analysis:an:1/u2").dataset_id == "u2"
    with pytest.raises(CatalogError, match="no dataset of yours is called"):
        store.resolve("my.payroll")
    with pytest.raises(CatalogError, match="has no version 7"):
        store.resolve("u1@7")
    with pytest.raises(CatalogError):
        store.resolve("")


def test_slug_is_how_chat_names_a_dataset():
    assert slug("Orders 2026.csv") == "orders_2026"
    assert slug("Q3 Layoffs (final).xlsx") == "q3_layoffs_final"
    assert slug("") == "dataset"


# --------------------------------------------------------------------------- #
# Locators carry no secrets; events carry no names
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("locator", [
    {"dsn": _uri_with_password()},
    {"connection_id": "c1", "password": "hunter2"},
    {"bucket": "b", "aws_secret_access_key": "x"},
    {"url": "https://acct.blob.core.windows.net/c/k?sig=abc"},
    {"url": "s3://b/k?X-Amz-Signature=deadbeef"},
    {"nested": {"token": "t"}},
    {"path": _uri_with_password()},
])
def test_a_locator_that_carries_a_credential_is_refused(store, locator):
    """Bite: accept any mapping."""
    with pytest.raises(CatalogError):
        check_locator(locator)
    with pytest.raises(CatalogError):
        store.register("db_table", locator, ORDERS)
    assert store.list() == []


def test_a_plain_locator_passes_and_a_bad_kind_does_not(store):
    assert check_locator({"connection_id": "c1", "schema": "public", "table": "orders"})
    with pytest.raises(CatalogError, match="unknown dataset kind"):
        store.register("spreadsheet", LOCATOR, ORDERS)
    with pytest.raises(CatalogError):
        store.register("file", LOCATOR, ORDERS, row_count_kind="guess")


def test_events_are_a_closed_vocabulary(store, events):
    """Bite: add the display name or the locator to an event."""
    store.register("file", LOCATOR, ORDERS, dataset_id="u1", display_name="patient_hiv_status.csv",
                   row_count=250_000, row_count_kind="exact", tags=["q3_layoffs_final"])
    assert events == [{"event": "catalog.register", "kind": "file", "outcome": "new_dataset",
                       "rows": "<10m", "columns": "<1k"}]
    assert set(events[0]) == {"event", "kind", "outcome", "rows", "columns"}
    assert size_bucket(None) == "unknown" and size_bucket(999) == "<1k" and size_bucket(10**8) == ">=10m"
    with pytest.raises(CatalogError):
        catalog_event("register", kind="spreadsheet", outcome="x")
    assert json.dumps(events[0])   # serialisable, and nothing in it is a name


def test_events_never_fail_a_registration(db_url):
    def boom(event):
        raise RuntimeError("sink down")
    store = CatalogStore("user-a", on_event=boom)
    assert store.register("file", LOCATOR, ORDERS, dataset_id="u1").outcome == "new_dataset"


def test_bindings_and_registrations_are_plain_data(store):
    reg = store.register("file", LOCATOR, ORDERS, dataset_id="u1", display_name="orders_2026.csv")
    binding = store.resolve("my.orders_2026")
    assert pickle.loads(pickle.dumps(binding)) == binding
    assert json.loads(json.dumps(binding.as_dict()))["version_id"] == reg.version_id
    assert KINDS[0] == "file" and "derived" in KINDS


# --------------------------------------------------------------------------- #
# T16.4 (unit shape) delete
# --------------------------------------------------------------------------- #

def test_forget_removes_everything_of_mine_and_leaves_a_nameless_tombstone(store, db_url, events):
    """Bite: skip the vault rows; keep the name on the tombstone; delete the
    derived dataset too; reuse the tombstone's id for a re-registration."""
    store.register("file", LOCATOR, ORDERS, dataset_id="u1", display_name="orders_2026.csv",
                   analysis_id="an:1", tags=["finance"])
    store.register("file", LOCATOR, ORDERS + [("region", "object")], dataset_id="u1", analysis_id="an:2")
    store.add_terms("u1", ["revenue"], column="amount")
    graph = LineageStore("user-a")
    graph.record_dataset("u1", label="orders_2026.csv", columns=ORDERS, connection_id="conn:upload")
    graph.record_dataset("ds:out", label="churn.csv", derived_from="u1", analysis_id="an:1",
                         columns=[("churned", "int64")])
    graph.close()

    report = store.forget("u1")

    assert report["tombstone"] is True and report["derived_kept"] == ["ds:out"]
    assert report["removed"]["versions"] == 2 and report["removed"]["uses"] == 2
    assert report["removed"]["terms"] == 1 and report["removed"]["snapshots"] == 2
    assert report["removed"]["lineage"] == {"nodes": 4, "edges": 5, "vault": 3}
    assert report["source_system"] == "untouched" and "upload object" in report["not_reached"]
    # the tombstone: id, kind, time -- no name, locator, tags
    (row,) = _rows(db_url, "catalog_dataset")
    assert row["deleted_at"] is not None and row["display_name"] == "" and row["locator"] == "{}" \
        and row["tags"] == "[]"
    assert _rows(db_url, "catalog_dataset_version") == [] and _rows(db_url, "catalog_term") == []
    assert _rows(db_url, "catalog_dataset_use") == [] and _rows(db_url, "source_schema_snapshots") == []
    # the graph: the dataset and its columns are gone, the derived dataset stays
    graph = LineageStore("user-a")
    assert graph.get_node("u1") is None and graph.get_node("ds:out") is not None
    assert graph.resolve_column(graph.column_id("u1", "customer_email")) is None
    assert graph.dataset_lineage("ds:out")["parents"] == []
    graph.close()
    # reads: gone from the list, a tombstone on get, not resolvable
    assert store.list() == [] and store.get("u1")["tombstone"] is True
    with pytest.raises(CatalogError):
        store.resolve("my.orders_2026")
    assert store.forget("u1") == {"dataset_id": "u1", "already_deleted": True}
    assert events[-1] == {"event": "catalog.forget", "kind": "file", "outcome": "deleted",
                          "rows": "unknown", "columns": "unknown"}
    # a re-registration of the same place is a new dataset with a new id
    again = store.register("file", LOCATOR, ORDERS, display_name="orders_2026.csv")
    assert again.outcome == "new_dataset" and again.dataset_id != "u1"


def test_forget_keeps_a_snapshot_another_dataset_still_uses(store, db_url):
    store.register("file", LOCATOR, ORDERS, dataset_id="u1")
    store.register("file", {"dataset_id": "u2", "filename": "orders_copy.csv"}, ORDERS, dataset_id="u2")
    assert len(_rows(db_url, "source_schema_snapshots")) == 1
    assert store.forget("u1")["removed"]["snapshots"] == 0
    assert len(_rows(db_url, "source_schema_snapshots")) == 1
    assert store.get("u2")["columns"] == 3


def test_forget_touches_nothing_of_another_user(store, other):
    store.register("file", LOCATOR, ORDERS, dataset_id="u1", display_name="orders_2026.csv")
    other.register("file", {"dataset_id": "u9", "filename": "orders_2026.csv"}, ORDERS, dataset_id="u9",
                   display_name="orders_2026.csv")
    store.forget("u1")
    assert [d["dataset_id"] for d in other.list()] == ["u9"]
    assert other.resolve("my.orders_2026").dataset_id == "u9"
