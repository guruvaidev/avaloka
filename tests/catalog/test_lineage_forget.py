"""T16.5: ``LineageStore.forget`` removes the dataset, its columns and vault
entries, and nothing of another tenant. Bite: delete by id without tenant."""

from __future__ import annotations

from sqlalchemy import create_engine, text

from app.core.lineage import LineageStore

COLUMNS = [("customer_email", "object"), ("amount", "float64")]


def _count(url, table):
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
    finally:
        engine.dispose()


def _chain(store):
    store.record_dataset("ds:raw", label="sales.csv", columns=COLUMNS, connection_id="conn:s3",
                         pii={"customer_email": "email"})
    store.record_dataset("ds:cleaned", derived_from="ds:raw", analysis_id="an:1", columns=COLUMNS)
    store.record_model("model:churn", trained_on="ds:cleaned", feature_columns=["amount"])


def test_forget_removes_the_dataset_its_columns_their_vault_rows_and_every_edge(db_url):
    a = LineageStore("tenant-a")
    _chain(a)
    before = {t: _count(db_url, t) for t in ("lineage_node", "lineage_edge", "lineage_name_vault")}
    email_id = a.column_id("ds:raw", "customer_email")
    assert a.resolve_column(email_id) == "customer_email"

    removed = a.forget("ds:raw")

    # the dataset node + 2 column nodes; HAS_COLUMN x2, LOADED_FROM, CLASSIFIED_AS, DERIVED_FROM (from cleaned)
    assert removed == {"nodes": 3, "edges": 5, "vault": 2}
    assert a.get_node("ds:raw") is None and a.get_node(email_id) is None
    assert a.resolve_column(email_id) is None
    assert a.get_node("ds:cleaned") is not None and a.get_node("model:churn") is not None
    assert a.dataset_lineage("ds:cleaned")["parents"] == []
    assert a.ancestors("ds:cleaned") == []
    after = {t: _count(db_url, t) for t in ("lineage_node", "lineage_edge", "lineage_name_vault")}
    assert before["lineage_node"] - after["lineage_node"] == 3
    assert before["lineage_edge"] - after["lineage_edge"] == 5
    assert before["lineage_name_vault"] - after["lineage_name_vault"] == 2
    assert a.forget("ds:raw") == {"nodes": 0, "edges": 0, "vault": 0}
    a.close()


def test_forget_is_tenant_scoped(db_url):
    """The same id under tenant B is another dataset and must survive."""
    a, b = LineageStore("tenant-a"), LineageStore("tenant-b")
    _chain(a)
    _chain(b)
    b_email = b.column_id("ds:raw", "customer_email")

    assert a.forget("ds:raw")["nodes"] == 3

    assert b.get_node("ds:raw") is not None
    assert b.resolve_column(b_email) == "customer_email"
    assert b.dataset_lineage("ds:cleaned")["parents"][0].id == "ds:raw"
    assert b.models_touching_pii() and b.models_touching_pii()[0]["model"] == "model:churn"
    a.close()
    b.close()


def test_forget_on_an_inert_store_is_a_no_op(db_url, monkeypatch):
    monkeypatch.delenv("AVALOKA_LINEAGE_DB_URL")
    assert LineageStore("tenant-a").forget("ds:raw") == {"nodes": 0, "edges": 0, "vault": 0}
    assert LineageStore(None, db_url).forget("ds:raw") == {"nodes": 0, "edges": 0, "vault": 0}
