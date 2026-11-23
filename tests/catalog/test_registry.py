"""T15.8 and the catalog half of 15b: every read through a connector writes
one catalog version (on change only) and one LOADED_FROM edge with the
source-version and query digests, and never the query text or a credential.

Runs on sqlite (SQLAlchemyConnector with a planted secret URL and an
injected engine) and on a local CSV (FileConnector). Bites: register a
version on every read; store the SQL text; put the DSN in the locator;
record the edge without the digest.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy import create_engine, text

from app.catalog.store import CatalogError, CatalogStore, CatalogUnavailable
from app.connectors import registry
from app.core.lineage import LineageStore
from avaloka.connectors import Credentials
from avaloka.connectors.files import FileConnector
from avaloka.connectors.postgres import SQLAlchemyConnector

SECRET = "S3cret-" + "Registry-77"
TABLE_SQL_MARKER = "orders_q3_2026"   # the table name: fine in the catalog, and the SQL text must not be stored


def _frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({"id": np.arange(n), "region": rng.choice(["north", "south"], n),
                         "amount": rng.normal(100, 25, n).round(2)})


@pytest.fixture
def sqlite_connector(tmp_path):
    frame = _frame()
    db = tmp_path / "shop.db"
    engine = create_engine(f"sqlite:///{db}")
    frame.to_sql(TABLE_SQL_MARKER, engine, index=False)
    answering = create_engine(f"sqlite:///{db}")
    c = SQLAlchemyConnector(Credentials(f"postgresql://u:{SECRET}@db.internal:5432/x"), table=TABLE_SQL_MARKER,
                            engine=answering)
    yield c, frame, answering
    c.close()
    answering.dispose()
    engine.dispose()


@pytest.fixture
def csv_connector(tmp_path):
    frame = _frame()
    path = tmp_path / "orders_q3_2026.csv"
    frame.to_csv(path, index=False)
    return FileConnector(path), frame, path


def _rows(url, table):
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return conn.execute(text(f"SELECT * FROM {table}")).mappings().all()
    finally:
        engine.dispose()


def test_a_read_registers_one_version_and_a_rereads_registers_none(db_url, sqlite_connector):
    c, frame, _ = sqlite_connector
    first = registry.record_read("user-a", c, c.read(), connection_id="conn-1", analysis_id="an:1")
    assert first.outcome == "new_dataset" and first.version_no == 1
    again = registry.record_read("user-a", c, c.read(), connection_id="conn-1", analysis_id="an:2")
    assert again.outcome == "unchanged" and again.version_id == first.version_id
    store = CatalogStore("user-a")
    detail = store.get(first.dataset_id)
    assert detail["kind"] == "db_table" and detail["display_name"] == TABLE_SQL_MARKER
    assert detail["locator"] == {"backend": "sqlite", "table": TABLE_SQL_MARKER, "connection_id": "conn-1"}
    assert [v["version_no"] for v in detail["versions"]] == [1]
    assert detail["versions"][0]["row_count"] == len(frame) and detail["versions"][0]["row_count_kind"] == "exact"
    assert detail["versions"][0]["source_version"]["marker"]["row_count"] == len(frame)
    assert detail["analyses"] == ["an:1", "an:2"]
    assert [c for c, _ in [tuple(x) for x in detail["versions"][0]["columns"]]] == ["id", "region", "amount"]


def test_a_changed_source_is_a_new_version_pinned_by_the_analysis_that_read_it(db_url, sqlite_connector):
    """Bite: version on every read; or ignore the receipt's source_version."""
    c, frame, engine = sqlite_connector
    v1 = registry.record_read("user-a", c, c.read(), connection_id="conn-1", analysis_id="an:old")
    with engine.begin() as conn:
        conn.exec_driver_sql(f'INSERT INTO "{TABLE_SQL_MARKER}" (id, region, amount) VALUES (9999, "north", 1.0)')
    v2 = registry.record_read("user-a", c, c.read(), connection_id="conn-1", analysis_id="an:new")
    assert v2.outcome == "new_version" and v2.version_no == 2 and v2.dataset_id == v1.dataset_id
    store = CatalogStore("user-a")
    assert store.resolve("analysis:an:old").version_no == 1
    assert store.resolve("analysis:an:new").version_no == 2
    assert store.resolve(f"my.{TABLE_SQL_MARKER}").version_no == 2
    versions = store.get(v1.dataset_id)["versions"]
    assert versions[0]["source_version"]["marker"]["row_count"] == len(frame)
    assert versions[1]["source_version"]["marker"]["row_count"] == len(frame) + 1


def test_a_truncated_read_records_a_floor_not_an_exact_count(db_url, csv_connector):
    c, frame, _ = csv_connector
    reg = registry.record_read("user-a", c, c.read(max_rows=10, max_bytes=1 << 30))
    v = CatalogStore("user-a").get(reg.dataset_id)["versions"][0]
    assert v["row_count"] == 10 and v["row_count_kind"] == "estimate"
    full = registry.record_read("user-a", c, c.read())
    assert full.outcome == "unchanged", "the schema and content did not change: the bound is not a version"


def test_the_lineage_edge_carries_digests_and_never_the_query_text_or_the_secret(db_url, sqlite_connector):
    """T15.8. Bite: store query text; put the DSN in the locator or the label."""
    c, frame, _ = sqlite_connector
    q = c.pushdown([{"column": "region", "operator": "eq", "value": "north"}], ["id", "amount"])
    result = c.read(q)
    reg = registry.record_read("user-a", c, result, connection_id="conn-1", analysis_id="an:1")
    graph = LineageStore("user-a")
    node = graph.get_node(reg.dataset_id)
    assert node is not None and node.kind.value == "dataset" and node.attrs["kind"] == "db_table"
    assert node.attrs["query_digest"] == result.receipt.query_digest and len(node.attrs["query_digest"]) == 24
    assert node.attrs["source_version"] == result.receipt.source_version.digest
    graph.close()
    everything = json.dumps([[dict(r) for r in _rows(db_url, t)] for t in
                             ("lineage_node", "lineage_edge", "lineage_name_vault", "catalog_dataset",
                              "catalog_dataset_version", "source_schema_snapshots")], default=str)
    assert SECRET not in everything and "db.internal" not in everything
    assert "SELECT" not in everything and "WHERE" not in everything, "query text must not be stored"
    assert TABLE_SQL_MARKER in everything, "the table's name is the dataset's name; it belongs in the catalog"
    edges = [dict(r) for r in _rows(db_url, "lineage_edge")]
    loaded = [e for e in edges if e["kind"] == "loaded_from"]
    assert len(loaded) == 1 and loaded[0]["src"] == reg.dataset_id and loaded[0]["dst"] == "conn-1"
    attrs = json.loads(loaded[0]["attrs"])
    assert attrs == {"source_version": result.receipt.source_version.digest, "query_digest": result.receipt.query_digest}


def test_a_file_read_registers_a_file_dataset_with_its_name_and_format(db_url, csv_connector):
    c, frame, path = csv_connector
    result, reg = registry.read_and_record("user-a", c, analysis_id="an:7")
    assert len(result) == len(frame) and reg is not None and reg.outcome == "new_dataset"
    store = CatalogStore("user-a")
    item = store.get(reg.dataset_id)
    assert item["kind"] == "file" and item["locator"] == {"filename": "orders_q3_2026.csv", "format": "csv"}
    assert item["display_name"] == "orders_q3_2026" and store.resolve("my.orders_q3_2026").pinned_by is None
    assert item["versions"][0]["source_version"]["marker"]["sha256"]
    with path.open("a") as fh:
        fh.write("9999,north,1.0\n")
    _, reg2 = registry.read_and_record("user-a", c, analysis_id="an:8")
    assert reg2.outcome == "new_version" and store.resolve("analysis:an:7").version_no == 1


def test_an_unknown_connector_kind_is_refused_not_guessed(db_url, csv_connector):
    c, _, _ = csv_connector

    class Odd(FileConnector):
        name = "carrier-pigeon"

    odd = Odd(c.path)
    with pytest.raises(CatalogError, match="no catalog dataset kind"):
        registry.record_read("user-a", odd, odd.read())
    assert registry.try_record_read("user-a", odd, odd.read()) is None
    assert CatalogStore("user-a").list() == []


def test_a_locator_that_carries_a_credential_is_refused_even_from_a_connector(db_url, csv_connector):
    c, _, _ = csv_connector

    class Leaky(FileConnector):
        def locator_for(self, ref=None):
            return {"filename": self.path.name, "dsn": "postgresql://u:p@h/db"}

    leaky = Leaky(c.path)
    with pytest.raises(CatalogError, match="credential"):
        registry.record_read("user-a", leaky, leaky.read())
    assert CatalogStore("user-a").list() == []
    fallback = registry.locator_for(type("Bare", (), {"name": "file", "describe": lambda self: "file:x.csv"})(), "t")
    assert fallback == {"connector": "file", "ref": "t", "source": "file:x.csv"}


def test_try_record_read_never_raises_without_a_catalog(db_url, csv_connector, monkeypatch):
    c, _, _ = csv_connector
    monkeypatch.delenv("AVALOKA_LINEAGE_DB_URL")
    assert registry.try_record_read("user-a", c, c.read()) is None
    assert registry.try_record_read(None, c, c.read()) is None
    with pytest.raises(CatalogUnavailable):
        registry.record_read("user-a", c, c.read())


def test_reads_are_the_callers_own(db_url, csv_connector):
    c, _, _ = csv_connector
    registry.record_read("user-a", c, c.read())
    assert CatalogStore("user-b").list() == []
    assert LineageStore("user-b").get_node(CatalogStore("user-a").list()[0]["dataset_id"]) is None
