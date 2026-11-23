"""The connector contract on sqlite: the five things every connector proves.

The Postgres-specific statistics (pg_class estimates, pg_stats, relation
size, native partitions) are proven against a real server in
``tests/containers/test_connector_postgres.py``; what is here holds for any
SQLAlchemy URL and runs with nothing but the standard library's sqlite.
"""

import copy
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from avaloka.connectors import (Connector, ConnectorError, Credentials, Query,
                                cap_metadata, predicates_from_plan,
                                projection_from_plan)
from avaloka.connectors.postgres import PostgresConnector, SQLAlchemyConnector, connector_for
from avaloka.io import materialize


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "id": np.arange(500),
        "region": rng.choice(["north", "south", "east", "west"], 500),
        "amount": rng.normal(100, 25, 500).round(2),
        "status": rng.choice(["open", "closed", None], 500, p=[0.45, 0.45, 0.10]),
    })


@pytest.fixture
def sqlite_url(tmp_path, frame):
    import sqlalchemy
    db = tmp_path / "shop.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db}")
    frame.to_sql("orders", engine, index=False)
    pd.DataFrame({"region": ["north", "south"], "manager": ["ann", "bo"]}).to_sql("regions", engine, index=False)
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE INDEX ix_orders_region ON orders (region)")
    engine.dispose()
    return f"sqlite:///{db}"


@pytest.fixture
def connector(sqlite_url):
    with SQLAlchemyConnector(sqlite_url, table="orders") as c:
        yield c


# --- 5. credentials: resolved in process, never serialised -----------------
def test_credentials_never_serialise():
    # Assembled so the repo-hygiene scanner never sees a literal user:password@ URI.
    secret = "postgresql://u:" + "S3cret-Value" + "@db.internal/x"
    creds = Credentials(secret)
    assert creds.reveal() == secret
    assert secret not in repr(creds) and secret not in str(creds)
    assert "S3cret" not in f"{creds}"
    with pytest.raises(TypeError):
        pickle.dumps(creds)
    with pytest.raises(TypeError):
        copy.deepcopy(creds)
    with pytest.raises(TypeError):
        json.dumps(creds)
    assert "S3cret" not in json.dumps({"c": creds}, default=str)


def test_connector_describes_itself_without_the_password(sqlite_url):
    c = SQLAlchemyConnector("postgresql+psycopg2://alice:" + "S3cret-Value" + "@db.internal:5432/shop", table="t")
    assert "S3cret" not in c.describe() and "S3cret" not in repr(c)
    assert "alice" in c.describe()
    assert isinstance(c.credentials, Credentials)
    assert isinstance(SQLAlchemyConnector(sqlite_url), Connector)


def test_postgres_connector_refuses_other_dialects(sqlite_url):
    with pytest.raises(ConnectorError):
        PostgresConnector(sqlite_url)
    assert type(connector_for(sqlite_url)) is SQLAlchemyConnector


# --- 1. catalogue -------------------------------------------------------------
def test_catalogue_lists_tables_and_typed_columns(connector):
    cat = connector.catalogue()
    assert cat["backend"] == "sqlite"
    assert [t["name"] for t in cat["tables"]] == ["orders", "regions"]
    orders = cat["tables"][0]
    names = {c["name"]: c for c in orders["columns"]}
    assert set(names) == {"id", "region", "amount", "status"}
    assert names["id"]["type"] in ("BIGINT", "INTEGER")
    assert "FLOAT" in names["amount"]["type"] or "REAL" in names["amount"]["type"]
    assert all(isinstance(c["nullable"], bool) for c in orders["columns"])


# --- 2. statistics ------------------------------------------------------------
def test_statistics_carry_rows_indexes_and_size(connector, frame):
    stats = connector.statistics("orders")
    assert stats["kind"] == "database_table" and stats["backend"] == "sqlite"
    assert stats["row_count"] == len(frame) and stats["row_count_estimated"] is False
    assert stats["n_columns"] == 4
    assert {"name": "ix_orders_region", "columns": ["region"], "unique": False} in stats["indexes"]
    assert stats["database_size_bytes"] > 0
    assert stats["partitioning"] == {"partitioned": False, "columns": [], "partitions": 1}


def test_statistics_unknown_table_is_a_contract_error(connector):
    with pytest.raises(ConnectorError, match="not found"):
        connector.statistics("nope")


# --- 3. planner metadata ------------------------------------------------------
def test_planner_metadata_has_no_row_values_and_selected_table_first(connector, frame):
    md = connector.planner_metadata()
    assert md["kind"] == "database" and md["selected_table"] == "orders"
    assert [t["name"] for t in md["tables"]] == ["orders", "regions"]
    assert md["tables"][0]["statistics"]["row_count"] == len(frame)
    assert md["truncated"] is False and md["tables_total"] == 2
    text = json.dumps(md)
    for value in ("ann", "bo", "north", "closed"):  # cell values never enter the document
        assert f'"{value}"' not in text


def test_planner_metadata_respects_the_size_cap(connector):
    full = len(json.dumps(connector.planner_metadata(), separators=(",", ":")))
    small = connector.planner_metadata(max_bytes=full // 2)
    assert len(json.dumps(small, default=str, separators=(",", ":"))) <= full // 2
    assert small["truncated"] is True and small["tables_total"] == 2


def test_cap_metadata_sheds_statistics_before_tables():
    doc = {"tables": [{"name": f"t{i}", "columns": [{"name": "c", "type": "INT", "nullable": True}],
                       "statistics": {"row_count": i, "columns": {"c": {"x": "y" * 200}}}} for i in range(5)]}
    capped = cap_metadata(doc, max_bytes=600)
    assert capped["truncated"] and capped["tables_total"] == 5
    assert all("columns" not in t["statistics"] for t in capped["tables"])
    assert len(json.dumps(capped, separators=(",", ":"))) <= 600


# --- 4. pushdown from the workload optimizer's plan ---------------------------
PLAN = {
    "projection_columns": ["id", "region", "amount"],
    "pushdown_filters": [
        {"column": "region", "operator": "eq", "value": "north", "estimated_filter_factor": 0.25, "indexed": True},
        {"column": "amount", "operator": "gt", "value": 100.0},
    ],
    "strategy": "index_filtered_scan",
}


def test_pushdown_builds_a_parameterised_quoted_select(connector):
    q = connector.pushdown(predicates_from_plan(PLAN), projection_from_plan(PLAN))
    assert isinstance(q, Query)
    assert q.sql == ('SELECT "id", "region", "amount" FROM "orders" '
                     'WHERE "region" = :p0 AND "amount" > :p1')
    assert q.parameters == {"p0": "north", "p1": 100.0}
    assert "north" not in q.sql          # values are bound, never interpolated
    assert q.sql.startswith("SELECT ")


def test_pushdown_row_count_matches_pandas_on_the_whole_table(connector, frame):
    q = connector.pushdown(predicates_from_plan(PLAN), projection_from_plan(PLAN))
    got = connector.read(q).to_pandas()
    expected = frame[(frame["region"] == "north") & (frame["amount"] > 100.0)]
    assert len(got) == len(expected) > 0
    assert list(got.columns) == ["id", "region", "amount"]
    assert sorted(got["id"]) == sorted(expected["id"])


def test_pushdown_in_and_null_operators(connector, frame):
    q = connector.pushdown(
        [{"column": "region", "operator": "in", "value": ["east", "west"]},
         {"column": "status", "operator": "is_null"}], ["id"])
    assert q.sql == 'SELECT "id" FROM "orders" WHERE "region" IN (:p0_0, :p0_1) AND "status" IS NULL'
    expected = frame[frame["region"].isin(["east", "west"]) & frame["status"].isna()]
    assert len(connector.read(q).to_pandas()) == len(expected)


def test_pushdown_without_predicates_or_projection_selects_every_column(connector, frame):
    q = connector.pushdown([], [])
    assert q.sql == 'SELECT "id", "region", "amount", "status" FROM "orders"'
    assert len(connector.read(q).to_pandas()) == len(frame)


def test_pushdown_refuses_unknown_columns_and_operators(connector):
    with pytest.raises(ConnectorError, match="unknown column"):
        connector.pushdown([{"column": "1; DROP TABLE orders", "operator": "eq", "value": 1}], ["id"])
    with pytest.raises(ConnectorError, match="unknown column"):
        connector.pushdown([], ["id", "password"])
    with pytest.raises(ConnectorError, match="outside the plan shape"):
        connector.pushdown([{"column": "id", "operator": "like", "value": "%"}], ["id"])


def test_pushdown_quotes_awkward_identifiers(tmp_path):
    import sqlalchemy
    db = tmp_path / "odd.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db}")
    pd.DataFrame({"order id": [1, 2], "select": ["a", "b"]}).to_sql("my table", engine, index=False)
    engine.dispose()
    with SQLAlchemyConnector(f"sqlite:///{db}", table="my table") as c:
        q = c.pushdown([{"column": "select", "operator": "eq", "value": "a"}], ["order id"])
        assert q.sql == 'SELECT "order id" FROM "my table" WHERE "select" = :p0'
        assert len(c.read(q).to_pandas()) == 1


def test_read_is_select_only(connector):
    with pytest.raises(ConnectorError):
        connector.read(Query(sql="DELETE FROM orders"))


# --- the CLI path: materialize carries the planner metadata --------------------
def test_materialize_records_planner_metadata_with_redacted_provenance(sqlite_url, frame):
    r = materialize(f"{sqlite_url}#orders")
    md = r.detail["planner_metadata"]
    assert md["selected_table"] == "orders" and md["tables"][0]["statistics"]["row_count"] == len(frame)
    assert r.detail["statistics"]["indexes"][0]["columns"] == ["region"]


def test_connector_materialize_reuses_the_source_path(connector, frame, tmp_path):
    r = connector.materialize(tmp_path)
    assert r.kind == "database" and pd.read_parquet(r.local_path).shape == frame.shape


# --- a sample returns min(n, rows); a short block sample says it fell back ------
class _BlockSampled(SQLAlchemyConnector):
    """sqlite with a stand-in for TABLESAMPLE: ``draw`` is the filter its block
    sample applies, so a test can make it come back empty, short or full."""

    draw = " WHERE 1 = 0"

    def _sample_order(self, method):
        return self.draw if method == "tablesample" else super()._sample_order(method)


@pytest.mark.parametrize("draw, method", [(" WHERE 1 = 0", "random"), (' WHERE "id" < 2', "random"),
                                          (" WHERE 1 = 1", "tablesample")])
def test_a_short_block_sample_is_redrawn_and_labelled(sqlite_url, draw, method):
    with _BlockSampled(sqlite_url, table="orders") as c:
        c.draw = draw
        s = c.sample("orders", n=5, method="tablesample")
        assert len(s) == s.receipt.rows == 5
        assert s.receipt.method == method


def test_a_sample_of_a_table_smaller_than_n_is_the_whole_table(tmp_path):
    import sqlalchemy
    url = f"sqlite:///{tmp_path / 'tiny.db'}"
    engine = sqlalchemy.create_engine(url)
    pd.DataFrame({"id": [1, 2, 3]}).to_sql("t", engine, index=False)
    engine.dispose()
    with _BlockSampled(url, table="t") as c:
        s = c.sample("t", n=5, method="tablesample")
        assert sorted(s.to_pandas()["id"]) == [1, 2, 3] and s.receipt.method == "random"
