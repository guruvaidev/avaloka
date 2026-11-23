"""The Postgres connector against a real ``postgres:16``.

What sqlite cannot show (``tests/avaloka/test_connectors.py`` covers the rest):
the planner's row estimate from ``pg_class``, ``pg_stats`` null and distinct
fractions after ANALYZE, relation size, a native partitioned table's
partition columns and count, and a pushed-down predicate whose row count
equals pandas over the whole table. Needs ``AVALOKA_IT_POSTGRES_URL`` (the
``postgres_url`` fixture); it carries ``integration`` and ``containers`` and,
like everything in this directory, never skips.
"""

from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from avaloka.connectors import ConnectorError, predicates_from_plan, projection_from_plan
from avaloka.connectors.postgres import PostgresConnector, connector_for

from tests.avaloka.connector_conformance import CHECKS, Case


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    n = 2000
    return pd.DataFrame({
        "id": np.arange(n),
        "region": rng.choice(["north", "south", "east", "west"], n),
        "amount": rng.normal(100, 25, n).round(2),
        "status": rng.choice(["open", "closed", None], n, p=[0.45, 0.45, 0.10]),
        "year": rng.choice([2023, 2024], n),
    })


@pytest.fixture(scope="module")
def schema(postgres_url, frame):
    """A schema of this module's own, with a plain table and a partitioned one."""
    name = "it_" + uuid.uuid4().hex[:12]
    engine = create_engine(postgres_url)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {name}"))
        conn.execute(text(f"CREATE TABLE {name}.orders (id BIGINT PRIMARY KEY, region TEXT NOT NULL, "
                          f"amount DOUBLE PRECISION, status TEXT, year INTEGER)"))
        conn.execute(text(f"CREATE INDEX ix_orders_region ON {name}.orders (region)"))
        conn.execute(text(f"CREATE TABLE {name}.events (id BIGINT, year INTEGER NOT NULL, region TEXT) "
                          f"PARTITION BY LIST (year)"))
        for year in (2023, 2024):
            conn.execute(text(f"CREATE TABLE {name}.events_{year} PARTITION OF {name}.events "
                              f"FOR VALUES IN ({year})"))
    frame.to_sql("orders", engine, schema=name, index=False, if_exists="append")
    frame[["id", "year", "region"]].to_sql("events", engine, schema=name, index=False, if_exists="append")
    with engine.begin() as conn:
        conn.execute(text(f"ANALYZE {name}.orders"))
        conn.execute(text(f"ANALYZE {name}.events"))
    engine.dispose()
    yield name
    engine = create_engine(postgres_url)
    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA {name} CASCADE"))
    engine.dispose()


@pytest.fixture
def connector(postgres_url, schema):
    with PostgresConnector(postgres_url, schema=schema, table="orders") as c:
        yield c


def test_connector_for_picks_the_proving_connector(postgres_url):
    c = connector_for(postgres_url)
    assert type(c) is PostgresConnector and c.name == "postgresql"
    assert "example" not in c.describe()  # the runner's password never shows
    c.close()


def test_catalogue_lists_tables_and_typed_columns(connector):
    cat = connector.catalogue()
    assert cat["backend"] == "postgresql"
    names = [t["name"] for t in cat["tables"]]
    assert "orders" in names and "events" in names and "events_2023" in names
    orders = next(t for t in cat["tables"] if t["name"] == "orders")
    cols = {c["name"]: c for c in orders["columns"]}
    assert cols["id"]["type"] == "BIGINT" and cols["id"]["nullable"] is False
    assert cols["amount"]["type"] == "DOUBLE PRECISION"
    assert cols["region"]["nullable"] is False and cols["status"]["nullable"] is True


def test_statistics_expose_what_postgres_knows(connector, frame):
    stats = connector.statistics("orders")
    assert stats["backend"] == "postgresql" and stats["kind"] == "database_table"
    # pg_class.reltuples after ANALYZE is exact for a table this small.
    assert stats["row_count"] == len(frame) and stats["row_count_estimated"] is True
    assert stats["size_bytes"] > 0
    by_name = {i["name"]: i for i in stats["indexes"]}
    assert by_name["ix_orders_region"]["columns"] == ["region"]
    primary = next(i for i in stats["indexes"] if i.get("primary"))
    assert primary["columns"] == ["id"] and primary["unique"] is True
    # pg_stats: null fraction of status is about the 10% we planted; region has 4 distinct values.
    assert abs(stats["columns"]["status"]["null_fraction"] - frame["status"].isna().mean()) < 0.05
    assert stats["columns"]["region"]["n_distinct"] == 4.0
    assert stats["partitioning"]["partitioned"] is False


def test_statistics_see_native_partitions(connector):
    stats = connector.statistics("events")
    assert stats["partitioning"] == {"partitioned": True, "columns": ["year"], "partitions": 2, "kind": "native"}


def test_planner_metadata_is_capped_and_carries_no_rows(connector, frame):
    md = connector.planner_metadata(max_bytes=4096)
    assert md["backend"] == "postgresql" and md["selected_table"] == "orders"
    assert md["tables"][0]["name"] == "orders"
    assert md["tables"][0]["statistics"]["row_count"] == len(frame)
    assert len(json.dumps(md, default=str, separators=(",", ":"))) <= 4096
    text_ = json.dumps(md)
    assert '"north"' not in text_ and '"closed"' not in text_


PLAN = {
    "projection_columns": ["id", "region", "amount"],
    "pushdown_filters": [
        {"column": "region", "operator": "eq", "value": "north", "indexed": True},
        {"column": "amount", "operator": "gte", "value": 110.0},
        {"column": "status", "operator": "not_null"},
    ],
}


def test_pushdown_row_count_equals_pandas_on_the_whole_table(connector, frame, schema):
    q = connector.pushdown(predicates_from_plan(PLAN), projection_from_plan(PLAN))
    assert q.sql == (f'SELECT "id", "region", "amount" FROM "{schema}"."orders" '
                     f'WHERE "region" = :p0 AND "amount" >= :p1 AND "status" IS NOT NULL')
    assert q.parameters == {"p0": "north", "p1": 110.0}
    got = connector.read(q).to_pandas()
    expected = frame[(frame["region"] == "north") & (frame["amount"] >= 110.0) & frame["status"].notna()]
    assert len(got) == len(expected) > 0
    assert sorted(got["id"]) == sorted(expected["id"])


def test_pushdown_on_the_partitioned_table_prunes_by_partition_column(connector, frame):
    q = connector.pushdown([{"column": "year", "operator": "in", "value": [2024]}], ["id"], table="events")
    got = connector.read(q).to_pandas()
    assert len(got) == int((frame["year"] == 2024).sum())
    # Postgres only scans the 2024 partition for this predicate.
    with connector.engine.connect() as conn:
        plan = "\n".join(r[0] for r in conn.execute(
            text("EXPLAIN " + q.sql), {"p0_0": 2024}).fetchall())
    assert "events_2024" in plan and "events_2023" not in plan


def test_pushdown_refuses_what_the_plan_shape_does_not_allow(connector):
    with pytest.raises(ConnectorError):
        connector.pushdown([{"column": "region", "operator": "like", "value": "n%"}], ["id"])
    with pytest.raises(ConnectorError):
        connector.pushdown([], ["id", "no_such_column"])


# --------------------------------------------------------------------------- #
# T15.1 / T15.9: the conformance suite against a real server. The same seven
# checks tests/avaloka/test_connector_conformance.py runs on sqlite and files;
# PostgresConnector is CONTAINER_ONLY there and proves itself here.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("check", CHECKS, ids=lambda f: f.__name__.replace("check_", ""))
def test_conformance(postgres_url, schema, frame, check):
    from urllib.parse import urlsplit
    secret = urlsplit(postgres_url).password
    engine = create_engine(postgres_url)

    def mutate() -> None:
        with engine.begin() as conn:
            conn.execute(text(f"INSERT INTO {schema}.orders (id, region, amount, status, year) "
                              f"VALUES (900000, 'north', 1.0, 'open', 2024)"))

    with PostgresConnector(postgres_url, schema=schema, table="orders") as c:
        try:
            check(Case(connector=c, frame=frame, ref="orders", secret=secret or None, mutate=mutate,
                       sample_methods=("head", "random", "tablesample")))
        finally:
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {schema}.orders WHERE id = 900000"))
            engine.dispose()



# --------------------------------------------------------------------------- #
# A sample returns min(n, rows). TABLESAMPLE SYSTEM keeps whole pages at a
# fixed rate, so on a table of one page it returns every row or none; the
# connector re-draws by ``random`` when the block sample comes back short and
# says so in the receipt. Ten draws each, so a sampler that only sometimes
# comes back short cannot pass by luck.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rows", [3, 20])
def test_tablesample_on_a_small_table_returns_min_of_n_and_rows(postgres_url, schema, rows):
    engine = create_engine(postgres_url)
    table = f"small_{rows}"
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE {schema}.{table} (id BIGINT PRIMARY KEY, region TEXT)"))
        conn.execute(text(f"INSERT INTO {schema}.{table} SELECT g, 'north' FROM generate_series(1, {rows}) g"))
        conn.execute(text(f"ANALYZE {schema}.{table}"))
    try:
        with PostgresConnector(postgres_url, schema=schema, table=table) as c:
            for _ in range(10):
                s = c.sample(table, n=5, method="tablesample")
                assert len(s) == s.receipt.rows == min(5, rows)
                assert s.receipt.method in ("tablesample", "random")
                assert len(set(s.to_pandas()["id"])) == len(s)
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE {schema}.{table}"))
        engine.dispose()
