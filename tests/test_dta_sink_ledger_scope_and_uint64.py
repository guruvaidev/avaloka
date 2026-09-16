"""Two silent-corruption edges in the SQL sink's retry ledger / unsigned downcast.

Both are cases where the guard itself could turn a loud failure into a wrong result
reported as success — the exact failure mode the ledger and the downcast exist to
prevent:

1. The ledger is shared across every transfer to a destination and is NEVER pruned,
   so its key must identify the job *and* what it wrote. Keyed on job_id alone, a
   collision against any past row makes a NEW transfer skip its insert and report the
   old row count as success — an empty table, "successfully" transferred.
2. _downcast_unsigned_ints sees EVERY unsigned column, not just counts/ranks. A real
   uint64 >= 2**63 (snowflake/hash ids) cast to int64 wraps negative with no error.

See tests/test_dta_sink_idempotency.py for the primary retry-dedup contract; this
file only pins the edges. Same file-based SQLite setup (see that module's docstring).
"""
import numpy as np
import pandas as pd
import sqlalchemy as sa
from daft.io.sink import WriteResult

from app.agents.data_transfer_agent.datasink.sql_sink import (
    PostgresDataSink,
    _downcast_unsigned_ints,
)


def _url(tmp_path):
    return f"sqlite:///{tmp_path / 'dest.db'}"


def _wr(n):
    return [WriteResult(result={"status": "success"}, rows_written=n, bytes_written=0)]


def _make_table(url, table):
    eng = sa.create_engine(url)
    with eng.begin() as c:
        c.exec_driver_sql(f"CREATE TABLE {table} (id INTEGER, name TEXT)")
    eng.dispose()


def _sink(url, job_id, table):
    return PostgresDataSink(url, table, write_mode="append",
                            columns=["id", "name"], job_id=job_id)


def _stage(sink, url, rows):
    eng = sa.create_engine(url)
    stg = sink._qi(sink._staging_table)
    with eng.begin() as c:
        for rid, name in rows:
            c.exec_driver_sql(f"INSERT INTO {stg} (id, name) VALUES ({rid}, '{name}')")
    eng.dispose()


def _count(url, table):
    eng = sa.create_engine(url)
    try:
        with eng.begin() as c:
            return c.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        eng.dispose()


# ── ledger scoping ────────────────────────────────────────────────────────────

def test_same_job_id_to_a_different_table_still_inserts(tmp_path):
    """A job-id collision must not cancel an insert into an unrelated table.

    Keyed on job_id alone this silently no-ops: 'orders' is stamped, so the transfer
    into 'customers' skips its insert and reports the old count — customers ends up
    empty and the run reports success.
    """
    url = _url(tmp_path)
    _make_table(url, "orders")
    _make_table(url, "customers")

    a = _sink(url, "job-dup", "orders")
    _stage(a, url, [(1, "a"), (2, "b")])
    a.finalize(_wr(2))
    assert _count(url, "orders") == 2

    # Same id, DIFFERENT target: a real insert, not a retry of the one above.
    b = _sink(url, "job-dup", "customers")
    _stage(b, url, [(9, "z")])
    out = b.finalize(_wr(1)).to_pydict()

    assert _count(url, "customers") == 1, "collision silently swallowed the insert"
    assert out["rows_inserted"] == [1]


def test_retry_dedup_still_holds_per_table(tmp_path):
    """Scoping the key must not weaken the retry guarantee it exists for."""
    url = _url(tmp_path)
    _make_table(url, "orders")

    first = _sink(url, "job-r", "orders")
    _stage(first, url, [(1, "a"), (2, "b")])
    first.finalize(_wr(2))

    retry = _sink(url, "job-r", "orders")
    _stage(retry, url, [(1, "a"), (2, "b")])
    out = retry.finalize(_wr(2)).to_pydict()

    assert _count(url, "orders") == 2        # still no duplicate
    assert out["rows_inserted"] == [2]


# ── unsigned downcast ─────────────────────────────────────────────────────────

def test_counts_and_ranks_are_downcast(tmp_path):
    """The AGG-4b backstop: ordinary UInt64 aggregation output casts to int64."""
    df = pd.DataFrame({"member_count": pd.Series([1, 2, 3], dtype="uint64")})
    out = _downcast_unsigned_ints(df)
    assert out["member_count"].dtype == np.dtype("int64")
    assert out["member_count"].tolist() == [1, 2, 3]


def test_uint64_above_int64_max_is_not_silently_wrapped():
    """A real uint64 id >= 2**63 must NOT be cast — int64 would wrap negative.

    Leaving it unsigned makes pandas to_sql raise its original ValueError, which is
    the correct outcome: a loud failure beats a negative id written as success.
    """
    big = 2**63 + 5
    df = pd.DataFrame({"snowflake_id": pd.Series([1, big], dtype="uint64")})

    out = _downcast_unsigned_ints(df)

    assert out["snowflake_id"].dtype == np.dtype("uint64"), "cast anyway → wraps negative"
    assert out["snowflake_id"].tolist() == [1, big]      # value preserved exactly
    assert (out["snowflake_id"] >= 0).all()


def test_empty_unsigned_column_is_handled(tmp_path):
    """max() on an empty column must not throw the guard off (no rows → nothing to wrap)."""
    df = pd.DataFrame({"c": pd.Series([], dtype="uint64")})
    out = _downcast_unsigned_ints(df)
    assert out["c"].dtype == np.dtype("int64")
