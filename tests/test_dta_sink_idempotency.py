"""Retry-idempotency tests for the SQL sink (append-duplicate-on-retry bug).

``container_file.py`` retries ``run_pipeline()`` up to 3× on ANY exception. Each
retry re-reads the source and re-stages every row. Before the fix, an append whose
``finalize()`` had already COMMITTED but then failed downstream would insert the
rows a SECOND time on retry — duplicates in the destination, still reported as
success. The sink now records each ``JOB_ID`` in a ledger committed in the SAME
transaction as the insert, so a retry with the same job id is a no-op.

Backed by a file-based SQLite db: the sink opens a fresh engine per operation, so
an in-memory ``:memory:`` url would be a different database on each connection.
``PostgresDataSink`` only fixes the identifier-quote char to ``"`` (SQLite-compatible);
the SQLAlchemy dialect is chosen by the connection url, so it drives SQLite fine.
"""
import pandas as pd
import pytest
import sqlalchemy as sa
from daft.io.sink import WriteResult

from app.agents.data_transfer_agent.datasink.sql_sink import (
    PostgresDataSink,
    _downcast_unsigned_ints,
)


def _url(tmp_path):
    return f"sqlite:///{tmp_path / 'dest.db'}"


def _make_target(url):
    eng = sa.create_engine(url)
    with eng.begin() as c:
        c.exec_driver_sql("CREATE TABLE orders (id INTEGER, name TEXT)")
    eng.dispose()


def _stage(sink, url, rows):
    """Populate a freshly-constructed sink's staging table (what write() does)."""
    eng = sa.create_engine(url)
    stg = sink._qi(sink._staging_table)
    with eng.begin() as c:
        for rid, name in rows:
            c.exec_driver_sql(f"INSERT INTO {stg} (id, name) VALUES ({rid}, '{name}')")
    eng.dispose()


def _count(url, table="orders"):
    eng = sa.create_engine(url)
    try:
        with eng.begin() as c:
            return c.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        eng.dispose()


def _wr(n):
    return [WriteResult(result={"status": "success"}, rows_written=n, bytes_written=0)]


def _sink(url, job_id):
    return PostgresDataSink(url, "orders", write_mode="append",
                            columns=["id", "name"], job_id=job_id)


def test_append_retry_with_same_job_id_does_not_duplicate(tmp_path):
    url = _url(tmp_path)
    _make_target(url)

    # First attempt: stage 2 rows and finalize → 2 rows committed, ledger stamped.
    first = _sink(url, "job-1")
    _stage(first, url, [(1, "a"), (2, "b")])
    out1 = first.finalize(_wr(2)).to_pydict()
    assert _count(url) == 2
    assert out1["rows_inserted"] == [2]

    # Retry: a FRESH sink (new staging table) with the SAME job id re-stages the
    # same rows. finalize must be a no-op — the target stays at 2, not 4.
    retry = _sink(url, "job-1")
    _stage(retry, url, [(1, "a"), (2, "b")])
    out2 = retry.finalize(_wr(2)).to_pydict()
    assert _count(url) == 2                 # no duplicate insert
    assert out2["rows_inserted"] == [2]     # original committed count reported


def test_distinct_job_ids_each_append(tmp_path):
    """Two genuinely different transfers (different job ids) both append."""
    url = _url(tmp_path)
    _make_target(url)

    a = _sink(url, "job-A")
    _stage(a, url, [(1, "a")])
    a.finalize(_wr(1))

    b = _sink(url, "job-B")
    _stage(b, url, [(2, "b")])
    b.finalize(_wr(1))

    assert _count(url) == 2


def test_no_job_id_falls_back_to_plain_append(tmp_path):
    """Empty job id disables the ledger → previous (non-deduped) behavior. Documents
    that the dedup guarantee is scoped to a stable job id from the runner."""
    url = _url(tmp_path)
    _make_target(url)

    a = _sink(url, "")
    assert a.job_id == ""       # ledger disabled
    _stage(a, url, [(1, "a")])
    a.finalize(_wr(1))

    b = _sink(url, "")
    _stage(b, url, [(1, "a")])
    b.finalize(_wr(1))

    assert _count(url) == 2      # appended twice — no ledger to dedup


def test_overwrite_retry_is_idempotent(tmp_path):
    url = _url(tmp_path)
    _make_target(url)

    first = PostgresDataSink(url, "orders", write_mode="overwrite",
                             columns=["id", "name"], job_id="job-ow")
    _stage(first, url, [(1, "a"), (2, "b")])
    first.finalize(_wr(2))
    assert _count(url) == 2

    retry = PostgresDataSink(url, "orders", write_mode="overwrite",
                             columns=["id", "name"], job_id="job-ow")
    _stage(retry, url, [(1, "a"), (2, "b")])
    retry.finalize(_wr(2))
    assert _count(url) == 2      # still 2 (overwrite dedups anyway; ledger no-ops)


# ---------------------------------------------------------------------------
# Unsigned-integer downcast guard (Daft count()/rank() → UInt64 → sink crash)
# ---------------------------------------------------------------------------

def test_raw_uint64_to_sql_reproduces_the_error(tmp_path):
    """Baseline: pandas to_sql on a UInt64 column raises the exact sink error."""
    url = f"sqlite:///{tmp_path / 'u.db'}"
    eng = sa.create_engine(url)
    df = pd.DataFrame({"status": ["a", "b"], "member_count": pd.array([1, 2], dtype="uint64")})
    with pytest.raises(ValueError, match="Unsigned 64 bit integer datatype is not supported"):
        df.to_sql("agg", eng, if_exists="fail", index=False)
    eng.dispose()


def test_downcast_makes_uint64_writable(tmp_path):
    """The guard downcasts UInt64 → int64 so the same write now succeeds losslessly."""
    url = f"sqlite:///{tmp_path / 'u2.db'}"
    eng = sa.create_engine(url)
    df = pd.DataFrame({"status": ["a", "b"], "member_count": pd.array([1, 2], dtype="uint64")})

    fixed = _downcast_unsigned_ints(df)
    assert str(fixed["member_count"].dtype) == "int64"
    fixed.to_sql("agg", eng, if_exists="fail", index=False)   # no longer raises

    back = pd.read_sql("SELECT member_count FROM agg ORDER BY member_count", eng)
    assert list(back["member_count"]) == [1, 2]               # values preserved
    eng.dispose()


def test_downcast_leaves_signed_and_non_int_columns_untouched():
    df = pd.DataFrame({
        "signed": pd.array([1, 2], dtype="int64"),
        "text": ["x", "y"],
        "flt": [1.5, 2.5],
        "u32": pd.array([3, 4], dtype="uint32"),
    })
    out = _downcast_unsigned_ints(df)
    assert str(out["signed"].dtype) == "int64"
    assert str(out["text"].dtype) == "object"
    assert str(out["flt"].dtype) == "float64"
    assert str(out["u32"].dtype) == "int64"   # any unsigned width → int64
