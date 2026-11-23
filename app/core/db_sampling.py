"""Random samples of a database table, and the words that disclose them.

The API analyses a database table through a bounded sample, fetched over the
MCP ``query`` tool. It used to be ``SELECT * FROM t LIMIT 1000``: the first
thousand rows in whatever order the table is stored, presented as the table.
On a table stored with every positive first, that reported a stroke rate of
0.50 where the whole table's is 0.13.

So there is no head sample here, ever:

* the row total comes from ``COUNT(*)``, or, if that cannot run, from the
  catalog's own estimate, labelled as one;
* a table no bigger than the cap is read whole, which is exact;
* a bigger one is sampled at random with each dialect's own primitive, with a
  cheap random pre-filter in front of the sort on large tables;
* a dialect with no known random primitive falls back to a key-spaced sample
  (``MOD(key, stride) = offset``, random offset) on an integer key column;
* when none of that works, the caller gets :class:`DbSampleError` rather than
  the first rows.

The SQL is built for the MCP server, which wraps every query as
``SELECT * FROM (<sql>) AS _mcp_sub LIMIT <max_rows>`` (``SELECT TOP`` on
MSSQL, ``ROWNUM`` on Oracle), so each statement here must stay valid inside
that wrapper.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# Above this many rows the random sort is preceded by a random pre-filter that
# keeps roughly PREFILTER_OVERSAMPLE x the sample, so the database sorts
# thousands of rows instead of the whole table.
PREFILTER_MIN_ROWS = int(os.getenv("AVALOKA_DB_SAMPLE_PREFILTER_ROWS", "1000000"))
PREFILTER_OVERSAMPLE = 3.0

# How many rows of a table the API reads for an analysis. A table this size or
# smaller is read whole, which is exact; a bigger one gets a random sample of
# this many rows, disclosed as "N of M". It was 1,000, so the quickstart's
# 2,000-row ``orders`` table was analysed as a sample. The rows go to a CSV on
# disk (the session keeps a capped preview), so the bound is the API's memory
# while one read is in flight; 50,000 matches the file sampler's default
# (DEFAULT_SAMPLE_MAX_ROWS). It is also the per-query row cap a new saved
# connection registers with, since a lower cap would cut the read short.
ANALYSIS_MAX_ROWS = int(os.getenv("AVALOKA_DB_ANALYSIS_MAX_ROWS", "50000"))

RANDOM_SAMPLE_DIALECTS = ("postgresql", "mysql", "sqlite", "mssql", "oracle")

_KEY_NAME_RE = re.compile(r"^(id|.*_id|.*id)$", re.IGNORECASE)

RunQuery = Callable[[str], Awaitable[Dict[str, Any]]]


class DbSampleError(RuntimeError):
    """No honest sample could be drawn; the message is safe to show the user."""


@dataclass
class DbSample:
    """What the analysis rows are, relative to the whole table."""

    method: str                    # "full" | "random" | "keyed"
    n: int                         # rows the analysis holds
    total: Optional[int]           # rows in the table, or None when unknown
    total_is_estimate: bool = False
    dialect: str = ""

    @property
    def is_partial(self) -> bool:
        if self.method == "full":
            return False
        return self.total is None or self.n < self.total

    def as_session(self) -> Dict[str, Any]:
        return {
            "method": self.method, "n": self.n, "total": self.total,
            "total_is_estimate": self.total_is_estimate, "dialect": self.dialect,
        }

    @classmethod
    def from_session(cls, raw: Any) -> Optional["DbSample"]:
        if not isinstance(raw, dict) or raw.get("method") not in ("full", "random", "keyed"):
            return None
        try:
            n = int(raw.get("n") or 0)
            total = raw.get("total")
            total = int(total) if total is not None else None
        except (TypeError, ValueError):
            return None
        return cls(method=raw["method"], n=n, total=total,
                   total_is_estimate=bool(raw.get("total_is_estimate")),
                   dialect=str(raw.get("dialect") or ""))


def normalize_dialect(database_type: Optional[str]) -> str:
    d = (database_type or "").strip().lower()
    if d in ("postgres", "postgresql", "pg"):
        return "postgresql"
    if d in ("mysql", "mariadb"):
        return "mysql"
    if d in ("sqlite", "sqlite3"):
        return "sqlite"
    if d in ("mssql", "sqlserver", "sql_server"):
        return "mssql"
    return d


def count_sql(source: str) -> str:
    return f"SELECT COUNT(*) AS row_count FROM {source}"


def probe_sql(source: str) -> str:
    """Columns and dialect, no rows."""
    return f"SELECT * FROM {source} WHERE 1 = 0"


def full_read_sql(source: str) -> str:
    return f"SELECT * FROM {source}"


def catalog_estimate_sql(table: str, dialect: str) -> Optional[str]:
    """The catalog's row estimate for a plain table name, or None if there is none."""
    if not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)?", table or ""):
        return None
    d = normalize_dialect(dialect)
    bare = table.split(".")[-1]
    if d == "postgresql":
        return f"SELECT reltuples::bigint AS row_count FROM pg_class WHERE oid = to_regclass('{table}')"
    if d == "mysql":
        return ("SELECT TABLE_ROWS AS row_count FROM information_schema.tables "
                f"WHERE table_schema = DATABASE() AND table_name = '{bare}'")
    if d == "mssql":
        return ("SELECT SUM(row_count) AS row_count FROM sys.dm_db_partition_stats "
                f"WHERE object_id = OBJECT_ID('{table}') AND index_id IN (0, 1)")
    if d == "oracle":
        return f"SELECT num_rows AS row_count FROM user_tables WHERE table_name = UPPER('{bare}')"
    return None


def random_sample_sql(source: str, n: int, dialect: str, *,
                      total_rows: Optional[int] = None, source_is_table: bool = True) -> Optional[str]:
    """A uniform random sample of ``n`` rows of ``source``, or None for an unknown dialect.

    ``source`` is a table name or a parenthesised, aliased subquery. Every
    statement orders by the dialect's random function before its bound, so
    which rows come back never depends on storage order.
    """
    d = normalize_dialect(dialect)
    n = max(1, int(n))
    big = bool(total_rows and total_rows > PREFILTER_MIN_ROWS)
    frac = min(1.0, PREFILTER_OVERSAMPLE * n / total_rows) if big else 1.0
    if d == "postgresql":
        if big and source_is_table:
            # Row-level Bernoulli, not SYSTEM: SYSTEM keeps whole pages, so a
            # sorted table would hand back runs of neighbouring rows.
            return (f"SELECT * FROM (SELECT * FROM {source} TABLESAMPLE BERNOULLI ({frac * 100:.6f})) "
                    f"AS _avk_s ORDER BY random() LIMIT {n}")
        where = f" WHERE random() < {frac:.9f}" if big else ""
        return f"SELECT * FROM {source}{where} ORDER BY random() LIMIT {n}"
    if d == "mysql":
        where = f" WHERE RAND() < {frac:.9f}" if big else ""
        return f"SELECT * FROM {source}{where} ORDER BY RAND() LIMIT {n}"
    if d == "sqlite":
        # random() is a signed 64-bit integer; map it onto [0, 1).
        where = f" WHERE (random() / 18446744073709551616.0 + 0.5) < {frac:.9f}" if big else ""
        return f"SELECT * FROM {source}{where} ORDER BY random() LIMIT {n}"
    if d == "mssql":
        # Nested so the MCP wrapper's own "SELECT TOP" lands on the outer SELECT.
        where = f" WHERE ABS(CHECKSUM(NEWID())) % 1000000 < {int(frac * 1_000_000)}" if big else ""
        return f"SELECT * FROM (SELECT TOP {n} * FROM {source}{where} ORDER BY NEWID()) AS _avk_s"
    if d == "oracle":
        sample = f" SAMPLE ({max(frac * 100, 0.000001):.6f})" if big and source_is_table else ""
        return f"SELECT * FROM (SELECT * FROM {source}{sample} ORDER BY DBMS_RANDOM.VALUE) WHERE ROWNUM <= {n}"
    return None


def keyed_sample_sql(source: str, key: str, n: int, total_rows: int,
                     rng: Optional[random.Random] = None) -> str:
    """Every ``stride``-th key value from a random offset: spread over the key range, not the head."""
    stride = max(1, -(-int(total_rows) // max(1, int(n))))
    offset = (rng or random).randrange(stride)
    return f"SELECT * FROM {source} WHERE MOD({key}, {stride}) = {offset}"


def _first_int(result: Dict[str, Any]) -> Optional[int]:
    rows = (result or {}).get("rows") or []
    if not rows or not rows[0]:
        return None
    try:
        value = rows[0][0]
        if value is None or isinstance(value, bool):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _ok(result: Any) -> bool:
    return isinstance(result, dict) and not result.get("error") and "rows" in result


def _key_column(columns: List[str]) -> Optional[str]:
    for col in columns:
        if re.fullmatch(r"[A-Za-z_]\w*", str(col)) and _KEY_NAME_RE.match(str(col)):
            return str(col)
    return None


async def table_row_total(run_query: RunQuery, source: str, dialect: str,
                          table: Optional[str] = None) -> Tuple[Optional[int], bool]:
    """``(rows, is_estimate)``: COUNT(*) first, then the catalog estimate, else ``(None, False)``."""
    try:
        result = await run_query(count_sql(source))
    except Exception:
        result = None
    if _ok(result):
        total = _first_int(result)
        if total is not None:
            return total, False
    est_sql = catalog_estimate_sql(table, dialect) if table else None
    if est_sql:
        try:
            result = await run_query(est_sql)
        except Exception:
            result = None
        if _ok(result):
            total = _first_int(result)
            if total is not None and total >= 0:
                return total, True
    return None, False


async def draw_table_sample(run_query: RunQuery, source: str, n: int, *,
                            table: Optional[str] = None, source_is_table: bool = True,
                            ) -> Tuple[List[str], List[List[Any]], DbSample]:
    """Columns, rows and a :class:`DbSample` for at most ``n`` rows of ``source``.

    Raises :class:`DbSampleError` when no honest sample is possible; never
    falls back to the first rows.
    """
    probe = await run_query(probe_sql(source))
    if not isinstance(probe, dict) or probe.get("error"):
        raise DbSampleError(f"I couldn't read it: {(probe or {}).get('error', 'no response')}")
    dialect = normalize_dialect(((probe.get("meta") or {}).get("database_type")))
    columns = list(probe.get("columns") or [])

    total, is_estimate = await table_row_total(run_query, source, dialect, table if source_is_table else None)

    if total is not None and not is_estimate and total <= n:
        result = await run_query(full_read_sql(source))
        if not _ok(result):
            raise DbSampleError(f"I couldn't read it: {(result or {}).get('error', 'no rows')}")
        rows = list(result.get("rows") or [])
        cols = list(result.get("columns") or columns)
        if len(rows) == total:
            return cols, rows, DbSample("full", len(rows), total, False, dialect)
        # The server's own row cap cut the read short: that is a head, so
        # take a random sample of the same size instead.
        n = max(1, len(rows))

    sql = random_sample_sql(source, n, dialect, total_rows=total, source_is_table=source_is_table)
    if sql:
        result = await run_query(sql)
        if _ok(result) and (result.get("rows") or total == 0):
            rows = list(result.get("rows") or [])[:n]
            return list(result.get("columns") or columns), rows, DbSample(
                "random", len(rows), total, is_estimate, dialect)

    key = _key_column(columns)
    if key and total:
        result = await run_query(keyed_sample_sql(source, key, n, total))
        if _ok(result) and result.get("rows"):
            rows = list(result.get("rows") or [])
            if len(rows) > n:
                rows = random.sample(rows, n)
            return list(result.get("columns") or columns), rows, DbSample(
                "keyed", len(rows), total, is_estimate, dialect)

    raise DbSampleError(
        "I couldn't draw a random sample from it on this database, and I won't "
        "analyse its first rows as if they were the whole table."
    )


def sample_scope(sample: DbSample) -> str:
    """'a random sample of 1,000 of 5,110 rows' and its estimate/unknown variants."""
    kind = "key-spaced" if sample.method == "keyed" else "random"
    if sample.total is None:
        return f"a {kind} sample of {sample.n:,} rows (the table's row count could not be read)"
    if sample.total_is_estimate:
        return (f"a {kind} sample of {sample.n:,} of about {sample.total:,} rows "
                "(the database's own estimate of the table size)")
    return f"a {kind} sample of {sample.n:,} of {sample.total:,} rows"


def sample_context_line(sample: DbSample) -> str:
    """One line for the planner's [Analysis context]: what the rows in hand are."""
    if not sample.is_partial:
        return f"Read the whole database table ({sample.n:,} rows); figures are exact."
    return (f"Analysis rows are {sample_scope(sample)} of the database table; every "
            "count, rate or mean computed on them is a sample estimate, not the table's figure.")
