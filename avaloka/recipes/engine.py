"""The DuckDB side of a recipe run: one connection, one relation named ``t``.

DuckDB is part of the ``avaloka[recipes]`` extra, not a base dependency
(S1.09), so it is imported inside :func:`connect` and its absence is reported
in the same plain words the ``[formats]`` loader uses, never as a traceback
from an import at module load.

The budget (:mod:`avaloka.recipes.budget`) is enforced here: :func:`connect`
sets the engine's memory and thread caps, :func:`register` counts the source
and replaces ``t`` with a deterministic sample when the source is over the
row or byte limit, and :func:`timed` is the wall clock that interrupts the
connection when the run's time is spent.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from avaloka.workload import SAMPLED_MAX_BYTES, SAMPLED_MAX_ROWS

from .budget import DEFAULT, Budget

INSTALL_HINT = "DuckDB is not installed. Install the recipe extra: pip install 'avaloka[recipes]'"

#: How ``t`` becomes its sample: :func:`register` runs these, and a sampled
#: run's notebook runs the same statements on ``SAMPLE_ROWS`` / ``SAMPLE_SEED``
#: (``library._shared.load_cell``), so the notebook reads the rows the recipe read.
SAMPLE_SQL = "CREATE TEMP TABLE t_sample AS SELECT * FROM t USING SAMPLE reservoir({rows} ROWS) REPEATABLE ({seed})"
RENAME_SQL = "ALTER TABLE t_sample RENAME TO t"


class RecipesDependencyError(ImportError):
    """DuckDB is absent; ``str(error)`` is the install message in plain words."""


class SourceTooLarge(ValueError):
    """The source is over the sampled lane's ceiling (``avaloka.workload``);
    the run refuses ``too_large``. The message is for the log, not the user."""


@dataclass(frozen=True)
class SourceInfo:
    """What :func:`register` found: the source's full size, and whether ``t``
    is a sample of it."""
    total_rows: int
    total_bytes: int
    sampled: bool
    rows_analyzed: int


def missing_dependency() -> str | None:
    """The install message when DuckDB is absent, else None."""
    try:
        import duckdb  # noqa: F401
    except ImportError:
        return INSTALL_HINT
    return None


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise RecipesDependencyError(INSTALL_HINT) from exc
    return duckdb


def is_parquet(source: str | Path) -> bool:
    return str(source).lower().endswith(".parquet")


def connect(source: str | Path | pd.DataFrame, *, budget: Budget = DEFAULT,
            memory_limit: str | None = None, threads: int | None = None) -> Any:
    """A DuckDB connection with the dataset registered as the relation ``t``.

    ``source`` is a CSV or Parquet path (chosen by extension) or a pandas
    DataFrame. The budget's memory and thread caps are set on the connection
    so a recipe cannot take the host down on a large file; ``memory_limit``
    and ``threads`` given by name override the budget's (the pre-budget
    signature). ``t`` is the whole source until :func:`register` decides
    whether the run samples.
    """
    duckdb = _duckdb()
    if memory_limit is not None or threads is not None:
        budget = replace(budget, **{k: v for k, v in (("memory_limit", memory_limit), ("threads", threads))
                                    if v is not None})
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{budget.memory_limit}'")
    con.execute(f"SET threads={int(budget.threads)}")
    _register_full(con, source)
    return con


def _register_full(con: Any, source: str | Path | pd.DataFrame) -> None:
    if isinstance(source, pd.DataFrame):
        con.register("t", source)
    elif is_parquet(source):
        con.read_parquet(str(source)).create_view("t")
    else:
        con.read_csv(str(source)).create_view("t")


def _drop_full(con: Any, source: str | Path | pd.DataFrame) -> None:
    if isinstance(source, pd.DataFrame):
        con.unregister("t")
    else:
        con.execute("DROP VIEW t")


def source_size(con: Any, source: str | Path | pd.DataFrame) -> tuple[int, int]:
    """``(rows, bytes)`` of the source: a frame's length and deep memory; a
    file's DuckDB row count (Parquet answers from its metadata, CSV by a
    scan) and its size on disk."""
    if isinstance(source, pd.DataFrame):
        return len(source), int(source.memory_usage(deep=True).sum())
    rows = int(con.execute("SELECT count(*) FROM t").fetchone()[0])
    return rows, int(os.stat(source).st_size)


def register(con: Any, source: str | Path | pd.DataFrame, budget: Budget = DEFAULT) -> SourceInfo:
    """Size the source against the budget and leave ``t`` as what the run reads.

    Over ``budget.max_rows`` or ``budget.max_bytes``, ``t`` becomes a
    materialised temporary table holding a reservoir sample of
    ``budget.sample_rows`` rows, ``REPEATABLE`` on ``budget.seed``, so the
    same source samples the same rows on every connection and every recipe
    query reads the sample. Over the sampled lane's ceiling the source is
    refused (:class:`SourceTooLarge`). Call once, on a connection from
    :func:`connect`.
    """
    total_rows, total_bytes = source_size(con, source)
    if total_rows > SAMPLED_MAX_ROWS or total_bytes > SAMPLED_MAX_BYTES:
        raise SourceTooLarge(f"{total_rows} rows, {total_bytes} bytes; the sampled lane stops at "
                             f"{SAMPLED_MAX_ROWS} rows / {SAMPLED_MAX_BYTES} bytes")
    if total_rows <= budget.max_rows and total_bytes <= budget.max_bytes:
        return SourceInfo(total_rows, total_bytes, False, total_rows)
    n = min(int(budget.sample_rows), total_rows)
    con.execute(SAMPLE_SQL.format(rows=n, seed=int(budget.seed)))
    _drop_full(con, source)
    con.execute(RENAME_SQL)
    rows_analyzed = int(con.execute("SELECT count(*) FROM t").fetchone()[0])
    return SourceInfo(total_rows, total_bytes, True, rows_analyzed)


class Clock:
    """The wall clock :func:`timed` hands out: ``elapsed`` seconds so far and
    whether the limit ``expired`` (the connection was interrupted)."""

    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self.started = time.monotonic()
        self.expired = False

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


@contextmanager
def timed(con: Any, seconds: float) -> Iterator[Clock]:
    """Interrupt ``con`` when ``seconds`` have passed; cancel the timer on exit.

    A DuckDB statement running at expiry raises ``InterruptException``
    (:func:`interrupted`); the clock's ``expired`` flag records that the
    limit passed even if the work was in Python at the time, so the caller
    can still refuse a run that overran. An interrupt on an idle connection
    is a no-op, so the next statement is not affected.
    """
    clock = Clock(seconds)

    def expire() -> None:
        clock.expired = True
        con.interrupt()

    timer = threading.Timer(clock.seconds, expire)
    timer.daemon = True
    timer.start()
    try:
        yield clock
    finally:
        timer.cancel()


def interrupted(exc: BaseException) -> bool:
    """True when ``exc`` is DuckDB's interrupt (the wall clock fired)."""
    try:
        import duckdb
    except ImportError:
        return False
    return isinstance(exc, duckdb.InterruptException)


def out_of_memory(exc: BaseException) -> bool:
    """True when ``exc`` is DuckDB's out-of-memory error (the memory cap bit)."""
    try:
        import duckdb
    except ImportError:
        return False
    return isinstance(exc, duckdb.OutOfMemoryException)


def query(con: Any, sql: str) -> pd.DataFrame:
    """Run one statement and return its result as a DataFrame."""
    return con.execute(sql).df()


def ident(name: str) -> str:
    """A column name quoted as a DuckDB identifier."""
    return '"' + str(name).replace('"', '""') + '"'
