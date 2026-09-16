"""
loader/sql.py — SQL Dataset Loader
====================================
Loads tabular data from any SQLAlchemy-compatible database into either a
Ray Dataset (when Ray is available) or a pandas DataFrame (local fallback).

Supported URI schemes
---------------------
  postgresql://user:pass@host:5432/db
  mysql+pymysql://user:pass@host:3306/db
  sqlite:///path/to/file.db
  mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17+for+SQL+Server

URI format expected by ray_job.py / local_trainer.py
-----------------------------------------------------
The DATA_SOURCE_URI env var must include the table name appended after a
pipe character so the loader knows which table to query:

    postgresql://user:pass@host/db|my_table
    sqlite:///data.db|my_table

The pipe separator is used because SQLAlchemy connection strings may already
contain query-string parameters (e.g. ?sslmode=require) that make appending
with a query param ambiguous.

Alternatively, the table name can be supplied via the TABLE_NAME env var and
the URI passed without a pipe — useful when the URI comes from a secret store.

Streaming / memory notes (Ray path)
-------------------------------------
  - Uses SQLAlchemy server-side cursors (yield_per=1000) so the full table is
    never loaded into driver memory.
  - Rows are yielded in chunks and handed to ray.data.from_pandas_refs() to
    distribute blocks across the cluster.
  - Chunk size is controlled by SQL_CHUNK_SIZE env var (default: 10_000).

Local path
----------
  - Uses pandas.read_sql with the same chunked iteration and concatenates into
    a single DataFrame. Suitable for datasets that fit in local RAM.
  - For very large tables set SQL_CHUNK_SIZE to a smaller value to control
    peak memory usage during loading.

Environment variables
---------------------
  SQL_CHUNK_SIZE   — rows per chunk / Ray block (default: 10_000)
  TABLE_NAME       — fallback table name when not encoded in the URI with |
"""
from __future__ import annotations

import os
from typing import Iterator, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Ray is optional — the loader works without it for local_trainer.py
try:
    import ray
    import ray.data as rd
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False


# =========================================================
# Helpers
# =========================================================

_DEFAULT_CHUNK_SIZE = int(os.getenv("SQL_CHUNK_SIZE", 10_000))


def _parse_uri_and_table(uri: str) -> tuple[str, str]:
    """
    Split a URI of the form  <connection_string>|<table_name>  into its parts.

    Falls back to the TABLE_NAME env var when no pipe is present.

    Raises
    ------
    ValueError if neither the pipe separator nor TABLE_NAME is provided.
    """
    if "|" in uri:
        conn_str, table_name = uri.rsplit("|", 1)
        return conn_str.strip(), table_name.strip()

    table_name = os.getenv("TABLE_NAME", "").strip()
    if not table_name:
        raise ValueError(
            "SQL loader requires a table name. "
            "Either append it to the URI with a pipe  (e.g. postgresql://…/db|my_table) "
            "or set the TABLE_NAME environment variable."
        )
    return uri.strip(), table_name


def _make_engine(conn_str: str) -> Engine:
    """
    Create a SQLAlchemy engine with sane defaults.

    pool_pre_ping  — validates connections before use (avoids stale-connection errors).
    execution_options(stream_results=True)  — enables server-side cursor where
      the dialect supports it (PostgreSQL, MySQL). SQLite ignores this gracefully.
    """
    engine = create_engine(
        conn_str,
        pool_pre_ping=True,
        execution_options={"stream_results": True},
    )
    return engine


def _iter_chunks(
    engine: Engine,
    table_name: str,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    """
    Yield successive DataFrame chunks from the table using server-side cursors.

    yield_per(chunk_size) on the SQLAlchemy result set keeps only chunk_size rows
    in memory at a time, regardless of the total table size.
    """
    query = f"SELECT * FROM {table_name}"  # noqa: S608 — table name is caller-validated

    with engine.connect() as conn:
        result = conn.execution_options(yield_per=chunk_size).execute(text(query))
        keys   = list(result.keys())

        chunk: List[dict] = []
        for row in result:
            chunk.append(dict(zip(keys, row)))
            if len(chunk) >= chunk_size:
                yield pd.DataFrame(chunk)
                chunk = []
        if chunk:
            yield pd.DataFrame(chunk)


# =========================================================
# Public API
# =========================================================


def load_dataset_from_sql(
    uri: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    *,
    use_ray: Optional[bool] = None,
):
    """
    Load a SQL table into a Ray Dataset (if Ray is available) or a pandas
    DataFrame (local fallback).

    Parameters
    ----------
    uri        : Connection URI with optional |table_name suffix, or a plain
                 SQLAlchemy URI when TABLE_NAME env var is set.
    chunk_size : Rows per chunk / Ray block. Controls peak memory per worker.

    Returns
    -------
    ray.data.Dataset  — when Ray is available and initialised.
    pd.DataFrame      — otherwise, or when ``use_ray=False``.

    ``use_ray`` lets local training select pandas explicitly instead of
    inheriting unrelated process-global Ray state.
    """
    conn_str, table_name = _parse_uri_and_table(uri)
    print(f"[SQL Loader] Connecting to {_redact_uri(conn_str)}  table={table_name}  chunk_size={chunk_size}")

    engine = _make_engine(conn_str)

    # ---- Ray path ----
    ray_requested = ray.is_initialized() if use_ray is None and _RAY_AVAILABLE else bool(use_ray)
    if ray_requested:
        if not _RAY_AVAILABLE or not ray.is_initialized():
            raise RuntimeError("Ray loading was requested, but Ray is not initialized.")
        return _load_ray(engine, table_name, chunk_size)

    # ---- Local / pandas path ----
    return _load_pandas(engine, table_name, chunk_size)


# =========================================================
# Ray path
# =========================================================


def _load_ray(engine: Engine, table_name: str, chunk_size: int) -> "rd.Dataset":
    """
    Stream the table in chunks and assemble a Ray Dataset from pandas refs.

    Each chunk becomes one Ray Dataset block — Ray distributes them across
    the cluster automatically. The driver only holds one chunk at a time.
    """
    import ray.data as rd  # local import keeps the module importable without Ray

    refs = []
    total_rows = 0

    for chunk_df in _iter_chunks(engine, table_name, chunk_size):
        refs.append(ray.put(chunk_df))
        total_rows += len(chunk_df)
        print(f"[SQL Loader] Loaded {total_rows} rows ...", end="\r")

    print(f"\n[SQL Loader] Total rows loaded: {total_rows}  blocks={len(refs)}")

    if not refs:
        raise ValueError(f"[SQL Loader] Table '{table_name}' returned no rows.")

    return rd.from_pandas_refs(refs)


# =========================================================
# Local / pandas path
# =========================================================


def _load_pandas(engine: Engine, table_name: str, chunk_size: int) -> pd.DataFrame:
    """
    Concatenate all chunks into a single DataFrame for local training.
    """
    chunks: List[pd.DataFrame] = []
    total_rows = 0

    for chunk_df in _iter_chunks(engine, table_name, chunk_size):
        chunks.append(chunk_df)
        total_rows += len(chunk_df)
        print(f"[SQL Loader] Loaded {total_rows} rows ...", end="\r")

    print(f"\n[SQL Loader] Total rows loaded: {total_rows}")

    if not chunks:
        raise ValueError(f"[SQL Loader] Table '{table_name}' returned no rows.")

    return pd.concat(chunks, ignore_index=True)


# =========================================================
# Utility
# =========================================================


def _redact_uri(uri: str) -> str:
    """
    Redact password from a connection URI for safe logging.
    e.g. postgresql://user:secret@host/db  →  postgresql://user:***@host/db
    """
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(uri)
        if parsed.password:
            netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return uri
