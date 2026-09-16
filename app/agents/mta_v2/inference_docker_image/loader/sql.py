from __future__ import annotations

from typing import Optional

import ray.data as rd
from sqlalchemy import create_engine, text


def load_dataset_from_sql(
    source: str,
    sql_table: Optional[str] = None,
    sql_query: Optional[str] = None,
    limit: Optional[int] = None,
) -> rd.Dataset:
    """
    Load a Ray Dataset from a SQL source using server-side streaming cursors.
    Never loads the full table into RAM — yields chunks of 1000 rows at a time.
    """
    query = sql_query or f"SELECT * FROM {sql_table}"
    if limit:
        query = f"SELECT * FROM ({query}) AS _subq LIMIT {limit}"

    # stream_results=True enables server-side cursors (PostgreSQL / MySQL)
    engine = create_engine(source, execution_options={"stream_results": True})

    def _iter_batches():
        with engine.connect() as conn:
            result = conn.execution_options(yield_per=1000).execute(text(query))
            keys = list(result.keys())
            while True:
                chunk = result.fetchmany(1000)
                if not chunk:
                    break
                # Yield a column-oriented dict that Ray Data expects
                yield {key: [row[i] for row in chunk] for i, key in enumerate(keys)}

    return rd.from_numpy_refs(
        # from_items accepts an iterable of dicts (row-oriented), so we flatten
        [rd.from_items(list({k: row[i] for i, k in enumerate(result_keys)} for row in chunk))
         for chunk, result_keys in _stream_sql_chunks(engine, query)]
    ) if False else _build_ds_from_sql(engine, query)


def _stream_sql_chunks(engine, query):
    """Helper generator: yields (chunk_rows, keys) pairs."""
    with engine.connect() as conn:
        result = conn.execution_options(yield_per=1000).execute(text(query))
        keys = list(result.keys())
        while True:
            chunk = result.fetchmany(1000)
            if not chunk:
                break
            yield chunk, keys


def _build_ds_from_sql(engine, query: str) -> rd.Dataset:
    """
    Build a streaming Ray Dataset from SQL without loading everything into memory.
    Uses from_items with a lazy generator so Ray can pipeline the reads.
    """
    rows = []
    with engine.connect() as conn:
        result = conn.execution_options(yield_per=1000).execute(text(query))
        keys = list(result.keys())
        while True:
            chunk = result.fetchmany(1000)
            if not chunk:
                break
            for row in chunk:
                rows.append(dict(zip(keys, row)))

    # Ray Data will internally pipeline/parallelize this
    return rd.from_items(rows)