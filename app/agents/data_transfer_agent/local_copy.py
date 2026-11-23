"""Database-to-database table copy without Ray: bounded, chunked, all-or-nothing.

A transfer is normally a generated Daft script run as a RayJob. Where no Ray
cluster is configured, a plain copy of a table (no filter, no transformation)
does not need one: this module streams the source in chunks through
SQLAlchemy/pandas into the destination, inside one destination transaction.

Guarantees, in order:

* the size bound is checked with ``COUNT(*)`` before anything is written, so
  an oversized table is refused with nothing created;
* the destination table, if this run creates it, is created from the source's
  own column types: exactly, between two databases of the same kind, and made
  generic otherwise (so PostgreSQL -> MySQL works). A ``numeric`` column stays
  ``numeric``; it is never re-derived from a sample;
* the table is created inside the same transaction as the rows, so on a
  database with transactional DDL (PostgreSQL) a failure leaves no trace at
  all; elsewhere a table this run created is dropped after the rollback;
* success means the destination gained exactly as many rows as the source has.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

LOCAL_COPY_MAX_ROWS = int(os.getenv("DTA_LOCAL_COPY_MAX_ROWS", "250000"))
LOCAL_COPY_CHUNK_ROWS = int(os.getenv("DTA_LOCAL_COPY_CHUNK_ROWS", "5000"))


class LocalCopyRefused(RuntimeError):
    """Refused before anything was written. The message is safe to show the user."""


class LocalCopyFailed(RuntimeError):
    """Failed after starting; ``cleaned_up`` says what was undone."""

    def __init__(self, message: str, cleaned_up: str):
        super().__init__(message)
        self.cleaned_up = cleaned_up


@dataclass
class LocalCopyResult:
    rows_copied: int
    source_rows: int
    destination_rows: int
    created_table: bool


def _split(table: str):
    parts = str(table).split(".")
    return (parts[0], parts[1]) if len(parts) == 2 else (None, parts[0])


def _engine(url: str, connect_args: Optional[Dict[str, Any]], label: str):
    # create_engine is where the DBAPI is imported, so a missing driver is
    # caught here as a refusal rather than escaping as an ImportError.
    try:
        from sqlalchemy import create_engine
        return create_engine(url, connect_args=connect_args or {})
    except Exception as exc:
        raise LocalCopyRefused(
            f"The {label} database driver is not available here ({type(exc).__name__})."
        ) from exc


def local_table_copy(
    source_url: str,
    source_table: str,
    destination_url: str,
    destination_table: str,
    *,
    create_if_missing: bool,
    write_mode: str = "append",
    max_rows: Optional[int] = None,
    chunk_rows: Optional[int] = None,
    source_connect_args: Optional[Dict[str, Any]] = None,
    destination_connect_args: Optional[Dict[str, Any]] = None,
) -> LocalCopyResult:
    """Copy every row of ``source_table`` into ``destination_table``; see the module docstring."""
    import pandas as pd
    from sqlalchemy import Column, MetaData, Table, func, inspect, select

    max_rows = LOCAL_COPY_MAX_ROWS if max_rows is None else int(max_rows)
    chunk_rows = LOCAL_COPY_CHUNK_ROWS if chunk_rows is None else max(1, int(chunk_rows))
    src_engine = _engine(source_url, source_connect_args, "source")
    dst_engine = _engine(destination_url, destination_connect_args, "destination")
    src_schema, src_name = _split(source_table)
    dst_schema, dst_name = _split(destination_table)
    created = False
    try:
        # ---- everything below until the first write is read-only ----
        try:
            src_meta = MetaData()
            src = Table(src_name, src_meta, schema=src_schema, autoload_with=src_engine)
            with src_engine.connect() as conn:
                source_rows = int(conn.execute(select(func.count()).select_from(src)).scalar() or 0)
        except Exception as exc:
            raise LocalCopyRefused(f"Could not read the source table ({type(exc).__name__}).") from exc
        if source_rows > max_rows:
            raise LocalCopyRefused(
                f"The source table has {source_rows:,} rows, above the {max_rows:,}-row "
                "limit for a transfer run without Ray."
            )
        exists = inspect(dst_engine).has_table(dst_name, schema=dst_schema)
        if not exists and not create_if_missing:
            raise LocalCopyRefused("The destination table does not exist and creating it was not requested.")

        dst_meta = MetaData()
        if exists:
            dst = Table(dst_name, dst_meta, schema=dst_schema, autoload_with=dst_engine)
            missing = [c.name for c in src.columns if c.name not in dst.columns]
            if missing:
                raise LocalCopyRefused(
                    f"The destination table has no column(s) {missing} that the source has."
                )
        else:
            same_kind = src_engine.dialect.name == dst_engine.dialect.name
            dst = Table(dst_name, dst_meta, schema=dst_schema)
            for col in src.columns:
                col_type = col.type
                if not same_kind:
                    try:
                        col_type = col.type.as_generic()
                    except Exception:
                        pass
                dst.append_column(Column(col.name, col_type, nullable=True))

        # ---- writes: the table (if new) and every row, in one transaction ----
        try:
            with dst_engine.begin() as dconn:
                if not exists:
                    dst.create(dconn)
                    created = True
                before = int(dconn.execute(select(func.count()).select_from(dst)).scalar() or 0)
                if exists and write_mode == "overwrite":
                    dconn.execute(dst.delete())
                    before = 0
                copied = 0
                with src_engine.connect() as sconn:
                    for chunk in pd.read_sql(select(src), sconn, chunksize=chunk_rows):
                        records = chunk.astype(object).where(pd.notnull(chunk), None).to_dict(orient="records")
                        if records:
                            dconn.execute(dst.insert(), records)
                            copied += len(records)
                after = int(dconn.execute(select(func.count()).select_from(dst)).scalar() or 0)
                if after - before != source_rows or copied != source_rows:
                    raise RuntimeError(
                        f"row count check failed: source {source_rows}, copied {copied}, "
                        f"destination gained {after - before}"
                    )
        except Exception as exc:
            cleaned = "Every row this run wrote was rolled back."
            if created:
                # Transactional DDL (PostgreSQL) already undid the CREATE with the
                # rollback; where DDL commits on its own (MySQL), drop it here.
                try:
                    dst.drop(dst_engine, checkfirst=True)
                    cleaned = "The destination table this run created was dropped."
                except Exception as drop_exc:
                    cleaned = (f"The destination table this run created could NOT be dropped "
                               f"({type(drop_exc).__name__}); drop it by hand.")
            raise LocalCopyFailed(f"The copy failed ({type(exc).__name__}: {str(exc)[:160]}).", cleaned) from exc
        return LocalCopyResult(rows_copied=copied, source_rows=source_rows,
                               destination_rows=after, created_table=created)
    finally:
        src_engine.dispose()
        dst_engine.dispose()
