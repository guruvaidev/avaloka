from __future__ import annotations

import os
from typing import Optional, List

import re
import json
import fsspec
import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text


# -------------------------------
# Format Detection (URL)
# -------------------------------
def detect_format_from_url(url: str, sniff_bytes: int = 4096) -> str:
    clean_url = url.split("?")[0]
    fs, fs_path = fsspec.core.url_to_fs(url)

    ext = os.path.splitext(fs_path)[1].lower()
    if ext in (".csv", ".json", ".parquet", ".jsonl", ".ndjson"):
        fmt = ext.lstrip(".")
        return "jsonl" if fmt in ("jsonl", "ndjson") else fmt

    # fallback: sniff content
    with fs.open(fs_path, "rb") as f:
        content = f.read(sniff_bytes)

    if content[:4] == b"PAR1":
        return "parquet"

    stripped = content.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        return "json"

    if b"," in content and b"\n" in content:
        return "csv"

    raise ValueError(f"Unable to detect format for {url}")


def _read_json_dataset(url: str, fmt: str, limit: Optional[int] = None) -> pd.DataFrame:
    if fmt == "jsonl":
        return pd.read_json(url, lines=True, nrows=limit)

    try:
        df = pd.read_json(url)
    except ValueError:
        df = pd.read_json(url, lines=True, nrows=limit)
        return df

    return df.head(limit) if limit else df


# -------------------------------
# File Column Extraction (NO LOAD)
# -------------------------------
def get_columns_from_url(url: str) -> List[str]:
    fmt = detect_format_from_url(url)

    if fmt == "csv":
        df = pd.read_csv(url, nrows=0)
        return list(df.columns)

    elif fmt in ("json", "jsonl"):
        df = _read_json_dataset(url, fmt, limit=1)
        return list(df.columns)

    elif fmt == "parquet":
        schema = pq.read_schema(url)
        return schema.names

    else:
        raise ValueError(f"Unsupported format: {fmt}")


# -------------------------------
# File Data Load
# -------------------------------
def load_dataset_from_url(url: str, limit: Optional[int]) -> pd.DataFrame:
    fmt = detect_format_from_url(url)

    if fmt == "csv":
        return pd.read_csv(url, nrows=limit)

    elif fmt in ("json", "jsonl"):
        return _read_json_dataset(url, fmt, limit=limit)

    elif fmt == "parquet":
        df = pd.read_parquet(url)
        return df.head(limit) if limit else df

    else:
        raise ValueError(f"Unsupported format: {fmt}")


# -------------------------------
# SQL Helpers
# -------------------------------
def _extract_table_name(uri: str) -> str:
    """
    Assumes last path segment is table/view name.
    Example:
        postgresql://user:pass@host:5432/db/table_name
    """
    return uri.rstrip("/").split("/")[-1]


def _apply_limit(sql: str, limit: int, dialect: str) -> str:
    if limit is None:
        return sql

    if dialect.startswith("mssql"):
        return sql.replace("SELECT", f"SELECT TOP {limit}", 1)
    else:
        return f"{sql} LIMIT {limit}"


# -------------------------------
# SQL Column Extraction (NO LOAD)
# -------------------------------
def get_columns_from_sql(connection_uri: str) -> List[str]:
    table = _extract_table_name(connection_uri)
    engine = create_engine(connection_uri)

    sql = f"SELECT * FROM {table} WHERE 1=0"

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    return list(df.columns)


# -------------------------------
# SQL Data Load
# -------------------------------
def load_dataset_from_sql(
    connection_uri: str,
    limit: Optional[int]
) -> pd.DataFrame:
    table = _extract_table_name(connection_uri)
    engine = create_engine(connection_uri)

    dialect = engine.dialect.name
    base_sql = f"SELECT * FROM {table}"
    sql = _apply_limit(base_sql, limit, dialect)

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    return df


# -------------------------------
# Unified API
# -------------------------------
def get_columns(data_source_uri: str) -> List[str]:
    """
    Returns column names WITHOUT loading full dataset.
    """

    if data_source_uri.startswith(("postgresql://", "mysql://", "sqlite://", "mssql://")):
        return get_columns_from_sql(data_source_uri)
    else:
        return get_columns_from_url(data_source_uri)


def load_dataset(
    data_source_uri: str,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Loads dataset with optional row limit.
    """

    if data_source_uri.startswith(("postgresql://", "mysql://", "sqlite://", "mssql://")):
        return load_dataset_from_sql(data_source_uri, limit=limit)
    else:
        return load_dataset_from_url(data_source_uri, limit=limit)


# -------------------------------
# Get JSON data from content (for agent message parsing)
# -------------------------------
def extract_json_from_content(content: str):
    """Extracts a JSON object from a string, handling markdown code blocks."""
    try:
        # First, try to find the JSON block and load it directly (in case it's already valid JSON)
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in content")
            
        json_str = content[start:end + 1]
        
        # Fast path: if it's already valid JSON, return it
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
        # Fallback: attempt to clean up python-dict-like syntax
        # Instead of a blind replace, let's use ast.literal_eval for python-like dicts
        import ast
        try:
            # We need to replace true/false/null to True/False/None for ast if they are unquoted
            py_str = re.sub(r"\btrue\b", "True", json_str)
            py_str = re.sub(r"\bfalse\b", "False", py_str)
            py_str = re.sub(r"\bnull\b", "None", py_str)
            parsed_dict = ast.literal_eval(py_str)
            if isinstance(parsed_dict, dict):
                return parsed_dict
        except (SyntaxError, ValueError):
            pass

        # If both fail, raise the original error for debugging
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Could not parse JSON from text: {e}\nContent was: {content[:200]}...") from e
