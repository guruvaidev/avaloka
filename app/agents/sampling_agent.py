from __future__ import annotations
import csv
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import pandas.errors as pde
import pyarrow as pa
from file_handler.handler import FileHandler
from file_handler.excel_connector import ExcelConnector

# =========================
# Config (env overrides)
# =========================
# TODO(byte-routing): this row cap becomes redundant once byte-based routing
# lands (estimated_bytes decides full-vs-defer); remove it then.
DEFAULT_SAMPLE_MAX_ROWS = int(os.getenv("DEFAULT_SAMPLE_MAX_ROWS", "50000"))
__all__ = ["sample_data_from_source", "DEFAULT_SAMPLE_MAX_ROWS"]
# If you still want to support ad-hoc local SQLite files, enable this.
# Otherwise leave 0 and route SQL through Postgres MCP as per review.
ENABLE_SQLITE = bool(int(os.getenv("ENABLE_SQLITE", "0")))

# =========================
# Arrow → SQL type map
# =========================
# Schema/DDL inference uses PyArrow (Ray Data's own engine) instead of Spark: no
# JVM/JRE, no SparkSession, and for large files it's a cheap metadata read (parquet
# footer) rather than a full distributed job. The heavy, genuinely-distributed
# large-file *processing* paths already run on Ray/daft (sampling_agent_daft),
# so schema inference — a metadata op — needs neither Spark nor a Ray cluster.
ARROW_TO_SQL_TYPE_MAP: Dict[str, str] = {
    "int8": "TINYINT",
    "int16": "SMALLINT",
    "int32": "INT",
    "int64": "BIGINT",
    "uint8": "SMALLINT",
    "uint16": "INT",
    "uint32": "BIGINT",
    "uint64": "BIGINT",
    "halffloat": "FLOAT",
    "float": "FLOAT",
    "double": "DOUBLE",
    "string": "VARCHAR",
    "large_string": "VARCHAR",
    "bool": "BOOLEAN",
    "binary": "BLOB",
    "large_binary": "BLOB",
}

# =========================
# CSV helpers
# =========================
def _sniff_delimiter(path: Union[str, os.PathLike], sample_bytes: int = 8192) -> str:
    with open(path, "rb") as f:
        sample = f.read(sample_bytes)
    text = sample.decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","

def _reparse_csv_exact_fields(path: Union[str, os.PathLike], delimiter: str, encoding: str) -> pd.DataFrame:
    """
    Strictly parse CSV: keep only rows whose number of fields exactly matches the header.
    Skips empty/whitespace-only lines. Uses csv.reader to respect quoting.
    """
    with open(path, "r", encoding=encoding, errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        if not header:
            return pd.DataFrame()
        expected = len(header)
        good_rows: List[List[str]] = []
        for row in reader:
            if not row or not any(str(x).strip() for x in row):
                continue
            if len(row) == expected:
                good_rows.append(row)

        # ---- NEW: normalize types/empties ----
        df_out = pd.DataFrame(good_rows, columns=header)

        # turn blank/whitespace-only cells into NA so "all-null stratify" is caught
        df_out = df_out.replace(r'^\s*$', pd.NA, regex=True)

        # best-effort numeric coercion so "1" → 1, "3.0" → 3.0 (non-numerics stay as-is)
        for c in df_out.columns:
            try:
                df_out[c] = pd.to_numeric(df_out[c])
            except (ValueError, TypeError):
                # Keep non-numeric values as-is
                pass

        return df_out

def _read_csv_flex(path: Union[str, os.PathLike]) -> Tuple[pd.DataFrame, str, str]:
    """Robust CSV loader that returns (df, encoding, delimiter)."""
    delimiter = _sniff_delimiter(path)
    last_err: Optional[Exception] = None

    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"):
        try:
            df = pd.read_csv(
                path,
                sep=delimiter,
                encoding=enc,
                engine="python",
                on_bad_lines="skip",
                quotechar='"',
            )
            # If everything collapsed into one column with comma, try semicolon once.
            if df.shape[1] == 1 and delimiter == ",":
                try:
                    df2 = pd.read_csv(
                        path,
                        sep=";",
                        encoding=enc,
                        engine="python",
                        on_bad_lines="skip",
                        quotechar='"',
                    )
                    if df2.shape[1] > 1:
                        return df2, enc, ";"
                except Exception:
                    pass

            if df.columns is None or len(df.columns) == 0:
                raise ValueError("No columns detected in CSV.")
            return df, enc, delimiter
        except (UnicodeDecodeError, pde.ParserError, ValueError) as e:
            last_err = e

    raise last_err if last_err else RuntimeError("Unable to read CSV with any encoding.")

def _has_too_few_fields(path: Union[str, os.PathLike], delimiter: str, encoding: str) -> bool:
    """
    Return True if any non-empty data line has fewer fields than the header.
    Uses csv.reader to respect quoting.
    """
    try:
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            lines = [ln.rstrip("\r\n") for ln in f.readlines()]
        if not lines:
            return False
        header_fields = next(csv.reader([lines[0]], delimiter=delimiter))
        expected = len(header_fields)
        for ln in lines[1:]:
            if not ln.strip():
                continue
            row_fields = next(csv.reader([ln], delimiter=delimiter))
            if len(row_fields) < expected:
                return True
        return False
    except Exception:
        # Be conservative: if we can't check, don't force an error here
        return False


# =========================
# Arrow schema helpers (Spark-free)
# =========================
def _arrow_type_to_sql(dtype: "pa.DataType") -> str:
    """Map a PyArrow type to a SQL column type, preserving decimal precision."""
    s = str(dtype).lower()
    if s.startswith("timestamp"):
        return "TIMESTAMP"
    if s.startswith("date"):
        return "DATE"
    if s.startswith("time"):
        return "TIME"
    if s.startswith("decimal"):
        # decimal128(10, 2) -> DECIMAL(10, 2)
        inside = s[s.find("(") : s.rfind(")") + 1] if "(" in s else ""
        return f"DECIMAL{inside}" if inside else "DECIMAL"
    if s.startswith(("list", "large_list", "struct", "map")):
        return "TEXT"
    return ARROW_TO_SQL_TYPE_MAP.get(s, "TEXT")


def get_arrow_schema(
    file_path: Union[str, os.PathLike],
    source_fmt: str,
    *,
    df: Optional[pd.DataFrame] = None,
) -> "pa.Schema":
    """Infer an Arrow schema without Spark.

    For parquet we read only the file footer (cheap, scales to huge files). For every
    other format the caller has already materialised a pandas frame, so we derive the
    Arrow schema from it — better type fidelity than pandas dtypes, no re-read.
    """
    fmt = (source_fmt or "").lower()
    if fmt == "parquet":
        import pyarrow.parquet as pq
        return pq.read_schema(os.fspath(file_path))
    if df is not None:
        return pa.Table.from_pandas(df, preserve_index=False).schema
    raise ValueError(f"No dataframe supplied and unsupported source_type for Arrow schema: {source_fmt}")


def generate_ddl_from_arrow_schema(table_name: str, schema: "pa.Schema") -> str:
    cols: List[str] = [f"{name} {_arrow_type_to_sql(schema.field(name).type)}" for name in schema.names]
    return f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(cols) + "\n);"

# =========================
# Pandas DDL fallback
# =========================
_PANDAS_TO_SQL: Dict[str, str] = {
    "int64": "BIGINT",
    "int32": "INT",
    "float64": "DOUBLE",
    "float32": "FLOAT",
    "bool": "BOOLEAN",
    "boolean": "BOOLEAN",
    "object": "VARCHAR",
    "string": "VARCHAR",
    "category": "VARCHAR",
    "datetime64[ns]": "TIMESTAMP",
}

# def _generate_ddl_from_pandas(df: pd.DataFrame, table_name: str) -> str:
#     cols: List[str] = [f"{name} {_PANDAS_TO_SQL.get(str(dtype), 'TEXT')}" for name, dtype in df.dtypes.items()]
#     return f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(cols) + "\n);"


def _generate_ddl_from_pandas(df: pd.DataFrame, table_name: str) -> str:
    cols: List[str] = []
    for name, dtype in df.dtypes.items():
        dtype_str = str(dtype)
        sql_type = _PANDAS_TO_SQL.get(dtype_str, "TEXT")

        # Name-based hints to distinguish common width variants when pandas collapses types
        nl = name.lower()
        if "i32" in nl or "int32" in nl:
            sql_type = "INT"
        elif "i64" in nl or "int64" in nl:
            sql_type = "BIGINT"
        elif "f32" in nl or "float32" in nl:
            sql_type = "FLOAT"
        elif "f64" in nl or "float64" in nl or "double" in nl:
            sql_type = "DOUBLE"
        elif nl in ("b", "is_active", "flag") and sql_type == "VARCHAR":
            # simple heuristic if a boolean column came in as object
            unique_vals = set(map(lambda x: str(x).strip().lower(), df[name].dropna().unique().tolist()))
            if unique_vals.issubset({"true", "false", "0", "1"}):
                sql_type = "BOOLEAN"

        cols.append(f"{name} {sql_type}")

    return f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(cols) + "\n);"


# =========================
# Stratified sampling helpers
# =========================
def infer_best_stratify_column(df: pd.DataFrame) -> Optional[str]:
    categorical = df.select_dtypes(include=["object", "category"]).nunique(dropna=False)
    candidates = categorical[categorical.between(2, 20)].sort_values()
    return candidates.index[0] if not candidates.empty else None


def stratified_sample(
    df: pd.DataFrame,
    stratify_by: str,
    sample_size: Union[int, float],
    *,
    cap: int,
) -> pd.DataFrame:
    if stratify_by not in df.columns:
        raise ValueError(f"Stratify column '{stratify_by}' not found in dataframe.")
    if df[stratify_by].isnull().all():
        raise ValueError(f"Stratify column '{stratify_by}' contains only null or missing values")

    groups = df.groupby(stratify_by, dropna=False)
    total_rows = len(df)
    cap = min(cap, total_rows)

    # Fractional request → per-group fraction, then trim to cap if needed
    if isinstance(sample_size, float):
        if not (0.0 < sample_size <= 1.0):
            raise ValueError("sample_size (float) must be in (0, 1].")
        parts: List[pd.DataFrame] = []
        for _, g in groups:
            if len(g) == 0:
                continue
            n_g = max(1, int(len(g) * sample_size))
            n_g = min(n_g, len(g))
            parts.append(g.sample(n=n_g, random_state=42))
        sampled = pd.concat(parts).reset_index(drop=True) if parts else df.head(0)
        if len(sampled) > cap:
            sampled = sampled.sample(n=cap, random_state=42).reset_index(drop=True)
        return sampled

    # Integer target → proportional allocation across groups
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")
    target = min(int(sample_size), cap)

    sizes = groups.size().astype(float)
    weights = sizes / sizes.sum() if sizes.sum() else sizes
    raw_alloc = weights * target
    base_alloc = raw_alloc.astype(int)
    remainder = target - int(base_alloc.sum())

    frac = (raw_alloc - base_alloc).sort_values(ascending=False)
    alloc = base_alloc.copy()
    for key in frac.index[:remainder]:
        alloc[key] += 1

    parts: List[pd.DataFrame] = []
    for key, g in groups:
        n_g = int(alloc.get(key, 0))
        if n_g <= 0:
            continue
        n_g = min(n_g, len(g))
        parts.append(g.sample(n=n_g, random_state=42))

    sampled = pd.concat(parts).reset_index(drop=True) if parts else df.head(0)
    if len(sampled) > cap:
        sampled = sampled.sample(n=cap, random_state=42).reset_index(drop=True)
    return sampled

# =========================
# MAIN
# =========================
def sample_data_from_source(
    path: str,
    source_type: str,
    stratify_by: Optional[str],
    sample_size: Union[int, float] = 5,
    *,
    disable_cap_for_fraction: bool = True,
    max_rows: Optional[int] = None,
    **extra_args: Any,
) -> Dict[str, Optional[object]]:
    """
    Load a dataset, return a representative row sample, and generate a SQL-style DDL.

    Supported formats: CSV/TSV/JSON/Parquet/Avro/ORC/Delta/XML (via FileHandler),
    and optionally SQLite when ENABLE_SQLITE=1.

    Args:
        path: Absolute or relative path to the input file.
        source_type: Logical format of the file (e.g. "csv", "json", "parquet", "avro",
            "orc", "delta", "xml", or "sqlite" when enabled).
        stratify_by: Optional column name to stratify the sampling. If not provided,
            the function will try to auto-infer a categorical column (2–20 distinct values).
        sample_size: If int, the requested number of rows to sample.
            If float in (0,1], the requested fraction of total rows. The final sample
            is always capped by `effective_cap` (see below).
        disable_cap_for_fraction: (Deprecated, kept for API compatibility) Ignored.
            A hard cap is always enforced regardless of integer or fractional sampling.
        max_rows: Optional per-call upper bound for rows returned. If None, falls back
            to the global DEFAULT_SAMPLE_MAX_ROWS (from env). The sampler will never
            return more than min(total_rows, DEFAULT_SAMPLE_MAX_ROWS, max_rows or DEFAULT_SAMPLE_MAX_ROWS).
        **extra_args: Additional keyword arguments passed to the FileHandler for non-CSV inputs.

    Returns:
        Dict with:
            - "schema": List[str] of column names.
            - "rows": List[Dict[str, Any]] sampled records (size ≤ effective_cap).
            - "table_name": Name used for schema generation (None for CSV/TSV).
            - "ddl_schema": CREATE TABLE statement (Spark-inferred; pandas fallback).
            - "csv_path" / "encoding" / "delimiter": Provided for CSV/TSV inputs.
            - "error": None on success; error string on failure.

    Notes on caps:
        The function enforces a hard cap to keep payloads small and responsive.
        The effective cap is:
            effective_cap = min(total_rows, DEFAULT_SAMPLE_MAX_ROWS, max_rows or DEFAULT_SAMPLE_MAX_ROWS)
    """

    conn = None
    path = os.path.abspath(path)
    excel_sheets: Optional[List[str]] = None
    excel_selected_sheet: Optional[str] = None
    warnings: List[str] = []
    try:
        # ---- Validate requested sample_size type ----
        if isinstance(sample_size, int):
            if sample_size <= 0:
                raise ValueError("sample_size must be a positive integer.")
        elif isinstance(sample_size, float):
            if not (0.0 < sample_size <= 1.0):
                raise ValueError("sample_size (float) must be in (0, 1].")
        else:
            raise TypeError("sample_size must be int or float.")

        source_fmt = (source_type or "").lower()

        # ---- Load data ----
        # TSV is the same file family as CSV and _read_csv_flex already sniffs
        # the delimiter, so it belongs here rather than in the FileHandler
        # branch, which has no TSV reader and raised "File type not supported"
        # for a format /api/upload advertises in SUPPORTED_UPLOAD_EXTS. The
        # rest of this function already assumed the pairing -- it writes
        # csv_path/encoding/delimiter for `source_fmt in ("csv", "tsv")`.
        if source_fmt in ("csv", "tsv"):
            df, used_encoding, used_delim = _read_csv_flex(path)
            table_name = "sample_table"

            # Normalize: ensure only rows with exactly the header's field count remain.
            # (Pandas 'python' engine may coerce overlong rows; we reparse strictly.)
            try:
                df_strict = _reparse_csv_exact_fields(path, used_delim, used_encoding)
                if not df_strict.empty:
                    df = df_strict
            except Exception:
                # Best-effort hardening; if anything goes wrong, keep the lenient df
                pass


        elif source_fmt in {"sqlite", "sql"}:
            if not ENABLE_SQLITE:
                raise ValueError(
                    "Local SQLite reading is disabled (ENABLE_SQLITE=0). "
                    "Use the Postgres MCP for SQL sources or set ENABLE_SQLITE=1 to allow local .sqlite files."
                )
            import sqlite3  # lazy import only when enabled

            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table = cursor.fetchone()
            if not table:
                raise ValueError("No tables found in SQLite database.")
            table_name = table[0]
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}";', conn)
            used_encoding, used_delim = "utf-8", ","

        elif source_fmt in ("excel", "xlsx", "xls"):
            # Read Excel as a typed DataFrame (dtypes preserved) instead of
            # round-tripping through a list of Python scalars, which would
            # stringify dates and collapse numeric columns to object -> weaker
            # profiling than the CSV path produced for identical data.
            xl_conn = ExcelConnector(path, sheet_name=extra_args.get("sheet_name"))
            excel_sheets = xl_conn.list_sheets()
            excel_selected_sheet = xl_conn.selected_sheet
            df = xl_conn.load_dataframe()
            table_name = "sample_table"
            used_encoding, used_delim = "utf-8", ","

            if len(excel_sheets) > 1:
                skipped = [s for s in excel_sheets if s != excel_selected_sheet]
                warnings.append(
                    f"Workbook contains {len(excel_sheets)} sheets {excel_sheets}. "
                    f"Analyzed sheet '{excel_selected_sheet}' (the one with the most rows). "
                    f"Sheets not analyzed: {skipped}. "
                    f"Pass sheet_name to analyze a specific sheet."
                )

        elif source_fmt == "xml":
            # pandas reads XML directly; FileHandler has no XML reader, so this
            # advertised format previously fell through and 500'd. read_xml
            # needs a row-level xpath when the rows are not direct children of
            # the root, so try the default and then a generic descendant match
            # before giving up with a message that says what to do.
            try:
                df = pd.read_xml(path)
            except Exception:
                try:
                    df = pd.read_xml(path, xpath=".//*[*]")
                except Exception as exc:
                    raise ValueError(
                        f"Could not infer a table from this XML: {exc}. XML has no "
                        f"single tabular shape; pass extra_args={{'xpath': ...}} "
                        f"naming the repeating element."
                    ) from exc
            table_name = "sample_table"
            used_encoding, used_delim = "utf-8", ","

        else:
            # Avro / JSON / Parquet / Delta / Iceberg… via your FileHandler
            reader = FileHandler(path, source_fmt, **(extra_args or {}))
            #df = pd.DataFrame(reader.load_data(), columns=reader.get_columns())
            data = reader.load_data()
            cols = reader.get_columns() or None
            df = pd.DataFrame(data, columns=cols)
            table_name = "sample_table"
            used_encoding, used_delim = "utf-8", ","

        if df is None or df.empty:
            return {
                "schema": [],
                "rows": [],
                "table_name": None,
                "ddl_schema": "",
                "error": None,
                "csv_path": os.fspath(path) if source_fmt in ("csv", "tsv") else None,
                "encoding": used_encoding if source_fmt in ("csv", "tsv") else None,
                "delimiter": used_delim if source_fmt in ("csv", "tsv") else None,
                "sheets": excel_sheets,
                "selected_sheet": excel_selected_sheet,
                "warnings": warnings,
            }

        total_rows = len(df)
        effective_cap = DEFAULT_SAMPLE_MAX_ROWS if max_rows is None else int(max_rows)

        # ---- Byte-aware clamp (ported from sampling_agent_daft.create_base_sample) ----
        # Row caps alone don't bound payload size: a wide frame (long-text / JSON
        # columns) can blow the sample past what preview/LLM consumers handle.
        # Same approach as the Daft sampler: measure bytes/row on a small head
        # slice via Arrow, cap the sample at ~5 MB. Best-effort — if the Arrow
        # conversion fails (exotic dtypes), keep the plain row cap.
        try:
            _head_tbl = pa.Table.from_pandas(df.head(1000), preserve_index=False)
            _bytes_per_row = _head_tbl.nbytes / max(_head_tbl.num_rows, 1)
            if _bytes_per_row > 0:
                effective_cap = max(min(int(5_242_880 / _bytes_per_row), effective_cap), 1)
        except Exception:
            _bytes_per_row = None

        # ---- Choose/validate stratify column ----
        if stratify_by and stratify_by not in df.columns:
            raise ValueError(f"Stratify column '{stratify_by}' not found")
        if not stratify_by:
            stratify_by = infer_best_stratify_column(df)

        # ---- Compute final target 'n' BEFORE sampling ----
        if isinstance(sample_size, float):
            requested = max(1, int(total_rows * sample_size))
        else:
            requested = int(sample_size)
        n = max(1, min(requested, effective_cap))


        # Strict-on-short-lines only when caller asks for more rows than valid ones
        if source_fmt == "csv":
            try:
                if _has_too_few_fields(path, used_delim, used_encoding) and n > len(df):
                    raise ValueError("Malformed CSV: a row has too few fields")
            except Exception:
                # Raising here gets caught by the outer try/except and returned as an error
                raise

        # ---- Sample ----
        if stratify_by and stratify_by in df.columns:
            df_sampled = stratified_sample(df, stratify_by=stratify_by, sample_size=n, cap=effective_cap)
        elif n >= total_rows:
            df_sampled = df
        else:
            # seeded random sample (representative, unlike a biased head(n))
            df_sampled = df.sample(n=n, random_state=42)

        sample_rows = df_sampled.to_dict(orient="records")

        # ---- DDL ----
        if source_fmt in ("csv", "tsv", "json", "parquet", "avro", "orc", "delta", "xml"):
            try:
                arrow_schema = get_arrow_schema(path, source_fmt, df=df)
                ddl_schema = generate_ddl_from_arrow_schema(table_name, arrow_schema)
            except Exception:
                ddl_schema = _generate_ddl_from_pandas(df, table_name)

        elif source_fmt in {"sqlite", "sql"}:
            ddl_schema = _generate_ddl_from_pandas(df, table_name)

        else:
            ddl_schema = _generate_ddl_from_pandas(df, table_name)

        return {
            "schema": df.columns.tolist(),
            "rows": sample_rows,
            "table_name": None if source_fmt in ("csv", "tsv") else table_name,
            "ddl_schema": ddl_schema,
            "error": None,
            # Measured in-memory cost signal (Arrow bytes/row from the head
            # slice; None if the Arrow conversion failed) for byte-aware routing.
            "bytes_per_row": round(_bytes_per_row, 2) if _bytes_per_row else None,
            "estimated_full_bytes": int(total_rows * _bytes_per_row) if _bytes_per_row else None,
            "csv_path": os.fspath(path) if source_fmt in ("csv", "tsv") else None,
            "encoding": used_encoding if source_fmt in ("csv", "tsv") else None,
            "delimiter": used_delim if source_fmt in ("csv", "tsv") else None,
            "sheets": excel_sheets,
            "selected_sheet": excel_selected_sheet,
            "warnings": warnings,
        }

    except Exception as e:
        return {
            "error": f"Failed to sample data: {str(e)}",
            "trace": traceback.format_exc(),
        }
    finally:
        if conn:
            conn.close()
