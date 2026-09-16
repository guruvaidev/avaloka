# =============================================================================
# data_transfer_agent.py
# =============================================================================
# Orchestrates the full ETL pipeline:
#   1. Credential validation & connection probing
#   2. Schema deduction (source + destination)
#   3. Sample data fetch
#   4. AI-driven code generation, validation, and execution
#   5. Schema compatibility check
#   6. Injection-script assembly
# =============================================================================

import logging
import os
import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import daft

from app.agents.data_transfer_agent.dta_state import (
    DaftCodingAgentState as CodingAgentState,
    DaftETLState as ETLState,
    cloud_storage_credentials,
    db_credentials,
)
from app.agents.data_transfer_agent import daft_coder
from app.agents.data_transfer_agent.daft_coder import (
    daft_coder_node,
    daft_pseudocode_node,
    daft_retrieval_node,
    empty_source_guard_node,
    is_stub_code,
    sanitize_user_prompt_for_transformations,
    strip_runtime_analysis_context,
)
from app.agents.data_transfer_agent.daft_execution import execution_agent_node_local
from app.agents.data_transfer_agent.daft_validator import logical_semantic_validator_node
from app.agents.validator import static_semantic_validator_node, syntactic_validator_node

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger(__name__)

# Shared type groups — defined once and reused everywhere to avoid typos
_DB_TYPES = {"mysql", "postgresql", "postgres"}
_FILE_TYPES = {"csv", "json", "parquet"}

# Timestamp helper

def _ts() -> str:
    """Returns the current UTC time as a compact ISO-8601 string."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")


def log_step(step: str, detail: str = "") -> None:
    """Logs a clearly delimited pipeline step boundary."""
    msg = f"──── [{_ts()}] {step}"
    if detail:
        msg += f" | {detail}"
    logger.info(msg)

# Custom exception

class PipelineError(RuntimeError):
    """
    Raised when a non-recoverable error occurs inside the pipeline.
    Callers catch this at the top level and surface it as ``error_message``
    in the returned state so the caller gets a clean, human-readable reason.
    """

# Connection building

def build_connection_string(
    credentials: Union[db_credentials, cloud_storage_credentials, dict],
    source_type: str,
) -> Tuple[Optional[str], Any]:
    """
    Converts credentials into a connection URL (or cloud URI) plus any
    supplementary config object required by Daft / SQLAlchemy.

    Returns
    -------
    (url_or_uri, connect_args_or_io_config)
      - Databases   → (sqlalchemy_url,  connect_args dict)
      - Cloud files → (gs:// / s3:// / az:// URI,  daft IOConfig)
      - Local files → (local_path,  empty IOConfig)

    Raises
    ------
    PipelineError
        When the cloud provider or source type is not supported.
    """
    from daft.io import AzureConfig, GCSConfig, IOConfig, S3Config

    # Normalise credentials to a plain dict where possible
    if isinstance(credentials, db_credentials):
        creds: Any = {
            "user": credentials.user,
            "password": credentials.password.get_secret_value(),
            "host": credentials.host,
            "port": credentials.port,
            "database": credentials.database,
            "table": credentials.table,
            "schema_name": credentials.schema_name,
            "sslmode": credentials.sslmode,
        }
    elif isinstance(credentials, cloud_storage_credentials):
        creds = credentials  # handled in the cloud branch below
    else:
        creds = credentials  # plain dict — legacy callers

    # Built Database connections string
    if source_type in _DB_TYPES:
        user = creds.get("user")
        password = urllib.parse.quote_plus(creds.get("password", ""))
        host = creds.get("host", "localhost")
        database = creds.get("database")

        if source_type == "mysql":
            port = creds.get("port", 3306)
            ssl_params: Dict[str, str] = {}
            if isinstance(creds.get("ssl"), dict):
                ssl_params = {k: str(v) for k, v in creds["ssl"].items()}
            qs = f"?{urllib.parse.urlencode(ssl_params)}" if ssl_params else ""
            url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}{qs}"
            logger.info("[%s] MySQL URL built — host=%s, db=%s.", _ts(), host, database)
            return url, {}

        # postgresql / postgres
        port = creds.get("port", 5432)
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        # psycopg2 takes sslmode/sslrootcert directly; an ssl.SSLContext is not a
        # valid connect_arg for it and raises TypeError.
        connect_args: Dict[str, Any] = {}
        sslmode = creds.get("sslmode")
        if sslmode:
            connect_args["sslmode"] = sslmode
            if creds.get("sslrootcert"):
                connect_args["sslrootcert"] = creds["sslrootcert"]
            logger.info("[%s] PostgreSQL SSL mode: %s.", _ts(), sslmode)

        logger.info("[%s] PostgreSQL URL built — host=%s, db=%s.", _ts(), host, database)
        return url, connect_args

    # Cloud / local file connections 
    if source_type in _FILE_TYPES:
        if isinstance(creds, cloud_storage_credentials):
            provider = creds.provider.lower()
            uri = creds.get_cloud_uri(source_type)
            logger.info("[%s] Cloud provider=%s | URI=%s.", _ts(), provider, uri)

            if provider == "gcp":
                if creds.gcp_service_account_info:
                    import json as _json
                    gcs_cfg = GCSConfig(credentials=_json.dumps(creds.gcp_service_account_info))
                    logger.info("[%s] GCSConfig: explicit service-account credentials.", _ts())
                else:
                    gcs_cfg = GCSConfig()
                    logger.info("[%s] GCSConfig: auto-detect via GOOGLE_APPLICATION_CREDENTIALS.", _ts())
                return uri, IOConfig(gcs=gcs_cfg)

            if provider == "aws":
                s3_cfg = S3Config(
                    key_id=creds.access_key,
                    access_key=creds.secret_key.get_secret_value() if creds.secret_key else None,
                    region_name=creds.region,
                )
                logger.info("[%s] S3Config built — region=%s.", _ts(), creds.region)
                return uri, IOConfig(s3=s3_cfg)

            if provider == "azure":
                az_cfg = AzureConfig(
                    storage_account=creds.az_storage_account,
                    access_key=creds.az_access_key.get_secret_value() if creds.az_access_key else None,
                    sas_token=creds.az_sas_token.get_secret_value() if creds.az_sas_token else None,
                    tenant_id=creds.az_tenant_id,
                    client_id=creds.az_client_id,
                    client_secret=creds.az_client_secret.get_secret_value() if creds.az_client_secret else None,
                )
                logger.info("[%s] AzureConfig built — account=%s.", _ts(), creds.az_storage_account)
                return uri, IOConfig(azure=az_cfg)

            raise PipelineError(f"Unsupported cloud provider: '{provider}'.")

        # Local file path
        local_path = creds.get("file_path") if isinstance(creds, dict) else None
        logger.info("[%s] Local file path: %s.", _ts(), local_path)
        return local_path, IOConfig()

    raise PipelineError(f"Unsupported source_type: '{source_type}'.")

# DB connection probe  ← the missing safety net

def probe_db_connection(
    connection_string: str, label: str, connect_args: Optional[Dict[str, Any]] = None
) -> None:
    """
    Opens a real TCP connection to the database, runs ``SELECT 1``, and
    immediately closes it.

    **Why this exists:** ``build_connection_string`` only assembles a URL —
    it never touches the network.  Without an early probe, a wrong password,
    unreachable host, or firewall block would silently pass through schema
    deduction (which swallows its own exceptions) and only surface as a
    cryptic error deep inside the AI repair loop — after wasting LLM calls.
    This probe catches the problem at Step 2, before any AI work begins.

    Parameters
    ----------
    connection_string : str
        SQLAlchemy-compatible URL.
    label : str
        Human-readable label for log messages, e.g. ``"source"`` or
        ``"destination"``.

    Raises
    ------
    PipelineError
        With a clear, actionable message when the connection cannot be
        established.
    """
    safe_url = connection_string.split("@")[-1]
    log_step(f"DB connection probe — {label}", safe_url)

    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(connection_string, connect_args=connect_args or {})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.info("[%s] DB connection successful for %s.", _ts(),label.capitalize(),)

    except Exception as exc:
        logger.error("[%s] DB connection FAILED for %s: %s",_ts(),label.capitalize(),exc,)

        raise PipelineError(
            f"Cannot connect to {label} database. "
            f"Check host, port, credentials, and firewall rules. "
            f"Detail: {exc}"
        ) from exc

def _url_with_ssl_params(url: str, connect_args: Optional[Dict[str, Any]]) -> str:
    """daft.read_sql opens its own psycopg2 connection and takes no connect_args,
    so fold sslmode/sslrootcert into the URL query string instead."""
    ssl_params = {
        k: connect_args[k]
        for k in ("sslmode", "sslrootcert")
        if connect_args and connect_args.get(k)
    }
    if not ssl_params:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urllib.parse.urlencode(ssl_params)}"


def _read_daft_df(
    data_type: str,
    conn_str: str,
    table: str = None,
    io_config=None,
    limit: int = None,
    connect_args: Optional[Dict[str, Any]] = None,
):
    """
    Centralized reader for DB + file sources.
    """
    logger.info(
        "[%s] Reading data | type=%s | table=%s | limit=%s",
        _ts(), data_type, table, limit
    )

    if data_type in _DB_TYPES:
        if not table:
            raise ValueError("Table name required for DB source")

        query = f"SELECT * FROM {table}"
        if limit:
            query += f" LIMIT {limit}"

        return daft.read_sql(query, conn=_url_with_ssl_params(conn_str, connect_args))

    elif data_type in _FILE_TYPES:
        readers = {
            "csv": daft.read_csv,
            "json": daft.read_json,
            "parquet": daft.read_parquet,
        }

        reader = readers.get(data_type)
        if not reader:
            raise ValueError(f"Unsupported file type: {data_type}")

        df = reader(conn_str, io_config=io_config)

        return df.limit(limit) if limit else df

    else:
        raise ValueError(f"Unsupported data type: {data_type}")


# Substrings that mean the object is genuinely absent (a legitimate "new
# table/file") rather than a transient read/permission error. DB indicators are
# kept strict so a connection/host error is NOT mistaken for a missing table.
_DB_MISSING_INDICATORS = ("does not exist", "doesn't exist", "no such table", "undefinedtable")
_FILE_MISSING_INDICATORS = (
    "does not exist", "not found", "no such file", "no files", "nosuchkey", "nosuchbucket", "404",
)


def _is_missing_object_error(exc: Exception, data_type: str) -> bool:
    """True when *exc* means the table/file is absent, as opposed to a transient
    connection/permission error that must not silently skip the schema check."""
    if isinstance(exc, FileNotFoundError):
        return True
    msg = str(exc).lower()
    indicators = _DB_MISSING_INDICATORS if data_type in _DB_TYPES else _FILE_MISSING_INDICATORS
    return any(ind in msg for ind in indicators)


def deduce_schema(
    data_type: str,
    conn_str: str,
    table: str = None,
    io_config=None,
):
    log_step("deduce_schema", f"type={data_type} table={table}")

    try:
        df = _read_daft_df(
            data_type,
            conn_str,
            table=table,
            io_config=io_config,
            limit=2,
        )

        schema = df.schema()

        # Always normalise to a {column: dtype} dict, for every source type.
        # Previously only file sources were parsed, so a DB source returned a
        # raw Daft Schema object. Once shared consumers began reading
        # "source_schema" (e.g. static_semantic_validator_node calling
        # schema.keys()), that raw Schema crashed with
        # "'Schema' object has no attribute 'keys'" on D2C/D2D transfers.
        schema = parse_schema(schema)

        logger.info("[%s] Schema deduced successfully.", _ts())
        return schema

    except Exception as exc:
        logger.warning(
            "[%s] Could not deduce schema (type=%s, table=%s): %s",
            _ts(),
            data_type,
            table,
            exc,
        )
        if _is_missing_object_error(exc, data_type):
            # Genuinely absent → legitimate "new table/file"; callers handle it.
            return None if data_type in _DB_TYPES else {}
        # A transient read/permission error on a possibly-existing object — do
        # not let callers silently treat it as "new" and skip the schema check.
        raise

# Schema utilities

def parse_schema(schema: Any) -> Dict[str, str]:
    """
    Normalises any schema representation into a ``{column_name: dtype}`` dict.

    Handles:
    - Daft/Polars pipe-delimited string output (from ``df.show()``)
    - Daft Schema objects with ``to_name_and_dtype_list()``
    - Daft/PyArrow Field iterables
    - Plain dicts
    - pandas-style ``dtypes`` attributes
    - Generic column-name iterables (dtype set to ``"Unknown"``)
    """
    if schema is None:
        return {}

    # Pipe-delimited string (e.g. output of df.show())
    if isinstance(schema, str) and "│" in schema:
        result: Dict[str, str] = {}
        for line in schema.strip().split("\n"):
            if "│" in line and "┆" in line:
                parts = line.split("┆")
                col = parts[0].replace("│", "").strip()
                dtype = parts[1].replace("│", "").strip()
                if col and col not in {"column_name", "Column Name"}:
                    result[col] = dtype
        return result

    try:
        if hasattr(schema, "to_name_and_dtype_list"):
            return {n: str(d) for n, d in schema.to_name_and_dtype_list()}

        schema_list = list(schema)

        if schema_list and hasattr(schema_list[0], "name") and hasattr(schema_list[0], "dtype"):
            return {str(f.name): str(f.dtype) for f in schema_list}

        if isinstance(schema, dict):
            return {str(k): str(v) for k, v in schema.items()}

        if hasattr(schema, "dtypes"):
            return {str(k): str(v) for k, v in dict(schema.dtypes).items()}

        return {str(col): "Unknown" for col in schema_list}

    except Exception as exc:
        logger.warning("[%s] parse_schema failed: %s.", _ts(), exc)
        return {}


def _norm_col(name: str) -> str:
    """Normalise a column name for comparison: trimmed, unquoted, lower-cased."""
    return str(name).strip().strip('"`[]').lower()


def check_schema_match(
    generated_columns: List[str],
    destination_columns: List[str],
) -> bool:
    """Return ``True`` if the transfer can write into the destination table.

    The generated columns must be a **subset** of the destination columns
    (case-insensitively). We do NOT require an exact match:

    * Extra destination columns are fine — an auto-increment primary key, a
      column with a ``DEFAULT``, or any nullable column the source doesn't
      produce will simply take its default/NULL on insert. Requiring exact
      equality wrongly rejected these very common real-world tables.
    * Comparison is case-insensitive so ``Longitude`` matches ``longitude``.

    A generated column that does NOT exist in the destination is a real
    mismatch (there is nowhere to write it), so that still returns ``False``.
    """
    gen = {_norm_col(c) for c in generated_columns}
    dest = {_norm_col(c) for c in destination_columns}
    return gen.issubset(dest)


def _dtype_category(dtype: str) -> str:
    """Bucket a Daft dtype string into a coarse category, so we compare
    categories (Int32 vs Int64 is fine) rather than exact dtypes."""
    d = str(dtype).strip().lower()
    if d in ("", "unknown"):
        return "unknown"
    # Complex types (List[...], Struct, Map, …) → "other"; checked first because
    # "int" would otherwise match inside "List[Int64]".
    if "[" in d or any(t in d for t in ("list", "struct", "map", "tensor", "embedding", "fixedsize")):
        return "other"
    if "bool" in d:
        return "boolean"
    if any(t in d for t in ("timestamp", "datetime", "date", "time", "duration", "interval")):
        return "temporal"
    if any(t in d for t in ("utf8", "string", "str", "text", "char")):
        return "string"
    if any(t in d for t in ("binary", "bytes", "blob")):
        return "binary"
    if any(t in d for t in ("int", "float", "double", "decimal", "number", "numeric", "real")):
        return "numeric"
    return "other"


def _sql_type_for_daft_dtype(dtype: str, db_type: str) -> str:
    """Map a Daft dtype string to a portable SQL column type for CREATE TABLE."""
    d = str(dtype).strip().lower()
    is_pg = db_type in ("postgresql", "postgres")
    cat = _dtype_category(dtype)
    if cat == "numeric":
        if any(t in d for t in ("float", "double", "decimal", "real")):
            return "DOUBLE PRECISION" if is_pg else "DOUBLE"
        return "BIGINT"
    if cat == "boolean":
        return "BOOLEAN" if is_pg else "TINYINT(1)"
    if cat == "temporal":
        # Order matters: "timestamp" starts with "time", so match it first.
        if "timestamp" in d or "datetime" in d:
            return "TIMESTAMP" if is_pg else "DATETIME"
        if "date" in d:
            return "DATE"
        if "time" in d or "duration" in d or "interval" in d:
            return "TIME"
        return "TIMESTAMP" if is_pg else "DATETIME"
    if cat == "binary":
        return "BYTEA" if is_pg else "LONGBLOB"
    return "TEXT"  # string / other / unknown → TEXT (widest, safe default)


def _quote_sql_ident(ident: str, db_type: str) -> str:
    ident = str(ident).replace('"', "").replace("`", "")
    return f'"{ident}"' if db_type in ("postgresql", "postgres") else f"`{ident}`"


def _daft_schema_to_sql_ddl(table: str, schema: Dict[str, str], db_type: str) -> str:
    """Build a CREATE TABLE from a transformed-output ``{column: daft_dtype}`` schema."""
    if not schema:
        raise ValueError("cannot create a table from an empty schema")
    cols = ", ".join(
        f"{_quote_sql_ident(c, db_type)} {_sql_type_for_daft_dtype(t, db_type)}"
        for c, t in schema.items()
    )
    return f"CREATE TABLE {_quote_sql_ident(table, db_type)} ({cols})"


def _create_destination_table(conn_str: str, table: str, schema: Dict[str, str],
                              db_type: str, connect_args: Optional[dict] = None) -> str:
    """Create the destination table from the transformed schema. Returns the executed DDL."""
    from sqlalchemy import create_engine
    ddl = _daft_schema_to_sql_ddl(table, schema, db_type)
    engine = create_engine(conn_str, connect_args=connect_args or {})
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(ddl)
    finally:
        engine.dispose()
    return ddl


# Only these categories are compared strictly; a conflict between two of them
# (e.g. String into a numeric column) reliably fails at insert time. Other
# categories are skipped to avoid false positives.
_STRICT_DTYPE_CATEGORIES = {"numeric", "string", "temporal"}


def incompatible_column_types(
    generated_schema: Dict[str, str],
    destination_schema: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    """Return ``(column, generated_dtype, destination_dtype)`` for columns whose
    generated type category clearly conflicts with the destination column's.

    Complements :func:`check_schema_match` (name-only): a string-vs-int mismatch
    passes the name check but blows up at DB insert time with a confusing error.
    Only hard cross-category conflicts among {numeric, string, temporal} are
    reported; unknown/other/binary/boolean are skipped to stay conservative.
    """
    dest_norm = {_norm_col(k): v for k, v in destination_schema.items()}
    conflicts: List[Tuple[str, str, str]] = []
    for col, gen_dtype in generated_schema.items():
        dest_dtype = dest_norm.get(_norm_col(col))
        if dest_dtype is None:
            continue  # extra dest columns / unmatched — handled by name check
        gcat = _dtype_category(gen_dtype)
        dcat = _dtype_category(dest_dtype)
        if (
            gcat in _STRICT_DTYPE_CATEGORIES
            and dcat in _STRICT_DTYPE_CATEGORIES
            and gcat != dcat
        ):
            conflicts.append((col, gen_dtype, dest_dtype))
    return conflicts


def columns_missing_from_destination(
    generated_columns: List[str],
    destination_columns: List[str],
) -> List[str]:
    """Generated columns (original casing) that have no home in the destination."""
    dest = {_norm_col(c) for c in destination_columns}
    return [c for c in generated_columns if _norm_col(c) not in dest]


# Error-state helpers  (used by the code-generation pipeline)

def build_composite_feedback(coder_def: dict) -> Optional[str]:
    """
    Combines all active error signals into one feedback string that is fed
    back to the coder LLM on the next repair attempt.

    Execution stderr is always listed first — it is the ground-truth signal
    at runtime and must take priority over static-analysis feedback.
    """
    parts: List[str] = []

    if coder_def.get("execution_error"):
        parts.append(
            "EXECUTION ERROR (exact runtime stderr — trust over any other signal):\n"
            + coder_def["execution_error"].strip()
        )
    if coder_def.get("logical_review_feedback"):
        parts.append("LOGIC REVIEW FEEDBACK:\n" + coder_def["logical_review_feedback"].strip())

    # The shared validators set these flags to True and put the message in
    # code_validation_feedback — which the caller overwrites with our return value.
    # Read the message from wherever it actually is, or the repair runs blind.
    def _flag_message(flag_key: str) -> Optional[str]:
        flag = coder_def.get(flag_key)
        if not flag:
            return None
        if isinstance(flag, str):
            return flag
        pending = coder_def.get("code_validation_feedback")
        if isinstance(pending, str) and pending and pending != "APPROVED":
            return pending
        return None

    if (syntax_msg := _flag_message("syntax_error")):
        parts.append("SYNTAX ERROR:\n" + syntax_msg.strip())
    if (schema_msg := _flag_message("static_semantic_error")):
        # Don't repeat the identical message when both flags point at the same
        # pending code_validation_feedback text.
        if schema_msg.strip() not in {p.split(":\n", 1)[-1] for p in parts}:
            parts.append("SCHEMA VIOLATION:\n" + schema_msg.strip())

    return "\n\n---\n\n".join(parts) if parts else None


def reset_error_flags(coder_def: dict) -> None:
    """Clears all error flags in-place so each repair cycle starts clean."""
    coder_def.update({
        "execution_error": None,
        "logical_semantic_error": False,
        "syntax_error": False,
        "static_semantic_error": False,
        "logical_review_feedback": None,
        "execution_stdout": None,
        "execution_stderr": None,
    })
    logger.debug("[%s] Error flags reset.", _ts())


def _brief_error(detail, limit: int = 220) -> str:
    """One customer-readable line from an arbitrary error/traceback.

    Customers should never see a wall of technical output in chat — the full
    detail always goes to the server logs first. A Python traceback's LAST line
    is the actual error, so surface that; otherwise the first non-empty line.
    """
    text = str(detail or "").strip()
    if not text:
        return "an unknown error occurred"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line = lines[-1] if "Traceback (most recent call last)" in text else lines[0]
    if len(line) > limit:
        line = line[:limit].rstrip() + "…"
    return line


def _surface_logic_review_warning(final_state: dict) -> None:
    """Append an advisory logic-review finding to warnings in-place.

    In non-strict mode (the default) the reviewer records the finding but does
    not block, so without this it is silently dropped instead of reaching the
    user. The full reviewer paragraph goes to the logs; chat gets one sentence —
    customers don't need the reviewer's whole essay.
    """
    logic_feedback = final_state.get("coder_definition", {}).get("logical_review_feedback")
    if not logic_feedback:
        return
    logger.warning("[%s] Logic reviewer flagged (advisory): %s", _ts(), logic_feedback)
    first_sentence = re.split(r"(?<=[.!?])\s+", str(logic_feedback).strip())[0]
    if len(first_sentence) > 220:
        first_sentence = first_sentence[:220].rstrip() + "…"
    final_state["warnings"] = (final_state.get("warnings") or []) + [
        f"Advisory review note: {first_sentence}"
    ]

# Code-generation pipeline

def daft_code_generation_pipeline(state: ETLState) -> ETLState:
    """
    Runs the full Daft code-generation pipeline:

      1. Prompt sanitisation
      2. Pseudocode → retrieval → initial code generation
      3. Syntax validation  (+ one immediate repair if needed)
      4. Static-semantic / schema validation  (+ one immediate repair if needed)
      5. Execution + logic repair loop  (up to 5 attempts)

    On unrecoverable failure, the state is returned as-is with error flags
    set.  Callers check those flags rather than catching exceptions.
    """
    start = _ts()
    logger.info("[%s] >>>>>> Code generation pipeline START.", start)

    # ── 1. Sanitise prompt ─────
    log_step("1 / Sanitise prompt")
    raw_prompt = state["coder_definition"].get("user_prompt", "")
    state["coder_definition"]["user_prompt"] = sanitize_user_prompt_for_transformations(
        raw_prompt, daft_coder.coder_llm
    )
    logger.info("[%s] Prompt sanitised.", _ts())

    # ── 2. Initial code generation ───────
    log_step("2 / Initial code generation")
    state["coder_definition"].update(daft_pseudocode_node(state["coder_definition"]))

    # ── 2a. Empty source guard ─
    # Runs after pseudocode (so the filter condition is known) but before any
    # code generation, retrieval, or LLM calls.  If the filter matches 0 rows
    # in the source, the pipeline aborts here with a clear message instead of
    # generating and executing code that writes an empty table.
    log_step("2a / Empty source guard")

    # Forward source connection info into coder_definition so the guard can
    # reach the DB.  These keys live on the ETL state, not the coder state.
    # source_type  → guard skips automatically for non-DB sources (csv/parquet/json)
    # source_connect_args → SSL contexts etc. forwarded to SQLAlchemy
    state["coder_definition"]["data_source_location"] = state.get("source_connection_string", "")
    state["coder_definition"]["source_table"] = state.get("source_table", "")
    state["coder_definition"]["source_type"] = state.get("source_type", "")
    state["coder_definition"]["source_connect_args"] = state.get("source_connect_args") or {}

    guard_result = empty_source_guard_node(state["coder_definition"])
    state["coder_definition"].update(guard_result)

    if state["coder_definition"].get("empty_source"):
        abort_reason = state["coder_definition"].get(
            "pipeline_abort_reason",
            "Source dataset is empty after applying the filter condition.",
        )
        logger.warning("[%s] Pipeline aborted by empty source guard: %s", _ts(), abort_reason)
        state["error_message"] = abort_reason
        return state

    logger.info("[%s] Empty source guard passed — continuing to code generation.", _ts())

    state["coder_definition"].update(daft_retrieval_node(state["coder_definition"]))
    state["coder_definition"].update(daft_coder_node(state["coder_definition"]))
    logger.info("[%s] Initial code generated.", _ts())

    # ── 3. Syntax validation ───
    log_step("3 / Syntax validation")
    state["coder_definition"].update(syntactic_validator_node(state["coder_definition"]))

    if state["coder_definition"].get("syntax_error"):
        logger.warning("[%s] Syntax error on initial generation — one immediate repair.", _ts())
        state["coder_definition"]["code_validation_feedback"] = build_composite_feedback(state["coder_definition"])
        state["coder_definition"].update(daft_coder_node(state["coder_definition"]))
        state["coder_definition"].update(syntactic_validator_node(state["coder_definition"]))

        if state["coder_definition"].get("syntax_error"):
            logger.error("[%s] Syntax error persists after repair. Aborting.", _ts())
            return state
        logger.info("[%s] Syntax error resolved.", _ts())

    # ── 4. Static-semantic (schema) validation ────────────────────────────────
    log_step("4 / Static-semantic validation")
    state["coder_definition"].update(static_semantic_validator_node(state["coder_definition"]))

    if state["coder_definition"].get("static_semantic_error"):
        logger.warning("[%s] Schema violation on initial generation — one immediate repair.", _ts())
        state["coder_definition"]["code_validation_feedback"] = build_composite_feedback(state["coder_definition"])
        state["coder_definition"].update(daft_coder_node(state["coder_definition"]))
        state["coder_definition"].update(static_semantic_validator_node(state["coder_definition"]))

        if state["coder_definition"].get("static_semantic_error"):
            logger.error("[%s] Schema error persists after repair. Aborting.", _ts())
            return state
        logger.info("[%s] Schema error resolved.", _ts())

    # ── 5. Execution + logic repair loop ─
    log_step("5 / Execution + logic repair loop")
    max_retries = 5

    for attempt in range(max_retries):
        label = f"{attempt + 1}/{max_retries}"
        logger.info("[%s] Execution attempt %s.", _ts(), label)

        state = execution_agent_node_local(state)
        execution_failed = bool(state["coder_definition"].get("execution_error"))

        if not execution_failed:
            logger.info("[%s] Execution passed (attempt %s) — running logic validator.", _ts(), label)
            state["coder_definition"].update(logical_semantic_validator_node(state["coder_definition"]))

            if not state["coder_definition"].get("logical_semantic_error"):
                logger.info("[%s]  Code generation successful after attempt %s.", _ts(), label)
                break

            logger.warning(
                "[%s]  Logic validator flagged issues (attempt %s): %s",
                _ts(), label, state["coder_definition"].get("logical_review_feedback"),
            )
        else:
            logger.error(
                "[%s]  Execution failed (attempt %s): %s",
                _ts(), label,
                (state["coder_definition"].get("execution_error") or "").strip(),
            )

        if attempt == max_retries - 1:
            logger.error("[%s] All %d attempts exhausted — code generation failed.", _ts(), max_retries)
            return state

        # Build feedback, reset flags, regenerate
        state["coder_definition"]["code_validation_feedback"] = build_composite_feedback(state["coder_definition"])
        reset_error_flags(state["coder_definition"])

        logger.info("[%s] Starting repair cycle %s.", _ts(), label)
        state["coder_definition"].update(daft_coder_node(state["coder_definition"]))

        # Syntax check on repaired code
        state["coder_definition"].update(syntactic_validator_node(state["coder_definition"]))
        if state["coder_definition"].get("syntax_error"):
            logger.warning("[%s] Repair %s produced invalid syntax — retrying.", _ts(), label)
            state["coder_definition"]["code_validation_feedback"] = build_composite_feedback(state["coder_definition"])
            reset_error_flags(state["coder_definition"])
            continue

        # Schema check on repaired code
        state["coder_definition"].update(static_semantic_validator_node(state["coder_definition"]))
        if state["coder_definition"].get("static_semantic_error"):
            logger.warning("[%s] Repair %s produced schema violation — retrying.", _ts(), label)
            state["coder_definition"]["code_validation_feedback"] = build_composite_feedback(state["coder_definition"])
            reset_error_flags(state["coder_definition"])
            continue

    else:
        # Loop completed without a `break` — all attempts failed
        logger.error("[%s] Code generation failed — max retries (%d) exceeded.", _ts(), max_retries)
        return state

    logger.info("[%s] <<<<<< Code generation pipeline END (started %s).", _ts(), start)
    return state

# Injection-script assembly

def _make_io_config_code(provider: str, var_prefix: str) -> str:
    """
    Returns a Python function-definition string that builds an ``IOConfig``
    for ``provider`` at pod runtime using environment variables.

    ``var_prefix`` is ``"source"`` or ``"dest"`` so the generated helper
    has a unique name within the assembled script.
    """
    fn = f"_build_{var_prefix}_io_config"

    if provider == "gcp":
        if var_prefix == "dest":
            # Bare GCSConfig() authenticates as the ambient (SOURCE) SA; parquet
            # writes through this io_config, so the dest SA must be honored here.
            return (
                f"def {fn}():\n"
                f"    import os as _os\n"
                f"    _dest_sa = _os.environ.get('DEST_GCP_SA_JSON')\n"
                f"    if _dest_sa:\n"
                f"        return IOConfig(gcs=GCSConfig(credentials=_dest_sa))\n"
                f"    # Same-SA transfer: ambient GOOGLE_APPLICATION_CREDENTIALS.\n"
                f"    return IOConfig(gcs=GCSConfig())\n"
            )
        return (
            f"def {fn}():\n"
            f"    # GOOGLE_APPLICATION_CREDENTIALS mounted by Kubernetes secret.\n"
            f"    return IOConfig(gcs=GCSConfig())\n"
        )
    # Dest helpers prefer DEST_-prefixed creds (fallback: shared name) so a
    # cross-account same-provider write doesn't reuse the source account.
    d = "DEST_" if var_prefix == "dest" else ""
    if provider == "aws":
        return (
            f"def {fn}():\n"
            f"    return IOConfig(s3=S3Config(\n"
            f"        key_id=os.environ.get('{d}AWS_ACCESS_KEY_ID') or os.environ['AWS_ACCESS_KEY_ID'],\n"
            f"        access_key=os.environ.get('{d}AWS_SECRET_ACCESS_KEY') or os.environ['AWS_SECRET_ACCESS_KEY'],\n"
            f"        region_name=os.environ.get('{d}AWS_REGION') or os.environ.get('AWS_REGION', 'us-east-1'),\n"
            f"    ))\n"
        )
    if provider == "azure":
        return (
            f"def {fn}():\n"
            f"    return IOConfig(azure=AzureConfig(\n"
            f"        storage_account=os.environ.get('{d}AZURE_STORAGE_ACCOUNT') or os.environ['AZURE_STORAGE_ACCOUNT'],\n"
            f"        access_key=os.environ.get('{d}AZURE_ACCESS_KEY') or os.environ.get('AZURE_ACCESS_KEY'),\n"
            f"        sas_token=os.environ.get('{d}AZURE_SAS_TOKEN') or os.environ.get('AZURE_SAS_TOKEN'),\n"
            f"        tenant_id=os.environ.get('AZURE_TENANT_ID'),\n"
            f"        client_id=os.environ.get('AZURE_CLIENT_ID'),\n"
            f"        client_secret=os.environ.get('AZURE_CLIENT_SECRET'),\n"
            f"    ))\n"
        )
    return f"def {fn}():\n    return IOConfig()\n"


def _make_cloud_upload_helper_code(provider: str) -> str:
    """Return source for a provider-generic ``_upload_file_to_cloud`` helper.

    The generated helper uploads a single local file to ``blob_path`` in
    ``bucket_name`` using credentials resolved from environment variables at pod
    runtime (the same env vars ``_make_io_config_code`` reads). Used by the
    single-file CSV/JSON cloud writers so they work for GCS, S3 and Azure — not
    just GCS. Parquet cloud writes go through Daft's native ``write_parquet`` and
    do not need this.
    """
    if provider == "gcp":
        return (
            "def _upload_file_to_cloud(local_path, bucket_name, blob_path):\n"
            "    import os, json as _json\n"
            "    from google.cloud import storage as _gcs\n"
            "    # Write as the destination connection's own SA when supplied, so a\n"
            "    # cloud->cloud transfer doesn't reuse the source SA (which need not\n"
            "    # have write access to the destination bucket). Falls back to the\n"
            "    # ambient GOOGLE_APPLICATION_CREDENTIALS identity when unset.\n"
            "    _dest_sa = os.environ.get('DEST_GCP_SA_JSON')\n"
            "    if _dest_sa:\n"
            "        _client = _gcs.Client.from_service_account_info(_json.loads(_dest_sa))\n"
            "    else:\n"
            "        _client = _gcs.Client()\n"
            "    _client.bucket(bucket_name).blob(blob_path).upload_from_filename(local_path)\n"
        )
    if provider == "aws":
        return (
            "def _upload_file_to_cloud(local_path, bucket_name, blob_path):\n"
            "    import os\n"
            "    import boto3\n"
            "    _s3 = boto3.client(\n"
            "        's3',\n"
            "        aws_access_key_id=os.environ.get('DEST_AWS_ACCESS_KEY_ID') or os.environ.get('AWS_ACCESS_KEY_ID'),\n"
            "        aws_secret_access_key=os.environ.get('DEST_AWS_SECRET_ACCESS_KEY') or os.environ.get('AWS_SECRET_ACCESS_KEY'),\n"
            "        region_name=os.environ.get('DEST_AWS_REGION') or os.environ.get('AWS_REGION', 'us-east-1'),\n"
            "    )\n"
            "    _s3.upload_file(local_path, bucket_name, blob_path)\n"
        )
    if provider == "azure":
        return (
            "def _upload_file_to_cloud(local_path, bucket_name, blob_path):\n"
            "    import os\n"
            "    from azure.storage.blob import ContainerClient\n"
            "    _account = os.environ.get('DEST_AZURE_STORAGE_ACCOUNT') or os.environ['AZURE_STORAGE_ACCOUNT']\n"
            "    _cred = (os.environ.get('DEST_AZURE_ACCESS_KEY') or os.environ.get('DEST_AZURE_SAS_TOKEN')\n"
            "             or os.environ.get('AZURE_ACCESS_KEY') or os.environ.get('AZURE_SAS_TOKEN'))\n"
            "    _cc = ContainerClient(\n"
            "        account_url=f'https://{_account}.blob.core.windows.net',\n"
            "        container_name=bucket_name,\n"
            "        credential=_cred,\n"
            "    )\n"
            "    with open(local_path, 'rb') as _f:\n"
            "        _cc.get_blob_client(blob_path).upload_blob(_f, overwrite=True)\n"
        )
    return (
        "def _upload_file_to_cloud(local_path, bucket_name, blob_path):\n"
        f"    raise ValueError('Unsupported cloud provider for upload: {provider}')\n"
    )


def _make_cloud_download_helper_code(provider: str) -> str:
    """Return source for a provider-generic ``_download_file_from_cloud`` helper.

    Returns ``True`` if the object existed and was downloaded, ``False`` if it does
    not exist. Needed by the single-object CSV/JSON writers so ``write_mode="append"``
    can genuinely append to what is already there (matching the SQL sink's append
    semantics) instead of silently replacing it.
    """
    if provider == "gcp":
        return (
            "def _download_file_from_cloud(local_path, bucket_name, blob_path):\n"
            "    import os, json as _json\n"
            "    from google.cloud import storage as _gcs\n"
            "    _dest_sa = os.environ.get('DEST_GCP_SA_JSON')\n"
            "    if _dest_sa:\n"
            "        _client = _gcs.Client.from_service_account_info(_json.loads(_dest_sa))\n"
            "    else:\n"
            "        _client = _gcs.Client()\n"
            "    _blob = _client.bucket(bucket_name).blob(blob_path)\n"
            "    if not _blob.exists():\n"
            "        return False\n"
            "    _blob.download_to_filename(local_path)\n"
            "    return True\n"
        )
    if provider == "aws":
        return (
            "def _download_file_from_cloud(local_path, bucket_name, blob_path):\n"
            "    import os\n"
            "    import boto3\n"
            "    from botocore.exceptions import ClientError\n"
            "    _s3 = boto3.client(\n"
            "        's3',\n"
            "        aws_access_key_id=os.environ.get('DEST_AWS_ACCESS_KEY_ID') or os.environ.get('AWS_ACCESS_KEY_ID'),\n"
            "        aws_secret_access_key=os.environ.get('DEST_AWS_SECRET_ACCESS_KEY') or os.environ.get('AWS_SECRET_ACCESS_KEY'),\n"
            "        region_name=os.environ.get('DEST_AWS_REGION') or os.environ.get('AWS_REGION', 'us-east-1'),\n"
            "    )\n"
            "    try:\n"
            "        _s3.download_file(bucket_name, blob_path, local_path)\n"
            "        return True\n"
            "    except ClientError as _e:\n"
            "        if _e.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey'):\n"
            "            return False\n"
            "        raise\n"
        )
    if provider == "azure":
        return (
            "def _download_file_from_cloud(local_path, bucket_name, blob_path):\n"
            "    import os\n"
            "    from azure.storage.blob import ContainerClient\n"
            "    from azure.core.exceptions import ResourceNotFoundError\n"
            "    _account = os.environ.get('DEST_AZURE_STORAGE_ACCOUNT') or os.environ['AZURE_STORAGE_ACCOUNT']\n"
            "    _cred = (os.environ.get('DEST_AZURE_ACCESS_KEY') or os.environ.get('DEST_AZURE_SAS_TOKEN')\n"
            "             or os.environ.get('AZURE_ACCESS_KEY') or os.environ.get('AZURE_SAS_TOKEN'))\n"
            "    _cc = ContainerClient(\n"
            "        account_url=f'https://{_account}.blob.core.windows.net',\n"
            "        container_name=bucket_name,\n"
            "        credential=_cred,\n"
            "    )\n"
            "    try:\n"
            "        with open(local_path, 'wb') as _f:\n"
            "            _cc.get_blob_client(blob_path).download_blob().readinto(_f)\n"
            "        return True\n"
            "    except ResourceNotFoundError:\n"
            "        try:\n"
            "            os.remove(local_path)\n"
            "        except OSError:\n"
            "            pass\n"
            "        return False\n"
        )
    return (
        "def _download_file_from_cloud(local_path, bucket_name, blob_path):\n"
        f"    raise ValueError('Unsupported cloud provider for download: {provider}')\n"
    )


# Appending to a JSON-array object means reopening it and dropping the closing "]"
# so more elements can be written. Done by seeking from the end rather than parsing
# the file, so a large destination is never loaded into memory.
_JSON_APPEND_HELPER = (
    # json.dumps defaults to allow_nan=True, which emits bare NaN / Infinity /
    # -Infinity. Python reads those back, but they are NOT valid JSON and every
    # strict parser rejects them -- including Daft, which is what reads this same
    # file back on the next transfer. A CSV column with a blank numeric cell
    # (e.g. a missing Latitude) therefore produced a destination that looked
    # like a success and then failed the NEXT run with
    # "DaftError::External JSON deserialization error: InternalError(TapeError)
    # at character N ('N')" -- corruption written by one run, discovered by
    # another. Map non-finite floats to null instead.
    "def _json_safe(_obj):\n"
    "    \"\"\"Recursively replace non-finite floats with None (valid JSON null).\"\"\"\n"
    "    import math\n"
    "    if isinstance(_obj, dict):\n"
    "        return {_k: _json_safe(_v) for _k, _v in _obj.items()}\n"
    "    if isinstance(_obj, (list, tuple)):\n"
    "        return [_json_safe(_v) for _v in _obj]\n"
    "    try:\n"
    "        # Covers float and numpy scalars alike; bool/int are exact and str\n"
    "        # would raise, so they are excluded up front.\n"
    "        if not isinstance(_obj, (str, bytes, bool, int)) and (\n"
    "            math.isnan(_obj) or math.isinf(_obj)\n"
    "        ):\n"
    "            return None\n"
    "    except (TypeError, ValueError):\n"
    "        pass\n"
    "    return _obj\n"
    "\n"
    "def _json_open_for_append(path):\n"
    "    \"\"\"Truncate the closing ']' of a JSON array file so rows can be appended.\n"
    "    Returns True if the array already contained at least one element.\"\"\"\n"
    "    import os\n"
    "    with open(path, 'rb+') as _f:\n"
    "        _f.seek(0, os.SEEK_END)\n"
    "        _pos = _f.tell() - 1\n"
    "        while _pos >= 0:\n"
    "            _f.seek(_pos)\n"
    "            _ch = _f.read(1)\n"
    "            if _ch == b']':\n"
    "                _f.seek(_pos)\n"
    "                _f.truncate()\n"
    "                break\n"
    "            if not _ch.isspace():\n"
    "                raise ValueError('Destination JSON is not a JSON array; cannot append.')\n"
    "            _pos -= 1\n"
    "        else:\n"
    "            raise ValueError('Destination JSON is empty or malformed; cannot append.')\n"
    "    with open(path, 'rb') as _f:\n"
    "        _f.seek(0, os.SEEK_END)\n"
    "        _f.seek(max(0, _f.tell() - 64))\n"
    "        _tail = _f.read().strip()\n"
    "    return not _tail.endswith(b'[')\n"
)


def _read_sql_sink_source() -> str:
    """
    Returns the source of ``datasink/sql_sink.py`` so it can be inlined into the
    generated injection script.

    The Ray workers only receive the single ``sample_code.py`` file (mounted via
    ConfigMap, with ``PYTHONPATH=/home/ray``); the ``app`` package is NOT present
    there. Any ``from app.agents...sql_sink import ...`` therefore dies with
    ``ModuleNotFoundError: No module named 'app'``. Embedding the sink source
    keeps the script genuinely self-contained while ``sql_sink.py`` stays the
    single source of truth (read at generation time, on the app host).
    """
    sink_path = os.path.join(os.path.dirname(__file__), "datasink", "sql_sink.py")
    with open(sink_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# Destination formats written by materializing to pandas, then writing locally via
# pandas/pyarrow/fastavro and (for cloud) uploading. Overwrite-only — no append.
# tsv/orc need no extra deps; xlsx/xml/avro need openpyxl/lxml/fastavro in the DTA
# runtime image (app/data_transfer_docker/runtime-requirements.txt).
_PANDAS_DEST_FORMATS = ("tsv", "xlsx", "xml", "avro", "orc")
# Cloud formats whose write reuses the provider-generic _upload_file_to_cloud helper.
_CLOUD_UPLOAD_FORMATS = ("csv", "json") + _PANDAS_DEST_FORMATS
# pandas write statement per format; `{path}` is a Python expression for the target path.
_PANDAS_WRITE_EXPR = {
    "tsv":  '_pdf.to_csv({path}, sep="\\t", index=False)',
    "xlsx": '_pdf.to_excel({path}, index=False)',
    "xml":  '_pdf.to_xml({path}, index=False)',
    "orc":  '_pdf.to_orc({path}, index=False)',
    "avro": '_write_avro(_pdf, {path})',
}
# Top-level helper injected into the generated script only when writing Avro (no
# pandas/Daft native writer exists). Infers a nullable schema from the frame's dtypes.
_AVRO_WRITE_HELPER = (
    "def _write_avro(pdf, path):\n"
    "    import fastavro, math\n"
    "    def _t(dt):\n"
    "        return {'i': 'long', 'u': 'long', 'f': 'double', 'b': 'boolean'}.get(getattr(dt, 'kind', 'O'), 'string')\n"
    "    schema = {'type': 'record', 'name': 'AvalokaExport',\n"
    "              'fields': [{'name': str(c), 'type': ['null', _t(pdf[c].dtype)]} for c in pdf.columns]}\n"
    "    def _clean(v):\n"
    "        if v is None:\n"
    "            return None\n"
    "        try:\n"
    "            if isinstance(v, float) and math.isnan(v):\n"
    "                return None\n"
    "        except Exception:\n"
    "            pass\n"
    "        return v if isinstance(v, (int, float, bool)) else str(v)\n"
    "    records = [{str(k): _clean(val) for k, val in row.items()} for row in pdf.to_dict(orient='records')]\n"
    "    with open(path, 'wb') as _f:\n"
    "        fastavro.writer(_f, schema, records)\n"
)


_SCRIPT_LITERAL_BREAKERS = ('"', "'", "\\", "\n", "\r", "\x00")


def _unsafe_script_literals(final_state: dict) -> List[Tuple[str, str]]:
    """Values that would break out of the generated script's string literals."""
    checks = [
        ("source table", final_state.get("source_table")),
        ("destination table", final_state.get("destination_table")),
        ("source file", final_state.get("source_file")),
        ("destination file", final_state.get("destination_file")),
    ]
    for label, creds in (
        ("source cloud path", final_state.get("source_cloud_credentials")),
        ("destination cloud path", final_state.get("destination_cloud_credentials")),
    ):
        if isinstance(creds, cloud_storage_credentials):
            checks.append((label, creds.file_path))
            checks.append((label.replace("path", "bucket"), getattr(creds, "bucket_name", None)))
    return [
        (label, value) for label, value in checks
        if isinstance(value, str) and any(ch in value for ch in _SCRIPT_LITERAL_BREAKERS)
    ]


def make_injection_script(final_state: dict) -> str:
    """
    Assembles the complete, self-contained Python script that a Ray pod will
    execute to perform the actual data transfer.

    The script contains:
    - Cloud IO-config helpers  (credentials resolved via env vars at pod runtime)
    - ``read_source()``        (reads from the source)
    - ``transform_data(df)``   (the AI-generated transformation)
    - ``write_destination(df)`` (writes to the destination)
    - ``run_pipeline()``       (orchestrates the three functions above)

    Why is the write-body logic kept inline here?
    ---------------------------------------------
    The previous version extracted it into ``_build_write_body()``.
    That function was only ever called once, from this one place, and it
    returned a (string, list) tuple that needed to be unpacked and merged
    back in.  There was no reuse, no testability gain — just an extra
    indirection.  Keeping it inline means you can read the full script
    assembly top-to-bottom without jumping between functions.
    """
    log_step("make_injection_script")

    # ── Unpack state ───────────
    source_type: str = final_state.get("source_type", "").lower()
    dest_type: str = final_state.get("destination_type", "").lower()
    source_conn: str = final_state.get("source_connection_string", "")
    dest_conn: str = final_state.get("destination_connection_string", "")
    source_table: str = final_state.get("source_table", "source_table")
    dest_table: str = final_state.get("destination_table", "target_table")
    source_loc: str = final_state.get("source_file", "input_data")
    dest_loc: str = final_state.get("destination_file", "output_data.csv")
    source_cloud_creds: Optional[cloud_storage_credentials] = final_state.get("source_cloud_credentials")
    dest_cloud_creds: Optional[cloud_storage_credentials] = final_state.get("destination_cloud_credentials")
    sslmode: str = final_state.get("destination_sslmode", "")

    coder_def: dict = final_state.get("coder_definition", {})
    generated_transform_code: str = coder_def.get(
        "generated_code", "def transform_data(df):\n    return df"
    ).replace("def main(df):", "def transform_data(df):")

    # write_mode is threaded into the SQL sink and the Daft file writers below.
    write_mode: str = (final_state.get("write_mode") or "append").strip().lower()
    if write_mode not in {"append", "overwrite"}:
        write_mode = "append"
    generated_columns: List[str] = list(
        parse_schema(coder_def.get("generated_output_schema", "")).keys()
    )

    # Database URLs contain the password in plaintext. The generated script is
    # persisted to pipeline_runs/<job>/injection_script.py and uploaded as the Ray
    # working directory, so an inlined URL would leave credentials on disk and in
    # the cluster's object store. Hand them to the runtime as environment variables
    # instead — the same mechanism the cloud credentials already use. The runners
    # merge this into the container env / RayJob runtimeEnv.
    secret_env: Dict[str, str] = {}
    if source_type in _DB_TYPES | {"sql"} and source_conn:
        secret_env["DTA_SOURCE_CONN_STR"] = source_conn
    if dest_type in _DB_TYPES | {"sql"} and dest_conn:
        secret_env["DTA_DEST_CONN_STR"] = dest_conn
    final_state["runtime_secret_env"] = secret_env

    source_is_cloud = isinstance(source_cloud_creds, cloud_storage_credentials)
    dest_is_cloud = isinstance(dest_cloud_creds, cloud_storage_credentials)

    # ── Base imports ───────────
    imports: List[str] = [
        "import daft",
        "import os",
        "import pandas as pd",
        "from pathlib import Path",
        "from daft import col, lit",
    ]
    if source_is_cloud or dest_is_cloud:
        imports.append("from daft.io import IOConfig, GCSConfig, S3Config, AzureConfig")

    # ── Cloud IO-config helpers
    source_io_helper = (
        _make_io_config_code(source_cloud_creds.provider.lower(), "source")
        if source_is_cloud else ""
    )
    dest_io_helper = (
        _make_io_config_code(dest_cloud_creds.provider.lower(), "dest")
        if dest_is_cloud else ""
    )

    # ── Provider-generic single-file upload helper ──
    # Parquet cloud writes use Daft's native write_parquet + io_config and don't need
    # this. Emitted for CSV/JSON and the pandas-materialized formats (tsv/xlsx/xml/avro/orc).
    dest_upload_helper = (
        _make_cloud_upload_helper_code(dest_cloud_creds.provider.lower())
        if (dest_is_cloud and dest_type in _CLOUD_UPLOAD_FORMATS) else ""
    )
    # write_mode="append" on a single CSV/JSON object requires reading back what is
    # already there, so the new rows are added rather than replacing the object. The
    # pandas-materialized formats are overwrite-only, so they need no download helper.
    dest_download_helper = (
        _make_cloud_download_helper_code(dest_cloud_creds.provider.lower())
        if (dest_is_cloud and dest_type in ("csv", "json")) else ""
    )
    json_append_helper = _JSON_APPEND_HELPER if (dest_is_cloud and dest_type == "json") else ""
    # Avro needs a top-level writer helper (no pandas/Daft native writer).
    avro_helper = _AVRO_WRITE_HELPER if (dest_type == "avro") else ""

    # ── read_source() body ─────
    if source_is_cloud:
        cloud_uri = source_cloud_creds.get_cloud_uri(source_type)
        if source_type == "csv":
            read_body = (
                f'_io_config = _build_source_io_config()\n'
                f'    return daft.read_csv("{cloud_uri}", io_config=_io_config)'
            )
        elif source_type == "parquet":
            read_body = (
                f'_io_config = _build_source_io_config()\n'
                f'    return daft.read_parquet("{cloud_uri}", io_config=_io_config)'
            )
        elif source_type == "json":
            raise PipelineError("JSON is not supported as a cloud source. Use csv or parquet.")
        else:
            read_body = f'raise ValueError("Unsupported cloud source type: {source_type}")'
    else:
        if source_type == "csv":
            read_body = f'return daft.read_csv("{source_loc}")'
        elif source_type == "parquet":
            read_body = f'return daft.read_parquet("{source_loc}")'
        elif source_type == "json":
            read_body = f'return daft.read_json("{source_loc}")'
        elif source_type in _DB_TYPES | {"sql"}:
            # Read the URL from the environment — see `secret_env` above.
            read_body = (
                f'source_table = "{source_table}"\n'
                f'    return daft.read_sql(f"SELECT * FROM {{source_table}}", os.environ["DTA_SOURCE_CONN_STR"])'
            )
        else:
            read_body = f'raise ValueError("Unsupported source type: {source_type}")'

    # ── write_destination() body ─────────
    extra_imports: List[str] = []
    sink_source_block: str = ""

    if dest_is_cloud:
        cloud_dest_uri = dest_cloud_creds.get_cloud_uri(dest_type)

        if dest_type == "csv":
            # Stream rows into a single local CSV via Daft's iter_rows(), then upload
            # via the provider-generic _upload_file_to_cloud helper. iter_rows() is
            # Daft's driver-side streaming API — it drives the distributed execution
            # with lineage-based recovery and buffers only a few rows at a time.
            # (Do NOT ray.get() Daft partition refs directly: a lost object then
            # cannot be reconstructed → ObjectLostError on node churn.)
            # write_mode is honoured the same way the SQL sink honours it: "append"
            # adds rows to whatever is already in the object, "overwrite" replaces it.
            # Column names come from the frame's schema, not the first row, so a
            # zero-row result still produces a valid header instead of an empty file.
            write_body = (
                f'import tempfile, os, csv as _csv\n'
                f'    dest_uri = "{cloud_dest_uri}"\n'
                f'    bucket_name = "{dest_cloud_creds.bucket_name}"\n'
                f'    final_blob_path = "{dest_cloud_creds.file_path}"\n'
                f'    _write_mode = "{write_mode}"\n'
                f'    _fieldnames = list(df.column_names)\n'
                f'    _rows = 0\n'
                f'    with tempfile.TemporaryDirectory() as tmpdir:\n'
                f'        combined_path = os.path.join(tmpdir, "output.csv")\n'
                f'        _existing = False\n'
                f'        if _write_mode == "append":\n'
                f'            _existing = _download_file_from_cloud(combined_path, bucket_name, final_blob_path)\n'
                f'        if _existing:\n'
                f'            with open(combined_path, "r", newline="", encoding="utf-8") as _f:\n'
                f'                _header = next(_csv.reader(_f), None)\n'
                f'            if _header:\n'
                f'                if set(_header) != set(_fieldnames):\n'
                f'                    raise ValueError(\n'
                f'                        f"Cannot append to {{final_blob_path}}: existing columns "\n'
                f'                        f"{{_header}} do not match the transformed columns {{_fieldnames}}."\n'
                f'                    )\n'
                f'                _fieldnames = _header\n'
                f'        with open(combined_path, "a" if _existing else "w", newline="", encoding="utf-8") as outfile:\n'
                f'            _writer = _csv.DictWriter(outfile, fieldnames=_fieldnames)\n'
                f'            if not _existing:\n'
                f'                _writer.writeheader()\n'
                f'            for row in df.iter_rows():\n'
                f'                _writer.writerow(row)\n'
                f'                _rows += 1\n'
                f'        if _rows == 0 and _write_mode == "overwrite":\n'
                f'            raise ValueError(\n'
                f'                f"Refusing to overwrite {{final_blob_path}}: the transformation "\n'
                f'                f"produced 0 rows, which would replace the destination with an empty file."\n'
                f'            )\n'
                f'        if _rows == 0 and _existing:\n'
                f'            print(f"AVALOKA_RESULT rows_inserted=0 table={{final_blob_path}}")\n'
                f'            return {{"status": "success", "type": "cloud_csv", "uri": dest_uri, "rows_inserted": 0}}\n'
                f'        _upload_file_to_cloud(combined_path, bucket_name, final_blob_path)\n'
                f'    print(f"AVALOKA_RESULT rows_inserted={{_rows}} table={{final_blob_path}}")\n'
                f'    return {{"status": "success", "type": "cloud_csv", "uri": dest_uri, "rows_inserted": _rows}}'
            )

        elif dest_type == "parquet":
            write_body = (
                f'_io_config = _build_dest_io_config()\n'
                f'    df.write_parquet("{cloud_dest_uri}", write_mode="{write_mode}", io_config=_io_config)\n'
                f'    return {{"status": "success", "type": "cloud_parquet", "uri": "{cloud_dest_uri}"}}'
            )

        elif dest_type == "json":
            # Stream rows into a single JSON array via Daft's iter_rows(), then
            # upload via the provider-generic _upload_file_to_cloud helper (see the
            # CSV note above re: iter_rows vs. ray.get on Daft partition refs).
            # As with CSV: "append" adds elements to the existing JSON array (the
            # closing "]" is truncated by seeking from the end, so a large destination
            # is never parsed into memory), "overwrite" writes a fresh array.
            write_body = (
                f'import tempfile, os, json\n'
                f'    dest_uri = "{cloud_dest_uri}"\n'
                f'    bucket_name = "{dest_cloud_creds.bucket_name}"\n'
                f'    final_blob_path = "{dest_cloud_creds.file_path}"\n'
                f'    _write_mode = "{write_mode}"\n'
                f'    _rows = 0\n'
                f'    with tempfile.TemporaryDirectory() as tmpdir:\n'
                f'        combined_path = os.path.join(tmpdir, "output.json")\n'
                f'        _existing = False\n'
                f'        if _write_mode == "append":\n'
                f'            _existing = _download_file_from_cloud(combined_path, bucket_name, final_blob_path)\n'
                f'        first = True\n'
                f'        if _existing:\n'
                f'            first = not _json_open_for_append(combined_path)\n'
                f'        with open(combined_path, "a" if _existing else "w", encoding="utf-8") as outfile:\n'
                f'            if not _existing:\n'
                f'                outfile.write("[")\n'
                f'            for row in df.iter_rows():\n'
                # allow_nan=False turns any non-finite value _json_safe somehow
                # missed into an immediate, obvious failure rather than silently
                # writing a file that only breaks on a later run.
                f'                outfile.write(("" if first else ",") + json.dumps(_json_safe(row), default=str, allow_nan=False))\n'
                f'                first = False\n'
                f'                _rows += 1\n'
                f'            outfile.write("]")\n'
                f'        if _rows == 0 and _write_mode == "overwrite":\n'
                f'            raise ValueError(\n'
                f'                f"Refusing to overwrite {{final_blob_path}}: the transformation "\n'
                f'                f"produced 0 rows, which would replace the destination with an empty file."\n'
                f'            )\n'
                f'        if _rows == 0 and _existing:\n'
                f'            print(f"AVALOKA_RESULT rows_inserted=0 table={{final_blob_path}}")\n'
                f'            return {{"status": "success", "type": "cloud_json", "uri": dest_uri, "rows_inserted": 0}}\n'
                f'        _upload_file_to_cloud(combined_path, bucket_name, final_blob_path)\n'
                f'    print(f"AVALOKA_RESULT rows_inserted={{_rows}} table={{final_blob_path}}")\n'
                f'    return {{"status": "success", "type": "cloud_json", "uri": dest_uri, "rows_inserted": _rows}}'
            )

        elif dest_type in _PANDAS_DEST_FORMATS:
            # Materialize to pandas, write to a local temp file, then upload via the
            # provider-generic helper. Overwrite-only (these formats have no append).
            _pw = _PANDAS_WRITE_EXPR[dest_type].format(path="combined_path")
            write_body = (
                f'import tempfile, os\n'
                f'    dest_uri = "{cloud_dest_uri}"\n'
                f'    bucket_name = "{dest_cloud_creds.bucket_name}"\n'
                f'    final_blob_path = "{dest_cloud_creds.file_path}"\n'
                f'    _pdf = df.to_pandas()\n'
                f'    _rows = len(_pdf)\n'
                f'    if _rows == 0:\n'
                f'        raise ValueError(f"Refusing to write {{final_blob_path}}: the transformation produced 0 rows.")\n'
                f'    with tempfile.TemporaryDirectory() as tmpdir:\n'
                f'        combined_path = os.path.join(tmpdir, "output.{dest_type}")\n'
                f'        {_pw}\n'
                f'        _upload_file_to_cloud(combined_path, bucket_name, final_blob_path)\n'
                f'    print(f"AVALOKA_RESULT rows_inserted={{_rows}} table={{final_blob_path}}")\n'
                f'    return {{"status": "success", "type": "cloud_{dest_type}", "uri": dest_uri, "rows_inserted": _rows}}'
            )

        else:
            write_body = f'raise ValueError("Unsupported cloud destination type: {dest_type}")'

    else:
        # Local / database destination
        if dest_type in _DB_TYPES | {"sql"}:
            sink_class = "PostgresDataSink" if dest_type in {"postgresql", "postgres"} else "MySQLDataSink"
            # Inline the sink source instead of importing from app.* — the Ray
            # workers don't have the `app` package on their path. See
            # _read_sql_sink_source() for details.
            sink_source_block = _read_sql_sink_source()
            if sslmode and dest_type in {"postgresql", "postgres"}:
                connect_args_code = f"_connect_args = {{'sslmode': '{sslmode}'}}"
            elif sslmode and dest_type == "mysql":
                connect_args_code = "_connect_args = {'ssl': {'check_hostname': False, 'verify_mode': 'CERT_NONE'}}"
            else:
                connect_args_code = "_connect_args = {}"

            write_body = (
                f'conn_str = os.environ["DTA_DEST_CONN_STR"]\n'
                f'    table_name = "{dest_table}"\n'
                f'    {connect_args_code}\n'
                f'    destination = {sink_class}(conn_str, table_name, connect_args=_connect_args, '
                f'write_mode="{write_mode}", columns={generated_columns!r})\n'
                f'    df.write_sink(destination)\n'
                f'    return {{"status": "success", "type": "database", "table": "{dest_table}"}}'
            )

        elif dest_type == "csv":
            write_body = (
                f'df.write_csv("{dest_loc}", write_mode="{write_mode}")\n'
                f'    return {{"status": "success", "type": "csv", "file": "{dest_loc}"}}'
            )
        elif dest_type == "parquet":
            write_body = (
                f'df.write_parquet("{dest_loc}", write_mode="{write_mode}")\n'
                f'    return {{"status": "success", "type": "parquet", "file": "{dest_loc}"}}'
            )
        elif dest_type == "json":
            # to_json always truncates, so append must merge with the existing records
            # first — every other sink honors write_mode.
            write_body = (
                f'import os as _os, pandas as _pd\n'
                f'    _pdf = df.to_pandas()\n'
                f'    if "{write_mode}" == "append" and _os.path.exists("{dest_loc}") and _os.path.getsize("{dest_loc}"):\n'
                f'        _pdf = _pd.concat([_pd.read_json("{dest_loc}", orient="records"), _pdf], ignore_index=True)\n'
                f'    _pdf.to_json("{dest_loc}", orient="records")\n'
                f'    return {{"status": "success", "type": "json", "file": "{dest_loc}"}}'
            )
        elif dest_type in _PANDAS_DEST_FORMATS:
            _pw = _PANDAS_WRITE_EXPR[dest_type].format(path=f'"{dest_loc}"')
            write_body = (
                f'_pdf = df.to_pandas()\n'
                f'    {_pw}\n'
                f'    return {{"status": "success", "type": "{dest_type}", "file": "{dest_loc}"}}'
            )
        else:
            write_body = (
                f'print(df.show())\n'
                f'    return {{"status": "success", "type": "stdout"}}'
            )

    # ── Assemble script ────────
    imports.extend(extra_imports)
    imports_str = "\n".join(imports)
    io_helpers_block = (
        f"\n{source_io_helper}\n{dest_io_helper}\n{dest_upload_helper}"
        f"\n{dest_download_helper}\n{json_append_helper}\n{avro_helper}"
        if (source_io_helper or dest_io_helper or dest_upload_helper
            or dest_download_helper or json_append_helper or avro_helper) else ""
    )
    sink_block = f"\n# --- INLINED SQL SINK (datasink/sql_sink.py) ---\n{sink_source_block}\n# ----------------------------------------------\n" if sink_source_block else ""

    script = f"""{imports_str}
{io_helpers_block}{sink_block}
def read_source():
    {read_body}

# --- GENERATED TRANSFORMATION CODE ---
{generated_transform_code}
# -------------------------------------

def write_destination(df):
    {write_body}

def run_pipeline():
    print("Starting data pipeline execution...")
    df = read_source()
    df = transform_data(df)
    result = write_destination(df)
    print("Pipeline execution complete.")
    return result

if __name__ == "__main__":
    run_pipeline()
"""
    logger.info("[%s] Injection script assembled successfully.", _ts())
    return script

# Main entry point

def data_transfer_pipeline(
    user_prompt: str,
    source_type: str,
    destination_type: str,
    source_file: Optional[str] = None,
    source_table: Optional[str] = None,
    destination_file: Optional[str] = None,
    destination_table: Optional[str] = None,
    source_credentials: Optional[Union[db_credentials, cloud_storage_credentials, dict]] = None,
    destination_credentials: Optional[Union[db_credentials, cloud_storage_credentials, dict]] = None,
    write_mode: Optional[str] = "append",
    create_if_missing: bool = False,
) -> Tuple[ETLState, Optional[str]]:
    """
    Top-level orchestrator for the full data-transfer pipeline.

    Accepts ``db_credentials`` / ``cloud_storage_credentials`` Pydantic objects
    or plain dicts for backwards compatibility.

    Returns
    -------
    (final_state, injection_script)
        ``injection_script`` is ``None`` when code generation or schema checks fail.
    """
    from daft.io import IOConfig

    pipeline_start = _ts()
    # Strip the chat endpoint's appended "[Analysis context] …" metadata at the DTA
    # boundary so no downstream consumer (prompt sanitiser, pseudocode, guards) can
    # misread its fidelity/sample words as a transformation and drop rows.
    user_prompt = strip_runtime_analysis_context(user_prompt)
    logger.info(
    f"[{pipeline_start}] Data_Transfer_Pipeline START : "
    f"source_type={source_type} | dest_type={destination_type} | write_mode={write_mode}"
    )
    injection_script: Optional[str] = None
    warnings_list: List[str] = []

    # Reject unsupported source combinations before spending any schema/sample/
    # LLM work. The generated reader can't read a cloud JSON source, so without
    # this it would fail only at injection-script assembly — after the full
    # pipeline has already run.
    if source_type == "json" and isinstance(source_credentials, cloud_storage_credentials):
        logger.error("[%s] Cloud JSON source is not supported — rejecting up front.", _ts())
        return {
            "error_message": (
                "JSON is not supported as a cloud source. Use a CSV or Parquet source "
                "object instead (JSON is supported as a destination)."
            )
        }, None

    # Step 0: Pre-flight — fail fast if the coder LLM is not configured.
    # coder_llm is built at import time; the key may have been set since (e.g. a
    # late-loaded .env), so re-attempt the build before failing.
    if daft_coder.coder_llm is None:
        from dotenv import load_dotenv
        load_dotenv()
        daft_coder.coder_llm = daft_coder._build_coder_llm()
    if daft_coder.coder_llm is None:
        raise RuntimeError(
            "GROQ_API_KEY_CODING_AGENT is not set — the code-generation LLM could "
            "not be initialised. Set it in your environment (or .env) before running "
            "the data-transfer pipeline."
        )

    # Step 1: Build connection strings
    log_step("Step 1:Build connection strings")
    try:
        src_res = (
            build_connection_string(source_credentials, source_type)
            if source_credentials else (None, IOConfig())
        )
        dst_res = (
            build_connection_string(destination_credentials, destination_type)
            if destination_credentials else (None, IOConfig())
        )
    except PipelineError as exc:
        logger.error("[%s] Failed to build connection string: %s", _ts(), exc)
        return {"error_message": str(exc)}, None

    source_conn_str, source_conn_extra = src_res if isinstance(src_res, tuple) else (src_res, IOConfig())
    destination_conn_str, dest_conn_extra = dst_res if isinstance(dst_res, tuple) else (dst_res, IOConfig())

    source_io_config = source_conn_extra if isinstance(source_conn_extra, IOConfig) else IOConfig()
    source_connect_args: Dict[str, Any] = source_conn_extra if isinstance(source_conn_extra, dict) else {}
    dest_io_config = dest_conn_extra if isinstance(dest_conn_extra, IOConfig) else IOConfig()
    dest_connect_args: Dict[str, Any] = dest_conn_extra if isinstance(dest_conn_extra, dict) else {}

    # Step 2: Probe DB connections  (fail-fast before any LLM work) 
    log_step("Step 2:DB connection probes")
    try:
        if source_type in _DB_TYPES and source_conn_str:
            probe_db_connection(source_conn_str, "source", source_connect_args)
        if destination_type in _DB_TYPES and destination_conn_str:
            probe_db_connection(destination_conn_str, "destination", dest_connect_args)
    except PipelineError as exc:
        logger.error("[%s] Connection probe failed: %s", _ts(), exc)
        return {"error_message": str(exc)}, None

    # Step 3: Deduce source schema
    log_step("Step 3: Deduce source schema")

    try:
        source_table = (
            source_credentials.table
            if isinstance(source_credentials, db_credentials)
            else None                                      # cloud files have no table
            if isinstance(source_credentials, cloud_storage_credentials)
            else (source_credentials or {}).get("table")  # plain dict fallback
        )

        source_schema = deduce_schema(
            source_type,
            source_conn_str,
            table=source_table,
            io_config=source_io_config,
        )

        logger.info("[%s] Source schema ready.", _ts())

    except Exception as exc:
        logger.error("[%s] Could not deduce source schema: %s", _ts(), exc)
        return {"error_message": f"Connection Error: Could not read source schema. Detail: {exc}"}, None

    # The connection was already probed in Step 2, so for a DB source an empty
    # schema means the named table simply does not exist (mirror of the Case-3
    # guard on the destination). Fail fast with a clear message instead of
    # feeding a None schema into the code generator.
    if source_type in _DB_TYPES and source_table and not source_schema:
        logger.error("[%s] Source table '%s' does not exist.", _ts(), source_table)
        return {
            "error_message": (
                f"Source table `{source_table}` does not exist in the source database.\n\n"
                f"*Please check the table name (and schema) on the source connection.*"
            )
        }, None

    # Step 4: Deduce destination schema
    log_step("Step 4: Deduce destination schema")

    try:
        destination_table = (
            destination_credentials.table
            if isinstance(destination_credentials, db_credentials)
            else None                                           # cloud files have no table
            if isinstance(destination_credentials, cloud_storage_credentials)
            else (destination_credentials or {}).get("table")  # plain dict fallback
        )

        # Overwriting a cloud FILE replaces it wholesale, so whatever it holds
        # now has no bearing on compatibility -- and insisting on parsing it
        # first makes an unreadable destination permanently un-writable, with no
        # way out from the chat. (A database destination is different: overwrite
        # there still writes into an existing table, whose columns must match.)
        _dest_is_cloud_file = isinstance(destination_credentials, cloud_storage_credentials)
        if _dest_is_cloud_file and write_mode == "overwrite":
            destination_schema = {}
            logger.info(
                "[%s] Destination schema skipped — overwrite replaces the whole "
                "cloud file, so its current contents are irrelevant.", _ts(),
            )
        else:
            destination_schema = deduce_schema(
                destination_type,
                destination_conn_str,
                table=destination_table,
                io_config=dest_io_config,
            )

            logger.info("[%s] Destination schema ready.", _ts())

    except Exception as exc:
        # A read/permission error (not an absent table — deduce_schema returns
        # empty for that) means we cannot verify compatibility. Abort with a
        # clear retry message instead of silently skipping the schema check.
        logger.error("[%s] Could not read destination schema: %s", _ts(), exc)
        # Distinguish "the destination file is unreadable/corrupt" from a
        # genuinely transient fault. Telling someone to retry a malformed
        # destination just reproduces the same error forever -- they need to
        # know the file itself is the problem and how to get past it.
        _detail = str(exc)
        _malformed = any(
            token in _detail.lower()
            for token in ("deserialization", "tapeerror", "parse", "malformed", "invalid json")
        )
        if _malformed:
            _guidance = (
                "The destination exists but could not be parsed, so its schema "
                "can't be checked — it is most likely malformed, or not the format "
                "this transfer expects. Retrying as-is will fail the same way. "
                "Either delete/rename the destination file, or repeat the request "
                "with **overwrite** (which replaces the file wholesale and skips "
                "this check)."
            )
        else:
            _guidance = (
                "This looks like a transient connection or permissions issue "
                "(not a missing table) — please retry."
            )
        return {
            "error_message": (
                "Could not read the destination schema to verify compatibility. "
                f"{_guidance}\n\nDetail: {exc}"
            )
        }, None

    # Step 5: Fetch source sample data
    log_step("Step 5:Fetch sample data and load in pandas dataframe")
    try:
        df_daft = _read_daft_df(
            source_type,
            source_conn_str,
            table=source_table,
            io_config=source_io_config,
            limit=100,
            connect_args=source_connect_args,
        )

        input_sample_data = df_daft.to_pandas()

        logger.info(
            "[%s] Sample data fetched — %d rows, %d columns.",
            _ts(),
            len(input_sample_data),
            len(input_sample_data.columns),
        )

    except Exception as exc:
        logger.error("[%s] Failed to fetch sample data: %s", _ts(), exc)
        return {
            "error_message": f"Connection Error: Could not fetch sample data. Detail: {exc}"
        }, None

    # Step 6: Build ETL state
    log_step("Step 6:Build ETL state")

    coder_def = CodingAgentState(
        user_prompt=user_prompt,
        source_schema=source_schema,
        data_source_location=source_conn_str,
        generated_code="",
        syntax_error=False,
        static_semantic_error=False,
        logical_semantic_error=False,
        execution_error=False,
        code_validation_feedback=None,
        retry_count=0,
        input_sample_data=input_sample_data,
        generated_output_schema=None,
    )

    etl_state = ETLState(
        user_query=user_prompt,
        source_type=source_type,
        destination_type=destination_type,
        source_connection_string=source_conn_str,
        source_schema=source_schema,
        source_connect_args=source_connect_args,
        source_io_config=source_io_config,
        destination_connection_string=destination_conn_str,
        destination_schema=destination_schema,
        destination_connect_args=dest_connect_args,
        destination_io_config=dest_io_config,
        write_mode=write_mode,
        destination_schema_compatible=False,
        coder_definition=coder_def,
        messages=[],
        error_message=None,
        warnings=warnings_list or None,
    )

    # Wizard flag: create the destination DB table from the transformed schema if absent.
    etl_state["create_if_missing"] = bool(create_if_missing)

    if source_type in _DB_TYPES:
        etl_state["source_db_credentials"] = source_credentials
        etl_state["source_table"] = source_table
    elif source_type in _FILE_TYPES:
        etl_state["source_cloud_credentials"] = (
            source_credentials if isinstance(source_credentials, cloud_storage_credentials) else None
        )
        etl_state["source_file"] = source_file or source_conn_str

    if destination_type in _DB_TYPES:
        etl_state["destination_db_credentials"] = destination_credentials
        etl_state["destination_table"] = destination_table
    elif destination_type in _FILE_TYPES:
        etl_state["destination_cloud_credentials"] = (
            destination_credentials if isinstance(destination_credentials, cloud_storage_credentials) else None
        )
        etl_state["destination_file"] = destination_file or destination_conn_str

    logger.info("[%s] ETLState built.", _ts())

    #  Step 7: Code generation
    log_step("Step 7:Code generation pipeline")
    final_state = daft_code_generation_pipeline(etl_state)
    _surface_logic_review_warning(final_state)

    #  Step 8: Check generation success
    log_step("Step 8:Check code generation success")
    error_flags = ["syntax_error", "static_semantic_error", "execution_error", "logical_semantic_error"]
    generation_failed = any(final_state["coder_definition"].get(k) for k in error_flags)

    # A no-op stub / empty codegen passes every validator but ships untransformed
    # data, so treat it as a failure — via the flag or structural detection.
    gen_code = final_state["coder_definition"].get("generated_code")
    stub_used = bool(final_state["coder_definition"].get("stub_fallback_used")) or is_stub_code(gen_code)

    # Guard: the request references columns that don't exist in the SOURCE. The
    # pseudocode marks these as "[MISSING: <col>]". Without this, the coder emits a
    # no-op transform (returns the frame unchanged) and the pipeline ships the whole
    # untransformed table as a "success" — the requested filter/projection silently
    # dropped. That's almost always the wrong source table, so fail loudly and show
    # the columns the source actually has.
    _pseudocode = str(final_state["coder_definition"].get("coder_pseudocode") or "")
    missing_cols = sorted({
        c.strip().strip("`'\"")
        for grp in re.findall(r"\[MISSING:\s*([^\]]+)\]", _pseudocode, flags=re.IGNORECASE)
        for c in grp.split(",")
        if c.strip()
    })

    if missing_cols:
        _src_cols = list(parse_schema(final_state.get("source_schema") or {}).keys())
        _src_tbl = final_state.get("source_table")
        final_state["code_generated_successfully"] = False
        final_state["error_message"] = (
            "The request references column(s) that don't exist in the source"
            + (f" table `{_src_tbl}`" if _src_tbl else "")
            + f": `{missing_cols}`.\n\n"
            f"* **Available source columns:** `{_src_cols}`\n\n"
            "I won't transfer an unfiltered/unselected copy of the data. Point the "
            "transfer at a source (or table) that actually has these columns, or "
            "adjust the request to use columns that exist."
        )
        logger.error(
            "[%s] Aborting: requested column(s) %s absent from source table '%s' (have %s).",
            _ts(), missing_cols, _src_tbl, _src_cols,
        )
    elif generation_failed or stub_used:
        final_state["code_generated_successfully"] = False
        if stub_used and not generation_failed:
            final_state["error_message"] = (
                "Code Generation Failed: the transformation could not be generated "
                "and fell back to a no-op stub. Aborting instead of transferring "
                "untransformed data as a successful run. Please retry, and check "
                "that the coding LLM is configured and the request is well-formed."
            )
            logger.error("[%s] Code generation fell back to a no-op stub. Aborting.", _ts())
        else:
            # NOTE: use `or`, not .get(key, default): code_validation_feedback is
            # reset to None on every generation, so the key exists with value None
            # and a .get() default would never fire (-> "Code Generation Failed: None").
            # Fall through to the real execution error instead.
            cdef = final_state["coder_definition"]
            feedback = (
                cdef.get("code_validation_feedback")
                or cdef.get("execution_error")
                or cdef.get("execution_stderr")
                or "Unknown error."
            )
            # Full feedback (often a traceback) goes to the logs; chat gets one line.
            logger.error("[%s] Code generation failed. Feedback: %s", _ts(), feedback)
            final_state["error_message"] = (
                "The transformation could not be generated for this request.\n"
                f"Reason: {_brief_error(feedback)}\n\n"
                "Please rephrase the request or try again — the technical details "
                "have been recorded in the logs."
            )
    else:
        final_state["code_generated_successfully"] = True
        logger.info("[%s] Code generation succeeded. Proceeding to schema check.", _ts())

    #  Step 9: Schema compatibility check 
    log_step("Step 9:Schema compatibility check")

    if final_state.get("code_generated_successfully"):
        dest_schema_raw = final_state.get("destination_schema")
        dest_type = final_state.get("destination_type")
        gen_schema = parse_schema(final_state["coder_definition"].get("generated_output_schema", ""))
        dest_schema = parse_schema(dest_schema_raw)
        gen_cols = list(gen_schema.keys())
        dest_cols = list(dest_schema.keys())

        if not dest_schema_raw:
            # Step 2 already verified the destination connection, so for a DB
            # destination an unreadable schema means the named table is absent.
            dest_table_name = final_state.get("destination_table")
            if dest_type in _DB_TYPES and dest_table_name and final_state.get("create_if_missing"):
                # Wizard "create new table" path: build the table from the TRANSFORMED
                # output schema (columns + types), then proceed to load into it.
                try:
                    created_ddl = _create_destination_table(
                        final_state.get("destination_connection_string", ""),
                        dest_table_name,
                        gen_schema,
                        dest_type,
                        connect_args=final_state.get("destination_connect_args") or {},
                    )
                    final_state["destination_schema_compatible"] = True
                    final_state["created_destination_table"] = {
                        "table": dest_table_name, "schema": gen_schema, "ddl": created_ddl,
                    }
                    logger.info(
                        "[%s] Created destination table '%s' from transformed schema.",
                        _ts(), dest_table_name,
                    )
                except Exception as exc:
                    logger.error(
                        "[%s] Failed to create destination table '%s': %s",
                        _ts(), dest_table_name, exc,
                    )
                    final_state["destination_schema_compatible"] = False
                    final_state["error_message"] = (
                        f"Could not create the destination table `{dest_table_name}`.\n\n"
                        f"* **Generated Columns:** `{gen_cols}`\n\n"
                        f"Reason: {_brief_error(exc)}"
                    )
            elif dest_type in _DB_TYPES and dest_table_name:
                logger.error(
                    "[%s] Destination table '%s' does not exist.", _ts(), dest_table_name,
                )
                final_state["destination_schema_compatible"] = False
                final_state["error_message"] = (
                    f"Destination table `{dest_table_name}` does not exist in the "
                    f"destination database.\n\n"
                    f"* **Generated Columns:** `{gen_cols}`\n\n"
                    f"*Please create the table `{dest_table_name}` (with the columns "
                    f"above) in the destination database first, or say 'create a new table "
                    f"{dest_table_name}' and I'll build it for you.*"
                )
            else:
                logger.info(
                    "[%s] Destination schema empty (new file) — schema check bypassed.",
                    _ts(),
                )
                final_state["destination_schema_compatible"] = True

        elif not gen_cols and dest_cols:
            # Undetermined generated schema would pass the subset check vacuously
            # (∅ ⊆ anything) and ship unverified data — refuse instead.
            logger.error(
                "[%s]  Generated output schema is empty/undetermined — cannot "
                "verify destination compatibility. Aborting.", _ts(),
            )
            final_state["destination_schema_compatible"] = False
            final_state["error_message"] = (
                "Could not determine the transformed data's schema, so its "
                "compatibility with the destination could not be verified. "
                "Aborting instead of transferring unverified data. Please retry."
            )

        elif dest_type in _DB_TYPES | _FILE_TYPES:
            if check_schema_match(gen_cols, dest_cols):
                # Names match; also guard against type conflicts (e.g. String
                # into a numeric column) that fail at insert time.
                type_conflicts = incompatible_column_types(gen_schema, dest_schema)
                if type_conflicts:
                    logger.error(
                        "[%s]  SCHEMA TYPE MISMATCH — %s", _ts(), type_conflicts,
                    )
                    final_state["destination_schema_compatible"] = False
                    conflict_lines = "\n".join(
                        f"    * `{col}`: generated `{g}` vs destination `{d}`"
                        for col, g, d in type_conflicts
                    )
                    final_state["error_message"] = (
                        f"Schema Type Mismatch Detected!\n"
                        f"The column names match, but these column types are "
                        f"incompatible and would fail on insert:\n{conflict_lines}\n\n"
                        f"*Suggestion: cast these column(s) to the destination type "
                        f"in your prompt (e.g. convert text to a number), or align "
                        f"the destination table's column types.*"
                    )
                else:
                    logger.info("[%s]  Schema columns and types match.", _ts())
                    final_state["destination_schema_compatible"] = True
            else:
                logger.error(
                    "[%s]  SCHEMA MISMATCH — generated=%s | expected=%s",
                    _ts(), gen_cols, dest_cols,
                )
                missing = columns_missing_from_destination(gen_cols, dest_cols)
                final_state["destination_schema_compatible"] = False
                final_state["error_message"] = (
                    f"Schema Mismatch Detected!\n"
                    f"* **Generated Columns:** `{gen_cols}`\n"
                    f"* **Destination Columns:** `{dest_cols}`\n"
                    f"* **Columns not present in the destination:** `{missing}`\n\n"
                    f"*Suggestion: these column(s) don't exist in the destination table. "
                    f"Rename them in your prompt to match the destination, or add them to "
                    f"the destination table schema.*"
                )
        else:
            logger.warning("[%s] Unknown destination type '%s' — cannot verify schema.", _ts(), dest_type)
            final_state["destination_schema_compatible"] = False

    # Step 10: Assemble injection script 
    log_step("Step 10:Assemble injection script")

    if final_state.get("destination_schema_compatible"):
        # Table names and object paths are interpolated into the generated script's
        # string literals; a quote/backslash/newline would break out of them.
        unsafe = _unsafe_script_literals(final_state)
        if unsafe:
            final_state["destination_schema_compatible"] = False
            final_state["error_message"] = (
                "Transfer aborted: "
                + "; ".join(f"{what} contains unsupported characters: {val!r}" for what, val in unsafe)
                + ". Quotes, backslashes and newlines are not allowed in table names or file paths."
            )
            logger.error("[%s] %s", _ts(), final_state["error_message"])
            return final_state, None
        dest_creds = final_state.get("destination_db_credentials") or {}
        final_state["destination_sslmode"] = (
            (dest_creds.sslmode or "") if isinstance(dest_creds, db_credentials)
            else (dest_creds.get("sslmode") or "")
        )
        injection_script = make_injection_script(final_state)
        logger.info("[%s] Injection script ready.", _ts())
    else:
        logger.info("[%s] Skipping injection script — schema mismatch or generation failure.", _ts())

    logger.info(
        "[%s] Data_transfer_pipeline END (started %s) ", _ts(), pipeline_start,)
    return final_state, injection_script