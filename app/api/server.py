from __future__ import annotations

# Import torch FIRST, before any other heavy libraries.
# On Windows, torch's c10.dll uses static thread-local storage (TLS). Once the
# process has loaded many other DLLs (fastapi, langgraph, daft, chromadb, ...),
# the limited static-TLS slots can be exhausted, and a later/lazy `import torch`
# fails with: OSError [WinError 1114] DLL initialization routine failed (c10.dll).
# Loading torch up front claims its TLS slots early and avoids that crash.
import torch  # noqa: F401  (imported for side effect: early DLL init)

import datetime
import base64
import io
import csv
import os
import sys
import re
import json
import math
import uuid
import time
import hashlib
import random
import logging
import shutil
import concurrent.futures
import pandas as pd
import asyncio
from asyncio import sleep
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
import httpx
from botocore.exceptions import ClientError
from fastapi import (
    Depends,
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    status,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from starlette.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from fastapi.responses import FileResponse, JSONResponse
import asyncio
from app.api.integrations import (
    SUPPORTED_INTEGRATION_PROVIDERS,
    GitHubConnectIn,
    GitHubUpdateIn,
    REPO_RE,
    get_integration_connection,
    list_integration_connections,
    save_integration_connection,
    delete_integration_connection,
    validate_github_repo_access,
    resolve_github_config,
)

load_dotenv()

# -------------------------------------------------------------------
# App-wide logging (so logs from planner/validator/execution_agent show)
# -------------------------------------------------------------------
LOG_LEVEL = os.getenv("AVALOKA_LOG_LEVEL", "INFO").upper()
UPLOAD_CHUNK_SIZE = 1 << 20  # 1MB
MAX_UPLOAD_FILES = int(os.getenv("AVALOKA_MAX_UPLOAD_FILES", "10"))
MAX_UPLOAD_FILE_BYTES = int(os.getenv("AVALOKA_MAX_UPLOAD_FILE_BYTES", str(100 * 1024 * 1024)))  # 100MB
MAX_UPLOAD_TOTAL_BYTES = int(os.getenv("AVALOKA_MAX_UPLOAD_TOTAL_BYTES", str(200 * 1024 * 1024)))  # 200MB
# A multi-sheet Excel upload becomes one dataset per sheet; cap how many
# sheets a single workbook may expand into (largest sheets win).
MAX_XLSX_SHEETS = int(os.getenv("AVALOKA_MAX_XLSX_SHEETS", "10"))
# Windows consoles default to cp1252: any emoji in a log line or print()
# raises UnicodeEncodeError — logging swallows it as "--- Logging error ---"
# noise, but a bare print() inside a graph node kills the whole request.
# Reconfigure stdout/stderr to UTF-8 (replacing unencodable chars) so
# encoding can never take down a request.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    stream=sys.stdout,
    force=True,  # important: overrides uvicorn/default handlers
)

# # Ensure our package loggers propagate to root
# for _name in ("app", "app.agents", "app.graph", "app.api"):
#     _lg = logging.getLogger(_name)
#     _lg.setLevel(LOG_LEVEL)
#     _lg.propagate = True


# Ensure our package loggers propagate to root
for _name in ("app", "app.agents", "app.graph", "app.api"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(LOG_LEVEL)
    _lg.propagate = True

# Daft emits every per-operator tracing span through Python logging at INFO
# (~1,000 lines per upload). Each one costs GIL time and terminal I/O while
# sampling runs. Keep warnings and errors, drop the span noise.
for _noisy in ("tracing", "tracing.span"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# A bare "train" is not a training request ("which train line is busiest"):
# require a model as the object, or "model training" / "start training".
_TRAINING_TURN_RE = re.compile(
    r"\b(?:re-?)?train(?:ing)?\s+(?:(?:a|an|the|my|new)\s+)?(?:\w+\s+){0,2}models?\b"
    r"|\b(?:fit|build)\s+(?:(?:a|an|the|my|new)\s+)?(?:\w+\s+){0,2}models?\b"
    r"|\bmodel\s+training\b"
    r"|\bstart\s+training\b",
    re.I,
)


logging.getLogger(__name__).info("Avaloka logging configured (AVALOKA_LOG_LEVEL=%s)", LOG_LEVEL)

def _sanitize_training_plan_for_response(plan: Any) -> Any:
    if not isinstance(plan, dict):
        return plan

    sanitized = dict(plan)
    if sanitized.get("model_type") not in {"classification", "regression"}:
        sanitized["model_type"] = "classification"
    if not sanitized.get("model_name"):
        sanitized["model_name"] = "my_model"
    if not sanitized.get("model_description"):
        sanitized["model_description"] = "This is a model for..."
    if not sanitized.get("model_version"):
        sanitized["model_version"] = "v1.0"
    return sanitized


def _json_safe_payload(value: Any) -> Any:
    """Recursively convert non-JSON values like NaN/Inf/pandas NA to null."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_payload(item) for item in value]
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return _json_safe_payload(value.item())
        except Exception:
            pass

    return value


# storage/cache abstractions
from app.core.settings import Settings
from app.core.storage import ResourceNotFoundError, IBlobStore, GCSBlobStore, S3BlobStore
from app.agents.sampling_agent import sample_data_from_source, DEFAULT_SAMPLE_MAX_ROWS
from app.agents.sampling_agent_daft import COMPLEX_SOURCE_TYPES, sample_with_profiling
from file_handler.excel_connector import describe_sheet_choice, export_workbook_sheets_to_csv
from app.agents.sampling_async import sample_quick
from app.agents.profiling_agent import profile_full, usable_full_profile
from app.agents.sampling_persistence import persist_full_profile, load_portfolio, load_profile, upsert_profile
from app.agents.execution_agent import execution_agent_node_local, execution_agent_node_ray
from app.agents.execution_agent import RAY_NO_CONNECTION_REASON
from app.agents.planner import review_edited_code
from app.agents.validator import database_write_errors, run_code_validation
from app.agents.analysis_outcome import failure_detail, plain_failure_reason
from app.core.user_errors import looks_like_raw_error, plain_or, scrub_run_logs
from app.agents.claim_verifier import presentable_segments
from app.api import reply_segments
from app.core.celery_app import (
    AvalokaScheduler,
    AvalokaEntry,
    celery_app,
    _logs_from_final,
    _ray_job_from_final,
    _run_outcome_from_final,
    _run_status_from_final,
    _training_failure_from_final,
    scheduled_failure_message,
)
from app.agents.mta_v2.failure_diagnostics import (
    TRAINING_FAILURE_MESSAGE,
    build_training_failure,
    format_training_failure,
)
from app.core.task_metadata import scheduled_task_metadata
from app.core.db_sampling import (
    ANALYSIS_MAX_ROWS,
    DbSample,
    DbSampleError,
    draw_table_sample,
    sample_context_line,
    sample_scope,
)
from app.agents.drivers_route import (STATE_KEY as DRIVERS_STATE_KEY, answer_drivers_turn, inputs_from_session,
                                      is_drivers_question)
from app.core.db_pushdown import (
    parse_aggregate_intent,
    pushdown_output_table,
    render_pushdown_answer,
    run_pushdown,
)
from app.core import run_truth
from app.utils import convert_message_dicts_to_objects
from app.core import celery_results
from celery.result import AsyncResult
from celery.schedules import crontab
from redbeat.schedulers import get_redis

# NEW: shared services
from app.services import session_service, storage_service
from app.services import notification_service
from app.core import job_ledger, job_notes, long_jobs
from app.services.session_service import (
    _note_cache_failure,
    _mget_safe,
    _json_default,
    _jsonify,
    _k_session,
    _k_user_sessions,
    _k_thread_session,
    _k_session_threads,
    save_session,
    update_session,
    delete_session,
    get_session,
    refresh_session_ttl,
    find_session_by_dataset_for_user,
    db_credential_handles_for_user,
    db_credential_handle_for_user,
    bind_thread_session,
    get_thread_session,
    _user_datasets,
    _init_cache,
    _cache_unhealthy
)
from app.core.cache import ICache, RedisCache
from app.services.storage_service import (
    _strip_prefix,
    _store_and_key_from_uri,
    _key_from_uri,
    init_blob_store,
)

import app.api.config as api_config
from app.api.config import (
    logger,
    COOKIE_NAME,
    MCP_SERVER_URL,
    MCP_TIMEOUT,
    LANGGRAPH_API_URL,
    UPSTREAM_TIMEOUT,
    TMP_ROOT,
    SESSION_TTL_SECONDS,
    MAX_CONTEXT_TURNS,
    allow_origins,
    lifespan as config_lifespan,
)

from app.api.helpers import suggest_join_keys
from app.api.database_compat import (
    database_query_text,
    legacy_connect_fields,
    legacy_query_fields,
)
from app.api.legacy_database_connections import (
    LegacyConnectionResolutionError,
    resolve_legacy_database_credentials,
)
from app.api import chat_swarm, swarm_gate
from app.api.swarm_bridge import (
    build_swarm_state,
    consume_connection_event,
    database_connection_event,
)
from app.api.ui_contract import (
    api_capabilities as _api_capabilities,
    build_execution_context as _build_execution_context,
    build_execution_context_from_state as _build_execution_context_from_state,
)
from app.api.helpers import (
    THREAD_META,
    _filename_ext,
    _posix,
    _safe_rmtree,
    _new_id,
    _now_iso,
    _normalize_schema_for_state,
    _write_rows_to_csv,
    _ddl_from_schema,
    _schema_from_db_rows,
    _fallback_sample_csv,
    _datauri_csv_to_records,
    _compact_messages,
    _limit_messages,
    read_thread_msgs,
    _ensure_thread_local,
    hydrate_thread_history,
    persist_thread_history,
    delete_thread_history,
    _schema_to_columns,
    _alias_from_filename,
    _maybe_json_load,
    _pick_best_dataset_id,

)

from app.core.typed_secrets import redact_typed_secrets, seal_typed_secrets
from app.api.cloud_connections import (
    get_cloud_connection,
    connection_belongs_to_user,
    redact_connection,
    encrypt_secret,
    decrypt_secret,
    _store_from_connection_uri,
    normalize_storage_uri,
    detect_folder_table_type,
)
# GRAPH / GRAPH_READY are compiled once, authoritatively, in the try-block below.
# (app.api.graph_runtime is intentionally NOT imported here: it eagerly compiled a
#  second graph that this module always shadowed, which is exactly what left the
#  runtime graph without a checkpointer.)
from app.api.schemas import (
    ChatResponse,
    UploadResponse,
    ThreadCreateIn,
    ThreadOut,
    MessageCreateIn,
    DatasetListItem,
    DatasetPreviewResponse,
    RegisterExistingIn,
    BucketListResponse,
    BucketFolder,
    DatabaseConnectRequest,
    DatabaseChatRequest,
    DatabaseChatResponse,
    MultiUploadResponse, UploadedDatasetOut,
    # Asset Persistence retrieval models
    AssetResponse, CodeAsset, OutputAsset, VizAsset, JobAsset, AssetEntry,
    InsightRewriteOut, InsightRewriteIn,
    InsightFeedbackIn,
    InsightFeedbackOut,
    InsightChartIn,        
    InsightTurnIn,         
    InsightExplainIn,      
    InsightExplainOut, 
)
from app.services.persistence_service import (
    _persist_assets_background,
    generate_signed_url as _generate_asset_signed_url,
)
from app.agents.visualization_agent import build_visualization_config_from_sample
from app.agents.chart_explainer import explain_charts
from app.core.plan_limits import resolve_plan, limits_for, check_sizes, limit_error
from app.core.plan_limits import PLAN_FREE, PLAN_ENTERPRISE, PLAN_PROFESSIONAL


# Data transfer, model training and model inference are COST gates in
# app/core/editions.py; app/core/plan_gate.py decides them (planner.py in chat,
# the model endpoints below directly). Self-hosted and undeclared deployments
# are never gated. On a hosted deployment a failed plan lookup is refused as
# unavailable (503), never as an upgrade, the same policy as the planner.
from app.core.plan_gate import check_feature, upgrade_message, unavailable_message, FEATURE_CAPABILITY


def _plan_allows_premium(plan: Optional[str], feature: str = "model_training") -> bool:
    """True when ``plan`` includes ``feature`` on this deployment."""
    return check_feature(feature, plan=plan).allowed


async def _require_premium_plan(user_id: str, feature: str) -> None:
    """403 with an upgrade payload unless the user's plan includes the feature;
    503 "plan check unavailable" when a hosted plan lookup failed."""
    decision = await asyncio.to_thread(check_feature, feature, user_id=user_id)
    if decision.allowed:
        return
    if decision.unavailable:
        # The failed lookup is cached for _FAILURE_TTL_S (plan_limits), so that
        # is when a retry can first get a different answer.
        from app.core.plan_limits import _FAILURE_TTL_S
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {
                "reason": "plan_check_unavailable",
                "feature": feature,
                "message": unavailable_message(decision),
            },
            headers={"Retry-After": str(int(_FAILURE_TTL_S))},
        )
    raise HTTPException(
        status.HTTP_403_FORBIDDEN,
        {
            "reason": "plan_upgrade_required",
            "feature": feature,
            "plan": decision.plan,
            "allowed_plans": decision.allowed_plans(),
            "message": upgrade_message(decision),
        },
    )

import jwt
from jwt import PyJWKClient, InvalidTokenError

JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")  # used only for legacy HS256 tokens

_SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
_jwks_client = (
    PyJWKClient(f"{_SUPABASE_URL}/auth/v1/.well-known/jwks.json", cache_keys=True)
    if _SUPABASE_URL else None
)

# With HS256 an empty secret would verify any token an attacker signs with the
# empty string, so a missing secret must never allow authentication. Set this to
# start the server anyway (every request stays unauthenticated / 401).
ALLOW_INSECURE_AUTH = os.getenv("AVALOKA_ALLOW_INSECURE_AUTH", "").lower() in ("1", "true", "yes")

from app.core import local_auth as _local_auth


def _validate_auth_config() -> None:
    # AVALOKA_AUTH_MODE=local (app/core/local_auth.py) is refused outright on a
    # deployment that is not explicitly self-hosted: raise, even with
    # AVALOKA_ALLOW_INSECURE_AUTH, so a hosted install cannot boot with it.
    if _local_auth.load() is not None:
        logger.warning(
            "[auth] AVALOKA_AUTH_MODE=local: one local account, tokens signed by this "
            "server. For a self-hosted laptop install only; Supabase tokens are not accepted."
        )
        return
    if not JWT_SECRET:
        msg = (
            "SUPABASE_JWT_SECRET is not set. HS256 verification against an empty "
            "secret would accept forged tokens, so authentication is disabled."
        )
        if ALLOW_INSECURE_AUTH:
            logger.critical("%s All requests will be rejected as unauthenticated.", msg)
        else:
            raise RuntimeError(
                msg + " Set SUPABASE_JWT_SECRET, or set AVALOKA_ALLOW_INSECURE_AUTH=1 "
                "to start with authentication disabled."
            )

# -------------------------------------------------------------------
# Basic constants
# -------------------------------------------------------------------

settings = Settings()

ASSISTANT_ID = os.getenv("ASSISTANT_ID", "avaloka")
def _read_version() -> str:
    """The release number, from the VERSION file at the repo root.

    Hardcoding it here meant /version answered "1.5" for the whole 1.6 line:
    nothing sets APP_VERSION in the chart, so the literal default was what
    shipped, and it drifted the moment the branch did. Reading the file keeps
    one source of truth that a release bump actually moves. The env var still
    wins so a build can stamp something more specific.
    """
    env = (os.getenv("APP_VERSION") or "").strip()
    if env:
        return env
    # The OSS distribution used to carry its own line (1.0.x while the internal
    # branch was on 1.6). From 2.0 the public release and the internal branch
    # are the same version, so the OSS edition reads the VERSION file like
    # every other build. AVALOKA_OSS_VERSION remains an explicit override for
    # a public build that must identify itself differently.
    if (os.getenv("AVALOKA_EDITION") or "").strip().lower() in ("oss", "community"):
        oss = (os.getenv("AVALOKA_OSS_VERSION") or "").strip()
        if oss:
            return oss
    for candidate in (
        os.path.join(os.path.dirname(__file__), "..", "..", "VERSION"),
        "/app/VERSION",
    ):
        try:
            with open(candidate) as fh:
                value = fh.read().strip()
            if value:
                return value
        except OSError:
            continue
    return "unknown"


APP_VERSION = _read_version()

# Supported file formats. jsonl/ndjson joined on 2026-10-17: the CLI loader
# read them already and the sampler's JSON reader is line-delimited, so the
# only thing refusing them was this set.
SUPPORTED_UPLOAD_EXTS = {
    "csv", "tsv", "json", "jsonl", "ndjson", "xml", "parquet", "avro", "orc", "xls", "xlsx",
    "pdf",
    "jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp",
    "mp4", "mov", "mkv", "avi", "webm", "m4v",
}

#: Formats the product knows about but POST /api/upload does not take, each
#: with the one-line reason /api/capabilities shows. A deferral belongs here,
#: next to the accepted set, so the two cannot drift apart unnoticed. These
#: are table and dataset layouts Daft can read (0.7.25) that no entry point
#: serves yet; a multi-file layout cannot arrive as one multipart file.
DEFERRED_UPLOAD_FORMATS: Dict[str, str] = {
    "iceberg": "a table directory, not a file: register it through the existing-folder route, which detects Iceberg metadata",
    "delta": "a table directory with a transaction log; Daft reads it but no upload or folder route serves it yet",
    "hudi": "a table directory with a timeline; Daft reads it but no upload or folder route serves it yet",
    "lance": "a columnar dataset directory; Daft reads it but no upload or folder route serves it yet",
    "paimon": "a lake table directory; Daft reads it but no upload or folder route serves it yet",
    "mcap": "a robotics bag of typed channels, not a table; Daft reads it but the row contract (one row per message?) is undecided",
    "warc": "a web archive of records; Daft reads it but no entry point serves it and no benchmark truth exists",
    "webdataset": "a tar of samples; Daft reads it but no entry point serves it and no benchmark truth exists",
    "huggingface": "a hub dataset id, not a file: needs a source URI route, not an upload",
}


def upload_format_status() -> Dict[str, Dict[str, Any]]:
    """Per-format status for /api/capabilities.

    Every accepted extension is ``enabled``; ``statistics`` says whether the
    sampler's statistics contract (data_shape, column_statistics,
    data_quality, source) is produced for it -- the bounded complex formats
    return rows but no statistics. Every deferral carries its reason.
    """
    status: Dict[str, Dict[str, Any]] = {
        ext: {"status": "enabled", "statistics": ext not in COMPLEX_SOURCE_TYPES, "reason": None}
        for ext in SUPPORTED_UPLOAD_EXTS
    }
    for ext, reason in DEFERRED_UPLOAD_FORMATS.items():
        status[ext] = {"status": "deferred", "statistics": False, "reason": reason}
    return status


CONTENT_TYPE_TO_EXT: Dict[str, Optional[str]] = {
    "text/csv": "csv",
    "application/vnd.ms-excel": "xlsx",
    "text/tab-separated-values": "tsv",
    "application/json": "json",
    "application/x-ndjson": "ndjson",
    "application/jsonl": "jsonl",
    "application/xml": "xml",
    "text/xml": "xml",
    "application/pdf": "pdf",
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/tiff": "tiff",
    "image/bmp": "bmp",
    "image/webp": "webp",
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/webm": "webm",
    "video/x-msvideo": "avi",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    # parquet/avro/orc often come as octet-stream; rely on filename in those cases
    "application/octet-stream": None,
}

CACHE_CALL_TIMEOUT = float(os.getenv("CACHE_CALL_TIMEOUT", "0.6"))  # seconds per cache op

# # Per-dataset profiling + portfolio cache (populated on upload, read on preview)
# _profile_meta: Dict[str, Dict[str, Any]] = {}
# MAX_DB_SAMPLE_ROWS = 1000

# Per-dataset profiling + portfolio cache (populated on upload, read on preview)
_profile_meta: Dict[str, Dict[str, Any]] = {}

# Rows of each sample sent to the browser and kept in the Redis session.
# The FULL samples stay on disk (sample_input.csv), in-process (_profile_meta)
# and in Supabase (dataset_samples); only the copies shipped around are capped.
PREVIEW_RESPONSE_ROWS = int(os.getenv("AVALOKA_PREVIEW_RESPONSE_ROWS", "500"))

def _strip_blank_header_columns(path: Path) -> List[int]:
    """Remove columns whose header is blank AND whose every value is empty
    (e.g. a trailing comma in the header). Rewrites the file in place,
    preserving cell text verbatim. Returns dropped column indexes ([] if
    nothing was removed, the file isn't UTF-8, or a blank column holds data)."""
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            try:
                delim = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
            except Exception:
                delim = ","
            header = next(csv.reader(f, delimiter=delim), None)
    except (UnicodeDecodeError, OSError):
        return []
    if not header:
        return []

    blank = {i for i, h in enumerate(header) if not str(h).strip()}
    if not blank:
        return []
    keep = [i for i in range(len(header)) if i not in blank]

    tmp = path.with_name(path.name + ".clean")
    try:
        with open(path, "r", encoding="utf-8", newline="") as src, \
             open(tmp, "w", encoding="utf-8", newline="") as dst:
            reader = csv.reader(src, delimiter=delim)
            writer = csv.writer(dst, delimiter=delim, lineterminator="\n")
            for row in reader:
                # A "blank" column that actually holds data is real data: keep the file as-is.
                if any(i < len(row) and str(row[i]).strip() for i in blank):
                    tmp.unlink(missing_ok=True)
                    return []
                writer.writerow([row[i] for i in keep if i < len(row)])
        os.replace(tmp, path)
        return sorted(blank)
    except Exception:
        logger.warning("[upload] blank-column strip failed for %s; keeping original", path, exc_info=True)
        tmp.unlink(missing_ok=True)
        return []


from collections import Counter  # noqa: E402  (header-row detection below)

# Header-row detection. A file whose column names are not on row 1 (a report
# title, an "exported on ..." line, blank rows above the table) was read with
# row 1 as the header, so every column became "Unnamed: 0, 1, ..." and the
# charts and chat inherited those names.
HEADER_SCAN_ROWS = 50          # rows read to find the header
HEADER_MAX_OFFSET = 20         # the header must be within the first 20 rows
_NUMBERISH_RE = re.compile(
    r"^[\s$€£₹(+-]*\d[\d,.\s]*%?\)?$"                       # 1,200 / -5.5% / (30)
    r"|^\d{1,4}[-/.]\d{1,2}([-/.]\d{1,4})?([ T]\d{1,2}:\d{2}(:\d{2})?)?$"   # dates
)


def _sniff_delimiter_by_consistency(lines: List[str]) -> str:
    """The delimiter that gives the most rows the same number of fields.
    csv.Sniffer is thrown off by a title line above the table."""
    best, best_score = ",", (-1, -1)
    for d in (",", ";", "\t", "|"):
        widths = [len(r) for r in csv.reader(lines, delimiter=d) if len(r) > 1]
        if not widths:
            continue
        width, freq = Counter(widths).most_common(1)[0]
        if (freq, width) > best_score:
            best, best_score = d, (freq, width)
    return best


# pandas writes "Unnamed: 3" for a header cell that was empty. A sheet that was
# exported with row 1 as the header (a merged title row) arrives with those
# placeholders already in the file, so they count as empty here.
_UNNAMED_RE = re.compile(r"^Unnamed: \d+(?:_level_\d+)?$")


def _is_blank_cell(c: Optional[str]) -> bool:
    return not c or not c.strip() or bool(_UNNAMED_RE.match(c.strip()))


def _filled(row: List[str]) -> List[str]:
    return [c.strip() for c in row if not _is_blank_cell(c)]


def _looks_like_header(row: List[str], width: int) -> bool:
    cells = _filled(row)
    if len(cells) < max(2, int(0.8 * width + 0.5)):
        return False
    if len(set(c.lower() for c in cells)) != len(cells):
        return False                                  # header names are distinct
    text = [c for c in cells if not _NUMBERISH_RE.match(c)]
    return len(text) >= 0.6 * len(cells)              # mostly words, not values


def _find_header_row(rows: List[List[str]]) -> Optional[int]:
    """Index of the header row, or None when the file should be left alone.

    The table's width is the most common number of filled cells in the lower
    part of the scan (the data). The header is the first row that is about as
    wide, mostly text and has distinct names. Every row above it must be
    sparse (a title, a note, a blank line); otherwise the guess is not safe."""
    if len(rows) < 3:
        return None
    body = rows[len(rows) // 3:] or rows
    widths = [len(_filled(r)) for r in body if len(_filled(r)) > 1]
    if not widths:
        return None
    width = Counter(widths).most_common(1)[0][0]
    if width < 2:
        return None
    for i, row in enumerate(rows[: HEADER_MAX_OFFSET + 1]):
        if _looks_like_header(row, width):
            above = rows[:i]
            if all(len(_filled(r)) <= max(1, width // 2) for r in above):
                # ...and the next non-empty row must look like data of that width.
                nxt = next((r for r in rows[i + 1:] if _filled(r)), None)
                if nxt is not None and len(_filled(nxt)) >= max(2, int(0.5 * width)):
                    return i
            return None
    return None


# The Excel reader names every column column_1..column_N when it cannot find
# the header itself (a merged note row above it, e.g. "a" or "SE-01 is an
# active ..." spanning A1:M1). The real names are then one of the data rows.
_GENERATED_HEADER_RE = re.compile(r"^column_\d+$")


def _fix_header_row(path: Path, generated_header: bool = False) -> Optional[Dict[str, Any]]:
    """Make the real header row the first row of a CSV/TSV, in place.

    Also names header cells that are blank but have data under them
    (column_N), which pandas would call "Unnamed: N" -- e.g. the index column
    written by DataFrame.to_csv, or a merged cell in an Excel export.

    generated_header: the first row is column_1..column_N made up by the Excel
    reader (it said "No header row was found"). The real header is looked for
    in the rows below it; row numbers in the result are then sheet rows.

    Returns {"header_row": n (1-based), "skipped": [...], "named": [...]} when the
    file was changed, plus "replaced_generated": True when a made-up column_N header
    row was replaced; else None. Never raises."""
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            head_lines = []
            for _ in range(HEADER_SCAN_ROWS):
                line = f.readline()
                if not line:
                    break
                head_lines.append(line)
    except (UnicodeDecodeError, OSError):
        return None
    if not head_lines:
        return None

    try:
        delim = _sniff_delimiter_by_consistency(head_lines)
        rows = list(csv.reader(head_lines, delimiter=delim))
        replaced_generated = False
        if (generated_header and len(rows) > 1 and len(rows[0]) > 1
                and all(_GENERATED_HEADER_RE.match((c or "").strip()) for c in rows[0])):
            found = _find_header_row(rows[1:])
            if found is not None:
                h = found + 1
                replaced_generated = True
            else:
                h = 0
        else:
            h = _find_header_row(rows) or 0
    except Exception:
        logger.warning("[upload] header-row scan failed for %s", path, exc_info=True)
        return None
    header = rows[h] if h < len(rows) else []
    blank_named = [i for i, c in enumerate(header) if _is_blank_cell(c)]
    if h == 0 and not blank_named:
        return None

    tmp = path.with_name(path.name + ".hdr")
    named: List[int] = []
    try:
        # Blank header cells: only name the ones with data below them; fully
        # empty columns are left blank for _strip_blank_header_columns to drop.
        has_data = set()
        if blank_named:
            with open(path, "r", encoding="utf-8-sig", newline="") as src:
                for n, row in enumerate(csv.reader(src, delimiter=delim)):
                    if n <= h:
                        continue
                    for i in blank_named:
                        if i < len(row) and row[i].strip():
                            has_data.add(i)
                    if len(has_data) == len(blank_named):
                        break
        new_header = list(header)
        cleared = False
        for i in blank_named:
            if i in has_data:
                new_header[i] = f"column_{i + 1}"
                named.append(i + 1)
            elif new_header[i].strip():
                new_header[i] = ""        # an "Unnamed: N" over an empty column:
                cleared = True            # blank it so the strip step drops it
        if h == 0 and not named and not cleared:
            return None

        with open(path, "r", encoding="utf-8-sig", newline="") as src, \
             open(tmp, "w", encoding="utf-8", newline="") as dst:
            reader = csv.reader(src, delimiter=delim)
            writer = csv.writer(dst, delimiter=delim, lineterminator="\n")
            for n, row in enumerate(reader):
                if n < h:
                    continue
                writer.writerow(new_header if n == h else row)
        os.replace(tmp, path)
    except Exception:
        logger.warning("[upload] header-row fix failed for %s; keeping original", path, exc_info=True)
        tmp.unlink(missing_ok=True)
        return None

    first = 1 if replaced_generated else 0          # the made-up row is not a sheet row
    skipped = [" ".join(_filled(r))[:80] for r in rows[first:h] if _filled(r)]
    result = {"header_row": h + 1 - first, "skipped": skipped, "named": named}
    if replaced_generated:                          # only present when the made-up header was replaced
        result["replaced_generated"] = True
    return result


def _cap_rows(rows: Any, n: int = PREVIEW_RESPONSE_ROWS) -> Any:
       
    return rows[:n] if isinstance(rows, list) else rows


def _cap_portfolio(portfolio: Any, n: int = PREVIEW_RESPONSE_ROWS) -> Any:
    if not isinstance(portfolio, dict):
        return portfolio
    return {k: (v[:n] if isinstance(v, list) else v) for k, v in portfolio.items()}


# Rows read from a database table for analysis; see db_sampling.ANALYSIS_MAX_ROWS.
MAX_DB_SAMPLE_ROWS = ANALYSIS_MAX_ROWS
SMALL_FILE_PERSISTENCE_MAX_WORKERS = int(os.getenv("AVALOKA_SMALL_FILE_PERSISTENCE_MAX_WORKERS", "2"))
SMALL_FILE_PERSISTENCE_MAX_INFLIGHT = int(os.getenv("AVALOKA_SMALL_FILE_PERSISTENCE_MAX_INFLIGHT", "2"))

small_file_persistence_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
small_file_persistence_semaphore: Optional[asyncio.Semaphore] = None
small_file_persistence_inflight: set[str] = set()

DATASET_SESSION_SNAPSHOT_KEYS = (
    "work_local_input",
    "active_data_source_location",
    "active_data_source_location_cloud",
    "active_data_source_location_local",
    "uploaded_csv_columns",
    "uploaded_csv_preview",
    "schema",
    "ddl_schema",
    "file_size_bytes",
    "file_size_mb",
    "complex_structure",
)

LATEST_OUTPUT_SESSION_KEYS = (
    "latest_output_location",
    "latest_output_location_local",
    "latest_output_columns",
    "latest_output_row_count",
    "latest_output_created_at",
    "latest_output_is_trainable",
)


def _truthy_session_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


_RESET_DATASET_RE = re.compile(
    r"\b(?:reset|restore)\s+(?:the\s+)?(?:original\s+)?dataset\b"
    r"|\buse\s+(?:the\s+)?original\s+dataset\b"
    r"|\bgo\s+back\s+to\s+(?:the\s+)?original\s+(?:dataset|data)\b",
    re.IGNORECASE,
)


def _snapshot_original_dataset_session(sess: Dict[str, Any]) -> bool:
    if sess.get("original_dataset_snapshot"):
        return False

    sess["original_dataset_snapshot"] = _jsonify({
        key: sess.get(key)
        for key in DATASET_SESSION_SNAPSHOT_KEYS
    })
    return True


def _restore_original_dataset_session(sess: Dict[str, Any]) -> bool:
    snapshot = _maybe_json_load(sess.get("original_dataset_snapshot")) or {}
    # No snapshot means nothing to restore; falling through would wipe the
    # session's dataset keys via the pop() branch below.
    if not isinstance(snapshot, dict) or not snapshot:
        return False

    changed = False
    for key in DATASET_SESSION_SNAPSHOT_KEYS:
        if key in snapshot and snapshot[key] is not None:
            if sess.get(key) != snapshot[key]:
                sess[key] = snapshot[key]
                changed = True
        elif key in sess:
            sess.pop(key, None)
            changed = True

    if _truthy_session_value(sess.get("data_source_was_modified")):
        sess["data_source_was_modified"] = False
        changed = True

    if "original_dataset_snapshot" in sess:
        sess.pop("original_dataset_snapshot", None)
        changed = True

    return changed


def _merge_session_with_drops(
    cur: Dict[str, Any],
    sess: Dict[str, Any],
    baseline: Dict[str, Any],
    drop: Iterable[str] = (),
) -> None:
    """Merge a turn's session copy into the stored one, deletions included.

    ``cur.update(sess)`` adds and overwrites but never deletes, so a key the
    turn popped from ``sess`` (a resolved clarification, cleared training
    state, a restored dataset's keys) survived in the stored session and was
    read back next turn. A key in ``baseline`` (the session as the turn read
    it) that ``sess`` no longer has is dropped from ``cur`` while ``cur``
    still holds the value the turn read, so a value written meanwhile by
    another writer is kept. Keys in ``drop`` that ``sess`` lacks are dropped
    whatever ``cur`` holds.

    The deferred-turn keys (``_PENDING_TURN_KEYS``) are never merged: only
    their own mutators write them. The turn's copy was read before its running
    record existed, so merging it back restored the previous turn's record, and
    the poll for this turn answered "superseded" while it was finishing.
    """
    cur.update({k: v for k, v in sess.items() if k not in _PENDING_TURN_KEYS})
    for key, old in baseline.items():
        if key in _PENDING_TURN_KEYS:
            continue
        if key not in sess and key in cur and cur[key] == old:
            cur.pop(key, None)
    for key in drop:
        if key in _PENDING_TURN_KEYS:
            continue
        if key not in sess:
            cur.pop(key, None)


def _training_finished_successfully(final_state: Dict[str, Any]) -> bool:
    if not final_state.get("training_completed"):
        return False
    result = final_state.get("training_result")
    if not isinstance(result, dict):
        return False
    if result.get("error") or result.get("execution_error"):
        return False
    return bool(result.get("mlflow_run_id"))


def _replan_fresh_state() -> Dict[str, Any]:
    from app.agents.replan import fresh_state

    return fresh_state()


def _friendly_turn_error(exc: BaseException) -> str:
    """Map an unhandled turn failure to a specific, user-safe message.
    The full traceback is logged; the user gets a reason, never a stack trace."""
    text = str(exc).lower()
    try:
        import groq as _groq
    except Exception:
        _groq = None

    model_gone = (
        (_groq and isinstance(exc, getattr(_groq, "NotFoundError", ())))
        or "model_not_found" in text
        or "does not exist or you do not have access" in text
    )
    if model_gone:
        return ("⚠️ The analysis model is temporarily unavailable — the configured AI model "
                "was changed or retired by the provider. This has been logged for the team; "
                "please try again shortly.")

    if (_groq and isinstance(exc, getattr(_groq, "AuthenticationError", ()))) or "invalid api key" in text:
        return ("⚠️ The analysis service rejected our credentials. This is a configuration issue "
                "on our side and has been logged for the team.")

    if (_groq and isinstance(exc, getattr(_groq, "RateLimitError", ()))) or "rate limit" in text:
        return "⚠️ The analysis service is busy right now. Please wait a moment and try again."

    return ("⚠️ Something went wrong while running this analysis. The error has been logged for "
            "the team — please try again, and contact support if it keeps happening.")

# -------------------------------------------------------------------
# Utility: LangGraph HTTP helpers
# -------------------------------------------------------------------

# ====== HTTP Client / Cache / Storage (globals bound in lifespan) ======
http_client: Optional[httpx.AsyncClient] = None
cache: Optional[ICache] = None
blob_store: Optional[IBlobStore] = None

async def append_thread_msg(thread_id: str, role: str, content: str) -> None:
    return


# =============== Graph wiring (with safe planner) ===============
try:
    here = Path(__file__).resolve()
    for c in [here.parent, here.parent.parent, here.parent.parent.parent]:
        p = str(c)
        if p not in sys.path:
            sys.path.append(p)

    from app.agents import planner as _planner_mod
    from app.agents.planner import (
        build_fidelity_prompt_message as _planner_build_fidelity_prompt,
        build_mode_switch_confirmation as _planner_build_mode_switch_confirmation,
        build_sample_switch_confirmation as _planner_build_sample_switch_confirmation,
        estimate_large_dataset_runtime_hint as _planner_estimated_runtime_hint,
        parse_fidelity_from_control_text as _planner_parse_fidelity_from_text,
        parse_selected_sample_from_control_text as _planner_parse_selected_sample_from_text,
        is_explicit_mode_switch_message as _planner_is_mode_switch_message,
    )

    _orig_plan = _planner_mod.plan_etl_job

    def _fallback_decide(user_text: str) -> dict:
        return {"ready_to_summarize": False, "ready_to_code": False,
                "ai_message": "I could not route that request because the planner failed. Please try again."}

    def _plan_etl_job_safe(state):
        msgs = state.get("messages", [])
        last = msgs[-1] if msgs else None
        user_text = last.content if isinstance(last, HumanMessage) else (getattr(last, "content", "") or "")

        try:
            new_state = _orig_plan(state)
        except Exception:
            decision = _fallback_decide(user_text)
            out = dict(state)
            if decision.get("ai_message"):
                out["messages"] = msgs + [_said(decision["ai_message"])]
            if decision.get("infrastructure_request"):
                out["infrastructure_request"] = decision["infrastructure_request"]
            out["ready_to_summarize"] = bool(decision.get("ready_to_summarize", False))
            out["ready_to_code"] = bool(decision.get("ready_to_code", False))
            return out

        if "ready_to_summarize" not in new_state:
            new_state["ready_to_summarize"] = False
        if "ready_to_code" not in new_state:
            new_state["ready_to_code"] = False
        return new_state

    _planner_mod.plan_etl_job = _plan_etl_job_safe

    from app.api.workflow import build_graph
    from langgraph.checkpoint.memory import MemorySaver

    # A process-wide checkpointer so config thread_id actually persists top-level
    # state across turns. (The internal coding subgraph is invoked without a
    # thread_id, so build_coding_graph deliberately does NOT inherit this. The
    # langgraph_app.py deployment is separate and lets the LangGraph API handle
    # persistence, so it compiles without one.)
    checkpointer = MemorySaver()
    GRAPH = build_graph(checkpointer=checkpointer).compile(checkpointer=checkpointer)
    GRAPH_READY = True
    logger.info("LangGraph build_graph loaded successfully with patched planner.")
except Exception as e:
    logger.warning("[warn] build_graph failed (%s); using echo fallback.", e)
    GRAPH_READY = False

    def _planner_build_fidelity_prompt(num_bytes: int) -> str:
        return "This is a large dataset. You are currently in `quick_sample` mode by default."

    def _planner_build_mode_switch_confirmation(mode: str, num_bytes: int) -> str:
        return f"Switched to mode {mode.replace('_', ' ')}."

    def _planner_build_sample_switch_confirmation(sample_name: str) -> str:
        return f"Switched to {sample_name} sample."

    def _planner_estimated_runtime_hint(num_bytes: int) -> str:
        return "roughly minutes to hours depending on query complexity"

    def _planner_parse_fidelity_from_text(text: str) -> Optional[str]:
        return None

    def _planner_parse_selected_sample_from_text(text: str) -> Optional[str]:
        return None

    def _planner_is_mode_switch_message(text: str) -> bool:
        return False

    def _echo_handler(state: Dict[str, Any]) -> Dict[str, Any]:
        msgs = state.get("messages") or []
        text = ""
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                text = m.content
                break
        out = (state.get("messages") or []) + [AIMessage(content=f"(fallback graph) you said: {text}")]
        return {"messages": out, "ready_to_summarize": False, "ready_to_code": False}

    GRAPH = RunnableLambda(_echo_handler)

# =============== Models ===============
# ChatResponse and UploadResponse are imported from app.api.schemas

def get_output_from_state(state, tmp_root: Optional[Path] = None):
    """
    Return (output_file_data, output_json) from the final graph state.
    - output_file_data: data-uri csv blob if available
    - output_json: list[dict] parsed from either data-uri, markdown table, or output_location csv
    """
    final_all = state.get("messages", []) or []
    ai_msg = next((m for m in reversed(final_all) if isinstance(m, AIMessage)), None)

    output_file_data = None
    output_json = None

    # Structured aggregate execution returns typed tables directly. Its
    # output_json is the first table for older clients, not a CSV with a
    # synthetic table marker.
    if state.get("structured_action_plan"):
        return _json_safe_payload(state.get("output_file_data")), _json_safe_payload(state.get("output_json"))

    # 1) Data-uri output
    candidate_output = state.get("output_file_data") or None
    candidate_output = _maybe_json_load(candidate_output)
    if isinstance(candidate_output, dict) and candidate_output.get("content"):
        output_file_data = candidate_output
        try:
            output_json = _datauri_csv_to_records(output_file_data["content"])
        except Exception:
            output_json = None

    # 2) Markdown table output (if model printed a table)
    if output_json is None and ai_msg and isinstance(ai_msg.content, str):
        blocks = re.findall(r"(?:^\|.*\|\s*\n?)+", ai_msg.content, flags=re.MULTILINE)
        if blocks:
            lines = [ln.strip() for ln in blocks[0].strip().splitlines()]
            if len(lines) >= 3:
                headers = [h.strip() for h in lines[0].strip("|").split("|")]
                data_lines = [ln for ln in lines[2:] if ln.startswith("|")]
                drop_first = (len(headers) > 0 and headers[0] in ("", "#", "index"))
                if drop_first:
                    headers = headers[1:]
                rows = []
                for ln in data_lines:
                    cells = [c.strip() for c in ln.strip("|").split("|")]
                    if drop_first and cells:
                        cells = cells[1:]
                    if len(cells) == len(headers):
                        rows.append(dict(zip(headers, cells)))
                output_json = rows or None

    # 3) Disk fallback: output_location CSV (this was previously unreachable)
    if output_json is None:
        out_path = state.get("output_location")
        if out_path:
            p = Path(out_path)
            if p.exists() and p.is_file():
                try:
                    df = pd.read_csv(p)
                    output_json = df.to_dict(orient="records")
                except Exception as e:
                    logger.warning("Failed to read output_location CSV (%s): %s", out_path, e)

    return _json_safe_payload(output_file_data), _json_safe_payload(output_json)


def _full_cloud_uri(sess: dict) -> str | None:
    folder_read_path = sess.get("folder_read_path")
    if folder_read_path:
        return folder_read_path

    base = (sess.get("data_source_location") or "").rstrip("/")
    key = (sess.get("object_name") or "").lstrip("/")
    if not base:
        return None
    if key and (base.endswith("/" + key) or base.endswith(key)):
        return base
    if key and "://" in base:
        return f"{base}/{key}"
    return base


# ---------------------------------------------------------------------------
# NEW: re-materialize local input files when TMP_ROOT was wiped / ingest ran
# in a different process. Called by send_message before every local (non-ray)
# execution. Rebuilds files AT THE SESSION'S EXISTING PATHS so the already-built
# state_in references stay valid.
# ---------------------------------------------------------------------------
async def _ensure_local_inputs(
    sess: Dict[str, Any],
    *,
    need_full: bool = False,
    rows_override: Optional[List[Dict[str, Any]]] = None,
) -> None:
    # 1) Ensure work_dir exists.
    work_dir_str = sess.get("work_dir")
    if work_dir_str:
        work_dir = Path(work_dir_str)
    else:
        work_dir = TMP_ROOT / (sess.get("session_id") or _new_id())
        sess["work_dir"] = str(work_dir)
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.warning("[ensure-local] could not create work_dir %s", work_dir, exc_info=True)

    rows = rows_override if isinstance(rows_override, list) else None

    # 2) Rebuild the sample CSV if it's gone (recreate at the SAME path).
    sample_path_str = sess.get("sample_local_input")
    sample_path = Path(sample_path_str) if sample_path_str else (work_dir / "sample_input.csv")
    if not sample_path.exists():
        if rows:
            try:
                _write_rows_to_csv(rows, sample_path)
                sess["sample_local_input"] = _posix(sample_path)
                logger.info(
                    "[ensure-local] rebuilt sample_input.csv (%d rows) -> %s",
                    len(rows), sample_path,
                )
            except Exception:
                logger.warning(
                    "[ensure-local] failed to rebuild sample CSV at %s", sample_path, exc_info=True
                )
        else:
            logger.info(
                "[ensure-local] sample CSV missing and no rows to rebuild it: %s", sample_path
            )

    # 3) Ensure a working input exists.
    work_input_str = sess.get("work_local_input")
    work_input = Path(work_input_str) if work_input_str else None
    # work_local_input may have been repointed at the sample or the streamed
    # head. For a sub-1 GB full run that is not the full file: fetch it again.
    is_stand_in = bool(work_input) and (
        work_input == sample_path
        or work_input.name in ("sample_input.csv", "streamed_head.csv")
    )
    small_file = int(sess.get("file_size_bytes") or 0) < LARGE_DATASET_THRESHOLD_BYTES
    work_input_ok = bool(
        work_input and work_input.exists()
        and not (need_full and is_stand_in and small_file)
    )

    # ENTIRE-local needs the full object; sample suffices otherwise.
    if need_full and not work_input_ok:
        storage_uri = sess.get("data_source_location") or ""
        object_name = sess.get("object_name")
        # db:// sources have no downloadable object; the sample is all we have.
        can_download = (
            isinstance(storage_uri, str)
            and "://" in storage_uri
            and not storage_uri.startswith("db://")
        )
        if can_download:
            ext = (sess.get("input_data_type") or "csv").lower()
            dest = work_input if (work_input and not is_stand_in) else (
                work_dir / f"input_{sess.get('dataset_id') or _new_id()}.{ext}"
            )
            try:
                conn_id = sess.get("connection_id")
                if conn_id:
                    conn = await get_cloud_connection(conn_id)
                    store_for_read, _ = await _store_from_connection_uri(storage_uri, conn)
                    key_for_read = (object_name or _key_from_uri(storage_uri) or "").lstrip("/")
                    await asyncio.to_thread(store_for_read.get_file, key_for_read, dest)
                else:
                    store_for_read, key_for_read = _store_and_key_from_uri(storage_uri, object_name)
                    await asyncio.to_thread(
                        store_for_read.get_file,
                        key_for_read or object_name or _key_from_uri(storage_uri),
                        dest,
                    )
                sess["work_local_input"] = _posix(dest)
                work_input, work_input_ok = dest, True
                logger.info("[ensure-local] re-downloaded full source for ENTIRE run -> %s", dest)
            except Exception:
                logger.warning(
                    "[ensure-local] failed to re-download full source from %s; "
                    "falling back to sample for this run", storage_uri, exc_info=True,
                )

    # 4) Last resort: point work_local_input at the sample so the executor has
    #    something to read instead of FileNotFoundError.
    if not work_input_ok and sample_path.exists():
        sess["work_local_input"] = _posix(sample_path)
        logger.info("[ensure-local] work_local_input falling back to sample CSV: %s", sample_path)

async def _ensure_dataset_local_sample(ds_sess: Dict[str, Any]) -> Optional[str]:
    """Guarantee readable local input for one dataset in a multi-dataset group.

    Rebuilds sample_input.csv from stored preview rows when TMP_ROOT was wiped,
    AND repoints work_local_input at that sample when the original full-input
    file (input_{dsid}.csv) is gone. Without the repoint, section 7 advertises a
    deleted path as csv_path and the join FileNotFounds (e.g. customers.csv).
    """
    work_dir_str = ds_sess.get("work_dir")
    work_dir = (
        Path(work_dir_str) if work_dir_str
        else (TMP_ROOT / (ds_sess.get("session_id") or _new_id()))
    )
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.warning("[ensure-sibling] could not create work_dir %s", work_dir, exc_info=True)

    work_input_str = ds_sess.get("work_local_input")
    work_input_ok = bool(work_input_str and Path(work_input_str).exists())

    sample_str = ds_sess.get("sample_local_input")
    sample_path = Path(sample_str) if sample_str else (work_dir / "sample_input.csv")
    sample_ok = sample_path.exists()

    # Rebuild the sample from preview rows if it's gone.
    if not sample_ok:
        rows = (
            _maybe_json_load(ds_sess.get("uploaded_csv_preview"))
            or ds_sess.get("uploaded_csv_preview")
            or []
        )
        if isinstance(rows, list) and rows:
            try:
                _write_rows_to_csv(rows, sample_path)
                ds_sess["sample_local_input"] = _posix(sample_path)
                sample_ok = True
                logger.info("[ensure-sibling] rebuilt sample CSV -> %s", sample_path)
            except Exception:
                logger.warning("[ensure-sibling] failed to rebuild sample CSV at %s",
                               sample_path, exc_info=True)
        else:
            logger.info("[ensure-sibling] no preview rows to rebuild sample for %s", work_dir)

    # The critical line: if the full working input is gone, point it at the
    # (now-present) sample so every path advertised to the model reads a real
    # file. The sample is a faithful stand-in for a sample-fidelity join.
    if not work_input_ok and sample_ok:
        ds_sess["work_local_input"] = _posix(sample_path)
        logger.info("[ensure-sibling] repointed dead work_local_input -> %s", sample_path)

    return _posix(sample_path) if sample_ok else None

def _assistant_entry(content: str, segments: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """The assistant entry of a chat response: the reply with its commentary
    marked, and its segments (app/api/reply_segments.py). Without *segments*
    the server wrote *content* itself: plain."""
    return {"role": "assistant", **reply_segments.presented(
        segments if segments is not None else reply_segments.plain(content))}


def _said(content: str) -> AIMessage:
    """A reply the server writes itself (a guard, a prompt, a notice): plain."""
    return reply_segments.tag(AIMessage(content=content), reply_segments.PLAIN)


def _assistant_messages_from_state(state: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = state.get("messages") or []
    if messages and isinstance(messages[0], dict):
        messages = convert_message_dicts_to_objects(messages)

    result: List[Dict[str, str]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            segs = reply_segments.of_message(message)
            result.append(_assistant_entry(reply_segments.text_of(segs), segs))
    return result

async def lg_request(method: str, path: str, **kw) -> httpx.Response:
    """
    Low-level HTTP call to the LangGraph server, using the shared http_client from config
    if available; otherwise a one-shot client. Every call carries the service
    token the server requires (api_config.langgraph_auth_headers).
    """
    kw["headers"] = {**(kw.get("headers") or {}), **api_config.langgraph_auth_headers()}
    try:
        if api_config.http_client is None:
            timeout = httpx.Timeout(
                connect=5.0,
                read=UPSTREAM_TIMEOUT,
                write=UPSTREAM_TIMEOUT,
                pool=UPSTREAM_TIMEOUT,
            )
            http_client_local = httpx.AsyncClient(
                base_url=LANGGRAPH_API_URL,
                timeout=timeout,
            )
            resp = await http_client_local.request(method, path, **kw)
            await http_client_local.aclose()
            return resp

        return await api_config.http_client.request(method, path, **kw)
    except httpx.TimeoutException as e:
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT,
            f"LangGraph timeout contacting {path}: {e}",
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"LangGraph network error contacting {path}: {e}",
        )


async def lg_json(method: str, path: str, **kw) -> Dict[str, Any]:
    r = await lg_request(method, path, **kw)
    if not r.is_success:
        raise HTTPException(
            r.status_code,
            f"LangGraph {path} -> {r.status_code}: {r.text}",
        )
    ctype = r.headers.get("content-type", "")
    return r.json() if "application/json" in ctype else {}


# -------------------------------------------------------------------
# Session / user helpers
# -------------------------------------------------------------------


def _resolve_session_id(req: Request, body_session_id: Optional[str]) -> Optional[str]:
    if body_session_id:
        return body_session_id
    hdr = req.headers.get("X-Avaloka-Session")
    if hdr:
        return hdr
    return req.cookies.get(COOKIE_NAME)


# ── Deferred (long-running) turn support ─────────────────────────────
# A synchronous training turn runs ~271s inside GRAPH.invoke, but the edge
# proxy aborts at ~90s. Race every turn against a soft deadline: fast turns
# (all analysis) return in-turn as before; a turn that overruns returns a
# "running" handle BEFORE the proxy aborts, finishes in the background, and
# stashes its final ChatResponse on the thread-bound session for polling.
TURN_SYNC_DEADLINE_S = float(os.getenv("AVALOKA_TURN_SYNC_DEADLINE_S", "70"))

# Strong refs so backgrounded turns aren't garbage-collected mid-run.
_deferred_turn_tasks: set[asyncio.Task] = set()


# A thread's deferred turns. ``pending_turn`` is the newest turn (what a poll
# without an id answers). ``pending_turns`` keeps the turns it displaced, by id,
# so a turn that finishes after a newer one started is still answered by id:
# one slot let the newer turn's "running" (or a stale session copy) hide a
# finished answer, and the poll called it "superseded".
_PENDING_TURNS_KEEP = 4
# Written only by the _pending_turn_* mutators. A turn's own session copy was
# read before its running record existed, so merging that copy back must never
# carry these keys (it restored the previous turn and lost this one).
_PENDING_TURN_KEYS = ("pending_turn", "pending_turns")


def _pending_turn_slots(cur: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    pt = cur.get("pending_turn")
    pt = _maybe_json_load(pt) if isinstance(pt, str) else pt
    turns = cur.get("pending_turns")
    turns = _maybe_json_load(turns) if isinstance(turns, str) else turns
    return (pt if isinstance(pt, dict) else None), dict(turns) if isinstance(turns, dict) else {}


def _store_pending_turns(cur: Dict[str, Any], pt: Optional[Dict[str, Any]], turns: Dict[str, Any]) -> None:
    if pt is not None:
        cur["pending_turn"] = pt
        turns.pop(pt.get("id"), None)
    for stale in list(turns)[: max(0, len(turns) - _PENDING_TURNS_KEEP)]:
        turns.pop(stale, None)
    if turns:
        cur["pending_turns"] = turns
    else:
        cur.pop("pending_turns", None)


def _find_pending_turn(cur: Dict[str, Any], turn_id: str) -> Optional[Dict[str, Any]]:
    pt, turns = _pending_turn_slots(cur)
    if pt is not None and pt.get("id") == turn_id:
        return pt
    rec = turns.get(turn_id)
    return rec if isinstance(rec, dict) else None


def _pending_turn_running(deferred_id: str, kind: str = "analysis", queued: bool = False):
    """``queued`` says the turn is waiting for a slot on the long-jobs pool
    (app/core/long_jobs.py) rather than running; the poll shows it."""
    def _apply(cur: Dict[str, Any]) -> None:
        pt, turns = _pending_turn_slots(cur)
        known = _find_pending_turn(cur, deferred_id)
        if known is not None and known.get("status") in ("done", "error"):
            return
        if pt is not None and pt.get("id") != deferred_id:
            turns[pt.get("id")] = pt    # displaced, not lost: still answerable by id
        _store_pending_turns(cur, {
            "id": deferred_id, "status": "running", "queued": bool(queued),
            "kind": kind, "created_at": _now_iso(),
        }, turns)
    return _apply


def _carry_pending_kind(cur: Dict[str, Any], deferred_id: Optional[str] = None) -> str:
    if deferred_id is not None:
        prev = _find_pending_turn(cur, deferred_id)
    else:
        prev, _ = _pending_turn_slots(cur)
    return (prev.get("kind") if isinstance(prev, dict) else None) or "analysis"


def _finish_pending_turn(cur: Dict[str, Any], record: Dict[str, Any]) -> None:
    """File a finished turn: newest slot if it is the newest turn, else by id."""
    pt, turns = _pending_turn_slots(cur)
    prev = _find_pending_turn(cur, record["id"]) or {}
    if prev.get("created_at"):
        record["created_at"] = prev["created_at"]
    newer_elsewhere = (
        pt is not None and pt.get("id") != record["id"]
        and (pt.get("status") == "running"
             or not record.get("created_at")
             or str(pt.get("created_at") or "") > str(record["created_at"]))
    )
    if newer_elsewhere:
        turns[record["id"]] = record
        _store_pending_turns(cur, None, turns)
        return
    if pt is not None and pt.get("id") != record["id"]:
        turns[pt.get("id")] = pt
    _store_pending_turns(cur, record, turns)


def _pending_turn_done(deferred_id: str, result: Dict[str, Any]):
    # "done" means the turn returned; outcome says whether its run succeeded.
    truth = (result or {}).get("run_truth") or {}
    def _apply(cur: Dict[str, Any]) -> None:
        _finish_pending_turn(cur, {
            "id": deferred_id, "status": "done", "kind": _carry_pending_kind(cur, deferred_id),
            "outcome": truth.get("outcome"),
            "result": _json_safe_payload(result), "finished_at": _now_iso(),
        })
    return _apply


def _pending_turn_error(deferred_id: str, message: str):
    truth = run_truth.failed("exception", kind="deferred", detail=message)
    def _apply(cur: Dict[str, Any]) -> None:
        _finish_pending_turn(cur, {
            "id": deferred_id, "status": "error", "kind": _carry_pending_kind(cur, deferred_id),
            "outcome": truth["outcome"], "run_truth": truth,
            "message": message, "finished_at": _now_iso(),
        })
    return _apply


# In-process copy of each thread's pending turn. The session in Redis is the
# primary store, but when Redis is down (save_session times out) the finished
# result was lost and the poll said "running" forever. This copy lets
# GET /threads/{id}/pending-turn still answer from the same API process.
# It is not shared between replicas and not kept across restarts; the
# notifications row (D4) is the durable pointer.
_PENDING_TURN_MEM: Dict[str, Dict[str, Any]] = {}
_PENDING_TURN_MEM_MAX = 500


def _pending_turn_mem_session(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The in-process entry in the session's shape, for the mutators and the poll."""
    if not isinstance(entry, dict):
        return {}
    pt = {k: v for k, v in entry.items() if k not in ("user_id", "turns")}
    cur: Dict[str, Any] = {"pending_turn": pt} if pt else {}
    if entry.get("turns"):
        cur["pending_turns"] = dict(entry["turns"])
    return cur


def _remember_pending_turn(thread_id: str, user_id: Optional[str], mutator) -> None:
    """Apply a _pending_turn_* mutator to the in-process copy too."""
    if not thread_id:
        return
    try:
        cur = _pending_turn_mem_session(_PENDING_TURN_MEM.get(thread_id))
        mutator(cur)
        pt = cur.get("pending_turn")
        if isinstance(pt, dict):
            _PENDING_TURN_MEM[thread_id] = {
                **pt, "user_id": user_id, "turns": dict(cur.get("pending_turns") or {}),
            }
            if len(_PENDING_TURN_MEM) > _PENDING_TURN_MEM_MAX:
                # Drop the oldest entries (dicts keep insertion order).
                for key in list(_PENDING_TURN_MEM)[: len(_PENDING_TURN_MEM) - _PENDING_TURN_MEM_MAX]:
                    _PENDING_TURN_MEM.pop(key, None)
    except Exception:
        logger.debug("[deferred] could not record pending turn in memory", exc_info=True)


def _turn_job_label(question: str, dataset_label: str) -> Optional[str]:
    """What the bell quotes for a deferred chat turn: the question's first
    80 characters and the dataset's alias, both already shown in the thread.
    job_ledger scrubs and clips it again on the way in."""
    q = " ".join(redact_typed_secrets(question or "")[0].split())
    q = (q[:77] + "…") if len(q) > 80 else q
    parts = [f"\u201c{q}\u201d"] if q else []
    if dataset_label:
        parts.append(f"on {dataset_label}")
    return " ".join(parts) or None


def _turn_outcome(resp: "ChatResponse") -> Tuple[str, Optional[str], Optional[str]]:
    """(status, fail_reason, error_line) for the ledger from a turn's run truth."""
    truth = resp.run_truth or {}
    outcome = str(truth.get("outcome") or "")
    if outcome == run_truth.FAILED:
        reason = str(truth.get("reason") or "")
        return "failed", ("refused" if truth.get("kind") == "refused" or "refus" in reason.lower() else "failed"), reason or None
    return "succeeded", None, None


async def _finish_turn_job(job_id: Optional[str], status: str, *, fail_reason: Optional[str] = None,
                           error_line: Optional[str] = None, quiet: bool = False) -> None:
    """Finish the ledger row of a deferred or long chat turn off the loop;
    the ledger rings the bell and queues the chat note (neither when
    ``quiet``: the row turned out to be an ordinary turn). Never raises."""
    if not job_id:
        return
    try:
        await asyncio.to_thread(job_ledger.finish, job_id, status, fail_reason=fail_reason,
                                error_line=error_line, notify=not quiet, chat_note=not quiet)
    except Exception:  # noqa: BLE001
        logger.warning("[jobs] could not finish turn job %s", str(job_id)[:8], exc_info=True)


async def _settle_turn_job(turn_holder: Dict[str, Any], final: Dict[str, Any], deferred: bool,
                           status: str, fail_reason: Optional[str], error_line: Optional[str]) -> None:
    """Decide what the turn's ledger row is now that the graph has returned.

    * The turn queued its work on a worker (``task_info.task_id`` is set):
      the worker owns the row -- it travelled in the Celery state -- so this
      turn must not finish it.
    * The row was classified long but no producer claimed it (no resource, no
      Ray job): it was an ordinary turn. It becomes an ``analysis`` row and,
      unless the turn was deferred (then the user was told to wait and is owed
      a bell), ends quietly: no bell, no chat note.
    * Otherwise the producers' claims stand and the row ends with this turn's
      outcome; a producer that already finished it made this a no-op.
    """
    job_id = turn_holder.get("ledger_job_id")
    if not job_id:
        return
    if (final.get("task_info") or {}).get("task_id"):
        logger.info("[jobs] turn job %s handed to the worker as task %s", job_id[:8],
                    str((final.get("task_info") or {}).get("task_id"))[:8])
        turn_holder["ledger_job_id"] = None
        return
    quiet = False
    if turn_holder.get("long_kind"):
        row = await asyncio.to_thread(job_ledger.get, job_id)
        if row is not None and not row.terminal and not row.resource_type and not row.ray_job_name:
            await asyncio.to_thread(job_ledger.claim, job_id, kind="analysis")
            quiet = not deferred
    await _finish_turn_job(job_id, status, fail_reason=fail_reason, error_line=error_line, quiet=quiet)


async def _run_execution_node(node, state_in: Dict[str, Any], use_ray: bool) -> Dict[str, Any]:
    """Run an execution node off the loop: on the long-jobs pool when it will
    wait on a RayJob, on the default pool otherwise (app/core/long_jobs.py)."""
    if use_ray:
        return await long_jobs.get_pool().submit(lambda: node(state_in))
    return await asyncio.get_event_loop().run_in_executor(None, lambda: node(state_in))


JOB_NOTES_FLUSH_TIMEOUT_S = float(os.getenv("AVALOKA_JOB_NOTES_FLUSH_TIMEOUT_S", "5"))


async def _deliver_job_notes(thread_id: str, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Deliver the thread's finished-job notes at a turn boundary.

    app/core/job_notes.py holds the outbox; this is the only place that turns
    queued notes into history. It runs after the turn's own reply is in
    THREAD_META (so a note never lands between a question and its answer), is
    bounded, never raises, and calls no model. The flush is skipped by the
    outbox itself while a turn is in flight on the thread. Returns the batch
    the UI appends after the reply, or None.
    """
    if not thread_id:
        return None
    try:
        batch = await asyncio.wait_for(
            asyncio.to_thread(job_notes.flush_sync, thread_id, user_id=user_id),
            timeout=JOB_NOTES_FLUSH_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("[jobs] note flush for thread %s timed out; notes stay pending", thread_id[:8])
        return None
    except Exception:  # noqa: BLE001 - delivery must never fail the request
        logger.warning("[jobs] note flush for thread %s failed; notes stay pending", thread_id[:8], exc_info=True)
        return None
    if not batch:
        return None
    try:
        _ensure_thread_local(thread_id)
        lc = THREAD_META[thread_id].get("lc_msgs", [])
        lc.append(job_notes.make_note_message(batch["text"], batch["job_ids"], batch["batch_id"]))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
    except Exception:  # noqa: BLE001 - the transcript row and the bell already carry it
        logger.warning("[jobs] note batch %s not appended to history of %s", batch.get("batch_id", "")[:8],
                       thread_id[:8], exc_info=True)
    return batch


def _resolve_user_id(req: Request) -> Optional[str]:
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    # Local auth mode: only this server's own tokens count, and nothing falls
    # through to the Supabase path below. active_config() is None whenever the
    # mode is off or refused, so every other mode is untouched.
    local_cfg = _local_auth.active_config()
    if local_cfg is not None:
        claims = _local_auth.verify(auth_header.split(" ", 1)[1].strip(), local_cfg)
        return claims.get("sub") if claims else None

    if not JWT_SECRET:
        # Fail closed: never verify against an empty secret (forgeable).
        logger.error("Rejecting request: SUPABASE_JWT_SECRET is not configured.")
        return None

    token = auth_header.split(" ", 1)[1].strip()
    try:
        alg = jwt.get_unverified_header(token).get("alg", "HS256")
        if alg in ("ES256", "RS256"):
            if not _jwks_client:
                logger.warning("JWT alg=%s but SUPABASE_URL is not set; cannot fetch JWKS", alg)
                return None
            signing_key = _jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256", "RS256"],
                options={"verify_aud": False},
            )
        else:
            payload = jwt.decode(
                token,
                JWT_SECRET,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )
    except InvalidTokenError as exc:
        logger.warning("Invalid JWT in request: %s", exc)
        return None
    except Exception as exc:
        # e.g. JWKS fetch failure — return 401 instead of crashing with a 500
        logger.warning("JWT verification failed: %s", exc)
        return None

    return payload.get("sub") or payload.get("email")

# -------------------------------------------------------------------
# FastAPI app + CORS + lifespan
# -------------------------------------------------------------------

# @asynccontextmanager
# async def server_lifespan(app: FastAPI):
#     global small_file_persistence_executor, small_file_persistence_semaphore, small_file_persistence_inflight

#     _validate_auth_config()
#     # A service-role key for the wrong project answers every request with
#     # `401 Invalid API key`, which reads like a rotated key rather than a
#     # mismatch. Say so once at boot instead of once per failed request.
#     try:
#         from app.agents.sampling_persistence import warn_on_supabase_credential_mismatch
#         warn_on_supabase_credential_mismatch()
#     except Exception as _exc:  # never let a diagnostic stop the server
#         logger.debug("supabase credential check skipped: %s", _exc)

#     async with config_lifespan(app):
@asynccontextmanager
async def server_lifespan(app: FastAPI):
    global small_file_persistence_executor, small_file_persistence_semaphore, small_file_persistence_inflight

    _validate_auth_config()
    # Every sealed credential is AES-GCM under DB_ENCRYPTION_KEY. Provisioned
    # from .env when missing (app/core/provisioning.py); the API refuses to start
    # only if it still has none (app/core/encryption_key.py).
    from app.core.encryption_key import require_db_encryption_key
    require_db_encryption_key("The Avaloka API")
    # A service-role key for the wrong project answers every request with
    # `401 Invalid API key`, which reads like a rotated key rather than a
    # mismatch. Say so once at boot instead of once per failed request.
    try:
        from app.agents.sampling_persistence import warn_on_supabase_credential_mismatch
        warn_on_supabase_credential_mismatch()
    except Exception as _exc:  # never let a diagnostic stop the server
        logger.debug("supabase credential check skipped: %s", _exc)

    # Load torch + the local embedding model once, on the main thread, at boot.
    # Loading it lazily inside a request worker thread (memory injection node,
    # or the OpenAI -> sentence-transformers fallback) crashed the process with
    # SIGSEGV (exit 139). Set AVALOKA_EMBED_WARMUP=0 to skip.
    if os.getenv("AVALOKA_EMBED_WARMUP", "1").lower() not in ("0", "false", "no"):
        try:
            torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
            from app.services.embedding_utils import warmup_local_model
            _t_warm = time.monotonic()
            if warmup_local_model():
                logger.info("[startup] embedding model warmed up in %.2fs",
                            time.monotonic() - _t_warm)
        except Exception as _exc:  # a failed warm-up must never stop the server
            logger.warning("[startup] embedding warm-up skipped: %s", _exc)

    # Cloud keys written before migration 20261116130000 are plaintext; seal
    # them now, under the key just required above. Counts only in the log.
    try:
        from app.api import cloud_connections as _cc
        _sealed = await asyncio.to_thread(_cc.seal_at_startup)
        if _sealed:
            logger.info("[startup] sealed %d cloud connection(s) that carried plaintext keys", _sealed)
    except Exception as _exc:  # never let the sealer stop the server
        logger.warning("[startup] cloud key sealing skipped: %s", type(_exc).__name__)

    async with config_lifespan(app):
        small_file_persistence_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=SMALL_FILE_PERSISTENCE_MAX_WORKERS,
            thread_name_prefix="small-file-persist",
        )
        small_file_persistence_semaphore = asyncio.Semaphore(SMALL_FILE_PERSISTENCE_MAX_INFLIGHT)
        small_file_persistence_inflight = set()
        try:
            yield
        finally:
            small_file_persistence_inflight.clear()
            small_file_persistence_semaphore = None
            if small_file_persistence_executor is not None:
                small_file_persistence_executor.shutdown(wait=False, cancel_futures=True)
                small_file_persistence_executor = None

app = FastAPI(title="Avaloka UI API", lifespan=server_lifespan)

# The final pass on what a person is shown: no sealed credential token, and
# no fragment of one, in any thread response, history, trace or stream.
from app.core.typed_secrets import RedactRepliesMiddleware, install_model_scrub  # noqa: E402

install_model_scrub()
app.add_middleware(RedactRepliesMiddleware)

# Auth before validation: an anonymous request gets 401 before FastAPI parses and
# validates its body or query, so a 422 (or a router's own 200/400 store and body
# checks) can never answer a caller with no token. Unlike 1.6 it gates reads too:
# a GET with a required query field leaks its schema the same way. Added before
# CORS so the 401 still carries CORS headers. The resolver is looked up per
# request so tests that patch it are honoured.
from app.api.auth_gate import GATED_METHODS, AuthBeforeBodyMiddleware  # noqa: E402
from app.core import editions as _editions  # noqa: E402

_AUTH_GATE_ANY = ("GET", "POST", "PUT", "PATCH", "DELETE")
AUTH_GATE_EXEMPT = {
    # Liveness and readiness probes (Helm, compose) and the release string.
    ("GET", "/health"),
    ("GET", "/version"),
    # Public by owner ruling, on hosted and self-hosted alike: feature discovery
    # a client may read before signing in. An anonymous caller is told the
    # least-privileged plan's limits (_caller_edition) and nothing user-specific.
    ("GET", "/api/capabilities"),
    # Tells a signed-out browser enabled=false, reason="unauthenticated", so the
    # UI stops rather than retrying (app/analytics/collector.py).
    ("GET", "/analytics/config"),
    # The GoTrue subset for AVALOKA_AUTH_MODE=local is how a caller gets a token,
    # so it cannot require one. Every route 404s unless local mode is on, and the
    # routes that need a session check the bearer themselves and answer in the
    # GoTrue shape supabase-js parses (app/api/local_auth_routes.py).
    *((m, "/local-auth/{rest:path}") for m in _AUTH_GATE_ANY),
}
# Public everywhere except on the hosted deployment, where they need a login (owner
# ruling). Self-hosted and undeclared deployments keep them public, as before the
# deployment could be declared. Asked per request (editions.deployment_state).
AUTH_GATE_EXEMPT_UNLESS_HOSTED = {
    # FastAPI's schema and docs pages. Nothing depends on reading them without a
    # token: the tests that read the schema send one.
    ("GET", "/openapi.json"),
    ("GET", "/docs"),
    ("GET", "/docs/oauth2-redirect"),
    ("GET", "/redoc"),
    # Answers {"user_id": null} to an anonymous caller: it reports who the token
    # resolves to.
    ("GET", "/debug/whoami"),
}
app.add_middleware(
    AuthBeforeBodyMiddleware,
    resolve_user_id=lambda req: _resolve_user_id(req),
    exempt=AUTH_GATE_EXEMPT,
    methods=GATED_METHODS,
    conditional_exempt=AUTH_GATE_EXEMPT_UNLESS_HOSTED,
    conditional_applies=lambda: _editions.deployment_state() != "hosted",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Product analytics collector (docs/analytics/DESIGN.md). Events are stored in
# this deployment's own database; nothing leaves the cluster from these routes.
from app.analytics.collector import build_router as _build_analytics_router

app.include_router(_build_analytics_router(_resolve_user_id))

# GoTrue subset for AVALOKA_AUTH_MODE=local; every route 404s when it is off.
from app.api.local_auth_routes import router as _local_auth_router

app.include_router(_local_auth_router)

# Saved database connections: the UI's only way to list, save and delete them,
# in every auth mode (app/api/db_connections.py).
from app.api import db_connections as _db_connections

app.include_router(_db_connections.build_router(_resolve_user_id))
# The personal dataset catalog: /catalog/datasets (docs/design/1.7-DATA-PLATFORM.md part C).
from app.catalog.api import build_router as _build_catalog_router  # noqa: E402
app.include_router(_build_catalog_router(_resolve_user_id))
# Cloud connections: the one route that stores an object-store key, sealed in
# this process (app/api/cloud_connections.py; migration 20261116130000 refuses
# plaintext keys). At start-up, rows written before that migration are sealed.
from app.api import cloud_connections as _cloud_connections  # noqa: E402

app.include_router(_cloud_connections.build_router(_resolve_user_id))


# Rows written before migration 20261116130000 are sealed at start-up by
# server_lifespan (seal_at_startup), not here.

# Ray jobs panel: the caller's own Ray jobs (status, list, detail, logs, stop).
# The browser never reaches Ray directly; see app/api/ray_jobs.py.
from app.api.ray_jobs import build_router as _build_ray_jobs_router

app.include_router(_build_ray_jobs_router(_resolve_user_id, _require_premium_plan))

# Trained models, registry records, serving endpoints and ML jobs. Owner-scoped;
# slow operations answer 202 with a job handle (app/api/models_routes.py).
from app.api.models_routes import build_router as _build_models_router

app.include_router(_build_models_router(_resolve_user_id, _require_premium_plan))

# ---------- PNA preflight ----------
@app.middleware("http")
async def handle_private_network_preflight(request: Request, call_next):
    if request.method == "OPTIONS":
        origin = request.headers.get("Origin")
        wants_pna = (
            request.headers.get("Access-Control-Request-Private-Network", "")
            .lower()
            == "true"
        )
        if origin and wants_pna and ("*" in allow_origins or origin in allow_origins):
            resp = Response(status_code=204)
            resp.headers["Access-Control-Allow-Private-Network"] = "true"
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Headers"] = request.headers.get(
                "Access-Control-Request-Headers", "*"
            )
            return resp
    return await call_next(request)


@app.middleware("http")
async def _upload_timing_middleware(request: Request, call_next):
    if request.url.path != "/api/upload":
        return await call_next(request)
    t = time.monotonic()
    resp = await call_next(request)
    logger.info(
        "[upload-timing] request_total  %.2fs (body parse + handler + response build)",
        time.monotonic() - t,
    )
    return resp


@app.middleware("http")
async def _upload_plan_precheck(request: Request, call_next):
    """Refuse an over-limit upload from its Content-Length, before the body is read.

    The upload handler only runs after FastAPI has received and parsed the whole
    multipart body, so a 240 MB file on a Free plan would otherwise be fully
    transferred and written to disk just to be rejected. This middleware sits
    outside CORSMiddleware, so it adds the CORS headers itself; without them the
    browser reports a CORS error instead of the 413.
    """
    if request.method != "POST" or request.url.path != "/api/upload":
        return await call_next(request)
    try:
        declared = int(request.headers.get("content-length") or 0)
    except ValueError:
        declared = 0
    user_id = _resolve_user_id(request) if declared else None
    if user_id:
        limits = limits_for(await asyncio.to_thread(resolve_plan, user_id))
        # Multipart framing adds bytes beyond the files themselves.
        slack = int(os.getenv("AVALOKA_UPLOAD_CL_SLACK_BYTES", str(1 << 20)))
        if declared > limits.max_total_bytes + slack:
            body = limit_error(limits, "total_size", declared)
            logger.info("[upload-limits] refused before body: user=%s plan=%s bytes=%d",
                        user_id, limits.plan, declared)
            resp = JSONResponse(status_code=413, content={"detail": body})
            origin = request.headers.get("origin")
            if origin and ("*" in allow_origins or origin in allow_origins):
                resp.headers["Access-Control-Allow-Origin"] = origin
                resp.headers["Access-Control-Allow-Credentials"] = "true"
                resp.headers["Vary"] = "Origin"
            return resp
    return await call_next(request)


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------


@app.get("/health")
async def health():
    ok = False
    try:
        r = await lg_request("GET", "/ok")
        ok = r.status_code < 500
    except HTTPException:
        ok = False

    try:
        redis_ok = await session_service.cache.ping() if session_service.cache else False
    except Exception as e:
        _note_cache_failure("ping", e)
        redis_ok = False

    return {
        "status": "ok",
        "graph_ready": GRAPH_READY,
        "langgraph_url": LANGGRAPH_API_URL,
        "assistant_id": ASSISTANT_ID,
        "upstream_reachable": ok,
        "upstream_timeout_s": UPSTREAM_TIMEOUT,
        "redis_connected": redis_ok,
        "redis_mode": (
            getattr(session_service.cache, "mode", "single")
            if session_service.cache
            else "none"
        ),
    }

@app.get("/version")
async def get_version():
    return {"version": APP_VERSION}


async def _caller_edition(request: Request) -> str:
    """The edition whose limits apply to this caller.

    An open-source install is ``oss``. Otherwise the caller's plan decides, and
    an anonymous caller gets the least-privileged hosted plan.
    """
    if (os.getenv("AVALOKA_EDITION") or "").strip().lower() in ("oss", "community"):
        return "oss"
    user_id = _resolve_user_id(request)
    if not user_id:
        return "free"
    return await asyncio.to_thread(resolve_plan, user_id)


#: The swarm console gate (app/api/swarm_gate.py): 401 without a user, 403 on
#: oss/free, the caller's user_id on Professional and Enterprise. Every
#: swarm/trace route takes it as ``user_id: str = Depends(require_swarm_console)``.
require_swarm_console = swarm_gate.dependency(_resolve_user_id, _caller_edition)


async def _note_thread_mission(thread_id: str, mission_id: str) -> bool:
    """Record a swarm mission on the thread's session so the next chat turn
    tells the team's story. The one hook for every path that fans out on a
    thread's dataset: the /swarm page today, the planner's own fan-out next."""
    try:
        sid = await get_thread_session(thread_id)
        if not sid:
            return False
        await update_session(sid, lambda cur: chat_swarm.remember(cur, mission_id, told=False))
        return True
    except Exception:  # noqa: BLE001 - the story is additive; the mission is not
        logger.warning("[chat-swarm] could not note mission %s on thread %s", mission_id, thread_id)
        return False


async def _swarm_stories(sess: Dict[str, Any], user_id: str, edition: Optional[str],
                         dataset_label: Optional[str]) -> tuple[List[Dict[str, Any]], bool]:
    """The team stories owed to this chat: one ``swarm`` prompt per mission that
    ran for the thread's dataset and has not been told. Marks finished ones
    told; an unfinished one is told so far and stays pending. Returns the
    prompts and whether the session's record changed."""
    prompts: List[Dict[str, Any]] = []
    changed = False
    for mid in chat_swarm.pending_missions(sess):
        try:
            mission = await chat_swarm.records_for(user_id, mid)
        except Exception:  # noqa: BLE001
            logger.warning("[chat-swarm] could not read mission %s", mid, exc_info=True)
            continue
        if mission is None:  # nothing serves it any more: forget it
            chat_swarm.remember(sess, mid, told=True)
            changed = True
            continue
        prompts.append(chat_swarm.summary_prompt(
            mission["records"], edition=edition, mission_id=mid,
            finished=mission["finished"], dataset_label=dataset_label))
        if mission["finished"]:
            chat_swarm.remember(sess, mid, told=True)
            changed = True
    return prompts, changed


def _voice_after_upload(**kwargs) -> Optional[List[Dict[str, Any]]]:
    """Avaloka's first word after the auto-insights (app/api/chat_voice.py).
    Never fails an upload: a gap in the statistics means she says nothing."""
    try:
        from app.api import chat_voice

        return chat_voice.prompts_after_upload(**kwargs) or None
    except Exception:  # noqa: BLE001 - the prompts are additive
        logger.warning("[chat-voice] could not build the upload prompts", exc_info=True)
        return None


def _voice_after_turn(final: Mapping[str, Any], *, question: str,
                      execution_context: Optional[Mapping[str, Any]],
                      sess: Mapping[str, Any],
                      profile_meta: Optional[Mapping[str, Any]] = None,
                      datasets: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
    """Avaloka talking through the steps the turn took (app/api/chat_voice.py),
    grounded in the dataset's statistics: the session's when it carries them
    (register-existing does), else this process's upload profile
    (``_profile_meta``). Never fails a chat turn."""
    try:
        from app.api import chat_voice

        meta = dict(profile_meta or {})
        stats = (_maybe_json_load(sess.get("sample_statistics")) or sess.get("sample_statistics")
                 or meta.get("sample_statistics"))
        schema = _maybe_json_load(sess.get("uploaded_csv_columns")) or sess.get("uploaded_csv_columns")
        return chat_voice.prompts_after_turn(
            final, question=question, execution_context=execution_context,
            sample_statistics=stats if isinstance(stats, dict) else None,
            profiling_result=(sess.get("full_profiling_result") or sess.get("profiling_result")
                              or meta.get("full_profiling_result") or meta.get("profiling_result")),
            schema=schema if isinstance(schema, list) else None,
            datasets=datasets,
        ) or None
    except Exception:  # noqa: BLE001 - the prompts are additive
        logger.warning("[chat-voice] could not build the turn prompts", exc_info=True)
        return None


@app.get("/api/capabilities")
async def get_api_capabilities(request: Request):
    """Feature discovery for new clients; the 1.5.2 UI need not call it."""
    from app.core.editions import swarm_limits

    edition = await _caller_edition(request)
    return _api_capabilities(
        upload_formats=SUPPORTED_UPLOAD_EXTS,
        upload_format_status=upload_format_status(),
        swarm_limits=swarm_limits(edition).as_dict(),
        swarm_console=swarm_gate.console_capability(edition),
    )


@app.get("/api/limits")
async def get_plan_limits(request: Request):
    """The caller's plan and upload limits, so the UI can check file sizes
    before uploading and show the upgrade dialog without sending anything."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    plan = await asyncio.to_thread(resolve_plan, user_id)
    # The same decision the chat and model endpoints make, for the UI to hide or
    # label paid actions before the user tries them.
    decisions = {
        feature: await asyncio.to_thread(check_feature, feature, user_id=user_id)
        for feature in FEATURE_CAPABILITY
    }
    return {
        # Free while a lookup is failing (resolve_plan's upload-limit policy);
        # plan_check_unavailable says when that is the case on a hosted deployment.
        "plan": plan,
        "limits": limits_for(plan).as_dict(),
        "features": {feature: d.allowed for feature, d in decisions.items()},
        # "not_on_plan" is an upgrade case; "unavailable" (the plan could not be
        # checked) is not: the UI says to try again and offers no upgrade.
        "feature_states": {
            feature: "allowed" if d.allowed else "unavailable" if d.unavailable else "not_on_plan"
            for feature, d in decisions.items()
        },
        "plan_check_unavailable": any(d.unavailable for d in decisions.values()),
    }

@app.get("/debug/whoami")
async def whoami(request: Request):
    return {"user_id": _resolve_user_id(request)}


@app.post("/api/missions/plan")
async def plan_mission_route(request: Request, intent: dict):
    """Plan a mission from an interface-agnostic intent dict (D4 / R2).

    Returns ``planned_to_dict()`` — the same shared representation the CLI produces
    in-process and the MCP server returns — so the CLI's ``--endpoint`` remote mode
    is byte-identical to its local mode for the same intent.
    """
    if not _resolve_user_id(request):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    from app.interfaces.service import plan_mission, planned_to_dict

    try:
        return planned_to_dict(plan_mission(intent or {}))
    except Exception as exc:  # noqa: BLE001 - surface a clean 400, not a 500 stack
        logger.warning("Mission planning failed for intent %s: %s", intent, exc)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Could not plan mission: {exc}")


# -------------------------------------------------------------------
# Upload (multi-format) -> BlobStore
# -------------------------------------------------------------------
from typing import Union, Optional, List, Dict, Any
async def _read_upload_to_path_with_limit(
    up: UploadFile,
    dest_path: Path,
    per_file_limit: int,
) -> int:
    """
    Streams UploadFile to dest_path enforcing per-file byte limit.
    Returns number of bytes written.
    """
    written = 0
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as outfp:
        while True:
            chunk = await up.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            written += len(chunk)
            if written > per_file_limit:
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"File '{up.filename}' exceeds max size ({per_file_limit} bytes)."
                )
            outfp.write(chunk)
    return written


from contextlib import contextmanager


@contextmanager
def _stage(name: str, dsid: str = ""):
    """Logs how long a block took: [upload-timing] <dsid> <name> <seconds>."""
    t = time.monotonic()
    try:
        yield
    finally:
        logger.info("[upload-timing] %s %-14s %.2fs", dsid, name, time.monotonic() - t)


async def _put_upload_object(
    src: Path, object_name: str, dest_uri: Optional[str], conn: Dict[str, Any]
) -> str:
    """Upload a local file to the configured store; returns the object URI.
    Same destination logic the old inline PUT block used."""
    with _stage("gcs_put", object_name):
        if dest_uri and "://" in dest_uri:
            if conn:
                store, _ = await _store_from_connection_uri(dest_uri, conn)
                return await asyncio.to_thread(store.put_file, src, object_name.lstrip("/"))
            if dest_uri.startswith("az://"):
                try:
                    _scheme, rest = dest_uri.split("://", 1)
                    parts = rest.split("/", 2)
                    prefix_path = parts[2].strip("/") if len(parts) >= 3 else ""
                except Exception:
                    prefix_path = ""
                key = f"{prefix_path}/{object_name}".lstrip("/") if prefix_path else object_name
                return await asyncio.to_thread(storage_service.blob_store.put_file, src, key)
            store, base_key = _store_and_key_from_uri(dest_uri, None)
            key_base = (base_key or "").rstrip("/")
            key = f"{key_base}/{object_name}".lstrip("/") if key_base else object_name
            return await asyncio.to_thread(store.put_file, src, key)
        return await asyncio.to_thread(storage_service.blob_store.put_file, src, object_name)


# ---- Background storage finalization for uploads ----------------------------
# The upload response no longer waits for the storage push. The push keeps
# running after the response; the finalizer below records the object URI on
# the dataset session once it lands.

# Strong refs so finalizers aren't garbage-collected after the response is sent.
_upload_storage_tasks: set = set()

# A request that read the session before the push landed and then saves the
# whole dict back (e.g. save_session(sid, stale_sess)) would drop
# data_source_location again. Re-check a few times and re-apply if it vanished.
_UPLOAD_REASSERT_DELAYS_S = (20, 60, 180)


def _delete_orphan_upload(uri: str, object_name: str) -> None:
    """Best-effort delete of an object whose dataset no longer exists."""
    try:
        store, key = _store_and_key_from_uri(uri, object_name)
        if store and key:
            store.delete(key)
            logger.info("[upload-storage] deleted orphaned object %s", uri)
    except Exception:
        logger.warning("[upload-storage] could not delete orphaned object %s", uri, exc_info=True)


async def _finalize_upload_storage(
    put_task: "asyncio.Task",
    session_id: str,
    dsid: str,
    object_name: str,
) -> None:
    # 1) Wait for the push itself.
    try:
        uri = await put_task
    except asyncio.CancelledError:
        logger.warning("[upload-storage] %s push was cancelled", dsid)
        return
    except Exception as e:
        logger.error("[upload-storage] %s push to storage FAILED: %s", dsid, e, exc_info=True)
        try:
            sess = await get_session(session_id)
            if isinstance(sess, dict) and sess.get("dataset_id") == dsid:
                err = str(e)[:500]
                await update_session(session_id, lambda cur: cur.update({"storage_error": err}))
        except Exception:
            logger.debug("[upload-storage] could not record storage_error", exc_info=True)
        return

    # 2) Record the URI on the session (and keep it there).
    for attempt, delay in enumerate((0, *_UPLOAD_REASSERT_DELAYS_S)):
        if delay:
            await asyncio.sleep(delay)
        try:
            sess = await get_session(session_id)
        except Exception:
            logger.debug("[upload-storage] session read failed for %s", session_id, exc_info=True)
            continue

        if not isinstance(sess, dict) or sess.get("dataset_id") != dsid:
            if attempt == 0:
                # Upload rolled back or dataset deleted before the push finished.
                await asyncio.to_thread(_delete_orphan_upload, uri, object_name)
            return

        if sess.get("data_source_location"):
            continue  # already recorded (or re-recorded); check again later

        def _apply(cur: Dict[str, Any], _uri: str = uri) -> None:
            if not cur.get("data_source_location"):
                cur["data_source_location"] = _uri
                cur.pop("storage_error", None)
                # Let send_message re-save the durable snapshot with the cloud URI.
                cur["snapshot_persisted"] = False

        try:
            await update_session(session_id, _apply)
            logger.info(
                "[upload-storage] %s recorded %s on session %s (check %d)",
                dsid, uri, session_id, attempt,
            )
        except Exception:
            logger.warning("[upload-storage] failed to record uri for %s", dsid, exc_info=True)


def _spawn_upload_finalizer(
    put_task: "asyncio.Task", session_id: str, dsid: str, object_name: str
) -> None:
    t = asyncio.create_task(_finalize_upload_storage(put_task, session_id, dsid, object_name))
    _upload_storage_tasks.add(t)
    t.add_done_callback(_upload_storage_tasks.discard)


# ---- Deferred profile for uploads (D5) --------------------------------------
# The semantic profile (three LLM calls) and the model-written Auto Insights
# ran inside the upload request; on a local model a ~50 MB file took over
# 600 s. The route now starts both as one background job and waits only a
# short soft deadline for it. A job that finishes in time is returned in the
# response as before; one that does not leaves data-only charts in the
# response, profiling_status="pending", and lands its result on the dataset
# session (GET /api/datasets/{id}/profile) when it finishes.
UPLOAD_PROFILE_SYNC_DEADLINE_S = float(os.getenv("AVALOKA_UPLOAD_PROFILE_SYNC_DEADLINE_S", "2"))

PROFILE_PENDING = "pending"
PROFILE_READY = "full_profile"
PROFILE_FAILED = "failed"
PROFILE_SKIPPED = "skipped"

_PROFILE_FAILED_MSG = (
    "The data profile could not be built, so Auto Insights describe the data "
    "without it. Charts, tables and questions still work."
)
_INSIGHTS_FAILED_MSG = (
    "The data profile is ready, but Auto Insights could not be rewritten with "
    "it. The charts shown are built from the data alone."
)

# Strong refs so landers aren't garbage-collected after the response is sent.
_upload_profile_tasks: set = set()


async def _run_upload_profile(
    *,
    dsid: str,
    schema_list: List[str],
    schema_any: Any,
    samples: List[Dict[str, Any]],
    sample_statistics: Optional[Dict[str, Any]],
    used_agent: bool,
    input_path: str,
    ext: str,
) -> Dict[str, Any]:
    """The profile, then the charts written with it. Never raises: a failure
    comes back as profiling_status="failed" with a message for the user."""
    out: Dict[str, Any] = {
        "profiling_status": PROFILE_SKIPPED,
        "full_profiling_result": None,
        "profiling_error": None,
        "visualization_config": None,
        "visualization_status": None,
    }
    prof: Optional[Dict[str, Any]] = None
    if used_agent and sample_statistics:
        try:
            with _stage("profile_full", dsid):
                res = await asyncio.to_thread(
                    profile_full, schema=schema_list, sample_rows=samples,
                    sample_statistics=sample_statistics,
                ) or {}
            result = res.get("full_profiling_result")
            if res.get("profiling_status") == PROFILE_SKIPPED:
                pass  # no profiling model configured: nothing failed
            elif (
                res.get("profiling_status") == "error"
                or not isinstance(result, dict)
                or result.get("status") == "error"
            ):
                logger.warning(
                    "[upload-profile] %s profile_full returned an error: %s",
                    dsid, (result or {}).get("reason") if isinstance(result, dict) else result,
                )
                out.update(profiling_status=PROFILE_FAILED, profiling_error=_PROFILE_FAILED_MSG)
            else:
                prof = result
                out.update(profiling_status=PROFILE_READY, full_profiling_result=result)
        except Exception as e:
            logger.warning("[upload-profile] %s profile_full failed: %s", dsid, e, exc_info=True)
            out.update(profiling_status=PROFILE_FAILED, profiling_error=_PROFILE_FAILED_MSG)

    # Auto Insights are written in the dataset's domain terms, so the charts
    # wait for the profile. A failed profile must not cost the charts.
    try:
        with _stage("viz", dsid):
            viz = await asyncio.to_thread(
                build_visualization_config_from_sample,
                dataset_id=dsid, sample_rows=samples, schema=schema_any,
                task_type="unsupervised", target_column=None,
                # Charts chosen from the sample, numbers computed on the
                # whole local file (only the charted columns are read).
                full_data_path=input_path if used_agent else None,
                full_data_type=ext,
                total_rows=((sample_statistics or {}).get("data_shape") or {}).get("rows"),
                profiling_result=prof,
            )
        out["visualization_config"] = viz or {}
        out["visualization_status"] = (viz or {}).get("visualization_status", "ready")
    except Exception as e:
        logger.warning("[upload-profile] %s visualization failed: %s", dsid, e, exc_info=True)
        if out["profiling_status"] == PROFILE_READY:
            out.update(profiling_status=PROFILE_FAILED, profiling_error=_INSIGHTS_FAILED_MSG)
    return out


def _deferred_profile_fields(r: Dict[str, Any]) -> Dict[str, Any]:
    """What a landed job writes on the dataset session."""
    fields: Dict[str, Any] = {
        "profiling_status": r.get("profiling_status"),
        "profiling_error": r.get("profiling_error"),
        "full_profiling_result": (
            _jsonify(r["full_profiling_result"]) if r.get("full_profiling_result") else None
        ),
        "profiling_finished_at": _now_iso(),
    }
    if r.get("visualization_config"):
        fields["visualization_config"] = _jsonify(r["visualization_config"])
        fields["visualization_status"] = r.get("visualization_status") or "ready"
    return fields


async def _notify_deferred_profile(
    *, user_id: Optional[str], dsid: str, thread_id: Optional[str],
    filename: str, r: Dict[str, Any],
) -> None:
    status_ = r.get("profiling_status")
    if status_ == PROFILE_READY:
        n_type, title = "insights.generated", "Auto Insights ready"
        body = f"The data profile for {filename} is ready." if filename else "The data profile is ready."
    elif status_ == PROFILE_FAILED:
        n_type, title = "insights.failed", "Data profile failed"
        body = (f"{filename}: " if filename else "") + (r.get("profiling_error") or _PROFILE_FAILED_MSG)
    else:
        return
    await notification_service.create_notification(
        user_id, n_type, title, body,
        thread_id=thread_id or None, dataset_id=dsid, ref=f"upload-profile:{dsid}",
    )


async def _land_upload_profile(
    job: "asyncio.Future",
    *,
    session_id: str,
    dsid: str,
    user_id: Optional[str],
    thread_id: Optional[str],
    filename: str,
) -> None:
    try:
        r = await job
    except asyncio.CancelledError:
        logger.warning("[upload-profile] %s job was cancelled", dsid)
        return
    except Exception as e:  # _run_upload_profile catches its own errors
        logger.error("[upload-profile] %s job failed: %s", dsid, e, exc_info=True)
        r = {"profiling_status": PROFILE_FAILED, "profiling_error": _PROFILE_FAILED_MSG}

    meta = _profile_meta.get(dsid) or {}
    meta["deferred_profile"] = dict(r)
    if r.get("full_profiling_result"):
        meta["full_profiling_result"] = r["full_profiling_result"]
    _profile_meta[dsid] = meta

    fields = _deferred_profile_fields(r)
    logger.info("[upload-profile] %s landed: %s", dsid, fields["profiling_status"])

    # Same re-assert as the storage finalizer: a request that read the session
    # while the job ran and saves it back would put "pending" back.
    for attempt, delay in enumerate((0, *_UPLOAD_REASSERT_DELAYS_S)):
        if delay:
            await asyncio.sleep(delay)
        try:
            sess = await get_session(session_id)
        except Exception:
            logger.debug("[upload-profile] session read failed for %s", session_id, exc_info=True)
            continue
        if not isinstance(sess, dict) or sess.get("dataset_id") != dsid:
            return  # dataset deleted (or upload rolled back) before the job finished
        if attempt and sess.get("profiling_status") != PROFILE_PENDING:
            continue

        def _apply(cur: Dict[str, Any]) -> None:
            if cur.get("dataset_id") == dsid:
                cur.update(fields)

        try:
            await update_session(session_id, _apply)
        except Exception:
            logger.warning("[upload-profile] could not record the profile for %s", dsid, exc_info=True)
            continue
        if attempt == 0:
            if r.get("full_profiling_result"):
                try:
                    await asyncio.to_thread(
                        upsert_profile, dataset_id=dsid, phase="semantic_profile",
                        profiling_result=r["full_profiling_result"],
                    )
                except Exception:
                    logger.warning("[upload-profile] could not persist the profile for %s", dsid, exc_info=True)
            try:
                await _notify_deferred_profile(
                    user_id=user_id, dsid=dsid, thread_id=thread_id, filename=filename, r=r,
                )
            except Exception:
                logger.warning("[upload-profile] could not notify for %s", dsid, exc_info=True)


def _spawn_upload_profile_lander(job: "asyncio.Future", **kw: Any) -> None:
    t = asyncio.create_task(_land_upload_profile(job, **kw))
    _upload_profile_tasks.add(t)
    t.add_done_callback(_upload_profile_tasks.discard)


@app.post("/api/upload", response_model=Union[UploadResponse, MultiUploadResponse])
async def upload_csv(
    request: Request,
    response: Response,

    # backward compat: allow single "file"
    file: Optional[UploadFile] = File(None),
    # new: multi files
    files: Optional[List[UploadFile]] = File(None),

    schema_json: Optional[str] = Form(None),
    storage_uri: Optional[str] = Form(None),
    backend: Optional[str] = Form(None),
    bucket: Optional[str] = Form(None),
    prefix: Optional[str] = Form(None),
    connection_id: Optional[str] = Form(None),
    # "+ Add" under the chat: add these files to an EXISTING chat instead of
    # starting a new analysis (no new thread, no new group, cookie unchanged).
    attach_to_thread_id: Optional[str] = Form(None),
    ):
    _t_handler = time.monotonic()
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Session storage (Redis) is not available.")
    if not storage_service.blob_store:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Storage backend not initialized.")

    # collect incoming files (supports both file + files)
    incoming: List[UploadFile] = []
    if files is not None:
        incoming.extend(files)
    if file is not None:
        incoming.append(file)
    if not incoming:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No files provided (use 'files' or 'file').")

    _limits = limits_for(await asyncio.to_thread(resolve_plan, user_id))
    # Starlette knows each file's size once the body is parsed: check the plan
    # before copying anything to disk. The streaming checks below stay as a backstop.
    _sizes = [getattr(up, "size", None) for up in incoming]
    _err = (
        check_sizes(_limits, _sizes)
        if all(s is not None for s in _sizes)
        else (limit_error(_limits, "file_count", len(incoming))
              if len(incoming) > _limits.max_files else None)
    )
    if _err:
        logger.info("[upload-limits] refused: plan=%s reason=%s actual=%s",
                    _limits.plan, _err["reason"], _err["actual"])
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, _err)

    used_files_field = files is not None  # did caller use multi field

    # ---- 1) load cloud connection once ----
    conn: Dict[str, Any] = {}
    if connection_id:
        conn = await get_cloud_connection(connection_id)

    # ---- 2) build destination storage_uri once ----
    dest_uri = storage_uri or request.headers.get("X-Storage-URI")
    if not dest_uri and backend:
        bdl = backend.lower()
        if bdl == "s3":
            b = (bucket or settings.s3_bucket).strip()
            p = (prefix or settings.s3_prefix).strip().strip("/")
            dest_uri = f"s3://{b}/{p}" if p else f"s3://{b}"
        elif bdl in ("gcs", "gs"):
            b = (bucket or settings.gcs_bucket).strip()
            p = (prefix or settings.gcs_prefix).strip().strip("/")
            dest_uri = f"gs://{b}/{p}" if p else f"gs://{b}"
        elif bdl in ("azure", "az"):
            account = (os.getenv("AZURE_ACCOUNT") or getattr(settings, "azure_account", "")).strip()
            container = (bucket or getattr(settings, "azure_container", "")).strip()
            p = (prefix or getattr(settings, "azure_prefix", "")).strip().strip("/")

            if conn:
                if not account:
                    account = (conn.get("access_key") or conn.get("account_name") or "").strip()
                if not container:
                    container = (conn.get("bucket_name") or conn.get("container") or "").strip()

            if not (account and container):
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    "Azure requires account and container (set env/config or use a connection with those fields).",
                )
            dest_uri = f"az://{account}/{container}" + (f"/{p}" if p else "")
        else:
            raise HTTPException(400, f"Unknown backend '{backend}'. Use s3 | gcs | azure.")

    datasets_out: List[UploadedDatasetOut] = []
    dataset_ids: List[str] = []
    dataset_session_map: Dict[str, str] = {}

    group_session_id: Optional[str] = None
    thread_id: str = ""

    # Attach mode: the new datasets join the chat's existing dataset group.
    attach_group_sid: Optional[str] = None
    if attach_to_thread_id:
        attach_group_sid, _attach_group_sess = await _resolve_thread_group_for_user(
            attach_to_thread_id, user_id
        )
        if not attach_group_sid:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found for this user")
        group_session_id = attach_group_sid
        thread_id = attach_to_thread_id

    # for backward-compat UploadResponse (single upload)
    first_schema_any: Union[List[str], Dict[str, str], Dict[str, Any]] = {}
    first_samples: List[Dict[str, Any]] = []
    first_ddl: str = ""
    first_viz_config: Dict[str, Any] = {}
    first_viz_status: str = "ready"
    first_dataset_id: str = ""
    first_portfolio_samples: Optional[Dict[str, List[Dict[str, Any]]]] = None
    first_available_samples: Optional[List[str]] = None
    first_profiling_result: Optional[Dict[str, Any]] = None
    first_full_profiling_result: Optional[Dict[str, Any]] = None
    first_profiling_status: Optional[str] = None
    first_profiling_error: Optional[str] = None
    first_sample_statistics: Optional[Dict[str, Any]] = None
    first_complex_structure: Optional[Dict[str, Any]] = None

    # Track created resources so we can rollback if a later file fails
    created_session_ids: List[str] = []
    created_work_dirs: List[Path] = []
    created_tmp_uploads: List[Path] = []
    # CHANGED: in-flight background storage pushes, with what the finalizer needs
    pending_puts: List[Dict[str, Any]] = []
    # Background profile jobs: every one started (cancelled on rollback), and
    # the ones still running when the response goes out (landed afterwards).
    profile_jobs: List["asyncio.Future"] = []
    pending_profiles: List[Dict[str, Any]] = []
    total_bytes_written = 0
    try:
        # ---- Stage A: materialize each incoming file to a tmp path and expand
        # Excel workbooks into one logical CSV dataset per sheet.
        logical_items: List[Dict[str, Any]] = []
        for f_idx, up in enumerate(incoming):
            filename = up.filename or f"file_{f_idx}"
            alias = _alias_from_filename(filename)

            ext = (_filename_ext(filename) or "").lower()
            if not ext:
                ct = (up.content_type or "").lower()
                ext = (CONTENT_TYPE_TO_EXT.get(ct) or "") if ct else ""
            if ext not in SUPPORTED_UPLOAD_EXTS:
                supported_list = ", ".join(f".{e}" for e in sorted(SUPPORTED_UPLOAD_EXTS))
                label = f".{ext}" if ext else "this file type"
                raise HTTPException(
                    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    f"{label} is not supported yet; currently supported: {supported_list}.",
                )

            tmp_upload = TMP_ROOT / f"upload_{_new_id()}.{ext}"
            created_tmp_uploads.append(tmp_upload)

            try:
                file_bytes = await _read_upload_to_path_with_limit(
                    up=up,
                    dest_path=tmp_upload,
                    per_file_limit=_limits.max_file_bytes,
                )
            except HTTPException as exc:
                if exc.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE:
                    raise HTTPException(
                        exc.status_code,
                        limit_error(_limits, "file_size",
                                    getattr(up, "size", None) or _limits.max_file_bytes + 1),
                    )
                raise
            total_bytes_written += file_bytes
            if total_bytes_written > _limits.max_total_bytes:
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    limit_error(_limits, "total_size", total_bytes_written),
                )
            try:
                await up.close()
            except Exception:
                pass

            if ext in ("xls", "xlsx"):
                try:
                    exports, sheet_notes = await asyncio.to_thread(
                        export_workbook_sheets_to_csv,
                        str(tmp_upload),
                        TMP_ROOT,
                        MAX_XLSX_SHEETS,
                    )
                except Exception as e:
                    logger.exception("[upload] failed to read Excel workbook %s: %s", filename, e)
                    raise HTTPException(
                        status.HTTP_422_UNPROCESSABLE_ENTITY,
                        f"Could not read Excel workbook '{filename}': {e}",
                    )
                if not exports:
                    raise HTTPException(
                        status.HTTP_422_UNPROCESSABLE_ENTITY,
                        f"Excel workbook '{filename}' has no non-empty sheets.",
                    )
                multi_sheet = len(exports) > 1
                stem = Path(filename).stem
                # What the reader assumed, said to the person who uploaded the
                # file. These name their sheets and columns: response only,
                # never a telemetry or analytics field.
                workbook_notes = list(sheet_notes) + describe_sheet_choice(exports)
                for e_idx, exp in enumerate(exports):
                    created_tmp_uploads.append(Path(exp.csv_path))
                    sheet_alias = (
                        "".join(ch if ch.isalnum() else "_" for ch in exp.sheet_name).strip("_")
                        or f"sheet{e_idx}"
                    )
                    logical_items.append({
                        "filename": f"{stem} [{exp.sheet_name}].csv" if multi_sheet else filename,
                        "alias": f"{alias}_{sheet_alias}" if multi_sheet else alias,
                        "ext": "csv",
                        "tmp_path": Path(exp.csv_path),
                        "sheet_name": exp.sheet_name,
                        "source_workbook": filename,
                        "notes": (workbook_notes if e_idx == 0 else []) + list(exp.notes),
                    })
                if multi_sheet or sheet_notes:
                    logger.info(
                        "[upload] workbook %s expanded into %d sheet dataset(s): %s | notes=%s",
                        filename, len(exports), [e.sheet_name for e in exports], sheet_notes,
                    )
                try:
                    tmp_upload.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                logical_items.append({
                    "filename": filename,
                    "alias": alias,
                    "ext": ext,
                    "tmp_path": tmp_upload,
                    "sheet_name": None,
                    "source_workbook": None,
                    "notes": [],
                })

        # Aliases must be unique within the upload group
        seen_aliases: Dict[str, int] = {}
        for item in logical_items:
            base_alias = item["alias"]
            n = seen_aliases.get(base_alias, 0)
            seen_aliases[base_alias] = n + 1
            if n:
                item["alias"] = f"{base_alias}_{n + 1}"

        # ---- Stage B: one dataset per logical item ----
        for idx, item in enumerate(logical_items):
            filename = item["filename"]
            alias = item["alias"]
            ext = item["ext"]
            tmp_upload = item["tmp_path"]

            dsid = _new_id()
            object_name = f"{dsid}.{ext}"
            if idx == 0:
                first_dataset_id = dsid
            session_id = _new_id()
            if idx == 0 and not attach_group_sid:
                group_session_id = session_id
            else:
                assert group_session_id is not None

            # ---- keep the local file; push to storage in the background ----
            work_dir = (TMP_ROOT / session_id).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
            created_work_dirs.append(work_dir)

            input_copy = work_dir / f"input_{dsid}.{ext}"
            await asyncio.to_thread(shutil.move, str(tmp_upload), str(input_copy))  # same disk -> rename

            # Column names not on row 1 (a report title, an "exported on" line or
            # blank rows above the table) made every column "Unnamed: N" on the
            # chart axes and in chat. Make the real header row the first row, and
            # name blank header cells that have data under them (column_N).
            # Runs before the blank-column strip below; Excel sheets arrive here
            # as CSV too, so they are covered.
            if ext in ("csv", "tsv"):
                # A sheet the Excel reader could not find a header in arrives
                # with made-up column_1..column_N names; look below them.
                _no_header_note = [
                    n for n in (item["notes"] or [])
                    if isinstance(n, str) and n.startswith("No header row was found")
                ]
                _hdr = await asyncio.to_thread(
                    _fix_header_row, input_copy, bool(item.get("sheet_name")) and bool(_no_header_note)
                )
                if _hdr:
                    _hdr_notes = [
                        n for n in (item["notes"] or [])
                        if not (_hdr.get("replaced_generated") and n in _no_header_note)
                    ]
                    if _hdr["header_row"] > 1:
                        _hdr_notes.append(
                            f"Column names were found on row {_hdr['header_row']}; the "
                            f"{_hdr['header_row'] - 1} row(s) above them (title, notes or "
                            f"blank lines) were not loaded as data."
                        )
                    if _hdr["named"]:
                        _hdr_notes.append(
                            f"Named {len(_hdr['named'])} column(s) that had data but no "
                            f"header: " + ", ".join(f"column_{i}" for i in _hdr["named"]) + "."
                        )
                    item["notes"] = _hdr_notes
                    logger.info(
                        "[upload] %s header fix: header_row=%s skipped=%s named=%s",
                        dsid, _hdr["header_row"], _hdr["skipped"], _hdr["named"],
                    )

            # Trailing-comma headers create an empty unnamed column. The sampler
            # drops it, but the executor reads this file directly, so strip it
            # here or analyses see one extra column (85 vs 84).
            if ext in ("csv", "tsv"):
                _dropped_idx = await asyncio.to_thread(_strip_blank_header_columns, input_copy)
                if _dropped_idx:
                    item["notes"] = list(item["notes"] or []) + [
                        f"Removed {len(_dropped_idx)} empty column(s) with no header "
                        f"(usually a trailing comma in the header row)."
                    ]
                    logger.info("[upload] %s stripped blank columns at %s", dsid, _dropped_idx)

            put_task = asyncio.create_task(
                _put_upload_object(input_copy, object_name, dest_uri, conn)
            )
            pending_puts.append({                       # CHANGED
                "task": put_task,
                "session_id": session_id,
                "dsid": dsid,
                "object_name": object_name,
            })

            # ---- sample + schema via portfolio sampler (reads the LOCAL file) ----
            samples: List[Dict[str, Any]] = []
            schema_any: Union[List[str], Dict[str, str], Dict[str, Any]] = {}
            ddl: str = ""
            used_agent = False
            agent_res: Optional[Dict[str, Any]] = None
            portfolio_samples: Optional[Dict[str, List[Dict[str, Any]]]] = None
            available_samples: Optional[List[str]] = None
            profiling_result: Optional[Dict[str, Any]] = None
            sample_statistics: Optional[Dict[str, Any]] = None
            complex_structure: Optional[Dict[str, Any]] = None

            try:
                with _stage("sample", dsid):
                    agent_res = await asyncio.to_thread(
                        sample_with_profiling,
                        path=str(input_copy),
                        source_type=ext,
                        sample_size=DEFAULT_SAMPLE_MAX_ROWS,
                        use_ray=False,
                    )
                if isinstance(agent_res, dict) and not agent_res.get("error"):
                    schema_any = agent_res.get("schema") or []
                    ddl = agent_res.get("ddl_schema") or ""
                    profiling_result = agent_res.get("profiling_result")
                    sample_statistics = agent_res.get("sample_statistics")
                    complex_structure = agent_res.get("complex_structure")
                    portfolio_samples = agent_res.get("portfolio_samples") or {}
                    available_samples = list(portfolio_samples.keys()) if portfolio_samples else []
                    samples = (portfolio_samples.get("random_baseline") or [])[:DEFAULT_SAMPLE_MAX_ROWS]
                    used_agent = True
                else:
                    logger.warning("[upload] sample_with_profiling error: %s", (agent_res or {}).get("error"))
            except Exception as e:
                logger.warning("[upload] sample_with_profiling failed -> fallback: %s", e)

            if not used_agent:
                if ext == "csv":
                    samples_all, schema_dict, ddl = await asyncio.to_thread(
                        _fallback_sample_csv, str(input_copy)
                    )
                    samples = samples_all[:DEFAULT_SAMPLE_MAX_ROWS]
                    schema_any = schema_dict
                else:
                    raise HTTPException(
                        500,
                        f"Failed to sample .{ext} file. File type not supported or sampling failed.",
                    )

            if not samples:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    "Empty Dataset Provided — the file has no data rows.",
                )

            # Optional caller-provided schema override (before the LLM calls)
            if schema_json:
                try:
                    parsed = json.loads(schema_json)
                    if isinstance(parsed, dict):
                        schema_any = parsed
                        ddl = _ddl_from_schema(dsid, parsed)
                    elif isinstance(parsed, list):
                        schema_any = parsed
                        ddl = _ddl_from_schema(dsid, {c: "string" for c in parsed})
                except Exception:
                    logger.warning("schema_json parse failed; ignoring override", exc_info=True)

            # ---- semantic profile + model-written charts: a background job ----
            schema_list = list(schema_any.keys()) if isinstance(schema_any, dict) else list(schema_any)

            # The profile and the insight model can take minutes on a local
            # model (D5), so they run as one job raced against a soft
            # deadline. Data-only charts from the sample are built meanwhile
            # and returned if the job is still running.
            _profile_deadline = time.monotonic() + UPLOAD_PROFILE_SYNC_DEADLINE_S
            profile_job = asyncio.ensure_future(_run_upload_profile(
                dsid=dsid, schema_list=schema_list, schema_any=schema_any,
                samples=samples, sample_statistics=sample_statistics,
                used_agent=used_agent, input_path=str(input_copy), ext=ext,
            ))
            profile_jobs.append(profile_job)

            async def _data_only_viz():
                with _stage("viz_data_only", dsid):
                    return await asyncio.to_thread(
                        build_visualization_config_from_sample,
                        dataset_id=dsid, sample_rows=samples, schema=schema_any,
                        task_type="unsupervised", target_column=None,
                        total_rows=((sample_statistics or {}).get("data_shape") or {}).get("rows"),
                        use_llm=False,
                    )

            output_path = work_dir / "output.csv"
            sample_input_path = work_dir / "sample_input.csv"

            async def _write_sample():
                with _stage("sample_csv", dsid):
                    await asyncio.to_thread(_write_rows_to_csv, samples, sample_input_path)

            viz_out, csv_out = await asyncio.gather(
                _data_only_viz(), _write_sample(), return_exceptions=True
            )
            if isinstance(csv_out, Exception):
                logger.warning("Failed to write sample_input.csv; continuing: %s", csv_out)
            await asyncio.wait(
                {profile_job}, timeout=max(0.0, _profile_deadline - time.monotonic())
            )

            profile_deferred = not profile_job.done()
            if profile_deferred:
                profiling_status: Optional[str] = PROFILE_PENDING
                profiling_error: Optional[str] = None
                full_profiling_result: Optional[Dict[str, Any]] = None
                logger.info("[upload-profile] %s still running; deferred past the response", dsid)
            else:
                _job = profile_job.result()
                profiling_status = _job["profiling_status"]
                profiling_error = _job["profiling_error"]
                full_profiling_result = _job["full_profiling_result"]
                if _job["visualization_config"] is not None:
                    viz_out = _job["visualization_config"]

            if isinstance(viz_out, Exception):
                logger.warning("Failed to build visualization_config: %s", viz_out)
                visualization_config, visualization_status = {}, "error"
            else:
                visualization_config = viz_out or {}
                visualization_status = visualization_config.get("visualization_status", "ready")

            _profile_meta[dsid] = {
                "portfolio_samples": portfolio_samples,
                "available_samples": available_samples,
                "profiling_result": profiling_result,
                "full_profiling_result": full_profiling_result,
                "sample_statistics": sample_statistics,
                "complex_structure": complex_structure,
            }

            if used_agent and agent_res:
                _schedule_small_file_persistence(
                    job_key=f"upload:{session_id}",
                    dataset_id=dsid,
                    full_result=agent_res,
                    user_id=user_id,
                    source_path=str(input_copy),
                    source_type=ext,
                    full_profiling_result=full_profiling_result,
                )

            cols = list(schema_any.keys()) if isinstance(schema_any, dict) else list(schema_any)
            if not cols and samples:
                cols = list(samples[0].keys())

            # CHANGED: do NOT wait for the storage push. If it already finished,
            # use its URI now; if it already failed, fail the upload as before.
            # Otherwise the finalizer records the URI after the response.
            uri: Optional[str] = None
            if put_task.done():
                try:
                    uri = put_task.result()
                except Exception as e:
                    logger.exception("failed to upload to storage: %s", e)
                    raise HTTPException(500, "Failed to upload to storage")
            logger.info(
                "[upload-timing] %s gcs_put_state  %s",
                dsid, "done" if uri else "still running (finishing in background)",
            )

            try:
                _upload_size_bytes = input_copy.stat().st_size
            except Exception:
                _upload_size_bytes = 0

            # ---- save per-dataset session ----
            session_data = {
                "user_id": user_id,
                "dataset_id": dsid,
                "created_at": _now_iso(),
                "work_dir": str(work_dir),
                # CHANGED: data_source_location is added below only when known.
                # The key must be ABSENT (not None) while the push is running, so a
                # stale session copy merged back later can't overwrite the real URI.
                "object_name": object_name,
                "source_kind": "uploaded",
                "work_local_input": _posix(input_copy),
                "output_location": _posix(output_path),
                "sample_local_input": _posix(sample_input_path),
                "uploaded_csv_preview": _jsonify(_cap_rows(samples)),
                "uploaded_csv_columns": cols,
                "schema": _jsonify(schema_any),
                "ddl_schema": ddl,
                "assistant_id": ASSISTANT_ID,
                "input_data_type": ext,
                "file_size_bytes": _upload_size_bytes,
                "file_size_mb": round(_upload_size_bytes / (1024 * 1024), 2),
                "connection_id": connection_id,
                "filename": filename,
                "alias": alias,
                "sheet_name": item["sheet_name"],
                "source_workbook": item["source_workbook"],
                "ingest_notes": item["notes"] or None,
                "group_session_id": group_session_id or session_id,
                "visualization_config": _jsonify(visualization_config),
                "visualization_status": visualization_status,
                "portfolio_samples": None,
                "profiling_result": _jsonify(profiling_result) if profiling_result else None,
                "full_profiling_result": _jsonify(full_profiling_result) if full_profiling_result else None,
                "complex_structure": _jsonify(complex_structure) if complex_structure else None,
                "available_samples": available_samples,
                "analysis_fidelity": FIDELITY_PORTFOLIO,
                # Without these, send_message sees sample_status="" -> portfolio
                # "not ready" -> demotes to QUICK -> runs on sample_input.csv
                # (7,013 rows) instead of the full local file (91,713 rows).
                "sample_status": "full_sample" if used_agent else "quick_sample",
                # "pending" while the background profile job runs; it sets
                # full_profile, failed (with profiling_error) or skipped.
                "profiling_status": profiling_status,
                "profiling_error": profiling_error,
                "total_rows_exact": (
                    ((sample_statistics or {}).get("data_shape") or {}).get("rows")
                    if used_agent else None
                ),
                "selected_sample_name": (
                    DEFAULT_SAMPLE_NAME
                    if (available_samples and DEFAULT_SAMPLE_NAME in available_samples)
                    else (available_samples[0] if available_samples else None)
                ),
                # Picked up by avaloka_agent on the next chat turn for proactive
                # onboarding, then cleared by send_message once consumed.
                "connection_event": {
                    "kind": "upload",
                    "dataset_id": dsid,
                    "filename": filename,
                    "alias": alias,
                    "input_data_type": ext,
                    "schema_columns": cols[:32] if isinstance(cols, list) else [],
                    "created_at": _now_iso(),
                },
            }
            if uri:
                session_data["data_source_location"] = uri
                # Same rule as the session: the location is recorded only once
                # known, never as a placeholder.
                session_data["connection_event"]["data_source_location"] = uri
            with _stage("session_save", dsid):
                _saved = await save_session(session_id, session_data)
            if not _saved:
                _safe_rmtree(work_dir)
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to persist session. Please retry.")
            created_session_ids.append(session_id)

            # Attach mode keeps the chat's cookie: switching it to the new
            # session made the next message resolve only the added file.
            if idx == 0 and not attach_group_sid:
                secure_cookie_env = os.getenv("COOKIE_SECURE", "").lower()
                secure_cookie = (
                    True if secure_cookie_env == "true" else
                    False if secure_cookie_env == "false" else
                    (request.url.scheme == "https")
                )
                response.set_cookie(
                    key=COOKIE_NAME,
                    value=session_id,
                    httponly=True,
                    samesite="lax",
                    secure=secure_cookie,
                    max_age=SESSION_TTL_SECONDS,
                )

            if idx == 0 and not attach_group_sid:
                try:
                    payload = {
                        "metadata": {
                            "source": "upload",
                            "assistant_id": ASSISTANT_ID,
                            "user_id": user_id,
                            "dataset_id": dsid,
                            "session_id": session_id,
                            "schema": schema_any,
                        }
                    }
                    with _stage("lg_threads", dsid):
                        thread_res = await lg_json("POST", "/threads", json=payload)
                    thread_id = thread_res.get("thread_id") or thread_res.get("id") or ""
                    if thread_id:
                        await bind_thread_session(thread_id, session_id)
                        session_data["thread_id"] = thread_id
                        await save_session(session_id, session_data)
                except HTTPException as e:
                    logger.warning("[upload] upstream /threads create failed (soft): %s", e.detail)

            if (idx != 0 or attach_group_sid) and thread_id:
                try:
                    session_data["thread_id"] = thread_id
                    await save_session(session_id, session_data)
                except Exception:
                    pass

            dataset_ids.append(dsid)
            dataset_session_map[dsid] = session_id
            if profile_deferred:
                pending_profiles.append({
                    "job": profile_job, "session_id": session_id, "dsid": dsid,
                    "filename": filename,
                })

            datasets_out.append(
                UploadedDatasetOut(
                    dataset_id=dsid,
                    filename=filename,
                    alias=alias,
                    columns=cols,
                    rows=_jsonify(_cap_rows(samples)),
                    visualization_config=visualization_config,
                    visualization_status=visualization_status,
                    portfolio_samples=_cap_portfolio(portfolio_samples),
                    available_samples=available_samples,
                    profiling_result=full_profiling_result,
                    profiling_status=profiling_status,
                    profiling_error=profiling_error,
                    sample_statistics=sample_statistics,
                    complex_structure=complex_structure,
                    analysis_fidelity=FIDELITY_PORTFOLIO,
                    selected_sample_name=(
                        DEFAULT_SAMPLE_NAME
                        if (available_samples and DEFAULT_SAMPLE_NAME in available_samples)
                        else (available_samples[0] if available_samples else None)
                    ),
                    sheet_name=item["sheet_name"],
                    source_workbook=item["source_workbook"],
                    notes=item["notes"] or None,
                )
            )

            if idx == 0:
                first_schema_any = schema_any
                first_samples = samples
                first_ddl = ddl
                first_viz_config = visualization_config
                first_viz_status = visualization_status
                first_portfolio_samples = portfolio_samples
                first_available_samples = available_samples
                first_profiling_result = profiling_result
                first_full_profiling_result = full_profiling_result
                first_profiling_status = profiling_status
                first_profiling_error = profiling_error
                first_sample_statistics = sample_statistics
                first_complex_structure = complex_structure

    except HTTPException:
        for p in created_tmp_uploads:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        for wd in created_work_dirs:
            try:
                _safe_rmtree(wd)
            except Exception:
                pass
        for sid in created_session_ids:
            try:
                await delete_session(sid)
            except Exception:
                pass
        for job in profile_jobs:
            job.cancel()
        # CHANGED: sessions are gone now, so each finalizer sees no session
        # and deletes any object its push still uploads (no GCS orphans).
        for p in pending_puts:
            _spawn_upload_finalizer(p["task"], p["session_id"], p["dsid"], p["object_name"])
        raise

    except Exception as e:
        logger.exception("Upload failed mid-batch, rolling back: %s", e)
        for p in created_tmp_uploads:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        for wd in created_work_dirs:
            try:
                _safe_rmtree(wd)
            except Exception:
                pass
        for sid in created_session_ids:
            try:
                await delete_session(sid)
            except Exception:
                pass
        for job in profile_jobs:
            job.cancel()
        for p in pending_puts:                           # CHANGED
            _spawn_upload_finalizer(p["task"], p["session_id"], p["dsid"], p["object_name"])
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Upload failed")

    # (only runs if upload succeeded)
    assert group_session_id is not None
    try:
        group_sess = await get_session(group_session_id)
        if isinstance(group_sess, dict):
            # Merge, never replace: in attach mode the group already lists the
            # chat's earlier datasets.
            _merge_group_index(group_sess, group_session_id, dataset_session_map)
            group_sess["thread_id"] = thread_id or group_sess.get("thread_id")
            await save_session(group_session_id, group_sess)
            if attach_group_sid:
                logger.info(
                    "[upload] attached %d dataset(s) to thread %s (group %s): %s",
                    len(dataset_ids), thread_id, group_session_id, dataset_ids,
                )
    except Exception:
        logger.warning("Failed to persist group dataset index", exc_info=True)

    for p in pending_puts:
        _spawn_upload_finalizer(p["task"], p["session_id"], p["dsid"], p["object_name"])
    # Last, so no later save in this handler can overwrite what a job lands.
    for p in pending_profiles:
        _spawn_upload_profile_lander(
            p["job"], session_id=p["session_id"], dsid=p["dsid"], user_id=user_id,
            thread_id=thread_id, filename=p["filename"],
        )

    logger.info("[upload-timing] handler_total  %.2fs", time.monotonic() - _t_handler)

    if not used_files_field and len(incoming) == 1:
        return UploadResponse(
            dataset_id=first_dataset_id,
            session_id=group_session_id,
            thread_id=thread_id,
            schema=first_schema_any,
            samples=_jsonify(_cap_rows(first_samples)),
            ddl_schema=first_ddl,
            rows_sampled=len(first_samples),
            total_rows=((first_sample_statistics or {}).get("data_shape") or {}).get("rows"),
            # samples=_jsonify(_cap_rows(first_samples)),
            # ddl_schema=first_ddl,
            # rows_sampled=len(first_samples),
            visualization_config=first_viz_config,
            visualization_status=first_viz_status,
            portfolio_samples=_cap_portfolio(first_portfolio_samples),
            available_samples=first_available_samples,
            profiling_result=first_full_profiling_result,
            profiling_status=first_profiling_status,
            profiling_error=first_profiling_error,
            sample_statistics=first_sample_statistics,
            complex_structure=first_complex_structure,
            analysis_fidelity=FIDELITY_PORTFOLIO,
            selected_sample_name=(
                DEFAULT_SAMPLE_NAME
                if (first_available_samples and DEFAULT_SAMPLE_NAME in first_available_samples)
                else (first_available_samples[0] if first_available_samples else None)
            ),
            datasets=datasets_out if len(datasets_out) > 1 else None,
            notes=(datasets_out[0].notes if datasets_out else None),
            assistant_prompts=_voice_after_upload(
                sample_statistics=first_sample_statistics, profiling_result=first_full_profiling_result,
                schema=(list(first_schema_any.keys()) if isinstance(first_schema_any, dict) else first_schema_any),
                filename=(datasets_out[0].filename if datasets_out else None),
                total_rows=((first_sample_statistics or {}).get("data_shape") or {}).get("rows"),
            ),
        )

    return MultiUploadResponse(
        session_id=group_session_id,
        thread_id=thread_id,
        datasets=datasets_out,
    )


@app.post("/api/register-existing-storage", response_model=UploadResponse)
async def register_existing(
    request: Request,
    response: Response,
    body: RegisterExistingIn,
):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.",
        )
    if not storage_service.blob_store:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Storage backend not initialized.",
        )

    ext = Path(body.key).suffix.lower().lstrip(".")
    if ext not in SUPPORTED_UPLOAD_EXTS:
        supported_list = ", ".join(f".{e}" for e in sorted(SUPPORTED_UPLOAD_EXTS))
        label = f".{ext}" if ext else "this file type"
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{label} is not supported yet; currently supported: {supported_list}.",
        )

    dsid = _new_id()
    session_id = _new_id()
    _reg_filename = Path(body.key).name or body.key
    _reg_alias = _alias_from_filename(_reg_filename)

    work_dir = (TMP_ROOT / session_id).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    local_input = work_dir / f"input_{dsid}.{ext}"
    conn = await get_cloud_connection(body.connection_id)
    object_name = body.key.lstrip("/")

    # Resolve/validate the URI up front: a malformed account should 400 here,
    # not after Ray sampling and profiling have already run.
    storage_uri_normalized = await normalize_storage_uri(body.storage_uri, conn)

    # ── FAST PATH: stream sample directly from GCS without downloading full file ──
    # NOTE: We still write the streamed head to a local CSV so the downstream
    # portfolio sampler (`sample_with_profiling`) can run on a normal file path.
    # If streaming fails or the source is not GCS, we download the full object
    # to `local_input` and run the sampler on that.
    samples: List[Dict[str, Any]] = []
    schema_any: Union[List[str], Dict[str, str], Dict[str, Any]] = {}
    ddl: str = ""
    used_agent = False
    gcs_streamed = False
    file_size_bytes = 0
    portfolio_samples: Optional[Dict[str, List[Dict[str, Any]]]] = None
    available_samples: Optional[List[str]] = None
    profiling_result: Optional[Dict[str, Any]] = None
    sample_statistics: Optional[Dict[str, Any]] = None
    complex_structure: Optional[Dict[str, Any]] = None
    agent_res: Optional[Dict[str, Any]] = None
    sampling_path = local_input
    sampler_source_path = str(local_input)
    sampler_source_type = ext
    sampler_extra_args: Dict[str, Any] = {}
    persistence_source_path = str(local_input)
    full_cloud_uri: Optional[str] = None

    if (
        ext in {"csv", "tsv"}
        and (body.storage_uri.startswith("gs://") or body.storage_uri.startswith("gcs://"))
    ):
        try:
            _sa_json = conn.get("secret_key") or conn.get("service_account_json") or ""

            import gcsfs  # type: ignore

            _fs = None
            if _sa_json:
                _sa_json = _sa_json.replace("\\n", "\n").replace("\\r", "").replace("\\t", "\t")
                _sa_json = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", _sa_json)
                try:
                    _token_info = json.loads(_sa_json)
                except json.JSONDecodeError:
                    _token_info = json.loads(_sa_json, strict=False)
                _fs = await asyncio.to_thread(gcsfs.GCSFileSystem, token=_token_info)
            else:
                _fs = await asyncio.to_thread(gcsfs.GCSFileSystem)  # uses ADC

            # Preserve any bucket/URI prefix. Keeping only storage_uri's first
            # path segment drops it, so the stream 404s and falls back to a full
            # download of the whole object.
            from urllib.parse import urlparse as _urlparse

            _parsed_uri = _urlparse(body.storage_uri or "")
            bucket_name = _parsed_uri.netloc or ""
            _base_prefix = (_parsed_uri.path or "").strip("/")
            if not bucket_name:
                _conn_bucket_raw = (
                    conn.get("bucket_name") or conn.get("bucket") or ""
                ).strip().strip("/")
                _parts = _conn_bucket_raw.split("/", 1)
                bucket_name = _parts[0]
                _extra_prefix = _parts[1].strip("/") if len(_parts) > 1 else ""
                if _extra_prefix:
                    _base_prefix = (
                        f"{_extra_prefix}/{_base_prefix}".strip("/")
                        if _base_prefix
                        else _extra_prefix
                    )
            gcs_path = "/".join(p for p in (bucket_name, _base_prefix, object_name) if p)
            full_cloud_uri = f"gs://{gcs_path}"

            logger.info(
                "[register-existing] Streaming head sample from gs://%s (no full download)",
                gcs_path,
            )

            def _stream_gcs_head(fs, path, max_rows):
                # Byte guard: max_rows alone doesn't bound memory — 50k rows of
                # wide JSON/long-text cells can be hundreds of MB. Stop early
                # once the materialized head reaches the byte budget.
                max_bytes = int(os.getenv("STREAM_HEAD_MAX_BYTES", str(64 * 1024 * 1024)))
                approx_bytes = 0
                rows: List[Dict[str, Any]] = []
                with fs.open(path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if i >= max_rows:
                            break
                        rows.append(dict(row))
                        approx_bytes += sum(len(v) for v in row.values() if isinstance(v, str)) + 24 * len(row)
                        if approx_bytes >= max_bytes:
                            logger.info(
                                "[register-existing] head stream stopped at %d rows (~%d MB byte budget)",
                                len(rows), max_bytes // (1024 * 1024),
                            )
                            break
                return rows

            head_rows = await asyncio.to_thread(
                _stream_gcs_head, _fs, gcs_path, DEFAULT_SAMPLE_MAX_ROWS
            )

            if head_rows:
                streamed_path = work_dir / "streamed_head.csv"
                _write_rows_to_csv(head_rows, streamed_path)
                sampling_path = streamed_path
                gcs_streamed = True
                logger.info(
                    "[register-existing] Streamed %d rows from GCS -> %s",
                    len(head_rows),
                    streamed_path,
                )
                try:
                    file_info = await asyncio.to_thread(_fs.info, gcs_path)
                    file_size_bytes = int(file_info.get("size", 0))
                    logger.info(
                        "[register-existing] File size: %s bytes (%.2f MB)",
                        file_size_bytes,
                        file_size_bytes / (1024 * 1024),
                    )
                except Exception as e:
                    logger.warning(
                        "[register-existing] could not estimate file size: %s",
                        e,
                    )
                    file_size_bytes = 0
        except Exception as e:
            logger.warning(
                "[register-existing] GCS head stream failed; falling back to full download: %s",
                e,
            )
            gcs_streamed = False

    # ── FALLBACK: full download for S3/Azure or if GCS stream failed ──
    if not gcs_streamed:
        store_for_read, _base_prefix = await _store_from_connection_uri(
            storage_uri=body.storage_uri,
            conn=conn,
        )
        try:
            await asyncio.to_thread(store_for_read.get_file, object_name, local_input)
        except ResourceNotFoundError:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Object not found at {body.storage_uri} with key={body.key}",
            )
        except Exception as e:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Unexpected storage error: {e}",
            )
        sampling_path = local_input

        try:
            file_size_bytes = local_input.stat().st_size
            logger.info(
                "[register-existing] Downloaded file size: %s bytes (%.2f MB)",
                file_size_bytes,
                file_size_bytes / (1024 * 1024),
            )
        except Exception as e:
            logger.warning(
                "[register-existing] could not stat downloaded file: %s",
                e,
            )
            file_size_bytes = 0

    _is_complex_file = ext in {
        "pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp",
        "mp4", "mov", "mkv", "avi", "webm", "m4v",
    }
    _is_large_file = file_size_bytes >= LARGE_DATASET_THRESHOLD_BYTES and not _is_complex_file
    background_task_id: Optional[str] = None
    _sample_status: Optional[str] = None
    _profiling_status: Optional[str] = None
    _analysis_fidelity: str = FIDELITY_QUICK if _is_large_file else FIDELITY_PORTFOLIO

    if _is_large_file:
        logger.info(
            "[register-existing] File %.2f GB >= 1 GB — using quick sample and waiting for fidelity choice",
            file_size_bytes / (1024 ** 3),
        )
        try:
            agent_res = await asyncio.to_thread(
                sample_quick,
                path=str(sampling_path),
                source_type=("csv" if gcs_streamed else ext),
                # None => let should_use_ray() decide from size. This branch is
                # already >= 1 GB so it will normally still choose Ray; forcing
                # it would only rob the heuristic of its RAY_AVAILABLE check.
                use_ray=None,
            )
            if isinstance(agent_res, dict) and not agent_res.get("error"):
                schema_any = agent_res.get("schema") or []
                ddl = agent_res.get("ddl_schema") or ""
                profiling_result = agent_res.get("profiling_result")
                sample_statistics = agent_res.get("sample_statistics")
                complex_structure = agent_res.get("complex_structure")
                samples = (agent_res.get("rows") or [])[:DEFAULT_SAMPLE_MAX_ROWS]
                portfolio_samples = agent_res.get("portfolio_samples") or {}
                if not portfolio_samples and samples:
                    portfolio_samples = {"random_baseline": samples}
                available_samples = list(portfolio_samples.keys()) if portfolio_samples else []
                _sample_status = "quick_sample"
                _profiling_status = "quick_profile"
                used_agent = True
            else:
                logger.warning("[register-existing] sample_quick error: %s", (agent_res or {}).get("error"))
        except Exception as e:
            logger.warning("[register-existing] sample_quick failed -> fallback: %s", e)
    else:
        if gcs_streamed and full_cloud_uri:
            sampler_source_path = full_cloud_uri
            sampler_source_type = ext
            sampler_extra_args["cloud_credentials"] = conn
            persistence_source_path = full_cloud_uri
            # Keep a local working copy for the existing local-analysis workflow.
            try:
                store_for_read, _base_prefix = await _store_from_connection_uri(
                    storage_uri=body.storage_uri,
                    conn=conn,
                )
                await asyncio.to_thread(store_for_read.get_file, object_name, local_input)
            except ResourceNotFoundError:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND,
                    f"Object not found at {body.storage_uri} with key={body.key}",
                )
            except Exception as e:
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    f"Unexpected storage error: {e}",
                )
        else:
            sampler_source_path = str(sampling_path)
            sampler_source_type = "csv" if gcs_streamed else ext
            persistence_source_path = str(sampling_path)

        try:
            agent_res = await asyncio.to_thread(
                sample_with_profiling,
                path=sampler_source_path,
                source_type=sampler_source_type,
                sample_size=DEFAULT_SAMPLE_MAX_ROWS,
                # None => let should_use_ray() decide. Forcing True spun up a
                # local Ray cluster for every registration regardless of size --
                # for a 0.19 MB / 995-row CSV that is pure overhead, and where
                # the ray[default] extras are missing it stalls ~60s failing to
                # start the dashboard, which reads in the UI as a hung
                # "Registering...". The heuristic already requires >= 256 MB or
                # >= 50k rows before Ray is worth its startup cost.
                use_ray=None,
                **sampler_extra_args,
            )

            # If direct cloud sampling fails for a small cloud file, retry against the
            # already-downloaded local working copy so the flow still matches normal upload.
            if (
                (not isinstance(agent_res, dict) or agent_res.get("error"))
                and gcs_streamed
                and local_input.exists()
            ):
                logger.warning(
                    "[register-existing] direct cloud sample_with_profiling failed; retrying with local working copy %s",
                    local_input,
                )
                agent_res = await asyncio.to_thread(
                    sample_with_profiling,
                    path=str(local_input),
                    source_type=ext,
                    sample_size=DEFAULT_SAMPLE_MAX_ROWS,
                    use_ray=None,  # as above: size decides, not the call site
                )

            if isinstance(agent_res, dict) and not agent_res.get("error"):
                schema_any = agent_res.get("schema") or []
                ddl = agent_res.get("ddl_schema") or ""
                profiling_result = agent_res.get("profiling_result")
                sample_statistics = agent_res.get("sample_statistics")
                complex_structure = agent_res.get("complex_structure")
                portfolio_samples = agent_res.get("portfolio_samples") or {}
                available_samples = list(portfolio_samples.keys()) if portfolio_samples else []
                samples = (portfolio_samples.get("random_baseline") or [])[:DEFAULT_SAMPLE_MAX_ROWS]
                _sample_status = "full_sample"
                _profiling_status = "full_profile"
                used_agent = True
            else:
                logger.warning(
                    "[register-existing] sample_with_profiling error: %s",
                    (agent_res or {}).get("error"),
                )
        except Exception as e:
            logger.warning("[register-existing] sample_with_profiling failed -> fallback: %s", e)

    if not used_agent:
        fallback_path = str(local_input if local_input.exists() else sampling_path)
        if ext == "csv" or gcs_streamed:
            samples_all, schema_dict, ddl = await asyncio.to_thread(
                _fallback_sample_csv, fallback_path
            )
            samples = (samples_all or [])[:DEFAULT_SAMPLE_MAX_ROWS]
            schema_any = schema_dict
        else:
            _safe_rmtree(work_dir)
            raise HTTPException(
                500,
                f"Failed to sample .{ext} file. File type not supported or sampling failed.",
            )

    if not samples:
        # Client error: the upload contained no data rows (see the note above).
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Empty Dataset Provided — the file has no data rows.",
        )
    
    cols = list(schema_any.keys()) if isinstance(schema_any, dict) else list(schema_any)

    

    # Optional caller-provided schema override (before the LLM calls)
    if body.schema_json:
        try:
            parsed = json.loads(body.schema_json)
            if isinstance(parsed, dict):
                schema_any = parsed
                ddl = _ddl_from_schema(dsid, parsed)
            elif isinstance(parsed, list):
                schema_any = parsed
                ddl = _ddl_from_schema(dsid, {c: "string" for c in parsed})
        except Exception:
            logger.warning(
                "[register-existing] schema_json parse failed; ignoring override",
                exc_info=True,
            )

    # ---- semantic profile + viz config run concurrently ----
    schema_list = list(schema_any.keys()) if isinstance(schema_any, dict) else list(schema_any)

    async def _profile():
        # Large (>=1 GB) files get the profile later, after the fidelity choice.
        if not (used_agent and sample_statistics) or _is_large_file:
            return None
        with _stage("profile_full", dsid):
            res = await asyncio.to_thread(
                profile_full, schema=schema_list, sample_rows=samples,
                sample_statistics=sample_statistics,
            )
        # profile_full returns its error in the profile's place; that is no profile.
        return usable_full_profile(res)

        # The viz step waits for the profile so insights use its domain context.
    prof_task = asyncio.ensure_future(_profile())

    async def _viz():
        try:
            prof_for_viz = await asyncio.shield(prof_task)
        except Exception:
            prof_for_viz = None
        with _stage("viz", dsid):
            return await asyncio.to_thread(
                build_visualization_config_from_sample,
                dataset_id=dsid, sample_rows=samples, schema=schema_any,
                task_type="unsupervised", target_column=None,
                # Sub-1 GB objects are downloaded to local_input above; >= 1 GB
                # keeps sample-based charts (only a streamed head is local).
                full_data_path=(str(local_input)
                                if (not _is_large_file and local_input.exists()) else None),
                full_data_type=ext,
                total_rows=(((sample_statistics or {}).get("data_shape") or {}).get("rows")
                            if not _is_large_file else None),
                profiling_result=prof_for_viz,
            )

    prof_out, viz_out = await asyncio.gather(prof_task, _viz(), return_exceptions=True)

    full_profiling_result: Optional[Dict[str, Any]] = None
    if isinstance(prof_out, Exception):
        logger.warning("[register-existing] profile_full failed (non-fatal): %s", prof_out)
    else:
        full_profiling_result = prof_out

    if isinstance(viz_out, Exception):
        logger.warning("Failed to build visualization_config (register-existing): %s", viz_out)
        visualization_config, visualization_status = {}, "error"
    else:
        visualization_config = viz_out or {}
        visualization_status = visualization_config.get("visualization_status", "ready")

    _profile_meta[dsid] = {
        "portfolio_samples": portfolio_samples,
        "available_samples": available_samples,
        "profiling_result": profiling_result,
        "full_profiling_result": full_profiling_result,
        "sample_statistics": sample_statistics,
        "complex_structure": complex_structure,
    }

    if used_agent and agent_res and not _is_large_file:
        _schedule_small_file_persistence(
            job_key=f"register:{session_id}",
            dataset_id=dsid,
            full_result=agent_res,
            user_id=user_id,
            source_path=persistence_source_path,
            source_type=ext,
            full_profiling_result=full_profiling_result,
        )

    sample_input_path = work_dir / "sample_input.csv"
    try:
        _write_rows_to_csv(samples, sample_input_path)
    except Exception:
        logger.warning(
            "[register-existing] Failed to write sample_input.csv; continuing",
            exc_info=True,
        )

    output_path = work_dir / "output.csv"

    # storage_uri_normalized = body.storage_uri.rstrip("/")
    # object_key = body.key.lstrip("/")
    object_key = body.key.lstrip("/")

    session_data = {
        "user_id": user_id,
        "dataset_id": dsid,
        "work_dir": str(work_dir),
        "data_source_location": storage_uri_normalized,
        "object_name": object_key,
        # The source object pre-exists in the customer's bucket; Avaloka only
        # references it, so deleting the dataset must NOT delete it.
        "source_kind": "registered",
        # Prefer the FULL local copy (downloaded above for <1GB objects), matching
        # the upload path — otherwise entire_dataset "full file" runs silently
        # execute on the 1,000-row streamed head. The head is only the fallback
        # for >=1GB objects, whose entire_dataset runs go to the cluster instead.
        "work_local_input": _posix(
            local_input if Path(local_input).exists() else sampling_path
        ),
        "output_location": _posix(output_path),
        "sample_local_input": _posix(sample_input_path),
        "uploaded_csv_preview": _jsonify(samples),
        "uploaded_csv_columns": cols,
        "schema": _jsonify(schema_any),
        "ddl_schema": ddl,
        "assistant_id": ASSISTANT_ID,
        "input_data_type": ext,
        "filename": _reg_filename,
        "alias": _reg_alias,
        "visualization_config": _jsonify(visualization_config),
        "visualization_status": visualization_status,
        "connection_id": body.connection_id,  # needed for ray secret injection
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        # Persist rich preview artifacts in Redis session so cloud small-file
        # registration survives the redirect to /preview without relying on
        # process-local _profile_meta.
        "portfolio_samples": _jsonify(portfolio_samples) if portfolio_samples else None,
        "profiling_result": _jsonify(profiling_result) if profiling_result else None,
        "full_profiling_result": _jsonify(full_profiling_result) if full_profiling_result else None,
        "sample_statistics": _jsonify(sample_statistics) if sample_statistics else None,
        "complex_structure": _jsonify(complex_structure) if complex_structure else None,
        "sample_status": _sample_status,
        "profiling_status": _profiling_status,
        "available_samples": available_samples,
        "analysis_fidelity": _analysis_fidelity,
        "selected_sample_name": (
            DEFAULT_SAMPLE_NAME
            if (available_samples and DEFAULT_SAMPLE_NAME in available_samples)
            else (available_samples[0] if available_samples else None)
        ),
        "portfolio_generation_status": ("not_requested" if _is_large_file else "ready"),
        "requires_fidelity_choice": bool(_is_large_file),
        "fidelity_prompt": (_build_fidelity_prompt(file_size_bytes) if _is_large_file else None),
        "estimated_runtime_hint": (_estimated_runtime_hint(file_size_bytes) if _is_large_file else None),
        "fidelity_prompt_shown": bool(_is_large_file),
        # Picked up by avaloka_agent on the next chat turn for proactive
        # onboarding, then cleared by send_message once consumed.
        "connection_event": {
            "kind": "cloud_storage",
            "dataset_id": dsid,
            "connection_id": body.connection_id,
            "input_data_type": ext,
            "data_source_location": storage_uri_normalized,
            "file_size_mb": round((file_size_bytes or 0) / (1024 * 1024), 2),
            "schema_columns": (cols[:32] if isinstance(cols, list) else []),
            "is_large_file": bool(_is_large_file),
            "created_at": _now_iso(),
        },
    }

    ok = await save_session(session_id, session_data)
    if not ok:
        _safe_rmtree(work_dir)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to persist session. Please retry.",
        )

    # NOTE: for large files, portfolio/full-profile generation is now explicit user choice.
    # Do not auto-schedule here; chat intent ("use portfolio") triggers scheduling.
    if _is_large_file and used_agent:
        logger.info(
            "[register-existing] Large dataset quick preview ready; waiting for user choice to schedule portfolio build."
        )

    secure_cookie_env = os.getenv("COOKIE_SECURE", "").lower()
    secure_cookie = (
        True
        if secure_cookie_env == "true"
        else False
        if secure_cookie_env == "false"
        else (request.url.scheme == "https")
    )
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=secure_cookie,
        max_age=SESSION_TTL_SECONDS,
    )

    thread_id = ""
    try:
        payload = {
            "metadata": {
                "source": "upload",
                "assistant_id": ASSISTANT_ID,
                "user_id": user_id,
                "dataset_id": dsid,
                "session_id": session_id,
                "schema": schema_any,
            }
        }
        thread_res = await lg_json("POST", "/threads", json=payload)
        thread_id = thread_res.get("thread_id") or thread_res.get("id") or ""
        if thread_id:
            await bind_thread_session(thread_id, session_id)
            session_data["thread_id"] = thread_id
            await save_session(session_id, session_data)
    except HTTPException as e:
        logger.warning(
            "[register-existing] upstream /threads create failed (soft): %s",
            e.detail,
        )

    return UploadResponse(
        dataset_id=dsid,
        session_id=session_id,
        thread_id=thread_id,
        schema=schema_any,
        samples=_jsonify(_cap_rows(samples)),
        ddl_schema=ddl,
        rows_sampled=len(samples),
        visualization_config=visualization_config,
        visualization_status=visualization_status,
        file_size_bytes=file_size_bytes,
        file_size_mb=round(file_size_bytes / (1024 * 1024), 2),
        portfolio_samples=_cap_portfolio(portfolio_samples),
        available_samples=available_samples,
        profiling_result=full_profiling_result,
        sample_statistics=sample_statistics,
        complex_structure=complex_structure,
        sample_status=_sample_status,
        profiling_status=_profiling_status,
        background_task_id=background_task_id,
        analysis_fidelity=_analysis_fidelity,
        selected_sample_name=(
            DEFAULT_SAMPLE_NAME
            if (available_samples and DEFAULT_SAMPLE_NAME in available_samples)
            else (available_samples[0] if available_samples else None)
        ),
        requires_fidelity_choice=bool(_is_large_file),
        fidelity_prompt=(_build_fidelity_prompt(file_size_bytes) if _is_large_file else None),
        estimated_runtime_hint=(_estimated_runtime_hint(file_size_bytes) if _is_large_file else None),
        assistant_prompts=_voice_after_upload(
            sample_statistics=sample_statistics, profiling_result=full_profiling_result,
            schema=cols, filename=Path(object_name).name if object_name else None,
            total_rows=((sample_statistics or {}).get("data_shape") or {}).get("rows"),
        ),
    )



class RegisterFolderIn(BaseModel):
    storage_uri: str          # bucket root, e.g. gs://my-bucket
    folder: str               # bucket-relative prefix from /buckets/list
    connection_id: str
    schema_json: Optional[str] = None
    table_type: Optional[str] = None   # optional UI-supplied override


@app.post("/api/register-existing-folder", response_model=UploadResponse)
async def register_existing_folder(
    request: Request,
    response: Response,
    body: RegisterFolderIn,
):
    """Register a whole folder as ONE dataset for partitioned tables
    (Parquet directory / Hive-partitioned / Iceberg). Unlike register-existing,
    which streams one object, this classifies the folder and hands Daft the whole
    tree via a glob (Parquet) or metadata.json (Iceberg)."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Session storage (Redis) is not available.")
    if not storage_service.blob_store:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Storage backend not initialized.")

    conn = await get_cloud_connection(body.connection_id)
    if not connection_belongs_to_user(conn, user_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Cloud connection not found")

    folder_uri = f"{body.storage_uri.rstrip('/')}/{body.folder.strip('/')}"
    folder_uri = await normalize_storage_uri(folder_uri, conn)

    # List once (keys relative to folder root) → classify + size.
    store, _base_prefix = await _store_from_connection_uri(folder_uri, conn)
    try:
        listing = await asyncio.to_thread(lambda: list(store.list("")))
    except Exception as e:
        logger.exception("[register-folder] listing failed for %s", folder_uri)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Failed to list folder: {e}")

    rel_keys = [k for (k, _s, _u) in listing]
    total_bytes = sum(int(s or 0) for (_k, s, _u) in listing)

    detection = detect_folder_table_type(rel_keys)
    table_type = body.table_type or detection.get("table_type")
    if table_type in (None, "unknown", "empty"):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Not a recognized Parquet or Iceberg table — open the folder and select files instead.",
        )

    if table_type == "iceberg":
        metadata_key = detection.get("metadata_key")
        if not metadata_key:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Iceberg table detected but no metadata.json was found in the folder.",
            )
        sampler_source_path = folder_uri
        sampler_source_type = "iceberg"
        sampler_extra: Dict[str, Any] = {
            "cloud_credentials": conn,
            "metadata_uri": f"{folder_uri.rstrip('/')}/{metadata_key.lstrip('/')}",
        }
    else:  # hive_parquet | parquet_dir
        sampler_source_path = f"{folder_uri.rstrip('/')}/{detection.get('glob', '**/*.parquet')}"
        sampler_source_type = "parquet"
        sampler_extra = {
            "cloud_credentials": conn,
            "hive_partitioning": bool(detection.get("hive_partitioning")),
        }

    dsid = _new_id()
    session_id = _new_id()
    work_dir = (TMP_ROOT / session_id).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)


    # Direct cloud sampling (no head-stream: you can't head-stream a partitioned
    # table as CSV). Ray parallelises the multi-file read, but only pays off for
    # genuinely large folders — small partitioned tables read faster on Daft's
    # native runner and skip cluster startup.
    _use_ray = total_bytes >= LARGE_DATASET_THRESHOLD_BYTES
    try:
        agent_res = await asyncio.to_thread(
            sample_with_profiling,
            path=sampler_source_path,
            source_type=sampler_source_type,
            sample_size=DEFAULT_SAMPLE_MAX_ROWS,
            use_ray=_use_ray,
            **sampler_extra,
        )


    except Exception as e:
        _safe_rmtree(work_dir)
        logger.exception("[register-folder] sample_with_profiling raised for %s", folder_uri)
        raise HTTPException(500, f"Failed to read {table_type} table: {e}")

    if not isinstance(agent_res, dict) or agent_res.get("error"):
        _safe_rmtree(work_dir)
        raise HTTPException(500, f"Failed to read {table_type} table: {(agent_res or {}).get('error')}")

    schema_any: Union[List[str], Dict[str, str], Dict[str, Any]] = agent_res.get("schema") or []
    ddl = agent_res.get("ddl_schema") or ""
    profiling_result = agent_res.get("profiling_result")
    sample_statistics = agent_res.get("sample_statistics")
    portfolio_samples = agent_res.get("portfolio_samples") or {}
    available_samples = list(portfolio_samples.keys()) if portfolio_samples else []
    samples = (portfolio_samples.get(DEFAULT_SAMPLE_NAME) or [])[:DEFAULT_SAMPLE_MAX_ROWS]

    if not samples:
        _safe_rmtree(work_dir)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Empty table (no rows sampled).")

    # Optional caller-provided schema override
    if body.schema_json:
        try:
            parsed = json.loads(body.schema_json)
            if isinstance(parsed, dict):
                schema_any = parsed
                ddl = _ddl_from_schema(dsid, parsed)
            elif isinstance(parsed, list):
                schema_any = parsed
                ddl = _ddl_from_schema(dsid, {c: "string" for c in parsed})
        except Exception:
            logger.warning("[register-folder] schema_json parse failed; ignoring override", exc_info=True)

    # Semantic profiling (best-effort)
    full_profiling_result: Optional[Dict[str, Any]] = None
    if sample_statistics:
        try:
            schema_list = list(schema_any.keys()) if isinstance(schema_any, dict) else list(schema_any)
            prof_res = await asyncio.to_thread(
                profile_full,
                schema=schema_list,
                sample_rows=samples,
                sample_statistics=sample_statistics,
            )
            full_profiling_result = usable_full_profile(prof_res)
        except Exception as e:
            logger.warning("[register-folder] profile_full failed (non-fatal): %s", e)

    _profile_meta[dsid] = {
        "portfolio_samples": portfolio_samples,
        "available_samples": available_samples,
        "profiling_result": profiling_result,
        "full_profiling_result": full_profiling_result,
        "sample_statistics": sample_statistics,
    }

    _schedule_small_file_persistence(
        job_key=f"register-folder:{session_id}",
        dataset_id=dsid,
        full_result=agent_res,
        user_id=user_id,
        source_path=sampler_source_path,
        source_type=sampler_source_type,
        full_profiling_result=full_profiling_result,
    )

    cols = list(schema_any.keys()) if isinstance(schema_any, dict) else list(schema_any)
    if not cols and samples:
        cols = list(samples[0].keys())

    sample_input_path = work_dir / "sample_input.csv"
    try:
        _write_rows_to_csv(samples, sample_input_path)
    except Exception:
        logger.warning("[register-folder] Failed to write sample_input.csv; continuing", exc_info=True)
    output_path = work_dir / "output.csv"

    try:
        visualization_config = build_visualization_config_from_sample(
            dataset_id=dsid,
            sample_rows=samples,
            schema=schema_any,
            task_type="unsupervised",
            target_column=None,
            profiling_result=full_profiling_result,
        )
        visualization_status = visualization_config.get("visualization_status", "ready")
    except Exception as e:
        logger.warning("[register-folder] Failed to build visualization_config: %s", e)
        visualization_config = {}
        visualization_status = "error"

    session_data = {
        "user_id": user_id,
        "dataset_id": dsid,
        "work_dir": str(work_dir),
        "data_source_location": folder_uri,
        "object_name": None,
        # The source folder pre-exists in the customer's bucket; Avaloka only
        # references it, so deleting the dataset must NOT delete it.
        "source_kind": "registered",
        # sample_input.csv is the local working copy used by local analysis; the
        # real source for cloud/Ray reads is folder_read_path below.
        "work_local_input": _posix(sample_input_path),
        "output_location": _posix(output_path),
        "sample_local_input": _posix(sample_input_path),
        "uploaded_csv_preview": _jsonify(samples),
        "uploaded_csv_columns": cols,
        "schema": _jsonify(schema_any),
        "ddl_schema": ddl,
        "assistant_id": ASSISTANT_ID,
        "input_data_type": ("iceberg" if table_type == "iceberg" else "parquet"),
        "connection_id": body.connection_id,
        # Partitioned-table provenance so downstream reads reconstruct the exact
        # source instead of treating folder_uri as a single object.
        "folder_table_type": table_type,          # iceberg | hive_parquet | parquet_dir
        "folder_read_path": sampler_source_path,   # exact glob / URI Daft must reuse
        "iceberg_metadata_uri": sampler_extra.get("metadata_uri"),
        "hive_partitioning": bool(sampler_extra.get("hive_partitioning")),
        "file_size_bytes": total_bytes,
        "file_size_mb": round(total_bytes / (1024 * 1024), 2),
        "visualization_config": _jsonify(visualization_config),
        "visualization_status": visualization_status,
        "portfolio_samples": _jsonify(portfolio_samples) if portfolio_samples else None,
        "profiling_result": _jsonify(profiling_result) if profiling_result else None,
        "full_profiling_result": _jsonify(full_profiling_result) if full_profiling_result else None,
        "sample_statistics": _jsonify(sample_statistics) if sample_statistics else None,
        "available_samples": available_samples,
        "analysis_fidelity": FIDELITY_PORTFOLIO,
        "sample_status": "full_sample",
        "profiling_status": "full_profile" if full_profiling_result else None,
        "selected_sample_name": (
            DEFAULT_SAMPLE_NAME
            if (available_samples and DEFAULT_SAMPLE_NAME in available_samples)
            else (available_samples[0] if available_samples else None)
        ),
    }

    if not await save_session(session_id, session_data):
        _safe_rmtree(work_dir)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to persist session. Please retry.")

    secure_cookie_env = os.getenv("COOKIE_SECURE", "").lower()
    secure_cookie = (
        True if secure_cookie_env == "true"
        else False if secure_cookie_env == "false"
        else (request.url.scheme == "https")
    )
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=secure_cookie,
        max_age=SESSION_TTL_SECONDS,
    )

    thread_id = ""
    try:
        payload = {
            "metadata": {
                "source": "register-folder",
                "assistant_id": ASSISTANT_ID,
                "user_id": user_id,
                "dataset_id": dsid,
                "session_id": session_id,
                "schema": schema_any,
            }
        }
        thread_res = await lg_json("POST", "/threads", json=payload)
        thread_id = thread_res.get("thread_id") or thread_res.get("id") or ""
        if thread_id:
            await bind_thread_session(thread_id, session_id)
            session_data["thread_id"] = thread_id
            await save_session(session_id, session_data)
    except HTTPException as e:
        logger.warning("[register-folder] upstream /threads create failed (soft): %s", e.detail)

    return UploadResponse(
        dataset_id=dsid,
        session_id=session_id,
        thread_id=thread_id,
        schema=schema_any,
        samples=_jsonify(samples),
        ddl_schema=ddl,
        rows_sampled=len(samples),
        visualization_config=visualization_config,
        visualization_status=visualization_status,
        file_size_bytes=total_bytes,
        file_size_mb=round(total_bytes / (1024 * 1024), 2),
        portfolio_samples=portfolio_samples,
        available_samples=available_samples,
        profiling_result=full_profiling_result,
        sample_statistics=sample_statistics,
        analysis_fidelity=FIDELITY_PORTFOLIO,
        selected_sample_name=(
            DEFAULT_SAMPLE_NAME
            if (available_samples and DEFAULT_SAMPLE_NAME in available_samples)
            else (available_samples[0] if available_samples else None)
        ),
        assistant_prompts=_voice_after_upload(
            sample_statistics=sample_statistics, profiling_result=full_profiling_result, schema=cols,
            total_rows=((sample_statistics or {}).get("data_shape") or {}).get("rows"),
        ),
    )





# -------------------------------------------------------------------
# Datasets attached to a chat ("+ Add" under the chat box)
# -------------------------------------------------------------------


async def _resolve_thread_group_for_user(
    thread_id: str, user_id: str
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """The dataset group (session id, session) a chat uses, if the caller owns it.

    send_message binds a thread to its GROUP session, whose dataset_ids /
    dataset_session_map decide which datasets the chat can see.
    """
    if not thread_id:
        return None, None
    sid = await get_thread_session(thread_id)
    sess = await get_session(sid) if sid else None
    if not isinstance(sess, dict) or sess.get("user_id") != user_id:
        return None, None
    gsid = sess.get("group_session_id") or sid
    if gsid != sid:
        gsess = await get_session(gsid)
        if isinstance(gsess, dict) and gsess.get("user_id") == user_id:
            return gsid, gsess
    return sid, sess


def _merge_group_index(
    group_sess: Dict[str, Any], group_sid: str, new_map: Dict[str, str]
) -> List[str]:
    """Add datasets to a group's index in place, keeping the ones already there.

    A group created by a single upload, a cloud registration or a DB sample may
    have no dataset_ids yet; its own dataset is then the first member.
    """
    ids = _maybe_json_load(group_sess.get("dataset_ids")) or group_sess.get("dataset_ids") or []
    ids = list(ids) if isinstance(ids, list) else []        # copy: never mutate a shared list
    smap = (_maybe_json_load(group_sess.get("dataset_session_map"))
            or group_sess.get("dataset_session_map") or {})
    smap = dict(smap) if isinstance(smap, dict) else {}
    own = group_sess.get("dataset_id")
    if own and own not in ids:
        ids.insert(0, own)
    if own:
        smap.setdefault(own, group_sid)
    for dsid, ds_sid in new_map.items():
        if dsid not in ids:
            ids.append(dsid)
        smap[dsid] = ds_sid
    group_sess["dataset_ids"] = ids
    group_sess["dataset_session_map"] = smap
    return ids


def _group_selected_ids(group_sess: Dict[str, Any]) -> List[str]:
    """The datasets a chat uses when a message does not name any.

    Every dataset of the group except the ones the user switched off
    (excluded_dataset_ids). Stored as exclusions so a sheet or file added later
    ("+ Add") is in use straight away. Never empty: if every dataset was
    excluded, all of them are used.
    """
    ids = _maybe_json_load(group_sess.get("dataset_ids")) or group_sess.get("dataset_ids") or []
    ids = list(dict.fromkeys(ids)) if isinstance(ids, list) else []
    excluded = _maybe_json_load(group_sess.get("excluded_dataset_ids")) or group_sess.get("excluded_dataset_ids") or []
    excluded = set(excluded) if isinstance(excluded, list) else set()
    selected = [d for d in ids if d not in excluded]
    return selected or ids


async def _group_datasets_out(group_sess: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The chips for a chat: one entry per dataset in its group."""
    ids = group_sess.get("dataset_ids") or []
    smap = group_sess.get("dataset_session_map") or {}
    selected = set(_group_selected_ids(group_sess))
    out: List[Dict[str, Any]] = []
    for dsid in ids:
        ds_sid = smap.get(dsid)
        ds_sess = await get_session(ds_sid) if ds_sid else None
        ds_sess = ds_sess if isinstance(ds_sess, dict) else {}
        out.append({
            "dataset_id": dsid,
            "filename": ds_sess.get("filename"),
            "alias": ds_sess.get("alias"),
            "sheet_name": ds_sess.get("sheet_name"),
            # Sheets of one workbook share this; separate files have None.
            "source_workbook": ds_sess.get("source_workbook"),
            "rows": ds_sess.get("total_rows_exact"),
            "selected": dsid in selected,
            "available": bool(ds_sess),
        })
    return out


class AttachDatasetsIn(BaseModel):
    # Either list, or both. The "+ Add" picker lists ANALYSES (Supabase
    # analyses rows), so it sends analysis_ids; every dataset of each picked
    # analysis is added to this chat.
    dataset_ids: List[str] = Field(default_factory=list)
    analysis_ids: List[str] = Field(default_factory=list)


async def _datasets_of_analysis(analysis_id: str, user_id: str) -> Dict[str, str]:
    """dataset_id -> session_id for every dataset of an analysis the caller owns.

    An analysis points at its group session; a multi-file or multi-sheet
    analysis has several datasets there. A session evicted from Redis is
    rebuilt from the durable analyses row first.
    """
    row = await _resolve_session_from_aid(analysis_id, user_id)   # 503 if Supabase is down
    if not row or not row.get("session_id"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Analysis {analysis_id} not found for this user")
    sid = row["session_id"]
    sess = await get_session(sid)
    if not (isinstance(sess, dict) and sess.get("user_id")):
        sess = await _rehydrate_session_from_analysis(
            {"id": analysis_id, "thread_id": row.get("thread_id")}, sid
        )
    if not isinstance(sess, dict) or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"The data for analysis {analysis_id} is no longer available")
    index = dict(sess)                                # do not modify the other analysis
    ids = _merge_group_index(index, sid, {})
    smap = index["dataset_session_map"]
    out: Dict[str, str] = {}
    for dsid in ids:
        ds_sid = smap.get(dsid) or await find_session_by_dataset_for_user(dsid, user_id)
        if ds_sid:
            out[dsid] = ds_sid
    if not out:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Analysis {analysis_id} has no datasets")
    return out


@app.get("/threads/{thread_id}/datasets", tags=["threads"])
async def list_thread_datasets(request: Request, thread_id: str):
    """The datasets this chat can use (the chips under the chat box)."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    gsid, gsess = await _resolve_thread_group_for_user(thread_id, user_id)
    if not gsid or gsess is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found for this user")
   
    _merge_group_index(gsess, gsid, {})      # read-only normalize of older groups
    return {"thread_id": thread_id, "datasets": await _group_datasets_out(gsess),
            "selected_dataset_ids": _group_selected_ids(gsess)}


class SelectDatasetsIn(BaseModel):
    # The datasets of this chat to use from now on (at least one).
    dataset_ids: List[str] = Field(default_factory=list)


@app.put("/threads/{thread_id}/selected-datasets", tags=["threads"])
async def select_thread_datasets(request: Request, thread_id: str, body: SelectDatasetsIn):
    """Choose which of the chat's datasets are in use.

    Upload of several files: the files the user ticked in the review step.
    Chips under the chat: switching one on or off. A workbook's sheets are all
    in use until the user switches one off. Messages that name no dataset_ids
    use this selection; a message that names dataset_ids still wins.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Session storage (Redis) is not available.")
    gsid, gsess = await _resolve_thread_group_for_user(thread_id, user_id)
    if not gsid or gsess is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found for this user")

    ids = _merge_group_index(gsess, gsid, {})
    wanted = [d for d in dict.fromkeys(body.dataset_ids or []) if d]
    if not wanted:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Select at least one dataset.")
    unknown = [d for d in wanted if d not in ids]
    if unknown:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"Not datasets of this chat: {', '.join(unknown)}")

    gsess["excluded_dataset_ids"] = [d for d in ids if d not in set(wanted)]
    if not await save_session(gsid, gsess):
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            "Failed to save the chat's dataset selection.")
    logger.info("[select] thread=%s group=%s selected=%d of %d",
                thread_id, gsid, len(wanted), len(ids))
    return {"thread_id": thread_id, "datasets": await _group_datasets_out(gsess),
            "selected_dataset_ids": _group_selected_ids(gsess)}


@app.post("/threads/{thread_id}/datasets", tags=["threads"])
async def attach_datasets_to_thread(request: Request, thread_id: str, body: AttachDatasetsIn):
    """Add existing datasets / analyses to this chat, keeping the ones it has.

    "+ Add" -> pick an existing analysis or dataset. New files go through
    /api/upload with attach_to_thread_id instead.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Session storage (Redis) is not available.")

    gsid, gsess = await _resolve_thread_group_for_user(thread_id, user_id)
    if not gsid or gsess is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found for this user")

    requested = [d for d in dict.fromkeys(body.dataset_ids or []) if d]
    analyses = [a for a in dict.fromkeys(body.analysis_ids or []) if a]
    if not requested and not analyses:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Provide at least one dataset_id or analysis_id.")

    new_map: Dict[str, str] = {}
    for dsid in requested:
        ds_sid = await find_session_by_dataset_for_user(dsid, user_id)
        if not ds_sid:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Dataset {dsid} not found for this user")
        new_map[dsid] = ds_sid
    for aid in analyses:
        new_map.update(await _datasets_of_analysis(aid, user_id))

    if len(new_map) > MAX_UPLOAD_FILES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"Too many datasets. Max is {MAX_UPLOAD_FILES}.")

    ids = _merge_group_index(gsess, gsid, new_map)
    if not await save_session(gsid, gsess):
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            "Failed to save the chat's datasets.")
    for ds_sid in new_map.values():
        try:
            await refresh_session_ttl(ds_sid)
        except Exception:
            pass

    logger.info("[attach] thread=%s group=%s added=%s (from analyses=%s) total=%d",
                thread_id, gsid, list(new_map), analyses, len(ids))
    return {"thread_id": thread_id, "dataset_ids": ids,
            "datasets": await _group_datasets_out(gsess)}
# -------------------------------------------------------------------
# Threads
# -------------------------------------------------------------------


@app.post("/threads", response_model=ThreadOut)
async def create_thread(request: Request, body: ThreadCreateIn):
    # Same check as every other /threads route. Without it an anonymous caller
    # (model code in the executor could reach api:9000) created LangGraph
    # threads and bound them to whatever X-Avaloka-Session it named.
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    sid = _resolve_session_id(request, None)
    if sid:
        sess = await get_session(sid)
        if sess and sess.get("user_id") != user_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Session belongs to another user")
    data = await lg_json(
        "POST",
        "/threads",
        json={"metadata": {"assistant_id": ASSISTANT_ID, **(body.metadata or {})}},
    )
    thread_id = data.get("thread_id") or data.get("id")
    if not thread_id:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "LangGraph did not return a thread_id",
        )
    _ensure_thread_local(thread_id, body.metadata or {})
    if sid:
        await bind_thread_session(thread_id, sid)
    return ThreadOut(
        thread_id=thread_id,
        created_at=_now_iso(),
        metadata=body.metadata or {},
    )


@app.get("/threads", response_model=List[ThreadOut])
async def list_threads(
    request: Request,
    limit: int = 50,
    q: Optional[str] = None,
    page_token: Optional[str] = None,
):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    payload: Dict[str, Any] = {"limit": max(1, min(limit, 200))}
    if q:
        payload["query"] = q
    if page_token:
        payload["page_token"] = page_token

    res = await lg_json("POST", "/threads/search", json=payload)
    if isinstance(res, list):
        items = res
    elif isinstance(res, dict):
        items = res.get("items") or res.get("threads") or []
    else:
        items = []

    out: List[ThreadOut] = []
    for it in items:
        if isinstance(it, dict):
            tid = (
                it.get("thread_id")
                or it.get("id")
                or it.get("thread")
                or it.get("uuid")
            )
            created = it.get("created_at") or _now_iso()
            meta = it.get("metadata") or {}
        else:
            tid = str(it)
            created = _now_iso()
            meta = {}
        if not tid:
            continue
        out.append(
            ThreadOut(
                thread_id=tid,
                created_at=created,
                metadata=meta,
            )
        )

    # Only return threads owned by the caller; threads without a resolvable
    # session binding are dropped rather than leaked to everyone.
    async def _owned_by_caller(t: ThreadOut) -> bool:
        sid = await get_thread_session(t.thread_id)
        if not sid:
            return False
        sess = await get_session(sid)
        return bool(sess) and sess.get("user_id") == user_id

    owned_flags = await asyncio.gather(*(_owned_by_caller(t) for t in out))
    return [t for t, owned in zip(out, owned_flags) if owned]

async def task_sse_iter(
    task_id: str,
    redbeat_key: str,
    session: Optional[Dict[str, Any]] = None,
):
    try:
        entry = AvalokaEntry.from_key(redbeat_key, celery_app)
        task_metadata = _scheduled_task_metadata_from_entry(entry)
    except Exception:
        entry = None
        task_metadata = scheduled_task_metadata({})

    task = AsyncResult(task_id, app=celery_app)
    current_state = task.state
    if current_state not in ("FAILURE", "SUCCESS", "REVOKED"):
        yield "event: message\ndata: " + json.dumps({
            "status": task.state,
            "message": task_metadata["running_message"],
            **task_metadata,
        }) + "\n\n"
    
    polls = 0
    ended_without_result: Optional[str] = None
    while (task := AsyncResult(task_id, app=celery_app)).state not in ("FAILURE", "SUCCESS", "REVOKED"):
        if task.state != current_state:
            yield "event: message\ndata: " + json.dumps({
                "status": task.state,
                "message": task_metadata["running_message"],
                **task_metadata,
            }) + "\n\n"
            current_state = task.state
        if polls % _SSE_DURABLE_CHECK_EVERY == 0:
            ended_without_result = await asyncio.to_thread(_run_finished_state, task_id)
            if ended_without_result:
                break
        polls += 1
        await sleep(1)

    if entry is None:
        entry = AvalokaEntry.from_key(redbeat_key, celery_app)
    last_run_at = AvalokaScheduler.get_last_run_at(redbeat_key, generate_key=False)
    next_due_at = (
        entry.schedule.remaining_estimate(last_run_at).total_seconds() + 2
        if entry.max_runs > 0 and entry.total_run_count < entry.max_runs
        else -1
    )

    if ended_without_result:
        # The run is over but its result is no longer stored.
        yield "event: message\ndata: " + json.dumps({
            "status": "SUCCESS" if ended_without_result == "SUCCESS" else "FAILED",
            "message": "The run finished; its stored result is no longer available.",
            "next_due_at": next_due_at, **task_metadata,
        }) + "\n\n"
    elif task.state == "REVOKED":
        yield "event: message\ndata: " + json.dumps({"status": "CANCELLED", "message": _SCHEDULED_CANCELLED_TEXT, "next_due_at": next_due_at, **task_metadata}) + "\n\n"
    elif task.failed():
        # The Celery traceback is the worker's log, never the user's message.
        _failed_msg = _scheduled_failure_text(task, task_metadata)
        _details = await asyncio.to_thread(_failed_run_details, task, task_id, task_metadata)
        yield "event: message\ndata: " + json.dumps({"status": "FAILED", "message": _failed_msg, "error": _failed_msg, "traceback": _failed_msg, "next_due_at": next_due_at, **task_metadata, **_details}) + "\n\n"
    elif _scheduled_run_failed(task.result, task_metadata):
        _failed_msg = _scheduled_task_completion_message(task.result or {}, [], task_metadata)
        _details = await asyncio.to_thread(_failed_run_details, task, task_id, task_metadata)
        yield "event: message\ndata: " + json.dumps({"status": "FAILED", "message": _failed_msg, "error": _failed_msg, "traceback": _failed_msg, "next_due_at": next_due_at, **task_metadata, **_details}) + "\n\n"
    else:
        view = await _finished_run_view(task.result, task_id, task_metadata)
        final = view["final"]
        assistant_messages = view["assistant_messages"]
        output_file_data, output_json = view["output_file_data"], view["output_json"]
        completion_message = view["completion_message"]
        # Same builder as the synchronous response and /tasks/{id}/result.
        execution_context = _build_execution_context_from_state(final, session=session)
        # The outcome _record_run_finish writes for this run (item 8a): Celery
        # SUCCESS only means the task returned, not that it answered.
        _task_truth = view["run_truth"]
        yield "event: message\ndata: " + json.dumps({
            "status": "SUCCESS",
            "run_truth": _task_truth,
            "message": completion_message,
            "completion_message": completion_message,
            "messages": assistant_messages,
            "assistant_message": assistant_messages[-1] if assistant_messages else None,
            **task_metadata,
            "result": {
                "output_file_data": output_file_data,
                "output_json": output_json,
                "messages": assistant_messages,
                "assistant_message": assistant_messages[-1] if assistant_messages else None,
                "message": completion_message,
                "completion_message": completion_message,
                "execution_context": execution_context,
                "run_truth": _task_truth,
                "output_contract": view["output_contract"],
            },
            "next_due_at": next_due_at
        }) + "\n\n"

    yield "event: done\ndata: {}\n\n"


@app.delete("/threads/{thread_id}", status_code=204, tags=["threads"])
async def delete_thread(request: Request, thread_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    sid = await get_thread_session(thread_id)
    if sid:
        sess = await get_session(sid)
        if not sess:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread_id")
        if sess.get("user_id") != user_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Thread belongs to another user")
    else:
        await hydrate_thread_history(thread_id)
        if await read_thread_msgs(thread_id):
            # Orphaned history has no owner to authorize against; refuse to let
            # an arbitrary authenticated caller destroy it.
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread_id")

    fwd_headers: Dict[str, str] = {}
    for h in ("Authorization", "X-API-Key", "X-LangGraph-Api-Key"):
        if h in request.headers:
            fwd_headers[h] = request.headers[h]
    try:
        resp = await lg_request(
            "DELETE",
            f"/threads/{thread_id}",
            headers=fwd_headers,
        )
        if resp.status_code not in (200, 204, 404, 410):
            raise HTTPException(
                resp.status_code, f"Upstream delete failed: {resp.text}"
            )
    except HTTPException as e:
        if e.status_code not in (404, 410, 502, 504):
            raise

    if session_service.cache:
        await session_service.cache.delete(_k_thread_session(thread_id))
    await delete_thread_history(thread_id)
    THREAD_META.pop(thread_id, None)
    await _forget_thread_memory(user_id, thread_id, sid)
    return Response(status_code=204)


async def _forget_thread_memory(user_id: str, thread_id: str, sid: Optional[str] = None) -> None:
    """Delete what a deleted thread learned (app/services/memory_plane.py).
    Never raises: a memory store being down must not fail the delete; the
    memory plane logs which store was not cleared."""
    try:
        from app.services.memory_plane import forget_thread_memory
        await asyncio.to_thread(forget_thread_memory, user_id, thread_id, sid)
    except Exception:
        logger.warning("Memory cleanup failed for thread %s", thread_id, exc_info=True)


async def forget_account_memory(user_id: str) -> Dict[str, Any]:
    """Internal API: delete all of a user's learned memory, for account
    deletion. 1.6 has no account-deletion route; whatever deletes an account
    calls this. Covers every Milvus insight and Chroma signature filed under
    the user, rows filed under the user's sessions, and the Redis copies of
    the threads bound to those sessions. Returns {store: rows removed or
    "failed: ..."}; never raises."""
    sids: List[str] = []
    tids: set = set()
    cache = session_service.cache
    try:
        if cache:
            sids = sorted(await cache.smembers(_k_user_sessions(user_id)) or [])
            for sid in sids:
                tids.update(await cache.smembers(_k_session_threads(sid)) or [])
    except Exception:
        logger.warning("forget_account_memory: could not list the sessions of user %s",
                       user_id, exc_info=True)
    try:
        from app.services.memory_plane import forget_user_memory
        return await asyncio.to_thread(forget_user_memory, user_id, sorted(tids), sids)
    except Exception:
        logger.warning("Memory cleanup failed for user %s", user_id, exc_info=True)
        return {}


def _full_cloud_uri(sess: dict) -> str | None:
    # Folder-as-dataset (partitioned Parquet / Hive / Iceberg): the executable
    # cloud source is the glob / metadata URI resolved at registration, NOT the
    # bare folder. object_name is None for these, so the single-object logic
    # below would hand the reader an un-globbed directory.
    folder_read_path = sess.get("folder_read_path")
    if folder_read_path:
        return folder_read_path

    base = (sess.get("data_source_location") or "").rstrip("/")
    key = (sess.get("object_name") or "").lstrip("/")
    if not base:
        return None
    if key and (base.endswith("/" + key) or base.endswith(key)):
        return base
    if key and "://" in base:
        return f"{base}/{key}"
    return base

FIDELITY_QUICK = "quick_sample"
FIDELITY_PORTFOLIO = "portfolio_samples"
FIDELITY_ENTIRE = "entire_dataset"


def _session_total_rows(sess: Dict[str, Any]) -> Optional[int]:
    """Row count of the whole file as measured at ingest, or None if unknown.

    sample_with_profiling (uploads and sub-1 GB registrations) reads the whole
    file and records the exact count in ``sample_statistics.data_shape.rows``.
    The quick sampler used for >=1 GB objects writes an *estimate* to the same
    field, so callers must not trust it for large datasets.
    """
    stats = _maybe_json_load(sess.get("sample_statistics")) or sess.get("sample_statistics")
    if isinstance(stats, dict):
        rows = (stats.get("data_shape") or {}).get("rows")
        if isinstance(rows, int) and not isinstance(rows, bool) and rows > 0:
            return rows
    rows = sess.get("total_rows_exact")
    if isinstance(rows, int) and not isinstance(rows, bool) and rows > 0:
        return rows
    return None


def _analysis_ran_on_partial_sample(
    *,
    effective_fidelity: Optional[str],
    sample_n: int,
    total_rows: Optional[int],
    is_large_dataset: bool,
    entire_on_head: bool,
) -> bool:
    """Did this turn's figures come from a subset of the file?

    The honest test is rows analysed against rows in the file. Where the
    dataset lives is not evidence: an uploaded file sits at a gs:// URI too,
    and a quick-sample run reads every row of it when the file fits under the
    sample cap. Treating every cloud-origin session as sampled produced the
    "computed on a 1,000-row sample" note on 1,000-row files.
    """
    if effective_fidelity == FIDELITY_ENTIRE:
        # Entire-dataset semantics on a streamed head is the one way an
        # "entire" run can still be partial.
        return entire_on_head
    if is_large_dataset:
        # Only a head or a sample of a >=1 GB object is ever local, and the
        # recorded row total is itself an estimate taken from that head.
        return True
    if effective_fidelity == FIDELITY_PORTFOLIO:
        # Sub-1 GB portfolio runs execute on the full local copy (see the
        # routing ladder in send_message), so nothing was left out.
        return False
    # Quick sample: the sampler returned min(total, row cap, byte clamp) rows.
    if total_rows is not None:
        return sample_n < total_rows
    return sample_n >= DEFAULT_SAMPLE_MAX_ROWS


def _sample_fidelity_note(
    effective_fidelity: Optional[str],
    sample_rows: Any,
    sample_name: Optional[str],
    total_rows: Optional[int] = None,
    db_sample: Optional[DbSample] = None,
) -> str:
    """One-line disclosure appended to numeric answers computed on a partial sample."""
    n = len(sample_rows) if isinstance(sample_rows, list) else 0
    if db_sample is not None:
        return (
            f"\n\n> Note: these figures were computed from {sample_scope(db_sample)}, not the "
            f"entire table, so counts, rates, distinct values and min/max are sample estimates. "
            f"Ask for a rate, mean or count by one column and the database computes it "
            f"exactly over every row."
        )
    if effective_fidelity == FIDELITY_PORTFOLIO:
        scope = f"a portfolio sample ({sample_name})" if sample_name else "a portfolio sample"
    elif n and total_rows and total_rows > n:
        scope = f"a {n:,}-row sample of the {total_rows:,}-row file"
    else:
        scope = f"a {n:,}-row sample (the first rows of the file)" if n else "a partial sample of the file"
    return (
        f"\n\n> Note: these figures were computed on {scope}, not the entire dataset, so "
        f"counts, rates, distinct values and min/max may differ from the full file. "
        f'Reply "run this on the entire dataset" for exact numbers.'
    )
DEFAULT_SAMPLE_NAME = "random_baseline"
LARGE_DATASET_THRESHOLD_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
# Fidelity labels and the execution_context builder live in app.api.ui_contract.


# ── Byte-aware routing config ────────────────────────────────────────────────
# Shared with the samplers (which cannot import server.py) — single source of
# truth lives in app.core.analysis_limits; see its docstring for the threshold
# semantics (estimated UNCOMPRESSED in-memory working set, NOT source bytes)
# and the derivation of the default cap (unvalidated pending shadow data).
from app.core.analysis_limits import (
    analysis_max_inmemory_bytes as _analysis_max_inmemory_bytes,
    byte_routing_enabled as _byte_routing_enabled,
    routing_shadow_enabled as _routing_shadow_enabled,
)


def _log_routing_shadow(**fields: Any) -> None:
    """Shadow-mode observability: one structured line per request comparing the
    byte-based routing decision that WOULD have been taken against what actually
    ran. Behaviour is never changed here. The `band` field exists specifically
    to observe the cap..1GB range (deferred by the new gate, interactive under
    the old one) before the cap is locked in."""
    try:
        cap = fields.get("analysis_max_inmemory_bytes") or 0
        est = fields.get("estimated_full_bytes") or 0
        src = fields.get("file_size_bytes") or 0
        would_defer = bool(cap and est and est >= cap)
        old_large = src >= LARGE_DATASET_THRESHOLD_BYTES
        if not would_defer:
            band = "below_cap"
        elif not old_large:
            band = "cap_to_1gb"   # scheduling work the old gate ran interactively
        else:
            band = "above_1gb"
        fields.update(would_defer=would_defer, old_gate_large=old_large, band=band)
        logger.info("[routing_shadow] %s", json.dumps(fields, default=str, sort_keys=True))
    except Exception as exc:  # observability must never break the request
        logger.warning("[routing_shadow] logging failed: %s", exc)


# ── Metadata-only fast path (byte routing) ──────────────────────────────────
# Above the analysis cap we defer to a background job — but schema, column
# names, dtypes, row count and date range must never queue a job ("how many
# columns does this have" answers instantly regardless of size). Deliberately
# conservative: only fires when deferral would otherwise trigger, only on
# short, clearly structural questions, with analytic keywords excluded.
_METADATA_PATTERNS = [
    re.compile(r"\bhow many (columns|fields|variables)\b", re.I),
    re.compile(r"\b(what|which|list|show|name|describe)\b[^?.!]*\b(columns|fields|column names|schema|structure)\b", re.I),
    re.compile(r"\b(the |its )?schema\b", re.I),
    re.compile(r"\bdata ?types?\b|\bdtypes?\b", re.I),
    re.compile(r"\bhow many (rows|records|entries|observations)\b", re.I),
    re.compile(r"\b(row|record) count\b|\btotal (rows|records)\b|\bnumber of (rows|records)\b", re.I),
    re.compile(r"\bdate range\b|\btime range\b|\btime span\b", re.I),
]
_METADATA_EXCLUDE = re.compile(
    r"\b(where|group|filter|average|avg|mean|sum|median|min|max|"
    r"per|top|correlat\w*|distribut\w*|plot|chart|graph|compare|trend|predict|"
    r"change|convert|transform|create|make|write|save|export|join|merge|drop|delete)\b",
    re.I,
)
# "how many rows HAVE <condition>" is analytic; "how many columns does this
# have?" is metadata — reject the qualifier only when it introduces content.
_METADATA_QUALIFIED = re.compile(
    r"\b(have|with|that|whose|having|containing)\b\s+\S", re.I
)
# Task-operation messages (status/cancel/etc.) must reach the planner's task
# tools — the deferred-path early returns must not swallow them.
_TASK_OP_RE = re.compile(
    r"\b(cancel|abort|stop|pause|resume|status|progress|task|job)s?\b", re.I
)


def _is_metadata_only_question(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 160:
        return False
    if _METADATA_EXCLUDE.search(t) or _METADATA_QUALIFIED.search(t):
        return False
    return any(p.search(t) for p in _METADATA_PATTERNS)


async def _db_pushdown_answer(
    user_id: str, sess: Dict[str, Any], question: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """``(answer, output_table)`` for a plain group aggregate run on the whole DB table.

    Only for a session made from a table (``db_pushdown_table``), whose
    database the caller connected themselves: the api_key comes from the
    caller's own saved connection or connect session (#535's handles), never
    from the dataset session, so a collaborator on a shared analysis gets
    nothing. None means
    "not a question this can answer exactly"; the turn then runs on the
    sample and says so.
    """
    table = sess.get("db_pushdown_table")
    customer_id = (sess.get("customer_id") or "").strip()
    sample = DbSample.from_session(_maybe_json_load(sess.get("db_sample")) or sess.get("db_sample"))
    if not (table and customer_id and sample and _DB_IDENT_RE.match(str(table))):
        return None
    columns = _maybe_json_load(sess.get("uploaded_csv_columns")) or sess.get("uploaded_csv_columns") or []
    intent = parse_aggregate_intent(question, [str(c) for c in columns])
    if intent is None:
        return None
    try:
        handle = await db_credential_handle_for_user(user_id, customer_id)
    except Exception:
        handle = None
    api_key = decrypt_secret(handle) if handle else None
    if not api_key:
        return None

    async def _query(sql: str) -> Dict[str, Any]:
        return await _run_mcp_db_query(customer_id, api_key, sql)

    try:
        result = await run_pushdown(_query, str(table), intent, sample.dialect)
    except Exception as exc:
        logger.warning("[db-pushdown] aggregate not run: %s", type(exc).__name__)
        return None
    if result is None:
        return None
    return render_pushdown_answer(result, str(table)), pushdown_output_table(result)


def _drivers_inputs(sess: Dict[str, Any], datasets: int) -> Dict[str, Any]:
    """The rows this thread holds, for the one drivers routing decision
    (app/agents/drivers_route.answer_drivers_turn) the graph's first node
    makes too: the fast path below and that node read the same inputs."""
    return inputs_from_session(sess, total_rows=_session_total_rows(sess), datasets=datasets)


def _build_metadata_answer(sess: Dict[str, Any], question: str) -> Optional[str]:
    """Answer a metadata-only question from already-captured session state.

    Never fabricates exactness: row counts are labeled estimated unless the
    full profile ran; date ranges from a preview slice are labeled as such.
    Returns None when nothing trustworthy can be built (caller falls through
    to the normal defer path).
    """
    schema = _maybe_json_load(sess.get("schema")) or sess.get("schema")
    stats = _maybe_json_load(sess.get("sample_statistics")) or sess.get("sample_statistics") or {}
    data_shape = (stats.get("data_shape") or {}) if isinstance(stats, dict) else {}
    q = (question or "").lower()

    col_items: List[Tuple[str, str]] = []
    if isinstance(schema, dict):
        col_items = [(str(k), str(v)) for k, v in schema.items()]
    elif isinstance(schema, list):
        col_items = [(str(c), "") for c in schema]
    n_cols = len(col_items) or (data_shape.get("columns") or 0)

    total_rows = data_shape.get("rows") or sess.get("total_rows_exact") or sess.get("estimated_rows")
    rows_exact = str(sess.get("sample_status") or "") == "full_sample"

    parts: List[str] = []
    asks_rows = bool(re.search(r"\brows?\b|\brecords?\b|\bentries\b|\bobservations\b", q))
    asks_cols = bool(re.search(r"\bcolumns?\b|\bfields?\b|\bschema\b|\bstructure\b|\bd?types?\b|\bvariables\b", q))
    asks_range = bool(re.search(r"\b(date|time) (range|span)\b", q))

    if asks_cols and col_items:
        parts.append(f"The dataset has **{n_cols} columns**:")
        if any(dt for _, dt in col_items):
            parts.append("\n".join(f"- `{name}` — {dt or 'unknown'}" for name, dt in col_items))
        else:
            parts.append(", ".join(f"`{name}`" for name, _ in col_items))
    elif asks_cols and n_cols:
        parts.append(f"The dataset has **{n_cols} columns**.")

    if asks_rows and total_rows:
        # Partial-measurement guard: >=1GB cloud registrations profile only the
        # streamed head, so data_shape.rows is the HEAD count (e.g. ~50k for a
        # 10GB object). Detect it the same way routing does (measured estimate
        # below on-disk size) and never quote the head count as the dataset's.
        _file_bytes_meta = int(sess.get("file_size_bytes") or 0)
        _est_meta = stats.get("estimated_full_bytes") if isinstance(stats, dict) else None
        _partial_measurement = bool(
            _file_bytes_meta > 0
            and isinstance(_est_meta, (int, float))
            and 0 < _est_meta < _file_bytes_meta
        )
        if _partial_measurement:
            _bpr_meta = stats.get("bytes_per_row") if isinstance(stats, dict) else None
            if isinstance(_bpr_meta, (int, float)) and _bpr_meta > 0:
                parts.append(
                    f"This file is {_human_size(_file_bytes_meta)} on disk and only its first rows "
                    f"were profiled, so an exact count needs the full pass — roughly "
                    f"**{int(_file_bytes_meta / _bpr_meta):,}+ rows** based on file size."
                )
            else:
                parts.append(
                    f"This file is {_human_size(_file_bytes_meta)} on disk and only its first rows "
                    "were profiled — the exact row count comes from the full-dataset job."
                )
        elif rows_exact:
            parts.append(f"It has **{int(total_rows):,} rows** (exact).")
        else:
            parts.append(
                f"It has **approximately {int(total_rows):,} rows** (estimated from file "
                f"size — the exact count comes with the full profile)."
            )

    if asks_range:
        range_lines: List[str] = []
        col_stats = stats.get("column_statistics") or {} if isinstance(stats, dict) else {}
        for cname, cstat in col_stats.items():
            if not isinstance(cstat, dict):
                continue
            if re.search(r"date|time|_at$|_ts$", str(cname), re.I):
                lo, hi = cstat.get("min"), cstat.get("max")
                if lo is not None and hi is not None:
                    range_lines.append(f"- `{cname}`: {lo} → {hi}")
        if range_lines:
            parts.append(
                "Observed range in the preview sample (first rows only — the full-dataset "
                "range needs a complete pass):\n" + "\n".join(range_lines)
            )
        elif not parts:
            # a pure date-range ask we can't answer honestly from metadata
            return None

    return "\n\n".join(parts) if parts else None


def _build_deferred_job_card(
    task_id: str,
    dataset_name: str,
    interpreted_question: str,
    eta_hint: str,
) -> str:
    """Deterministic response for a byte-gated deferred analysis. Contains the
    job handle and NO analytical numbers — a sampled number presented as an
    answer is worse than no answer."""
    q = (interpreted_question or "").strip()
    if len(q) > 300:
        q = q[:297] + "..."
    return (
        "📋 **Full-dataset analysis scheduled**\n\n"
        f"- **Job ID**: `{task_id}`\n"
        f"- **Dataset**: {dataset_name}\n"
        f"- **Question (as interpreted)**: {q}\n"
        f"- **Estimated completion**: {eta_hint}\n\n"
        # Not "every row": the Ray batch job computes the answer on the first
        # RAY_ANALYSIS_SAMPLE_ROWS rows and reports the file's row count beside it
        # (MAJ-030, the Ray-honesty labelling). The result carries that scope.
        "This runs as a background job; nothing is shown until it finishes. The "
        "result will appear here, saying how many of the file's rows it analysed, "
        "and you'll get a notification."
    )


def _estimated_inmemory_bytes(sess: Dict[str, Any]) -> Tuple[int, float, str]:
    """Estimated uncompressed in-memory footprint of the WHOLE dataset.

    This is the real cost signal for routing: ``total_rows * bytes_per_row``
    measured by the samplers at ingest (Arrow bytes on a head slice). It is
    deliberately NOT ``file_size_bytes`` — that is on-disk/compressed source
    size, which understates Parquet by ~5-10x.

    Returns (estimated_full_bytes, bytes_per_row, source_tag). source_tag is
    "measured" when the sampler recorded the signal, "expansion" when we fell
    back to file_size_bytes * ANALYSIS_MEM_EXPANSION (default 4.0), "none"
    when nothing was available (both numbers 0).
    """
    file_bytes = int(sess.get("file_size_bytes") or 0)
    try:
        expansion = float(os.getenv("ANALYSIS_MEM_EXPANSION", "4.0"))
    except ValueError:
        expansion = 4.0

    stats = _maybe_json_load(sess.get("sample_statistics")) or sess.get("sample_statistics")
    measured = 0
    bpr = 0.0
    if isinstance(stats, dict):
        est = stats.get("estimated_full_bytes")
        raw_bpr = stats.get("bytes_per_row")
        if isinstance(est, (int, float)) and est > 0:
            measured = int(est)
            bpr = float(raw_bpr or 0.0)
        else:
            # older sessions may have bytes_per_row without the product
            rows = ((stats.get("data_shape") or {}).get("rows")
                    or sess.get("total_rows_exact") or sess.get("estimated_rows"))
            if isinstance(raw_bpr, (int, float)) and raw_bpr > 0 and isinstance(rows, (int, float)) and rows > 0:
                measured = int(rows * raw_bpr)
                bpr = float(raw_bpr)

    if measured > 0:
        # Partial-measurement floor: for >=1GB cloud registrations the sampler
        # ran on the streamed HEAD file, so rows*bytes_per_row reflects only the
        # head (a few MB for a multi-GB object). In-memory size can't be smaller
        # than the on-disk source — if the measurement is below it, the sampler
        # clearly never saw the whole file; floor to the expansion estimate.
        if file_bytes > 0 and measured < file_bytes:
            return max(measured, int(file_bytes * expansion)), bpr, "expansion_floor"
        return measured, bpr, "measured"

    if file_bytes > 0:
        return int(file_bytes * expansion), 0.0, "expansion"
    return 0, 0.0, "none"

def _human_size(num_bytes: int) -> str:
    size = float(max(int(num_bytes or 0), 0))
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


def _estimated_runtime_hint(num_bytes: int) -> str:
    return _planner_estimated_runtime_hint(num_bytes)


def _build_fidelity_prompt(num_bytes: int) -> str:
    return _planner_build_fidelity_prompt(num_bytes)


def _persist_small_file_profile_sync(
    *,
    dataset_id: str,
    full_result: Dict[str, Any],
    user_id: Optional[str],
    source_path: str,
    source_type: str,
    full_profiling_result: Optional[Dict[str, Any]],
) -> None:
    persist_full_profile(
        dataset_id=dataset_id,
        full_result=full_result,
        user_id=user_id,
        source_path=source_path,
        source_type=source_type,
    )
    if full_profiling_result:
        upsert_profile(
            dataset_id=dataset_id,
            phase="semantic_profile",
            profiling_result=full_profiling_result,
        )


async def _run_small_file_persistence_job(
    *,
    job_key: str,
    dataset_id: str,
    full_result: Dict[str, Any],
    user_id: Optional[str],
    source_path: str,
    source_type: str,
    full_profiling_result: Optional[Dict[str, Any]],
) -> None:
    global small_file_persistence_inflight

    executor = small_file_persistence_executor
    semaphore = small_file_persistence_semaphore
    if executor is None or semaphore is None:
        logger.warning("[small-file-persist] runtime not initialized; skipping dataset %s", dataset_id)
        small_file_persistence_inflight.discard(job_key)
        return

    try:
        async with semaphore:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                executor,
                lambda: _persist_small_file_profile_sync(
                    dataset_id=dataset_id,
                    full_result=full_result,
                    user_id=user_id,
                    source_path=source_path,
                    source_type=source_type,
                    full_profiling_result=full_profiling_result,
                ),
            )
            logger.info("[small-file-persist] completed dataset %s", dataset_id)
    except Exception:
        logger.warning("[small-file-persist] failed for dataset %s", dataset_id, exc_info=True)
    finally:
        small_file_persistence_inflight.discard(job_key)


def _schedule_small_file_persistence(
    *,
    job_key: str,
    dataset_id: str,
    full_result: Dict[str, Any],
    user_id: Optional[str],
    source_path: str,
    source_type: str,
    full_profiling_result: Optional[Dict[str, Any]],
) -> None:
    global small_file_persistence_inflight

    if job_key in small_file_persistence_inflight:
        logger.info("[small-file-persist] already in-flight for %s; skipping duplicate", job_key)
        return

    small_file_persistence_inflight.add(job_key)
    task = asyncio.create_task(
        _run_small_file_persistence_job(
            job_key=job_key,
            dataset_id=dataset_id,
            full_result=full_result,
            user_id=user_id,
            source_path=source_path,
            source_type=source_type,
            full_profiling_result=full_profiling_result,
        )
    )
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    logger.info("[small-file-persist] scheduled dataset %s", dataset_id)


async def _load_persisted_preview_artifacts(
    dataset_id: str,
) -> Tuple[Optional[Dict[str, List[Dict[str, Any]]]], Optional[Dict[str, Any]]]:
    stored_portfolio, stored_profile = await asyncio.gather(
        asyncio.to_thread(load_portfolio, dataset_id),
        asyncio.to_thread(load_profile, dataset_id),
    )
    return stored_portfolio, stored_profile


def _upload_sample_to_cloud_if_needed(sess: Dict[str, Any]) -> Optional[str]:
    """
    For portfolio-based analysis on k8s-ray, we want Ray to read from a
    sample file in cloud storage instead of scanning the entire dataset.
    This helper uploads the local sample_input.csv to the same bucket/prefix
    as the original object and returns its URI.
    """
    storage_uri = (sess.get("data_source_location") or "").rstrip("/")
    sample_path = sess.get("sample_local_input")
    object_name = sess.get("object_name") or _key_from_uri(storage_uri) or ""

    if not storage_uri or not sample_path:
        return None

    try:
        sample_path = Path(sample_path)
        if not sample_path.exists():
            return None
    except Exception:
        return None

    # Prefer precomputed URIs from Supabase/session if present
    uris = sess.get("portfolio_sample_uris")
    sel = sess.get("selected_sample_name")
    if isinstance(uris, str):
        try:
            uris = json.loads(uris)
        except Exception:
            uris = {}
    if isinstance(uris, dict) and sel and uris.get(sel):
        return str(uris[sel])

    # Derive a sample object key alongside the original file as a last resort.
    key_dir, _, fname = object_name.rpartition("/")
    sample_dir = f"{key_dir}/_avaloka_samples" if key_dir else "_avaloka_samples"
    sample_key = f"{sample_dir}/{fname or 'sample_portfolio.csv'}"

    try:
        scheme, rest = storage_uri.split("://", 1)
        bucket = rest.split("/", 1)[0]
        sample_uri = f"{scheme}://{bucket}/{sample_key}"
        store_for_write, rel_key = _store_and_key_from_uri(sample_uri, None)
        if not store_for_write:
            return None
        store_for_write.put_file(sample_path, rel_key)
        return sample_uri
    except Exception:
        logger.warning("[portfolio] Failed to upload sample to cloud for k8s-ray; falling back to full dataset", exc_info=True)
        return None


def _parse_fidelity_from_text(text: str) -> Optional[str]:
    return _planner_parse_fidelity_from_text(text)


def _is_mode_switch_message(text: str) -> bool:
    return _planner_is_mode_switch_message(text)


def _scheduled_task_is_in_flight(task_id: Optional[str]) -> bool:
    """Return True when a scheduled task should be treated as still active.

    Be conservative: if a task id exists but we cannot inspect RedBeat/Celery
    cleanly, assume it is still in flight so later chat turns fall back to
    sample-based analysis instead of trying to reschedule entire-dataset work.
    """
    if not task_id:
        return False

    try:
        redbeat_key = AvalokaEntry.generate_key(celery_app, task_id)
        AvalokaEntry.from_key(redbeat_key, celery_app)
    except Exception:
        return True

    try:
        execution_id = AvalokaScheduler.get_last_run_task_id(redbeat_key, False)
        if not execution_id:
            return True
        if AsyncResult(execution_id, app=celery_app).state in ("FAILURE", "SUCCESS", "REVOKED"):
            return False
        # Not final in Celery: a result that is no longer stored reads PENDING.
        return _run_finished_state(execution_id) is None
    except Exception:
        logger.debug("Could not inspect scheduled task state for %s", task_id, exc_info=True)
        return True

def _entire_job_in_flight(sess: Dict[str, Any]) -> bool:
    """A full-dataset job is genuinely running for this session.

    Time-bounded like the byte-routing guard: a purged, lost or expired Celery
    task reports PENDING forever, which used to lock every later "Full dataset"
    question onto the quick sample with no way out.
    """
    task_id = sess.get("analysis_task_id")
    if not task_id:
        return False
    scheduled_at = float(sess.get("analysis_task_scheduled_at") or 0)
    try:
        ttl = int(os.getenv("ANALYSIS_DEFER_INFLIGHT_TTL_S", "3600"))
    except ValueError:
        ttl = 3600
    if scheduled_at <= 0 or (time.time() - scheduled_at) >= ttl:
        return False
    return _scheduled_task_is_in_flight(task_id)


def _reported_last_run_at(entry: Any, run_count: int) -> Optional[str]:
    """When the task last ran, or None if it never has.

    entry.last_run_at is RedBeat's schedule anchor, not a run record: celery's
    ScheduleEntry seeds it with the creation time and AvalokaEntry.save()
    persists that, so a task that has never run would read "last run just now".
    Only a dispatch (which also bumps the run count) makes it a real run time.
    """
    if not run_count or not getattr(entry, "last_run_at", None):
        return None
    return entry.last_run_at.isoformat()


def _scheduled_task_metadata_from_entry(entry: Any) -> Dict[str, Any]:
    try:
        args = getattr(entry, "args", None) or []
        if args and isinstance(args[0], dict):
            return scheduled_task_metadata(args[0].get("task_schedule") or {})
    except Exception:
        logger.debug("Could not read scheduled task metadata", exc_info=True)
    return scheduled_task_metadata({})


def _scheduled_task_completion_message(
    final: Dict[str, Any],
    assistant_messages: List[Dict[str, str]],
    task_metadata: Dict[str, Any],
) -> str:
    if task_metadata.get("task_type") == "training":
        failure = _training_failure_from_final(final)
        if failure:
            return format_training_failure(failure)
        status, error = _run_status_from_final(final, task_type="training")
        if status == "failure":
            return TRAINING_FAILURE_MESSAGE
    task_type = task_metadata.get("task_type") or "execute"
    status, error = _run_status_from_final(final, task_type=task_type)
    if status == "failure":
        # A failed or empty run is never "complete" (celery_app decides, once).
        return scheduled_failure_message(final, task_type, error)
    if task_type in {"start_inference", "stop_inference"} and assistant_messages:
        latest = assistant_messages[-1].get("content")
        if latest:
            return latest
    return str(task_metadata.get("success_message") or "The scheduled task is complete.")


def _scheduled_run_failed(final: Dict[str, Any], task_metadata: Dict[str, Any]) -> bool:
    """A Celery SUCCESS whose final state is a failed or empty run.

    A failed run now raises ScheduledRunFailed and lands in FAILURE; results
    stored before that still read SUCCESS, and must not be sent as one.
    """
    if celery_results.is_summary(final):
        # Stored as a summary: a failed run is FAILURE (ScheduledRunFailed).
        return False
    status, _ = _run_status_from_final(final, task_type=task_metadata.get("task_type") or "execute")
    return status == "failure"


_SCHEDULED_CANCELLED_TEXT = "This run was cancelled. No output was produced."


def _scheduled_failure_text(task: Any, task_metadata: Dict[str, Any]) -> str:
    """The plain message for a run Celery recorded as FAILURE."""
    result = getattr(task, "result", None)
    if type(result).__name__ == "ScheduledRunFailed":
        if str(result):
            return str(result)
        # Loaded back from the durable store, which keeps the class only: the
        # plain-words message is the run record's error.
        run = celery_results.load_scheduled_run(getattr(task, "id", None)) or {}
        if run.get("error"):
            return str(run["error"])
    if task_metadata.get("task_type") == "training":
        return TRAINING_FAILURE_MESSAGE
    return "The scheduled task failed."


def _failed_run_details(task: Any, task_id: Optional[str], task_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """What a FAILED answer carries besides its message: the run truth
    _record_run_finish wrote for this run and the output contract's reasons
    (MAJ-030 / item 8a), from the Celery result when it still holds the final
    state, else from the durable scheduled_runs row. The route and the run
    record say the same thing; no rows are returned for a refused result."""
    final = getattr(task, "result", None)
    truth: Optional[Dict[str, Any]] = None
    contract: Optional[Dict[str, Any]] = None
    if isinstance(final, dict) and not celery_results.is_summary(final):
        truth = _scheduled_task_run_truth(final, task_metadata)
        contract = (final.get("execution_result") or {}).get("output_contract") if isinstance(final.get("execution_result"), dict) else None
    else:
        run = celery_results.load_scheduled_run(task_id) or {}
        run_result = run.get("result") or {}
        truth = run_result.get("run_truth") if isinstance(run_result.get("run_truth"), dict) else None
        contract = run_result.get("output_contract") if isinstance(run_result.get("output_contract"), dict) else None
    return {
        "run_truth": truth,
        "result": {"output_file_data": None, "output_json": None, "output_contract": contract,
                   "run_truth": truth},
    }


_EXEC_OK = ("success", "succeeded", "completed", "done")


async def _finished_run_view(final: Any, execution_id: Optional[str], task_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """What a finished (SUCCESS) run returns to the UI.

    A result stored as a summary (app/core/celery_results.py) holds no
    messages and no inline output; the run's reply, output table, logs and run
    truth come from its durable scheduled_runs row. A result stored before summaries (the
    full state) is read as it always was.
    """
    chat_only = _scheduled_task_has_chat_result_only(task_metadata)
    task_type = task_metadata.get("task_type")
    if not celery_results.is_summary(final):
        final = dict(final) if isinstance(final, dict) else {}
        final["messages"] = convert_message_dicts_to_objects(final.get("messages", []))
        assistant_messages = _assistant_messages_from_state(final)
        output_file_data, output_json = get_output_from_state(final)
        er = final.get("execution_result") or {}
        if chat_only or (str(er.get("status", "")).lower() not in _EXEC_OK and not final.get("output_file_data")):
            output_file_data = output_json = None
        run_status, error = (_run_status_from_final(final, task_type="training")
                             if task_type == "training" else (None, None))
        return {
            "final": final,
            "assistant_messages": assistant_messages,
            "output_file_data": output_file_data,
            "output_json": output_json,
            "completion_message": _scheduled_task_completion_message(final, assistant_messages, task_metadata),
            "run_truth": _scheduled_task_run_truth(final, task_metadata),
            "output_contract": er.get("output_contract"),
            "logs": _logs_from_final(final),
            "ray_job": _ray_job_from_final(final),
            "run_status": run_status,
            "error": error,
        }
    run = await asyncio.to_thread(celery_results.load_scheduled_run, execution_id) or {}
    run_result = run.get("result") or {}
    assistant_messages = celery_results.assistant_messages_from_logs(run_result.get("logs"))
    output_json = run_result.get("output_json") if isinstance(run_result.get("output_json"), list) else None
    output_file_data = celery_results.output_file_data_from_rows(output_json, final.get("output_location"))
    if chat_only:
        output_file_data = output_json = None
    run_status = str(run.get("status") or "").lower() or None
    if task_type == "training" and run_status == "failure":
        completion_message = run_result.get("message") or TRAINING_FAILURE_MESSAGE
    elif task_type in {"start_inference", "stop_inference"} and assistant_messages:
        completion_message = assistant_messages[-1]["content"]
    else:
        completion_message = str(task_metadata.get("success_message") or "The scheduled task is complete.")
    return {
        # The summary carries what the execution context reads (fidelity, sample).
        "final": dict(final),
        "assistant_messages": assistant_messages,
        "output_file_data": output_file_data,
        "output_json": output_json,
        "completion_message": completion_message,
        "run_truth": run_result.get("run_truth"),
        "output_contract": run_result.get("output_contract"),
        "logs": run_result.get("logs"),
        "ray_job": run_result.get("ray_job"),
        "run_status": run_status if task_type == "training" else None,
        "error": run.get("error") if task_type == "training" else None,
    }


def _run_finished_state(execution_id: Optional[str]) -> Optional[str]:
    """SUCCESS/FAILURE when the run's durable record says it is over, else None.

    For a Celery state that is not final: a result that is no longer stored
    reads as PENDING, which must not be shown as "running".
    """
    return celery_results.finished_run_state(celery_results.load_scheduled_run(execution_id))


#: The SSE checks the durable run record every this many polls while a task
#: is not final, so a run that ended without a stored result ends the stream.
_SSE_DURABLE_CHECK_EVERY = 15


def _scheduled_task_run_truth(final: Dict[str, Any], task_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """run_truth for a finished task's result: the same call _record_run_finish
    makes, so the route and the run record cannot disagree."""
    return _run_outcome_from_final(final, task_type=str(task_metadata.get("task_type") or "execute"))[2]


def _scheduled_task_has_chat_result_only(task_metadata: Dict[str, Any]) -> bool:
    return task_metadata.get("task_type") in {"start_inference", "stop_inference"}


def _rows_to_columns(rows: Any) -> List[str]:
    if not isinstance(rows, list) or not rows:
        return []
    first_row = rows[0]
    if isinstance(first_row, dict):
        return [str(column) for column in first_row.keys()]
    return []


def _count_csv_rows(path: Path) -> Optional[int]:
    try:
        with path.open("rb") as fh:
            line_count = sum(1 for _ in fh)
        return max(line_count - 1, 0)
    except Exception:
        return None


_AVALOKA_RESULT_RE = re.compile(
    r"<<<AVALOKA_RESULT>>>(.*?)<<<END_AVALOKA_RESULT>>>", re.DOTALL
)


def _iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def _scalar_result_from_state(final: Dict[str, Any], output_json: Any) -> Optional[Dict[str, Any]]:
    """Return the scalar answer if this turn produced one (count, mean, max...),
    else None. A scalar answer belongs in the chat message, not in the Data table."""
    # 1) Explicit sentinel emitted by avaloka_result()
    for text in _iter_strings(final.get("execution_result") or {}):
        m = _AVALOKA_RESULT_RE.search(text)
        if m:
            try:
                payload = json.loads(m.group(1))
                if isinstance(payload, dict) and "value" in payload:
                    return payload
            except (TypeError, ValueError):
                pass

    # 2) Fallback: a 1-row x 1-column result is a scalar wrapped in a table
    if (
        isinstance(output_json, list)
        and len(output_json) == 1
        and isinstance(output_json[0], dict)
        and len(output_json[0]) == 1
    ):
        (col, val), = output_json[0].items()
        return {"kind": "scalar", "value": val, "columns": [col]}
    return None

def _record_latest_tabular_output(
    sess: Dict[str, Any],
    output_path: Optional[str],
    rows_for_preview: Any = None,
) -> bool:
    """
    Remember the most recent generated dataframe without making it active.

    Active dataset replacement is still controlled by activate_output_as_dataset;
    this metadata only gives MTA a stable target when the user later asks to
    train on the generated/modified output.
    """
    if not output_path:
        return False

    output_path_obj = Path(output_path)
    if not output_path_obj.exists() or not output_path_obj.is_file():
        return False

    columns = _rows_to_columns(rows_for_preview)
    preview_rows = rows_for_preview if isinstance(rows_for_preview, list) else []
    row_count = len(preview_rows) if preview_rows else None

    if not columns or row_count is None:
        try:
            output_df_head = pd.read_csv(output_path_obj, nrows=DEFAULT_SAMPLE_MAX_ROWS)
        except Exception as exc:
            logger.info(
                "[send_message] Could not inspect latest tabular output %s: %s",
                output_path_obj,
                exc,
            )
            return False
        if output_df_head.empty:
            return False
        if not columns:
            columns = [str(column) for column in output_df_head.columns]
        if not preview_rows:
            preview_rows = output_df_head.to_dict(orient="records")
        if row_count is None:
            row_count = _count_csv_rows(output_path_obj)

    if not columns:
        return False

    active_path = _posix(output_path_obj)
    trainable = len(columns) >= 2 and (row_count is None or row_count >= 2)
    updates = {
        "latest_output_location": active_path,
        "latest_output_location_local": active_path,
        "latest_output_columns": _jsonify(columns),
        "latest_output_row_count": row_count,
        "latest_output_created_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "latest_output_is_trainable": trainable,
    }

    changed = False
    for key, value in updates.items():
        if sess.get(key) != value:
            sess[key] = value
            changed = True

    if preview_rows:
        latest_preview = _jsonify(preview_rows[:DEFAULT_SAMPLE_MAX_ROWS])
        if sess.get("latest_output_preview") != latest_preview:
            sess["latest_output_preview"] = latest_preview
            changed = True

    return changed


def _parse_selected_sample_from_text(text: str) -> Optional[str]:
    return _planner_parse_selected_sample_from_text(text)


def _resolve_selected_sample_rows(
    session_data: Dict[str, Any],
    dataset_id: Optional[str] = None,
    *,
    fallback_rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    portfolio = _maybe_json_load(session_data.get("portfolio_samples")) or session_data.get("portfolio_samples") or {}
    if not isinstance(portfolio, dict):
        portfolio = {}
    
    if not portfolio and dataset_id:
        portfolio = (_profile_meta.get(dataset_id) or {}).get("portfolio_samples") or {}
        if not portfolio:
            try:
                portfolio = load_portfolio(dataset_id) or {}
            except Exception:
                portfolio = {}
            if portfolio:
                # cache so later turns don't re-download the portfolio
                _meta = _profile_meta.get(dataset_id) or {}
                _meta["portfolio_samples"] = portfolio
                _profile_meta[dataset_id] = _meta
    available = _maybe_json_load(session_data.get("available_samples")) or session_data.get("available_samples") or []
    if not available and portfolio:
        available = list(portfolio.keys())
    if not isinstance(available, list):
        available = []

    selected = session_data.get("selected_sample_name")
    chosen: Optional[str] = None
    rows: List[Dict[str, Any]] = []

    if selected and selected in portfolio:
        maybe_rows = portfolio.get(selected) or []
        if isinstance(maybe_rows, list):
            rows = maybe_rows
            chosen = selected

    if not rows and DEFAULT_SAMPLE_NAME in portfolio:
        maybe_rows = portfolio.get(DEFAULT_SAMPLE_NAME) or []
        if isinstance(maybe_rows, list):
            rows = maybe_rows
            chosen = DEFAULT_SAMPLE_NAME

    if not rows and available:
        first = str(available[0])
        maybe_rows = portfolio.get(first) or []
        if isinstance(maybe_rows, list):
            rows = maybe_rows
            chosen = first

    if not rows:
        fallback = fallback_rows or []
        if isinstance(fallback, list):
            rows = fallback

    return rows, chosen


def _schedule_background_sampling_task(
    *,
    dataset_id: str,
    session_id: str,
    cloud_uri: str,
    connection_id: str,
    file_size_bytes: int,
    ray_namespace: str,
) -> Optional[str]:
    try:
        from uuid import uuid4 as _uuid4

        celery_state = {
            "task_schedule": {
                "task_type": "sample_profile",
                "schedule_type": "relative",
                "second": 0,
                "max_runs": 1,
            },
            "dataset_id": dataset_id,
            "session_id": session_id,
            "data_source_location_cloud": cloud_uri,
            "connection_id": connection_id,
            "file_size_bytes": int(file_size_bytes or 0),
            "execution_mode": "k8s-ray",
            "ray_namespace": ray_namespace or os.getenv("RAY_NAMESPACE", "default"),
            "ray_execution_profile": "batch_heavy",
            "messages": [],
        }
        _task_id = str(_uuid4())
        _entry = AvalokaEntry(
            _task_id,
            "app.core.celery_app.agent_task",
            0,
            args=[celery_state],
            app=celery_app,
            max_runs=1,
        ).save()
        return _entry.name
    except Exception:
        logger.warning("[sampling] Failed to schedule background sampling task", exc_info=True)
        return None


def _format_join_suggestions(sugs: list, limit: int = 5) -> str:
    lines = []
    for s in (sugs or [])[:limit]:
        left_ds = s.get("left_dataset") or s.get("left_alias") or "left"
        right_ds = s.get("right_dataset") or s.get("right_alias") or "right"
        left_key = s.get("left_key") or s.get("key") or "?"
        right_key = s.get("right_key") or s.get("key") or "?"
        conf = s.get("confidence")
        match_rate = s.get("match_rate")
        transform = s.get("suggested_transform")

        extra = []
        if conf is not None:
            extra.append(f"confidence={conf}")
        if match_rate is not None:
            extra.append(f"match_rate={match_rate}")
        if transform:
            extra.append(f"transform={transform}")

        suffix = f" ({', '.join(extra)})" if extra else ""
        lines.append(f"- {left_ds}.{left_key} ↔ {right_ds}.{right_key}{suffix}")
    return "\n".join(lines)

def _first_usable_path(*paths: Optional[str]) -> Optional[str]:
    """First path that is a cloud URI or an existing local file; else the first
    non-empty one (so we never hand the model a path we know is dead over one we
    haven't checked)."""
    for p in paths:
        if p and ("://" in p or Path(p).exists()):
            return p
    for p in paths:
        if p:
            return p
    return None
    
def _build_multi_dataset_context(
    multi_dataset_state: List[Dict[str, Any]],
    join_key_suggestions: Optional[list] = None,
) -> List[str]:
    """Build the per-dataset prompt context (paths, columns, preview) so a
    join/compare request actually receives the CSV paths the prompt promises."""
    parts: List[str] = [
        "\n\n[Context] Multiple datasets are available in this session. "
        "You can load and join/compare them using pandas by reading from the CSV paths below. "
        "Use the dataset aliases when reasoning about them."
    ]

    for meta in multi_dataset_state:
        alias = meta.get("alias") or meta.get("dataset_id")
        data_path = (
            meta.get("data_source_location")
            or meta.get("full_data_location")
            or meta.get("sample_data_location")
        )
        block = [f"\n[Dataset: {alias}]"]
        if meta.get("filename"):
            block.append(f"  filename: {meta['filename']}")
        if meta.get("sheet_name"):
            block.append(
                f"  sheet: '{meta['sheet_name']}' of workbook "
                f"{meta.get('source_workbook') or 'unknown'} (each sheet of that "
                f"workbook is a separate dataset in this session)"
            )
        if data_path:
            block.append(f"  csv_path: {data_path}")
        elif data_path:
            # File was cleaned up; don't advertise an unreadable path or the
            # model will silently fall back to the primary frame.
            block.append("  csv_path: (unavailable — dataset needs re-opening)")
        cols = meta.get("columns")
        if cols:
            block.append(
                f"  columns: {', '.join(map(str, cols)) if isinstance(cols, list) else cols}"
            )
        preview = meta.get("preview")
        if preview:
            try:
                preview_str = json.dumps(preview)
            except (TypeError, ValueError):
                preview_str = str(preview)
            block.append(f"  preview: {preview_str[:1000]}")
        parts.append("\n".join(block))

    if join_key_suggestions:
        parts.append(
            "\n[Suggested join keys]\n" + _format_join_suggestions(join_key_suggestions, limit=5)
        )

    return parts

def _is_new_checkpoint_worthy(sess: dict, new_output_json, exec_succeeded: bool) -> bool:
    if not exec_succeeded:
        return False  # failed runs are scratch, not versions
    last_output_key = sess.get("gcs_output_object_key")
    if not last_output_key:
        return True  # first successful run always counts
    new_fingerprint = hashlib.sha256(
        json.dumps(new_output_json or [], sort_keys=True, default=str).encode()
    ).hexdigest()
    if sess.get("last_output_fingerprint") == new_fingerprint:
        return False  # identical output — not a new version
    return True
#---- send message (NO execute_scope anywhere) ----
@app.post("/threads/{thread_id}/messages", response_model=ChatResponse)
async def send_message(
    request: Request,
    thread_id: str,
    body: MessageCreateIn,
):
    # _ensure_thread_local(thread_id)
    # # Restore chat history from Redis after a process restart (no-op while
    # # the in-process copy is already populated).
    # await hydrate_thread_history(thread_id)
    _t_req = time.monotonic()

    def _tmark(stage: str) -> None:
        # One line per stage, elapsed since the request arrived. Grep "[turn-timing]".
        logger.info("[turn-timing] thread=%s %-18s +%.2fs",
                    thread_id[:8], stage, time.monotonic() - _t_req)

    _ensure_thread_local(thread_id)
    # Restore chat history from Redis after a process restart (no-op while
    # the in-process copy is already populated).
    await hydrate_thread_history(thread_id)
    _tmark("history")

    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.",
        )

    # A password, key or token typed into the chat is sealed before anything
    # reads the message: the history, the logs, every model call and the graph
    # see a token, and only the data transfer agent opens it, to hand the
    # connection to the sealed store (app/core/typed_secrets.py).
    body.content, _sealed_count = seal_typed_secrets(body.content)
    if _sealed_count:
        logger.info("[send_message] sealed %d typed credential(s) in thread %s", _sealed_count, thread_id[:8])

    # optional hint from UI / frontend (for primary dataset)
    dataset_id_hint = (body.metadata or {}).get("dataset_id")
    # analysis id from the ?aid= in the Lovable URL (shared-analysis chats)
    analysis_id_hint = (body.metadata or {}).get("analysis_id") or (body.metadata or {}).get("aid")

    # Check the canonical thread even when the requester already has a session.
    # Profile/team grants carry an effective level; legacy Enterprise returns
    # no level and retains its existing authorization path.
    message_plan = await asyncio.to_thread(resolve_plan, user_id)
    shared_thread = (await _resolve_shared_session_for_thread(thread_id, user_id)
                     if message_plan != "enterprise" else None)
    if shared_thread and shared_thread.get("share_access_level", 3) < 3:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Edit access and a Professional plan are required to run a shared analysis.")

    # user_id is the REQUESTER (auth.uid()). owner_user_id is whose sessions we
    # actually read: same for the owner, but for an authorized collaborator it
    # becomes the analysis owner's auth.uid() so the existing owner-scoped
    # session/dataset guards below still pass (shared-thread model).
    owner_user_id = user_id

    # ----------------------------------------------------
    # 1) Resolve session: header/cookie -> thread binding -> dataset lookup
    # ----------------------------------------------------
    sid = _resolve_session_id(request, None) or await get_thread_session(thread_id)
    sess = await get_session(sid) if sid else None

    # reject stale session not owned by this user
    if sid and (not sess or sess.get("user_id") != user_id):
        sid, sess = None, None

    # fallback: find by dataset_id hint
    if not sid and dataset_id_hint:
        sid = await find_session_by_dataset_for_user(dataset_id_hint, user_id)
        if sid:
            sess = await get_session(sid)

    # fallback: SHARED ANALYSIS. A collaborator has no session of their own —
    # it lives under the OWNER's auth.uid(), so every lookup above misses and we
    # 400. The analyses row holds the canonical session_id/thread_id; resolve by
    # thread_id (or ?aid=), authorize, and rehydrate onto the owner's session.
    if not sid or not sess:
        shared = await _resolve_shared_session_for_thread(
            thread_id, user_id, analysis_id=analysis_id_hint
        )
        if shared and shared.get("share_access_level", 3) < 3:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Edit access is required to collaborate on this analysis.")
        if shared and shared.get("session_id"):
            candidate_sid = shared["session_id"]
            candidate_sess = await get_session(candidate_sid)

            # Redis TTL expired (session created ~a week ago, evicted) —
            # rebuild from the durable analyses row instead of 400ing.
            if not (isinstance(candidate_sess, dict) and candidate_sess.get("user_id")):
                candidate_sess = await _rehydrate_session_from_analysis(shared, candidate_sid)


            if isinstance(candidate_sess, dict) and candidate_sess.get("user_id"):
                sid, sess = candidate_sid, candidate_sess
                owner_user_id = candidate_sess["user_id"]
                if shared.get("role") == "collaborator":
                    logger.info(
                        "[send_message] collaborator=%s rehydrated onto shared "
                        "analysis session=%s owner=%s thread=%s",
                        user_id, sid, owner_user_id, thread_id,
                    )

    if not sid or not sess:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No session found. Pass X-Avaloka-Session header/cookie or include metadata.dataset_id that belongs to this user.",
        )
    
    _tmark("session_resolved")
    # ----------------------------------------------------
    # 2) Multi-upload group: group_session_id owns dataset index
    # ----------------------------------------------------
    group_sid = sess.get("group_session_id") or sid
    group_sess = sess
    if group_sid != sid:
        try:
            gs = await get_session(group_sid)
            #if isinstance(gs, dict) and gs.get("user_id") == user_id:
            if isinstance(gs, dict) and gs.get("user_id") == owner_user_id:
                group_sess = gs
        except Exception:
            group_sess = sess

    # bind thread to GROUP session (stable thread across all datasets in this upload group)
    await bind_thread_session(thread_id, group_sid)
    await refresh_session_ttl(group_sid)

    # ----------------------------------------------------
    # 3) Dataset selection (user-selected subset or all)
    # ----------------------------------------------------
    requested_ids = body.dataset_ids  # <-- MAIN ENTRY POINT FROM UI
    if requested_ids is not None and len(requested_ids) == 0:
        # treat empty list as "no preference" => use all in group
        requested_ids = None

    if requested_ids:
        # keep order, de-dupe
        selected_ids = list(dict.fromkeys(requested_ids))
    else:
        # default: the datasets in use for this chat (all of a workbook's
        # sheets; the files the user picked when several were uploaded)
        selected_ids = _group_selected_ids(group_sess)
        if not selected_ids:
            fallback = dataset_id_hint or sess.get("dataset_id")
            selected_ids = [fallback] if fallback else []

    # If metadata.dataset_id is provided, make that PRIMARY dataset
    explicit_primary = dataset_id_hint if isinstance(dataset_id_hint, str) else None
    if explicit_primary:
        if explicit_primary in selected_ids:
            selected_ids = [explicit_primary] + [x for x in selected_ids if x != explicit_primary]
        else:
            selected_ids = [explicit_primary] + selected_ids
            selected_ids = list(dict.fromkeys(selected_ids))

    if not selected_ids:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No dataset_ids available to load.")

    # dataset_id -> session_id map (from multi-upload)
    session_map = group_sess.get("dataset_session_map") or {}
    if not isinstance(session_map, dict):
        session_map = {}

    dataset_sessions: List[Tuple[str, str, Dict[str, Any]]] = []
    for dsid in selected_ids:
        ds_sid = session_map.get(dsid)
        if not ds_sid:
            ds_sid = await find_session_by_dataset_for_user(dsid, owner_user_id)
        if not ds_sid:
            continue

        ds_sess = await get_session(ds_sid)
        if not isinstance(ds_sess, dict) or ds_sess.get("user_id") != owner_user_id:
            continue

        dataset_sessions.append((dsid, ds_sid, ds_sess))
        try:
            await refresh_session_ttl(ds_sid)
        except Exception:
            pass

    if not dataset_sessions:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No datasets available to load.")

    # ----------------------------------------------------
    # 4) AUTO-ROUTE when user did NOT explicitly pick dataset_ids or dataset_id hint
    # ----------------------------------------------------
    auto_route = (requested_ids is None) and (explicit_primary is None) and (len(dataset_sessions) > 1)
    if auto_route:
        try:
            best = _pick_best_dataset_id(
                body.content,
                [(dsid, ds_sess) for (dsid, _sid, ds_sess) in dataset_sessions],
            )
            if best:
                dataset_sessions.sort(key=lambda x: 0 if x[0] == best else 1)
        except Exception:
            # best-effort; ignore failure
            pass

    # PRIMARY dataset for actual code execution
    primary_dsid, primary_sid, primary_sess = dataset_sessions[0]
    # The session as this turn read it: what a merge compares against to
    # carry the turn's deletions into the stored copy.
    _primary_baseline = dict(primary_sess)
    active_local_source = (
        primary_sess.get("active_data_source_location_local")
        or primary_sess.get("active_data_source_location")
    )
    active_cloud_source = primary_sess.get("active_data_source_location_cloud")

    # Durable snapshot for rebuild-on-miss. Write from the live session until it
    # lands (the frontend may create the analyses row only after the first turn),
    # then stop. Keyed by this request's thread_id, which matches the row.
    if not _truthy_session_value(primary_sess.get("snapshot_persisted")):
        # A durability backup for rebuild-on-miss: nothing in this turn reads
        # it, so the user must not wait on the Supabase round-trip.
        async def _persist_snapshot_bg(_tid=thread_id, _sid=primary_sid,
                                       _snap=dict(primary_sess)) -> None:
            try:
                if await persist_session_snapshot(_tid, _snap) > 0:
                    await update_session(
                        _sid, lambda cur: cur.update({"snapshot_persisted": True})
                    )
            except Exception:
                logger.debug("[snapshot] background persist failed", exc_info=True)

        _snap_task = asyncio.create_task(_persist_snapshot_bg())
        _upload_storage_tasks.add(_snap_task)          # keep a strong ref
        _snap_task.add_done_callback(_upload_storage_tasks.discard)
    _tmark("snapshot")

    # ----------------------------------------------------
    # 4.5) Fidelity + selected-sample resolution (chat-driven)
    # ----------------------------------------------------
    _text_requested_fidelity = _parse_fidelity_from_text(body.content)
    _text_requested_sample_name = _parse_selected_sample_from_text(body.content)
    _requested_fidelity = _text_requested_fidelity   # the UI switch writes the session directly
    _requested_sample_name = (
        _text_requested_sample_name
        or body.selected_sample_name
        or (body.metadata or {}).get("selected_sample_name")
    )

    _available_samples = _maybe_json_load(primary_sess.get("available_samples")) or primary_sess.get("available_samples") or []
    if not isinstance(_available_samples, list):
        _available_samples = []

    # If session available_samples is stale/empty, refresh from Supabase portfolio
    if _requested_sample_name and _requested_sample_name not in _available_samples:
        try:
            _fresh_portfolio = await asyncio.to_thread(load_portfolio, primary_dsid)
            if _fresh_portfolio:
                _available_samples = [k for k in _fresh_portfolio.keys() if k not in ("quick", "full")] or list(_fresh_portfolio.keys())
                primary_sess["available_samples"] = json.dumps(_available_samples)
                if not primary_sess.get("portfolio_samples") or isinstance(primary_sess.get("portfolio_samples"), str):
                    pass  # don't store full portfolio in session; it's in Supabase
                _meta = _profile_meta.get(primary_dsid) or {}
                _meta["portfolio_samples"] = _fresh_portfolio
                _meta["available_samples"] = _available_samples
                _profile_meta[primary_dsid] = _meta
                logger.info("[send_message] Refreshed available_samples from Supabase: %s", _available_samples)
        except Exception:
            logger.debug("[send_message] Could not refresh available_samples from Supabase", exc_info=True)

    _current_fidelity = primary_sess.get("analysis_fidelity")
    if _requested_fidelity in (FIDELITY_QUICK, FIDELITY_PORTFOLIO, FIDELITY_ENTIRE):
        _current_fidelity = _requested_fidelity
        primary_sess["analysis_fidelity"] = _current_fidelity

    if not _current_fidelity:
        _current_fidelity = (
            FIDELITY_QUICK
            if str(primary_sess.get("sample_status")) == "quick_sample"
            else FIDELITY_PORTFOLIO
        )
        primary_sess["analysis_fidelity"] = _current_fidelity

    if _requested_sample_name:
        if _available_samples and _requested_sample_name not in _available_samples:
            valid = ", ".join(str(x) for x in _available_samples)
            msg = (
                f"Sample '{_requested_sample_name}' is not available. "
                f"Available samples: {valid}"
            )
            lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
            lc_seed.append(HumanMessage(body.content))
            lc_seed.append(_said(msg))
            THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
            await persist_thread_history(thread_id)
            return ChatResponse(
                messages=[{"role": "user", "content": body.content}, _assistant_entry(msg)],
                datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
                active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
                analysis_fidelity=_current_fidelity,
                selected_sample_name=primary_sess.get("selected_sample_name"),
                execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            )
        primary_sess["selected_sample_name"] = _requested_sample_name

    _fallback_preview = _maybe_json_load(primary_sess.get("uploaded_csv_preview")) or primary_sess.get("uploaded_csv_preview") or []
   
    _resolved_rows, _resolved_name = await asyncio.to_thread(
        _resolve_selected_sample_rows,
        primary_sess,
        primary_dsid,
        fallback_rows=_fallback_preview if isinstance(_fallback_preview, list) else [],
    )
    if _resolved_name and primary_sess.get("selected_sample_name") != _resolved_name:
        primary_sess["selected_sample_name"] = _resolved_name
    _tmark("sample_rows")

    _file_size_bytes = int(primary_sess.get("file_size_bytes") or 0)
    _is_large_dataset = _file_size_bytes >= LARGE_DATASET_THRESHOLD_BYTES
    # Byte-aware routing signal — computed ONCE here, before the planner;
    # written into graph state below so downstream nodes read the stored value.
    _estimated_full_bytes, _bytes_per_row_val, _inmem_source = _estimated_inmemory_bytes(primary_sess)
    _byte_mode = _byte_routing_enabled()
    _is_folder_dataset_early = bool(primary_sess.get("folder_read_path"))
    # Two paths only under byte routing: below cap -> whole dataset, answer
    # normally; at/above cap -> defer, no partial result. Folder datasets keep
    # the legacy flow (multi-file cost can't be probed reliably).
    _defer_required = (
        _byte_mode
        and not _is_folder_dataset_early
        and _estimated_full_bytes > 0
        and _estimated_full_bytes >= _analysis_max_inmemory_bytes()
    )
    _bg_task_id = primary_sess.get("background_task_id")
    _runtime_hint = _estimated_runtime_hint(_file_size_bytes)
    _sample_status = str(primary_sess.get("sample_status") or "")
    _mode_switch_message = _is_mode_switch_message(body.content)
    _sample_switch_message = bool(_text_requested_sample_name)

    # ── The team, from the chat (app/api/chat_swarm.py) ──
    # "Swarm this", "send your team", "clone yourself": start a coordinated
    # mission on this thread's dataset through the same service the console
    # uses, capped by the caller's edition, and tell the story back from the
    # mission's own records. The live page is Professional and Enterprise; the
    # chat summary is every edition's.
    if chat_swarm.wants_swarm(body.content, body.metadata):
        _swarm_edition = await _caller_edition(request)
        _swarm_prompt: Optional[Dict[str, Any]] = None
        _swarm_text: Optional[str] = None
        _swarm_label = primary_sess.get("alias") or primary_sess.get("filename") or None
        _pending_mid = chat_swarm.pending_mission(primary_sess)
        if _pending_mid:
            _mission = await chat_swarm.records_for(user_id, _pending_mid)
            if _mission is None:
                chat_swarm.remember(primary_sess, _pending_mid, told=True)
            else:
                _swarm_prompt = chat_swarm.summary_prompt(
                    _mission["records"], edition=_swarm_edition, mission_id=_pending_mid,
                    finished=_mission["finished"], dataset_label=_swarm_label)
                chat_swarm.remember(primary_sess, _pending_mid, told=_mission["finished"])
        if _swarm_prompt is None and chat_swarm.is_status_ask(body.content):
            # Asked after the team with nothing pending: say so, start nothing.
            _swarm_text = chat_swarm.NOTHING_PENDING
        elif _swarm_prompt is None:
            _swarm_path = next((p for p in (
                primary_sess.get("work_local_input"),
                primary_sess.get("active_data_source_location_local"),
                primary_sess.get("sample_local_input"),
            ) if p and Path(str(p)).is_file()), None)
            if not _swarm_path:
                _swarm_text = ("I'd love to send the team, but I can't find a local working copy of this "
                               "dataset to shard. Upload it again and ask me once more.")
            else:
                _goal = chat_swarm.mission_goal(body.content)
                try:
                    _mission = await chat_swarm.run_mission(
                        user_id, dataset_path=str(_swarm_path), goal=_goal, edition=_swarm_edition)
                    _swarm_prompt = chat_swarm.summary_prompt(
                        _mission["records"], edition=_swarm_edition, mission_id=_mission["mission_id"],
                        finished=_mission["finished"], goal=_goal, dataset_label=_swarm_label)
                    chat_swarm.remember(primary_sess, _mission["mission_id"], told=_mission["finished"])
                except Exception as exc:  # noqa: BLE001 - the service refused, or the run died
                    logger.warning("[chat-swarm] mission did not start: %s", type(exc).__name__)
                    _swarm_text = ("I tried to send the team out, but the mission could not start on this "
                                   "dataset. Ask me for a plain analysis and I'll do it myself.")
        if _swarm_prompt is not None:
            _swarm_text = chat_swarm.summary_text(_swarm_prompt)
        await update_session(primary_sid, lambda cur: cur.update(
            {chat_swarm.SESSION_KEY: primary_sess.get(chat_swarm.SESSION_KEY)}))
        lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
        lc_seed.append(HumanMessage(body.content))
        lc_seed.append(_said(_swarm_text or ""))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, _assistant_entry(_swarm_text or "")],
            datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
            active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
            analysis_fidelity=_current_fidelity,
            selected_sample_name=primary_sess.get("selected_sample_name"),
            execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            assistant_prompts=[_swarm_prompt] if _swarm_prompt is not None else None,
        )

    # Ask once before triggering heavy cluster work for large datasets.
    # Under byte routing the prompt has no branch left to serve: below cap
    # "entire" is the only sensible option, above cap we defer — so skip it.
    if _is_large_dataset and not _byte_mode and not _requested_fidelity and not primary_sess.get("fidelity_prompt_shown"):
        primary_sess["fidelity_prompt_shown"] = True
        primary_sess["requires_fidelity_choice"] = True
        primary_sess["fidelity_prompt"] = _build_fidelity_prompt(_file_size_bytes)
        primary_sess["estimated_runtime_hint"] = _runtime_hint
        await update_session(primary_sid, lambda cur: cur.update(primary_sess))
        prompt_msg = _build_fidelity_prompt(_file_size_bytes)
        lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
        lc_seed.append(HumanMessage(body.content))
        lc_seed.append(_said(prompt_msg))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, _assistant_entry(prompt_msg)],
            datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
            active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
            analysis_fidelity=_current_fidelity,
            selected_sample_name=primary_sess.get("selected_sample_name"),
            execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            analysis_task_id=primary_sess.get("analysis_task_id"),
        )

    # ── Byte-gated pre-graph handling (flag on, dataset at/above the cap) ──
    if (
        _defer_required
        and not _mode_switch_message
        and not _sample_switch_message
        and not _TASK_OP_RE.search(body.content or "")  # task ops go to the planner
    ):
        _early_msg: Optional[str] = None

        # 1) Metadata-only questions answer instantly regardless of size —
        #    never queue a job for "how many columns does this have".
        if _is_metadata_only_question(body.content):
            _early_msg = _build_metadata_answer(primary_sess, body.content)

        # 2) A full-dataset job already in flight: return its handle instead of
        #    scheduling a duplicate. Time-bounded, so a task that was purged or
        #    lost (no result anywhere, no finished run record) cannot lock the
        #    dataset on a stale job card permanently.
        _sched_at = float(primary_sess.get("analysis_task_scheduled_at") or 0)
        try:
            _inflight_ttl = int(os.getenv("ANALYSIS_DEFER_INFLIGHT_TTL_S", "3600"))
        except ValueError:
            _inflight_ttl = 3600
        _job_recent = _sched_at > 0 and (time.time() - _sched_at) < _inflight_ttl
        if _early_msg is None and _job_recent and _scheduled_task_is_in_flight(primary_sess.get("analysis_task_id")):
            _early_msg = _build_deferred_job_card(
                task_id=str(primary_sess.get("analysis_task_id")),
                dataset_name=str(primary_sess.get("alias") or primary_sess.get("filename") or primary_dsid),
                interpreted_question=body.content,
                eta_hint=_estimated_runtime_hint(_file_size_bytes),
            ) + "\n\n_(This job was already running — no new job was scheduled.)_"

        # 3) No cloud path: we can neither run it interactively (over budget)
        #    nor schedule it. Say so plainly; no sampled numbers.
        _cloud_uri_early = _full_cloud_uri(primary_sess) or ""
        if _early_msg is None and not (_cloud_uri_early and primary_sess.get("connection_id")):
            _early_msg = (
                "This dataset's estimated in-memory size "
                f"({_human_size(_estimated_full_bytes)}) exceeds the interactive analysis "
                f"budget ({_human_size(_analysis_max_inmemory_bytes())}), and it has no "
                "cloud connection to run as a background job. Register it from cloud "
                "storage (or raise ANALYSIS_MAX_INMEMORY_BYTES) to analyze it in full. "
                "No partial or sampled results are shown for over-budget datasets."
            )

        if _early_msg is not None:
            lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
            lc_seed.append(HumanMessage(body.content))
            lc_seed.append(_said(_early_msg))
            THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
            await persist_thread_history(thread_id)
            return ChatResponse(
                messages=[{"role": "user", "content": body.content}, _assistant_entry(_early_msg)],
                datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
                active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
                analysis_fidelity=_current_fidelity,
                selected_sample_name=primary_sess.get("selected_sample_name"),
                execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
                analysis_task_id=primary_sess.get("analysis_task_id"),
            )

    if _sample_switch_message and _requested_sample_name:
        msg = _planner_build_sample_switch_confirmation(_requested_sample_name)
        await update_session(primary_sid, lambda cur: cur.update(primary_sess))
        lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
        lc_seed.append(HumanMessage(body.content))
        lc_seed.append(_said(msg))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[_assistant_entry(msg)],
            datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
            active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
            analysis_fidelity=_current_fidelity,
            selected_sample_name=primary_sess.get("selected_sample_name"),
            execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            analysis_task_id=primary_sess.get("analysis_task_id"),
        )

    # Explicit portfolio choice triggers background portfolio/profile creation once.
    _portfolio_scheduled_now = False
    if (
        _is_large_dataset
        and _current_fidelity == FIDELITY_PORTFOLIO
        and not _bg_task_id
        and str(primary_sess.get("sample_status")) == "quick_sample"
    ):
        _cloud_uri_for_bg = _full_cloud_uri(primary_sess) or ""
        _conn_for_bg = primary_sess.get("connection_id") or ""
        if _cloud_uri_for_bg and _conn_for_bg:
            _scheduled = _schedule_background_sampling_task(
                dataset_id=primary_dsid,
                session_id=primary_sid,
                cloud_uri=_cloud_uri_for_bg,
                connection_id=_conn_for_bg,
                file_size_bytes=_file_size_bytes,
                ray_namespace=os.getenv("RAY_NAMESPACE", "default"),
            )
            if _scheduled:
                primary_sess["background_task_id"] = _scheduled
                primary_sess["portfolio_generation_status"] = "scheduled"
                primary_sess["requires_fidelity_choice"] = False
                _bg_task_id = _scheduled
                _portfolio_scheduled_now = True
                logger.info(
                    "[send_message] Scheduled portfolio build %s for dataset %s after user choice",
                    _scheduled,
                    primary_dsid,
                )

    _resolved_data_source_label = (
        "entire_dataset"
        if _current_fidelity == FIDELITY_ENTIRE
        else (_resolved_name or "quick_sample")
    )
    # Sub-1 GB datasets always have their full file locally, so there is
    # nothing to wait for; only >=1 GB portfolios are built in the background.
    _portfolio_ready = _sample_status == "full_sample" or not _is_large_dataset
    _effective_fidelity = _current_fidelity
    _analysis_task_in_flight = False
    if not _mode_switch_message:
        if _current_fidelity == FIDELITY_PORTFOLIO and not _portfolio_ready:
            _effective_fidelity = FIDELITY_QUICK
        elif _current_fidelity == FIDELITY_ENTIRE:
            _analysis_task_in_flight = _entire_job_in_flight(primary_sess)
            if _analysis_task_in_flight:
                _effective_fidelity = FIDELITY_PORTFOLIO if _portfolio_ready else FIDELITY_QUICK
            elif primary_sess.get("analysis_task_id"):
                # Finished, failed or stale: release the lock so this turn runs in full.
                logger.info("[send_message] clearing stale analysis_task_id %s",
                            primary_sess.get("analysis_task_id"))
                primary_sess["analysis_task_id"] = None
                primary_sess["analysis_task_scheduled_at"] = None

    _effective_analysis_source_label = (
        "entire_dataset"
        if _effective_fidelity == FIDELITY_ENTIRE
        else (_resolved_name or "quick_sample")
    )
    primary_sess["resolved_analysis_source"] = _resolved_data_source_label
    if _requested_fidelity in (FIDELITY_QUICK, FIDELITY_PORTFOLIO, FIDELITY_ENTIRE):
        primary_sess["requires_fidelity_choice"] = False

    await update_session(primary_sid, lambda cur: cur.update(primary_sess))

    if _portfolio_scheduled_now and _mode_switch_message:
        msg = _planner_build_mode_switch_confirmation(FIDELITY_PORTFOLIO, _file_size_bytes)
        lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
        lc_seed.append(HumanMessage(body.content))
        lc_seed.append(_said(msg))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, _assistant_entry(msg)],
            datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
            active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
            analysis_fidelity=_current_fidelity,
            selected_sample_name=primary_sess.get("selected_sample_name"),
            execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            analysis_task_id=primary_sess.get("analysis_task_id"),
        )
    if _mode_switch_message and _current_fidelity == FIDELITY_PORTFOLIO and _portfolio_ready:
        msg = (
            "Switched to mode portfolio samples.\n"
            "Portfolio samples are ready. You can continue analysis using the selected portfolio sample."
        )
        lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
        lc_seed.append(HumanMessage(body.content))
        lc_seed.append(_said(msg))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, _assistant_entry(msg)],
            datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
            active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
            analysis_fidelity=_current_fidelity,
            selected_sample_name=primary_sess.get("selected_sample_name"),
            execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            analysis_task_id=primary_sess.get("analysis_task_id"),
        )
    if _mode_switch_message:
        msg = _planner_build_mode_switch_confirmation(_current_fidelity, _file_size_bytes)
        lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
        lc_seed.append(HumanMessage(body.content))
        lc_seed.append(_said(msg))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, _assistant_entry(msg)],
            datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
            active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
            analysis_fidelity=_current_fidelity,
            selected_sample_name=primary_sess.get("selected_sample_name"),
            execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            analysis_task_id=primary_sess.get("analysis_task_id"),
        )

    # ── "reset dataset" fast path: restore the pre-transformation snapshot ──
    if _RESET_DATASET_RE.search(body.content or ""):
        if _restore_original_dataset_session(primary_sess):
            def _merge_restored(cur):
                _merge_session_with_drops(cur, primary_sess, _primary_baseline,
                                          drop=("original_dataset_snapshot",))

            await update_session(primary_sid, _merge_restored)
            msg = (
                "Restored the original uploaded dataset. "
                "Your next questions will run against the full dataset again."
            )
        else:
            msg = "You're already working with the original uploaded dataset."
        lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
        lc_seed.append(HumanMessage(body.content))
        lc_seed.append(_said(msg))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, _assistant_entry(msg)],
            datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
            active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
            analysis_fidelity=_current_fidelity,
            selected_sample_name=primary_sess.get("selected_sample_name"),
            execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            analysis_task_id=primary_sess.get("analysis_task_id"),
        )


    # ── Metadata fast path for small datasets: row/column counts and schema are
    #    already known exactly from ingest, so answer without running code.
    if (
        len(dataset_sessions) == 1
        and not _is_large_dataset
        and not _truthy_session_value(primary_sess.get("data_source_was_modified"))  # stored shape would be stale
        and str(primary_sess.get("sample_status") or "") == "full_sample"           # row count is exact
        and _is_metadata_only_question(body.content)
    ):
        _meta_msg = _build_metadata_answer(primary_sess, body.content)
        if _meta_msg:
            lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
            lc_seed.append(HumanMessage(body.content))
            lc_seed.append(_said(_meta_msg))
            THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
            await persist_thread_history(thread_id)
            return ChatResponse(
                messages=[{"role": "user", "content": body.content},
                          _assistant_entry(_meta_msg)],
                datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
                active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
                analysis_fidelity=_current_fidelity,
                selected_sample_name=primary_sess.get("selected_sample_name"),
                execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            )

    # ── What predicts X: the shared driver ranking, said from a template ──
    #    (app/agents/drivers_route.py). Upload and DB-table threads alike; no
    #    coder, no LLM, so every figure shown is the ranking's own. A question
    #    that asks for a model ("train a random forest ...") falls through.
    #    The decision is the graph's first node's (answer_drivers_turn); this
    #    only answers before the graph state is built.
    if is_drivers_question(body.content):
        _drivers = await asyncio.to_thread(
            answer_drivers_turn, _drivers_inputs(primary_sess, len(dataset_sessions)), body.content)
        if _drivers:
            _drv_msg, _drv_table = _drivers
            lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
            lc_seed.append(HumanMessage(body.content))
            lc_seed.append(_said(_drv_msg))
            THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
            await persist_thread_history(thread_id)
            return ChatResponse(
                messages=[{"role": "user", "content": body.content},
                          _assistant_entry(_drv_msg)],
                output_tables=[_drv_table],
                output_json=_drv_table["rows"],
                datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
                active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
                analysis_fidelity=_current_fidelity,
                selected_sample_name=primary_sess.get("selected_sample_name"),
                execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            )

    # ── Database aggregate pushdown: a rate / mean / count by one group column,
    #    asked of a table analysed from a database, is computed by the database
    #    over every row instead of estimated from the analysis sample.
    if (
        len(dataset_sessions) == 1
        and primary_sess.get("db_pushdown_table")
        and not _truthy_session_value(primary_sess.get("data_source_was_modified"))
        and not (_maybe_json_load(primary_sess.get("pending_clarification")) or primary_sess.get("pending_clarification"))
    ):
        _pushdown = await _db_pushdown_answer(user_id, primary_sess, body.content)
        if _pushdown:
            _push_msg, _push_table = _pushdown
            lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
            lc_seed.append(HumanMessage(body.content))
            lc_seed.append(_said(_push_msg))
            THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
            await persist_thread_history(thread_id)
            return ChatResponse(
                messages=[{"role": "user", "content": body.content},
                          _assistant_entry(_push_msg)],
                output_tables=[_push_table],
                output_json=_push_table["rows"],
                datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
                active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
                analysis_fidelity=_current_fidelity,
                selected_sample_name=primary_sess.get("selected_sample_name"),
                execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            )

    # ----------------------------------------------------
    # 5) History (LangChain messages in memory)
    # ----------------------------------------------------
    tmeta = THREAD_META[thread_id]
    # # ----------------------------------------------------
    # # 5) History (LangChain messages in memory)
    # # ----------------------------------------------------
    # tmeta = THREAD_META[thread_id]
    lc_msgs: List[Any] = tmeta.get("lc_msgs", [])

    # ----------------------------------------------------
    # 6) Dataset dropdown + viz maps (for UI)
    # ----------------------------------------------------
    datasets_dropdown: List[Dict[str, Any]] = []
    visualization_configs: Dict[str, Any] = {}
    visualization_statuses: Dict[str, str] = {}

    def _get_viz(ds_sess: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        vc = _maybe_json_load(ds_sess.get("visualization_config"))
        vs = ds_sess.get("visualization_status")
        return vc, vs

    for dsid, _ds_sid, ds_sess in dataset_sessions:
        datasets_dropdown.append(
            {
                "dataset_id": dsid,
                "filename": ds_sess.get("filename"),
                "alias": ds_sess.get("alias"),
            }
        )
        vc, vs = _get_viz(ds_sess)
        visualization_configs[dsid] = vc or {}
        if vs:
            visualization_statuses[dsid] = vs


    # ----------------------------------------------------
    # 6.2) Re-materialize sibling local samples so every csv_path advertised in
    #      the multi-dataset prompt actually exists on disk. This is what makes
    #      "Open for analysis" on an existing group behave like a fresh upload;
    #      the primary is handled separately by _ensure_local_inputs below.
    # ----------------------------------------------------
    if len(dataset_sessions) > 1:
        for _dsid, _ds_sid, _ds_sess in dataset_sessions[1:]:
            _rebuilt = await _ensure_dataset_local_sample(_ds_sess)
            if _rebuilt:
                _patch = {
                    "sample_local_input": _ds_sess.get("sample_local_input"),
                    "work_local_input": _ds_sess.get("work_local_input"),
                }
                try:
                    await update_session(_ds_sid, lambda cur, p=_patch: cur.update(p))
                except Exception:
                    logger.debug("[ensure-sibling] persist failed for %s", _ds_sid, exc_info=True)

    # ----------------------------------------------------
    # 7) Build multi-dataset state (for cross-dataset join/compare)
    # ----------------------------------------------------
    multi_dataset_state: List[Dict[str, Any]] = []

    # We'll also use this inside the augmented prompt
    for i, (dsid, ds_sid, ds_sess) in enumerate(dataset_sessions):
        ds_alias = ds_sess.get("alias") or dsid
        ds_filename = ds_sess.get("filename")
        ds_sample = ds_sess.get("sample_local_input")
        ds_work = ds_sess.get("work_local_input")
        ds_active_local = (
            ds_sess.get("active_data_source_location_local")
            or ds_sess.get("active_data_source_location")
        )
        ds_active_cloud = ds_sess.get("active_data_source_location_cloud")

        ds_schema = _maybe_json_load(ds_sess.get("schema")) or ds_sess.get("schema")
        ds_cols = _maybe_json_load(ds_sess.get("uploaded_csv_columns")) or ds_sess.get("uploaded_csv_columns")
        ds_preview = _maybe_json_load(ds_sess.get("uploaded_csv_preview")) or ds_sess.get("uploaded_csv_preview")
        ds_preview_head = (ds_preview or [])[:5] if isinstance(ds_preview, list) else ds_preview

        sample_path = ds_sample and _posix(ds_sample)
        full_path = ds_work and _posix(ds_work)
        active_path = (
            (ds_active_cloud and _posix(ds_active_cloud))
            or (ds_active_local and _posix(ds_active_local))
        )

        ds_stats = _maybe_json_load(ds_sess.get("sample_statistics")) or ds_sess.get("sample_statistics")
        ds_row_count = None
        if isinstance(ds_stats, dict):
            ds_row_count = (ds_stats.get("data_shape") or {}).get("rows")

        multi_dataset_state.append(
            {
                "dataset_id": dsid,
                "session_id": ds_sid,
                "alias": ds_alias,
                "filename": ds_filename,
                "columns": ds_cols,
                "preview": ds_preview_head,
                "schema": ds_schema,
                "ddl_schema": ds_sess.get("ddl_schema"),
                "row_count": ds_row_count,
                "sample_data_location": sample_path,
                "full_data_location": full_path,
                "active_data_source_location": active_path,
                "active_data_source_location_cloud": ds_active_cloud,
                #"data_source_location": active_path or full_path or sample_path,
                "data_source_location": _first_usable_path(active_path, full_path, sample_path),
                # Excel-sheet provenance (set when this dataset is one sheet of
                # an uploaded multi-sheet workbook)
                "sheet_name": ds_sess.get("sheet_name"),
                "source_workbook": ds_sess.get("source_workbook"),
            }
        )
    # ---- Join key suggestions (for compare/join flows) ----
    join_key_suggestions = await asyncio.to_thread(suggest_join_keys, multi_dataset_state)
    _tmark("join_keys")


    # ----------------------------------------------------
    # 7.5) Deterministic routing for compare/join (before planner)
    # ----------------------------------------------------
    user_text = (body.content or "").lower()

    wants_compare = ("compare" in user_text) and ("dataset" in user_text or "datasets" in user_text)
    # wants_join = any(k in user_text for k in ["join", "merge"])

    # # detect explicit join key instructions like: "join on customer_id"
    # has_explicit_keys = bool(re.search(r"\b(join|merge)\s+(on|by)\s+\w+", user_text))

    wants_join = bool(re.search(r"\b(join|merge)\b", user_text))

    # detect: "join ... on customer_id" (allows words between join and on/by)
    has_explicit_keys = bool(
        re.search(r"\b(join|merge)\b.*\b(on|by)\b\s+[a-zA-Z_][\w\.]*(\s*,\s*[a-zA-Z_][\w\.]*)*", user_text)
    )


    if len(multi_dataset_state) > 1 and wants_compare:
        msg = "Compare datasets (join-key discovery):\n\n"
        if join_key_suggestions:
            msg += "Suggested join keys:\n" + _format_join_suggestions(join_key_suggestions, limit=5)
            msg += "\n\nReply like: `Join on customer_id`"
        else:
            msg += "No reliable join keys found from sample. Please choose join columns manually."

        # keep thread history consistent
        lc_msgs.append(HumanMessage(body.content))
        lc_msgs.append(_said(msg))
        tmeta["lc_msgs"] = _limit_messages(_compact_messages(lc_msgs), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)

        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, _assistant_entry(msg)],
            datasets=datasets_dropdown,
            active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
            visualization_configs=visualization_configs,
            visualization_statuses=visualization_statuses,
            visualization_config=visualization_configs.get(primary_dsid),
            visualization_status=visualization_statuses.get(primary_dsid),
            analysis_fidelity=_current_fidelity,
            selected_sample_name=primary_sess.get("selected_sample_name"),
            execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
        )

    if len(multi_dataset_state) > 1 and wants_join and not has_explicit_keys:
        msg = "Join without a key isn’t possible. Which columns should I join on?\n\n"
        if join_key_suggestions:
            msg += "Suggested join keys:\n" + _format_join_suggestions(join_key_suggestions, limit=5)
        else:
            msg += "Suggested join keys:\n- No reliable join keys found from sample."
        msg += "\n\nReply like: `Join on customer_id`"

        # keep thread history consistent
        lc_msgs.append(HumanMessage(body.content))
        lc_msgs.append(_said(msg))
        tmeta["lc_msgs"] = _limit_messages(_compact_messages(lc_msgs), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)

        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, _assistant_entry(msg)],
            datasets=datasets_dropdown,
            active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
            visualization_configs=visualization_configs,
            visualization_statuses=visualization_statuses,
            visualization_config=visualization_configs.get(primary_dsid),
            visualization_status=visualization_statuses.get(primary_dsid),
            analysis_fidelity=_current_fidelity,
            selected_sample_name=primary_sess.get("selected_sample_name"),
            execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
        )

    # ----------------------------------------------------
    # 8) AUGMENT user message with context for ALL selected datasets
    #     (including file paths so LLM can join/compare)
    # ----------------------------------------------------
    augmented_parts = [body.content]
    augmented_parts.append(
        "\n[Analysis context] "
        f"Current fidelity: {_current_fidelity}; "
        f"Effective fidelity for this request: {_effective_fidelity}; "
        f"Selected sample: {primary_sess.get('selected_sample_name') or 'none'}; "
        f"Resolved analysis source: {_effective_analysis_source_label}; "
        f"Dataset size bytes: {int(primary_sess.get('file_size_bytes') or 0)}"
    )
    _ctx_db_sample = DbSample.from_session(
        _maybe_json_load(primary_sess.get("db_sample")) or primary_sess.get("db_sample"))
    if _ctx_db_sample:
        augmented_parts[-1] += "; " + sample_context_line(_ctx_db_sample)

    if len(multi_dataset_state) > 1:
        augmented_parts.extend(
            _build_multi_dataset_context(multi_dataset_state, join_key_suggestions)
        )

    augmented_content = "\n".join(augmented_parts)
    lc_msgs.append(HumanMessage(augmented_content))

    lc_msgs = _compact_messages(lc_msgs)
    lc_msgs = _limit_messages(lc_msgs, MAX_CONTEXT_TURNS)
    tmeta["lc_msgs"] = lc_msgs
    await persist_thread_history(thread_id)

    # Section 9 — Primary dataset paths & schema
    sample_local_input = primary_sess.get("sample_local_input")
    work_local_input   = primary_sess.get("work_local_input")

    local_path = (
        (active_local_source and _posix(active_local_source))
        or (work_local_input and _posix(work_local_input))
        or (sample_local_input and _posix(sample_local_input))
    )
    cloud_uri = active_cloud_source or _full_cloud_uri(primary_sess)

    # ── SAFE DEFAULTS (prevent UnboundLocalError if early-return paths skip assignments) ──
    schema_for_state = _normalize_schema_for_state(
        _maybe_json_load(primary_sess.get("schema")) or primary_sess.get("schema")
    )

    if primary_sess.get("work_dir") and not Path(primary_sess["work_dir"]).exists():
        Path(primary_sess["work_dir"]).mkdir(parents=True, exist_ok=True)

    meta             = body.metadata or {}
    execution_mode   = meta.get("execution_mode")
    ray_workers      = int(meta.get("ray_workers") or 3)
    ray_namespace    = meta.get("ray_namespace") or "ray-training"
    ray_timeout_s    = int(meta.get("ray_timeout_s") or 1800)

    if not execution_mode:
        if _current_fidelity == FIDELITY_QUICK:
            execution_mode = "local"
        elif _current_fidelity == FIDELITY_PORTFOLIO:
            execution_mode = (
                "k8s-ray"
                if (_is_large_dataset and cloud_uri and primary_sess.get("connection_id"))
                else "local"
            )
        elif _current_fidelity == FIDELITY_ENTIRE:
            execution_mode = (
                "k8s-ray"
                if (_is_large_dataset and cloud_uri and primary_sess.get("connection_id"))
                else "local"
            )
        else:
            execution_mode = "local"

    local_path = (
        (active_local_source and _posix(active_local_source))
        or (work_local_input and _posix(work_local_input))
        or (sample_local_input and _posix(sample_local_input))
    )
    cloud_uri = active_cloud_source or _full_cloud_uri(primary_sess)

    _ray_fallback_reason = None
    if execution_mode == "k8s-ray":
        conn_id = primary_sess.get("connection_id")
        
        # ── GUARD: local file or missing connection → fall back to local execution ──
        if not cloud_uri or not conn_id:
            logger.warning(
                "send_message: execution_mode=k8s-ray but cloud_uri=%s conn_id=%s "
                "-> falling back to local execution",
                cloud_uri, conn_id
            )
            execution_mode = "local"
            # Visible, not quiet: the local node labels the reply and the run.
            _ray_fallback_reason = (
                RAY_NO_CONNECTION_REASON if cloud_uri
                else "the dataset has no cloud copy for Ray to read"
            )


        if primary_sess.get("work_dir") and not Path(primary_sess["work_dir"]).exists():
            Path(primary_sess["work_dir"]).mkdir(parents=True, exist_ok=True)

        # schema may be JSON-encoded; decode then normalize
        schema_for_state = _normalize_schema_for_state(
            _maybe_json_load(primary_sess.get("schema")) or primary_sess.get("schema")
        )

    # ----------------------------------------------------
    # 10) Persisted training state
    # ----------------------------------------------------
    # Intent/routing decisions belong to planner. The server only hydrates
    # existing training context so planner can make that decision.
    enable_training = primary_sess.get("enable_training", False)
    pending_training_plan = _maybe_json_load(primary_sess.get("training_plan")) or primary_sess.get("training_plan")
    pending_training_result = _maybe_json_load(primary_sess.get("training_result")) or primary_sess.get("training_result")
    pending_training_task = _maybe_json_load(primary_sess.get("training_task")) or primary_sess.get("training_task")
    pending_training_completed = bool(
        _maybe_json_load(primary_sess.get("training_completed"))
        if primary_sess.get("training_completed") is not None
        else False
    )
    training_plan_reply_action = "none"

    # The active DB-sample session stores its origin as `db://{customer_id}/{dsid}?sample=true`,
    # but `data_source_location` below is overwritten with the local sample CSV path — so the
    # customer_id (the database the user is analyzing) would be lost. Capture it here so the DTA
    # planner can use it as the implicit transfer source. No api_key needed: the transfer fetches
    # full creds from customer_id alone via customer_dbs.
    _active_db_customer_id: Optional[str] = None
    _active_db_source_table: Optional[str] = None   # the table the user analyzed
    # Prefer the explicit customer_id persisted on the DB-sample session.
    _sess_customer = (primary_sess.get("customer_id") or "").strip()
    if _sess_customer:
        _active_db_customer_id = _sess_customer
        _active_db_source_table = primary_sess.get("source_table") or None
    else:
        # Fall back to parsing any db:// location visible on the session.
        for _loc_key in ("data_source_location", "full_data_location",
                         "sample_data_location", "output_location"):
            _loc = primary_sess.get(_loc_key) or ""
            if isinstance(_loc, str) and _loc.startswith("db://"):
                _db_loc_match = re.match(r"^db://([^/]+)/", _loc)
                if _db_loc_match:
                    _active_db_customer_id = _db_loc_match.group(1)
                    _active_db_source_table = primary_sess.get("source_table") or None
                    break
    # No fallback to "the user's most-recent active database" from another
    # session: a transfer's source is what THIS thread analysed or what the user
    # names. That scan picked a database the user was not looking at, a silent
    # default on a write path (the same class as the J1 ghost copy).
    # The onboarding API's credential routes answer only the customer's own
    # api_key, so the planner carries it sealed (ciphertext under
    # DB_ENCRYPTION_KEY, the same blob the DB session stores) and opens it in
    # process when it calls. Nothing in state is a plaintext credential.
    # Every database the CALLER connected gets a handle, so a transfer can name
    # a destination (or a source) other than the active one; the map is built
    # from the caller's own saved connections (/api/connections, what the UI
    # saves) and connect sessions only. ``user_id`` is the requester,
    # not ``owner_user_id``: a collaborator on a shared analysis gets no handle
    # for the owner's databases, the owner's session blob included.
    _db_credential_handles: Dict[str, Any] = {}
    try:
        _db_credential_handles = await db_credential_handles_for_user(user_id) or {}
    except Exception as _e:
        logger.warning("DTA source bridge: credential handle lookup failed: %s", type(_e).__name__)
    _active_db_credential_handle: Optional[Dict[str, Any]] = None
    if _active_db_customer_id:
        if (_sess_customer == _active_db_customer_id and primary_sess.get("api_key_enc")
                and primary_sess.get("user_id") == user_id):
            _active_db_credential_handle = primary_sess.get("api_key_enc")
        else:
            _active_db_credential_handle = _db_credential_handles.get(_active_db_customer_id)
    _tmark("dta_bridge")
    logger.info(
        "DTA source bridge: input_data_type=%s data_source_location=%r customer_id=%r "
        "source_table=%r connection_id=%r -> active_db_customer_id=%r active_db_source_table=%r",
        primary_sess.get("input_data_type"),
        primary_sess.get("data_source_location"),
        primary_sess.get("customer_id"),
        primary_sess.get("source_table"),
        primary_sess.get("connection_id"),
        _active_db_customer_id,
        _active_db_source_table,
    )

    # The requester's plan gates data transfer and model training in the planner.
    # None when it cannot be looked up: the gate then refuses it as unavailable
    # on a hosted deployment (allows it elsewhere) instead
    # of treating the user as Free.
    try:
        from app.core.plan_limits import try_resolve_plan
        _user_plan = await asyncio.to_thread(try_resolve_plan, user_id)
    except Exception:
        logger.warning("[plan-gate] plan lookup failed for user=%s", user_id, exc_info=True)
        _user_plan = None

    # ----------------------------------------------------
    # 11) Build state for LangGraph (now multi-dataset aware)
    # ----------------------------------------------------
    state_in: Dict[str, Any] = {
        "user_id": user_id,
        # session_id and thread_id stay distinct (avaloka_agent loads Redis
        # context by thread id), plus the one-shot connection_event left by
        # upload / register-existing-storage / database/connect.
        **build_swarm_state(
            thread_id=thread_id, session_id=sid, session=primary_sess
        ),
        "user_plan": _user_plan,
        "messages": lc_msgs,
        "sample_data": _resolved_rows or _fallback_preview or [],
        "uploaded_csv_columns": _maybe_json_load(primary_sess.get("uploaded_csv_columns"))
        or primary_sess.get("uploaded_csv_columns"),
        "uploaded_csv_preview": _resolved_rows or _fallback_preview or [],
        "schema": schema_for_state,
        "complex_structure": _maybe_json_load(primary_sess.get("complex_structure"))
        or primary_sess.get("complex_structure"),
        "ddl_schema": primary_sess.get("ddl_schema"),
        "sample_data_location": sample_local_input and _posix(sample_local_input),
        "full_data_location": work_local_input and _posix(work_local_input),

        "data_source_location_local": local_path,
        "data_source_location_cloud": cloud_uri,
        "full_data_source_location_cloud": cloud_uri,
        "data_source_location": (cloud_uri if execution_mode == "k8s-ray" else local_path),
        "active_data_source_location": (
            (active_cloud_source and _posix(active_cloud_source))
            or (active_local_source and _posix(active_local_source))
        ),
        "active_data_source_location_cloud": active_cloud_source and _posix(active_cloud_source),
        "active_data_source_location_local": active_local_source and _posix(active_local_source),
        "activate_output_as_dataset": False,
        "data_source_was_modified": _truthy_session_value(primary_sess.get("data_source_was_modified")),
        "latest_output_location": primary_sess.get("latest_output_location")
        and _posix(primary_sess["latest_output_location"]),
        "latest_output_location_local": primary_sess.get("latest_output_location_local")
        and _posix(primary_sess["latest_output_location_local"]),
        "latest_output_columns": (
            _maybe_json_load(primary_sess.get("latest_output_columns"))
            or primary_sess.get("latest_output_columns")
            or []
        ),
        "latest_output_row_count": primary_sess.get("latest_output_row_count"),
        "latest_output_created_at": primary_sess.get("latest_output_created_at"),
        "latest_output_is_trainable": (
            _truthy_session_value(primary_sess.get("latest_output_is_trainable"))
            if primary_sess.get("latest_output_is_trainable") is not None
            else None
        ),
        "connection_id": primary_sess.get("connection_id"),
        "active_db_customer_id": _active_db_customer_id,
        "active_db_source_table": _active_db_source_table,
        "active_db_credential_handle": _active_db_credential_handle,
        # Set on every turn (empty when none) so a checkpointed thread never
        # keeps a handle for a database the caller has since disconnected.
        "db_credential_handles": _db_credential_handles,
        "dataset_id": primary_dsid,
        "file_size_bytes": primary_sess.get("file_size_bytes", 0),
        "dataset_size_bytes": primary_sess.get("file_size_bytes", 0),
        "analysis_fidelity": _effective_fidelity,
        "selected_sample_name": primary_sess.get("selected_sample_name"),
        "resolved_analysis_source": _effective_analysis_source_label,


        "execution_mode": execution_mode,
        "ray_workers": ray_workers,
        "ray_namespace": ray_namespace,
        "ray_timeout_s": ray_timeout_s,
        "ray_execution_profile": "batch_heavy",
        "ray_fallback_reason": _ray_fallback_reason,


        
        "output_location": primary_sess.get("output_location")
        and _posix(primary_sess["output_location"]),
        # "input_data_type": primary_sess.get("input_data_type", "csv"),
        # "execution_result": {},
        "input_data_type": primary_sess.get("input_data_type", "csv"),
        # Partitioned folder-as-dataset provenance so the executor rebuilds the
        # exact Daft read (hive_partitioning / iceberg metadata) instead of
        # treating folder_read_path as a single object.
        "folder_table_type": primary_sess.get("folder_table_type"),
        "folder_read_path": primary_sess.get("folder_read_path"),
        "hive_partitioning": _truthy_session_value(primary_sess.get("hive_partitioning")),
        "iceberg_metadata_uri": primary_sess.get("iceberg_metadata_uri"),
        "execution_result": {},
        "execution_stdout": None,
        "execution_stderr": None,
        "execution_error": None,
        "fresh_output_produced": None,
        # The replan loop's record is this turn's only (app/agents/replan.py).
        **_replan_fresh_state(),
        "output_file_data": None,
        "output_json": None,
        "output_tables": None,
        "structured_action_plan": None,
        "multi_action_actions": None,
        "multi_action_missing": None,
        "multi_action_status": None,
        "multi_action_notes": None,
        "deploy_on_k8s": False,

        # Training flags (persist across turns)
        "enable_training": enable_training,
        "training_plan": pending_training_plan,
        "training_result": pending_training_result,
        "ready_to_train": primary_sess.get("ready_to_train", False),
        "training_task": pending_training_task,
        "training_completed": pending_training_completed,
        "mlflow_run_id": primary_sess.get("mlflow_run_id"),
        "training_plan_reply_action": training_plan_reply_action,
        "clear_training_state": False,
        "pending_clarification": (
            _maybe_json_load(primary_sess.get("pending_clarification"))
            or primary_sess.get("pending_clarification")
        ),
        "skip_to_training": False,
        "ready_to_code": False,

        # NEW: multi-dataset info for join/compare
        "active_dataset_id": primary_dsid,
        "active_dataset_ids": [dsid for dsid, _, _ in dataset_sessions],
        "datasets_context": multi_dataset_state,
        "multi_dataset_state": multi_dataset_state,
        
    }

    
    training_state_keys = (
        "training_plan",    # v1.2
        "training_result",  # v1.2
        "training_task",
        "ready_to_train",
        "training_completed",
        "training_metrics",
        "model_artifacts",
        "mlflow_run_id",
        "enable_training",
        "training_plan_reply_action",
    )

    for key in training_state_keys:
        if key in primary_sess:
            state_in[key] = _maybe_json_load(primary_sess.get(key)) or primary_sess.get(key)
    # reset per-run fields
    state_in.update(
        {
            "planner_definition": {},
            # Set by the planner only on the turn a paid feature is refused.
            "plan_upgrade_required": None,
            "planner_graph_path": None,
            "planner_graph_status": None,
            "task_info": {},
            "task_schedule": None,
            "ready_to_summarize": False,
            "ready_to_code": False,
            "coder_definition": {},
            "training_plan_reply_action": training_plan_reply_action,
            "clear_training_state": False,
            # MemorySaver checkpointing retains omitted keys across turns —
            # a stale defer decision must never leak into a later run.
            "analysis_defer_required": None,
        }
    )

    existing_tasks = primary_sess.get("tasks")
    if isinstance(existing_tasks, str):
        try:
            existing_tasks = json.loads(existing_tasks)
        except Exception:
            existing_tasks = []
    if not isinstance(existing_tasks, list):
        existing_tasks = []

    state_in["task_list"] = existing_tasks
    # The graph's first node decides "what predicts X" from these same inputs.
    state_in[DRIVERS_STATE_KEY] = _drivers_inputs(primary_sess, len(dataset_sessions))

    # Byte-aware routing signals: decided once (pre-planner, above); downstream
    # nodes must read these from state instead of recomputing.
    state_in["bytes_per_row"] = _bytes_per_row_val or None
    state_in["estimated_full_bytes"] = _estimated_full_bytes or None

    # ----------------------------------------------------
    # 11.5) Fidelity-based execution policy
    # ----------------------------------------------------

    # if _effective_fidelity == FIDELITY_QUICK:
    _cloud_uri_check = state_in.get("data_source_location_cloud") or ""
    _conn_id_check = state_in.get("connection_id")

    # A partitioned folder table must be read via its cloud folder_read_path
    # (glob / iceberg metadata) for portfolio/entire runs — never the local
    # sample CSV — regardless of the 1 GB threshold.
    _is_folder_dataset = bool(primary_sess.get("folder_read_path"))
    _folder_read_path = primary_sess.get("folder_read_path")

    # ── Byte routing: two paths only (flag on, non-folder). Below cap: whole
    #    dataset, answer normally. At/above cap: defer as a scheduled job —
    #    never execute, never return a partial result. The legacy quick/
    #    portfolio/entire ladder below stays intact for flag-off rollback.
    if (
        _byte_mode
        and not _is_folder_dataset
        and not _mode_switch_message
        and _inmem_source != "none"  # no size signal (e.g. db:// samples) -> legacy path
    ):
        _byte_local_candidate = (
            (active_local_source and _posix(active_local_source))
            or (work_local_input and _posix(work_local_input))
            or local_path
        )
        # A head/sample file is NOT the full frame — entire-dataset semantics
        # on it would resurrect the undisclosed-sampling bug. If the full
        # object lives in the cloud, defer to the full-data job instead.
        _local_is_partial = bool(_byte_local_candidate) and str(_byte_local_candidate).endswith(
            ("streamed_head.csv", "sample_input.csv")
        )
        _route_defer = _defer_required or (
            _local_is_partial and bool(_cloud_uri_check and _conn_id_check)
        )
        if _route_defer:
            # Cloud path is guaranteed: the no-cloud case early-returned above.
            # NOTE: task_schedule is deliberately NOT pre-set here — a pre-set
            # schedule makes route_planner_output fire schedule_task for ANY
            # turn (chitchat included) before the planner ever reads intent.
            # coding_subgraph_node injects the schedule only when the planner
            # actually produced an analysis (analysis_defer_required=True).
            _effective_fidelity = FIDELITY_ENTIRE
            execution_mode = "k8s-ray"
            state_in["execution_mode"] = "k8s-ray"
            state_in["ray_execution_profile"] = "batch_heavy"
            state_in["data_source_location"] = cloud_uri or _cloud_uri_check
            state_in["analysis_defer_required"] = True
            logger.info(
                "[byte_routing] defer: estimated_full_bytes=%s >= cap=%s -> scheduled job",
                _estimated_full_bytes, _analysis_max_inmemory_bytes(),
            )
        else:
            # Below cap with a genuinely full local file: run on the WHOLE
            # frame locally (entire-dataset semantics;
            # _ensure_local_inputs(need_full=True) below re-materializes the
            # full file if the local copy went missing).
            _effective_fidelity = FIDELITY_ENTIRE
            execution_mode = "local"
            state_in["execution_mode"] = "local"
            state_in["task_schedule"] = None
            state_in["ray_execution_profile"] = "batch_heavy"
            state_in["analysis_defer_required"] = False
            state_in["data_source_location"] = _byte_local_candidate
        state_in["analysis_fidelity"] = FIDELITY_ENTIRE

    elif _effective_fidelity == FIDELITY_QUICK:
        execution_mode = "local"
        state_in["execution_mode"] = "local"
        state_in["task_schedule"] = None
        state_in["ray_execution_profile"] = "batch_heavy"
        state_in["data_source_location"] = (
            (active_local_source and _posix(active_local_source))
            or (sample_local_input and _posix(sample_local_input))
            or local_path
        )
    

    elif _effective_fidelity == FIDELITY_PORTFOLIO:
        if _is_folder_dataset and _folder_read_path and _conn_id_check:
            # Read the whole partitioned table via Ray, not a single-CSV sample.
            execution_mode = "k8s-ray"
            state_in["execution_mode"] = "k8s-ray"
            state_in["ray_execution_profile"] = "batch_heavy"
            state_in["data_source_location_cloud"] = _folder_read_path
            state_in["data_source_location"] = _folder_read_path
            state_in["task_schedule"] = None
        else:
            sample_cloud_uri = (
                await asyncio.to_thread(_upload_sample_to_cloud_if_needed, primary_sess)
                if _is_large_dataset else None
            )
            if _is_large_dataset and sample_cloud_uri and _conn_id_check:
                execution_mode = "k8s-ray"
                state_in["execution_mode"] = "k8s-ray"
                state_in["ray_execution_profile"] = "interactive_sample_analysis"
                state_in["data_source_location_cloud"] = sample_cloud_uri
                state_in["data_source_location"] = sample_cloud_uri
                state_in["task_schedule"] = None
                state_in["ray_workers"] = 1
                state_in["ray_timeout_s"] = min(int(ray_timeout_s or 1800), 600)
            else:
                # Sub-1 GB: run locally on the FULL file (work_local_input).
                execution_mode = "local"
                state_in["execution_mode"] = "local"
                state_in["task_schedule"] = None
                state_in["ray_execution_profile"] = "batch_heavy"
                state_in["data_source_location"] = (
                    (active_local_source and _posix(active_local_source))
                    or (work_local_input and _posix(work_local_input))
                    or (sample_local_input and _posix(sample_local_input))
                    or local_path
                )
   
    elif _effective_fidelity == FIDELITY_ENTIRE:
        execution_mode = (
            "local"
            if active_local_source
            else (
                "k8s-ray"
                if ((_is_large_dataset or _is_folder_dataset) and _cloud_uri_check and _conn_id_check)
                else "local"
            )
        )
        state_in["execution_mode"] = execution_mode
        state_in["ray_execution_profile"] = "batch_heavy"
        state_in["data_source_location"] = (
            cloud_uri if execution_mode == "k8s-ray" else local_path
        )
        if execution_mode == "k8s-ray":
            state_in["task_schedule"] = {
                "task_type": "execute",
                "schedule_type": "relative",
                "second": 0,
                "max_runs": 1,
            }
    if execution_mode == "k8s-ray":
        # Routing above settled on Ray after all: there is no fallback to label.
        state_in["ray_fallback_reason"] = None
    logger.info(
        "[fidelity] mode=%s resolved execution_mode=%s cloud_uri=%s task_schedule=%s",
        f"{_current_fidelity}->{_effective_fidelity}" if _current_fidelity != _effective_fidelity else _current_fidelity,
        execution_mode,
        bool(_cloud_uri_check),
        state_in.get("task_schedule"),
    )

    # Re-materialize the local input if TMP_ROOT was wiped (host reboot / tmp
    # cleaner) or a different process handled ingest. Faithful for QUICK/PORTFOLIO
    # (rebuild from the resolved sample rows); ENTIRE-local needs the full object back.
    if execution_mode != "k8s-ray":
        await _ensure_local_inputs(
            primary_sess,
            need_full=(
                _effective_fidelity == FIDELITY_ENTIRE
                or (_effective_fidelity == FIDELITY_PORTFOLIO and not _is_large_dataset)
            ),
            rows_override=(_resolved_rows or _fallback_preview or None),
        )
        # If the re-download changed work_local_input, point this run at it.
        if (
            _effective_fidelity in (FIDELITY_PORTFOLIO, FIDELITY_ENTIRE)
            and not _is_large_dataset
            and not active_local_source
            and primary_sess.get("work_local_input")
        ):
            state_in["data_source_location"] = _posix(primary_sess["work_local_input"])
            state_in["full_data_location"] = _posix(primary_sess["work_local_input"])
    if execution_mode != "k8s-ray":
        await _ensure_local_inputs(
            primary_sess,
            need_full=(_effective_fidelity == FIDELITY_ENTIRE),
            rows_override=(_resolved_rows or _fallback_preview or None),
        )



    _tmark("pre_graph_done")
    # # ----------------------------------------------------
    # # 12) Run workflow graph
    # # ----------------------------------------------------
    bound_sid = group_sid          # thread is bound to the group session (step 2)
    _turn_holder: Dict[str, Any] = {}

    # The job-note outbox (app/core/job_notes.py) must not deliver into this
    # thread while this turn runs; open it here, close it once, from whichever
    # path ends the turn (normal return, in-turn error, deferred failure).
    job_notes.turn_started(thread_id)
    _turn_holder["notes_open"] = True

    def _close_notes_turn() -> None:
        if _turn_holder.pop("notes_open", False):
            job_notes.turn_finished(thread_id)

    async def _run_graph_and_finalize() -> ChatResponse:
        _t0 = time.monotonic()
        deferred_id = _new_id()
        _turn_holder["deferred_id"] = deferred_id
        _turn_holder["datasets"] = datasets_dropdown
        _turn_holder["active_dataset_ids"] = [d for d, _, _ in dataset_sessions]
        _turn_holder["analysis_fidelity"] = _current_fidelity
        _turn_holder["selected_sample_name"] = primary_sess.get("selected_sample_name")

        # Shadow-mode observability: measure what the byte router *would* have
        # decided against what actually happened. Costs one rss read per turn
        # and is off unless AVALOKA_ROUTING_SHADOW is set.
        _shadow_on = _routing_shadow_enabled()
        _t_graph_start = time.monotonic()
        _rss_before = 0
        if _shadow_on:
            try:
                import psutil
                _rss_before = psutil.Process().memory_info().rss
            except Exception:
                pass

        def _graph_call():
            return GRAPH.invoke(
                state_in,
                config={"configurable": {"thread_id": f"ui:{thread_id}"}},
            )

        # Long producers (training, transfer, Ray execution, batch prediction)
        # wait inside the graph for the whole job. They get their own bounded
        # pool so N of them never starve the default pool every chat turn uses;
        # the job ledger shows them queued until a worker starts them.
        _long = long_jobs.classify_long_turn(body.content, state_in, _TRAINING_TURN_RE)
        if _long is not None:
            _pool = long_jobs.get_pool()
            _queued = _pool.would_queue()
            _turn_holder["long_kind"] = _long.kind
            _turn_holder["long_queued"] = _queued
            _job_id = await asyncio.to_thread(
                job_ledger.start, user_id, _long.kind, job_id=deferred_id, thread_id=thread_id,
                session_id=primary_sid, status="queued" if _queued else "running", origin="api",
                label=_turn_job_label(body.content or "", primary_sess.get("alias") or primary_sess.get("filename") or ""),
            )
            if _job_id:
                _turn_holder["ledger_job_id"] = _job_id
                state_in["ledger_job_id"] = _job_id
                state_in["ledger_job_kind"] = _long.kind
            _loop = asyncio.get_running_loop()

            def _on_start() -> None:          # worker thread: the job left the queue
                _turn_holder["long_queued"] = False
                job_ledger.running(_job_id)
                if _turn_holder.get("deferred_stashed"):
                    _mut = _pending_turn_running(deferred_id, _turn_holder.get("pending_kind") or "analysis", queued=False)
                    _remember_pending_turn(thread_id, user_id, _mut)
                    try:
                        asyncio.run_coroutine_threadsafe(update_session(bound_sid, _mut), _loop)
                    except Exception:  # noqa: BLE001
                        logger.debug("[long-jobs] could not update the pending turn", exc_info=True)

            logger.info("[long-jobs] turn %s classified %s (%s): %s; pool running=%d queued=%d",
                        deferred_id[:8], _long.kind, _long.reason, "queued" if _queued else "running",
                        _pool.running, _pool.queued)
            final = await _pool.submit(_graph_call, on_start=_on_start)
        else:
            final = await asyncio.get_event_loop().run_in_executor(None, _graph_call)
        _tmark("graph_done")

        if _shadow_on:
            try:
                import resource
                _ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # ru_maxrss is KB on Linux, bytes on macOS — normalize to bytes
                if sys.platform != "darwin":
                    _ru_maxrss *= 1024
            except Exception:
                _ru_maxrss = 0
            _rss_after = 0
            try:
                import psutil
                _rss_after = psutil.Process().memory_info().rss
            except Exception:
                pass
            _shadow_stats = _maybe_json_load(primary_sess.get("sample_statistics")) or {}
            _log_routing_shadow(
                thread_id=thread_id,
                dataset_id=primary_dsid,
                estimated_full_bytes=_estimated_full_bytes,
                bytes_per_row=_bytes_per_row_val,
                inmem_estimate_source=_inmem_source,
                total_rows=(_shadow_stats.get("data_shape") or {}).get("rows") if isinstance(_shadow_stats, dict) else None,
                file_size_bytes=_file_size_bytes,
                analysis_max_inmemory_bytes=_analysis_max_inmemory_bytes(),
                wall_clock_s=round(time.monotonic() - _t_graph_start, 3),
                rss_before_bytes=_rss_before,
                rss_after_bytes=_rss_after,
                peak_rss_bytes=_ru_maxrss,
                engine_used=final.get("execution_mode") or state_in.get("execution_mode"),
                effective_fidelity=_effective_fidelity,
                byte_routing_enabled=_byte_routing_enabled(),
            )

        # ----------------------------------------------------
        # 13) Planner graph metadata
        # ----------------------------------------------------
        _prev_artifacts = THREAD_META[thread_id].setdefault("artifacts", {})
        planner_graph_path = final.get("planner_graph_path") or _prev_artifacts.get("planner_graph_path")
        planner_graph_status = final.get("planner_graph_status") or _prev_artifacts.get("planner_graph_status")
        planner_graph_display_url = f"/threads/{thread_id}/planner-graph" if planner_graph_path else None

        THREAD_META[thread_id]["artifacts"]["planner_graph_path"] = planner_graph_path
        THREAD_META[thread_id]["artifacts"]["planner_graph_status"] = planner_graph_status

        # ----------------------------------------------------
        # 14) Visualization: primary dataset viz, plus map for all
        # ----------------------------------------------------
        primary_viz_config = final.get("visualization_config") or visualization_configs.get(primary_dsid) or {}
        primary_viz_status = final.get("visualization_status") or visualization_statuses.get(primary_dsid)

        visualization_configs[primary_dsid] = primary_viz_config
        if primary_viz_status:
            visualization_statuses[primary_dsid] = primary_viz_status

        THREAD_META[thread_id]["artifacts"]["visualization_config"] = primary_viz_config
        THREAD_META[thread_id]["artifacts"]["visualization_status"] = primary_viz_status or ""

        # ----------------------------------------------------
        # 15) Persist schema & training state on PRIMARY dataset session only
        # ----------------------------------------------------
        changed = False
        for k in ("schema", "ddl_schema", "uploaded_csv_preview", "uploaded_csv_columns"):
            if k in state_in and k in primary_sess:
                # Keep the preview capped in Redis; the full rows stay in
                # _profile_meta / Supabase / sample_input.csv.
                _new_val = _jsonify(
                    _cap_rows(state_in[k]) if k == "uploaded_csv_preview" else state_in[k]
                )
                if _new_val != primary_sess.get(k):
                    primary_sess[k] = _new_val
                    changed = True

        training_state_keys = (
            "training_plan",
            "training_result",
            "training_task",
            "ready_to_train",
            "training_completed",
            "training_metrics",
            "model_artifacts",
            "mlflow_run_id",
            "enable_training",
            "training_plan_reply_action",
        )
        if final.get("clear_training_state"):
            for k in training_state_keys:
                if k in primary_sess:
                    primary_sess.pop(k, None)
                    changed = True
            primary_sess["enable_training"] = False
            primary_sess["training_plan_reply_action"] = "none"
            changed = True

        for k in training_state_keys:
            if k in final and final.get(k) is not None:
                primary_sess[k] = _jsonify(final[k])
                changed = True

        if (
            _training_finished_successfully(final)
            and _truthy_session_value(primary_sess.get("data_source_was_modified"))
        ):
            if _restore_original_dataset_session(primary_sess):
                logger.info(
                    "[send_message] Restored original uploaded dataset after training completed for session %s",
                    primary_sid,
                )
                changed = True

        for k in ("pending_clarification",):
            if k in final:
                if final.get(k) is None:
                    primary_sess.pop(k, None)
                else:
                    primary_sess[k] = _jsonify(final[k])
                changed = True

        # The agent clears connection_event once it has onboarded the user.
        # Persist that, or every later turn re-reads the event from Redis and
        # onboards again.
        _consumed_event = primary_sess.get("connection_event")
        changed = consume_connection_event(primary_sess, final) or changed
        _drop_consumed_event = (
            _consumed_event is not None and "connection_event" not in primary_sess
        )

        task_info = final.get("task_info") or {}
        task_id = task_info.get("task_id")
        # True when this turn was a byte-gated deferral (scheduled, not executed).
        _byte_defer_active = bool(_byte_mode and state_in.get("analysis_defer_required"))
        existing_tasks = primary_sess.get("tasks")
        if isinstance(existing_tasks, str):
            try:
                existing_tasks = json.loads(existing_tasks)
            except Exception:
                existing_tasks = []
        if not isinstance(existing_tasks, list):
            existing_tasks = []
        primary_sess["tasks"] = existing_tasks
        if task_id is not None and task_id not in existing_tasks:
            primary_sess["tasks"].append(task_id)
            if _current_fidelity == FIDELITY_ENTIRE or _byte_defer_active:
                primary_sess["analysis_task_id"] = task_id
                # Timestamp bounds the duplicate-job guard (see pre-graph block)
                primary_sess["analysis_task_scheduled_at"] = time.time()
            changed = True

        if changed:
            def _merge_primary(cur: Dict[str, Any]) -> None:
                # Deletions too: a clarification left in the stored session
                # read every later question's first word as a table name, and
                # cleared training state came back on the next turn.
                _merge_session_with_drops(
                    cur, primary_sess, _primary_baseline,
                    drop=training_state_keys if final.get("clear_training_state") else (),
                )
                # update() never deletes, so a consumed event would survive the
                # merge. Drop it only if it is still the one this turn
                # consumed, so an event written meanwhile is kept.
                if _drop_consumed_event and cur.get("connection_event") == _consumed_event:
                    cur.pop("connection_event", None)

            await update_session(primary_sid, _merge_primary)

        # ----------------------------------------------------
        # 16) Update artifacts: coding agent
        # ----------------------------------------------------
        THREAD_META[thread_id]["artifacts"]["ready_to_code"] = bool(final.get("ready_to_code", False))
        THREAD_META[thread_id]["artifacts"]["coder_definition"] = final.get("coder_definition") or {}

        # ----------------------------------------------------
        # 17) Find AIMessage & update history (with training fallback)
        # ----------------------------------------------------
        final_all = final.get("messages", [])
        ai_msg = next((m for m in reversed(final_all) if isinstance(m, AIMessage)), None)
        # The claim verifier's verdict is enforced here, where the reply is
        # assembled: a claim it found contradicted or overreaching is withheld
        # (and the reply says so), in the response and in the thread history.
        # Each sentence keeps the kind the node that wrote it recorded; the
        # verifier's notes are plain; every commentary sentence is marked in
        # the text the response and the history carry (reply_segments).
        if ai_msg is not None and isinstance(ai_msg.content, str):
            _written = reply_segments.of_message(ai_msg)
            ai_msg = reply_segments.as_message(presentable_segments(
                reply_segments.text_of(_written), final.get("verification_report"), _written))

        if not ai_msg and final.get("ready_to_train") and not final.get("training_completed"):
            training_task = final.get("training_task")
            model_registry = final.get("model_registry")
            training_plan = None

            if training_task and isinstance(training_task, dict):
                training_plan = training_task.get("training_plan")
            if not training_plan and model_registry and isinstance(model_registry, dict):
                training_plan = model_registry.get("training_plan")

            if training_plan:
                goal = training_plan.get("goal", "Train a machine learning model")
                if isinstance(goal, str) and len(goal) > 200:
                    goal = goal[:200] + "..."
                model_suggestions = training_plan.get("model_suggestions", [])
                if isinstance(model_suggestions, list) and model_suggestions:
                    model_types = ", ".join(
                        [
                            m.get("model_type", str(m)) if isinstance(m, dict) else str(m)
                            for m in model_suggestions[:3]
                        ]
                    )
                else:
                    model_types = "Multiple models"

                fallback_message = f"""Model Training Agent: Training plan created successfully!

**Training Plan Summary:**
- Goal: {goal}
- Strategy: {training_plan.get('training_strategy', 'single_model')}
- Model Types: {model_types}
- Estimated Duration: {training_plan.get('estimated_duration', '15-30 minutes')}

**Next Steps:**
- Say "Start training" to begin model training

Ready to proceed with model training!"""

                ai_msg = AIMessage(content=fallback_message)
                lc_msgs.append(ai_msg)
                lc_msgs2 = _limit_messages(_compact_messages(lc_msgs), MAX_CONTEXT_TURNS)
                THREAD_META[thread_id]["lc_msgs"] = lc_msgs2
                final_all = final_all + [ai_msg]

        if ai_msg:
            lc_msgs.append(ai_msg)
            THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_msgs), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)

        ready_to_summarize = bool(final.get("ready_to_summarize", False))
        ready_to_code = bool(final.get("ready_to_code", False))

        # ----------------------------------------------------
        # 18) Output handling: get_output_from_state + CSV + markdown fallbacks
        # ----------------------------------------------------
        if not final.get("output_location") and not final.get("structured_action_plan"):
            final["output_location"] = state_in.get("output_location")

        try:
            output_file_data, output_json = get_output_from_state(final, tmp_root=TMP_ROOT)
        except TypeError:
            output_file_data, output_json = get_output_from_state(final)

        if output_json is None:
            candidate_output = output_file_data or final.get("output_file_data") or None
            if isinstance(candidate_output, dict) and candidate_output.get("content"):
                def _datauri_csv_to_records(data_uri: str) -> Optional[List[Dict[str, Any]]]:
                    if not data_uri or not data_uri.startswith("data:text/csv;base64,"):
                        return None
                    try:
                        b64 = data_uri.split("base64,", 1)[1]
                        csv_text = base64.b64decode(b64).decode()
                        sio = io.StringIO(csv_text)
                        reader = csv.DictReader(sio)
                        return [row for row in reader]
                    except Exception:
                        return None

                output_file_data = candidate_output
                output_json = _datauri_csv_to_records(output_file_data["content"])

        output_from_markdown = False

        if output_json is None and ai_msg and isinstance(ai_msg.content, str):
            blocks = re.findall(r"(?:^\|.*\|\s*\n?)+", reply_segments.unmarked(ai_msg), flags=re.MULTILINE)
            if blocks:
                lines = [ln.strip() for ln in blocks[0].strip().splitlines()]
                if len(lines) >= 3:
                    headers = [h.strip() for h in lines[0].strip("|").split("|")]
                    data_lines = [ln for ln in lines[2:] if ln.startswith("|")]
                    drop_first = len(headers) > 0 and headers[0] in ("", "#", "index")
                    if drop_first:
                        headers = headers[1:]
                    rows: List[Dict[str, Any]] = []
                    for ln in data_lines:
                        cells = [c.strip() for c in ln.strip("|").split("|")]
                        if drop_first and cells:
                            cells = cells[1:]
                        if len(cells) == len(headers):
                            rows.append(dict(zip(headers, cells)))
                    output_json = rows or None
                    output_from_markdown = bool(output_json)

        # ----------------------------------------------------
        # 18b) k8s-ray fallback: read artifact.json from GCS
        # ----------------------------------------------------
        _is_rayjob_response = (
            ai_msg
            and isinstance(ai_msg.content, str)
            and re.search(r"rayjob-[a-f0-9]+-\d+", ai_msg.content)
            and "success" in ai_msg.content.lower()
        )
        if output_json is None and (execution_mode == "k8s-ray" or _is_rayjob_response):
            try:
                rayjob_id = None
                if ai_msg and isinstance(ai_msg.content, str):
                    _m = re.search(r"(rayjob-[a-f0-9]+-\d+)", ai_msg.content)
                    if _m:
                        rayjob_id = _m.group(1)
                if not rayjob_id:
                    rayjob_id = (task_info or {}).get("rayjob_id") or (task_info or {}).get("ray_job_id")

                if rayjob_id:
                    _cloud_base = cloud_uri or primary_sess.get("data_source_location") or ""
                    _bucket_m = re.match(r"gs://([^/]+)", _cloud_base)
                    _gcs_bucket = _bucket_m.group(1) if _bucket_m else os.getenv("AVALOKA_ARTIFACT_GCS_BUCKET", "").strip()
                    if not _gcs_bucket:
                        raise HTTPException(
                            status.HTTP_500_INTERNAL_SERVER_ERROR,
                            "Could not determine the GCS bucket for the run artifact: the session "
                            "has no gs:// cloud URI and AVALOKA_ARTIFACT_GCS_BUCKET is not set.",
                        )
                    _artifact_path = f"gs://{_gcs_bucket}/test-input-data/_avaloka_runs/{rayjob_id}/artifact.json"
                    logger.info("[send_message] k8s-ray: fetching artifact gs://%s", _artifact_path)

                    try:
                        import gcsfs as _gcsfs  # type: ignore[import-not-found]
                    except Exception as e:
                        raise HTTPException(
                            status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Missing dependency for GCS artifact fetch (gcsfs): {e}",
                        )
                    _conn_id = primary_sess.get("connection_id")
                    _conn_creds = {}
                    if _conn_id:
                        try:
                            _conn_creds = await get_cloud_connection(_conn_id)
                        except Exception:
                            pass

                    _sa_json = _conn_creds.get("secret_key") or _conn_creds.get("service_account_json") or ""
                    if _sa_json:
                        _sa_json = _sa_json.replace("\\n", "\n").replace("\\r", "").replace("\\t", "\t")
                        _sa_json = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', _sa_json)
                        try:
                            _token = json.loads(_sa_json)
                        except json.JSONDecodeError:
                            _token = json.loads(_sa_json, strict=False)
                        _fs = await asyncio.to_thread(_gcsfs.GCSFileSystem, token=_token)
                    else:
                        _fs = await asyncio.to_thread(_gcsfs.GCSFileSystem)

                    def _read_artifact_json(fs, path):
                        with fs.open(path, "r", encoding="utf-8") as _f:
                            return json.load(_f)

                    _artifact = await asyncio.to_thread(_read_artifact_json, _fs, _artifact_path)
                    _out_rows = _artifact.get("output_rows") or []
                    _out_cols = _artifact.get("output_columns") or []

                    if _out_rows and isinstance(_out_rows[0], dict):
                        output_json = _out_rows
                    elif _out_rows and _out_cols:
                        output_json = [dict(zip(_out_cols, _r)) for _r in _out_rows]
                    elif _out_rows:
                        output_json = _out_rows

                    if output_json:
                        import io as _bio
                        _buf = _bio.StringIO()
                        _writer = csv.DictWriter(_buf, fieldnames=list(output_json[0].keys()))
                        _writer.writeheader()
                        _writer.writerows(output_json)
                        _csv_bytes = _buf.getvalue().encode("utf-8")
                        _b64 = base64.b64encode(_csv_bytes).decode("utf-8")
                        output_file_data = {
                            "filename": f"{rayjob_id}_output.csv",
                            "content": f"data:text/csv;base64,{_b64}",
                            "size": len(_csv_bytes),
                        }
                        logger.info(
                            "[send_message] k8s-ray: loaded %d rows from gs://%s",
                            len(output_json), _artifact_path,
                        )
            except Exception as _ray_err:
                logger.warning(
                    "[send_message] k8s-ray artifact fetch failed (output_json stays None): %s",
                    _ray_err, exc_info=True,
                )

        execution_status = str(((final.get("execution_result") or {}).get("status")) or "").lower()
        produced_fresh_output = (
            execution_status in ("success", "succeeded", "dryrun", "partial")
            or _is_rayjob_response
            or output_from_markdown
        )
        if output_json is not None and not produced_fresh_output:
            logger.info(
                "[send_message] Clearing stale output payload for non-execution turn execution_status=%s",
                execution_status or "<none>",
            )
            output_json = None
            output_file_data = None

        if (
            task_id
            and ai_msg
            and isinstance(ai_msg.content, str)
            and "scheduled" in ai_msg.content.lower()
        ):
            logger.info(
                "[send_message] Clearing stale output payload for scheduled task response task_id=%s",
                task_id,
            )
            output_json = None
            output_file_data = None

        # Scalar answers (row count, mean, max, "which category"...) are answered
        # in the chat message. Don't render them as a 1x1 Data table, and don't
        # let a chart config built from a 1-row output replace the dataset's charts.
        _scalar_answer = (
            _scalar_result_from_state(final, output_json)
            if produced_fresh_output and not final.get("activate_output_as_dataset")
            else None
        )
        if _scalar_answer is not None:
            logger.info(
                "[send_message] Scalar answer %s — suppressing table/chart payload",
                _scalar_answer,
            )
            output_json = None
            output_file_data = None
            primary_viz_config = (
                _maybe_json_load(primary_sess.get("visualization_config"))
                or primary_sess.get("visualization_config")
                or {}
            )
            primary_viz_status = primary_sess.get("visualization_status")
            visualization_configs[primary_dsid] = primary_viz_config
            if primary_viz_status:
                visualization_statuses[primary_dsid] = primary_viz_status
            THREAD_META[thread_id]["artifacts"]["visualization_config"] = primary_viz_config
            THREAD_META[thread_id]["artifacts"]["visualization_status"] = primary_viz_status or ""

        # ── Byte-gated deferral: the response is a deterministic job card, never
        #    LLM prose. The summarizer was bypassed in the graph (D-01), and the
        #    payload must carry NO analytical numbers — clear outputs
        #    unconditionally and replace the assistant message wholesale.
        # A "Full dataset" question on a large file is handed to a background
        # Ray job the same way: answer with the job card, not the summarizer's
        # JSON or the bare "task is scheduled" line.
        _entire_job_scheduled = bool(
            task_id
            and not _byte_defer_active
            and _effective_fidelity == FIDELITY_ENTIRE
            and state_in.get("execution_mode") == "k8s-ray"
        )
        if _byte_defer_active or _entire_job_scheduled:
            if task_id:
                _job_card = _build_deferred_job_card(
                    task_id=str(task_id),
                    dataset_name=str(primary_sess.get("alias") or primary_sess.get("filename") or primary_dsid),
                    # The user's own words, as the in-flight card (above) shows
                    # them: final["plan"] is the planner model's prose, and the
                    # card is plain.
                    interpreted_question=str(body.content),
                    eta_hint=_estimated_runtime_hint(_file_size_bytes),
                )
            else:
                _sched_err = final.get("execution_error") or "the scheduler did not return a job id"
                _job_card = (
                    "⚠️ The full-dataset job could not be scheduled "
                    f"({_sched_err}). No partial or sampled results are shown for "
                    "datasets above the in-memory analysis budget — please retry."
                )
            ai_msg = _said(_job_card)
            output_json = None
            output_file_data = None
            # Thread history was already persisted (section 17) with the graph's
            # raw scheduler message — rewrite it so refresh/history shows the same
            # deterministic job card the user saw.
            try:
                _lc = THREAD_META[thread_id].get("lc_msgs", [])
                if _lc and isinstance(_lc[-1], AIMessage):
                    _lc[-1] = _said(_job_card)
                else:
                    _lc.append(_said(_job_card))
                THREAD_META[thread_id]["lc_msgs"] = _lc
                await persist_thread_history(thread_id)
            except Exception as exc:
                logger.warning("[byte_routing] failed to rewrite thread history with job card: %s", exc)

        if produced_fresh_output and _scalar_answer is None and not final.get("structured_action_plan"):
            latest_output_path = final.get("output_location") or state_in.get("output_location")
            if _record_latest_tabular_output(primary_sess, latest_output_path, output_json):
                latest_updates = {
                    key: primary_sess.get(key)
                    for key in (*LATEST_OUTPUT_SESSION_KEYS, "latest_output_preview")
                    if key in primary_sess
                }
                await update_session(primary_sid, lambda cur: cur.update(latest_updates))
                logger.info(
                    "[send_message] Recorded latest tabular output for future turns: %s",
                    primary_sess.get("latest_output_location"),
                )

        if final.get("activate_output_as_dataset") and produced_fresh_output:
            output_path = final.get("output_location") or state_in.get("output_location")
            output_path_obj = Path(output_path) if output_path else None
            if output_path_obj and output_path_obj.exists() and output_path_obj.is_file():
                active_path = _posix(output_path_obj)
                rows_for_preview = output_json
                columns_for_schema: List[str] = []
                output_df_head = None
                output_has_rows = True
                if isinstance(rows_for_preview, list):
                    output_has_rows = bool(rows_for_preview)
                if not output_has_rows or rows_for_preview is None:
                    try:
                        output_df_head = pd.read_csv(output_path_obj, nrows=50)
                        output_has_rows = not output_df_head.empty
                    except Exception:
                        output_has_rows = False
                if not output_has_rows:
                    logger.info(
                        "[send_message] Skipping transformed-dataset activation because output is empty: %s",
                        active_path,
                    )
                else:
                    if output_df_head is None:
                        try:
                            output_df_head = pd.read_csv(output_path_obj, nrows=50)
                        except Exception:
                            output_df_head = None
                if isinstance(rows_for_preview, list) and rows_for_preview:
                    first_row = rows_for_preview[0]
                    if isinstance(first_row, dict):
                        columns_for_schema = list(first_row.keys())
                if not columns_for_schema:
                    if output_df_head is not None:
                        columns_for_schema = list(output_df_head.columns)
                        rows_for_preview = output_df_head.to_dict(orient="records")
                    else:
                        rows_for_preview = rows_for_preview if isinstance(rows_for_preview, list) else []

                _prior_columns = _maybe_json_load(primary_sess.get("uploaded_csv_columns")) \
                    or primary_sess.get("uploaded_csv_columns") or []
                _prior_col_set = {str(c) for c in _prior_columns} if isinstance(_prior_columns, list) else set()
                _new_col_set = {str(c) for c in columns_for_schema} if columns_for_schema else set()
                _preserves_columns = not _prior_col_set or (
                    bool(_new_col_set) and _prior_col_set.issubset(_new_col_set)
                )
                if output_has_rows and not _preserves_columns:
                    logger.info(
                        "[send_message] Not activating transformed dataset %s: output drops "
                        "%d of %d original columns (e.g. %s); keeping original source active.",
                        active_path,
                        len(_prior_col_set - _new_col_set),
                        len(_prior_col_set),
                        sorted(_prior_col_set - _new_col_set)[:5],
                    )
                if output_has_rows and _preserves_columns:
                    _snapshot_original_dataset_session(primary_sess)
                    primary_sess["active_data_source_location"] = active_path
                    primary_sess["active_data_source_location_local"] = active_path
                    primary_sess.pop("active_data_source_location_cloud", None)
                    primary_sess["work_local_input"] = active_path
                    primary_sess["data_source_was_modified"] = True
                    try:
                        primary_sess["file_size_bytes"] = output_path_obj.stat().st_size
                        primary_sess["file_size_mb"] = round(output_path_obj.stat().st_size / (1024 * 1024), 2)
                    except Exception:
                        pass
                    if columns_for_schema:
                        primary_sess["uploaded_csv_columns"] = _jsonify(columns_for_schema)
                        existing_schema = _maybe_json_load(primary_sess.get("schema")) or {}
                        if isinstance(existing_schema, dict):
                            primary_sess["schema"] = _jsonify({
                                column: existing_schema.get(column, "String")
                                for column in columns_for_schema
                            })
                    if isinstance(rows_for_preview, list):
                        primary_sess["uploaded_csv_preview"] = _jsonify(rows_for_preview[:DEFAULT_SAMPLE_MAX_ROWS])

                    for k in training_state_keys:
                        primary_sess.pop(k, None)
                    primary_sess["enable_training"] = False
                    primary_sess["training_plan_reply_action"] = "none"

                    def _merge_clearing_training(cur):
                        # Also drops active_data_source_location_cloud, popped
                        # above; it used to survive and outrank the new local
                        # source. enable_training / training_plan_reply_action
                        # are set in primary_sess, so they are kept, not popped.
                        _merge_session_with_drops(cur, primary_sess, _primary_baseline,
                                                  drop=training_state_keys)

                    await update_session(primary_sid, _merge_clearing_training)
                    logger.info(
                        "[send_message] Activated transformed dataset for future turns: %s",
                        active_path,
                    )

        # ----------------------------------------------------
        # 19) Training fields for response
        # ----------------------------------------------------
        training_scheduled = final.get("training_scheduled", False)
        inference_scheduled = final.get("inference_scheduled", False)
        configure_inference_service_scheduled = final.get("configure_inference_service_scheduled", False)
        stop_inference_service_scheduled = final.get("stop_inference_service_scheduled", False)
        training_plan = _sanitize_training_plan_for_response(final.get("training_plan"))
        training_result = final.get("training_result")
        training_task = final.get("training_task")
        training_metrics = final.get("training_metrics")
        model_artifacts = final.get("model_artifacts")
        mlflow_run_id = final.get("mlflow_run_id")
        if not mlflow_run_id and isinstance(training_result, dict):
            mlflow_run_id = training_result.get("mlflow_run_id")
        training_completed = final.get("training_completed", False)
        ready_to_train = final.get("ready_to_train", False)
        training_status = "completed" if training_completed else "pending" if ready_to_train else None

        _training_errored = (
            isinstance(training_result, dict)
            and (
                training_result.get("status") == "error"
                or training_result.get("error")
                or training_result.get("execution_error")
            )
            and not training_result.get("mlflow_run_id")
        )
        if _training_errored:
            _err_text = (
                training_result.get("error")
                or training_result.get("execution_error")
                or "Training failed."
            )
            if not ai_msg:
                # The raw error is for the log; the user gets the scheduled
                # path's wording (failure_diagnostics).
                logger.warning("[chat] model training failed: %s", _err_text)
                _failure = training_result.get("failure")
                ai_msg = _said(format_training_failure(
                    _failure if isinstance(_failure, dict) else build_training_failure(_err_text)
                ))
                lc_msgs.append(ai_msg)
                THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_msgs), MAX_CONTEXT_TURNS)
                await persist_thread_history(thread_id)
            training_status = "failed"
            training_result = None
            training_completed = False
            mlflow_run_id = None

        # ----------------------------------------------------
        # 20) Build ChatResponse (multi-dataset aware)
        # ----------------------------------------------------
        _sample_n = len(_resolved_rows) if isinstance(_resolved_rows, list) else 0
        _cloud_origin_session = str(
            primary_sess.get("data_source_location") or ""
        ).startswith(("gs://", "gcs://", "s3://", "az://", "abfs://"))
        _entire_on_head = (
            _cloud_origin_session
            and _effective_fidelity == FIDELITY_ENTIRE
            and str(primary_sess.get("work_local_input") or "").endswith("streamed_head.csv")
        )
        _total_rows = _session_total_rows(primary_sess)
        _ran_on_partial_sample = produced_fresh_output and _analysis_ran_on_partial_sample(
            effective_fidelity=_effective_fidelity,
            sample_n=_sample_n,
            total_rows=_total_rows,
            is_large_dataset=_is_large_dataset,
            entire_on_head=_entire_on_head,
        )
        # A database session knows exactly what its rows are: the whole table,
        # or a random sample of N of M rows. That record decides, not the file
        # heuristics above (which never fired for a DB sample: no row total).
        _db_sample = DbSample.from_session(
            _maybe_json_load(primary_sess.get("db_sample")) or primary_sess.get("db_sample"))
        if _db_sample is not None:
            _ran_on_partial_sample = produced_fresh_output and _db_sample.is_partial
        _assistant_text = ai_msg.content if ai_msg else None
        _assistant_segments = reply_segments.of_message(ai_msg) if ai_msg else None
        _structured = bool(final.get("structured_action_plan"))
        _multi_status = final.get("multi_action_status") if _structured else None
        _multi_actions = (final.get("multi_action_actions") or []) if _structured else []
        _multi_missing = (final.get("multi_action_missing") or []) if _structured else []
        _output_tables = (final.get("output_tables") or []) if _structured else []
        _multi_notes = (final.get("multi_action_notes") or []) if _structured else []
        if _structured and _output_tables:
            _missing_ids = {item.get("id") for item in _multi_missing}
            _completed_labels = [
                f"{item.get('operation')} of {item.get('column') or 'records'}"
                + (f" by {', '.join(item.get('group_by') or [])}" if item.get("group_by") else "")
                for item in _multi_actions if item.get("id") not in _missing_ids
            ]
            _completed_text = "Completed: " + "; ".join(_completed_labels) + ". " if _completed_labels else ""
            if _multi_missing:
                _missing_labels = "; ".join(
                    f"{item.get('label', 'metric')} ({item.get('reason', 'could not calculate')})"
                    for item in _multi_missing
                )
                _assistant_text = "Incomplete analysis. " + _completed_text + (
                    f"I could not complete {_missing_labels} after three calculation approaches. "
                    "Please check the source values and column types; data cleansing or clarification may be needed."
                )
            else:
                # Code checked that every planned metric was calculated; whether the
                # plan covers the request was the model reviewer's call, not claimed.
                _assistant_text = _completed_text + (
                    f"All {len(_multi_actions)} metrics in the plan were calculated. "
                    "The results are in the tables below.")
            if _multi_notes:
                _assistant_text += " Data preparation: " + " ".join(_multi_notes)
        elif _structured:
            _assistant_text = str((final.get("execution_result") or {}).get("message") or "Incomplete analysis.")
        # A Ray-to-local fallback is labelled by the local node; a structured
        # reply rebuilt above would drop that label, so it is restored here.
        _ray_fallback_note = ((final.get("execution_result") or {}).get("ray_fallback") or {}).get("note")
        if _ray_fallback_note and isinstance(_assistant_text, str) and _ray_fallback_note.strip() not in _assistant_text:
            _assistant_text = _assistant_text.rstrip() + _ray_fallback_note
        if _structured:
            # Written here from the action records: plain.
            _assistant_segments = reply_segments.plain(_assistant_text)
        if _ran_on_partial_sample and isinstance(_assistant_text, str) and _assistant_text.strip():
            _assistant_segments = reply_segments.join(
                reply_segments.rstrip(_assistant_segments or reply_segments.plain(_assistant_text)),
                reply_segments.plain(_sample_fidelity_note(
                    _effective_fidelity, _resolved_rows, primary_sess.get("selected_sample_name"),
                    total_rows=_total_rows, db_sample=_db_sample,
                )))
            _assistant_text = reply_segments.render(_assistant_segments)

        resp = ChatResponse(
            messages=[{"role": "user", "content": body.content}]
            + ([_assistant_entry(_assistant_text, _assistant_segments)] if _assistant_text else []),
            reasoning=final.get("reasoning_trace"),
            answered_by=final.get("answered_by"),
            plan_upgrade_required=final.get("plan_upgrade_required"),
            planner_definition=final.get("planner_definition", {}) or {},
            ready_to_summarize=ready_to_summarize,
            ready_to_code=ready_to_code,
            coder_definition=final.get("coder_definition", {}) or {},
            output_file_data=output_file_data,
            output_json=output_json,
            output_tables=_output_tables or None,
            multi_action_status=_multi_status,
            multi_action_actions=_multi_actions or None,
            multi_action_missing=_multi_missing or None,
            multi_action_notes=_multi_notes or None,
            planner_graph_path=planner_graph_path,
            planner_graph_status=planner_graph_status,
            planner_graph_display_url=planner_graph_display_url,
            task_info=task_info,
            training_scheduled=training_scheduled,
            inference_scheduled=inference_scheduled,
            configure_inference_service_scheduled=configure_inference_service_scheduled,
            stop_inference_service_scheduled=stop_inference_service_scheduled,
            training_plan=training_plan,
            training_result=training_result,
            training_task=training_task,
            training_metrics=training_metrics,
            model_artifacts=model_artifacts,
            mlflow_run_id=mlflow_run_id,
            training_completed=training_completed,
            ready_to_train=ready_to_train,
            training_status=training_status,
            visualization_config=primary_viz_config,
            visualization_status=primary_viz_status,
            datasets=datasets_dropdown,
            active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
            visualization_configs=visualization_configs,
            visualization_statuses=visualization_statuses,
            analysis_fidelity=_current_fidelity,
            selected_sample_name=primary_sess.get("selected_sample_name"),
            execution_context=_build_execution_context(
                _effective_fidelity,
                primary_sess.get("selected_sample_name"),
                rows_analyzed=(
                    (_sample_n if _ran_on_partial_sample else _total_rows)
                    if produced_fresh_output else None
                ),
                total_rows=_total_rows,
                swarm=final.get("avaloka_swarm"),
                swarm_message=final.get("avaloka_swarm_message"),
            ),
            analysis_task_id=(task_info or {}).get("task_id"),
            integrity_report=final.get("integrity_report"),
            # analysis_fidelity=_current_fidelity,
            # selected_sample_name=primary_sess.get("selected_sample_name"),
            # execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            # analysis_task_id=(task_info or {}).get("task_id"),
            # integrity_report=final.get("integrity_report"),
            integrity_safe_to_train=final.get("integrity_safe_to_train"),
            evaluation_report=final.get("evaluation_report"),
            evaluation_beats_baseline=final.get("evaluation_beats_baseline"),
            agent_errors=final.get("agent_errors"),
            verification_safe_to_present=final.get("verification_safe_to_present"),
            verification_report=final.get("verification_report"),
        )

        # Avaloka talks the user through the steps this turn took, grounded in
        # the session's statistics (app/api/chat_voice.py). Additive.
        resp.assistant_prompts = _voice_after_turn(
            final, question=body.content, execution_context=resp.execution_context, sess=primary_sess,
            profile_meta=_profile_meta.get(primary_dsid),
            # The turn's datasets, primary first: the basis names each one
            # the code read (a workbook's sheets combined), not only the first.
            datasets=[{"dataset_id": _dsid, "alias": ds_sess.get("alias"), "filename": ds_sess.get("filename"),
                       "sheet_name": ds_sess.get("sheet_name"),
                       "source_workbook": ds_sess.get("source_workbook"),
                       "rows": _session_total_rows(ds_sess)}
                      for _dsid, _sid, ds_sess in dataset_sessions],
        )
        # And the team's story, for any mission that ran on this thread's
        # dataset since the last turn, whatever started it (chat_swarm).
        if chat_swarm.pending_missions(primary_sess):
            _stories, _stories_changed = await _swarm_stories(
                primary_sess, user_id, await _caller_edition(request),
                primary_sess.get("alias") or primary_sess.get("filename") or None,
            )
            if _stories:
                resp.assistant_prompts = (resp.assistant_prompts or []) + _stories
            if _stories_changed:
                await update_session(primary_sid, lambda cur: cur.update(
                    {chat_swarm.SESSION_KEY: primary_sess.get(chat_swarm.SESSION_KEY)}))

        # ----------------------------------------------------
        # 20a) Agent trace: persist this turn (app/trace)
        # ----------------------------------------------------
        from app.trace.service import persist_from_state as _persist_trace

        _trace_turn = await _persist_trace(
            thread_id, user_id, final,
            execution_context=resp.execution_context, question=body.content,
            dataset_name=primary_sess.get("alias") or primary_sess.get("filename"),
        )
        if _trace_turn:
            resp.trace_turn_id = _trace_turn
            resp.trace_url = f"/threads/{thread_id}/trace?turn={_trace_turn}"

        # ----------------------------------------------------
        # 20b) Asset persistence (non-blocking background task)
        # ----------------------------------------------------
        _generated_code = (
            final.get("generated_code")
            or (final.get("coder_definition") or {}).get("code")
        )
        _exec_result = final.get("execution_result") or {}
        _exec_succeeded = str(_exec_result.get("status", "")).lower() == "success"

        _should_persist = (
            bool(_generated_code)
            or bool(output_json)
            or bool(final.get("ready_to_code"))
            or _exec_succeeded
        )

        if _should_persist and _is_new_checkpoint_worthy(primary_sess, output_json, _exec_succeeded):
            _storage_uri = (
                primary_sess.get("data_source_location")
                or state_in.get("data_source_location_cloud")
            )
            _new_fingerprint = hashlib.sha256(
                json.dumps(output_json or [], sort_keys=True, default=str).encode()
            ).hexdigest()
            primary_sess["last_output_fingerprint"] = _new_fingerprint
            # A merge, not save_session(primary_sess): that whole-dict write put
            # back the pending-turn record this turn read at its start.
            await update_session(
                primary_sid,
                lambda cur: _merge_session_with_drops(cur, primary_sess, _primary_baseline),
            )

            asyncio.create_task(
                _persist_assets_background(
                    user_id              = user_id,
                    session_id           = primary_sid,
                    dataset_id           = primary_dsid,
                    generated_code       = _generated_code,
                    planner_definition   = final.get("planner_definition") or {"plan": final.get("plan", "")},
                    coder_definition     = final.get("coder_definition"),
                    execution_result     = _exec_result,
                    exec_succeeded       = _exec_succeeded,
                    output_file_data     = output_file_data,
                    output_json          = output_json,
                    output_location      = state_in.get("output_location"),
                    visualization_config = final.get("visualization_config") or primary_viz_config,
                    connection_id        = primary_sess.get("connection_id"),
                    storage_uri          = _storage_uri,
                    dataset_label        = primary_sess.get("alias") or primary_sess.get("filename") or "",
                    analysis_label       = redact_typed_secrets(body.content or "")[0][:60],
                )
            )

        # If this turn overran the sync window, the HTTP response already went
        # back as a "running" handle — stash the finished result for polling.
        _deferred = (time.monotonic() - _t0) >= TURN_SYNC_DEADLINE_S
        # Run truth comes from code-written fields only; any run_truth the
        # graph state carries is ignored, and this replaces it on the response.
        # The replan loop's record rides on it (app/agents/replan.py).
        resp.run_truth = run_truth.classify_turn_with_recovery(
            final,
            deferred=_deferred,
            scope_defaults={
                "rows_processed": (
                    (_sample_n if _ran_on_partial_sample else _total_rows)
                    if produced_fresh_output else None
                ),
                "rows_total": _total_rows,
                "rows_total_kind": (
                    "unknown" if _total_rows is None
                    else "estimate" if _is_large_dataset else "exact"
                ),
                "sample": bool(_ran_on_partial_sample) if produced_fresh_output else None,
            },
            entire_on_head=bool(produced_fresh_output and _entire_on_head),
        )
        # This turn's own ledger row (a long or deferred turn) ends first, so
        # its note is among the ones delivered right after the reply. The
        # ledger rings the bell; a fast, unclassified turn has no row.
        _status, _fail_reason, _error_line = _turn_outcome(resp)
        _turn_holder["turn_outcome"] = (_status, _fail_reason, _error_line)
        await _settle_turn_job(_turn_holder, final if isinstance(final, dict) else {}, _deferred,
                               _status, _fail_reason, _error_line)
        # The turn's reply is in THREAD_META: close the turn for the outbox and
        # deliver what finished meanwhile, after the reply. A deferred turn
        # carries the batch in its stashed result, so the poll returns it too.
        _close_notes_turn()
        _notes_batch = await _deliver_job_notes(thread_id, user_id)
        if _notes_batch:
            resp.job_notes = [_notes_batch]
        if _deferred:
            _done_mutator = _pending_turn_done(deferred_id, resp.model_dump())
            # In-process copy first: it cannot fail, and the poll falls back to it.
            _remember_pending_turn(thread_id, user_id, _done_mutator)
            try:
                await update_session(bound_sid, _done_mutator)
            except Exception:
                logger.warning("[deferred] failed to stash finished turn %s", deferred_id, exc_info=True)

        return resp

    # ---- race the turn against the soft deadline ----
    _turn_task = asyncio.create_task(_run_graph_and_finalize())
    _turn_task.add_done_callback(lambda _t: _close_notes_turn())   # every ending closes the turn
    _done, _ = await asyncio.wait({_turn_task}, timeout=TURN_SYNC_DEADLINE_S)

    # if _turn_task in _done:
    #     # Fast turn (all analysis, and any training that fit): identical to before.
    #     resp = _turn_task.result()          # re-raises HTTPException/errors in-turn
    if _turn_task in _done:
        try:
            resp = _turn_task.result()      # re-raises HTTPException/errors in-turn
        except HTTPException:
            raise  # already structured (401/400/504…) — keep its status
        except Exception as exc:
            # An unexpected failure inside the graph (e.g. a retired provider model).
            # Log the full traceback for engineers; return the reason to the user as a
            # normal assistant message instead of a bare 500 "Message failed".
            logger.exception("[send_message] turn failed (thread=%s): %s", thread_id, exc)
            _friendly = _friendly_turn_error(exc)
            try:
                _lc = THREAD_META[thread_id].get("lc_msgs", [])
                _lc.append(_said(_friendly))
                THREAD_META[thread_id]["lc_msgs"] = _limit_messages(
                    _compact_messages(_lc), MAX_CONTEXT_TURNS
                )
                await persist_thread_history(thread_id)
            except Exception:
                logger.debug("[send_message] could not persist error message", exc_info=True)
            _close_notes_turn()
            _err_notes_batch = await _deliver_job_notes(thread_id, user_id)
            resp = ChatResponse(
                messages=[{"role": "user", "content": body.content},
                          _assistant_entry(_friendly)],
                datasets=datasets_dropdown,
                active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
                visualization_configs=visualization_configs,
                visualization_statuses=visualization_statuses,
                analysis_fidelity=_current_fidelity,
                selected_sample_name=primary_sess.get("selected_sample_name"),
                execution_context=_build_execution_context(
                    _current_fidelity, primary_sess.get("selected_sample_name")
                ),
                run_truth=run_truth.failed("exception", kind="interactive", detail=_friendly),
                job_notes=[_err_notes_batch] if _err_notes_batch else None,
            )
    else:
        # Slow turn (model training): return a handle before the proxy aborts.
        _deferred_turn_tasks.add(_turn_task)
        _deferred_id = _turn_holder.get("deferred_id") or _new_id()

        def _cleanup(t: asyncio.Task, _did=_deferred_id):
            _deferred_turn_tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                logger.error("[deferred] turn %s failed: %s", _did, t.exception())
                _err_text = _friendly_turn_error(t.exception())
                _err_mutator = _pending_turn_error(_did, _err_text)
                _remember_pending_turn(thread_id, user_id, _err_mutator)
                asyncio.create_task(update_session(bound_sid, _err_mutator))
                _turn_holder["turn_outcome"] = ("failed", "failed", _err_text)
                asyncio.create_task(_finish_turn_job(_turn_holder.get("ledger_job_id"), "failed",
                                                     fail_reason="failed", error_line=_err_text))
        _turn_task.add_done_callback(_cleanup)

        # Classify the deferred work ONCE, before anything is stored or returned,
        # so every surface agrees (pending-turn record, poll endpoint, response,
        # frontend). A slow ANALYSIS must never be labelled training — that was
        # the misleading "Training model…" popup on normal analyses.
        _is_training_turn = bool(
            _TRAINING_TURN_RE.search(body.content or "")
            or state_in.get("ready_to_train")
        )
        _pending_kind = "training" if _is_training_turn else "analysis"
        _turn_holder["pending_kind"] = _pending_kind

        _running_mutator = _pending_turn_running(_deferred_id, _pending_kind,
                                                 queued=bool(_turn_holder.get("long_queued")))
        _remember_pending_turn(thread_id, user_id, _running_mutator)
        await update_session(bound_sid, _running_mutator)
        _turn_holder["deferred_stashed"] = True

        # Producer #1: a deferred turn is a background job. A long-classified
        # turn already has its row (job id = deferred id); any other one gets
        # it here. If the task ended while the row was being written, finish
        # it now; otherwise the task finishes it when it ends.
        if not _turn_holder.get("ledger_job_id"):
            _turn_holder["ledger_job_id"] = await asyncio.to_thread(
                job_ledger.start, user_id, "training" if _is_training_turn else "analysis",
                job_id=_deferred_id, thread_id=thread_id, session_id=primary_sid, status="running", origin="api",
                label=_turn_job_label(body.content or "", primary_sess.get("alias") or primary_sess.get("filename") or ""),
            ) or None
            if _turn_task.done() and _turn_holder.get("turn_outcome"):
                _st, _fr, _el = _turn_holder["turn_outcome"]
                await _finish_turn_job(_turn_holder.get("ledger_job_id"), _st, fail_reason=_fr, error_line=_el)

        _pending_text = (
            "Model training is running — this can take a few minutes. "
            "The results will appear here automatically when it's done, "
            "and you'll get a notification."
            if _is_training_turn else
            "This analysis is taking longer than usual. "
            "The result will appear here when it finishes, "
            "and you'll get a notification."
        )

        resp = ChatResponse(
            messages=[
                {"role": "user", "content": body.content},
                _assistant_entry(_pending_text),
            ],
            # training_status is now a TRAINING-ONLY signal. The generic
            # "keep polling" signal is pending_status; pending_kind says what it is.
            training_status="running" if _is_training_turn else None,
            pending_status="running",
            pending_kind=_pending_kind,
            analysis_task_id=_deferred_id,
            datasets=_turn_holder.get("datasets") or datasets_dropdown,
            active_dataset_ids=_turn_holder.get("active_dataset_ids")
                or [dsid for dsid, _, _ in dataset_sessions],
            analysis_fidelity=_turn_holder.get("analysis_fidelity") or _current_fidelity,
            selected_sample_name=_turn_holder.get("selected_sample_name"),
            execution_context=_build_execution_context(
                _turn_holder.get("analysis_fidelity") or _current_fidelity,
                _turn_holder.get("selected_sample_name"),
            ),
        )

    _tmark("response_ready")
    # ----------------------------------------------------
    # 21) SSE streaming support
    # ----------------------------------------------------
    if body.stream:
        resp_dict = resp.model_dump()
        async def sse_iter():
            yield f"event: message\ndata: {json.dumps(resp_dict)}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(
            sse_iter(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return resp





@app.get("/threads/{thread_id}/pending-turn")
async def get_pending_turn(
    request: Request,
    thread_id: str,
    deferred_id: Optional[str] = Query(None),
):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    
    sid = await get_thread_session(thread_id)
    sess = await get_session(sid) if sid else None
    mem = _PENDING_TURN_MEM.get(thread_id)
    mem = mem if (isinstance(mem, dict) and mem.get("user_id") == user_id) else None
    mem_cur = _pending_turn_mem_session(mem)

    if sid and sess:
        if sess.get("user_id") != user_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread_id")
        stores = [sess, mem_cur]
    elif mem:
        # Redis unreachable: answer from this process's copy.
        stores = [mem_cur]
    else:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread_id")

    if deferred_id:
        # Answer for THIS turn wherever it is filed. A finished record beats a
        # "running" one: the session can hold a stale running record when its
        # done write failed, and the in-process copy then has the result.
        found = [rec for rec in (_find_pending_turn(c, deferred_id) for c in stores) if rec]
        pt = next((r for r in found if r.get("status") in ("done", "error")), None) or (found[0] if found else None)
        if pt is None:
            newest = next((p for p in (_pending_turn_slots(c)[0] for c in stores) if p), None)
            if newest is None:
                return {"status": "none"}
            # Not filed anywhere, and another turn is the newest: that turn
            # replaced it. Name it so the client can show its result instead.
            return {"status": "superseded", "superseded_by": newest.get("id")}
    else:
        pt = _pending_turn_slots(stores[0])[0]
        mem_pt = _pending_turn_slots(mem_cur)[0]
        # The finished result may only be in memory, when the Redis write
        # failed; prefer it over a stale "running" for the same turn.
        if mem_pt and (pt is None
                       or (pt.get("id") == mem_pt.get("id") and pt.get("status") == "running"
                           and mem_pt.get("status") in ("done", "error"))):
            pt = mem_pt
        if not isinstance(pt, dict):
            return {"status": "none"}

    status_val = pt.get("status")
    kind = pt.get("kind") or "analysis"
    if status_val == "done":
        return {"status": "done", "kind": kind, "result": pt.get("result") or {}}
    if status_val == "error":
        default_err = ("Training failed. Please try again." if kind == "training"
                                              else "The analysis failed. Please try again.")
        return {"status": "error", "kind": kind, "message": pt.get("message") or default_err}
    return {"status": "running", "kind": kind, "queued": bool(pt.get("queued"))}

# ── Notifications (design doc D4) ────────────────────────────────────
@app.get("/api/notifications")
async def get_notifications(
    request: Request,
    unread: bool = Query(False),
    limit: int = Query(20, ge=1, le=100),
):
    """The caller's notifications, newest first, plus the unread count for the bell."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    try:
        items, unread_count = await notification_service.list_notifications(
            user_id, unread_only=unread, limit=limit,
        )
    except Exception as exc:
        logger.warning("[notify] list failed for %s: %s", user_id[:8], exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Notifications are unavailable")
    return {"items": items, "unread_count": unread_count}


@app.post("/api/notifications/read-all")
async def read_all_notifications(request: Request):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    try:
        updated = await notification_service.mark_all_read(user_id)
    except Exception as exc:
        logger.warning("[notify] read-all failed for %s: %s", user_id[:8], exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Notifications are unavailable")
    return {"updated": updated}


@app.post("/api/notifications/{notification_id}/read")
async def read_notification(request: Request, notification_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    try:
        ok = await notification_service.mark_read(user_id, notification_id)
    except Exception as exc:
        logger.warning("[notify] read failed for %s: %s", user_id[:8], exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Notifications are unavailable")
    if not ok:
        # Someone else's row looks the same as a missing one.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown notification")
    return {"ok": True}

async def _require_owned_thread(request: Request, thread_id: str) -> str:
    """401 without a user, 503 without Redis, 404 unless the thread is theirs."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Session storage (Redis) is not available.")
    sid = await get_thread_session(thread_id)
    sess = await get_session(sid) if sid else None
    if not sid or not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread_id")
    return user_id


def _checkpoint_values(thread_id: str) -> Optional[Dict[str, Any]]:
    """The thread's latest graph state from the in-process checkpointer, if any."""
    try:
        snap = GRAPH.get_state({"configurable": {"thread_id": f"ui:{thread_id}"}})
        return dict(getattr(snap, "values", None) or {}) or None
    except Exception:  # noqa: BLE001 - echo graph, no checkpointer, unknown thread
        return None


@app.get("/threads/{thread_id}/trace")
async def get_thread_trace(request: Request, thread_id: str, turn: Optional[str] = None):
    """Every agent step of this analysis: decisions, calculations, inference, handoffs.

    The caller's own analysis, so it carries their columns and values. Agent
    timings, tokens and outcomes are aggregated under ``agents``. Open in every
    edition: this is how a user sees their prompt acted upon, not the swarm
    console (app/api/swarm_gate.py).
    """
    from app.trace.service import pick_turn, swarm_limits_for, thread_trace

    await _require_owned_thread(request, thread_id)
    doc = await thread_trace(
        thread_id,
        checkpoint_state=_checkpoint_values(thread_id),
        swarm_limits=swarm_limits_for(await _caller_edition(request)),
    )
    if turn:
        chosen = pick_turn(doc["turns"], turn)
        if chosen is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown turn for this thread")
        doc["turns"] = [chosen]
    return doc


@app.get("/api/agent-metrics")
async def get_agent_metrics(request: Request, user_id: str = Depends(require_swarm_console)):
    """Per-agent calls, latency, tokens, cost, outcomes and retries across the
    caller's analyses. Read from their stored traces; nothing leaves the cluster."""
    from app.trace.service import user_agent_metrics

    return await user_agent_metrics(user_id)


# ---------------------------------------------------------------------------
# Swarm missions: one trace per clone, served live (AGENT_TRACE.md §6).
# Owner-only like the thread trace: 401 without a user, 403 outside Professional
# and Enterprise (require_swarm_console), 404 for a mission that is not the
# caller's, so another user cannot learn that it exists.
# ---------------------------------------------------------------------------
class SwarmMissionIn(BaseModel):
    source: Dict[str, str]
    goal: str
    kind: str = "analyze"
    workers: Optional[int] = None
    max_parallel: Optional[int] = None
    target: Optional[str] = None
    metric: Optional[str] = None


async def _thread_dataset_path(thread_id: str, user_id: str) -> Optional[str]:
    """The local file the chat path would analyse for this thread, if the
    thread is the caller's: the working copy of the upload, else its sample."""
    if not session_service.cache:
        return None
    sid = await get_thread_session(thread_id)
    sess = await get_session(sid) if sid else None
    if not sid or not sess or sess.get("user_id") != user_id:
        return None
    for key in ("work_local_input", "active_data_source_location_local", "sample_local_input"):
        candidate = sess.get(key)
        if candidate and Path(str(candidate)).is_file():
            return str(candidate)
    return None


def _swarm_404() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "Unknown mission_id")


@app.post("/api/swarm/missions")
async def post_swarm_mission(request: Request, body: SwarmMissionIn,
                             user_id: str = Depends(require_swarm_console)):
    """Start a coordinated mission on a bundled sample, a thread's dataset or
    (only with AVALOKA_SWARM_ALLOW_PATHS=1) a server path. Returns at once;
    poll ``/api/swarm/missions/{id}/records?after=`` for the live trace."""
    from app.trace import swarm_service

    edition = await _caller_edition(request)
    try:
        started = await swarm_service.start_mission(
            user_id, source=body.source, goal=body.goal, kind=body.kind, workers=body.workers,
            max_parallel=body.max_parallel, target=body.target, metric=body.metric,
            edition=edition,
            thread_resolver=lambda tid: _thread_dataset_path(tid, user_id),
        )
    except swarm_service.SwarmRequestError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    # A mission on a thread's dataset is told back in that thread's chat on
    # its next turn, whatever started it (app/api/chat_swarm.py).
    if isinstance(body.source, dict) and body.source.get("thread_id"):
        await _note_thread_mission(str(body.source["thread_id"]), started["mission_id"])
    return started


@app.get("/api/swarm/missions")
async def list_swarm_missions(request: Request,
                              user_id: str = Depends(require_swarm_console)):
    from app.trace import swarm_service

    return {"missions": await swarm_service.list_missions(user_id)}


@app.get("/api/swarm/missions/{mission_id}")
async def get_swarm_mission(request: Request, mission_id: str, records: int = 0,
                            user_id: str = Depends(require_swarm_console)):
    """The mission document: grant, clones with their steps, merge. Add
    ``?records=1`` for the raw append-only records too."""
    from app.trace import swarm_service

    doc = await swarm_service.get_mission(user_id, mission_id, records=bool(records))
    if doc is None:
        raise _swarm_404()
    return doc


@app.get("/api/swarm/missions/{mission_id}/records")
async def get_swarm_mission_records(request: Request, mission_id: str, after: int = 0,
                                    limit: int = 500,
                                    user_id: str = Depends(require_swarm_console)):
    """The live endpoint. Records with ``seq > after`` (seq starts at 1, so
    the first poll omits ``after``); ``cursor`` is the last seq returned and
    ``done`` says the mission finished and nothing is left past the cursor."""
    from app.trace import swarm_service

    page = await swarm_service.records_after(user_id, mission_id, after=after, limit=limit)
    if page is None:
        raise _swarm_404()
    return page


@app.get("/api/swarm/missions/{mission_id}/clones/{clone_id}")
async def get_swarm_clone(request: Request, mission_id: str, clone_id: str,
                          user_id: str = Depends(require_swarm_console)):
    from app.trace import swarm_service

    doc = await swarm_service.get_clone(user_id, mission_id, clone_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown mission_id or clone_id")
    return doc


@app.get("/api/swarm/missions/{mission_id}/episodes")
async def get_swarm_episodes(request: Request, mission_id: str,
                             user_id: str = Depends(require_swarm_console)):
    """The RL view: one episode per clone, reward left null for the learned layer."""
    from app.trace import swarm_service

    doc = await swarm_service.get_episodes(user_id, mission_id)
    if doc is None:
        raise _swarm_404()
    return doc


@app.get("/api/swarm/missions/{mission_id}/telemetry")
async def get_swarm_telemetry(request: Request, mission_id: str,
                              user_id: str = Depends(require_swarm_console)):
    """Exactly the metadata-only records telemetry would send for this
    mission, so a reviewer can read them. Nothing is sent by this route."""
    from app.trace import swarm_service

    doc = await swarm_service.get_telemetry(user_id, mission_id)
    if doc is None:
        raise _swarm_404()
    return doc


@app.get("/api/swarm/missions/{mission_id}/understanding")
async def get_swarm_understanding(request: Request, mission_id: str, after: int = 0,
                                  limit: int = 500, user_id: str = Depends(require_swarm_console)):
    """What each agent understood about the data and decided, and the fleet's
    summary (``agent.understanding`` / ``swarm.understanding`` records), paged
    like ``/records``: ``seq > after``, ``cursor``, ``done``. The full-fidelity
    layer: it may name the caller's own columns and files."""
    from app.trace import swarm_service

    page = await swarm_service.understanding_after(user_id, mission_id, after=after, limit=limit)
    if page is None:
        raise _swarm_404()
    return page


@app.get("/api/swarm/missions/{mission_id}/understanding/telemetry")
async def get_swarm_understanding_telemetry(request: Request, mission_id: str,
                                            user_id: str = Depends(require_swarm_console)):
    """The metadata-only projection of the understanding records: buckets,
    histograms and counts, exactly as telemetry would send them. Nothing is
    sent by this route."""
    from app.trace import swarm_service

    doc = await swarm_service.get_understanding_telemetry(user_id, mission_id)
    if doc is None:
        raise _swarm_404()
    return doc


@app.delete("/api/swarm/missions/{mission_id}")
async def delete_swarm_mission(request: Request, mission_id: str,
                               user_id: str = Depends(require_swarm_console)):
    from app.trace import swarm_service

    if not await swarm_service.delete_mission(user_id, mission_id):
        raise _swarm_404()
    return {"mission_id": mission_id, "deleted": True}


async def _planner_graph_json(request: Request, thread_id: str, turn: Optional[str]) -> Dict[str, Any]:
    from app.trace.service import graph_for, pick_turn, thread_trace

    await _require_owned_thread(request, thread_id)
    doc = await thread_trace(thread_id, checkpoint_state=_checkpoint_values(thread_id))
    chosen = pick_turn(doc["turns"], turn)
    if chosen is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No recorded turn for this thread")
    plan_steps = None
    defn = (_checkpoint_values(thread_id) or {}).get("planner_definition")
    if isinstance(defn, dict):
        plan_steps = defn.get("transformations") or defn.get("steps") or None
    graph = graph_for(chosen, plan_steps=plan_steps)
    graph["thread_id"] = thread_id
    return graph


@app.get("/threads/{thread_id}/planner-graph")
async def get_planner_graph_image(request: Request, thread_id: str,
                                  format: str = "png", turn: Optional[str] = None):
    """The planner graph. ``format=json`` returns the granular, typed graph of a
    recorded turn (agents, decisions, data operations, checks, evidence);
    the default stays the rendered PNG the shipped UI reads."""
    if format == "json":
        return await _planner_graph_json(request, thread_id, turn)
    # ---- 1) Auth: require a valid user ----
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Missing or invalid auth token",
        )

    # You need Redis to check thread -> session -> user ownership
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.",
        )

    # ---- 2) Authorize: ensure this thread belongs to this user ----
    sid = await get_thread_session(thread_id)
    sess = await get_session(sid) if sid else None

    # If we can't find a session for this thread, or it's not this user's
    # session, pretend the thread doesn't exist (avoid leaking info).
    if not sid or not sess or sess.get("user_id") != user_id:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Unknown thread_id",
        )

    # ---- 3) Look up planner graph metadata in THREAD_META ----
    tmeta = THREAD_META.get(thread_id)
    if not tmeta:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Unknown thread_id",
        )

    artifacts = tmeta.get("artifacts") or {}
    path = artifacts.get("planner_graph_path")
    if not path:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Planner graph not available for this thread",
        )

    p = Path(path).resolve()
    if not p.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Planner graph file missing on server",
        )

    # ---- 4) Extra path safety: must live inside PLANNER_GRAPH_OUTPUT_DIR ----
    base_dir = os.getenv("PLANNER_GRAPH_OUTPUT_DIR")
    if base_dir:
        base = Path(base_dir).resolve()
        # Only allow the file if it is under base_dir (or exactly in it)
        if base not in p.parents and p.parent != base:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Invalid graph path",
            )

    # ---- 5) Serve the PNG ----
    return FileResponse(
        str(p),
        media_type="image/png",
        filename=p.name,
    )


def _session_tasks(sess: Optional[Dict[str, Any]]) -> List[str]:
    """Return a session's scheduled task ids as a list.

    Upload/register sessions never create a 'tasks' key (only a scheduled run
    does), so reads must tolerate its absence instead of raising KeyError.
    """
    tasks = (sess or {}).get("tasks")
    if isinstance(tasks, str):
        try:
            tasks = json.loads(tasks)
        except Exception:
            tasks = []
    return tasks if isinstance(tasks, list) else []


def _discover_session_tasks(sid: str, user_id: str) -> List[str]:
    """Recover RedBeat tasks that belong to a session but missed session state.

    The schedule is durable in Redis even if the API response was interrupted
    before its id was appended to the session. Filtering by both session and
    user prevents another user's schedules from becoming visible.
    """
    discovered: List[str] = []
    try:
        redis = get_redis(celery_app)
        keys = redis.zrange(celery_app.redbeat_conf.schedule_key, 0, -1)
        for raw_key in keys:
            key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
            try:
                entry = AvalokaEntry.from_key(key, celery_app)
                args = entry.args or []
                state = args[0] if args and isinstance(args[0], dict) else {}
                if state.get("session_id") == sid and state.get("user_id") == user_id:
                    discovered.append(str(entry.name))
            except Exception:
                logger.debug("Could not inspect RedBeat entry %s", key, exc_info=True)
    except Exception:
        logger.debug("Could not discover scheduled tasks from RedBeat", exc_info=True)
    return discovered


@app.get("/tasks")
async def get_task_list(request: Request):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.", 
        )
    
    sid = _resolve_session_id(request, None)
    sess = await get_session(sid) if sid else None
    if sid and (not sess or sess.get("user_id") != user_id):
        sid, sess = None, None
    if not sid or not sess:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No session found. Pass X-Avaloka-Session header to prove that the task belongs to this user."
        )

    stored_task_ids = _session_tasks(sess)
    task_ids = list(dict.fromkeys([
        *stored_task_ids,
        *_discover_session_tasks(sid, user_id),
    ]))
    if task_ids != stored_task_ids:
        await update_session(sid, lambda cur: cur.update(tasks=task_ids))

    tasks = []
    for task_id in task_ids:
        redbeat_key = AvalokaEntry.generate_key(celery_app, task_id)

        try:
            entry = AvalokaEntry.from_key(redbeat_key, celery_app)
        except Exception as e:
            logger.warning(f"Failed to retrieve task_id {task_id}", exc_info=e)
            continue

        if isinstance(entry.schedule, crontab):
            schedule = {
                "month_of_year": getattr(entry.schedule, "_orig_month_of_year", None),
                "day_of_month":  getattr(entry.schedule, "_orig_day_of_month",  None),
                "day_of_week":   getattr(entry.schedule, "_orig_day_of_week",   None),
                "hour":          getattr(entry.schedule, "_orig_hour",          None),
                "minute":        getattr(entry.schedule, "_orig_minute",        None),
                "second":        getattr(entry.schedule, "_orig_second",        None),
            }
        else:
            schedule = {
                "second": entry.schedule.run_every.total_seconds()
            }

        task_metadata = _scheduled_task_metadata_from_entry(entry)
        durable_runs = await _load_scheduled_runs(task_id)
        latest_run = durable_runs[0] if durable_runs else None
        tasks.append({
            "id": entry.name,
            "schedule": schedule,
            "max_runs": entry.max_runs,
            "total_run_count": entry.total_run_count,
            "last_run_at": _reported_last_run_at(entry, entry.total_run_count),
            "task_type": task_metadata.get("task_type"),
            "task_label": task_metadata.get("task_label"),
            "latest_run": latest_run,
        })

    return tasks

@app.get("/tasks/{task_id}/info")
async def get_task_info(request: Request, task_id: str, max_timestamp: int = 0):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.", 
        )
    
    sid = _resolve_session_id(request, None)
    sess = await get_session(sid) if sid else None
    if sid and (not sess or sess.get("user_id") != user_id):
        sid, sess = None, None
    if not sid or not sess:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No session found. Pass X-Avaloka-Session header to prove that the task belongs to this user."
        )

    if task_id not in _session_tasks(sess):
        raise HTTPException(status.HTTP_404_NOT_FOUND)

    redbeat_key = AvalokaEntry.generate_key(celery_app, task_id)

    try:
        entry = AvalokaEntry.from_key(redbeat_key, celery_app)
    except Exception:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    
    all_runs = AvalokaScheduler.get_task_ids_with_timestamp(redbeat_key, False)
    tasks_data = list(
        filter(
            lambda task: max_timestamp == 0 or task[1].timestamp() < max_timestamp, 
            all_runs
        )
    )

    if isinstance(entry.schedule, crontab):
        schedule = {
            "month_of_year": getattr(entry.schedule, "_orig_month_of_year", "*"),
            "day_of_month":  getattr(entry.schedule, "_orig_day_of_month",  "*"),
            "day_of_week":   getattr(entry.schedule, "_orig_day_of_week",   "*"),
            "hour":          getattr(entry.schedule, "_orig_hour",          "*"),
            "minute":        getattr(entry.schedule, "_orig_minute",        "*"),
        }
    else:
        schedule = {
            "second": entry.schedule.run_every.total_seconds()
        }

    return {
        "schedule": schedule,
        "max_runs": entry.max_runs,
        "total_run_count": len(tasks_data),
        "last_run_at": _reported_last_run_at(entry, len(all_runs))
    }

@app.get("/tasks/{task_id}/result/{index}")
async def get_task_result(request: Request, task_id: str, index: int):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.", 
        )
    
    sid = _resolve_session_id(request, None)
    sess = await get_session(sid) if sid else None
    if sid and (not sess or sess.get("user_id") != user_id):
        sid, sess = None, None
    if not sid or not sess:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No session found. Pass X-Avaloka-Session header to prove that the task belongs to this user."
        )

    if task_id not in _session_tasks(sess):
        raise HTTPException(status.HTTP_404_NOT_FOUND)

    scheduled_task_id = task_id

    def _clear_scheduled_task(cur):
        # Re-check on the latest session under the lock so a concurrent asset
        # persist isn't clobbered by this single-field clear.
        if cur.get("analysis_task_id") == scheduled_task_id:
            cur["analysis_task_id"] = None

    redbeat_key = AvalokaEntry.generate_key(celery_app, task_id)
    task_ids = AvalokaScheduler.get_task_ids(redbeat_key, False)
    if not task_ids:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    if index > len(task_ids) - 1:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "Index out of range"
        )
    if index < 0:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "Index out of range"
        )
   
    task_id = task_ids[index]

    entry = AvalokaEntry.from_key(redbeat_key, celery_app)
    task_metadata = _scheduled_task_metadata_from_entry(entry)
    last_run_at = AvalokaScheduler.get_last_run_at(redbeat_key, generate_key=False)
    next_due_at = entry.schedule.remaining_estimate(last_run_at).total_seconds()

    task = AsyncResult(task_id, app=celery_app)

    if task.state == "REVOKED":
        if sess.get("analysis_task_id") == scheduled_task_id:
            await update_session(sid, _clear_scheduled_task)
        return {"status": "CANCELLED", "message": _SCHEDULED_CANCELLED_TEXT, "next_due_at": next_due_at, **task_metadata}
    elif task.failed() or (task.state == "SUCCESS" and _scheduled_run_failed(task.result, task_metadata)):
        if sess.get("analysis_task_id") == scheduled_task_id:
            await update_session(sid, _clear_scheduled_task)
        # The Celery traceback is the worker's log, never the user's message.
        _failed_msg = (_scheduled_failure_text(task, task_metadata) if task.failed()
                       else _scheduled_task_completion_message(task.result or {}, [], task_metadata))
        _details = await asyncio.to_thread(_failed_run_details, task, task_id, task_metadata)
        return {"status": "FAILED", "message": _failed_msg, "error": _failed_msg, "traceback": _failed_msg,
                "next_due_at": next_due_at, **task_metadata, **_details}
    elif task.state not in ("SUCCESS",):
        finished = await asyncio.to_thread(_run_finished_state, task_id)
        if finished:
            # Over, but its result is no longer stored: never "running".
            if sess.get("analysis_task_id") == scheduled_task_id:
                await update_session(sid, _clear_scheduled_task)
            return {
                "status": "SUCCESS" if finished == "SUCCESS" else "FAILED",
                "message": "The run finished; its stored result is no longer available.",
                "next_due_at": next_due_at,
                **task_metadata,
            }
        return {
            "status": task.state,
            "message": task_metadata["running_message"],
            "next_due_at": next_due_at,
            **task_metadata,
        }
    else:
        view = await _finished_run_view(task.result, task_id, task_metadata)
        final = view["final"]
        assistant_messages = view["assistant_messages"]
        output_file_data, output_json = view["output_file_data"], view["output_json"]
        if sess.get("analysis_task_id") == scheduled_task_id:
            await update_session(sid, _clear_scheduled_task)

        # ── WBS 0.6: persist scheduled job outputs async ──────────────
        _generated_code = (
            final.get("generated_code")
            or (final.get("coder_definition") or {}).get("code")
        )
        _exec_result   = final.get("execution_result") or {}
        _exec_succeeded = str(_exec_result.get("status", "")).lower() in (
            "success", "succeeded", "completed", "done"
        )
        if _generated_code or output_json:
            _storage_uri = sess.get("data_source_location")
            asyncio.create_task(
                _persist_assets_background(
                    user_id              = sess.get("user_id", ""),
                    session_id           = sid,
                    dataset_id           = sess.get("dataset_id", ""),
                    generated_code       = _generated_code,
                    planner_definition   = final.get("planner_definition") or {"plan": final.get("plan", "")},
                    coder_definition     = final.get("coder_definition"),
                    execution_result     = _exec_result,
                    exec_succeeded       = _exec_succeeded,
                    output_file_data     = output_file_data,
                    output_json          = output_json,
                    output_location      = final.get("output_location"),
                    visualization_config = final.get("visualization_config"),
                    connection_id        = sess.get("connection_id"),
                    storage_uri          = _storage_uri,
                )
            )

        completion_message = view["completion_message"]
        # Same builder as the synchronous response and the task SSE stream.
        execution_context = _build_execution_context_from_state(final, session=sess)
        # The outcome _record_run_finish writes for this run (item 8a): Celery
        # SUCCESS only means the task returned, not that it answered.
        _task_truth = view["run_truth"]
        return {
            "status": "SUCCESS",
            "run_truth": _task_truth,
            "message": completion_message,
            "completion_message": completion_message,
            "messages": assistant_messages,
            "assistant_message": assistant_messages[-1] if assistant_messages else None,
            **task_metadata,
            "result": {
                "output_file_data": output_file_data,
                "output_json": output_json,
                "messages": assistant_messages,
                "assistant_message": assistant_messages[-1] if assistant_messages else None,
                "message": completion_message,
                "completion_message": completion_message,
                "execution_context": execution_context,
                "run_truth": _task_truth,
                "output_contract": view["output_contract"],
            },
            "next_due_at": next_due_at
        }

_CELERY_STATE_TO_RUN_STATUS = {
    "SUCCESS": "success", "FAILURE": "failure", "REVOKED": "failure",
    "STARTED": "running", "RETRY": "running",
    "RECEIVED": "pending", "PENDING": "pending",
}
_SCHEDULED_RUNS_TABLE = os.getenv("SUPABASE_SCHEDULED_RUNS_TABLE", "scheduled_runs")


def _run_iso(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.datetime.fromtimestamp(value, tz=datetime.timezone.utc).isoformat()
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value.isoformat()
    return str(value)


def _public_run(run: Any) -> Any:
    """A run-history row as the user may see it: no traceback in its error,
    message or log lines. The raw text stays on the row and in the worker log."""
    if not isinstance(run, dict):
        return run
    run = dict(run)
    if looks_like_raw_error(run.get("error")):
        run["error"] = "The scheduled run failed."
    result = run.get("result")
    if isinstance(result, dict):
        result = dict(result)
        if "logs" in result:
            result["logs"] = scrub_run_logs(result.get("logs"))
        if looks_like_raw_error(result.get("message")):
            result["message"] = "The scheduled run failed."
        run["result"] = result
    return run


async def _load_scheduled_runs(schedule_id: str) -> Optional[List[Dict[str, Any]]]:
    """Durable run history. None -> table missing/unavailable (use RedBeat fallback);
    [] -> table exists but empty (valid, do NOT fall back)."""
    def _q():
        client = get_supabase_client()
        res = (client.table(_SCHEDULED_RUNS_TABLE)
               .select("run_id,run_number,status,started_at,finished_at,duration_ms,result,error")
               .eq("schedule_id", schedule_id)
               .order("started_at", desc=True)
               .execute())
        return getattr(res, "data", None)
    try:
        return await asyncio.to_thread(_q)
    except Exception:
        logger.debug("[runs] scheduled_runs unavailable; RedBeat fallback", exc_info=True)
        return None


@app.get("/tasks/{task_id}/runs")
async def get_task_runs(request: Request, task_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Session storage (Redis) is not available.")

    sid = _resolve_session_id(request, None)
    sess = await get_session(sid) if sid else None
    if sid and (not sess or sess.get("user_id") != user_id):
        sid, sess = None, None
    if not sid or not sess:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No session found. Pass X-Avaloka-Session header to prove that the task belongs to this user.",
        )
    if task_id not in _session_tasks(sess):
        raise HTTPException(status.HTTP_404_NOT_FOUND)

    redbeat_key = AvalokaEntry.generate_key(celery_app, task_id)
    try:
        entry = AvalokaEntry.from_key(redbeat_key, celery_app)
    except Exception:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    task_metadata = _scheduled_task_metadata_from_entry(entry)

    # Prefer the durable table (it outlives the Redis result, CELERY_RESULT_EXPIRES).
    # Rows written before the run log was scrubbed still hold tracebacks, so
    # every row is cleaned on the way out.
    durable = await _load_scheduled_runs(task_id)
    if durable is not None:
        return [_public_run(run) for run in durable]

    # Fallback: RedBeat run list + live Celery result backend.
    runs_with_ts = list(AvalokaScheduler.get_task_ids_with_timestamp(redbeat_key, False))
    runs_with_ts.sort(key=lambda p: p[1], reverse=True)   # newest first
    total = len(runs_with_ts)

    out: List[Dict[str, Any]] = []
    for offset, (exec_id, ts) in enumerate(runs_with_ts):
        task = AsyncResult(exec_id, app=celery_app)
        state = task.state
        run_status = _CELERY_STATE_TO_RUN_STATUS.get(state, "pending")
        started_at = _run_iso(ts)
        finished_at = _run_iso(getattr(task, "date_done", None))  # None after expiry
        duration_ms = None
        if started_at and finished_at:
            try:
                duration_ms = int(
                    (datetime.datetime.fromisoformat(finished_at)
                     - datetime.datetime.fromisoformat(started_at)).total_seconds() * 1000
                )
            except Exception:
                duration_ms = None

        result_payload = None
        error = None
        if state not in ("SUCCESS", "FAILURE", "REVOKED") and _run_finished_state(exec_id):
            # Over, but its result is no longer stored: not "pending".
            run_status = "success" if _run_finished_state(exec_id) == "SUCCESS" else "failure"
            error = "The stored result of this run is no longer available."
        if state == "SUCCESS":
            final = task.result or {}
            if isinstance(final, dict):
                view = await _finished_run_view(final, exec_id, task_metadata)
                result_payload = {
                    "output_json": view["output_json"],
                    "output_file_data": view["output_file_data"],
                    "message": view["completion_message"],
                    "logs": view["logs"],
                    "ray_job": view["ray_job"],
                }
                if task_metadata.get("task_type") == "training":
                    run_status, error = view["run_status"] or run_status, view["error"]
                    if run_status == "failure":
                        result_payload["logs"] = None
        elif state == "FAILURE":
            # The traceback is the worker's log; the row carries plain words.
            logger.info("[runs] run %s failed: %s", exec_id, task.traceback or task.result)
            error = _scheduled_failure_text(task, task_metadata)
        elif state == "REVOKED":
            error = "Run was cancelled."

        out.append({
            "run_id":      exec_id,
            "run_number":  total - offset,
            "status":      run_status,
            "started_at":  started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "result":      result_payload,
            "error":       error,
        })

    return [_public_run(run) for run in out]

@app.get("/tasks/{task_id}/status")
async def get_task_status(request: Request, task_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.", 
        )
    
    sid = _resolve_session_id(request, None)
    sess = await get_session(sid) if sid else None
    if sid and (not sess or sess.get("user_id") != user_id):
        sid, sess = None, None
    if not sid or not sess:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No session found. Pass X-Avaloka-Session header to prove that the task belongs to this user."
        )

    if task_id not in _session_tasks(sess):
        raise HTTPException(status.HTTP_404_NOT_FOUND)

    redbeat_key = AvalokaEntry.generate_key(celery_app, task_id)

    try:
        entry = AvalokaEntry.from_key(redbeat_key, celery_app)
    except Exception:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task ID couldn't be found")
    task_metadata = _scheduled_task_metadata_from_entry(entry)

    task_id = AvalokaScheduler.get_last_run_task_id(redbeat_key, False)

    def sse_iter(next_due_at: int):
        yield "event: message\ndata: " + json.dumps({
            "status": "SCHEDULED",
            "message": task_metadata["scheduled_message"],
            "next_due_at": next_due_at,
            **task_metadata,
        }) + "\n\n"
        yield "event: done\ndata: {}\n\n"

    if task_id is None:
        next_due_at = (
            entry.schedule.remaining_estimate(entry.last_run_at or datetime.datetime.now()).total_seconds() + 2
            if entry.max_runs > 0 and entry.total_run_count < entry.max_runs
            else -1
        )
        return StreamingResponse(
            sse_iter(next_due_at),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # if AsyncResult(task_id).state in ("FAILURE", "SUCCESS", "REVOKED"):
    #     last_run_at = AvalokaScheduler.get_last_run_at(redbeat_key, generate_key=False)
    if (AsyncResult(task_id, app=celery_app).state in ("FAILURE", "SUCCESS", "REVOKED")
            or _run_finished_state(task_id)):
        last_run_at = AvalokaScheduler.get_last_run_at(redbeat_key, generate_key=False)
        return StreamingResponse(
            sse_iter(entry.schedule.remaining_estimate(last_run_at).total_seconds()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        task_sse_iter(task_id, redbeat_key, session=sess),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.delete("/tasks/{task_id}")
async def cancel_task(request: Request, task_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.", 
        )
    
    sid = _resolve_session_id(request, None)
    sess = await get_session(sid) if sid else None
    if sid and (not sess or sess.get("user_id") != user_id):
        sid, sess = None, None
    if not sid or not sess:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No session found. Pass X-Avaloka-Session header to prove that the task belongs to this user."
        )
    tasks = _session_tasks(sess)
    if task_id not in tasks:
        raise HTTPException(status.HTTP_404_NOT_FOUND)

    def _remove_task(cur):
        cur_tasks = _session_tasks(cur)
        if task_id in cur_tasks:
            cur_tasks.remove(task_id)
        cur["tasks"] = cur_tasks

    await update_session(sid, _remove_task)

    try:
        entry = AvalokaEntry.from_key(
            AvalokaEntry.generate_key(celery_app, task_id),
            celery_app
        )
    except Exception:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    entry.delete()
    return {"status": "CANCELLED"}

@app.get("/threads/{thread_id}/messages")
async def fetch_history(request: Request, thread_id: str):
    # History is durable now (survives restarts), so this endpoint must be
    # authenticated like send_message/delete_thread — otherwise anyone with
    # a thread_id could read a user's whole conversation from Redis.
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    sid = await get_thread_session(thread_id)
    if sid:
        sess = await get_session(sid)
        if not sess:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread_id")
        if sess.get("user_id") != user_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Thread belongs to another user")
    # Restore from Redis first so history survives process restarts.
    await hydrate_thread_history(thread_id)
    if sid:
        # A read is a turn boundary: deliver finished-job notes before serving.
        await _deliver_job_notes(thread_id, user_id)
    msgs = await read_thread_msgs(thread_id)
    if not sid and msgs:
        # History with no session binding has no owner to authorize against;
        # serving it would hand one user's chat to any authenticated caller.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread_id")
    return {"messages": msgs}


@app.get("/threads/{thread_id}/job-notes")
async def get_thread_job_notes(request: Request, thread_id: str):
    """Deliver and return the thread's finished-job notes (app/core/job_notes.py).

    The UI calls this when a notification for this thread arrives on its
    realtime channel. If a turn is in flight the notes stay queued and the
    response says so; they arrive with that turn's reply instead.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    sid = await get_thread_session(thread_id)
    sess = await get_session(sid) if sid else None
    if not sid or not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread_id")
    if job_notes.turn_in_flight(thread_id):
        return {"delivered": None, "deferred": True,
                "pending": await asyncio.to_thread(job_notes.pending_count, thread_id)}
    batch = await _deliver_job_notes(thread_id, user_id)
    return {"delivered": batch, "deferred": False,
            "pending": await asyncio.to_thread(job_notes.pending_count, thread_id)}


@app.get(
    "/threads/{thread_id}/code",
    summary="Get only the code string from coder_definition",
    tags=["code"],
)
async def get_thread_code_only(request: Request, thread_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    sid = await get_thread_session(thread_id)
    sess = await get_session(sid) if sid else None
    if not sid or not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread_id")

    t = THREAD_META.get(thread_id)
    if not t:
        raise HTTPException(404, "Unknown thread_id")
    artifacts = t.get("artifacts") or {}
    cd = artifacts.get("coder_definition") or {}
    code = cd.get("code") if isinstance(cd, dict) else None
    if not code or not isinstance(code, str) or not code.strip():
        raise HTTPException(
            404,
            "No code found in coder_definition for this thread",
        )
    return Response(content=code, media_type="text/plain")


# -------------------------------------------------------------------
# Datasets: list / get / preview / delete
# -------------------------------------------------------------------
@app.get("/datasets", response_model=List[DatasetListItem], tags=["datasets"])
async def list_datasets(request: Request):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.",
        )

    cache = session_service.cache

    # ---- 1) Collect this user's session ids (one scan, no 0.6s wait_for) ----
    try:
        sid_set_key = _k_user_sessions(user_id)
        sids: List[str] = []
        if hasattr(cache, "sscan_iter"):
            async for sid in cache.sscan_iter(sid_set_key, count=200, cap=5000):  # type: ignore[attr-defined]
                sids.append(sid)
        else:
            sids = list(await cache.smembers(sid_set_key))[:5000]
    except Exception as e:
        # A failure to even read the index is a real error — surface it as 503
        # so the frontend shows "failed to load / retry", NOT an empty tab.
        logger.exception("datasets: failed to read user session index: %s", e)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Could not load datasets, please retry.",
        )

    # The catalog is durable where the session store is not: its file and
    # derived datasets join the list, Redis items first (app/catalog/compat.py).
    from app.catalog.compat import catalog_items as _catalog_items, merge_dataset_items as _merge_items

    if not sids:
        extra = await asyncio.to_thread(_catalog_items, user_id, ())
        return _merge_items([], extra, DatasetListItem)  # no sessions: the catalog alone

    # ---- 2) Read every session blob ONCE (no per-dataset re-MGET, no
    #         get_session() so the circuit breaker can't blank the list) ----
    try:
        blobs = await _mget_safe([_k_session(s) for s in sids])
    except Exception as e:
        logger.exception("datasets: failed to load session blobs: %s", e)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Could not load datasets, please retry.",
        )

    # ---- 3) Build one item per distinct dataset, newest first ----
    items: List[DatasetListItem] = []
    seen: set[str] = set()
    for sid, blob in zip(sids, blobs):
        if not blob:
            continue  # expired/missing individual session — skip, don't fail
        try:
            sess = json.loads(blob)
        except Exception:
            continue
        if sess.get("user_id") != user_id:
            continue
        dsid = sess.get("dataset_id")
        if not dsid or dsid in seen:
            continue
        seen.add(dsid)

        size = int(sess.get("file_size_bytes") or 0)
        created = sess.get("created_at") or _now_iso()
        cols = _schema_to_columns(sess.get("schema"))

        items.append(
            DatasetListItem(
                dataset_id=dsid,
                created_at=created,
                size_bytes=size,
                columns=cols,
                # Grouping key + labels for the Uploaded Datasets tab. A dataset
                # uploaded alone falls back to its own session id as its group.
                session_id=sid,
                group_session_id=sess.get("group_session_id") or sid,
                filename=sess.get("filename"),
                alias=sess.get("alias"),
                source_kind=sess.get("source_kind"),
            )
        )
    # Newest first. created_at is ISO-8601, so lexical sort == chronological.
    extra = await asyncio.to_thread(_catalog_items, user_id, seen)
    return _merge_items(items, extra, DatasetListItem)

@app.get(
    "/datasets/{dataset_id}",
    response_model=DatasetListItem,
    tags=["datasets"],
)
async def get_dataset(request: Request, dataset_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.",
        )

    user_ds = await _user_datasets(user_id)
    if dataset_id not in user_ds:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Dataset not found for this user",
        )

    sid_for_ds = await find_session_by_dataset_for_user(dataset_id, user_id)
    if not sid_for_ds:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No session bound to this dataset",
        )

    sess = await get_session(sid_for_ds, bypass_circuit=True)
    if not sess:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Session not found for this dataset",
        )

    storage_uri = sess.get("data_source_location")
    object_name = sess.get("object_name") or _key_from_uri(storage_uri)

    size, created = 0, ""
    try:
        store_for_read, key_for_read = _store_and_key_from_uri(
            storage_uri,
            object_name,
        )
        if store_for_read and key_for_read:
            size, created = store_for_read.stat(key_for_read)
    except Exception as e:
        logger.warning(
            "get_dataset: stat failed for uri=%s key=%s (%s)",
            storage_uri,
            object_name,
            e,
        )

    if not created:
        created = _now_iso()

    cols = _schema_to_columns(sess.get("schema"))

    return DatasetListItem(
        dataset_id=dataset_id,
        created_at=created,
        size_bytes=size,
        columns=cols,
    )


@app.get(
    "/datasets/{dataset_id}/preview",
    response_model=DatasetPreviewResponse,
    tags=["datasets"],
)
async def preview_dataset(
    request: Request,
    dataset_id: str,
    limit: int = Query(
        DEFAULT_SAMPLE_MAX_ROWS,
        ge=1,
        le=DEFAULT_SAMPLE_MAX_ROWS,
    ),
    session_id: Optional[str] = Query(None),
    thread_id: Optional[str] = Query(None),
):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not storage_service.blob_store:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Storage backend not initialized.",
        )

    header_sid = _resolve_session_id(request, None)
    effective_sid = session_id or header_sid

    sess_for_ds: Optional[Dict[str, Any]] = None

    if effective_sid:
        sess_for_ds = await get_session(effective_sid, bypass_circuit=True)
        if not sess_for_ds:
            sess_for_ds = None
            effective_sid = None
        else:
            if (
                sess_for_ds.get("user_id") != user_id
                or sess_for_ds.get("dataset_id") != dataset_id
            ):
                logger.warning(
                    "preview: ignoring stale session_id=%s "
                    "(session.user_id=%s, session.dataset_id=%s) "
                    "for request user_id=%s dataset_id=%s",
                    effective_sid,
                    sess_for_ds.get("user_id"),
                    sess_for_ds.get("dataset_id"),
                    user_id,
                    dataset_id,
                )
                sess_for_ds = None
                effective_sid = None
            else:
                session_id = effective_sid

    if not sess_for_ds:
        user_ds = await _user_datasets(user_id)
        if dataset_id not in user_ds:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                "Dataset not found for this user",
            )

        sid_for_ds = await find_session_by_dataset_for_user(
            dataset_id,
            user_id,
        )
        session_id = sid_for_ds
        if sid_for_ds:
            sess_for_ds = await get_session(sid_for_ds, bypass_circuit=True)

    if not sess_for_ds:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No session bound to this dataset",
        )

    if not thread_id:
        saved_tid = sess_for_ds.get("thread_id")
        if saved_tid:
            thread_id = saved_tid
            if session_id:
                await bind_thread_session(thread_id, session_id)
        elif session_id:
            thread_id = f"ui:{session_id}"
            await bind_thread_session(thread_id, session_id)

    local_input = Path(sess_for_ds.get("work_local_input") or "")
    if not local_input.exists():
        sample_input = Path(sess_for_ds.get("sample_local_input") or "")
        if sample_input.exists():
            # Preview reads the sample, but work_local_input is left alone:
            # persisting the sample there made every later "Full dataset"
            # run use it. _ensure_local_inputs re-downloads the real file.
            local_input = sample_input
            logger.info(
                "[preview] using sample_local_input for preview only, dataset %s -> %s",
                dataset_id,
                sample_input,
            )
        elif not sess_for_ds.get("uploaded_csv_preview"):
            work_dir = Path(
                sess_for_ds.get("work_dir") or (TMP_ROOT / _new_id())
            )
            work_dir.mkdir(parents=True, exist_ok=True)
            ext = (sess_for_ds.get("input_data_type") or "csv").lower()
            local_input = work_dir / f"input_{dataset_id}.{ext}"

            store_for_read, key_for_read = _store_and_key_from_uri(
                sess_for_ds.get("data_source_location"),
                sess_for_ds.get("object_name"),
            )
            await asyncio.to_thread(
                store_for_read.get_file,
                key_for_read
                or (
                    sess_for_ds.get("object_name")
                    or _key_from_uri(sess_for_ds.get("data_source_location"))
                ),
                local_input,
            )
            logger.info(
                "[preview] pulled %s (key=%s) -> %s",
                sess_for_ds.get("data_source_location"),
                key_for_read,
                local_input,
            )

            sess_for_ds["work_local_input"] = _posix(local_input)
            ok = await save_session(session_id, sess_for_ds)
            if not ok:
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "Failed to persist session while preparing preview",
                )
        else:
            logger.info(
                "[preview] skipping full source download for dataset %s because preview rows are already available",
                dataset_id,
            )

    schema_any: Union[List[str], Dict[str, str]] = []
    samples: List[Dict[str, Any]] = []
    ddl: str = ""

    if sess_for_ds.get("uploaded_csv_preview"):
        schema_any = sess_for_ds.get("schema") or []
        #samples = list(sess_for_ds.get("uploaded_csv_preview") or [])[:limit]4
        raw_preview = sess_for_ds.get("uploaded_csv_preview")
        preview = _maybe_json_load(raw_preview) or raw_preview or []
        samples = preview[:limit] if isinstance(preview, list) else []

        ddl = sess_for_ds.get("ddl_schema") or _ddl_from_schema(
            dataset_id,
            {c: "string" for c in _schema_to_columns(schema_any)},
        )
    else:
        ext = (sess_for_ds.get("input_data_type") or "csv").lower()
        try:
            agent_res = await asyncio.to_thread(
                sample_data_from_source,
                path=str(local_input),
                source_type=ext,
                stratify_by=None,
                sample_size=1.0,
            )
            if isinstance(agent_res, dict) and not agent_res.get("error"):
                schema_any = agent_res.get("schema") or []
                samples = agent_res.get("rows") or []
                ddl = agent_res.get("ddl_schema") or ""
            else:
                if ext == "csv":
                    samples_all, schema_dict, ddl = _fallback_sample_csv(
                        str(local_input)
                    )
                    schema_any = schema_dict
                    samples = samples_all[:limit]
                else:
                    raise HTTPException(
                        status.HTTP_500_INTERNAL_SERVER_ERROR,
                        f"Failed to sample .{ext} for preview.",
                    )
        except Exception:
            if ext == "csv":
                samples_all, schema_dict, ddl = _fallback_sample_csv(
                    str(local_input)
                )
                schema_any = schema_dict
                samples = samples_all[:limit]
            else:
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    f"Failed to sample .{ext} for preview.",
                )

    _preview_sample_status = (sess_for_ds or {}).get("sample_status")
    meta = _profile_meta.get(dataset_id) or {}
    session_portfolio = _maybe_json_load((sess_for_ds or {}).get("portfolio_samples")) or (sess_for_ds or {}).get("portfolio_samples")
    session_available_samples = _maybe_json_load((sess_for_ds or {}).get("available_samples")) or (sess_for_ds or {}).get("available_samples")
    session_profiling_result = _maybe_json_load((sess_for_ds or {}).get("profiling_result")) or (sess_for_ds or {}).get("profiling_result")
    session_full_profiling_result = _maybe_json_load((sess_for_ds or {}).get("full_profiling_result")) or (sess_for_ds or {}).get("full_profiling_result")
    session_sample_statistics = _maybe_json_load((sess_for_ds or {}).get("sample_statistics")) or (sess_for_ds or {}).get("sample_statistics")

    # When a background task ran, the session portfolio is the stale quick-sample
    # data; do NOT seed meta from it — Supabase has the authoritative full portfolio.
    _has_bg_task = bool((sess_for_ds or {}).get("background_task_id"))
    _session_portfolio_is_stale = _has_bg_task and _preview_sample_status == "full_sample"

    if session_portfolio and not meta.get("portfolio_samples") and not _session_portfolio_is_stale:
        meta["portfolio_samples"] = session_portfolio
    if session_available_samples and not meta.get("available_samples"):
        meta["available_samples"] = session_available_samples
    elif session_portfolio and not meta.get("available_samples") and not _session_portfolio_is_stale:
        meta["available_samples"] = list(session_portfolio.keys())
    if session_profiling_result and not meta.get("profiling_result"):
        meta["profiling_result"] = session_profiling_result
    if session_full_profiling_result and not meta.get("full_profiling_result"):
        meta["full_profiling_result"] = session_full_profiling_result
    if session_sample_statistics and not meta.get("sample_statistics"):
        meta["sample_statistics"] = session_sample_statistics
    # If the shared session already carries the full small-file portfolio/profile,
    # trust it and skip the eager Supabase refresh. We require at least 2 named
    # samples (excluding quick/full) plus a semantic profile to consider the
    # session data complete and avoid overwriting it with a partial Supabase read.
    _session_named_keys = [k for k in (session_portfolio or {}).keys() if k not in ("quick", "full")]
    if (
        _preview_sample_status == "full_sample"
        and not _session_portfolio_is_stale
        and session_portfolio
        and session_full_profiling_result
        and session_sample_statistics
        and len(_session_named_keys) >= 2
    ):
        meta["_phase"] = "full"
    if meta:
        _profile_meta[dataset_id] = meta
    _preview_bg_task_id = (sess_for_ds or {}).get("background_task_id")
    _preview_analysis_task_id = (sess_for_ds or {}).get("analysis_task_id")
    _preview_requires_fidelity_choice = bool((sess_for_ds or {}).get("requires_fidelity_choice"))
    _preview_runtime_hint = (sess_for_ds or {}).get("estimated_runtime_hint")
    _preview_file_size = int((sess_for_ds or {}).get("file_size_bytes") or 0)
    _preview_fidelity_prompt = (sess_for_ds or {}).get("fidelity_prompt") or (
        _build_fidelity_prompt(_preview_file_size) if _preview_requires_fidelity_choice else None
    )
    
    _preview_analysis_fidelity = (sess_for_ds or {}).get("analysis_fidelity") or (
        FIDELITY_QUICK if _preview_sample_status == "quick_sample" else FIDELITY_PORTFOLIO
    )
    _preview_selected_sample_name = (sess_for_ds or {}).get("selected_sample_name")


    # Upload-time auto-viz (build_visualization_config_from_sample) is stored on
    # the session as a JSON string; decode it so open-for-analysis re-renders the
    # full 4-chart config instead of the frontend's 2-chart fallback.
    _preview_viz_config = (
        _maybe_json_load((sess_for_ds or {}).get("visualization_config"))
        or (sess_for_ds or {}).get("visualization_config")
        or {}
    )
    _preview_viz_status = (sess_for_ds or {}).get("visualization_status")

    # Fallback: if session still says 'quick_sample' but the background
    # Celery task has already succeeded, upgrade to 'full_sample' and
    # persist so future requests skip this check.
    if _preview_sample_status == "quick_sample" and _preview_bg_task_id:
        try:
            _rk = AvalokaEntry.generate_key(celery_app, _preview_bg_task_id)
            _eid = AvalokaScheduler.get_last_run_task_id(_rk, False)
            if _eid:
                _ar = AsyncResult(_eid,  app=celery_app)
                if (_ar.ready() and _ar.successful()) or (
                        not _ar.ready() and _run_finished_state(_eid) == "SUCCESS"):
                    _preview_sample_status = "full_sample"
                    sess_for_ds["sample_status"] = "full_sample"
                    sess_for_ds["profiling_status"] = "full_profile"
                    _celery_res = _ar.result if _ar.successful() and isinstance(_ar.result, dict) else {}
                    _celery_avail = _celery_res.get("available_samples") or []
                    if _celery_avail:
                        sess_for_ds["available_samples"] = json.dumps(_celery_avail) if not isinstance(_celery_avail, str) else _celery_avail
                    await save_session(session_id, sess_for_ds)
                    logger.info(
                        "[preview] Detected completed background task %s — upgraded to full_sample + %d available_samples",
                        _preview_bg_task_id, len(_celery_avail),
                    )
        except Exception:
            logger.debug("[preview] Could not check background task status", exc_info=True)

    stored_profile: Optional[Dict[str, Any]] = None
    stored_portfolio: Optional[Dict[str, List[Dict[str, Any]]]] = None

    _cached_available = meta.get("available_samples") or []
    _portfolio_looks_complete = (
        meta.get("_phase") == "full"
        and len(_cached_available) >= 2
        and meta.get("full_profiling_result")
    )
    _needs_refresh = (
        not meta
        or not meta.get("portfolio_samples")
        or (_preview_sample_status == "full_sample" and not _portfolio_looks_complete)
    )
    if _needs_refresh:
        try:
            stored_portfolio, stored_profile = await _load_persisted_preview_artifacts(
                dataset_id
            )
            meta = meta or {}
            if stored_portfolio:
                _fresh_available = [
                    k for k in stored_portfolio.keys() if k not in ("quick", "full")
                ] or [k for k in stored_portfolio.keys() if k != "quick"]
                meta["portfolio_samples"] = stored_portfolio
                meta["available_samples"] = _fresh_available
                if sess_for_ds is not None:
                    sess_for_ds["available_samples"] = json.dumps(_fresh_available)
            if stored_profile:
                meta["full_profiling_result"] = stored_profile.get("full_profiling_result")
                meta["sample_statistics"] = stored_profile.get("sample_statistics")
                meta["profiling_result"] = stored_profile.get("profiling_result")
                if stored_profile.get("portfolio_sample_uris"):
                    meta["portfolio_sample_uris"] = stored_profile.get("portfolio_sample_uris")
            _fresh_avail_count = len(meta.get("available_samples") or [])
            if (
                _preview_sample_status == "full_sample"
                and _fresh_avail_count >= 2
                and meta.get("full_profiling_result")
            ):
                meta["_phase"] = "full"
            else:
                meta.pop("_phase", None)
            if meta:
                _profile_meta[dataset_id] = meta
            if sess_for_ds is not None and session_id:
                await save_session(session_id, sess_for_ds)
        except Exception:
            logger.warning("[preview] load from Supabase fallback failed", exc_info=True)
    meta = meta or {}

    # ── Authoritative upgrade: Redis may still say quick_sample after background
    # finished (Celery AsyncResult check failed, worker used different Redis, or
    # session TTL race) while Supabase already has full_profile / portfolio rows.
    if _preview_sample_status == "quick_sample":
        try:
            if stored_profile is None:
                stored_profile = await asyncio.to_thread(load_profile, dataset_id)
            if stored_portfolio is None:
                stored_portfolio = await asyncio.to_thread(load_portfolio, dataset_id)
            has_full_profile = bool(
                stored_profile
                and (
                    stored_profile.get("profiling_result")
                    or stored_profile.get("full_profiling_result")
                )
            )
            has_non_quick_samples = bool(
                stored_portfolio
                and any(
                    k != "quick"
                    and isinstance(v, list)
                    and len(v) > 0
                    for k, v in stored_portfolio.items()
                )
            )
            if has_full_profile or has_non_quick_samples:
                _preview_sample_status = "full_sample"
                sess_for_ds["sample_status"] = "full_sample"
                sess_for_ds["profiling_status"] = "full_profile"
                meta = meta or {}
                if stored_portfolio:
                    _fresh_available = [
                        k for k in stored_portfolio.keys() if k not in ("quick", "full")
                    ] or [k for k in stored_portfolio.keys() if k != "quick"]
                    meta["portfolio_samples"] = stored_portfolio
                    meta["available_samples"] = _fresh_available
                    sess_for_ds["available_samples"] = json.dumps(_fresh_available)
                if stored_profile:
                    meta["full_profiling_result"] = stored_profile.get("full_profiling_result")
                    meta["sample_statistics"] = stored_profile.get("sample_statistics")
                    meta["profiling_result"] = stored_profile.get("profiling_result")
                    if stored_profile.get("portfolio_sample_uris"):
                        meta["portfolio_sample_uris"] = stored_profile.get("portfolio_sample_uris")
                _avail_count = len(meta.get("available_samples") or [])
                if _avail_count >= 2 and meta.get("full_profiling_result"):
                    meta["_phase"] = "full"
                else:
                    meta.pop("_phase", None)
                _profile_meta[dataset_id] = meta
                await save_session(session_id, sess_for_ds)
                logger.info(
                    "[preview] Supabase has full artifacts but session was quick_sample — upgraded to full_sample + %d available_samples",
                    _avail_count,
                )
        except Exception:
            logger.debug("[preview] Supabase authoritative upgrade failed", exc_info=True)

    # If we learned portfolio_sample_uris from Supabase, persist into session for send_message reuse.
    try:
        if session_id and sess_for_ds is not None and meta.get("portfolio_sample_uris") and not sess_for_ds.get("portfolio_sample_uris"):
            sess_for_ds["portfolio_sample_uris"] = json.dumps(meta.get("portfolio_sample_uris"))
            await save_session(session_id, sess_for_ds)
    except Exception:
        logger.debug("[preview] Could not persist portfolio_sample_uris into session", exc_info=True)

    # When full_sample is ready, show full portfolio rows in main table (not quick sample)
    if _preview_sample_status == "full_sample" and meta.get("portfolio_samples"):
        _portfolio = meta.get("portfolio_samples") or {}
        _selected = _preview_selected_sample_name or DEFAULT_SAMPLE_NAME
        _full_rows = _portfolio.get(_selected) or _portfolio.get(DEFAULT_SAMPLE_NAME) or (list(_portfolio.values()) or [[]])[0]
        if _full_rows:
            samples = (_full_rows if isinstance(_full_rows, list) else list(_full_rows))[:limit]

    _preview_cap = min(limit, PREVIEW_RESPONSE_ROWS)
    return DatasetPreviewResponse(
        dataset_id=dataset_id,
        schema=schema_any,
        samples=_jsonify(samples[:_preview_cap]),
        ddl_schema=ddl,
        rows_sampled=len(samples),
        session_id=session_id,
        thread_id=thread_id,
        portfolio_samples=_cap_portfolio(meta.get("portfolio_samples"), _preview_cap),
        available_samples=meta.get("available_samples"),
        profiling_result=meta.get("profiling_result"),
        full_profiling_result=meta.get("full_profiling_result"),
        sample_statistics=meta.get("sample_statistics"),
        sample_status=_preview_sample_status,
        background_task_id=_preview_bg_task_id,
        analysis_fidelity=_preview_analysis_fidelity,
        selected_sample_name=_preview_selected_sample_name,
        analysis_task_id=_preview_analysis_task_id,
        requires_fidelity_choice=_preview_requires_fidelity_choice,
        fidelity_prompt=_preview_fidelity_prompt,
        estimated_runtime_hint=_preview_runtime_hint,
        visualization_config=_preview_viz_config,
        visualization_status=_preview_viz_status,
        profiling_status=(sess_for_ds or {}).get("profiling_status"),
        profiling_error=(sess_for_ds or {}).get("profiling_error"),
    )


@app.get("/api/datasets/{dataset_id}/profile")
async def get_dataset_profile(request: Request, dataset_id: str):
    """The upload's background profile job: its status and, once landed, the
    profile and the charts written with it. The UI polls this while
    profiling_status is "pending"."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    sid = await find_session_by_dataset_for_user(dataset_id, user_id)
    sess = await get_session(sid, bypass_circuit=True) if sid else None
    if not isinstance(sess, dict) or sess.get("user_id") != user_id or sess.get("dataset_id") != dataset_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found for this user")

    out = {
        "dataset_id": dataset_id,
        "profiling_status": sess.get("profiling_status"),
        "profiling_error": sess.get("profiling_error"),
        "profiling_result": _maybe_json_load(sess.get("full_profiling_result")) or sess.get("full_profiling_result"),
        "visualization_config": _maybe_json_load(sess.get("visualization_config")) or sess.get("visualization_config"),
        "visualization_status": sess.get("visualization_status"),
    }
    # A stale session save can put "pending" back before the lander re-asserts;
    # this process's copy of the landed job is the fresher answer then.
    landed = (_profile_meta.get(dataset_id) or {}).get("deferred_profile")
    if out["profiling_status"] == PROFILE_PENDING and isinstance(landed, dict):
        out.update(
            profiling_status=landed.get("profiling_status"),
            profiling_error=landed.get("profiling_error"),
            profiling_result=landed.get("full_profiling_result"),
        )
        if landed.get("visualization_config"):
            out["visualization_config"] = landed["visualization_config"]
            out["visualization_status"] = landed.get("visualization_status") or "ready"
    return out


@app.get("/api/datasets/{dataset_id}/background-task-status")
async def get_background_task_status(request: Request, dataset_id: str):
    """Lightweight JSON endpoint for background sampling task status. Used by UI to poll until SUCCESS then fetch /preview once."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    sid = _resolve_session_id(request, None)
    sess = await get_session(sid, bypass_circuit=True) if sid else None
    if not sid or not sess or sess.get("user_id") != user_id or sess.get("dataset_id") != dataset_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Session not found or does not match dataset")
    bg_task_id = (sess or {}).get("background_task_id")
    if not bg_task_id:
        return {
            "state": "NONE",
            "sample_status": (sess or {}).get("sample_status"),
            "profiling_status": (sess or {}).get("profiling_status"),
            "background_task_id": (sess or {}).get("background_task_id"),
            "analysis_task_id": (sess or {}).get("analysis_task_id"),
            "analysis_fidelity": (sess or {}).get("analysis_fidelity"),
            "selected_sample_name": (sess or {}).get("selected_sample_name"),
        }
    redbeat_key = AvalokaEntry.generate_key(celery_app, bg_task_id)
    try:
        AvalokaEntry.from_key(redbeat_key, celery_app)
    except Exception:
        return {"state": "NONE"}
    execution_id = AvalokaScheduler.get_last_run_task_id(redbeat_key, False)
    if not execution_id:
        return {
            "state": "PENDING",
            "sample_status": (sess or {}).get("sample_status"),
            "profiling_status": (sess or {}).get("profiling_status"),
            "background_task_id": (sess or {}).get("background_task_id"),
            "analysis_task_id": (sess or {}).get("analysis_task_id"),
            "analysis_fidelity": (sess or {}).get("analysis_fidelity"),
            "selected_sample_name": (sess or {}).get("selected_sample_name"),
        }
    r = AsyncResult(execution_id, app=celery_app)
    # Not final in Celery: a result that is no longer stored reads PENDING, so
    # ask the durable run record before reporting RUNNING.
    finished = None if r.ready() else _run_finished_state(execution_id)
    if not r.ready() and not finished:
        return {
            "state": "RUNNING",
            "sample_status": (sess or {}).get("sample_status"),
            "profiling_status": (sess or {}).get("profiling_status"),
            "background_task_id": (sess or {}).get("background_task_id"),
            "analysis_task_id": (sess or {}).get("analysis_task_id"),
            "analysis_fidelity": (sess or {}).get("analysis_fidelity"),
            "selected_sample_name": (sess or {}).get("selected_sample_name"),
        }
    if (r.ready() and r.successful()) or finished == "SUCCESS":
        res = r.result if r.ready() and isinstance(r.result, dict) else {}
        out_sample_status = res.get("sample_status", "full_sample")
        out_profiling_status = res.get("profiling_status", (sess or {}).get("profiling_status"))
        out_available_samples = res.get("available_samples") or []
        # Keep Redis session aligned with Celery result (fixes stale quick_sample in /preview)
        if sess and out_sample_status == "full_sample":
            try:
                sess["sample_status"] = "full_sample"
                sess["profiling_status"] = out_profiling_status or "full_profile"
                if not sess.get("analysis_fidelity") or sess.get("analysis_fidelity") == "quick_sample":
                    sess["analysis_fidelity"] = "portfolio_samples"
                if out_available_samples:
                    sess["available_samples"] = json.dumps(out_available_samples) if not isinstance(out_available_samples, str) else out_available_samples
                await save_session(sid, sess)
                logger.info(
                    "[bg-task-status] Synced session %s to full_sample + %d available_samples after Celery SUCCESS",
                    sid, len(out_available_samples),
                )
            except Exception:
                logger.debug("[bg-task-status] session sync failed", exc_info=True)
        return {
            "state": "SUCCESS",
            "sample_status": out_sample_status,
            "profiling_status": out_profiling_status,
            "background_task_id": (sess or {}).get("background_task_id"),
            "analysis_task_id": (sess or {}).get("analysis_task_id"),
            "analysis_fidelity": (sess or {}).get("analysis_fidelity"),
            "selected_sample_name": (sess or {}).get("selected_sample_name"),
            "available_samples": out_available_samples,
        }
    return {
        "state": "FAILURE",
        "sample_status": (sess or {}).get("sample_status"),
        "profiling_status": (sess or {}).get("profiling_status"),
        "background_task_id": (sess or {}).get("background_task_id"),
        "analysis_task_id": (sess or {}).get("analysis_task_id"),
        "analysis_fidelity": (sess or {}).get("analysis_fidelity"),
        "selected_sample_name": (sess or {}).get("selected_sample_name"),
    }


class FidelityUpdateIn(BaseModel):
    analysis_fidelity: Optional[str] = None      # quick_sample | portfolio_samples | entire_dataset
    selected_sample_name: Optional[str] = None   # e.g. random_baseline, strat_brand


async def _dataset_session_for_user(request: Request, dataset_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    sid = await find_session_by_dataset_for_user(dataset_id, user_id)
    sess = await get_session(sid) if sid else None
    if not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found for this user")
    return user_id, sid, sess


def _fidelity_options(sess: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    size = int(sess.get("file_size_bytes") or 0)
    large = size >= LARGE_DATASET_THRESHOLD_BYTES
    sample_status = str(sess.get("sample_status") or "")
    portfolio_ready = sample_status == "full_sample" or not large
    has_cloud = bool(_full_cloud_uri(sess) and sess.get("connection_id"))
    samples = (_maybe_json_load(sess.get("available_samples")) or sess.get("available_samples")
               or (_profile_meta.get(dataset_id) or {}).get("available_samples") or [])
    building = bool(sess.get("background_task_id")) and not portfolio_ready

    return {
        "dataset_id": dataset_id,
        "analysis_fidelity": sess.get("analysis_fidelity")
            or (FIDELITY_QUICK if sample_status == "quick_sample" else FIDELITY_PORTFOLIO),
        "selected_sample_name": sess.get("selected_sample_name"),
        "file_size_bytes": size,
        "is_large_dataset": large,
        "analysis_task_id": sess.get("analysis_task_id"),
        "options": [
            {
                "value": FIDELITY_QUICK,
                "label": "Quick sample",
                "enabled": True,
                "detail": "First rows of the file. Answers in seconds.",
            },
            {
                "value": FIDELITY_PORTFOLIO,
                "label": "Portfolio samples",
                "enabled": portfolio_ready or (large and has_cloud),
                "status": "ready" if portfolio_ready else ("building" if building else "not_built"),
                "samples": samples,
                "detail": ("Representative samples across the file."
                           if portfolio_ready else
                           "Built in the background on the cluster when selected."),
                "reason": None if (portfolio_ready or has_cloud) else "Needs a cloud connection",
            },
            {
                "value": FIDELITY_ENTIRE,
                "label": "Full dataset",
                "enabled": (not large) or has_cloud,
                "runs_as_job": large,
                "detail": ("Each question runs as a background Ray job."
                           if large else "Runs on every row."),
                "reason": None if ((not large) or has_cloud)
                          else "Large files need a cloud connection to run on the cluster",
            },
        ],
    }


@app.get("/api/datasets/{dataset_id}/fidelity", tags=["datasets"])
async def get_dataset_fidelity(request: Request, dataset_id: str):
    _, _, sess = await _dataset_session_for_user(request, dataset_id)
    return _fidelity_options(sess, dataset_id)


@app.put("/api/datasets/{dataset_id}/fidelity", tags=["datasets"])
async def set_dataset_fidelity(request: Request, dataset_id: str, body: FidelityUpdateIn):
    _, sid, sess = await _dataset_session_for_user(request, dataset_id)
    current = _fidelity_options(sess, dataset_id)
    by_value = {o["value"]: o for o in current["options"]}

    fid = body.analysis_fidelity
    if body.selected_sample_name and not fid:
        fid = FIDELITY_PORTFOLIO               # picking a sample implies portfolio mode
    if fid and fid not in by_value:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Unknown fidelity '{fid}'")
    if fid and not by_value[fid]["enabled"]:
        raise HTTPException(status.HTTP_409_CONFLICT, by_value[fid].get("reason") or "Not available")

    samples = by_value[FIDELITY_PORTFOLIO]["samples"]
    if body.selected_sample_name and samples and body.selected_sample_name not in samples:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Sample '{body.selected_sample_name}' is not available")

    # Large file, portfolio not built yet: start the background build now.
    scheduled = None
    if (fid == FIDELITY_PORTFOLIO and current["is_large_dataset"]
            and str(sess.get("sample_status")) == "quick_sample"
            and not sess.get("background_task_id")):
        scheduled = _schedule_background_sampling_task(
            dataset_id=dataset_id,
            session_id=sid,
            cloud_uri=_full_cloud_uri(sess) or "",
            connection_id=sess.get("connection_id") or "",
            file_size_bytes=int(sess.get("file_size_bytes") or 0),
            ray_namespace=os.getenv("RAY_NAMESPACE", "default"),
        )

    def _apply(cur: Dict[str, Any]) -> None:
        if fid:
            cur["analysis_fidelity"] = fid
            cur["requires_fidelity_choice"] = False
            cur["fidelity_prompt_shown"] = True
        if body.selected_sample_name:
            cur["selected_sample_name"] = body.selected_sample_name
        if scheduled:
            cur["background_task_id"] = scheduled
            cur["portfolio_generation_status"] = "scheduled"

    await update_session(sid, _apply)
    logger.info("[fidelity-ui] dataset=%s -> %s sample=%s scheduled=%s",
                dataset_id, fid, body.selected_sample_name, scheduled)

    fresh = await get_session(sid) or sess
    return {**_fidelity_options(fresh, dataset_id), "background_task_id": scheduled}

@app.delete(
    "/datasets/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["datasets"],
)
async def delete_dataset(request: Request, dataset_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.",
        )
    if dataset_id not in (await _user_datasets(user_id)):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Dataset not found for this user",
        )

    try:
        sid_set_key = _k_user_sessions(user_id)

        user_sids: List[str] = []
        if hasattr(session_service.cache, "sscan_iter"):
            async for sid in session_service.cache.sscan_iter(  # type: ignore[attr-defined]
                sid_set_key,
                count=200,
                cap=5000,
            ):
                user_sids.append(sid)
        else:
            user_sids = list(
                await session_service.cache.smembers(sid_set_key)
            )[:5000]

        blobs = await _mget_safe([_k_session(s) for s in user_sids])

        target_sids: List[str] = []
        object_refs: List[Tuple[Optional[str], Optional[str]]] = []

        for sid, blob in zip(user_sids, blobs):
            if not blob:
                continue
            try:
                s = json.loads(blob)
            except Exception:
                continue
            if s.get("dataset_id") == dataset_id:
                target_sids.append(sid)
                object_refs.append(
                    (s.get("data_source_location"), s.get("object_name"), s.get("source_kind"))
                )

        for storage_uri, key, source_kind in object_refs or []:
            # Only delete storage objects Avaloka created. register-existing
            # datasets point at the customer's pre-existing source file, so
            # deleting it would destroy their data. Anything not explicitly
            # marked "uploaded" is left in place (fail safe).
            if source_kind != "uploaded":
                logger.info(
                    "delete_dataset: skipping storage delete for non-owned object "
                    "(source_kind=%s, uri=%s)",
                    source_kind,
                    storage_uri,
                )
                continue
            try:
                store_for_read, key_for_read = _store_and_key_from_uri(
                    storage_uri,
                    key,
                )
                if store_for_read and key_for_read:
                    store_for_read.delete(key_for_read)
            except Exception:
                logger.warning(
                    "Storage delete failed (uri=%s, key=%s)",
                    storage_uri,
                    key,
                    exc_info=True,
                )

        for sid in target_sids:
            await session_service.cache.delete(_k_session(sid))
            await session_service.cache.srem(sid_set_key, sid)

        # Collect threads bound to the deleted sessions: the durable reverse
        # index (survives restarts) plus the in-process scan (covers threads
        # bound before the index existed).
        doomed_tids: set = set()
        for sid in target_sids:
            try:
                if session_service.cache:
                    members = await session_service.cache.smembers(_k_session_threads(sid))
                    doomed_tids.update(members or [])
            except Exception:
                pass
        for tid in list(THREAD_META.keys()):
            try:
                bound_sid = await get_thread_session(tid)
                if bound_sid in target_sids:
                    doomed_tids.add(tid)
            except Exception:
                pass
        for tid in doomed_tids:
            try:
                # Delete the durable copies BEFORE popping the in-process
                # entry so a concurrent hydrate can't resurrect the history.
                await delete_thread_history(tid)
                if session_service.cache:
                    await session_service.cache.delete(_k_thread_session(tid))
                THREAD_META.pop(tid, None)
            except Exception:
                pass
            await _forget_thread_memory(user_id, tid)
        # The facts learned about this dataset go with it.
        try:
            from app.services.memory_plane import forget_dataset_memory
            await asyncio.to_thread(forget_dataset_memory, user_id, dataset_id)
        except Exception:
            logger.warning("Memory cleanup failed for dataset %s", dataset_id, exc_info=True)
        for sid in target_sids:
            try:
                if session_service.cache:
                    await session_service.cache.delete(_k_session_threads(sid))
            except Exception:
                pass

        for sid in target_sids:
            try:
                sess = await get_session(sid)
                wd = sess and sess.get("work_dir")
                if wd:
                    _safe_rmtree(wd)
            except Exception:
                pass

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("delete_dataset failed: %s", e)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to delete dataset",
        )


# -------------------------------------------------------------------
# Buckets listing
# -------------------------------------------------------------------

@app.get("/buckets/list", response_model=BucketListResponse)
async def list_bucket_objects(
    request: Request,
    backend: str = Query(
        ...,
        pattern="^(s3|gcs|gs|azure|az)$",
        description="Storage backend: s3 | gcs (or gs) | azure (or az)",
    ),
    bucket: str = Query(
        ...,
        description="S3 bucket / GCS bucket / Azure container name (NO prefix here)",
    ),
    prefix: str = Query(
        "",
        description="Optional folder/prefix inside the bucket/container",
    ),
    connection_id: Optional[str] = Query(
        None,
        description="Cloud connection id in Supabase (used for credentials/region)",
    ),
):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    backend_lower = backend.lower()
    normalized_prefix = prefix.strip().lstrip("/")

    conn: Dict[str, Any] = {}

    if backend_lower in ("s3", "azure", "az") and not connection_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "connection_id is required for S3 and Azure backends.",
        )
    if connection_id:
        conn = await get_cloud_connection(connection_id)
        if not connection_belongs_to_user(conn, user_id):
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Cloud connection not found for id={connection_id}",
            )
        logger.info(
            "[buckets/list] using connection_id=%s provider=%s bucket_name=%s row=%s",
            connection_id,
            (conn.get("provider") or conn.get("backend")),
            (conn.get("bucket_name") or conn.get("container")),
            redact_connection(conn),
        )

    if backend_lower == "s3":
        storage_uri = f"s3://{bucket}" + (
            f"/{normalized_prefix}" if normalized_prefix else ""
        )
    elif backend_lower in ("gcs", "gs"):
        storage_uri = f"gs://{bucket}" + (
            f"/{normalized_prefix}" if normalized_prefix else ""
        )
    elif backend_lower in ("azure", "az"):
        account_name = (
            conn.get("access_key")
            or conn.get("account_name")
            or conn.get("azure_account")
        )
        if not account_name:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Azure connection is missing the storage account name.",
            )
        storage_uri = (
            f"az://{account_name}/{bucket}"
            + (f"/{normalized_prefix}" if normalized_prefix else "")
        )
    else:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unknown backend '{backend}'. Use s3 | gcs | azure.",
        )

    try:
        store, base_prefix = await _store_from_connection_uri(
            storage_uri,
            conn or {},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[buckets/list] failed to initialize storage client")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to initialize storage client for {backend_lower}://{bucket}: {e}",
        )

    try:
        folder_prefixes, file_tuples = store.list_hierarchy("")
    except Exception as e:
        logger.exception(
            "[buckets/list] failed listing level for %s://%s/%s",
            backend_lower, bucket, normalized_prefix,
        )
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to list {backend_lower}://{bucket}/{normalized_prefix or ''}: {e}",
        )

    def _folder_out(rel: str) -> Dict[str, str]:
        name = rel.rstrip("/").rsplit("/", 1)[-1]
        # rel is relative to the store's baked-in prefix; the browsable
        # path we hand back is normalized_prefix + rel so navigation composes.
        nav = f"{normalized_prefix.rstrip('/')}/{rel}" if normalized_prefix else rel
        return {"name": name, "prefix": nav.lstrip("/")}

    folders = [_folder_out(p) for p in folder_prefixes]

    def _file_key_full(rel_key: str) -> str:
        # `store` is rooted at storage_uri (which includes normalized_prefix),
        # so list_hierarchy returns keys relative to that level (bare filename).
        # Prepend normalized_prefix ONCE to get the bucket-relative object key
        # so register-existing composes bucket + key with no missing folder.
        rel_key = rel_key.lstrip("/")
        return f"{normalized_prefix.rstrip('/')}/{rel_key}" if normalized_prefix else rel_key

    objects: List[Dict[str, str]] = []
    for key, size, updated in file_tuples:
        full_key = _file_key_full(key)
        ext = full_key.rsplit(".", 1)[-1].lower() if "." in full_key else ""
        if ext not in SUPPORTED_UPLOAD_EXTS:
            continue
        objects.append({"key": full_key, "size": str(size), "updated": str(updated)})

    return BucketListResponse(
        backend=backend_lower,
        bucket=bucket,
        prefix=normalized_prefix,
        folders=[BucketFolder(**f) for f in folders],
        objects=objects,
    )

# -------------------------------------------------------------------
# DB Connect & Chat (MCP-backed)
# -------------------------------------------------------------------
_DB_IDENT_RE = re.compile(r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)?$")  # schema.table too


async def _run_mcp_db_query(customer_id: str, api_key: str, sql: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        payload = {"method": "call_tool", "params": {"name": "query", "arguments": {"sql": sql}}}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = await client.post(f"{MCP_SERVER_URL}/call_tool", json=payload, headers=headers, timeout=MCP_TIMEOUT)
        resp.raise_for_status()
        return resp.json()


class TablesToAnalysisIn(BaseModel):
    tables: List[str]                       # table names selected in the modal
    customer_id: Optional[str] = None
    api_key: Optional[str] = None
    session_id: Optional[str] = None        # DB-connect session, to resolve creds
    limit: int = MAX_DB_SAMPLE_ROWS


@app.post("/api/database/tables-to-analysis", response_model=MultiUploadResponse)
async def db_tables_to_analysis(request: Request, response: Response, body: TablesToAnalysisIn):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Session storage (Redis) is not available.")

    # Resolve creds (same fallback as chat_db_sample_only).
    customer_id = body.customer_id
    api_key = body.api_key
    if not (customer_id and api_key):
        sid = _resolve_session_id(request, body.session_id)
        db_sess = await get_session(sid) if sid else None
        if db_sess and db_sess.get("user_id") == user_id and db_sess.get("type") == "database":
            customer_id = customer_id or db_sess.get("customer_id")
            if not api_key:
                api_key = decrypt_secret(db_sess.get("api_key_enc"))
    if not (customer_id and api_key):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Missing database credentials. Connect first, then pass its session_id.")

    tables = [t.strip() for t in (body.tables or []) if t and t.strip()]
    if not tables:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Provide at least one table.")
    if len(tables) > MAX_UPLOAD_FILES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"Too many tables. Max is {MAX_UPLOAD_FILES}.")
    for t in tables:
        if not _DB_IDENT_RE.match(t):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid table name: {t!r}")

    limit = max(1, min(int(body.limit or MAX_DB_SAMPLE_ROWS), MAX_DB_SAMPLE_ROWS))

    group_session_id: Optional[str] = None
    thread_id = ""
    dataset_ids: List[str] = []
    dataset_session_map: Dict[str, str] = {}
    datasets_out: List[UploadedDatasetOut] = []
    created_session_ids: List[str] = []
    created_work_dirs: List[Path] = []
    seen_aliases: Dict[str, int] = {}

    try:
        async def _query(sql: str) -> Dict[str, Any]:
            return await _run_mcp_db_query(customer_id, api_key, sql)

        for idx, table in enumerate(tables):
            # A random sample (or the whole table when it fits), never the
            # first rows: a head sample of a sorted table answered 0.50 for a
            # rate whose true value is 0.13. See app/core/db_sampling.py.
            try:
                cols, rows, db_sample = await draw_table_sample(_query, table, limit, table=table)
            except DbSampleError as exc:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Failed to read '{table}': {exc}")
            if not cols or not rows:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Table '{table}' returned no rows.")

            dict_rows = _rows_list_to_dicts(cols, rows[:limit])
            dsid = _new_id()
            session_id = _new_id()
            if idx == 0:
                group_session_id = session_id

            work_dir = (TMP_ROOT / session_id).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
            created_work_dirs.append(work_dir)

            sample_input_path = work_dir / "sample_input.csv"
            _write_rows_to_csv(dict_rows, sample_input_path)
            output_path = work_dir / "output.csv"

            schema_dict = _schema_from_db_rows(cols, dict_rows)
            ddl = _ddl_from_schema(dsid, schema_dict)

            base_alias = "".join(ch if ch.isalnum() else "_" for ch in table).strip("_") or f"table{idx}"
            n = seen_aliases.get(base_alias, 0)
            seen_aliases[base_alias] = n + 1
            alias = base_alias if n == 0 else f"{base_alias}_{n + 1}"

            try:
                viz = build_visualization_config_from_sample(
                    dataset_id=dsid, sample_rows=dict_rows, schema=schema_dict,
                    task_type="unsupervised", target_column=None,
                )
                viz_status = viz.get("visualization_status", "ready")
            except Exception:
                viz, viz_status = {}, "error"

            session_data = {
                "user_id": user_id,
                "dataset_id": dsid,
                "work_dir": str(work_dir),
                "data_source_location": f"db://{customer_id}/{dsid}?sample=true",
                "customer_id": customer_id,
                "source_table": table,
                "db_sample": db_sample.as_session(),
                # These rows came from the whole table, so a plain aggregate can
                # be pushed down to it (see _db_pushdown_answer).
                "db_pushdown_table": table,
                "created_at": _now_iso(),
                "object_name": None,
                "work_local_input": None,
                "output_location": _posix(output_path),
                "sample_local_input": _posix(sample_input_path),
                "uploaded_csv_preview": _jsonify(dict_rows),
                "uploaded_csv_columns": cols,
                "schema": _jsonify(schema_dict),
                "ddl_schema": ddl,
                "assistant_id": ASSISTANT_ID,
                "input_data_type": "db",
                "filename": table,
                "alias": alias,
                "group_session_id": group_session_id,
                "visualization_config": _jsonify(viz),
                "visualization_status": viz_status,
            }
            if not await save_session(session_id, session_data):
                _safe_rmtree(work_dir)
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to persist DB dataset session.")
            created_session_ids.append(session_id)

            if idx == 0:
                secure_cookie_env = os.getenv("COOKIE_SECURE", "").lower()
                secure_cookie = (True if secure_cookie_env == "true"
                                 else False if secure_cookie_env == "false"
                                 else (request.url.scheme == "https"))
                response.set_cookie(key=COOKIE_NAME, value=session_id, httponly=True,
                                    samesite="lax", secure=secure_cookie, max_age=SESSION_TTL_SECONDS)
                try:
                    thread_res = await lg_json("POST", "/threads", json={"metadata": {
                        "source": "db-tables", "assistant_id": ASSISTANT_ID, "user_id": user_id,
                        "dataset_id": dsid, "session_id": session_id, "schema": schema_dict,
                    }})
                    thread_id = thread_res.get("thread_id") or thread_res.get("id") or ""
                    if thread_id:
                        await bind_thread_session(thread_id, session_id)
                        session_data["thread_id"] = thread_id
                        await save_session(session_id, session_data)
                except HTTPException as e:
                    logger.warning("[db-tables] thread create failed (soft): %s", e.detail)
            elif thread_id:
                session_data["thread_id"] = thread_id
                await save_session(session_id, session_data)

            dataset_ids.append(dsid)
            dataset_session_map[dsid] = session_id
            datasets_out.append(UploadedDatasetOut(
                dataset_id=dsid, filename=table, alias=alias,
                columns=cols, rows=_jsonify(_cap_rows(dict_rows)),
                visualization_config=viz, visualization_status=viz_status,
            ))
    except HTTPException:
        for wd in created_work_dirs:
            try: _safe_rmtree(wd)
            except Exception: pass
        for s in created_session_ids:
            try: await delete_session(s)
            except Exception: pass
        raise
    except Exception as e:
        logger.exception("[db-tables] failed: %s", e)
        for wd in created_work_dirs:
            try: _safe_rmtree(wd)
            except Exception: pass
        for s in created_session_ids:
            try: await delete_session(s)
            except Exception: pass
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to create analysis from tables")

    assert group_session_id is not None
    try:
        group_sess = await get_session(group_session_id)
        if isinstance(group_sess, dict):
            group_sess["dataset_ids"] = dataset_ids
            group_sess["dataset_session_map"] = dataset_session_map
            group_sess["thread_id"] = thread_id or group_sess.get("thread_id")
            await save_session(group_session_id, group_sess)
    except Exception:
        logger.warning("[db-tables] failed to persist group index", exc_info=True)

    return MultiUploadResponse(session_id=group_session_id, thread_id=thread_id, datasets=datasets_out)

async def list_database_tables(customer_id: str, api_key: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "method": "call_tool",
                "params": {"name": "list_tables", "arguments": {}},
            }
            resp = await client.post(
                f"{MCP_SERVER_URL}/call_tool",
                json=payload,
                headers=headers,
                timeout=MCP_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"Failed to list tables: {resp.text}"}
    except Exception as e:
        return {"error": f"Error listing tables: {str(e)}"}


async def _bind_mcp_customer_owner(customer_id: Optional[str], user_id: str, api_key: Optional[str]) -> None:
    """Bind a registered database to the user who holds its api_key.

    The onboarding API scopes every /customers route to an owner, and releases a
    password to the planner only for a customer bound to the user it names (or
    to nobody yet). The UI registers anonymously, so this is where the binding
    happens: here the user is verified and the api_key is in hand. Best-effort;
    never fails the caller's request.
    """
    token = os.getenv("MCP_SERVICE_TOKEN", "")
    if not (token and customer_id and user_id and api_key):
        return
    base = os.getenv("ONBOARDING_API_URL", "http://localhost:8081").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{base}/customers/{customer_id}/owner",
                headers={
                    "X-Avaloka-Service-Token": token,
                    "X-Avaloka-User": str(user_id),
                    "Authorization": f"Bearer {api_key}",
                },
            )
        if resp.status_code != 200:
            logger.warning("[mcp-creds] owner bind for %s -> HTTP %s", customer_id, resp.status_code)
    except Exception as exc:
        logger.warning("[mcp-creds] owner bind for %s failed: %s", customer_id, exc)


@app.post("/api/database/connect")
async def connect_database(
    request: Request,
    body: DatabaseConnectRequest,
):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.",
        )

    customer_id = str(body.customer_id or "").strip()
    api_key = str(body.api_key or "").strip()
    connection_id = str(body.connection_id or "").strip()
    database_type_hint: Optional[str] = None

    if not (customer_id and api_key):
        if customer_id or api_key:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "customer_id and api_key must be supplied together.",
            )
        if not connection_id:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Provide customer_id and api_key, or a saved connection_id.",
            )
        try:
            legacy_credentials = await resolve_legacy_database_credentials(
                connection_id, user_id
            )
        except LegacyConnectionResolutionError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        customer_id = legacy_credentials.customer_id
        api_key = legacy_credentials.api_key
        database_type_hint = legacy_credentials.database_type

    response_connection_id = connection_id or customer_id

    test_result = await list_database_tables(customer_id, api_key)
    if "error" in test_result:
        raise HTTPException(
            400,
            f"Database connection failed: {test_result['error']}",
        )
    # The MCP server just accepted api_key for customer_id: that is the proof of
    # possession. A customer bound to nobody yet is bound to this user now, so
    # from here on only they can have the service transfer from it.
    await _bind_mcp_customer_owner(customer_id, user_id, api_key)

    session_id = _new_id()
    thread_id = _new_id()
    # Persist the api_key encrypted at rest so /api/v1/database/query can reuse
    # this session without the client resending raw credentials on every call.
    session_data = {
        "user_id": user_id,
        "type": "database",
        "customer_id": customer_id,
        "connection_id": response_connection_id,
        "thread_id": thread_id,
        "database_info": test_result.get("meta", {}),
        "created_at": _now_iso(),
        # Picked up by avaloka_agent on the next chat turn for proactive
        # onboarding. Carries no credential: api_key is stored only encrypted,
        # below, and never enters the event the agent puts in its prompt.
        # 1.6's connect takes customer_id + api_key only. When the legacy
        # saved-connection path (#248) lands, pass its connection_id and the
        # saved connection's database_type here as database_type_hint.
        "connection_event": database_connection_event(
            customer_id=customer_id,
            test_result=test_result,
            created_at=_now_iso(),
        ),
    }
    api_key_enc = encrypt_secret(api_key)
    if api_key_enc:
        session_data["api_key_enc"] = api_key_enc
    if not await save_session(session_id, session_data):
        raise HTTPException(500, "Failed to persist DB session")
    out = {
        "session_id": session_id,
        "customer_id": customer_id,
        "status": "connected",
        "credentials_saved": bool(api_key_enc),
        "tables_available": len(test_result.get("tables", [])),
        "database_type": (
            test_result.get("meta", {}).get("database_type") or database_type_hint
        ),
    }
    out.update(
        legacy_connect_fields(
            session_id=session_id,
            thread_id=thread_id,
            connection_id=response_connection_id,
            expires_in=SESSION_TTL_SECONDS,
        )
    )
    return out


def _source_table_from_sql(sql: str) -> Optional[str]:
    """Best-effort table name from a ``SELECT ... FROM <table>`` query.

    Lets a DTA transfer use the table the user actually analyzed as the implicit
    source, instead of the connection's auto-detected first table. Returns the
    first FROM target (schema-qualified name preserved), or None if not parseable.
    """
    m = re.search(r"\bfrom\s+([`\"\[]?[A-Za-z_][\w$.]*[`\"\]]?)", sql or "", re.IGNORECASE)
    if not m:
        return None
    return (m.group(1).strip().strip('`"[]') or None)


@app.post("/api/v1/database/query", response_model=DatabaseChatResponse)
async def chat_db_sample_only(
    request: Request,
    body: DatabaseChatRequest,
    response: Response,
):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    query_started = time.perf_counter()
    query_text = database_query_text(body.content, body.query)
    if not query_text:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Provide database query text in content or query.",
        )
    customer_id = body.customer_id or body.metadata.get("customer_id")
    api_key = (body.metadata or {}).get("api_key")
    connection_id = str(
        body.connection_id or body.metadata.get("connection_id") or ""
    ).strip()
    sid = _resolve_session_id(
        request, body.session_id or body.metadata.get("session_id")
    )
    db_sess = await get_session(sid) if sid else None
    owned_db_sess = (
        db_sess
        if db_sess
        and db_sess.get("user_id") == user_id
        and db_sess.get("type") == "database"
        else None
    )
    response_session_id = sid if owned_db_sess else None
    if (
        owned_db_sess
        and connection_id
        and owned_db_sess.get("connection_id")
        and connection_id != str(owned_db_sess["connection_id"])
    ):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The saved database session belongs to a different connection_id.",
        )

    # Fall back to credentials saved by /api/database/connect so a client that
    # followed the connect-then-query flow doesn't resend raw creds each call.
    if not (customer_id and api_key):
        if owned_db_sess:
            customer_id = customer_id or owned_db_sess.get("customer_id")
            connection_id = connection_id or str(
                owned_db_sess.get("connection_id") or ""
            ).strip()
            if not api_key:
                api_key = decrypt_secret(owned_db_sess.get("api_key_enc"))

    if not (customer_id and api_key) and connection_id:
        try:
            legacy_credentials = await resolve_legacy_database_credentials(
                connection_id, user_id
            )
        except LegacyConnectionResolutionError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        customer_id = customer_id or legacy_credentials.customer_id
        api_key = api_key or legacy_credentials.api_key

    if not (customer_id and api_key):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Missing database credentials. Provide customer_id and api_key in the "
            "request body, or call /api/database/connect first and pass its "
            "session_id.",
        )

    content_lower = query_text.lower()
    tool_name = "query"
    arguments: Dict[str, Any] = {"sql": query_text}
    if content_lower in ["list tables", "show tables"]:
        tool_name = "list_tables"
        arguments = {}
    elif content_lower.startswith("describe table"):
        tool_name = "describe_table"
        arguments = {"table_name": query_text.split()[-1]}

    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "method": "call_tool",
                "params": {"name": tool_name, "arguments": arguments},
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            resp = await client.post(
                f"{MCP_SERVER_URL}/call_tool",
                json=payload,
                headers=headers,
                timeout=MCP_TIMEOUT,
            )
            resp.raise_for_status()
            result = resp.json()
    except httpx.HTTPStatusError as e:
        msg = (
            f"Database server error: {e.response.status_code} {e.response.text}"
        )
        return DatabaseChatResponse(
            messages=[
                {"role": "user", "content": query_text},
                _assistant_entry(msg),
            ],
            query_result={"error": msg},
            connection_id=connection_id or None,
            session_id=response_session_id,
            thread_id=(
                owned_db_sess.get("thread_id") if owned_db_sess else None
            ),
            **legacy_query_fields(
                {"columns": [], "rows": []},
                execution_time_ms=round(
                    (time.perf_counter() - query_started) * 1000
                ),
            ),
        )

    out = DatabaseChatResponse(
        messages=[{"role": "user", "content": query_text}],
        query_result=result,
        connection_id=connection_id or None,
        session_id=response_session_id,
        thread_id=(owned_db_sess.get("thread_id") if owned_db_sess else None),
        database_info={
            "customer_id": customer_id,
            "query_executed": query_text,
        },
        **legacy_query_fields(
            result if isinstance(result, dict) else {},
            execution_time_ms=round((time.perf_counter() - query_started) * 1000),
        ),
    )

    if (
        tool_name == "query"
        and isinstance(result, dict)
        and "columns" in result
        and "rows" in result
    ):
        cols: List[str] = result.get("columns") or []
        rows: List[List[Any]] = result.get("rows") or []
        _row_limit = ((result.get("meta") or {}).get("row_limit")) or 0
        query_sample: Optional[DbSample] = DbSample(
            "full", len(rows), len(rows), False,
            str((result.get("meta") or {}).get("database_type") or ""),
        )
        if rows and (len(rows) > MAX_DB_SAMPLE_ROWS or (_row_limit and len(rows) >= _row_limit)):
            # The result was cut at a row cap, so what came back is its head.
            # Analyse a random sample of the whole result instead.
            async def _q(sql: str) -> Dict[str, Any]:
                return await _run_mcp_db_query(customer_id, api_key, sql)
            try:
                cols, rows, query_sample = await draw_table_sample(
                    _q, f"({query_text.strip().rstrip(';')}) AS _avk_q",
                    MAX_DB_SAMPLE_ROWS, source_is_table=False,
                )
            except DbSampleError as exc:
                logger.warning("[db-sample] no random sample of a capped result: %s", type(exc).__name__)
                out.messages.append(_assistant_entry(
                    "This result was larger than the analysis row cap and " + str(exc)
                    + " Narrow the query and try again."))
                cols, rows, query_sample = [], [], None

        if cols and rows:
            dsid, sid, tid = await _create_or_update_db_sample_session(
                request=request,
                user_id=user_id,
                customer_id=customer_id,
                columns=cols,
                rows=rows,
                preferred_dataset_id=body.dataset_id,
                preferred_session_id=response_session_id,
                preferred_thread_id=(
                    owned_db_sess.get("thread_id") if owned_db_sess else None
                ),
                source_table=_source_table_from_sql(query_text),
                db_sample=query_sample,
            )

            secure_cookie_env = os.getenv("COOKIE_SECURE", "").lower()
            secure_cookie = (
                True
                if secure_cookie_env == "true"
                else False
                if secure_cookie_env == "false"
                else (request.url.scheme == "https")
            )
            response.set_cookie(
                key=COOKIE_NAME,
                value=sid,
                httponly=True,
                samesite="lax",
                secure=secure_cookie,
                max_age=SESSION_TTL_SECONDS,
            )

            out.dataset_id = dsid
            out.session_id = sid
            out.thread_id = tid
            # pull visualization from the session we just saved
            viz_config: Dict[str, Any] = {}
            viz_status: Optional[str] = None

            try:
                sess = await get_session(sid)
                if sess:
                    raw_viz = sess.get("visualization_config")
                    if isinstance(raw_viz, str):
                        try:
                            viz_config = json.loads(raw_viz)
                        except Exception:
                            viz_config = {}
                    elif isinstance(raw_viz, dict):
                        viz_config = raw_viz or {}

                    viz_status = sess.get("visualization_status")
            except Exception as e:
                logger.warning(
                    "[db-sample] Failed to load visualization from session: %s", e
                )

            out.dataset_id = dsid
            out.session_id = sid
            out.thread_id = tid

            out.visualization_config = viz_config
            out.visualization_status = viz_status


            out.messages.append(_assistant_entry("Query executed. Sample materialized for preview & chat."))
        else:
            out.messages.append(_assistant_entry("Query executed successfully (no rows)."))

    else:
        out.messages.append(_assistant_entry(f"{tool_name} executed successfully."))

    return out


# -------------------------------------------------------------------
# The language models the agents use, and the fallback chain behind them
# (owner ruling 2026-11-02: the model screen). No credentials are returned.
# -------------------------------------------------------------------
@app.get("/api/llm-models")
async def list_llm_models(request: Request):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    from app.core.inference import describe_llm_setup

    return describe_llm_setup()


# -------------------------------------------------------------------
# Trained models, registry and serving endpoints: app/api/models_routes.py
# (mounted next to the Ray jobs router, see include_router above).
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# DB sample helpers
# -------------------------------------------------------------------


def _rows_list_to_dicts(
    columns: List[str],
    rows: List[List[Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = {
            columns[i]: (r[i] if i < len(r) else None)
            for i in range(len(columns))
        }
        out.append(d)
    return out


async def _create_or_update_db_sample_session(
    request: Request,
    user_id: str,
    customer_id: str,
    columns: List[str],
    rows: List[List[Any]],
    preferred_dataset_id: Optional[str] = None,
    preferred_session_id: Optional[str] = None,
    preferred_thread_id: Optional[str] = None,
    source_table: Optional[str] = None,
    db_sample: Optional[DbSample] = None,
) -> Tuple[str, str, str]:
    sample_rows = rows[:MAX_DB_SAMPLE_ROWS]
    dict_rows = _rows_list_to_dicts(columns, sample_rows)

    sid = preferred_session_id
    sess = await get_session(sid) if sid else None
    if sess and sess.get("user_id") != user_id:
        sid = None
        sess = None
    if not sess and preferred_dataset_id:
        sid = await find_session_by_dataset_for_user(
            preferred_dataset_id, user_id
        )
        sess = await get_session(sid) if sid else None

    dsid = (
        preferred_dataset_id
        or (sess.get("dataset_id") if sess else None)
        or _new_id()
    )

    if sess and sess.get("work_dir") and Path(sess["work_dir"]).exists():
        work_dir = Path(sess["work_dir"])
        session_id = sid
    else:
        session_id = sid or _new_id()
        work_dir = (TMP_ROOT / session_id).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

    sample_input_path = work_dir / "sample_input.csv"
    _write_rows_to_csv(dict_rows, sample_input_path)

    output_path = work_dir / "output.csv"

    schema_dict = _schema_from_db_rows(columns, dict_rows)
    ddl = _ddl_from_schema(dsid, schema_dict)

    # ---- Build visualization config for DB sample ----
    try:
        visualization_config = build_visualization_config_from_sample(
            dataset_id=dsid,
            sample_rows=dict_rows,
            schema=schema_dict,
            task_type="unsupervised",
            target_column=None,
        )
        visualization_status = visualization_config.get("visualization_status", "ready")
    except Exception as e:
        logger.warning("[db-sample] Failed to build visualization_config: %s", e)
        visualization_config = {}
        visualization_status = "error"

    session_data = {
        **(sess or {}),
        "user_id": user_id,
        "dataset_id": dsid,
        "work_dir": str(work_dir),
        "data_source_location": f"db://{customer_id}/{dsid}?sample=true",
        # Persist the source database explicitly so a DTA transfer can use it as the
        # implicit source without having to re-parse the db:// URI (see the source
        # bridge in send_message), for this thread only.
        "customer_id": customer_id,
        # The table the user analyzed (parsed from their query), so a DTA transfer
        # uses it as the implicit source instead of the auto-detected first table.
        "source_table": source_table,
        # What these rows are relative to the query's whole result (the answer
        # discloses a partial sample as "N of M rows").
        "db_sample": db_sample.as_session() if db_sample else None,
        "created_at": _now_iso(),
        "object_name": None,
        "work_local_input": None,
        "output_location": _posix(output_path),
        "sample_local_input": _posix(sample_input_path),
        "uploaded_csv_preview": _jsonify(dict_rows),
        "uploaded_csv_columns": columns,
        "schema": _jsonify(schema_dict),
        "ddl_schema": ddl,
        "assistant_id": ASSISTANT_ID,
        "input_data_type": "db",
        "visualization_config": _jsonify(visualization_config),
        "visualization_status": visualization_status,
    }

    if not await save_session(session_id, session_data):
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to persist DB sample session",
        )

    thread_id = ""
    try:
        saved_tid = preferred_thread_id or (
            sess.get("thread_id") if sess else None
        )
        if saved_tid:
            thread_id = saved_tid
        else:
            payload = {
                "metadata": {
                    "source": "db-sample",
                    "assistant_id": ASSISTANT_ID,
                    "user_id": user_id,
                    "dataset_id": dsid,
                    "session_id": session_id,
                    "schema": schema_dict,
                }
            }
            thread_res = await lg_json("POST", "/threads", json=payload)
            thread_id = (
                thread_res.get("thread_id") or thread_res.get("id") or ""
            )

        if thread_id:
            await bind_thread_session(thread_id, session_id)
            s = (await get_session(session_id)) or {}
            s["thread_id"] = thread_id
            await save_session(session_id, s)
    except HTTPException as e:
        logger.warning(
            "[db-sample] upstream /threads create failed (soft): %s", e.detail
        )

    if not thread_id:
        thread_id = f"ui:{session_id}"
        await bind_thread_session(thread_id, session_id)
        saved = (await get_session(session_id)) or {}
        saved["thread_id"] = thread_id
        if not await save_session(session_id, saved):
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Failed to persist DB sample thread binding",
            )

    return dsid, session_id, thread_id


# ─────────────────────────────────────────────────────────────────────────────
# WBS 0.6 v3.0 — Asset Persistence: Historical Retrieval Endpoints
# ─────────────────────────────────────────────────────────────────────────────

def _build_asset_history(sess: dict, list_key: str) -> List[AssetEntry]:
    """Parse the per-type asset history list from the session."""
    raw = sess.get(list_key) or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    return [
        AssetEntry(object_key=e["object_key"], prompt_ts=e["prompt_ts"])
        for e in raw
        if isinstance(e, dict) and e.get("object_key")
    ]


@app.get("/api/assets/{session_id}", response_model=AssetResponse, tags=["assets"])
async def get_session_assets(request: Request, session_id: str):
    """
    Return the full asset manifest for a session: signed download URLs for
    most-recent code / output / viz, full version history, and Git branch info.
    Frontend uses this on page-load/refresh to restore code editor + output table.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    sess = await get_session(session_id)
    if not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    code_key = sess.get("gcs_code_object_key")
    out_key = sess.get("gcs_output_object_key")
    viz_key = sess.get("gcs_viz_object_key")
    git_branch = sess.get("git_job_branch")

    conn_id = sess.get("connection_id")
    storage_uri = sess.get("data_source_location")

    code_url, out_url, viz_url = await asyncio.gather(
        _generate_asset_signed_url(code_key, conn_id, storage_uri) if code_key else asyncio.sleep(0),
        _generate_asset_signed_url(out_key, conn_id, storage_uri) if out_key else asyncio.sleep(0),
        _generate_asset_signed_url(viz_key, conn_id, storage_uri) if viz_key else asyncio.sleep(0),
    )

    return AssetResponse(
        session_id=session_id,
        dataset_id=sess.get("dataset_id"),
        persist_errors=sess.get("persist_errors") or [],
        code_asset=CodeAsset(
            available=bool(code_key),
            object_key=code_key,
            signed_url=code_url or None,
            history=_build_asset_history(sess, "code_assets"),
        ),
        output_asset=OutputAsset(
            available=bool(out_key),
            object_key=out_key,
            signed_url=out_url or None,
            history=_build_asset_history(sess, "output_assets"),
        ),
        viz_asset=VizAsset(
            available=bool(viz_key),
            object_key=viz_key,
            signed_url=viz_url or None,
            history=_build_asset_history(sess, "viz_assets"),
        ),
        job_asset=JobAsset(
            available=bool(git_branch),
            git_branch=git_branch,
            git_repo=os.getenv("GITHUB_JOB_REGISTRY_REPO", "avaloka/avaloka-job-registry"),
            github_url=(
                f"https://github.com/"
                f"{os.getenv('GITHUB_JOB_REGISTRY_REPO', 'avaloka/avaloka-job-registry')}"
                f"/tree/{git_branch}/jobs/{sess.get('user_id','')}/{session_id}"
            ) if git_branch else None,
        ),
    )


@app.get("/api/assets/{session_id}/code", tags=["assets"])
async def get_asset_code_url(
    request: Request,
    session_id: str,
    prompt_ts: Optional[str] = Query(None, description="Fetch a specific version by prompt_ts"),
):
    """
    Return a fresh 15-min signed URL for the generated .py script.
    Optionally pass ?prompt_ts=20260407_143022 to get a specific version.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    sess = await get_session(session_id)
    if not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    conn_id = sess.get("connection_id")
    storage_uri = sess.get("data_source_location")

    if prompt_ts:
        history = _build_asset_history(sess, "code_assets")
        entry = next((e for e in history if e.prompt_ts == prompt_ts), None)
        if not entry:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No code asset for prompt_ts={prompt_ts}")
        key = entry.object_key
    else:
        key = sess.get("gcs_code_object_key")

    if not key:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No generated code persisted for this session")

    url = await _generate_asset_signed_url(key, conn_id, storage_uri)
    if not url:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not generate signed URL")

    return {"available": True, "signed_url": url, "object_key": key}


@app.get("/api/assets/{session_id}/output", tags=["assets"])
async def get_asset_output_url(
    request: Request,
    session_id: str,
    prompt_ts: Optional[str] = Query(None, description="Fetch a specific version by prompt_ts"),
):
    """Return a fresh 15-min signed URL for the execution output CSV."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    sess = await get_session(session_id)
    if not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    conn_id = sess.get("connection_id")
    storage_uri = sess.get("data_source_location")

    if prompt_ts:
        history = _build_asset_history(sess, "output_assets")
        entry = next((e for e in history if e.prompt_ts == prompt_ts), None)
        if not entry:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No output asset for prompt_ts={prompt_ts}")
        key = entry.object_key
    else:
        key = sess.get("gcs_output_object_key")

    if not key:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No execution output persisted for this session")

    url = await _generate_asset_signed_url(key, conn_id, storage_uri)
    if not url:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not generate signed URL")

    return {"available": True, "signed_url": url, "object_key": key}


@app.get("/api/assets/{session_id}/job", tags=["assets"])
async def get_asset_job_definition(request: Request, session_id: str):
    """Return the Git repo branch + direct GitHub URL for the job definition."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    sess = await get_session(session_id)
    if not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    git_branch = sess.get("git_job_branch")
    if not git_branch:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No job definition in Git for this session (requires successful execution + planner save_job_to_git call)",
        )

    git_repo = os.getenv("GITHUB_JOB_REGISTRY_REPO", "avaloka/avaloka-job-registry")
    uid = sess.get("user_id", "")
    dataset_id = sess.get("dataset_id", "")

    return {
        "available": True,
        "git_branch": git_branch,
        "git_repo": git_repo,
        "github_url": f"https://github.com/{git_repo}/tree/{git_branch}/jobs/{uid}/{session_id}",
    }
 

_REWRITE_STYLES: Dict[str, str] = {
    "regenerate": "Regenerate the insights from scratch with a fresh angle; the user was not satisfied with the previous version.",
    "shorter":    "Rewrite more concisely — fewer words, only the most important points.",
    "detailed":   "Expand with more depth, supporting numbers, and context.",
    "business":   "Reframe for a business/executive audience — impact, trends, decisions; no jargon.",
    "technical":  "Reframe for a technical audience — statistical detail, methodology, column-level specifics.",
}






from app.agents.sampling_persistence import get_supabase_client
# auth.uid() -> profiles.id. Stable per account, so cache it.
_profile_id_cache: Dict[str, str] = {}


async def _resolve_profile_id(user_id: str) -> Optional[str]:
    """Map a JWT sub (auth.uid()) to its profiles.id.

    Ownership columns store profiles.id, NOT auth.uid() — every RLS policy on
    analyses reads:
        owner_id = (SELECT profiles.id FROM profiles WHERE profiles.user_id = auth.uid())
    The JWT only carries auth.uid(), so it must be translated before comparing.
    """
    cached = _profile_id_cache.get(user_id)
    if cached:
        return cached

    def _q():
        client = get_supabase_client()
        res = (
            client.table("profiles")
            .select("id")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        data = getattr(res, "data", None) or []
        return data[0]["id"] if data else None

    try:
        profile_id = await asyncio.to_thread(_q)
    except Exception as exc:
        logger.error("[profile-resolve] lookup failed for %s: %s", user_id, exc, exc_info=True)
        return None

    if profile_id:
        _profile_id_cache[user_id] = str(profile_id)
    else:
        logger.warning("[profile-resolve] no profiles row for auth.uid()=%s", user_id)
    return profile_id

async def _is_analysis_collaborator(
    analysis_id: Optional[str],
    profile_id: Optional[str],          # unused, kept for signature stability
    project_id: Optional[str],
    user_id: Optional[str] = None,      # auth.uid(), needed to resolve app_users.id
) -> bool:
    """
    True if the requester is a collaborator on this ANALYSIS or on the PROJECT it
    belongs to. The Share panel writes resource_type='analysis' rows for a shared
    analysis and resource_type='project' rows for a shared project (the projects
    RLS policy "collaborators read projects via analyses" relies on the former).
    Both key on app_users.id, so translate auth.uid() -> app_users.id first.
    """
    if not user_id or not (analysis_id or project_id):
        return False

    app_user_id = await _resolve_app_user_id(user_id)
    if not app_user_id:
        return False

    targets = [(t, rid) for t, rid in (("analysis", analysis_id), ("project", project_id)) if rid]

    def _q() -> bool:
        client = get_supabase_client()
        for resource_type, resource_id in targets:
            try:
                res = (client.table("resource_collaborators")
                       .select("id")
                       .eq("resource_type", resource_type)
                       .eq("resource_id", str(resource_id))
                       .eq("app_user_id", str(app_user_id))
                       .limit(1).execute())
            except Exception as exc:
                logger.warning("[collab] resource_collaborators check failed (%s=%s): %s",
                               resource_type, resource_id, exc)
                continue
            if getattr(res, "data", None):
                return True
        return False

    try:
        return await asyncio.to_thread(_q)
    except Exception as exc:
        logger.error("[collab] authorization check errored: %s", exc, exc_info=True)
        return False


async def _resolve_shared_session_for_thread(
    thread_id: str,
    user_id: str,
    analysis_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve a chat thread to its owning analysis' session when the requester is
    the OWNER or an authorized COLLABORATOR / project member.

    Redis sessions are keyed by the owner's auth.uid(); a collaborator has none,
    so header/cookie/dataset resolution all miss. The analyses row stores the
    canonical session_id + thread_id, so look it up by thread_id (or ?aid=),
    authorize, and hand back the owner's session_id for a shared-thread rehydrate.
    """
    def _q():
        client = get_supabase_client()
        q = client.table("analyses").select("id,owner_id,session_id,thread_id,project_id")
        q = q.eq("id", analysis_id) if analysis_id else q.eq("thread_id", thread_id)
        res = q.limit(1).execute()
        data = getattr(res, "data", None) or []
        return data[0] if data else None

    try:
        row = await asyncio.to_thread(_q)
    except Exception as exc:
        logger.error("[shared-session] analyses lookup failed (thread=%s aid=%s): %s",
                     thread_id, analysis_id, exc, exc_info=True)
        return None

    if not row:
        return None

    owner_id = str(row.get("owner_id") or "")

    profile_id = await _resolve_profile_id(user_id)
    requester_plan = await asyncio.to_thread(resolve_plan, user_id)
    if requester_plan == "enterprise" and not row.get("session_id"):
        return None
    if profile_id and requester_plan != "enterprise":
        from app.services.analysis_share_access import standalone_analysis_access

        try:
            access_level = await asyncio.to_thread(
                standalone_analysis_access, get_supabase_client(), row["id"], profile_id, requester_plan
            )
        except Exception as exc:
            logger.error("[shared-session] analysis grant lookup failed: %s", exc, exc_info=True)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Analysis sharing lookup is temporarily unavailable.") from exc
        if access_level:
            return {**row, "role": "collaborator", "share_access_level": access_level}


    # analyses.owner_id is not consistently populated: some rows store the
    # owner's profiles.id, others store the raw auth.uid() (the value the JWT
    # carries directly). Accept an owner match against EITHER id space so a
    # user reopening their own analysis is never misclassified as "not shared".
    if owner_id and owner_id == str(user_id):
        return {**row, "role": "owner"}

    if profile_id and owner_id == str(profile_id):
        return {**row, "role": "owner"}

    # Also cover owner_id stored as the app_users.id for this account.
    app_user_id = await _resolve_app_user_id(user_id)
    if app_user_id and owner_id == str(app_user_id):
        return {**row, "role": "owner"}


    if profile_id and await _is_analysis_collaborator(
        row["id"], profile_id, row.get("project_id"), user_id=user_id
    ):
        return {**row, "role": "collaborator"}

    logger.warning(
        "[shared-session] aid=%s owner_id=%s not matched to requester "
        "auth.uid()=%s profiles.id=%s app_users.id=%s",
        row.get("id"), owner_id, user_id, profile_id, app_user_id,
    )
    return None


# profiles.id -> auth.uid(). The rebuilt session's user_id must be the OWNER's
# auth.uid() so the owner-scoped guards in send_message still pass. Stable, cache it.
_auth_uid_by_profile_cache: Dict[str, str] = {}


async def _resolve_auth_uid_from_profile_id(profile_id: Optional[str]) -> Optional[str]:
    if not profile_id:
        return None
    cached = _auth_uid_by_profile_cache.get(str(profile_id))
    if cached:
        return cached

    def _q():
        client = get_supabase_client()
        res = (client.table("profiles").select("user_id")
               .eq("id", profile_id).limit(1).execute())
        data = getattr(res, "data", None) or []
        return data[0]["user_id"] if data else None

    try:
        uid = await asyncio.to_thread(_q)
    except Exception as exc:
        logger.error("[rehydrate] profiles reverse-lookup failed for %s: %s",
                     profile_id, exc, exc_info=True)
        return None
    if uid:
        _auth_uid_by_profile_cache[str(profile_id)] = str(uid)
    return uid


async def _rehydrate_session_from_analysis(
    shared: Dict[str, Any],
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """Rebuild an evicted Redis session from its durable analyses row.

    Redis is a TTL cache; the analyses row is durable. On a cache miss (week-old
    dataset) reconstruct the session from the row and write it back so later turns
    hit the cache. Prefers a full session_snapshot (cloud source + object_name +
    columns) when one was persisted; otherwise rebuilds sample fidelity from the
    row's schema + samples.
    """
    analysis_id = shared.get("id")
    if not analysis_id:
        return None

    def _q():
        client = get_supabase_client()
        res = (client.table("analyses")
               .select("id,owner_id,session_id,thread_id,project_id,"
                       "dataset_id,schema,samples,session_snapshot")
               .eq("id", analysis_id).limit(1).execute())
        data = getattr(res, "data", None) or []
        return data[0] if data else None

    try:
        row = await asyncio.to_thread(_q)
    except Exception as exc:
        logger.error("[rehydrate] analyses lookup failed for %s: %s",
                     analysis_id, exc, exc_info=True)
        return None
    if not row:
        return None

    owner_user_id = await _resolve_auth_uid_from_profile_id(row.get("owner_id"))
    if not owner_user_id:
        logger.warning("[rehydrate] no owner auth.uid() for analysis %s", analysis_id)
        return None

    snapshot = row.get("session_snapshot")
    if isinstance(snapshot, str):
        snapshot = _maybe_json_load(snapshot)

    dsid = row.get("dataset_id") or (snapshot or {}).get("dataset_id") or _new_id()
    schema_any = _maybe_json_load(row.get("schema")) or row.get("schema") or {}
    samples = _maybe_json_load(row.get("samples")) or row.get("samples") or []
    if not isinstance(samples, list):
        samples = []

    work_dir = (TMP_ROOT / session_id).resolve()
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.warning("[rehydrate] could not create work_dir %s", work_dir, exc_info=True)

    sample_input_path = work_dir / "sample_input.csv"
    if samples:
        try:
            _write_rows_to_csv(samples, sample_input_path)
        except Exception:
            logger.warning("[rehydrate] failed to rebuild sample CSV at %s",
                           sample_input_path, exc_info=True)

    cols = list(schema_any.keys()) if isinstance(schema_any, dict) else list(schema_any)
    if not cols and samples and isinstance(samples[0], dict):
        cols = list(samples[0].keys())
    ddl = _ddl_from_schema(dsid, {c: "string" for c in cols}) if cols else ""

    sess: Dict[str, Any] = {
        "user_id": owner_user_id,
        "dataset_id": dsid,
        "created_at": _now_iso(),
        "work_dir": str(work_dir),
        "assistant_id": ASSISTANT_ID,
        "thread_id": shared.get("thread_id"),
        "schema": _jsonify(schema_any),
        "ddl_schema": ddl,
        "uploaded_csv_columns": cols,
        "uploaded_csv_preview": _jsonify(samples),
        "sample_local_input": _posix(sample_input_path) if samples else None,
        "output_location": _posix(work_dir / "output.csv"),
        "input_data_type": "csv",
        "rehydrated_from_analysis": True,
    }

    # Full snapshot restores cloud source + object_name so "run on the entire
    # dataset" re-downloads the real file instead of silently using the sample.
    if isinstance(snapshot, dict) and snapshot:
        for k in (
            *DATASET_SESSION_SNAPSHOT_KEYS,
            "connection_id", "object_name", "source_kind", "input_data_type",
            "file_size_bytes", "file_size_mb", "data_source_location",
            "folder_read_path", "folder_table_type", "iceberg_metadata_uri",
            "hive_partitioning",
        ):
            v = snapshot.get(k)
            if v is not None:
                sess[k] = v
        # snapshot's work_local_input points at a wiped tmp path; drop it so
        # _ensure_local_inputs re-downloads from data_source_location on demand.
        sess.pop("work_local_input", None)

    if await save_session(session_id, sess):
        logger.info("[rehydrate] rebuilt session %s from analysis %s (snapshot=%s, %d rows)",
                    session_id, analysis_id, bool(snapshot), len(samples))
    else:
        logger.warning("[rehydrate] failed to persist rebuilt session %s", session_id)
    return sess


_SESSION_SNAPSHOT_KEYS = tuple(dict.fromkeys((
    *DATASET_SESSION_SNAPSHOT_KEYS,
    "dataset_id", "connection_id", "object_name", "source_kind",
    "input_data_type", "file_size_bytes", "file_size_mb",
    "data_source_location", "folder_read_path", "folder_table_type",
    "iceberg_metadata_uri", "hive_partitioning",
    "uploaded_csv_columns", "uploaded_csv_preview", "schema", "ddl_schema",
    # multi-file group index so a cold reopen restores ALL members, not one
    "dataset_ids", "dataset_session_map", "group_session_id", "group_datasets",
    "excluded_dataset_ids",
)))


_GROUP_MEMBER_KEYS = (
    "filename", "alias", "schema", "ddl_schema",
    "uploaded_csv_columns", "uploaded_csv_preview", "input_data_type",
    "sheet_name", "source_workbook", "data_source_location", "object_name",
    "source_kind", "connection_id", "file_size_bytes", "file_size_mb",
    "visualization_config", "visualization_status",
)


def _group_members_snapshot(dataset_sessions) -> list:
    """Per-member payload so a cold reopen can rebuild every sibling session."""
    out = []
    for dsid, ds_sid, ds_sess in dataset_sessions:
        member = {k: ds_sess.get(k) for k in _GROUP_MEMBER_KEYS if ds_sess.get(k) is not None}
        member["dataset_id"] = dsid
        member["session_id"] = ds_sid
        out.append(member)
    return out

async def persist_session_snapshot(thread_id: str, sess: Dict[str, Any]) -> int:
    """Write a durable snapshot of a LIVE session onto its analyses row.

    Returns the number of analyses rows updated (0 when the frontend hasn't
    created the row yet), so the caller marks it persisted only on a real hit
    instead of skipping forever on a first turn that predates the row.
    """
    snapshot = _jsonify({
        k: sess.get(k) for k in _SESSION_SNAPSHOT_KEYS if sess.get(k) is not None
    })
    if not snapshot:
        return 0

    def _update() -> int:
        client = get_supabase_client()
        res = (client.table("analyses")
               .update({"session_snapshot": snapshot})
               .eq("thread_id", thread_id).execute())
        return len(getattr(res, "data", None) or [])

    try:
        return await asyncio.to_thread(_update)
    except Exception as exc:
        from app.agents.sampling_persistence import log_supabase_skip
        log_supabase_skip(logger, f"[snapshot] persist failed for thread {thread_id}", exc)
        return 0

def _is_supabase_auth_failure(exc: BaseException) -> bool:
    """True when Supabase refused our credentials rather than our query.

    Worth separating because the operator action is completely different: a
    rejected key is a deployment fix, anything else is a retry. postgrest
    surfaces this as an APIError carrying code 401/403; a missing env var
    arrives as the RuntimeError from get_supabase_client().
    """
    code = getattr(exc, "code", None)
    try:
        if code is not None and int(code) in (401, 403):
            return True
    except (TypeError, ValueError):
        pass
    text = f"{exc}".lower()
    return ("invalid api key" in text
            or "supabase env vars not set" in text
            or "jwt" in text and "invalid" in text)


def _is_supabase_unconfigured(exc: BaseException) -> bool:
    """True when the Supabase credentials are unset, so nothing was sent.

    _is_supabase_auth_failure() counts this as a credential problem too (the
    fix is in the deployment either way), but the message must not say the
    credentials were rejected: none were offered.
    """
    from app.agents.sampling_persistence import SupabaseNotConfigured
    return (isinstance(exc, SupabaseNotConfigured)
            or "supabase env vars not set" in f"{exc}".lower())


async def _resolve_session_from_aid(analysis_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Resolve a Supabase analysis_id (aid) to its backend session.
    Returns {owner_id, session_id, thread_id} if found AND owned by user_id, else None.
    Service-role bypasses RLS, so we check ownership in code.
    """
    def _q():
        client = get_supabase_client()
        res = (
            client.table("analyses")
            .select("owner_id,session_id,thread_id")
            .eq("id", analysis_id)
            .limit(1)
            .execute()
        )
        data = getattr(res, "data", None) or []
        return data[0] if data else None

    try:
        row = await asyncio.to_thread(_q)
    except Exception as exc:
        # Returning None here is a lie. Every caller reads None as "no such
        # analysis" and answers 404, so a Supabase credential or connectivity
        # failure was reported to the client as `404 Analysis not found` -- an
        # answer about the DATA when the truth is that we never got to look.
        # That is what sent an "Invalid API key" 401 to be debugged as a
        # missing record.
        #
        # None still means "queried successfully, nothing there". A lookup that
        # could not run raises, and 503 says the honest thing: unknown, try again.
        detail = "Analysis lookup is temporarily unavailable."
        if _is_supabase_unconfigured(exc):
            # Unset is not rejected: nothing was sent, so there is no key to
            # check against the project -- the variables need setting.
            logger.error(
                "[aid-resolve] Supabase is not configured while resolving %s: %s "
                "(values are read from the RUNNING process's environment).",
                analysis_id, exc,
            )
            detail = ("Analysis lookup is unavailable: the backend's Supabase "
                      "credentials are not set (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY).")
        elif _is_supabase_auth_failure(exc):
            logger.error(
                "[aid-resolve] Supabase REJECTED the service-role credentials while "
                "resolving %s. Check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY: the "
                "key must belong to the same project as the URL, be the service_role "
                "key (not anon), and not have been rotated. NOTE: the values are read "
                "from the environment of the RUNNING process -- editing .env does "
                "nothing until the server is restarted. Underlying error: %s",
                analysis_id, exc,
            )
            detail = ("Analysis lookup is unavailable: the backend's Supabase "
                      "credentials were rejected.")
        else:
            logger.error("[aid-resolve] lookup failed for %s: %s",
                         analysis_id, exc, exc_info=True)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail) from exc

    if not row:
        return None

    # analyses.owner_id may hold the owner's profiles.id OR the raw auth.uid()
    # (or, on some rows, the app_users.id). Accept a match against any of the
    # three so a user's own analysis is never misread as belonging to someone else.
    owner_id = str(row.get("owner_id") or "")
    if owner_id and owner_id == str(user_id):
        return row

    profile_id = await _resolve_profile_id(user_id)
    if profile_id and owner_id == str(profile_id):
        return row

    app_user_id = await _resolve_app_user_id(user_id)
    if app_user_id and owner_id == str(app_user_id):
        return row

    logger.warning(
        "[aid-resolve] aid=%s owner_id=%s not matched to requester "
        "auth.uid()=%s profiles.id=%s app_users.id=%s",
        analysis_id, owner_id, user_id, profile_id, app_user_id,
    )
    return None

async def _fetch_code_text(object_key: str, sess: Dict[str, Any]) -> Optional[str]:
    """Fetch stored .py text from GCS via signed URL (server-side, same as /code)."""
    url = await _generate_asset_signed_url(
        object_key, sess.get("connection_id"), sess.get("data_source_location")
    )
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url)
            return r.text if r.status_code == 200 else None
    except Exception as exc:
        logger.warning("[refresh] failed to fetch code text for %s: %s", object_key, exc)
        return None

@app.get("/analysis/{analysis_id}/code", tags=["insights"])
async def get_analysis_code(
    request: Request,
    analysis_id: str,
    insight_id: Optional[str] = Query(None),
    prompt_ts: Optional[str] = Query(None, description="Fetch a specific version"),
):
    """
    Return the generated Python code for an analysis so the UI can open it
    in the code editor. Resolves aid -> session_id, then reuses the existing
    asset-persistence signed-URL logic.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    row = await _resolve_session_from_aid(analysis_id, user_id)
    if not row or not row.get("session_id"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found")

    session_id = row["session_id"]
    sess = await get_session(session_id)
    if not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found for this analysis")

    conn_id = sess.get("connection_id")
    storage_uri = sess.get("data_source_location")

    if prompt_ts:
        history = _build_asset_history(sess, "code_assets")
        entry = next((e for e in history if e.prompt_ts == prompt_ts), None)
        if not entry:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No code asset for prompt_ts={prompt_ts}")
        key = entry.object_key
    else:
        key = sess.get("gcs_code_object_key")

    if not key:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No generated code persisted for this analysis")

    url = await _generate_asset_signed_url(key, conn_id, storage_uri)
    if not url:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not generate signed URL")

    # Fetch the file server-side so the browser never makes a cross-origin
    # call to GCS (avoids needing bucket CORS config).
    code_text = None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url)
            if r.status_code == 200:
                code_text = r.text
            else:
                logger.warning("[code] signed-url fetch returned %s for %s", r.status_code, key)
    except Exception as exc:
        logger.warning("[code] failed to fetch code text for %s: %s", key, exc)

    return {
        "available": True,
        "analysis_id": analysis_id,
        "insight_id": insight_id,
        "code": code_text,        # inline text — UI shows this directly
        "signed_url": url,        # kept for an optional download button
        "object_key": key,
    }


@app.post("/analysis/{analysis_id}/feedback", response_model=InsightFeedbackOut, tags=["insights"])
async def submit_analysis_feedback(request: Request, analysis_id: str, body: InsightFeedbackIn):
    """
    Persist thumbs up/down + optional comment for an insight.
    Writes to analysis_feedback with owner_id from the JWT (service-role bypasses RLS,
    so ownership is set + checked in code).
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    if body.feedback_type not in ("positive", "negative"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "feedback_type must be 'positive' or 'negative'")

    # verify the analysis exists and belongs to this user
    row = await _resolve_session_from_aid(analysis_id, user_id)
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found")
    profile_id = await _resolve_profile_id(user_id)
    if not profile_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "No profile for this account")
    def _insert():
        client = get_supabase_client()
        res = (
            client.table("analysis_feedback")
            .insert({
                "analysis_id":   analysis_id,
                "insight_id":    body.insight_id,
                "feedback_type": body.feedback_type,
                "comment":       body.comment,
                "owner_id":      profile_id,
            })
            .execute()
        )
        data = getattr(res, "data", None) or []
        return data[0] if data else None

    try:
        inserted = await asyncio.to_thread(_insert)
    except Exception as exc:
        logger.error("[feedback] insert failed for aid=%s: %s", analysis_id, exc, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to save feedback")

    if not inserted:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Feedback insert returned no row")

    return InsightFeedbackOut(
        id=str(inserted["id"]),
        analysis_id=analysis_id,
        insight_id=inserted.get("insight_id"),
        feedback_type=inserted["feedback_type"],
        comment=inserted.get("comment"),
        created_at=str(inserted.get("created_at") or _now_iso()),
    )


# auth.uid() -> app_users.id. resource_collaborators.app_user_id references
# app_users(id), NOT profiles.id or auth.uid(), so sharing checks must translate
# into this id space. Stable per account, so cache it.
_app_user_id_cache: Dict[str, str] = {}


async def _resolve_app_user_id(user_id: str) -> Optional[str]:
    """Map a JWT sub (auth.uid()) to its app_users.id.

    The sharing tables (resource_collaborators.app_user_id) key on app_users.id.
    app_users links back to auth via app_users.auth_user_id, so translate the
    requester's auth.uid() through it before comparing against a share row.
    """
    cached = _app_user_id_cache.get(user_id)
    if cached:
        return cached

    def _q():
        client = get_supabase_client()
        res = (
            client.table("app_users")
            .select("id")
            .eq("auth_user_id", user_id)
            .limit(1)
            .execute()
        )
        data = getattr(res, "data", None) or []
        return data[0]["id"] if data else None

    try:
        app_user_id = await asyncio.to_thread(_q)
    except Exception as exc:
        logger.error("[app-user-resolve] lookup failed for %s: %s", user_id, exc, exc_info=True)
        return None

    if app_user_id:
        _app_user_id_cache[user_id] = str(app_user_id)
    else:
        logger.warning("[app-user-resolve] no app_users row for auth.uid()=%s", user_id)
    return app_user_id


def _prompt_ts_to_iso(ts: str) -> Optional[str]:
    try:
        return (datetime.datetime
                .strptime(ts, "%Y%m%d_%H%M%S")
                .replace(tzinfo=datetime.timezone.utc)
                .isoformat())
    except Exception:
        return None

@app.post("/analysis/{analysis_id}/refresh", tags=["insights"])
async def refresh_analysis(
    request: Request,
    analysis_id: str,
    prompt_ts: Optional[str] = Query(None, description="Re-run a specific version; default = latest"),
):
    """
    Re-execute the latest stored code for an analysis (refresh always executes
    the latest code). Runs the stored .py directly through the execution
    node, then fires _persist_assets_background so the run is versioned for rollback.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    row = await _resolve_session_from_aid(analysis_id, user_id)
    if not row or not row.get("session_id"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found")

    session_id = row["session_id"]
    sess = await get_session(session_id)
    if not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found for this analysis")

    # ---- resolve the code object key (latest, or a specific version) ----
    if prompt_ts:
        history = _build_asset_history(sess, "code_assets")
        entry = next((e for e in history if e.prompt_ts == prompt_ts), None)
        if not entry:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No code asset for prompt_ts={prompt_ts}")
        code_key = entry.object_key
    else:
        code_key = sess.get("gcs_code_object_key")
    if not code_key:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No code to refresh for this analysis")

    latest_code = await _fetch_code_text(code_key, sess)
    if not latest_code:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not load stored code")

    # ---- build the execution state (seed coder_definition.code) ----
    local_input = (
        (sess.get("work_local_input") and _posix(sess["work_local_input"]))
        or (sess.get("sample_local_input") and _posix(sess["sample_local_input"]))
    )
    output_location = sess.get("output_location") and _posix(sess["output_location"])
    schema_for_state = _normalize_schema_for_state(
        _maybe_json_load(sess.get("schema")) or sess.get("schema")
    )

    # decide local vs k8s-ray, mirroring send_message's fidelity policy
    file_size = int(sess.get("file_size_bytes") or 0)
    cloud_uri = _full_cloud_uri(sess)
    conn_id = sess.get("connection_id")
    fidelity = sess.get("analysis_fidelity")
    is_folder = bool(sess.get("folder_read_path"))
    use_ray = (
        bool(cloud_uri) and bool(conn_id)
        and fidelity in (FIDELITY_PORTFOLIO, FIDELITY_ENTIRE)
        and (file_size >= LARGE_DATASET_THRESHOLD_BYTES or is_folder)
    )

    state_in: Dict[str, Any] = {
        "user_id": user_id,
        "session_id": session_id,
        "dataset_id": sess.get("dataset_id"),
        "messages": [],
        "coder_definition": {"code": latest_code},
        "generated_code": latest_code,
        "schema": schema_for_state,
        "input_data_type": sess.get("input_data_type", "csv"),
        "folder_table_type": sess.get("folder_table_type"),
        "folder_read_path": sess.get("folder_read_path"),
        "hive_partitioning": bool(sess.get("hive_partitioning")),
        "iceberg_metadata_uri": sess.get("iceberg_metadata_uri"),
        "output_location": output_location,
        "connection_id": conn_id,
        "data_source_location": (cloud_uri if use_ray else local_input),
        "data_source_location_local": local_input,
        "data_source_location_cloud": cloud_uri,
        "execution_result": {},
        "output_file_data": None,
    }
    if use_ray:
        state_in.update({
            "execution_mode": "k8s-ray",
            "ray_namespace": os.getenv("RAY_NAMESPACE", "ray-training"),
            "ray_execution_profile": "batch_heavy",
        })

    node = execution_agent_node_ray if use_ray else execution_agent_node_local
    logger.info("[refresh] aid=%s session=%s mode=%s key=%s",
                analysis_id, session_id, "k8s-ray" if use_ray else "local", code_key)

    final = await _run_execution_node(node, state_in, use_ray)

    # ---- extract fresh output ----

    _exec_result = final.get("execution_result") or {}
    _exec_status = str(
        _exec_result.get("status", "")
    ).lower()

    # "completed" without a file is not a successful analysis result.
    _exec_ok = _exec_status in {
        "success",
        "succeeded",
        "dryrun",
    }

    _fresh_output = bool(
        final.get("fresh_output_produced")
        or _exec_result.get("fresh_output")
        or final.get("output_file_data")
    )

    if _exec_ok and _fresh_output:
        output_file_data, output_json = get_output_from_state(
            final,
            tmp_root=TMP_ROOT,
        )
    else:
        output_file_data = None
        output_json = None

    logger.info(
        "[save-and-execute] aid=%s executed_rows=%s "
        "fresh_output=%s output_file=%s",
        analysis_id,
        (
            len(output_json)
            if isinstance(output_json, list)
            else None
        ),
        _fresh_output,
        _exec_result.get("output_file"),
    )

    # ---- version this run (same persistence call send_message uses) ----
    old_output_key = sess.get("gcs_output_object_key")
    old_output_history_count = len(
        _build_asset_history(sess, "output_assets")
    )
    # ---- version this run (same persistence call send_message uses) ----
    _checkpoint_created = _is_new_checkpoint_worthy(sess, output_json, _exec_ok)
    if _checkpoint_created:
        try:
            _new_fingerprint = hashlib.sha256(
                json.dumps(output_json or [], sort_keys=True, default=str).encode()
            ).hexdigest()
            sess["last_output_fingerprint"] = _new_fingerprint
            await save_session(session_id, sess)

            await _persist_assets_background(
                user_id              = user_id,
                session_id           = session_id,
                dataset_id           = sess.get("dataset_id", ""),
                generated_code       = latest_code,
                planner_definition   = {"plan": sess.get("plan", "")},
                coder_definition     = {"code": latest_code},
                execution_result     = _exec_result,
                exec_succeeded       = _exec_ok,
                output_file_data     = output_file_data,
                output_json          = output_json,
                output_location      = output_location,
                visualization_config = _maybe_json_load(sess.get("visualization_config")) or {},
                connection_id        = conn_id,
                storage_uri          = sess.get("data_source_location"),
                dataset_label        = sess.get("alias") or sess.get("filename") or "",
                analysis_label       = (sess.get("version_prompts") or {}).get(prompt_ts) or "refresh"
            )
        except Exception as exc:
            logger.warning("[refresh] persistence failed (non-fatal): %s", exc)
            _checkpoint_created = False
    else:
        logger.info(
            "[refresh] Skipping checkpoint — output identical to last version or run failed (aid=%s session=%s)",
            analysis_id, session_id,
        )

  # re-read to surface the new version id for the rollback UI
    # fresh = await get_session(session_id) or sess
    # out_hist = _build_asset_history(fresh, "output_assets")
    # new_prompt_ts = out_hist[-1].prompt_ts if out_hist else None
    fresh = await get_session(session_id) or sess
    out_hist = _build_asset_history(fresh, "output_assets")

    latest_output = (
        max(out_hist, key=lambda entry: entry.prompt_ts)
        if out_hist
        else None
    )

    history_advanced = (
        len(out_hist) > old_output_history_count
    )

    new_output_key = fresh.get("gcs_output_object_key")
    pointer_advanced = bool(
        new_output_key
        and (
            new_output_key != old_output_key
            or history_advanced
        )
    )

    # Persistence may append history but fail to move the current pointer.
    # Repair it synchronously before returning to the frontend.
    if (
        history_advanced
        and latest_output
        and new_output_key != latest_output.object_key
    ):
        logger.warning(
            "[save-and-execute] repairing stale output pointer: "
            "old_current=%s latest_history=%s",
            new_output_key,
            latest_output.object_key,
        )

        fresh["gcs_output_object_key"] = latest_output.object_key

        if not await save_session(session_id, fresh):
            logger.error(
                "[save-and-execute] failed to repair output pointer "
                "for session=%s",
                session_id,
            )
        else:
            new_output_key = latest_output.object_key
            pointer_advanced = True

    _checkpoint_created = bool(
    _checkpoint_created
    and history_advanced
    and new_output_key
    )

    new_prompt_ts = (
        latest_output.prompt_ts
        if history_advanced and latest_output
        else None
    )

    return {
        "available": True,
        "analysis_id": analysis_id,
        "session_id": session_id,
        "status": _exec_result.get("status", "unknown"),
        "output_json": output_json,
        "output_file_data": output_file_data,
        "new_version_prompt_ts": new_prompt_ts,
        "code_object_key": fresh.get("gcs_code_object_key"),
        "output_object_key": fresh.get("gcs_output_object_key"),
        "checkpoint_created": _checkpoint_created,
        "output_row_count": (
            len(output_json)
            if isinstance(output_json, list)
            else None
        ),
    }

# -------------------------------------------------------------------
# Insights: Version listing + Restore (Checkpoint/Restore pattern)
# -------------------------------------------------------------------

class RestoreVersionIn(BaseModel):
    prompt_ts: str


def _merge_version_checkpoints(sess: dict) -> List[Dict[str, Any]]:
    """
    Merge code_assets / output_assets / viz_assets histories into one
    unified, time-ordered list of checkpoints keyed by prompt_ts.
    Each checkpoint may have any subset of {code, output, viz} available,
    since not every prompt necessarily produced all three artifact types.
    """
    code_hist = _build_asset_history(sess, "code_assets")
    output_hist = _build_asset_history(sess, "output_assets")
    viz_hist = _build_asset_history(sess, "viz_assets")

    by_ts: Dict[str, Dict[str, Any]] = {}

    for entry in code_hist:
        by_ts.setdefault(entry.prompt_ts, {"prompt_ts": entry.prompt_ts})["code_object_key"] = entry.object_key
    for entry in output_hist:
        by_ts.setdefault(entry.prompt_ts, {"prompt_ts": entry.prompt_ts})["output_object_key"] = entry.object_key
    for entry in viz_hist:
        by_ts.setdefault(entry.prompt_ts, {"prompt_ts": entry.prompt_ts})["viz_object_key"] = entry.object_key

    current_code_key = sess.get("gcs_code_object_key")
    current_output_key = sess.get("gcs_output_object_key")
    current_viz_key = sess.get("gcs_viz_object_key")

    prompt_labels = sess.get("version_prompts") or {}   # prompt_ts -> user prompt text

    checkpoints = []
    for ts, entry in by_ts.items():
        entry["is_current"] = (
            entry.get("code_object_key") == current_code_key
            and current_code_key is not None
        )
        entry["has_code"] = "code_object_key" in entry
        entry["has_output"] = "output_object_key" in entry
        entry["has_viz"] = "viz_object_key" in entry
        entry["prompt"] = prompt_labels.get(ts)         # NEW: the question this version answered
        entry["created_at"] = _prompt_ts_to_iso(ts)     # NEW: parseable timestamp for the UI
        checkpoints.append(entry)

    checkpoints.sort(key=lambda e: e["prompt_ts"], reverse=True)
    return checkpoints


@app.get("/analysis/{analysis_id}/versions", tags=["insights"])
async def list_analysis_versions(request: Request, analysis_id: str):
    """
    List saved checkpoints (versions) for an analysis, newest first, so the
    frontend can render a version history / rollback list (Lovable-style
    Checkpoint UI, per Poornima's suggestion).
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    row = await _resolve_session_from_aid(analysis_id, user_id)
    if not row or not row.get("session_id"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found")

    session_id = row["session_id"]
    sess = await get_session(session_id)
    if not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found for this analysis")

    checkpoints = _merge_version_checkpoints(sess)

    return {
        "analysis_id": analysis_id,
        "session_id": session_id,
        "current_prompt_ts": next(
            (c["prompt_ts"] for c in checkpoints if c["is_current"]), None
        ),
        "versions": checkpoints,
    }


@app.post("/analysis/{analysis_id}/restore", tags=["insights"])
async def restore_analysis_version(
    request: Request,
    analysis_id: str,
    body: RestoreVersionIn,
):
    """
    Set a chosen prior checkpoint as the CURRENT version (non-destructive —
    like a git checkout, not a delete). After restore:
      - gcs_code_object_key / gcs_output_object_key / gcs_viz_object_key
        point back at the restored checkpoint's artifacts.
      - History lists (code_assets/output_assets/viz_assets) are left intact,
        so nothing newer is lost — a subsequent /refresh will create a NEW
        checkpoint on top, same as normal flow.
      - The restored code text is fetched inline so the UI can immediately
        show it in the code editor without a second round-trip.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    row = await _resolve_session_from_aid(analysis_id, user_id)
    if not row or not row.get("session_id"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found")

    session_id = row["session_id"]
    sess = await get_session(session_id)
    if not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found for this analysis")

    checkpoints = _merge_version_checkpoints(sess)
    target = next((c for c in checkpoints if c["prompt_ts"] == body.prompt_ts), None)
    if not target:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No checkpoint found for prompt_ts={body.prompt_ts}",
        )

    # Repoint "current" pointers to the restored checkpoint's artifacts.
    # Only overwrite keys that this checkpoint actually has, so we don't
    # null out output/viz just because this particular prompt only touched code.
    changed = False
    if target.get("code_object_key"):
        sess["gcs_code_object_key"] = target["code_object_key"]
        changed = True
    if target.get("output_object_key"):
        sess["gcs_output_object_key"] = target["output_object_key"]
        changed = True
    if target.get("viz_object_key"):
        sess["gcs_viz_object_key"] = target["viz_object_key"]
        changed = True

    if changed:
        sess["restored_from_prompt_ts"] = body.prompt_ts
        sess["restored_at"] = _now_iso()
        ok = await save_session(session_id, sess)
        if not ok:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Failed to persist restore",
            )

    # Fetch restored code text inline for the editor (best-effort).
    restored_code_text = None
    if target.get("code_object_key"):
        restored_code_text = await _fetch_code_text(target["code_object_key"], sess)

    # Fresh signed URLs for output/viz so the UI can re-render them immediately.
    conn_id = sess.get("connection_id")
    storage_uri = sess.get("data_source_location")
    output_url = (
        await _generate_asset_signed_url(target["output_object_key"], conn_id, storage_uri)
        if target.get("output_object_key")
        else None
    )
    viz_url = (
        await _generate_asset_signed_url(target["viz_object_key"], conn_id, storage_uri)
        if target.get("viz_object_key")
        else None
    )

    logger.info(
        "[restore] analysis_id=%s session=%s restored prompt_ts=%s (code=%s output=%s viz=%s)",
        analysis_id, session_id, body.prompt_ts,
        bool(target.get("code_object_key")),
        bool(target.get("output_object_key")),
        bool(target.get("viz_object_key")),
    )

    return {
        "available": True,
        "analysis_id": analysis_id,
        "session_id": session_id,
        "restored_prompt_ts": body.prompt_ts,
        "code": restored_code_text,
        "code_object_key": target.get("code_object_key"),
        "output_object_key": target.get("output_object_key"),
        "output_signed_url": output_url,
        "viz_object_key": target.get("viz_object_key"),
        "viz_signed_url": viz_url,
    }



class SaveAndExecuteIn(BaseModel):
    code: str
    # Set false to skip the deterministic validators after the planner review
    # (advanced escape hatch); the planner review always runs, and so does the
    # NoDatabaseWrites safety rule.
    run_validation: Optional[bool] = True


#: What save-and-execute tells the user when a validation stage stops the edit.
_VALIDATION_STAGE_MESSAGES = {
    "syntax": "Your code has a Python syntax error, so it was not run. Check the lines you changed.",
    "static_semantic": "Your code didn't pass the safety and correctness checks, so it was not run.",
    "sample_execution": (
        "Your code stopped with an error when it was tried on a small sample of your data, "
        "so it was not run on the full data. Check the lines you changed."
    ),
}
_VALIDATION_FALLBACK = "Your code couldn't be checked, so it was not run."


def _plain_validation_feedback(stage: Optional[str], feedback: Any) -> str:
    """The validator's verdict on an edit, in words the user can be shown.

    The sample run's feedback is the exception text itself (str(e): data
    values, column names), so that stage always gets the plain sentence. The
    other stages' findings are code-written and are kept unless they carry a
    raw error. The raw feedback is in the server log.
    """
    plain = _VALIDATION_STAGE_MESSAGES.get(stage or "", _VALIDATION_FALLBACK)
    if stage == "sample_execution":
        return plain
    return plain_or(feedback, plain)


def _save_and_execute_failure_message(final: Dict[str, Any]) -> str:
    """What save-and-execute tells the user when the edited code did not run.

    The stderr traceback stays in the server log; the user hears the reason
    in her terms, the way a failed chat turn words it (analysis_outcome).
    """
    detail = failure_detail(final) if isinstance(final, dict) else ""
    if not detail:
        return ("Your code finished without producing a new result, so nothing was saved. "
                "The current version is unchanged.")
    return (f"Your code didn't finish: {plain_failure_reason(detail)}. Nothing was saved, "
            "and the current version is unchanged.")


@app.post("/analysis/{analysis_id}/save-and-execute", tags=["insights"])
async def save_and_execute_analysis(
    request: Request,
    analysis_id: str,
    body: SaveAndExecuteIn,
):
    """
    Save the user's EDITED code and run it, starting FROM THE PLANNER.

    Flow: edited code -> planner review (approves correct code, corrects wrong
    code, since non-technical users may edit in bugs) -> validator -> executor
    -> new version checkpoint. The planner is the verification gate; if it has to
    correct the code, the corrected code is what gets validated and executed.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    edited_code = (body.code or "").strip()
    if not edited_code:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No code provided to save and execute")

    row = await _resolve_session_from_aid(analysis_id, user_id)
    if not row or not row.get("session_id"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found")

    session_id = row["session_id"]
    sess = await get_session(session_id)
    if not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found for this analysis")

    schema_for_state = _normalize_schema_for_state(
        _maybe_json_load(sess.get("schema")) or sess.get("schema")
    )
    preview = _maybe_json_load(sess.get("uploaded_csv_preview")) or sess.get("uploaded_csv_preview") or []
    columns = _maybe_json_load(sess.get("uploaded_csv_columns")) or sess.get("uploaded_csv_columns") or []

    # Last user analysis prompt, so the planner reviews the edit against real intent.
    last_prompt = ""
    tid = sess.get("thread_id")
    if tid:
        try:
            await hydrate_thread_history(tid)
            for m in reversed(await read_thread_msgs(tid) or []):
                if isinstance(m, dict) and m.get("role") == "user":
                    last_prompt = m.get("content") or ""
                    break
        except Exception:
            logger.debug("[save-and-execute] could not read last prompt", exc_info=True)

    review_state = {
        "schema": schema_for_state,
        "uploaded_csv_columns": columns,
        "uploaded_csv_preview": preview,
    }

    # ---- 1) PLANNER REVIEW (the verification gate) ----
    review = await asyncio.to_thread(
        review_edited_code, review_state, edited_code, user_intent=last_prompt
    )
    reviewed_code = review.get("code") or edited_code
    planner_status = review.get("status", "approved")   # "approved" | "corrected"
    planner_notes = review.get("notes") or ""
    logger.info("[save-and-execute] aid=%s planner_status=%s", analysis_id, planner_status)

    # The planner-reviewed edit is the authoritative execution source.
    code_to_run = reviewed_code

    # NoDatabaseWrites is a safety rule, not a quality check, so it runs even
    # when the caller turned the validators off: run_validation=false used to
    # skip the whole validator, and with it the only thing between edited code
    # and a database write outside the transfer path.
    db_write_errors = database_write_errors(reviewed_code)
    if db_write_errors:
        logger.warning("[save-and-execute] aid=%s rejected by NoDatabaseWrites (run_validation=%s)",
                       analysis_id, body.run_validation)
        return {
            "available": False,
            "status": "validation_failed",
            "analysis_id": analysis_id,
            "session_id": session_id,
            "planner_status": planner_status,
            "planner_notes": planner_notes,
            "corrected_code": reviewed_code if planner_status == "corrected" else None,
            "validation_passed": False,
            "validation_stage": "static_semantic",
            "validation_feedback": "\n".join(db_write_errors),
            "output_json": None,
            "output_file_data": None,
            "checkpoint_created": False,
        }

    if body.run_validation:
        coding_state = {
            "generated_code": reviewed_code,
            "coder_definition": {"code": reviewed_code},
            "schema": schema_for_state,
            "source_schema": schema_for_state,

            # The original chat request must not be used to restore old analytical
            # choices such as row limits, thresholds, filters, or sort direction.
            "user_prompt": (
                "Validate the supplied user-edited code for syntax, schema, runtime, "
                "and safety defects only. The code's analytical choices are authoritative."
            ),

            "uploaded_csv_columns": columns,
            "uploaded_csv_preview": preview,
            "datasets_context": [],
            "multi_dataset_state": [],
        }

        validation = await asyncio.to_thread(
            run_code_validation,
            coding_state,
        )

        if not validation["ok"]:
            logger.info(
                "[save-and-execute] aid=%s validation failed at %s "
                "(planner_status=%s)",
                analysis_id,
                validation.get("stage"),
                planner_status,
            )

            return {
                "available": False,
                "status": "validation_failed",
                "analysis_id": analysis_id,
                "session_id": session_id,
                "planner_status": planner_status,
                "planner_notes": planner_notes,
                "corrected_code": (
                    reviewed_code
                    if planner_status == "corrected"
                    else None
                ),
                "validation_passed": False,
                "validation_stage": validation.get("stage"),
                "validation_feedback": _plain_validation_feedback(
                    validation.get("stage"), validation.get("feedback")
                ),
                "output_json": None,
                "output_file_data": None,
                "checkpoint_created": False,
            }

        validator_code = (
            validation.get("code") or reviewed_code
        ).strip()

        if validator_code != reviewed_code.strip():
            logger.warning(
                "[save-and-execute] aid=%s validator attempted to rewrite "
                "planner-approved user code; preserving reviewed code",
                analysis_id,
            )

        # Validation blocks unsafe/invalid code but does not silently change
        # intentional analytical choices.
        code_to_run = reviewed_code


    # ---- 3) EXECUTOR (local vs k8s-ray, same policy as /refresh) ----
    local_input = (
        (sess.get("work_local_input") and _posix(sess["work_local_input"]))
        or (sess.get("sample_local_input") and _posix(sess["sample_local_input"]))
    )
    output_location = sess.get("output_location") and _posix(sess["output_location"])

    file_size = int(sess.get("file_size_bytes") or 0)
    cloud_uri = _full_cloud_uri(sess)
    conn_id = sess.get("connection_id")
    fidelity = sess.get("analysis_fidelity")
    use_ray = (
        file_size >= LARGE_DATASET_THRESHOLD_BYTES
        and bool(cloud_uri) and bool(conn_id)
        and fidelity in (FIDELITY_PORTFOLIO, FIDELITY_ENTIRE)
    )

    state_in: Dict[str, Any] = {
        "user_id": user_id,
        "session_id": session_id,
        "dataset_id": sess.get("dataset_id"),
        "messages": [],
        #"coder_definition": {"code": validated_code},
        "coder_definition": {"code": code_to_run},
        #"generated_code": validated_code,
        "generated_code": code_to_run,
        "schema": schema_for_state,
        "input_data_type": sess.get("input_data_type", "csv"),
        "folder_table_type": sess.get("folder_table_type"),
        "folder_read_path": sess.get("folder_read_path"),
        "hive_partitioning": bool(sess.get("hive_partitioning")),
        "iceberg_metadata_uri": sess.get("iceberg_metadata_uri"),
        "output_location": output_location,
        "connection_id": conn_id,
        "data_source_location": (cloud_uri if use_ray else local_input),
        "data_source_location_local": local_input,
        "data_source_location_cloud": cloud_uri,
        "execution_result": {},
        "output_file_data": None,
    }
    if use_ray:
        state_in.update({
            "execution_mode": "k8s-ray",
            "ray_namespace": os.getenv("RAY_NAMESPACE", "ray-training"),
            "ray_execution_profile": "batch_heavy",
        })

    node = execution_agent_node_ray if use_ray else execution_agent_node_local
    logger.info(
        "[save-and-execute] aid=%s session=%s mode=%s",
        analysis_id, session_id, "k8s-ray" if use_ray else "local",
    )
    final = await _run_execution_node(node, state_in, use_ray)
    # ---- Extract and verify the newly executed output ----
    _exec_result = final.get("execution_result") or {}
    _exec_status = str(
        _exec_result.get("status", "")
    ).strip().lower()

    _exec_ok = _exec_status in {
        "success",
        "succeeded",
        "dryrun",
        "completed",
    }

    output_file_data, output_json = get_output_from_state(
        final,
        tmp_root=TMP_ROOT,
    )

    _has_output_file = bool(
        isinstance(output_file_data, dict)
        and output_file_data.get("content")
    )

    _has_output_json = output_json is not None

    # Define this BEFORE any persistence logic uses it.
    _fresh_output = bool(
        final.get("fresh_output_produced")
        or _exec_result.get("fresh_output")
        or _has_output_file
        or (_exec_ok and _has_output_json)
    )

    output_row_count = (
        len(output_json)
        if isinstance(output_json, list)
        else None
    )

    logger.info(
        "[save-and-execute] aid=%s status=%s "
        "fresh_output=%s output_rows=%s has_file=%s",
        analysis_id,
        _exec_status,
        _fresh_output,
        output_row_count,
        _has_output_file,
    )

    if not _exec_ok or not _fresh_output:
        logger.error(
            "[save-and-execute] execution produced no usable fresh output: "
            "aid=%s status=%s message=%s",
            analysis_id,
            _exec_status,
            _exec_result.get("message"),
        )

        return {
            "available": False,
            "status": "execution_failed",
            "analysis_id": analysis_id,
            "session_id": session_id,
            "planner_status": planner_status,
            "planner_notes": planner_notes,
            "final_code": (
                code_to_run
                if code_to_run.strip() != edited_code.strip()
                else None
            ),
            "validation_passed": True,
            # Plain words only: the message on the result carries the
            # script's stderr (a traceback), which is logged above.
            "execution_message": _save_and_execute_failure_message(final),
            "output_json": None,
            "output_file_data": None,
            "output_row_count": None,
            "new_version_prompt_ts": None,
            "checkpoint_created": False,
        }


    # ---- Persist the new 10-row output ----
    old_output_key = sess.get("gcs_output_object_key")
    old_output_history_count = len(
        _build_asset_history(sess, "output_assets")
    )

    checkpoint_created = False

    try:
        output_fingerprint = hashlib.sha256(
            json.dumps(
                output_json if output_json is not None else [],
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()

        sess["last_output_fingerprint"] = output_fingerprint
        await save_session(session_id, sess)

        await _persist_assets_background(
            user_id=user_id,
            session_id=session_id,
            dataset_id=sess.get("dataset_id", ""),
            generated_code=code_to_run,
            planner_definition={
                "plan": sess.get("plan", "")
            },
            coder_definition={
                "code": code_to_run
            },
            execution_result=_exec_result,
            exec_succeeded=True,
            output_file_data=output_file_data,
            output_json=output_json,
            output_location=output_location,
            visualization_config=(
                _maybe_json_load(
                    sess.get("visualization_config")
                ) or {}
            ),
            connection_id=conn_id,
            storage_uri=sess.get(
                "data_source_location"
            ),
            dataset_label=sess.get("alias") or sess.get("filename") or "",
            analysis_label=redact_typed_secrets(last_prompt)[0][:60] if last_prompt else "",
        )

        checkpoint_created = True

    except Exception as exc:
        # This is fatal for Save & Execute because the UI reads the
        # persisted asset/version after this request.
        logger.exception(
            "[save-and-execute] persistence failed: aid=%s",
            analysis_id,
        )

        return {
            "available": False,
            "status": "persistence_failed",
            "analysis_id": analysis_id,
            "session_id": session_id,
            "planner_status": planner_status,
            "planner_notes": planner_notes,
            "validation_passed": True,
            "execution_message": (
                f"Execution produced {output_row_count} rows, "
                "but saving the new output failed."
            ),
            "output_json": output_json,
            "output_file_data": output_file_data,
            "output_row_count": output_row_count,
            "new_version_prompt_ts": None,
            "checkpoint_created": False,
        }


    # ---- Confirm that persistence moved the current output pointer ----
    fresh = await get_session(session_id) or sess
    out_hist = _build_asset_history(
        fresh,
        "output_assets",
    )

    latest_output = (
        max(
            out_hist,
            key=lambda entry: entry.prompt_ts,
        )
        if out_hist
        else None
    )

    history_advanced = (
        len(out_hist) > old_output_history_count
    )

    new_output_key = fresh.get(
        "gcs_output_object_key"
    )

    # Repair a stale current-output pointer if history was added but
    # gcs_output_object_key was not moved.
    if (
        history_advanced
        and latest_output
        and new_output_key != latest_output.object_key
    ):
        logger.warning(
            "[save-and-execute] repairing output pointer: "
            "old=%s current=%s latest=%s",
            old_output_key,
            new_output_key,
            latest_output.object_key,
        )

        fresh["gcs_output_object_key"] = (
            latest_output.object_key
        )

        if await save_session(session_id, fresh):
            new_output_key = latest_output.object_key
        else:
            logger.error(
                "[save-and-execute] could not save repaired "
                "output pointer for session=%s",
                session_id,
            )

    new_prompt_ts = (
        latest_output.prompt_ts
        if latest_output
        else None
    )

    logger.info(
        "[save-and-execute] completed: aid=%s rows=%s "
        "old_output=%s new_output=%s checkpoint=%s",
        analysis_id,
        output_row_count,
        old_output_key,
        new_output_key,
        checkpoint_created,
    )

    return {
        "available": True,
        "status": _exec_result.get(
            "status",
            "success",
        ),
        "analysis_id": analysis_id,
        "session_id": session_id,
        "planner_status": planner_status,
        "planner_notes": planner_notes,
        "final_code": (
            code_to_run
            if code_to_run.strip() != edited_code.strip()
            else None
        ),
        "validation_passed": True,
        "output_json": output_json,
        "output_file_data": output_file_data,
        "output_row_count": output_row_count,
        "new_version_prompt_ts": new_prompt_ts,
        "code_object_key": fresh.get(
            "gcs_code_object_key"
        ),
        "output_object_key": new_output_key,
        "checkpoint_created": checkpoint_created,
    }
# -------------------------------------------------------------------
# MCP connection credentials (encrypt / decrypt)
#
# Replaces the `encrypt-mcp-credentials` Supabase edge function, which is not
# deployed in this project. Same AES-GCM scheme (SHA-256 of DB_ENCRYPTION_KEY)
# via encrypt_secret/decrypt_secret, so ciphertext stays byte-compatible with
# _maybe_decrypt_into() in cloud_connections.py.

# NOTE: mcp_connections.user_id stores auth.uid() (NOT profiles.id), so it is
# compared directly against the JWT sub — no _resolve_profile_id() hop here.
# -------------------------------------------------------------------

_MCP_SECRET_FIELDS = ("api_key", "username")


class McpCredentialsIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    api_key: Optional[str] = Field(None, alias="apiKey")
    username: Optional[str] = None


# Rows live wherever app.api.db_connections decides (Supabase when hosted, the
# API's own file in local auth mode), so these routes work in both.
def _mcp_conn_update(connection_id: str, patch: Dict[str, Any]) -> None:
    _db_connections.connection_store().update(connection_id, patch)


# Connection ids are uuids in both stores (the Supabase column is uuid; the file
# store mints uuid4). PostgREST rejects any other string with 22P02, which
# supabase-py raises, so a malformed id must stop here: it can match no row,
# and letting it through turned "not found" into a 500.
_MCP_CONN_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _mcp_conn_not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "Connection not found")


async def _mcp_conn_owned_row(connection_id: str, user_id: str) -> Dict[str, Any]:
    """The caller's own connection row (the service role bypasses RLS).

    Ownership is db_connections.owned_row, the rule /api/connections already
    follows: another user's connection, a missing one and a malformed id all
    get the same 404 and body, so the response never says whether an id exists.
    """
    if not _MCP_CONN_ID_RE.fullmatch(connection_id or ""):
        raise _mcp_conn_not_found()
    try:
        row = await asyncio.to_thread(_db_connections.owned_row, connection_id, user_id)
    except Exception as exc:
        logger.error("[mcp-creds] lookup failed for %s: %s", connection_id, exc, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Connection lookup failed")
    if not row:
        logger.warning("[mcp-creds] connection %s not found for requester %s", connection_id, user_id)
        raise _mcp_conn_not_found()
    return row


@app.post("/customers/register", tags=["mcp"])
async def register_customer_database(request: Request):
    """Register a database with the onboarding API, as the authenticated user.

    The web UI's connect modal posts here (the backend proxy forwards to this
    service only). The onboarding API accepts a registration only for a user
    it can verify, so the user resolved from the JWT is forwarded as a signed
    assertion and the customer is recorded as theirs. The body passes through
    untouched and is never logged: it carries the database password.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    import app.mcp_server.client as onboarding_client

    headers = onboarding_client.user_headers(user_id)
    if not headers:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "DB_ENCRYPTION_KEY is not configured; cannot register a database.",
        )
    headers["Content-Type"] = "application/json"
    try:
        async with httpx.AsyncClient() as client:
            upstream = await client.post(
                f"{onboarding_client.ONBOARDING_API_URL}/customers/register",
                content=await request.body(), headers=headers, timeout=60.0,
            )
    except httpx.HTTPError as exc:
        logger.warning("[mcp-register] onboarding API unreachable: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Registration service unavailable")
    try:
        payload = upstream.json()
    except ValueError:
        payload = {"detail": f"Registration service answered HTTP {upstream.status_code}"}
    return JSONResponse(status_code=upstream.status_code, content=payload)


@app.post("/api/mcp-connections/{connection_id}/encrypt", tags=["mcp"])
async def encrypt_mcp_connection_credentials(
    request: Request,
    connection_id: str,
    body: McpCredentialsIn,
):
    """Encrypt api_key / username into *_ciphertext + *_iv and blank the plaintext columns."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    row = await _mcp_conn_owned_row(connection_id, user_id)

    patch: Dict[str, Any] = {}
    encrypted: List[str] = []
    for field in _MCP_SECRET_FIELDS:
        value = getattr(body, field, None)
        if not value:
            continue
        blob = encrypt_secret(value)
        if not blob:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "DB_ENCRYPTION_KEY is not configured; cannot encrypt credentials.",
            )
        patch[f"{field}_ciphertext"] = blob["ciphertext"]
        patch[f"{field}_iv"] = blob["iv"]
        patch[field] = ""          # never keep plaintext in the row
        encrypted.append(field)

    if not patch:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Provide at least one of apiKey / username.",
        )

    try:
        await asyncio.to_thread(_mcp_conn_update, connection_id, patch)
    except Exception as exc:
        logger.error("[mcp-creds] update failed for %s: %s", connection_id, exc, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to store credentials")

    logger.info("[mcp-creds] encrypted %s for connection %s", encrypted, connection_id)
    await _bind_mcp_customer_owner(row.get("customer_id"), user_id, body.api_key)
    return {"success": True, "connection_id": connection_id, "encrypted": encrypted}


@app.post("/api/mcp-connections/{connection_id}/decrypt", tags=["mcp"])
async def decrypt_mcp_connection_credentials(request: Request, connection_id: str):
    """Return the decrypted api_key / username to the owning user."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    row = await _mcp_conn_owned_row(connection_id, user_id)

    out: Dict[str, Optional[str]] = {}
    for field in _MCP_SECRET_FIELDS:
        out[field] = decrypt_secret({
            "ciphertext": row.get(f"{field}_ciphertext"),
            "iv": row.get(f"{field}_iv"),
        })

    if row.get("api_key_ciphertext") and not out["api_key"]:
        # Ciphertext exists but will not decrypt — almost always a
        # DB_ENCRYPTION_KEY mismatch (e.g. rows migrated from another project).
        logger.error(
            "[mcp-creds] connection %s has api_key_ciphertext but decryption failed "
            "(DB_ENCRYPTION_KEY mismatch?)", connection_id,
        )
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Stored credentials could not be decrypted with the current "
            "DB_ENCRYPTION_KEY. Re-register this connection.",
        )

    await _bind_mcp_customer_owner(row.get("customer_id"), user_id, out["api_key"])
    return {"apiKey": out["api_key"], "username": out["username"]}



@app.get("/api/integrations", tags=["integrations"])
async def get_integrations(request: Request):
    """Render state for every provider card on Settings → Integrations.
    Never returns a token — only whether one is stored and the non-secret config."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
 
    try:
        rows = await asyncio.to_thread(list_integration_connections, user_id)
    except Exception as exc:
        logger.error("[integrations] list failed for %s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not load integrations")
 
    by_provider = {r.get("provider"): r for r in rows}
    env_github = bool(os.getenv("GITHUB_SYSTEM_TOKEN") and os.getenv("GITHUB_JOB_REGISTRY_REPO"))
 
    out = []
    for provider in sorted(SUPPORTED_INTEGRATION_PROVIDERS):
        row = by_provider.get(provider) or {}
        config = row.get("config") or {}
        out.append({
            "provider": provider,
            "enabled": bool(row.get("enabled")),
            "has_token": bool(row.get("token_ciphertext")),
            # GitHub still works off the env fallback until a user connects via UI;
            # surface that so the card can show "using system default".
            "using_system_default": (provider == "github" and not row and env_github),
            "config": {k: v for k, v in config.items() if k != "token"},
        })
    return {"integrations": out}
 
 
@app.post("/api/integrations/github/connect", tags=["integrations"])
async def connect_github_integration(request: Request, body: GitHubConnectIn):
    """Validate a GitHub PAT against a repo, then store it encrypted and enable it."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
 
    repo = (body.repo or "").strip()
    token = (body.token or "").strip()
    if not REPO_RE.match(repo):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "repo must be in 'owner/repo' format")
    if not token:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "token is required")
 
    check = await validate_github_repo_access(token, repo)
    if not check["ok"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, check["error"] or "GitHub validation failed")
    if not check["push"]:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "The token can read this repo but has no push (write) access; "
            "job persistence needs write access.",
        )
 
    blob = encrypt_secret(token)
    if not blob:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "DB_ENCRYPTION_KEY is not configured; cannot store the token securely.",
        )
 
    try:
        await asyncio.to_thread(
            save_integration_connection,
            user_id, "github",
            enabled=True,
            config_updates={"repo": check["full_name"] or repo},
            token_ciphertext=blob["ciphertext"],
            token_iv=blob["iv"],
        )
    except Exception as exc:
        logger.error("[integrations] github connect save failed for %s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to save the GitHub connection")
 
    return {
        "success": True,
        "provider": "github",
        "repo": check["full_name"] or repo,
        "enabled": True,
        "push_access": True,
    }
 
 
@app.patch("/api/integrations/github", tags=["integrations"])
async def update_github_integration(request: Request, body: GitHubUpdateIn):
    """Toggle the card on/off, or change the target repo (re-validated with the stored token)."""
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
 
    existing = await asyncio.to_thread(get_integration_connection, user_id, "github")
    if not existing and body.enabled:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Configure GitHub first (POST /api/integrations/github/connect) before enabling it.",
        )
 
    config_updates = None
    if body.repo is not None:
        repo = body.repo.strip()
        if not REPO_RE.match(repo):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "repo must be in 'owner/repo' format")
        token = (
            decrypt_secret({
                "ciphertext": existing.get("token_ciphertext"),
                "iv": existing.get("token_iv"),
            })
            if existing else None
        )
        if token:
            check = await validate_github_repo_access(token, repo)
            if not check["ok"]:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, check["error"] or "GitHub validation failed")
            if not check["push"]:
                raise HTTPException(status.HTTP_403_FORBIDDEN, "Token lacks push access to that repo.")
            repo = check["full_name"] or repo
        config_updates = {"repo": repo}
 
    try:
        row = await asyncio.to_thread(
            save_integration_connection,
            user_id, "github",
            enabled=body.enabled,
            config_updates=config_updates,
        )
    except Exception as exc:
        logger.error("[integrations] github update failed for %s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to update the GitHub connection")
 
    return {
        "success": True,
        "provider": "github",
        "enabled": bool(row.get("enabled")),
        "repo": (row.get("config") or {}).get("repo"),
    }
 
 
@app.delete("/api/integrations/github", status_code=status.HTTP_204_NO_CONTENT, tags=["integrations"])
async def disconnect_github_integration(request: Request):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    try:
        await asyncio.to_thread(delete_integration_connection, user_id, "github")
    except Exception as exc:
        logger.error("[integrations] github delete failed for %s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to remove the GitHub connection")
    return Response(status_code=status.HTTP_204_NO_CONTENT)



async def _load_insight_viz_context(
    session_id: Optional[str],
    dataset_id: Optional[str],
) -> Dict[str, Any]:
    """Dataset label + stored visualization profile for the analysis session.

    For multi-file uploads the analyses row points at the GROUP session, so a
    requested dataset_id is mapped to its own session via dataset_session_map.
    """
    if not session_id:
        return {}
    sess = await get_session(session_id)
    if not isinstance(sess, dict):
        return {}

    if dataset_id and sess.get("dataset_id") != dataset_id:
        smap = sess.get("dataset_session_map") or {}
        if isinstance(smap, str):
            smap = _maybe_json_load(smap) or {}
        ds_sid = smap.get(dataset_id) if isinstance(smap, dict) else None
        if ds_sid:
            ds_sess = await get_session(ds_sid)
            if isinstance(ds_sess, dict) and ds_sess.get("user_id") == sess.get("user_id"):
                sess = ds_sess

    viz = _maybe_json_load(sess.get("visualization_config")) or sess.get("visualization_config") or {}
    return {
        "dataset_name": sess.get("alias") or sess.get("filename"),
        "viz": viz if isinstance(viz, dict) else {},
    }





@app.post(
    "/analysis/{analysis_id}/insights/explain",
    response_model=InsightExplainOut,
    tags=["insights"],
)
async def explain_analysis_insights(
    request: Request,
    analysis_id: str,
    body: InsightExplainIn,
):
    """Spoken-style explanation of the Auto Insights charts for the avatar.

    Owners AND project collaborators can use it (the Auto Insights page is
    shared), so authorize through the shared-analysis resolver rather than the
    owner-only _resolve_session_from_aid.
    """
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")

    shared = await _resolve_shared_session_for_thread("", user_id, analysis_id=analysis_id)
    if not shared:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found")

    ctx = await _load_insight_viz_context(shared.get("session_id"), body.dataset_id)
    viz = ctx.get("viz") or {}

    charts = [c.model_dump() for c in body.charts]
    referenced = {
        str(f) for c in charts for f in (c.get("x_field"), c.get("y_field")) if f
    }

    column_profiles = [
        {
            "name": p.get("name"),
            "dtype": p.get("dtype"),
            "n_unique": p.get("n_unique"),
            "missing_ratio": p.get("missing_ratio"),
        }
        for p in (viz.get("columns") or [])
        if isinstance(p, dict) and str(p.get("name")) in referenced
    ]
    warnings = [
        w for w in (viz.get("warnings") or [])
        if isinstance(w, str) and any(f"'{f}'" in w for f in referenced)
    ]

    result = await asyncio.to_thread(
        explain_charts,
        body.question,
        charts,
        body.dataset_name or ctx.get("dataset_name") or "dataset",
        warnings,
        [h.model_dump() for h in body.history],
        body.focus_chart_id,
        column_profiles,
    )

    logger.info(
        "[insights-explain] aid=%s role=%s charts=%d focus=%s answered=%s",
        analysis_id, shared.get("role"), len(charts),
        body.focus_chart_id, bool(result.get("answer")),
    )
    return InsightExplainOut(**result)
 
# -------------------------------------------------------------------
# Local runner
# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run(
        "app.api.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "9000")),
        reload=DEBUG,
        proxy_headers=True,
        # Same loop as the image CMD (deploy/docker/Dockerfile.api).
        loop="asyncio",
    )
