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
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
import httpx
from botocore.exceptions import ClientError
from fastapi import (
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
from fastapi.responses import FileResponse
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

# Ensure our package loggers propagate to root
for _name in ("app", "app.agents", "app.graph", "app.api"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(LOG_LEVEL)
    _lg.propagate = True

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
from app.agents.sampling_agent_daft import sample_with_profiling
from file_handler.excel_connector import export_workbook_sheets_to_csv
from app.agents.sampling_async import sample_quick
from app.agents.profiling_agent import profile_full
from app.agents.sampling_persistence import persist_full_profile, load_portfolio, load_profile, upsert_profile
from app.agents.execution_agent import execution_agent_node_local, execution_agent_node_ray
from app.agents.planner import review_edited_code
from app.agents.validator import run_code_validation
from app.core.celery_app import (
    AvalokaScheduler,
    AvalokaEntry,
    celery_app,
    _logs_from_final,
    _ray_job_from_final,
    _run_status_from_final,
    _training_failure_from_final,
)
from app.agents.mta_v2.failure_diagnostics import (
    TRAINING_FAILURE_MESSAGE,
    format_training_failure,
)
from app.core.task_metadata import scheduled_task_metadata
from app.utils import convert_message_dicts_to_objects
from celery.result import AsyncResult
from celery.schedules import crontab
from redbeat.schedulers import get_redis

# NEW: shared services
from app.services import session_service, storage_service
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
    find_active_db_customer_for_user,
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
)
from app.services.persistence_service import (
    _persist_assets_background,
    generate_signed_url as _generate_asset_signed_url,
)
from app.agents.visualization_agent import build_visualization_config_from_sample

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


def _validate_auth_config() -> None:
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
    # The OSS distribution ships its own line: this is the first public
    # release, so it is 1.0.x there while the internal branch stays on the 1.6
    # line. AVALOKA_EDITION is already how the build tells the two apart.
    if (os.getenv("AVALOKA_EDITION") or "").strip().lower() in ("oss", "community"):
        oss = (os.getenv("AVALOKA_OSS_VERSION") or "1.0.0").strip()
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

# Supported file formats
SUPPORTED_UPLOAD_EXTS = {"csv", "tsv", "json", "xml", "parquet", "avro", "orc", "xls", "xlsx"}

CONTENT_TYPE_TO_EXT: Dict[str, Optional[str]] = {
    "text/csv": "csv",
    "application/vnd.ms-excel": "xlsx",
    "text/tab-separated-values": "tsv",
    "application/json": "json",
    "application/xml": "xml",
    "text/xml": "xml",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    # parquet/avro/orc often come as octet-stream; rely on filename in those cases
    "application/octet-stream": None,
}

CACHE_CALL_TIMEOUT = float(os.getenv("CACHE_CALL_TIMEOUT", "0.6"))  # seconds per cache op

# Per-dataset profiling + portfolio cache (populated on upload, read on preview)
_profile_meta: Dict[str, Dict[str, Any]] = {}
MAX_DB_SAMPLE_ROWS = 1000
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


def _training_finished_successfully(final_state: Dict[str, Any]) -> bool:
    if not final_state.get("training_completed"):
        return False
    result = final_state.get("training_result")
    if not isinstance(result, dict):
        return False
    if result.get("error") or result.get("execution_error"):
        return False
    return bool(result.get("mlflow_run_id"))

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
                out["messages"] = msgs + [AIMessage(content=decision["ai_message"])]
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
    work_input_ok = bool(work_input and work_input.exists())

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
            dest = work_input if work_input else (
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

def _assistant_messages_from_state(state: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = state.get("messages") or []
    if messages and isinstance(messages[0], dict):
        messages = convert_message_dicts_to_objects(messages)

    result: List[Dict[str, str]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            result.append({"role": "assistant", "content": str(message.content or "")})
    return result

async def lg_request(method: str, path: str, **kw) -> httpx.Response:
    """
    Low-level HTTP call to the LangGraph server, using the shared http_client from config
    if available; otherwise a one-shot client.
    """
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


def _pending_turn_running(deferred_id: str):
    def _apply(cur: Dict[str, Any]) -> None:
        pt = cur.get("pending_turn")
        pt = _maybe_json_load(pt) if isinstance(pt, str) else pt
        # Never downgrade a turn we've already finished.
        if isinstance(pt, dict) and pt.get("id") == deferred_id and pt.get("status") in ("done", "error"):
            return
        cur["pending_turn"] = {"id": deferred_id, "status": "running", "created_at": _now_iso()}
    return _apply


def _pending_turn_done(deferred_id: str, result: Dict[str, Any]):
    def _apply(cur: Dict[str, Any]) -> None:
        cur["pending_turn"] = {
            "id": deferred_id, "status": "done",
            "result": _json_safe_payload(result), "finished_at": _now_iso(),
        }
    return _apply


def _pending_turn_error(deferred_id: str, message: str):
    def _apply(cur: Dict[str, Any]) -> None:
        cur["pending_turn"] = {
            "id": deferred_id, "status": "error",
            "message": message, "finished_at": _now_iso(),
        }
    return _apply


def _resolve_user_id(req: Request) -> Optional[str]:
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

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

@asynccontextmanager
async def server_lifespan(app: FastAPI):
    global small_file_persistence_executor, small_file_persistence_semaphore, small_file_persistence_inflight

    _validate_auth_config()
    # A service-role key for the wrong project answers every request with
    # `401 Invalid API key`, which reads like a rotated key rather than a
    # mismatch. Say so once at boot instead of once per failed request.
    try:
        from app.agents.sampling_persistence import warn_on_supabase_credential_mismatch
        warn_on_supabase_credential_mismatch()
    except Exception as _exc:  # never let a diagnostic stop the server
        logger.debug("supabase credential check skipped: %s", _exc)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


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
            resp.headers["Access-Control-Allow-Methods"] = request.headers.get(
                "Access-Control-Request-Method", "*"
            )
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
):
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
    
    # ---- Upload count validation ----
    if len(incoming) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Too many files. Max allowed is {MAX_UPLOAD_FILES} per upload."
        )


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

    # Group session id = FIRST file’s session (this is what you return as session_id)
    group_session_id: Optional[str] = None
    thread_id: str = ""

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
    first_sample_statistics: Optional[Dict[str, Any]] = None
    
    # Track created resources so we can rollback if a later file fails
    created_session_ids: List[str] = []
    created_work_dirs: List[Path] = []
    created_tmp_uploads: List[Path] = []
    total_bytes_written = 0
    try:
        # ---- Stage A: materialize each incoming file to a tmp path (with size
        # validation) and expand Excel workbooks into one logical CSV dataset
        # per sheet, so a workbook's sheets become sibling datasets instead of
        # everything after the first sheet being silently dropped.
        logical_items: List[Dict[str, Any]] = []
        for f_idx, up in enumerate(incoming):
            filename = up.filename or f"file_{f_idx}"
            alias = _alias_from_filename(filename)

            # Resolve extension
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

            # ---- write upload to tmp file (with per-file + total size validation) ----
            tmp_upload = TMP_ROOT / f"upload_{_new_id()}.{ext}"
            created_tmp_uploads.append(tmp_upload)

            file_bytes = await _read_upload_to_path_with_limit(
                up=up,
                dest_path=tmp_upload,
                per_file_limit=MAX_UPLOAD_FILE_BYTES,
            )
            total_bytes_written += file_bytes
            if total_bytes_written > MAX_UPLOAD_TOTAL_BYTES:
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"Total upload size exceeds limit ({MAX_UPLOAD_TOTAL_BYTES} bytes).",
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
                        "notes": sheet_notes if e_idx == 0 else [],
                    })
                if multi_sheet or sheet_notes:
                    logger.info(
                        "[upload] workbook %s expanded into %d sheet dataset(s): %s | notes=%s",
                        filename,
                        len(exports),
                        [e.sheet_name for e in exports],
                        sheet_notes,
                    )
                # raw workbook tmp no longer needed once per-sheet CSVs exist
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

        # Aliases must be unique within the upload group (dataset auto-routing
        # and the UI dropdown key on them); suffix duplicates deterministically.
        seen_aliases: Dict[str, int] = {}
        for item in logical_items:
            base_alias = item["alias"]
            n = seen_aliases.get(base_alias, 0)
            seen_aliases[base_alias] = n + 1
            if n:
                item["alias"] = f"{base_alias}_{n + 1}"

        # ---- Stage B: one dataset (id + session + storage object + sampling)
        # per logical item.
        for idx, item in enumerate(logical_items):
            filename = item["filename"]
            alias = item["alias"]
            ext = item["ext"]
            tmp_upload = item["tmp_path"]

            dsid = _new_id()
            object_name = f"{dsid}.{ext}"
            if idx == 0:
                first_dataset_id = dsid

            # one “group” session returned to client; each dataset stored in its own session
            if idx == 0:
                session_id = _new_id()
                group_session_id = session_id
            else:
                session_id = _new_id()
                assert group_session_id is not None

            # Track session_id early so we can rollback if anything fails later
            #created_session_ids.append(session_id)

            # ---- PUT to storage ----
            try:
                if dest_uri and "://" in dest_uri:
                    if conn:
                        store_for_write, _base_prefix = await _store_from_connection_uri(dest_uri, conn)
                        key = object_name.lstrip("/")
                        uri = await asyncio.to_thread(store_for_write.put_file, tmp_upload, key)
                    else:
                        if dest_uri.startswith("az://"):
                            store_for_write = storage_service.blob_store
                            try:
                                _scheme, rest = dest_uri.split("://", 1)
                                parts = rest.split("/", 2)
                                prefix_path = parts[2].strip("/") if len(parts) >= 3 else ""
                            except Exception:
                                prefix_path = ""
                            key_base = prefix_path.rstrip("/")
                            key = f"{key_base}/{object_name}".lstrip("/") if key_base else object_name
                            uri = await asyncio.to_thread(store_for_write.put_file, tmp_upload, key)
                        else:
                            store_for_write, base_key = _store_and_key_from_uri(dest_uri, None)
                            key_base = (base_key or "").rstrip("/")
                            key = f"{key_base}/{object_name}".lstrip("/") if key_base else object_name
                            uri = await asyncio.to_thread(store_for_write.put_file, tmp_upload, key)
                else:
                    uri = await asyncio.to_thread(storage_service.blob_store.put_file, tmp_upload, object_name)
            except Exception as e:
                logger.exception("failed to upload to storage: %s", e)
                raise HTTPException(500, "Failed to upload to storage")
            finally:
                # tmp file no longer needed after storage upload succeeds/fails
                try:
                    tmp_upload.unlink(missing_ok=True)
                except Exception:
                    pass

            # ---- create work dir + download for sampling ----
            work_dir = (TMP_ROOT / session_id).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
            created_work_dirs.append(work_dir)

            input_copy = work_dir / f"input_{dsid}.{ext}"

            try:
                if conn and dest_uri:
                    store_for_read, _ = await _store_from_connection_uri(dest_uri, conn)
                    key_for_read = object_name.lstrip("/")
                    await asyncio.to_thread(store_for_read.get_file, key_for_read, input_copy)
                else:
                    store_for_read, key_for_read = _store_and_key_from_uri(uri, object_name)
                    await asyncio.to_thread(store_for_read.get_file, key_for_read or object_name, input_copy)
            except Exception as e:
                logger.exception("failed to download from storage for sampling: %s", e)
                _safe_rmtree(work_dir)
                raise HTTPException(500, "Failed to create working copy from storage")

            # ---- sample + schema via portfolio sampler ----
            samples: List[Dict[str, Any]] = []
            schema_any: Union[List[str], Dict[str, str], Dict[str, Any]] = {}
            ddl: str = ""
            used_agent = False
            portfolio_samples: Optional[Dict[str, List[Dict[str, Any]]]] = None
            available_samples: Optional[List[str]] = None
            profiling_result: Optional[Dict[str, Any]] = None
            sample_statistics: Optional[Dict[str, Any]] = None

            try:
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
                    _safe_rmtree(work_dir)
                    raise HTTPException(
                        500,
                        f"Failed to sample .{ext} file. File type not supported or sampling failed.",
                    )

            # Run semantic profiling if we have statistics from the sampler
            full_profiling_result: Optional[Dict[str, Any]] = None
            if used_agent and sample_statistics:
                try:
                    schema_list = list(schema_any.keys()) if isinstance(schema_any, dict) else list(schema_any)
                    prof_res = await asyncio.to_thread(
                        profile_full,
                        schema=schema_list,
                        sample_rows=samples,
                        sample_statistics=sample_statistics,
                    )
                    full_profiling_result = prof_res.get("full_profiling_result")
                except Exception as e:
                    logger.warning("[upload] profile_full failed (non-fatal): %s", e)

            _profile_meta[dsid] = {
                "portfolio_samples": portfolio_samples,
                "available_samples": available_samples,
                "profiling_result": profiling_result,
                "full_profiling_result": full_profiling_result,
                "sample_statistics": sample_statistics,
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

            if not samples:
                # The caller uploaded a file with no data rows. That is a client
                # error, not a server fault: a 500 pages an on-call engineer and
                # tells the UI something broke on our side.
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    "Empty Dataset Provided — the file has no data rows.",
                )

            # Optional caller-provided schema override (applied to each file)
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

            cols = list(schema_any.keys()) if isinstance(schema_any, dict) else list(schema_any)
            if not cols and samples:
                cols = list(samples[0].keys())

            output_path = work_dir / "output.csv"
            sample_input_path = work_dir / "sample_input.csv"
            try:
                _write_rows_to_csv(samples, sample_input_path)
            except Exception:
                logger.warning("Failed to write sample_input.csv; continuing", exc_info=True)

            # ---- per-dataset visualization config ----
            try:
                visualization_config = build_visualization_config_from_sample(
                    dataset_id=dsid,
                    sample_rows=samples,
                    schema=schema_any,
                    task_type="unsupervised",
                    target_column=None,
                )
                visualization_status = visualization_config.get("visualization_status", "ready")
            except Exception as e:
                logger.warning("Failed to build visualization_config: %s", e)
                visualization_config = {}
                visualization_status = "error"

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
                "data_source_location": uri,
                "object_name": object_name,
                # Avaloka created this storage object, so deleting the dataset may
                # delete it. (register-existing sets "registered" — see delete_dataset.)
                "source_kind": "uploaded",
                "work_local_input": _posix(input_copy),
                "output_location": _posix(output_path),
                "sample_local_input": _posix(sample_input_path),
                "uploaded_csv_preview": _jsonify(samples),
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
                # Excel-sheet provenance (None for non-workbook uploads): the
                # dataset object itself is the converted per-sheet CSV.
                "sheet_name": item["sheet_name"],
                "source_workbook": item["source_workbook"],
                "ingest_notes": item["notes"] or None,
                "group_session_id": group_session_id or session_id,
                "visualization_config": _jsonify(visualization_config),
                "visualization_status": visualization_status,
                # Keep full small-file preview artifacts in Redis session so /preview
                # does not depend on process-local _profile_meta.
                "portfolio_samples": _jsonify(portfolio_samples) if portfolio_samples else None,
                "profiling_result": _jsonify(profiling_result) if profiling_result else None,
                "full_profiling_result": _jsonify(full_profiling_result) if full_profiling_result else None,
                "sample_statistics": _jsonify(sample_statistics) if sample_statistics else None,
                "available_samples": available_samples,
                "analysis_fidelity": FIDELITY_PORTFOLIO,
                "selected_sample_name": (
                    DEFAULT_SAMPLE_NAME
                    if (available_samples and DEFAULT_SAMPLE_NAME in available_samples)
                    else (available_samples[0] if available_samples else None)
                ),
            }

            if not await save_session(session_id, session_data):
                _safe_rmtree(work_dir)
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to persist session. Please retry.")
            created_session_ids.append(session_id)
            # set cookie once (group session == first dataset session)
            if idx == 0:
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

            # create thread only once (first file)
            if idx == 0:
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
                    logger.warning("[upload] upstream /threads create failed (soft): %s", e.detail)

            # for subsequent dataset sessions, store same thread_id
            if idx != 0 and thread_id:
                try:
                    session_data["thread_id"] = thread_id
                    await save_session(session_id, session_data)
                except Exception:
                    pass

            dataset_ids.append(dsid)
            dataset_session_map[dsid] = session_id

            datasets_out.append(
                UploadedDatasetOut(
                    dataset_id=dsid,
                    filename=filename,
                    alias=alias,
                    columns=cols,
                    rows=_jsonify(samples),
                    visualization_config=visualization_config,
                    visualization_status=visualization_status,
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
                first_sample_statistics = sample_statistics

    except HTTPException:
        # rollback and re-raise
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

        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Upload failed")
    
    #  (only runs if upload succeeded)
    assert group_session_id is not None
    try:
        group_sess = await get_session(group_session_id)
        if isinstance(group_sess, dict):
            group_sess["dataset_ids"] = dataset_ids
            group_sess["dataset_session_map"] = dataset_session_map
            group_sess["thread_id"] = thread_id or group_sess.get("thread_id")
            await save_session(group_session_id, group_sess)
    except Exception:
        logger.warning("Failed to persist group dataset index", exc_info=True)


    # response shape selection (backward compat)
    if not used_files_field and len(incoming) == 1:
        return UploadResponse(
            dataset_id=first_dataset_id,
            session_id=group_session_id,
            thread_id=thread_id,
            schema=first_schema_any,
            samples=_jsonify(first_samples),
            ddl_schema=first_ddl,
            rows_sampled=len(first_samples),
            visualization_config=first_viz_config,
            visualization_status=first_viz_status,
            portfolio_samples=first_portfolio_samples,
            available_samples=first_available_samples,
            profiling_result=first_full_profiling_result,
            sample_statistics=first_sample_statistics,
            analysis_fidelity=FIDELITY_PORTFOLIO,
            selected_sample_name=(
                DEFAULT_SAMPLE_NAME
                if (first_available_samples and DEFAULT_SAMPLE_NAME in first_available_samples)
                else (first_available_samples[0] if first_available_samples else None)
            ),
            # A multi-sheet workbook expands into one dataset per sheet; the
            # top-level fields above describe the primary (largest) sheet.
            datasets=datasets_out if len(datasets_out) > 1 else None,
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

    # conn = await get_cloud_connection(body.connection_id)
    # object_name = body.key.lstrip("/")
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
    agent_res: Optional[Dict[str, Any]] = None
    sampling_path = local_input
    sampler_source_path = str(local_input)
    sampler_source_type = ext
    sampler_extra_args: Dict[str, Any] = {}
    persistence_source_path = str(local_input)
    full_cloud_uri: Optional[str] = None

    if body.storage_uri.startswith("gs://") or body.storage_uri.startswith("gcs://"):
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

    _is_large_file = file_size_bytes >= LARGE_DATASET_THRESHOLD_BYTES
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

    full_profiling_result: Optional[Dict[str, Any]] = None
    if used_agent and sample_statistics and not _is_large_file:
        try:
            schema_list = list(schema_any.keys()) if isinstance(schema_any, dict) else list(schema_any)
            prof_res = await asyncio.to_thread(
                profile_full,
                schema=schema_list,
                sample_rows=samples,
                sample_statistics=sample_statistics,
            )
            full_profiling_result = prof_res.get("full_profiling_result")
        except Exception as e:
            logger.warning("[register-existing] profile_full failed (non-fatal): %s", e)

    _profile_meta[dsid] = {
        "portfolio_samples": portfolio_samples,
        "available_samples": available_samples,
        "profiling_result": profiling_result,
        "full_profiling_result": full_profiling_result,
        "sample_statistics": sample_statistics,
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

    try:
        visualization_config = build_visualization_config_from_sample(
            dataset_id=dsid,
            sample_rows=samples,
            schema=schema_any,
            task_type="unsupervised",
            target_column=None,
        )
        visualization_status = visualization_config.get("visualization_status", "ready")
    except Exception as e:
        logger.warning(
            "Failed to build visualization_config (register-existing): %s",
            e,
        )
        visualization_config = {}
        visualization_status = "error"

    cols = list(schema_any.keys()) if isinstance(schema_any, dict) else list(schema_any)

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
        samples=_jsonify(samples),
        ddl_schema=ddl,
        rows_sampled=len(samples),
        visualization_config=visualization_config,
        visualization_status=visualization_status,
        file_size_bytes=file_size_bytes,
        file_size_mb=round(file_size_bytes / (1024 * 1024), 2),
        portfolio_samples=portfolio_samples,
        available_samples=available_samples,
        profiling_result=full_profiling_result,
        sample_statistics=sample_statistics,
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
            full_profiling_result = prof_res.get("full_profiling_result")
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
    )

# -------------------------------------------------------------------
# Threads
# -------------------------------------------------------------------


@app.post("/threads", response_model=ThreadOut)
async def create_thread(request: Request, body: ThreadCreateIn):
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
    sid = _resolve_session_id(request, None)
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

async def task_sse_iter(task_id: str, redbeat_key: str):
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
    
    while (task := AsyncResult(task_id, app=celery_app)).state not in ("FAILURE", "SUCCESS", "REVOKED"):
        if task.state != current_state:
            yield "event: message\ndata: " + json.dumps({
                "status": task.state,
                "message": task_metadata["running_message"],
                **task_metadata,
            }) + "\n\n"
            current_state = task.state
        await sleep(1)

    if entry is None:
        entry = AvalokaEntry.from_key(redbeat_key, celery_app)
    last_run_at = AvalokaScheduler.get_last_run_at(redbeat_key, generate_key=False)
    next_due_at = (
        entry.schedule.remaining_estimate(last_run_at).total_seconds() + 2
        if entry.max_runs > 0 and entry.total_run_count < entry.max_runs
        else -1
    )

    if task.state == "REVOKED":
        yield "event: message\ndata: " + json.dumps({"status": "CANCELLED", "traceback": str(task.traceback), "next_due_at": next_due_at, **task_metadata}) + "\n\n"
    elif task.failed():
        yield "event: message\ndata: " + json.dumps({"status": "FAILED", "traceback": str(task.traceback), "next_due_at": next_due_at, **task_metadata}) + "\n\n"
    else:
        final = task.result or {}
        final.update(
            messages=convert_message_dicts_to_objects(final.get("messages", []))
        )
        assistant_messages = _assistant_messages_from_state(final)
        output_file_data, output_json = get_output_from_state(final)
        execution_result = final.get("execution_result") or {}
        execution_succeeded = str(execution_result.get("status", "")).lower() in (
            "success", "succeeded", "completed", "done"
        )
        if _scheduled_task_has_chat_result_only(task_metadata) or (
            not execution_succeeded and not final.get("output_file_data")
        ):
            output_file_data = None
            output_json = None
        completion_message = _scheduled_task_completion_message(final, assistant_messages, task_metadata)
        yield "event: message\ndata: " + json.dumps({
            "status": "SUCCESS",
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
    return Response(status_code=204)


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


def _sample_fidelity_note(
    effective_fidelity: Optional[str],
    sample_rows: Any,
    sample_name: Optional[str],
) -> str:
    """One-line disclosure appended to numeric answers computed on a partial sample."""
    n = len(sample_rows) if isinstance(sample_rows, list) else 0
    if effective_fidelity == FIDELITY_PORTFOLIO:
        scope = f"a portfolio sample ({sample_name})" if sample_name else "a portfolio sample"
    else:
        scope = f"a {n:,}-row sample (the first rows of the file)" if n else "a partial sample of the file"
    return (
        f"\n\n> Note: these figures were computed on {scope}, not the entire dataset, so "
        f"counts, rates, distinct values and min/max may differ from the full file. "
        f'Reply "run this on the entire dataset" for exact numbers.'
    )
DEFAULT_SAMPLE_NAME = "random_baseline"
LARGE_DATASET_THRESHOLD_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB

_FIDELITY_LABELS = {
    FIDELITY_QUICK: "Quick Sample",
    FIDELITY_PORTFOLIO: "Portfolio Samples",
    FIDELITY_ENTIRE: "Entire Dataset",
}


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
        "This dataset is above the in-memory analysis budget, so no partial or "
        "sampled results are shown — the answer will come from the full run. "
        "Check the task panel for progress."
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


def _build_execution_context(fidelity: Optional[str], sample_name: Optional[str]) -> Dict[str, Any]:
    return {
        "mode": fidelity or FIDELITY_QUICK,
        "mode_label": _FIDELITY_LABELS.get(fidelity or "", fidelity or "Quick Sample"),
        "sample": sample_name,
    }


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
        return AsyncResult(execution_id, app=celery_app).state not in ("FAILURE", "SUCCESS", "REVOKED")
    except Exception:
        logger.debug("Could not inspect scheduled task state for %s", task_id, exc_info=True)
        return True


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
    if task_metadata.get("task_type") in {"start_inference", "stop_inference"} and assistant_messages:
        latest = assistant_messages[-1].get("content")
        if latest:
            return latest
    return str(task_metadata.get("success_message") or "The scheduled task is complete.")


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
    _ensure_thread_local(thread_id)
    # Restore chat history from Redis after a process restart (no-op while
    # the in-process copy is already populated).
    await hydrate_thread_history(thread_id)

    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Session storage (Redis) is not available.",
        )

    # optional hint from UI / frontend (for primary dataset)
    dataset_id_hint = (body.metadata or {}).get("dataset_id")
    # analysis id from the ?aid= in the Lovable URL (shared-analysis chats)
    analysis_id_hint = (body.metadata or {}).get("analysis_id") or (body.metadata or {}).get("aid")

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
        # default: all datasets in this group
        selected_ids = list(dict.fromkeys(group_sess.get("dataset_ids") or []))
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
    active_local_source = (
        primary_sess.get("active_data_source_location_local")
        or primary_sess.get("active_data_source_location")
    )
    active_cloud_source = primary_sess.get("active_data_source_location_cloud")

    # Durable snapshot for rebuild-on-miss. Write from the live session until it
    # lands (the frontend may create the analyses row only after the first turn),
    # then stop. Keyed by this request's thread_id, which matches the row.
    if not _truthy_session_value(primary_sess.get("snapshot_persisted")):
        if await persist_session_snapshot(thread_id, primary_sess) > 0:
            primary_sess["snapshot_persisted"] = True
            await update_session(
                primary_sid, lambda cur: cur.update({"snapshot_persisted": True})
            )

    # ----------------------------------------------------
    # 4.5) Fidelity + selected-sample resolution (chat-driven)
    # ----------------------------------------------------
    _text_requested_fidelity = _parse_fidelity_from_text(body.content)
    _text_requested_sample_name = _parse_selected_sample_from_text(body.content)
    _requested_fidelity = (
        _text_requested_fidelity
        or body.analysis_fidelity
        or (body.metadata or {}).get("analysis_fidelity")
    )
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
            lc_seed.append(AIMessage(content=msg))
            THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
            await persist_thread_history(thread_id)
            return ChatResponse(
                messages=[{"role": "user", "content": body.content}, {"role": "assistant", "content": msg}],
                datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
                active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
                analysis_fidelity=_current_fidelity,
                selected_sample_name=primary_sess.get("selected_sample_name"),
                execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            )
        primary_sess["selected_sample_name"] = _requested_sample_name

    _fallback_preview = _maybe_json_load(primary_sess.get("uploaded_csv_preview")) or primary_sess.get("uploaded_csv_preview") or []
    _resolved_rows, _resolved_name = _resolve_selected_sample_rows(
        primary_sess,
        dataset_id=primary_dsid,
        fallback_rows=_fallback_preview if isinstance(_fallback_preview, list) else [],
    )
    if _resolved_name and primary_sess.get("selected_sample_name") != _resolved_name:
        primary_sess["selected_sample_name"] = _resolved_name

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
        lc_seed.append(AIMessage(content=prompt_msg))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, {"role": "assistant", "content": prompt_msg}],
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
        #    scheduling a duplicate. Time-bounded: _scheduled_task_is_in_flight
        #    reports PENDING forever once the Celery result expires
        #    (result_expires=3600), which would otherwise lock the dataset on a
        #    stale job card permanently.
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
            lc_seed.append(AIMessage(content=_early_msg))
            THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
            await persist_thread_history(thread_id)
            return ChatResponse(
                messages=[{"role": "user", "content": body.content}, {"role": "assistant", "content": _early_msg}],
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
        lc_seed.append(AIMessage(content=msg))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[{"role": "assistant", "content": msg}],
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
    _portfolio_ready = _sample_status == "full_sample"
    _effective_fidelity = _current_fidelity
    _analysis_task_in_flight = False
    if not _mode_switch_message:
        if _current_fidelity == FIDELITY_PORTFOLIO and not _portfolio_ready:
            _effective_fidelity = FIDELITY_QUICK
        elif _current_fidelity == FIDELITY_ENTIRE:
            _analysis_task_in_flight = _scheduled_task_is_in_flight(primary_sess.get("analysis_task_id"))
            if _analysis_task_in_flight:
                _effective_fidelity = FIDELITY_PORTFOLIO if _portfolio_ready else FIDELITY_QUICK

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
        lc_seed.append(AIMessage(content=msg))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, {"role": "assistant", "content": msg}],
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
        lc_seed.append(AIMessage(content=msg))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, {"role": "assistant", "content": msg}],
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
        lc_seed.append(AIMessage(content=msg))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, {"role": "assistant", "content": msg}],
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
                cur.update(primary_sess)
                cur.pop("original_dataset_snapshot", None)

            await update_session(primary_sid, _merge_restored)
            msg = (
                "Restored the original uploaded dataset. "
                "Your next questions will run against the full dataset again."
            )
        else:
            msg = "You're already working with the original uploaded dataset."
        lc_seed = THREAD_META[thread_id].get("lc_msgs", [])
        lc_seed.append(HumanMessage(body.content))
        lc_seed.append(AIMessage(content=msg))
        THREAD_META[thread_id]["lc_msgs"] = _limit_messages(_compact_messages(lc_seed), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)
        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, {"role": "assistant", "content": msg}],
            datasets=[{"dataset_id": x[0], "filename": x[2].get("filename"), "alias": x[2].get("alias")} for x in dataset_sessions],
            active_dataset_ids=[dsid for dsid, _, _ in dataset_sessions],
            analysis_fidelity=_current_fidelity,
            selected_sample_name=primary_sess.get("selected_sample_name"),
            execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            analysis_task_id=primary_sess.get("analysis_task_id"),
        )

    # ----------------------------------------------------
    # 5) History (LangChain messages in memory)
    # ----------------------------------------------------
    tmeta = THREAD_META[thread_id]
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
        lc_msgs.append(AIMessage(content=msg))
        tmeta["lc_msgs"] = _limit_messages(_compact_messages(lc_msgs), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)

        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, {"role": "assistant", "content": msg}],
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
        lc_msgs.append(AIMessage(content=msg))
        tmeta["lc_msgs"] = _limit_messages(_compact_messages(lc_msgs), MAX_CONTEXT_TURNS)
        await persist_thread_history(thread_id)

        return ChatResponse(
            messages=[{"role": "user", "content": body.content}, {"role": "assistant", "content": msg}],
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
            # Don't raise — just quietly downgrade to local


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
    # Fallback: the db:// sample session is orphaned from the chat thread's dataset
    # group, so a "blind" transfer often lands on a non-DB (e.g. CSV) session. When the
    # active session yields NO source (no DB customer_id AND no cloud connection_id),
    # use the user's most-recent active database so D2D/C2D still find the DB they were
    # analyzing. Gated on the absence of a cloud connection so cloud sources are never
    # overridden.
    if not _active_db_customer_id and not (primary_sess.get("connection_id") or "").strip():
        try:
            _fallback = await find_active_db_customer_for_user(user_id)
        except Exception as _e:
            _fallback = None
            logger.warning("DTA source bridge: active-DB fallback failed: %s", _e)
        if _fallback:
            _active_db_customer_id, _active_db_source_table = _fallback
            logger.info(
                "DTA source bridge: active session had no source; fell back to the "
                "user's most-recent active database customer_id=%r table=%r",
                _active_db_customer_id, _active_db_source_table,
            )
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

    # ----------------------------------------------------
    # 11) Build state for LangGraph (now multi-dataset aware)
    # ----------------------------------------------------
    state_in: Dict[str, Any] = {
        "user_id": user_id,
        "session_id": sid,
        "thread_id": thread_id,
        "messages": lc_msgs,
        "sample_data": _resolved_rows or _fallback_preview or [],
        "uploaded_csv_columns": _maybe_json_load(primary_sess.get("uploaded_csv_columns"))
        or primary_sess.get("uploaded_csv_columns"),
        "uploaded_csv_preview": _resolved_rows or _fallback_preview or [],
        "schema": schema_for_state,
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
        "output_file_data": None,
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
            # ── existing sample-upload logic below, unchanged ──
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
            # Fallback: run locally against the sample CSV.
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
            need_full=(_effective_fidelity == FIDELITY_ENTIRE),
            rows_override=(_resolved_rows or _fallback_preview or None),
        )



    # # ----------------------------------------------------
    # # 12) Run workflow graph
    # # ----------------------------------------------------
    bound_sid = group_sid          # thread is bound to the group session (step 2)
    _turn_holder: Dict[str, Any] = {}

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

        final = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: GRAPH.invoke(
                state_in,
                config={"configurable": {"thread_id": f"ui:{thread_id}"}},
            ),
        )

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
            if k in state_in and k in primary_sess and state_in[k] != primary_sess.get(k):
                primary_sess[k] = _jsonify(state_in[k])
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
            await update_session(primary_sid, lambda cur: cur.update(primary_sess))

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
        if not final.get("output_location"):
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
            blocks = re.findall(r"(?:^\|.*\|\s*\n?)+", ai_msg.content, flags=re.MULTILINE)
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
            execution_status in ("success", "succeeded", "dryrun")
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

        # ── Byte-gated deferral: the response is a deterministic job card, never
        #    LLM prose. The summarizer was bypassed in the graph (D-01), and the
        #    payload must carry NO analytical numbers — clear outputs
        #    unconditionally and replace the assistant message wholesale.
        if _byte_defer_active:
            if task_id:
                _job_card = _build_deferred_job_card(
                    task_id=str(task_id),
                    dataset_name=str(primary_sess.get("alias") or primary_sess.get("filename") or primary_dsid),
                    interpreted_question=str(final.get("plan") or body.content),
                    eta_hint=_estimated_runtime_hint(_file_size_bytes),
                )
            else:
                _sched_err = final.get("execution_error") or "the scheduler did not return a job id"
                _job_card = (
                    "⚠️ The full-dataset job could not be scheduled "
                    f"({_sched_err}). No partial or sampled results are shown for "
                    "datasets above the in-memory analysis budget — please retry."
                )
            ai_msg = AIMessage(content=_job_card)
            output_json = None
            output_file_data = None
            # Thread history was already persisted (section 17) with the graph's
            # raw scheduler message — rewrite it so refresh/history shows the same
            # deterministic job card the user saw.
            try:
                _lc = THREAD_META[thread_id].get("lc_msgs", [])
                if _lc and isinstance(_lc[-1], AIMessage):
                    _lc[-1] = AIMessage(content=_job_card)
                else:
                    _lc.append(AIMessage(content=_job_card))
                THREAD_META[thread_id]["lc_msgs"] = _lc
                await persist_thread_history(thread_id)
            except Exception as exc:
                logger.warning("[byte_routing] failed to rewrite thread history with job card: %s", exc)

        if produced_fresh_output:
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
                        cur.update(primary_sess)
                        for k in training_state_keys:
                            cur.pop(k, None)

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
                ai_msg = AIMessage(content=f"Model training failed: {_err_text}")
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
        _ran_on_partial_sample = produced_fresh_output and (
            (_cloud_origin_session and _effective_fidelity != FIDELITY_ENTIRE)
            or _entire_on_head
            or (
                _effective_fidelity == FIDELITY_QUICK
                and (_is_large_dataset or _sample_n >= DEFAULT_SAMPLE_MAX_ROWS)
            )
            or (_effective_fidelity == FIDELITY_PORTFOLIO and _is_large_dataset)
        )
        _assistant_text = ai_msg.content if ai_msg else None
        if _ran_on_partial_sample and isinstance(_assistant_text, str) and _assistant_text.strip():
            _assistant_text = _assistant_text.rstrip() + _sample_fidelity_note(
                _effective_fidelity, _resolved_rows, primary_sess.get("selected_sample_name"),
            )

        resp = ChatResponse(
            messages=[{"role": "user", "content": body.content}]
            + ([{"role": "assistant", "content": _assistant_text}] if ai_msg else []),
            reasoning=final.get("reasoning_trace"),
            planner_definition=final.get("planner_definition", {}) or {},
            ready_to_summarize=ready_to_summarize,
            ready_to_code=ready_to_code,
            coder_definition=final.get("coder_definition", {}) or {},
            output_file_data=output_file_data,
            output_json=output_json,
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
            execution_context=_build_execution_context(_current_fidelity, primary_sess.get("selected_sample_name")),
            analysis_task_id=(task_info or {}).get("task_id"),
            integrity_report=final.get("integrity_report"),
            integrity_safe_to_train=final.get("integrity_safe_to_train"),
            evaluation_report=final.get("evaluation_report"),
            evaluation_beats_baseline=final.get("evaluation_beats_baseline"),
            agent_errors=final.get("agent_errors"),
            verification_safe_to_present=final.get("verification_safe_to_present"),
            verification_report=final.get("verification_report"),
        )

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
            await save_session(primary_sid, primary_sess)

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
                    analysis_label       = (body.content or "")[:60],
                )
            )

        # If this turn overran the sync window, the HTTP response already went
        # back as a "running" handle — stash the finished result for polling.
        if (time.monotonic() - _t0) >= TURN_SYNC_DEADLINE_S:
            try:
                await update_session(bound_sid, _pending_turn_done(deferred_id, resp.model_dump()))
            except Exception:
                logger.warning("[deferred] failed to stash finished turn %s", deferred_id, exc_info=True)

        return resp

    # ---- race the turn against the soft deadline ----
    _turn_task = asyncio.create_task(_run_graph_and_finalize())
    _done, _ = await asyncio.wait({_turn_task}, timeout=TURN_SYNC_DEADLINE_S)

    if _turn_task in _done:
        # Fast turn (all analysis, and any training that fit): identical to before.
        resp = _turn_task.result()          # re-raises HTTPException/errors in-turn
    else:
        # Slow turn (model training): return a handle before the proxy aborts.
        _deferred_turn_tasks.add(_turn_task)
        _deferred_id = _turn_holder.get("deferred_id") or _new_id()

        def _cleanup(t: asyncio.Task, _did=_deferred_id):
            _deferred_turn_tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                logger.error("[deferred] turn %s failed: %s", _did, t.exception())
                asyncio.create_task(
                    update_session(bound_sid, _pending_turn_error(_did, "Training failed. Please try again."))
                )
        _turn_task.add_done_callback(_cleanup)

        await update_session(bound_sid, _pending_turn_running(_deferred_id))

        resp = ChatResponse(
            messages=[
                {"role": "user", "content": body.content},
                {"role": "assistant", "content":
                    "Model training is running — this can take a few minutes. "
                    "The results will appear here automatically when it's done."},
            ],
            training_status="running",
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
    if not sid or not sess or sess.get("user_id") != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread_id")

    pt = sess.get("pending_turn")
    pt = _maybe_json_load(pt) if isinstance(pt, str) else pt
    if not isinstance(pt, dict):
        return {"status": "none"}
    if deferred_id and pt.get("id") != deferred_id:
        return {"status": "superseded"}   # a newer turn replaced this one

    status_val = pt.get("status")
    if status_val == "done":
        return {"status": "done", "result": pt.get("result") or {}}
    if status_val == "error":
        return {"status": "error", "message": pt.get("message") or "Training failed."}
    return {"status": "running"}

@app.get("/threads/{thread_id}/planner-graph")
async def get_planner_graph_image(request: Request, thread_id: str):
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
            "last_run_at": entry.last_run_at.isoformat() if entry.last_run_at else None,
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
    
    tasks_data = list(
        filter(
            lambda task: max_timestamp == 0 or task[1].timestamp() < max_timestamp, 
            AvalokaScheduler.get_task_ids_with_timestamp(redbeat_key, False)
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
        "last_run_at": entry.last_run_at.isoformat() if entry.last_run_at else None
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
        return {"status": "CANCELLED", "traceback": str(task.traceback), "next_due_at": next_due_at, **task_metadata}
    elif task.failed():
        if sess.get("analysis_task_id") == scheduled_task_id:
            await update_session(sid, _clear_scheduled_task)
        return {"status": "FAILED", "traceback": str(task.traceback), "next_due_at": next_due_at, **task_metadata}
    elif task.state not in ("SUCCESS",):
        return {
            "status": task.state,
            "message": task_metadata["running_message"],
            "next_due_at": next_due_at,
            **task_metadata,
        }
    else:
        final = task.result or {}
        final.update(
            messages=convert_message_dicts_to_objects(final.get("messages", []))
        )
        assistant_messages = _assistant_messages_from_state(final)
        output_file_data, output_json = get_output_from_state(final)
        execution_result = final.get("execution_result") or {}
        execution_succeeded = str(execution_result.get("status", "")).lower() in (
            "success", "succeeded", "completed", "done"
        )
        if _scheduled_task_has_chat_result_only(task_metadata) or (
            not execution_succeeded and not final.get("output_file_data")
        ):
            output_file_data = None
            output_json = None
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

        completion_message = _scheduled_task_completion_message(final, assistant_messages, task_metadata)
        return {
            "status": "SUCCESS",
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

    # Prefer the durable table (survives result_expires=3600).
    durable = await _load_scheduled_runs(task_id)
    if durable is not None:
        return durable

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
        if state == "SUCCESS":
            final = task.result or {}
            if isinstance(final, dict):
                final = dict(final)
                final["messages"] = convert_message_dicts_to_objects(final.get("messages", []))
                assistant_messages = _assistant_messages_from_state(final)
                output_file_data, output_json = get_output_from_state(final)
                exec_result = final.get("execution_result") or {}
                exec_ok = str(exec_result.get("status", "")).lower() in (
                    "success", "succeeded", "completed", "done"
                )
                if _scheduled_task_has_chat_result_only(task_metadata) or (
                    not exec_ok and not final.get("output_file_data")
                ):
                    output_file_data = None
                    output_json = None
                result_payload = {
                    "output_json": output_json,
                    "output_file_data": output_file_data,
                    "message": _scheduled_task_completion_message(final, assistant_messages, task_metadata),
                    "logs": _logs_from_final(final),
                    "ray_job": _ray_job_from_final(final),
                }
                if task_metadata.get("task_type") == "training":
                    run_status, error = _run_status_from_final(final, task_type="training")
                    if run_status == "failure":
                        result_payload["logs"] = None
        elif state == "FAILURE":
            raw_error = task.traceback or task.result or "Task failed"
            error = (
                format_training_failure(build_training_failure(raw_error))
                if task_metadata.get("task_type") == "training"
                else str(raw_error)
            )
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

    return out

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
    if AsyncResult(task_id, app=celery_app).state in ("FAILURE", "SUCCESS", "REVOKED"):
        last_run_at = AvalokaScheduler.get_last_run_at(redbeat_key, generate_key=False)
        return StreamingResponse(
            sse_iter(entry.schedule.remaining_estimate(last_run_at).total_seconds()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        task_sse_iter(task_id, redbeat_key),
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
    msgs = await read_thread_msgs(thread_id)
    if not sid and msgs:
        # History with no session binding has no owner to authorize against;
        # serving it would hand one user's chat to any authenticated caller.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread_id")
    return {"messages": msgs}


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

    if not sids:
        return []  # genuinely no sessions for this user

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
    items.sort(key=lambda it: it.created_at or "", reverse=True)

    return items

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
            local_input = sample_input
            sess_for_ds["work_local_input"] = _posix(sample_input)
            ok = await save_session(session_id, sess_for_ds)
            if not ok:
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "Failed to persist session while preparing preview",
                )
            logger.info(
                "[preview] using sample_local_input fallback for dataset %s -> %s",
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
                if _ar.ready() and _ar.successful():
                    _preview_sample_status = "full_sample"
                    sess_for_ds["sample_status"] = "full_sample"
                    sess_for_ds["profiling_status"] = "full_profile"
                    _celery_res = _ar.result or {}
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

    return DatasetPreviewResponse(
        dataset_id=dataset_id,
        schema=schema_any,
        samples=_jsonify(samples),
        ddl_schema=ddl,
        rows_sampled=len(samples),
        session_id=session_id,
        thread_id=thread_id,
        portfolio_samples=meta.get("portfolio_samples"),
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
    )
     

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
    if not r.ready():
        return {
            "state": "RUNNING",
            "sample_status": (sess or {}).get("sample_status"),
            "profiling_status": (sess or {}).get("profiling_status"),
            "background_task_id": (sess or {}).get("background_task_id"),
            "analysis_task_id": (sess or {}).get("analysis_task_id"),
            "analysis_fidelity": (sess or {}).get("analysis_fidelity"),
            "selected_sample_name": (sess or {}).get("selected_sample_name"),
        }
    if r.successful():
        res = r.result or {}
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
        for idx, table in enumerate(tables):
            result = await _run_mcp_db_query(customer_id, api_key, f"SELECT * FROM {table} LIMIT {limit}")
            if not isinstance(result, dict) or result.get("error") or "columns" not in result or "rows" not in result:
                raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                    f"Failed to read '{table}': {(result or {}).get('error', 'no rows')}")
            cols = result.get("columns") or []
            rows = result.get("rows") or []
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

            schema_dict = {c: "string" for c in cols}
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
                columns=cols, rows=_jsonify(dict_rows),
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

    customer_id = body.customer_id
    api_key = body.api_key

    test_result = await list_database_tables(customer_id, api_key)
    if "error" in test_result:
        raise HTTPException(
            400,
            f"Database connection failed: {test_result['error']}",
        )

    session_id = _new_id()
    # Persist the api_key encrypted at rest so /api/v1/database/query can reuse
    # this session without the client resending raw credentials on every call.
    session_data = {
        "user_id": user_id,
        "type": "database",
        "customer_id": customer_id,
        "database_info": test_result.get("meta", {}),
        "created_at": _now_iso(),
    }
    api_key_enc = encrypt_secret(api_key)
    if api_key_enc:
        session_data["api_key_enc"] = api_key_enc
    if not await save_session(session_id, session_data):
        raise HTTPException(500, "Failed to persist DB session")
    return {
        "session_id": session_id,
        "customer_id": customer_id,
        "status": "connected",
        "credentials_saved": bool(api_key_enc),
        "tables_available": len(test_result.get("tables", [])),
        "database_type": test_result.get("meta", {}).get("database_type"),
    }


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

    customer_id = body.customer_id or body.metadata.get("customer_id")
    api_key = (body.metadata or {}).get("api_key")

    # Fall back to credentials saved by /api/database/connect so a client that
    # followed the connect-then-query flow doesn't resend raw creds each call.
    if not (customer_id and api_key):
        sid = _resolve_session_id(
            request, body.session_id or body.metadata.get("session_id")
        )
        db_sess = await get_session(sid) if sid else None
        if (
            db_sess
            and db_sess.get("user_id") == user_id
            and db_sess.get("type") == "database"
        ):
            customer_id = customer_id or db_sess.get("customer_id")
            if not api_key:
                api_key = decrypt_secret(db_sess.get("api_key_enc"))

    if not (customer_id and api_key):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Missing database credentials. Provide customer_id and api_key in the "
            "request body, or call /api/database/connect first and pass its "
            "session_id.",
        )

    content_lower = (body.content or "").lower().strip()
    tool_name = "query"
    arguments: Dict[str, Any] = {"sql": (body.content or "").strip()}
    if content_lower in ["list tables", "show tables"]:
        tool_name = "list_tables"
        arguments = {}
    elif content_lower.startswith("describe table"):
        tool_name = "describe_table"
        arguments = {"table_name": body.content.strip().split()[-1]}

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
                {"role": "user", "content": body.content},
                {"role": "assistant", "content": msg},
            ],
            query_result={"error": msg},
        )

    out = DatabaseChatResponse(
        messages=[{"role": "user", "content": body.content}],
        query_result=result,
        database_info={
            "customer_id": customer_id,
            "query_executed": (body.content or "").strip(),
        },
    )

    if (
        tool_name == "query"
        and isinstance(result, dict)
        and "columns" in result
        and "rows" in result
    ):
        cols: List[str] = result.get("columns") or []
        rows: List[List[Any]] = result.get("rows") or []

        if cols and rows:
            dsid, sid, tid = await _create_or_update_db_sample_session(
                request=request,
                user_id=user_id,
                customer_id=customer_id,
                columns=cols,
                rows=rows,
                preferred_dataset_id=body.dataset_id,
                source_table=_source_table_from_sql(body.content),
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


            out.messages.append(
                {
                    "role": "assistant",
                    "content": "Query executed. Sample materialized for preview & chat.",
                }
            )
        else:
            out.messages.append(
                {
                    "role": "assistant",
                    "content": "Query executed successfully (no rows).",
                }
            )

    else:
        out.messages.append(
            {
                "role": "assistant",
                "content": f"{tool_name} executed successfully.",
            }
        )

    return out


# -------------------------------------------------------------------
# Get trained model list with metadata
# -------------------------------------------------------------------
@app.get("/api/models")
async def list_models(request: Request):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Session storage (Redis) is not available.")
    if not storage_service.blob_store:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Storage backend not initialized.")
    
    from app.agents.mta_v2.mlflow_manager import MLflowManager
    mlflow_manager = MLflowManager()
    models = await asyncio.to_thread(mlflow_manager.list_models)
    result = {
        "models": [
            model for model in models if model.get("user_id") == user_id
        ]
    }

    return result

# -------------------------------------------------------------------
# Get model details
# -------------------------------------------------------------------
@app.get("/api/models/{run_id}")
async def get_model_details(request: Request, run_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Session storage (Redis) is not available.")
    if not storage_service.blob_store:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Storage backend not initialized.")

    
    from app.agents.mta_v2.mlflow_manager import MLflowManager
    mlflow_manager = MLflowManager()
    model = await asyncio.to_thread(mlflow_manager.get_model_details, run_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found.")
    elif model.get("user_id") != user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You do not have permission to access this model.")
    return model

# -------------------------------------------------------------------
# Delete model by the run_id (MLflow's run_id)
# -------------------------------------------------------------------
@app.delete("/api/models/{run_id}")
async def delete_model(request: Request, run_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Session storage (Redis) is not available.")
    if not storage_service.blob_store:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Storage backend not initialized.")

    from app.agents.mta_v2.mlflow_manager import MLflowManager
    mlflow_manager = MLflowManager()
    run_details = await asyncio.to_thread(mlflow_manager.get_model_details, run_id)
    if run_details is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found.")
    elif run_details.get("user_id") != user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You do not have permission to delete this model.")
    else:
        if run_details.get("inference_service_details") is not None:
            from app.agents.mta_v2.inference_service_manager import InferenceServiceManager
            inferencer = InferenceServiceManager()
            await asyncio.to_thread(inferencer.stop_inference_service, run_id)
        await asyncio.to_thread(mlflow_manager.delete_model, run_id)
    return {"status": "deleted"}

# -------------------------------------------------------------------
# Inference through the configured service/gateway
# -------------------------------------------------------------------
@app.post("/api/models/{run_id}/inference")
async def inference(
    request: Request,
    run_id: str,
    input_data: Union[Dict[str, Any], List[Dict[str, Any]]],
):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Session storage (Redis) is not available.")
    if not storage_service.blob_store:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Storage backend not initialized.")

    from app.agents.mta_v2.mlflow_manager import MLflowManager
    mlflow_manager = MLflowManager()
    run_details = await asyncio.to_thread(mlflow_manager.get_model_details, run_id)
    if run_details is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found.")
    elif run_details.get("user_id") != user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You do not have permission to use this model.")
    if not run_details.get("inference_service_details"):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Inference service is not configured for this model.",
        )
    if isinstance(input_data, list) and (
        not input_data or not all(isinstance(row, dict) for row in input_data)
    ):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Batch inference requires a non-empty JSON array of row objects.",
        )
    try:
        from app.agents.mta_v2.inference_service_manager import InferenceServiceManager

        manager = InferenceServiceManager()
        result = await asyncio.to_thread(manager.inference, run_id, input_data)
        return {"result": result}
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    except Exception as exc:
        logger.error("Inference gateway failed for run '%s': %s", run_id, exc, exc_info=True)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Inference gateway failed: {exc}")


# -------------------------------------------------------------------
# Configure Inference Service
# -------------------------------------------------------------------
@app.post("/api/models/{run_id}/configure-inference-service")
async def configure_inference_service(request: Request, run_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Session storage (Redis) is not available.")
    if not storage_service.blob_store:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Storage backend not initialized.")

    from app.agents.mta_v2.mlflow_manager import MLflowManager
    mlflow_manager = MLflowManager()
    run_details = await asyncio.to_thread(mlflow_manager.get_model_details, run_id)
    if run_details is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found.")
    elif run_details.get("user_id") != user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You do not have permission to delete this model.")
    else:
        from app.agents.mta_v2.inference_service_manager import InferenceServiceManager
        inference_service_manager = InferenceServiceManager()
        try:
            details = await asyncio.to_thread(
                inference_service_manager.configure_inference_service,
                run_id,
            )
            return {"status": "running", "inference_service_details": details}
        except Exception as e:
            logger.error(f"Error occurred while configuring inference service - {e}")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Failed to configure inference service: {e}")


# -------------------------------------------------------------------
# Stop Inference Service
# -------------------------------------------------------------------
@app.post("/api/models/{run_id}/stop-inference-service")
async def stop_inference_service(request: Request, run_id: str):
    user_id = _resolve_user_id(request)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
    if not session_service.cache:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Session storage (Redis) is not available.")
    if not storage_service.blob_store:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Storage backend not initialized.")

    from app.agents.mta_v2.mlflow_manager import MLflowManager
    mlflow_manager = MLflowManager()
    run_details = await asyncio.to_thread(mlflow_manager.get_model_details, run_id)
    if run_details is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found.")
    elif run_details.get("user_id") != user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You do not have permission to delete this model.")
    else:
        from app.agents.mta_v2.inference_service_manager import InferenceServiceManager
        inference_service_manager = InferenceServiceManager()
        try:
            await asyncio.to_thread(inference_service_manager.stop_inference_service, run_id)
            return {"status": "stopped"}
        except Exception as e:
            logger.error(f"Error occurred while stopping inference service - {e}")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Failed to stop inference service: {e}")


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
    source_table: Optional[str] = None,
) -> Tuple[str, str, str]:
    sample_rows = rows[:MAX_DB_SAMPLE_ROWS]
    dict_rows = _rows_list_to_dicts(columns, sample_rows)

    dsid = preferred_dataset_id or _new_id()
    sid = (
        await find_session_by_dataset_for_user(dsid, user_id)
        if preferred_dataset_id
        else None
    )
    sess = await get_session(sid) if sid else None

    if sess and sess.get("work_dir") and Path(sess["work_dir"]).exists():
        work_dir = Path(sess["work_dir"])
        session_id = sid
    else:
        session_id = _new_id()
        work_dir = (TMP_ROOT / session_id).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

    sample_input_path = work_dir / "sample_input.csv"
    _write_rows_to_csv(dict_rows, sample_input_path)

    output_path = work_dir / "output.csv"

    schema_dict = {c: "string" for c in columns}
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
        "user_id": user_id,
        "dataset_id": dsid,
        "work_dir": str(work_dir),
        "data_source_location": f"db://{customer_id}/{dsid}?sample=true",
        # Persist the source database explicitly so a DTA transfer can use it as the
        # implicit source without having to re-parse the db:// URI (see the source
        # bridge in send_message). `created_at` lets the bridge pick the most-recent
        # active database when it has to fall back to a user-session scan.
        "customer_id": customer_id,
        # The table the user analyzed (parsed from their query), so a DTA transfer
        # uses it as the implicit source instead of the auto-detected first table.
        "source_table": source_table,
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
        saved_tid = sess.get("thread_id") if sess else None
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
    analysis_id: str,             # unused — sharing is project-level in this schema
    profile_id: str,              # unused now, kept for signature stability
    project_id: Optional[str],
    user_id: Optional[str] = None,   # auth.uid(), needed to resolve app_users.id
) -> bool:
    """
    True if the requester is a collaborator on the PROJECT this analysis belongs
    to. resource_collaborators only carries resource_type='project' (verified),
    and app_user_id references app_users(id) — so translate the requester's
    auth.uid() -> app_users.id and compare against that.
    """
    if not project_id or not user_id:
        return False

    app_user_id = await _resolve_app_user_id(user_id)
    if not app_user_id:
        return False

    def _q() -> bool:
        client = get_supabase_client()
        try:
            res = (client.table("resource_collaborators")
                   .select("id")
                   .eq("resource_type", "project")
                   .eq("resource_id", project_id)
                   .eq("app_user_id", app_user_id)
                   .limit(1).execute())
            return bool(getattr(res, "data", None))
        except Exception as exc:
            logger.warning("[collab] resource_collaborators check failed (project=%s): %s",
                           project_id, exc)
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

    if not row or not row.get("session_id"):
        return None

    owner_id = str(row.get("owner_id") or "")

    # analyses.owner_id is not consistently populated: some rows store the
    # owner's profiles.id, others store the raw auth.uid() (the value the JWT
    # carries directly). Accept an owner match against EITHER id space so a
    # user reopening their own analysis is never misclassified as "not shared".
    if owner_id and owner_id == str(user_id):
        return {**row, "role": "owner"}

    profile_id = await _resolve_profile_id(user_id)   # auth.uid() -> profiles.id
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
        logger.warning("[snapshot] persist failed for thread %s: %s", thread_id, exc)
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
        if _is_supabase_auth_failure(exc):
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
    Re-execute the latest stored code for an analysis (Leela: 'refresh always
    executes latest code'). Runs the stored .py directly through the execution
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

    final = await asyncio.get_event_loop().run_in_executor(None, lambda: node(state_in))

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

    checkpoint_created = bool(
        checkpoint_created
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
    # (advanced escape hatch); the planner review always runs.
    run_validation: Optional[bool] = True


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
                "validation_feedback": validation.get("feedback"),
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
    final = await asyncio.get_event_loop().run_in_executor(None, lambda: node(state_in))
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
            "execution_message": _exec_result.get(
                "message",
                "Execution did not produce fresh output",
            ),
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
            analysis_label=last_prompt[:60] if last_prompt else "",
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

SUPABASE_MCP_CONN_TABLE = os.getenv("SUPABASE_MCP_CONNECTIONS_TABLE", "mcp_connections")

_MCP_SECRET_FIELDS = ("api_key", "username")


class McpCredentialsIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    api_key: Optional[str] = Field(None, alias="apiKey")
    username: Optional[str] = None


def _mcp_conn_select(connection_id: str) -> Optional[Dict[str, Any]]:
    client = get_supabase_client()
    res = (
        client.table(SUPABASE_MCP_CONN_TABLE)
        .select(
            "id,user_id,api_key_ciphertext,api_key_iv,"
            "username_ciphertext,username_iv"
        )
        .eq("id", connection_id)
        .limit(1)
        .execute()
    )
    data = getattr(res, "data", None) or []
    return data[0] if data else None


def _mcp_conn_update(connection_id: str, patch: Dict[str, Any]) -> None:
    client = get_supabase_client()
    client.table(SUPABASE_MCP_CONN_TABLE).update(patch).eq("id", connection_id).execute()


async def _mcp_conn_owned_row(connection_id: str, user_id: str) -> Dict[str, Any]:
    """Fetch an mcp_connections row and enforce ownership (service role bypasses RLS)."""
    try:
        row = await asyncio.to_thread(_mcp_conn_select, connection_id)
    except Exception as exc:
        logger.error("[mcp-creds] lookup failed for %s: %s", connection_id, exc, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Connection lookup failed")
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Connection not found")
    if str(row.get("user_id") or "") != str(user_id):
        logger.warning(
            "[mcp-creds] connection %s owned by %s != requester %s",
            connection_id, row.get("user_id"), user_id,
        )
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Connection belongs to another user")
    return row


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

    await _mcp_conn_owned_row(connection_id, user_id)

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
 
# -------------------------------------------------------------------
# Local runner
# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run(
        "app.api.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8010")),
        reload=DEBUG,
        proxy_headers=True,
    )
