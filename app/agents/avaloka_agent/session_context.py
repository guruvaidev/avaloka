"""Load condensed Redis thread/session context for the Avaloka agent.

Reads (best-effort, never raises) from the same Redis-backed session store
``app.services.session_service`` already uses for dataset metadata.

The output is a small dict capped at ~2KB so it can be safely injected into
a Groq prompt without inflating context.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_PROFILE_CHARS = 600
_MAX_DATASETS = 5
_MAX_TASKS = 5


def _trim(s: Any, n: int = _MAX_PROFILE_CHARS) -> str:
    text = "" if s is None else (s if isinstance(s, str) else json.dumps(s, default=str))
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def _summarize_session(blob: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_id": blob.get("dataset_id"),
        "data_source_location": blob.get("data_source_location"),
        "input_data_type": blob.get("input_data_type"),
        "schema_columns": (blob.get("uploaded_csv_columns") or [])[:32],
        "file_size_mb": blob.get("file_size_mb"),
        "analysis_fidelity": blob.get("analysis_fidelity"),
        "selected_sample_name": blob.get("selected_sample_name"),
        "available_samples": blob.get("available_samples"),
        "profiling_summary": _trim(blob.get("profiling_result")),
        "connection_event": blob.get("connection_event"),
        "last_infra_platform": blob.get("last_infra_platform"),
        "thread_id": blob.get("thread_id"),
    }


async def load_thread_context(thread_id: Optional[str]) -> Dict[str, Any]:
    """Load condensed context for one thread. Empty dict if Redis is down or
    the thread is unknown."""
    if not thread_id:
        return {}
    # Keep the conversational agent importable in lightweight CLI/test builds
    # that do not install Redis. The API image includes this dependency; when it
    # is absent, session enrichment simply degrades to the documented empty
    # context instead of preventing the graph from importing.
    try:
        from app.services import session_service
    except Exception as exc:
        logger.debug("avaloka_agent: session service unavailable: %s", exc)
        return {}
    try:
        sid = await session_service.get_thread_session(thread_id)
    except Exception as exc:
        logger.debug("avaloka_agent: get_thread_session failed: %s", exc)
        return {}
    if not sid:
        return {}
    try:
        sess = await session_service.get_session(sid)
    except Exception as exc:
        logger.debug("avaloka_agent: get_session failed: %s", exc)
        return {}
    if not sess:
        return {}
    out = _summarize_session(sess)
    out["session_id"] = sid
    return out


async def load_user_recent_datasets(user_id: Optional[str]) -> List[Dict[str, Any]]:
    """Best-effort list of recent datasets the user has connected.

    Returns at most ``_MAX_DATASETS`` lightweight summaries; may be empty.
    """
    try:
        from app.services import session_service
    except Exception:
        return []
    if not user_id or not session_service.cache:
        return []
    try:
        ds_ids = await session_service._user_datasets(user_id)
    except Exception:
        return []
    return [{"dataset_id": d} for d in ds_ids[:_MAX_DATASETS]]


def render_for_prompt(ctx: Dict[str, Any]) -> str:
    """Compact, deterministic string rendering for prompt injection."""
    if not ctx:
        return "none"
    lines: List[str] = []
    if ctx.get("dataset_id"):
        lines.append(f"current_dataset_id: {ctx['dataset_id']}")
    if ctx.get("data_source_location"):
        lines.append(f"data_source: {ctx['data_source_location']}")
    if ctx.get("input_data_type"):
        lines.append(f"file_type: {ctx['input_data_type']}")
    if ctx.get("file_size_mb") is not None:
        lines.append(f"file_size_mb: {ctx['file_size_mb']}")
    if ctx.get("schema_columns"):
        lines.append(f"columns: {', '.join(map(str, ctx['schema_columns'][:24]))}")
    if ctx.get("analysis_fidelity"):
        lines.append(f"current_fidelity: {ctx['analysis_fidelity']}")
    if ctx.get("selected_sample_name"):
        lines.append(f"selected_sample: {ctx['selected_sample_name']}")
    if ctx.get("available_samples"):
        lines.append(f"available_samples: {ctx['available_samples']}")
    if ctx.get("profiling_summary"):
        lines.append(f"profile: {_trim(ctx['profiling_summary'], 400)}")
    if ctx.get("connection_event"):
        lines.append(f"connection_event: {_trim(ctx['connection_event'], 200)}")
    if ctx.get("last_infra_platform"):
        lines.append(f"last_infra_platform: {ctx['last_infra_platform']}")
    return "\n".join(lines) if lines else "none"
