"""Native-Python data-source discovery used as fallback when openclaw is
unavailable.

Reuses Avaloka's existing primitives so we don't add new heavy
dependencies:
- Schema/profile already produced by ``/api/upload`` and
  ``/api/register-existing-storage`` lives in the Redis session.
- DB introspection is provided by ``app.mcp_server`` (already in the repo).

Each function returns a JSON-serializable dict and never raises — on
failure it returns ``{"error": "..."}``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_COLUMNS = 64
_MAX_TABLES = 50


def _truncate_columns(cols: Any) -> List[str]:
    if not cols:
        return []
    if isinstance(cols, dict):
        cols = list(cols.keys())
    if not isinstance(cols, list):
        return []
    return [str(c) for c in cols[:_MAX_COLUMNS]]


def discover_dataset(redis_context: Dict[str, Any]) -> Dict[str, Any]:
    """Read whatever has already been profiled for the current dataset.

    This is the primary fallback path: in 95% of cases the connect endpoint
    has already produced schema + quick profile, so the agent can answer
    "what's in this dataset?" without calling out anywhere.
    """
    if not redis_context:
        return {"error": "no thread session bound", "source": "native"}
    return {
        "source": "native",
        "dataset_id": redis_context.get("dataset_id"),
        "data_source": redis_context.get("data_source_location"),
        "file_type": redis_context.get("input_data_type"),
        "file_size_mb": redis_context.get("file_size_mb"),
        "columns": _truncate_columns(redis_context.get("schema_columns")),
        "profiling_summary": redis_context.get("profiling_summary"),
        "available_samples": redis_context.get("available_samples"),
        "current_fidelity": redis_context.get("analysis_fidelity"),
    }


def discover_database(connection_id: Optional[str]) -> Dict[str, Any]:
    """List tables in the connected database via the existing MCP server.

    Falls back gracefully when the MCP module is unavailable (some local
    dev environments).
    """
    if not connection_id:
        return {"error": "missing connection_id", "source": "native"}
    try:
        # Lazy import: app.mcp_server may carry heavier transitive deps.
        from app.mcp_server.client import list_tables as mcp_list_tables  # type: ignore
    except Exception as exc:
        logger.debug("avaloka_agent: MCP client unavailable: %s", exc)
        return {
            "error": "MCP server client not available; use the database "
                     "connect endpoint to introspect tables.",
            "source": "native",
        }
    try:
        tables = mcp_list_tables(connection_id)
        if isinstance(tables, dict) and "tables" in tables:
            tables = tables["tables"]
        return {
            "source": "native",
            "connection_id": connection_id,
            "tables": list(tables or [])[:_MAX_TABLES],
        }
    except Exception as exc:
        logger.warning("avaloka_agent: native db discovery failed: %s", exc)
        return {"error": str(exc), "source": "native"}


def discover_object_store(uri: Optional[str], connection_id: Optional[str]) -> Dict[str, Any]:
    """List a small head of objects under the given cloud URI.

    Uses ``daft`` if available; otherwise just echoes the URI.
    """
    if not uri:
        return {"error": "missing uri", "source": "native"}
    info: Dict[str, Any] = {
        "source": "native",
        "uri": uri,
        "connection_id": connection_id,
        "objects": [],
    }
    try:
        import daft  # type: ignore
    except ImportError:
        info["error"] = "daft not installed; cannot list cloud objects"
        return info
    try:
        # ``daft.from_glob_path`` returns a small DataFrame of file metadata.
        files_df = daft.from_glob_path(uri)
        files = files_df.limit(20).to_pylist()
        info["objects"] = [
            {"path": f.get("path"), "size": f.get("size")}
            for f in files
        ]
    except Exception as exc:
        info["error"] = f"daft listing failed: {exc}"
    return info


def discover(redis_context: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """High-level dispatcher used by the agent when it wants discovery.

    Picks the right fallback based on what's in state.
    """
    conn_id = state.get("connection_id")
    cloud_uri = state.get("data_source_location_cloud") or state.get("data_source_location")

    if redis_context.get("dataset_id"):
        return discover_dataset(redis_context)

    if cloud_uri and str(cloud_uri).lower().startswith(("s3://", "gs://", "gcs://", "az://", "azure://")):
        return discover_object_store(cloud_uri, conn_id)

    if conn_id and not cloud_uri:
        return discover_database(conn_id)

    return {"source": "native", "note": "no datasource bound to this thread"}
