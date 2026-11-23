"""Catalog registration from the analysis path. Best-effort, like lineage.

The execution agent already calls ``record_analysis_lineage`` once an output
is validated; that function calls :func:`record_analysis` here with what it
resolved (the parent dataset ids, their metadata, the output and its id), so
there is one place that decides what an analysis touched and two stores that
hear about it. A catalog failure is logged once and never fails the analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from app.catalog.store import CatalogError, CatalogStore, CatalogUnavailable, normalise_columns

logger = logging.getLogger(__name__)

_FORMATS = {".csv": "csv", ".tsv": "tsv", ".json": "json", ".parquet": "parquet", ".xlsx": "xlsx",
            ".xls": "xls", ".avro": "avro", ".orc": "orc", ".xml": "xml"}


def _owner(state: Mapping[str, Any]) -> Optional[str]:
    user_id = state.get("user_id")
    return str(user_id).strip() or None if user_id else None


def _format(filename: str) -> Optional[str]:
    return _FORMATS.get(Path(str(filename)).suffix.lower()) if filename else None


def parent_registration(parent_id: str, metadata: Mapping[str, Any], state: Mapping[str, Any],
                        fallback_columns: Any = None) -> Dict[str, Any]:
    """What ``register`` is called with for one input dataset.

    A dataset that came through a connection is a ``db_table`` at that
    connection; anything else is a ``file`` (an upload). The locator names
    the upload or the table, never a path with a signed query string and
    never a DSN; those stay with the sealed connection.
    """
    connection_id = metadata.get("connection_id") or (state.get("connection_id") if metadata else None)
    filename = str(metadata.get("filename") or metadata.get("alias") or "")
    columns = normalise_columns(metadata.get("columns") or fallback_columns or [])
    if connection_id:
        locator: Dict[str, Any] = {"connection_id": str(connection_id), "dataset_id": parent_id}
        for key in ("schema", "table", "database"):
            if metadata.get(key):
                locator[key] = str(metadata[key])
        kind = "db_table"
    else:
        locator = {"dataset_id": parent_id}
        if filename:
            locator["filename"] = filename
            fmt = _format(filename)
            if fmt:
                locator["format"] = fmt
        kind = "file"
    source_version = metadata.get("content_hash") or metadata.get("etag") or None
    row_count = metadata.get("row_count") or metadata.get("rows")
    return {
        "kind": kind, "locator": locator, "columns": columns, "dataset_id": parent_id,
        "display_name": filename or metadata.get("table") or parent_id,
        "source_version": source_version,
        "row_count": int(row_count) if isinstance(row_count, (int, float)) and row_count >= 0 else None,
        "row_count_kind": "exact" if isinstance(row_count, (int, float)) else "unknown",
    }


def record_analysis(state: Mapping[str, Any], output_data: Any, *, output_id: str, analysis_id: str,
                    parents: List[str], metadata_by_id: Mapping[str, Mapping[str, Any]],
                    store: Optional[CatalogStore] = None) -> Optional[Dict[str, Any]]:
    """Register the inputs an analysis used (pinning their versions to it)
    and the output it produced. Returns what was registered, or ``None`` when
    the catalog is off, has no owner, or failed; never raises."""
    try:
        owner = _owner(state)
        if not owner:
            return None
        try:
            catalog = store or CatalogStore(owner)
        except (CatalogUnavailable, CatalogError) as exc:
            logger.debug("[catalog] not recording: %s", exc)
            return None
        registered: Dict[str, Any] = {"inputs": [], "output": None}
        fallback = state.get("uploaded_csv_columns") or []
        for parent_id in parents:
            spec = parent_registration(parent_id, metadata_by_id.get(parent_id, {}), state,
                                       fallback if parent_id == parents[0] else None)
            reg = catalog.register(analysis_id=analysis_id, **spec)
            registered["inputs"].append({"dataset_id": reg.dataset_id, "version_id": reg.version_id,
                                         "version_no": reg.version_no, "outcome": reg.outcome})
        dtypes = getattr(output_data, "dtypes", None)
        columns = normalise_columns(dtypes if dtypes is not None else [])
        file_data = state.get("output_file_data") or {}
        label = (file_data.get("filename") or Path(str(state.get("output_location") or "")).name
                 or "analysis output")
        try:
            rows: Optional[int] = int(len(output_data))
        except Exception:  # noqa: BLE001
            rows = None
        locator: Dict[str, Any] = {"analysis_id": analysis_id, "parent_dataset_id": parents[0] if parents else None}
        # Where the output file landed, so "use analysis:<id>"'s output can be
        # bound later (app/catalog/chat.py). A storage URI or path, never a
        # signed URL: check_locator refuses those at the door.
        output_location = state.get("output_location") or state.get("latest_output_location")
        if isinstance(output_location, str) and output_location.strip():
            locator["output_location"] = output_location.strip()
        reg = catalog.register(
            "derived", locator,
            columns, dataset_id=output_id, display_name=str(label), row_count=rows,
            row_count_kind="exact" if rows is not None else "unknown")
        registered["output"] = {"dataset_id": reg.dataset_id, "version_id": reg.version_id,
                                "version_no": reg.version_no, "outcome": reg.outcome}
        return registered
    except CatalogUnavailable as exc:
        logger.warning("[catalog] unavailable, analysis not registered: %s", exc)
        return None
    except Exception:  # noqa: BLE001
        logger.warning("[catalog] could not register an analysis", exc_info=True)
        return None


__all__ = ["parent_registration", "record_analysis"]
