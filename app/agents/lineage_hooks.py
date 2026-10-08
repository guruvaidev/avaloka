"""Best-effort lineage capture at dataset and model production boundaries."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.lineage import Edge, EdgeKind, LineageStore

logger = logging.getLogger(__name__)


def _tenant(state: Dict[str, Any]) -> Optional[str]:
    """Whose lineage this is.

    The store is one shared database, so nothing is written or read without an
    owner. ``user_id`` is the verified JWT subject and the only identity the
    graph state carries; the organisation a user belongs to lives in Supabase
    and never reaches the backend. The memory plane draws its tenant boundary
    the same way (app/services/db/postgres_client.py). No user, no lineage:
    there is no shared "default" bucket to fall into.
    """
    user_id = state.get("user_id")
    return str(user_id).strip() or None if user_id else None


def _store(state: Dict[str, Any]) -> LineageStore:
    return LineageStore(_tenant(state))


def _locations_match(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    if str(left) == str(right):
        return True
    if "://" in str(left) or "://" in str(right):
        return False
    return os.path.abspath(str(left)) == os.path.abspath(str(right))


def _pii_kinds(report: Any) -> Dict[str, str]:
    if not isinstance(report, dict):
        return {}
    return {
        str(finding["column"]): str(finding["kind"])
        for finding in report.get("findings", [])
        if isinstance(finding, dict)
        and finding.get("column")
        and finding.get("kind")
        and finding.get("kind") != "not_pii"
    }


def _columns(metadata: Dict[str, Any], pii: Dict[str, str], fallback: Any = None) -> List[Tuple[str, str]]:
    columns = metadata.get("columns") or fallback or []
    if isinstance(columns, dict):
        columns = [{"name": name, "dtype": dtype} for name, dtype in columns.items()]
    result: Dict[str, str] = {}
    for column in columns:
        if isinstance(column, dict):
            name = column.get("name") or column.get("column")
            dtype = column.get("dtype") or column.get("type") or "unknown"
        elif isinstance(column, (tuple, list)) and len(column) == 2:
            name, dtype = column
        else:
            name, dtype = column, "unknown"
        if name:
            result[str(name)] = str(dtype)
    for name in pii:
        result.setdefault(name, "unknown")
    return list(result.items())


def _dataset_parents(state: Dict[str, Any]) -> List[str]:
    latest_id = state.get("latest_output_dataset_id")
    input_location = (
        state.get("data_source_location")
        or state.get("active_data_source_location")
        or state.get("active_data_source_location_local")
    )
    latest_locations = (
        state.get("latest_output_location"),
        state.get("latest_output_location_local"),
        state.get("output_location"),
    )
    if latest_id and any(_locations_match(input_location, item) for item in latest_locations):
        return [str(latest_id)]

    ids = state.get("active_dataset_ids")
    if isinstance(ids, (list, tuple)) and ids:
        return list(dict.fromkeys(str(item) for item in ids if item))

    dataset_id = (
        state.get("active_dataset_id")
        or state.get("dataset_id")
        or state.get("cloud_dataset_id")
    )
    if not dataset_id and isinstance(state.get("dataset"), dict):
        dataset_id = state["dataset"].get("id") or state["dataset"].get("dataset_id")
    return [str(dataset_id)] if dataset_id else []


def _source_metadata(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    context = state.get("multi_dataset_state") or state.get("datasets_context") or []
    if not isinstance(context, list):
        return {}
    return {
        str(item["dataset_id"]): item
        for item in context
        if isinstance(item, dict) and item.get("dataset_id")
    }


def _state_pii_report(state: Dict[str, Any]) -> Dict[str, Any]:
    report = state.get("pii_report")
    if not isinstance(report, dict):
        report = (state.get("preparation_proposal") or {}).get("pii", {})
    return report if isinstance(report, dict) else {}


def record_dataset_pii(state: Dict[str, Any], dataframe: Any, report: Any) -> None:
    """Persist PII findings from the preparation agent against its input dataset."""
    try:
        parents = _dataset_parents(state)
        if not parents:
            return
        primary_id = str(state.get("active_dataset_id") or parents[0])
        if primary_id not in parents:
            primary_id = parents[0]
        pii = _pii_kinds(report)
        columns = [
            (str(name), str(dtype))
            for name, dtype in getattr(dataframe, "dtypes", {}).items()
        ]
        store = _store(state)
        try:
            with store.batch():
                existing = store.get_node(primary_id)
                store.record_dataset(
                    primary_id,
                    label=existing.label if existing else "",
                    columns=_columns({}, pii, columns),
                    pii=pii,
                    **(existing.attrs if existing else {}),
                )
        finally:
            store.close()
    except Exception:  # noqa: BLE001
        logger.warning("[lineage] could not record PII scan", exc_info=True)


def record_analysis_lineage(state: Dict[str, Any], output_data: Any) -> Optional[str]:
    """Record an output dataset only after the execution agent validates it."""
    try:
        parents = _dataset_parents(state)
        if not parents:
            return None
        metadata_by_id = _source_metadata(state)
        primary_id = str(state.get("active_dataset_id") or parents[0])
        if primary_id not in parents:
            primary_id = parents[0]
        pii_dataset_id = str(
            state.get("pii_dataset_id")
            or state.get("active_dataset_id")
            or parents[0]
        )
        pii = _pii_kinds(_state_pii_report(state)) if pii_dataset_id in parents else {}
        analysis_id = f"an:{uuid.uuid4().hex}"
        output_id = f"ds:output:{uuid.uuid4().hex}"
        output_columns = [
            (str(name), str(dtype))
            for name, dtype in getattr(output_data, "dtypes", {}).items()
        ]
        store = _store(state)
        if not store.available:
            # Returning an id for a dataset that was never stored would make
            # the next turn's "where did this come from?" look it up and miss.
            return None
        try:
            # One transaction for the whole event: every parent, the output,
            # its columns and all their edges.
            with store.batch():
                known = {node.id: node for node in store.get_nodes(parents)}
                for parent_id in parents:
                    metadata = metadata_by_id.get(parent_id, {})
                    existing = known.get(parent_id)
                    parent_pii = pii if parent_id == pii_dataset_id else _pii_kinds(metadata.get("pii_report"))
                    connection_id = metadata.get("connection_id") or state.get("connection_id")
                    if existing and not metadata and not parent_pii and not connection_id:
                        continue
                    schema = metadata.get("columns") or []
                    if not schema and existing is None and parent_id == primary_id:
                        schema = state.get("uploaded_csv_columns") or []
                    store.record_dataset(
                        parent_id,
                        label=str(metadata.get("filename") or metadata.get("alias") or (existing.label if existing else "")),
                        columns=_columns(metadata, parent_pii, schema),
                        pii=parent_pii,
                        connection_id=connection_id if parent_id == primary_id else None,
                        **(existing.attrs if existing else {}),
                    )

                output_path = state.get("output_location") or ""
                file_data = state.get("output_file_data") or {}
                output_label = (
                    file_data.get("filename")
                    or Path(str(output_path)).name
                    or "analysis output"
                )
                store.record_dataset(
                    output_id,
                    label=output_label,
                    columns=output_columns,
                    derived_from=parents[0],
                    analysis_id=analysis_id,
                    row_count=len(output_data),
                )
                for parent_id in parents[1:]:
                    store.add_edge(Edge(output_id, EdgeKind.DERIVED_FROM, parent_id,
                                        {"analysis_id": analysis_id}))
            return output_id
        finally:
            store.close()
    except Exception:  # noqa: BLE001
        logger.warning("[lineage] could not record successful analysis", exc_info=True)
        return None


def record_model_lineage(state: Dict[str, Any], training_result: Dict[str, Any]) -> Optional[str]:
    """Record a successful model run and its selected feature columns."""
    try:
        plan = state.get("training_plan") or {}
        data_config = plan.get("data_config") or {}
        training_uri = data_config.get("dataset_uri")
        latest_id = state.get("latest_output_dataset_id")
        latest_locations = (
            state.get("latest_output_location"),
            state.get("latest_output_location_local"),
            state.get("output_location"),
        )
        if latest_id and (
            not training_uri
            or any(_locations_match(training_uri, item) for item in latest_locations)
        ):
            source_id = str(latest_id)
        else:
            parents = _dataset_parents(state)
            source_id = parents[0] if parents else None
        run_id = training_result.get("mlflow_run_id") or training_result.get("task_id")
        if not source_id or not run_id:
            return None

        integrity = training_result.get("integrity_report") or state.get("integrity_report") or {}
        pii = _pii_kinds(integrity.get("pii", {}) if isinstance(integrity, dict) else {})
        features = [str(column) for column in data_config.get("feature_columns", [])]
        target = data_config.get("target_column")
        columns = list(dict.fromkeys([*features, *([str(target)] if target else []), *pii]))
        model_id = f"model:{run_id}"

        store = _store(state)
        if not store.available:
            return None
        try:
            with store.batch():
                existing = store.get_node(source_id)
                store.record_dataset(
                    source_id,
                    label=existing.label if existing else "",
                    columns=[(column, "unknown") for column in columns],
                    pii=pii,
                    **(existing.attrs if existing else {}),
                )
                store.record_model(
                    model_id,
                    trained_on=source_id,
                    feature_columns=features,
                    label=str(training_result.get("model_name") or ""),
                    model_type=training_result.get("model_type"),
                )
        finally:
            store.close()
        return model_id
    except Exception:  # noqa: BLE001
        logger.warning("[lineage] could not record successful model training", exc_info=True)
        return None


def format_lineage_reply(state: Dict[str, Any]) -> str:
    """Answer a provenance question for the latest or currently active dataset."""
    dataset_id = (
        state.get("latest_output_dataset_id")
        or state.get("active_dataset_id")
        or state.get("dataset_id")
    )
    if not dataset_id:
        return "I don't have a selected dataset to trace yet."

    store = _store(state)
    try:
        if not store.available:
            return "Lineage is unavailable right now; the analysis itself is unaffected."
        result = store.dataset_lineage(str(dataset_id))
        if result is None:
            return "I don't have recorded lineage for the current dataset yet."

        dataset = result["dataset"]
        label = dataset.label or dataset.id
        parents = result["parents"]
        ancestors = result["ancestors"]
        if not ancestors:
            return f"{label} is recorded as a source dataset; no upstream derivation is recorded."

        parent_text = ", ".join(
            f"{node.label or node.id} ({node.id})" for node in parents
        )
        upstream = [node for node in ancestors if node.id not in {parent.id for parent in parents}]
        details = f"Directly derived from: {parent_text or 'an unlabelled dataset'}"
        if upstream:
            upstream_text = ", ".join(
                f"{node.label or node.id} ({node.id})" for node in upstream
            )
            details += f". Earlier ancestors: {upstream_text}"
        return f"{label} ({dataset.id}) — {details}."
    except Exception:  # noqa: BLE001
        logger.warning("[lineage] chat query failed", exc_info=True)
        return "Lineage is unavailable right now; the analysis itself is unaffected."
    finally:
        store.close()