"""Pure adapters for the legacy 1.5.2 database workspace contract."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def database_query_text(content: str | None, query: str | None) -> str:
    """Return the modern content value, accepting the legacy ``query`` alias."""
    return str(content or query or "").strip()


def legacy_connect_fields(
    *, session_id: str, thread_id: str, connection_id: str, expires_in: int
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "thread_id": thread_id,
        "connection_id": connection_id,
        "expires_in": max(0, int(expires_in)),
    }


def _column_type(values: Sequence[Any]) -> str:
    value = next((item for item in values if item is not None), None)
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, (dict, list, tuple)):
        return "json"
    return "string"


def legacy_query_fields(
    result: Mapping[str, Any] | None, *, execution_time_ms: int
) -> dict[str, Any]:
    """Project an MCP query response into the active 1.5.2 UI result shape."""
    payload = dict(result or {})
    raw_columns = payload.get("columns") or []
    columns = [str(column) for column in raw_columns]
    raw_rows = payload.get("rows") or []
    rows: list[dict[str, Any]] = []
    for raw in raw_rows if isinstance(raw_rows, list) else []:
        if isinstance(raw, Mapping):
            row = {str(key): value for key, value in raw.items()}
            if not columns:
                columns = list(row)
        elif isinstance(raw, (list, tuple)):
            row = dict(zip(columns, raw))
        else:
            row = {columns[0] if columns else "value": raw}
            if not columns:
                columns = ["value"]
        rows.append(row)

    schema = [
        {
            "name": column,
            "type": _column_type([row.get(column) for row in rows]),
        }
        for column in columns
    ]
    declared_count = payload.get("row_count")
    row_count = (
        int(declared_count)
        if isinstance(declared_count, (int, float)) and declared_count >= 0
        else len(rows)
    )
    return {
        "results": rows,
        "schema": schema,
        "row_count": row_count,
        "execution_time_ms": max(0, int(execution_time_ms)),
    }
