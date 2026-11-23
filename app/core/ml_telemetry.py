"""Telemetry events for the training/inference pipeline, with a denylist.

Avaloka's standing privacy constraint: telemetry carries no customer data --
no model names, column names, file names, dataset URIs, class labels. These
events describe *that* something happened and how long it took, keyed by
opaque identifiers only. :func:`event` is the single way to build one, and it
drops or refuses anything on the denylist, so a future caller cannot leak by
passing the training state through.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger("avaloka.telemetry.ml")

#: Keys that may appear on an event. Anything else is dropped.
ALLOWED_KEYS = frozenset({
    "event", "kind", "status", "job_id", "run_id", "user_hash", "backend", "task_type",
    "duration_s", "n_rows", "n_features", "n_classes", "k", "attempts", "error_code",
    "server", "warm", "endpoint_status", "retry_after",
})

#: Keys that must never appear, even nested, and whose *values* must not leak either.
DENYLIST = frozenset({
    "model_name", "model_description", "registered_name", "feature_columns", "feature_names",
    "feature_cols", "model_feature_names", "columns", "column_names", "target_column", "target_col",
    "dataset_uri", "data_uri", "filename", "file_name", "path", "output_location",
    "class_names", "labels", "messages", "error", "traceback", "logs", "stdout", "stderr",
})

_HASH_RE = re.compile(r"^[0-9a-f]{12}$")


def user_hash(user_id: Any) -> Optional[str]:
    """A short, non-reversible handle for grouping, never the user id itself."""
    if not user_id:
        return None
    import hashlib

    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:12]


class TelemetryLeak(ValueError):
    """A denylisted key was passed explicitly."""


def _scrub_values(payload: Mapping[str, Any], forbidden_values: set) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str) and any(v and v in value for v in forbidden_values):
            continue
        out[key] = value
    return out


def event(name: str, *, state: Optional[Mapping[str, Any]] = None, **fields: Any) -> Dict[str, Any]:
    """Build one event. ``state`` may be a raw training/job state: only safe
    numbers are derived from it (row/feature/class counts), nothing is copied.

    Raises :class:`TelemetryLeak` when a denylisted key is passed explicitly in
    ``fields`` -- that is a programming error, not something to scrub quietly.
    """
    leaked = DENYLIST.intersection(fields)
    if leaked:
        raise TelemetryLeak(f"telemetry must not carry {sorted(leaked)}")

    out: Dict[str, Any] = {"event": str(name)}
    if "user_id" in fields:
        out["user_hash"] = user_hash(fields.pop("user_id"))
    for key, value in fields.items():
        if key in ALLOWED_KEYS and value is not None:
            out[key] = value

    forbidden_values: set = set()
    if state:
        result = state.get("training_result") if isinstance(state.get("training_result"), Mapping) else {}
        plan = state.get("training_plan") if isinstance(state.get("training_plan"), Mapping) else {}
        data_cfg = plan.get("data_config") if isinstance(plan.get("data_config"), Mapping) else {}
        for src in (state, result, plan, data_cfg):
            for key in DENYLIST:
                val = src.get(key)
                if isinstance(val, str):
                    forbidden_values.add(val)
                elif isinstance(val, (list, tuple)):
                    forbidden_values.update(str(v) for v in val if isinstance(v, str))
        n_rows = result.get("rows_processed") or state.get("n_rows")
        if isinstance(n_rows, (int, float)):
            out.setdefault("n_rows", int(n_rows))
        for src_key, out_key in (("num_features", "n_features"), ("num_classes", "n_classes")):
            v = result.get(src_key)
            if isinstance(v, (int, float)):
                out.setdefault(out_key, int(v))
        if isinstance(result.get("model_type"), str) and "task_type" not in out:
            out["task_type"] = "regression" if "regression" in result["model_type"].lower() else "classification"
        if "user_hash" not in out and state.get("user_id"):
            out["user_hash"] = user_hash(state.get("user_id"))

    out = _scrub_values(out, forbidden_values)
    return out


def emit(name: str, *, state: Optional[Mapping[str, Any]] = None, **fields: Any) -> Dict[str, Any]:
    """Build and log the event as one JSON line."""
    payload = event(name, state=state, **fields)
    logger.info(json.dumps(payload, sort_keys=True, default=str))
    return payload
