"""Size-aware fidelity / execution-mode reasoning.

Translates dataset size + cloud presence + intent into the state shape that
``route_planner_output`` already expects. The planner's existing routing
sees the richer pre-configured state and behaves correctly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Thresholds, all in bytes
_KB = 1024
_MB = 1024 * _KB
_GB = 1024 * _MB

SMALL_THRESHOLD = 100 * _MB
MEDIUM_THRESHOLD = 1 * _GB
LARGE_THRESHOLD = 10 * _GB

FIDELITY_QUICK = "quick_sample"
FIDELITY_PORTFOLIO = "portfolio_samples"
FIDELITY_ENTIRE = "entire_dataset"


def _human_size(num_bytes: int) -> str:
    try:
        value = float(num_bytes or 0)
    except (TypeError, ValueError):
        value = 0.0
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.1f} {units[idx]}" if idx > 0 else f"{int(value)} {units[idx]}"


def _runtime_hint(num_bytes: int) -> str:
    try:
        gb = float(num_bytes or 0) / float(_GB)
    except (TypeError, ValueError):
        gb = 0.0
    if gb <= 0:
        return "roughly minutes to hours depending on query complexity"
    if gb < 5:
        return "roughly 10-30 minutes"
    if gb < 20:
        return "roughly 30-90 minutes"
    if gb < 100:
        return "tens of minutes to a few hours"
    return "hours to potentially days"


def _size_class(num_bytes: int) -> str:
    if num_bytes < SMALL_THRESHOLD:
        return "small"
    if num_bytes < MEDIUM_THRESHOLD:
        return "medium"
    if num_bytes < LARGE_THRESHOLD:
        return "large"
    return "huge"


def decide_execution_profile(
    state: Dict[str, Any],
    intent: str,
) -> Dict[str, Any]:
    """Decide fidelity, execution mode, and (when needed) a Celery schedule
    based on dataset size, cloud availability, and intent.

    Returns a dict suitable for shallow-merge into ``state``. Never overrides
    a fidelity the user (or a prior turn) explicitly set.
    """
    size = int(state.get("dataset_size_bytes") or state.get("file_size_bytes") or 0)
    has_cloud = bool(
        state.get("data_source_location_cloud") and state.get("connection_id")
    )
    user_fidelity = state.get("analysis_fidelity")

    # No evidence of a bound dataset → leave existing state alone.
    # Returning {} preserves the planner's defaults (e.g. execution_mode="cloud"
    # which is needed for infrastructure-provisioning routing).
    if size <= 0 and not has_cloud and not user_fidelity:
        return {}

    if user_fidelity in (FIDELITY_QUICK, FIDELITY_PORTFOLIO, FIDELITY_ENTIRE):
        fidelity = user_fidelity
    else:
        # Size first, then read the WHOLE thing unless it is genuinely large.
        #
        # SMALL_THRESHOLD was previously defined and never used here: the first
        # branch tested MEDIUM_THRESHOLD, so everything under 1 GB was sampled
        # -- including a 12-row CSV, which the user was then told had been
        # "computed on a 12-row sample". Sampling a file small enough to read
        # entirely costs accuracy and credibility and saves nothing.
        if size <= SMALL_THRESHOLD:
            fidelity = FIDELITY_ENTIRE
        elif size < MEDIUM_THRESHOLD:
            fidelity = FIDELITY_PORTFOLIO
        elif size < LARGE_THRESHOLD:
            fidelity = FIDELITY_PORTFOLIO if has_cloud else FIDELITY_QUICK
        else:
            fidelity = FIDELITY_ENTIRE if has_cloud else FIDELITY_PORTFOLIO

    if fidelity == FIDELITY_QUICK:
        exec_mode = "local"
    elif fidelity == FIDELITY_PORTFOLIO:
        exec_mode = "k8s-ray" if has_cloud else "local"
    else:  # entire_dataset
        # A small file read whole is a local job; dispatching a 12-row CSV to
        # the cluster would trade seconds for minutes.
        exec_mode = "local" if size <= SMALL_THRESHOLD else (
            "k8s-ray" if has_cloud else "local")

    if intent in ("ml_training", "ml_inference") and size > MEDIUM_THRESHOLD and has_cloud:
        exec_mode = "k8s-ray"

    updates: Dict[str, Any] = {
        "analysis_fidelity": fidelity,
        "execution_mode": exec_mode,
    }

    if (
        fidelity == FIDELITY_ENTIRE
        and exec_mode == "k8s-ray"
        and not (state.get("task_schedule") or {}).get("task_type")
    ):
        updates["task_schedule"] = {
            "task_type": "execute",
            "schedule_type": "relative",
            "second": 0,
            "max_runs": 1,
        }

    recommendation = {
        "size_bytes": size,
        "size_human": _human_size(size),
        "size_class": _size_class(size),
        "fidelity": fidelity,
        "execution_mode": exec_mode,
        "eta_text": _runtime_hint(size) if exec_mode == "k8s-ray" else "seconds to minutes",
        "cost_note": (
            "k8s-ray runs incur cluster compute time; entire-dataset analyses "
            "are scheduled via Celery and may take "
            f"{_runtime_hint(size)}."
            if exec_mode == "k8s-ray"
            else "Local execution; no cluster cost."
        ),
        "user_overrode_fidelity": bool(user_fidelity),
        # Large inputs are where full fidelity stops being free, so that is
        # where the user gets asked rather than told.
        "confirm_before_full_run": bool(
            fidelity == FIDELITY_ENTIRE and size > LARGE_THRESHOLD),
    }
    updates["avaloka_recommendation"] = recommendation
    return updates
