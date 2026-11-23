"""
Task specifications for the benchmark.

A *task* is a unit of work the Avaloka CLI is asked to perform on a dataset,
plus the objective criteria used to score the result. Tasks span the lifecycle
in the spec:

    sample -> analyze features -> build features -> transform -> train -> serve

Tasks are declarative: the harness hands ``brief`` + ``source`` to the CLI and
checks the returned artifact against ``checks``. Each check is a pure function
``(result, context) -> CheckOutcome`` so the same task scores identically across
the reference client and the real CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Stage(str, Enum):
    SAMPLE = "sample"
    ANALYZE = "analyze_features"
    BUILD = "build_features"
    TRANSFORM = "transform"
    TRAIN = "train"
    SERVE = "serve"


class Severity(str, Enum):
    CRITICAL = "critical"  # failing this means the task fails outright
    MAJOR = "major"
    MINOR = "minor"


@dataclass
class CheckOutcome:
    name: str
    passed: bool
    severity: Severity
    detail: str = ""
    # Optional numeric measurement (e.g. accuracy) for the scorecard.
    measured: Optional[float] = None
    threshold: Optional[float] = None


# A check inspects the CLI's result dict + a context (dataset handle, oracle).
Check = Callable[[Dict[str, Any], "TaskContext"], CheckOutcome]


@dataclass
class TaskContext:
    """Everything a check may need beyond the CLI result."""

    dataset_key: str
    target: Optional[str]
    local_path: str
    source_kind: str
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkTask:
    task_id: str
    stage: Stage
    dataset_key: str
    brief: str
    checks: List[Check]
    # Source kinds this task should be run against (defaults to local only).
    source_kinds: List[str] = field(default_factory=lambda: ["local"])
    # Marks the task as requiring the heavy DL path.
    heavy: bool = False
    weight: float = 1.0
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.checks:
            raise ValueError(f"Task {self.task_id} has no checks")


# --------------------------------------------------------------------------- #
# Reusable check builders
# --------------------------------------------------------------------------- #


def check_no_error(severity: Severity = Severity.CRITICAL) -> Check:
    def _c(result, ctx):
        err = result.get("error")
        return CheckOutcome(
            "no_error", passed=err is None, severity=severity, detail=str(err or "")
        )

    return _c


def check_keys_present(keys: List[str], severity=Severity.CRITICAL) -> Check:
    def _c(result, ctx):
        missing = [k for k in keys if k not in result or result[k] is None]
        return CheckOutcome(
            f"keys_present:{','.join(keys)}",
            passed=not missing,
            severity=severity,
            detail=f"missing={missing}" if missing else "",
        )

    return _c


def check_rows_nonempty(key: str = "rows", severity=Severity.CRITICAL) -> Check:
    def _c(result, ctx):
        rows = result.get(key) or []
        n = len(rows) if hasattr(rows, "__len__") else 0
        return CheckOutcome(
            f"rows_nonempty:{key}", passed=n > 0, severity=severity, detail=f"n={n}"
        )

    return _c


def check_ddl_well_formed(severity=Severity.MAJOR) -> Check:
    def _c(result, ctx):
        ddl = result.get("ddl_schema") or ""
        ok = "CREATE TABLE" in ddl and ddl.strip().endswith(";")
        return CheckOutcome(
            "ddl_well_formed", passed=ok, severity=severity, detail=ddl[:80]
        )

    return _c


def check_schema_covers_columns(min_cols: int = 2, severity=Severity.MAJOR) -> Check:
    def _c(result, ctx):
        schema = result.get("schema") or []
        return CheckOutcome(
            "schema_covers_columns",
            passed=len(schema) >= min_cols,
            severity=severity,
            detail=f"n_cols={len(schema)}",
            measured=float(len(schema)),
            threshold=float(min_cols),
        )

    return _c


def check_metric_at_least(
    metric_key: str, threshold: float, severity=Severity.CRITICAL
) -> Check:
    """The CLI's reported metric must meet/exceed a threshold (e.g. accuracy)."""

    def _c(result, ctx):
        metrics = result.get("metrics") or {}
        val = metrics.get(metric_key)
        ok = isinstance(val, (int, float)) and float(val) >= threshold
        return CheckOutcome(
            f"{metric_key}>={threshold}",
            passed=bool(ok),
            severity=severity,
            measured=float(val) if isinstance(val, (int, float)) else None,
            threshold=threshold,
            detail=f"{metric_key}={val}",
        )

    return _c


def check_metric_at_most(
    metric_key: str, threshold: float, severity=Severity.CRITICAL
) -> Check:
    """For error metrics (RMSE/MAE): reported value must be <= threshold."""

    def _c(result, ctx):
        metrics = result.get("metrics") or {}
        val = metrics.get(metric_key)
        ok = isinstance(val, (int, float)) and float(val) <= threshold
        return CheckOutcome(
            f"{metric_key}<={threshold}",
            passed=bool(ok),
            severity=severity,
            measured=float(val) if isinstance(val, (int, float)) else None,
            threshold=threshold,
            detail=f"{metric_key}={val}",
        )

    return _c


def check_beats_baseline(metric_key: str, margin: float = 0.0, severity=Severity.MAJOR) -> Check:
    """
    The CLI's model must beat the harness-computed baseline oracle stored in
    ``ctx.extras['baseline'][metric_key]`` by at least ``margin``.
    Direction is inferred: error metrics (rmse/mae/logloss) are 'lower better'.
    """

    lower_better = any(k in metric_key.lower() for k in ("rmse", "mae", "loss", "error"))

    def _c(result, ctx):
        baseline = (ctx.extras.get("baseline") or {}).get(metric_key)
        val = (result.get("metrics") or {}).get(metric_key)
        if not isinstance(val, (int, float)) or not isinstance(baseline, (int, float)):
            return CheckOutcome(
                f"beats_baseline:{metric_key}", passed=False, severity=severity,
                detail=f"missing val={val} baseline={baseline}",
            )
        ok = (val + margin <= baseline) if lower_better else (val >= baseline - margin)
        return CheckOutcome(
            f"beats_baseline:{metric_key}",
            passed=bool(ok),
            severity=severity,
            measured=float(val),
            threshold=float(baseline),
            detail=f"val={val} baseline={baseline} lower_better={lower_better}",
        )

    return _c


def check_predicate(name: str, fn: Callable[[Dict, TaskContext], bool], severity=Severity.MAJOR) -> Check:
    """Escape hatch for bespoke assertions."""

    def _c(result, ctx):
        try:
            ok = bool(fn(result, ctx))
            detail = ""
        except Exception as e:  # a throwing predicate is a failed check
            ok, detail = False, f"{type(e).__name__}: {e}"
        return CheckOutcome(name, passed=ok, severity=severity, detail=detail)

    return _c
