"""Typed deterministic statistics for evidence-backed analytical missions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StatisticalPlan:
    method: str
    variables: tuple[str, ...] = ()
    grouping: tuple[str, ...] = ()
    estimand: str = ""
    unit_of_analysis: tuple[str, ...] = ()
    weights: str | None = None
    missing_policy: str = "drop_required"
    parameters: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StatisticalResult:
    method: str
    estimate: Any
    sample_size: int
    statistic: float | None = None
    degrees_of_freedom: float | None = None
    p_value: float | None = None
    confidence_interval: tuple[float, float] | None = None
    diagnostics: tuple[dict[str, Any], ...] = ()
    plan_hash: str = ""


def _required_frame(frame: pd.DataFrame, plan: StatisticalPlan) -> pd.DataFrame:
    required = list(dict.fromkeys([*plan.variables, *plan.grouping] + ([plan.weights] if plan.weights else [])))
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"statistical inputs not found: {missing}")
    data = frame[required].copy() if required else frame.copy()
    if plan.missing_policy == "drop_required" and required:
        data = data.dropna(subset=required)
    elif plan.missing_policy not in {"drop_required", "preserve"}:
        raise ValueError(f"unsupported missing policy: {plan.missing_policy}")
    return data


def _chi_square(table: np.ndarray) -> tuple[float, int, float | None, np.ndarray]:
    if table.ndim != 2 or min(table.shape) < 2:
        raise ValueError("chi-square requires at least a 2x2 contingency table")
    total = float(table.sum())
    if total <= 0:
        raise ValueError("chi-square table is empty")
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    if np.any(expected <= 0):
        raise ValueError("chi-square expected counts contain zero")
    statistic = float(np.sum((table - expected) ** 2 / expected))
    df = int((table.shape[0] - 1) * (table.shape[1] - 1))
    p_value: float | None = None
    try:
        from scipy.stats import chi2  # optional; scikit-learn environments usually include scipy
        p_value = float(chi2.sf(statistic, df))
    except Exception:  # pragma: no cover - optional dependency boundary
        pass
    return statistic, df, p_value, expected


def bootstrap_mean_interval(
    values: pd.Series | np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("bootstrap requires at least one finite value")
    if not 0 < confidence < 1 or resamples < 100:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        means[index] = float(rng.choice(array, size=len(array), replace=True).mean())
    alpha = 1 - confidence
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def analytic_mean_interval(
    values: pd.Series | np.ndarray, *, confidence: float = 0.95,
) -> tuple[float, float]:
    """The normal-approximation interval for a mean, mean +/- z * s / sqrt(n).

    Constant cost whatever ``n``: a 2,000-resample bootstrap over every row
    of a 3M-row frame is billions of draws, and at that size the bootstrap
    and the analytic interval agree to the reported digits anyway."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("an interval requires at least one finite value")
    if not 0 < confidence < 1:
        raise ValueError("invalid confidence level")
    from statistics import NormalDist

    mean = float(array.mean())
    if len(array) < 2:
        return mean, mean
    half = NormalDist().inv_cdf(0.5 + confidence / 2) * float(array.std(ddof=1)) / math.sqrt(len(array))
    return mean - half, mean + half


def compute_statistic(frame: pd.DataFrame, plan: StatisticalPlan) -> StatisticalResult:
    """Execute a supported plan with explicit population and denominator semantics."""
    data = _required_frame(frame, plan)
    method = plan.method
    params = dict(plan.parameters)
    diagnostics: list[dict[str, Any]] = []
    estimate: Any
    statistic = df = p_value = None
    interval = None

    if method == "count":
        estimate = int(len(data))
    elif method == "distinct_count":
        estimate = int(data[plan.variables[0]].nunique(dropna=True))
    elif method in {"sum", "mean", "median", "quantile"}:
        values = pd.to_numeric(data[plan.variables[0]], errors="coerce").dropna()
        if method == "sum":
            estimate = float(values.sum())
        elif method == "mean":
            estimate = float(values.mean())
            if params.get("interval") == "analytic":
                interval = analytic_mean_interval(
                    values, confidence=float(params.get("confidence", 0.95)))
            elif params.get("bootstrap"):
                interval = bootstrap_mean_interval(
                    values, confidence=float(params.get("confidence", 0.95)),
                    resamples=int(params.get("resamples", 2000)), seed=int(params.get("seed", 0)),
                )
        elif method == "median":
            estimate = float(values.median())
        else:
            estimate = float(values.quantile(float(params.get("q", 0.5))))
        data = data.loc[values.index]
    elif method == "weighted_mean":
        values = pd.to_numeric(data[plan.variables[0]], errors="coerce")
        weights = pd.to_numeric(data[plan.weights or ""], errors="coerce")
        valid = values.notna() & weights.notna() & (weights >= 0)
        if not valid.any() or float(weights[valid].sum()) <= 0:
            raise ValueError("weighted mean requires positive total valid weight")
        estimate = float(np.average(values[valid], weights=weights[valid]))
        data = data.loc[valid]
    elif method == "proportion":
        values = data[plan.variables[0]]
        if "equals" in params:
            successes = values.eq(params["equals"])
        else:
            successes = values.astype(bool)
        denominator = int(successes.notna().sum())
        if denominator == 0:
            raise ValueError("proportion denominator is zero")
        numerator = int(successes.fillna(False).sum())
        estimate = numerator / denominator
        diagnostics.append({"numerator": numerator, "denominator": denominator})
    elif method == "grouped_aggregate":
        operation = str(params.get("operation", "mean"))
        value = plan.variables[0] if plan.variables else None
        grouped = data.groupby(list(plan.grouping), dropna=False)
        if operation == "count":
            series = grouped.size()
        elif operation == "distinct_count":
            series = grouped[value].nunique(dropna=True)
        elif operation in {"mean", "sum", "median", "max", "min"}:
            series = getattr(grouped[value], operation)()
        else:
            raise ValueError(f"unsupported grouped aggregation: {operation}")
        estimate = []
        for group_key, aggregate_value in series.items():
            keys = group_key if isinstance(group_key, tuple) else (group_key,)
            row = dict(zip(plan.grouping, keys))
            row["value"] = (
                float(aggregate_value)
                if isinstance(aggregate_value, (int, float, np.number))
                else aggregate_value
            )
            estimate.append(row)
    elif method == "correlation":
        left = pd.to_numeric(data[plan.variables[0]], errors="coerce")
        right = pd.to_numeric(data[plan.variables[1]], errors="coerce")
        valid = left.notna() & right.notna()
        kind = str(params.get("kind", "pearson"))
        estimate = float(left[valid].corr(right[valid], method=kind))
        data = data.loc[valid]
    elif method == "chi_square":
        table = pd.crosstab(data[plan.variables[0]], data[plan.variables[1]], dropna=False)
        minimum = int(params.get("minimum_marginal", 0))
        if minimum:
            keep_rows = table.sum(axis=1) > minimum
            keep_cols = table.sum(axis=0) > minimum
            table = table.loc[keep_rows, keep_cols]
        statistic, df, p_value, expected = _chi_square(table.to_numpy(dtype=float))
        estimate = statistic
        diagnostics.extend([
            {"contingency_shape": list(table.shape)},
            {"minimum_expected_count": float(expected.min()), "expected_below_5": int((expected < 5).sum())},
        ])
    elif method == "ema":
        values = pd.to_numeric(data[plan.variables[0]], errors="coerce")
        alpha = float(params["alpha"])
        if not 0 < alpha <= 1:
            raise ValueError("ema alpha must be in (0, 1]")
        estimate = values.ewm(alpha=alpha, adjust=False).mean().tolist()
    elif method == "volatility":
        values = pd.to_numeric(data[plan.variables[0]], errors="coerce").dropna()
        definition = str(params.get("definition", "std"))
        if definition == "intraday" and len(plan.variables) >= 2:
            high = pd.to_numeric(data[plan.variables[0]], errors="coerce")
            low = pd.to_numeric(data[plan.variables[1]], errors="coerce")
            estimate = float((high - low).mean())
        else:
            estimate = float(values.std(ddof=int(params.get("ddof", 1))))
        data = data.loc[values.index]
    elif method == "period_return":
        values = pd.to_numeric(data[plan.variables[0]], errors="coerce").dropna()
        if len(values) < 2 or values.iloc[0] == 0:
            raise ValueError("period return requires two values and a non-zero start")
        estimate = float(values.iloc[-1] / values.iloc[0] - 1)
        data = data.loc[values.index]
    else:
        raise ValueError(f"unsupported statistical method: {method}")

    if isinstance(estimate, float) and (math.isnan(estimate) or math.isinf(estimate)):
        raise ValueError(f"{method} produced a non-finite estimate")
    return StatisticalResult(
        method=method, estimate=estimate, sample_size=len(data),
        statistic=statistic, degrees_of_freedom=df, p_value=p_value,
        confidence_interval=interval, diagnostics=tuple(diagnostics),
        plan_hash=plan.fingerprint(),
    )
