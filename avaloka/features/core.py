"""Deterministic, evidence-preserving analytical feature transformations."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransformStep:
    operation: str
    output: str | None = None
    inputs: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureRecipe:
    steps: tuple[TransformStep, ...] = ()
    output_grain: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JoinDiagnostics:
    left_rows: int
    right_rows: int
    output_rows: int
    matched_left_rows: int
    unmatched_left_rows: int
    left_key_unique: bool
    right_key_unique: bool
    cardinality: str
    multiplication_ratio: float
    normalized: bool


@dataclass(frozen=True)
class FeatureResult:
    frame: pd.DataFrame
    recipe_hash: str
    diagnostics: tuple[dict[str, Any], ...]


_IDENTIFIER_SEPARATORS = re.compile(r"[^a-z0-9]+")


def normalize_identifier(value: Any) -> str | None:
    """Normalize a join identifier without turning missing values into keys."""
    is_container = isinstance(value, (list, tuple, dict, set, np.ndarray, pd.Series))
    if value is None or (not is_container and pd.isna(value)):
        return None
    normalized = _IDENTIFIER_SEPARATORS.sub("", str(value).strip().casefold())
    return normalized or None


def parse_structured_string(value: Any) -> Any:
    """Parse JSON/Python literal containers; return the original value on failure."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{(\"'":
        return value
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return value
        return parsed if isinstance(parsed, (dict, list, tuple, str, int, float, bool, type(None))) else value


def project_nested(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, str):
            current = parse_structured_string(current)
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
    return current


def verified_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    how: str = "inner",
    normalize: bool = True,
    max_multiplication_ratio: float = 1.05,
    allow_many_to_many: bool = False,
) -> tuple[pd.DataFrame, JoinDiagnostics]:
    """Join with key normalization and explicit cardinality/multiplication checks."""
    if left_on not in left or right_on not in right:
        missing = [name for name, frame in ((left_on, left), (right_on, right)) if name not in frame]
        raise KeyError(f"join key(s) not found: {missing}")
    ldf, rdf = left.copy(), right.copy()
    lk, rk = left_on, right_on
    if normalize:
        lk, rk = "__avaloka_left_key", "__avaloka_right_key"
        ldf[lk] = ldf[left_on].map(normalize_identifier)
        rdf[rk] = rdf[right_on].map(normalize_identifier)
    left_valid = ldf[lk].notna()
    right_valid = rdf[rk].notna()
    left_unique = not ldf.loc[left_valid, lk].duplicated().any()
    right_unique = not rdf.loc[right_valid, rk].duplicated().any()
    if left_unique and right_unique:
        cardinality = "one_to_one"
    elif right_unique:
        cardinality = "many_to_one"
    elif left_unique:
        cardinality = "one_to_many"
    else:
        cardinality = "many_to_many"
    if cardinality == "many_to_many" and not allow_many_to_many:
        raise ValueError("many-to-many join requires explicit allow_many_to_many=True")

    valid_right = rdf.loc[right_valid].copy()
    merged = ldf.merge(valid_right, how=how, left_on=lk, right_on=rk, indicator="__avaloka_merge")
    matched = int((merged["__avaloka_merge"] == "both").sum())
    unmatched = int((merged["__avaloka_merge"] == "left_only").sum())
    baseline = max(1, len(ldf))
    ratio = len(merged) / baseline
    if ratio > max_multiplication_ratio and not allow_many_to_many:
        raise ValueError(
            f"join multiplied rows by {ratio:.3f}; limit is {max_multiplication_ratio:.3f}"
        )
    diagnostics = JoinDiagnostics(
        left_rows=len(left), right_rows=len(right), output_rows=len(merged),
        matched_left_rows=matched, unmatched_left_rows=unmatched,
        left_key_unique=left_unique, right_key_unique=right_unique,
        cardinality=cardinality, multiplication_ratio=ratio, normalized=normalize,
    )
    helper_columns = [lk, rk] if normalize else []
    merged = merged.drop(columns=[*helper_columns, "__avaloka_merge"], errors="ignore")
    return merged, diagnostics


def execute_feature_recipe(frame: pd.DataFrame, recipe: FeatureRecipe) -> FeatureResult:
    """Apply a reviewed transform vocabulary and retain per-step diagnostics."""
    out = frame.copy()
    diagnostics: list[dict[str, Any]] = []
    for index, step in enumerate(recipe.steps):
        before_rows = len(out)
        op, params = step.operation, dict(step.parameters)
        output = step.output
        inputs = list(step.inputs)
        if any(name not in out.columns for name in inputs):
            missing = [name for name in inputs if name not in out.columns]
            raise KeyError(f"step {index} {op}: missing input columns {missing}")

        if op == "normalize_identifier":
            out[output or inputs[0]] = out[inputs[0]].map(normalize_identifier)
        elif op == "parse_structured_string":
            out[output or inputs[0]] = out[inputs[0]].map(parse_structured_string)
        elif op == "project_nested":
            path = str(params["path"])
            out[output or path.replace(".", "_")] = out[inputs[0]].map(lambda value: project_nested(value, path))
        elif op == "explode_array":
            out = out.explode(inputs[0], ignore_index=True)
        elif op == "derive_date_part":
            part = str(params.get("part", "year"))
            dates = pd.to_datetime(out[inputs[0]], errors="coerce", utc=params.get("utc", False))
            accessor = getattr(dates.dt, part, None)
            if accessor is None:
                raise ValueError(f"unsupported date part: {part}")
            out[output or f"{inputs[0]}_{part}"] = accessor
        elif op == "derive_duration":
            start = pd.to_datetime(out[inputs[0]], errors="coerce", utc=True)
            end = pd.to_datetime(out[inputs[1]], errors="coerce", utc=True)
            unit = str(params.get("unit", "days"))
            divisor = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}.get(unit)
            if divisor is None:
                raise ValueError(f"unsupported duration unit: {unit}")
            out[output or f"{inputs[1]}_minus_{inputs[0]}_{unit}"] = (end - start).dt.total_seconds() / divisor
        elif op == "log10":
            values = pd.to_numeric(out[inputs[0]], errors="coerce")
            out[output or f"log10_{inputs[0]}"] = np.where(values > 0, np.log10(values), np.nan)
        elif op in {"ratio", "percentage"}:
            numerator = pd.to_numeric(out[inputs[0]], errors="coerce")
            denominator = pd.to_numeric(out[inputs[1]], errors="coerce").replace(0, np.nan)
            values = numerator / denominator
            out[output or f"{inputs[0]}_per_{inputs[1]}"] = values * (100.0 if op == "percentage" else 1.0)
        elif op == "ema":
            alpha = float(params["alpha"])
            if not 0 < alpha <= 1:
                raise ValueError("ema alpha must be in (0, 1]")
            out[output or f"ema_{inputs[0]}"] = pd.to_numeric(out[inputs[0]], errors="coerce").ewm(alpha=alpha, adjust=False).mean()
        elif op == "intraday_range":
            high = pd.to_numeric(out[inputs[0]], errors="coerce")
            low = pd.to_numeric(out[inputs[1]], errors="coerce")
            out[output or "intraday_range"] = high - low
        elif op == "period_return":
            values = pd.to_numeric(out[inputs[0]], errors="coerce")
            out[output or f"return_{inputs[0]}"] = values.pct_change(fill_method=None)
        elif op == "deduplicate_at_grain":
            keys = inputs or list(recipe.output_grain)
            if not keys:
                raise ValueError("deduplicate_at_grain requires keys")
            out = out.drop_duplicates(subset=keys, keep=str(params.get("keep", "first"))).reset_index(drop=True)
        else:
            raise ValueError(f"unsupported feature operation: {op}")

        diagnostics.append({
            "step": index, "operation": op, "rows_before": before_rows,
            "rows_after": len(out), "output": output,
        })
    if recipe.output_grain and out.duplicated(subset=list(recipe.output_grain)).any():
        raise ValueError(f"output grain is not unique: {list(recipe.output_grain)}")
    return FeatureResult(out, recipe.fingerprint(), tuple(diagnostics))
