"""Deterministic statistics used across fireflies.

These functions produce the *facts* (schema, distributions, quality issues,
leakage signals) that the rest of the system reasons about. Keeping them pure
and LLM-free is deliberate: a defensible analysis report must rest on measured
numbers, not generated ones.
"""

from __future__ import annotations

from typing import Any

import re
import warnings

import numpy as np
import pandas as pd

# Heuristic thresholds. Centralised so they are inspectable and tunable.
HIGH_MISSING_PCT = 0.30
HIGH_CARDINALITY_RATIO = 0.90
ID_UNIQUE_RATIO = 0.98
SMALL_SAMPLE_ROWS = 200


def infer_role(series: pd.Series, n_rows: int) -> str:
    """Classify a column into a modelling role from its content."""
    non_null = series.dropna()
    n_unique = non_null.nunique()
    if n_unique <= 1:
        return "constant"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        # Numeric but near-unique integer-like -> probably an identifier.
        if n_rows and n_unique / max(1, len(non_null)) >= ID_UNIQUE_RATIO and _looks_integral(non_null):
            return "id"
        return "numeric"
    # object / category
    # Dates arrive from CSV as strings, and a daily date column is near-unique,
    # so the identifier rule below used to claim it. That is not cosmetic: the
    # planner picks a time-based split only when it can see a datetime column,
    # so a time-ordered dataset was validated with a RANDOM split -- training on
    # future rows to predict past ones, which flatters every metric it reports.
    if _looks_like_dates(non_null):
        return "datetime"
    if n_rows and n_unique / max(1, len(non_null)) >= ID_UNIQUE_RATIO:
        return "id"
    avg_len = non_null.astype(str).str.len().mean() if len(non_null) else 0
    if avg_len and avg_len > 40:
        return "text"
    return "categorical"


#: A value must carry a four-digit year and two separators to be considered for
#: date parsing. Without the year requirement, a semantic version ("1.2.0") and
#: an order number both parse as timestamps, and an identifier column becomes
#: the thing the train/test split is ordered by -- a worse error than missing a
#: date column, because the metrics still look fine.
_DATE_HINT_RE = re.compile(
    r"^\s*(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})(?:[T\s]|$)"
)


def _looks_like_dates(non_null: pd.Series, threshold: float = 0.95) -> bool:
    """True when a string column is really a date column.

    Deliberately strict. Mistaking an identifier for a date would order the
    split by it, which is a worse error than missing a date column: the metrics
    would still look fine.
    """
    if not len(non_null):
        return False
    try:
        text = non_null.astype(str)
    except Exception:
        return False
    sample = text.head(200)
    if (sample.str.match(_DATE_HINT_RE).mean() or 0) < threshold:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    except Exception:
        return False
    return bool(parsed.notna().mean() >= threshold)


def _looks_integral(series: pd.Series) -> bool:
    try:
        return bool(np.all(np.equal(np.mod(series.to_numpy(dtype="float64"), 1), 0)))
    except (TypeError, ValueError):
        return False


def profile_column(series: pd.Series, n_rows: int) -> dict[str, Any]:
    role = infer_role(series, n_rows)
    missing = int(series.isna().sum())
    n_unique = int(series.dropna().nunique())
    prof: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "role": role,
        "missing": missing,
        "missing_pct": round(missing / n_rows, 4) if n_rows else 0.0,
        "n_unique": n_unique,
        "unique_pct": round(n_unique / n_rows, 4) if n_rows else 0.0,
    }

    non_null = series.dropna()
    if role == "numeric" and len(non_null):
        arr = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(arr):
            q1, q3 = float(arr.quantile(0.25)), float(arr.quantile(0.75))
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            prof["stats"] = {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": round(float(arr.mean()), 6),
                "median": round(float(arr.median()), 6),
                "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 6),
                "p25": q1,
                "p75": q3,
                "skew": round(float(arr.skew()) if len(arr) > 2 else 0.0, 4),
                "n_outliers": int(((arr < lo) | (arr > hi)).sum()),
            }
    elif role in {"categorical", "boolean", "id"} and len(non_null):
        vc = non_null.astype(str).value_counts().head(10)
        prof["top_values"] = [{"value": k, "count": int(v)} for k, v in vc.items()]

    prof["sample_values"] = [
        _jsonable(v) for v in non_null.head(5).tolist()
    ]
    return prof


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return str(v) if not isinstance(v, (int, float, bool, str)) else v


def quality_report(frame: pd.DataFrame, columns: list[dict[str, Any]]) -> dict[str, Any]:
    """Score data quality 0–100 and enumerate concrete, actionable issues."""
    n_rows = len(frame)
    issues: list[dict[str, Any]] = []

    duplicates = int(frame.duplicated().sum())
    if duplicates:
        issues.append({
            "severity": "medium",
            "column": None,
            "kind": "duplicate_rows",
            "detail": f"{duplicates} duplicate rows ({duplicates / n_rows:.1%}).",
        })

    constant_cols, high_missing, high_card, id_cols = [], [], [], []
    for col in columns:
        name = col["name"]
        if col["role"] == "constant":
            constant_cols.append(name)
            issues.append({"severity": "low", "column": name, "kind": "constant_column",
                           "detail": "Column has a single value; no predictive content."})
        if col["missing_pct"] >= HIGH_MISSING_PCT:
            high_missing.append(name)
            issues.append({"severity": "high", "column": name, "kind": "high_missingness",
                           "detail": f"{col['missing_pct']:.1%} missing."})
        if col["role"] == "categorical" and col["unique_pct"] >= HIGH_CARDINALITY_RATIO:
            high_card.append(name)
            issues.append({"severity": "medium", "column": name, "kind": "high_cardinality",
                           "detail": f"{col['n_unique']} distinct values ({col['unique_pct']:.1%})."})
        if col["role"] == "id":
            id_cols.append(name)

    # Score: start at 100, deduct for issues weighted by severity, plus a small
    # penalty for tiny samples (limits statistical confidence downstream).
    weights = {"high": 12, "medium": 6, "low": 2}
    score = 100 - sum(weights[i["severity"]] for i in issues)
    if n_rows < SMALL_SAMPLE_ROWS:
        score -= 10
        issues.append({"severity": "medium", "column": None, "kind": "small_sample",
                       "detail": f"Only {n_rows} rows; statistical conclusions are low-confidence."})
    score = max(0, min(100, score))

    return {
        "score": score,
        "n_rows": n_rows,
        "n_cols": len(columns),
        "duplicate_rows": duplicates,
        "constant_columns": constant_cols,
        "high_missing_columns": high_missing,
        "high_cardinality_columns": high_card,
        "id_columns": id_cols,
        "issues": issues,
    }


def detect_target_leakage(
    frame: pd.DataFrame, target: str, columns: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Flag features that are suspiciously predictive of the target.

    A correlation/association at or near 1.0 usually means the feature encodes
    the outcome (or is computed from it) — a classic leakage source. This is the
    Validator's core defensive check.
    """
    findings: list[dict[str, Any]] = []
    if target not in frame.columns:
        return findings
    y = frame[target]
    y_is_numeric = pd.api.types.is_numeric_dtype(y)

    for col in columns:
        name = col["name"]
        if name == target:
            continue
        # The profile describes the dataset as loaded; the frame has been
        # transformed since, and the data engineer drops constant columns. A
        # dropped column cannot leak, and indexing the frame for it raised
        # KeyError -- which crashed the whole training mission on any dataset
        # carrying a single-valued column, which is most real ones.
        if name not in frame.columns:
            continue
        # Identifier-like / near-unique columns trivially "determine" the target
        # (each row is its own group), so they are not genuine leakage signals.
        if col.get("role") == "id" or col.get("unique_pct", 0.0) >= 0.5:
            continue
        s = frame[name]
        try:
            if y_is_numeric and pd.api.types.is_numeric_dtype(s):
                corr = abs(float(pd.to_numeric(s, errors="coerce").corr(pd.to_numeric(y, errors="coerce"))))
                if corr >= 0.98:
                    findings.append({"column": name, "kind": "near_perfect_correlation",
                                     "value": round(corr, 4), "severity": "high",
                                     "detail": f"|corr| with target = {corr:.3f}; likely leakage."})
            else:
                # Genuine categorical: does each value map to exactly one label?
                grp = frame.groupby(s.astype("string"))[target].nunique(dropna=True)
                if len(grp) > 1 and (grp <= 1).mean() >= 0.99:
                    findings.append({"column": name, "kind": "deterministic_mapping",
                                     "value": 1.0, "severity": "high",
                                     "detail": "Feature value determines the target; likely leakage."})
        except Exception:
            continue
    return findings


def correlation_pairs(frame: pd.DataFrame, threshold: float = 0.9, top: int = 15) -> list[dict[str, Any]]:
    """Top absolute pairwise correlations among numeric columns."""
    num = frame.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return []
    corr = num.corr(numeric_only=True).abs()
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append({"a": cols[i], "b": cols[j], "corr": round(float(v), 4)})
    pairs.sort(key=lambda p: p["corr"], reverse=True)
    flagged = [p for p in pairs if p["corr"] >= threshold]
    return (flagged or pairs)[:top]
