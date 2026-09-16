"""Portable data-integrity checks shared by local and Ray training.

This module intentionally depends only on pandas.  It is copied into the
standalone Ray training image as well as the full Avaloka image, which keeps the
last pre-training safety gate identical in both execution paths.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import pandas as pd


DUPLICATE_BLOCKING_RATIO = 0.01
LEAKAGE_CORRELATION_THRESHOLD = 0.98
_IDENTIFIER_NAME = re.compile(
    r"(?i:(?:^|[_\W])(id|uuid|guid|key|index)(?:$|[_\W]))|(?:[a-z0-9](?:Id|ID))$"
)
_ROW_IDENTIFIER_STEMS = {
    "id", "index", "row", "record", "uuid", "guid", "passenger",
    "event", "request", "session", "order", "transaction", "invoice",
    "receipt", "booking", "reservation", "trip",
}
_GROUP_IDENTIFIER_STEMS = {
    "user", "customer", "account", "patient", "employee", "member",
    "device", "household",
}
_ENTITY_IDENTIFIER_STEMS = {
    "product", "item", "sku", "store", "vendor", "merchant", "brand",
    "category", "department", "aisle", "channel", "location", "warehouse",
    "movie", "book", "game", "campaign", "ratecode",
}


def _categorical_codes(series: pd.Series) -> pd.Series | None:
    """Return stable codes for a low-cardinality categorical column."""
    try:
        if series.nunique(dropna=True) > 50:
            return None
        return series.astype("category").cat.codes
    except Exception:
        return None


def find_target_leakage(
    df: pd.DataFrame,
    target: str,
    feature_columns: Sequence[str],
    *,
    threshold: float = LEAKAGE_CORRELATION_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Return features that duplicate or almost perfectly encode ``target``."""
    if not target or target not in df.columns:
        return []

    matches: List[Dict[str, Any]] = []
    y = df[target]
    for column in feature_columns:
        if column == target:
            matches.append({
                "column": column,
                "method": "target_in_features",
                "correlation": 1.0,
            })
            continue
        if column not in df.columns:
            continue

        x = df[column]
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 2:
            continue
        x_valid = x.loc[valid]
        y_valid = y.loc[valid]
        if y_valid.nunique(dropna=True) < 2:
            continue

        try:
            exact_copy = bool(
                x_valid.reset_index(drop=True)
                .eq(y_valid.reset_index(drop=True))
                .all()
            )
        except Exception:
            exact_copy = False
        if exact_copy:
            matches.append({
                "column": column,
                "method": "exact_copy",
                "correlation": 1.0,
            })
            continue

        x_numeric = (
            x_valid
            if pd.api.types.is_numeric_dtype(x_valid)
            else _categorical_codes(x_valid)
        )
        y_numeric = (
            y_valid
            if pd.api.types.is_numeric_dtype(y_valid)
            else _categorical_codes(y_valid)
        )
        if x_numeric is None or y_numeric is None:
            continue
        try:
            correlation = abs(float(x_numeric.corr(y_numeric)))
        except Exception:
            continue
        if correlation != correlation:
            continue
        if correlation >= threshold:
            matches.append({
                "column": column,
                "method": "near_perfect_correlation",
                "correlation": round(correlation, 6),
            })
    return matches


def target_leakage_error(target: str, matches: Sequence[Dict[str, Any]]) -> str:
    """Build a deterministic target-leakage error for failure diagnosis."""
    columns = ", ".join(repr(str(match["column"])) for match in matches)
    return (
        f"Target leakage detected: feature column(s) {columns} duplicate or almost "
        f"perfectly encode target {target!r}. Remove these columns from the training "
        "plan before training."
    )


def find_duplicate_training_rows(
    df: pd.DataFrame,
    target: str,
    feature_columns: Sequence[str],
    *,
    threshold: float = DUPLICATE_BLOCKING_RATIO,
) -> Dict[str, Any] | None:
    """Find exact repeated source records.

    Repeated values after projecting onto selected model features are not enough
    to establish duplication: multiple real observations can legitimately have
    the same inputs and target. Comparing the complete source row avoids blocking
    valid low-cardinality and entity-level datasets.
    """
    if not target or target not in df.columns or len(df) < 2:
        return None

    if not any(
        column != target and column in df.columns
        for column in feature_columns
    ):
        return None

    compared_columns = list(df.columns)
    try:
        duplicate_count = int(df.duplicated(subset=compared_columns, keep="first").sum())
    except Exception:
        return None
    duplicate_ratio = duplicate_count / max(int(len(df)), 1)
    if duplicate_count == 0 or duplicate_ratio < threshold:
        return None
    return {
        "duplicate_count": duplicate_count,
        "duplicate_ratio": round(duplicate_ratio, 6),
        "compared_columns": [str(column) for column in compared_columns],
        "excluded_identifier_columns": [],
    }


def looks_like_identifier_name(column: str) -> bool:
    """Return whether a column name resembles any kind of identifier."""
    return bool(_IDENTIFIER_NAME.search(str(column).strip()))


def _identifier_stem(column: str) -> str:
    """Return the semantic portion of an ID-like column name."""
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(column).strip())
    tokens = [
        token for token in re.split(r"[^a-z0-9]+", value.lower())
        if token not in {"id", "uuid", "guid", "key", "index"}
    ]
    return tokens[-1] if tokens else value.lower()


def classify_identifier_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    uniqueness_threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    """Classify ID-looking columns using names *and* sampled values.

    A name containing ``id`` is not enough to discard a feature. Near-unique
    row keys and occurrence/group keys are excluded, while repeated entity
    keys such as ``product_id`` or ``store_id`` are retained. The latter can
    carry real predictive information across observations.

    ``df`` may be a bounded sample. All column names are considered so the
    recorded evidence can show the other identifiers present in the dataset.
    """
    row_count = int(len(df))
    identifier_columns = [
        str(column) for column in columns if looks_like_identifier_name(str(column))
    ]
    decisions: List[Dict[str, Any]] = []

    for column in identifier_columns:
        stem = _identifier_stem(column)
        non_null_count = 0
        unique_count = 0
        if column in df.columns and row_count:
            try:
                non_null_count = int(df[column].notna().sum())
                unique_count = int(df[column].nunique(dropna=True))
            except Exception:
                non_null_count = 0
                unique_count = 0

        uniqueness_ratio = (
            unique_count / non_null_count if non_null_count else None
        )
        repeated_values = max(non_null_count - unique_count, 0)

        if stem in _ROW_IDENTIFIER_STEMS:
            role = "row_or_occurrence_identifier"
            recommendation = "exclude"
            reason = "its name denotes a row, event, or transaction key"
        elif stem in _GROUP_IDENTIFIER_STEMS:
            role = "group_identifier"
            recommendation = "exclude"
            reason = (
                "its name denotes a person/account grouping key that can leak "
                "identity across validation rows"
            )
        elif uniqueness_ratio is not None and uniqueness_ratio >= uniqueness_threshold:
            role = "near_unique_identifier"
            recommendation = "exclude"
            reason = (
                f"{unique_count} of {non_null_count} sampled non-null values are unique"
            )
        elif stem in _ENTITY_IDENTIFIER_STEMS and (
            uniqueness_ratio is None or repeated_values > 0
        ):
            role = "repeated_entity_identifier"
            recommendation = "keep"
            reason = (
                "its name denotes a reusable entity and sampled values repeat"
                if uniqueness_ratio is not None
                else "its name denotes a reusable entity; no sample was available"
            )
        elif uniqueness_ratio is not None and repeated_values >= max(2, int(non_null_count * 0.05)):
            role = "repeated_entity_identifier"
            recommendation = "keep"
            reason = (
                f"sampled values repeat {repeated_values} times, so it does not "
                "behave like a row key"
            )
        else:
            role = "unresolved_identifier"
            recommendation = "exclude"
            reason = "there is not enough evidence that it is a reusable entity key"

        decisions.append({
            "column": column,
            "role": role,
            "recommendation": recommendation,
            "reason": reason,
            "sample_rows": row_count,
            "non_null_count": non_null_count,
            "unique_count": unique_count,
            "uniqueness_ratio": (
                round(float(uniqueness_ratio), 6)
                if uniqueness_ratio is not None
                else None
            ),
            "related_identifier_columns": [
                other for other in identifier_columns if other != column
            ],
        })

    return decisions


def find_identifier_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    uniqueness_threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    """Return selected identifier columns that should not be model inputs."""
    return [
        decision
        for decision in classify_identifier_features(
            df,
            feature_columns,
            uniqueness_threshold=uniqueness_threshold,
        )
        if decision["recommendation"] == "exclude"
    ]


def identifier_features_error(matches: Sequence[Dict[str, Any]]) -> str:
    """Build a deterministic identifier-feature error for failure diagnosis."""
    columns = ", ".join(repr(str(match["column"])) for match in matches)
    return (
        f"Identifier-like feature detected: feature column(s) {columns} behave as "
        "row, occurrence, or grouping identifiers and should not be used directly "
        "for prediction. Remove these columns from the training plan before training."
    )


def duplicate_rows_error(match: Dict[str, Any]) -> str:
    """Build a deterministic duplicate-record error for failure diagnosis."""
    duplicate_count = int(match.get("duplicate_count") or 0)
    duplicate_ratio = float(match.get("duplicate_ratio") or 0.0)
    excluded = match.get("excluded_identifier_columns") or []
    identifier_note = ""
    if excluded:
        labels = ", ".join(repr(str(column)) for column in excluded)
        identifier_note = f" after excluding identifier column(s) {labels}"
    return (
        f"Duplicate training rows detected: {duplicate_count} repeated row(s) "
        f"({duplicate_ratio:.1%}){identifier_note}. Training was blocked because "
        "copied records can leak across the training and validation split."
    )


def build_training_integrity_report(
    df: pd.DataFrame,
    target: str,
    feature_columns: Sequence[str],
) -> Dict[str, Any]:
    """Return portable integrity evidence recorded with every MTA run."""
    duplicate_match = find_duplicate_training_rows(df, target, feature_columns)
    identifier_matches = find_identifier_features(df, feature_columns)

    findings: List[Dict[str, Any]] = []
    if duplicate_match:
        findings.append({
            "check": "duplicate_rows",
            "severity": "blocker",
            "column": None,
            "detail": duplicate_rows_error(duplicate_match),
            "evidence": dict(duplicate_match),
        })
    for match in identifier_matches:
        findings.append({
            "check": "identifier_feature",
            "severity": "blocker",
            "column": str(match["column"]),
            "detail": identifier_features_error([match]),
            "evidence": dict(match),
        })

    return {
        "safe_to_train": not findings,
        "rows": int(len(df)),
        "checked_columns": len(feature_columns),
        "n_findings": len(findings),
        "n_blockers": len(findings),
        "findings": findings,
    }
