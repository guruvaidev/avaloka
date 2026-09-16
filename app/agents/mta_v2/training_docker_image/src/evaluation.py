"""Portable holdout-baseline evaluation shared by local and Ray training."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


TIME_ORDER_COLUMN = "__mta_time_order__"
DEFAULT_TIME_SERIES_FOLDS = 5


def add_time_order_column(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """Return a copy with a validated, sortable numeric time key.

    Numeric sequence/timestamp columns retain their numeric order. Other values
    must all parse as datetimes. Missing or malformed timestamps are rejected so
    validation cannot silently mix unknown dates into a chronological split.
    """
    if not time_column or time_column not in frame.columns:
        raise ValueError(
            f"Time column {time_column!r} is not present in the training dataset."
        )

    values = frame[time_column]
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        order_values = numeric
    else:
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        invalid = int(parsed.isna().sum())
        if invalid:
            raise ValueError(
                f"Time column {time_column!r} contains {invalid} missing or invalid "
                "value(s). Every row needs a valid timestamp for time-ordered training."
            )
        order_values = parsed.astype("int64")

    result = frame.copy()
    result[TIME_ORDER_COLUMN] = order_values.to_numpy()
    return result


def time_series_split_metadata(
    n_rows: int,
    time_column: str,
    n_splits: int = DEFAULT_TIME_SERIES_FOLDS,
) -> Dict[str, Any]:
    """Describe the final expanding-window fold used for model training.

    MTA trains one model, so it uses the final fold from ``TimeSeriesSplit``:
    the largest available historical training window and the latest validation
    window. The returned row counts are also usable by Ray's distributed split.
    """
    if n_rows < 3:
        raise ValueError(
            "Time-ordered training requires at least 3 rows to create an earlier "
            "training window and a later validation window."
        )
    effective_splits = min(max(2, int(n_splits)), n_rows - 1)
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    train_index, validation_index = list(splitter.split(range(n_rows)))[-1]
    return {
        "splitter": "TimeSeriesSplit",
        "splitter_reason": (
            f"rows are ordered by {time_column!r}; the final expanding-window fold "
            "trains only on earlier rows and validates on later rows, preventing "
            "future-to-past leakage"
        ),
        "time_column": time_column,
        "n_folds": effective_splits,
        "n_train_rows": int(len(train_index)),
        "n_validation_rows": int(len(validation_index)),
    }


def _finite_numbers(values: Iterable[Any]) -> List[float]:
    numbers: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def build_holdout_evaluation_report(
    train_targets: Iterable[Any],
    validation_targets: Iterable[Any],
    model_type: str,
    model_metrics: Dict[str, Any],
    *,
    splitter: str = "DeterministicHashHoldout",
    splitter_reason: Optional[str] = None,
    n_folds: int = 1,
    time_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare validation performance with a trivial training-only baseline.

    Classification always predicts the majority class learned from the
    training split. Regression always predicts the training-target mean.
    Neither baseline reads validation statistics while being fitted.
    """
    task = "regression" if "regression" in str(model_type).lower() else "classification"
    train = list(train_targets)
    validation = list(validation_targets)
    report: Dict[str, Any] = {
        "task": task,
        "splitter": splitter,
        "splitter_reason": splitter_reason or (
            "the trainer uses a deterministic 80/20 row-hash split; "
            "baseline parameters are fitted on training rows only"
        ),
        "n_folds": int(n_folds),
        "time_column": time_column,
        "n_train_rows": len(train),
        "n_validation_rows": len(validation),
        "metrics": {},
        "baseline": {},
        "primary_metric": "rmse" if task == "regression" else "accuracy",
        "beats_baseline": None,
        "warnings": [],
    }

    if not train or not validation:
        report["warnings"].append("baseline comparison skipped because a split is empty")
        return report

    if task == "classification":
        train_labels = [str(value) for value in train]
        validation_labels = [str(value) for value in validation]
        majority = Counter(train_labels).most_common(1)[0][0]
        baseline_accuracy = sum(
            label == majority for label in validation_labels
        ) / len(validation_labels)
        model_accuracy = model_metrics.get("accuracy")
        report["baseline"] = {
            "strategy": "most_frequent_training_class",
            "predicted_class": majority,
            "accuracy": round(float(baseline_accuracy), 6),
        }
        if model_accuracy is not None:
            model_accuracy = float(model_accuracy)
            report["metrics"]["accuracy"] = round(model_accuracy, 6)
            report["beats_baseline"] = model_accuracy > baseline_accuracy
    else:
        train_numbers = _finite_numbers(train)
        validation_numbers = _finite_numbers(validation)
        if not train_numbers or not validation_numbers:
            report["warnings"].append(
                "baseline comparison skipped because target values are not numeric"
            )
            return report
        train_mean = sum(train_numbers) / len(train_numbers)
        baseline_mae = sum(abs(value - train_mean) for value in validation_numbers) / len(validation_numbers)
        baseline_rmse = math.sqrt(
            sum((value - train_mean) ** 2 for value in validation_numbers)
            / len(validation_numbers)
        )
        model_rmse = model_metrics.get("val_rmse")
        report["baseline"] = {
            "strategy": "training_target_mean",
            "prediction": round(train_mean, 6),
            "mae": round(baseline_mae, 6),
            "rmse": round(baseline_rmse, 6),
        }
        if model_rmse is not None:
            model_rmse = float(model_rmse)
            report["metrics"]["rmse"] = round(model_rmse, 6)
            if model_metrics.get("val_mae") is not None:
                report["metrics"]["mae"] = round(float(model_metrics["val_mae"]), 6)
            if model_metrics.get("val_r2_score") is not None:
                report["metrics"]["r2"] = round(float(model_metrics["val_r2_score"]), 6)
            report["beats_baseline"] = model_rmse < baseline_rmse

    if report["beats_baseline"] is False:
        report["warnings"].append(
            f"model does not beat the trivial baseline on {report['primary_metric']}"
        )
    return report
