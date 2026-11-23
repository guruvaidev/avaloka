"""Small, auditable split-conformal calibration primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConformalPlan:
    task: str  # regression | classification
    target_coverage: float = 0.9
    method: str = "split"


@dataclass(frozen=True)
class ConformalResult:
    task: str
    target_coverage: float
    quantile: float
    predictions: Any
    empirical_coverage: float | None
    efficiency: float
    diagnostics: tuple[dict[str, Any], ...]


def _validate_coverage(coverage: float, calibration_size: int) -> float:
    if not 0 < coverage < 1:
        raise ValueError("target coverage must be in (0, 1)")
    if calibration_size < 2:
        raise ValueError("conformal calibration requires at least two observations")
    rank = math.ceil((calibration_size + 1) * coverage)
    return min(1.0, rank / calibration_size)


def split_conformal_regression(
    calibration_y: np.ndarray,
    calibration_prediction: np.ndarray,
    test_prediction: np.ndarray,
    *,
    target_coverage: float = 0.9,
    test_y: np.ndarray | None = None,
) -> ConformalResult:
    """Create symmetric split-conformal regression intervals."""
    y = np.asarray(calibration_y, dtype=float)
    prediction = np.asarray(calibration_prediction, dtype=float)
    test_prediction = np.asarray(test_prediction, dtype=float)
    if y.shape != prediction.shape:
        raise ValueError("calibration labels and predictions must have identical shape")
    valid = np.isfinite(y) & np.isfinite(prediction)
    scores = np.abs(y[valid] - prediction[valid])
    quantile_level = _validate_coverage(target_coverage, len(scores))
    quantile = float(np.quantile(scores, quantile_level, method="higher"))
    intervals = np.column_stack((test_prediction - quantile, test_prediction + quantile))
    empirical = None
    if test_y is not None:
        actual = np.asarray(test_y, dtype=float)
        if actual.shape != test_prediction.shape:
            raise ValueError("test labels and predictions must have identical shape")
        empirical = float(((actual >= intervals[:, 0]) & (actual <= intervals[:, 1])).mean())
    diagnostics = ({
        "kind": "exchangeability",
        "detail": "Coverage is marginal and assumes calibration/test exchangeability.",
    },)
    return ConformalResult(
        task="regression", target_coverage=target_coverage, quantile=quantile,
        predictions=intervals, empirical_coverage=empirical,
        efficiency=float(np.mean(intervals[:, 1] - intervals[:, 0])),
        diagnostics=diagnostics,
    )


def split_conformal_classification(
    calibration_y: np.ndarray,
    calibration_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    *,
    target_coverage: float = 0.9,
    test_y: np.ndarray | None = None,
) -> ConformalResult:
    """Create label sets from 1 - true-class probability nonconformity."""
    labels = np.asarray(calibration_y, dtype=int)
    calibration = np.asarray(calibration_probabilities, dtype=float)
    test = np.asarray(test_probabilities, dtype=float)
    if calibration.ndim != 2 or test.ndim != 2 or calibration.shape[1] != test.shape[1]:
        raise ValueError("probability inputs must be 2D with matching class count")
    if len(labels) != len(calibration) or np.any(labels < 0) or np.any(labels >= calibration.shape[1]):
        raise ValueError("calibration labels do not match probability rows/classes")
    if not np.allclose(calibration.sum(axis=1), 1.0, atol=1e-5) or not np.allclose(test.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("classification probabilities must sum to one")
    scores = 1.0 - calibration[np.arange(len(labels)), labels]
    quantile_level = _validate_coverage(target_coverage, len(scores))
    quantile = float(np.quantile(scores, quantile_level, method="higher"))
    tolerance = np.finfo(float).eps * max(1.0, abs(quantile)) * 8
    included = (1.0 - test) <= quantile + tolerance
    prediction_sets = [np.flatnonzero(row).tolist() for row in included]
    empirical = None
    if test_y is not None:
        actual = np.asarray(test_y, dtype=int)
        if len(actual) != len(prediction_sets):
            raise ValueError("test labels do not match test probabilities")
        empirical = float(np.mean([label in values for label, values in zip(actual, prediction_sets)]))
    diagnostics = ({
        "kind": "marginal_coverage",
        "detail": "Group-conditional coverage is not guaranteed; audit important segments.",
    },)
    return ConformalResult(
        task="classification", target_coverage=target_coverage, quantile=quantile,
        predictions=prediction_sets, empirical_coverage=empirical,
        efficiency=float(np.mean([len(values) for values in prediction_sets])),
        diagnostics=diagnostics,
    )
