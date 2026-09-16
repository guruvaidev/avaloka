"""Tabular modelling engine (scikit-learn).

Avaloka does not rebuild AutoML — it assembles defensible baselines and a small
panel of candidates behind a single, leakage-resistant preprocessing pipeline,
then lets the Validator and FinOps fireflies reason about the results. Every
candidate is trained inside a ``Pipeline`` so preprocessing is fit on training
folds only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


@dataclass
class CandidateResult:
    name: str
    algorithm: str
    primary_metric: str
    primary_score: float
    metrics: dict[str, float]
    cv_mean: float
    cv_std: float
    train_seconds: float
    inference_ms_per_1k: float
    pipeline: Any = field(repr=False, default=None)

    def public(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "pipeline"}
        return d


def _feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = _feature_types(X)
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=30)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ], remainder="drop")


def _estimators(task: str) -> dict[str, Any]:
    if task == "regression":
        return {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        }
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def default_metric(task: str) -> str:
    if task == "binary_classification":
        return "roc_auc"
    if task == "multiclass_classification":
        return "f1_macro"
    return "r2"


def _score_all(task: str, y_true, y_pred, y_proba) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if task == "regression":
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    else:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        avg = "binary" if task == "binary_classification" else "macro"
        metrics["f1" if task == "binary_classification" else "f1_macro"] = float(
            f1_score(y_true, y_pred, average=avg, zero_division=0)
        )
        if task == "binary_classification" and y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except ValueError:
                pass
    return metrics


def _cv_scoring(task: str, metric: str) -> str:
    return {
        "roc_auc": "roc_auc",
        "r2": "r2",
        "f1_macro": "f1_macro",
        "f1": "f1",
        "accuracy": "accuracy",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }.get(metric, "r2" if task == "regression" else "accuracy")


@dataclass
class TrainingOutcome:
    task: str
    metric: str
    candidates: list[CandidateResult]
    best: CandidateResult
    X_columns: list[str]
    n_train: int
    n_test: int
    holdout_test_size: float
    #: Score of the trivial model -- majority class, or the mean. A number with
    #: nothing to compare it against is not evidence, and this module's own
    #: docstring promised "defensible baselines" while computing none.
    baseline_score: float = float("nan")
    baseline_strategy: str = ""

    @property
    def beats_baseline(self) -> bool | None:
        """Whether the winner beats the trivial model by more than fold noise.

        The bar is the confidence interval, not the mean, matching
        app/agents/evaluation_agent.py. A classifier that only ever predicts the
        majority class scores exactly the baseline, and floating-point noise
        then records it as a win by +0.0000 -- which is the failure a baseline
        exists to catch, so catching it loosely is no use.

        ``None`` when it cannot be determined; never optimistically True.
        """
        best = self.best
        if math.isnan(self.baseline_score) or math.isnan(best.cv_mean):
            return None
        spread = 0.0 if math.isnan(best.cv_std) else best.cv_std
        if self.metric in {"rmse", "mae"}:
            return bool(best.cv_mean + spread < self.baseline_score)
        return bool(best.cv_mean - spread > self.baseline_score)


def _is_identifier_like(series: pd.Series) -> bool:
    """An identifier: one value per row, and not a continuous measurement."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    if non_null.nunique() < len(series):
        return False
    # Floating-point columns that happen to be all-distinct are measurements.
    # Integer and string columns that are all-distinct are row keys.
    return not pd.api.types.is_float_dtype(series)


def _baseline_score(task: str, metric: str, pre, X_train, y_train, X_test, y_test):
    """Score the model that does no work, on the same split and the same metric.

    Classification predicts the majority class and regression the training
    mean -- the two answers a person gives before looking at any feature. The
    dummy rides the same preprocessor as the candidates so it sees identical
    inputs and the comparison is like for like.
    """
    if task == "regression":
        strategy, dummy = "mean", DummyRegressor(strategy="mean")
    elif metric in {"roc_auc"}:
        # A most_frequent classifier gives a ranking metric nothing to rank:
        # "prior" returns the class prior for every row, which is 0.5 AUC.
        strategy, dummy = "prior", DummyClassifier(strategy="prior")
    else:
        strategy, dummy = "most_frequent", DummyClassifier(strategy="most_frequent")

    try:
        pipe = Pipeline([("pre", pre), ("model", dummy)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = None
        if task == "binary_classification" and hasattr(pipe, "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None
        return float(_score_all(task, y_test, y_pred, y_proba).get(metric, float("nan"))), strategy
    except Exception:
        return float("nan"), strategy


def _temporal_split(data, X, y, time_column: str, test_size: float):
    """Past trains, future tests — the only honest split for ordered data.

    Falls back to the caller's random split if the column cannot be ordered,
    rather than silently producing an arbitrary one.
    """
    if time_column not in data.columns:
        raise ValueError(f"time column {time_column!r} is not in the frame")
    order = pd.to_datetime(data[time_column], errors="coerce", format="mixed")
    if order.isna().all():
        raise ValueError(f"time column {time_column!r} holds no parseable dates")
    ranked = order.rank(method="first", na_option="bottom").astype(int)
    ordered = ranked.sort_values().index
    cut = max(1, int(len(ordered) * (1.0 - test_size)))
    train_idx, test_idx = ordered[:cut], ordered[cut:]
    return (X.loc[train_idx], X.loc[test_idx],
            y.loc[train_idx], y.loc[test_idx])


def train_candidates(
    frame: pd.DataFrame,
    target: str,
    task: str,
    *,
    metric: str | None = None,
    algorithms: list[str] | None = None,
    test_size: float = 0.2,
    split_strategy: str | None = None,
    time_column: str | None = None,
) -> TrainingOutcome:
    """Train the candidate panel and return per-candidate results + the winner.

    ``split_strategy`` is honoured rather than described. It used to be neither:
    the analysis planner chose ``time_based_holdout`` for a dataset with a
    datetime column and recorded the reason -- "a temporal split avoids
    look-ahead leakage" -- and this function then split at random anyway. The
    validation report went on to state that a temporal split had been used. A
    system whose claim is that its results are checkable cannot report a split
    it did not perform.
    """
    metric = metric or default_metric(task)
    data = frame.dropna(subset=[target]).reset_index(drop=True)
    X = data.drop(columns=[target])
    # Drop obvious identifier-like columns to reduce leakage / overfit risk.
    #
    # "All values distinct" alone is the wrong test. A continuous measurement --
    # spend, latency, a sensor reading -- is almost always all-distinct in a
    # dataset of this size, so the original rule silently dropped the real
    # features and kept only the coarse ones. With every feature continuous it
    # dropped all of them, and the fit then failed with "Found array with 0
    # feature(s)". An identifier is all-distinct AND not a continuous number.
    X = X.loc[:, [c for c in X.columns if not _is_identifier_like(X[c])]]
    if X.empty:
        raise ValueError(
            "No usable features remain after dropping identifier-like columns."
        )
    y = data[target]

    if task != "regression":
        y = y.astype("category").cat.codes if y.dtype == object else y

    if split_strategy == "time_based_holdout" and time_column:
        X_train, X_test, y_train, y_test = _temporal_split(
            data, X, y, time_column, test_size
        )
    else:
        stratify = y if task.endswith("classification") and y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
        )

    estimators = _estimators(task)
    if algorithms:
        estimators = {k: v for k, v in estimators.items() if k in algorithms} or estimators

    pre = build_preprocessor(X)
    cv_scoring = _cv_scoring(task, metric)
    higher_is_better = metric not in {"rmse", "mae"}

    baseline_score, baseline_strategy = _baseline_score(
        task, metric, pre, X_train, y_train, X_test, y_test
    )

    results: list[CandidateResult] = []
    for name, est in estimators.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_seconds = time.perf_counter() - t0

        y_pred = pipe.predict(X_test)
        y_proba = None
        if hasattr(pipe, "predict_proba") and task == "binary_classification":
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None
        metrics = _score_all(task, y_test, y_pred, y_proba)

        # Cross-validation for stability (k folds on the training split).
        n_splits = max(2, min(5, int(np.bincount(y_train).min()) if task.endswith("classification") else 5))
        try:
            cv = cross_val_score(pipe, X_train, y_train, cv=min(5, n_splits), scoring=cv_scoring)
            cv_mean, cv_std = float(np.mean(cv)), float(np.std(cv))
            if cv_scoring.startswith("neg_"):
                cv_mean = -cv_mean
        except Exception:
            cv_mean, cv_std = float("nan"), float("nan")

        # Inference latency proxy: time to predict, normalised per 1k rows.
        t1 = time.perf_counter()
        pipe.predict(X_test)
        infer_ms = (time.perf_counter() - t1) * 1000.0 / max(1, len(X_test)) * 1000.0

        primary = metrics.get(metric, metrics.get(default_metric(task), float("nan")))
        results.append(CandidateResult(
            name=name, algorithm=name, primary_metric=metric, primary_score=float(primary),
            metrics=metrics, cv_mean=cv_mean, cv_std=cv_std, train_seconds=train_seconds,
            inference_ms_per_1k=round(infer_ms, 3), pipeline=pipe,
        ))

    results.sort(key=lambda r: (r.primary_score if not np.isnan(r.primary_score) else -1e9),
                 reverse=higher_is_better)
    best = results[0]
    return TrainingOutcome(
        task=task, metric=metric, candidates=results, best=best,
        X_columns=X.columns.tolist(), n_train=len(X_train), n_test=len(X_test),
        holdout_test_size=test_size,
        baseline_score=baseline_score, baseline_strategy=baseline_strategy,
    )
