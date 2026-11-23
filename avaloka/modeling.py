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
                              RandomForestClassifier, RandomForestRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
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
    #: Classifiers only: the labelled confusion matrix and per-class precision
    #: and recall on the holdout. ``metrics`` stays flat so every reader that
    #: formats it as numbers keeps working; the matrix is a table, not a number.
    classification: dict[str, Any] | None = None

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
        # Macro, so a rare class that is never predicted drags the number
        # down instead of hiding behind the majority. Accuracy alone cannot
        # tell a classifier that found every class from one that found none
        # but the largest.
        metrics["precision_macro"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0))
        if task == "binary_classification" and y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except ValueError:
                pass
    return metrics


def classification_detail(y_true, y_pred, label_names: list[str] | None = None,
                          classes: list[Any] | None = None) -> dict[str, Any]:
    """The confusion matrix and per-class precision/recall, labelled.

    Rows are the actual class, columns the predicted one, and every row sums
    to that class's holdout count -- which is the check a reader (and the
    test) can do by eye. The first row and column carry the labels so the
    JSON is readable without this docstring. Every class present in either
    the truth or the predictions gets a row; a class the model never
    predicted still shows, as a row of misses, because dropping it would make
    the matrix look better than the model.

    ``classes`` are the label values ``y`` actually holds, in the order of
    ``label_names``. A text target is encoded, so its values are the codes
    0..k-1 and positions and values coincide; a numeric target is not, so a
    taxi ``payment_type`` of {1, 2, 3, 4} holds the values 1..4, and reading
    ``label_names`` by value instead of by position shifted every header one
    to the left ("2 | 3 | 4 | 4"). Without ``classes`` the values are taken
    to be codes, which is what a direct caller with encoded labels passes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    codes = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))
    if label_names is None:
        name_of: dict[Any, str] = {}
    elif classes is None:
        name_of = {i: str(n) for i, n in enumerate(label_names)}
    else:
        name_of = {c: str(n) for c, n in zip(classes, label_names)}
    names = [name_of.get(c, str(c)) for c in codes]
    matrix = confusion_matrix(y_true, y_pred, labels=codes)
    precision = precision_score(y_true, y_pred, labels=codes, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=codes, average=None, zero_division=0)
    table: list[list[Any]] = [["actual \\ predicted", *names]]
    for name, row in zip(names, matrix):
        table.append([name, *[int(v) for v in row]])
    return {
        "labels": names,
        "confusion_matrix": table,
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "support": int(matrix[i].sum())}
            for i, (name, p, r) in enumerate(zip(names, precision, recall))
        },
        "precision_macro": float(np.mean(precision)) if len(codes) else float("nan"),
        "recall_macro": float(np.mean(recall)) if len(codes) else float("nan"),
    }


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


def _exceeds_fold_noise(cv_mean: float, cv_std: float, reference: float,
                        lower_is_better: bool) -> bool | None:
    """Whether a cross-validated score beats ``reference`` by more than fold noise.

    The bar is the confidence interval, not the mean, matching
    app/agents/evaluation_agent.py: the interval ``cv_mean +/- cv_std`` must
    lie wholly on the winning side of the reference. ``None`` when either
    side is unknown; never optimistically True.
    """
    if math.isnan(reference) or math.isnan(cv_mean):
        return None
    spread = 0.0 if math.isnan(cv_std) else cv_std
    if lower_is_better:
        return bool(cv_mean + spread < reference)
    return bool(cv_mean - spread > reference)


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
    #: The ensemble-versus-best-single verdict (see :func:`ensemble_verdict`),
    #: or None when fewer than two candidates were trained.
    ensemble_vs_single: dict[str, Any] | None = None
    #: Class names in code order for classifiers (None for regression), so a
    #: reader of the confusion matrix sees "churned", not "1".
    label_names: list[str] | None = None
    #: The label values ``y`` holds, aligned with ``label_names``: the codes
    #: 0..k-1 for an encoded text target, the values themselves (e.g. 1..4)
    #: for a numeric one.
    classes: list[Any] | None = None
    #: What every candidate was ranked on: ``"cv"`` (the mean over the same
    #: folds) or ``"holdout"`` when cross-validation failed for any of them.
    selection_basis: str = "cv"
    #: The holdout rows, kept so a later firefly (the conformal calibrator)
    #: can calibrate on data no candidate was fitted on.
    X_test: Any = field(repr=False, default=None)
    y_test: Any = field(repr=False, default=None)

    @property
    def beats_baseline(self) -> bool | None:
        """Whether the winner beats the trivial model by more than fold noise.

        A classifier that only ever predicts the majority class scores exactly
        the baseline, and floating-point noise then records it as a win by
        +0.0000 -- which is the failure a baseline exists to catch, so catching
        it loosely is no use. Hence the interval, not the mean.

        ``None`` when it cannot be determined; never optimistically True.
        """
        best = self.best
        return _exceeds_fold_noise(best.cv_mean, best.cv_std, self.baseline_score,
                                   lower_is_better=self.metric in {"rmse", "mae"})


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
    ensemble: bool = True,
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
        # Encode any label that is not already a number. Testing `dtype ==
        # object` missed text held as pandas' string or category dtype, and
        # the words then reached np.bincount unencoded.
        if not pd.api.types.is_numeric_dtype(y):
            categorical = y.astype("category")
            label_names = [str(c) for c in categorical.cat.categories]
            y = categorical.cat.codes
            classes = list(range(len(label_names)))
        else:
            classes = sorted(y.dropna().unique().tolist())
            label_names = [str(c) for c in classes]
    else:
        label_names = classes = None

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
    folds = _cv_folds(task, y_train)

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
        results.append(_evaluate_candidate(
            name, name, pipe, task, metric, cv_scoring, folds, train_seconds,
            X_train, y_train, X_test, y_test, y_pred, y_proba, label_names, classes,
        ))

    # One selection rule for every candidate: the CV mean of the primary
    # metric over the same folds. The best single used to be picked on the
    # holdout and then compared with the ensemble on CV -- two different
    # numbers -- so on nyc_taxi the ensemble (CV 0.5343) "beat" logistic
    # regression's CV 0.5133 and was selected, while gradient boosting (CV
    # 0.5660) and random forest (CV 0.5548) both beat it on that same CV and
    # logistic regression (0.5278) beat its 0.4887 on the holdout.
    basis = "cv" if all(not math.isnan(r.cv_mean) for r in results) else "holdout"
    lower_is_better = metric in {"rmse", "mae"}
    results.sort(key=lambda r: selection_key(r, basis, lower_is_better), reverse=True)
    best_single = results[0]

    # One more candidate: the panel voting together. It is scored on exactly
    # the same holdout and folds as the singles, and it is only *selected*
    # when its CV gain over the best single (best on CV, by the rule above)
    # clears fold noise -- an ensemble that wins by less than the folds
    # disagree among themselves has not earned its serving cost or its
    # opacity.
    verdict = None
    best = best_single
    voted = (_ensemble_candidate(task, metric, cv_scoring, folds, results,
                                 X_train, y_train, X_test, y_test, label_names, classes)
             if ensemble else None)
    if voted is not None:
        results.append(voted)
        verdict = ensemble_verdict(voted, best_single, metric)
        if verdict["selected"] == "ensemble":
            best = voted

    return TrainingOutcome(
        task=task, metric=metric, candidates=results, best=best,
        X_columns=X.columns.tolist(), n_train=len(X_train), n_test=len(X_test),
        holdout_test_size=test_size,
        baseline_score=baseline_score, baseline_strategy=baseline_strategy,
        ensemble_vs_single=verdict, label_names=label_names, classes=classes,
        selection_basis=basis, X_test=X_test, y_test=y_test,
    )


def selection_key(c: CandidateResult, basis: str, lower_is_better: bool) -> tuple:
    """Sort key, larger is better: the CV mean on ``"cv"``, else the holdout.

    A NaN sorts last. Ties on CV fall to the holdout so the order is total;
    FinOps ranks with this same key so its "Maximum quality" is the model the
    selection would pick on quality alone.
    """
    sign = -1.0 if lower_is_better else 1.0

    def _v(x: float) -> float:
        return -math.inf if math.isnan(x) else sign * x

    primary = c.cv_mean if basis == "cv" else c.primary_score
    return (_v(primary), _v(c.primary_score))


def _cv_folds(task: str, y_train):
    """The one fold split every candidate, the ensemble included, is scored on.

    Unshuffled, so it is what ``cross_val_score(cv=k)`` would build -- but
    built once, so "the same folds" is a fact of the code and not of a
    default. k is capped by the rarest class, counted by value: the
    ``np.bincount`` this replaces counted the absent value 0 of a {1..4}
    target as a class of size zero and fell back to two folds.
    """
    if task.endswith("classification"):
        rarest = int(pd.Series(np.asarray(y_train)).value_counts().min())
        return StratifiedKFold(n_splits=max(2, min(5, rarest)))
    return KFold(n_splits=5)


def _evaluate_candidate(name, algorithm, pipe, task, metric, cv_scoring, folds, train_seconds,
                        X_train, y_train, X_test, y_test, y_pred, y_proba,
                        label_names, classes) -> CandidateResult:
    """Holdout metrics, k-fold CV on the training split, and a latency proxy."""
    metrics = _score_all(task, y_test, y_pred, y_proba)
    detail = (classification_detail(y_test, y_pred, label_names, classes)
              if task.endswith("classification") else None)

    # Cross-validation for stability (the shared folds on the training split).
    try:
        cv = cross_val_score(pipe, X_train, y_train, cv=folds, scoring=cv_scoring)
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
    return CandidateResult(
        name=name, algorithm=algorithm, primary_metric=metric, primary_score=float(primary),
        metrics=metrics, cv_mean=cv_mean, cv_std=cv_std, train_seconds=train_seconds,
        inference_ms_per_1k=round(infer_ms, 3), pipeline=pipe, classification=detail,
    )


ENSEMBLE_NAME = "ensemble"


def _ensemble_candidate(task, metric, cv_scoring, folds, singles, X_train, y_train, X_test,
                        y_test, label_names, classes) -> CandidateResult | None:
    """Soft-voting (classification) or averaging (regression) over the panel.

    Built from the panel's pipelines so each member keeps its own fitted
    preprocessing; sklearn's voting estimators clone and refit them, which is
    what makes the ensemble's cross-validation honest -- every fold refits
    every member. Needs at least two members; a vote of one is the single.
    """
    members = [(c.name, c.pipeline) for c in singles if c.pipeline is not None]
    if len(members) < 2:
        return None
    if task == "regression":
        est = VotingRegressor(members)
    else:
        soft = all(hasattr(p, "predict_proba") for _, p in members)
        est = VotingClassifier(members, voting="soft" if soft else "hard")
    t0 = time.perf_counter()
    try:
        est.fit(X_train, y_train)
    except Exception:
        return None
    train_seconds = time.perf_counter() - t0
    y_pred = est.predict(X_test)
    y_proba = None
    if task == "binary_classification" and hasattr(est, "predict_proba"):
        try:
            y_proba = est.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
    return _evaluate_candidate(
        ENSEMBLE_NAME, ENSEMBLE_NAME, est, task, metric, cv_scoring, folds, train_seconds,
        X_train, y_train, X_test, y_test, y_pred, y_proba, label_names, classes,
    )


def ensemble_verdict(ensemble: CandidateResult, single: CandidateResult, metric: str) -> dict[str, Any]:
    """Ensemble versus best single, decided on the CV interval, not the mean.

    The same bar as :meth:`TrainingOutcome.beats_baseline`: the winner's
    ``cv_mean +/- cv_std`` must lie wholly beyond the loser's mean. A margin
    inside that band is fold noise, and then the single is selected -- it is
    cheaper to serve and easier to explain, and the ensemble has not shown it
    is better. ``exceeds_fold_noise`` is None when CV failed on either side.
    """
    lower_is_better = metric in {"rmse", "mae"}
    margin = ensemble.cv_mean - single.cv_mean
    if lower_is_better:
        margin = -margin
    if math.isnan(margin):
        cv_winner = "undetermined"
    elif margin > 0:
        cv_winner = "ensemble"
    elif margin < 0:
        cv_winner = "single"
    else:
        cv_winner = "tie"
    winner, loser = (ensemble, single) if cv_winner == "ensemble" else (single, ensemble)
    exceeds = (None if cv_winner in {"undetermined", "tie"} else
               _exceeds_fold_noise(winner.cv_mean, winner.cv_std, loser.cv_mean, lower_is_better))
    selected = "ensemble" if cv_winner == "ensemble" and exceeds else "single"

    def _interval(c: CandidateResult) -> list[float]:
        spread = 0.0 if math.isnan(c.cv_std) else c.cv_std
        return [c.cv_mean - spread, c.cv_mean + spread]

    if cv_winner == "undetermined":
        reason = "Cross-validation failed for one side; the single is selected by default."
    elif selected == "ensemble":
        reason = (f"The ensemble's CV {metric} {ensemble.cv_mean:.4f} beats the best single "
                  f"'{single.name}' ({single.cv_mean:.4f}) by {abs(margin):.4f}, more than "
                  f"its fold spread of {ensemble.cv_std:.4f}.")
    elif cv_winner == "ensemble":
        reason = (f"The ensemble's CV {metric} {ensemble.cv_mean:.4f} is ahead of "
                  f"'{single.name}' ({single.cv_mean:.4f}) by {abs(margin):.4f}, within its "
                  f"fold spread of {ensemble.cv_std:.4f}; the gain is fold noise, so the "
                  f"simpler single model is selected.")
    else:
        reason = (f"The best single '{single.name}' (CV {metric} {single.cv_mean:.4f}) is "
                  f"{'ahead of' if cv_winner == 'single' else 'level with'} the ensemble "
                  f"({ensemble.cv_mean:.4f}); the ensemble is not selected.")
    return {
        "metric": metric,
        "ensemble": ensemble.name,
        "best_single": single.name,
        "ensemble_cv": {"mean": ensemble.cv_mean, "std": ensemble.cv_std,
                        "interval": _interval(ensemble)},
        "single_cv": {"mean": single.cv_mean, "std": single.cv_std,
                      "interval": _interval(single)},
        "cv_margin": margin,
        "cv_winner": cv_winner,
        "exceeds_fold_noise": exceeds,
        "selected": selected,
        "reason": reason,
    }
