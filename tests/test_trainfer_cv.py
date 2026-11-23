"""Cross-validation (1.7/1.8): the fold rule, the TI-36 oracle, sums and labels,
regression without a matrix, skips, seeds, time order, persistence and the API.

All error on develop-1.7 @ 9e3caed0 (app.agents.mta_v2.cross_validation absent),
except where noted: the API test fails by assertion on the base router... which is
also absent, so E. The oracle itself is TI-36's hand-computed matrix.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.core import ml_store


@pytest.fixture
def store():
    s = ml_store.InMemoryMLStore(); ml_store.set_ml_store(s)
    yield s
    ml_store.set_ml_store(None)


def _cv():
    from app.agents.mta_v2 import cross_validation as cv
    return cv


# T-29 / TI-37: the fold-count rule at its boundaries ---------------------------------

@pytest.mark.parametrize("n,expected", [(3, 2), (4, 2), (9, 4), (10, 5), (11, 5), (30, 5), (1000, 5)])
def test_k_eff_boundaries(n, expected):
    assert _cv().k_eff(n, 5) == expected


def test_k_eff_honours_a_requested_k():
    cv = _cv()
    assert cv.k_eff(100, 3) == 3 and cv.k_eff(100, 10) == 10 and cv.k_eff(7, 10) == 3


# TI-36: the oracle ---------------------------------------------------------------------
# 12 rows, 3 classes (A, B, C), 4 per class. The predictor gets 3 of each class right
# and confuses the fourth cyclically (A->B, B->C, C->A). With 4 stratified folds each
# fold holds one row of every class, so every fold sums to 3 and the sum of the folds is
# exactly the hand-computed matrix.

ORACLE = [[3, 1, 0], [0, 3, 1], [1, 0, 3]]
LABELS = ["A", "B", "C"]


def _oracle_data():
    y = np.array([0] * 4 + [1] * 4 + [2] * 4)
    pred = y.copy()
    pred[3], pred[7], pred[11] = 1, 2, 0       # the fourth row of each class is confused
    return y, pred


def test_ti36_the_matrix_equals_the_hand_computed_one():
    cv = _cv()
    y, pred = _oracle_data()
    report = cv.run_folds(y, task="classification", labels=LABELS, fit_predict=lambda tr, va: pred[va], k=4, seed=0, min_rows=12)
    assert report["status"] == "succeeded" and report["k"] == 4 and report["splitter"] == "StratifiedKFold"
    agg = report["aggregate"]
    assert agg["matrix"] == ORACLE
    assert agg["n"] == 12 and agg["labels"] == LABELS
    assert all(f["n"] == 3 for f in report["folds"]), "each fold holds one row per class"
    # pooled on the summed matrix: accuracy 9/12, macro F1 0.75, every class 3/4 precision and recall
    assert agg["accuracy"] == pytest.approx(0.75) and agg["f1_macro"] == pytest.approx(0.75)
    assert agg["precision"] == [0.75, 0.75, 0.75] and agg["recall"] == [0.75, 0.75, 0.75]
    # fold means describe the spread: three folds with one miss (2/3) and one perfect fold average 0.75
    assert agg["accuracy_mean"] == pytest.approx(0.75)
    assert agg["normalized"] == [[0.75, 0.25, 0.0], [0.0, 0.75, 0.25], [0.25, 0.0, 0.75]]
    summed = cv.confusion(y, pred, LABELS)
    assert summed["matrix"] == ORACLE and summed["f1_macro"] == pytest.approx(0.75)


# TI-33 / TI-34: sums and labels -------------------------------------------------------------

def test_fold_matrices_sum_to_the_aggregate_rows_sum_to_class_counts_and_labels_hold_order():
    cv = _cv()
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, size=90)
    pred = np.where(rng.random(90) < 0.7, y, rng.integers(0, 3, size=90))
    report = cv.run_folds(y, task="classification", labels=["zero", "one", "two"], fit_predict=lambda tr, va: pred[va], k=5, seed=1)
    folds = report["folds"]; agg = report["aggregate"]
    assert np.array_equal(np.sum([np.array(f["matrix"]) for f in folds], axis=0), np.array(agg["matrix"]))
    assert agg["n"] == 90 == sum(f["n"] for f in folds)
    assert [sum(r) for r in agg["matrix"]] == [int((y == c).sum()) for c in range(3)], "row totals are the true-class counts"
    for row in agg["normalized"]:
        assert sum(row) == pytest.approx(1.0, abs=0.01)
    assert agg["labels"] == ["zero", "one", "two"] and all(f["labels"] == ["zero", "one", "two"] for f in folds)
    # every training row validates exactly once
    seen = np.concatenate([f["val_index"] for f in folds])
    assert sorted(seen.tolist()) == list(range(90))


# TI-35: regression has no matrix ----------------------------------------------------------------

def test_regression_reports_fold_metrics_and_no_confusion_matrix():
    cv = _cv()
    rng = np.random.default_rng(5)
    y = rng.normal(100, 10, size=60)
    report = cv.run_folds(y, task="regression", labels=None, fit_predict=lambda tr, va: y[va] + 1.0, k=5, seed=1)
    assert report["splitter"] == "KFold"
    assert report["aggregate"]["confusion_matrix"] is None
    assert all({"rmse", "mae", "r2"} <= set(f) for f in report["folds"])
    assert report["aggregate"]["mae_mean"] == pytest.approx(1.0) and report["aggregate"]["rmse_mean"] == pytest.approx(1.0)
    assert set(cv.run_metrics({**report, "task": "regression"})) == {"cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_mae_std", "cv_r2_mean", "cv_r2_std"}


# TI-37: too few rows skips with the reason; a thin class skips naming it ----------------------------

def test_too_few_rows_skips_with_the_floor_in_the_reason():
    cv = _cv()
    y = np.array([0, 1] * 10)
    report = cv.run_folds(y, task="classification", labels=["a", "b"], fit_predict=lambda tr, va: y[va], k=5)
    assert report["status"] == "skipped" and report["folds"] == [] and report["aggregate"] is None
    assert "20 rows" in report["reason"] and "30" in report["reason"]


def test_a_class_thinner_than_k_is_refused_not_silently_emptied():
    cv = _cv()
    y = np.array([0] * 38 + [1] * 2)
    report = cv.run_folds(y, task="classification", labels=["a", "b"], fit_predict=lambda tr, va: y[va], k=5)
    assert report["status"] == "skipped" and "2 rows" in report["reason"]


# TI-38: seeds ------------------------------------------------------------------------------------

def test_seeded_folds_are_reproducible_and_a_new_seed_moves_rows():
    cv = _cv()
    y = np.array([0, 1] * 25)
    a = cv.run_folds(y, task="classification", labels=["a", "b"], fit_predict=lambda tr, va: y[va], k=5, seed=7)
    b = cv.run_folds(y, task="classification", labels=["a", "b"], fit_predict=lambda tr, va: y[va], k=5, seed=7)
    c = cv.run_folds(y, task="classification", labels=["a", "b"], fit_predict=lambda tr, va: y[va], k=5, seed=8)
    assert [f["val_index"] for f in a["folds"]] == [f["val_index"] for f in b["folds"]]
    assert [f["val_index"] for f in a["folds"]] != [f["val_index"] for f in c["folds"]]


# TI-56: a time column selects TimeSeriesSplit --------------------------------------------------------

def test_time_ordered_uses_increasing_non_overlapping_windows():
    cv = _cv()
    y = np.arange(60) % 2
    report = cv.run_folds(y, task="classification", labels=["a", "b"], fit_predict=lambda tr, va: y[va], k=4, time_ordered=True)
    assert report["splitter"] == "TimeSeriesSplit"
    windows = [f["val_index"] for f in report["folds"]]
    for earlier, later in zip(windows, windows[1:]):
        assert max(earlier) < min(later), "validation windows increase and never overlap"
    assert all(min(w) > 0 for w in windows), "every fold validates later than its fit rows"


# The real recipe on a tiny frame (MLP refit per fold) -------------------------------------------------

def test_cross_validate_plan_refits_the_recipe_per_fold_on_the_training_split_only():
    cv = _cv()
    rng = np.random.default_rng(0)
    n = 120
    X = rng.standard_normal((n, 3))
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["target"] = np.where(X[:, 0] + X[:, 1] > 0, "yes", "no")
    plan = {"model_type": "classification", "model_name": "m", "hyperparameter_config": {"epochs": 2, "batch_size": 16, "hidden_layer_sizes": [8], "random_seed": 1},
            "data_config": {"dataset_uri": "unused", "feature_columns": ["f1", "f2", "f3"], "target_column": "target", "class_names": ["no", "yes"]},
            "cross_validation": {"k": 3}}
    report = cv.cross_validate_plan(plan, df=df)
    assert report["status"] == "succeeded" and report["k"] == 3 and report["task"] == "classification"
    assert report["aggregate"]["labels"] == ["no", "yes"]
    assert report["n_rows"] < n, "the holdout split is removed before folding"
    assert report["aggregate"]["n"] == report["n_rows"]
    artifacts = cv.fold_artifacts(report)
    assert set(artifacts) == {"cv/fold_0/confusion_matrix.json", "cv/fold_1/confusion_matrix.json", "cv/fold_2/confusion_matrix.json", "cv/confusion_matrix.json"}


# Persistence + the worker hook + the API -----------------------------------------------------------------

def test_run_cross_validation_persists_the_row_and_logs_mlflow(store):
    from app.infra import training_tasks as t
    fake_cv = MagicMock()
    fake_cv.DEFAULT_K = 5
    fake_cv.cross_validate_plan.return_value = {"status": "succeeded", "k": 4, "splitter": "StratifiedKFold", "reason": None, "task": "classification",
                                                "n_rows": 12, "duration_s": 0.1, "aggregate": {"labels": LABELS, "matrix": ORACLE, "n": 12, "accuracy_mean": 0.75},
                                                "folds": [{"fold": i, "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "n": 3, "val_index": [0, 1, 2]} for i in range(4)]}
    fake_cv.fold_artifacts.return_value = {"cv/confusion_matrix.json": {}}
    fake_cv.run_metrics.return_value = {"cv_accuracy_mean": 0.75}
    logged = {}
    state = {"training_plan": {"execution_backend": "local", "data_config": {"feature_columns": ["a"], "target_column": "t"}, "cross_validation": {"k": 4}},
             "training_result": {"mlflow_run_id": "run-1"}}
    out = t.run_cross_validation("u1", "run-1", state, store=store, cv_module=fake_cv, mlflow_log=lambda rid, arts, mets: logged.update(rid=rid, arts=arts, mets=mets))
    assert out["status"] == "succeeded" and out["aggregate"]["matrix"] == ORACLE
    row = store.get_cross_validation("run-1")
    assert row.k == 4 and row.folds and "val_index" not in row.folds[0], "row indices stay out of Postgres"
    assert logged["rid"] == "run-1" and logged["mets"] == {"cv_accuracy_mean": 0.75}
    assert fake_cv.cross_validate_plan.call_args.kwargs["k"] == 4


def test_a_ray_trained_run_is_recorded_as_skipped_with_the_reason(store):
    from app.infra import training_tasks as t
    state = {"training_plan": {"execution_backend": "ray", "data_config": {"feature_columns": ["a"], "target_column": "t"}},
             "training_result": {"mlflow_run_id": "run-2", "ray_job_name": "avaloka-train-x"}}
    out = t.run_cross_validation("u1", "run-2", state, store=store, cv_module=MagicMock(DEFAULT_K=5))
    assert out["status"] == "skipped" and "Ray" in out["reason"]


def test_cross_validation_route_is_owner_scoped(store):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.models_routes import build_router
    async def premium(u, f): return None
    app = FastAPI(); app.include_router(build_router(lambda r: r.headers.get("X-User"), premium))
    c = TestClient(app)
    assert c.get("/api/models/run-1/cross-validation", headers={"X-User": "u1"}).status_code == 404
    store.upsert_cross_validation(ml_store.MLCrossValidation(mlflow_run_id="run-1", user_id="u1", status="succeeded", k=4, splitter="StratifiedKFold",
                                                              aggregate={"labels": LABELS, "matrix": ORACLE}, folds=[{"fold": 0, "matrix": ORACLE}]))
    r = c.get("/api/models/run-1/cross-validation", headers={"X-User": "u1"})
    assert r.status_code == 200 and r.json()["aggregate"]["matrix"] == ORACLE and len(r.json()["folds"]) == 1
    assert c.get("/api/models/run-1/cross-validation", headers={"X-User": "u2"}).status_code == 404, "another user learns nothing"
