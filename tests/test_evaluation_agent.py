"""Cross-validated evaluation, splitter selection, and the baseline gate."""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
sklearn = pytest.importorskip("sklearn")

from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402

from app.agents.evaluation_agent import (MetricResult, baseline_scores,  # noqa: E402
                                         choose_splitter, evaluate_model)


# --------------------------------------------------------------------------- #
# Splitter selection is a correctness decision
# --------------------------------------------------------------------------- #

def test_time_ordered_data_gets_a_time_series_split():
    name, reason = choose_splitter(task="regression", n_rows=500, time_ordered=True)
    assert name == "TimeSeriesSplit" and "future" in reason


def test_grouped_data_gets_group_kfold():
    name, reason = choose_splitter(task="classification", n_rows=500,
                                   groups=["a", "a", "b", "b"])
    assert name == "GroupKFold" and "entity" in reason


def test_time_ordering_outranks_grouping():
    """Both present: training on the future is the worse error."""
    name, _ = choose_splitter(task="classification", n_rows=500,
                              groups=["a", "b"], time_ordered=True)
    assert name == "TimeSeriesSplit"


def test_classification_stratifies_by_default():
    assert choose_splitter(task="classification", n_rows=500)[0] == "StratifiedKFold"


def test_plain_regression_gets_kfold():
    assert choose_splitter(task="regression", n_rows=500)[0] == "KFold"


def test_a_single_group_does_not_trigger_group_kfold():
    assert choose_splitter(task="regression", n_rows=100, groups=["a"] * 100)[0] == "KFold"


def test_every_splitter_choice_carries_a_reason():
    for kwargs in ({"task": "classification"}, {"task": "regression"},
                   {"task": "regression", "time_ordered": True},
                   {"task": "regression", "groups": ["a", "b"]}):
        _, reason = choose_splitter(n_rows=100, **kwargs)
        assert reason and len(reason) > 10


# --------------------------------------------------------------------------- #
# The baseline every model must clear
# --------------------------------------------------------------------------- #

def test_classification_baseline_is_the_majority_class():
    y = [0] * 95 + [1] * 5
    base = baseline_scores(y, "classification")
    assert base["accuracy"] == pytest.approx(0.95)


def test_macro_f1_baseline_exposes_what_accuracy_hides():
    """95% accuracy on 95/5 data is worthless; macro-F1 says so."""
    base = baseline_scores([0] * 95 + [1] * 5, "classification")
    assert base["accuracy"] == pytest.approx(0.95)
    assert base["f1_macro"] < 0.5


def test_regression_baseline_predicts_the_mean():
    y = [10.0, 20.0, 30.0]
    base = baseline_scores(y, "regression")
    assert base["rmse"] == pytest.approx(math.sqrt(200 / 3))
    assert base["r2"] == 0.0


def test_empty_target_yields_no_baseline():
    assert baseline_scores([], "classification") == {}


# --------------------------------------------------------------------------- #
# Metric aggregation and uncertainty
# --------------------------------------------------------------------------- #

def test_metric_reports_mean_std_and_interval():
    m = MetricResult("accuracy", [0.80, 0.82, 0.78, 0.81, 0.79])
    assert m.mean == pytest.approx(0.80)
    low, high = m.confidence_interval()
    assert low < m.mean < high


def test_a_single_fold_has_no_spread():
    m = MetricResult("accuracy", [0.9])
    assert m.std == 0.0 and m.confidence_interval() == (0.9, 0.9)


# --------------------------------------------------------------------------- #
# End-to-end evaluation
# --------------------------------------------------------------------------- #

def separable_classification(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.4 * X[:, 1] > 0).astype(int)
    return X, y


def test_a_real_model_beats_the_baseline():
    X, y = separable_classification()
    report = evaluate_model(LogisticRegression(max_iter=500), X, y, task="classification")
    assert report.beats_baseline is True
    assert report.metrics["accuracy"].mean > 0.8
    assert report.n_folds == 5


def test_a_useless_model_is_caught_by_the_baseline_gate():
    """The point of the gate: a model that learns nothing must not pass."""
    X, y = separable_classification()
    report = evaluate_model(DummyClassifier(strategy="most_frequent"), X, y,
                            task="classification")
    assert report.beats_baseline is False
    assert any("does not beat" in w for w in report.warnings)


def test_evaluation_reports_the_splitter_it_used_and_why():
    X, y = separable_classification()
    report = evaluate_model(LogisticRegression(max_iter=500), X, y, task="classification")
    assert report.splitter == "StratifiedKFold"
    assert report.splitter_reason


def test_regression_metrics_are_returned_unnegated():
    """sklearn negates error metrics; a reported RMSE must be positive."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] * 2 + rng.normal(0, 0.1, 200)
    report = evaluate_model(LinearRegression(), X, y, task="regression")
    assert report.metrics["rmse"].mean > 0
    assert report.metrics["mae"].mean > 0
    assert report.beats_baseline is True         # lower RMSE than predicting the mean


def test_tiny_datasets_warn_instead_of_pretending():
    X, y = separable_classification(n=10)
    report = evaluate_model(LogisticRegression(max_iter=500), X, y, task="classification")
    assert report.n_folds == 0
    assert any("too noisy" in w for w in report.warnings)
    assert report.baseline                      # baseline is still reported


def test_a_failing_estimator_is_reported_not_raised():
    class Broken:
        def get_params(self, deep=True): return {}
        def set_params(self, **kw): return self
        def fit(self, X, y): raise ValueError("cannot fit")
        def predict(self, X): raise ValueError("cannot predict")

    X, y = separable_classification(n=100)
    report = evaluate_model(Broken(), X, y, task="classification")
    assert any("cross-validation failed" in w for w in report.warnings)
    assert report.beats_baseline is None        # unknown, never optimistically True


def test_report_serialises_for_the_api():
    X, y = separable_classification()
    payload = evaluate_model(LogisticRegression(max_iter=500), X, y,
                             task="classification").as_dict()
    for key in ("task", "splitter", "splitter_reason", "metrics", "baseline",
                "beats_baseline", "primary_metric"):
        assert key in payload
    assert "ci_low" in payload["metrics"]["accuracy"]


def test_summary_states_the_baseline_verdict():
    X, y = separable_classification()
    text = evaluate_model(DummyClassifier(strategy="most_frequent"), X, y,
                          task="classification").summary()
    assert "DOES NOT BEAT" in text


def test_node_returns_declared_keys_only():
    from app.agents.evaluation_agent import evaluation_agent_node
    X, y = separable_classification()
    out = evaluation_agent_node({
        "trained_estimator": LogisticRegression(max_iter=500),
        "eval_X": X, "eval_y": y, "task_kind": "classification",
    })
    assert set(out) == {"evaluation_report", "evaluation_beats_baseline"}
    assert out["evaluation_beats_baseline"] is True
