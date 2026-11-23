"""
Data-science benchmark tasks.

These exercise the CLI's ability to build features and train models that meet
quality bars -- the "data scientist" half of the swarm. Model quality is scored
against a harness-computed baseline oracle (``ctx.extras['baseline']``) so the
CLI must produce a genuinely useful model, not merely a well-formed one.

The heavy (real DL) variants are marked ``heavy=True`` and are skipped unless
``AVALOKA_BENCH_HEAVY=1`` (see conftest).
"""

from __future__ import annotations

from ..tasks import (
    BenchmarkTask,
    Severity,
    Stage,
    check_beats_baseline,
    check_keys_present,
    check_metric_at_least,
    check_no_error,
    check_predicate,
)


def _features_nonempty(result, ctx):
    return result.get("n_features", 0) >= 3 and len(result.get("feature_names", [])) >= 3


def _no_target_leakage(result, ctx):
    # The target column must not appear among engineered feature names.
    if not ctx.target:
        return True
    return all(not name.startswith(f"{ctx.target}=") and name != ctx.target
               for name in result.get("feature_names", []))


DS_TASKS = [
    # --- feature engineering ----------------------------------------------- #
    BenchmarkTask(
        task_id="ds.build.titanic",
        stage=Stage.BUILD,
        dataset_key="titanic",
        brief="Engineer a numeric feature matrix; encode categoricals, scale numerics.",
        checks=[
            check_no_error(),
            check_keys_present(["feature_names", "n_features", "matrix_path"]),
            check_predicate("enough_features", _features_nonempty, Severity.MAJOR),
            check_predicate("no_target_leakage", _no_target_leakage, Severity.CRITICAL),
        ],
        tags=["feature-engineering"],
    ),
    # --- training: classification (light path, must beat baseline) --------- #
    BenchmarkTask(
        task_id="ds.train.titanic.clf",
        stage=Stage.TRAIN,
        dataset_key="titanic",
        brief="Train a survival classifier; report accuracy/macro_f1/logloss.",
        checks=[
            check_no_error(),
            check_keys_present(["metrics", "model_kind"]),
            check_metric_at_least("accuracy", 0.62, Severity.CRITICAL),
            # Allow the CLI to land within 2% of the oracle baseline.
            check_beats_baseline("accuracy", margin=0.02, severity=Severity.MAJOR),
        ],
        tags=["classification", "training"],
    ),
    BenchmarkTask(
        task_id="ds.train.churn.clf",
        stage=Stage.TRAIN,
        dataset_key="telco_churn",
        brief="Train a churn classifier on an imbalanced target.",
        checks=[
            check_no_error(),
            check_metric_at_least("accuracy", 0.70, Severity.CRITICAL),
            check_metric_at_least("macro_f1", 0.45, Severity.MAJOR),
        ],
        tags=["classification", "imbalanced", "training"],
    ),
    # --- training: regression ---------------------------------------------- #
    BenchmarkTask(
        task_id="ds.train.house.reg",
        stage=Stage.TRAIN,
        dataset_key="house_prices",
        brief="Train a price regressor; report rmse/mae/r2.",
        checks=[
            check_no_error(),
            check_keys_present(["metrics"]),
            check_metric_at_least("r2", 0.30, Severity.CRITICAL),
            check_beats_baseline("rmse", margin=0.0, severity=Severity.MAJOR),
        ],
        tags=["regression", "training"],
    ),
    # --- heavy: real deep-learning training -------------------------------- #
    BenchmarkTask(
        task_id="ds.train.titanic.deep",
        stage=Stage.TRAIN,
        dataset_key="titanic",
        brief="Train a deep tabular MLP (PyTorch) and beat the linear baseline.",
        heavy=True,
        checks=[
            check_no_error(),
            check_keys_present(["metrics", "model_kind"]),
            check_metric_at_least("accuracy", 0.62, Severity.CRITICAL),
            check_predicate("used_deep_model",
                            lambda r, c: "torch" in (r.get("model_kind") or ""),
                            Severity.MAJOR),
        ],
        tags=["classification", "deep-learning", "heavy"],
    ),
    BenchmarkTask(
        task_id="ds.train.house.deep",
        stage=Stage.TRAIN,
        dataset_key="house_prices",
        brief="Train a deep tabular MLP regressor (PyTorch).",
        heavy=True,
        checks=[
            check_no_error(),
            check_metric_at_least("r2", 0.30, Severity.CRITICAL),
            check_predicate("used_deep_model",
                            lambda r, c: "torch" in (r.get("model_kind") or ""),
                            Severity.MAJOR),
        ],
        tags=["regression", "deep-learning", "heavy"],
    ),
    # --- serve: inference as a service over k8s ---------------------------- #
    BenchmarkTask(
        task_id="ds.serve.titanic.k8s",
        stage=Stage.SERVE,
        dataset_key="titanic",
        brief="Provision the trained model as a k8s inference service.",
        checks=[
            check_no_error(),
            check_keys_present(["endpoint", "predict_url", "status"]),
            check_predicate("service_ready", lambda r, c: r.get("ready") is True,
                            Severity.CRITICAL),
            check_predicate("k8s_endpoint",
                            lambda r, c: "svc.cluster.local" in (r.get("endpoint") or ""),
                            Severity.MINOR),
        ],
        tags=["serving", "k8s", "inference"],
    ),
]
