"""The built-in benchmark suite.

Two families, each grounded in ground truth:

* **data_engineering** — ingestion parity across file/database/object-storage,
  workload routing, role inference, data-quality detection, reproducible
  transforms, and exact batch convergence.
* **data_science** — beating a naive baseline, leakage detection, deployment
  gating, economics sanity, correct firefly activation, and multi-Avaloka
  convergence.

Synthetic tasks always run (no network). Kaggle tasks reuse the repo's existing
download convention and skip gracefully when the dataset isn't present.
"""

from __future__ import annotations

from avaloka.benchmark import datasets as ds
from avaloka.benchmark.spec import BenchmarkTask, DatasetSpec, Expectations

ALL_SOURCES = ("file", "sqlite", "object_storage")


# --------------------------------------------------------------------------
def data_engineering_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            name="de_ingestion_parity_churn",
            family="data_engineering", kind="analyze",
            goal="Profile the churn dataset identically from any source.",
            dataset=DatasetSpec("churn", "synthetic", builder=ds.build_churn),
            source_schemes=ALL_SOURCES,
            expectations=Expectations(
                expected_lane="online",
                expected_roles={
                    "customer_id": "id", "tenure": "numeric",
                    "monthly_charges": "numeric", "contract": "categorical",
                },
            ),
        ),
        BenchmarkTask(
            name="de_data_quality_dirty",
            family="data_engineering", kind="analyze",
            goal="Detect the quality problems in a messy operational export.",
            dataset=DatasetSpec("dirty", "synthetic", builder=ds.build_dirty),
            source_schemes=("file", "sqlite"),
            expectations=Expectations(
                expected_roles={
                    "amount": "numeric", "region": "categorical",
                    "status_flag": "constant",
                },
                # The 5% duplicated rows make record_id non-unique, so it reads as
                # a high-cardinality categorical rather than a pure identifier.
                expect_quality_flags={
                    "constant_columns": ["status_flag"],
                    "high_missing_columns": ["income"],
                    "high_cardinality_columns": ["record_id"],
                },
                max_quality=90,
            ),
        ),
        BenchmarkTask(
            name="de_workload_routing_sampled",
            family="data_engineering", kind="analyze",
            goal="Route a >100k-row dataset to the sampled lane.",
            dataset=DatasetSpec("wide_sampled", "synthetic", builder=ds.build_wide_sampled),
            source_schemes=("file",),
            expectations=Expectations(expected_lane="sampled_online"),
        ),
        BenchmarkTask(
            name="de_batch_exact_convergence",
            family="data_engineering", kind="batch",
            goal="Prove the Ray map-reduce converges to the exact full-dataset profile.",
            dataset=DatasetSpec("batch_exact", "synthetic", builder=ds.build_batch_exact),
            expectations=Expectations(exact_columns=["value", "weight"]),
        ),
    ]


def data_science_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            name="ds_classification_churn",
            family="data_science", kind="train",
            goal="Train a churn classifier that beats a naive baseline and package it.",
            dataset=DatasetSpec("churn", "synthetic", builder=ds.build_churn, target="churned"),
            target="churned", metric="roc_auc", deployable=True,
            source_schemes=("file", "sqlite"),
            expectations=Expectations(
                min_score=0.60, min_gain_over_baseline=0.05,
                expected_fireflies=["data_scout", "model_scientist", "validator",
                                    "finops", "ml_engineer"],
            ),
        ),
        BenchmarkTask(
            name="ds_regression_housing",
            family="data_science", kind="train",
            goal="Fit a house-price regressor that clearly beats the mean predictor.",
            dataset=DatasetSpec("housing", "synthetic", builder=ds.build_housing, target="price"),
            target="price", metric="r2",
            source_schemes=("file",),
            expectations=Expectations(
                min_score=0.55, min_gain_over_baseline=0.4,
                expected_roles={"location": "categorical", "area_sqft": "numeric"},
                expected_fireflies=["model_scientist", "validator"],
                forbidden_fireflies=["ml_engineer"],   # not deployable
            ),
        ),
        BenchmarkTask(
            name="ds_leakage_detection",
            family="data_science", kind="train",
            goal="Catch target leakage and refuse to certify the model for production.",
            dataset=DatasetSpec("leaky", "synthetic", builder=ds.build_leaky, target="target"),
            target="target", metric="roc_auc",
            source_schemes=("file",),
            expectations=Expectations(
                must_detect_leakage=["leak_score", "outcome_label"],
                expected_verdict="fail", expected_max_level=1,
            ),
        ),
        # ── Cases the suite used to have no shape for ──────────────────────
        #
        # Each of these scored 1.00 before the defect it covers was fixed,
        # because no task carried the shape that triggers it. They are here so
        # the benchmark can fail for the right reason next time.
        BenchmarkTask(
            name="ds_temporal_split_on_time_ordered_data",
            family="data_science", kind="train",
            goal="Validate time-ordered data by training on the past and testing on the future.",
            dataset=DatasetSpec("timeseries", "synthetic",
                                builder=ds.build_timeseries, target="revenue"),
            target="revenue", metric="rmse",
            source_schemes=("file",),
            expectations=Expectations(
                expected_roles={"order_date": "datetime"},
                expected_task="regression",
                expected_split="time_based_holdout",
                require_baseline_reported=True,
                expect_beats_baseline=True,
            ),
        ),
        BenchmarkTask(
            name="ds_constant_column_does_not_break_training",
            family="data_science", kind="train",
            goal="Train on a dataset carrying a single-valued column.",
            dataset=DatasetSpec("constant_and_target", "synthetic",
                                builder=ds.build_constant_and_target, target="churned"),
            target="churned", metric="roc_auc",
            source_schemes=("file",),
            expectations=Expectations(
                expected_roles={"country": "constant", "row_id": "id"},
                expect_quality_flags={"constant_columns": ["country"]},
                require_baseline_reported=True,
            ),
        ),
        BenchmarkTask(
            name="ds_continuous_features_reach_the_model",
            family="data_science", kind="train",
            goal="Fit a model where every feature is a continuous measurement.",
            dataset=DatasetSpec("continuous_only", "synthetic",
                                builder=ds.build_continuous_only, target="throughput"),
            target="throughput", metric="rmse",
            source_schemes=("file",),
            expectations=Expectations(
                must_use_features=["latency_ms", "cpu_load"],
                require_baseline_reported=True,
                expect_beats_baseline=True,
            ),
        ),
        BenchmarkTask(
            name="ds_no_signal_is_reported_as_no_signal",
            family="data_science", kind="train",
            goal="Report honestly that a dataset with no relationship yields no model.",
            dataset=DatasetSpec("no_signal", "synthetic",
                                builder=ds.build_no_signal, target="outcome"),
            target="outcome", metric="roc_auc",
            source_schemes=("file",),
            expectations=Expectations(
                require_baseline_reported=True,
                expect_beats_baseline=False,
                expected_verdict="fail",
                expected_max_level=1,
            ),
        ),
        BenchmarkTask(
            name="ds_analysis_gating_and_swarm",
            family="data_science", kind="analyze",
            goal="Run an analysis mission and converge it across a multi-Avaloka fleet.",
            dataset=DatasetSpec("churn", "synthetic", builder=ds.build_churn),
            source_schemes=("file",),
            coordinate_workers=4,
            expectations=Expectations(
                expected_max_level=1,               # analysis is a research artifact
                expected_fireflies=["data_scout", "sampling_specialist",
                                    "analysis_planner", "validator", "reporter"],
                forbidden_fireflies=["model_scientist", "ml_engineer"],
                exact_columns=["tenure", "monthly_charges", "support_tickets"],
            ),
        ),
    ]


def kaggle_tasks() -> list[BenchmarkTask]:
    """Optional tasks over real Kaggle datasets (skipped if not downloaded)."""
    return [
        BenchmarkTask(
            name="kaggle_telco_churn_analyze",
            family="data_science", kind="analyze",
            goal="Profile the Telco customer-churn dataset and surface segments.",
            dataset=DatasetSpec("telco", "kaggle", kaggle_slug="blastchar/telco-customer-churn"),
            source_schemes=("file",),
            expectations=Expectations(
                expected_fireflies=["data_scout", "reporter"],
            ),
        ),
        BenchmarkTask(
            name="kaggle_iris_analyze",
            family="data_engineering", kind="analyze",
            goal="Ingest and profile the Iris dataset from a Kaggle source.",
            dataset=DatasetSpec("iris", "kaggle", kaggle_slug="uciml/iris"),
            source_schemes=("file", "sqlite"),
            expectations=Expectations(expected_lane="online"),
        ),
    ]


def all_tasks(*, include_kaggle: bool = False) -> list[BenchmarkTask]:
    tasks = data_engineering_tasks() + data_science_tasks()
    if include_kaggle:
        tasks += kaggle_tasks()
    return tasks
