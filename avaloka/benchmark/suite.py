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
from avaloka.benchmark.datasets_formats import FORMAT_DATASETS
from avaloka.benchmark.datasets_spreadsheet import SPREADSHEET_DATASETS
from avaloka.benchmark.spec import (BenchmarkTask, DatasetSpec, Expectations,
                                    GroundTruth)

ALL_SOURCES = ("file", "sqlite", "object_storage")


# --------------------------------------------------------------------------
# Comprehension expectations are derived from the ground truth the builder
# declared, rather than retyped beside each task. Retyping is how a benchmark
# ends up asserting what the reader does instead of what the file contains:
# the two copies drift, the task's copy is the one that gets "fixed" when it
# goes red, and the check quietly becomes a mirror.
# --------------------------------------------------------------------------
def _comprehension_expectations(truth: GroundTruth, **extra) -> Expectations:
    dtypes = {c: "numeric" for c in truth.numeric_columns}
    dtypes.update({c: "datetime" for c in truth.date_columns})
    dtypes.update({c: "text" for c in truth.text_columns})
    return Expectations(
        no_placeholder_column_names=True,
        header_row_identified=truth.header_row,
        sheet_identified=truth.sheet_name,
        expected_columns_resolved=list(truth.columns),
        expected_dtypes=dtypes,
        expected_row_count=truth.n_rows,
        narration_names_only_real_columns=True,
        narration_columns_from_ground_truth=True,
        expected_answer=dict(truth.planted_answer),
        **extra,
    )


def format_tasks() -> list[BenchmarkTask]:
    """One analysis task per declared upload format other than CSV.

    ``source_schemes`` is file-only on purpose. Re-publishing the dataset
    through the sqlite or object-storage connector would hand the mission a
    frame that something else had already parsed, so the task would stop being
    about the format and start being about pandas. The file *is* the test.
    """
    tasks: list[BenchmarkTask] = []
    for name, (builder, ext, truth) in sorted(FORMAT_DATASETS.items()):
        tasks.append(BenchmarkTask(
            name=f"fmt_reads_{ext}",
            family="format_comprehension", kind="analyze",
            goal=f"Read a {ext.upper()} file and profile the table that is in it.",
            dataset=DatasetSpec(name, "synthetic", builder=builder,
                                suffix=f".{ext}", declared_format=ext,
                                ground_truth=truth,
                                notes=truth.shape),
            source_schemes=("file",),
            expectations=_comprehension_expectations(truth),
        ))
    return tasks


def nested_document_tasks() -> list[BenchmarkTask]:
    """The nesting rule, graded clause by clause.

    JSON and XML are trees; a dataframe is a rectangle. The rule is stated in
    ``avaloka/io/loader.py`` and these tasks check it: scalar nesting must
    flatten to dotted names, and a shape with no single correct reading must be
    refused by name rather than guessed at. Without these the reader could
    quietly pick any interpretation and the suite would score it 1.00.
    """
    from avaloka.benchmark.datasets_formats import NESTED_DATASETS
    tasks: list[BenchmarkTask] = []
    for name, (builder, ext, truth, must_name) in sorted(NESTED_DATASETS.items()):
        spec = DatasetSpec(name, "synthetic", builder=builder, suffix=f".{ext}",
                           declared_format=ext, ground_truth=truth,
                           notes=truth.shape)
        if must_name:
            expectations = Expectations(expected_refusal=list(must_name))
        else:
            expectations = _comprehension_expectations(truth)
        tasks.append(BenchmarkTask(
            name=name, family="format_comprehension", kind="analyze",
            goal=("Read a nested document under the stated rule, or refuse it "
                  "by name."),
            dataset=spec, source_schemes=("file",), expectations=expectations))
    return tasks


def spreadsheet_tasks() -> list[BenchmarkTask]:
    """The workbook shapes that actually break readers.

    Every one of these is a shape a person produced in a real spreadsheet, and
    the suite had no dataset carrying any of them -- so the reader's behaviour
    on all of them was unmeasured while the scorecard read 1.00.
    """
    tasks: list[BenchmarkTask] = []
    for name, (builder, truth) in sorted(SPREADSHEET_DATASETS.items()):
        tasks.append(BenchmarkTask(
            name=f"xlsx_{name}",
            family="format_comprehension", kind="analyze",
            goal=("Find the table inside the workbook and profile it, "
                  "naming the columns the file actually has."),
            dataset=DatasetSpec(name, "synthetic", builder=builder,
                                suffix=".xlsx", declared_format="xlsx",
                                ground_truth=truth, notes=truth.shape),
            source_schemes=("file",),
            expectations=_comprehension_expectations(truth),
        ))
    return tasks


# --------------------------------------------------------------------------
def data_engineering_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            name="de_ingestion_parity_churn",
            family="data_engineering", kind="analyze",
            goal="Profile the churn dataset identically from any source.",
            dataset=DatasetSpec("churn", "synthetic", builder=ds.build_churn, declared_format="csv"),
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
            dataset=DatasetSpec("dirty", "synthetic", builder=ds.build_dirty, declared_format="csv"),
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
            dataset=DatasetSpec("wide_sampled", "synthetic", builder=ds.build_wide_sampled, declared_format="csv"),
            source_schemes=("file",),
            expectations=Expectations(expected_lane="sampled_online"),
        ),
        BenchmarkTask(
            name="de_batch_exact_convergence",
            family="data_engineering", kind="batch",
            goal="Prove the Ray map-reduce converges to the exact full-dataset profile.",
            dataset=DatasetSpec("batch_exact", "synthetic", builder=ds.build_batch_exact, declared_format="csv"),
            expectations=Expectations(exact_columns=["value", "weight"]),
        ),
        # One defect, read off two artifacts, plus the guard against its
        # obvious wrong repair. A payment method called "None" is a category.
        # The grouped total is 5500.0 whether or not the categories survive,
        # so nothing here compares a total.
        BenchmarkTask(
            name="de_null_words_stay_categories",
            family="data_engineering", kind="analyze",
            goal="Total revenue by payment_method",
            dataset=DatasetSpec("null_labelled_payments", "synthetic",
                                builder=ds.build_null_labelled_payments),
            source_schemes=("file",),
            expectations=Expectations(
                # the answer: every planted group, its exact sum, nothing else
                expected_groups={
                    label: count * ds.NULL_LABELLED_REVENUE_PER_ROW
                    for label, count in ds.NULL_LABELLED_PAYMENT_COUNTS.items()
                },
                forbid_unexpected_groups=True,
                # the profile: the same loss, on the artifact a fix to the
                # goal analyzer alone would leave wrong
                expected_category_counts={
                    "payment_method": dict(ds.NULL_LABELLED_PAYMENT_COUNTS),
                },
                expected_missing={"payment_method": 0},
                # the guard: keep_default_na=False would keep the labels and
                # turn `amount` into text. Holds today; not evidence of a fix.
                expected_roles={"amount": "numeric"},
                guard_roles=("amount",),
            ),
        ),
        # A sheet with a title row, a blank row, an empty first column and a
        # frozen pane. The reader finds the header today, so all three checks
        # are guards: they hold, and are here to fail a change that loses the
        # header search or the drop of the empty column.
        BenchmarkTask(
            name="de_spreadsheet_header_is_found",
            family="data_engineering", kind="analyze",
            goal="Profile the quarterly sales sheet.",
            dataset=DatasetSpec("titled_sheet", "synthetic", suffix=".xlsx",
                                builder=ds.build_titled_sheet),
            source_schemes=("file",),
            expectations=Expectations(
                expected_columns=list(ds.TITLED_SHEET_HEADER),
                forbid_unnamed_columns=True,
                expected_n_rows=3 * ds.TITLED_SHEET_ROWS_PER_REGION,
                guard_checks=("columns_exact", "no_unnamed_columns", "n_rows"),
            ),
        ),
        # The same kind of sheet, with a region that is called "None". A
        # separate task from the guard above, so that a header that is found
        # cannot make a lost category look like a pass, or the other way round.
        BenchmarkTask(
            name="de_spreadsheet_null_words_stay_categories",
            family="data_engineering", kind="analyze",
            goal="Total revenue by region",
            dataset=DatasetSpec(
                "titled_sheet_null_word", "synthetic", suffix=".xlsx",
                builder=lambda dest: ds.build_titled_sheet(
                    dest, regions=ds.TITLED_SHEET_REGIONS_WITH_NULL_WORD)),
            source_schemes=("file",),
            expectations=Expectations(
                expected_groups={
                    region: ds.TITLED_SHEET_ROWS_PER_REGION * ds.TITLED_SHEET_REVENUE_PER_ROW
                    for region in ds.TITLED_SHEET_REGIONS_WITH_NULL_WORD
                },
                forbid_unexpected_groups=True,
                expected_category_counts={"region": {
                    region: ds.TITLED_SHEET_ROWS_PER_REGION
                    for region in ds.TITLED_SHEET_REGIONS_WITH_NULL_WORD
                }},
                expected_missing={"region": 0},
            ),
        ),
    ]


def data_science_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            name="ds_classification_churn",
            family="data_science", kind="train",
            goal="Train a churn classifier that beats a naive baseline and package it.",
            dataset=DatasetSpec("churn", "synthetic", builder=ds.build_churn, target="churned", declared_format="csv"),
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
            dataset=DatasetSpec("housing", "synthetic", builder=ds.build_housing, target="price", declared_format="csv"),
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
            dataset=DatasetSpec("leaky", "synthetic", builder=ds.build_leaky, target="target", declared_format="csv"),
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
            dataset=DatasetSpec("timeseries", "synthetic", builder=ds.build_timeseries, target="revenue", declared_format="csv"),
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
            dataset=DatasetSpec("constant_and_target", "synthetic", builder=ds.build_constant_and_target, target="churned", declared_format="csv"),
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
            dataset=DatasetSpec("continuous_only", "synthetic", builder=ds.build_continuous_only, target="throughput", declared_format="csv"),
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
            dataset=DatasetSpec("no_signal", "synthetic", builder=ds.build_no_signal, target="outcome", declared_format="csv"),
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
            dataset=DatasetSpec("churn", "synthetic", builder=ds.build_churn, declared_format="csv"),
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
        # "revenue" names neither gross_revenue nor net_revenue, so there is
        # no measure to total. Counting rows per region instead is a different
        # question, and labelling it "answered" is what this refuses. It does
        # not say what the right reply is beyond that: not "answered".
        BenchmarkTask(
            name="ds_unresolved_metric_is_not_answered",
            family="data_science", kind="analyze",
            goal="Total revenue by region",
            dataset=DatasetSpec("two_revenue_columns", "synthetic",
                                builder=ds.build_two_revenue_columns),
            source_schemes=("file",),
            expectations=Expectations(expected_goal_status="limited"),
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


def edgar_tasks() -> list[BenchmarkTask]:
    """One real SEC filing, fetched only when ``--edgar`` is passed.

    The 10-K task that always runs is the synthetic workbook in
    ``spreadsheet_tasks``; this is the same shape checked against an actual
    filing. It reaches the network, so it is opt-in exactly as Kaggle is, and
    it skips cleanly when the filing or the required contact header is absent.
    Ground truth cannot be planted in someone else's filing, so this task
    grades the shape questions only -- no placeholder names, narration that
    names real columns -- and not a planted figure.
    """
    from avaloka.benchmark.edgar import EDGAR_SHAPE_TRUTH
    return [
        BenchmarkTask(
            name="edgar_10k_financial_report",
            family="format_comprehension", kind="analyze",
            goal=("Read a real SEC Financial_Report.xlsx and name the statement's "
                  "line items and period columns."),
            dataset=DatasetSpec(
                "edgar_financial_report", "edgar",
                suffix=".xlsx", declared_format="xlsx",
                # Apple's FY2023 10-K: large, stable and well-formed.
                edgar_cik="320193", edgar_accession="0000320193-23-000106",
                ground_truth=EDGAR_SHAPE_TRUTH,
                notes="a real filing: shape graded, no planted figure"),
            source_schemes=("file",),
            expectations=Expectations(
                no_placeholder_column_names=True,
                narration_names_only_real_columns=True,
        narration_columns_from_ground_truth=True,
            ),
        ),
    ]


def all_tasks(*, include_kaggle: bool = False,
              include_edgar: bool = False) -> list[BenchmarkTask]:
    tasks = (data_engineering_tasks() + data_science_tasks()
             + format_tasks() + spreadsheet_tasks()
             + nested_document_tasks())
    if include_kaggle:
        tasks += kaggle_tasks()
    if include_edgar:
        tasks += edgar_tasks()
    return tasks
