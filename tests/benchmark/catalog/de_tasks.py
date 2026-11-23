"""
Data-engineering benchmark tasks.

These exercise the CLI's ability to ingest, sample, profile, and transform data
from heterogeneous sources -- the "data engineer" half of the swarm. Tasks are
scored on structural correctness (schema/DDL, row counts, transformation
semantics), not model quality.
"""

from __future__ import annotations

from ..sources import ALL_SOURCE_KINDS
from ..tasks import (
    BenchmarkTask,
    Severity,
    Stage,
    check_ddl_well_formed,
    check_keys_present,
    check_no_error,
    check_predicate,
    check_rows_nonempty,
    check_schema_covers_columns,
)


def _has_feature_roles(result, ctx):
    feats = result.get("features") or []
    roles = {f.get("role") for f in feats}
    return {"numeric", "categorical"} & roles and (
        ctx.target is None or any(f["role"] == "target" for f in feats)
    )


def _missing_fraction_reported(result, ctx):
    feats = result.get("features") or []
    return all("missing_fraction" in f and 0.0 <= f["missing_fraction"] <= 1.0
               for f in feats) and len(feats) > 0


def _transform_dropped_nulls(result, ctx):
    # After fillna there must be no row loss; output must exist and be readable.
    import pandas as pd
    out = result.get("output_path")
    return bool(out) and len(pd.read_csv(out)) == result.get("rows")


def _groupby_reduced_rows(result, ctx):
    return result.get("rows", 10**9) <= ctx.extras.get("n_rows", 0)


DE_TASKS = [
    # --- sampling across every source kind --------------------------------- #
    BenchmarkTask(
        task_id="de.sample.multisource",
        stage=Stage.SAMPLE,
        dataset_key="titanic",
        brief="Sample 10% of the dataset stratified by the target; emit DDL + schema.",
        source_kinds=ALL_SOURCE_KINDS,  # local, sqlite, postgres, gcs, s3, azure
        checks=[
            check_no_error(),
            check_keys_present(["schema", "rows", "ddl_schema"]),
            check_rows_nonempty(),
            check_schema_covers_columns(min_cols=5),
            check_ddl_well_formed(),
        ],
        tags=["ingest", "sampling", "multisource"],
    ),
    BenchmarkTask(
        task_id="de.sample.churn.sql",
        stage=Stage.SAMPLE,
        dataset_key="telco_churn",
        brief="Sample from a SQL source; verify schema parity with the CSV source.",
        source_kinds=["sqlite", "postgres"],
        checks=[
            check_no_error(),
            check_rows_nonempty(),
            check_schema_covers_columns(min_cols=8),
        ],
        tags=["ingest", "sql"],
    ),
    # --- feature profiling -------------------------------------------------- #
    BenchmarkTask(
        task_id="de.analyze.profile",
        stage=Stage.ANALYZE,
        dataset_key="telco_churn",
        brief="Profile every column: dtype, role, missingness, cardinality.",
        checks=[
            check_no_error(),
            check_keys_present(["features"]),
            check_predicate("feature_roles_assigned", _has_feature_roles, Severity.MAJOR),
            check_predicate("missingness_reported", _missing_fraction_reported, Severity.MAJOR),
        ],
        tags=["profiling", "eda"],
    ),
    # --- transformations ---------------------------------------------------- #
    BenchmarkTask(
        task_id="de.transform.clean",
        stage=Stage.TRANSFORM,
        dataset_key="telco_churn",
        brief="Coerce TotalCharges to numeric then fill missing values with 0.",
        checks=[
            check_no_error(),
            check_keys_present(["output_path", "rows", "columns"]),
            check_predicate("no_row_loss_after_fillna", _transform_dropped_nulls, Severity.MAJOR),
        ],
        tags=["cleaning", "type-coercion"],
    ),
    BenchmarkTask(
        task_id="de.transform.aggregate",
        stage=Stage.TRANSFORM,
        dataset_key="nyc_taxi",
        brief="Group taxi trips by pickup zone and sum total_amount (DE aggregation).",
        checks=[
            check_no_error(),
            check_keys_present(["output_path", "rows"]),
            check_predicate("aggregation_reduced_rows", _groupby_reduced_rows, Severity.MAJOR),
        ],
        tags=["aggregation", "large"],
    ),
]
