"""The workload optimizer's plan is executed, and its estimate is measured.

Until 2.0 the optimizer wrote ``workload_optimization.json`` and nobody read
it: no production code set ``analysis_filters``, so every plan was a projected
scan with zero factors, and the Data Engineer never looked at the plan. These
tests drive the real fireflies in mission order on a dataset and a goal that
names a predicate, and assert both the estimate the optimizer recorded and the
rows the engineer actually touched. A Data Engineer that ignores the plan
fails them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from avaloka.fireflies.analysis_planner import AnalysisPlanner
from avaloka.fireflies.data_engineer import (IN_MEMORY, PUSHDOWN_PYARROW,
                                             DataEngineer)
from avaloka.fireflies.data_scout import DataScout
from avaloka.fireflies.sampling_specialist import SamplingSpecialist
from avaloka.fireflies.workload_optimizer import WorkloadOptimizer
from avaloka.goal_analysis import apply_filters, goal_filters
from avaloka.io import materialize
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze

N_ROWS = 1_000


def _sales(n: int = N_ROWS, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "region": rng.choice(["west", "east", "north"], n, p=[0.2, 0.5, 0.3]),
        "amount": rng.normal(100, 30, n).round(2),
        "units": rng.integers(1, 20, n),
        "channel": rng.choice(["web", "store"], n),
        "currency": ["USD"] * n,                      # constant: the plan projects it away
    })


def _mission(tmp_path, source, goal, *, blackboard=None):
    ctx = MissionContext(MissionKind.ANALYZE, goal, str(source), tmp_path / "out", Budget())
    ctx.blackboard["resolved_source"] = materialize(str(source)).as_dict()
    ctx.blackboard["source_statistics"] = materialize(str(source)).detail.get("statistics", {})
    ctx.blackboard.update(blackboard or {})
    for firefly in (DataScout, SamplingSpecialist, WorkloadOptimizer, DataEngineer):
        firefly(ctx).activate()
    plan = json.loads((tmp_path / "out" / "workload_optimization.json").read_text())
    return ctx, plan


# ----------------------------------------------------------- the goal parser
def test_goal_filters_read_the_predicates_a_goal_names(tmp_path):
    frame = _sales()
    source = tmp_path / "sales.csv"
    frame.to_csv(source, index=False)
    ctx = MissionContext(MissionKind.ANALYZE, "x", str(source), tmp_path / "out", Budget())
    DataScout(ctx).activate()
    profile = ctx.blackboard["profile"]
    assert goal_filters("west sales", profile) == [{"column": "region", "operator": "eq", "value": "west"}]
    assert goal_filters("total amount where region is West", profile) == [
        {"column": "region", "operator": "eq", "value": "west"}]
    assert goal_filters("average amount for units above 12", profile) == [
        {"column": "units", "operator": "gt", "value": 12}]
    assert goal_filters("amount between 80 and 120 by region", profile) == [
        {"column": "amount", "operator": "gte", "value": 80},
        {"column": "amount", "operator": "lte", "value": 120}]
    assert goal_filters("compare east and west", profile) == [
        {"column": "region", "operator": "in", "value": ["east", "west"]}]
    assert goal_filters("rows where channel is missing", profile) == [
        {"column": "channel", "operator": "is_null", "value": None}]
    # Nothing is guessed from words that are not values or columns.
    assert goal_filters("what drives amount", profile) == []
    assert goal_filters("Understand churn", profile) == []
    # And the mask agrees with pandas.
    west = apply_filters(frame, goal_filters("west sales", profile))
    assert len(west) == int((frame["region"] == "west").sum())


# --------------------------------------------------------- estimate vs. measured
@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_estimate_and_measurement_are_both_recorded_and_agree(tmp_path, fmt):
    frame = _sales()
    source = tmp_path / f"sales.{fmt}"
    (frame.to_csv if fmt == "csv" else frame.to_parquet)(source, index=False)
    expected_rows = int((frame["region"] == "west").sum())

    ctx, plan = _mission(tmp_path, source, "west sales")

    # The optimizer derived the predicate from the goal and estimated from the profile.
    assert plan["filters_derived_from"] == "goal"
    assert plan["analysis_filters"] == [{"column": "region", "operator": "eq", "value": "west"}]
    assert plan["predicate_order"] == ["region"]
    assert plan["projection_columns"] == ["region", "amount", "units", "channel"]   # currency is constant
    # Equality on a categorical column is estimated from its top values: exact.
    assert plan["filter_factors"][0]["estimated_filter_factor"] == pytest.approx(expected_rows / N_ROWS)
    assert plan["estimated_rows_after_filters"] == expected_rows

    # The engineer applied it and measured it.
    applied = plan["applied"]
    assert applied["rows_in"] == N_ROWS and applied["rows_out"] == expected_rows
    assert applied["columns_in"] == 5 and applied["columns_out"] == 4
    assert applied["estimated_rows_after_filters"] == expected_rows
    assert applied["estimate_error_rows"] == 0
    assert applied["selectivity_measured"] == pytest.approx(applied["selectivity_estimated"], abs=1e-6)
    assert applied["filters_applied"] == plan["analysis_filters"]
    if fmt == "parquet":
        assert applied["strategy_applied"] == PUSHDOWN_PYARROW
        assert applied["pushed_down"]["applied"] is True and applied["pushed_down"]["engine"] == "pyarrow"
        assert applied["pushed_down"]["columns"] == plan["projection_columns"]
        assert applied["pushed_down"]["rows_read"] == expected_rows
        assert plan["strategy"] == "projected_scan"                   # no index, no partition
    else:
        assert applied["strategy_applied"] == IN_MEMORY
        assert applied["pushed_down"] == {"applied": False, "reason": "not a local Parquet file"}

    # The prepared data is the filtered, projected frame ...
    prepared = pd.read_parquet(tmp_path / "out" / "transformed_dataset.parquet")
    assert len(prepared) == expected_rows and "currency" not in prepared.columns
    assert set(prepared["region"]) == {"west"}
    # ... and the measurement rode the firefly's unit into the ledger (the
    # swarm trace reads it from there into clone.step.completed.metadata).
    unit = next(u for u in ctx.ledger.units if u.firefly == "data_engineer")
    assert unit.metadata["workload"]["rows_out"] == expected_rows
    assert unit.metadata["workload"]["strategy_applied"] == applied["strategy_applied"]
    ops = [t["op"] for t in ctx.blackboard["transformations"]]
    assert "apply_filters" in ops


def test_the_planner_answers_the_goal_over_the_rows_the_plan_kept(tmp_path):
    frame = _sales()
    source = tmp_path / "sales.csv"
    frame.to_csv(source, index=False)
    ctx = MissionContext(MissionKind.ANALYZE, "total amount for west sales", str(source),
                         tmp_path / "out", Budget())
    run_analyze(ctx)
    answer = json.loads((tmp_path / "out" / "goal_analysis.json").read_text())
    west = frame[frame["region"] == "west"]
    assert answer["filters"] == [{"column": "region", "operator": "eq", "value": "west"}]
    assert f"{len(west):,} where region = west" in answer["scope"]
    assert answer["metrics"]["amount"]["sum"] == pytest.approx(west["amount"].sum(), abs=1e-3)
    # And the whole bundle's plan carries the measurement.
    plan = json.loads((tmp_path / "out" / "workload_optimization.json").read_text())
    assert plan["applied"]["rows_out"] == len(west)


def test_a_caller_supplied_filter_set_is_used_as_given(tmp_path):
    frame = _sales()
    source = tmp_path / "sales.csv"
    frame.to_csv(source, index=False)
    given = [{"column": "units", "operator": "gte", "value": 10}]
    _, plan = _mission(tmp_path, source, "west sales", blackboard={"analysis_filters": given})
    assert plan["filters_derived_from"] == "caller" and plan["analysis_filters"] == given
    assert plan["applied"]["rows_out"] == int((frame["units"] >= 10).sum())


def test_a_goal_over_every_row_leaves_the_frame_whole(tmp_path):
    frame = _sales()
    source = tmp_path / "sales.csv"
    frame.to_csv(source, index=False)
    _, plan = _mission(tmp_path, source, "what drives amount")
    assert plan["analysis_filters"] == [] and plan["estimated_rows_after_filters"] == N_ROWS
    applied = plan["applied"]
    assert applied["rows_out"] == applied["rows_in"] == N_ROWS
    assert applied["strategy_applied"] == IN_MEMORY        # the projection still ran
    assert applied["columns_out"] == 4


def test_the_sample_is_not_replaced_by_a_pushdown_read(tmp_path):
    """Above the sampling threshold the engineer must work on the sample the
    Sampling Specialist drew, so the plan is applied in memory and the
    estimate is compared on selectivity, not on rows."""
    frame = _sales(n=120_000, seed=3)
    source = tmp_path / "sales.parquet"
    frame.to_parquet(source, index=False)
    _, plan = _mission(tmp_path, source, "west sales")
    applied = plan["applied"]
    assert applied["strategy_applied"] == IN_MEMORY
    assert applied["pushed_down"] == {"applied": False, "reason": "working frame is a sample"}
    assert applied["rows_in"] == 50_000 and applied["rows_out"] < applied["rows_in"]
    assert applied["selectivity_measured"] == pytest.approx(applied["selectivity_estimated"], abs=0.02)


def test_the_generated_script_repeats_the_scan(tmp_path):
    frame = _sales()
    source = tmp_path / "sales.csv"
    frame.to_csv(source, index=False)
    ctx, plan = _mission(tmp_path, source, "west sales with units above 12")
    script = (tmp_path / "out" / "analysis.py").read_text()
    namespace: dict = {}
    exec(compile(script, "analysis.py", "exec"), namespace)
    again = namespace["transform"](namespace["load"]())
    assert len(again) == plan["applied"]["rows_out"] == int(((frame["region"] == "west") & (frame["units"] > 12)).sum())
    assert list(again.columns) == list(ctx.blackboard["transformed_frame"].columns)
