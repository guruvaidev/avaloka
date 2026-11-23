"""A goal maps to the calculation it asks for, its numbers are honest, and PASS is earned.

Four defects from the 2.0 hand-graded CLI run, each pinned here:

* "What predicts default?" became the filter ``loan_status = Default`` and
  analysed the two rows it kept. The outcome of a predicts-goal is a target.
* "Average revenue by region" printed a SUM of a 50,000-row sample of 4M.
* "payment_type=4: mean fare_amount=-1.37" where the full data gives +1.33
  (a rare group's sample mean), next to "mean of payment_type: 1.16".
* "Validation verdict: PASS" on a run that answered nothing.
"""

import json

import numpy as np
import pandas as pd
import pytest

import avaloka.fireflies.analysis_planner as planner_module
import avaloka.fireflies.sampling_specialist as sampling_module
import avaloka.goal_analysis as goal_module
from avaloka.fireflies.validator import reconcile_figures
from avaloka.goal_analysis import analyze_goal, driver_target, goal_filters, is_categorical_code
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze
from avaloka.specialist_plans import statistical_plan_from_goal
from avaloka.stats import profile_column


def _loans(n=600, seed=3):
    rng = np.random.default_rng(seed)
    income = rng.normal(60_000, 15_000, n).round(0)
    status = np.where(rng.random(n) < 0.15, "Default", rng.choice(["Current", "Fully Paid"], n))
    # Lower income, higher default: there is something to find.
    status = np.where(income < 40_000, "Default", status)
    return pd.DataFrame({
        "loan_amnt": rng.integers(1_000, 30_000, n),
        "annual_inc": income,
        "grade": rng.choice(list("ABCD"), n),
        "loan_status": status,
    })


def _profile(frame):
    return {"dataset": {"n_rows": len(frame)},
            "columns": [profile_column(frame[c], len(frame)) for c in frame.columns]}


def _mission(tmp_path, frame, goal, name="out"):
    source = tmp_path / f"{name}.csv"
    frame.to_csv(source, index=False)
    ctx = MissionContext(MissionKind.ANALYZE, goal, str(source), tmp_path / name, Budget())
    result = run_analyze(ctx)
    return ctx, result, json.loads((tmp_path / name / "goal_analysis.json").read_text())


# --- (a) a predicts-goal makes X the target, never a filter -----------------
def test_a_predicts_goal_targets_the_outcome_and_filters_nothing_on_it():
    frame = _loans()
    profile = _profile(frame)
    assert goal_filters("what predicts default?", profile) == []
    target = driver_target("what predicts default?", profile)
    assert (target["column"], target["value"]) == ("loan_status", "Default")

    result = analyze_goal(frame, "what predicts default?", profile)
    assert result["kind"] == "drivers" and result["status"] == "answered"
    assert result["metrics"]["target"] == "loan_status"
    assert result["metrics"]["target_value"] == "Default"
    assert "filters" not in result
    assert result["metrics"]["positives"] == int((frame["loan_status"] == "Default").sum())
    assert result["metrics"]["drivers"][0]["column"] == "annual_inc"


def test_a_predicts_mission_keeps_every_row(tmp_path):
    frame = _loans()
    ctx, _, answer = _mission(tmp_path, frame, "what predicts default?")
    applied = ctx.blackboard["workload_optimization"]["applied"]
    assert applied["filters_applied"] == []
    assert len(ctx.blackboard["transformed_frame"]) == len(frame)
    assert answer["kind"] == "drivers" and answer["metrics"]["target"] == "loan_status"


def test_a_filter_on_another_column_survives_a_predicts_goal():
    frame = _loans()
    filters = goal_filters("what predicts default for grade = B", _profile(frame))
    assert filters == [{"column": "grade", "operator": "eq", "value": "B"}]


def test_a_rare_outcome_says_so_and_names_the_other_values():
    frame = _loans()
    frame.loc[frame["loan_status"] == "Default", "loan_status"] = "Charged Off"
    frame.loc[:4, "loan_status"] = "Default"
    result = analyze_goal(frame, "what predicts default?", _profile(frame))
    assert result["status"] == "limited"
    assert "too few" in result["findings"][0] and "Charged Off" in result["findings"][0]


# --- (b) a mean-by-group goal computes a mean, and says sample or full -------
def _sales(n=4_000, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"region": rng.choice(["north", "south", "east"], n),
                         "revenue": rng.gamma(2, 50, n).round(2)})


def test_b_average_by_group_is_a_mean_over_the_full_data_when_held():
    full = _sales()
    sample = full.sample(500, random_state=1).reset_index(drop=True)
    result = analyze_goal(sample, "average revenue by region", _profile(full), sampled=True,
                          population=full)
    assert result["metrics"]["aggregation"] == "mean"
    exact = full.groupby("region")["revenue"].mean()
    assert {g["group"]: g["value"] for g in result["metrics"]["groups"]} == pytest.approx(exact.to_dict())
    assert result["basis"] == {"sampled": False, "rows_used": 4_000, "source_rows": 4_000, "full_pass": True}
    assert any("Computed over all 4,000 rows" in f for f in result["findings"])
    assert not any(" sum " in f for f in result["findings"])


def test_b_without_the_full_data_the_mean_is_labelled_an_estimate():
    full = _sales()
    sample = full.sample(500, random_state=1).reset_index(drop=True)
    result = analyze_goal(sample, "average revenue by region", _profile(full), sampled=True)
    assert result["metrics"]["aggregation"] == "mean"
    assert result["basis"]["sampled"] is True
    assert "Estimated from a 500-row sample of 4,000 rows." in result["findings"]
    assert all("ci95" in g for g in result["metrics"]["groups"])


def test_b_a_sample_total_is_never_presented_as_a_total():
    full = _sales()
    sample = full.sample(500, random_state=1).reset_index(drop=True)
    result = analyze_goal(sample, "total revenue by region", _profile(full), sampled=True)
    assert result["metrics"]["aggregation"] == "sum"
    assert all("sample sum revenue=" in f for f in result["findings"][:3])
    assert any("not dataset totals" in f for f in result["findings"])


# --- (c) the taxi-like fixture keeps its sign --------------------------------
def _taxi(seed=11):
    """A rare dispute group whose fares are half refunds: full mean +1.3."""
    rng = np.random.default_rng(seed)
    common = pd.DataFrame({"payment_type": rng.choice([1, 2], 20_000), "fare_amount": rng.gamma(3, 6, 20_000)})
    dispute = rng.normal(0, 25, 600)
    dispute = dispute - dispute.mean() + 1.3
    rare = pd.DataFrame({"payment_type": 4, "fare_amount": dispute})
    return pd.concat([common, rare], ignore_index=True)


def test_c_the_rare_groups_sign_comes_from_the_full_data():
    full = _taxi()
    rare = full[full["payment_type"] == 4]
    # A sample whose dispute rows happen to be the refunds: its mean is negative.
    unlucky = pd.concat([full[full["payment_type"] != 4].sample(2_000, random_state=0),
                         rare.nsmallest(40, "fare_amount")], ignore_index=True)
    assert unlucky.loc[unlucky["payment_type"] == 4, "fare_amount"].mean() < 0
    result = analyze_goal(unlucky, "average fare_amount by payment_type", _profile(full), sampled=True,
                          population=full)
    group4 = next(g for g in result["metrics"]["groups"] if g["group"] == "4")
    assert group4["value"] == pytest.approx(1.3, abs=1e-6)
    assert group4["value"] > 0


def test_c_a_sampled_mean_whose_interval_spans_zero_does_not_claim_a_sign():
    full = _taxi()
    sample = full.sample(3_000, random_state=5).reset_index(drop=True)
    result = analyze_goal(sample, "average fare_amount by payment_type", _profile(full), sampled=True)
    line = next(f for f in result["findings"] if f.startswith("payment_type=4:"))
    assert "the sign is not established" in line


# --- (d) a categorical code is never averaged --------------------------------
def test_d_a_code_column_is_recognised():
    full = _taxi()
    assert is_categorical_code("payment_type", full["payment_type"])
    assert is_categorical_code("RatecodeID", pd.Series([1.0, 2.0, 5.0]))
    assert not is_categorical_code("fare_amount", full["fare_amount"])
    assert not is_categorical_code("passenger_count", pd.Series([1, 2, 3]))


def test_d_the_statistician_averages_the_measure_not_the_code():
    full = _taxi()
    plan, _ = statistical_plan_from_goal("average fare_amount by payment_type", full)
    assert plan["variables"] == ["fare_amount"]
    plan, why = statistical_plan_from_goal("average payment_type", full)
    assert plan is None and "category codes" in why["reason"]


def test_d_goal_analysis_never_reports_a_mean_of_codes():
    full = _taxi()
    full["region"] = np.where(np.arange(len(full)) % 2, "east", "west")
    grouped = analyze_goal(full, "mean payment_type by region", _profile(full))
    assert grouped["status"] == "limited" and "groups" not in grouped["metrics"]
    assert "category codes" in grouped["findings"][0]
    summary = analyze_goal(full, "what is the average payment_type?", _profile(full))
    assert "mean" not in summary["metrics"]["payment_type"]
    assert summary["metrics"]["payment_type"]["categorical_code"] is True


# --- (e) PASS needs an answer that reconciles; a non-answer is WARN -----------
def test_e_an_answered_reconciled_goal_passes(tmp_path):
    ctx, result, _ = _mission(tmp_path, _sales(600), "average revenue by region")
    checks = {c["name"]: c for c in ctx.blackboard["validation"]["checks"]}
    assert checks["goal_answered"]["status"] == "pass"
    assert checks["goal_reconciliation"]["status"] == "pass"
    assert result.summary["verdict"] == "pass"


def test_e_a_non_answer_is_warn_in_plain_words(tmp_path):
    ctx, result, answer = _mission(tmp_path, _sales(600), "What caused the CEO to resign?")
    assert answer["status"] == "limited"
    checks = {c["name"]: c for c in ctx.blackboard["validation"]["checks"]}
    assert checks["goal_answered"]["status"] == "warn"
    assert checks["goal_answered"]["detail"].startswith("Not answered:")
    assert result.summary["verdict"] == "warn"


# --- (f) a planted mismatch fails reconciliation -----------------------------
def test_f_reconciliation_catches_a_planted_mismatch():
    full = _sales()
    result = analyze_goal(full, "average revenue by region", _profile(full))
    assert reconcile_figures(result["figures"], full) == []
    planted = [dict(f) for f in result["figures"]]
    planted[1] = {**planted[1], "value": planted[1]["value"] + 0.5}
    mismatches = reconcile_figures(planted, full)
    assert [m["label"] for m in mismatches] == [planted[1]["label"]]


def test_f_a_mission_reporting_a_number_the_data_does_not_give_fails(tmp_path, monkeypatch):
    real = planner_module.analyze_goal

    def sum_reported_as_mean(*args, **kwargs):
        result = real(*args, **kwargs)
        for figure in result["figures"]:
            figure["value"] = figure["value"] * 600 / 3          # a total where a mean belongs
        return result

    monkeypatch.setattr(planner_module, "analyze_goal", sum_reported_as_mean)
    ctx, result, _ = _mission(tmp_path, _sales(600), "average revenue by region")
    checks = {c["name"]: c for c in ctx.blackboard["validation"]["checks"]}
    assert checks["goal_reconciliation"]["status"] == "fail"
    assert "do not match a recomputation" in checks["goal_reconciliation"]["detail"]
    assert result.summary["verdict"] == "fail"


# --- the reporter's sample limitation follows the result's own basis ----------
def test_the_working_sample_limitation_follows_the_goal_basis(tmp_path, monkeypatch):
    monkeypatch.setattr(sampling_module, "SAMPLE_THRESHOLD_ROWS", 100)
    monkeypatch.setattr(sampling_module, "DEFAULT_SAMPLE_ROWS", 150)
    line = "Goal-specific figures use the working sample, not every source row."
    ctx, _, full = _mission(tmp_path, _sales(600), "average revenue by region", "full")
    assert ctx.blackboard["sampling"]["sampled"] and full["basis"]["full_pass"]
    assert line not in ctx.blackboard["limitations"]

    monkeypatch.setattr(goal_module, "FULL_PASS_MAX_ROWS", 10)
    ctx, _, sampled = _mission(tmp_path, _sales(600), "average revenue by region", "sampled")
    assert sampled["basis"]["sampled"]
    assert line in ctx.blackboard["limitations"]
    assert any("Estimated from a 150-row sample of 600 rows." == f for f in sampled["findings"])


# --- the refusals the e2e run graded unwarranted ------------------------------
def _retail(n=400, seed=2):
    rng = np.random.default_rng(seed)
    items = np.array(["MUG", "LANTERN", "CANDLE", "TOWEL"])
    return pd.DataFrame({
        "InvoiceNo": np.arange(n) + 536_000,
        "Description": items[rng.choice(4, n, p=[0.1, 0.2, 0.3, 0.4])],
        "Quantity": rng.integers(1, 12, n),
        "UnitPrice": rng.choice([1.25, 2.5, 4.95], n),
        "Country": rng.choice(["United Kingdom", "France", "Spain"], n, p=[0.8, 0.1, 0.1]),
    })


def test_what_sells_most_ranks_items_by_units_sold():
    frame = _retail()
    result = analyze_goal(frame, "what sells most?", _profile(frame))
    assert result["kind"] == "contribution" and result["status"] == "answered"
    rows = result["metrics"]["dimensions"]["Description"]
    exact = frame.groupby("Description")["Quantity"].sum().sort_values(ascending=False)
    assert [r["group"] for r in rows] == list(exact.index)
    assert rows[0]["value"] == exact.iloc[0]
    assert reconcile_figures(result["figures"], frame) == []


def test_which_products_and_countries_drive_revenue_uses_quantity_times_price():
    frame = _retail()
    result = analyze_goal(frame, "which products and countries drive revenue?", _profile(frame))
    assert result["kind"] == "contribution" and result["status"] == "answered"
    assert set(result["metrics"]["dimensions"]) == {"Description", "Country"}
    revenue = frame["Quantity"] * frame["UnitPrice"]
    exact = revenue.groupby(frame["Country"]).sum()
    top = result["metrics"]["dimensions"]["Country"][0]
    assert top["group"] == "United Kingdom" and top["value"] == pytest.approx(exact["United Kingdom"])
    assert "Quantity x UnitPrice" in result["metrics"]["measure"]
    assert reconcile_figures(result["figures"], frame) == []


def test_which_patients_are_readmitted_targets_the_outcome_value():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"Glucose": rng.normal(120, 20, 500),
                          "Outcome": rng.choice(["Recovered", "readmitted", "Readmitted", "Deceased"], 500)})
    frame.loc[frame["Glucose"] > 140, "Outcome"] = "Readmitted"
    profile = _profile(frame)
    assert goal_filters("which patients are readmitted?", profile) == []
    result = analyze_goal(frame, "which patients are readmitted?", profile)
    assert result["kind"] == "drivers" and result["status"] == "answered"
    assert result["metrics"]["target"] == "Outcome"
    # Case-insensitive: both spellings of the outcome count as it.
    assert result["metrics"]["positives"] == int(frame["Outcome"].str.lower().eq("readmitted").sum())
    assert reconcile_figures(result["figures"], frame) == []


def test_what_drives_a_column_named_by_a_word_of_it():
    frame = _taxi().sample(5_000, random_state=0).reset_index(drop=True)
    frame["trip_distance"] = frame["fare_amount"] / 3 + np.random.default_rng(1).normal(0, 1, len(frame))
    result = analyze_goal(frame, "what drives the fare?", _profile(frame))
    assert result["metrics"]["target"] == "fare_amount"
    assert result["metrics"]["drivers"][0]["column"] == "trip_distance"
    assert reconcile_figures(result["figures"], frame) == []


def test_when_is_a_measure_highest_says_which_column_it_could_mean():
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=96, freq="h"),
                          "Global_active_power": np.tile(np.r_[np.ones(18), np.full(6, 5.0)], 4)})
    unnamed = analyze_goal(frame, "when is consumption highest?", _profile(frame))
    assert unnamed["kind"] == "peak_timing" and unnamed["status"] == "limited"
    assert "Global_active_power" in unnamed["findings"][0]
    named = analyze_goal(frame, "when is Global_active_power highest?", _profile(frame))
    assert named["status"] == "answered" and named["metrics"]["peaks"]["hour"]["at"] == "18"
    assert reconcile_figures(named["figures"], frame) == []
