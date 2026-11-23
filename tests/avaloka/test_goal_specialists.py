"""The analytical specialists are woken by the goal, not by a seeded blackboard.

The Experimenter, Statistician and ConformalCalibrator ran only when
``experiment_plan``, ``statistical_plan`` or ``conformal_config`` was already
on the blackboard, and no production code wrote them. The planner now writes
them from the goal (and the task), the mission slots the specialist in ahead
of the Validator, and the report states what they found.
"""

import json

import numpy as np
import pandas as pd
import pytest

from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze, run_train
from avaloka.specialist_plans import (conformal_config_for,
                                      experiment_plan_from_goal,
                                      statistical_plan_from_goal)


def _two_arm_frame(n=2000, seed=21, rate_a=0.20, rate_b=0.30):
    """A planted experiment: two balanced arms, B converts more."""
    rng = np.random.default_rng(seed)
    variant = np.where(rng.random(n) < 0.5, "A", "B")
    rate = np.where(variant == "B", rate_b, rate_a)
    return pd.DataFrame({
        "user_id": range(n),
        "variant": variant,
        "converted": (rng.random(n) < rate).astype(int),
        "age": rng.integers(18, 70, n),
    })


def _run(tmp_path, frame, goal, name="ab", **kw):
    source = tmp_path / f"{name}.csv"
    frame.to_csv(source, index=False)
    out = tmp_path / f"{name}-out"
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal=goal, data_source=str(source),
                         output_dir=out, budget=Budget(), **kw)
    result = run_analyze(ctx)
    return out, result


class TestExperimentPlanFromGoal:
    def test_reads_the_arms_and_the_outcome_from_the_goal(self):
        plan, why = experiment_plan_from_goal("is variant B better than A on converted", _two_arm_frame())
        assert plan == {
            "treatment_column": "variant", "outcome": "converted",
            "control": "A", "treatment": "B", "metric": "proportion",
            "randomized": True, "alpha": 0.05, "expected_treatment_share": 0.5,
        }
        assert why["planned"] and "randomised" in why["assumptions"][0]

    def test_finds_the_arm_column_by_name_when_the_goal_does_not_say_it(self):
        frame = _two_arm_frame().rename(columns={"variant": "treatment_group"})
        plan, _ = experiment_plan_from_goal("does the treatment lift converted?", frame)
        assert plan["treatment_column"] == "treatment_group"
        assert (plan["control"], plan["treatment"]) == ("A", "B")

    def test_a_mean_outcome_and_an_unlabelled_design(self):
        frame = pd.DataFrame({"cohort": ["old", "new"] * 50, "revenue": np.arange(100.0)})
        plan, why = experiment_plan_from_goal("compare revenue for new vs old", frame)
        assert plan["metric"] == "mean"
        assert (plan["treatment"], plan["control"]) == ("new", "old")

    def test_no_plan_without_a_two_arm_question_or_an_arm_column(self):
        frame = _two_arm_frame()
        assert experiment_plan_from_goal("what is the average age", frame)[0] is None
        plan, why = experiment_plan_from_goal("is variant B better than A on converted",
                                              frame.drop(columns=["variant"]))
        assert plan is None and "arm" in why["reason"]


class TestStatisticalPlanFromGoal:
    def test_names_the_statistic_and_its_columns(self):
        frame = pd.DataFrame({"tenure": [1.0, 2.0, 3.0], "monthly": [3.0, 2.0, 1.0],
                              "contract": ["a", "b", "a"], "churned": [0, 1, 0]})
        plan, _ = statistical_plan_from_goal("what is the correlation between tenure and monthly", frame)
        assert plan["method"] == "correlation" and plan["variables"] == ["tenure", "monthly"]
        plan, _ = statistical_plan_from_goal("chi-square: is contract independent of churned", frame)
        assert plan["method"] == "chi_square" and set(plan["variables"]) == {"contract", "churned"}
        plan, _ = statistical_plan_from_goal("what proportion of customers churned", frame)
        assert plan["method"] == "proportion" and plan["parameters"] == {"equals": 1}
        plan, _ = statistical_plan_from_goal("average monthly with an interval", frame)
        assert plan["method"] == "mean" and plan["parameters"]["bootstrap"] is True
        assert statistical_plan_from_goal("correlation of tenure", frame)[0] is None
        assert statistical_plan_from_goal("show me everything", frame)[0] is None


class TestABMission:
    def test_an_ab_goal_runs_the_experimenter_and_the_report_states_lift_ci_srm_power(self, tmp_path):
        """The proof in the brief: the goal produces an experiment_plan, the
        Experimenter runs, and the report carries the lift with its CI, the
        SRM check and the power. A mutation that stops writing
        ``experiment_plan`` leaves the ledger without an experimenter."""
        out, result = _run(tmp_path, _two_arm_frame(), "is variant B better than A on converted")
        activated = [u["firefly"] for u in result.summary["fireflies"]]
        assert "experimenter" in activated
        assert activated.index("experimenter") < activated.index("validator")

        plan = json.loads((out / "planner_graph.json").read_text())
        assert plan["specialists"]["experiment"]["plan"]["treatment"] == "B"
        assert "experimenter" in [n["id"] for n in plan["execution_graph"]["nodes"]]

        experiment = json.loads((out / "experiment_result.json").read_text())
        assert experiment["status"] == "pass"
        assert 0.05 < experiment["effect"] < 0.15
        low, high = experiment["confidence_interval"]
        assert low > 0.0 and high > low
        assert experiment["sample_ratio_mismatch"] is False
        assert experiment["power"]["achieved_power"] > 0.9
        assert experiment["power"]["minimum_detectable_effect"] > 0
        assert experiment["plan"]["control"] == "A"

        summary = result.summary["experiment"]
        assert summary["effect"] == pytest.approx(experiment["effect"])
        findings = "\n".join(result.context.blackboard["findings"])
        assert "lift" in findings and "95% CI" in findings
        assert "Sample ratio check passed" in findings
        assert "Power:" in findings
        assert "'B' is better than 'A' on 'converted'" in findings
        for name in ("README.md", "executive_report.html"):
            text = (out / name).read_text()
            assert "lift" in text and "Sample ratio" in text and "Power" in text
        limitations = result.context.blackboard["limitations"]
        assert any("randomised" in l for l in limitations)

    def test_a_null_experiment_is_reported_as_no_difference(self, tmp_path):
        out, result = _run(tmp_path, _two_arm_frame(rate_b=0.20, seed=4),
                           "is variant B better than A on converted", name="null")
        findings = "\n".join(result.context.blackboard["findings"])
        assert "interval includes zero" in findings
        experiment = json.loads((out / "experiment_result.json").read_text())
        assert experiment["confidence_interval"][0] < 0 < experiment["confidence_interval"][1]


class TestStatisticianMission:
    def test_a_correlation_goal_runs_the_statistician(self, churn_csv, tmp_path):
        ctx = MissionContext(kind=MissionKind.ANALYZE,
                             goal="what is the correlation between tenure and monthly",
                             data_source=str(churn_csv), output_dir=tmp_path / "corr", budget=Budget())
        result = run_analyze(ctx)
        activated = [u["firefly"] for u in result.summary["fireflies"]]
        assert "statistician" in activated and "experimenter" not in activated
        statistic = json.loads((tmp_path / "corr" / "statistical_result.json").read_text())
        assert statistic["method"] == "correlation"
        assert -1.0 <= statistic["estimate"] <= 1.0
        assert result.summary["statistic"]["sample_size"] == statistic["sample_size"]
        findings = "\n".join(result.context.blackboard["findings"])
        assert "pearson correlation of tenure and monthly" in findings


class TestConformalMission:
    def test_every_train_mission_calibrates_a_90_percent_label_set(self, trained_package):
        result = json.loads((trained_package / "conformal_result.json").read_text())
        assert result["task"] == "classification"
        assert result["target_coverage"] == 0.9
        assert result["empirical_coverage"] is not None
        assert result["efficiency"] >= 1.0
        mission = json.loads((trained_package / "mission.json").read_text())
        assert mission["conformal"]["target_coverage"] == 0.9
        assert "conformal_calibrator" in [u["firefly"] for u in mission["fireflies"]]
        plan = json.loads((trained_package / "planner_graph.json").read_text())
        assert plan["specialists"]["conformal"]["planned"] is True
        assert "conformal_calibrator" in [n["id"] for n in plan["execution_graph"]["nodes"]]

    def test_a_regression_mission_calibrates_an_interval(self, tmp_path):
        rng = np.random.default_rng(9)
        frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
        frame["y"] = 3 * frame.a - frame.b + rng.normal(0, 0.5, 300)
        source = tmp_path / "reg.csv"
        frame.to_csv(source, index=False)
        ctx = MissionContext(kind=MissionKind.TRAIN, goal="Predict y", data_source=str(source),
                             output_dir=tmp_path / "reg", budget=Budget(limit_usd=0.06), target="y")
        result = run_train(ctx)
        conformal = json.loads((tmp_path / "reg" / "conformal_result.json").read_text())
        assert conformal["task"] == "regression"
        assert conformal["efficiency"] > 0
        assert 0.7 <= conformal["empirical_coverage"] <= 1.0
        assert any("Conformal interval at 90%" in f for f in result.context.blackboard["findings"])

    def test_config_only_for_modelling_tasks(self):
        assert conformal_config_for("regression") == {"task": "regression", "target_coverage": 0.9,
                                                      "method": "split"}
        assert conformal_config_for("binary_classification")["task"] == "classification"
        assert conformal_config_for("exploratory_analysis") is None
