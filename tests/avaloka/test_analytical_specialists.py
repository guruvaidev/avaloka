"""Hermetic correctness tests for Avaloka analytical specialists."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avaloka.data_agent import (DataMissionRequest, DataQueryMission,
                                ModelResponse, ModelToolCall, ModelUsage,
                                ToolObservation, build_question_plan,
                                build_source_map, review_answer,
                                route_specialists)
from avaloka.experiments import (ExperimentPlan, analyze_ab_test,
                                 minimum_sample_size_per_arm)
from avaloka.features import (FeatureRecipe, TransformStep,
                              execute_feature_recipe, normalize_identifier,
                              verified_join)
from avaloka.statistics import StatisticalPlan, compute_statistic
from avaloka.uncertainty import (split_conformal_classification,
                                 split_conformal_regression)


def test_verified_join_normalizes_keys_and_reports_cardinality():
    left = pd.DataFrame({"id": ["ACME-01", " BETA_2 "], "value": [1, 2]})
    right = pd.DataFrame({"customer_id": ["acme01", "beta2"], "segment": ["a", "b"]})
    joined, diagnostics = verified_join(
        left, right, left_on="id", right_on="customer_id", how="left"
    )
    assert joined["segment"].tolist() == ["a", "b"]
    assert diagnostics.cardinality == "one_to_one"
    assert diagnostics.unmatched_left_rows == 0
    assert normalize_identifier(None) is None


def test_verified_join_rejects_unapproved_many_to_many():
    left = pd.DataFrame({"id": ["a", "a"]})
    right = pd.DataFrame({"id": ["A", "A"]})
    with pytest.raises(ValueError, match="many-to-many"):
        verified_join(left, right, left_on="id", right_on="id")


def test_feature_recipe_applies_domain_transforms_with_lineage():
    frame = pd.DataFrame({
        "raw_id": ["A-1", "B-2"], "amount": [10.0, 100.0],
        "numerator": [1, 3], "denominator": [2, 4],
        "date": ["2024-01-01", "2025-02-03"],
    })
    recipe = FeatureRecipe(steps=(
        TransformStep("normalize_identifier", "entity_id", ("raw_id",)),
        TransformStep("log10", "log_amount", ("amount",)),
        TransformStep("percentage", "share_pct", ("numerator", "denominator")),
        TransformStep("derive_date_part", "year", ("date",), {"part": "year"}),
    ))
    result = execute_feature_recipe(frame, recipe)
    assert result.frame["entity_id"].tolist() == ["a1", "b2"]
    assert result.frame["log_amount"].tolist() == [1.0, 2.0]
    assert result.frame["share_pct"].tolist() == [50.0, 75.0]
    assert result.frame["year"].tolist() == [2024, 2025]
    assert len(result.recipe_hash) == 64
    assert len(result.diagnostics) == 4


def test_statistician_preserves_proportion_denominator_and_group_grain():
    frame = pd.DataFrame({"state": ["IN", "IN", "OH"], "wifi": [True, False, True], "rating": [4, 2, 5]})
    proportion = compute_statistic(
        frame, StatisticalPlan("proportion", ("wifi",), parameters={"equals": True})
    )
    assert proportion.estimate == pytest.approx(2 / 3)
    assert proportion.diagnostics[0] == {"numerator": 2, "denominator": 3}

    grouped = compute_statistic(
        frame, StatisticalPlan(
            "grouped_aggregate", ("rating",), grouping=("state",),
            parameters={"operation": "mean"},
        )
    )
    assert grouped.estimate == [{"state": "IN", "value": 3.0}, {"state": "OH", "value": 5.0}]


def test_statistician_chi_square_has_expected_diagnostics():
    frame = pd.DataFrame({
        "histology": ["a"] * 10 + ["b"] * 10,
        "mutation": [True] * 10 + [False] * 10,
    })
    result = compute_statistic(frame, StatisticalPlan("chi_square", ("histology", "mutation")))
    assert result.statistic == pytest.approx(20.0)
    assert result.degrees_of_freedom == 1
    assert result.diagnostics[0]["contingency_shape"] == [2, 2]


def test_experimenter_refuses_causal_claim_without_randomization():
    frame = pd.DataFrame({"arm": ["c", "c", "t", "t"], "outcome": [1, 2, 4, 5]})
    result = analyze_ab_test(
        frame, ExperimentPlan("arm", "outcome", "c", "t", randomized=False)
    )
    assert result.status == "cannot_verify"
    assert result.effect is None
    assert result.diagnostics[0]["kind"] == "causal_guard"


def test_experimenter_estimates_effect_and_power_requirement():
    frame = pd.DataFrame({
        "arm": ["c"] * 4 + ["t"] * 4,
        "outcome": [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0],
    })
    result = analyze_ab_test(
        frame, ExperimentPlan("arm", "outcome", "c", "t", randomized=True, expected_treatment_share=0.5)
    )
    assert result.status == "pass"
    assert result.effect == pytest.approx(2.0)
    assert result.confidence_interval[0] < result.effect < result.confidence_interval[1]
    assert minimum_sample_size_per_arm(1.0, 0.5) > 0


def test_split_conformal_regression_and_classification():
    regression = split_conformal_regression(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.1, 1.8, 3.2, 3.9]),
        np.array([2.5, 5.0]), target_coverage=0.8,
        test_y=np.array([2.6, 5.1]),
    )
    assert regression.predictions.shape == (2, 2)
    assert regression.empirical_coverage == 1.0

    calibration_probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    classification = split_conformal_classification(
        np.array([0, 1, 0, 1]), calibration_probabilities,
        np.array([[0.85, 0.15], [0.15, 0.85]]), target_coverage=0.75,
        test_y=np.array([0, 1]),
    )
    assert classification.empirical_coverage == 1.0
    assert classification.efficiency >= 1.0


def test_question_plan_routes_dab_specialists_without_claiming_new_tools():
    plan = build_question_plan(
        "Which state has the highest proportion of businesses offering WiFi? Return only the state.",
        "MongoDB nested business records and SQLite reviews",
    )
    source_map = build_source_map("MongoDB nested business records and SQLite reviews")
    route = route_specialists(plan, source_map)
    assert plan.statistical_intent == "descriptive"
    assert plan.join_required
    assert {"cartographer", "transformer", "statistician", "skeptic"}.issubset(route)


def test_answer_reviewer_requests_exact_output_repair():
    plan = build_question_plan("How many records matched? Return only the number.")
    verdict = review_answer(
        plan, "Here is the answer: seven", evidence_refs=["tool-1"], successful_evidence=1
    )
    assert verdict.status == "repair"
    assert any("numeric" in finding for finding in verdict.findings)
    assert any("return-only" in finding for finding in verdict.findings)


class _Provider:
    def __init__(self):
        self.responses = [
            ModelResponse("r1", None, [ModelToolCall("q1", "query", {})], ModelUsage()),
            ModelResponse("r2", None, [ModelToolCall("a1", "return_answer", {"answer": "Here is the answer: 7"})], ModelUsage()),
            ModelResponse("r3", None, [ModelToolCall("a2", "return_answer", {"answer": "7", "evidence_refs": ["q1"]})], ModelUsage()),
        ]

    def complete(self, messages, tools, *, timeout_s):
        return self.responses.pop(0)


class _Tools:
    def specs(self):
        return [{"type": "function", "function": {"name": "query", "parameters": {"type": "object"}}}]

    def execute(self, name, arguments, *, call_id):
        return ToolObservation(True, '[{"count":7}]', result_key="result", rows_read=1, bytes_read=12)

    def close(self):
        return None


def test_data_query_mission_exposes_specialists_and_repairs_answer_once():
    request = DataMissionRequest(
        task_id="fixture", question="How many records matched? Return only the number.",
        database_description="MongoDB collection: records", model="fake",
    )
    result = DataQueryMission(_Provider(), _Tools()).run(request)
    assert result.answer == "7"
    assert result.review_status == "pass"
    assert result.review_repairs == 1
    assert {"cartographer", "statistician", "skeptic"}.issubset(result.specialists)
    assert result.question_plan["return_only"] is True
