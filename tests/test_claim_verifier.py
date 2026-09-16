"""Claim verification: does the evidence support what we told the user?

Each test builds the specific way an automated analysis misleads, so a
regression reads as "we stopped catching causal overreach" rather than a
number moving.
"""

from __future__ import annotations

import pytest

from app.agents.claim_verifier import (CHECKS, Verdict, check_absolute_language,
                                       check_causal_overreach,
                                       check_contradicted_by_metrics,
                                       check_invented_numbers,
                                       check_unsupported_significance,
                                       claim_verifier_node, split_claims,
                                       verify_claims)

CORRELATIONAL = {"correlation": 0.62, "n_rows": 4200}
EXPERIMENTAL = {"ab_test": {"treatment": 0.31, "control": 0.24}, "p_value": 0.003}


# --------------------------------------------------------------------------- #
# The headline failure: causation asserted from correlation
# --------------------------------------------------------------------------- #

def test_causal_language_from_correlation_is_overreach():
    report = verify_claims(
        "Reducing delivery time causes a large increase in repeat purchases.",
        CORRELATIONAL)
    assert not report.safe_to_present
    assert report.findings[0].verdict is Verdict.OVERREACH
    assert report.findings[0].check == "causal_overreach"


@pytest.mark.parametrize("phrasing", [
    "Late deliveries drive churn among premium customers.",
    "Churn increased because of the price change last quarter.",
    "The discount led to higher basket sizes across all regions.",
    "Support delays are responsible for the drop in satisfaction.",
])
def test_common_causal_phrasings_are_caught(phrasing):
    assert verify_claims(phrasing, CORRELATIONAL).problems


def test_causal_language_is_allowed_with_experimental_evidence():
    """The check must not punish a claim the evidence actually licenses."""
    report = verify_claims(
        "The treatment causes a 7 point lift in conversion.", EXPERIMENTAL)
    causal = [f for f in report.findings if f.check == "causal_overreach"]
    assert not causal


def test_correlational_wording_is_left_alone():
    report = verify_claims(
        "Delivery time is associated with repeat purchase rate.", CORRELATIONAL)
    assert report.safe_to_present


def test_the_word_causal_in_a_heading_does_not_trip_the_check():
    """Whole-word matching: 'causal analysis' as a label is not a claim."""
    assert not check_causal_overreach("This section covers causal analysis methods.",
                                      {"correlation": 0.4})


# --------------------------------------------------------------------------- #
# Numbers that appear nowhere in the evidence
# --------------------------------------------------------------------------- #

def test_a_number_absent_from_the_evidence_is_unsupported():
    finding = check_invented_numbers(
        "Revenue grew by 47.3% year over year.",
        {"growth_rate": 12.1, "revenue": 980000})
    assert finding is not None and finding.verdict is Verdict.UNSUPPORTED


def test_a_number_present_in_the_evidence_passes():
    assert check_invented_numbers(
        "Revenue grew by 12.1% year over year.",
        {"growth_rate": 12.1}) is None


def test_numbers_are_matched_within_tolerance():
    assert check_invented_numbers("Accuracy reached 0.873.",
                                  {"accuracy": 0.8731}) is None


def test_small_counts_in_prose_are_not_treated_as_statistics():
    """'3 segments' is prose, not an invented statistic."""
    assert check_invented_numbers("We identified 3 segments.",
                                  {"silhouette": 0.44}) is None


def test_small_percentage_is_not_exempt_from_verification():
    """A percentage remains a statistic even when its value is an integer."""
    finding = check_invented_numbers(
        "Churn is 8.0%.",
        {"churn_rate": 35.9},
    )
    assert finding is not None
    assert finding.verdict is Verdict.UNSUPPORTED
    assert finding.check == "invented_number"


def test_numbers_nested_deep_in_evidence_are_found():
    assert check_invented_numbers(
        "The model reached 0.913 accuracy.",
        {"evaluation_report": {"metrics": {"accuracy": {"mean": 0.913}}}}) is None


def test_statistic_with_empty_evidence_is_unsupported():
    """A numeric claim must not pass merely because evidence is empty."""
    finding = check_invented_numbers("Churn was 47.3%.", {})
    assert finding is not None
    assert finding.verdict is Verdict.UNSUPPORTED
    assert finding.check == "invented_number"
    assert "no computed numeric evidence" in finding.detail


def test_prose_counts_and_years_still_pass_with_empty_evidence():
    """The empty-evidence guard must not turn metadata into statistics."""
    assert check_invented_numbers(
        "We identified 3 segments in 2024.",
        {},
    ) is None


# --------------------------------------------------------------------------- #
# Claims the model's own metrics contradict
# --------------------------------------------------------------------------- #

def test_claiming_reliability_when_the_baseline_was_not_beaten():
    """Ties the verifier to the Evaluation agent's verdict."""
    report = verify_claims(
        "The model accurately predicts churn for new customers.",
        {"evaluation_report": {"beats_baseline": False, "primary_metric": "accuracy"}})
    assert not report.safe_to_present
    assert report.findings[0].verdict is Verdict.CONTRADICTED
    assert report.findings[0].evidence_ref == "evaluation_report.beats_baseline"


def test_the_same_claim_is_fine_when_the_baseline_was_beaten():
    report = verify_claims(
        "The model accurately predicts churn for new customers.",
        {"evaluation_report": {"beats_baseline": True}})
    contradicted = [f for f in report.findings if f.verdict is Verdict.CONTRADICTED]
    assert not contradicted


def test_a_neutral_statement_is_not_contradicted():
    assert check_contradicted_by_metrics(
        "The dataset contains 4,200 rows.",
        {"evaluation_report.beats_baseline": False}) is None


# --------------------------------------------------------------------------- #
# Absolutes and unsupported significance
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("claim", [
    "Churn will always rise when prices increase.",
    "This guarantees higher retention next quarter.",
    "Premium customers never churn.",
])
def test_absolute_language_is_overreach(claim):
    finding = check_absolute_language(claim, {})
    assert finding is not None and finding.verdict is Verdict.OVERREACH


def test_hedged_language_passes():
    assert check_absolute_language(
        "Churn tends to rise when prices increase.", {}) is None


def test_significant_without_a_test_is_overreach():
    finding = check_unsupported_significance(
        "There is a significant difference between the two segments.",
        {"mean_a": 4.1, "mean_b": 3.8})
    assert finding is not None and finding.verdict is Verdict.OVERREACH


def test_significant_with_a_p_value_is_allowed():
    assert check_unsupported_significance(
        "There is a significant difference between the two segments.",
        {"p_value": 0.01}) is None


def test_significant_with_a_confidence_interval_is_allowed():
    assert check_unsupported_significance(
        "The lift is significant.", {"ci_low": 0.02, "ci_high": 0.09}) is None


# --------------------------------------------------------------------------- #
# Report behaviour
# --------------------------------------------------------------------------- #

def test_a_clean_narrative_is_safe_to_present():
    report = verify_claims(
        "The dataset contains 4200 rows across 3 regions. "
        "Delivery time is associated with repeat purchase rate.",
        {"n_rows": 4200, "correlation": 0.62})
    assert report.safe_to_present
    assert report.claims_checked == 2
    assert not report.problems


def test_an_empty_narrative_checks_nothing():
    report = verify_claims("", {"anything": 1})
    assert report.claims_checked == 0 and report.safe_to_present


def test_each_claim_yields_at_most_one_finding():
    """A sentence with several problems reports the strongest, not a pile."""
    report = verify_claims(
        "The price change causes a significant 47.3% increase and always will.",
        {"correlation": 0.3})
    assert len(report.findings) == 1


def test_contradiction_outranks_the_softer_checks():
    """Check order matters: the metric contradiction is the important one."""
    report = verify_claims(
        "The model accurately predicts churn and this guarantees retention.",
        {"evaluation_report": {"beats_baseline": False}})
    assert report.findings[0].check == "contradicted_by_metrics"


def test_split_claims_ignores_fragments():
    claims = split_claims("Yes. The revenue grew steadily across every region. No.")
    assert len(claims) == 1


def test_report_serialises_for_the_api():
    import json
    payload = verify_claims("Late delivery causes churn.", CORRELATIONAL).as_dict()
    json.dumps(payload)
    assert payload["safe_to_present"] is False
    assert set(payload["findings"][0]) == {
        "claim", "verdict", "check", "detail", "evidence_ref"}


def test_node_returns_only_declared_keys():
    out = claim_verifier_node({
        "analysis_narrative": "Delivery delays cause churn.",
        "evaluation_report": {"beats_baseline": True},
    })
    assert set(out) == {"verification_report", "verification_safe_to_present"}
    assert out["verification_safe_to_present"] is False


def test_node_handles_a_missing_narrative():
    out = claim_verifier_node({})
    assert out["verification_safe_to_present"] is True


def test_verifier_needs_no_llm():
    """The floor must not depend on a model that can itself hallucinate."""
    import inspect

    import app.agents.claim_verifier as mod
    source = inspect.getsource(mod)
    for forbidden in ("ChatGroq", "build_chat_model", "openai", "invoke("):
        assert forbidden not in source, f"claim verifier must stay deterministic ({forbidden})"


def test_all_checks_accept_the_same_signature():
    for check in CHECKS:
        assert check("some claim text here", {}) is None or True


# --------------------------------------------------------------------------- #
# "No analysis ran" is not the same state as "analysis produced no numbers"
# --------------------------------------------------------------------------- #

def test_conversational_turn_is_not_verified_against_an_empty_evidence_set():
    """The node is the `end` target from the conversational route.

    Ordinary chat turns reach it with every evidence key None. Verifying a true
    sentence against nothing marks it invented: "This dataset has 12 columns"
    can never be supported, because the profile is not among the keys the node
    collects. Those findings cost nothing in safety -- safe_to_present only
    trips on CONTRADICTED/OVERREACH -- and everything in signal.
    """
    out = claim_verifier_node({
        "analysis_narrative": "I scanned 1,247 rows and found no issues.",
    })
    report = out["verification_report"]
    assert report["findings"] == [], (
        f"a chat turn with no analysis produced findings: {report['findings']}")
    assert out["verification_safe_to_present"] is True


def test_skipped_turn_says_it_was_skipped():
    """Silence and "nothing to check" must be distinguishable downstream.

    A report of zero findings because verification ran, and one because it was
    never attempted, mean different things to anything reading the artifact.
    """
    out = claim_verifier_node({"analysis_narrative": "Your file is 45.2 MB."})
    assert out["verification_report"].get("skipped"), (
        "the report must record that verification was skipped, not imply a pass")


def test_one_piece_of_evidence_is_enough_to_verify():
    """The guard must be narrow: ANY evidence means the claim is checkable.

    Otherwise a partially-populated state would silently skip verification --
    which would be a far worse failure than the false positives this fixes.
    """
    out = claim_verifier_node({
        "analysis_narrative": "Accuracy was 0.913.",
        "evaluation_report": {"metrics": {"accuracy": 0.42}},
    })
    report = out["verification_report"]
    assert "skipped" not in report, "verification must run when evidence exists"
    assert report["findings"], "a contradicted figure should still be caught"


def test_direct_callers_still_get_fail_closed_behaviour():
    """The short-circuit lives in the NODE, not in the check.

    A caller who deliberately passes an empty evidence set is asserting there
    is nothing to support the claim, and should still be told so.
    """
    finding = check_invented_numbers("Churn was 47.3%.", {})
    assert finding is not None
    assert finding.verdict is Verdict.UNSUPPORTED


def test_analysis_that_produced_no_numbers_is_still_verified():
    """The distinction the guard turns on.

    Evidence exists but contains no numbers -- an analysis DID run and reported
    nothing numeric. A figure in the narrative is then genuinely unsupported,
    and must not be waved through by the skip path.
    """
    out = claim_verifier_node({
        "analysis_narrative": "Churn was 47.3%.",
        "evaluation_report": {"note": "no numeric metrics were computed"},
    })
    report = out["verification_report"]
    assert "skipped" not in report, (
        "evidence was present, so verification must not be skipped")
    assert report["findings"], "an unsupported figure should be reported"
