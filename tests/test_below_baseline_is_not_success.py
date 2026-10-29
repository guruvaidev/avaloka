"""A model that does not beat a trivial baseline must not be presented as a success.

Found by running the HTTP API end to end: training churn.csv returned
evaluation_beats_baseline=False, yet the reply said "The training was
successful", invited predictions, and verification_safe_to_present was True.
And an analysis narrative said "the west region has the lowest at 68.12" next
to a result table whose minimum was 67.73, and the verifier passed it.
"""
from __future__ import annotations

from app.agents.claim_verifier import Verdict, verify_claims
from app.agents.mta_v2.agent import ModelTrainingAgent

_BELOW = {
    "task": "classification", "primary_metric": "accuracy", "beats_baseline": False,
    "metrics": {"accuracy": 0.627066},
    "baseline": {"strategy": "most_frequent_training_class", "accuracy": 0.705},
    "splitter": "DeterministicHashHoldout",
}
_RESULT = {
    "status": "success", "mlflow_run_id": "run-1", "model_name": "my_model",
    "model_type": "classification", "final_accuracy": 0.627066, "final_f1_score": 0.632795,
}


def _report(evaluation):
    return ModelTrainingAgent()._format_training_result_report(
        {**_RESULT, "evaluation_report": evaluation}, use_local=True)


def test_a_below_baseline_model_is_reported_as_below_baseline() -> None:
    text = _report(_BELOW)
    assert "does not beat the trivial baseline" in text
    assert "0.627066 vs 0.705" in text
    assert "was successful" not in text
    assert "run a prediction" not in text


def test_a_model_that_beats_its_baseline_is_reported_as_before() -> None:
    text = _report({**_BELOW, "beats_baseline": True, "baseline": {"accuracy": 0.5}})
    assert "The training was successful" in text
    assert "run a prediction" in text


def test_no_evaluation_report_keeps_the_old_wording() -> None:
    text = _report(None)
    assert "The training was successful" in text


def test_the_verifier_rejects_success_language_for_a_below_baseline_model() -> None:
    report = verify_claims(
        "The training was successful, with an accuracy of 0.627. You can now run a prediction in chat.",
        {"evaluation_report": _BELOW},
    )
    assert not report.safe_to_present
    assert any(f.check == "contradicted_by_metrics" for f in report.findings)


_TABLE = {"execution_output": [
    {"region": "east", "avg_monthly_charges": 68.7788},
    {"region": "north", "avg_monthly_charges": 67.7303},
    {"region": "south", "avg_monthly_charges": 71.1677},
    {"region": "west", "avg_monthly_charges": 68.1178},
]}


def test_a_lowest_claim_the_table_contradicts_is_caught() -> None:
    report = verify_claims(
        "The south region has the highest average monthly charges at 71.17, "
        "while the west region has the lowest at 68.12.", _TABLE)
    assert not report.safe_to_present
    finding = next(f for f in report.findings if f.check == "ordering_contradicted")
    assert finding.verdict is Verdict.CONTRADICTED and "67.7303" in finding.detail


def test_a_correct_ordering_claim_passes() -> None:
    report = verify_claims(
        "The South region has the highest average monthly charges at about 71.17, "
        "while the North is the lowest at roughly 67.73.", _TABLE)
    assert report.safe_to_present
    assert not [f for f in report.findings if f.check == "ordering_contradicted"]


def test_a_superlative_with_no_figure_from_the_table_is_left_alone() -> None:
    report = verify_claims("Most customers sit in the east, and the lowest churn is in two-year contracts.", _TABLE)
    assert not [f for f in report.findings if f.check == "ordering_contradicted"]
