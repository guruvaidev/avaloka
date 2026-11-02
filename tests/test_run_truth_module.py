"""app/core/run_truth.py on its own: the closed vocabulary and the classifiers.

New module in item 8a, so these error on the base with ModuleNotFoundError by
design; tests/test_run_truth_8a.py holds the bite tests that fail there on
assertions.
"""
import pytest

from app.core import run_truth as rt


def test_every_reason_code_has_a_template_and_one_outcome():
    assert set(rt.REASON_TEMPLATES) == set(rt.OUTCOME_OF_REASON)
    assert set(rt.OUTCOME_OF_REASON.values()) == set(rt.OUTCOMES)


def test_an_unknown_reason_code_is_rejected():
    with pytest.raises(KeyError):
        rt.make("looked_fine", kind="scheduled")
    with pytest.raises(ValueError):
        rt.RunOutcome(outcome="succeeded", reason_code="looked_fine", reason="", kind="scheduled")


def test_a_reason_cannot_end_in_the_wrong_outcome():
    with pytest.raises(ValueError):
        rt.RunOutcome(outcome="succeeded", reason_code="nothing_ran", reason="", kind="scheduled")


def test_scope_rejects_unknown_engines_and_kinds():
    with pytest.raises(ValueError):
        rt.build_scope(engine="gpu-farm")
    with pytest.raises(ValueError):
        rt.build_scope(rows_total=1, rows_total_kind="guess")


def test_unknown_rows_stay_none_never_zero():
    scope = rt.build_scope()
    assert scope["rows_processed"] is None and scope["rows_total"] is None
    assert scope["rows_total_kind"] == "unknown"


@pytest.mark.parametrize("final,code", [
    (None, "nothing_ran"),
    ({}, "nothing_ran"),
    ({"execution_result": {"status": "dryrun"}}, "nothing_ran"),
    ({"execution_result": {"status": "success"}}, "ok"),
    ({"execution_result": {"status": "success"}, "execution_error": "x"}, "exception"),
    ({"execution_result": {"status": "success", "ray_fallback": {"reason": "no connection", "rows_used": 9}}}, "fallback"),
    # MAJ-030: a result the output contract refused (app/agents/output_contract.py).
    ({"execution_result": {"status": "refused", "output_contract": {
        "status": "refused", "violations": [{"rule": "missing_measure", "detail": "no 'price' column"}]}}},
     "output_contract"),
])
def test_classify_scheduled(final, code):
    truth = rt.classify_scheduled(final, task_type="execute")
    assert truth["reason_code"] == code
    assert truth["outcome"] == rt.OUTCOME_OF_REASON[code]


def test_a_turn_with_missing_metrics_is_partial():
    final = {"execution_result": {"status": "success"}, "structured_action_plan": True,
             "multi_action_missing": [{"id": "m1", "label": "median of x"}]}
    truth = rt.classify_turn(final)
    assert truth["outcome"] == "partial"
    assert truth["reason_code"] == "partial_metrics"
    assert "median of x" in truth["reason"]


def test_a_failed_training_turn_is_failed():
    truth = rt.classify_turn({"training_result": {"status": "error", "error": "OOM"}}, deferred=True)
    assert truth["outcome"] == "failed" and truth["kind"] == "deferred"


def test_legacy_status_keeps_the_ui_words():
    assert rt.legacy_status("failed") == "failure"
    assert rt.legacy_status("succeeded") == "success"
    assert rt.legacy_status("partial") == "success"
