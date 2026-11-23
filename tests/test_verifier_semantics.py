"""Does the answer answer the QUESTION, not only match the code's output?

Each case is one the rc5 web re-check found: a figure that was in the result
and still answered a different question. Replies and code are stubbed; no LLM
runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.claim_verifier import (DataUsed, WITHHELD_ANSWER, claim_verifier_node, code_filters,
                                       condition_mismatch, presentable_reply, question_conditions,
                                       verify_claims, withheld_claims)
from app.api import chat_voice

WITHIN_30 = "What share of encounters were readmitted within 30 days?"


def _readmission_code(literal: str) -> str:
    return (
        "import pandas as pd\n"
        "def main(df):\n"
        f"    hit = df[df['readmitted'].astype(str).str.strip() == '{literal}']\n"
        "    share = len(hit) / len(df)\n"
        "    return pd.DataFrame({'share_of_readmissions_within_30_days': [share]})\n")


def _turn(question: str, reply: str, code: str, rows: list, **extra) -> dict:
    state = {"messages": [HumanMessage(content=question), AIMessage(content=reply)],
             "coder_definition": {"code": code},
             "execution_result": {"status": "success"}, "output_json": rows}
    state.update(extra)
    return state


# --------------------------------------------------------------------------- #
# (a) the filter answers another question
# --------------------------------------------------------------------------- #

def test_counting_over_30_when_asked_within_30_days_is_withheld_with_the_reason():
    """gemma, rc5: the code kept readmitted == '>30', the reply said 0.35, and
    "all of them hold". The truth is 0.1116."""
    reply = "The share of encounters readmitted within 30 days was approximately 0.35."
    out = claim_verifier_node(_turn(WITHIN_30, reply, _readmission_code(">30"),
                                    [{"share_of_readmissions_within_30_days": 0.3493}]))
    report = out["verification_report"]
    assert out["verification_safe_to_present"] is False and report["verified"] is False
    assert [f["check"] for f in report["findings"]] == ["condition_mismatch"]
    shown = presentable_reply(reply, report)
    assert "0.35" not in shown and shown.startswith(WITHHELD_ANSWER)
    assert "Why: The analysis counted readmitted '>30', but you asked about within 30 days." in shown


def test_the_conditions_are_read_from_the_question_and_the_code():
    assert question_conditions(WITHIN_30) == [("<", 30.0, "within 30 days")]
    assert question_conditions("loans with no more than 2 delinquencies") == [("<=", 2.0, "no more than 2 delinquencies")]
    code = "x = df.query(\"days > 30\")\ny = df[df.age.ge(65)]\nz = df['r'].isin(['<30'])"
    assert sorted(code_filters(code), key=str) == [("age", ">=", 65), ("days", ">", 30), ("r", "in", ("<30",))]


def test_a_numeric_filter_the_wrong_way_round_is_a_mismatch():
    code = "def main(df):\n    return df[df['days_to_return'] > 30]\n"
    assert condition_mismatch("How many orders were returned within 30 days?", code) == (
        "the analysis counted days_to_return > 30, but you asked about within 30 days")
    assert condition_mismatch("How many orders were returned within 30 days?",
                              code.replace("> 30", "<= 30")) is None


def test_a_different_category_from_the_one_asked_is_a_mismatch():
    used = DataUsed(frames=[("loans", pd.DataFrame({"loan_status": ["Charged Off", "Fully Paid", "Current"]}))])
    code = "def main(df):\n    return (df['loan_status'] == 'Fully Paid').mean()\n"
    assert condition_mismatch("What share of loans have loan_status Charged Off?", code, used) == (
        "the analysis counted loan_status = 'Fully Paid', but you asked about 'Charged Off'")
    right = code.replace("Fully Paid", "Charged Off")
    assert condition_mismatch("What share of loans have loan_status Charged Off?", right, used) is None


# --------------------------------------------------------------------------- #
# (b) no condition in the question, nothing flagged
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question", [
    "What drives readmission?",
    "How does readmission differ by age group?",
    "Summarise the readmitted column.",
])
def test_a_question_with_no_condition_is_never_flagged(question):
    reply = "Readmitted encounters make up approximately 0.35 of the total."
    out = claim_verifier_node(_turn(question, reply, _readmission_code(">30"),
                                    [{"share_of_readmissions_within_30_days": 0.35}]))
    assert out["verification_report"]["findings"] == []
    assert presentable_reply(reply, out["verification_report"]) == reply


# --------------------------------------------------------------------------- #
# (c) a wrong "strongest predictor"
# --------------------------------------------------------------------------- #

def _stroke_frame(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    age = rng.uniform(1, 85, n)
    married = np.where(rng.random(n) < 0.5 + age / 300, "Yes", "No")
    stroke = (rng.random(n) < 0.01 + (age / 85) ** 3 * 0.25).astype(int)
    return pd.DataFrame({"age": age.round(1), "ever_married": married,
                         "gender": rng.choice(["Male", "Female"], n), "stroke": stroke})


STROKE_IMPORTANCES = [{"feature": "ever_married_Yes", "importance": 1.2858},
                      {"feature": "gender_Female", "importance": 1.1773},
                      {"feature": "age", "importance": 0.0686}]


def test_a_wrong_strongest_predictor_is_withheld_and_the_computed_ranking_shown():
    """groq, rc5: "The strongest predictor of stroke is marital status (1.29)";
    age has the strongest association (r=0.245)."""
    used = DataUsed(frames=[("stroke", _stroke_frame())])
    reply = ("The strongest predictor of stroke is marital status (ever_married_Yes, importance 1.29). "
             "Age has very low importance (0.07). "
             "Gender is also included in the model.")
    report = verify_claims(reply, {"execution_output": STROKE_IMPORTANCES}, used,
                           question="what predicts a stroke?").as_dict()
    checks = {f["claim"]: f["check"] for f in report["findings"]}
    assert checks == {
        "The strongest predictor of stroke is marital status (ever_married_Yes, importance 1.29).":
            "driver_ranking_contradicted",
        "Age has very low importance (0.07).": "driver_ranking_contradicted"}
    shown = presentable_reply(reply, report)
    assert "1.29" not in shown and "very low importance" not in shown
    assert "Gender is also included in the model." in shown
    assert "age has the strongest association with stroke (r=" in shown
    assert "ranked by association: age (r=" in shown


def test_the_right_strongest_predictor_passes():
    used = DataUsed(frames=[("stroke", _stroke_frame())])
    reply = "Age is the strongest driver of stroke. Gender has little association with stroke."
    report = verify_claims(reply, {"execution_output": STROKE_IMPORTANCES}, used,
                           question="what predicts a stroke?")
    assert not report.findings


# --------------------------------------------------------------------------- #
# (d) a join said to match nothing
# --------------------------------------------------------------------------- #

def test_a_false_empty_join_claim_is_contradicted(tmp_path):
    """groq, rc5: "a join on this key would return an empty result", while the
    same join gives exact segment averages. No analysis ran on that turn."""
    customers, orders = tmp_path / "customers.csv", tmp_path / "orders.csv"
    pd.DataFrame({"customer_id": range(1, 51), "segment": ["smb", "consumer"] * 25}).to_csv(customers, index=False)
    pd.DataFrame({"order_id": range(100), "customer_id": [i % 50 + 1 for i in range(100)],
                  "amount": range(100)}).to_csv(orders, index=False)
    reply = ("The natural key is customers.customer_id and orders.customer_id. "
             "In the current sample no rows match, so a join on this key would return an empty result.")
    state = {"messages": [HumanMessage(content="How do my tables relate?"), AIMessage(content=reply)],
             "datasets_context": [{"alias": "customers", "data_source_location": str(customers)},
                                  {"alias": "orders", "data_source_location": str(orders)}]}
    out = claim_verifier_node(state)
    report = out["verification_report"]
    assert out["verification_safe_to_present"] is False
    assert [f["check"] for f in report["findings"]] == ["join_claim_contradicted"]
    shown = presentable_reply(reply, report)
    assert "empty result" not in shown
    assert "50 customer_id values appear in both customers (50 distinct) and orders (50 distinct)" in shown


def test_an_empty_join_claim_that_is_true_stands(tmp_path):
    customers, orders = tmp_path / "customers.csv", tmp_path / "orders.csv"
    pd.DataFrame({"customer_id": range(1, 51)}).to_csv(customers, index=False)
    pd.DataFrame({"customer_id": range(900, 950)}).to_csv(orders, index=False)
    reply = "No customer_id values match, so a join on customer_id would return an empty result."
    state = {"messages": [AIMessage(content=reply)],
             "datasets_context": [{"alias": "customers", "data_source_location": str(customers)},
                                  {"alias": "orders", "data_source_location": str(orders)}]}
    assert claim_verifier_node(state)["verification_report"]["findings"] == []


# --------------------------------------------------------------------------- #
# (e) a model metric quoted when CV does not beat the baseline
# --------------------------------------------------------------------------- #

STROKE_EVALUATION = {
    "metrics": {"accuracy": 0.963027}, "baseline": {"accuracy": 0.957265}, "primary_metric": "accuracy",
    "beats_baseline": True,
    "cross_validation": {"n_folds": 5, "primary_metric": "accuracy",
                         "metrics": {"accuracy": {"mean": 0.947745, "std": 0.002392}},
                         "baseline": {"accuracy": 0.949717}, "beats_baseline": False, "cv_mean": 0.947745},
}


def test_a_cv_result_that_does_not_beat_the_baseline_gets_the_warning():
    reply = ("* **Final Accuracy:** 0.963027\n"
             "* **Cross-validation:** accuracy 0.9477 ± 0.0024 over 5-fold StratifiedKFold")
    out = claim_verifier_node({"messages": [AIMessage(content=reply)], "evaluation_report": STROKE_EVALUATION})
    shown = presentable_reply(reply, out["verification_report"])
    assert shown.startswith(reply)
    assert shown.endswith("The model does not reliably beat the trivial baseline: across 5-fold "
                          "cross-validation its accuracy is 0.9477, against 0.9497 for the trivial baseline.")


def test_a_reply_that_already_says_so_is_left_alone():
    reply = "Accuracy is 0.963, but across folds it does not beat the trivial baseline."
    out = claim_verifier_node({"messages": [AIMessage(content=reply)], "evaluation_report": STROKE_EVALUATION})
    assert presentable_reply(reply, out["verification_report"]) == reply


def test_the_chat_voice_does_not_say_beats_when_the_folds_do_not():
    final = {"training_plan": {"model_name": "stroke", "data_config": {"target_column": "stroke"}},
             "training_completed": True, "evaluation_beats_baseline": True,
             "evaluation_report": STROKE_EVALUATION}
    ex = next(p for p in chat_voice.prompts_after_turn(final, question="train") if p["step"] == "execution")
    assert "by more than fold noise" not in ex["message"]
    assert "does not reliably beat the trivial baseline" in ex["message"]


# --------------------------------------------------------------------------- #
# (f) a correct reply is unchanged
# --------------------------------------------------------------------------- #

def test_a_correct_within_30_days_answer_is_unchanged():
    """groq, rc5: '<30' counted, 0.1116 stated: right, and shown as written."""
    reply = "The share of encounters that were readmitted within 30 days is **0.1116** (about 11.2%)."
    out = claim_verifier_node(_turn(WITHIN_30, reply, _readmission_code("<30"),
                                    [{"share_of_readmissions_within_30_days": 0.111599}]))
    assert out["verification_report"]["verified"] is True
    assert not withheld_claims(out["verification_report"])
    assert presentable_reply(reply, out["verification_report"]) == reply


def test_a_beaten_baseline_quoted_needs_no_warning():
    evaluation = dict(STROKE_EVALUATION, cross_validation=dict(STROKE_EVALUATION["cross_validation"],
                                                               beats_baseline=True))
    reply = "* **Final Accuracy:** 0.963027"
    out = claim_verifier_node({"messages": [AIMessage(content=reply)], "evaluation_report": evaluation})
    assert presentable_reply(reply, out["verification_report"]) == reply
