"""Avaloka talks the user through each step, grounded in measured statistics.

Pure tests of app/api/chat_voice.py: the statistics shape is the one
sampling_async writes (type, missing_ratio, n_unique, top_values, skewness),
and the finished state is the shape the trace tests drive the graph with.
"""

from __future__ import annotations

import pytest

from app.api import chat_voice
from app.api.ui_contract import PROMPT_STEPS, api_capabilities, assistant_prompt


def _stats(rows=55_500):
    def cat(values, total=rows):
        top = [{"value": v, "count": c} for v, c in values]
        return {"type": "categorical", "null_count": 0, "missing_ratio": 0.0, "total_count": total,
                "n_unique": len(values), "top_values": top}

    def num(mean, mn, mx, skew=0.0, missing=0.0):
        return {"type": "numeric", "null_count": int(rows * missing), "missing_ratio": missing,
                "total_count": rows, "n_unique": 1000, "mean": mean, "min": mn, "max": mx, "skewness": skew}

    return {
        "data_shape": {"rows": rows, "columns": 8},
        "column_statistics": {
            "Name": {"type": "categorical", "null_count": 0, "missing_ratio": 0.0, "total_count": rows,
                     "n_unique": rows - 12, "top_values": [{"value": "Bobby JacksOn", "count": 2}]},
            "Age": num(51.5, 13, 89),
            "Gender": cat([("Male", 27_800), ("Female", 27_700)]),
            "Medical Condition": cat([("Arthritis", 9_300), ("Diabetes", 9_300), ("Hypertension", 9_200),
                                      ("Obesity", 9_200), ("Cancer", 9_200), ("Asthma", 9_200)]),
            "Date of Admission": {"type": "categorical", "null_count": 0, "missing_ratio": 0.0,
                                  "total_count": rows, "n_unique": 1800, "top_values": []},
            "Billing Amount": num(25_500.0, -2008.0, 52_764.0, skew=0.02, missing=0.12),
            "Admission Type": cat([("Elective", 18_650), ("Urgent", 18_550), ("Emergency", 18_300)]),
            "Test Results": cat([("Abnormal", 18_600), ("Normal", 18_500), ("Inconclusive", 18_400)]),
        },
        "data_quality": {},
    }


PROFILE = {"domain": {"category": "Healthcare", "confidence": 0.9},
           "predictive_values": {"Medical Condition": "high", "Test Results": "high"}}


# ------------------------------------------------------------------ the read
def test_the_read_classifies_columns_from_the_statistics_alone():
    read = chat_voice.read_statistics(_stats(), PROFILE)
    assert read.n_rows == 55_500 and read.n_cols == 8 and read.domain == "Healthcare"
    assert "Name" in read.id_columns                      # one value per row
    assert "Date of Admission" in read.time_columns       # the name says so
    assert "Age" in read.numeric and "Billing Amount" in read.numeric
    assert read.missing == [("Billing Amount", 0.12)]
    assert read.target == "Test Results"                  # outcome-named, low cardinality, not dominated
    assert read.segment[0] in ("Gender", "Admission Type") and read.segment[2] > 0.3


def test_the_read_survives_an_empty_or_odd_statistics_document():
    assert chat_voice.read_statistics(None).columns == []
    assert chat_voice.read_statistics({"column_statistics": "nope"}).n_cols is None
    read = chat_voice.read_statistics({"column_statistics": {"x": {"error": "boom"}}}, schema=["x"])
    assert read.columns == ["x"] and read.numeric == [] and read.categorical == []


# ---------------------------------------------------------------- the upload
def test_after_upload_she_names_what_she_saw_and_asks_about_the_outcome():
    prompts = chat_voice.prompts_after_upload(sample_statistics=_stats(), profiling_result=PROFILE,
                                              filename="healthcare_dataset.csv")
    assert len(prompts) == 1
    p = prompts[0]
    assert p["step"] == "auto_insights"
    assert "55,500 rows across 8 columns" in p["message"] and "healthcare data" in p["message"]
    assert "`Billing Amount` is 12% empty" in p["message"]
    assert "`Test Results`" in p["question"]
    labels = [r["label"] for r in p["quick_replies"]]
    assert "What drives Test Results?" in labels and "Trend over Date of Admission" in labels
    assert 2 <= len(p["message"].split(". ")) <= 4, p["message"]  # succinct: two or three sentences
    assert len(p["quick_replies"]) <= chat_voice.MAX_REPLIES


def test_after_upload_without_statistics_she_says_nothing_rather_than_something_generic():
    assert chat_voice.prompts_after_upload(sample_statistics=None) == []
    assert chat_voice.prompts_after_upload(sample_statistics={}) == []


def test_upload_questions_are_grounded_not_generic():
    """Two different datasets get two different questions."""
    a = chat_voice.prompts_after_upload(sample_statistics=_stats(), profiling_result=PROFILE)[0]
    stats_b = {"data_shape": {"rows": 300, "columns": 2}, "column_statistics": {
        "revenue": {"type": "numeric", "missing_ratio": 0.0, "total_count": 300, "n_unique": 300, "mean": 5.0},
        "cost": {"type": "numeric", "missing_ratio": 0.0, "total_count": 300, "n_unique": 290, "mean": 2.0}}}
    b = chat_voice.prompts_after_upload(sample_statistics=stats_b)[0]
    assert a["question"] != b["question"]
    assert "Explain revenue" in [r["label"] for r in b["quick_replies"]]


# ------------------------------------------------------------------ the turn
FINAL = {
    "avaloka_intent": "statistical_analysis",
    "planner_definition": {"transformations": [{"operation": "group_by", "column": "Admission Type"},
                                               {"operation": "mean", "column": "Billing Amount"}]},
    "execution_result": {"status": "success", "message": "done", "output_row_count": 3},
    "output_json": [{"Admission Type": "Elective", "avg": 25_400.1}] * 3,
    "verification_report": {"safe_to_present": True, "claims_checked": 2, "n_problems": 0, "findings": []},
}
CTX = {"mode": "portfolio_samples", "mode_label": "Portfolio Samples", "rows_analyzed": 7_013,
       "total_rows": 55_500, "is_sampled": True}


def test_a_turn_yields_one_prompt_per_step_in_order_each_with_one_question():
    prompts = chat_voice.prompts_after_turn(
        FINAL, question="What is the average Billing Amount by Admission Type?",
        execution_context=CTX, sample_statistics=_stats(), profiling_result=PROFILE)
    steps = [p["step"] for p in prompts]
    assert steps == ["understanding", "sampling", "plan", "execution", "verification", "result"]
    assert [s for s in PROMPT_STEPS if s in steps] == steps  # contract order
    for p in prompts:
        assert p["question"].endswith("?") and 1 <= len(p["quick_replies"]) <= chat_voice.MAX_REPLIES
        assert len(p["message"]) < 320, p["message"]  # succinct


def test_the_steps_are_grounded_in_the_question_and_the_statistics():
    prompts = {p["step"]: p for p in chat_voice.prompts_after_turn(
        FINAL, question="What is the average Billing Amount by Admission Type?",
        execution_context=CTX, sample_statistics=_stats(), profiling_result=PROFILE)}
    assert "`Billing Amount`" in prompts["understanding"]["message"]           # the columns she touched
    assert "7,013 of 55,500 rows" in prompts["sampling"]["message"]            # the real sample
    assert prompts["sampling"]["quick_replies"][0]["send"] == "Use the entire dataset"
    assert "2 steps" in prompts["plan"]["message"] and "group by" in prompts["plan"]["message"]
    assert "3 rows" in prompts["execution"]["message"]
    assert prompts["verification"]["message"] == "I checked 2 claims in my answer against the numbers."
    # The result step offers what the question did not already cover, and the team.
    labels = [r["label"] for r in prompts["result"]["quick_replies"]]
    assert "Train a model on Test Results" in labels and "Send my team at it" in labels
    swarm = next(r for r in prompts["result"]["quick_replies"] if r["label"] == "Send my team at it")
    assert swarm["send"].startswith("Swarm this: What is the average")


def test_a_chitchat_turn_says_nothing_and_a_failed_run_says_it_failed():
    assert chat_voice.prompts_after_turn({}, question="hi there", sample_statistics=_stats()) == []
    failed = {"execution_result": {"status": "failed", "message": "KeyError: 'Billng Amount'\nTraceback..."}}
    prompts = chat_voice.prompts_after_turn(failed, question="avg billing", sample_statistics=_stats())
    ex = next(p for p in prompts if p["step"] == "execution")
    # Plain words, like the reply (#522's failed-step language): what went
    # wrong for her, not the exception class or the traceback.
    assert ex["message"] == "I couldn't finish the analysis: the code asked for a column the data doesn't have."
    assert "KeyError" not in ex["message"] and "Traceback" not in ex["message"]
    assert ex["quick_replies"][0]["send"] == "Retry"
    # A failed run is not an answer, and nothing about it was verified.
    steps = {p["step"]: p for p in prompts}
    assert "verification" not in steps
    assert "That's my answer" not in steps["result"]["message"]


def test_a_flagged_verification_asks_to_redo_the_statement():
    final = {"verification_report": {"safe_to_present": False, "claims_checked": 3, "n_problems": 1}}
    v = chat_voice.prompts_after_turn(final, question="q")[0]
    assert v["step"] == "verification" and "1 did not hold" in v["message"]
    assert v["quick_replies"][0]["send"] == "Redo it"


def test_a_clarification_turn_ends_on_its_question_with_nothing_to_verify():
    """The conversational agent offered options and ran nothing: she says how
    she read the ask and stops. No 'nothing to verify', no 'that's my answer'."""
    final = {"avaloka_intent": "clarification",
             "pending_suggestions": ["Average Billing Amount by Test Results", "Test Results by Admission Type"],
             "verification_report": {"safe_to_present": True, "claims_checked": 0, "n_problems": 0}}
    prompts = chat_voice.prompts_after_turn(final, question="What drives Test Results?",
                                            sample_statistics=_stats(), profiling_result=PROFILE)
    assert [p["step"] for p in prompts] == ["understanding"]
    assert "pin down before I run anything" in prompts[0]["message"]
    assert "`Test Results`" in prompts[0]["message"]


def test_a_training_turn_is_honest_about_the_baseline():
    final = {"training_plan": {"model_name": "churn_predictor", "data_config": {"target_column": "churned"}},
             "training_completed": True, "evaluation_beats_baseline": False}
    ex = next(p for p in chat_voice.prompts_after_turn(final, question="train") if p["step"] == "execution")
    assert "`churned`" in ex["message"] and "does not beat the trivial baseline" in ex["message"]


# ------------------------------------------------------------- the contract
def test_the_contract_shape_and_the_capability_flag():
    p = assistant_prompt("result", " hello ", question="why? ", quick_replies=["A", {"label": "B", "send": "do b"}],
                         link={"label": "x", "href": "/swarm?mission=1"})
    assert p == {"step": "result", "message": "hello", "question": "why?",
                 "message_segments": [{"text": "hello", "kind": "plain"}],
                 "quick_replies": [{"label": "A", "send": "A"}, {"label": "B", "send": "do b"}],
                 "link": {"label": "x", "href": "/swarm?mission=1"}}
    with pytest.raises(ValueError):
        assistant_prompt("nope", "x")
    caps = api_capabilities()["features"]["assistant_prompts"]
    assert caps["enabled"] is True and caps["optional"] is True and caps["steps"] == list(PROMPT_STEPS)


def test_a_numbered_pick_and_a_plan_read_as_the_job_not_the_number():
    """"1" picked from her list; the planner ran Correlation Analysis; the coder's
    result failed her checks. She says the job, not "I read '1' as…"."""
    final = {"avaloka_intent": "clarification",  # stale from the turn before
             "planner_definition": {"transformations": [{"operation": "Correlation Analysis"}]},
             "contract_violations": ["'Date of Admission' was declared datetime but holds object",
                                     "'Discharge Date' was declared datetime but holds object"]}
    prompts = {p["step"]: p for p in chat_voice.prompts_after_turn(
        final, question="1", sample_statistics=_stats(), profiling_result=PROFILE)}
    assert "I read “option 1” as a job for my planner." in prompts["understanding"]["message"]
    # The step's name is the planner model's: the plan step says it, marked (c-strict).
    assert "My planner named it: Correlation Analysis. (commentary — not checked)" in prompts["plan"]["message"]
    assert "didn't pass my own checks" in prompts["execution"]["message"]
    assert "Date of Admission" in prompts["execution"]["message"]
    assert prompts["execution"]["quick_replies"][0]["send"] == "Try again"
    assert "result" in prompts  # something was attempted, so she offers a next step
    # ...but the code never ran, so it is not "my answer".
    assert prompts["result"]["message"].startswith("I don't have an answer to")
