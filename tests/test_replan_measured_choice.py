"""Missing values in the measured column: the loop asks, and the next turn resumes.

The owner's ruling: when the column a calculation sums or averages has
missing values, the replan loop neither fills them nor leaves them out on its
own. It asks, with code-written counts; "median" fills them and says what that
does to the average; "leave them out" leaves them out and says how many; a run
no one can answer (a scheduled run) ends with the question and nothing
changed. The pause and resume reuse the planner's pending-question pattern
(pending_clarification, as the infrastructure confirmation does).

Turn one goes through the graph build_graph() registers from the coder's
hand-over (as in tests/test_replan_on_execution_error.py). Turn two goes in
the way the server sends it: the reply, the persisted pending question, into
memory_injection's successor, so the real avaloka_agent, plan_etl, routers,
executor, replan node and verifier run. No model is called.

On the tree before the ruling (1d87f175) the loop left the rows out on its
own, so every test here fails there at its first assertion: no question.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tests.test_replan_on_execution_error import (  # noqa: F401  (graph is a fixture)
    FIXTURES, TOTAL_CODE, TRACEBACK_MARKERS, _reply, _rows, _run, _run_truth, graph,
)

QUESTION = ("12 of 300 rows have no value in revenue, the column this calculation measures. "
            "Should I fill them with the median, or leave them out?")


def _ask(graph, tmp_path):
    final, data = _run(graph, tmp_path, "revenue.csv", TOTAL_CODE, "Total revenue?")
    reply = _reply(final)
    assert QUESTION in reply, reply
    return final, data


def _answer(graph, tmp_path, first, data, answer: str):
    """Turn two, as the server sends it: the reply and the persisted pending question."""
    _, compiled = graph
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    compiled.update_state(config, {
        "messages": [HumanMessage(content="Total revenue?"), HumanMessage(content=answer)],
        "user_prompt": answer,
        "pending_clarification": first.get("pending_clarification"),
        "data_source_location": str(data),
        "output_location": str(tmp_path / "out2" / "output.csv"),
        "execution_mode": "local",
        "input_data_type": "csv",
        "agent_errors": [],
        "coder_definition": {},
    }, as_node="memory_injection")
    compiled.invoke(None, config)
    return compiled.get_state(config).values


def test_the_question_is_asked_with_exact_counts_and_nothing_is_imputed(graph, tmp_path):
    final, data = _ask(graph, tmp_path)
    reply = _reply(final)
    assert "Nothing has been filled or left out yet" in reply
    for marker in TRACEBACK_MARKERS:
        assert marker not in reply
    assert not final.get("replan_transforms"), "something was repaired before the user answered"
    assert not final.get("replan_input_overrides")
    assert not list((tmp_path / "out").glob(".replan/*")), "a repaired copy was written before the answer"
    assert final["execution_result"]["status"] == "failed"
    pending = final.get("pending_clarification") or {}
    assert pending.get("type") == "replan_measured_choice"
    assert (pending["column"], pending["rows_affected"], pending["rows_total"]) == ("revenue", 12, 300)
    assert pending["code"] == TOTAL_CODE


def test_answer_median_fills_and_states_the_bias(graph, tmp_path):
    first, data = _ask(graph, tmp_path)
    final = _answer(graph, tmp_path, first, data, "median")
    raw = pd.read_csv(data)
    median = float(raw["revenue"].median())
    reply = _reply(final)
    assert (f"12 of 300 rows had no value in revenue; I filled them with the median ({median:,.2f}), "
            f"as you chose. This pulls the average of revenue toward {median:,.2f}.") in reply, reply
    assert [int(float(r["total_revenue"])) for r in _rows(final)] == [
        int(raw["revenue"].fillna(median).astype(int).sum())]
    assert not final.get("pending_clarification")


def test_answer_leave_out_excludes_and_discloses(graph, tmp_path):
    first, data = _ask(graph, tmp_path)
    final = _answer(graph, tmp_path, first, data, "leave them out")
    raw = pd.read_csv(data)
    reply = _reply(final)
    assert ("I left out 12 of 300 rows (4.0%) that had no value in revenue, as you chose; "
            "the result is computed over the other 288.") in reply, reply
    assert [int(float(r["total_revenue"])) for r in _rows(final)] == [int(raw["revenue"].dropna().sum())]


def test_run_truth_records_the_question_and_the_choice(graph, tmp_path):
    first, data = _ask(graph, tmp_path)
    asked = _run_truth(first)
    assert (asked["outcome"], asked["reason_code"]) == ("failed", "needs_decision")
    assert asked["recovery"]["stopped"]["question"] == QUESTION
    assert asked["recovery"]["stopped"]["choice"] == "no_answer"
    assert asked["recovery"]["attempts"] == []

    final = _answer(graph, tmp_path, first, data, "median")
    truth = _run_truth(final)
    assert truth["outcome"] == "succeeded"
    [attempt] = truth["recovery"]["attempts"]
    assert attempt["decision"] == {"question": QUESTION, "column": "revenue", "choice": "median",
                                   "chosen_by": "user"}
    [transform] = attempt["transforms"]
    assert (transform["op"], transform["rows_affected"], transform["chosen_by"]) == ("impute", 12, "user")


def test_a_stale_resume_is_refused_and_nothing_is_changed(graph, tmp_path):
    first, data = _ask(graph, tmp_path)
    # The dataset changed between the question and the answer.
    with open(data, "a") as handle:
        handle.write("north,5,\n")
    final = _answer(graph, tmp_path, first, data, "median")
    reply = _reply(final)
    assert "The data has changed since I asked, so I haven't applied that choice" in reply, reply
    assert not final.get("replan_transforms") and not final.get("replan_attempts")
    assert not (final.get("execution_result") or {}).get("status"), "the run was resumed anyway"


def test_a_resume_on_another_dataset_is_refused(graph, tmp_path):
    first, data = _ask(graph, tmp_path)
    other = tmp_path / "other.csv"
    other.write_bytes(data.read_bytes())
    final = _answer(graph, tmp_path, first, other, "leave them out")
    assert "You've switched datasets since I asked" in _reply(final)
    assert not final.get("replan_transforms")


def test_a_scheduled_run_does_not_guess_and_ends_with_the_question(tmp_path, monkeypatch):
    from app.core import celery_app

    monkeypatch.delenv("AVALOKA_REPLAN_MEASURED_CHOICE", raising=False)
    monkeypatch.delenv("AVALOKA_REPLAN_MEASURED_POLICY", raising=False)
    # The worker's own executor; the default scheduled executor is the pod.
    monkeypatch.setenv("AVALOKA_SCHEDULED_EXECUTOR", "local")
    data = tmp_path / "revenue.csv"
    data.write_bytes((FIXTURES / "revenue.csv").read_bytes())
    final = celery_app._run_agent_task({
        "messages": [{"type": "human", "content": "Total revenue?"}],
        "coder_definition": {"code": TOTAL_CODE},
        "data_source_location": str(data),
        "output_location": str(tmp_path / "out" / "output.csv"),
        "execution_mode": "local",
        "task_schedule": {"task_type": "execute"},
    })
    texts = [m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
             for m in final.get("messages") or []]
    reply = texts[-1]
    assert QUESTION in reply, reply
    assert "This run could not wait for an answer, so I haven't filled or left out anything" in reply
    assert "AVALOKA_REPLAN_MEASURED_CHOICE=median" in reply
    assert not final.get("replan_transforms")
    assert final["execution_result"]["status"] == "failed"
    for marker in TRACEBACK_MARKERS:
        assert marker not in reply


def test_a_scheduled_run_applies_an_answer_given_up_front(tmp_path, monkeypatch):
    from app.core import celery_app

    monkeypatch.setenv("AVALOKA_REPLAN_MEASURED_CHOICE", "leave_out")
    monkeypatch.setenv("AVALOKA_SCHEDULED_EXECUTOR", "local")
    data = tmp_path / "revenue.csv"
    data.write_bytes((FIXTURES / "revenue.csv").read_bytes())
    final = celery_app._run_agent_task({
        "messages": [{"type": "human", "content": "Total revenue?"}],
        "coder_definition": {"code": TOTAL_CODE},
        "data_source_location": str(data),
        "output_location": str(tmp_path / "out" / "output.csv"),
        "execution_mode": "local",
        "task_schedule": {"task_type": "execute"},
    })
    assert final["execution_result"]["status"] in ("success", "succeeded"), final.get("execution_result")
    [t] = final["replan_transforms"]
    assert (t["op"], t["rows_affected"], t["chosen_by"]) == ("exclude_rows", 12, "env")
