"""scheduler._run_now_inline presents a table only for a run that succeeded.

Carried over from the declined #617's review (note 1): a Ray run never sets
output_location itself, so on a failed or contract-refused run the value in the
result is whatever the session carried in (an earlier turn's file), and the chat
would show it as this question's answer.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

scheduler = pytest.importorskip("app.agents.scheduler")
celery_app = pytest.importorskip("app.core.celery_app")

STALE = "s3://bucket/outputs/earlier-turn.csv"


def _state() -> dict:
    return {"messages": [HumanMessage(content="lowest price per brand")], "output_location": STALE,
            "output_file_data": None, "task_schedule": {"task_type": "execute"}}


def _run(monkeypatch, result: dict) -> dict:
    monkeypatch.setattr(celery_app, "agent_task", lambda _state: result)
    return scheduler._run_now_inline(_state())


@pytest.mark.parametrize("execution_result", [
    {"status": "refused", "output_contract": {"status": "refused", "reason": "no price column"}},
    {"status": "error", "error": "Ray cluster unreachable"},
], ids=["refused", "failed"])
def test_a_failed_or_refused_run_shows_no_table(monkeypatch, execution_result):
    out = _run(monkeypatch, {
        "messages": [{"role": "ai", "content": "I couldn't answer that."}],
        "output_location": STALE, "output_file_data": None, "execution_result": execution_result,
    })
    assert out["output_location"] is None and out["output_file_data"] is None
    assert out["messages"][-1].content == "I couldn't answer that."


def test_a_successful_run_keeps_its_output(monkeypatch):
    table = {"rows": [{"brand": "a", "min_price": 1.0}]}
    out = _run(monkeypatch, {
        "messages": [{"role": "ai", "content": "Here it is."}],
        "output_location": "s3://bucket/outputs/this-turn.csv", "output_file_data": table,
        "execution_result": {"status": "success"},
    })
    assert out["output_file_data"] == table
    assert out["output_location"] == "s3://bucket/outputs/this-turn.csv"
