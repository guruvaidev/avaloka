"""A scheduling request typed in chat must reach the planner's schedule_task tool.

Seen on develop-1.6 (kind deployment, 2026-09-25): "Calculate the total amount
per category, and schedule this to run every Sunday at 10 pm" came back in 13 s
with no assistant message and no task. The conversational agent matched the
keyword "schedule", pre-set a run-once task_schedule without reading the
sentence, and the planner -- seeing a schedule already present -- skipped its
LLM; the router then found nothing to do and the graph ended silently.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.avaloka_agent.agent as agent_mod
import app.agents.planner as planner_mod
from app.agents.avaloka_agent.intent_classifier import IntentResult, classify_intent

SUNDAY = "Calculate the total amount per category, and schedule this to run every Sunday at 10 pm"
MONDAY = "Every Monday 9am recompute fraud rate by ProductCD"   # docs/USER_GUIDE.md §9 wording

STRAY_RUN_ONCE = {"task_type": "execute", "schedule_type": "relative", "second": 0, "max_runs": 1}


# --------------------------------------------------------------------------- #
# The conversational agent must not invent a schedule
# --------------------------------------------------------------------------- #

def test_the_keyword_classifier_still_routes_schedule_requests_to_the_planner():
    intent = classify_intent([HumanMessage(content=SUNDAY)])
    assert intent.intent == "schedule" and intent.should_delegate()


def test_preconfigure_does_not_fabricate_a_task_schedule():
    updates = agent_mod._preconfigure_for_planner(
        {"messages": [HumanMessage(content=SUNDAY)]},
        IntentResult(intent="schedule", confidence=0.6),
    )
    assert "task_schedule" not in updates


def test_the_agent_node_delegates_a_schedule_request_without_a_schedule(monkeypatch):
    monkeypatch.setattr(agent_mod, "_load_redis_context", lambda _s: {})
    monkeypatch.setattr(agent_mod, "_run_discovery", lambda *_a, **_k: None)
    monkeypatch.setattr(agent_mod, "classify_intent",
                        lambda *_a, **_k: IntentResult(intent="schedule", confidence=0.6))
    updates = agent_mod.avaloka_agent_node({"messages": [HumanMessage(content=SUNDAY)]})
    assert updates["delegate_to_planner"] is True
    assert updates.get("task_schedule") in (None, {})


# --------------------------------------------------------------------------- #
# ...and the planner must still ask its LLM even if a stray schedule is present
# --------------------------------------------------------------------------- #

class _FakeToolCall:
    def __init__(self, name, args):
        self.tool_calls = [{"name": name, "args": args}]


class _FakeLLM:
    def __init__(self, name, args):
        self.calls, self._resp = 0, _FakeToolCall(name, args)

    def invoke(self, *_a, **_k):
        self.calls += 1
        return self._resp


CRON_ARGS = {"task_type": "execute", "schedule_type": "repetitive", "day_of_week": 0,
             "hour": 22, "minute": 0, "max_runs": -1}


def _plan(monkeypatch, message, extra):
    llm = _FakeLLM("schedule_task", CRON_ARGS)
    monkeypatch.setattr(planner_mod, "llm", llm)
    import app.agents.scheduler as sched
    monkeypatch.setattr(sched, "is_celery_worker_running", lambda: True)
    state = {"messages": [HumanMessage(content=message)], "user_id": "u", "session_id": "s"}
    state.update(extra)
    return llm, planner_mod.plan_etl_job(state)


VAGUE = "Calculate the total amount per category, and schedule this to run on the usual cadence"


def test_a_stray_run_once_schedule_no_longer_mutes_the_planner(monkeypatch):
    """What the old conversational agent injected must not short-circuit planning.

    VAGUE names no cadence the parser can turn into a crontab, so the LLM is
    the one that has to be asked.
    """
    llm, out = _plan(monkeypatch, VAGUE, {"task_schedule": dict(STRAY_RUN_ONCE)})
    assert llm.calls >= 1, "the planner LLM was bypassed"
    ts = out.get("task_schedule") or {}
    assert ts.get("schedule_type") == "repetitive" and ts.get("day_of_week") == 0 and ts.get("hour") == 22
    assert out.get("ready_to_code") is True


def test_a_schedule_that_arrived_with_a_plan_still_bypasses(monkeypatch):
    """The deferral path sets ready_to_code together with the schedule; keep that fast path."""
    llm, out = _plan(monkeypatch, SUNDAY, {"task_schedule": dict(CRON_ARGS), "ready_to_code": True,
                                           "plan": "already planned", "coder_definition": {"code": "x"}})
    assert llm.calls == 0
    assert out.get("task_schedule") == CRON_ARGS


def test_the_user_guide_wording_produces_a_cron_without_asking_the_llm(monkeypatch):
    llm, out = _plan(monkeypatch, MONDAY, {"plan": "stale plan from an earlier turn"})
    assert llm.calls == 0, "a weekday-and-time cadence is parsed, not guessed"
    ts = out.get("task_schedule") or {}
    assert (ts.get("schedule_type"), ts.get("day_of_week"), ts.get("hour"), ts.get("minute"), ts.get("max_runs")) \
        == ("repetitive", 1, 9, 0, -1)
    assert out.get("ready_to_code") is True
    assert out["user_prompt"].startswith("recompute fraud rate by ProductCD")
    assert "do not implement timing" in out["user_prompt"]
    assert out.get("plan") is None, "the earlier turn's plan would outrank the request in the coder"


def test_the_llm_path_also_hands_the_coder_a_cadence_free_request(monkeypatch):
    """When the parser is not concrete and the LLM calls schedule_task."""
    llm, out = _plan(monkeypatch, VAGUE, {})
    assert llm.calls >= 1
    assert out["user_prompt"].startswith("Calculate the total amount per category")


# --------------------------------------------------------------------------- #
# The deterministic cadence parser
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    (MONDAY, (1, 9, 0)),
    (SUNDAY, (0, 22, 0)),
    ("Refresh the dashboard numbers every weekday at 8:30am", ("1-5", 8, 30)),
    ("recompute totals daily at 6am", ("*", 6, 0)),
    ("every day at 18:15 refresh the summary", ("*", 18, 15)),
    ("10 pm every Sunday recompute the totals", (0, 22, 0)),
    ("Every Friday recompute the totals", (5, 0, 0)),
    ("each night at 11 rebuild the report", ("*", 23, 0)),
    ("Every Monday 12am recompute the totals", (1, 0, 0)),
])
def test_calendar_cadences_become_crontab_fields(text, expected):
    schedule, concrete = planner_mod._parse_task_schedule_from_text(text.lower())
    assert concrete is True
    assert (schedule["day_of_week"], schedule["hour"], schedule["minute"]) == expected
    assert schedule["schedule_type"] == "repetitive" and schedule["max_runs"] == -1
    assert schedule["task_type"] == "execute"


def test_every_n_minutes_still_parses_as_before():
    schedule, concrete = planner_mod._parse_task_schedule_from_text(
        "recompute the fraud rate by productcd every 2 minutes")
    assert concrete and schedule["minute"] == "*/2" and schedule["hour"] == "*"


@pytest.mark.parametrize("text", [
    "What is the average sales every day?",
    "Show daily sales by region",
    "compute weekly revenue by store",
    "how many transactions happen each morning",
])
def test_analysis_questions_with_calendar_words_are_not_schedules(text):
    assert planner_mod._parse_task_schedule_from_text(text.lower()) is None


def test_a_cadence_without_a_time_is_not_concrete_unless_it_names_a_day():
    assert planner_mod._parse_task_schedule_from_text("run this every morning") is None
    schedule, concrete = planner_mod._parse_task_schedule_from_text("schedule this every morning")
    assert concrete is False


def test_the_llm_path_is_forced_to_schedule_for_weekday_wording():
    import re
    pattern = planner_mod._SCHEDULE_TRIGGER_RE
    assert pattern.search(MONDAY.lower()) and pattern.search("each weekday at 8am refresh it")
    assert not pattern.search("average sales every day")


@pytest.mark.parametrize("text,expected", [
    (MONDAY, "recompute fraud rate by ProductCD"),
    (SUNDAY, "Calculate the total amount per category"),
    ("Recompute the fraud rate by ProductCD every 2 minutes", "Recompute the fraud rate by ProductCD"),
    ("recompute totals daily at 6am", "recompute totals"),
    ("schedule this to run every day at 6", ""),
    ("run it every 5 minutes", ""),
])
def test_the_coder_gets_the_analysis_without_the_cadence(text, expected):
    assert planner_mod._analysis_request_without_cadence(text) == expected


# --------------------------------------------------------------------------- #
# The fabrication guard must not call a derivable metric invented
# --------------------------------------------------------------------------- #
# The guide's sentence was blocked on the fraud fixture: "I couldn't map these
# to the data: fraud rate" -- by the guard's LLM judge, one turn in two.

FRAUD_COLUMNS = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card4", "isFraud"]


def test_a_rate_of_a_column_the_dataset_has_is_anchored():
    vocab = planner_mod._fab_dataset_vocab(FRAUD_COLUMNS, [])
    assert planner_mod._fab_candidate_anchored("fraud rate", FRAUD_COLUMNS, vocab) is True


def test_a_metric_of_something_the_dataset_lacks_is_still_unanchored():
    vocab = planner_mod._fab_dataset_vocab(FRAUD_COLUMNS, [])
    assert planner_mod._fab_candidate_anchored("engagement rate", FRAUD_COLUMNS, vocab) is False
    assert planner_mod._fab_candidate_anchored("churn score", FRAUD_COLUMNS, vocab) is False
    assert planner_mod._fab_candidate_anchored("rate", FRAUD_COLUMNS, vocab) is False
