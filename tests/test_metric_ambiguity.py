"""Clarify genuinely ambiguous measures using the active dataset schema."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.planner import (
    _detect_ambiguous_prompt_details,
    _resolve_pending_clarification,
    plan_etl_job,
)


@pytest.mark.parametrize("columns,prompt,measure", [
    (["gross_revenue", "net_revenue", "month"], "What was revenue last month?", "revenue"),
    (["base_salary", "bonus_salary"], "Show salary by department", "salary"),
    (["ActualCost", "BudgetCost"], "Calculate cost", "cost"),
    (["gross_revenue", "net_revenue"], "Why did revenue fall?", "revenue"),
    (["gross_revenue", "net_revenue"], "revenue by region", "revenue"),
])
def test_multiple_qualified_measures_require_a_choice(columns, prompt, measure):
    details = _detect_ambiguous_prompt_details(prompt, {"uploaded_csv_columns": columns})
    assert details is not None
    assert details["pending_clarification"]["type"] == "metric_choice"
    assert details["pending_clarification"]["measure"] == measure
    assert details["pending_clarification"]["columns"] == columns[:2]


@pytest.mark.parametrize("columns,prompt", [
    (["gross_revenue", "net_revenue"], "What was gross revenue last month?"),
    (["gross_revenue", "net_revenue"], "Compare gross and net revenue"),
    (["gross_revenue", "net_revenue"], "Show all revenue columns"),
    (["gross_revenue", "net_revenue"], "Show revenue columns"),
    (["gross_revenue", "net_revenue"], "Show all revenue by region"),
    (["gross_revenue", "net_revenue"], "How many rows are there?"),
    (["revenue"], "What was revenue last month?"),
    (["revenue", "net_revenue"], "What was revenue last month?"),
    (["gross_revenue", "net_revenue"], "Train a model to predict revenue"),
])
def test_explicit_or_unambiguous_requests_do_not_ask(columns, prompt):
    assert _detect_ambiguous_prompt_details(prompt, {"uploaded_csv_columns": columns}) is None


def test_does_not_combine_columns_from_unrelated_datasets():
    state = {"multi_dataset_state": [
        {"columns": ["gross_revenue"]}, {"columns": ["net_revenue"]},
    ]}
    assert _detect_ambiguous_prompt_details("What was revenue?", state) is None


def test_ambiguity_uses_schema_when_uploaded_columns_are_absent():
    state = {"schema": {"gross_revenue": "float", "net_revenue": "float"}}
    assert _detect_ambiguous_prompt_details("What was revenue?", state) is not None


def test_generic_all_and_comparison_do_not_hide_ambiguity():
    state = {"uploaded_csv_columns": ["gross_revenue", "net_revenue", "cost"]}
    assert _detect_ambiguous_prompt_details("What was revenue for all customers?", state)
    assert _detect_ambiguous_prompt_details("What was revenue for both regions?", state)
    assert _detect_ambiguous_prompt_details("Compare revenue and cost", state)
    assert _detect_ambiguous_prompt_details("What was revenue where region is east?", state)


def test_non_data_intent_does_not_ask_despite_schema_overlap():
    state = {
        "uploaded_csv_columns": ["gross_revenue", "net_revenue"],
        "avaloka_intent": "chit_chat",
    }
    assert _detect_ambiguous_prompt_details("I liked the revenue chart", state) is None


def test_metric_choice_resolves_to_the_selected_column():
    state = {"uploaded_csv_columns": ["gross_revenue", "net_revenue"]}
    pending = _detect_ambiguous_prompt_details("What was revenue?", state)["pending_clarification"]
    resolved = _resolve_pending_clarification(pending, "gross")
    assert "gross_revenue" in resolved
    assert _detect_ambiguous_prompt_details(resolved, state) is None
    assert _resolve_pending_clarification(pending, "gross or net") is None


def test_planner_asks_before_calculation_then_resolves_choice():
    state = {
        "messages": [HumanMessage(content="What was revenue last month?")],
        "uploaded_csv_columns": ["gross_revenue", "net_revenue", "month"],
    }
    asked = plan_etl_job(state)
    assert asked["pending_clarification"]["type"] == "metric_choice"
    assert asked["ready_to_code"] is False
    assert asked["plan"] is None
    assert any(isinstance(msg, AIMessage) and "gross revenue" in msg.content
               and "net revenue" in msg.content for msg in asked["messages"])

    followup = dict(asked)
    followup["messages"] = [HumanMessage(content="net")]
    resolved = plan_etl_job(followup)
    assert resolved.get("pending_clarification") is None
    assert any(isinstance(msg, HumanMessage) and "net_revenue" in msg.content
               for msg in resolved["messages"])


def test_planner_does_not_ask_with_only_one_measure():
    state = {
        "messages": [HumanMessage(content="What was revenue last month?")],
        "uploaded_csv_columns": ["revenue", "month"],
    }
    result = plan_etl_job(state)
    assert result.get("pending_clarification") is None
    assert not any(isinstance(msg, AIMessage) and "Which revenue" in msg.content
                   for msg in result.get("messages", []))


def test_unrecognised_short_reply_reasks_but_new_request_supersedes_choice():
    state = {
        "messages": [HumanMessage(content="What was revenue?")],
        "uploaded_csv_columns": ["gross_revenue", "net_revenue"],
    }
    asked = plan_etl_job(state)
    retry = plan_etl_job({**asked, "messages": [HumanMessage(content="unsure")]})
    assert retry["pending_clarification"]["type"] == "metric_choice"
    assert "Please choose one" in retry["messages"][-1].content

    fresh = plan_etl_job({**asked, "messages": [HumanMessage(content="How many rows are there?")]})
    assert fresh.get("pending_clarification") is None


def test_short_choice_reaches_planner_even_if_classified_as_chat(monkeypatch):
    from app.agents.avaloka_agent import agent as conversational_agent
    from app.agents.avaloka_agent.intent_classifier import IntentResult

    monkeypatch.setattr(conversational_agent, "_load_redis_context", lambda _state: {})
    monkeypatch.setattr(
        conversational_agent, "classify_intent",
        lambda *_args, **_kwargs: IntentResult(intent="chit_chat", confidence=1.0),
    )
    monkeypatch.setattr(conversational_agent, "decide_execution_profile", lambda *_args: {})
    monkeypatch.setattr(conversational_agent, "resolve_infra_preference", lambda *_args: {})
    monkeypatch.setattr(conversational_agent, "_preconfigure_for_planner", lambda *_args: {})
    monkeypatch.setattr(conversational_agent, "plan_swarm", lambda *_args: None)

    state = {
        "messages": [HumanMessage(content="net")],
        "pending_clarification": {
            "type": "metric_choice", "measure": "revenue",
            "columns": ["gross_revenue", "net_revenue"],
            "original_prompt": "What was revenue?",
        },
    }
    result = conversational_agent.avaloka_agent_node(state)
    assert result["delegate_to_planner"] is True
