"""Production-graph wiring for the conversational router and ClaimVerifier."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

pytest.importorskip("celery", reason="production workflow requires the API scheduler dependencies")

from app.api import workflow


def test_direct_conversation_runs_router_then_claim_verifier_without_planner():
    planner = Mock(side_effect=AssertionError("direct conversational reply reached planner"))

    def conversational(state):
        return {
            "messages": list(state.get("messages") or [])
            + [AIMessage(content="Hello — I can help with your data.")],
            "avaloka_mode": "exploring",
            "avaloka_intent": "chit_chat",
            "delegate_to_planner": False,
        }

    with patch.object(workflow, "memory_injection_node", return_value={}), \
            patch.object(workflow, "avaloka_agent_node", side_effect=conversational), \
            patch.object(workflow, "plan_etl_job", planner):
        graph = workflow.build_graph().compile()
        result = graph.invoke({
            "messages": [HumanMessage(content="hello")],
            "planner_definition": {},
        })

    planner.assert_not_called()
    assert result["avaloka_intent"] == "chit_chat"
    assert result["verification_safe_to_present"] is True
    assert result["verification_report"]["claims_checked"] == 1


def test_data_request_delegates_to_planner_then_runs_claim_verifier():
    def conversational(_state):
        return {
            "avaloka_mode": "analyzing",
            "avaloka_intent": "statistical_analysis",
            "delegate_to_planner": True,
        }

    planner = Mock(return_value={
        "messages": [AIMessage(content="The observed columns are correlated.")],
        "ready_to_summarize": False,
        "ready_to_code": False,
    })

    with patch.object(workflow, "memory_injection_node", return_value={}), \
            patch.object(workflow, "avaloka_agent_node", side_effect=conversational), \
            patch.object(workflow, "plan_etl_job", planner):
        graph = workflow.build_graph().compile()
        result = graph.invoke({
            "messages": [HumanMessage(content="show correlation")],
            "planner_definition": {},
        })

    planner.assert_called_once()
    assert result["verification_safe_to_present"] is True
    assert result["verification_report"]["claims_checked"] == 1
