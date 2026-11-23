"""A question about verified results already in hand is answered, not delegated.

The conversational product benchmark drove its eleven probes through the graph
with verified results in ``avaloka_recommendation`` -- the field the reply
step forwards as ``<recommendation>`` -- and every one of them was delegated to
the planner, which never sees that field. "Churn fell -- can you confirm?" ended
in a planning failure or an execution error with the answer sitting in state.

The rule: with verified results in hand, a message the keyword table does not
claim (``clarification``) and that asks for no new work becomes
``discussion``, answered directly. Everything else routes as before: a keyword
match keeps its route, new work goes to the planner, and with no results in
hand nothing changes at all.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.agents.avaloka_agent.intent_classifier import (
    _heuristic_intent,
    asks_about_results,
    classify_intent,
)

# Asked about results in hand: confirm, correct, interpret, or decline.
QUESTIONS = [
    "What was revenue last quarter?",
    "How many rows are in this dataset?",
    "How many distinct aisles are there? A sampled estimate is fine.",
    "Churn fell this quarter — can you confirm?",
    "That's wrong, it's 42.5. Please correct it.",
    "What's churn for the enterprise segment?",
    "How did the model do?",
    "So increasing ad spend will increase revenue?",
]

# New work: these need the pipeline even with results in hand.
NEW_WORK = [
    "Compute distinct aisles on the entire dataset.",
    "You excluded refunds. Include them and recompute.",
    "Impute the missing total_bedrooms with the median.",
    "filter to the north region and re-run it",
    "count the orders per day",
]


def _classify(text: str, results_in_hand: bool) -> str:
    return classify_intent([HumanMessage(content=text)], results_in_hand=results_in_hand).intent


@pytest.mark.parametrize("text", QUESTIONS)
def test_a_question_about_results_in_hand_is_discussion(text: str) -> None:
    assert asks_about_results(text)
    result = classify_intent([HumanMessage(content=text)], results_in_hand=True)
    assert result.intent == "discussion"
    assert not result.should_delegate()


@pytest.mark.parametrize("text", QUESTIONS)
def test_without_results_in_hand_nothing_changes(text: str) -> None:
    assert _classify(text, False) == _heuristic_intent(text, False).intent


@pytest.mark.parametrize("text", NEW_WORK)
def test_new_work_still_goes_to_the_planner(text: str) -> None:
    assert not asks_about_results(text)
    result = classify_intent([HumanMessage(content=text)], results_in_hand=True)
    assert result.intent != "discussion" and result.should_delegate()


@pytest.mark.parametrize("text,intent", [
    ("Now chart median_house_value by ocean_proximity on the cleaned dataset.", "visualization"),
    ("train a model to predict churn", "ml_training"),
    ("what is in this dataset?", "exploration"),
    ("what drives churn?", "statistical_analysis"),
    ("schedule this every day", "schedule"),
])
def test_a_keyword_route_is_kept_with_results_in_hand(text: str, intent: str) -> None:
    assert _classify(text, True) == intent
