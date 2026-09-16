"""Routing tests for the conversational intent classifier.

These cover the two defects found by driving a live kind deployment as the UI
does, not by reading code: a schema question that never reached the data, and a
training request that routed to inference. Both were invisible to the existing
suite because nothing asserted on ``should_delegate`` at all.
"""
from __future__ import annotations

import pytest

from app.agents.avaloka_agent.intent_classifier import _heuristic_intent


def _intent(text: str) -> str:
    return _heuristic_intent(text, False).intent


def _delegates(text: str) -> bool:
    return _heuristic_intent(text, False).should_delegate()


# These are answered from the discovery pass on the cheap direct-reply path,
# NOT by delegating to the planner -- delegating "what is in this dataset?"
# starts a 70s training run to report column names. What matters is that they
# classify as exploration, so the renderer knows to describe the data.
@pytest.mark.parametrize("message", [
    "what is in this dataset?",
    "what's in this file?",
    "show me the schema",
    "list columns",
    "describe the columns",
    "sample rows",
])
def test_questions_about_the_data_classify_as_exploration(message: str) -> None:
    assert _intent(message) == "exploration"
    assert not _delegates(message), (
        f"{message!r} is answerable from discovery alone; delegating it sends a "
        f"question about column names through the training pipeline")


def test_exploration_is_answered_from_discovery_without_an_llm() -> None:
    """The no-LLM fallback must describe the data, not ask what the user wants.

    This is the whole point of the deterministic path: an install with no model
    key still answers the commonest first question. It previously fell through
    to "tell me a little about what you're trying to figure out" while holding
    the schema.
    """
    from app.agents.avaloka_agent.agent import _describe_dataset

    out = _describe_dataset(
        {"schema_columns": ["region", "units", "revenue"],
         "schema_types": {"region": "object", "units": "int64",
                          "revenue": "float64"},
         "row_count": 8},
        None,
    )
    assert "region" in out and "revenue" in out
    assert "float64" in out
    assert "trying to figure out" not in out


def test_describe_dataset_says_so_when_nothing_is_bound() -> None:
    out = _describe_dataset_import()({}, None)
    assert "don't have a dataset" in out


def _describe_dataset_import():
    from app.agents.avaloka_agent.agent import _describe_dataset
    return _describe_dataset


# "predict" belongs to ml_inference and "train a model" to ml_training; when a
# message contains both, the longer and more specific one decides it. Ordering
# of the rule table must not.
@pytest.mark.parametrize("message", [
    "train a model to predict churn",
    "build me a model that predicts revenue from the other columns",
    "fit a model to predict revenue",
])
def test_training_requests_are_not_routed_to_inference(message: str) -> None:
    assert _intent(message) == "ml_training", (
        f"{message!r} asks to BUILD a model; routing it to ml_inference asks "
        f"the system to run one that does not exist yet")


@pytest.mark.parametrize("message,expected", [
    ("score this with the model", "ml_inference"),
    ("run inference", "ml_inference"),
    ("train a model", "ml_training"),
    ("hyperparameter search", "ml_training"),
])
def test_unambiguous_ml_intents_are_unchanged(message: str, expected: str) -> None:
    assert _intent(message) == expected


# Guard the regression the fix itself could cause: purely conversational turns
# must NOT start pulling data.
@pytest.mark.parametrize("message", ["hello", "thanks", "who are you?"])
def test_chit_chat_still_does_not_touch_data(message: str) -> None:
    assert not _heuristic_intent(message, False).is_data_touching()


# ---------------------------------------------------------------------------
# Sampling-mode switches, and the interrogative "where"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "use entire dataset",
    "use full dataset",
    "switch to entire dataset",
    "switch to full dataset",
    "run this on the entire dataset",
    "use quick sample",
    "use portfolio samples",
])
def test_sampling_switches_reach_the_planner(message: str) -> None:
    """These are instructions to the pipeline, not questions about the data.

    The planner already owns the machinery that honours them
    (``parse_fidelity_from_control_text`` / ``is_explicit_mode_switch_message``
    / ``build_mode_switch_confirmation``). They used to classify as
    ``exploration``, which does not delegate, so the planner never saw them and
    the sampling mode silently never changed -- while the agent's own caveat
    tells users to say exactly these words to get exact numbers.
    """
    assert _intent(message) == "sampling_mode"
    assert _delegates(message), (
        f"{message!r} changes how much data is scanned; answered from chat it "
        f"leaves the mode untouched while implying it changed")


@pytest.mark.parametrize("message", [
    "I want to understand why people are leaving. Where should I start?",
    "Where do I begin?",
    "Where can I see the churn breakdown?",
])
def test_interrogative_where_is_not_a_filter_clause(message: str) -> None:
    """"Where should I start?" is a person asking for guidance, not SQL.

    The ambiguous-filter pattern captures whatever follows "where", so an
    interrogative produced the garbled "What should i start should I filter
    by?" -- in response to the friendliest question a new user asks.
    """
    from app.agents.planner import _detect_ambiguous_prompt_details

    assert _detect_ambiguous_prompt_details(message) is None


@pytest.mark.parametrize("message", ["filter where region", "show rows where region"])
def test_real_filter_clauses_still_ask(message: str) -> None:
    from app.agents.planner import _detect_ambiguous_prompt_details

    result = _detect_ambiguous_prompt_details(message)
    assert result is not None and "region" in result["message"]
