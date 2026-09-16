"""Recovering an answer from a rejected tool call.

Under ``tool_choice="required"`` Groq returns a 400 (``tool_use_failed``) when
the model writes a reply without wrapping it as a tool call. The reply itself is
right there in ``error.failed_generation``; showing the user "I ran into a
temporary problem" instead is a self-inflicted loss.
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from app.agents import planner as planner_mod
from app.agents.planner import (_extract_failed_respond_to_user_text,
                                _failed_generation, plan_etl_job)
from app.graph.etl_state import ETLState

# The sibling recovery path for a truncated NoParams call. It exists in the
# working tree it was written in but on no branch yet, so this file has to work
# either way rather than fail to import.
_salvage_noparam_tool_call = getattr(planner_mod, "_salvage_noparam_tool_call", None)


def _tool_use_failed(failed_generation: str) -> Exception:
    """A Groq 400 stringified the way the SDK stringifies it: a Python repr."""
    return Exception(
        "Error code: 400 - "
        + repr({
            "error": {
                "message": "Tool choice is required, but model did not call a tool",
                "type": "invalid_request_error",
                "code": "tool_use_failed",
                "failed_generation": failed_generation,
            }
        })
    )


# The exact payload from the 2026-08-20 00:00:04 log, apostrophe and all.
REAL_GENERATION = (
    '{\n  "response_text": "The dataset contains information about YouTube '
    "channels, including fields such as Youtuber name, subscriber count, total "
    "video views, category, channel type, number of uploads, country, and "
    "various demographic and economic indicators for the channel's country.\"\n}"
)


# --------------------------------------------------------------------------- #
# The shape that was being dropped: bare arguments, no wrapper, no tool name
# --------------------------------------------------------------------------- #

def test_a_bare_arguments_object_is_recovered():
    recovered = _extract_failed_respond_to_user_text(_tool_use_failed(REAL_GENERATION))
    assert recovered is not None, "the model's answer was thrown away"
    assert recovered.startswith("The dataset contains information about YouTube channels")


def test_the_recovered_text_survives_repr_escaping():
    """The generation arrives through a Python repr; an apostrophe inside it is
    escaped on the way and must not still be escaped on the way out."""
    recovered = _extract_failed_respond_to_user_text(_tool_use_failed(REAL_GENERATION))
    assert "channel's country" in recovered
    assert "\\'" not in recovered


def test_failed_generation_is_read_back_exactly():
    assert _failed_generation(_tool_use_failed(REAL_GENERATION)) == REAL_GENERATION


# --------------------------------------------------------------------------- #
# What must NOT be recovered
# --------------------------------------------------------------------------- #

def test_another_tools_arguments_are_not_read_as_an_answer():
    """`response_text` belongs to exactly one tool schema. A generation for any
    other tool has to fall through to the normal error path."""
    exc = _tool_use_failed('{"suggestions": ["look at churn", "look at revenue"]}')
    assert _extract_failed_respond_to_user_text(exc) is None


def test_a_400_that_is_not_tool_use_failed_is_left_alone():
    exc = Exception("Error code: 400 - {'error': {'message': 'bad request', 'code': 'invalid'}}")
    assert _extract_failed_respond_to_user_text(exc) is None
    assert _failed_generation(exc) is None


def test_an_empty_response_text_is_not_an_answer():
    assert _extract_failed_respond_to_user_text(_tool_use_failed('{"response_text": "   "}')) is None


def test_unparseable_generation_does_not_raise():
    assert _extract_failed_respond_to_user_text(_tool_use_failed("{not json at all")) is None


# --------------------------------------------------------------------------- #
# The older shape must keep working
# --------------------------------------------------------------------------- #

def test_the_malformed_wrapper_shape_still_recovers():
    exc = Exception(
        'Error code: 400 - tool_use_failed '
        '<function=respond_to_user {"response_text": "hello there"}</function>'
    )
    assert _extract_failed_respond_to_user_text(exc) == "hello there"


@pytest.mark.skipif(_salvage_noparam_tool_call is None,
                    reason="_salvage_noparam_tool_call is not on this branch")
def test_noparam_salvage_is_untouched():
    """The sibling recovery path keys off the tool NAME and is unaffected."""
    exc = _tool_use_failed('{"name": "generate_code", "arguments": "{trunca')
    assert _salvage_noparam_tool_call(exc) == "generate_code"
    exc = _tool_use_failed('{"name": "initiate_transfer", "arguments": "{trunca')
    assert _salvage_noparam_tool_call(exc) is None, "a tool with parameters must not be guessed"


# --------------------------------------------------------------------------- #
# End to end through the planner
# --------------------------------------------------------------------------- #

@patch("app.agents.planner.llm")
def test_the_user_gets_the_answer_not_the_apology(mock_llm):
    """The reported turn: 'what does this dataset talk about?'"""
    mock_llm.invoke.side_effect = _tool_use_failed(REAL_GENERATION)

    result = plan_etl_job(ETLState(
        messages=[HumanMessage(content="what does this dataset talk about?")]
    ))

    reply = result["messages"][-1].content
    assert "temporary problem" not in reply
    assert "YouTube channels" in reply
    # A recovered prose answer is a finished turn, not a reason to run code.
    assert result["ready_to_code"] is False
    assert result["ready_to_summarize"] is False


@patch("app.agents.planner.llm")
def test_an_unrecoverable_failure_still_apologises(mock_llm):
    """The fallback message has to survive for the failures it was written for."""
    mock_llm.invoke.side_effect = Exception("Error code: 503 - service unavailable")

    result = plan_etl_job(ETLState(messages=[HumanMessage(content="hello")]))

    assert "temporary problem" in result["messages"][-1].content
    assert result["ready_to_code"] is False


# --------------------------------------------------------------------------- #
# Log level: a handled turn must not read as an outage
# --------------------------------------------------------------------------- #

@patch("app.agents.planner.llm")
def test_a_recovered_turn_logs_no_error(mock_llm, caplog):
    """The user saw a correct answer; the log said ERROR. That combination is
    what makes an operator chase a non-incident, and would trip any alerting
    keyed on ERROR."""
    mock_llm.invoke.side_effect = _tool_use_failed(REAL_GENERATION)

    with caplog.at_level("DEBUG", logger="app.agents.planner"):
        plan_etl_job(ETLState(
            messages=[HumanMessage(content="what does this dataset talk about?")]
        ))

    errors = [r for r in caplog.records
              if r.levelname == "ERROR" and r.name == "app.agents.planner"]
    assert not errors, f"recovered turn logged ERROR: {[r.getMessage() for r in errors]}"

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("RECOVERED" in m for m in warnings), \
        "the rejection should still be visible, just not as an error"


@patch("app.agents.planner.llm")
def test_an_unrecovered_failure_still_logs_error(mock_llm, caplog):
    """The ERROR has to survive for the failures it was written for."""
    mock_llm.invoke.side_effect = Exception("Error code: 503 - service unavailable")

    with caplog.at_level("DEBUG", logger="app.agents.planner"):
        plan_etl_job(ETLState(messages=[HumanMessage(content="hello")]))

    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert any("Structured Output/Tool Call failed" in m for m in errors)
