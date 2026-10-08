"""Per-agent backend selection and the summarizer's tuned call (PR #420).

PR #420 added ``backend_for`` / ``SUPPORTED_BACKENDS`` / ``_ROUTABLE_AGENTS`` to
app/core/model_config.py, moved the visualization default to gpt-oss-120b, and
rewrote how app/agents/summarizer.py calls its model (plain message list,
token budget, reasoning effort, a distinct error when reasoning used the whole
budget). Neither change came with a test.

No LLM: the summarizer is driven by a scripted fake, so this covers what it
sends and how it reacts, not what a real model writes.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents import summarizer
from app.core import model_config as mc


# ---------------------------------------------------------------------------
# backend_for
# ---------------------------------------------------------------------------


def test_every_agent_defaults_to_groq(monkeypatch):
    for agent in mc.DEFAULT_MODELS:
        monkeypatch.delenv(f"AVALOKA_{agent.upper()}_BACKEND", raising=False)
    assert {mc.backend_for(agent) for agent in mc.DEFAULT_MODELS} == {"groq"}


@pytest.mark.parametrize("value,expected", [
    ("openrouter", "openrouter"), (" Local ", "local"), ("GROQ", "groq"), ("", "groq"),
])
def test_the_visualization_agent_honours_its_backend_variable(monkeypatch, value, expected):
    monkeypatch.setenv("AVALOKA_VISUALIZATION_BACKEND", value)
    assert mc.backend_for("visualization") == expected


def test_an_unknown_backend_is_an_error_naming_the_variable(monkeypatch):
    monkeypatch.setenv("AVALOKA_VISUALIZATION_BACKEND", "bedrock")
    with pytest.raises(mc.UnsupportedModelError, match="AVALOKA_VISUALIZATION_BACKEND='bedrock'"):
        mc.backend_for("visualization")


def test_a_backend_set_on_a_groq_only_agent_is_ignored_and_says_so(monkeypatch, caplog):
    """Returning 'openrouter' here would be a lie: the planner builds ChatGroq."""
    monkeypatch.setenv("AVALOKA_PLANNER_BACKEND", "openrouter")
    with caplog.at_level(logging.WARNING, logger=mc.logger.name):
        assert mc.backend_for("planner") == "groq"
    assert "AVALOKA_PLANNER_BACKEND=openrouter ignored" in caplog.text


# ---------------------------------------------------------------------------
# resolve / describe_all
# ---------------------------------------------------------------------------

_FOREIGN = "gpt-5"


def test_the_test_model_really_is_foreign_to_groq():
    """Guards the three tests below: if this id became Groq-serveable they
    would pass without exercising the routing at all."""
    assert mc.provider_for_model(_FOREIGN) is not None


def test_a_foreign_model_on_groq_fails_with_a_hint_for_a_routable_agent(monkeypatch):
    monkeypatch.setenv("AVALOKA_VISUALIZATION_MODEL", _FOREIGN)
    monkeypatch.delenv("AVALOKA_VISUALIZATION_BACKEND", raising=False)
    with pytest.raises(mc.UnsupportedModelError, match="AVALOKA_VISUALIZATION_BACKEND=openrouter"):
        mc.resolve("visualization")


def test_a_foreign_model_on_groq_gives_no_such_hint_for_a_groq_only_agent(monkeypatch):
    monkeypatch.setenv(mc.env_var_for("planner"), _FOREIGN)
    with pytest.raises(mc.UnsupportedModelError) as err:
        mc.resolve("planner")
    assert "BACKEND=openrouter" not in str(err.value)


def test_a_foreign_model_resolves_once_the_agent_is_routed_off_groq(monkeypatch):
    monkeypatch.setenv("AVALOKA_VISUALIZATION_MODEL", _FOREIGN)
    monkeypatch.setenv("AVALOKA_VISUALIZATION_BACKEND", "openrouter")
    assert mc.resolve("visualization") == _FOREIGN


def test_describe_all_reports_each_agents_own_backend(monkeypatch):
    monkeypatch.setenv("AVALOKA_VISUALIZATION_MODEL", _FOREIGN)
    monkeypatch.setenv("AVALOKA_VISUALIZATION_BACKEND", "openrouter")
    monkeypatch.delenv("AVALOKA_PLANNER_BACKEND", raising=False)

    described = mc.describe_all()

    assert described["visualization"]["backend"] == "openrouter"
    assert described["visualization"]["serveable_by_backend"] is True
    assert described["planner"]["backend"] == "groq"
    assert set(described) == set(mc.DEFAULT_MODELS)


def test_the_narrator_has_no_model_entry_so_it_uses_the_summarizers():
    """result_narrator._resolve_narrator_model depends on this KeyError."""
    from app.agents import result_narrator

    assert "narrator" not in mc.DEFAULT_MODELS
    with pytest.raises(KeyError):
        mc.resolve("narrator")
    assert result_narrator._NARRATOR_MODEL == mc.model_for("summarizer").model


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


class _Reply:
    def __init__(self, content, finish_reason="stop"):
        self.content = content
        self.response_metadata = {"finish_reason": finish_reason}


class _ScriptedLLM:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


_GOOD = '```json\n{"job_name": "load fares", "notes": "none"}\n```'


def _state() -> dict:
    return {"messages": [
        HumanMessage(content='load {"table": "fares"} into the warehouse'),
        AIMessage(content="Plan: read fares, write to warehouse."),
    ]}


@pytest.fixture
def scripted(monkeypatch):
    def install(*outcomes, reasoning=True):
        llm = _ScriptedLLM(*outcomes)
        monkeypatch.setattr(summarizer, "summarizer_llm", llm)
        monkeypatch.setattr(summarizer, "_IS_REASONING", reasoning)
        return llm
    return install


def test_literal_braces_in_the_conversation_reach_the_model_untouched():
    """The old ChatPromptTemplate treated '{...}' in the chat as a variable."""
    messages = summarizer._create_summary_messages(_state()["messages"])

    assert isinstance(messages[0], SystemMessage) and isinstance(messages[1], HumanMessage)
    assert '{"table": "fares"}' in messages[1].content
    assert summarizer._SCHEMA_JSON_STR in messages[0].content


def test_the_previous_error_is_fed_back_into_the_prompt():
    messages = summarizer._create_summary_messages(_state()["messages"], "No valid JSON block found.")
    assert "The error was: No valid JSON block found." in messages[0].content


def test_the_summarizer_sends_a_budget_and_low_effort(scripted):
    llm = scripted(_Reply(_GOOD))

    out = summarizer.summarize_etl_job(_state())

    assert llm.calls[0][1] == {
        "max_tokens": summarizer._SUMMARIZER_MAX_TOKENS, "reasoning_effort": "low",
    }
    assert out["planner_definition"]["job_name"] == "load fares"
    assert out["ready_to_code"] is True


def test_an_answer_eaten_by_reasoning_is_retried_and_named_in_the_next_prompt(scripted):
    llm = scripted(_Reply("", finish_reason="length"), _Reply(_GOOD))

    out = summarizer.summarize_etl_job(_state())

    assert len(llm.calls) == 2
    assert "token budget was used up by reasoning" in llm.calls[1][0][0].content
    assert out["ready_to_code"] is True


def test_two_bad_answers_stop_the_pipeline_before_the_coder(scripted):
    llm = scripted(_Reply("no json here"), _Reply("still none"))

    out = summarizer.summarize_etl_job(_state())

    assert len(llm.calls) == 2
    assert out["ready_to_code"] is False
    assert "Failed to generate a valid ETL job summary after 2 attempts" in out["messages"][-1].content


def test_rejected_tuning_kwargs_fall_back_to_a_plain_call(scripted):
    llm = scripted(TypeError("unexpected keyword 'reasoning_effort'"), _Reply(_GOOD))

    out = summarizer.summarize_etl_job(_state())

    assert llm.calls[1][1] == {}
    assert out["ready_to_code"] is True
