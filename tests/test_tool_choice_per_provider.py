"""tool_choice must match the provider, not be one hard-coded string.

OpenRouter rejects tool_choice="required". Measured against the live API:
openai/gpt-oss-120b returned finish_reason="error" with empty content on 3/3
attempts with 19 tools and 2/3 with only 3 tools, so tool count was not the
factor. Every planner turn then raised "Planner response returned no tool calls"
and the user saw "I ran into a temporary problem". With "auto" a live four-prompt
planner run went from 0/4 working to 3/4.
"""
from __future__ import annotations

import importlib

import pytest

from app.core.inference import forced_tool_choice_for_provider, tool_choice_for_provider

CASES = [
    ("groq", "required"),        # works there, and dispatch relies on it
    ("openrouter", "auto"),      # "required" -> finish_reason="error", 0 tool calls
    ("local", "required"),       # Ollama: "auto" returns prose, "required" works
    ("openai", "required"),
    ("vertex", "any"),           # Vertex's spelling of the same idea
    ("", "required"),            # unknown/disabled -- a string, never unbound
]


@pytest.mark.parametrize("provider,expected", CASES)
def test_tool_choice_for_provider(provider, expected):
    assert tool_choice_for_provider(provider) == expected


def test_openrouter_and_local_disagree():
    """Both build ChatOpenAI, so keying on the client class would be wrong.

    This is the regression guard: a single value for all OpenAI-compatible
    providers breaks one of them. Measured -- OpenRouter errors on "required",
    Ollama returns no tool call on "auto".
    """
    assert tool_choice_for_provider("openrouter") == "auto"
    assert tool_choice_for_provider("local") == "required"
    assert tool_choice_for_provider("openrouter") != tool_choice_for_provider("local")


def test_vertex_recognised_by_client_when_provider_is_unknown():
    assert tool_choice_for_provider("", "ChatVertexAI") == "any"


@pytest.mark.parametrize("module", ["app.agents.planner", "app.agents.mta_v2.agent"])
def test_agents_bind_tool_choice_from_the_helper(module):
    """Bound unconditionally, and consistent with the helper.

    Binding it only in an `else` branch meant that with no API key the name was
    never created and the first invoke raised NameError -- 57 test failures.
    """
    mod = importlib.import_module(module)
    assert isinstance(getattr(mod, "TOOL_CHOICE", None), str)
    assert mod.TOOL_CHOICE in {'required', 'auto', 'any'}


FORCED_CASES = [
    ("groq", "required"),        # accepted and enforced
    ("openai", "required"),
    ("local", "required"),       # accepted by Ollama, not enforced (gemma4:e4b answered in prose)
    ("vertex", "any"),
    ("openrouter", None),        # rejects "required": finish_reason="error"
    ("bedrock", None),           # not measured: never a guess
    ("", None),
]


@pytest.mark.parametrize("provider,expected", FORCED_CASES)
def test_forced_tool_choice_for_provider(provider, expected):
    """What the no-tool-call retry may insist with, per provider."""
    assert forced_tool_choice_for_provider(provider) == expected


def test_openrouter_is_never_forced():
    """The one provider measured to reject forcing gets None, so the retry
    keeps the portable "auto" and relies on the instruction instead."""
    assert forced_tool_choice_for_provider("openrouter") is None
    assert tool_choice_for_provider("openrouter") == "auto"


def test_mta_binds_the_forced_choice_from_the_helper():
    import importlib

    mod = importlib.import_module("app.agents.mta_v2.agent")
    assert mod.FORCED_TOOL_CHOICE in {"required", "auto", "any"}
    assert mod.FORCED_TOOL_CHOICE == (
        forced_tool_choice_for_provider(mod.MTA_PROVIDER, type(mod.llm).__name__) or mod.TOOL_CHOICE
    )
