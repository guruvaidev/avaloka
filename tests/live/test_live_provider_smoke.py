"""Live smoke: does the configured provider actually dispatch a tool?

Every one of the 2557 hermetic tests passes without an LLM, and that is the
point of them. It is also why a real defect shipped: TOOL_CHOICE was hard-coded
to "required", OpenRouter answers that with finish_reason="error" and no tool
call, and every planner turn returned "I ran into a temporary problem while
planning that request." No offline test could see it -- the provider's handling
of one request parameter is not something a stub can model.

So this suite asks the only question a stub cannot: with a real endpoint, does a
turn come back? It is provider-agnostic on purpose -- the same file runs against
OpenRouter weekly and a local Ollama model nightly, because the two disagree
about tool_choice and a tier that only ever exercised one would miss the other.

Marked `integration`, so the hermetic gate's `-m "not integration"` excludes it.
Skips cleanly when nothing is configured, so a contributor without keys is never
asked to care.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

#: Turns that must produce a reply. Deliberately mundane -- the failure mode
#: being guarded is "no turn works at all", not subtle answer quality.
TURNS = [
    "what can you help me with?",
    "run 3",
    "remember my favourite column is reordered",
]

#: Substrings that mean the pipeline gave up rather than answered.
GAVE_UP = ("temporary problem", "could not route", "planner failed")


def _configured() -> str:
    """The provider this run can actually reach, or "" if none."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("INFERENCE_LOCAL_BASE_URL"):
        return "local"
    if os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY_PLANNING_AGENT"):
        return "groq"
    return ""


skip_unconfigured = pytest.mark.skipif(
    not _configured(),
    reason="no live provider configured (set OPENROUTER_API_KEY, "
           "INFERENCE_LOCAL_BASE_URL, or GROQ_API_KEY)",
)


@skip_unconfigured
def test_provider_builds_an_llm():
    """The factory returns a model, and TOOL_CHOICE is bound."""
    import app.agents.planner as planner

    assert planner.llm is not None, (
        "the factory returned no LLM despite a configured provider -- this is the "
        "shape of the regression where an OpenRouter default silently disabled a "
        "Groq-only deployment"
    )
    assert isinstance(planner.TOOL_CHOICE, str) and planner.TOOL_CHOICE, (
        "TOOL_CHOICE must be bound; leaving it unbound raised NameError at the "
        "first invoke instead of reporting a disabled LLM"
    )


@skip_unconfigured
def test_the_model_will_dispatch_a_tool():
    """tool_choice is accepted and a tool call comes back.

    This is the assertion that would have caught the OpenRouter failure: the
    request is well-formed, the key is valid, and the provider still returns no
    tool call.
    """
    import app.agents.planner as planner

    response = planner.llm.invoke(
        [{"role": "user", "content": "what can you help me with?"}],
        tools=planner.tools,
        tool_choice=planner.TOOL_CHOICE,
    )
    finish = (getattr(response, "response_metadata", {}) or {}).get("finish_reason")
    calls = getattr(response, "tool_calls", None) or []
    content = str(getattr(response, "content", "") or "")

    assert finish != "error", (
        f"provider rejected the request: finish_reason={finish!r}, "
        f"tool_choice={planner.TOOL_CHOICE!r}, {len(planner.tools)} tools. "
        f"This is what 'required' does on OpenRouter."
    )
    assert calls or content, (
        f"provider returned neither a tool call nor content "
        f"(finish_reason={finish!r}, tool_choice={planner.TOOL_CHOICE!r})"
    )


@skip_unconfigured
@pytest.mark.parametrize("prompt", TURNS)
def test_a_turn_produces_a_reply(prompt):
    """A full planner turn answers instead of surfacing an internal failure."""
    from langchain_core.messages import AIMessage, HumanMessage

    import app.agents.planner as planner

    result = planner.plan_etl_job({
        "messages": [HumanMessage(content=prompt)],
        "user_id": "live-smoke",
        "session_id": "live-smoke",
        "planner_definition": {},
    })
    replies = [m.content for m in result.get("messages", []) if isinstance(m, AIMessage)]
    assert replies, f"{prompt!r} produced no reply at all"

    last = str(replies[-1])
    hit = [marker for marker in GAVE_UP if marker in last]
    assert not hit, (
        f"{prompt!r} surfaced an internal failure ({hit[0]!r}) instead of an "
        f"answer, on provider {_configured()!r} with "
        f"tool_choice={planner.TOOL_CHOICE!r}: {last[:160]}"
    )
