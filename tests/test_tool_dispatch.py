"""Regression tests: every tool advertised to the planner LLM must have a
handler, and an unhandled tool call must never end the turn silently.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.planner as planner_mod
from app.agents.planner import plan_etl_job, tool_name_to_param, tools


class FakeToolCallResponse:
    def __init__(self, name, args):
        self.tool_calls = [{"name": name, "args": args}]


class FakeLLM:
    def __init__(self, name, args):
        self._response = FakeToolCallResponse(name, args)

    def invoke(self, *_a, **_k):
        return self._response


def _run_tool_turn(monkeypatch, name, args, state_extra=None):
    monkeypatch.setattr(planner_mod, "llm", FakeLLM(name, args))
    state = {
        "messages": [HumanMessage(content="what similar analyses have we done before?")],
        "user_id": "u",
        "session_id": "s",
    }
    state.update(state_extra or {})
    return plan_etl_job(state)


def _last_ai_content(result):
    replies = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
    return replies[-1].content if replies else ""


def test_every_advertised_tool_is_mapped_to_params():
    advertised = {t["function"]["name"] for t in tools}
    assert advertised <= set(tool_name_to_param), (
        "tools advertised to the LLM without a parameter mapping: "
        f"{advertised - set(tool_name_to_param)}"
    )


def test_retrieve_historical_analysis_replies_with_memory_hints(monkeypatch):
    monkeypatch.setattr(
        planner_mod, "retrieve_memory",
        lambda query, state: {"memory_hints": ["Grouped sales by region last week"]},
        raising=False,
    )
    import app.services.memory_plane as mp
    monkeypatch.setattr(
        mp, "retrieve_memory",
        lambda query, state: {"memory_hints": ["Grouped sales by region last week"]},
    )
    result = _run_tool_turn(
        monkeypatch, "retrieve_historical_analysis", {"query": "similar analyses"}
    )
    reply = _last_ai_content(result)
    assert "Grouped sales by region last week" in reply
    assert not result.get("ready_to_code")


def test_retrieve_historical_analysis_replies_even_with_no_memory(monkeypatch):
    import app.services.memory_plane as mp
    monkeypatch.setattr(mp, "retrieve_memory", lambda query, state: {"memory_hints": []})
    result = _run_tool_turn(
        monkeypatch, "retrieve_historical_analysis", {"query": "similar analyses"}
    )
    assert "couldn't find any similar past analyses" in _last_ai_content(result)


def test_retrieve_historical_analysis_falls_back_to_state_hints(monkeypatch):
    import app.services.memory_plane as mp

    def boom(query, state):
        raise RuntimeError("memory plane down")

    monkeypatch.setattr(mp, "retrieve_memory", boom)
    result = _run_tool_turn(
        monkeypatch, "retrieve_historical_analysis", {"query": "similar analyses"},
        state_extra={"memory_hints": ["Cleaned nulls in the orders dataset"]},
    )
    assert "Cleaned nulls in the orders dataset" in _last_ai_content(result)


# -------------------------
# LLM failures must not fabricate a plan and route to code generation
# -------------------------

class ExplodingLLM:
    def __init__(self, exc):
        self._exc = exc

    def invoke(self, *_a, **_k):
        raise self._exc


def _run_failing_turn(monkeypatch, prompt, exc):
    monkeypatch.setattr(planner_mod, "llm", ExplodingLLM(exc))
    return plan_etl_job({
        "messages": [HumanMessage(content=prompt)],
        "user_id": "u",
        "session_id": "s",
    })


def test_transient_llm_error_does_not_route_pleasantry_to_coder(monkeypatch):
    monkeypatch.delenv(planner_mod.FORCE_PLAN_ENV, raising=False)
    result = _run_failing_turn(monkeypatch, "thanks!", RuntimeError("rate limit exceeded"))
    assert not result.get("ready_to_code")
    assert not result.get("plan")
    assert result.get("coder_definition") == {}
    assert "try again" in _last_ai_content(result).lower()

    import app.api.workflow as wf
    assert wf.route_planner_output(result) == "continue_planning"


def test_tool_parse_error_does_not_fabricate_a_plan(monkeypatch):
    monkeypatch.delenv(planner_mod.FORCE_PLAN_ENV, raising=False)
    monkeypatch.setattr(planner_mod, "llm", FakeLLM("nonexistent_tool", {}))
    result = plan_etl_job({
        "messages": [HumanMessage(content="hello there")],
        "user_id": "u",
        "session_id": "s",
    })
    assert not result.get("ready_to_code")
    assert not result.get("plan")


def test_recovered_respond_to_user_text_is_surfaced(monkeypatch):
    exc = RuntimeError(
        '<function=respond_to_user {"response_text": "Happy to help with analysis!"} </function>'
    )
    result = _run_failing_turn(monkeypatch, "thanks!", exc)
    assert "Happy to help with analysis!" in _last_ai_content(result)
    assert not result.get("ready_to_code")


def test_force_plan_env_still_fabricates_on_failure(monkeypatch):
    monkeypatch.setenv(planner_mod.FORCE_PLAN_ENV, "1")
    result = _run_failing_turn(monkeypatch, "sum sales by region", RuntimeError("boom"))
    assert result.get("ready_to_code") is True
    assert result.get("plan")


def test_unhandled_tool_call_still_produces_a_reply(monkeypatch):
    # Simulate a future advertised-but-unhandled tool by dispatching an
    # existing parseable tool the elif chain knows, then removing its branch
    # coverage is impractical — instead register a fake mapping.
    from pydantic import BaseModel

    class NoArgs(BaseModel):
        pass

    monkeypatch.setitem(planner_mod.tool_name_to_param, "future_tool", NoArgs)
    monkeypatch.setattr(
        planner_mod.ToolCall, "model_fields", planner_mod.ToolCall.model_fields, raising=False
    )

    class LooseToolCall:
        def __init__(self):
            self.name = "future_tool"
            self.parameters = NoArgs()

    monkeypatch.setattr(planner_mod, "parse_tool_call", lambda tc: LooseToolCall())
    result = _run_tool_turn(monkeypatch, "future_tool", {})
    reply = _last_ai_content(result)
    assert reply, "unhandled tool call must not end the turn silently"
    assert "future_tool" in reply
