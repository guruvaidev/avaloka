"""A provider that rejects the model's tool call gets a re-prompt, not a give-up.

DataAgentBench, first column on integration/2.0 (Groq openai/gpt-oss-120b):
17 of 54 queries ended in HTTP 400 ``tool_use_failed`` -- Groq's parser could
not read the generation (a phantom "json" tool) -- and the runtime retried the
identical payload three times, then the mission ended as ``provider_error``.
The same defect class as the MTA's empty tool-call reply: the model did not
produce a usable tool call, and nobody told it so.

Hermetic: a scripted provider raises ``ToolCallRejected`` the way
``scripts/avaloka_dab_agent.py::OpenAICompatibleProvider.complete`` now does
on that 400, and the mission is the real one.
"""

from __future__ import annotations

import copy

import pytest

from avaloka.data_agent import (DataMissionRequest, DataQueryMission, MissionPolicy,
                                ModelResponse, ModelToolCall, ModelUsage,
                                ProviderError, ToolCallRejected, ToolObservation)
from avaloka.data_agent.contracts import tool_use_failed_text

PHANTOM = '{"name": "json", "arguments": {"query": "select count(*) from acme_payroll"}}'


class RejectingProvider:
    """Raises ToolCallRejected for each leading ``None`` in the script, then
    answers from the remaining responses."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def complete(self, messages, tools, *, timeout_s):
        self.calls.append(copy.deepcopy(messages))
        item = self.script.pop(0)
        if item is None:
            raise ToolCallRejected(
                "provider rejected the model's tool call: BadRequestError: "
                "Error code: 400 - {'error': {'code': 'tool_use_failed'}}",
                failed_generation=PHANTOM,
            )
        return item


class FakeTools:
    def __init__(self):
        self.calls = []
        self.closed = False

    def specs(self):
        return [{"type": "function",
                 "function": {"name": "query_db", "parameters": {"type": "object"}}}]

    def execute(self, name, arguments, *, call_id):
        self.calls.append((name, arguments, call_id))
        return ToolObservation(True, 'result_key=var-1; result=[{"n":7}]',
                               result_key="var-1", rows_read=1, bytes_read=8)

    def close(self):
        self.closed = True


def _response(call_id, name, arguments):
    return ModelResponse(response_id=f"response-{call_id}", content=None,
                         tool_calls=[ModelToolCall(call_id, name, arguments)],
                         usage=ModelUsage(prompt_tokens=8, completion_tokens=2, total_tokens=10))


def _request(**kwargs):
    return DataMissionRequest(task_id="fixture:1", question="What is n?",
                              database_description="Read-only fixture", model="fake-model",
                              **kwargs)


def test_a_rejected_tool_call_is_re_prompted_and_the_mission_completes():
    provider = RejectingProvider([
        None,                                                     # 400 tool_use_failed
        _response("query-1", "query_db", {"query": "select n"}),  # after the re-prompt
        _response("final", "return_answer", {"answer": "7", "evidence_refs": ["query-1"]}),
    ])
    tools = FakeTools()

    result = DataQueryMission(provider, tools).run(_request())

    assert result.answer == "7"
    assert result.error is None and result.error_kind is None
    assert result.retries == 1
    assert len(provider.calls) == 3
    # The re-prompt tells the model what went wrong and what a valid reply is.
    nudge = provider.calls[1][-1]
    assert nudge["role"] == "user"
    assert "could not parse it as a tool call" in nudge["content"]
    assert "exact name" in nudge["content"] and "return_answer" in nudge["content"]
    assert provider.calls[0] == provider.calls[1][:-1]
    events = [e for e in result.trajectory if e["kind"] == "tool_call_rejected"]
    assert events == [{"kind": "tool_call_rejected", "iteration": 1, "count": 1}]
    assert tools.closed


def test_rejections_beyond_the_bound_end_as_provider_error_with_the_class():
    provider = RejectingProvider([None, None, None])
    tools = FakeTools()

    result = DataQueryMission(provider, tools).run(
        _request(policy=MissionPolicy(max_tool_call_rejections=2)))

    assert result.answer is None
    assert result.termination_reason == "provider_error"
    assert result.error_kind == "provider_error"
    assert result.error.startswith("provider_error: ToolCallRejected: ")
    assert result.retries == 2 and len(provider.calls) == 3
    assert [e["count"] for e in result.trajectory if e["kind"] == "tool_call_rejected"] == [1, 2, 3]
    assert tools.closed


def test_the_models_failed_generation_never_enters_the_trajectory():
    provider = RejectingProvider([None, None, None])
    result = DataQueryMission(provider, FakeTools()).run(_request())
    import json
    assert "acme_payroll" not in json.dumps(result.trajectory)
    assert "acme_payroll" not in (result.error or "")


def test_tool_call_rejected_is_a_provider_error_so_old_boundaries_still_hold():
    assert issubclass(ToolCallRejected, ProviderError)


@pytest.mark.parametrize("body", [
    {"error": {"code": "tool_use_failed", "message": "Failed to call a function.",
               "failed_generation": PHANTOM}},
    {"code": "tool_use_failed", "failed_generation": PHANTOM},
])
def test_classifier_reads_the_groq_body_shapes(body):
    class BadRequestError(Exception):
        pass

    exc = BadRequestError("Error code: 400")
    exc.body = body
    assert tool_use_failed_text(exc) == PHANTOM


def test_classifier_leaves_other_errors_alone():
    class RateLimitError(Exception):
        pass

    exc = RateLimitError("429")
    exc.body = {"error": {"code": "rate_limit_exceeded"}}
    assert tool_use_failed_text(exc) is None
    assert tool_use_failed_text(ValueError("x")) is None
