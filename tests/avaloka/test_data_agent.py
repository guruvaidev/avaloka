"""Offline tests for Avaloka's bounded evidence reasoning mission."""

from __future__ import annotations

import copy

from avaloka.data_agent import (DataMissionRequest, DataQueryMission,
                                MissionLimits, MissionPolicy, ModelResponse,
                                ModelToolCall, ModelUsage, PricingPolicy,
                                ToolObservation)
from avaloka.data_agent.mission import SYSTEM_PROMPT


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools, *, timeout_s):
        self.calls.append(copy.deepcopy(messages))
        return self.responses.pop(0)


class FakeTools:
    def __init__(self, observations=None):
        self.calls = []
        self.closed = False
        self.observations = list(observations or [])

    def specs(self):
        return [{
            "type": "function",
            "function": {"name": "query_db", "parameters": {"type": "object"}},
        }]

    def execute(self, name, arguments, *, call_id):
        self.calls.append((name, arguments, call_id))
        if self.observations:
            return self.observations.pop(0)
        return ToolObservation(
            True, 'result_key=var-1; result=[{"n":7}]',
            result_key="var-1", rows_read=1, bytes_read=8,
        )

    def close(self):
        self.closed = True


def _response(call_id, name, arguments, *, tokens=10):
    return ModelResponse(
        response_id=f"response-{call_id}",
        content=None,
        tool_calls=[ModelToolCall(call_id, name, arguments)],
        usage=ModelUsage(
            prompt_tokens=tokens - 2, completion_tokens=2, total_tokens=tokens
        ),
    )


def _request(**kwargs):
    return DataMissionRequest(
        task_id="fixture:1",
        question="What is n?",
        database_description="Read-only fixture with nested records",
        model="fake-model",
        **kwargs,
    )


def test_final_answer_requires_successful_tool_evidence():
    provider = ScriptedProvider([
        _response("premature", "return_answer", {"answer": "7"}),
        _response("query-1", "query_db", {"query": "select n"}),
        _response(
            "final", "return_answer",
            {"answer": "7", "evidence_refs": ["query-1"]},
        ),
    ])
    tools = FakeTools()

    result = DataQueryMission(provider, tools).run(_request())

    assert result.answer == "7"
    assert result.successful_tool_calls == 1
    assert result.evidence[0].tool_call_id == "query-1"
    assert result.retries == 1
    assert tools.closed


def test_tool_observation_is_bounded_but_full_content_is_hashed():
    huge = "header:" + ("x" * 10_000) + ":tail"
    provider = ScriptedProvider([
        _response("query-1", "query_db", {"query": "inspect"}),
        _response("final", "return_answer", {"answer": "done"}),
    ])
    tools = FakeTools([ToolObservation(True, huge, result_key="large")])

    result = DataQueryMission(provider, tools).run(_request(
        policy=MissionPolicy(max_prompt_tool_result_bytes=200)
    ))

    tool_message = provider.calls[1][-1]["content"]
    assert "truncated" in tool_message
    assert len(tool_message.encode()) < 260  # SUCCESS prefix plus bounded content
    assert result.evidence[0].content_truncated is True
    assert len(result.evidence[0].content_sha256) == 64


def test_data_scan_ratio_stops_after_observed_read():
    provider = ScriptedProvider([
        _response("query-1", "query_db", {"query": "scan"}),
    ])
    tools = FakeTools([
        ToolObservation(True, "rows", rows_read=10, bytes_read=60)
    ])

    result = DataQueryMission(provider, tools).run(_request(
        source_bytes=100,
        limits=MissionLimits(max_data_scanned_ratio=0.5),
    ))

    assert result.termination_reason == "budget_exceeded"
    assert result.data_scanned_ratio == 0.6
    assert result.error_kind == "budget_exceeded"


def test_repeated_equivalent_calls_stop_nonconvergence():
    arguments = {"query": "select n"}
    provider = ScriptedProvider([
        _response("one", "query_db", arguments),
        _response("two", "query_db", arguments),
        _response("three", "query_db", arguments),
    ])
    result = DataQueryMission(provider, FakeTools()).run(_request(
        policy=MissionPolicy(max_iterations=10, repeated_call_limit=3,
                             max_strategy_switches=0)
    ))

    assert result.termination_reason == "repeated_tool_call"
    assert result.error_kind == "tool_error"
    assert len(provider.calls) == 3


def test_tiered_pricing_is_computed_per_model_call():
    policy = PricingPolicy(
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
        threshold_tokens=5,
        input_usd_per_mtok_over_threshold=3.0,
        output_usd_per_mtok_over_threshold=4.0,
        version="test",
    )
    usage = ModelUsage(prompt_tokens=6, completion_tokens=2, total_tokens=8)
    assert policy.cost(usage) == (6 * 3.0 + 2 * 4.0) / 1_000_000


def test_prompt_directs_bounded_semantic_classification_without_package_probes():
    assert "semantic classification" in SYSTEM_PROMPT
    assert "bounded JSON batches" in SYSTEM_PROMPT
    assert "install packages" in SYSTEM_PROMPT


def test_progress_guidance_requests_synthesis_after_repeated_tool_errors():
    provider = ScriptedProvider([
        _response("good", "query_db", {"query": "select evidence"}),
        _response("bad-1", "query_db", {"query": "bad 1"}),
        _response("bad-2", "query_db", {"query": "bad 2"}),
        _response("bad-3", "query_db", {"query": "bad 3"}),
        _response("final", "return_answer", {"answer": "best supported"}),
    ])
    tools = FakeTools([
        ToolObservation(True, "one valid row", rows_read=1),
        ToolObservation(False, "bad query"),
        ToolObservation(False, "bad query"),
        ToolObservation(False, "bad query"),
    ])

    result = DataQueryMission(provider, tools).run(_request(
        policy=MissionPolicy(max_iterations=10),
    ))

    assert result.answer == "best supported"
    progress_messages = [
        message["content"] for message in provider.calls[4]
        if message.get("role") == "user"
        and message.get("content", "").startswith("PROGRESS CHECK")
    ]
    assert len(progress_messages) == 1
    assert "multiple tool attempts have failed" in progress_messages[0]
