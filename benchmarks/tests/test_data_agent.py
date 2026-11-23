"""Offline tests for Avaloka's benchmark-facing reasoning mission."""

from __future__ import annotations

from types import SimpleNamespace

from avaloka.data_agent import (DataMissionRequest, DataQueryMission, MissionLimits,
                                MissionPolicy, ModelResponse, ModelToolCall,
                                ModelUsage, PricingPolicy, ProviderError,
                                ToolObservation)


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, tools, *, timeout_s):
        self.calls += 1
        return self.responses.pop(0)


class FakeTools:
    def __init__(self):
        self.calls = []
        self.closed = False

    def specs(self):
        return [{"type": "function", "function": {"name": "query_db",
                "parameters": {"type": "object"}}}]

    def execute(self, name, arguments, *, call_id):
        self.calls.append((name, arguments, call_id))
        return ToolObservation(True, "result_key=var_call-1; result=[{\"n\": 7}]",
                               result_key="var_call-1", rows_read=1, bytes_read=8)

    def close(self):
        self.closed = True


def _response(call_id, name, arguments, *, tokens=10):
    return ModelResponse(
        response_id=f"response-{call_id}",
        content=None,
        tool_calls=[ModelToolCall(call_id, name, arguments)],
        usage=ModelUsage(prompt_tokens=tokens - 2, completion_tokens=2,
                         total_tokens=tokens),
    )


def _request(**kwargs):
    return DataMissionRequest(
        task_id="dab:fixture:1",
        question="What is n?",
        database_description="db is a read-only fixture",
        model="fake-model",
        **kwargs,
    )


def test_mission_preserves_tool_call_ids_and_returns_supported_answer():
    provider = ScriptedProvider([
        _response("call-1", "query_db", {"db_name": "db", "query": "select n"}),
        _response("call-2", "return_answer", {"answer": "7"}),
    ])
    tools = FakeTools()
    result = DataQueryMission(provider, tools).run(_request())
    assert result.answer == "7"
    assert result.termination_reason == "return_answer"
    assert tools.calls[0][2] == "call-1"
    assert [event["tool_call_ids"] for event in result.trajectory
            if event["kind"] == "model"] == [["call-1"], ["call-2"]]
    assert tools.closed


def test_repeated_equivalent_tool_calls_terminate_early():
    call = {"db_name": "db", "query": "select n"}
    provider = ScriptedProvider([
        _response("call-1", "query_db", call),
        _response("call-2", "query_db", call),
        _response("call-3", "query_db", call),
    ])
    result = DataQueryMission(provider, FakeTools()).run(_request(
        policy=MissionPolicy(max_iterations=10, repeated_call_limit=3,
                             max_strategy_switches=0)
    ))
    assert result.error_kind == "tool_error"
    assert result.termination_reason == "repeated_tool_call"
    assert provider.calls == 3


def test_token_budget_stops_after_the_call_that_reports_usage():
    provider = ScriptedProvider([
        _response("call-1", "query_db", {"db_name": "db", "query": "select n"},
                  tokens=11),
    ])
    result = DataQueryMission(provider, FakeTools()).run(_request(
        limits=MissionLimits(max_tokens=10)
    ))
    assert result.error_kind == "budget_exceeded"
    assert result.total_tokens == 11


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


def test_provider_error_returns_a_classified_result():
    class FailingProvider:
        def complete(self, messages, tools, *, timeout_s):
            raise ProviderError("provider unavailable")

    result = DataQueryMission(FailingProvider(), FakeTools()).run(_request())
    assert result.termination_reason == "provider_error"
    assert result.error_kind == "provider_error"
    assert "provider unavailable" in result.error


def test_assistant_message_replays_opaque_provider_tool_metadata():
    response = ModelResponse(
        response_id="response-1",
        content=None,
        tool_calls=[ModelToolCall(
            "call-1",
            "query_db",
            {"db_name": "db", "query": "select n"},
            provider_metadata={
                "extra_content": {"google": {"thought_signature": "opaque"}}
            },
        )],
    )
    message = DataQueryMission._assistant_tool_message(response)
    assert message["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "opaque"}
    }


def test_openai_compatible_adapter_extracts_gemini_thought_signature(tmp_path):
    from scripts.avaloka_dab_agent import OpenAICompatibleProvider

    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="query_db", arguments='{"query":"x"}'),
        model_dump=lambda **kwargs: {
            "extra_content": {"google": {"thought_signature": "opaque"}}
        },
    )
    response = SimpleNamespace(
        id="response-1",
        usage=SimpleNamespace(
            prompt_tokens=4,
            completion_tokens=2,
            total_tokens=6,
            prompt_tokens_details=None,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
        ))],
    )

    class Completions:
        @staticmethod
        def create(**kwargs):
            return response

    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    provider.model = "gemini-3.1-pro-preview"
    provider.log_path = tmp_path / "model_calls.jsonl"
    provider.max_output_tokens = 128
    provider.retries = 0

    parsed = provider.complete([], [], timeout_s=1)

    assert parsed.tool_calls[0].provider_metadata == {
        "extra_content": {"google": {"thought_signature": "opaque"}}
    }
