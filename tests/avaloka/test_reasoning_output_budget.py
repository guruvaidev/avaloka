"""The declared completion budget must be enforced after every model call."""

from avaloka.data_agent import (DataMissionRequest, DataQueryMission,
                                MissionPolicy, ModelResponse, ModelUsage)


class OversizedProvider:
    def complete(self, messages, tools, *, timeout_s):
        return ModelResponse(
            response_id="oversized",
            content="unsupported answer",
            tool_calls=[],
            usage=ModelUsage(
                prompt_tokens=2, completion_tokens=11, total_tokens=13
            ),
        )


class NoopTools:
    def specs(self):
        return []

    def execute(self, name, arguments, *, call_id):
        raise AssertionError("no tool should execute")

    def close(self):
        pass


def test_completion_budget_stops_oversized_response():
    request = DataMissionRequest(
        task_id="output-budget",
        question="answer",
        database_description="fixture",
        model="fake",
        policy=MissionPolicy(max_output_tokens=10),
    )
    result = DataQueryMission(OversizedProvider(), NoopTools()).run(request)
    assert result.termination_reason == "budget_exceeded"
    assert result.completion_tokens == 11
    assert "completion tokens" in (result.error or "")
