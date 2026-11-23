"""Regression coverage for benchmark-derived timeout and privacy failures."""

from __future__ import annotations

import time

from avaloka.data_agent import (DataMissionRequest, DataQueryMission,
                                MissionLimits, MissionPolicy, PricingPolicy,
                                ToolObservation)
from avaloka.io import normalize_records
from avaloka.io.sources import _redact_uri


class NeverCalledProvider:
    def __init__(self):
        self.called = False

    def complete(self, messages, tools, *, timeout_s):
        self.called = True
        raise AssertionError("provider must not run")


class SlowTools:
    def __init__(self, delay=0.25):
        self.delay = delay
        self.closed = False

    def specs(self):
        return [{
            "type": "function",
            "function": {"name": "query_db", "parameters": {"type": "object"}},
        }]

    def execute(self, name, arguments, *, call_id):
        time.sleep(self.delay)
        return ToolObservation(True, "late")

    def close(self):
        self.closed = True


class OneToolProvider:
    def complete(self, messages, tools, *, timeout_s):
        from avaloka.data_agent import ModelResponse, ModelToolCall

        return ModelResponse(
            response_id="slow",
            content=None,
            tool_calls=[ModelToolCall("slow-1", "query_db", {"query": "slow"})],
        )


def _request(**kwargs):
    return DataMissionRequest(
        task_id="safety:1",
        question="read the source",
        database_description="fixture",
        model="fake",
        **kwargs,
    )


def test_cost_limit_fails_closed_without_configured_pricing():
    provider = NeverCalledProvider()
    result = DataQueryMission(provider, SlowTools()).run(_request(
        limits=MissionLimits(max_cost_usd=0.01)
    ))
    assert result.termination_reason == "pricing_unconfigured"
    assert result.error_kind == "pricing_unconfigured"
    assert provider.called is False


def test_synchronous_tool_call_cannot_block_past_policy_timeout():
    tools = SlowTools(delay=0.25)
    started = time.perf_counter()
    result = DataQueryMission(OneToolProvider(), tools).run(_request(
        policy=MissionPolicy(tool_timeout_s=0.01, max_iterations=2)
    ))
    elapsed = time.perf_counter() - started
    assert result.termination_reason == "timeout"
    assert result.error_kind == "timeout"
    assert elapsed < 0.20
    assert tools.closed


def test_tiny_tool_context_limit_is_exact():
    observation = ToolObservation(True, "x" * 1_000)
    bounded, truncated, digest = DataQueryMission._bound_observation(
        observation, 7
    )
    assert len(bounded.encode()) <= 7
    assert truncated is True
    assert len(digest) == 64


def test_lineage_redacts_users_queries_and_document_filters():
    sql = _redact_uri(
        "postgresql://analyst:secret@db.example/prod#query=SELECT%20email%20FROM%20users"
    )
    mongo = _redact_uri(
        "mongodb://analyst:secret@db.example/prod?authSource=admin"
        "#collection=users&filter=%7B%22email%22%3A%22private%40x.com%22%7D"
    )
    for redacted in (sql, mongo):
        assert "analyst" not in redacted
        assert "secret" not in redacted
        assert "email" not in redacted
        assert "private" not in redacted
    assert sql.endswith("#query=redacted")
    assert mongo.endswith("#collection=users")


def test_set_normalization_is_deterministic():
    first, _ = normalize_records([{"tags": {"b", "a", "c"}}])
    second, _ = normalize_records([{"tags": {"c", "b", "a"}}])
    assert first.loc[0, "tags"] == '["a","b","c"]'
    assert second.loc[0, "tags"] == first.loc[0, "tags"]
