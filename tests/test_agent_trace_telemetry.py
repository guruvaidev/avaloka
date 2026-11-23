"""The telemetry projection of the agent trace carries no customer content.

The trace itself is the user's analysis and is full of their data. The
telemetry projection must not be: these tests plant column names, a file name,
cell values, a question and an exception message across every field a step
has, run a real graph, and check the projected records (a) are accepted by the
ingest service's own validator and (b) contain none of what was planted.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage

from app.trace.recorder import instrument
from app.trace.telemetry import KNOWN_AGENTS, KNOWN_MODELS, KNOWN_PROVIDERS, agent_events
from telemetry_service import schema as ingest

PLANTED = [
    "salary_band_x91", "acme_payroll_2026.csv", "Zanzibar-Ward-7", "987654.321",
    "who earns the most in Zanzibar-Ward-7", "customer_secret_token",
]


def _planted_graph():
    from tests.test_agent_trace import CODE, _fake_llm, build_test_graph

    llm = instrument(_fake_llm(text="salary_band_x91 is highest in Zanzibar-Ward-7"),
                     agent="conversational", provider="openrouter", model="acme-private-finetune")
    graph = build_test_graph(llm=llm, fail_first=True)
    final = graph.invoke(
        {"messages": [HumanMessage(content="who earns the most in Zanzibar-Ward-7")],
         "user_id": "u", "session_id": "s",
         "data_source_location": "/data/acme_payroll_2026.csv",
         "uploaded_csv_columns": ["salary_band_x91"],
         "sample_data": [{"salary_band_x91": "987654.321", "ward": "Zanzibar-Ward-7"}],
         "analysis_fidelity": "entire_dataset"},
        config={"configurable": {"thread_id": "planted"}},
    )
    return final


def test_projection_is_accepted_by_the_ingest_schema():
    steps = _planted_graph()["agent_trace"]
    events = agent_events(steps)
    assert events, "nothing projected"
    ingest.validate_batch(events)  # raises Rejected on any unapproved field or value
    completed = [e for e in events if e["event_type"] == "agent.completed"]
    assert {e["agent"] for e in completed} >= {"memory_injection", "avaloka_agent", "code_etl", "coder"}


def test_projection_carries_none_of_the_planted_content():
    final = _planted_graph()
    # The trace does carry it -- it is the user's own analysis ...
    trace_text = json.dumps(final["agent_trace"], default=str)
    assert "Zanzibar-Ward-7" in trace_text and "acme_payroll_2026.csv" in trace_text
    # ... and the telemetry projection carries none of it.
    wire = json.dumps(agent_events(final["agent_trace"]))
    for planted in PLANTED + ["acme-private-finetune", "monthly_charges", "region", "churn.csv"]:
        assert planted not in wire, planted


def test_unknown_model_is_sent_as_other_and_the_thread_id_never_travels():
    final = _planted_graph()
    events = agent_events(final["agent_trace"])
    lead = next(e for e in events if e["event_type"] == "agent.completed" and e["agent"] == "avaloka_agent")
    assert lead["llm_model"] == "other" and lead["llm_provider"] == "openrouter"
    assert lead["tokens_in"] == 120 and lead["tokens_out"] == 30
    wire = json.dumps(events)
    assert final["trace_turn_id"] not in wire
    for step in final["agent_trace"]:
        assert step["step_id"] not in wire


def test_a_failure_sends_its_class_never_its_message():
    steps = [{"kind": "agent", "node": "execute_locally", "status": "error", "latency_ms": 5,
              "error_class": "ValueError", "inference": [],
              "decision": {"reason": "customer_secret_token in acme_payroll_2026.csv"}}]
    (started, completed) = agent_events(steps)
    assert completed["outcome"] == "failed" and completed["error_class"] == "ValueError"
    ingest.validate_batch([started, completed])
    assert "customer_secret_token" not in json.dumps([started, completed])


def test_a_node_the_ingest_does_not_know_is_dropped_not_renamed():
    assert agent_events([{"kind": "agent", "node": "acme_custom_node", "status": "ok"}]) == []


@pytest.mark.parametrize("bad_class", ["Error in /data/acme_payroll_2026.csv", "x" * 200])
def test_an_unshaped_error_class_is_replaced(bad_class):
    (_, completed) = agent_events([{"kind": "agent", "node": "coder", "status": "error",
                                    "error_class": bad_class, "inference": []}])
    assert completed["error_class"] == "Exception"
    ingest.validate(completed)


def test_vocabularies_agree_with_the_ingest_service():
    """Every value the projection can emit is one the ingest accepts.

    Agents are a subset, not equal: the ingest also names CLI fireflies and
    other specialists that are not graph nodes and never pass through this
    projection. test_every_traced_graph_node_can_be_reported covers the other
    direction. Providers and models are restated whole and held equal.
    """
    assert KNOWN_AGENTS <= set(ingest.AGENTS.values), KNOWN_AGENTS - set(ingest.AGENTS.values)
    assert KNOWN_PROVIDERS == set(ingest.LLM_PROVIDERS.values)
    assert KNOWN_MODELS == set(ingest.LLM_MODELS.values) - {"other"}


def test_every_traced_graph_node_can_be_reported():
    from app.api import workflow

    builder = workflow.build_graph()
    names = (set(builder.nodes) | set(workflow.build_coding_graph().get_graph().nodes)) - {"__start__", "__end__"}
    assert names <= KNOWN_AGENTS, names - KNOWN_AGENTS
