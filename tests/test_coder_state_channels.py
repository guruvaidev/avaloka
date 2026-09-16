"""Regression tests: coder state keys must not be dropped by LangGraph.

LangGraph silently drops any input or returned key that is not a declared
channel of the graph's state schema. These tests pin every key that must
survive the ETL graph and the coding subgraph boundaries.
"""
import ast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

import app.api.workflow as wf
from app.agents.coder import _format_datasets_context
from app.agents.state import CodingAgentState
from app.agents.validator import JoinSafetyVisitor
from app.graph.etl_state import ETLState

DATASETS = [
    {"dataset_id": "a", "alias": "left", "columns": ["id", "x"], "row_count": 10},
    {"dataset_id": "b", "alias": "right", "columns": ["id", "y"], "row_count": 20},
]


def _probe_graph(state_schema, keys):
    """Compile a one-node graph over state_schema that records which of
    `keys` survive channel filtering on input."""
    seen = {}

    def probe(state):
        seen.update({k: state.get(k) for k in keys})
        return {}

    g = StateGraph(state_schema)
    g.add_node("probe", probe)
    g.set_entry_point("probe")
    g.add_edge("probe", END)
    return g.compile(), seen


# -------------------------
# MAJ-028: multi-dataset channels reach the coding subgraph
# -------------------------

def test_coding_state_preserves_multi_dataset_and_fidelity_channels():
    graph, seen = _probe_graph(CodingAgentState, [
        "multi_dataset_state", "datasets_context",
        "analysis_fidelity", "selected_sample_name",
    ])
    graph.invoke({
        "user_prompt": "join them",
        "multi_dataset_state": DATASETS,
        "datasets_context": DATASETS,
        "analysis_fidelity": "entire_dataset",
        "selected_sample_name": "random_baseline",
    })
    assert seen["multi_dataset_state"] == DATASETS
    assert seen["datasets_context"] == DATASETS
    assert seen["analysis_fidelity"] == "entire_dataset"
    assert seen["selected_sample_name"] == "random_baseline"

    # is_multi logic in coder_node must now see both datasets
    ds = seen["datasets_context"] or seen["multi_dataset_state"] or []
    assert isinstance(ds, list) and len(ds) > 1


def test_format_datasets_context_prefers_datasets_context():
    out = _format_datasets_context({"datasets_context": DATASETS})
    assert "left" in out and "right" in out
    out = _format_datasets_context({"multi_dataset_state": DATASETS})
    assert "left" in out and "right" in out
    assert _format_datasets_context({}) == "[]"


# -------------------------
# MAJ-028: JoinSafetyVisitor with real metadata
# -------------------------

CROSS_JOIN_CODE = "import pandas as pd\nout = pd.merge(df, df2, how='cross')\n"


def _visit(schema, metadata, code):
    visitor = JoinSafetyVisitor(schema, metadata)
    visitor.visit(ast.parse(code))
    return visitor.errors


def test_small_cross_join_with_known_row_counts_is_allowed():
    assert _visit({"id": "int64"}, DATASETS, CROSS_JOIN_CODE) == []


def test_huge_cross_join_is_still_blocked():
    big = [{"dataset_id": "a", "row_count": 2_000_000},
           {"dataset_id": "b", "row_count": 2_000_000}]
    assert _visit({"id": "int64"}, big, CROSS_JOIN_CODE)


def test_cross_join_with_unknown_row_counts_is_still_blocked():
    assert _visit({"id": "int64"}, [], CROSS_JOIN_CODE)


def test_join_key_from_secondary_dataset_is_allowed():
    errors = _visit({"id": "int64", "x": "int64"}, DATASETS,
                    "merged = df2.merge(df3, on='y')\n")
    assert errors == []


def test_unknown_join_key_is_still_blocked():
    errors = _visit({"id": "int64", "x": "int64"}, DATASETS,
                    "merged = df.merge(df2, on='nope')\n")
    assert errors


# -------------------------
# MAJ-028/029: ETLState channels seeded by server.py must survive
# -------------------------

def test_etl_state_preserves_server_seeded_channels():
    graph, seen = _probe_graph(ETLState, [
        "datasets_context", "multi_dataset_state",
        "active_dataset_id", "active_dataset_ids",
        "file_size_bytes", "dataset_size_bytes",
        "memory_hints", "session_logic_signature",
    ])
    graph.invoke({
        "user_id": "u",
        "session_id": "s",
        "datasets_context": DATASETS,
        "multi_dataset_state": DATASETS,
        "active_dataset_id": "a",
        "active_dataset_ids": ["a", "b"],
        "file_size_bytes": 2 * 1024 ** 3,
        "dataset_size_bytes": 2 * 1024 ** 3,
        "memory_hints": ["prefers z-score"],
        "session_logic_signature": "sig-1",
    })
    assert seen["datasets_context"] == DATASETS
    assert seen["multi_dataset_state"] == DATASETS
    assert seen["active_dataset_id"] == "a"
    assert seen["active_dataset_ids"] == ["a", "b"]
    assert seen["file_size_bytes"] == 2 * 1024 ** 3
    assert seen["dataset_size_bytes"] == 2 * 1024 ** 3
    assert seen["memory_hints"] == ["prefers z-score"]
    assert seen["session_logic_signature"] == "sig-1"


# -------------------------
# MAJ-029: memory / messages / review-feedback channels in the subgraph
# -------------------------

def test_coding_state_preserves_memory_and_message_channels():
    messages = [HumanMessage(content="normalize C1"), AIMessage(content="ok")]
    graph, seen = _probe_graph(CodingAgentState, [
        "memory_hints", "session_logic_signature", "messages",
    ])
    graph.invoke({
        "user_prompt": "normalize C1",
        "memory_hints": ["user prefers min-max"],
        "session_logic_signature": "sig-2",
        "messages": messages,
    })
    assert seen["memory_hints"] == ["user prefers min-max"]
    assert seen["session_logic_signature"] == "sig-2"
    assert seen["messages"] == messages


def test_logical_review_feedback_survives_node_return():
    def reviewer(state):
        return {"logical_review_feedback": "consider dropna scope"}

    g = StateGraph(CodingAgentState)
    g.add_node("reviewer", reviewer)
    g.set_entry_point("reviewer")
    g.add_edge("reviewer", END)
    final = g.compile().invoke({"user_prompt": "x"})
    assert final.get("logical_review_feedback") == "consider dropna scope"


# -------------------------
# coding_subgraph_node boundary: keys passed in, results passed up
# -------------------------

@pytest.fixture
def fake_subgraph(monkeypatch):
    captured = {}
    result = {"generated_code": "x = 1", "llm_raw_response": None}

    class FakeGraph:
        def invoke(self, initial):
            captured.update(initial)
            return dict(result)

    monkeypatch.setattr(wf, "build_coding_graph", lambda *a, **k: FakeGraph())
    monkeypatch.setattr(wf, "coding_graph", None, raising=False)
    monkeypatch.setattr(wf, "coder_node_is_mock", False, raising=False)
    monkeypatch.setattr(wf, "_detect_ambiguous_prompt_details", lambda prompt: None)
    return captured, result


def test_coding_subgraph_node_passes_multi_dataset_and_memory_keys(fake_subgraph):
    captured, _ = fake_subgraph
    wf.coding_subgraph_node({
        "user_prompt": "join them",
        "messages": [],
        "schema": {},
        "multi_dataset_state": DATASETS,
        "datasets_context": DATASETS,
        "analysis_fidelity": "entire_dataset",
        "selected_sample_name": "random_baseline",
        "memory_hints": ["hint"],
        "session_logic_signature": "sig-3",
    })
    assert captured["multi_dataset_state"] == DATASETS
    assert captured["datasets_context"] == DATASETS
    assert captured["analysis_fidelity"] == "entire_dataset"
    assert captured["selected_sample_name"] == "random_baseline"
    assert captured["memory_hints"] == ["hint"]
    assert captured["session_logic_signature"] == "sig-3"


def test_coding_subgraph_node_propagates_logical_review_feedback(fake_subgraph):
    _, result = fake_subgraph
    result["logical_review_feedback"] = "soft note"
    updates = wf.coding_subgraph_node({
        "user_prompt": "clean data",
        "messages": [],
        "schema": {},
    })
    assert updates["logical_review_feedback"] == "soft note"


def test_entire_dataset_on_large_cloud_dataset_offloads_to_ray(fake_subgraph):
    updates = wf.coding_subgraph_node({
        "user_prompt": "full scan",
        "messages": [],
        "schema": {},
        "analysis_fidelity": "entire_dataset",
        "dataset_size_bytes": wf.LARGE_DATASET_THRESHOLD_BYTES,
        "data_source_location_cloud": "gs://bucket/data.csv",
        "connection_id": "conn-1",
    })
    assert updates["execution_mode"] == "k8s-ray"
    assert updates["task_schedule"]["task_type"] == "execute"


def test_entire_dataset_on_small_dataset_stays_local(fake_subgraph):
    updates = wf.coding_subgraph_node({
        "user_prompt": "full scan",
        "messages": [],
        "schema": {},
        "analysis_fidelity": "entire_dataset",
        "dataset_size_bytes": 1024,
    })
    assert updates["execution_mode"] == "local"
