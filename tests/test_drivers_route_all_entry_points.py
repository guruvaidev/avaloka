"""rc12's "what are the strongest predictors of stroke?", from every entry point.

PR #581 answered the question from the shared driver ranking only in the web
chat's fast path. The planner graph -- the ``langgraph dev`` assistant / runs
API (app/api/langgraph_app.py), and the web chat's own GRAPH once a turn
reaches it -- still sent it through memory, the conversational agent, the
planner and the coder, and the swarm's merged answer never showed the table.

Here the same p1 question goes through the web chat, the langgraph_app graph
(with the state the web chat hands it, and with a bare runs-API state), and
the swarm, and all of them must give the same drivers table and figures.
Every graph node past the first raises if it is reached, so the coder, the
narrator and the verifier cannot have produced or rewritten a figure. An
explicit model request still reaches the coder through the graph. No
network, no LLM.
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.api.server as server

pytest.importorskip("celery", reason="production workflow requires the API scheduler dependencies")

try:
    from tests.server_harness import FakeGraph, SESSIONS, client, make_auth_headers, _quiet_send  # noqa: F401
    from tests.test_predictors_deterministic_route import (
        MODEL_ASK, QUESTION, _MustNotRunGraph, _ask, _db_thread, _table, _text, _upload_thread)
except ImportError:  # tests/ on sys.path
    from server_harness import FakeGraph, SESSIONS, client, make_auth_headers, _quiet_send  # noqa: F401
    from test_predictors_deterministic_route import (
        MODEL_ASK, QUESTION, _MustNotRunGraph, _ask, _db_thread, _table, _text, _upload_thread)


class CoderReached(AssertionError):
    """The question went past the graph's first node to the analysis pipeline."""


def _must_not_run(name: str):
    def node(state, *args, **kwargs):
        raise CoderReached(f"{name} ran: a drivers question must be answered by the graph's first node")
    return node


#: Every node a drivers question must not reach, and the coder itself.
DOWNSTREAM = ("memory_injection_node", "avaloka_agent_node", "plan_etl_job", "summarize_etl_job",
              "coding_subgraph_node", "coder_node", "result_narrator_node", "claim_verifier_node",
              "execution_agent_node_local", "visualization_agent_node")


@pytest.fixture(autouse=True)
def no_llm_key(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "GROQ_API_KEY", "OPENROUTER_API_KEY",
                 "OPENAI_API_KEY", "OPENAI_BASE_URL", "OLLAMA_HOST", "INFERENCE_LOCAL_BASE_URL",
                 "AVALOKA_LLM_PROVIDER", "AVALOKA_LLM_MODEL", "AVALOKA_LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def langgraph_app():
    """app/api/langgraph_app.py's ``graph`` -- what ``langgraph dev`` serves --
    built with every node past the first raising."""
    from app.api import workflow

    with patch.multiple(workflow, **{n: _must_not_run(n) for n in DOWNSTREAM}):
        import app.api.langgraph_app as module

        module = importlib.reload(module)
        yield module.graph
    importlib.reload(module)


class _CapturingGraph(FakeGraph):
    """Records the graph state the web chat builds for a turn."""

    def __init__(self):
        self.states: List[Dict[str, Any]] = []

    def invoke(self, state_in, config=None):
        self.states.append(copy.deepcopy(state_in))
        return super().invoke(state_in, config)


def _web_answer_and_state(client, monkeypatch, tmp_path, kind: str, user: str):
    """The web chat's answer to QUESTION on a thread, and the graph state the
    web chat builds on that thread (captured from a turn that reaches it),
    with QUESTION as the turn's message."""
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    thread = (_upload_thread(client, user, tmp_path) if kind == "upload"
              else _db_thread(client, monkeypatch, user, tmp_path, cap=2000))
    web = _ask(client, thread, QUESTION, user)

    capture = _CapturingGraph()
    monkeypatch.setattr(server, "GRAPH", capture, raising=False)
    _ask(client, thread, MODEL_ASK, user)
    assert capture.states, "the model request did not reach the graph"
    state = capture.states[-1]
    last = state["messages"][-1]
    assert MODEL_ASK in last.content
    # The same turn, asked as the drivers question (the web chat's run
    # context stays appended, as it is on every turn).
    state["messages"] = [HumanMessage(content=last.content.replace(MODEL_ASK, QUESTION))]
    return thread, web, state


def _final_text(final: Dict[str, Any]) -> str:
    return next(m for m in reversed(final["messages"]) if isinstance(m, AIMessage)).content


def _graph_table(final: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables = [t for t in final.get("output_tables") or [] if t.get("action_ids") == ["drivers_route"]]
    assert tables, "the graph's final state carries no drivers table"
    return tables[0]["rows"]


def _figures(rows: List[Dict[str, Any]]):
    return [(r["column"], r["measure"], r["association"]) for r in rows]


def _table_block(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if line.startswith("|"))


@pytest.mark.parametrize("kind", ["upload", "db"])
def test_the_graph_given_the_web_chats_state_says_exactly_what_the_web_chat_says(
        client, monkeypatch, tmp_path, langgraph_app, kind):
    _thread, web, state = _web_answer_and_state(client, monkeypatch, tmp_path, kind, f"drv-all-{kind}")

    final = langgraph_app.invoke(state)

    assert _figures(_graph_table(final)) == _figures(_table(web))
    assert final["output_json"] == _table(web)
    # Word for word, the basis line (N of M on the sampled DB thread) included.
    assert _final_text(final) == _text(web)
    if kind == "db":
        assert "a random sample of 2,000 of 5,110 rows" in _final_text(final)
    # The turn ended at the first node: nothing verified or rewrote it.
    assert final["verification_report"]["claims_checked"] == 0


def test_a_runs_api_state_gets_the_same_table_from_the_langgraph_app(client, monkeypatch, tmp_path, langgraph_app):
    """What a ``langgraph dev`` caller sends: the question and the file."""
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    thread = _upload_thread(client, "drv-all-runs", tmp_path)
    web = _ask(client, thread, QUESTION, "drv-all-runs")
    sess = next(s for s in SESSIONS.values() if s.get("dataset_id") == thread["dataset_id"])

    final = langgraph_app.invoke({"messages": [HumanMessage(content=QUESTION)],
                                  "data_source_location": str(sess["work_local_input"])})

    assert _figures(_graph_table(final)) == _figures(_table(web))
    assert _table_block(_final_text(final)) == _table_block(_text(web))
    # Same sentence, same figures; only the basis line may say less (the
    # runs API hands over no ingest row count).
    assert _final_text(final).splitlines()[:-1] == _text(web).splitlines()[:-1]
    assert _final_text(final).splitlines()[-1] == "Computed over all 5,110 rows."


def test_the_swarm_merges_to_the_same_table(client, monkeypatch, tmp_path):
    """"swarm this: <question>" from the chat: the team's merged answer, and
    the reply the chat sends, carry the web chat's table."""
    from app.api import chat_swarm
    from avaloka.coordinator import MainAvaloka
    from avaloka.swarm_trace import SwarmTraceRecorder

    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    thread = _upload_thread(client, "drv-all-swarm", tmp_path)
    web = _ask(client, thread, QUESTION, "drv-all-swarm")
    sess = next(s for s in SESSIONS.values() if s.get("dataset_id") == thread["dataset_id"])

    goal = chat_swarm.mission_goal(f"swarm this: {QUESTION}")
    rec = SwarmTraceRecorder()
    with patch("app.agents.coder.coder_node", _must_not_run("coder_node")):
        cr = MainAvaloka().coordinate(str(sess["work_local_input"]), goal=goal, output_dir=tmp_path / "swarm",
                                      workers=2, recorder=rec)

    assert cr.answer["kind"] == "drivers" and cr.answer["status"] == "answered"
    assert _figures(cr.answer.get("table") or []) == _figures(_table(web))
    reply = chat_swarm.summary_prompt(list(rec.records), edition=None, mission_id=None, goal=goal)["message"]
    assert _table_block(_text(web)) in reply, reply


def test_an_explicit_model_request_still_reaches_the_coder_through_the_graph(client, monkeypatch, tmp_path):
    from app.api import workflow

    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    thread = _upload_thread(client, "drv-all-model", tmp_path)
    sess = next(s for s in SESSIONS.values() if s.get("dataset_id") == thread["dataset_id"])
    reached: List[str] = []

    def coder(state):
        reached.append("code_etl")
        raise CoderReached("the coder ran")

    with patch.object(workflow, "memory_injection_node", return_value={}), \
            patch.object(workflow, "avaloka_agent_node", return_value={"delegate_to_planner": True}), \
            patch.object(workflow, "plan_etl_job", return_value={"ready_to_code": True}), \
            patch.object(workflow, "coding_subgraph_node", side_effect=coder):
        import app.api.langgraph_app as module

        graph = importlib.reload(module).graph
        with pytest.raises(CoderReached):
            graph.invoke({"messages": [HumanMessage(content=MODEL_ASK)],
                          "data_source_location": str(sess["work_local_input"])})
    importlib.reload(module)
    assert reached == ["code_etl"]
