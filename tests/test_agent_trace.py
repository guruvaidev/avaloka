"""The agent trace: recorded through a real LangGraph graph, served over HTTP.

Every test here drives real code paths: a ``StateGraph(ETLState)`` with the real
reducer and checkpointer, real chat-model callbacks, the real describers and
the real server routes. Nothing asserts on a hand-built trace.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.state import CodingAgentState
from app.graph.etl_state import ETLState
from app.trace import recorder
from app.trace.build import build_turn, planner_graph
from app.trace.describe import extract_operations
from app.trace.recorder import instrument, traced_node, traced_router

CODE = (
    "import pandas as pd\n"
    "df = pd.read_csv('churn.csv')\n"
    "df = df[df['monthly_charges'] > 0]\n"
    "df['annual'] = df['monthly_charges'] * 12\n"
    "result = df.groupby('region')['monthly_charges'].mean().reset_index()\n"
    "result.to_csv('out.csv', index=False)\n"
)
# What the coder wrote first when the first attempt fails validation: a
# different derivation and a different aggregate, so a graph built from the
# wrong attempt is told apart from one built from the code that ran.
CODE_ATTEMPT_1 = (
    "import pandas as pd\n"
    "df = pd.read_csv('churn.csv')\n"
    "df['annual_guess'] = df['monthly_charges'] * 10\n"
    "result = df.groupby('region')['monthly_charges'].median()\n"
)
RESULT = [{"region": "East", "monthly_charges": 61.5}, {"region": "West", "monthly_charges": 72.25}]
COLUMNS = ["customer_id", "tenure_months", "monthly_charges", "region", "churned"]
TRAINING_RESULT = {
    "model_name": "churn_predictor", "model_type": "classification", "runtime_s": 12.5,
    "model": {"target_column": "churned", "feature_names": ["tenure_months", "monthly_charges"]},
    "metrics": {"accuracy": 0.91, "f1": 0.88, "final_val_loss": 0.31, "rows_processed": 400},
    "evaluation_report": {
        "task": "classification", "splitter": "StratifiedKFold", "n_folds": 5, "n_rows": 400,
        "primary_metric": "accuracy",
        "metrics": {"accuracy": {"name": "accuracy", "mean": 0.91, "std": 0.01},
                    "f1_macro": {"name": "f1_macro", "mean": 0.88, "std": 0.02}},
        "baseline": {"accuracy": 0.73, "f1_macro": 0.42}, "beats_baseline": True,
        "lift_over_baseline": 0.18, "warnings": [],
    },
}


def _fake_llm(text: str = "ok", tokens=(120, 30)):
    msg = AIMessage(content=text, usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1],
                                                  "total_tokens": sum(tokens)})
    return GenericFakeChatModel(messages=iter([msg] * 10))


# --------------------------------------------------------------------------- #
# A small but real graph: entry, LLM-using agent, router, nested sub-graph
# --------------------------------------------------------------------------- #

def _coding_subgraph(fail_first: bool):
    attempts = {"n": 0}

    def coder(state):
        attempts["n"] += 1
        code = CODE_ATTEMPT_1 if (fail_first and attempts["n"] == 1) else CODE
        return {"generated_code": code, "retry_count": (state.get("retry_count") or 0) + (1 if attempts["n"] > 1 else 0)}

    def static(state):
        failed = fail_first and attempts["n"] == 1
        return {"static_semantic_error": failed,
                "code_validation_feedback": "groupby result is not reset" if failed else None}

    def route(state):
        return "refine" if state.get("static_semantic_error") else "end"

    g = StateGraph(CodingAgentState)
    g.add_node("coder", traced_node("coder", coder, nested=True))
    g.add_node("validator_static", traced_node("validator_static", static, nested=True))
    g.set_entry_point("coder")
    g.add_edge("coder", "validator_static")
    g.add_conditional_edges("validator_static",
                            traced_router("check_validation_status", route,
                                          signals=("static_semantic_error", "retry_count")),
                            {"refine": "coder", "end": END})
    return g.compile()


def build_test_graph(*, llm=None, fail_first=False, explode=False, rerun=False, training=False):
    """``rerun``: the route skips the coding team and runs the code already on
    the state. ``training``: the route goes to the model trainer instead."""
    sub = _coding_subgraph(fail_first)

    def memory(state):
        return {"memory_hints": ["region is a column"]}

    def lead(state):
        if llm is not None:
            llm.invoke([HumanMessage(content="classify")])
        return {"avaloka_intent": "statistical_analysis", "delegate_to_planner": True,
                "avaloka_intent_reasoning": "asks for an average by group",
                "avaloka_intent_confidence": 0.9,
                "avaloka_swarm": [{"clone": "coder", "name": "Ava-Coder", "role": "Analytics engineer",
                                   "focus": "writes the code", "status": "pending"},
                                  {"clone": "scientist", "name": "Ava-Scientist", "role": "ML",
                                   "focus": "", "status": "pending"}]}

    def code_etl(state):
        out = sub.invoke({"user_prompt": "avg", "plan": "", "retry_count": 0})
        return {"generated_code": out["generated_code"]}

    def execute(state):
        if explode:
            raise ValueError("secret_customer_value in /data/acme_payroll.csv")
        return {"execution_result": {"status": "success", "message": "done", "output_row_count": 2},
                "output_json": RESULT}

    def verify(state):
        return {"verification_report": {"safe_to_present": True, "claims_checked": 2, "n_problems": 0,
                                        "findings": [{"claim": "East averages 61.5", "verdict": "supported",
                                                      "check": "number", "detail": "", "evidence_ref": None}]}}

    def train(state):
        return {"training_plan": {"model_name": "churn_predictor", "model_type": "classification",
                                  "data_config": {"target_column": "churned",
                                                  "feature_columns": ["tenure_months", "monthly_charges"]}},
                "training_result": TRAINING_RESULT,
                "evaluation_report": TRAINING_RESULT["evaluation_report"],
                "evaluation_beats_baseline": True, "training_completed": True}

    g = StateGraph(ETLState)
    g.add_node("memory_injection", traced_node("memory_injection", memory, entry=True))
    g.add_node("avaloka_agent", traced_node("avaloka_agent", lead))
    g.add_node("code_etl", traced_node("code_etl", code_etl))
    g.add_node("execute_locally", traced_node("execute_locally", execute))
    g.add_node("train_models", traced_node("train_models", train))
    g.add_node("verify_claims", traced_node("verify_claims", verify))
    g.set_entry_point("memory_injection")
    g.add_edge("memory_injection", "avaloka_agent")
    first = "train_models" if training else ("execute_locally" if rerun else "code_etl")
    g.add_conditional_edges("avaloka_agent",
                            traced_router("route_avaloka",
                                          lambda s: "delegate_to_planner" if s.get("delegate_to_planner") else "end",
                                          signals=("avaloka_intent", "delegate_to_planner")),
                            {"delegate_to_planner": first, "end": "verify_claims"})
    g.add_edge("code_etl", "execute_locally")
    g.add_edge("execute_locally", "verify_claims")
    g.add_edge("train_models", "verify_claims")
    g.add_edge("verify_claims", END)
    return g.compile(checkpointer=MemorySaver())


def _run(graph, text="What is the average monthly_charges by region?", thread="t1", **state):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)], "user_id": "u", "session_id": "s",
         "data_source_location": "/tmp/churn.csv", "analysis_fidelity": "entire_dataset",
         "selected_sample_name": "random_baseline", "uploaded_csv_columns": COLUMNS, **state},
        config={"configurable": {"thread_id": thread}},
    )


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #

def test_every_node_and_route_lands_on_the_state_channel_in_order():
    llm = instrument(_fake_llm(), agent="conversational", provider="openrouter", model="openai/gpt-oss-120b")
    final = _run(build_test_graph(llm=llm))
    trace = final["agent_trace"]
    top = [(e["kind"], e["node"]) for e in trace if e.get("parent_id") is None]
    assert top == [("agent", "memory_injection"), ("agent", "avaloka_agent"),
                   ("decision", "route_avaloka"), ("agent", "code_etl"),
                   ("agent", "execute_locally"), ("agent", "verify_claims")]
    assert [e["seq"] for e in trace] == sorted(e["seq"] for e in trace)
    assert len({e["turn_id"] for e in trace}) == 1 == len({final["trace_turn_id"]})


def test_inference_is_attributed_to_the_step_that_made_the_call():
    llm = instrument(_fake_llm(tokens=(120, 30)), agent="conversational", provider="openrouter",
                     model="openai/gpt-oss-120b")
    final = _run(build_test_graph(llm=llm))
    by_node = {e["node"]: e for e in final["agent_trace"] if e["kind"] == "agent"}
    calls = by_node["avaloka_agent"]["inference"]
    assert len(calls) == 1
    call = calls[0]
    assert (call["provider"], call["model"], call["tokens_in"], call["tokens_out"]) == (
        "openrouter", "openai/gpt-oss-120b", 120, 30)
    assert call["outcome"] == "success" and call["latency_ms"] is not None
    assert call["cost_usd"] is None and call["cost_source"] is None  # unpriced, not guessed
    assert by_node["memory_injection"]["inference"] == []


def test_configured_price_is_applied_and_labelled(monkeypatch):
    monkeypatch.setenv("AVALOKA_MODEL_PRICES", json.dumps({"openai/gpt-oss-120b": [1.0, 2.0]}))
    llm = instrument(_fake_llm(tokens=(1_000_000, 500_000)), agent="c", provider="groq",
                     model="openai/gpt-oss-120b")
    final = _run(build_test_graph(llm=llm))
    call = next(e for e in final["agent_trace"] if e["node"] == "avaloka_agent")["inference"][0]
    assert call["cost_usd"] == pytest.approx(2.0) and call["cost_source"] == "price_table"


def test_subgraph_steps_nest_under_code_etl_and_record_the_retry():
    final = _run(build_test_graph(fail_first=True))
    code = next(e for e in final["agent_trace"] if e["node"] == "code_etl")
    kids = [e for e in final["agent_trace"] if e.get("parent_id") == code["step_id"]]
    assert [(k["kind"], k["node"]) for k in kids] == [
        ("agent", "coder"), ("agent", "validator_static"), ("decision", "check_validation_status"),
        ("agent", "coder"), ("agent", "validator_static"), ("decision", "check_validation_status")]
    first_check = kids[1]["decision"]
    assert first_check == {"type": "check", "check": "Static semantics", "value": "fail",
                           "reason": "groupby result is not reset"}
    assert kids[2]["decision"]["chosen"] == "refine"
    assert "back to the coder" in kids[2]["decision"]["reason"]
    assert kids[4]["decision"]["value"] == "pass" and kids[5]["decision"]["chosen"] == "end"


def test_route_records_its_reason_and_signals():
    final = _run(build_test_graph())
    route = next(e for e in final["agent_trace"] if e["kind"] == "decision" and e["node"] == "route_avaloka")
    d = route["decision"]
    assert d["chosen"] == "delegate_to_planner" and route["after"] == "avaloka_agent"
    assert "hands it to the planner" in d["reason"]
    assert d["signals"] == {"avaloka_intent": "statistical_analysis", "delegate_to_planner": True}


def test_decisions_carry_the_agents_own_reasons():
    final = _run(build_test_graph())
    lead = next(e for e in final["agent_trace"] if e["node"] == "avaloka_agent")
    assert lead["decision"]["value"] == "statistical_analysis"
    assert lead["decision"]["reason"] == "asks for an average by group"
    assert lead["decision"]["confidence"] == 0.9


def test_a_new_turn_replaces_the_checkpointed_trace():
    graph = build_test_graph()
    first = _run(graph, thread="same")
    second = _run(graph, text="again", thread="same")
    assert first["trace_turn_id"] != second["trace_turn_id"]
    assert {e["turn_id"] for e in second["agent_trace"]} == {second["trace_turn_id"]}
    assert len(second["agent_trace"]) == len(first["agent_trace"])


def test_a_failing_node_is_recorded_by_class_only_and_still_raises():
    graph = build_test_graph(explode=True)
    with pytest.raises(ValueError):
        _run(graph, thread="boom")
    snap = graph.get_state({"configurable": {"thread_id": "boom"}}).values
    parked = recorder.drain(snap["trace_turn_id"])
    failed = [e for e in parked if e.get("node") == "execute_locally"]
    assert failed and failed[0]["status"] == "error" and failed[0]["error_class"] == "ValueError"
    assert "secret_customer_value" not in json.dumps(failed[0])
    assert "acme_payroll" not in json.dumps(failed[0])


def test_recording_never_changes_what_a_node_returns():
    def node(state):
        return {"ready_to_code": True}

    wrapped = traced_node("plan_etl", node)
    out = wrapped({"trace_turn_id": "x", "ready_to_code": False})
    assert out["ready_to_code"] is True
    assert set(out) == {"ready_to_code", "agent_trace"}
    assert out["agent_trace"][-1]["written"] == ["ready_to_code"]


def test_the_real_graph_is_built_with_every_node_and_router_traced():
    """Fails if build_graph() stops wrapping a node: the trace would silently lose it."""
    from app.api import workflow

    builder = workflow.build_graph()
    for name, spec in builder.nodes.items():
        fn = getattr(spec.runnable, "func", None) or getattr(spec, "runnable", None)
        assert getattr(fn, "__avaloka_traced__", False), f"node {name} is not traced"
    for source, branches in builder.branches.items():
        for branch in branches.values():
            path = getattr(branch.path, "func", None) or branch.path
            assert getattr(path, "__avaloka_traced__", False), f"router after {source} is not traced"


# --------------------------------------------------------------------------- #
# Describing the calculation
# --------------------------------------------------------------------------- #

def test_operations_follow_the_data_not_the_text():
    """Seen on a real run: the script's entry point is at the bottom."""
    code = (
        "import pandas as pd\n"
        "def main(df):\n"
        "    df['monthly_charges'] = pd.to_numeric(df['monthly_charges'], errors='coerce')\n"
        "    return df.groupby('region', as_index=False).agg(avg=('monthly_charges', 'mean'))\n"
        "if __name__ == '__main__':\n"
        "    df = pd.read_csv('input.csv')\n"
        "    main(df).to_csv('output.csv', index=False)\n"
    )
    ops = extract_operations(code)
    assert [o["op"] for o in ops] == ["load", "cast", "group_aggregate", "write"]
    assert ops[1]["column"] == "monthly_charges"
    assert ops[2]["group_by"] == ["region"] and "agg(avg=('monthly_charges', 'mean'))" in ops[2]["aggregations"][0]


def test_operations_are_extracted_with_their_formulas():
    ops = extract_operations(CODE)
    kinds = [o["op"] for o in ops]
    assert kinds == ["load", "filter", "derive", "group_aggregate", "write"]
    agg = ops[3]
    assert agg["group_by"] == ["region"] and agg["aggregations"] == ["mean"]
    assert agg["formula"] == "df.groupby('region')['monthly_charges'].mean().reset_index()"
    assert ops[2]["column"] == "annual" and "* 12" in ops[2]["formula"]
    assert extract_operations("def broken(:") == []


def test_executor_step_carries_scope_and_small_result():
    final = _run(build_test_graph())
    ex = next(e for e in final["agent_trace"] if e["node"] == "execute_locally")
    calc = ex["calculation"]
    assert calc["values"]["output_rows"] == 2
    assert calc["result"]["columns"] == ["region", "monthly_charges"]
    assert calc["result"]["rows"] == [["East", 61.5], ["West", 72.25]]
    assert calc["data_scope"]["fidelity"] == "entire_dataset"
    assert calc["data_scope"]["sample"] == "random_baseline"


# --------------------------------------------------------------------------- #
# The API shapes
# --------------------------------------------------------------------------- #

def test_turn_has_handoffs_clones_and_totals():
    llm = instrument(_fake_llm(tokens=(100, 20)), agent="c", provider="openrouter", model="openai/gpt-oss-120b")
    final = _run(build_test_graph(llm=llm, fail_first=True))
    turn = build_turn(final["agent_trace"], turn_id=final["trace_turn_id"],
                      swarm_plan=final["avaloka_swarm"])
    top = [h for h in turn["handoffs"] if h["parent_id"] is None]
    assert top[1]["from"] == "avaloka_agent" and top[1]["to"] == "code_etl"
    assert top[1]["route"] == "delegate_to_planner"
    assert "avaloka_intent" in top[1]["passed"]
    clones = {c["clone"]: c for c in turn["swarm"]["clones"]}
    assert clones["coder"]["status"] == "done" and len(clones["coder"]["agent_steps"]) >= 2
    assert clones["scientist"]["status"] == "not_run"
    assert turn["totals"]["llm_calls"] == 1 and turn["totals"]["tokens_in"] == 100
    assert turn["totals"]["cost_usd"] is None and turn["totals"]["unpriced_calls"] == 1


def test_planner_graph_is_typed_layered_and_granular():
    final = _run(build_test_graph(fail_first=True))
    turn = build_turn(final["agent_trace"], turn_id=final["trace_turn_id"],
                      data_scope={"rows_analyzed": 400, "total_rows": 400})
    g = planner_graph(turn)
    types = {n["type"] for n in g["nodes"]}
    assert {"question", "agent", "decision", "data", "operation", "check", "result", "evidence"} <= types
    ids = {n["id"]: n for n in g["nodes"]}
    for e in g["edges"]:
        assert e["source"] in ids and e["target"] in ids
        assert ids[e["target"]]["layer"] > ids[e["source"]]["layer"], e
    retry = [e for e in g["edges"] if e["type"] == "retry"]
    assert len(retry) == 1 and ids[retry[0]["target"]]["label"] == "Coder · attempt 2"
    assert ids[retry[0]["source"]]["value"] == "fail"
    # The retry replaces, not duplicates, the edge between those two nodes.
    assert [e["type"] for e in g["edges"]
            if (e["source"], e["target"]) == (retry[0]["source"], retry[0]["target"])] == ["retry"]
    agg = next(n for n in g["nodes"] if n.get("op") == "group_aggregate")
    assert agg["label"] == "Group by region → mean"
    assert agg["formula"].startswith("df.groupby('region')")
    result = next(n for n in g["nodes"] if n["type"] == "result")
    assert result["table"]["rows"][0] == ["East", 61.5]
    checks = [n for n in g["nodes"] if n["type"] == "check"]
    assert [c["value"] for c in checks] == ["fail", "pass"]
    assert all(n.get("step_id") for n in g["nodes"] if n["type"] in ("agent", "check", "decision"))


def test_a_verifier_that_flagged_claims_does_not_read_as_plain_safe():
    """Seen on a real run: 'safe' alone hid three flagged claims."""
    steps = [{"kind": "agent", "step_id": "v", "parent_id": None, "seq": 1, "node": "verify_claims",
              "agent": "Claim verifier", "status": "ok", "latency_ms": 3, "inference": [],
              "decision": {"type": "evidence", "value": "safe"},
              "calculation": {"values": {"claims_checked": 5, "problems": 3}}}]
    g = planner_graph(build_turn(steps))
    node = next(n for n in g["nodes"] if n["type"] == "evidence")
    assert node["subtitle"] == "safe · 3 of 5 claims flagged"


def _ops(graph: Dict[str, Any]) -> list:
    return [n for n in graph["nodes"] if n["type"] == "operation"]


def test_operations_come_from_the_code_that_ran_not_the_failed_attempt():
    """Attempt 1 failed validation and was never run; attempt 2 was. The graph
    must show attempt 2's operations only, read from the executor's step."""
    final = _run(build_test_graph(fail_first=True))
    turn = build_turn(final["agent_trace"], turn_id=final["trace_turn_id"])
    g = planner_graph(turn)
    executor = next(s for s in turn["steps"] if s["node"] == "execute_locally")
    ops = _ops(g)
    assert ops, "no operation nodes"
    assert {o["source"] for o in ops} == {"executed"}
    assert {o["step_id"] for o in ops} == {executor["step_id"]}
    formulas = " ".join(o["formula"] for o in ops)
    assert ".mean()" in formulas and "df['annual'] = " in formulas
    assert "median" not in formulas and "annual_guess" not in formulas
    # The failed attempt is still in the trace, so it was there to be picked wrongly.
    first = next(s for s in turn["steps"] if s["node"] == "coder")
    assert any("median" in o["formula"] for o in first["calculation"]["operations"])


def test_a_rerun_without_a_coding_step_still_shows_what_ran():
    """Seen on a real run: 'Code already exists; run it' drew no operations at all."""
    final = _run(build_test_graph(rerun=True), generated_code=CODE)
    turn = build_turn(final["agent_trace"], turn_id=final["trace_turn_id"])
    g = planner_graph(turn)
    assert not any(n.get("node") == "code_etl" for n in g["nodes"])
    assert [o["op"] for o in _ops(g)] == ["filter", "derive", "group_aggregate"]
    assert {o["source"] for o in _ops(g)} == {"executed"}
    # The write is plumbing: it folds into the result rather than standing as a step.
    result = next(n for n in g["nodes"] if n["type"] == "result")
    assert result["values"]["written_to"] == "out.csv"


def _live_turn() -> Dict[str, Any]:
    import os

    path = os.path.join(os.path.dirname(__file__), "fixtures", "agent_trace_live_salaries.json")
    with open(path) as fh:
        return json.load(fh)["turn"]


def test_live_turn_operations_follow_the_executed_attempt_not_the_plan():
    """A real run (gemma4:e4b, the salaries sample). The plan named a column
    ``department`` that does not exist; the code that ran grouped by
    ``job_title``. The coder took two attempts, the first failing the logical
    review. The graph must read the second, and never the plan."""
    turn = _live_turn()
    g = planner_graph(turn, plan_steps=[{"name": "Calculate Average Salary by Department",
                                         "type": "aggregation", "columns": ["department", "salary"]}])
    ops = _ops(g)
    coders = [s for s in turn["steps"] if s["node"] == "coder"]
    assert len(coders) == 2
    # This trace predates the executor recording its code, so the last validated attempt is next in line.
    assert {o["source"] for o in ops} == {"validated"}
    assert {o["step_id"] for o in ops} == {coders[1]["step_id"]}
    text = json.dumps([n for n in g["nodes"] if n["lane"] == "data"])
    assert "department" not in text.lower().replace("top paying department", "")
    agg = next(o for o in ops if o["op"] == "group_aggregate")
    assert agg["label"].startswith("Group by job_title") and agg["columns"] == ["job_title", "salary"]
    labels = [o["label"] for o in ops]
    assert "Join" not in labels and "Write" not in labels
    assert "Append rows" in labels
    assert next(n for n in g["nodes"] if n["type"] == "result")["values"]["written_to"] == "output.csv"


def test_live_turn_data_lane_is_a_spine_not_an_afterthought():
    """Before: 35 nodes, 6 of them data. The data lane must be a connected
    path from the question to the evidence that the UI can show on its own."""
    g = planner_graph(_live_turn())
    spine = [n for n in g["nodes"] if n["type"] == "question" or n["lane"] == "data"]
    ids = {n["id"] for n in spine}
    edges = [e for e in g["edges"] if e["source"] in ids and e["target"] in ids]
    assert [n["type"] for n in sorted(spine, key=lambda n: n["layer"])] == [
        "question", "data", "data", "operation", "operation", "result"]
    reached = {"q"}
    for _ in spine:
        reached |= {e["target"] for e in edges if e["source"] in reached}
    assert reached == ids, "the spine is not connected from the question"
    assert any(e["target"] == next(n["step_id"] for n in g["nodes"] if n["type"] == "evidence")
               and e["type"] == "evidence" for e in g["edges"])


def test_operation_nodes_carry_the_columns_they_touch():
    final = _run(build_test_graph())
    g = planner_graph(build_turn(final["agent_trace"], turn_id=final["trace_turn_id"]))
    by_op = {o["op"]: o for o in _ops(g)}
    assert by_op["filter"]["columns"] == ["monthly_charges"]
    assert by_op["derive"]["columns"] == ["annual", "monthly_charges"]
    assert by_op["group_aggregate"]["columns"] == ["region", "monthly_charges"]
    assert by_op["filter"]["label"] == "Filter on monthly_charges"
    load = next(n for n in g["nodes"] if n["id"] == "data:load")
    assert load["columns"] == COLUMNS and load["values"]["columns"] == len(COLUMNS)
    assert "'churn.csv'" in load["formula"]


def test_the_question_points_at_the_data_it_acts_on():
    final = _run(build_test_graph())
    g = planner_graph(build_turn(final["agent_trace"], turn_id=final["trace_turn_id"]))
    assert any((e["source"], e["target"], e["type"]) == ("q", "data:load", "data") for e in g["edges"])


def test_a_training_turn_yields_train_and_evaluate_nodes_carrying_the_metric():
    final = _run(build_test_graph(training=True), text="Predict churned from the other columns")
    turn = build_turn(final["agent_trace"], turn_id=final["trace_turn_id"])
    g = planner_graph(turn)
    train = [n for n in g["nodes"] if n["type"] == "train"]
    assert len(train) == 1
    assert train[0]["label"] == "Train churn_predictor" and train[0]["subtitle"] == "classification"
    assert train[0]["columns"] == ["churned", "tenure_months", "monthly_charges"]
    assert train[0]["values"]["accuracy"] == 0.91 and train[0]["values"]["target"] == "churned"
    ev = next(n for n in g["nodes"] if n["type"] == "evaluate")
    assert ev["label"] == "Evaluate · accuracy" and ev["value"] == "beats_baseline"
    assert ev["values"]["accuracy"] == 0.91 and ev["values"]["baseline accuracy"] == 0.73
    assert ev["evaluation"]["primary_metric"] == "accuracy" and ev["evaluation"]["beats_baseline"] is True
    ids = {n["id"]: n for n in g["nodes"]}
    assert any((e["source"], e["target"]) == (train[0]["id"], ev["id"]) for e in g["edges"])
    assert any(e["source"] == ev["id"] and ids[e["target"]]["type"] == "evidence" for e in g["edges"])
    assert not any(n["type"] in ("operation", "result") for n in g["nodes"])
    trainer = next(s for s in turn["steps"] if s["node"] == "train_models")
    assert trainer["calculation"]["models"][0]["metrics"]["f1"] == 0.88


def test_query_filters_name_their_columns():
    ops = extract_operations("df = pd.read_csv('a.csv')\nout = df.query('tenure_months > 12 and churned == 1')\n",
                             known=COLUMNS)
    assert [o["op"] for o in ops] == ["load", "filter"]
    assert ops[1]["columns"] == ["tenure_months", "churned"]
    assert ops[0]["columns"] == []


# --------------------------------------------------------------------------- #
# Over HTTP
# --------------------------------------------------------------------------- #

try:  # tests/ as a package, or tests/ on sys.path
    from tests.server_harness import (  # noqa: F401
        FakeGraph, client, make_auth_headers, _do_upload, _quiet_send,
    )
except ImportError:  # pragma: no cover
    from server_harness import (  # noqa: F401
        FakeGraph, client, make_auth_headers, _do_upload, _quiet_send,
    )


class _TracingGraph(FakeGraph):
    """The harness's FakeGraph, plus the trace a real run would leave."""

    def __init__(self):
        self.graph = build_test_graph(fail_first=True)

    def invoke(self, state_in, config=None):
        final = super().invoke(state_in, config)
        traced = self.graph.invoke({"messages": state_in["messages"], "user_id": "u", "session_id": "s"},
                                   config={"configurable": {"thread_id": "http"}})
        for key in ("agent_trace", "trace_turn_id", "avaloka_swarm"):
            final[key] = traced[key]
        return final

    def get_state(self, config):
        return self.graph.get_state({"configurable": {"thread_id": "http"}})


def _chat(client, monkeypatch, user="trace-user"):
    import app.api.server as server

    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _TracingGraph(), raising=False)
    upload = _do_upload(client, user_id=user)
    resp = client.post(f"/threads/{upload['thread_id']}/messages",
                       json={"content": "What is the average monthly_charges by region?",
                             "metadata": {"dataset_id": upload["dataset_id"]}},
                       headers=make_auth_headers(user))
    assert resp.status_code == 200, resp.text
    return upload["thread_id"], resp.json()


def test_chat_response_links_its_trace_and_the_route_serves_it(client, monkeypatch):
    thread_id, body = _chat(client, monkeypatch)
    assert body["trace_turn_id"] and body["trace_url"].startswith(f"/threads/{thread_id}/trace")
    doc = client.get(body["trace_url"], headers=make_auth_headers("trace-user")).json()
    assert doc["schema_version"] == "1.0" and doc["thread_id"] == thread_id
    (turn,) = doc["turns"]
    assert turn["turn_id"] == body["trace_turn_id"]
    assert turn["question"] == "What is the average monthly_charges by region?"
    assert any(s["node"] == "code_etl" for s in turn["steps"])
    assert {a["node"] for a in doc["agents"]} >= {"avaloka_agent", "code_etl", "coder"}
    assert next(a for a in doc["agents"] if a["node"] == "coder")["retries"] == 1
    assert doc["swarm_limits"]["max_clones_per_mission"] == 2


def test_planner_graph_json_route_and_png_default(client, monkeypatch):
    thread_id, _ = _chat(client, monkeypatch)
    headers = make_auth_headers("trace-user")
    g = client.get(f"/threads/{thread_id}/planner-graph?format=json", headers=headers)
    assert g.status_code == 200, g.text
    doc = g.json()
    assert doc["thread_id"] == thread_id and any(n["type"] == "operation" for n in doc["nodes"])
    # The default is still the PNG route, which has no PNG for this thread.
    assert client.get(f"/threads/{thread_id}/planner-graph", headers=headers).status_code == 404


def test_trace_is_private_to_its_owner(client, monkeypatch):
    thread_id, _ = _chat(client, monkeypatch)
    other = make_auth_headers("someone-else")
    assert client.get(f"/threads/{thread_id}/trace", headers=other).status_code == 404
    assert client.get(f"/threads/{thread_id}/planner-graph?format=json", headers=other).status_code == 404
    assert client.get(f"/threads/{thread_id}/trace").status_code == 401


def test_agent_metrics_aggregate_the_callers_threads(client, monkeypatch):
    _chat(client, monkeypatch)
    doc = client.get("/api/agent-metrics", headers=make_auth_headers("trace-user")).json()
    assert doc["threads"] == 1 and doc["turns"] == 1
    coder = next(a for a in doc["agents"] if a["node"] == "coder")
    assert coder["activations"] == 2 and coder["succeeded"] == 2 and coder["retries"] == 1
    assert set(coder["latency_ms"]) == {"total", "p50", "max"}
    assert client.get("/api/agent-metrics", headers=make_auth_headers("nobody")).json()["agents"] == []


def test_build_chat_model_attaches_the_usage_recorder(monkeypatch):
    """The one place every agent's model is built must instrument it."""
    from app.core import inference

    monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-real-key")
    llm = inference.build_chat_model(role="planning", agent="PLANNER", groq_model="x", groq_api_key=None)
    assert llm is not None
    recorders = [h for h in (llm.callbacks or []) if isinstance(h, recorder.UsageRecorder)]
    assert len(recorders) == 1
    assert (recorders[0].agent, recorders[0].provider) == ("planner", "openrouter")
    assert recorders[0].model == "openai/gpt-oss-120b"


def test_a_string_join_is_not_a_data_join():
    """``", ".join(missing)`` in a guard clause once drew a "Join on ," node."""
    from app.trace.describe import extract_operations

    code = (
        "import os, pandas as pd\n"
        "df = pd.read_csv('salaries.csv')\n"
        "missing = [c for c in ['salary'] if c not in df.columns]\n"
        "if missing:\n"
        "    raise SystemExit(', '.join(missing))\n"
        "out = os.path.join('out', 'result.csv')\n"
        "joined = df.merge(df, on='job_title')\n"
        "joined.to_csv(out)\n"
    )
    ops = [o["op"] for o in extract_operations(code)]
    assert ops.count("join") == 1, ops
