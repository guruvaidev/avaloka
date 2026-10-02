"""Regression tests: stub fallback reachability and the validation gate.

The coder's deterministic-stub branch fires only on hard failures at
retry_count >= 2, so the router must re-enter the coder at that count instead
of ending; and when the subgraph still ends with validation errors,
coding_subgraph_node must refuse to package the code for execution.
"""
import ast

import pytest

import app.agents.coder as coder_mod
import app.api.workflow as wf

PREVIEW = [
    {"OrderID": 1, "Product": "Laptop", "Price": 1200},
    {"OrderID": 2, "Product": "Mouse", "Price": 25},
]
SCHEMA = {"OrderID": "int64", "Product": "object", "Price": "int64"}


# -------------------------
# check_validation_status routing
# -------------------------

def test_clean_state_ends():
    assert wf.check_validation_status({"retry_count": 1}) == "end"


@pytest.mark.parametrize("retry", [0, 1])
def test_any_error_below_cap_refines(retry):
    assert wf.check_validation_status(
        {"syntax_error": True, "retry_count": retry}) == "refine"
    assert wf.check_validation_status(
        {"logical_semantic_error": True, "retry_count": retry}) == "refine"


def test_hard_error_at_two_refines_so_stub_is_reachable():
    assert wf.check_validation_status(
        {"syntax_error": True, "retry_count": 2}) == "refine"
    assert wf.check_validation_status(
        {"static_semantic_error": True, "retry_count": 2}) == "refine"


def test_soft_error_at_two_also_gets_a_final_constrained_pass():
    """Logical failures at the cap used to end here; they now refine once more.

    The coder constrains itself to plain pandas over schema columns when
    logical_semantic_error is set at this depth, so one flaky retry no longer
    becomes a permanent "failed validation after all retries" on a well-formed
    ask. retry_count 3 is still the hard stop -- see test_any_error_at_three_ends.
    """
    assert wf.check_validation_status(
        {"logical_semantic_error": True, "retry_count": 2}) == "refine"


def test_any_error_at_three_ends():
    assert wf.check_validation_status(
        {"syntax_error": True, "retry_count": 3}) == "end"


# -------------------------
# Coder stub branch fires on the extra pass
# -------------------------

def test_coder_emits_stub_on_hard_failure_at_retry_two(monkeypatch):
    class ExplodingLLM:
        def invoke(self, *_a, **_k):
            raise AssertionError("stub branch must return before the LLM is called")

    monkeypatch.setattr(coder_mod, "coder_llm", ExplodingLLM())
    result = coder_mod.coder_node({
        "plan": "Read the CSV file and save it to the output location.",
        "user_prompt": "Read the CSV file and save it.",
        "schema": SCHEMA,
        "uploaded_csv_preview": PREVIEW,
        "data_source_location": "input.csv",
        "output_location": "out.csv",
        "syntax_error": True,
        "retry_count": 2,
    })
    assert "stub" in result["llm_raw_response"].lower()
    tree = ast.parse(result["generated_code"])
    assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in ast.walk(tree))


# -------------------------
# Full subgraph loop: broken LLM output ends in a validated stub
# -------------------------

def test_loop_recovers_with_stub_after_repeated_hard_failures(monkeypatch):
    class Msg:
        content = "```python\ndef main(df:\n```"

    class BrokenLLM:
        def invoke(self, *_a, **_k):
            return Msg()

    monkeypatch.setattr(coder_mod, "coder_llm", BrokenLLM())
    graph = wf.build_coding_graph()
    final = graph.invoke({
        "plan": "Read the CSV file and save it to the output location.",
        "user_prompt": "Read the CSV file and save it.",
        "schema": SCHEMA,
        "uploaded_csv_preview": PREVIEW,
        "data_source_location": "input.csv",
        "output_location": "out.csv",
        "retry_count": 0,
    })
    assert not final.get("syntax_error")
    assert not final.get("static_semantic_error")
    assert not final.get("logical_semantic_error")
    ast.parse(final["generated_code"])


# -------------------------
# coding_subgraph_node refuses to package code that failed validation
# -------------------------

@pytest.fixture
def failing_subgraph(monkeypatch):
    class FakeGraph:
        def invoke(self, initial):
            return {
                "generated_code": "df['nope'].sum(",
                "llm_raw_response": "bad code",
                "syntax_error": True,
                "code_validation_feedback": "SyntaxError: unexpected EOF",
            }

    monkeypatch.setattr(wf, "build_coding_graph", lambda *a, **k: FakeGraph())
    monkeypatch.setattr(wf, "coding_graph", None, raising=False)
    monkeypatch.setattr(wf, "coder_node_is_mock", False, raising=False)
    monkeypatch.setattr(wf, "_detect_ambiguous_prompt_details", lambda prompt: None)


def test_empty_execution_result_is_checkpoint_serializable():
    # A raw DataFrame in graph state crashes the checkpointer's msgpack
    # serializer and 500s the request; the empty-result branch must emit
    # serializable records like every other branch.
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from app.agents.validator import execute_code_node

    code = (
        "import pandas as pd\n"
        "def main(df):\n"
        "    return df[df['Product'] == 'Nonexistent']\n"
    )
    result = execute_code_node({
        "generated_code": code,
        "uploaded_csv_preview": PREVIEW,
    })
    assert result["execution_error"] == "Execution returned an empty DataFrame."
    assert result["execution_output_data"] == []
    assert result["execution_output_preview"] == []
    JsonPlusSerializer().dumps_typed(result)


MATH_ERR = (
    "Cannot perform mathematical operation ('sum') on the 'country' column "
    "because it contains text/string values, not numbers."
)


@pytest.fixture
def blocking_subgraph(monkeypatch):
    class FakeGraph:
        def invoke(self, initial):
            return {
                "generated_code": "def main(df):\n    print('blocked')\n    return df",
                "llm_raw_response": MATH_ERR,
                "coder_blocked_reason": MATH_ERR,
                "code_validation_feedback": MATH_ERR,
            }

    monkeypatch.setattr(wf, "build_coding_graph", lambda *a, **k: FakeGraph())
    monkeypatch.setattr(wf, "coding_graph", None, raising=False)
    monkeypatch.setattr(wf, "coder_node_is_mock", False, raising=False)
    monkeypatch.setattr(wf, "_detect_ambiguous_prompt_details", lambda prompt: None)


def test_coder_block_becomes_the_chat_reply_and_skips_execution(blocking_subgraph):
    updates = wf.coding_subgraph_node({
        "user_prompt": "sum of country",
        "messages": [],
        "schema": {},
    })
    # No stub packaged, so the parent router ends instead of executing.
    assert updates["coder_definition"] == {}
    assert updates["generated_code"] == ""
    assert wf.route_after_code(updates) == "end"
    # The guard message is the LAST AI message — what the server shows in chat.
    assert getattr(updates["messages"][-1], "content", "") == MATH_ERR


def test_normal_coder_run_clears_stale_blocked_reason(monkeypatch):
    # The subgraph can run under the parent checkpointer, so a blocked-turn
    # value could carry into the next run; every non-blocked coder return
    # must overwrite it or old refusals would replay on healthy turns.
    import app.agents.coder as coder_mod

    monkeypatch.setattr(coder_mod, "coder_llm", None)
    result = coder_mod.coder_node({
        "plan": "Read the CSV file and save it to the output location.",
        "user_prompt": "Read the CSV file and save it.",
        "schema": SCHEMA,
        "uploaded_csv_preview": PREVIEW,
        "retry_count": 0,
        "coder_blocked_reason": "stale refusal from a previous run",
    })
    assert result["coder_blocked_reason"] is None


def test_coder_blocked_reason_survives_the_subgraph_boundary():
    from langgraph.graph import StateGraph, END
    from app.agents.state import CodingAgentState

    def blocker(state):
        return {"coder_blocked_reason": MATH_ERR}

    g = StateGraph(CodingAgentState)
    g.add_node("blocker", blocker)
    g.set_entry_point("blocker")
    g.add_edge("blocker", END)
    final = g.compile().invoke({"user_prompt": "x"})
    assert final.get("coder_blocked_reason") == MATH_ERR


def test_failed_validation_is_not_packaged_for_execution(failing_subgraph):
    updates = wf.coding_subgraph_node({
        "user_prompt": "sum a missing column",
        "messages": [],
        "schema": {},
    })
    assert updates["coder_definition"] == {}
    assert updates["generated_code"] == ""

    message = updates["execution_error"]
    # The user is told the request was not executed...
    assert "nothing was executed" in message
    # ...but the coder-facing feedback must NOT be pasted into chat. This used
    # to read "Last validation feedback: SyntaxError: ..." — a raw traceback
    # line in a user's conversation.
    assert "SyntaxError" not in message
    assert "Traceback" not in message

    # With no packaged code the parent router must end, not execute.
    assert wf.route_after_code(updates) == "end"
    assert any(message == getattr(m, "content", "") for m in updates["messages"])
