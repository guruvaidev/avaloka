import pytest
from types import SimpleNamespace

import app.api.workflow as workflow_mod
from app.agents import validator as validator_mod


GOOD_CODE = """
import pandas as pd

def main(df):
    df["total"] = df["a"] + df["b"]
    print("computed total")
    return df
"""

CRASH_CODE = """
import pandas as pd

def main(df):
    raise ValueError("BOOM_MARKER")
"""

SAMPLE_CSV = "a,b\n1,2\n3,4\n"
SCHEMA = {"a": "int64", "b": "int64"}


def _fake_coder_returning(code):
    def fake_coder(state):
        return {
            "generated_code": code,
            "retry_count": (state.get("retry_count") or 0) + 1,
            "llm_raw_response": "generated",
        }
    return fake_coder


class FakeReviewer:
    """Approves unless the prompt shows the BOOM_MARKER execution crash."""

    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        if "BOOM_MARKER" in prompt:
            content = (
                '{"is_logically_correct": false, "is_fabricated": false,'
                ' "rationale": "runtime crash detected"}'
            )
        else:
            content = (
                '{"is_logically_correct": true, "is_fabricated": false,'
                ' "rationale": "ok"}'
            )
        return SimpleNamespace(content=content)


# -------------------------
# Graph wiring
# -------------------------

def test_execute_code_node_is_wired_into_coding_graph():
    graph = workflow_mod.build_coding_graph()
    nodes = list(graph.get_graph().nodes)
    edges = [(e.source, e.target) for e in graph.get_graph().edges]

    assert "execute_code" in nodes
    assert ("validator_static", "execute_code") in edges
    # The contract check sits between execution and the LLM reviewer: it reads
    # the executed result and gives a deterministic verdict before any tokens
    # are spent on judging.
    assert "validator_contract" in nodes
    assert ("execute_code", "validator_contract") in edges
    assert ("validator_contract", "validator_logical") in edges
    # The refine loop re-enters at the coder, so retried code is re-executed.
    assert ("validator_logical", "coder") in edges


def test_coding_graph_populates_execution_keys(monkeypatch):
    monkeypatch.setattr(workflow_mod, "coder_node", _fake_coder_returning(GOOD_CODE))
    monkeypatch.setattr(validator_mod, "validator_llm", None)
    monkeypatch.delenv("AVALOKA_STRICT_LOGIC_VALIDATION", raising=False)

    graph = workflow_mod.build_coding_graph()
    final = graph.invoke(
        {
            "user_prompt": "Add a total column",
            "plan": "Add a total column",
            "schema": SCHEMA,
            "sample_data": SAMPLE_CSV,
            "retry_count": 0,
        }
    )

    assert final["execution_error"] is None
    assert "computed total" in (final["execution_stdout"] or "")
    preview = final["execution_output_preview"]
    assert isinstance(preview, list) and preview
    assert preview[0]["total"] == 3
    assert final["execution_output_data"][1]["total"] == 7


def test_refine_loop_reexecutes_after_crash(monkeypatch):
    calls = []

    def fake_coder(state):
        calls.append(dict(state))
        code = CRASH_CODE if len(calls) == 1 else GOOD_CODE
        return {
            "generated_code": code,
            "retry_count": (state.get("retry_count") or 0) + 1,
        }

    reviewer = FakeReviewer()
    monkeypatch.setattr(workflow_mod, "coder_node", fake_coder)
    monkeypatch.setattr(validator_mod, "validator_llm", reviewer)
    monkeypatch.setenv("AVALOKA_STRICT_LOGIC_VALIDATION", "1")

    graph = workflow_mod.build_coding_graph()
    final = graph.invoke(
        {
            "user_prompt": "Add a total column",
            "plan": "Add a total column",
            "schema": SCHEMA,
            "sample_data": SAMPLE_CSV,
            "retry_count": 0,
        }
    )

    assert len(calls) == 2
    # The crash rationale must reach the coder as refinement feedback.
    assert calls[1].get("code_validation_feedback") == "runtime crash detected"
    # Validators may reformat the code, so compare content rather than the exact string.
    assert "BOOM_MARKER" not in final["generated_code"]
    assert "total" in final["generated_code"]
    assert final["execution_error"] is None
    assert final["execution_output_preview"][0]["total"] == 3
    assert final["retry_count"] == 2


# -------------------------
# execute_code_node behavior on realistic coder output
# -------------------------

def test_main_guard_stays_inert_during_sample_execution():
    code = """
import pandas as pd

def main(df):
    print("rows:", len(df))
    return df

if __name__ == "__main__":
    df = pd.read_csv("/nonexistent/real_data.csv")
    df = main(df)
    df.to_csv("/nonexistent/out.csv", index=False)
"""
    state = {
        "generated_code": code,
        "sample_data": SAMPLE_CSV,
        "syntax_error": False,
        "static_semantic_error": False,
    }
    out = validator_mod.execute_code_node(state)
    assert out["execution_error"] is None
    assert "rows: 2" in out["execution_stdout"]


def test_top_level_helpers_resolve_inside_main():
    code = """
import pandas as pd

def double_a(frame):
    frame["a"] = frame["a"] * 2
    return frame

def main(df):
    return double_a(df)
"""
    state = {
        "generated_code": code,
        "sample_data": SAMPLE_CSV,
        "syntax_error": False,
        "static_semantic_error": False,
    }
    out = validator_mod.execute_code_node(state)
    assert out["execution_error"] is None
    assert out["execution_output_preview"][0]["a"] == 2


def test_sample_unavailable_reported_not_raised():
    state = {
        "generated_code": GOOD_CODE,
        "sample_data": "",
        "uploaded_csv_preview": None,
        "syntax_error": False,
        "static_semantic_error": False,
    }
    out = validator_mod.execute_code_node(state)
    assert out["execution_error"] == "Sample data unavailable for execution"
    assert out["execution_output_data"] is None


def test_runtime_error_is_captured_in_state():
    state = {
        "generated_code": CRASH_CODE,
        "sample_data": SAMPLE_CSV,
        "syntax_error": False,
        "static_semantic_error": False,
    }
    out = validator_mod.execute_code_node(state)
    assert "BOOM_MARKER" in out["execution_error"]
    assert out["execution_output_data"] is None


def test_missing_main_is_reported():
    state = {
        "generated_code": "import pandas as pd\nx = 1\n",
        "sample_data": SAMPLE_CSV,
        "syntax_error": False,
        "static_semantic_error": False,
    }
    out = validator_mod.execute_code_node(state)
    assert "main()" in out["execution_error"]


def test_non_dataframe_return_is_flagged():
    code = """
import pandas as pd

def main(df):
    return df.to_dict(orient="records")
"""
    state = {
        "generated_code": code,
        "sample_data": SAMPLE_CSV,
        "syntax_error": False,
        "static_semantic_error": False,
    }
    out = validator_mod.execute_code_node(state)
    assert "must return a pandas DataFrame" in out["execution_error"]


# -------------------------
# Logical reviewer consumes execution results
# -------------------------

def test_logical_reviewer_prompt_includes_execution_results(monkeypatch):
    reviewer = FakeReviewer()
    monkeypatch.setattr(validator_mod, "validator_llm", reviewer)
    monkeypatch.delenv("AVALOKA_STRICT_LOGIC_VALIDATION", raising=False)

    state = {
        "generated_code": GOOD_CODE,
        "user_prompt": "Add a total column",
        "syntax_error": False,
        "static_semantic_error": False,
        "execution_stdout": "STDOUT_SENTINEL_ROWS_2",
        "execution_stderr": "",
        "execution_error": None,
        "execution_output_preview": [{"a": 1, "b": 2, "total": 3}],
    }
    out = validator_mod.logical_semantic_validator_node(state)

    assert len(reviewer.prompts) == 1
    assert "STDOUT_SENTINEL_ROWS_2" in reviewer.prompts[0]
    assert out["code_validation_feedback"] == "APPROVED"


def test_logical_reviewer_rejects_on_real_crash(monkeypatch):
    reviewer = FakeReviewer()
    monkeypatch.setattr(validator_mod, "validator_llm", reviewer)
    monkeypatch.setenv("AVALOKA_STRICT_LOGIC_VALIDATION", "1")

    state = {
        "generated_code": CRASH_CODE,
        "user_prompt": "Add a total column",
        "syntax_error": False,
        "static_semantic_error": False,
        "execution_stdout": "",
        "execution_stderr": "",
        "execution_error": "BOOM_MARKER",
        "execution_output_preview": [],
    }
    out = validator_mod.logical_semantic_validator_node(state)

    assert out["logical_semantic_error"] is True
    assert out["code_validation_feedback"] == "runtime crash detected"
