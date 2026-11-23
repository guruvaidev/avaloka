"""A crash on the sample run refines the code even with no LLM reviewer.

Before: ``check_validation_status`` did not read the sample run's
``execution_error``. A crash sent the code back only when the reviewer read it
and said so; with no reviewer configured the reviewer step approved, and code
that had already raised on the sample went on to the full run.
"""

import app.api.workflow as workflow_mod
from app.agents import validator as validator_mod

CRASH_CODE = """
import pandas as pd

def main(df):
    raise ValueError("BOOM_MARKER")
"""

GOOD_CODE = """
import pandas as pd

def main(df):
    df["total"] = df["a"] + df["b"]
    return df
"""

SAMPLE_CSV = "a,b\n1,2\n3,4\n"
SCHEMA = {"a": "int64", "b": "int64"}


def _run(monkeypatch, codes):
    calls = []

    def fake_coder(state):
        calls.append(dict(state))
        code = codes[min(len(calls) - 1, len(codes) - 1)]
        return {"generated_code": code, "retry_count": (state.get("retry_count") or 0) + 1}

    monkeypatch.setattr(workflow_mod, "coder_node", fake_coder)
    monkeypatch.setattr(validator_mod, "validator_llm", None)
    monkeypatch.delenv("AVALOKA_STRICT_LOGIC_VALIDATION", raising=False)
    final = workflow_mod.build_coding_graph().invoke({
        "user_prompt": "Add a total column", "plan": "Add a total column",
        "schema": SCHEMA, "sample_data": SAMPLE_CSV, "retry_count": 0,
    })
    return calls, final


def test_sample_crash_without_reviewer_goes_back_to_the_coder(monkeypatch):
    calls, final = _run(monkeypatch, [CRASH_CODE, GOOD_CODE])
    assert len(calls) == 2, "a crash on the sample was approved and sent to the full run"
    feedback = calls[1].get("code_validation_feedback") or ""
    assert "BOOM_MARKER" in feedback and "sample" in feedback
    assert final["execution_error"] is None
    assert final["execution_output_preview"][0]["total"] == 3


def test_a_clean_sample_run_is_not_sent_back(monkeypatch):
    calls, final = _run(monkeypatch, [GOOD_CODE])
    assert len(calls) == 1
    assert not final.get("sample_crash_error")


def test_a_crash_that_never_clears_stops_at_the_budget_and_still_goes_forward(monkeypatch):
    calls, final = _run(monkeypatch, [CRASH_CODE])
    # The same budget as the other validators: three coder passes, then end.
    assert len(calls) == 3
    assert final["sample_crash_error"] is True
    assert workflow_mod.check_validation_status(final) == "end"


def test_an_empty_result_is_not_a_crash(monkeypatch):
    empty = "import pandas as pd\n\ndef main(df):\n    return df.iloc[0:0]\n"
    calls, _ = _run(monkeypatch, [empty])
    assert len(calls) == 1
