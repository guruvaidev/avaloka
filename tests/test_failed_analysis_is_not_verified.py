"""A failed analysis is reported as failed, and nothing is "verified" about it.

From the web e2e (taxi Parquet, gemma4:e4b, 2026-10-18): the analysis failed
to read its input, and the turn's prompts then said

    "The run didn't finish: Script failed with exit status 1"
    "I checked 4 claims in my answer against the numbers; all of them hold."
    "That's my answer to “What is the average fare_amount by payment_type”."

The four "claims" were the sentences of the failure message, checked against
the validator's dry run on the sample. The trace marked execute_locally ok.

The failing case here is driven through the nodes build_graph() registers --
the same wrappers the running app uses -- with a real subprocess failure, not
a hand-made failed state, so a fix applied on some other path does not pass.
No LLM is involved: the coder's code is fixed.
"""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import HumanMessage

from app.agents.claim_verifier import Verdict, claim_verifier_node
from app.api import chat_voice

QUESTION = "What is the average fare_amount by payment_type?"
CODE = (
    "import pandas as pd\n"
    "def main(df):\n"
    "    return df.groupby('payment_type', as_index=False).agg("
    "average_fare_amount=('fare_amount', 'mean'))\n"
)
#: What the validator's dry run on the sample left on the state before execution.
DRY_RUN = [{"payment_type": 1, "average_fare_amount": 99.5}]


@pytest.fixture(scope="module")
def registered_nodes():
    from app.api import workflow

    builder = workflow.build_graph()

    def node(name):
        spec = builder.nodes[name]
        return getattr(spec.runnable, "func", None) or spec.runnable

    return workflow, node


def _texts(prompts):
    return " ".join(p.get("message", "") + " " + (p.get("question") or "") for p in prompts)


def test_a_failed_analysis_is_a_failed_step_in_plain_words(registered_nodes, tmp_path):
    workflow, node = registered_nodes
    broken = tmp_path / "trips.csv"
    # Line 3 has four fields under a two-column header: the ParserError of the e2e.
    broken.write_text("payment_type,fare_amount\n1,10.5\n2,7.25,8,9\n1,12.0\n")
    state = {
        "messages": [HumanMessage(content=QUESTION)],
        "coder_definition": {"code": CODE},
        "data_source_location": str(broken),
        "output_location": str(tmp_path / "out" / "output.csv"),
        "execution_output_data": DRY_RUN,
        "integrity_report": {"safe_to_train": True},
        "agent_errors": [],
    }

    after_exec = node("execute_locally")(state)

    # The trace marks the step failed, with a class and no value-bearing text.
    step = [s for s in after_exec["agent_trace"] if s.get("node") == "execute_locally"][-1]
    assert step["status"] == "error", step
    assert step["error_class"] == "AnalysisStepFailed"
    assert after_exec["execution_result"]["status"] == "failed"
    assert workflow.route_after_execution(after_exec) == "end"

    # The reply says what went wrong, not the mechanism.
    reply = after_exec["messages"][-1].content
    assert "I couldn't read the data file" in reply
    assert "ParserError" not in reply and "Expected 2 fields" not in reply

    verified = node("verify_claims")(after_exec)
    report = verified["verification_report"]
    assert report["claims_checked"] == 0 and report["verified"] is False
    assert report["analysis"] == "failed"
    assert verified["verification_safe_to_present"] is False

    prompts = chat_voice.prompts_after_turn({**after_exec, **verified}, question=QUESTION)
    text = _texts(prompts)
    assert "all of them hold" not in text
    assert "That's my answer" not in text
    assert "verification" not in [p["step"] for p in prompts]
    execution = [p for p in prompts if p["step"] == "execution"]
    assert execution and execution[0]["message"] == (
        "I couldn't finish the analysis: I couldn't read the data file.")


def test_the_verifier_refuses_when_the_result_is_empty():
    """A success with no rows has nothing to check a number against.

    The dry run carries the very number the narrative cites; it is not the
    result and must not make the claim "hold".
    """
    out = claim_verifier_node({
        "analysis_narrative": "The average fare for card payments is 99.5 across the trips.",
        "execution_result": {"status": "success", "output_row_count": 0},
        "output_json": [],
        "execution_output_data": DRY_RUN,
    })
    report = out["verification_report"]
    assert report["verified"] is False
    assert report["analysis"] == "no_result"
    assert report["claims_checked"] == 0
    assert not any(f["verdict"] == Verdict.SUPPORTED.value for f in report["findings"])
    assert out["verification_safe_to_present"] is False
    prompts = chat_voice.prompts_after_turn({**out, "execution_result": {"status": "success"},
                                             "output_json": []}, question=QUESTION)
    assert "all of them hold" not in _texts(prompts)
    assert "That's my answer" not in _texts(prompts)


def _answered(narrative):
    return claim_verifier_node({
        "analysis_narrative": narrative,
        "execution_result": {"status": "success", "output_row_count": 2},
        "output_json": [{"payment_type": 1, "average_fare_amount": 11.5},
                        {"payment_type": 2, "average_fare_amount": 7.38}],
        "execution_output_data": DRY_RUN,
    })


def test_a_claim_whose_number_is_not_in_the_result_is_rejected():
    """99.5 is in the dry run, not in the result: rejected, not verified."""
    out = _answered("The average fare for card payments is 99.5 across the trips.")
    report = out["verification_report"]
    assert report["goal_answered"] is True
    assert report["claims_reconciled"] is False
    assert report["verified"] is False
    assert [f["check"] for f in report["findings"]] == ["invented_number"]
    assert report["findings"][0]["verdict"] == Verdict.UNSUPPORTED.value


def test_a_claim_whose_number_is_in_the_result_is_verified_and_said_so():
    """The positive control: the gate is not just closed."""
    out = _answered("The average fare for card payments is 11.5 across the trips.")
    report = out["verification_report"]
    assert report["verified"] is True and report["claims_reconciled"] is True
    final = {**out, "execution_result": {"status": "success", "output_row_count": 2},
             "output_json": [{"payment_type": 1, "average_fare_amount": 11.5}]}
    text = _texts(chat_voice.prompts_after_turn(final, question=QUESTION))
    assert re.search(r"I checked \d+ claims? in my answer against the numbers\.", text), text
    assert "all of them hold" not in text  # the owner's ruling: the count, no verdict
    assert "That's my answer" in text
