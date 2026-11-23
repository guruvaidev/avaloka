"""The verifier's question is the user's words, not the context block the web chat appends.

claim_verifier._question returned the latest human message as the graph holds
it: the user's question followed by "[Analysis context]" and each dataset's
columns and preview rows (app/api/server.py). Every number in that block was
then "a number the question stated" (question_numbers), so a reply citing one
was taken for a restatement and never checked. On the rc15 related turn the
preview holds ``"amount": 327.43``: a reply saying SMB orders average $327.43
(the data says $182.27) was shown as fact.

The turn is the stored rc15 one (tests/fixtures/filter_disclosure_live), driven
through claim_verifier_node and presentable_reply to the text the user sees.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.claim_verifier import claim_verifier_node, presentable_reply

TURN = json.loads((Path(__file__).parent / "fixtures" / "filter_disclosure_live" / "rc15_turns.json")
                  .read_text(encoding="utf-8"))["turns"]["rw7"]

#: A figure that is in the context block's preview (one order's amount) and
#: nowhere in the result: SMB's mean is 182.27.
FALSE_CLAIM = "- SMB customers' orders average $327.43."


def _state(human: str, narrative: str):
    return {"messages": [HumanMessage(content=human), AIMessage(content=narrative)],
            "coder_definition": {"code": TURN["code"]}, "planner_definition": TURN["planner_definition"],
            "output_json": TURN["output_json"],
            "execution_result": {"status": "success", "fresh_output": True, "output_row_count": 3, "reads": []}}


def _shown(human: str, narrative: str):
    state = _state(human, narrative)
    report = claim_verifier_node(state)["verification_report"]
    return presentable_reply(narrative, report), report


def test_the_preview_number_is_in_the_live_message_only():
    assert '"amount": 327.43' in TURN["graph_human_message"]
    assert "327.43" not in TURN["question"]
    assert all(row["average_order_amount"] != "327.43" for row in TURN["output_json"])


def test_a_figure_from_the_context_preview_is_checked_not_taken_as_restated():
    narrative = TURN["narrative"] + "  \n" + FALSE_CLAIM
    text, report = _shown(TURN["graph_human_message"], narrative)
    assert "327.43" not in text
    assert "$545.08" in text                       # the supported figures stay
    assert any(f["claim"].startswith("- SMB customers' orders average") and f["check"] == "invented_number"
               for f in report["findings"])


def test_the_live_message_and_the_bare_question_get_the_same_verdicts():
    narrative = TURN["narrative"] + "  \n" + FALSE_CLAIM
    live_text, live = _shown(TURN["graph_human_message"], narrative)
    bare_text, bare = _shown(TURN["question"], narrative)
    assert live_text == bare_text
    assert [(f["claim"], f["verdict"], f["check"]) for f in live["findings"]] == \
        [(f["claim"], f["verdict"], f["check"]) for f in bare["findings"]]


def test_a_number_the_user_typed_is_still_a_restatement():
    human = TURN["graph_human_message"].replace(
        TURN["question"], "How does the average order amount differ by customer segment, against a $327.43 target?", 1)
    narrative = TURN["narrative"] + "  \n- None of the segments reaches the $327.43 target except enterprise."
    text, report = _shown(human, narrative)
    assert "$327.43 target except enterprise" in text
    assert not any("327.43" in f["claim"] and f["check"] == "invented_number" for f in report["findings"])
