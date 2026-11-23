""""So I offered options" is said only of a turn that offered options (2.0).

rc18b live regrade: the understanding card said "I read “What is the average
Glucose, and how many Glucose values are missing” as something to pin down
before I run anything, so I offered options" on a turn that ran code and
answered (rw7c nullxlsx), and on every run's "What tables do I have and how do
they relate?", which the planner answered directly. ``avaloka_intent ==
"clarification"`` is the keyword classifier's "no keyword matched": the planner
takes the turn. Options are what the turn records when it shows them
(``pending_suggestions``, ``pending_clarification``), and both outlive their
turn, so they count only when a step of this turn wrote them.

The turns are the rweb18b verify dumps' minimum fields
(tests/fixtures/chat_voice_live/rweb18b_offered.json).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from app.api import chat_voice

LIVE = json.loads((Path(__file__).parent / "fixtures" / "chat_voice_live" / "rweb18b_offered.json").read_text())
OFFERED = "so I offered options"


def _understanding(name: str, **final_overrides) -> str:
    case = copy.deepcopy(LIVE[name])
    case["final"].update(final_overrides)
    prompts = chat_voice.prompts_after_turn(case["final"], question=case["question"])
    card = next(p for p in prompts if p["step"] == "understanding")
    assert [s["kind"] for s in card["message_segments"]] == ["plain"]
    return card["message"]


def test_a_turn_that_ran_code_did_not_offer_options():
    case = LIVE["runc_nullxlsx_ask"]
    assert any(OFFERED in m for m in case["shown"])
    final = case["final"]
    assert final["avaloka_intent_reasoning"] == "no keyword matched"
    assert final["coder_definition"] and final["execution_result"]["status"] == "success"
    assert final["planner_definition"] == {}           # no named steps, so the intent label showed

    said = _understanding("runc_nullxlsx_ask")
    assert OFFERED not in said and "before I run anything" not in said, said
    assert said == "I read “What is the average Glucose, and how many Glucose values are missing” as a question for my planner."


def test_a_turn_answered_directly_did_not_offer_options():
    case = LIVE["runa_related_ask_tables"]
    assert any(OFFERED in m for m in case["shown"])
    nodes = [s["node"] for s in case["final"]["agent_trace"]]
    assert "plan_etl" in nodes and "code_etl" not in nodes
    said = _understanding("runa_related_ask_tables")
    assert OFFERED not in said, said
    assert said == "I read “What tables do I have and how do they relate” as a question for my planner."


def test_options_count_only_when_this_turn_wrote_them():
    final = LIVE["runa_related_ask_tables"]["final"]
    turn = final["trace_turn_id"]
    options = ["Show the full schema of both tables", "Join on customer_id"]
    # Left over from an earlier turn: present on the state, written by no step of this one.
    said = _understanding("runa_related_ask_tables", pending_suggestions=options)
    assert OFFERED not in said, said
    # Written this turn (the planner's suggest_analysis): she offered them.
    trace = final["agent_trace"] + [{"turn_id": turn, "node": "plan_etl", "written": ["pending_suggestions"]}]
    said = _understanding("runa_related_ask_tables", pending_suggestions=options, agent_trace=trace)
    assert said == ("I read “What tables do I have and how do they relate” as something to pin down before I run "
                    "anything, so I offered options.")
    # A clarifying question with its choices, asked this turn, is offering options too.
    clar = {"type": "metric_choice", "measure": "amount", "columns": ["amount", "net_amount"]}
    trace = final["agent_trace"] + [{"turn_id": turn, "node": "plan_etl", "written": ["pending_clarification"]}]
    assert OFFERED in _understanding("runa_related_ask_tables", pending_clarification=clar, agent_trace=trace)
    # ...but not on a turn that also wrote code.
    trace = final["agent_trace"] + [{"turn_id": turn, "node": "plan_etl", "written": ["pending_suggestions"]}]
    assert OFFERED not in _understanding("runa_related_ask_tables", pending_suggestions=options, agent_trace=trace,
                                         coder_definition={"code": "def main(df):\n    return df"})
