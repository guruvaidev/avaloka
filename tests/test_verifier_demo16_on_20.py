"""The 1.6 demo guide's two false replies, end to end through 2.x's verifier.

The 1.6 demo run (develop-1.6 @ 7fd77f09, gpt-oss-120b) said "The lowest
average is for Autos & Vehicles at about 17.9\u202fM" (the table's lowest is
Travel & Events, 12,500,000) and "No category exceeds 30 million" (Trailers'
median is 39,000,000). On integration/2.0 @ c3b27651 this test failed 4 of 4:
Extremes read "17.9 M" as 17.9 and found no measure, and no check read a
threshold. Flagged means a finding on that sentence whose verdict is
contradicted or overreach, so safe_to_present is False and the sentence is
held back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.claim_verifier import claim_verifier_node

TURNS = json.loads((Path(__file__).parent / "fixtures" / "verifier_superlatives_16" / "turns.json").read_text())


def _report(turn, reply=None):
    state = {"messages": [HumanMessage(content=turn["question"]), AIMessage(content=reply or turn["reply"])],
             "execution_result": {"status": "success"},
             "output_json": turn["rows"]}
    out = claim_verifier_node(state)
    return out["verification_report"], out["verification_safe_to_present"]


@pytest.mark.parametrize("name", ["mean", "median"])
@pytest.mark.parametrize("gap", [" ", " "])
def test_demo16_false_claim_is_flagged_on_20(name, gap):
    turn = TURNS[name]
    reply = turn["reply"].replace(" ", gap)
    claim = turn["false_claim"].replace(" ", gap).lstrip("- ")
    report, safe = _report(turn, reply)
    hits = [f for f in report["findings"] if claim in f["claim"] and f["verdict"] in ("contradicted", "overreach")]
    print(json.dumps(report["findings"], indent=1, ensure_ascii=False))
    assert hits, f"{name}: not flagged; findings={[(f['claim'][:50], f['check'], f['verdict']) for f in report['findings']]}"
    assert safe is False
