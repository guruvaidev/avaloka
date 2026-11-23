"""The execution profile is merged into ``avaloka_recommendation``, not put in
its place.

The field reaches the reply as ``<recommendation>`` and can carry verified
analysis results beside the profile. ``decide_execution_profile`` replaced it
whenever a dataset was bound, so a question asked with results in hand lost
them on the way to the reply.
"""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage

from app.agents.avaloka_agent.execution_profile import SMALL_THRESHOLD, decide_execution_profile

EVIDENCE = {"source": "verified analysis results for the connected dataset",
            "results": {"true_value": 37.31, "asserted_value": 42.5}}


def _bound(**extra: Any) -> Dict[str, Any]:
    return {"dataset_size_bytes": SMALL_THRESHOLD - 1, **extra}


def test_the_evidence_survives_the_profile() -> None:
    """Fails on the base: the profile replaced the field and the results were gone."""
    out = decide_execution_profile(_bound(avaloka_recommendation=dict(EVIDENCE)), "statistical_analysis")
    rec = out["avaloka_recommendation"]
    assert rec.get("results") == EVIDENCE["results"]
    assert rec.get("source") == EVIDENCE["source"]
    assert rec["fidelity"] == "entire_dataset" and rec["execution_mode"] == "local"


def test_the_profile_keys_are_this_turns() -> None:
    stale = {**EVIDENCE, "fidelity": "quick_sample", "execution_mode": "k8s-ray", "eta_text": "hours"}
    rec = decide_execution_profile(_bound(avaloka_recommendation=stale), "statistical_analysis")[
        "avaloka_recommendation"]
    assert rec["fidelity"] == "entire_dataset" and rec["execution_mode"] == "local"
    assert rec["eta_text"] == "seconds to minutes"
    assert rec.get("results") == EVIDENCE["results"]


def test_the_callers_dict_is_not_mutated() -> None:
    original = dict(EVIDENCE)
    decide_execution_profile(_bound(avaloka_recommendation=original), "statistical_analysis")
    assert original == EVIDENCE


def test_no_recommendation_yet_is_the_profile_alone() -> None:
    rec = decide_execution_profile(_bound(), "statistical_analysis")["avaloka_recommendation"]
    assert "results" not in rec and rec["fidelity"] == "entire_dataset"


def test_no_bound_dataset_leaves_the_field_alone() -> None:
    assert decide_execution_profile({"avaloka_recommendation": dict(EVIDENCE)}, "statistical_analysis") == {}


def test_the_evidence_reaches_the_planner_state_through_the_node(monkeypatch) -> None:
    """End to end through the agent node, on the delegated path where the
    profile runs: the update it returns still carries the evidence."""
    from app.agents.avaloka_agent.agent import avaloka_agent_node

    async def _empty(_thread_id):
        return {}
    monkeypatch.setattr("app.agents.avaloka_agent.agent.load_thread_context", _empty)
    out = avaloka_agent_node(_bound(
        messages=[HumanMessage(content="train a model on this data")],
        avaloka_recommendation=dict(EVIDENCE)))
    assert out["delegate_to_planner"] is True
    assert out["avaloka_recommendation"].get("results") == EVIDENCE["results"]
