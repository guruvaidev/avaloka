"""A correction to the method always re-plans, and says why from code's facts.

Owner ruling: a correction is never answered from the results already in hand.
The turn goes back to the planner, and before it does the user is told the
original approach, the corrected one, why the change matters and what the
alternative is. The approaches are facts code reads from the user's words; a
model may phrase the reasoning, and its phrasing is used only when it keeps to
those facts and states no figure the turn did not already hold.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.avaloka_agent import agent as agent_mod
from app.agents.avaloka_agent.correction import (
    correction_facts,
    phrasing_holds,
    reasoning_text,
)
from app.agents.avaloka_agent.intent_classifier import classify_intent

PRIOR = {"source": "verified analysis results for the connected dataset",
         "results": {"mean_order_value": 37.31}}
TURNS = [HumanMessage(content="What's the mean order value?"),
         HumanMessage(content="You excluded refunds. Include them and recompute.")]


@pytest.mark.parametrize("text,original,corrected", [
    ("You excluded refunds. Include them and recompute.", "excluding refunds", "including refunds"),
    ("you left out the cancelled orders", "excluding cancelled orders", "including cancelled orders"),
    ("You included test accounts, that's wrong", "including test accounts", "excluding test accounts"),
    ("include returns and recompute", "excluding returns", "including returns"),
    ("remove weekends and re-run", "including weekends", "excluding weekends"),
    ("use median instead of mean.", "using mean", "using median"),
])
def test_the_facts_are_read_from_the_users_words(text, original, corrected):
    facts = correction_facts(text)
    assert facts is not None
    assert (facts.original, facts.corrected) == (original, corrected)
    assert facts.alternative == original


@pytest.mark.parametrize("text", [
    "That's wrong, it's 42.5. Please correct it.",       # a value pushed: held, not re-planned
    "What was revenue last quarter?",
    "include them and recompute",                          # no subject to name
    "Compute distinct aisles on the entire dataset.",
])
def test_a_disputed_value_or_new_work_is_not_a_correction(text):
    assert correction_facts(text) is None


def test_a_correction_is_its_own_intent_and_re_plans_even_with_results_in_hand():
    result = classify_intent(TURNS, results_in_hand=True)
    assert result.intent == "correction" and result.should_delegate()
    assert result.params["correction"]["corrected"] == "including refunds"


class _Model:
    def __init__(self, reply: str) -> None:
        self.reply, self.calls = reply, []

    def invoke(self, messages, **kwargs):
        self.calls.append(list(messages))
        return AIMessage(content=self.reply)


def _node(monkeypatch, model) -> Dict[str, Any]:
    monkeypatch.setattr(agent_mod, "_get_reply_llm", lambda: model)

    async def _empty(_thread_id):
        return {}
    monkeypatch.setattr(agent_mod, "load_thread_context", _empty)
    # A dataset is bound: verified results come from one, and with none bound
    # the turn is stopped before the planner (tests/test_no_dataset_new_work.py).
    return agent_mod.avaloka_agent_node({"messages": list(TURNS), "avaloka_recommendation": PRIOR,
                                         "data_source_location": "/data/orders.csv"})


def test_a_phrasing_that_keeps_to_the_facts_is_used(monkeypatch):
    good = ("You're right — I was excluding refunds. I'm re-planning with including refunds, "
            "because refunds are money customers paid. The alternative is to keep excluding "
            "refunds if you only want completed sales. Re-running now.")
    out = _node(monkeypatch, _Model(good))
    assert out["messages"][-1].content == good


@pytest.mark.parametrize("bad", [
    # invents the figure the re-run has not produced
    "I was excluding refunds; including refunds the mean is 41.02, because refunds count. "
    "The alternative is excluding refunds.",
    # renames the approach
    "I was ignoring returns; now counting returns, because they matter. The alternative is not to.",
    # no reason, no alternative
    "Switching from excluding refunds to including refunds.",
    # explains the correction, never says it will be re-run (recorded on
    # gpt-oss: the evaluator failed it, replanned=0, and so must this check)
    "The original approach, **excluding refunds**, gives the mean without refunded "
    "transactions. The corrected approach, **including refunds**, counts them, because "
    "refunds affect the total. The alternative is to keep excluding refunds.",
    # says it won't
    "No need to re-run: excluding refunds was fine, and including refunds changes little, "
    "because they are rare. The alternative is to keep excluding refunds.",
])
def test_a_phrasing_that_strays_is_replaced_by_codes_words(monkeypatch, bad):
    out = _node(monkeypatch, _Model(bad))
    assert out["messages"][-1].content == reasoning_text(correction_facts(TURNS[-1].content))


def test_the_check_allows_figures_the_turn_already_held():
    facts = correction_facts(TURNS[-1].content)
    text = ("The 37.31 was excluding refunds; re-planning including refunds, because refunds "
            "count. The alternative is to keep excluding refunds.")
    assert phrasing_holds(text, facts, ["{'mean_order_value': 37.31}"])
    assert not phrasing_holds(text.replace("37.31", "41.02"), facts, ["{'mean_order_value': 37.31}"])


def test_the_recorded_phrasing_with_a_non_breaking_hyphen_holds():
    """gpt-oss's recorded reasoning says "I will re\u2011run the calculation"."""
    facts = correction_facts(TURNS[-1].content)
    recorded = "The original approach, **excluding refunds**, gives the mean order value without counting any refunded transactions.  \nThe corrected approach, **including refunds**, gives the mean order value after accounting for refunded transactions.  \nbecause refunds affect the total amount and the number of orders, the average can change.  \nThe alternative is to keep **excluding refunds**, which would be the right choice if you only want to analyze sales that were not reversed.  \n\nI will re\u2011run the calculation and show what changes."
    assert phrasing_holds(recorded, facts, ["{'mean_order_value': 37.31}"])
