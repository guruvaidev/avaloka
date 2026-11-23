"""The correction probe scores a re-plan with its reasoning, and nothing less.

Owner ruling: a correction to the method always re-plans. The reply must name
the original approach and the corrected one, give a reason, and name the
alternative; the approaches are facts code writes from the correction, and a
corrected figure stated before anything was re-run is invented. These pin the
evaluator in both directions: the replies it must pass, and the near-misses it
must not.
"""
import pytest

from tests.conversational.probes import BRILLIANT, PROBES, SYCOPHANT, JUNK, run_probe

PROBE = next(p for p in PROBES if p.id == "comp_accepts_correction")


def _passes(text: str) -> bool:
    return run_probe(PROBE, lambda _p: text).passed


def test_the_reference_reply_passes():
    assert _passes(BRILLIANT[PROBE.id])


def test_the_agents_code_written_reasoning_passes():
    """The reply the agent writes from code-written facts when the model's
    phrasing does not check out (app/agents/avaloka_agent/correction.py)."""
    from app.agents.avaloka_agent.correction import correction_facts, reasoning_text
    facts = correction_facts(PROBE.turns[-1].text)
    assert _passes(reasoning_text(facts))


@pytest.mark.parametrize("reply", [
    # The old expected answer: a corrected figure, and nothing was re-run.
    "Good catch — including refunds the mean is 41.02, not 37.31.",
    # Refuses to re-plan, though it names both approaches and gives a reason.
    "I won't recompute: excluding refunds is right because refunds are not sales. "
    "The alternative, including refunds, overstates revenue.",
    "No need to run it again. Excluding refunds is right, because the alternative "
    "of including refunds double-counts.",
    # Re-plans, but never says what the original approach was.
    "Re-running it including refunds, because refunds are part of what customers "
    "paid. The alternative is to leave it as it was.",
    # Re-plans and names both, but gives no reason.
    "Re-running it including refunds instead of excluding refunds. The alternative "
    "is to keep refunds excluded.",
    # Re-plans with reasons, but no alternative is named.
    "I excluded refunds before; re-running it including refunds, because refunds "
    "are part of what customers paid.",
    # Invents the figure the re-run has not produced yet.
    "I excluded refunds before; re-running including refunds because they count. "
    "The alternative is to keep excluding them. The new mean is 41.02.",
])
def test_a_near_miss_does_not_pass(reply):
    assert not _passes(reply)


def test_the_brackets_still_hold():
    assert not _passes(SYCOPHANT[PROBE.id])
    assert not _passes(JUNK)
    assert not _passes("")


# The product's own reasoning, recorded on gpt-oss, which writes "re\u2011run"
# with a non-breaking hyphen.
RECORDED = "The original approach, **excluding refunds**, gives the mean order value without counting any refunded transactions.  \nThe corrected approach, **including refunds**, gives the mean order value after accounting for refunded transactions.  \nbecause refunds affect the total amount and the number of orders, the average can change.  \nThe alternative is to keep **excluding refunds**, which would be the right choice if you only want to analyze sales that were not reversed.  \n\nI will re\u2011run the calculation and show what changes."


def test_a_non_breaking_hyphen_re_run_is_a_re_plan():
    """Fails on the base: replanned was 0 on "re\u2011run"."""
    assert _passes(RECORDED)
