"""The planner must offer a way forward, and honour it on the next turn.

Reported from a live develop-1.6 deployment: asked how to analyse a dataset,
Avaloka returned nine numbered steps and stopped. Nothing indicated it could
run any of them, and the obvious reply -- "do number 5" -- meant nothing.
"""
from __future__ import annotations

import pytest

from app.agents.planner import _render_suggestions, resolve_suggestion_reference

SUGGESTIONS = [
    "Compute task durations from Start and Due",
    "Summarise mean, median and spread of durations",
    "Run a Monte Carlo simulation over total project time",
]


# --------------------------------------------------------------------------- #
# The reply has to invite a next move
# --------------------------------------------------------------------------- #

def test_suggestions_are_numbered_so_the_user_can_point_at_one():
    """"Run 3" is a reply anyone will type; restating a bullet is not."""
    out = _render_suggestions(SUGGESTIONS)
    for n in (1, 2, 3):
        assert f"{n}." in out, f"suggestion {n} is not numbered"


def test_suggestions_offer_to_carry_one_out():
    """The old rendering ended on the final bullet with no way forward."""
    out = _render_suggestions(SUGGESTIONS).lower()
    assert "run" in out, "the reply never offers to run anything"
    assert out.rstrip()[-1] not in {"-", "*"}, "reply still ends on a bare bullet"


def test_empty_suggestions_ask_rather_than_apologise_into_silence():
    out = _render_suggestions([])
    assert out.strip(), "an empty suggestion list produced an empty reply"
    assert "?" in out or "tell me" in out.lower(), (
        "with nothing to suggest, the agent must still ask for direction")


# --------------------------------------------------------------------------- #
# ...and the invitation has to be honoured
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("reply", ["3", "run 3", "Run 3", "do 3", "option 3",
                                   "number 3", "#3", "please run 3", "run 3."])
def test_a_numeric_reply_resolves_to_the_suggestion(reply: str):
    """Offering a numbered list and then not understanding the number is the
    same defect as documenting a phrase that does nothing."""
    state = {"pending_suggestions": SUGGESTIONS}
    assert resolve_suggestion_reference(reply, state) == SUGGESTIONS[2]


def test_a_reference_with_extra_instruction_is_left_alone():
    """"run 3 but only for the north region" is a NEW instruction.

    Rewriting it to the bare suggestion would silently discard the constraint
    the user just added -- worse than not resolving it at all.
    """
    state = {"pending_suggestions": SUGGESTIONS}
    original = "run 3 but only for the north region"
    assert resolve_suggestion_reference(original, state) == original


@pytest.mark.parametrize("reply", ["7", "run 99", "0"])
def test_out_of_range_numbers_are_not_resolved(reply: str):
    state = {"pending_suggestions": SUGGESTIONS}
    assert resolve_suggestion_reference(reply, state) == reply


def test_numbers_mean_nothing_without_a_pending_list():
    """A bare "3" in an unrelated conversation must not be rewritten."""
    assert resolve_suggestion_reference("3", {}) == "3"
    assert resolve_suggestion_reference("3", {"pending_suggestions": []}) == "3"


def test_ordinary_messages_are_untouched():
    state = {"pending_suggestions": SUGGESTIONS}
    for message in ("what is this dataset about?",
                    "plot revenue by region",
                    "run the monte carlo simulation"):
        assert resolve_suggestion_reference(message, state) == message
