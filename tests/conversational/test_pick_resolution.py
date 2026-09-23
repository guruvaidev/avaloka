"""A number the user types must mean what we offered -- or be asked about.

The reported failure: Avaloka offered seven numbered analyses, the user replied
"run 3", and the training agent answered "I can only help with model-training
and inference tasks". Three separate things had to hold for that to happen, so
there are three groups of tests here.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.suggestions import (looks_like_a_pick, parse_offered_options,
                                    pick_suggestion,
                                    recover_pending_suggestions,
                                    resolve_pick_with_recovery)

OFFERED = [
    "Descriptive summary of the parameters and their values",
    "Extract numeric values and compute aggregates such as total man hours",
    "Create a pivot table comparing management effort against target volumes",
    "Identify missing or inconsistent entries",
]

REPLY = (
    "Here is what I would look at first:\n\n"
    + "\n".join(f"{n}. {s}" for n, s in enumerate(OFFERED, 1))
    + "\n\nWould you like me to start with one of these?"
)


# -- 1. the pick resolves when state survives ------------------------------

@pytest.mark.parametrize("message", [
    "run 3", "3", "#3", "option 3", "number 3", "please run 3", "do 3", "run 3.",
])
def test_every_natural_phrasing_of_a_pick_resolves(message):
    state = {"pending_suggestions": OFFERED}
    assert resolve_pick_with_recovery(message, state) == OFFERED[2]


# -- 2. the pick still resolves when the checkpoint is gone ----------------
# MemorySaver is process-local RAM. A restart or a second worker loses
# pending_suggestions, and before this the number silently meant nothing.

def test_pick_recovers_from_transcript_when_checkpoint_is_empty():
    state = {
        "pending_suggestions": None,
        "messages": [HumanMessage(content="what can I look at?"), AIMessage(content=REPLY)],
    }
    assert pick_suggestion("run 3", state) is None          # state alone cannot
    assert resolve_pick_with_recovery("run 3", state) == OFFERED[2]


def test_recovery_reads_the_most_recent_offer_not_the_first():
    old = "Here are options:\n\n1. Old A\n2. Old B\n3. Old C\n"
    state = {"messages": [AIMessage(content=old), HumanMessage(content="ok"),
                          AIMessage(content=REPLY)]}
    assert recover_pending_suggestions(state["messages"]) == OFFERED


def test_recovery_ignores_numbers_the_user_typed():
    """A user's own numbered list is not an offer we made."""
    state = {"messages": [HumanMessage(content="1. me\n2. myself\n3. I")]}
    assert recover_pending_suggestions(state["messages"]) == []


# -- 3. an unresolvable pick is never guessed at --------------------------

def test_pick_with_nothing_to_resolve_against_returns_none():
    assert resolve_pick_with_recovery("run 3", {}) is None
    assert resolve_pick_with_recovery("run 3", {"pending_suggestions": None,
                                                "messages": []}) is None


def test_out_of_range_pick_is_not_resolved():
    assert resolve_pick_with_recovery("run 9", {"pending_suggestions": OFFERED}) is None


def test_a_pick_carrying_extra_instruction_is_left_alone():
    """"run 3 but only for the north region" adds a constraint. Rewriting it to
    the bare suggestion would silently discard what the user just asked for."""
    msg = "run 3 but only for the north region"
    assert not looks_like_a_pick(msg)
    assert resolve_pick_with_recovery(msg, {"pending_suggestions": OFFERED}) is None


@pytest.mark.parametrize("message", [
    "what columns are there?", "train a model to predict score",
    "run the training plan", "show me the top 3 domains",
])
def test_ordinary_messages_are_not_mistaken_for_picks(message):
    assert not looks_like_a_pick(message)
    assert resolve_pick_with_recovery(message, {"pending_suggestions": OFFERED}) is None


def test_parse_requires_a_list_that_starts_at_one_and_increments():
    assert parse_offered_options(
        "1. summarise\n2. aggregate\n3. pivot") == ["summarise", "aggregate", "pivot"]
    # A run that skips or restarts is prose containing numbers, not a menu.
    assert parse_offered_options("2. aggregate\n5. pivot") == []
    # Single-character items are noise, not options.
    assert parse_offered_options("1. a\n2. b\n3. c") == []
