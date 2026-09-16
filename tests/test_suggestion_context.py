"""The number the user types has to mean what we offered.

The conversational prompt ends replies with a numbered list and the line
"Say the number and I'll start". That is a promise, and it was only half
implemented: the planner recorded the options it offered through its
``suggest_analysis`` tool, but the conversational agent — the surface a user
actually talks to, and the one the prompt instructs to list options — recorded
nothing. A reply of "5" arrived as the bare string "5" with no record of what
had been offered.

Two further context losses sat behind it: an offer was never cleared, so a "2"
typed many turns later resolved against a menu nobody remembered; and the
planner's context-limit retry dropped every AI message, discarding the list the
user was answering exactly when the session was long enough to need it.
"""
from __future__ import annotations

import pytest

from app.agents.suggestions import (parse_offered_options, pick_suggestion,
                                    remember_offered_options,
                                    resolve_suggestion_reference)

OFFER = (
    "Churn concentrates in the basic plan. Worth looking at:\n"
    "\n"
    "1. Break churn down by plan and tenure_months\n"
    "2. Correlate support_tickets with churn\n"
    "3. Segment by region\n"
    "4. Model last_login_days against churn\n"
    "5. Compare monthly_spend for churned vs retained\n"
    "\n"
    "Which of those would you like me to run? Say the number and I'll start."
)


# ── Recording what was offered ──────────────────────────────────────────────

def test_a_conversational_reply_records_the_options_it_offered():
    """The defect: this path offered options and remembered none of them."""
    updates: dict = {}
    remember_offered_options(updates, OFFER)

    assert updates["pending_suggestions"] is not None
    assert len(updates["pending_suggestions"]) == 5
    assert updates["pending_suggestions"][4] == "Compare monthly_spend for churned vs retained"


def test_the_closing_menu_wins_over_earlier_numbered_prose():
    """Replies often number their explanation before numbering the choices."""
    reply = (
        "The pipeline does three things:\n"
        "1. load the file\n"
        "2. clean the columns\n"
        "3. write parquet\n"
        "\n"
        "Where would you like to go next?\n"
        "1. Churn by plan\n"
        "2. Tickets versus churn\n"
        "3. Region split\n"
    )
    assert parse_offered_options(reply) == [
        "Churn by plan", "Tickets versus churn", "Region split",
    ]


def test_prose_that_merely_contains_a_number_is_not_a_menu():
    assert parse_offered_options("There are 12 rows and 8 columns.") == []
    assert parse_offered_options("") == []


def test_a_list_that_does_not_start_at_one_is_not_recorded():
    """Mis-recording resolves a pick to something never offered.

    That is worse than not resolving it, so the parser is deliberately strict.
    """
    assert parse_offered_options("3. third\n4. fourth\n") == []


# ── Resolving the pick ──────────────────────────────────────────────────────

@pytest.mark.parametrize("message,expected_index", [
    ("5", 4), ("run 5", 4), ("option 3", 2), ("#1", 0),
    ("please run 2 thanks", 1), ("number 4", 3),
])
def test_a_bare_reference_resolves_to_the_option(message, expected_index):
    options = parse_offered_options(OFFER)
    state = {"pending_suggestions": options}
    assert resolve_suggestion_reference(message, state) == options[expected_index]


@pytest.mark.parametrize("message", [
    "run 5 but only for the north region",
    "what about region?",
    "9",
])
def test_anything_that_is_not_purely_a_reference_is_left_alone(message):
    """Rewriting these would discard what the user actually asked for."""
    state = {"pending_suggestions": parse_offered_options(OFFER)}
    assert resolve_suggestion_reference(message, state) == message


def test_pick_reports_no_pick_rather_than_echoing_the_input():
    """The caller needs to distinguish "picked" from "unchanged"."""
    state = {"pending_suggestions": parse_offered_options(OFFER)}
    assert pick_suggestion("what about region?", state) is None
    assert pick_suggestion("3", state) == "Segment by region"


# ── An offer is good for one turn ───────────────────────────────────────────

def test_an_offer_does_not_survive_being_taken():
    """A "2" typed twenty turns later used to resolve against a dead menu."""
    state = {"pending_suggestions": parse_offered_options(OFFER)}

    assert pick_suggestion("5", state) is not None
    state["pending_suggestions"] = None          # what the agent does on a pick

    assert pick_suggestion("2", state) is None


def test_a_reply_offering_nothing_clears_a_stale_list():
    updates = {"pending_suggestions": ["old", "options"]}
    remember_offered_options(updates, "The dataset has 12 rows and 8 columns.")
    assert updates["pending_suggestions"] is None


def test_no_options_means_no_resolution():
    assert resolve_suggestion_reference("3", {}) == "3"
    assert resolve_suggestion_reference("3", {"pending_suggestions": []}) == "3"
    assert resolve_suggestion_reference("3", {"pending_suggestions": None}) == "3"
