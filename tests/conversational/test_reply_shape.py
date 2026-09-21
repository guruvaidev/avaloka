"""What a reply must look like before we call this a conversation.

These are cheap, deterministic contracts on the text Avaloka sends back. They
exist because the reported failure was not only a misroute: the same exchange
also offered seven options with no closing question, which is a menu to work
through rather than a recommendation to act on.
"""
import re

import pytest

from app.agents.suggestions import (MAX_OFFERED_SUGGESTIONS,
                                    render_suggestions as _render_suggestions)

IDEAS = [
    "Descriptive summary of the parameters and their values",
    "Extract numeric values and compute aggregates such as total man hours",
    "Create a pivot table comparing management effort against target volumes",
    "Identify missing or inconsistent entries and suggest cleaning steps",
    "Map each activity domain to target vs actual hours",
    "Visualise the distribution of hours per domain",
    "Consistency-check total man hrs against domains x target hours",
]


def _numbered(text):
    return re.findall(r"^\s*(\d+)\.\s+\S", text, flags=re.MULTILINE)


def test_offers_between_three_and_five_ideas():
    """Seven is a backlog. Three to five reads as advice."""
    assert MAX_OFFERED_SUGGESTIONS == 5
    shown = _numbered(_render_suggestions(IDEAS))
    assert len(shown) == 5, f"offered {len(shown)} ideas"


def test_never_offers_more_than_it_can_resolve():
    """The rendered list and the remembered list must be the same length, or a
    number the user reads off the screen resolves to something else."""
    for n in range(1, len(IDEAS) + 1):
        shown = _numbered(_render_suggestions(IDEAS[:n]))
        assert len(shown) == min(n, MAX_OFFERED_SUGGESTIONS)
        assert shown == [str(i) for i in range(1, len(shown) + 1)]


def test_the_list_is_numbered_so_a_number_can_answer_it():
    assert _numbered(_render_suggestions(IDEAS[:3])) == ["1", "2", "3"]


def test_reply_ends_by_asking_how_to_continue():
    """A reply that stops on its last bullet leaves the user nowhere to go."""
    text = _render_suggestions(IDEAS)
    assert text.rstrip().endswith((".", "?")) and "?" in text, text[-200:]
    tail = text[text.rindex("5."):]
    assert "?" in tail, "the closing question must follow the list, not precede it"


def test_the_offer_names_how_to_accept_it():
    text = _render_suggestions(IDEAS)
    assert "number" in text.lower()
    assert "own words" in text.lower(), "must also accept a free-text answer"


def test_empty_suggestions_still_asks_a_question():
    text = _render_suggestions([])
    assert "?" in text or "Tell me" in text, text
