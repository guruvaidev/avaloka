"""`avaloka chat` has to be answerable, not just talkative.

The session ends every dataset load with a numbered list and the line "Say the
number and I will start". That is a promise, and a promise the loop has to
keep: a reply of "2" must run the second suggestion. The same affordance was
missing on the server side and shipped as a phrase that did nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avaloka.chat import _PICK_RE, render_suggestions, suggest_next_steps
from avaloka.stats import correlation_pairs, profile_column, quality_report


@pytest.fixture
def profile():
    rng = np.random.default_rng(7)
    n = 300
    frame = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "region": rng.choice(["north", "south"], n),
        "tenure_months": rng.integers(1, 60, n),
        "monthly_spend": rng.gamma(2.0, 40.0, n),
        "churned": (rng.random(n) < 0.3).astype(int),
    })
    cols = [profile_column(frame[c], len(frame)) for c in frame.columns]
    return {
        "dataset": {"source": "x.csv", "format": "csv", "sha256": "", "n_rows": n,
                    "n_cols": frame.shape[1], "delimiter": ","},
        "columns": cols,
        "quality": quality_report(frame, cols),
        "correlations": correlation_pairs(frame),
    }


def test_suggestions_name_columns_this_dataset_actually_has(profile):
    """Generic advice is advice about data science, not about your data."""
    ideas = suggest_next_steps(profile)

    assert ideas, "a loaded dataset must always yield something to do next"
    names = {c["name"] for c in profile["columns"]}
    assert any(any(name in idea for name in names) for idea in ideas), ideas


def test_a_binary_column_is_offered_as_a_training_target(profile):
    ideas = suggest_next_steps(profile)
    assert any(idea.startswith("train churned") for idea in ideas), ideas


def test_suggestions_are_runnable_commands_not_prose(profile):
    """Each line is fed straight back into the loop when the user picks it."""
    for idea in suggest_next_steps(profile):
        assert idea.split()[0] in {"train", "analyze"}, idea


def test_an_identifier_is_never_offered_as_a_target(profile):
    ideas = suggest_next_steps(profile)
    assert not any(idea.startswith("train customer_id") for idea in ideas), ideas


def test_suggestions_do_not_repeat_themselves(profile):
    ideas = suggest_next_steps(profile)
    assert len(ideas) == len({i.lower() for i in ideas})


@pytest.mark.parametrize("reply", ["2", "run 2", "option 2", "#2", "please run 2 thanks"])
def test_the_number_the_user_types_is_understood(reply):
    match = _PICK_RE.match(reply)
    assert match and int(match.group(1)) == 2


@pytest.mark.parametrize("reply", [
    "analyze churn by region",
    "run 2 but only for the north",
    "what columns do you have?",
])
def test_a_real_instruction_is_not_mistaken_for_a_pick(reply):
    """Rewriting these would throw away what the user actually asked."""
    assert _PICK_RE.match(reply) is None


def test_the_list_closes_by_inviting_a_choice(profile, capsys):
    """A numbered list with no invitation is a dead end."""
    from rich.console import Console

    from avaloka.persona import Avaloka

    console = Console(force_terminal=False, width=100)
    render_suggestions(console, Avaloka(llm=False), suggest_next_steps(profile))
    out = capsys.readouterr().out

    assert "1." in out and "2." in out
    assert "number" in out.lower(), "must tell the user how to answer"


def test_nothing_is_rendered_when_there_is_nothing_to_suggest(capsys):
    from rich.console import Console

    from avaloka.persona import Avaloka

    console = Console(force_terminal=False, width=100)
    render_suggestions(console, Avaloka(llm=False), [])
    assert capsys.readouterr().out == "", "an empty list must not print an empty invitation"
