"""A goal's outcome value of several words is read whole.

On the LendingClub file, "what predicts loan_status = Charged Off" was read
as loan_status = "Charged" (the bare value stopped at the first word), held
by 0 of 497,955 rows; "what predicts Charged Off loans" named no value at
all, because the goal was matched word by word and "Charged" alone is none.
The longest distinct value of a column that appears in the goal text is
matched, case-insensitively, as whole words.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avaloka.goal_analysis import analyze_goal, driver_target

STATUSES = ["Fully Paid", "Charged Off", "Current", "In Grace Period", "Late (31-120 days)",
            "Charged Off After Review"]


@pytest.fixture(scope="module")
def loans() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 1200
    status = rng.choice(STATUSES, n, p=[0.45, 0.2, 0.2, 0.08, 0.05, 0.02])
    rate = rng.normal(12, 3, n) + np.where(status == "Charged Off", 4.0, 0.0)
    return pd.DataFrame({"int_rate": rate.round(2), "grade": rng.choice(list("ABCDE"), n),
                         "loan_status": status})


@pytest.mark.parametrize("goal", [
    "what predicts Charged Off loans",
    "what predicts charged off loans?",
    "what predicts loan_status = Charged Off",
    "what predicts loan_status = charged off for grade A",
])
def test_charged_off_is_read_whole(loans, goal):
    target = driver_target(goal, {}, loans)
    assert (target["column"], target["value"]) == ("loan_status", "Charged Off"), target


def test_a_value_that_begins_with_a_qualifier_word_is_read_whole(loans):
    target = driver_target("what predicts In Grace Period loans", {}, loans)
    assert (target["column"], target["value"]) == ("loan_status", "In Grace Period"), target


def test_the_longest_value_wins(loans):
    target = driver_target("what predicts Charged Off After Review loans", {}, loans)
    assert target["value"] == "Charged Off After Review", target


def test_the_answer_ranks_against_charged_off(loans):
    result = analyze_goal(loans, "what predicts Charged Off loans", {})
    assert result["kind"] == "drivers" and result["status"] == "answered", result
    assert result["metrics"]["target_value"] == "Charged Off"
    assert result["findings"][0].startswith("What moves with loan_status = Charged Off")
    assert result["metrics"]["positives"] == int((loans["loan_status"] == "Charged Off").sum())


def test_single_word_values_still_resolve_word_by_word(loans):
    target = driver_target("what predicts Current loans", {}, loans)
    assert (target["column"], target["value"]) == ("loan_status", "Current"), target
