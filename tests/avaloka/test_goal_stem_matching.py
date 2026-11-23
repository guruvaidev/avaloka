"""Goal words name columns through simple stems, whole words only.

README check (rc3 web): "What drives returns?" and "What is the return rate by
channel?" are asked of ``orders``, whose outcome column is ``returned``.
"returns", "returned" and "return" share the stem ``return``; "channels" is
``channel``. Matching is case-insensitive and whole-token: an exact name wins,
and "turn" never names ``returned``.
"""

import pytest

from avaloka.goal_analysis import canonical_column_words, column_mentions, driver_target, word_stems

ORDERS = ["order_id", "order_date", "region", "channel", "product", "units",
          "unit_price", "discount_pct", "returned"]
_ROLES = {"order_id": "id", "order_date": "datetime", "region": "categorical", "channel": "categorical",
          "product": "categorical", "units": "numeric", "unit_price": "numeric",
          "discount_pct": "numeric", "returned": "boolean"}
PROFILE = {"columns": [{"name": c, "role": _ROLES[c], "n_unique": 2 if c == "returned" else 10}
                       for c in ORDERS]}


@pytest.mark.parametrize("goal", ["What drives returns?", "what drives RETURNED", "What drives return?",
                                  "drivers of returns", "what predicts returns by channel"])
def test_what_drives_returns_targets_returned(goal):
    target = driver_target(goal, PROFILE)
    assert target and target["column"] == "returned", target


def test_return_rate_by_channel_names_returned_and_channel():
    text = "What is the return rate by channels?"
    assert [c for _s, _e, c in column_mentions(text, ORDERS)] == ["returned", "channel"]
    assert canonical_column_words(text, ORDERS) == "What is the returned rate by channel?"


@pytest.mark.parametrize("text", ["what drives turn", "the turnover by region", "turned by channel",
                                  "TURNS by region"])
def test_turn_never_names_returned(text):
    assert "returned" not in {c for _s, _e, c in column_mentions(text, ORDERS)}
    assert canonical_column_words(text, ORDERS).lower().count("returned") == 0


def test_turn_is_not_a_driver_target():
    assert driver_target("what drives turn", PROFILE)["column"] is None
    assert driver_target("what drives turnover", PROFILE)["column"] is None


def test_an_exact_name_wins_over_an_inflected_one():
    cols = ["return", "returned"]
    assert [c for _s, _e, c in column_mentions("return by region", cols)] == ["return"]
    assert [c for _s, _e, c in column_mentions("returned by region", cols)] == ["returned"]
    # Both columns could be "returns": a guess would be a coin toss, so neither.
    assert column_mentions("returns by region", cols) == []


def test_a_longer_column_beats_a_shorter_one_inside_it():
    assert [c for _s, _e, c in column_mentions("unit prices by region", ORDERS)] == ["unit_price", "region"]


def test_stems_are_whole_and_never_shorter_than_three_letters():
    assert "return" in word_stems("returns") & word_stems("returned")
    assert not word_stems("turn") & word_stems("returned")
    assert word_stems("red") == {"red"}
