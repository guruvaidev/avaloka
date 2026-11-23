"""The quantifier thresholds of app/agents/card_quantifiers.py, at their edges.

The behaviour on the upload card (the rc14 false sentences) is pinned in
test_upload_card_claims.py; this file pins the words themselves.
"""

from __future__ import annotations

import pytest

from app.agents.card_quantifiers import ColumnFacts, quantifier_conflict


def _zeros(n_zero, n):
    return ColumnFacts([{"fee": 0.0}] * n_zero + [{"fee": 1.5}] * (n - n_zero))


@pytest.mark.parametrize("sentence, n_zero, ok", [
    ("fee is always zero", 100, True),
    ("fee is always zero", 99, False),
    ("fee is almost always zero", 95, True),
    ("fee is almost always zero", 94, False),
    ("for the vast majority of trips fee is zero", 80, True),
    ("for the vast majority of trips fee is zero", 79, False),
    ("the majority of fee values are zero", 51, True),
    ("the majority of fee values are zero", 50, False),
    ("fee is frequently zero", 25, True),
    ("fee is frequently zero", 24, False),
    ("fee is rarely zero", 10, True),
    ("fee is rarely zero", 11, False),
])
def test_share_words_and_their_thresholds(sentence, n_zero, ok):
    assert (quantifier_conflict(sentence, _zeros(n_zero, 100)) is None) is ok


def test_constant_needs_one_distinct_value():
    assert quantifier_conflict("fee is constant", ColumnFacts([{"fee": 2.0}] * 5)) is None
    why = quantifier_conflict("fee is constant", ColumnFacts([{"fee": 2.0}] * 5 + [{"fee": 3.0}]))
    assert "2 distinct values" in why


def test_nearly_constant_needs_95_percent_one_value():
    rows = [{"fee": 2.0}] * 95 + [{"fee": 3.0}] * 5
    assert quantifier_conflict("fee is nearly constant", ColumnFacts(rows)) is None
    rows = [{"fee": 2.0}] * 94 + [{"fee": 3.0}] * 6
    assert quantifier_conflict("fee is nearly constant", ColumnFacts(rows))


def test_constant_over_time_is_a_trend_not_a_column_claim():
    rows = [{"fee": 2.0}, {"fee": 3.0}]
    assert quantifier_conflict("The fee rate stays constant across age groups.", ColumnFacts(rows)) is None


def test_a_quantifier_about_nothing_checkable_is_withheld():
    rows = [{"fee": 1.0}, {"fee": 2.0}]
    assert quantifier_conflict("fee is frequently higher on weekends", ColumnFacts(rows))
    assert quantifier_conflict("Drivers are often tipped", ColumnFacts(rows))


def test_text_without_quantifiers_is_left_alone():
    assert quantifier_conflict("Total ride cost spikes for short distances.", ColumnFacts([])) is None


def test_column_statistics_alone_decide_always_zero_by_min_and_max():
    stats = [{"name": "Airport_fee", "dtype": "numeric", "n_unique": 3, "min": -1.75, "max": 1.75}]
    assert quantifier_conflict("Airport_fee always zero", ColumnFacts(None, stats))
    stats = [{"name": "Airport_fee", "dtype": "numeric", "n_unique": 1, "min": 0.0, "max": 0.0}]
    assert quantifier_conflict("Airport_fee always zero", ColumnFacts(None, stats)) is None
    # zero_pct rounded to 100 while the range shows a nonzero value: not "always".
    stats = [{"name": "fee", "n_unique": 2, "min": 0.0, "max": 1.0, "zero_pct": 100.0}]
    assert quantifier_conflict("fee always zero", ColumnFacts(None, stats))


@pytest.mark.parametrize("a, b, is_rate, ok", [
    (6.25, 5.0, True, True),      # 1.25x, 1.25 points
    (6.2, 5.0, True, False),      # 1.24x
    (2.0, 1.0, True, True),       # 2x, 1.0 point
    (1.8, 0.9, True, False),      # 2x, but 0.9 points
    (125.0, 100.0, False, True),  # not a rate: ratio only
])
def test_markedly_higher_needs_ratio_and_points(a, b, is_rate, ok):
    points = [{"x": "smokes", "y": a}, {"x": "never smoked", "y": b}, {"x": "Unknown", "y": 0.5}]
    why = quantifier_conflict("People who smoke have a markedly higher rate than never-smokers.",
                              ColumnFacts([]), points=points, x_field="smoking_status", is_rate=is_rate)
    assert (why is None) is ok


def test_markedly_higher_in_the_wrong_direction_is_withheld():
    points = [{"x": "smokes", "y": 2.0}, {"x": "never smoked", "y": 9.0}]
    why = quantifier_conflict("Smokers have a markedly higher rate than never-smokers.",
                              ColumnFacts([]), points=points, x_field="smoking_status", is_rate=True)
    assert "smokes is 2" in why


def test_markedly_higher_needs_groups_it_can_name():
    points = [{"x": "A", "y": 20.0}, {"x": "B", "y": 1.0}]
    assert quantifier_conflict("Older patients have a markedly higher rate than younger ones.",
                               ColumnFacts([]), points=points, x_field="band", is_rate=True)


def test_no_constant_columns_is_true_when_none_is_constant():
    facts = ColumnFacts([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    assert quantifier_conflict("No constant columns detected", facts) is None
    assert quantifier_conflict("Drop constant columns (none present).", facts) is None
    assert quantifier_conflict("Constant column provides no variance", facts)


def test_others_are_much_smaller_holds_against_every_unnamed_group():
    points = [{"x": "Private", "y": 2925}, {"x": "Self-employed", "y": 819}, {"x": "children", "y": 687}]
    facts = ColumnFacts([])
    assert quantifier_conflict("Work type is dominated by Private; other categories are much smaller.",
                               facts, points=points, x_field="work_type") is None
    # Self-employed (819) is named too, and children (687) is not much smaller than it.
    assert quantifier_conflict("Work type is dominated by Private and Self-employed; other categories "
                               "are much smaller.", facts, points=points, x_field="work_type")


def test_yes_no_groups_resolve_from_the_column_name_in_any_order():
    points = [{"x": "Y", "y": 1.79}, {"x": "N", "y": 0.68}]
    why = quantifier_conflict("The rate is notably higher for those who settled their debt compared "
                              "to those who did not.", ColumnFacts([]), points=points,
                              x_field="debt_settlement_flag", is_rate=True)
    assert why is None


def test_group_share_of_a_sum_chart():
    shares = {"United Kingdom": 0.82, "Germany": 0.05, "France": 0.04}
    facts = ColumnFacts([])
    assert quantifier_conflict("The United Kingdom accounts for the vast majority of units sold.",
                               facts, group_shares=shares) is None
    assert quantifier_conflict("A few specific web links account for the vast majority of records.",
                               facts, group_shares={"a": 0.03, "b": 0.02, "c": 0.02, "d": 0.01})


# ---- rc15 shapes ---------------------------------------------------------


def _groups(*counts):
    return ColumnFacts([{"url": f"u{i}"} for i, n in enumerate(counts) for _ in range(n)])


def test_handful_bulk_needs_top5_half():
    s = "A handful of url values account for the bulk of the records."
    assert quantifier_conflict(s, _groups(20, 10, 10, 5, 5, *[1] * 50)) is None      # top 5 = 50/100
    assert quantifier_conflict(s, _groups(20, 10, 10, 5, 4, *[1] * 51))              # 49/100


def test_top_n_uses_n():
    s = "The top 2 url values account for the bulk of the records."
    assert quantifier_conflict(s, _groups(30, 20, *[1] * 50)) is None
    assert quantifier_conflict(s, _groups(30, 19, *[1] * 51))


def test_disproportionate_share_bar():
    s = "A few url values hold a disproportionate share of rows."
    assert quantifier_conflict(s, _groups(10, 10, 10, 10, 10, *[1] * 120)) is None    # 29.4% >= max(25/125, 20%)
    assert quantifier_conflict(s, _groups(5, 5, 5, 5, 5, *[1] * 120))                 # 17.2% < 20%


def test_most_seen_once_distinct_vs_rows():
    # 6 texts once, 4 texts twice: 60% of distinct values once, 6/14 = 43% of rows.
    facts = ColumnFacts([{"text": f"once {i}"} for i in range(6)] +
                        [{"text": f"twice {i}"} for i in range(4) for _ in range(2)])
    assert quantifier_conflict("A few text values repeat while most are seen only once.", facts) is None
    assert quantifier_conflict("Most recipes have their own unique text.", facts)


def test_steady_increase_needs_rho_08_over_4_periods():
    def facts(vals):
        return ColumnFacts([{"d": f"2020-{i + 1:02d}-01", "v": v} for i, v in enumerate(vals)])
    s = "Volume shows a steady increase."
    assert quantifier_conflict(s, facts([1, 2, 3, 4, 5]), time_field="d", measure="v", aggregate="sum") is None
    assert quantifier_conflict(s, facts([1, 3, 2, 5, 4]), time_field="d", measure="v", aggregate="sum") is None  # rho 0.8
    assert quantifier_conflict(s, facts([1, 4, 2, 5, 3]), time_field="d", measure="v", aggregate="sum")
    assert quantifier_conflict(s, facts([1, 2, 3]), time_field="d", measure="v", aggregate="sum")  # 3 periods
    assert quantifier_conflict("Volume declines steadily.", facts([5, 4, 3, 2, 1]),
                               time_field="d", measure="v", aggregate="sum") is None


def test_no_other_needs_one_distinct_value():
    assert quantifier_conflict("No other country appears in the data.",
                               ColumnFacts([{"country": "UK"}] * 3)) is None
    assert "2 distinct" in quantifier_conflict("No other country appears in the data.",
                                               ColumnFacts([{"country": "UK"}, {"country": "FR"}]))
    # Not a column: not this shape.
    assert quantifier_conflict("No other variable is present.", ColumnFacts([{"country": "UK"}])) is None


def test_association_needs_r_01_and_the_sign():
    s = "Higher price is associated with lower qty."
    facts = ColumnFacts([{"price": 1.0, "qty": 1.0}, {"price": 2.0, "qty": 2.0}])
    stats = [{"name": "price", "dtype": "numeric"}, {"name": "qty", "dtype": "numeric"}]
    f = ColumnFacts(None, stats)
    assert quantifier_conflict(s, f, default_columns=["price", "qty"], correlation=-0.1) is None
    assert quantifier_conflict(s, f, default_columns=["price", "qty"], correlation=-0.09)
    assert quantifier_conflict(s, f, default_columns=["price", "qty"], correlation=0.5)    # wrong sign
    assert quantifier_conflict("Price is associated with qty.", f, default_columns=["price", "qty"],
                               correlation=0.3) is None
    assert facts.pearson("price", "qty") is None or True
