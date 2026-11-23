"""The generic binder: preconditions, defaults, one Ask, one Refusal, no causal refusal."""

from __future__ import annotations

import re

import pytest

from avaloka.outcomes.semantics import ColumnSemantics
from avaloka.recipes import Ask, Bound, Intent, Refusal
from avaloka.recipes.intent import ASK_LIMIT, bind
from avaloka.recipes.specs import SPECS

ROWS = {"n_rows": 500}


def sem(**cols: str | tuple) -> dict[str, ColumnSemantics]:
    out = {}
    for name, spec in cols.items():
        role, extra = (spec, {}) if isinstance(spec, str) else (spec[0], spec[1])
        out[name] = ColumnSemantics(name=name, role=role, **extra)
    return out


def test_ask_when_two_measures_fit():
    s = sem(revenue="measure", cost="measure", region="dimension")
    out = bind(Intent("typical value?", "describe"), SPECS["describe"], s, ROWS)
    assert isinstance(out, Ask)
    assert out.parameter == "measure" and out.options == ("revenue", "cost")
    assert not re.search(r"\d", out.text)


def test_single_candidate_is_bound_without_asking():
    s = sem(revenue="measure", region="dimension")
    out = bind(Intent("typical value?", "describe"), SPECS["describe"], s, ROWS)
    assert out == Bound("describe", "1.0", {"measure": "revenue"})


def test_too_many_candidates_ask_without_options():
    s = sem(**{f"m{i}": "measure" for i in range(ASK_LIMIT + 1)})
    out = bind(Intent("typical value?", "describe"), SPECS["describe"], s, ROWS)
    assert isinstance(out, Ask) and out.options == ()


def test_refusal_no_measure():
    s = sem(region="dimension", order_date="time")
    out = bind(Intent("average?", "describe"), SPECS["describe"], s, ROWS)
    assert out == Refusal("no_measure", out.text) and not re.search(r"\d", out.text)


def test_refusal_no_time_column():
    s = sem(revenue="measure")
    out = bind(Intent("trend", "trend", {"measure": "revenue"}), SPECS["trend"], s, ROWS)
    assert isinstance(out, Refusal) and out.reason == "no_time_column"


def test_refusal_insufficient_rows_from_profile():
    s = sem(revenue="measure", cost="measure")
    for profile in ({"n_rows": 12}, {"dataset": {"n_rows": 29}}):
        out = bind(Intent("correlated?", "correlate"), SPECS["correlate"], s, profile)
        assert isinstance(out, Refusal) and out.reason == "insufficient_rows"
    assert isinstance(bind(Intent("correlated?", "correlate"), SPECS["correlate"], s, {"n_rows": 30}), Bound)


def test_missing_counts_are_left_to_the_gate():
    s = sem(revenue="measure", cost="measure", order_date="time")
    assert isinstance(bind(Intent("correlated?", "correlate"), SPECS["correlate"], s, {}), Bound)
    assert isinstance(bind(Intent("trend", "trend", {"measure": "revenue"}), SPECS["trend"], s, {"n_periods": 3}), Refusal)
    assert isinstance(bind(Intent("trend", "trend", {"measure": "revenue"}), SPECS["trend"], s, {"n_periods": 4}), Bound)


def test_defaults_for_optional_params():
    s = sem(revenue="measure", order_date=("time", {"grain": "month"}), region="dimension")
    trend = bind(Intent("trend", "trend", {"measure": "revenue"}), SPECS["trend"], s, ROWS)
    assert trend == Bound("trend", "1.0", {"measure": "revenue", "time": "order_date", "grain": "month"})
    cmp = bind(Intent("vs last", "compare_periods", {"measure": "revenue"}), SPECS["compare_periods"], s, ROWS)
    assert isinstance(cmp, Bound) and cmp.params["window"] == "last_period"
    assert "dimension" not in cmp.params          # an optional column is never guessed
    grainless = sem(revenue="measure", order_date="time")
    out = bind(Intent("trend", "trend", {"measure": "revenue"}), SPECS["trend"], grainless, ROWS)
    assert isinstance(out, Bound) and "grain" not in out.params


def test_explicit_params_win_over_defaults_and_candidates():
    s = sem(revenue="measure", cost="measure", order_date=("time", {"grain": "day"}))
    out = bind(Intent("q", "trend", {"measure": "cost", "grain": "week"}), SPECS["trend"], s, ROWS)
    assert out == Bound("trend", "1.0", {"measure": "cost", "time": "order_date", "grain": "week"})


def test_many_param_takes_every_candidate_when_few():
    s = sem(a="measure", b="measure", c="measure", region="dimension")
    out = bind(Intent("move together?", "correlate"), SPECS["correlate"], s, ROWS)
    assert out == Bound("correlate", "1.0", {"measures": ("a", "b", "c")})


def test_causal_flag_passes_through_without_refusal():
    s = sem(churned="measure", sessions="measure", plan="dimension")
    intent = Intent("what drives churn?", "regress", {"outcome": "churned"}, 0.9, flags=("causal_question",))
    out = bind(intent, SPECS["regress"], s, ROWS)
    assert isinstance(out, Bound) and out.params["outcome"] == "churned"
    assert "causal_question" in intent.flags          # the caller reads it; nothing here consumed it


def test_arm_needs_a_few_valued_dimension():
    many = sem(converted="measure", variant=("dimension", {"distinct": 9}))
    out = bind(Intent("a/b", "ab_test"), SPECS["ab_test"], many, ROWS)
    assert isinstance(out, Refusal) and out.reason == "unsupported_question"
    two = sem(converted="measure", variant=("dimension", {"distinct": 2}))
    assert bind(Intent("a/b", "ab_test"), SPECS["ab_test"], two, ROWS) == \
        Bound("ab_test", "1.0", {"arm": "variant", "outcome": "converted"})


def test_bound_parameter_of_the_wrong_role_is_rejected_not_repaired():
    from avaloka.recipes.intent import IntentRejected
    s = sem(revenue="measure", region="dimension")
    with pytest.raises(IntentRejected):
        bind(Intent("q", "describe", {"measure": "region"}), SPECS["describe"], s, ROWS)
