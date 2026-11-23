"""B3: the §4.2 recommender rule table. Each row's cue picks its question
type, its candidates come back in the table's order, and each "Refused when"
refuses.

Bite (run by hand, in the PR): make ``is_rate`` return False and
``test_composition_refuses_a_rate`` and ``test_pie_is_not_offered_for_a_rate``
go red (a pie of conversion rates is offered).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avaloka.outcomes import semantics as S
from avaloka.outcomes.recommend import cue_matches, question_type, recommend

rng = np.random.default_rng(11)


def _rec(goal, df, **kw):
    return recommend(goal, df, S.infer(df), **kw)


def _kinds(rec):
    return [c.chart_kind for c in rec.candidates]


def _sales(periods=8, regions=("north", "south", "east")):
    dates = pd.date_range("2025-01-01", periods=periods, freq="MS")
    rows = [(d, r, float(rng.integers(50, 150))) for d in dates for r in regions]
    return pd.DataFrame(rows, columns=["month", "region", "revenue"])


@pytest.mark.parametrize("goal,q", [
    ("how has revenue moved over time", "trend"),
    ("what is the channel mix", "composition"),
    ("show the spread of order values", "distribution"),
    ("price vs units", "relationship"),
    ("where do users drop-off in the funnel", "funnel"),
    ("vintage curves for each cohort", "cohort"),
    ("which country sells most, on a map", "geo"),
    ("why did revenue fall yoy", "bridge"),
    ("customer journey from ad to purchase", "flow"),
    ("top products by revenue", "ranking"),
])
def test_each_rows_cue_names_its_question(goal, q):
    assert cue_matches(goal)[0][0] == q


def test_ties_go_to_the_first_row_in_table_order():
    # "trend" (row 1) and "top" (row 10) both match.
    df = _sales()
    assert question_type("top regions trend", S.infer(df), df)[0] == "trend"


def test_trend_candidates_in_order():
    rec = _rec("revenue trend", _sales(8))
    assert rec.question_type == "trend"
    assert _kinds(rec) == ["line", "area", "bar"]
    assert rec.candidates[0].encoding == {"x": "month", "y": "revenue", "series": "region"}


def test_trend_with_three_periods_is_refused():
    rec = _rec("revenue trend", _sales(3))
    assert rec.candidates == []
    assert rec.refused[0]["question_type"] == "trend" and "3 periods" in rec.refused[0]["reason"]


def test_trend_bars_stop_at_twelve_periods():
    rec = _rec("revenue trend", _sales(13))
    assert "bar" not in _kinds(rec)
    assert any(r.get("chart_kind") == "bar" for r in rec.refused)


def test_composition_of_an_additive_measure():
    df = pd.DataFrame({"channel": list("abcd") * 10, "spend": rng.uniform(1, 9, 40)})
    rec = _rec("spend breakdown by channel", df)
    assert _kinds(rec) == ["bar", "pie"]


def test_composition_refuses_a_rate():
    df = pd.DataFrame({"channel": list("abcd") * 10, "conversion_rate": rng.uniform(0, 0.2, 40)})
    rec = _rec("conversion rate breakdown by channel", df)
    assert rec.candidates == []
    assert "rate" in rec.refused[0]["reason"] and "not additive" in rec.refused[0]["reason"]


def test_pie_is_not_offered_for_a_rate():
    df = pd.DataFrame({"channel": list("abc") * 10, "ctr": rng.uniform(0, 0.1, 30)})
    assert "pie" not in _kinds(_rec("ctr share by channel", df))


def test_pie_stops_at_five_parts():
    df = pd.DataFrame({"channel": list("abcdef") * 10, "spend": rng.uniform(1, 9, 60)})
    rec = _rec("spend share by channel", df)
    assert "pie" not in _kinds(rec)
    assert any(r.get("chart_kind") == "pie" for r in rec.refused)


def test_distribution_needs_twenty_rows():
    df = pd.DataFrame({"order_value": rng.normal(50, 5, 19)})
    assert _rec("distribution of order value", df).candidates == []
    df = pd.DataFrame({"order_value": rng.normal(50, 5, 20), "region": list("ab") * 10})
    assert _kinds(_rec("distribution of order value", df)) == ["histogram", "boxplot"]


def test_relationship_samples_above_two_thousand_points():
    df = pd.DataFrame({"price": rng.normal(10, 1, 2500), "units_sold_x": rng.normal(5, 1, 2500)})
    rec = _rec("price vs units", df)
    assert _kinds(rec) == ["scatter", "heatmap"]
    assert rec.candidates[0].encoding["sample"] == 2000 and rec.candidates[0].note == "sampled"


def test_funnel_refused_when_stage_means_are_not_monotone():
    ok = pd.DataFrame({"impressions": [1000, 800], "clicks": [60, 50], "conversions": [6, 4]})
    assert _kinds(_rec("funnel drop-off", ok)) == ["funnel", "bar"]
    bad = pd.DataFrame({"impressions": [1000, 800], "clicks": [60, 50], "conversions": [600, 900]})
    rec = _rec("funnel drop-off", bad)
    assert rec.candidates == [] and "do not fall" in rec.refused[0]["reason"]


def test_cohort_needs_three_cohorts():
    def frame(n):
        cohorts = pd.date_range("2025-01-01", periods=n, freq="QS")
        return pd.DataFrame([(c, m, float(rng.uniform(0, 5))) for c in cohorts for m in range(1, 7)],
                            columns=["origination_quarter", "months_on_book", "dq30_pct"])
    assert _kinds(_rec("vintage curves", frame(4))) == ["cohort_heatmap", "vintage_lines"]
    rec = _rec("vintage curves", frame(2))
    assert rec.candidates == [] and "2 cohorts" in rec.refused[0]["reason"]


def test_geo_choropleth_for_states():
    df = pd.DataFrame({"state": ["CA"] * 12 + ["NY"] * 12 + ["TX"] * 12, "sales": rng.uniform(1, 9, 36)})
    rec = _rec("sales by state", df)
    assert _kinds(rec) == ["choropleth"]
    assert rec.candidates[0].encoding["geo_kind"] == "us_state"


def test_geo_zip_is_detected_but_not_mapped_in_this_release():
    df = pd.DataFrame({"zip": ["94103", "10001", "73301"] * 12, "sales": rng.uniform(1, 9, 36)})
    rec = _rec("sales by zip on a map", df)
    assert rec.candidates == [] and "us_zip maps are not in this release" in rec.refused[0]["reason"]


def test_bridge_is_a_waterfall():
    df = pd.DataFrame({"year": [2024] * 3 + [2025] * 3, "component": list("abc") * 2,
                       "revenue": [1.0, 2, 3, 2, 2, 4]})
    assert _kinds(_rec("why did revenue change yoy", df)) == ["waterfall"]


def test_flow_refuses_sankey_on_a_cycle():
    acyclic = pd.DataFrame({"from_step": ["ad", "ad", "site"] * 4, "to_step": ["site", "app", "buy"] * 4,
                            "users": rng.integers(1, 9, 12)})
    assert _kinds(_rec("journey from ad to buy", acyclic)) == ["sankey", "heatmap"]
    cyclic = pd.DataFrame({"from_step": ["ad", "site", "app"] * 4, "to_step": ["site", "app", "ad"] * 4,
                           "users": rng.integers(1, 9, 12)})
    rec = _rec("journey from ad to buy", cyclic)
    assert _kinds(rec) == ["heatmap"]
    assert "cycle" in rec.refused[0]["reason"]


def test_flow_refuses_sankey_above_thirty_nodes():
    df = pd.DataFrame({"src": [f"s{i}" for i in range(20)] * 2, "dst": [f"d{i}" for i in range(20)] * 2,
                       "users": rng.integers(1, 9, 40)})
    rec = _rec("path from source to destination", df)
    assert "sankey" not in _kinds(rec) and "40 nodes" in rec.refused[0]["reason"]


def test_ranking_is_top_ten_plus_other():
    df = pd.DataFrame({"product": [f"p{i}" for i in range(30)] * 2, "revenue": rng.uniform(1, 9, 60)})
    rec = _rec("top products", df)
    assert _kinds(rec) == ["bar"] and rec.candidates[0].encoding["top"] == 10


def test_same_input_same_output():
    df = _sales(8)
    assert _rec("revenue trend", df).to_dict() == _rec("revenue trend", df).to_dict()


def test_no_cue_falls_back_on_roles():
    assert _rec("tell me about this", _sales(8)).question_type == "trend"
