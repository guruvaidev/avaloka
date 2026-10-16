"""Sheet rows 37-40: geographic codes, year columns, scatter pairings, heavy skew."""
import random

import pytest

from app.agents import visualization_agent as va


def _rows(n=600, seed=7):
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        loan = rnd.lognormvariate(12, 0.6)
        if rnd.random() < 0.01:
            loan *= 40                      # a few extreme loans (up to millions)
        rows.append({
            "activity_year": rnd.choice([2018, 2019, 2020, 2021, 2022, 2023, 2024]),
            "census_tract": round(rnd.uniform(1e9, 6e10), 1),   # 21204460900.8-style
            "state_fips": rnd.choice([6, 36, 48, 12]),
            "loan_amount": round(loan, 2),
            "interest_rate": round(rnd.uniform(2.5, 8.0), 3),
            "loan_type": rnd.choice(["Conventional", "FHA", "VA"]),
        })
    return rows


def _profiles(rows):
    return {p["name"]: p for p in va.profile_columns(rows)}


# Row 37 -------------------------------------------------------------------
@pytest.mark.parametrize("name", ["census_tract", "Census Tract", "tract", "state_fips",
                                  "geoid", "GEO_ID", "msamd", "cbsa"])
def test_geographic_codes_are_identifiers(name):
    rows = [{name: 1e9 + i * 37.5, "x": i} for i in range(100)]
    assert _profiles(rows)[name]["role"] == "identifier"


def test_ordinary_measures_are_not_identifiers():
    p = _profiles(_rows())
    assert p["loan_amount"]["role"] == "measure"
    assert p["interest_rate"]["role"] == "measure"


# Row 38 -------------------------------------------------------------------
def test_year_column_is_a_category_not_a_measure():
    p = _profiles(_rows())["activity_year"]
    assert p["dtype"] == "categorical" and p["role"] == "dimension" and p["is_year"]
    assert {t["value"] for t in p["stats"]["top_k"]} == {str(y) for y in range(2018, 2025)}


def test_year_detected_without_year_in_name_when_few_distinct():
    rows = [{"period": 2015 + (i % 6), "v": i * 1.5} for i in range(120)]
    assert _profiles(rows)["period"].get("is_year")


def test_wide_integer_measure_in_year_range_is_not_a_year():
    rows = [{"weight": 1900 + i, "v": i} for i in range(150)]   # 150 distinct, no "year" in name
    assert not _profiles(rows)["weight"].get("is_year")


def test_counts_per_year_are_in_calendar_order():
    rows = _rows()
    points, _, _ = va._ground_categories(rows, "activity_year", None, "count", 15, False)
    assert [p["x"] for p in points] == [str(y) for y in range(2018, 2025)]


def test_llm_histogram_of_a_year_becomes_a_count_bar():
    rows = _rows()
    prof = {"dataset": {"rows_sampled": len(rows)}, "columns": va.profile_columns(rows)}
    out = va._charts_from_llm_suggestions(
        [{"type": "histogram", "title": "Years", "x_field": "activity_year"}], prof)
    assert out[0]["type"] == "bar"
    assert out[0]["encodings"]["y"] == {"aggregate": "count", "type": "quantitative"}


# Row 39 -------------------------------------------------------------------
def test_scatter_of_year_vs_code_is_dropped():
    rows = _rows()
    prof = {"dataset": {"rows_sampled": len(rows)}, "columns": va.profile_columns(rows)}
    out = va._charts_from_llm_suggestions(
        [{"type": "scatter", "title": "a", "x_field": "activity_year", "y_field": "state_fips"}], prof)
    assert out == []


def test_scatter_with_census_tract_is_dropped_as_identifier():
    rows = _rows()
    prof = {"dataset": {"rows_sampled": len(rows)}, "columns": va.profile_columns(rows)}
    out = va._charts_from_llm_suggestions(
        [{"type": "scatter", "title": "a", "x_field": "activity_year", "y_field": "census_tract"}], prof)
    assert out == []


def test_scatter_of_category_and_measure_becomes_average_bar():
    rows = _rows()
    prof = {"dataset": {"rows_sampled": len(rows)}, "columns": va.profile_columns(rows)}
    out = va._charts_from_llm_suggestions(
        [{"type": "scatter", "title": "Rates over years", "x_field": "interest_rate",
          "y_field": "activity_year", "insight": "rates correlate with year"}], prof)
    ch = out[0]
    assert ch["type"] == "bar"
    assert ch["encodings"]["x"]["field"] == "activity_year"
    assert ch["encodings"]["y"]["field"] == "interest_rate"
    assert ch["encodings"]["y"]["aggregate"] == "mean"
    assert ch["insight"] is None            # the correlation sentence is not reused


def test_scatter_of_two_measures_is_kept():
    rows = _rows()
    prof = {"dataset": {"rows_sampled": len(rows)}, "columns": va.profile_columns(rows)}
    out = va._charts_from_llm_suggestions(
        [{"type": "scatter", "title": "a", "x_field": "loan_amount", "y_field": "interest_rate"}], prof)
    assert out[0]["type"] == "scatter"


# Row 40 -------------------------------------------------------------------
def test_skewed_histogram_folds_the_tail_into_the_last_bar():
    rows = _rows()
    points, meta, text = va._ground_histogram(rows, "loan_amount", 30)
    vals = sorted(r["loan_amount"] for r in rows)
    assert meta["capped"]["high"] is not None
    assert points[-1]["x"].endswith("+")
    assert points[-1]["x1"] == vals[-1]                      # the bar reaches the true max
    assert sum(p["y"] for p in points) == len(rows)          # no row lost
    assert "grouped in the last bar" in text
    # The body is spread over many bars instead of one.
    assert sum(1 for p in points if p["y"] > 0) >= 10


def test_symmetric_histogram_is_unchanged():
    rows = [{"v": float(i % 100)} for i in range(1000)]
    points, meta, text = va._ground_histogram(rows, "v", 20)
    assert "capped" not in meta
    assert points[0]["x0"] == 0 and points[-1]["x1"] == 99
    assert "grouped" not in text


def test_skewed_scatter_hides_extremes_but_measures_r_on_all_rows():
    rows = _rows()
    points, meta, text = va._ground_scatter(rows, "loan_amount", "interest_rate")
    assert meta["points_hidden"] > 0
    assert meta["n"] == len(rows)
    cap = sorted(r["loan_amount"] for r in rows)[int(0.995 * len(rows))]
    assert max(p["x"] for p in points) <= cap
    assert "left off the plot" in text


def test_full_pipeline_never_charts_codes_or_scatters_years():
    rows = _rows()
    cfg = va.build_visualization_config_from_sample("ds", rows)
    for ch in cfg["charts"]:
        fields = va._chart_fields(ch)
        assert "census_tract" not in fields and "state_fips" not in fields, ch["title"]
        if ch["type"] in ("scatter", "histogram"):
            assert "activity_year" not in fields, ch["title"]
    stats = {r["name"]: r for r in cfg["column_stats"]}
    assert stats["activity_year"]["dtype"] == "categorical"
    assert stats["census_tract"]["role"] == "identifier"