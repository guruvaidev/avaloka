"""Multi-sheet / multi-file Auto Insights: no charts on made-up column names,
and an importance score the UI uses to pick the top insights across datasets."""
import pytest

from app.agents import visualization_agent as va


@pytest.mark.parametrize("name,expected", [
    ("column_3", True),
    ("Column_12", True),
    ("Unnamed: 4", True),
    ("unnamed:0", True),
    ("Unnamed: 2_level_1", True),
    ("column", False),
    ("column_name", False),
    ("loan_column_3", False),
    ("Area", False),
    (None, False),
])
def test_placeholder_column_names(name, expected):
    assert va._is_placeholder_column(name) is expected


def _bar(title, field, rank, **extra):
    chart = {"title": title, "type": "bar", "rank": rank,
             "encodings": {"x": {"field": field}, "y": {"aggregate": "count"}}}
    chart.update(extra)
    return chart


def test_charts_on_unnamed_columns_are_left_out_and_ranks_renumbered():
    charts = [
        _bar("Records by column_3", "column_3", 1),
        _bar("Records by Area", "Area", 2),
        {"title": "Pass rate by column_2", "type": "bar", "rank": 3,
         "encodings": {"x": {"field": "Area"}},
         "config": {"rate_by_value": {"field": "column_2", "value": "Pass"}}},
        _bar("Records by Status", "Status", 4),
    ]
    kept, left_out = va._drop_placeholder_charts(charts)
    assert [c["title"] for c in kept] == ["Records by Area", "Records by Status"]
    assert [c["rank"] for c in kept] == [1, 2]
    assert [x["title"] for x in left_out] == ["Records by column_3", "Pass rate by column_2"]
    assert "column_3" in left_out[0]["reason"]


def _headerless_rows(n=40):
    areas = ["MTA", "DTA", "Preparation", "Scheduler", "Planner"]
    return [{"column_1": f"TC-{i:03d}", "column_2": ["P0", "P1", "P2"][i % 3],
             "column_3": areas[i % 5], "column_4": ["Pass", "Fail", None][i % 3]}
            for i in range(n)]


def test_headerless_sheet_gets_no_charts_on_column_n(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)
    out = va.build_visualization_config_from_sample("ds1", _headerless_rows())
    for ch in out["charts"]:
        assert not any(va._is_placeholder_column(f) for f in va._chart_fields(ch)), ch["title"]
    assert out.get("charts_left_out"), "the left-out charts are listed for transparency"


def test_named_sheet_charts_carry_an_importance(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)
    rows = [{"Area": ["MTA", "DTA", "Planner"][i % 3], "Priority": ["P0", "P1"][i % 2],
             "Duration_min": 5 + (i * 7) % 40, "Defects": (i * 3) % 5}
            for i in range(60)]
    out = va.build_visualization_config_from_sample("ds2", rows)
    assert out["charts"]
    for ch in out["charts"]:
        assert isinstance(ch["importance"], int) and 0 <= ch["importance"] <= 100


def _scored(chart, rows=500):
    chart.setdefault("rank", 1)
    return va._chart_importance(chart, rows)


def test_importance_orders_kinds_of_finding():
    rate = _scored({"type": "bar", "derived_data": {"aggregate": "rate", "n": 500}})
    measure = _scored({"type": "bar", "derived_data": {"aggregate": "mean", "n": 500}})
    hist = _scored({"type": "histogram", "derived_data": {"n": 500}})
    counts = _scored({"type": "bar", "derived_data": {"aggregate": "count", "n_rows": 500}})
    missing = _scored({"type": "pie", "config": {"missing_vs_present": True}, "derived_data": {"n": 500}})
    assert rate > measure > hist > counts > missing


def test_importance_of_a_scatter_follows_r():
    weak = _scored({"type": "scatter", "derived_data": {"r": 0.05, "n": 500}})
    strong = _scored({"type": "scatter", "derived_data": {"r": -0.8, "n": 500}})
    assert strong > weak


def test_importance_penalises_rank_small_data_and_rewards_checked_findings():
    base = {"type": "bar", "derived_data": {"aggregate": "mean", "n": 500}}
    first = _scored(dict(base, rank=1))
    third = _scored(dict(base, rank=3))
    tiny = _scored({"type": "bar", "derived_data": {"aggregate": "mean", "n": 8}})
    few = _scored({"type": "bar", "derived_data": {"aggregate": "mean", "n": 25}})
    llm = _scored(dict(base, insight_source="llm"))
    assert first - third == 6
    assert first > few > tiny
    assert llm == first + 5


def test_profile_overview_drops_unnamed_high_value_columns():
    overview = va.build_profile_overview({
        "status": "ok",
        "predictive_values": {"column_1": "high", "Cases": "high", "Unnamed: 3": "high", "Area": "low"},
    })
    assert overview["high_value_columns"] == ["Cases"]