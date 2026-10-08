"""Chart grounding and column roles: what PR #420 added to visualization_agent.py.

The PR roughly doubled this module (1,155 -> 2,266 lines). The new part computes
every chart's points and its insight sentence from the same sample rows
(``_ground_*``), classifies columns into roles (measure / dimension / time /
identifier / label), guarantees a pie, and routes the model through
``invoke_viz_llm``. None of it had a test.

Plain tests pin the arithmetic: a grounded sentence is only worth having if the
numbers in it are the numbers in the rows. ``defect`` tests assert intended
behaviour against defects confirmed by running the merged code (73a67a48); they
are xfail(strict=True) and must be removed when the defect is.

No LLM: ``viz_llm`` is None without a GROQ key, so config building takes the
deterministic path, and the ``invoke_viz_llm`` tests install a scripted fake.
"""

from __future__ import annotations

import base64

import pytest

from app.agents import visualization_agent as va


def _profiles(rows, schema=None):
    return {p["name"]: p for p in va.profile_columns(rows, schema=schema)}


def _bar(x, y=None, aggregate=None, ctype="bar", **config):
    enc = {"x": {"field": x}}
    if y or aggregate:
        enc["y"] = {"field": y, "aggregate": aggregate}
    return {"title": f"{ctype} {x}", "type": ctype, "encodings": enc, "config": config}


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


def test_histogram_bins_account_for_every_numeric_row():
    rows = [{"fare": v} for v in range(100)] + [{"fare": None}, {"fare": "n/a"}]

    points, meta, text = va._ground_histogram(rows, "fare", 10)

    assert len(points) == 10
    assert sum(p["y"] for p in points) == 100
    assert (meta["n"], meta["min"], meta["max"]) == (100, 0.0, 99.0)
    assert meta["median"] == 49.5
    assert "ranges from 0 to 99" in text
    # None is missing; "n/a" is present but not a number, so it is not "missing".
    assert "1 rows (1.0%) have no value" in text


def test_histogram_of_a_constant_column_is_one_bar_not_a_division_by_zero():
    points, meta, text = va._ground_histogram([{"v": 5}] * 4, "v", 30)

    assert points == [{"x": "5", "x0": 5.0, "x1": 5.0, "y": 4}]
    assert meta["skewness"] == 0.0
    assert "ranges from 5 to 5" in text


def test_histogram_without_numbers_has_nothing_to_plot():
    assert va._ground_histogram([{"v": "a"}, {"v": None}], "v", 30) is None


def test_histogram_bin_count_is_clamped():
    rows = [{"v": i} for i in range(200)]
    assert len(va._ground_histogram(rows, "v", 1)[0]) == 5
    assert len(va._ground_histogram(rows, "v", 500)[0]) == 50


def test_a_right_skewed_column_is_described_as_right_skewed():
    rows = [{"v": 1}] * 50 + [{"v": 1000}]
    assert "right-skewed" in va._ground_histogram(rows, "v", 10)[2]


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------


def test_scatter_reports_the_correlation_the_rows_have():
    rows = [{"a": i, "b": 2 * i + 1} for i in range(20)]

    points, meta, text = va._ground_scatter(rows, "a", "b")

    assert meta["n"] == 20 and meta["r"] == pytest.approx(1.0)
    assert "strong positive relationship (r = 1.00)" in text
    assert points[3] == {"x": 3.0, "y": 7.0}


def test_scatter_needs_three_complete_pairs():
    rows = [{"a": 1, "b": 2}, {"a": 2, "b": None}, {"a": 3, "b": 4}]
    assert va._ground_scatter(rows, "a", "b") is None


def test_scatter_with_a_constant_axis_says_there_is_nothing_to_measure():
    rows = [{"a": i, "b": 7} for i in range(5)]
    _, meta, text = va._ground_scatter(rows, "a", "b")
    assert meta["r"] is None
    assert "constant" in text


def test_scatter_points_are_capped_but_n_is_not():
    rows = [{"a": i, "b": i % 13} for i in range(5000)]
    points, meta, _ = va._ground_scatter(rows, "a", "b")
    assert len(points) == va.SCATTER_MAX_POINTS
    assert meta["n"] == 5000 and meta["points_shown"] == va.SCATTER_MAX_POINTS


# ---------------------------------------------------------------------------
# Categories (bar / pie)
# ---------------------------------------------------------------------------

_REGION_ROWS = (
    [{"region": "East", "sales": 10}] * 6
    + [{"region": "West", "sales": 40}] * 3
    + [{"region": None, "sales": 5}]
)


def test_count_bar_orders_groups_and_reports_shares_of_the_present_rows():
    points, meta, text = va._ground_categories(_REGION_ROWS, "region", None, "count", 15, False)

    assert points == [{"x": "East", "y": 6.0}, {"x": "West", "y": 3.0}]
    assert meta == {"n_groups": 2, "aggregate": "count", "measure": None}
    assert "East (6 rows, 66.7%)" in text and "West (3 rows, 33.3%)" in text
    assert "1 rows (10.0%) have no region and are left out of the chart" in text


def test_missing_labels_never_become_a_bar():
    points, _, _ = va._ground_categories(_REGION_ROWS, "region", None, "count", 15, False)
    assert "(missing)" not in [p["x"] for p in points]


def test_sum_bar_adds_the_measure_per_group():
    points, _, text = va._ground_categories(_REGION_ROWS, "region", "sales", "sum", 15, False)
    assert points == [{"x": "West", "y": 120.0}, {"x": "East", "y": 60.0}]
    assert text.startswith("Total sales by region")


def test_mean_bar_reports_highest_and_lowest_not_shares():
    points, _, text = va._ground_categories(_REGION_ROWS, "region", "sales", "mean", 15, False)
    assert points == [{"x": "West", "y": 40.0}, {"x": "East", "y": 10.0}]
    assert "highest is West (40), lowest is East (10)" in text
    assert "The top group is 4.0× the lowest." in text
    assert "%" not in text.split("Rows with no")[0]


def test_pie_slices_add_up_to_the_whole():
    rows = [{"c": f"k{i % 12}"} for i in range(120)] + [{"c": "big"}] * 30

    points, _, _ = va._ground_categories(rows, "c", None, "count", 7, True)

    assert len(points) == 8 and points[-1]["x"] == "Other"
    assert sum(p["y"] for p in points) == 150


def test_an_averaged_pie_gets_no_other_slice():
    """Averages do not add up, so an 'Other' slice would be a made-up number."""
    rows = [{"c": f"k{i % 12}", "v": i} for i in range(120)]
    points, _, _ = va._ground_categories(rows, "c", "v", "mean", 7, True)
    assert "Other" not in [p["x"] for p in points]


def test_a_column_that_is_entirely_missing_has_nothing_to_plot():
    assert va._ground_categories([{"c": None}, {"c": ""}], "c", None, "count", 15, False) is None


# ---------------------------------------------------------------------------
# Monthly trend
# ---------------------------------------------------------------------------


def test_monthly_series_is_in_calendar_order_whatever_order_the_rows_came_in():
    rows = [
        {"d": "2024-03-05", "v": 3}, {"d": "2024-01-20", "v": 1},
        {"d": "2024-02-11", "v": 2}, {"d": "2024-01-02", "v": 10},
        {"d": "not a date", "v": 99},
    ]
    points, meta, _ = va._ground_monthly(rows, "d", "v", "sum")
    assert points == [
        {"x": "2024-01", "y": 11.0}, {"x": "2024-02", "y": 2.0}, {"x": "2024-03", "y": 3.0},
    ]
    assert meta["n_months"] == 3


def test_a_single_month_is_reported_as_no_trend():
    rows = [{"d": "2024-01-02"}, {"d": "2024/1/30 10:15"}]
    points, meta, text = va._ground_monthly(rows, "d", None, "count")
    assert points == [{"x": "2024-01", "y": 2.0}]
    assert "no trend to show" in text


def test_a_partly_covered_last_month_is_left_out_of_the_peak_and_low():
    rows = []
    for month, n in (("01", 30), ("02", 32), ("03", 31), ("04", 29), ("05", 4)):
        rows += [{"d": f"2024-{month}-10"}] * n

    _, meta, text = va._ground_monthly(rows, "d", None, "count")

    assert meta["partial_months"] == ["2024-05"]
    assert "low 2024-04 (29)" in text
    assert "2024-05 is much lower than usual" in text


# ---------------------------------------------------------------------------
# _ground_chart / _ground_all
# ---------------------------------------------------------------------------


def test_points_are_keyed_by_the_charts_own_column_names():
    """The frontend reads point[<field name>]; x/y-only points rendered as
    '(missing)' and 0."""
    rows = [{"region": "East", "sales": 10}] * 6 + [{"region": "West", "sales": 40}] * 3
    chart = _bar("region", "sales", "sum")

    assert va._ground_chart(chart, rows, _profiles(rows)) is True

    first = chart["derived_data"]["points"][0]
    assert first["region"] == "West" and first["sales"] == 120.0
    assert (first["x"], first["y"], first["label"], first["value"]) == ("West", 120.0, "West", 120.0)
    assert (chart["derived_data"]["x_key"], chart["derived_data"]["y_key"]) == ("region", "sales")
    assert chart["insight_source"] == "data"


def test_a_count_pie_points_theta_at_count_not_at_its_own_label_column():
    rows = [{"region": "East"}] * 6 + [{"region": "West"}] * 3
    chart = {
        "title": "pie", "type": "pie", "config": {},
        "encodings": {"theta": {"field": "region", "aggregate": "count"},
                      "color": {"field": "region"}},
    }

    assert va._ground_chart(chart, rows, _profiles(rows)) is True

    assert chart["encodings"]["theta"]["field"] == "count"
    assert chart["derived_data"]["points"][0]["count"] == 6.0
    assert chart["derived_data"]["points"][0]["region"] == "East"


@pytest.mark.parametrize("ctype", ["bar", "line"])
@pytest.mark.parametrize("y", [
    {"field": "region", "aggregate": "count"},   # the label column itself
    {"field": "patients", "aggregate": "sum"},   # not a column of the rows
])
def test_a_bar_or_line_that_was_counted_says_count_in_its_y_encoding(ctype, y):
    """When the y field is discarded the points are row counts under y_key
    'count'. The encoding must say so too: the UI and the chart explainer read
    yField / aggregate from it, and 'sum of patients' is not what was plotted."""
    rows = [{"region": "East"}] * 6 + [{"region": "West"}] * 3
    chart = {
        "title": ctype, "type": ctype, "config": {},
        "encodings": {"x": {"field": "region"}, "y": dict(y)},
    }

    assert va._ground_chart(chart, rows, _profiles(rows)) is True

    assert chart["derived_data"]["y_key"] == "count"
    assert chart["derived_data"]["points"][0]["count"] == 6.0
    assert chart["encodings"]["y"] == {"aggregate": "count", "type": "quantitative"}


def test_a_measure_bar_keeps_its_y_encoding():
    rows = [{"region": "East", "sales": 10}] * 6 + [{"region": "West", "sales": 40}] * 3
    chart = _bar("region", "sales", "sum")

    assert va._ground_chart(chart, rows, _profiles(rows)) is True

    assert chart["encodings"]["y"] == {"field": "sales", "aggregate": "sum"}
    assert chart["derived_data"]["y_key"] == "sales"


def test_an_existing_insight_is_not_overwritten_by_grounding():
    rows = [{"region": "East"}] * 6 + [{"region": "West"}] * 3
    chart = _bar("region")
    chart["insight"] = "written by the model"
    chart["insight_source"] = "llm"

    va._ground_chart(chart, rows, _profiles(rows))

    assert chart["insight"] == "written by the model"
    assert chart["insight_source"] == "llm"
    assert chart["derived_data"]["points"]


def test_a_chart_with_nothing_to_plot_is_dropped_and_ranks_are_renumbered():
    rows = [{"region": "East", "sales": 10}] * 6 + [{"region": "West", "sales": 40}] * 3
    charts = [_bar("no_such_column"), _bar("region"), {"title": "odd", "type": "heatmap"}]

    kept = va._ground_all(charts, rows, _profiles(rows))

    assert [c["title"] for c in kept] == ["bar region"]
    assert kept[0]["rank"] == 1


def test_a_chart_whose_grounding_raises_is_kept(monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("bad rows")

    monkeypatch.setattr(va, "_ground_categories", boom)
    rows = [{"region": "East"}] * 3

    kept = va._ground_all([_bar("region")], rows, _profiles(rows))

    assert [c["title"] for c in kept] == ["bar region"]
    assert "derived_data" not in kept[0]


# ---------------------------------------------------------------------------
# Column roles
# ---------------------------------------------------------------------------


def test_iso_dates_profile_as_a_time_column():
    rows = [{"d": f"2024-0{1 + i % 3}-1{i % 9}"} for i in range(30)]
    prof = _profiles(rows)["d"]
    assert (prof["dtype"], prof["role"]) == ("datetime", "time")
    assert prof["stats"]["n_months"] == 3
    assert sum(prof["stats"]["by_month"].values()) == 30


@pytest.mark.parametrize("name", ["Patient ID", "zip_code", "Room Number", "order_no", "phone"])
def test_an_identifier_named_column_is_never_a_measure(name):
    rows = [{name: 100 + (i % 7)} for i in range(40)]
    assert _profiles(rows)[name]["role"] == "identifier"


@pytest.mark.parametrize("name", ["num_countries", "Casino", "Monkey", "unit_price"])
def test_names_that_merely_contain_an_identifier_word_stay_measures(name):
    rows = [{name: 100 + (i % 7)} for i in range(40)]
    assert _profiles(rows)[name]["role"] == "measure"


def test_the_key_of_a_small_result_table_is_a_label_not_an_identifier():
    rows = [{"niche": f"niche {i}", "views": 1000 - i} for i in range(20)]
    assert _profiles(rows)["niche"]["role"] == "label"


def test_a_name_like_column_in_a_large_table_is_an_identifier():
    rows = [{"name": f"person {i}"} for i in range(200)]
    assert _profiles(rows)["name"]["role"] == "identifier"


def test_a_label_bar_shows_each_rows_own_number():
    rows = [{"niche": f"niche {i}", "views": 1000 - i} for i in range(20)]
    chart = _bar("niche", "views", "sum", top_k=5)

    assert va._ground_chart(chart, rows, _profiles(rows)) is True

    assert chart["derived_data"]["aggregate"] == "value"
    assert chart["derived_data"]["points"][0]["views"] == 1000.0
    assert chart["insight"].startswith("views by niche: highest is niche 0 (1,000)")


# ---------------------------------------------------------------------------
# The guaranteed pie
# ---------------------------------------------------------------------------


def _pie_inputs():
    rows = [{"plan": ("free", "pro", "team")[i % 3], "seats": i} for i in range(60)]
    profiles = va.profile_columns(rows)
    return rows, profiles, va.compute_unsupervised_feature_ranking(profiles)


def test_an_extra_pie_is_built_from_the_profiles_real_counts():
    rows, profiles, ranking = _pie_inputs()

    pie = va._build_extra_pie_chart([], profiles, ranking, len(rows))

    assert pie["id"] == "pie_plan" and pie["type"] == "pie"
    assert "free (20 rows, 33.3%)" in pie["insight"]


def test_no_second_pie_and_no_pie_of_a_column_already_charted():
    rows, profiles, ranking = _pie_inputs()
    assert va._build_extra_pie_chart([{"type": "pie"}], profiles, ranking, len(rows)) is None
    assert va._build_extra_pie_chart([_bar("plan")], profiles, ranking, len(rows)) is None


def test_the_deterministic_path_grounds_every_chart_it_ships():
    rows = [
        {"plan": ("free", "pro", "team")[i % 3], "seats": (i * 7) % 23, "mrr": (i * 13) % 101}
        for i in range(90)
    ]

    cfg = va.build_visualization_config_from_sample("ds", rows)

    assert cfg["charts"], "no charts for a table with a categorical and two measures"
    for chart in cfg["charts"]:
        assert chart["derived_data"]["points"], f"{chart['title']!r} shipped without points"
        assert chart["insight"], f"{chart['title']!r} shipped without an insight"
    assert [c["rank"] for c in cfg["charts"]] == list(range(1, len(cfg["charts"]) + 1))
    assert cfg["insight_model"]["used"] is False
    assert cfg["insight_model"]["error"] == "visualization LLM not configured"


def test_the_pie_is_not_guaranteed_when_the_only_categorical_is_already_a_bar():
    """The section is headed 'Guaranteed pie insight'. It is not guaranteed: a
    pie never repeats a charted column, and a bar is only converted when there
    are two or more count bars. Pinned so the heading is not taken at its word."""
    rows = [
        {"plan": ("free", "pro", "team")[i % 3], "seats": (i * 7) % 23, "mrr": (i * 13) % 101}
        for i in range(90)
    ]

    cfg = va.build_visualization_config_from_sample("ds", rows)

    assert "plan" in set().union(*(va._chart_fields(c) for c in cfg["charts"]))
    assert [c for c in cfg["charts"] if c["type"] == "pie"] == []


def test_with_two_count_bars_the_lower_ranked_one_becomes_the_pie():
    rows = [
        {"plan": ("free", "pro", "team")[i % 3], "tier": ("a", "b")[i % 2], "seats": (i * 7) % 23}
        for i in range(90)
    ]

    cfg = va.build_visualization_config_from_sample("ds", rows)

    pies = [c for c in cfg["charts"] if c["type"] == "pie"]
    assert len(pies) == 1
    pie_field = pies[0]["encodings"]["color"]["field"]
    bar_fields = {c["encodings"]["x"]["field"] for c in cfg["charts"] if c["type"] == "bar"}
    assert pie_field in {"plan", "tier"}
    assert pie_field not in bar_fields, "the same column is shown as a bar and as a pie"
    assert sum(p["y"] for p in pies[0]["derived_data"]["points"]) == 90


# ---------------------------------------------------------------------------
# invoke_viz_llm
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def invoke(self, prompt, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _StatusError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def test_invoke_without_a_model_raises_rather_than_returning_nothing(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)
    with pytest.raises(RuntimeError, match="not configured"):
        va.invoke_viz_llm("prompt")


def test_a_reasoning_model_gets_token_headroom_and_the_requested_effort(monkeypatch):
    llm = _ScriptedLLM("answer")
    monkeypatch.setattr(va, "viz_llm", llm)
    monkeypatch.setattr(va, "_VIZ_IS_REASONING", True)
    monkeypatch.setattr(va, "_VIZ_BACKEND", "groq")

    assert va.invoke_viz_llm("prompt", effort="medium") == "answer"
    assert llm.calls == [{"max_tokens": va._VIZ_MAX_TOKENS, "reasoning_effort": "medium"}]


def test_openrouter_takes_the_effort_as_a_reasoning_object(monkeypatch):
    llm = _ScriptedLLM("answer")
    monkeypatch.setattr(va, "viz_llm", llm)
    monkeypatch.setattr(va, "_VIZ_IS_REASONING", True)
    monkeypatch.setattr(va, "_VIZ_BACKEND", "openrouter")

    va.invoke_viz_llm("prompt", effort="low")

    assert llm.calls == [{
        "max_tokens": va._VIZ_MAX_TOKENS,
        "extra_body": {"reasoning": {"effort": "low"}},
    }]


def test_rejected_tuning_kwargs_fall_back_to_a_plain_call(monkeypatch):
    llm = _ScriptedLLM(_StatusError("unknown parameter: reasoning_effort", 400), "answer")
    monkeypatch.setattr(va, "viz_llm", llm)
    monkeypatch.setattr(va, "_VIZ_IS_REASONING", True)

    assert va.invoke_viz_llm("prompt") == "answer"
    assert llm.calls[1] == {}


@pytest.mark.parametrize("error", [
    _StatusError("nope", 401),
    _StatusError("nope", 403),
    RuntimeError("Invalid API Key"),
    RuntimeError("User not found."),
])
def test_a_rejected_key_is_not_retried(monkeypatch, error):
    llm = _ScriptedLLM(error, "answer")
    monkeypatch.setattr(va, "viz_llm", llm)

    with pytest.raises(type(error)):
        va.invoke_viz_llm("prompt")
    assert len(llm.calls) == 1


def test_the_old_private_name_still_resolves():
    assert va._invoke_viz_llm is va.invoke_viz_llm


# ---------------------------------------------------------------------------
# User context for the insight prompt
# ---------------------------------------------------------------------------


def test_user_context_drops_the_fidelity_footer_and_caps_memory():
    ctx = va._collect_user_context({
        "user_prompt": "which borough pays most?\n[Analysis context] sample=10%",
        "memory_hints": ["h" * 900] + [f"hint {i}" for i in range(20)],
    })
    assert ctx["question"] == "which borough pays most?"
    assert len(ctx["memory"]) == 10 and len(ctx["memory"][0]) == 500


def test_user_context_falls_back_to_the_last_human_message():
    ctx = va._collect_user_context({"messages": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]})
    assert ctx == {"question": "second"}


# ---------------------------------------------------------------------------
# Confirmed defects
# ---------------------------------------------------------------------------


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT PR420-V1: profile_columns classes any run of >= 20 distinct "
        "consecutive integers as a row number (role 'identifier'), and identifiers "
        "are never charted. That is also exactly what the x axis of a per-hour "
        "(0..23), per-day (1..31) or per-year result looks like, so 'trips by hour' "
        "yields one histogram of trips and no chart over hour at all. Remove this "
        "xfail when fixed."
    ),
)
@pytest.mark.parametrize("axis,values", [
    ("hour", range(24)),
    ("year", range(2000, 2024)),
])
def test_a_time_axis_of_consecutive_integers_is_still_charted(axis, values):
    rows = [{axis: v, "trips": 100 + 7 * v * (v % 5)} for v in values]

    cfg = va.build_visualization_config_from_sample("ds", rows, schema=[axis, "trips"])

    charted = set()
    for chart in cfg["charts"]:
        charted |= va._chart_fields(chart)
    assert axis in charted, f"no chart uses {axis!r}; charted {sorted(charted)}"


_SIGNED_ROWS = (
    [{"region": "East", "profit": 500}] * 4
    + [{"region": "West", "profit": -300}] * 4
    + [{"region": "North", "profit": 50}] * 4
)


def test_a_signed_measure_is_not_described_as_shares_of_a_total():
    """Was DEFECT PR420-V2: every sum was treated as a share of the total, so a
    measure that can be negative read 'East (2,000, 200.0%) ... West (-1,200,
    -120.0%)'. A signed sum is described by its highest and lowest group."""
    points, meta, text = va._ground_categories(_SIGNED_ROWS, "region", "profit", "sum", 15, False)

    assert "%" not in text, text
    assert text == (
        "Total profit by region: highest is East (2,000), lowest is West (-1,200) "
        "in the sample."
    )
    # The bar itself is still drawn, with the real signed values.
    assert [(p["x"], p["y"]) for p in points] == [
        ("East", 2000.0), ("North", 200.0), ("West", -1200.0),
    ]


@pytest.mark.parametrize("agg", ["sum", "mean"])
def test_a_pie_of_a_signed_measure_is_not_drawn(agg):
    """Was the second half of PR420-V2: the pie got a NEGATIVE 'Other' slice. A
    pie is shares of a whole, so with any negative group there is nothing
    honest to draw and the chart is declined (the caller drops it)."""
    assert va._ground_categories(_SIGNED_ROWS, "region", "profit", agg, 2, True) is None

    chart = {
        "title": "pie", "type": "pie", "config": {"top_k": 2},
        "encodings": {"color": {"field": "region"},
                      "theta": {"field": "profit", "aggregate": agg}},
    }
    assert va._ground_chart(chart, _SIGNED_ROWS, _profiles(_SIGNED_ROWS)) is False
    assert "derived_data" not in chart


def test_shares_are_still_reported_when_no_group_is_negative():
    """The guard must not switch shares off for ordinary non-negative sums,
    including a group that sums to exactly zero."""
    rows = (
        [{"region": "East", "sales": 30}] * 2
        + [{"region": "West", "sales": 20}] * 2
        + [{"region": "North", "sales": 0}] * 2
    )
    points, _, text = va._ground_categories(rows, "region", "sales", "sum", 2, True)

    assert "East (60, 60.0%)" in text and "West (40, 40.0%)" in text, text
    assert [(p["x"], p["y"]) for p in points] == [("East", 60.0), ("West", 40.0), ("Other", 0.0)]


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT PR420-V3: _keyed_points writes each point's values under the "
        "chart's column names INTO the same dict that holds the generic x/y/"
        "label/name/value keys, so a column whose name is one of those keys can "
        "overwrite the generic one. It needs the names CROSSED: a column called "
        "'y' on the x axis (as below), or one called 'value', 'label' or 'name' "
        "on the opposite axis (x field 'value' leaves point['value'] holding the "
        "label). Columns 'x' and 'y' on their own axes are harmless. Remove this "
        "xfail when fixed."
    ),
)
def test_generic_point_keys_survive_columns_named_like_them():
    points = va._keyed_points([{"x": 1.0, "y": 2.0}], x_key="y", y_key="x")

    assert (points[0]["x"], points[0]["y"]) == (1.0, 2.0)


def test_a_single_row_summary_returns_no_config_so_the_api_keeps_the_session_charts():
    """Intended behaviour, and unchanged by PR #420 (identical at its parent
    a97fb70c). This was marked DEFECT PR420-V4 on the reading that 'nothing is
    kept by this function'. Nothing is kept HERE on purpose: the node returns
    {} and None so that app/api/server.py falls back to the session's
    upload-time config --

        final.get("visualization_config") or visualization_configs.get(primary_dsid)

    -- and so a stale checkpointed config is overwritten. PR #420 deleted the
    comment that said so; the behaviour predates it. Returning the state's own
    config from the node instead would be a design change, not a fix.
    """
    existing = {"charts": [{"id": "upload_overview"}], "visualization_status": "ready"}
    csv_text = "avg_fare,avg_tip\n12.5,2.1\n"
    state = {
        "dataset_id": "ds",
        "schema": {"fare": "float64", "tip": "float64"},
        "uploaded_csv_columns": ["fare", "tip"],
        "uploaded_csv_preview": [["fare", "tip"], [10, 2], [15, 3]],
        "output_file_data": {
            "content": "data:text/csv;base64," + base64.b64encode(csv_text.encode()).decode(),
        },
        "visualization_config": existing,
        "visualization_status": "ready",
    }

    out = va.visualization_agent_node(state)

    assert out["visualization_config"] == {}
    assert out["visualization_status"] is None
    # The falsy pair is what makes the API's `or` fall through to the session.
    assert not out["visualization_config"] and not out["visualization_status"]


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT V5 (response time; PRE-EXISTING, not introduced by PR #420 -- "
        "formerly labelled PR420-V5): invoke_viz_llm retries every non-auth "
        "failure with a second, untuned call, including a timeout and a 429. "
        "The retry was written for a provider that rejects max_tokens / "
        "reasoning_effort; a timed-out call is instead repeated in full. The "
        "retry-everything logic is identical at #420's parent (a97fb70c), where "
        "even a 401 was retried; #420 stopped retrying auth errors. What #420 "
        "did add is the cost of each repeated call: max_tokens 4096 -> 8192 and "
        "reasoning effort low -> high. Remove this xfail when fixed."
    ),
)
@pytest.mark.parametrize("error", [
    TimeoutError("read timed out"),
    _StatusError("rate limit exceeded", 429),
])
def test_a_timeout_or_rate_limit_is_not_repeated_as_a_plain_call(monkeypatch, error):
    llm = _ScriptedLLM(error, "late answer")
    monkeypatch.setattr(va, "viz_llm", llm)

    with pytest.raises(type(error)):
        va.invoke_viz_llm("prompt")
    assert len(llm.calls) == 1
