"""The auto-viz agent must profile the table it is charting, not the uploaded file.

Two defects, both of which produced the same user-visible symptom: a correct
answer displayed next to a chart of columns that are not in it.

  1. `profile_columns` trusts the schema over the rows. It iterates
     `schema.keys()` and `row.get()`s each name, so any column in the schema but
     absent from the rows profiles as 100% missing with a single "nan" value --
     a phantom column. `visualization_agent_node` always passed the *uploaded*
     schema, so the moment it charted a computed result every input column
     became a phantom, and the LLM picked chart fields from that menu.

  2. The node only used the computed result when it had >= 2 rows. Every
     single-row answer -- a count, an average, any scalar the user asked for --
     fell through to the uploaded-file preview instead, so the chart beside the
     answer described a different table entirely.

Observed in tests/test_global_youtube_analysis_statistics.py:

    single_how_many_countries   exec_status='success' rows=1 charts=4
    AssertionError: Charts reference columns absent from the result table
      ['num_countries']: ["llm_0.x -> 'video views'", ...]

The analysis was right. `video views` is an input column that is nowhere in the
one-column result.

No LLM and no dataset needed: with no GROQ key the chart selection falls back to
its deterministic path, and these tests assert on the profile, which is built
before any model is consulted.
"""

from __future__ import annotations

import base64

from app.agents.visualization_agent import profile_columns, visualization_agent_node


#: What the user uploaded: the full YouTube file.
UPLOADED_SCHEMA = {
    "Youtuber": "object",
    "subscribers": "int64",
    "video views": "float64",
    "Country": "object",
    "uploads": "int64",
}

#: What "how many countries are in this data?" actually computes. One row, one
#: column, and none of the uploaded columns.
RESULT_ROWS = [{"num_countries": 49}]


def _datauri(csv_text: str) -> str:
    return "data:text/csv;base64," + base64.b64encode(csv_text.encode()).decode()


def _state(result_csv: str) -> dict:
    return {
        "dataset_id": "ds_test",
        "schema": dict(UPLOADED_SCHEMA),
        "uploaded_csv_columns": list(UPLOADED_SCHEMA),
        "uploaded_csv_preview": [
            list(UPLOADED_SCHEMA),
            ["T-Series", 245000000, 2.28e11, "India", 20082],
            ["MrBeast", 166000000, 2.836e10, "United States", 741],
        ],
        "output_file_data": {"content": _datauri(result_csv)},
        "execution_result": {"status": "success"},
    }


def _names(profiles) -> set:
    return {p["name"] for p in profiles}


# ---------------------------------------------------------------------------
# Defect 1: the schema overrides the rows
# ---------------------------------------------------------------------------


def test_the_uploaded_schema_invents_phantom_columns_over_a_result_table():
    """Documents the defect. If this stops holding, defect 1 is gone and the
    guard below is asserting nothing."""
    profiles = profile_columns(RESULT_ROWS, schema=UPLOADED_SCHEMA)

    # Every uploaded column appears, though none is in the result.
    assert _names(profiles) == set(UPLOADED_SCHEMA)
    assert "num_countries" not in _names(profiles), (
        "the column the result actually has is missing from its own profile"
    )
    phantom = [p for p in profiles if p["missing_ratio"] == 1.0]
    assert len(phantom) == len(UPLOADED_SCHEMA), (
        "expected every uploaded column to profile as entirely missing"
    )


def test_the_result_columns_profile_the_result():
    """The fix, at the profiler level: hand it the columns the rows actually have."""
    profiles = profile_columns(RESULT_ROWS, schema=list(RESULT_ROWS[0].keys()))
    assert _names(profiles) == {"num_countries"}
    assert profiles[0]["missing_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Defect 2: single-row results fell through to the uploaded file
# ---------------------------------------------------------------------------


def _charted_fields(state_out) -> set:
    fields = set()
    for chart in (state_out.get("visualization_config") or {}).get("charts") or []:
        for enc in (chart.get("encodings") or {}).values():
            if isinstance(enc, dict) and enc.get("field"):
                fields.add(str(enc["field"]))
    return fields


def test_a_single_row_result_is_charted_from_the_result_not_the_upload():
    """The reported case: one row, one column, correct answer, wrong chart."""
    out = visualization_agent_node(_state("num_countries\n49\n"))

    charted = _charted_fields(out)
    assert charted, f"no chart fields at all: {out.get('visualization_status')!r}"
    assert charted <= {"num_countries"}, (
        f"charts reference columns that are not in the result table: "
        f"{sorted(charted - {'num_countries'})}"
    )


def test_a_multi_row_result_is_charted_from_the_result():
    """The path that already worked stays working."""
    out = visualization_agent_node(
        _state("category,total_views\nMusic,120\nGaming,90\nNews,45\n")
    )
    charted = _charted_fields(out)
    assert charted <= {"category", "total_views"}, (
        f"charts reference columns absent from the result: "
        f"{sorted(charted - {'category', 'total_views'})}"
    )


def test_an_echoed_result_still_falls_back_to_the_uploaded_overview():
    """When the result IS the uploaded file, charting it as an overview is right.

    The echo branch is deliberately kept: the columns are the same either way, so
    it cannot produce a dangling chart field, and an unmodified passthrough is
    more useful shown as a dataset overview.
    """
    echo = "Youtuber,subscribers,video views,Country,uploads\nT-Series,245000000,2.28e11,India,20082\n"
    out = visualization_agent_node(_state(echo))
    charted = _charted_fields(out)
    assert charted <= set(UPLOADED_SCHEMA), (
        f"charts reference unknown columns: {sorted(charted - set(UPLOADED_SCHEMA))}"
    )


# ---------------------------------------------------------------------------
# Backstop: a hallucinated field is dropped even when the menu is correct
# ---------------------------------------------------------------------------


def test_a_suggestion_naming_an_unknown_column_is_dropped():
    """_charts_from_llm_suggestions used to build whatever it was handed.

    Correcting the profile removes the usual cause of a dangling field, but the
    model can still name a column that was never on the menu. A chart that plots
    one renders blank in the UI while every stage reports success, so it is
    dropped here rather than shipped.
    """
    from app.agents.visualization_agent import _charts_from_llm_suggestions

    viz_profile = {
        "dataset": {"rows_sampled": 1},
        "columns": [{"name": "num_countries"}],
    }
    suggestions = [
        {"type": "histogram", "title": "Countries", "x_field": "num_countries"},
        {"type": "bar", "title": "Views", "x_field": "Country", "y_field": "video views"},
        {"type": "bar", "title": "Half real", "x_field": "num_countries", "y_field": "uploads"},
    ]

    charts = _charts_from_llm_suggestions(suggestions, viz_profile)

    titles = [c["title"] for c in charts]
    assert titles == ["Countries"], (
        f"expected only the chart plotting a real column, got {titles}"
    )


def test_a_histogram_without_a_y_field_is_kept():
    """y_field is legitimately None for a histogram -- absent is not unknown."""
    from app.agents.visualization_agent import _charts_from_llm_suggestions

    charts = _charts_from_llm_suggestions(
        [{"type": "histogram", "title": "Dist", "x_field": "num_countries", "y_field": None}],
        {"dataset": {"rows_sampled": 1}, "columns": [{"name": "num_countries"}]},
    )
    assert [c["title"] for c in charts] == ["Dist"]


# ---------------------------------------------------------------------------
# Defect 3: a result whose categoricals group well produced no charts at all
#
# Surfaced only after defects 1 and 2 were fixed. While the agent charted the
# uploaded file, `op_null_drop_rows` got charts of the upload's numeric columns
# and the emptiness went unnoticed; once it charted its own result -- (Youtuber,
# Country), no numeric column -- select_charts returned nothing, because the
# categorical branch gated on an absolute `n_unique <= 25` and Country has 49.
# The chart it builds already renders only the top 15 categories, so the cap was
# never what kept the chart readable.
# ---------------------------------------------------------------------------


def _charts(result_csv: str):
    return (visualization_agent_node(_state(result_csv)).get("visualization_config")
            or {}).get("charts") or []


def _many_rows(n: int, distinct: int) -> str:
    body = "\n".join(f"chan{i},Country{i % distinct}" for i in range(n))
    return f"Youtuber,Country\n{body}\n"


def test_a_categorical_that_groups_is_charted_even_above_the_cardinality_cap():
    """49 countries across 939 rows: 19 rows per country, a useful bar chart."""
    charts = _charts(_many_rows(939, 49))
    fields = {e.get("field") for c in charts for e in (c.get("encodings") or {}).values()
              if isinstance(e, dict) and e.get("field")}
    assert charts, "a result with a well-grouped categorical produced no charts"
    assert "Country" in fields, f"expected a chart on Country, got {fields}"


def test_an_identifier_column_is_still_not_charted():
    """The cap existed for a reason: 939 unique Youtubers means every bar is 1.

    Widening the gate must not turn an identifier into a chart -- which is why
    the rule is "values repeat", not a bigger number.
    """
    charts = _charts(_many_rows(939, 939))
    fields = {e.get("field") for c in charts for e in (c.get("encodings") or {}).values()
              if isinstance(e, dict) and e.get("field")}
    assert "Youtuber" not in fields, (
        f"charted a column with one row per value: {fields}"
    )
