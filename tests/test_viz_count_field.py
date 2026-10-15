"""The model names the row count "count"; that must chart as a count, not be dropped."""
import pytest

from app.agents import visualization_agent as va

COLS = [
    {"name": "Medical Condition", "dtype": "string", "role": "categorical"},
    {"name": "Insurance Provider", "dtype": "string", "role": "categorical"},
    {"name": "Date of Admission", "dtype": "datetime", "role": "temporal"},
    {"name": "Billing Amount", "dtype": "float", "role": "numeric"},
]
PROFILE = {"dataset": {"rows_sampled": 100}, "columns": COLS}


def _one(s, profile=PROFILE):
    out = va._charts_from_llm_suggestions([s], profile)
    assert len(out) == 1, out
    return out[0]


@pytest.mark.parametrize("name", ["count", "Count", " rows ", "frequency", "row_count"])
def test_bar_with_count_y_is_a_row_count_chart(name):
    ch = _one({"type": "bar", "title": "t", "x_field": "Medical Condition", "y_field": name})
    assert ch["encodings"]["y"] == {"aggregate": "count", "type": "quantitative"}


def test_pie_with_count_y_counts_categories():
    ch = _one({"type": "pie", "title": "t", "x_field": "Insurance Provider", "y_field": "count"})
    assert ch["encodings"]["theta"]["aggregate"] == "count"
    assert ch["encodings"]["theta"]["field"] == "Insurance Provider"


def test_line_over_dates_with_count_y_is_monthly_count():
    ch = _one({"type": "line", "title": "t", "x_field": "Date of Admission", "y_field": "count"})
    assert ch["encodings"]["x"]["type"] == "temporal"
    assert ch["encodings"]["y"]["aggregate"] == "count"


def test_real_column_named_count_is_used_as_measure():
    prof = {"dataset": {}, "columns": COLS + [{"name": "count", "dtype": "int", "role": "numeric"}]}
    ch = _one({"type": "bar", "title": "t", "x_field": "Medical Condition", "y_field": "count"}, prof)
    assert ch["encodings"]["y"]["field"] == "count"


def test_unknown_column_is_still_dropped():
    assert va._charts_from_llm_suggestions(
        [{"type": "bar", "title": "t", "x_field": "Medical Condition", "y_field": "Revenue"}], PROFILE
    ) == []