"""1.6 release suite: Bugs-1.6 visualization titles.

BG05: an Auto Insights chart titled "Weekly Sales Versus Department" read as a
comparison of two things when it showed one measure broken down by a
category. When the model's own title is rejected, the agent now writes a
neutral title that only names what is plotted (``_neutral_title``); "vs" is
kept for scatter plots, where two measures really are set against each other.
"""
from __future__ import annotations

import pytest

from app.agents.visualization_agent import _neutral_title

ENC = {"x": {"field": "Dept"}, "y": {"field": "Weekly_Sales"}}


@pytest.mark.parametrize("ctype", ["bar", "line", "box", "area"])
def test_bg05_category_breakdown_title_reads_by_not_versus(ctype):
    """BG05: a measure broken down by a category is titled '<measure> by
    <category>', never 'versus' or 'vs'."""
    title = _neutral_title("", ctype, ENC)
    assert title == "Weekly_Sales by Dept"
    assert " vs" not in title.lower() and "versus" not in title.lower()


def test_bg05_scatter_keeps_vs_and_histogram_names_the_distribution():
    """BG05: the neutral title still distinguishes the chart kinds: a scatter of
    two measures is '<y> vs <x>', a histogram is 'Distribution of <x>'."""
    assert _neutral_title("", "scatter", ENC) == "Weekly_Sales vs Dept"
    assert _neutral_title("", "histogram", {"x": {"field": "Weekly_Sales"}}) == "Distribution of Weekly_Sales"


def test_bg05_data_sentence_lead_in_wins_over_axis_names():
    """BG05: when the grounded data sentence has a lead-in, the title is that
    lead-in (what the chart shows), not a constructed comparison."""
    text = "Weekly Sales by Department: department 92 has the highest average (75,204)."
    assert _neutral_title(text, "bar", ENC) == "Weekly Sales by Department"
