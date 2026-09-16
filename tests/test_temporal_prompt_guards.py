"""Regression tests: temporal group-by requests must not trip the
missing-filter-value guards.

"show average sales per day" is an aggregation, not a filter with a missing
value — neither the planner's clarification guard nor the validator's
assumed-weekday check may block it. Genuinely ambiguous filter requests
("filter by day") must still be caught.
"""
import pytest

from app.agents.planner import _detect_ambiguous_prompt_details
from app.agents.validator import _find_assumed_weekday_errors

DAY_ORDER_CODE = (
    "import pandas as pd\n"
    "def main(df):\n"
    "    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',"
    " 'Friday', 'Saturday', 'Sunday']\n"
    "    out = df.groupby('day')['sales'].mean().reindex(order).reset_index()\n"
    "    return out\n"
)

GROUP_BY_PROMPTS = [
    "show average sales per day",
    "show average sales per month",
    "select sales by day",
    "only show average sales per day",
    "plot revenue for each day",
    "group by day and compute the mean",
    "show me the trend per weekday",
]

AMBIGUOUS_PROMPTS = [
    "filter by day",
    "filter rows by date",
    "where status",
    "only keep rows where region",
]


# -------------------------
# Planner clarification guard
# -------------------------

@pytest.mark.parametrize("prompt", GROUP_BY_PROMPTS)
def test_planner_guard_allows_temporal_group_by(prompt):
    assert _detect_ambiguous_prompt_details(prompt) is None


@pytest.mark.parametrize("prompt", AMBIGUOUS_PROMPTS)
def test_planner_guard_still_catches_missing_filter_values(prompt):
    details = _detect_ambiguous_prompt_details(prompt)
    assert details is not None
    assert "filter by" in details["message"]


def test_planner_guard_passes_when_value_supplied():
    assert _detect_ambiguous_prompt_details("filter by day monday") is None
    assert _detect_ambiguous_prompt_details("show sales where region = US") is None


# -------------------------
# Validator assumed-weekday guard
# -------------------------

@pytest.mark.parametrize("prompt", GROUP_BY_PROMPTS)
def test_weekday_guard_allows_day_order_lists_for_group_by(prompt):
    assert _find_assumed_weekday_errors(DAY_ORDER_CODE, prompt) == []


def test_weekday_guard_still_blocks_assumed_weekday_filter():
    code = (
        "def main(df):\n"
        "    return df[df['day'] == 'Monday']\n"
    )
    errors = _find_assumed_weekday_errors(code, "filter rows by day")
    assert errors and "Monday" in errors[0]


def test_weekday_guard_passes_when_user_names_the_day():
    code = (
        "def main(df):\n"
        "    return df[df['day'] == 'Friday']\n"
    )
    assert _find_assumed_weekday_errors(code, "only keep rows for day friday") == []
