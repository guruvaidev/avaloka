"""The math-on-string guard must defer to the blueprint's declared intent.

Measured defect (3/3 runs, `tests/blueprint_contract/harness.py`):

    "Build a monthly time series of admission counts from the date of
     admission, and add a 3-month rolling average of that count."

was refused outright with

    Cannot perform mathematical operation ('average') on the 'Date of
    Admission' column because it contains text/string values, not numbers.

Nothing in that request averages a date. The guard
(``app/agents/planner.py::_detect_math_on_string_column``) decides by regex
which column an aggregation verb targets; here the only column *named* is a
date, and 'average' appears, so it fires. The refusal is terminal and becomes
the chat reply, so a routine time-series question is reported as impossible.

The blueprint now states the result columns, so when none of them is the
flagged column the guard's premise is demonstrably false.

Deterministic: no LLM, no network.
"""

from __future__ import annotations

import pytest

from app.agents.blueprint_contract import BlueprintContract
from app.agents.coder import _contract_contradicts_math_guard

MATH_ERR = (
    "Cannot perform mathematical operation ('average') on the 'Date of Admission' "
    "column because it contains text/string values, not numbers. Mathematical "
    "operations like mean, sum, average, standard deviation, etc. can only be "
    "applied to numeric columns. Please select a numeric column for this operation."
)

# What the blueprint says for the monthly-admissions request: the date is read,
# but the numbers produced are a count and its rolling mean.
TIMESERIES_CONTRACT = BlueprintContract.parse({
    "result_kind": "aggregate",
    "source_columns": ["Date of Admission"],
    "result_columns": [
        {"name": "month", "role": "key", "dtype": "string"},
        {"name": "admissions", "role": "metric", "dtype": "numeric"},
        {"name": "rolling_avg", "role": "metric", "dtype": "numeric"},
    ],
    "row_relation": {"type": "one_per_group", "group_by": ["month"]},
})


def test_guard_is_overridden_when_the_plan_averages_something_else():
    assert _contract_contradicts_math_guard(TIMESERIES_CONTRACT, MATH_ERR, {}) is True


def test_guard_stands_when_the_plan_really_does_average_the_text_column():
    """The case the guard exists for: "what is the average Youtuber?". A
    blueprint for that names the text column as the thing being averaged, so the
    guard's premise holds and the block must survive."""
    contract = BlueprintContract.parse({
        "result_kind": "scalar",
        "source_columns": ["Date of Admission"],
        "result_columns": [
            {"name": "average_date_of_admission", "role": "metric", "dtype": "numeric"},
        ],
    })
    assert _contract_contradicts_math_guard(contract, MATH_ERR, {}) is False


def test_no_contract_means_the_guard_keeps_its_original_behaviour():
    """Conservative by construction: without a contract there is no better
    evidence than the regex, so nothing changes."""
    assert _contract_contradicts_math_guard(None, MATH_ERR, {}) is False


def test_contract_without_a_numeric_metric_does_not_override():
    contract = BlueprintContract.parse({
        "result_kind": "subset",
        "result_columns": [{"name": "patient", "role": "key", "dtype": "string"}],
        "row_relation": {"type": "fewer_than_input"},
    })
    assert _contract_contradicts_math_guard(contract, MATH_ERR, {}) is False


def test_unparseable_guard_message_does_not_override():
    assert _contract_contradicts_math_guard(TIMESERIES_CONTRACT, "something else", {}) is False
    assert _contract_contradicts_math_guard(TIMESERIES_CONTRACT, "", {}) is False


@pytest.mark.parametrize("declared", [
    "Date of Admission",
    "date_of_admission",
    "DATE OF ADMISSION",
    "avg_date_of_admission",
])
def test_name_matching_is_normalised_so_the_guard_is_not_trivially_evaded(declared):
    """Re-spelling the flagged column must not slip past the guard."""
    contract = BlueprintContract.parse({
        "result_kind": "scalar",
        "result_columns": [{"name": declared, "role": "metric", "dtype": "numeric"}],
    })
    assert _contract_contradicts_math_guard(contract, MATH_ERR, {}) is False
