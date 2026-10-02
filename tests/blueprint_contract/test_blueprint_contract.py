"""Unit tests for the blueprint contract vocabulary, parser and checker.

Deterministic: no LLM, no network, no fixtures beyond small in-memory frames.
These run in the default hermetic suite.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.agents.blueprint_contract import (
    BlueprintContract,
    ContractError,
    check_contract,
    extract_contract_block,
    format_violations,
    parse_blueprint,
)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _blueprint(contract: dict, steps: str = "1. Do a thing.\n2. Do another.") -> str:
    return f"{steps}\n\n```json\n{json.dumps(contract)}\n```"


AGG_CONTRACT = {
    "result_kind": "aggregate",
    "source_columns": ["hospital", "test_result"],
    "result_columns": [
        {"name": "hospital", "role": "key", "dtype": "string"},
        {"name": "abnormal_rate", "role": "metric", "dtype": "numeric", "min": 0, "max": 1},
    ],
    "row_relation": {"type": "one_per_group", "group_by": ["hospital"]},
    "unique_key": ["hospital"],
}


def test_parses_steps_and_contract():
    steps, contract, problems = parse_blueprint(_blueprint(AGG_CONTRACT))
    assert "1. Do a thing." in steps
    assert "```" not in steps
    assert contract is not None
    assert contract.result_kind == "aggregate"
    assert contract.row_relation.type == "one_per_group"
    assert contract.row_relation.group_by == ("hospital",)
    assert problems == []


def test_parses_bare_trailing_object_without_fences():
    text = "1. Group the rows.\n\n" + json.dumps(AGG_CONTRACT)
    _steps, contract, _ = parse_blueprint(text)
    assert contract is not None
    assert contract.result_kind == "aggregate"


def test_accepts_contract_nested_under_a_contract_key():
    text = _blueprint({"contract": AGG_CONTRACT})
    _steps, contract, _ = parse_blueprint(text)
    assert contract is not None
    assert contract.result_kind == "aggregate"


def test_missing_contract_degrades_to_none_not_an_error():
    steps, contract, problems = parse_blueprint("1. Just some prose steps.")
    assert steps.startswith("1. Just some prose")
    assert contract is None
    assert problems and "did not include a contract" in problems[0]


def test_vacuous_contract_is_discarded():
    """A contract asserting nothing is worse than none: it looks like a gate
    while passing everything."""
    _steps, contract, problems = parse_blueprint(
        _blueprint({"result_kind": "row_transform"})
    )
    assert contract is None
    assert any("no checkable postcondition" in p for p in problems)


def test_unknown_result_kind_is_rejected():
    _steps, contract, problems = parse_blueprint(
        _blueprint({"result_kind": "teleport", "result_columns": ["x"]})
    )
    assert contract is None
    assert any("contract rejected" in p for p in problems)


def test_one_per_group_without_group_by_degrades_to_fewer_than_input():
    """A group relation with no group is not a constraint; it must not silently
    read as one."""
    _steps, contract, _ = parse_blueprint(_blueprint({
        "result_columns": ["total"],
        "row_relation": {"type": "one_per_group"},
    }))
    assert contract is not None
    assert contract.row_relation.type == "fewer_than_input"


def test_source_columns_absent_from_schema_are_reported():
    _steps, contract, problems = parse_blueprint(
        _blueprint(AGG_CONTRACT),
        schema={"hospital": "object", "billing": "float64"},
    )
    assert contract is not None  # reported, not fatal
    assert any("test_result" in p for p in problems)


def test_garbage_in_the_json_block_does_not_raise():
    steps, contract, problems = parse_blueprint("1. Steps.\n```json\n{not json\n```")
    assert contract is None
    assert steps
    assert problems


def test_extract_contract_block_prefers_the_last_block():
    text = (
        "```json\n" + json.dumps({"result_kind": "scalar", "result_columns": ["a"]}) + "\n```\n"
        "```json\n" + json.dumps(AGG_CONTRACT) + "\n```"
    )
    raw = extract_contract_block(text)
    assert raw["result_kind"] == "aggregate"


def test_round_trips_through_to_dict():
    contract = BlueprintContract.parse(AGG_CONTRACT)
    again = BlueprintContract.parse(contract.to_dict())
    assert again.result_kind == contract.result_kind
    assert again.row_relation == contract.row_relation
    assert [c.name for c in again.result_columns] == [c.name for c in contract.result_columns]


def test_parse_rejects_non_object():
    with pytest.raises(ContractError):
        BlueprintContract.parse(["not", "an", "object"])


# ---------------------------------------------------------------------------
# checking
# ---------------------------------------------------------------------------

SOURCE = pd.DataFrame({
    "hospital": ["A", "A", "B", "B", "C"],
    "test_result": ["Abnormal", "Normal", "Normal", "Normal", "Abnormal"],
    "billing": [10.0, 20.0, 30.0, 40.0, 50.0],
})


def _check(contract_dict, result):
    return check_contract(BlueprintContract.parse(contract_dict), result, SOURCE)


def test_correct_aggregate_passes():
    result = pd.DataFrame({"hospital": ["A", "B", "C"], "abnormal_rate": [0.5, 0.0, 1.0]})
    assert _check(AGG_CONTRACT, result) == []


def test_returning_the_raw_rows_for_an_aggregation_is_caught():
    """This is the failure the coder prompt's four-part 'AGGREGATION / CHART
    OUTPUT RULE' was written for. One structured postcondition replaces it and
    catches it deterministically."""
    violations = _check(AGG_CONTRACT, SOURCE.copy())
    rules = {v.rule for v in violations}
    assert "not_aggregated" in rules or "duplicate_group_rows" in rules


def test_duplicate_group_rows_are_caught():
    result = pd.DataFrame({
        "hospital": ["A", "A", "B"],
        "abnormal_rate": [0.5, 0.4, 0.0],
    })
    assert any(v.rule == "duplicate_group_rows" for v in _check(AGG_CONTRACT, result))


def test_missing_promised_column_is_caught():
    result = pd.DataFrame({"hospital": ["A", "B", "C"], "something_else": [1, 2, 3]})
    violations = _check(AGG_CONTRACT, result)
    assert any(v.rule == "missing_column" for v in violations)


def test_rate_outside_declared_range_is_caught():
    """A 'rate' of 42 is a real, silent, frequently-shipped defect: a percentage
    computed where a proportion was promised, or a sum where a mean was meant."""
    result = pd.DataFrame({"hospital": ["A", "B", "C"], "abnormal_rate": [0.5, 42.0, 1.0]})
    assert any(v.rule == "out_of_range" for v in _check(AGG_CONTRACT, result))


def test_wrong_dtype_is_caught():
    result = pd.DataFrame({"hospital": ["A", "B", "C"], "abnormal_rate": ["a", "b", "c"]})
    assert any(v.rule == "wrong_dtype" for v in _check(AGG_CONTRACT, result))


def test_column_matching_is_normalised_not_literal():
    """The contract states intent, not spelling. Holding generated code to an
    exact label would measure naming luck and produce false rejections."""
    result = pd.DataFrame({"Hospital": ["A", "B", "C"], "Abnormal Rate": [0.5, 0.0, 1.0]})
    assert _check(AGG_CONTRACT, result) == []


def test_same_as_input_relation():
    contract = {
        "result_kind": "row_transform",
        "result_columns": ["billing"],
        "row_relation": {"type": "same_as_input"},
    }
    ok = SOURCE.assign(billing_doubled=SOURCE["billing"] * 2)
    assert _check(contract, ok) == []
    assert any(v.rule == "row_count_changed"
               for v in _check(contract, SOURCE.head(2)))


def test_exactly_and_at_most_relations():
    at_most = {"result_kind": "aggregate", "result_columns": ["billing"],
               "row_relation": {"type": "at_most", "n": 2}}
    assert _check(at_most, SOURCE.head(2)) == []
    assert any(v.rule == "too_many_rows" for v in _check(at_most, SOURCE))

    exactly = {"result_kind": "aggregate", "result_columns": ["billing"],
               "row_relation": {"type": "exactly", "n": 3}}
    assert any(v.rule == "wrong_row_count" for v in _check(exactly, SOURCE.head(2)))


def test_scalar_result_must_be_one_row():
    contract = {"result_kind": "scalar", "result_columns": ["billing"]}
    assert _check(contract, SOURCE.head(1)) == []
    assert any(v.rule == "not_scalar" for v in _check(contract, SOURCE))


def test_not_null_policy():
    contract = {
        "result_kind": "aggregate",
        "result_columns": ["hospital"],
        "row_relation": {"type": "fewer_than_input"},
        "not_null": ["hospital"],
    }
    bad = pd.DataFrame({"hospital": ["A", None]})
    assert any(v.rule == "unexpected_nulls" for v in _check(contract, bad))


def test_unique_key_policy():
    contract = {
        "result_kind": "aggregate",
        "result_columns": ["hospital"],
        "row_relation": {"type": "fewer_than_input"},
        "unique_key": ["hospital"],
    }
    bad = pd.DataFrame({"hospital": ["A", "A"]})
    assert any(v.rule == "key_not_unique" for v in _check(contract, bad))


def test_non_dataframe_result_is_a_violation_not_a_crash():
    contract = BlueprintContract.parse(AGG_CONTRACT)
    assert check_contract(contract, None, SOURCE)
    assert check_contract(contract, 42, SOURCE)


def test_series_result_is_accepted_as_a_frame():
    contract = {"result_kind": "scalar", "result_columns": ["billing"]}
    s = pd.Series([1.0], name="billing")
    assert check_contract(BlueprintContract.parse(contract), s, SOURCE) == []


def test_violation_feedback_names_the_postcondition_and_the_contract():
    violations = _check(AGG_CONTRACT, SOURCE.copy())
    text = format_violations(violations, BlueprintContract.parse(AGG_CONTRACT))
    assert "acceptance criteria" in text
    assert "one row per" in text or "raw rows" in text
    # The coder must be able to see what it promised.
    assert "one_per_group" in text
    assert "Do not change the contract to fit the code." in text


# ---------------------------------------------------------------------------
# sample-degeneracy regressions
#
# Validation runs against a SAMPLE. Every shape postcondition therefore has to
# hold at small n, or the gate rejects correct code. These are the cases that
# actually fired during measurement.
# ---------------------------------------------------------------------------

def test_near_unique_group_key_is_not_reported_as_unaggregated():
    """The regression that made the first cut of this gate worse than nothing.

    In a 100-row sample of a real dataset a key like Hospital is near-unique, so
    a correct group-by returns roughly as many rows as it was given. Comparing
    the result to the input ROW COUNT called that "you returned the raw rows".
    The comparison has to be against the number of DISTINCT GROUPS.
    """
    source = pd.DataFrame({
        "hospital": [f"H{i}" for i in range(50)],
        "test_result": ["Abnormal", "Normal"] * 25,
    })
    # Correct aggregation: one row per hospital. 50 rows in, 50 rows out.
    result = pd.DataFrame({
        "hospital": [f"H{i}" for i in range(50)],
        "abnormal_rate": [0.0, 1.0] * 25,
    })
    violations = check_contract(BlueprintContract.parse(AGG_CONTRACT), result, source)
    assert violations == [], [str(v) for v in violations]


def test_genuine_failure_to_aggregate_is_still_caught_at_small_n():
    """The relaxation above must not blunt the check it exists for."""
    source = pd.DataFrame({
        "hospital": ["A"] * 25 + ["B"] * 25,
        "test_result": ["Abnormal", "Normal"] * 25,
    })
    violations = check_contract(
        BlueprintContract.parse(AGG_CONTRACT), source.copy(), source
    )
    assert any(v.rule in {"not_aggregated", "duplicate_group_rows"} for v in violations)


def test_fewer_than_input_tolerates_an_equal_count_on_a_sample():
    contract = {
        "result_kind": "aggregate",
        "result_columns": ["billing"],
        "row_relation": {"type": "fewer_than_input"},
    }
    assert _check(contract, SOURCE.copy()) == []
    doubled = pd.concat([SOURCE, SOURCE], ignore_index=True)
    assert any(v.rule == "not_reduced" for v in _check(contract, doubled))


def test_derived_group_key_absent_from_source_falls_back_to_row_count():
    """When the grouping column was derived (an hour, a month, a band) it is not
    in the input, so the distinct-group yardstick is unavailable."""
    source = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=48, freq="h")})
    contract = BlueprintContract.parse({
        "result_kind": "aggregate",
        "result_columns": [{"name": "hour", "role": "key"}],
        "row_relation": {"type": "one_per_group", "group_by": ["hour"]},
    })
    good = pd.DataFrame({"hour": list(range(24)), "n": [2] * 24})
    assert check_contract(contract, good, source) == []

    bad = pd.DataFrame({"hour": list(range(48)), "n": [1] * 48})
    assert any(v.rule == "not_aggregated" for v in check_contract(contract, bad, source))


def test_data_dependent_column_labels_are_not_required_literally():
    """A pivot names its output columns from the data, so the blueprint can only
    write a placeholder. Demanding it literally rejected correct pivots."""
    from app.agents.blueprint_contract import is_placeholder_column

    for name in ["<any medical condition>", "<Medical Condition>", "one column per condition",
                 "{condition}", "each condition"]:
        assert is_placeholder_column(name), name
    for name in ["hospital", "abnormal_rate", "Age Band", "total_records"]:
        assert not is_placeholder_column(name), name


def test_pivot_with_placeholder_columns_passes():
    source = pd.DataFrame({
        "age": [21, 35, 47, 52, 63],
        "condition": ["Cancer", "Asthma", "Cancer", "Obesity", "Asthma"],
    })
    contract = BlueprintContract.parse({
        "result_kind": "reshape",
        "result_columns": [
            {"name": "age_band", "role": "key", "dtype": "string"},
            {"name": "<any condition>", "role": "metric", "dtype": "numeric"},
        ],
        "row_relation": {"type": "at_most", "n": 30},
    })
    result = pd.DataFrame({
        "age_band": ["20-29", "30-39", "40-49", "50-59", "60-69"],
        "Cancer": [1, 0, 1, 0, 0],
        "Asthma": [0, 1, 0, 0, 1],
        "Obesity": [0, 0, 0, 1, 0],
    })
    assert check_contract(contract, result, source) == []


def test_a_real_named_column_is_still_required_alongside_placeholders():
    source = pd.DataFrame({"age": [21, 35], "condition": ["Cancer", "Asthma"]})
    contract = BlueprintContract.parse({
        "result_kind": "reshape",
        "result_columns": [
            {"name": "age_band", "role": "key", "dtype": "string"},
            {"name": "<any condition>", "role": "metric", "dtype": "numeric"},
        ],
        "row_relation": {"type": "at_most", "n": 30},
    })
    missing_key = pd.DataFrame({"Cancer": [1], "Asthma": [1]})
    assert any(v.rule == "missing_column" for v in check_contract(contract, missing_key, source))
