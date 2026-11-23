"""A min/max bound is HARD only when the plan can back it up (rc16 taxi).

"What is the average fare_amount by payment_type?" was refused in all three
rweb16 runs: the blueprint model declared ``average_fare_amount`` min 0.0, and
taxi fares include refunds and adjustments, so payment type 4's mean on the
validation sample is -1.52 and a correct result was HARD-rejected. The fixture
(tests/fixtures/rc16_taxi_avg_fare.json) holds that turn's blueprint contract
and code verbatim and the validator's 100-row sample, reduced to the two
columns the plan reads.

Deterministic: no LLM, no network.
"""

from __future__ import annotations

import json
import logging
import os

import pandas as pd
import pytest

from app.agents import blueprint_contract as bc
from app.agents.blueprint_contract import BlueprintContract, parse_blueprint

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "..", "fixtures", "rc16_taxi_avg_fare.json"), encoding="utf-8") as _fh:
    TAXI = json.load(_fh)


def _taxi_contract() -> BlueprintContract:
    _, contract, problems = parse_blueprint(TAXI["blueprint"], schema=["payment_type", "fare_amount"])
    assert contract is not None and problems == []
    return contract


def _avg_fare_contract(**bounds) -> BlueprintContract:
    return BlueprintContract.parse({
        "result_kind": "aggregate", "source_columns": ["payment_type", "fare_amount"],
        "result_columns": [{"name": "payment_type", "role": "key"},
                           {"name": "average_fare_amount", "dtype": "numeric", **bounds}],
        "row_relation": {"type": "one_per_group", "group_by": ["payment_type"]}})


# --- bite: the rweb16 taxi turn through the real validator path ---------------

def test_taxi_average_fare_is_answered_with_an_unverified_range_advisory(monkeypatch, caplog):
    from app.agents import validator

    monkeypatch.delenv("AVALOKA_EXECUTOR_URL", raising=False)
    assert "declared min 0.0 but has -1.51" in TAXI["refusal"]
    state = {
        "user_prompt": TAXI["question"],
        "generated_code": TAXI["code"],
        "uploaded_csv_preview": TAXI["sample_rows"],
        "coder_contract": _taxi_contract().to_dict(),
    }
    executed = validator.execute_code_node(state)
    assert executed["execution_error"] is None, executed
    state.update(executed)
    by_type = {r["payment_type"]: r["average_fare_amount"] for r in executed["execution_output_data"]}
    assert by_type[4] == pytest.approx(-1.5166666666666668)

    with caplog.at_level(logging.WARNING, logger="app.agents.blueprint_contract"):
        out = validator.contract_validator_node(state)

    assert out["contract_error"] is False, out
    assert out["contract_violations"] == []
    assert [a["rule"] for a in out["contract_advisories"]] == ["unverified_range"]
    (bound,) = out["contract_bounds"]
    assert (bound["column"], bound["side"], bound["value"], bound["source"]) == (
        "average_fare_amount", "min", 0.0, "model")
    # The sample's own fares reach below zero, so nothing backs min 0.
    assert bound["derived"]["data"][0] < 0
    assert any(r.levelno == logging.WARNING and "declared min 0.0" in r.message for r in caplog.records)


# --- guards: a derivable bound that is violated is still HARD -----------------

@pytest.mark.parametrize("column, bounds, values", [
    ("abnormal_rate", {"min": 0, "max": 1}, [0.2, 1.5]),
    ("share_of_trips", {"max": 1}, [0.4, 1.2]),
    ("trip_count", {"min": 0}, [3, -1]),
    ("pct_cash", {"max": 100}, [40.0, 130.0]),
])
def test_domain_bound_violated_is_rejected(column, bounds, values):
    contract = BlueprintContract.parse({
        "result_kind": "aggregate", "result_columns": [{"name": column, "dtype": "numeric", **bounds}],
        "row_relation": "unconstrained"})
    report = bc.evaluate_contract(contract, pd.DataFrame({column: values}),
                                  pd.DataFrame({"x": [1, 2, 3]}))
    assert [v.rule for v in report.violations] == ["out_of_range"]
    assert {b.source for b in report.bounds} == {"domain"}


def test_change_rate_is_not_a_domain_fraction():
    contract = BlueprintContract.parse({
        "result_kind": "aggregate",
        "result_columns": [{"name": "growth_rate", "dtype": "numeric", "min": 0}]})
    report = bc.evaluate_contract(contract, pd.DataFrame({"growth_rate": [-0.1, 0.3]}),
                                  pd.DataFrame({"x": [1]}))
    assert report.violations == [] and [a.rule for a in report.advisories] == ["unverified_range"]


def test_question_bound_violated_is_rejected():
    source = pd.DataFrame(TAXI["sample_rows"])
    result = source.groupby("payment_type", as_index=False).agg(average_fare_amount=("fare_amount", "mean"))
    question = "What is the average fare_amount by payment_type, counting only positive fares?"
    report = bc.evaluate_contract(_avg_fare_contract(min=0.0), result, source, question=question)
    assert [v.rule for v in report.violations] == ["out_of_range"]
    assert report.bounds[0].source == "question"


def test_question_qualifier_on_another_column_does_not_bound_this_one():
    source = pd.DataFrame(TAXI["sample_rows"])
    result = source.groupby("payment_type", as_index=False).agg(average_fare_amount=("fare_amount", "mean"))
    question = "What is the average fare_amount by payment_type for trips over 2 miles?"
    report = bc.evaluate_contract(_avg_fare_contract(min=2.0), result, source, question=question)
    assert report.violations == [] and report.bounds[0].source == "model"


def test_data_bound_violated_is_rejected():
    # Every fare in this input is >= 3, so a correct mean is >= 3 and a
    # declared min 0 is implied by the data; a negative mean is a wrong result.
    source = pd.DataFrame({"payment_type": [1, 1, 2, 2], "fare_amount": [3.0, 9.0, 4.0, 20.0]})
    bad = pd.DataFrame({"payment_type": [1, 2], "average_fare_amount": [6.0, -1.0]})
    report = bc.evaluate_contract(_avg_fare_contract(min=0.0), bad, source)
    assert [v.rule for v in report.violations] == ["out_of_range"]
    assert report.bounds[0].source == "data" and report.bounds[0].derived["data"] == [3.0, 20.0]

    # The same for a max: a mean above the column's observed maximum.
    too_high = pd.DataFrame({"payment_type": [1, 2], "average_fare_amount": [6.0, 25.0]})
    report = bc.evaluate_contract(_avg_fare_contract(max=20.0), too_high, source)
    assert [v.rule for v in report.violations] == ["out_of_range"]
    assert report.bounds[0].source == "data"

    good = pd.DataFrame({"payment_type": [1, 2], "average_fare_amount": [6.0, 12.0]})
    assert bc.check_contract(_avg_fare_contract(min=0.0, max=20.0), good, source) == []
