"""An ``exactly n`` row count is HARD only when the plan can back it up (rc11 p1).

"what are the strongest predictors of stroke?" on a DB thread got no answer:
the blueprint model wrote ``row_relation: exactly n=11`` -- its source_columns
count, the target ``stroke`` included -- and the correct 10-predictor result
was HARD-rejected as wrong_row_count. The fixture is p1's blueprint verbatim
(tests/fixtures/rc11_p1_blueprint.txt).

Deterministic: no LLM, no network.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pytest

from app.agents import blueprint_contract as bc
from app.agents.blueprint_contract import BlueprintContract, parse_blueprint

HERE = os.path.dirname(os.path.abspath(__file__))
P1_BLUEPRINT = os.path.join(HERE, "..", "fixtures", "rc11_p1_blueprint.txt")
P1_QUESTION = "what are the strongest predictors of stroke?"
PREDICTORS = ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type",
              "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]


def _patients(n: int = 60) -> pd.DataFrame:
    """The patients table's shape: an id column, ten predictors and stroke."""
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "id": rng.choice(np.arange(1000, 99999), size=n, replace=False),
        "gender": rng.choice(["Male", "Female"], size=n),
        "age": rng.uniform(1, 90, size=n).round(0),
        "hypertension": rng.integers(0, 2, size=n),
        "heart_disease": rng.integers(0, 2, size=n),
        "ever_married": rng.choice(["Yes", "No"], size=n),
        "work_type": rng.choice(["Private", "Self-employed", "Govt_job"], size=n),
        "Residence_type": rng.choice(["Urban", "Rural"], size=n),
        "avg_glucose_level": rng.uniform(55, 270, size=n).round(2),
        "bmi": rng.uniform(15, 45, size=n).round(1),
        "smoking_status": rng.choice(["never smoked", "smokes", "formerly smoked"], size=n),
        "stroke": rng.integers(0, 2, size=n),
    })


def _importance(features) -> pd.DataFrame:
    return pd.DataFrame({"feature": list(features),
                         "importance": np.linspace(1.0, 0.1, len(features))})


def _p1_contract(**overrides) -> BlueprintContract:
    with open(P1_BLUEPRINT, encoding="utf-8") as fh:
        _, contract, problems = parse_blueprint(fh.read(), schema=list(_patients().columns))
    assert contract is not None and problems == []
    if overrides:
        contract = BlueprintContract.parse({**contract.to_dict(), **overrides})
    return contract


# --- 1. p1 exactly: an invented count is advisory ---------------------------------

def test_p1_blueprint_and_ten_predictors_is_not_rejected(caplog):
    contract = _p1_contract()
    assert (contract.row_relation.type, contract.row_relation.n) == ("exactly", 11)

    with caplog.at_level(logging.WARNING, logger="app.agents.blueprint_contract"):
        report = bc.evaluate_contract(contract, _importance(PREDICTORS), _patients(), question=P1_QUESTION)

    assert report.violations == []
    assert [a.rule for a in report.advisories] == ["unverified_row_count"]
    assert report.row_count.source == "model" and not report.row_count.hard
    # The plan's own columns, target excluded, give 10 -- not the 11 it stated.
    assert report.row_count.derived == {"plan_columns": 10}
    assert any(r.levelno == logging.WARNING and "exactly 11" in r.message for r in caplog.records)


def test_p1_through_the_validator_node_answers_and_records_the_source():
    from app.agents import validator

    patients = _patients()
    state = {
        "user_prompt": P1_QUESTION,
        "coder_contract": _p1_contract().to_dict(),
        "execution_error": None,
        "execution_output_data": _importance(PREDICTORS).to_dict(orient="records"),
        "uploaded_csv_preview": patients.to_dict(orient="records"),
    }
    out = validator.contract_validator_node(state)
    assert out["contract_error"] is False, out
    assert out["contract_violations"] == []
    assert out["contract_row_count"]["source"] == "model"
    assert [a["rule"] for a in out["contract_advisories"]] == ["unverified_row_count"]


def test_invented_count_without_any_derivation_is_advisory():
    contract = BlueprintContract.parse({
        "result_kind": "aggregate", "result_columns": ["feature"],
        "row_relation": {"type": "exactly", "n": 7}})
    assert bc.check_contract(contract, _importance(PREDICTORS[:4]), _patients(), question="rank them") == []


# --- 2. guard: a derivable count that is violated is still HARD -------------------

def test_group_by_distinct_count_violated_is_rejected():
    source = pd.DataFrame({"hospital": ["A", "A", "B", "C", "C"], "bill": [1, 2, 3, 4, 5]})
    contract = BlueprintContract.parse({
        "result_kind": "aggregate", "source_columns": ["hospital", "bill"],
        "result_columns": [{"name": "hospital", "role": "key"}, "total"],
        "row_relation": {"type": "exactly", "n": 3, "group_by": ["hospital"]}})
    short = pd.DataFrame({"hospital": ["A", "B"], "total": [3, 3]})
    report = bc.evaluate_contract(contract, short, source)
    assert [v.rule for v in report.violations] == ["wrong_row_count"]
    assert report.row_count.source == "data"
    full = pd.DataFrame({"hospital": ["A", "B", "C"], "total": [3, 3, 9]})
    assert bc.check_contract(contract, full, source) == []


@pytest.mark.parametrize("question", ["show the top 5 hospitals by billing",
                                      "which five highest bills were paid?"])
def test_question_count_violated_is_rejected(question):
    source = pd.DataFrame({"hospital": list("ABCDEFG"), "bill": range(7)})
    contract = BlueprintContract.parse({
        "result_kind": "subset", "result_columns": ["hospital", "bill"],
        "row_relation": {"type": "exactly", "n": 5}})
    report = bc.evaluate_contract(contract, source.head(4), source, question=question)
    assert [v.rule for v in report.violations] == ["wrong_row_count"]
    assert report.row_count.source == "question"
    assert bc.check_contract(contract, source.head(5), source, question=question) == []


# --- 3. guard: the plan's column list, target (and identifiers) excluded ----------

@pytest.mark.parametrize("with_id", [False, True], ids=["p1-columns", "plus-id-column"])
def test_plan_columns_count_is_enforced(with_id):
    p1 = _p1_contract().to_dict()
    source_columns = p1["source_columns"] + (["id"] if with_id else [])
    contract = _p1_contract(source_columns=source_columns, row_relation={"type": "exactly", "n": 10})
    assert bc.exact_count_decision(contract, _patients(), P1_QUESTION).source == "plan_columns"

    ok = bc.evaluate_contract(contract, _importance(PREDICTORS), _patients(), question=P1_QUESTION)
    assert ok.violations == [] and ok.advisories == []

    short = bc.evaluate_contract(contract, _importance(PREDICTORS[:9]), _patients(), question=P1_QUESTION)
    assert [v.rule for v in short.violations] == ["wrong_row_count"]
    assert "derived from the plan columns" in short.violations[0].detail


# --- 4. one identifier rule: #573's IDENTIFIER_RULE ------------------------------

def test_dense_integer_key_without_an_id_name_is_an_identifier():
    """The case where #577's local rule and #573's rule disagreed.

    ``mrn`` (a medical record number) is a dense, all-distinct whole-number
    column whose name carries no id token. #573 calls it an identifier (rule
    (c): near-unique whole numbers packed into a range under twice the row
    count); the old local copy called every numeric column a measure, so the
    plan's column count kept it and derived 11 instead of 10.
    """
    patients = _patients()
    patients["mrn"] = np.random.default_rng(3).permutation(np.arange(100000, 100000 + len(patients)))
    assert bc.is_identifier_column("mrn", patients)

    p1 = _p1_contract().to_dict()
    contract = _p1_contract(source_columns=p1["source_columns"] + ["mrn"],
                            row_relation={"type": "exactly", "n": 10})
    decision = bc.exact_count_decision(contract, patients, P1_QUESTION)
    assert (decision.source, decision.derived) == ("plan_columns", {"plan_columns": 10})

    short = bc.evaluate_contract(contract, _importance(PREDICTORS[:9]), patients, question=P1_QUESTION)
    assert [v.rule for v in short.violations] == ["wrong_row_count"]
