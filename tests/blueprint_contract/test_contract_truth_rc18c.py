"""The contract's sentences say only what its evidence shows (rc18c).

Two false sentences from the rweb18c run:

* "what kinds of recipes dominate?" split each recipe name into tokens and
  counted them, one row per token. The token column is derived, so the check
  fell back to row counts and refused the answer with "the result has 174 rows
  against 100 input rows -- the raw rows were returned instead of the
  aggregate". 100 rows cannot come back as 174 raw rows.
* "What is the average fare_amount by payment_type?" got the advisory "data
  gives [-27.5, 94.0]); has -1.5166666666666668". Both numbers are the 100-row
  validation sample's; the full file spans [-404.1, 1616.5] and no payment
  type's full-data mean is negative.

The fixture (tests/fixtures/rc18c_contract_truth/turns.json) holds each turn's
question, blueprint contract, executed code and shown sentence. The tests
drive ``execute_code_node`` and ``contract_validator_node``, the path that
wrote them.

Deterministic: no LLM, no network.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from app.agents.blueprint_contract import BlueprintContract

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
with open(os.path.join(FIXTURES, "rc18c_contract_truth", "turns.json"), encoding="utf-8") as _fh:
    TURNS = json.load(_fh)
with open(os.path.join(FIXTURES, "rc16_taxi_avg_fare.json"), encoding="utf-8") as _fh:
    TAXI_SAMPLE = json.load(_fh)["sample_rows"]


def _run(monkeypatch, question: str, contract: dict, code: str, rows: list) -> dict:
    from app.agents import validator

    monkeypatch.delenv("AVALOKA_EXECUTOR_URL", raising=False)
    state = {
        "user_prompt": question,
        "generated_code": code,
        "uploaded_csv_preview": rows,
        "coder_contract": BlueprintContract.parse(contract).to_dict(),
    }
    executed = validator.execute_code_node(state)
    assert executed["execution_error"] is None, executed
    state.update(executed)
    out = validator.contract_validator_node(state)
    out["_rows"] = len(executed["execution_output_data"])
    return out


def _text(out: dict) -> str:
    return " ".join(v["detail"] for v in (out.get("contract_violations") or [])
                    + (out.get("contract_advisories") or []))


# --- bite: the recipes turn ----------------------------------------------------

def test_one_row_per_token_is_not_refused_as_raw_rows(monkeypatch):
    turn = TURNS["recipes"]
    assert "the raw rows were returned" in turn["shown"]
    rows = [{"Name": n} for n in turn["recipe_names"]]

    out = _run(monkeypatch, turn["question"], turn["contract"], turn["code"], rows)

    assert out["_rows"] > len(rows)          # more rows than the input: not its raw rows
    assert out["contract_error"] is False, out
    assert out["contract_violations"] == []
    assert "raw rows" not in _text(out)
    (note,) = [a for a in out["contract_advisories"] if a["rule"] == "aggregation_not_judged"]
    assert f"the result has {out['_rows']} rows from 100 input rows" in note["detail"]


def test_derived_key_with_as_many_rows_as_the_input_says_only_what_it_saw(monkeypatch):
    """Exactly the input's row count is still refused, in words the count backs."""
    turn = TURNS["recipes"]
    rows = [{"Name": f"Rezept {i}"} for i in range(100)]
    code = ("import pandas as pd\n\ndef main(df):\n"
            "    out = df.copy()\n"
            "    out['token'] = out['Name'].str.lower()\n"
            "    out['count'] = 1\n"
            "    return out[['token', 'count', 'Name']]\n")

    out = _run(monkeypatch, turn["question"], turn["contract"], code, rows)

    assert out["contract_error"] is True
    (v,) = [v for v in out["contract_violations"] if v["rule"] == "not_aggregated"]
    assert "raw rows were returned" not in v["detail"]
    assert "exactly as many rows as the 100-row validation sample (100) and all 1 of its columns" in v["detail"]


# --- bite: the taxi turn -------------------------------------------------------

def test_sample_range_and_value_are_named_as_the_samples(monkeypatch):
    turn = TURNS["taxi"]
    assert "data gives [-27.5, 94.0]" in turn["shown"]

    out = _run(monkeypatch, turn["question"], turn["contract"], turn["code"], TAXI_SAMPLE)

    assert out["contract_error"] is False, out
    (adv,) = [a for a in out["contract_advisories"] if a["rule"] == "unverified_range"]
    assert "data gives" not in adv["detail"]
    assert "the 100-row validation sample gives [-27.5, 94.0]" in adv["detail"]
    assert "the result on the validation sample has -1.5166666666666668" in adv["detail"]


def test_sample_group_count_is_named_as_the_samples(monkeypatch):
    """The distinct-group yardstick is the sample's too, and says so."""
    turn = TURNS["taxi"]
    code = "import pandas as pd\n\ndef main(df):\n    return df.rename(columns={'fare_amount': 'average_fare_amount'})\n"

    out = _run(monkeypatch, turn["question"], turn["contract"], code, TAXI_SAMPLE)

    (v,) = [v for v in out["contract_violations"] if v["rule"] == "not_aggregated"]
    assert v["detail"].startswith("the plan promised one row per ['payment_type']; "
                                  "the 100-row validation sample holds ")
