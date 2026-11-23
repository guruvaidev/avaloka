"""The result contract compares the result only with the rows it came from (rc18b).

The validator runs ``main(df)`` on a 100-row sample and checks the result
against that sample. Asked "what sells most?" of a two-sheet workbook, the
code ignored ``df`` and read both full sheets by path; a correct
one-row-per-product summary (2,776 rows) was withheld because "the input holds
73 distinct group(s)" -- the 73 (StockCode, Description) pairs of the sample.
"who are your best customers" was refused the same way ("100 distinct
group(s) but 1999 rows"), and a third run, which used ``df`` and one full
sheet, got a HARD ``out_of_range`` from a min 0 "derived from the data" of a
sample whose quantities were all positive.

The fixture (tests/fixtures/rc18b_contract_sample/turns.json) holds each
turn's question, blueprint contract, executed code and the sentence the user
was shown, from the rweb18b run. The tests write data files of the same shape,
point the code's paths at them and drive ``execute_code_node`` and
``contract_validator_node`` -- the path that produced the refusals.

Deterministic: no LLM, no network.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from app.agents.blueprint_contract import BlueprintContract, evaluate_contract

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "..", "fixtures", "rc18b_contract_sample", "turns.json"), encoding="utf-8") as _fh:
    TURNS = json.load(_fh)["contract_turns"]


def _code(turn: dict, paths: dict) -> str:
    code = turn["code"]
    for name, path in paths.items():
        code = code.replace("{" + name + "}", str(path))
    assert "{" + next(iter(paths)) + "}" not in code
    return code


def _retail(pairs: range, first_pairs: int = 0, quantity=lambda i: 2) -> pd.DataFrame:
    """Invoice lines over products ``P<i>``; the first 100 rows repeat
    ``first_pairs`` products, as the sample of the live sheet did."""
    rows = []
    if first_pairs:
        rows += [(i % first_pairs) for i in range(100)]
    rows += [i for i in pairs if not first_pairs or i >= first_pairs]
    return pd.DataFrame({
        "InvoiceNo": [f"5{n:05d}" for n in range(len(rows))],
        "StockCode": [f"{20000 + i}" for i in rows],
        "Description": [f"PRODUCT {i}" for i in rows],
        "Quantity": [quantity(i) for i in rows],
        "InvoiceDate": "2010-12-01 08:26:00",
        "UnitPrice": 1.25,
    })


def _run(monkeypatch, turn: dict, code: str, sample: pd.DataFrame) -> dict:
    from app.agents import validator

    monkeypatch.delenv("AVALOKA_EXECUTOR_URL", raising=False)
    state = {
        "user_prompt": turn["question"],
        "generated_code": code,
        "sample_data": sample.to_csv(index=False),
        "coder_contract": BlueprintContract.parse(turn["contract"]).to_dict(),
    }
    executed = validator.execute_code_node(state)
    assert executed["execution_error"] is None, executed
    state.update(executed)
    out = validator.contract_validator_node(state)
    out["_rows"] = len(executed["execution_output_data"])
    return out


def _all_text(out: dict) -> str:
    return " ".join(v["detail"] for v in (out.get("contract_violations") or [])
                    + (out.get("contract_advisories") or []))


# --- bite: the three live refusals -------------------------------------------

def test_messy_two_sheet_summary_is_not_refused_with_the_sample_group_count(tmp_path, monkeypatch):
    turn = TURNS["messy_rw7b"]
    assert "73 distinct group(s) but 2776 rows" in turn["shown"]
    sheet_2010 = _retail(range(0, 2000), first_pairs=73)
    sheet_2011 = _retail(range(1500, 2776))
    p2010, p2011 = tmp_path / "2010.csv", tmp_path / "2011.csv"
    sheet_2010.to_csv(p2010, index=False)
    sheet_2011.to_csv(p2011, index=False)
    sample = sheet_2010.head(100)
    assert len(sample[["StockCode", "Description"]].drop_duplicates()) == 73

    out = _run(monkeypatch, turn, _code(turn, {"path_2010": p2010, "path_2011": p2011}), sample)

    assert out["_rows"] == 2776
    assert out["contract_error"] is False, out
    assert out["contract_violations"] == []
    assert "distinct group(s)" not in _all_text(out)
    assert [a["rule"] for a in out["contract_advisories"]] == ["input_not_compared"]


def test_best_customers_join_is_not_refused_with_the_sample_group_count(tmp_path, monkeypatch):
    turn = TURNS["customers"]
    assert "100 distinct group(s) but 1999 rows" in turn["shown"]
    ids = range(1, 2001)
    customers = pd.DataFrame({"customer_id": list(ids),
                              "region": [("West", "East", "North")[i % 3] for i in ids],
                              "signup_year": [2019 + i % 6 for i in ids],
                              "segment": [("consumer", "smb", "enterprise")[i % 3] for i in ids]})
    # Customer 2000's only order was returned, so 1,999 customers have spend.
    orders = pd.DataFrame({"customer_id": list(ids) + [7],
                           "amount": [10.0 + i % 50 for i in ids] + [99.0],
                           "returned": [i == 2000 for i in ids] + [False]})
    pc, po = tmp_path / "customers.csv", tmp_path / "orders.csv"
    customers.to_csv(pc, index=False)
    orders.to_csv(po, index=False)

    out = _run(monkeypatch, turn, _code(turn, {"customers_path": pc, "orders_path": po}), customers.head(100))

    assert out["_rows"] == 1999
    assert out["contract_error"] is False, out
    assert "distinct group(s)" not in _all_text(out)


def test_sample_plus_full_sheet_gets_no_bound_derived_from_the_sample(tmp_path, monkeypatch):
    turn = TURNS["messy_rw7a"]
    assert "declared min 0.0 but has -366.0 (derived from the data)" in turn["shown"]
    sheet_2010 = _retail(range(0, 400), first_pairs=73)
    # Q1 holds returns: product 3000 nets -366.
    sheet_q1 = _retail(range(2900, 3100), quantity=lambda i: -366 if i == 3000 else 4)
    pq1 = tmp_path / "q1.csv"
    sheet_q1.to_csv(pq1, index=False)
    sample = sheet_2010.head(100)
    assert (sample["Quantity"] > 0).all()

    out = _run(monkeypatch, turn, _code(turn, {"q1_path": pq1}), sample)

    assert out["contract_error"] is False, out
    assert out["contract_violations"] == []
    assert "derived from the data" not in _all_text(out)
    assert "distinct group(s)" not in _all_text(out)
    qty = [b for b in out["contract_bounds"] if b["column"] == "TotalQuantity"]
    assert qty and all(b["source"] == "model" and "data" not in b["derived"] for b in qty)
    assert any(a["rule"] == "unverified_range" and "not the result's only input" in a["detail"]
               for a in out["contract_advisories"])


def test_result_keys_the_input_lacks_are_not_compared_with_its_group_count():
    """Without the code: a per-group result holding keys the input does not
    was computed on other rows, so its row count says nothing about them."""
    contract = BlueprintContract.parse(TURNS["messy_rw7b"]["contract"])
    source = _retail(range(0, 100), first_pairs=73).head(100)
    result = (_retail(range(0, 2776)).groupby(["StockCode", "Description"], as_index=False)
              .agg(TotalQuantity=("Quantity", "sum")))
    result["TotalRevenue"] = result["TotalQuantity"] * 1.25

    report = evaluate_contract(contract, result, source)

    assert [v.rule for v in report.violations] == []
    assert "distinct group(s)" not in " ".join(v.detail for v in report.violations + report.advisories)


# --- guards: a wrong result on the sample is still refused, truthfully ---------

def test_raw_sample_rows_are_still_not_aggregated(monkeypatch):
    turn = TURNS["messy_rw7b"]
    sample = _retail(range(0, 100), first_pairs=73).head(100)
    code = "import pandas as pd\n\ndef main(df):\n    return df[['StockCode', 'Description', 'Quantity']]\n"

    out = _run(monkeypatch, turn, code, sample)

    assert out["contract_error"] is True
    rules = [v["rule"] for v in out["contract_violations"]]
    assert "not_aggregated" in rules and "duplicate_group_rows" in rules
    assert any("holds 73 distinct group(s) but 100 rows" in v["detail"] for v in out["contract_violations"])


def test_duplicate_group_rows_are_caught_whatever_the_code_read(tmp_path, monkeypatch):
    turn = TURNS["messy_rw7b"]
    sheet = _retail(range(0, 300), first_pairs=73)
    path = tmp_path / "sheet.csv"
    sheet.to_csv(path, index=False)
    code = ("import pandas as pd\n\ndef main(df):\n"
            f"    full = pd.read_csv({str(path)!r})\n"
            "    full['TotalQuantity'] = full['Quantity']\n"
            "    full['TotalRevenue'] = full['Quantity'] * full['UnitPrice']\n"
            "    return full[['StockCode', 'Description', 'TotalQuantity', 'TotalRevenue']]\n")

    out = _run(monkeypatch, turn, code, sheet.head(100))

    rules = [v["rule"] for v in out["contract_violations"]]
    assert "duplicate_group_rows" in rules
    assert "not_aggregated" not in rules


@pytest.mark.parametrize("raw", [85123, 85123.0, "85123", " 85123 "])
def test_one_key_read_differently_is_still_one_key(raw):
    """The live StockCode came back an int from one frame and a float from
    another; that is not a foreign key, so the sample comparison still runs."""
    contract = BlueprintContract.parse({
        "result_kind": "aggregate", "source_columns": ["StockCode", "Quantity"],
        "result_columns": [{"name": "StockCode", "role": "key"}],
        "row_relation": {"type": "one_per_group", "group_by": ["StockCode"]}})
    source = pd.DataFrame({"StockCode": [raw, raw], "Quantity": [1, 2]})
    result = pd.DataFrame({"StockCode": [85123, 85123]})

    rules = [v.rule for v in evaluate_contract(contract, result, source).violations]

    assert "not_aggregated" in rules
