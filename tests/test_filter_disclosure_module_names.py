"""A scope line never names a module or a function as a column (rc18b).

Asked "What drives Outcome?", the code computed its correlations in a helper,
``mask = pd.notna(x) & pd.notna(y); pearsonr(x[mask], y[mask])``, and the
reply said "Scope: ... (only where pd has a value and pd has a value)". ``pd``
is the pandas module: the null test was read off ``pd.notna``'s receiver, not
its argument. The fixture (tests/fixtures/rc18b_contract_sample/turns.json,
``scope_turn``) holds that turn's question, executed code and shown sentence.

Deterministic: no LLM, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.filter_disclosure import detect_row_filters, scope_filters

FIXTURE = Path(__file__).parent / "fixtures" / "rc18b_contract_sample" / "turns.json"
TURN = json.loads(FIXTURE.read_text(encoding="utf-8"))["scope_turn"]
CODE = TURN["code"].replace("{path_2022}", "/data/a.csv").replace("{path_2023}", "/data/b.csv")


def test_live_nullxlsx_scope_names_the_tested_columns():
    assert "only where pd has a value" in TURN["shown"]

    lines = [f.line() for f in scope_filters(CODE, TURN["question"], "")]

    assert TURN["shown"] not in lines
    assert not any("pd has" in line for line in lines), lines
    assert any("only where Glucose_num has a value and Outcome has a value" in line for line in lines), lines


def test_live_nullxlsx_filters_hold_no_module_column():
    filters = detect_row_filters(CODE)
    assert filters
    for flt in filters:
        assert "pd" not in flt.columns and "np" not in flt.columns, flt.as_dict()
        assert "pd " not in (flt.readable or "") and "pd" not in (flt.excluded or "").split()


@pytest.mark.parametrize("mask, words", [
    ("pd.notna(df['Glucose'])", "Glucose has a value"),
    ("~pd.isna(df['Glucose'])", "Glucose has a value"),
    ("pd.notnull(df.Glucose)", "Glucose has a value"),
    ("df['Glucose'].notna()", "Glucose has a value"),
])
def test_function_form_null_test_names_its_argument(mask, words):
    code = ("import pandas as pd\n\ndef main(df):\n"
            f"    kept = df[{mask}]\n"
            "    return kept.groupby('Outcome', as_index=False)['Glucose'].mean()\n")

    (flt,) = detect_row_filters(code)

    assert flt.readable == words
    assert flt.columns == ["Glucose"]


def test_a_null_test_on_no_column_says_it_could_not_be_named():
    code = ("import numpy as np\nimport pandas as pd\n\ndef main(df):\n"
            "    score = np.log(df['Glucose']) + np.log(df['Age'])\n"
            "    kept = df[pd.notna(score)]\n"
            "    return kept.groupby('Outcome', as_index=False)['Glucose'].mean()\n")

    (flt,) = detect_row_filters(code)
    line = flt.line()

    assert "could not be named" in line
    for name in ("pd", "np", "score"):
        assert f"{name} has" not in line and name not in flt.columns, line
