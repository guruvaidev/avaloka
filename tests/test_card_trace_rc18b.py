"""rc18b (438656fe) upload-card items from the rweb18b regrade.

The fixture (tests/fixtures/card_truth/rweb18b_card_trace.json) holds the
sentences rc18b showed and the columns behind them; the notes column is the
repo's own null-words workbook (tests/fixtures/null_words/three_sheets.xlsx),
the file the regrade uploaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.agents import visualization_agent as va

HERE = Path(__file__).parent
CASE = json.loads((HERE / "fixtures" / "card_truth" / "rweb18b_card_trace.json").read_text(encoding="utf-8"))
WORKBOOK = HERE / "fixtures" / "null_words" / "three_sheets.xlsx"


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)


def _notes_rows():
    """The 2022 sheet's cells as the upload's sample rows carry them: raw strings."""
    raw = pd.read_excel(WORKBOOK, sheet_name="2022", dtype=str, keep_default_na=False)
    return raw.to_dict("records")


# --------------------------------------------------------------------------- #
# 3. Null words follow the loader; a third-place tie is named
# --------------------------------------------------------------------------- #
def test_notes_null_words_are_labels_as_the_loader_reads_them():
    from avaloka.io.loader import load_dataset  # the CLI copy of the same rule

    shown = CASE["shown"]["notes"]
    assert "18 rows (30.0%) have no Notes" in shown and "n/a (9 rows" in shown
    loaded = load_dataset(str(WORKBOOK)).frame["Notes"]
    _p, _m, text = va._ground_categories(_notes_rows(), "Notes", None, "count", 7, True)
    # The loader keeps "none" and "NULL" in a text column as labels, like "n/a";
    # only the blank cell is missing.
    assert int(loaded.isna().sum()) == 1 and "1 row (1.7%) has no Notes" in text, text
    assert f"out of the {int(loaded.notna().sum())} rows with a value for Notes" in text, text
    assert "none (9 rows, 15.3%)" in text, text


def test_a_tie_at_the_third_place_names_every_tied_group():
    # The cells rc18b counted (its null words dropped), so the 8-row tie is the one it showed.
    rows = [r for r in _notes_rows() if r["Notes"] not in ("none", "NULL", "")]
    _p, _m, text = va._ground_categories(rows, "Notes", None, "count", 7, True)
    assert "- (8 rows, 19.0%) (" in CASE["shown"]["notes"]
    assert "then a tie at 8 rows (19.0% each) between -, ? and called back" in text, text


def test_bmi_na_is_missing_in_a_numeric_column():
    rows = [{"bmi": v} for v in CASE["stroke_bmi"]]
    assert CASE["shown"]["bmi"].startswith("Records by bmi: the largest groups are N/A (201 rows")
    _p, _m, text = va._ground_categories(rows, "bmi", None, "count", 15, False)
    assert "N/A (" not in text, text
    assert "201 rows (3.9%) have no bmi and are left out of the chart" in text, text


# --------------------------------------------------------------------------- #
# 4. An average over a handful of rows says so
# --------------------------------------------------------------------------- #
def test_a_one_row_average_says_it_is_one_row():
    groups = CASE["messy_quantity_by_stockcode"]
    assert groups["17084R"] == [1, 1440.0]
    rows = [{"StockCode": k, "Quantity": mean} for k, (n, mean) in groups.items() for _ in range(n)]
    _p, _m, text = va._ground_categories(rows, "StockCode", "Quantity", "mean", 15, False)
    assert CASE["shown"]["stockcode"].startswith("Average Quantity by StockCode: highest is 17084R (1,440), ")
    assert "highest is 17084R (1,440, from 1 row)" in text, text
    big = [{"g": g, "v": v} for g, v, n in (("a", 10.0, 30), ("b", 4.0, 25)) for _ in range(n)]
    _p, _m, same = va._ground_categories(big, "g", "v", "mean", 15, False)
    assert "highest is a (10), lowest is b (4)" in same and "row" not in same    # large groups: unchanged
