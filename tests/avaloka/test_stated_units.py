"""Units the file states are carried; units it does not state never appear.

Two benchmark tasks were declared known defects until this change
(xlsx_thousands_scaling_in_title, xlsx_sec_financial_report). A sheet that
says "Revenue by region (£ thousands)" once, in its title, had that title
discarded with the rest of the preamble, so the total came out a thousand
times small and no artifact said in what. No Avaloka artifact emitted a unit
at all.

The rule (avaloka/units.py): a unit is read out of the file's own text -- a
title row or a header cell -- through fixed tables, carries where it came
from, and is reported beside the figures, which stay as written. A file that
states nothing gets nothing, and neither narration path may add a unit.

The mission-level tests at the top fail on the reader before this change by
assertion.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("openpyxl", reason="writing the workbooks needs openpyxl")

from avaloka.benchmark.datasets_spreadsheet import (  # noqa: E402
    _thousands_total,
    build_parenthesised_negatives,
    build_thousands_scaling_in_title,
)
from avaloka.io.loader import load_dataset  # noqa: E402
from avaloka.mission.budget import Budget  # noqa: E402
from avaloka.mission.context import MissionContext, MissionKind  # noqa: E402
from avaloka.missions import run_analyze  # noqa: E402


def _mission(tmp_path, source, goal):
    out = tmp_path / "bundle"
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal=goal, data_source=str(source),
                         output_dir=out, budget=Budget(limit_usd=None))
    run_analyze(ctx)
    return ctx, out


def _titled(tmp_path, title, columns, name="t.xlsx"):
    path = tmp_path / name
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(columns).to_excel(writer, index=False, startrow=1)
        writer.sheets["Sheet1"].cell(row=1, column=1, value=title)
    return path


# --------------------------------------------------------------------------
# Through the mission
# --------------------------------------------------------------------------
def test_a_total_off_a_thousands_sheet_is_given_at_full_scale_and_reconciles(tmp_path):
    book = build_thousands_scaling_in_title(tmp_path / "rev.xlsx")
    ctx, out = _mission(tmp_path, book, "What is the total Revenue?")
    revenue = json.loads((out / "goal_analysis.json").read_text())["metrics"]["Revenue"]

    assert revenue.get("sum") == pytest.approx(_thousands_total() / 1000)      # as written
    assert revenue.get("sum_at_full_scale") == pytest.approx(_thousands_total())
    assert revenue.get("unit") == "GBP thousands"
    assert revenue.get("unit_at_full_scale") == "GBP"
    checks = {c["name"]: c for c in json.loads((out / "validation_report.json").read_text())["checks"]}
    assert checks["goal_reconciliation"]["status"] == "pass", checks["goal_reconciliation"]


def test_the_profile_says_the_unit_and_where_the_file_says_it(tmp_path):
    book = build_thousands_scaling_in_title(tmp_path / "rev.xlsx")
    ctx, out = _mission(tmp_path, book, "What is the total Revenue?")
    profile = json.loads((out / "data_quality.json").read_text())
    columns = {c["name"]: c for c in profile["columns"]}

    assert columns["Revenue"].get("unit", {}).get("source") == {
        "where": "title", "sheet_row": 1, "text": "Revenue by region (£ thousands)"}
    assert "unit" not in columns["Region"]
    kinds = {i["kind"]: i for i in profile["quality"]["issues"]}
    assert "GBP thousands (the title on row 1)" in kinds.get("unit_stated_in_title", {}).get("detail", "")
    # The figures are not multiplied: the profile reads what the cells hold.
    assert columns["Revenue"]["stats"]["min"] == pytest.approx(482.5)


def test_a_file_that_states_no_unit_gets_none_anywhere(tmp_path):
    book = build_parenthesised_negatives(tmp_path / "ledger.xlsx")
    ctx, out = _mission(tmp_path, book, "What is the total Movement?")
    for artifact in ("goal_analysis.json", "data_quality.json", "executive_report.html",
                     "technical_report.html"):
        text = (out / artifact).read_text()
        for token in ("£", "GBP", "USD", "thousand", "full_scale", '"unit"'):
            assert token not in text, f"{artifact} carries {token!r} the file never stated"


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------
def test_count_and_rate_columns_are_not_in_the_title_currency(tmp_path):
    path = _titled(tmp_path, "Branch results (£ thousands)", {
        "Branch": ["a", "b"], "Revenue": [10.0, 20.0], "Headcount": [3, 4], "Margin %": [0.1, 0.2]})
    units = load_dataset(str(path)).structure["units"]
    assert set(units["columns"]) == {"Revenue"}
    assert sorted(units["exempt"]) == ["Headcount", "Margin %"]


def test_a_stated_code_beats_a_bare_dollar_and_a_bare_dollar_stays_a_symbol(tmp_path):
    from avaloka.units import read_unit

    assert read_unit("CONSOLIDATED STATEMENTS - USD ($) $ in Thousands") == {
        "scale": 1000, "scale_word": "thousands", "currency": "USD", "symbol": "$"}
    assert read_unit("Sales ($ in thousands)")["currency"] == "$"
    assert read_unit("Revenue (€ millions)") == {
        "scale": 1_000_000, "scale_word": "millions", "currency": "EUR", "symbol": "€"}
    assert read_unit("Quarterly summary") is None
    assert read_unit("Units sold (m)") is None          # a bare "m" is not read as millions


def test_a_header_that_states_its_own_unit_wins(tmp_path):
    path = _titled(tmp_path, "Costs (USD thousands)", {"Item": ["a"], "Cost (£000)": [5.0],
                                                       "Spend": [7.0]})
    columns = load_dataset(str(path)).structure["units"]["columns"]
    assert columns["Cost (£000)"]["currency"] == "GBP" and columns["Cost (£000)"]["source"]["where"] == "header"
    assert columns["Spend"]["currency"] == "USD"


def test_a_rephrasing_may_not_add_a_unit():
    from avaloka.llm.guard import clean_rephrasing

    original = "Revenue: total=21,861 (from 24 non-null rows)."
    assert clean_rephrasing(original, "Revenue totals 21,861 USD across 24 rows.") == original
    assert clean_rephrasing(original, "Revenue totals 21,861 across 24 rows.") == \
        "Revenue totals 21,861 across 24 rows."


def test_narration_may_not_add_a_unit():
    from avaloka.llm.client import narrate

    class Model:
        available, last_cost_usd = True, 0.0

        def __init__(self, reply):
            self.reply = reply

        def complete(self, system, prompt):
            return self.reply

    base = "Revenue totals 21,861."
    assert narrate(Model("Revenue totals £21,861 thousand."), "s", base, base)[0] == base
    assert narrate(Model("Total revenue was 21,861."), "s", base, base)[0] == "Total revenue was 21,861."
