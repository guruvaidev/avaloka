"""Text Avaloka's own code writes must be true: rc16's false system notes.

Under the 2.0 "c-strict" display, a sentence the model wrote is marked as
commentary, and what the code writes -- a scope line, a "Why:" reason, the
layout of a reply with a sentence held back -- is shown plain. The rc16 web
regrade (rweb16) found such text false:

* a scope line "only where every column has a value" for ``df['Country']
  .dropna()``, a Series dropna that dropped nothing (messy run 1; recipes run 2,
  0 of 12,190);
* no scope line for ``return df_src[df_src['Quantity'] > 0]`` inside a helper
  (messy run 3);
* "how many were left out could not be counted" for a null drop the replay can
  count (nullxlsx run 3: 4 of 120);
* "no count, total or mean equal to 10,132.48" for 85123A, whose total
  Quantity x UnitPrice over its sales IS 10,132.48, while the same claim as a
  bullet ("Highest revenue: 85123A - $10,132.48", false: 85123A ranks 4th) was
  shown (messy runs 2 and 3);
* a reason about a customer_id join for a withheld sentence about the signup
  year (related run 3), whose paragraph break went with it.

Every test drives a stored rweb16 turn (tests/fixtures/rc16_system_notes,
minimum fields of states/rweb16/<run>/<cell>/<turn>.json: the augmented human
message, the code that ran, its reads and result rows) through
claim_verifier_node and presentable_reply to the text the user sees. No LLM
runs.

Data: the null-words workbook is tests/fixtures/null_words/three_sheets.xlsx.
The messy sheets are the workbook's own rows of every item the result, the
claims or the truth name (messy_items.csv), and filler rows that keep each
sheet's 20,000 rows and its United Kingdom count (19,012 and 18,869), which
the answers state. The recipes and the related tables are samples of the
originals (120 recipes; 100 customers of every signup year and their orders).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.claim_verifier import (
    DataUsed, check_group_figure, claim_verifier_node, data_used, presentable_reply,
)
from app.agents.filter_disclosure import detect_row_filters

FIXTURES = Path(__file__).parent / "fixtures"
HERE = FIXTURES / "rc16_system_notes"
TURNS = json.loads((HERE / "turns.json").read_text(encoding="utf-8"))["turns"]

#: Each sheet's rows and United Kingdom rows in the workbook.
MESSY_SHEETS = {"2010 Q4": (20_000, 19_012), "2011 Q1": (20_000, 18_869)}


def _messy_sheet(tmp_path: Path, sheet: str) -> Path:
    path = tmp_path / f"messy_{sheet.replace(' ', '_')}.csv"
    if path.exists():
        return path
    items = pd.read_csv(HERE / "messy_items.csv", dtype={"StockCode": str, "InvoiceNo": str})
    real = items[items.pop("sheet") == sheet]
    rows, uk = MESSY_SHEETS[sheet]
    n = rows - len(real)
    uk_fill = uk - int((real["Country"] == "United Kingdom").sum())
    filler = pd.DataFrame({
        "InvoiceNo": "F0", "StockCode": [f"FILL{i % 50}" for i in range(n)],
        "Description": [f"FILLER ITEM {i % 50}" for i in range(n)], "Quantity": 1,
        "InvoiceDate": "2010-12-01 08:26:00", "UnitPrice": 0.01, "CustomerID": 12345.0,
        "Country": ["United Kingdom"] * uk_fill + ["France"] * (n - uk_fill)})
    pd.concat([real, filler], ignore_index=True).to_csv(path, index=False)
    return path


def _null_words_sheet(tmp_path: Path, sheet: str) -> Path:
    path = tmp_path / f"null_words_{sheet}.csv"
    pd.read_excel(FIXTURES / "null_words" / "three_sheets.xlsx", sheet_name=sheet, dtype=str,
                  keep_default_na=False).to_csv(path, index=False)
    return path


def _local(tmp_path: Path, alias: str, sheet) -> Path:
    if alias.startswith("ecommerce_messy"):
        return _messy_sheet(tmp_path, sheet)
    if alias.startswith("null_words"):
        return _null_words_sheet(tmp_path, sheet)
    if alias == "recipes":
        return HERE / "recipes_sample.jsonl"
    return HERE / f"{alias}.csv"


def _live_state(turn: dict, tmp_path: Path) -> dict:
    """The state at verify time as the server and graph build it: the
    augmented human message, the code that ran with the reads the executor
    recorded, or (a chat about the tables) the turn's datasets."""
    state = {"messages": [HumanMessage(content=turn["graph_human_message"]), AIMessage(content=turn["narrative"])]}
    if "code" in turn:
        code, reads = turn["code"], []
        for read in turn["reads"]:
            local = str(_local(tmp_path, read["alias"], read.get("sheet_name")))
            code = code.replace(read["path"], local)
            reads.append(dict(read, path=local))
        state.update(coder_definition={"code": code}, output_json=turn["output_json"],
                     execution_result={"status": "success", "fresh_output": True,
                                       "output_row_count": len(turn["output_json"]), "reads": reads})
    else:
        state["datasets_context"] = [dict(meta, data_source_location=str(_local(tmp_path, meta["alias"], None)))
                                     for meta in turn["datasets_context"]]
    return state


def _shown(turn: str, tmp_path: Path):
    state = _live_state(TURNS[turn], tmp_path)
    report = claim_verifier_node(state)["verification_report"]
    return presentable_reply(state["messages"][-1].content, report), report, state


def test_the_fixture_is_the_live_shape():
    for name, turn in TURNS.items():
        assert turn["graph_human_message"].startswith(turn["question"]), name
        assert "[Analysis context]" in turn["graph_human_message"], name
    # What rc16 showed, the defects these tests are about.
    assert "only where every column has a value" in TURNS["messy_ask_run1"]["shown_on_rc16"]
    assert "only where every column has a value" in TURNS["recipes_chat_run2"]["shown_on_rc16"]
    assert "Scope:" not in TURNS["messy_chat_run3"]["shown_on_rc16"]
    assert "could not be counted" in TURNS["nullxlsx_reply_run3"]["shown_on_rc16"]
    assert "- Highest revenue: 85123A – $10,132.48" in TURNS["messy_chat_run2"]["shown_on_rc16"]
    assert "no count, total or mean equal to 10,132.48" in TURNS["messy_chat_run2"]["withheld_on_rc16"][0]["detail"]
    assert "signup" in TURNS["related_tables_run3"]["withheld_on_rc16"][0]["claim"]
    assert "customer_id" in TURNS["related_tables_run3"]["withheld_on_rc16"][0]["detail"]


def test_the_messy_sheets_keep_the_figures_the_answers_state(tmp_path):
    frames = [pd.read_csv(_messy_sheet(tmp_path, s), dtype={"StockCode": str}) for s in MESSY_SHEETS]
    both = pd.concat(frames, ignore_index=True)
    assert [len(f) for f in frames] == [20_000, 20_000]
    assert int((both["Country"] == "United Kingdom").sum()) == 37_881
    sales = both[both["Quantity"] > 0].assign(s=lambda d: d["Quantity"] * d["UnitPrice"]).groupby("StockCode")["s"].sum()
    assert round(sales["85123A"], 2) == 10_132.48
    assert sales.sort_values(ascending=False).index[:4].tolist() == ["22423", "DOT", "AMAZONFEE", "85123A"]


# --------------------------------------------------------------------------- #
# 1a: a Series dropna is no row filter; a filter that drops nothing gets no line
# --------------------------------------------------------------------------- #

def test_messy_run1_a_series_dropna_gets_no_scope_line(tmp_path):
    """``countries = df_all[required_col].dropna()`` drops blanks of Country
    only (none here): the rows counted are every row of both sheets."""
    text, report, _state = _shown("messy_ask_run1", tmp_path)
    assert "The United Kingdom has the most rows, with 37,881 entries." in text
    assert "Scope:" not in text
    assert "every column has a value" not in text
    assert not report.get("scoped")


def test_recipes_run2_a_series_dropna_gets_no_scope_line(tmp_path):
    """``ingredients_exploded.dropna()`` and ``df['Month'].dropna()``: Series,
    each counted on its own values; nothing was filtered from the frame."""
    text, report, _state = _shown("recipes_chat_run2", tmp_path)
    assert "Scope:" not in text
    assert "every column has a value" not in text
    assert not report.get("scoped")


def test_messy_run3_ask_a_null_drop_that_drops_nothing_gets_no_scope_line(tmp_path):
    """``combined.dropna(subset=['Country'])`` on two concatenated sheets, after
    ``astype(str)``: it drops no row, and is now counted as dropping none."""
    text, report, state = _shown("messy_ask_run3", tmp_path)
    assert "Scope:" not in text
    filters = detect_row_filters(state["coder_definition"]["code"], data_used(state))
    assert [(f.kind, f.rows_before, f.rows_dropped) for f in filters] == [("dropna", 40_000, 0)]


# --------------------------------------------------------------------------- #
# 1b: a filter in a helper function
# --------------------------------------------------------------------------- #

def test_messy_run3_chat_a_filter_inside_a_helper_gets_its_scope_line(tmp_path):
    """``def clean(df_src): return df_src[df_src['Quantity'] > 0].dropna(...)``,
    called on each sheet: the frames it returns are the filtered ones."""
    text, report, _state = _shown("messy_chat_run3", tmp_path)
    items = pd.read_csv(HERE / "messy_items.csv")
    dropped = int((~(items["Quantity"] > 0)).sum())
    line = (f"Scope: excludes {dropped:,} rows (Quantity <= 0), a filter the question did not ask for; "
            f"the figures above are computed on the remaining {40_000 - dropped:,} of 40,000 rows.")
    assert line in text
    assert report["scoped"] is True


def test_a_real_row_filter_in_main_keeps_its_scope_line_now_counted(tmp_path):
    """Guard: ``df = df[df['Quantity'] > 0]`` after the concat (messy run 2)
    is still disclosed, with its count."""
    text, _report, _state = _shown("messy_chat_run2", tmp_path)
    items = pd.read_csv(HERE / "messy_items.csv")
    dropped = int((~(items["Quantity"] > 0)).sum())
    assert (f"Scope: excludes {dropped:,} rows (Quantity <= 0), a filter the question did not ask for; "
            f"the figures above are computed on the remaining {40_000 - dropped:,} of 40,000 rows.") in text
    assert "could not be counted" not in text


# --------------------------------------------------------------------------- #
# 1c: a null drop the replay can count is counted
# --------------------------------------------------------------------------- #

def test_nullxlsx_run3_the_null_drop_is_counted_not_called_uncountable(tmp_path):
    """``df[df['Outcome'].notna() & df['Glucose_num'].notna()]`` on two
    concatenated sheets drops 4 of 120 rows (Glucose_num blank). Counted, it
    is under the 5% null-handling threshold, so no line; never "could not be
    counted"."""
    text, _report, state = _shown("nullxlsx_reply_run3", tmp_path)
    assert "could not be counted" not in text
    filters = detect_row_filters(state["coder_definition"]["code"], data_used(state))
    assert [(f.null_only, f.rows_before, f.rows_dropped) for f in filters] == [(True, 120, 4)]
    assert "36 readmitted vs 80 recovered" in text          # 116 = 120 - 4


# --------------------------------------------------------------------------- #
# 2a: 85123A's total, and the same false claim shown as a bullet
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("turn", ["messy_chat_run2", "messy_chat_run3"])
def test_messy_85123a_is_withheld_everywhere_with_a_true_reason(tmp_path, turn):
    text, report, _state = _shown(turn, tmp_path)
    reply = text.split("I've held back")[0]
    assert "10,132.48" not in reply                      # the sentence and its bullet
    assert "Highest revenue: 85123A" not in reply
    # Four: with #603 the "Lowest revenue: 84077 – $1,160.21" bullet is withheld
    # too (the lowest of a top-10-by-quantity table; over the kept rows it is 0).
    assert "Lowest revenue: 84077" not in reply
    assert "I've held back 4 statements" in text
    reasons = {f["claim"]: f["detail"] for f in report["findings"]}
    headline = next(c for c in reasons if c.startswith("The item with the highest sales"))
    bullet = "- Highest revenue: 85123A – $10,132.48"
    for claim in (headline, bullet):
        # The reason is the claim's own and true: 10,132.48 is 85123A's
        # total, and it ranks 4th over the rows the analysis kept.
        assert "no count, total or mean equal" not in reasons[claim]
        assert re.search(r"over all rows with Quantity > 0 the highest TotalSales \(sum of Quantity × UnitPrice "
                         r"per (StockCode|Description)\) is (22423|REGENCY CAKESTAND 3 TIER) \(25,588.57\); "
                         r"(85123A|WHITE HANGING HEART T-LIGHT HOLDER) ranks 4 \(10,132.48\)", reasons[claim])
    assert "Why: 85123A is the highest only among the 10 StockCode values the analysis kept" in text
    # The true bullet stays.
    assert "- Largest quantity sold: 84077 – 5,147 units" in reply


def test_85123a_total_over_its_sales_is_a_total_of_the_group(tmp_path):
    """Guard on the candidate itself: 10,132.48 is 85123A's Quantity x
    UnitPrice over its rows with a positive Quantity."""
    used = DataUsed(frames=[(s, pd.read_csv(_messy_sheet(tmp_path, s), dtype={"StockCode": str}))
                            for s in MESSY_SHEETS])
    assert check_group_figure("85123A had total sales of $10,132.48.", {}, used) is None
    assert check_group_figure("85123A had total sales of $10,999.00.", {}, used) is not None


def test_a_withheld_sentence_repeated_verbatim_is_withheld_too():
    """The report names one sentence; its repeat (here in bold, as a bullet)
    is the same sentence and is never shown."""
    claim = "The item with the highest sales is 85123A, generating $10,132.48 in revenue."
    reply = (f"{claim}\n\n- **The item with the highest sales is 85123A, generating $10,132.48 in revenue.**\n"
             "- Largest quantity sold: 84077 – 5,147 units")
    report = {"findings": [{"claim": claim, "verdict": "contradicted", "check": "superlative_of_subset",
                            "detail": "22423 is the highest"}], "claims_reconciled": False}
    text = presentable_reply(reply, report)
    assert "10,132.48" not in text
    assert "- Largest quantity sold: 84077 – 5,147 units" in text
    assert "I've held back 2 statements" in text


# --------------------------------------------------------------------------- #
# 2b / 2c: the reason is the withheld sentence's own; the reply keeps its layout
# --------------------------------------------------------------------------- #

def test_related_run3_the_reason_is_about_the_year_match_it_withholds(tmp_path):
    text, report, _state = _shown("related_tables_run3", tmp_path)
    (finding,) = report["findings"]
    assert "signup" in finding["claim"]
    assert "customer_id" not in finding["detail"]
    assert finding["detail"].startswith("it says matching the year of signup_year to the year of order_date "
                                        "finds no match, but 1 year (2024) appears in both customers and orders")
    why = [line for line in text.splitlines() if line.startswith("Why:")]
    assert why and all("customer_id" not in line for line in why)


def test_related_run3_withholding_the_last_sentence_keeps_the_paragraph_break(tmp_path):
    text, _report, _state = _shown("related_tables_run3", tmp_path)
    assert "(high confidence, direct key‑to‑key match).\n\n**Next actions** (pick one):" in text


def test_related_run1_a_correctly_attributed_reason_is_unchanged(tmp_path):
    """Guard: the withheld sentence's own key is customer_id (written as code);
    its reason, its neighbours and the layout stay as they were. The primary
    relation was inside the withheld sentence itself ("The natural join key
    is customers.customer_id ↔ orders.customer_id (high confidence, but ...
    the match rate is 0 %)"): no neighbour was dropped."""
    text, report, _state = _shown("related_tables_run1", tmp_path)
    (finding,) = report["findings"]
    assert finding["detail"] == ("it says a join on customer_id finds no match, but 100 customer_id values appear "
                                 "in both customers (100 distinct) and orders (100 distinct), so the join returns rows")
    assert "**How they relate**\n\nA secondary possible link is" in text
    assert "(moderate confidence).\n\n**Next steps you can take**" in text


def test_related_seg_run1_returns_is_not_the_returned_column(tmp_path):
    """Guard: "The join on `customer_id` ... returns no matching rows" names
    no column once its code is removed; "returns" does not name ``returned``."""
    _text, report, _state = _shown("related_seg_run1", tmp_path)
    assert [f["detail"].split(",")[0] for f in report["findings"]] == ["it says a join on customer_id finds no match"]


def test_related_run2_all_true_is_shown_whole(tmp_path):
    text, report, _state = _shown("related_tables_run2", tmp_path)
    assert text == TURNS["related_tables_run2"]["narrative"]
    assert not report.get("findings")
