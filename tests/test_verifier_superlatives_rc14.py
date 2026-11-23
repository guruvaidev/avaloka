"""Superlatives, trends, bounds and paired lists, as rc14's replies stated them.

messy_xlsx ("what sells most?"): the code kept the top 10 items by QUANTITY
and the reply called the best of those by revenue the best overall. Every
figure was in the result, so every claim was shown. Over all rows REGENCY
CAKESTAND 3 TIER sold 24,615.97 and DOTCOM POSTAGE 20,910.16; WHITE HANGING
HEART T-LIGHT HOLDER (9,570.72) ranks 3rd and PAPER CHAIN KIT 50'S CHRISTMAS
(8,793.20) 4th.

nullxlsx ("What drives Outcome?"): a table of glucose bins narrated as a
trend it does not show, a bin called the largest recovery share when another
bin's is larger, and a true "lowest readmission share (≈ 15 %)" withheld as
an invented number.

esc_groq ("train a random forest to predict stroke and show feature
importance"): three true sentences withheld (a bound, scientific notation,
and a paired list given a reason that contradicts its pairing), and one
false one shown (a range said of every member of two groups).

Replies, code and result rows are the ones those runs produced
(results/rw7/messy_xlsx/chat.json, results/rw7/nullxlsx/reply.json,
results/esc_groq/up1/q0.json); no LLM runs. The messy frames reproduce the
workbook's per-item quantity and revenue for every item the claims or the
truth name.
"""

from __future__ import annotations

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.claim_verifier as cv
from app.agents.claim_verifier import DataUsed, claim_verifier_node, presentable_reply, verify_claims, withheld_claims


def _found(report, claim):
    return next((f for f in report.findings if f.claim == claim), None)


# --------------------------------------------------------------------------- #
# B: messy_xlsx -- a superlative over a subset said of the whole
# --------------------------------------------------------------------------- #

MESSY_QUESTION = "what sells most?"
MESSY_HEADLINE = "The item with the highest sales is **WHITE HANGING HEART T‑LIGHT HOLDER** with $9,570.72."
MESSY_HIGHEST = "- Highest revenue: $9,570.72 (WHITE HANGING HEART T‑LIGHT HOLDER)."
MESSY_SECOND = "- Second highest: $8,793.20 (PAPER CHAIN KIT 50'S CHRISTMAS)."
MESSY_QUANTITY = "- Largest quantity sold: 5,147 units (WORLD WAR 2 GLIDERS ASSTD DESIGNS)."
MESSY_REPLY = (f"{MESSY_HEADLINE}\n\n{MESSY_HIGHEST}  \n{MESSY_SECOND}  \n{MESSY_QUANTITY}  \n"
               "- The top‑selling item also has a moderate quantity (3,238), showing high unit price.")
MESSY_CODE = """import pandas as pd
import numpy as np

def main(df):
    path_2010 = '/var/tmp/avaloka/29a48a97-56d7-4995-a553-e61c53a0c440/input_523cabbf-ab99-488c-9b1c-a590b5576469.csv'
    path_2011 = '/var/tmp/avaloka/a99c3c03-0bf3-43e6-ac4c-960dbc9a4c55/input_36765f15-26a7-43b0-b46c-f9330d9aefcc.csv'
    df_2010 = pd.read_csv(path_2010, parse_dates=['InvoiceDate'])
    df_2011 = pd.read_csv(path_2011, parse_dates=['InvoiceDate'])
    cols = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
    df_2010 = df_2010[cols]
    df_2011 = df_2011[cols]
    for d in (df_2010, df_2011):
        d['Quantity'] = pd.to_numeric(d['Quantity'], errors='coerce').fillna(0).round().astype('Int64')
        d['UnitPrice'] = pd.to_numeric(d['UnitPrice'], errors='coerce').fillna(0.0).astype('float')
    df_all = pd.concat([df_2010, df_2011], ignore_index=True)
    df_all['SalesValue'] = df_all['Quantity'].astype(float) * df_all['UnitPrice']
    df_all = df_all.dropna(subset=['Description'])
    agg_df = df_all.groupby('Description', as_index=False).agg(TotalQuantity=('Quantity', 'sum'), TotalSales=('SalesValue', 'sum'))
    agg_df = agg_df.sort_values(by='TotalQuantity', ascending=False)
    top_n = 10
    top_products = agg_df.head(top_n).reset_index(drop=True)
    return top_products
"""
#: output_json as the API returned it: text cells, read back from the CSV.
MESSY_ROWS = [
    {"Description": "WORLD WAR 2 GLIDERS ASSTD DESIGNS", "TotalQuantity": "5147", "TotalSales": "1160.21"},
    {"Description": "PACK OF 72 RETROSPOT CAKE CASES", "TotalQuantity": "3707", "TotalSales": "2479.11"},
    {"Description": "WHITE HANGING HEART T-LIGHT HOLDER", "TotalQuantity": "3238", "TotalSales": "9570.72"},
    {"Description": "HAND WARMER BABUSHKA DESIGN", "TotalQuantity": "3232", "TotalSales": "4246.62"},
    {"Description": "SMALL POPCORN HOLDER", "TotalQuantity": "2648", "TotalSales": "2508.38"},
    {"Description": "MINI PAINT SET VINTAGE ", "TotalQuantity": "2602", "TotalSales": "1546.6000000000001"},
    {"Description": "GROW A FLYTRAP OR SUNFLOWER IN TIN", "TotalQuantity": "2593", "TotalSales": "2479.01"},
    {"Description": "PAPER CHAIN KIT 50'S CHRISTMAS ", "TotalQuantity": "2386", "TotalSales": "8793.2"},
    {"Description": "ASSORTED COLOUR BIRD ORNAMENT", "TotalQuantity": "2339", "TotalSales": "3931.79"},
    {"Description": "CREAM HEART CARD HOLDER", "TotalQuantity": "2279", "TotalSales": "5721.21"},
]
#: Items the top 10 by quantity left out: (quantity, revenue) over both sheets.
MESSY_OTHERS = {"REGENCY CAKESTAND 3 TIER": (1912, 24615.97), "DOTCOM POSTAGE": (38, 20910.16),
                "VINTAGE UNION JACK MEMOBOARD": (1500, 6650.42), "JUMBO BAG RED RETROSPOT": (2100, 4120.50),
                "RABBIT NIGHT LIGHT": (900, 1830.00), "LUNCH BAG RED RETROSPOT": (1100, 1790.25)}

HEADLINE_REASON = ("WHITE HANGING HEART T-LIGHT HOLDER is the highest only among the 10 Description values the "
                   "analysis kept; over all rows the highest TotalSales (sum of Quantity × UnitPrice per "
                   "Description) is REGENCY CAKESTAND 3 TIER (24,615.97); WHITE HANGING HEART T-LIGHT HOLDER "
                   "ranks 3 (9,570.72)")
SECOND_REASON = ("PAPER CHAIN KIT 50'S CHRISTMAS is the second highest only among the 10 Description values the "
                 "analysis kept; over all rows the second highest TotalSales (sum of Quantity × UnitPrice per "
                 "Description) is DOTCOM POSTAGE (20,910.16); PAPER CHAIN KIT 50'S CHRISTMAS ranks 4 (8,793.20)")


def messy_frames() -> list:
    """The two sheets, as invoice lines: each item's quantity and revenue
    split over a line in each, so only the two sheets together give the
    totals the result shows."""
    items = {r["Description"]: (int(r["TotalQuantity"]), float(r["TotalSales"])) for r in MESSY_ROWS}
    items.update(MESSY_OTHERS)
    q4, q1 = [], []
    for n, (desc, (qty, sales)) in enumerate(items.items()):
        price = round(sales / qty, 2)
        rest = round(sales - (qty - 1) * price, 2)
        line = {"InvoiceNo": str(536365 + n), "StockCode": f"S{n:04d}", "Description": desc,
                "InvoiceDate": "2010-12-01 08:26:00", "CustomerID": 17850.0 + n, "Country": "United Kingdom"}
        q4.append({**line, "Quantity": qty - 1, "UnitPrice": price})
        q1.append({**line, "Quantity": 1, "UnitPrice": rest})
    return [("2010 Q4", pd.DataFrame(q4)), ("2011 Q1", pd.DataFrame(q1))]


@pytest.fixture(scope="module")
def messy_used() -> DataUsed:
    return DataUsed(frames=messy_frames())


def test_the_messy_frames_reproduce_the_result_and_the_truth(messy_used):
    both = pd.concat([f for _l, f in messy_used.frames])
    revenue = (both["Quantity"] * both["UnitPrice"]).groupby(both["Description"].str.strip()).sum()
    assert revenue.idxmax() == "REGENCY CAKESTAND 3 TIER"
    assert round(revenue["WHITE HANGING HEART T-LIGHT HOLDER"], 2) == 9570.72
    assert list(revenue.sort_values(ascending=False).index[:4]) == [
        "REGENCY CAKESTAND 3 TIER", "DOTCOM POSTAGE", "WHITE HANGING HEART T-LIGHT HOLDER",
        "PAPER CHAIN KIT 50'S CHRISTMAS"]
    quantity = both.groupby(both["Description"].str.strip())["Quantity"].sum()
    assert quantity.idxmax() == "WORLD WAR 2 GLIDERS ASSTD DESIGNS"


def test_messy_subset_superlatives_are_withheld_saying_what_is_highest_over_all_rows(messy_used):
    report = verify_claims(MESSY_REPLY, {"execution_output": MESSY_ROWS}, messy_used, question=MESSY_QUESTION,
                           code=MESSY_CODE)
    found = {f.claim: (f.check, f.detail) for f in report.findings}
    assert found == {MESSY_HEADLINE: ("superlative_of_subset", HEADLINE_REASON),
                     MESSY_HIGHEST: ("superlative_of_subset", HEADLINE_REASON),
                     MESSY_SECOND: ("superlative_of_subset", SECOND_REASON)}
    # "Largest quantity sold: 5,147 units" is true over all rows, and stays.
    assert _found(report, MESSY_QUANTITY) is None
    shown = presentable_reply(MESSY_REPLY, report.as_dict())
    for line in (MESSY_HEADLINE, MESSY_HIGHEST, MESSY_SECOND):
        assert line not in shown
    assert MESSY_QUANTITY in shown
    assert f"Why: {HEADLINE_REASON}." in shown
    assert f"Why: {SECOND_REASON}." in shown


def test_messy_is_withheld_on_the_node_path(tmp_path, messy_used):
    """As the graph runs it: the sheets the run read come from execution_result.reads."""
    reads = []
    for label, frame in messy_used.frames:
        path = tmp_path / f"{label.replace(' ', '_')}.csv"
        frame.to_csv(path, index=False)
        reads.append({"path": str(path), "used": True})
    state = {"messages": [HumanMessage(content=MESSY_QUESTION), AIMessage(content=MESSY_REPLY)],
             "coder_definition": {"code": MESSY_CODE},
             "execution_result": {"status": "success", "reads": reads},
             "output_json": MESSY_ROWS}
    report = claim_verifier_node(state)["verification_report"]
    assert report["verified"] is False
    assert withheld_claims(report) == {MESSY_HEADLINE, MESSY_HIGHEST, MESSY_SECOND}


def test_a_true_superlative_over_all_rows_is_shown(messy_used):
    """The same question answered by revenue: the top 10 by revenue, and its top item."""
    both = pd.concat([f for _l, f in messy_used.frames])
    revenue = (both["Quantity"] * both["UnitPrice"]).groupby(both["Description"]).sum().sort_values(ascending=False)
    rows = [{"Description": d, "TotalSales": str(round(v, 2))} for d, v in revenue.head(10).items()]
    code = MESSY_CODE.replace("by='TotalQuantity'", "by='TotalSales'")
    reply = ("The item with the highest sales is **REGENCY CAKESTAND 3 TIER** with $24,615.97.\n"
             "- Second highest: $20,910.16 (DOTCOM POSTAGE).")
    report = verify_claims(reply, {"execution_output": rows}, messy_used, question=MESSY_QUESTION, code=code)
    assert report.findings == []


def test_a_superlative_said_of_the_selection_is_shown(messy_used):
    reply = ("Among the 10 items sold in the largest quantities, the highest sales is WHITE HANGING HEART "
             "T‑LIGHT HOLDER ($9,570.72).")
    evidence = {"execution_output": MESSY_ROWS, "result_shape": {"rows": 10, "columns": 3}}    # as the node gives it
    report = verify_claims(reply, evidence, messy_used, question=MESSY_QUESTION, code=MESSY_CODE)
    assert report.findings == []


def test_a_measure_the_data_cannot_reproduce_checks_nothing_beyond_the_result(messy_used):
    """A filtered or reshaped measure is not recomputed: the claim is left to the result's own checks."""
    rows = [dict(r, TotalSales=str(round(float(r["TotalSales"]) * 1.37, 2))) for r in MESSY_ROWS]
    reply = "The item with the highest sales is WHITE HANGING HEART T‑LIGHT HOLDER with $13,111.89."
    report = verify_claims(reply, {"execution_output": rows}, messy_used, question=MESSY_QUESTION, code=MESSY_CODE)
    assert report.findings == []


# --------------------------------------------------------------------------- #
# E: nullxlsx -- trends and shares over glucose bins
# --------------------------------------------------------------------------- #

NULLX_QUESTION = "What drives Outcome?"
NULLX_TREND = "Higher glucose levels increase the likelihood of readmission."
NULLX_HIGHEST = "- The highest readmission share (≈ 47 %) occurs in the (107.5, 130.0] bin."
NULLX_LARGEST = "- The (130.0, 174.0] bin shows the largest recovery share (≈ 71 %)."
NULLX_NAN = "- The “nan” bin has 100 % readmission, but only 4 cases."
NULLX_LOWEST = "- The lowest readmission share (≈ 15 %) is in the (88.0, 107.5] bin."
NULLX_REPLY = f"{NULLX_TREND}  \n{NULLX_HIGHEST}  \n{NULLX_LARGEST}  \n{NULLX_NAN}\n{NULLX_LOWEST}"
NULLX_ROWS = [
    {"Glucose_bin": "(107.5, 130.0]", "Outcome_std": "Readmitted", "count": "14", "outcome_share": "0.4666666666666667"},
    {"Glucose_bin": "(107.5, 130.0]", "Outcome_std": "Recovered", "count": "16", "outcome_share": "0.5333333333333333"},
    {"Glucose_bin": "(130.0, 174.0]", "Outcome_std": "Readmitted", "count": "8", "outcome_share": "0.2857142857142857"},
    {"Glucose_bin": "(130.0, 174.0]", "Outcome_std": "Recovered", "count": "20", "outcome_share": "0.7142857142857143"},
    {"Glucose_bin": "(70.999, 88.0]", "Outcome_std": "Readmitted", "count": "10", "outcome_share": "0.3125"},
    {"Glucose_bin": "(70.999, 88.0]", "Outcome_std": "Recovered", "count": "22", "outcome_share": "0.6875"},
    {"Glucose_bin": "(88.0, 107.5]", "Outcome_std": "Readmitted", "count": "4", "outcome_share": "0.15384615384615385"},
    {"Glucose_bin": "(88.0, 107.5]", "Outcome_std": "Recovered", "count": "22", "outcome_share": "0.8461538461538461"},
    {"Glucose_bin": "nan", "Outcome_std": "Readmitted", "count": "4", "outcome_share": "1.0"},
]
TREND_REASON = ("it says outcome_share of Readmitted rises with Glucose_bin, but across the Glucose_bin groups it "
                "is not monotonic: (70.999, 88.0] 31.2%, (88.0, 107.5] 15.4%, (107.5, 130.0] 46.7%, "
                "(130.0, 174.0] 28.6%; and it asserts causation ('increase the likelihood') from evidence that is "
                "correlational; no experiment, treatment/control or causal estimate is present")
LARGEST_REASON = ("(130.0, 174.0]'s outcome_share (71.4%) is not the largest among rows with Outcome_std = "
                  "Recovered: (88.0, 107.5] has 84.6%")


def test_nullxlsx_trend_and_largest_share_are_withheld_and_the_true_lines_stay():
    report = verify_claims(NULLX_REPLY, {"execution_output": NULLX_ROWS}, None, question=NULLX_QUESTION)
    found = {f.claim: (f.check, f.detail) for f in report.findings}
    assert found == {NULLX_TREND: ("trend_contradicted", TREND_REASON),
                     NULLX_LARGEST: ("superlative_contradicted", LARGEST_REASON)}
    shown = presentable_reply(NULLX_REPLY, report.as_dict())
    for line in (NULLX_HIGHEST, NULLX_NAN, NULLX_LOWEST):
        assert line in shown
    assert f"Why: {TREND_REASON[0].upper()}{TREND_REASON[1:]}." in shown
    assert f"Why: {LARGEST_REASON}." in shown


def test_lowest_readmission_share_rounded_to_a_whole_percent_is_not_invented():
    """≈ 15 % is 0.1538 rounded to the percent it is written in; 1% of 15 is 0.15 and missed it."""
    report = verify_claims(NULLX_LOWEST, {"execution_output": NULLX_ROWS}, None, question=NULLX_QUESTION)
    assert report.findings == []
    wrong = verify_claims(NULLX_LOWEST.replace("15", "37"), {"execution_output": NULLX_ROWS}, None)
    assert [f.check for f in wrong.findings] == ["invented_number"]


MONOTONIC_ROWS = [
    {"Glucose_bin": b, "Outcome_std": o, "outcome_share": str(s)}
    for b, r in (("(70.999, 88.0]", 0.15), ("(88.0, 107.5]", 0.25), ("(107.5, 130.0]", 0.30), ("(130.0, 174.0]", 0.45))
    for o, s in (("Readmitted", r), ("Recovered", round(1 - r, 2)))
]


def test_a_monotonic_increase_is_shown_and_a_causal_verb_on_it_is_still_overreach():
    true = "Readmission share increases with glucose: from 15% in the lowest bin to 45% in the highest."
    assert verify_claims(true, {"execution_output": MONOTONIC_ROWS}).findings == []
    falls = verify_claims("Recovery share rises with glucose across the bins.", {"execution_output": MONOTONIC_ROWS})
    assert [f.check for f in falls.findings] == ["trend_contradicted"]
    causal = verify_claims(NULLX_TREND, {"execution_output": MONOTONIC_ROWS})
    assert [(f.check, f.verdict.value) for f in causal.findings] == [("causal_overreach", "overreach")]


# --------------------------------------------------------------------------- #
# #582 follow-ups: esc_groq q0
# --------------------------------------------------------------------------- #

ESC_QUESTION = "train a random forest to predict stroke and show feature importance"
ESC_HEADLINE = ("The most influential predictor for stroke is **avg_glucose_level** (0.263), followed closely by "
                "**age** (0.242) and **bmi** (0.218).")
ESC_TOP3 = ("- **Top three**: avg_glucose_level, age, bmi – together they account for over 70 % of the "
            "model’s importance.")
ESC_GAP = ("- **Surprising gap**: hypertension and heart_disease have similar importance (~0.028 and 0.025) but "
           "are far below the top three, indicating lifestyle factors dominate.")
ESC_PATTERN = ("- **Consistent pattern**: all smoking_status categories and work_type categories contribute "
               "modestly (≈0.02–0.017), showing that medical history and occupation are less predictive than "
               "metabolic and demographic factors.")
ESC_LOWEST = ("- **Lowest**: work_type_Never_worked (5.1 × 10⁻⁶) and gender_Other "
              "(8.7 × 10⁻⁹) are essentially irrelevant.")
ESC_REPLY = "\n\n".join((ESC_HEADLINE, ESC_TOP3, ESC_GAP, ESC_PATTERN, ESC_LOWEST))
ESC_ROWS = [{"feature": f, "importance": v} for f, v in (
    ("avg_glucose_level", "0.26325507507001217"), ("age", "0.24214964805712694"), ("bmi", "0.21847743216672794"),
    ("hypertension", "0.02784047623054107"), ("heart_disease", "0.02502175327484755"),
    ("Residence_type_Rural", "0.02017940441193851"), ("smoking_status_never smoked", "0.020170903945041556"),
    ("gender_Male", "0.020008204799897965"), ("work_type_Self-employed", "0.01960674802594639"),
    ("Residence_type_Urban", "0.01908396832102431"), ("work_type_Private", "0.019010734586165885"),
    ("gender_Female", "0.018351172231460267"), ("smoking_status_formerly smoked", "0.0174468372594403"),
    ("smoking_status_smokes", "0.01726861292206202"), ("smoking_status_Unknown", "0.016913800035262466"),
    ("work_type_Govt_job", "0.013261307670985094"), ("ever_married_Yes", "0.01082355758163406"),
    ("ever_married_No", "0.010335465347529913"), ("work_type_children", "0.0007897832861215145"),
    ("work_type_Never_worked", "5.106106557818471e-06"), ("gender_Other", "8.669676127518158e-09"))]
PATTERN_REASON = ("it puts every smoking_status and work_type category at 0.02–0.017, but work_type_Govt_job "
                  "is 0.01326, work_type_children is 0.0007898, work_type_Never_worked is 5.106e-06 (importance)")


def _esc(reply=ESC_REPLY, rows=ESC_ROWS):
    return verify_claims(reply, {"execution_output": rows}, None, question=ESC_QUESTION)


def test_esc_true_lines_are_shown_and_the_range_over_two_groups_is_withheld():
    report = _esc()
    assert {f.claim: (f.check, f.detail) for f in report.findings} == {
        ESC_PATTERN: ("range_contradicted", PATTERN_REASON)}
    shown = presentable_reply(ESC_REPLY, report.as_dict())
    for line in (ESC_HEADLINE, ESC_TOP3, ESC_GAP, ESC_LOWEST):
        assert line in shown
    assert "hypertension's importance, not heart_disease's" not in shown


def test_a_range_every_member_of_the_group_is_in_is_shown():
    true = "- all smoking_status categories contribute modestly (≈0.017–0.02)."
    assert _esc(true).findings == []


def test_a_bound_holds_on_what_the_claim_names_and_a_false_one_does_not():
    assert _esc(ESC_TOP3).findings == []                      # the three sum to 72.4%
    false = _esc(ESC_TOP3.replace("over 70", "over 80"))
    assert [(f.check, f.detail) for f in false.findings] == [
        ("invented_number", "cites over 80 %, which does not appear in the computed evidence within tolerance")]
    under = _esc("- **Top three**: avg_glucose_level, age, bmi – together they account for under 70% of the "
                 "importance.")
    assert [f.check for f in under.findings] == ["invented_number"]


@pytest.mark.parametrize("raw, value", [
    ("5.1 × 10⁻⁶", 5.1e-6), ("8.7×10⁻⁹", 8.7e-9), ("5.1e-06", 5.1e-6), ("5.1E-6", 5.1e-6),
    ("3.2 x 10^-4", 3.2e-4), ("1.5·10³", 1500.0)])
def test_scientific_notation_is_one_figure(raw, value):
    found = [m.group(0) for m in cv._NUMBER.finditer(f"importance ({raw}) here")]
    assert found == [raw]
    assert cv._parse_number(raw) == pytest.approx(value)


def test_a_paired_list_is_paired_in_order():
    assert _esc(ESC_GAP).findings == []
    # Swapped, each figure is the other's: the reason says so the right way round.
    swapped = _esc(ESC_GAP.replace("~0.028 and 0.025", "~0.025 and 0.028"))
    assert [(f.check, f.detail) for f in swapped.findings] == [
        ("figure_misattributed", "0.025 is heart_disease's importance, not hypertension's (0.02784)")]
    three = "- avg_glucose_level, age and bmi lead the ranking (0.263, 0.242 and 0.218)."
    assert _esc(three).findings == []


def test_a_paired_list_that_cannot_be_paired_is_given_to_no_one():
    """Three entities, two figures: no pairing is safe, so no reason is given that may contradict it."""
    ambiguous = "- hypertension, heart_disease and bmi have similar importance (~0.028 and 0.025)."
    assert [f.check for f in _esc(ambiguous).findings if f.check == "figure_misattributed"] == []


def test_most_revenue_said_of_an_item_that_is_not_is_withheld_on_the_whole_result():
    """rweb3 groq messy_xlsx: the result held every item; the reply's "most
    revenue" item is 1,136.04 against 24,615.97. Its figure is the item's
    revenue (a sum of quantity × price), so the group check no longer
    withholds it on the figure: the superlative does, on the result."""
    rows = [{"StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER", "total_quantity": "1912",
             "total_sales": "24615.969999999998"},
            {"StockCode": "DOT", "Description": "DOTCOM POSTAGE", "total_quantity": "38", "total_sales": "20910.16"},
            {"StockCode": "15056BL", "Description": "EDWARDIAN PARASOL BLACK", "total_quantity": "187",
             "total_sales": "1136.04"},
            {"StockCode": "15056bl", "Description": "EDWARDIAN PARASOL BLACK", "total_quantity": "10",
             "total_sales": "127.2"},
            {"StockCode": "15056N", "Description": "EDWARDIAN PARASOL NATURAL", "total_quantity": "130",
             "total_sales": "752.9200000000001"}]
    claim = ("The item that generated the most revenue is the **EDWARDIAN PARASOL BLACK (StockCode 15056BL)** "
             "with total sales of **$1,136.04**.")
    report = verify_claims(claim, {"execution_output": rows})
    assert [(f.check, f.detail) for f in report.findings] == [(
        "superlative_contradicted",
        "15056BL's total_sales (1,136.04) is not the most: 22423 (REGENCY CAKESTAND 3 TIER) has 24,615.97")]
