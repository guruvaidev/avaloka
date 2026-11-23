"""Superlatives said of an item named by its code (rweb16 messy_xlsx, rw7b).

"what sells most?": the code dropped returns (Quantity > 0), kept the top 10
StockCodes by QUANTITY, and the reply ranked revenue within those ten:
"Lowest revenue: 84077 – $1,160.21". Over every item with a sale the lowest
revenue is $0.00, so the line is false; it was shown unmarked on d199d120.
84077 is a StockCode, but an all-digit cell read as a number, never as a
name, so the superlative check saw no entity in the claim. And with two text
columns (StockCode, Description) and a filter in the code, the check over
all rows bailed out before it began.

Key-column codes are entities now, digits only or not (claim_figures.
key_columns); a filter the code applies with a literal is applied to the
recompute; and the description beside each code is another name of the row,
not a subgroup. A plain number that is no key (a year, a count) is still no
entity.

Reply, code and rows are rweb16's (states/rweb16/rw7b/messy_xlsx/chat.json);
the invoice lines are synthetic and reproduce the result's totals only when
the returns are left out.
"""

from __future__ import annotations

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.claim_verifier import claim_verifier_node, presentable_reply, withheld_claims

QUESTION = "what sells most?"
REPLY = 'The item with the highest sales is the **WHITE HANGING HEART T‑LIGHT HOLDER (StockCode\u202f85123A)**, generating **$10,132.48** in revenue.  \n\n- Highest revenue: 85123A – $10,132.48  \n- Lowest revenue: 84077 – $1,160.21  \n- Largest quantity sold: 84077 – 5,147 units  \n- Highest quantity sold: 85123A – 3,648 units, yet its revenue far exceeds the next best (22086 – $8,813.85).'
CODE = "import pandas as pd\n\ndef main(df):\n    path_q4 = '/var/tmp/avaloka/13695437-5a6a-40c8-a6d9-675ebb20f4aa/input_ba374879-160a-4ccb-8134-c8d39d4fc00e.csv'\n    path_q1 = '/var/tmp/avaloka/dd2fdae6-ee5b-4707-ab61-1091e6646b61/input_f71652fa-c980-41b0-b24c-6bfdd979a988.csv'\n    df_q4 = pd.read_csv(path_q4, parse_dates=['InvoiceDate'])\n    df_q1 = pd.read_csv(path_q1, parse_dates=['InvoiceDate'])\n    df = pd.concat([df_q4, df_q1], ignore_index=True)\n    df = df[df['Quantity'] > 0]\n    df = df.dropna(subset=['StockCode', 'Description'])\n    df['SalesAmount'] = df['Quantity'] * df['UnitPrice']\n    agg_df = df.groupby(['StockCode', 'Description'], as_index=False).agg(TotalQuantity=('Quantity', 'sum'), TotalSales=('SalesAmount', 'sum'))\n    agg_df = agg_df.sort_values(by='TotalQuantity', ascending=False)\n    N = 10\n    result = agg_df.head(N).reset_index(drop=True)\n    return result\nif __name__ == '__main__':\n    dummy_df = pd.DataFrame()\n    output_df = main(dummy_df)\n    output_df.to_csv('output.csv', index=False)"
ROWS = [
    {"StockCode": "84077", "Description": "WORLD WAR 2 GLIDERS ASSTD DESIGNS", "TotalQuantity": "5147", "TotalSales": "1160.21"},
    {"StockCode": "21212", "Description": "PACK OF 72 RETROSPOT CAKE CASES", "TotalQuantity": "3737", "TotalSales": "2495.61"},
    {"StockCode": "85123A", "Description": "WHITE HANGING HEART T-LIGHT HOLDER", "TotalQuantity": "3648", "TotalSales": "10132.48"},
    {"StockCode": "22834", "Description": "HAND WARMER BABUSHKA DESIGN", "TotalQuantity": "3352", "TotalSales": "4438.62"},
    {"StockCode": "22197", "Description": "SMALL POPCORN HOLDER", "TotalQuantity": "2655", "TotalSales": "2514.0699999999997"},
    {"StockCode": "22492", "Description": "MINI PAINT SET VINTAGE ", "TotalQuantity": "2602", "TotalSales": "1546.6000000000001"},
    {"StockCode": "22693", "Description": "GROW A FLYTRAP OR SUNFLOWER IN TIN", "TotalQuantity": "2593", "TotalSales": "2479.01"},
    {"StockCode": "22086", "Description": "PAPER CHAIN KIT 50'S CHRISTMAS ", "TotalQuantity": "2393", "TotalSales": "8813.85"},
    {"StockCode": "84879", "Description": "ASSORTED COLOUR BIRD ORNAMENT", "TotalQuantity": "2339", "TotalSales": "3931.79"},
    {"StockCode": "22189", "Description": "CREAM HEART CARD HOLDER", "TotalQuantity": "2279", "TotalSales": "5721.21"},
]
LOWEST = "- Lowest revenue: 84077 – $1,160.21"
HIGHEST = "- Highest revenue: 85123A – $10,132.48"
LARGEST_QTY = "- Largest quantity sold: 84077 – 5,147 units"
#: Items the top 10 by quantity left out: (StockCode, Description, quantity, revenue).
OTHERS = [("22423", "REGENCY CAKESTAND 3 TIER", 1912, 25588.57), ("20849", "PAPER CRAFT LITTLE BIRDIE", 3, 0.0),
          ("21000", "ROSE SCENT CANDLE", 5, 4.25)]


def _lines():
    """Invoice lines over two sheets; every item's quantity and revenue split
    between them, and a return (negative quantity) the code's filter drops."""
    items = [(r["StockCode"], r["Description"], int(r["TotalQuantity"]), float(r["TotalSales"])) for r in ROWS]
    q4, q1 = [], []
    for n, (code, desc, qty, sales) in enumerate(items + OTHERS):
        price = round(sales / qty, 2)
        rest = round(sales - (qty - 1) * price, 2)
        line = {"InvoiceNo": str(536365 + n), "StockCode": code, "Description": desc,
                "InvoiceDate": "2010-12-01 08:26:00", "CustomerID": 17850.0 + n, "Country": "United Kingdom"}
        q4.append({**line, "Quantity": qty - 1, "UnitPrice": price})
        q1.append({**line, "Quantity": 1, "UnitPrice": rest})
        q1.append({**line, "InvoiceNo": "C" + line["InvoiceNo"], "Quantity": -12, "UnitPrice": price})
    # An item that was only ever returned: the lowest of all without the filter.
    q1.append({"InvoiceNo": "C540000", "StockCode": "79323W", "Description": "WHITE CHERRY LIGHTS",
               "InvoiceDate": "2011-01-04 10:00:00", "CustomerID": 12346.0, "Country": "United Kingdom",
               "Quantity": -80995, "UnitPrice": 2.08})
    return pd.DataFrame(q4), pd.DataFrame(q1)


def _turn(tmp_path, reply=REPLY, code=CODE, rows=ROWS, frames=None):
    reads = []
    for name, frame in zip(("q4", "q1"), frames or _lines()):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        reads.append({"path": str(path), "used": True})
    state = {"messages": [HumanMessage(content=QUESTION), AIMessage(content=reply)],
             "coder_definition": {"code": code},
             "execution_result": {"status": "success", "reads": reads},
             "output_json": rows}
    report = claim_verifier_node(state)["verification_report"]
    return report, presentable_reply(reply, report)


def test_the_lines_reproduce_the_result_only_without_the_returns():
    q4, q1 = _lines()
    both = pd.concat([q4, q1])
    kept = both[both["Quantity"] > 0]
    revenue = (kept["Quantity"] * kept["UnitPrice"]).groupby(kept["StockCode"]).sum()
    assert round(revenue["84077"], 2) == 1160.21 and revenue.min() == 0.0
    assert revenue.idxmax() == "22423"


def test_lowest_revenue_of_a_digit_only_code_is_withheld(tmp_path):
    report, shown = _turn(tmp_path)
    assert LOWEST in withheld_claims(report)
    assert LOWEST not in shown
    finding = next(f for f in report["findings"] if f["claim"] == LOWEST)
    assert finding["check"] == "superlative_of_subset"
    assert ("84077 is the lowest only among the 10 StockCode values the analysis kept; over all rows with "
            "Quantity > 0 the lowest TotalSales (sum of Quantity × UnitPrice per StockCode) is 20849 (0)"
            ) in finding["detail"]


def test_highest_revenue_of_an_alphanumeric_code_beside_its_description_is_withheld(tmp_path):
    report, shown = _turn(tmp_path)
    assert HIGHEST in withheld_claims(report)
    assert HIGHEST not in shown


def test_largest_quantity_of_a_digit_only_code_that_is_true_still_shows(tmp_path):
    report, shown = _turn(tmp_path)
    assert LARGEST_QTY not in withheld_claims(report)
    assert LARGEST_QTY in shown


def test_a_year_is_a_number_not_an_entity(tmp_path):
    """Years key this result, but a year is a value, not a code: "2011" in the
    claim names no entity, so no superlative is checked over all rows."""
    rows = [{"Year": "2010", "TotalSales": "500.0"}, {"Year": "2011", "TotalSales": "800.0"},
            {"Year": "2012", "TotalSales": "650.0"}]
    frame = pd.DataFrame({"Year": [2009, 2010, 2011, 2012], "Sales": [100.0, 500.0, 800.0, 650.0]})
    code = ("import pandas as pd\n\ndef main(df):\n    out = df.groupby('Year', as_index=False)"
            ".agg(TotalSales=('Sales', 'sum'))\n    return out.sort_values('TotalSales').tail(3)\n")
    reply = "- Lowest revenue: 2010 – $500.00"
    report, shown = _turn(tmp_path, reply=reply, code=code, rows=rows, frames=(frame, frame.iloc[:0]))
    assert not any(f["check"].startswith("superlative") for f in report["findings"])
