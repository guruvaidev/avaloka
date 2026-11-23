"""A superlative said in words no column has, and a ratio no check read (rc18d web check, D13.1/D13.2).

"what sells most?": the code kept the top 10 products by REVENUE
(``agg.sort_values(by='TotalRevenue', ...).head(TOP_N)``) and the reply said:

    The white hanging heart t-light holder sold the most units.
    - It moved 3,648 pieces, more than double the next highest item.

The verifier reported 5 claims checked and 0 problems, and the chat said "I
checked 5 claims". Over all rows WORLD WAR 2 GLIDERS ASSTD DESIGNS (84077) sold
5,147 units and is not in a revenue top 10. And within the result the next
highest quantity is 2,393: 3,648 is 1.52 times it, not "more than double".

Why it passed: "units" names no column (there was no measure lexicon); the
figure was in the next sentence, which starts with "It" and so names no
entity; no check read a ratio; ``claims_checked`` counted sentences, not
checks; and the scope line did not list the ``head(10)``.

Reply, code and rows are the live turn's (tests/fixtures/rc18d_superlatives,
minimum fields); the invoice lines are synthetic and reproduce the result's
totals with the returns left out (as in the workbook, no sold line lacks a
price). Product names are
from the public UCI Online Retail data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.claim_verifier import claim_verifier_node, presentable_reply, withheld_claims
from app.api import chat_voice

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "rc18d_superlatives" / "turn.json").read_text())
QUESTION, CODE, ROWS, REPLY = FIXTURE["question"], FIXTURE["code"], FIXTURE["rows"], FIXTURE["reply"]
MOST_UNITS = "The white hanging heart t‑light holder sold the most units."
DOUBLE = "- It moved 3,648 pieces, more than double the next highest item."
#: Items the revenue top 10 left out: (StockCode, Description, quantity, revenue).
OTHERS = [("84077", "WORLD WAR 2 GLIDERS ASSTD DESIGNS", 5147, 1160.21),
          ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 3737, 2495.61),
          ("20849", "PAPER CRAFT LITTLE BIRDIE", 3, 0.0)]


def _lines():
    """Invoice lines over the two sheets; every item's quantity and revenue
    split between them, and a return (negative quantity) the code drops."""
    items = [(r["StockCode"], r["Description_clean"].upper(), int(r["TotalQuantity"]), float(r["TotalRevenue"]))
             for r in ROWS]
    q4, q1 = [], []
    for n, (code, desc, qty, revenue) in enumerate(items + OTHERS):
        price = round(revenue / qty, 2)
        rest = round(revenue - (qty - 1) * price, 2)
        line = {"InvoiceNo": str(536365 + n), "StockCode": code, "Description": desc,
                "InvoiceDate": "2010-12-01 08:26:00", "CustomerID": 17850.0 + n, "Country": "United Kingdom"}
        q4.append({**line, "Quantity": qty - 1, "UnitPrice": price})
        q1.append({**line, "Quantity": 1, "UnitPrice": rest})
        q1.append({**line, "InvoiceNo": "C" + line["InvoiceNo"], "Quantity": -12, "UnitPrice": price})
    return pd.DataFrame(q4), pd.DataFrame(q1)


def _turn(tmp_path, reply=REPLY, code=CODE, rows=ROWS, frames=None, reads=True):
    paths = []
    for name, frame in zip(("q4", "q1"), frames or _lines()):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append({"path": str(path), "used": True})
    state = {"messages": [HumanMessage(content=QUESTION), AIMessage(content=reply)],
             "coder_definition": {"code": code},
             "execution_result": {"status": "success", "reads": paths if reads else []},
             "output_json": rows}
    report = claim_verifier_node(state)["verification_report"]
    return state, report, presentable_reply(reply, report)


def _finding(report, claim):
    return next((f for f in report["findings"] if f["claim"] == claim), None)


def test_the_lines_reproduce_the_result_and_gliders_sold_the_most():
    q4, q1 = _lines()
    both = pd.concat([q4, q1]).dropna(subset=["UnitPrice"])
    kept = both[both["Quantity"] > 0]
    qty = kept.groupby("StockCode")["Quantity"].sum()
    revenue = (kept["Quantity"] * kept["UnitPrice"]).groupby(kept["StockCode"]).sum()
    assert qty.idxmax() == "84077" and qty["84077"] == 5147 and qty["85123A"] == 3648
    assert round(revenue["85123A"], 2) == 10132.48
    assert "84077" not in revenue.sort_values(ascending=False).head(10).index


def test_t13_1_the_most_units_over_a_revenue_top_10_is_withheld(tmp_path):
    """"Units" is a quantity: the claim is checked over all rows, where 84077 sold 5,147."""
    _state, report, shown = _turn(tmp_path)
    finding = _finding(report, MOST_UNITS)
    assert finding is not None, report["findings"]
    assert finding["check"] == "superlative_of_subset"
    assert "84077 (5,147)" in finding["detail"] and "85123A ranks 3 (3,648)" in finding["detail"]
    assert MOST_UNITS in withheld_claims(report)
    assert "sold the most units" not in shown


def test_t13_1b_more_than_double_the_next_is_a_ratio_the_result_contradicts(tmp_path):
    """3,648 / 2,393 = 1.52: "It" is the item the sentence before named."""
    _state, report, shown = _turn(tmp_path)
    finding = _finding(report, DOUBLE)
    assert finding is not None, report["findings"]
    assert finding["check"] == "ratio_contradicted"
    assert "1.52 times" in finding["detail"] and "2,393" in finding["detail"]
    assert DOUBLE in withheld_claims(report)
    assert "It moved 3,648 pieces" not in shown


def test_t13_1c_a_true_superlative_over_every_group_is_not_flagged(tmp_path):
    """The same data with no cut: gliders did sell the most units."""
    code = CODE.replace("    top_products = agg.head(TOP_N).reset_index(drop=True)\n",
                        "    top_products = agg.reset_index(drop=True)\n")
    q4, q1 = _lines()
    both = pd.concat([q4, q1]).dropna(subset=["UnitPrice"])
    kept = both[both["Quantity"] > 0].assign(Revenue=lambda d: d["Quantity"] * d["UnitPrice"])
    agg = kept.groupby(["StockCode", "Description"], as_index=False).agg(
        TotalQuantity=("Quantity", "sum"), TotalRevenue=("Revenue", "sum"))
    rows = [{"StockCode": r.StockCode, "Description_clean": r.Description.lower(),
             "TotalQuantity": str(r.TotalQuantity), "TotalRevenue": str(round(r.TotalRevenue, 2))}
            for r in agg.sort_values("TotalRevenue", ascending=False).itertuples()]
    # 5,147 / 3,737 (21212) = 1.38: "about 1.4 times" is true, and 1.4 is a
    # ratio of the ranking, not a figure the result must hold.
    reply = ("World war 2 gliders asstd designs sold the most units.\n"
             "- It moved 5,147 pieces, about 1.4 times the next highest item.")
    _state, report, shown = _turn(tmp_path, reply=reply, code=code, rows=rows)
    assert report["findings"] == [], report["findings"]
    assert report["claims_checked"] == 2 and report.get("claims_seen") == 2
    assert "sold the most units." in shown and "about 1.4 times" in shown
    assert "keeps only" not in shown                      # no cut, no truncation line


def test_t13_1c_the_most_units_over_a_quantity_top_10_is_not_flagged(tmp_path):
    """Cut by the measure the superlative is about, the top of the cut is the top of all."""
    code = CODE.replace("by='TotalRevenue'", "by='TotalQuantity'")
    q4, q1 = _lines()
    both = pd.concat([q4, q1]).dropna(subset=["UnitPrice"])
    kept = both[both["Quantity"] > 0].assign(Revenue=lambda d: d["Quantity"] * d["UnitPrice"])
    agg = kept.groupby(["StockCode", "Description"], as_index=False).agg(
        TotalQuantity=("Quantity", "sum"), TotalRevenue=("Revenue", "sum"))
    rows = [{"StockCode": r.StockCode, "Description_clean": r.Description.lower(),
             "TotalQuantity": str(r.TotalQuantity), "TotalRevenue": str(round(r.TotalRevenue, 2))}
            for r in agg.sort_values("TotalQuantity", ascending=False).head(10).itertuples()]
    reply = "World war 2 gliders asstd designs sold the most units, 5,147."
    _state, report, _shown = _turn(tmp_path, reply=reply, code=code, rows=rows)
    assert report["findings"] == [], report["findings"]
    assert report["claims_checked"] == 1


def test_units_comma_figure_keeps_the_figure_with_its_measure(tmp_path):
    """"sold the most units, 3,648": the comma does not end what "the most" says."""
    reply = "The white hanging heart t‑light holder sold the most units, 3,648."
    _state, report, _shown = _turn(tmp_path, reply=reply)
    finding = _finding(report, reply)
    assert finding is not None and finding["check"] == "superlative_of_subset", report["findings"]


def test_t13_1d_an_unconfirmable_superlative_over_a_cut_is_not_counted_and_the_cut_is_listed(tmp_path):
    """The frames the run read are gone: "the most units" over a revenue top 10
    cannot be checked over all rows. It is not counted as checked, the answer
    is not verified, the reply says it was not checked over all rows, and the
    scope line says what the code cut."""
    reply = ("The white hanging heart t\u2011light holder sold the most units (3,648).\n"
             "- The regency cakestand 3 tier brought in 25,588.57.")
    _state, report, shown = _turn(tmp_path, reply=reply, reads=False)
    first = reply.split("\n")[0]
    assert report["claims_checked"] == 1 and report.get("claims_seen") == 2
    assert report["verified"] is False
    assert report.get("unconfirmed_claims") == [first]
    assert report["findings"] == []
    line = "Scope: the result keeps only the top 10 rows by TotalRevenue"
    assert any(c.startswith(line) for c in report["caveats"]), report.get("caveats")
    assert line in shown
    assert ("Not checked over all rows: white hanging heart t-light holder has the most TotalQuantity among the 10 "
            "rows the result keeps (the top 10 by TotalRevenue)") in shown


def test_the_rc18d_turn_counts_only_what_a_check_resolved_and_lists_the_cut(tmp_path):
    _state, report, shown = _turn(tmp_path)
    # Five statements; "The top revenue item still lagged behind the top quantity
    # seller in volume." is one no check can resolve.
    assert report["claims_checked"] == 4 and report.get("claims_seen") == 5
    assert report["n_problems"] == 2
    assert any(c.startswith("Scope: the result keeps only the top 10 rows by TotalRevenue")
               for c in report["caveats"])
    assert "Scope: the result keeps only the top 10 rows by TotalRevenue" in shown


def test_the_step_card_says_k_of_n(tmp_path):
    state, report, _shown = _turn(tmp_path)
    final = {**state, "verification_report": report, "verification_safe_to_present": report["safe_to_present"]}
    card = next(p for p in chat_voice.prompts_after_turn(final, question=QUESTION) if p["step"] == "verification")
    assert card["message"].startswith(
        "I checked 4 of the 5 statements in my answer against the numbers; 2 did not hold and I've flagged them."), \
        card["message"]
    assert "I checked 5" not in card["message"]


def test_the_card_never_claims_more_than_were_checked():
    """A report with the seen count: K < N says K of N; K == N keeps the old card."""
    def card(report):
        return chat_voice.prompts_after_turn({"verification_report": report}, question="q")[0]["message"]

    assert card({"safe_to_present": True, "claims_checked": 3, "claims_seen": 5, "n_problems": 0}) == \
        "I checked 3 of the 5 statements in my answer against the numbers."
    assert card({"safe_to_present": True, "claims_checked": 1, "claims_seen": 2, "n_problems": 0}) == \
        "I checked 1 of the 2 statements in my answer against the numbers."
    assert card({"safe_to_present": True, "claims_checked": 2, "claims_seen": 2, "n_problems": 0}) == \
        "I checked 2 claims in my answer against the numbers."


def test_a_superlative_without_a_measure_word_takes_the_figure_of_the_next_sentence(tmp_path):
    """"X sold the most. / It moved 3,648 pieces": the figure gives the measure;
    the finding is the first sentence's, which says "the most"."""
    first = "The white hanging heart t‑light holder sold the most."
    reply = f"{first}\n- It moved 3,648 pieces."
    _state, report, shown = _turn(tmp_path, reply=reply)
    finding = _finding(report, first)
    assert finding is not None and finding["check"] == "superlative_of_subset", report["findings"]
    assert "84077 (5,147)" in finding["detail"]
    assert report["claims_checked"] == 2


def test_it_with_another_rows_figure_is_misattributed(tmp_path):
    """"It moved 2,393 pieces" of the heart holder: 2,393 is the paper chain kit's."""
    second = "- It moved 2,393 pieces."
    reply = f"The white hanging heart t‑light holder is popular.\n{second}"
    _state, report, _shown = _turn(tmp_path, reply=reply)
    finding = _finding(report, second)
    assert finding is not None and finding["check"] == "figure_misattributed", report["findings"]
    assert "22086" in finding["detail"] or "paper chain kit" in finding["detail"]


def test_a_pronoun_reaches_only_the_claim_just_before(tmp_path):
    reply = ("The white hanging heart t‑light holder is popular.\n"
             "- Shoppers buy many lights before Christmas.\n"
             "- It moved 2,393 pieces.")
    _state, report, _shown = _turn(tmp_path, reply=reply)
    assert not any(f["check"] == "figure_misattributed" for f in report["findings"]), report["findings"]


def test_an_ambiguous_measure_word_is_no_match():
    from app.agents.claim_extremes import _lexicon

    assert _lexicon(["units"], ["TotalQuantity", "TotalRevenue"]) == ["TotalQuantity"]
    assert _lexicon(["revenue"], ["TotalQuantity", "TotalRevenue"]) == ["TotalRevenue"]
    assert _lexicon(["units"], ["TotalQuantity", "AvgQuantity"]) == []          # two candidates: no guess
    assert _lexicon(["units"], ["UnitPrice", "TotalRevenue"]) == []             # a price is no quantity
    assert _lexicon(["orders"], ["OrderCount", "TotalQuantity"]) == ["OrderCount"]
    assert _lexicon(["units"], ["OrderCount", "TotalQuantity"]) == ["TotalQuantity"]
    assert _lexicon(["popular"], ["TotalQuantity"]) == []                       # not a measure word
    assert _lexicon(["number", "of", "units"], ["TotalQuantity"]) == ["TotalQuantity"]


def test_the_cut_is_read_from_the_code():
    from app.agents.claim_extremes import result_truncation

    cut = result_truncation(CODE, 10)
    assert (cut["n"], cut["by"], cut["top"]) == (10, "TotalRevenue", True)
    assert result_truncation(CODE, 11) is None                  # the result is not the cut's
    assert result_truncation("def main(df):\n    return df.nlargest(5, 'Sales')\n", 5)["by"] == "Sales"
    low = result_truncation("def main(df):\n    return df.sort_values('Sales').head(3)\n", 3)
    assert (low["by"], low["top"]) == ("Sales", False)
    top = result_truncation("def main(df):\n    return df.sort_values('Sales').tail(3)\n", 3)
    assert top["top"] is True
    iloc = result_truncation("def main(df):\n    out = df.sort_values(by=['Qty'], ascending=[False])\n"
                             "    return out.iloc[:4]\n", 4)
    assert (iloc["by"], iloc["top"]) == ("Qty", True)
    assert result_truncation("def main(df):\n    print(df.head())\n    return df.groupby('a').sum()\n", 3) is None


def test_a_cut_the_answer_gives_is_not_listed_again(tmp_path):
    reply = "Of the top 10 products by revenue, the regency cakestand 3 tier brought in 25,588.57."
    _state, report, shown = _turn(tmp_path, reply=reply)
    assert report["truncation"]["disclosed_in_answer"] is True
    assert "keeps only the top 10" not in shown


def test_a_true_ratio_is_counted_and_not_read_as_an_invented_figure(tmp_path):
    """Within the result 85123A (3,648) is 1.52 times 22086 (2,393)."""
    reply = "The white hanging heart t‑light holder moved 3,648 pieces, about 1.5 times the next highest item."
    _state, report, _shown = _turn(tmp_path, reply=reply,
                                   code=CODE.replace("by='TotalRevenue'", "by='TotalQuantity'"),
                                   rows=sorted(ROWS, key=lambda r: -int(r["TotalQuantity"])))
    assert not any(f["check"] in ("ratio_contradicted", "invented_number") for f in report["findings"]), \
        report["findings"]


def test_a_cut_to_the_one_row_a_most_question_asks_for_is_not_a_scope(tmp_path):
    """rweb18d "what sells most?" kept head(1): the question asked for the top one."""
    from app.agents.filter_disclosure import apply_truncation_scope, truncation_line

    code = "def main(df):\n    return df.sort_values('total_revenue', ascending=False).head(1)\n"
    rows = [{"Description": "REGENCY CAKESTAND 3 TIER", "total_revenue": 25588.57}]
    state = {"messages": [HumanMessage(content="what sells most?")]}
    out = apply_truncation_scope({}, state, "Regency cakestand sells most.", code, rows)
    assert out["truncation"]["asked"] is True and "caveats" not in out
    state = {"messages": [HumanMessage(content="show me the sales")]}
    out = apply_truncation_scope({}, state, "Regency cakestand.", code, rows)
    assert out["caveats"] == [truncation_line(out["truncation"])]
    assert out["caveats"][0].startswith("Scope: the result keeps only the top row by total_revenue,")
