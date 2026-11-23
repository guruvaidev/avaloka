"""rc15 live web check: what the verifier and the presenter got wrong.

- An en dash between a label and a figure was read as a minus sign:
  "Highest sales: 22423 – $24,615.97" was withheld as "cites -24,615.97".
- The sentence splitter split "($545.08 vs. $182.27)" at "vs.". It withheld
  the first half and SHOWED the second, "$182.27).", to the user.
- "The gap between enterprise and SMB is the widest, over $360" (545.08 -
  182.27) and "... over $480" (545.33 - 59.79) were withheld as invented.
- "the UK also leads with 19,012 rows in 2010 Q4 and 18,869 rows in 2011 Q1"
  was withheld: 18,869 was given to 2010 Q4, the entity named before it,
  though "rows in 2011 Q1" says whose it is.

Replies and rows are the ones those runs produced (rweb15 results/rw7,
rw7b, rw7c: related/ask_seg.json, messy_xlsx/chat.json, messy_xlsx/ask.json);
no LLM runs.
"""

from __future__ import annotations

import re

import pytest

from app.agents.claim_verifier import presentable_reply, split_claims, verify_claims


def _report(reply, rows, question=None):
    return verify_claims(reply, {"execution_output": rows}, None, question=question)


# --------------------------------------------------------------------------- #
# An en dash between a label and a figure
# --------------------------------------------------------------------------- #

DASH_ROWS = [{"StockCode": "84077", "Description": "WORLD WAR 2 GLIDERS ASSTD DESIGNS", "total_quantity": "5147",
              "total_sales": "1160.21"},
             {"StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER", "total_quantity": "1899",
              "total_sales": "24615.969999999998"}]
DASH_HIGHEST = "- Highest sales: 22423 – $24,615.97"
DASH_LOWEST = "- Lowest sales: 84077 – $1,160.21"
DASH_REPLY = ("The REGENCY CAKESTAND 3 TIER (StockCode 22423) sells the most, with total sales of **$24,615.97** "
              "versus **$1,160.21** for the WORLD WAR 2 GLIDERS ASSTD DESIGNS (StockCode 84077).  \n\n"
              f"{DASH_HIGHEST}  \n{DASH_LOWEST}  \n"
              "- Quantity sold: 22423 – 1,899 units; 84077 – 5,147 units  \n"
              "- The item with the highest quantity (84077) is not the top seller.")


def test_an_en_dash_between_a_label_and_a_figure_is_not_a_minus():
    report = _report(DASH_REPLY, DASH_ROWS, "what sells most?")
    assert report.findings == []
    assert DASH_HIGHEST in presentable_reply(DASH_REPLY, report.as_dict())


@pytest.mark.parametrize("claim", ["The coefficient for fare is −1.52 per mile.",
                                   "The coefficient for fare is -1.52 per mile.",
                                   "The coefficient for fare is –1.52 per mile."])
def test_a_real_minus_is_still_a_minus(claim):
    rows = [{"feature": "fare", "coefficient": "-1.52"}]
    assert _report(claim, rows).findings == []
    positive = [{"feature": "fare", "coefficient": "1.52"}]
    assert [f.check for f in _report(claim, positive).findings] == ["invented_number"]


# --------------------------------------------------------------------------- #
# "vs." is not the end of a sentence; nothing but whole sentences is shown
# --------------------------------------------------------------------------- #

VS_ROWS = [{"segment": "enterprise", "average_amount": "545.08"}, {"segment": "smb", "average_amount": "182.27"},
           {"segment": "consumer", "average_amount": "59.77"}]
VS_LEADS = "- Enterprise leads by a wide margin over SMB ($545.08 vs. $182.27)."
VS_REPLY = ("Enterprise customers spend the most on average, with $545.08 per order, followed by SMBs at $182.27, "
            "while consumers average only $59.77.  \n\n"
            f"{VS_LEADS}  \n"
            "- SMB is roughly three times higher than consumers ($182.27 vs. $59.77).  \n"
            "- The gap between enterprise and consumer is the largest ($485.31).  \n"
            "- No other segments are present in the data.")


def test_vs_does_not_split_a_sentence_and_the_comparison_is_paired_in_order():
    assert VS_LEADS in split_claims(VS_REPLY)
    report = _report(VS_REPLY, VS_ROWS, "How does the average order amount differ by customer segment?")
    assert report.findings == []
    shown = presentable_reply(VS_REPLY, report.as_dict())
    assert shown == VS_REPLY
    # The comparison swapped is still caught, with the right reason.
    swapped = _report(VS_LEADS.replace("over SMB", "over consumer"), VS_ROWS)
    assert [(f.check, f.detail) for f in swapped.findings] == [
        ("figure_misattributed", "182.27 is smb's average_amount, not consumer's (59.77)")]


@pytest.mark.parametrize("first, second", [
    ("Revenue rose, e.g. in Q4.", "It fell again in the first quarter."),
    ("Most orders are small, i.e. under $60.", "Some are far larger than that."),
    ("Sales peak at 9 a.m. on Mondays.", "They dip later in the afternoon."),
    ("See Fig. 2 for the trend.", "It is flat across the year."),
    ("Item No. 5 sells most.", "It is a three-tier cake stand."),
    ("It is approx. 3 times larger.", "That is the whole difference."),
    ("Revenue was $545.08 vs. $182.27 last year.", "It held steady this year."),
])
def test_abbreviations_do_not_end_a_sentence(first, second):
    assert split_claims(f"{first} {second}") == [first, second]


_ABBREVIATIONS = ("vs.", "e.g.", "i.e.", "approx.", "etc.", "No.", "Fig.", "a.m.", "p.m.")


def _starts_a_sentence(original: str, at: int) -> bool:
    before = original[:at]
    if not before.strip() or before.endswith("\n"):
        return True
    head = before.rstrip()
    if len(head) == len(before) or head[-1] not in ".!?":
        return False
    return not any(head.endswith(a) for a in _ABBREVIATIONS)


def _ends_a_sentence(original: str, at: int) -> bool:
    rest = original[at:]
    return not rest.strip() or rest[0] in " \n\t" and original[:at].rstrip()[-1:] in ".!?)*" or rest.startswith("\n")


def _shown_spans(shown: str) -> list:
    body = shown.split("\n\nI've held back")[0]
    return [line.strip() for line in body.split("\n") if line.strip() and not line.startswith("Why: ")]


@pytest.mark.parametrize("withheld", [
    "- Enterprise leads by a wide margin over SMB ($545.08 vs.",         # the rc15 report, split at "vs."
    VS_LEADS,
    "- SMB is roughly three times higher than consumers ($182.27 vs. $59.77).",
    "- The gap between enterprise and consumer is the largest ($485.31).",
])
def test_only_whole_sentences_of_the_reply_are_ever_shown(withheld):
    """The invariant: whatever is withheld, every span shown is a substring of
    the reply that starts and ends at a sentence boundary of it. rc15 showed
    "$182.27)." on a line of its own."""
    report = {"safe_to_present": False, "claims_checked": 5, "claims_reconciled": False,
              "findings": [{"claim": withheld, "verdict": "contradicted", "check": "figure_misattributed",
                            "detail": "545.08 is enterprise's average_amount, not smb's (182.3)"}]}
    shown = presentable_reply(VS_REPLY, report)
    for span in _shown_spans(shown):
        if span.startswith("I've held back my answer"):
            continue
        for piece in re.split(r"(?<=\S)  +(?=\S)", span):
            at = VS_REPLY.find(piece)
            assert at >= 0, piece
            assert _starts_a_sentence(VS_REPLY, at), piece
            assert _ends_a_sentence(VS_REPLY, at + len(piece)), piece
    assert "\n$182.27)." not in shown


# --------------------------------------------------------------------------- #
# A gap between two figures of the result
# --------------------------------------------------------------------------- #

GAP_ROWS_RW7 = [{"segment": "enterprise", "average_order_amount": "545.08"},
                {"segment": "smb", "average_order_amount": "182.27"},
                {"segment": "consumer", "average_order_amount": "59.77"}]
GAP_ROWS_RW7B = [{"segment": "enterprise", "average_amount": "545.33"}, {"segment": "smb", "average_amount": "182.17"},
                 {"segment": "consumer", "average_amount": "59.79"}]
GAP_360 = "- The gap between enterprise and SMB is the widest, over $360."
GAP_480 = "- The gap between enterprise and consumer is over $480."


def test_a_bounded_gap_between_two_result_figures_is_supported():
    reply = ("Enterprise customers spend the most on average, with $545.08 per order, compared to SMBs at $182.27 "
             "and consumers at $59.77.  \n\n- Enterprise leads by a factor of roughly 3× SMB and 9× consumers.  \n"
             f"- SMB orders are about three times larger than consumer orders.  \n{GAP_360}  \n"
             "- Consumer orders are the smallest, under $60 on average.")
    assert _report(reply, GAP_ROWS_RW7).findings == []
    assert _report(GAP_480, GAP_ROWS_RW7B).findings == []


def test_a_bounded_gap_that_does_not_hold_is_still_withheld():
    wrong = _report(GAP_360.replace("over $360", "over $400"), GAP_ROWS_RW7)
    assert [f.check for f in wrong.findings] == ["invented_number"]


# --------------------------------------------------------------------------- #
# "18,869 rows in 2011 Q1" is 2011 Q1's
# --------------------------------------------------------------------------- #

UK_ROWS = [{"Dataset": "2010 Q4", "TopCountry": "United Kingdom", "RowCount": "19012"},
           {"Dataset": "2011 Q1", "TopCountry": "United Kingdom", "RowCount": "18869"},
           {"Dataset": "Combined", "TopCountry": "United Kingdom", "RowCount": "37881"}]
UK_QUARTERS = ("- In the quarterly snapshots, the UK also leads with 19,012 rows in 2010 Q4 and 18,869 rows in "
               "2011 Q1.")
UK_REPLY = ("United Kingdom has the highest row count across all datasets.  \n\n"
            f"{UK_QUARTERS}  \n"
            "- The combined dataset shows 37,881 rows for the UK, the largest total.  \n"
            "- No other country appears in the table, so the UK is the sole leader.")


def test_a_figure_followed_by_in_entity_is_that_entitys():
    assert _report(UK_REPLY, UK_ROWS, "Which Country has the most rows?").findings == []
    swapped = _report(UK_QUARTERS.replace("19,012", "@").replace("18,869", "19,012").replace("@", "18,869"), UK_ROWS)
    assert [(f.check, f.detail) for f in swapped.findings] == [
        ("figure_misattributed", "18,869 is 2011 Q1's RowCount, not 2010 Q4's (19,012)")]
