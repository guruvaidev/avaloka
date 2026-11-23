"""An answer that filtered rows the question did not ask to filter says so.

rc7 to rc14, related tables: asked "How does the average order amount differ
by customer segment?", the code dropped the 1,522 returned orders first and the
reply gave the means of the other 18,478 ("$545.08 per order"; over every order
it is $545.33). The filter showed only as a plan step, and the verifier called
the figures verified. Only the verifier's public surface is used here
(claim_verifier_node, presentable_reply), so these run against any version.
"""

from __future__ import annotations

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.claim_verifier import claim_verifier_node, presentable_reply

QUESTION = "How does the average order amount differ by customer segment?"

#: The rc14 reply, verbatim.
ANSWER = ("Enterprise customers spend the most on average, with $545.08 per order, followed by SMBs at $182.27, "
          "while consumer orders average only $59.77.")

#: The code that ran on rc14 (rweb14 rw7 related, ask_seg), verbatim.
CODE = """import pandas as pd

def main(_):
    customers_path = '/var/tmp/avaloka/802f1837-8149-4dcc-bd19-ee4a10eb9db1/input_c08faab8-d2be-44b2-88d4-aee81e05dc70.csv'
    orders_path = '/var/tmp/avaloka/f9e5e036-e61f-4e28-9b31-30201190d748/input_1992bc95-000a-424c-93ea-41a4d9a3bea7.csv'
    customers = pd.read_csv(customers_path, dtype={'customer_id': 'Int64', 'region': 'string', 'signup_year': 'Int64', 'segment': 'string'})
    orders = pd.read_csv(orders_path)
    customers['customer_id'] = customers['customer_id'].astype('Int64')
    orders['customer_id'] = pd.to_numeric(orders['customer_id'], errors='coerce').astype('Int64')
    orders['amount'] = pd.to_numeric(orders['amount'], errors='coerce')
    orders['returned'] = orders['returned'].astype('bool')
    orders = orders[orders['returned'] == False]
    merged = pd.merge(orders, customers, on='customer_id', how='inner')
    result = merged.groupby('segment', as_index=False).agg(avg_order_amount=('amount', 'mean'))
    result = result.sort_values(by='avg_order_amount', ascending=False).reset_index(drop=True)
    return result
if __name__ == '__main__':
    final_df = main(None)
    final_df.to_csv('output.csv', index=False)
"""

FILTER_LINE = "    orders = orders[orders['returned'] == False]\n"

PLAN = {"job_name": "Average Order Amount by Customer Segment", "transformations": [
    {"name": "Join Customers and Orders", "type": "cleaning", "columns": ["customer_id"]},
    {"name": "Filter Non-Returned Orders", "type": "filter", "columns": ["returned"]},
    {"name": "Group by Segment and Compute Avg Amount", "type": "aggregation", "columns": ["segment", "amount"]}]}

#: (segment, orders kept, mean kept, orders returned, all-order mean): 20,000
#: orders, 1,522 returned, the rc14 figures on the kept ones and the true
#: all-order means (545.33 / 182.17 / 59.79) over every order.
SEGMENTS = (("enterprise", 1850, 545.08, 150, 545.33), ("smb", 5550, 182.27, 450, 182.17),
            ("consumer", 11078, 59.77, 922, 59.79))


def _data(tmp_path, null_amounts: int = 0):
    customers, orders = [], []
    for i, (segment, kept, mean_kept, returned, mean_all) in enumerate(SEGMENTS):
        customer_id = i + 1
        customers.append({"customer_id": customer_id, "region": "West", "signup_year": 2022, "segment": segment})
        mean_returned = (mean_all * (kept + returned) - mean_kept * kept) / returned
        orders += [{"customer_id": customer_id, "amount": mean_kept, "returned": False}] * kept
        orders += [{"customer_id": customer_id, "amount": mean_returned, "returned": True}] * returned
    frame = pd.DataFrame(orders)
    frame.insert(0, "order_id", range(1, len(frame) + 1))
    if null_amounts:
        frame.loc[frame.index[:null_amounts], "amount"] = None
    c_path, o_path = tmp_path / "customers.csv", tmp_path / "orders.csv"
    pd.DataFrame(customers).to_csv(c_path, index=False)
    frame.to_csv(o_path, index=False)
    return c_path, o_path


def _turn(tmp_path, question=QUESTION, answer=ANSWER, code=CODE, rows=None, null_amounts=0):
    c_path, o_path = _data(tmp_path, null_amounts)
    rows = rows or [{"segment": s, "avg_order_amount": kept_mean} for s, _k, kept_mean, _r, _a in SEGMENTS]
    return {"messages": [HumanMessage(content=question), AIMessage(content=answer)],
            "coder_definition": {"code": code}, "output_json": rows, "planner_definition": PLAN,
            "execution_result": {"status": "success", "reads": [
                {"path": str(c_path), "alias": "customers"}, {"path": str(o_path), "alias": "orders"}]}}


def _report(state):
    return claim_verifier_node(state)["verification_report"]


def test_the_rc14_answer_discloses_the_returned_orders_it_left_out(tmp_path):
    report = _report(_turn(tmp_path))
    line = ("Scope: excludes 1,522 returned orders (returned = True), a filter the question did not ask for; "
            "the figures above are computed on the remaining 18,478 of 20,000 orders.")
    shown = presentable_reply(ANSWER, report)
    assert shown == f"{ANSWER}\n\n{line}"
    assert "$545.08" in shown and "$182.27" in shown and "$59.77" in shown


def test_the_rc14_figures_are_scoped_not_plainly_verified(tmp_path):
    report = _report(_turn(tmp_path))
    assert report.get("verified") is False
    assert report.get("scoped") is True
    assert report["scope"]["filters"][0]["rows_dropped"] == 1522
    assert report["scope"]["plan_steps"] == ["Filter Non-Returned Orders"]
    assert any("$545.08" in claim for claim in report["scoped_claims"])


def test_a_filter_the_question_asks_for_adds_no_line(tmp_path):
    question = "What is the average order value by segment, excluding returns?"
    report = _report(_turn(tmp_path, question=question))
    assert not report.get("scoped") and report.get("verified") is True
    assert presentable_reply(ANSWER, report) == ANSWER


def test_a_filter_for_one_segment_the_question_names_adds_no_line(tmp_path):
    code = CODE.replace(FILTER_LINE, "    customers = customers[customers['segment'] == 'consumer']\n")
    answer = "Consumer orders average $59.79."
    report = _report(_turn(tmp_path, question="What is the average order amount for consumers?", answer=answer,
                           code=code, rows=[{"segment": "consumer", "avg_order_amount": 59.79}]))
    assert not report.get("scoped")
    assert presentable_reply(answer, report) == answer


def test_no_filter_adds_no_line(tmp_path):
    answer = ("Enterprise customers spend the most on average, with $545.33 per order, followed by SMBs at "
              "$182.17, while consumer orders average only $59.79.")
    rows = [{"segment": s, "avg_order_amount": all_mean} for s, _k, _m, _r, all_mean in SEGMENTS]
    report = _report(_turn(tmp_path, answer=answer, code=CODE.replace(FILTER_LINE, ""), rows=rows))
    assert not report.get("scoped") and report.get("verified") is True
    assert presentable_reply(answer, report) == answer


def test_an_answer_that_already_says_it_dropped_returns_adds_no_line(tmp_path):
    answer = ("Among non-returned orders, enterprise customers spend the most on average, with $545.08 per order, "
              "followed by SMBs at $182.27, while consumer orders average only $59.77.")
    report = _report(_turn(tmp_path, answer=answer))
    assert not report.get("scoped") and report.get("verified") is True
    assert presentable_reply(answer, report) == answer


def test_a_null_drop_under_five_percent_adds_no_line(tmp_path):
    code = CODE.replace(FILTER_LINE, "    orders = orders.dropna(subset=['amount'])\n")
    report = _report(_turn(tmp_path, code=code, null_amounts=400))       # 2% of 20,000
    assert not report.get("scoped")
    assert presentable_reply(ANSWER, report) == ANSWER


def test_a_null_drop_over_five_percent_is_disclosed(tmp_path):
    code = CODE.replace(FILTER_LINE, "    orders = orders.dropna(subset=['amount'])\n")
    report = _report(_turn(tmp_path, code=code, null_amounts=2000))      # 10% of 20,000
    assert report.get("scoped") is True and report.get("verified") is False
    assert presentable_reply(ANSWER, report).endswith(
        "Scope: excludes 2,000 orders with no value in amount (10.0%); "
        "the figures above are computed on the remaining 18,000 of 20,000 orders.")
