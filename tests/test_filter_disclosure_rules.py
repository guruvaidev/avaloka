"""The row-filter detector (app/agents/filter_disclosure) rule by rule."""

from __future__ import annotations

import pandas as pd
import pytest

from app.agents.claim_verifier import DataUsed
from app.agents.filter_disclosure import (NULL_DISCLOSE_SHARE, asked_for, detect_row_filters, disclosed_in,
                                          scope_filters)


def _orders(n=100):
    return pd.DataFrame({"order_id": range(n), "status": ["returned" if i % 10 == 0 else "shipped" for i in range(n)],
                         "amount": [float(i) for i in range(n)], "region": ["EU" if i % 2 else "US" for i in range(n)]})


def _used(frame=None):
    return DataUsed(frames=[("orders.csv", _orders() if frame is None else frame)], sources=["/x/orders.csv"])


def _one(code, used=None):
    filters = detect_row_filters(code, used if used is not None else _used())
    assert len(filters) == 1, [f.as_dict() for f in filters]
    return filters[0]


@pytest.mark.parametrize("line", [
    "orders = orders[orders['status'] != 'returned']",
    "orders = orders.loc[orders['status'] != 'returned', ['amount', 'region']]",
    "orders = orders.query(\"status != 'returned'\")",
    "orders = orders.drop(orders[orders['status'] == 'returned'].index)",
    "keep = orders.status != 'returned'\n    orders = orders[keep]",
])
def test_each_form_of_a_row_filter_is_found_and_counted(line):
    code = f"import pandas as pd\ndef main(orders):\n    {line}\n    return orders.groupby('region')['amount'].mean()\n"
    flt = _one(code)
    assert (flt.rows_before, flt.rows_dropped) == (100, 10)
    assert not flt.null_only


def test_the_dropped_rows_are_described():
    flt = _one("def main(orders):\n    orders = orders[orders['status'] != 'returned']\n    return orders.amount.mean()\n")
    assert flt.excluded == "status = returned"
    assert flt.line() == ("Scope: excludes 10 orders (status = returned), a filter the question did not ask for; "
                          "the figures above are computed on the remaining 90 of 100 orders.")


def test_sql_where_in_the_code_is_a_filter():
    code = ("import pandas as pd\ndef main(_):\n    q = \"SELECT region, AVG(amount) AS a FROM orders "
            "WHERE status <> 'returned' GROUP BY region\"\n    return pd.read_sql(q, conn)\n")
    flt = _one(code, DataUsed())
    assert flt.kind == "sql_where" and flt.condition == "status <> 'returned'"
    assert "filtered subset" in flt.line()
    assert detect_row_filters(code.replace("status <> 'returned'", "amount IS NOT NULL"), DataUsed())[0].null_only


def test_a_head_before_an_aggregate_is_a_subset_and_after_one_is_not():
    before = "def main(orders):\n    first = orders.head(20)\n    return first.groupby('region')['amount'].mean()\n"
    flt = _one(before)
    assert flt.kind == "subset" and flt.rows_dropped == 80 and flt.reportable()
    after = "def main(orders):\n    out = orders.groupby('region')['amount'].mean()\n    return out.nlargest(1)\n"
    assert detect_row_filters(after, _used()) == []
    listing = "def main(orders):\n    return orders.sort_values('amount').head(5)\n"
    assert not any(f.reportable() for f in detect_row_filters(listing, _used()))


def test_a_filter_whose_rows_never_reach_the_result_is_ignored():
    code = ("def main(orders):\n    eu = orders[orders['region'] == 'EU']\n    print(len(eu))\n"
            "    return orders.groupby('status')['amount'].mean()\n")
    assert detect_row_filters(code, _used()) == []


def test_null_handling_is_disclosed_only_above_the_threshold():
    assert NULL_DISCLOSE_SHARE == 0.05
    frame = _orders()
    frame.loc[:3, "amount"] = None                     # 4%
    code = "def main(orders):\n    orders = orders[orders['amount'].notna()]\n    return orders.amount.mean()\n"
    flt = _one(code, _used(frame))
    assert flt.null_only and flt.rows_dropped == 4 and not flt.reportable()
    frame.loc[:5, "amount"] = None                     # 6%
    flt = _one(code, _used(frame))
    assert flt.rows_dropped == 6 and flt.reportable()


def test_an_inplace_dropna_on_rows_is_found_and_on_columns_is_not():
    frame = _orders()
    frame.loc[:9, "region"] = None
    rows = "def main(orders):\n    orders.dropna(subset=['region'], inplace=True)\n    return orders.amount.sum()\n"
    assert _one(rows, _used(frame)).rows_dropped == 10
    cols = "def main(orders):\n    orders = orders.dropna(axis=1)\n    return orders.amount.sum()\n"
    assert detect_row_filters(cols, _used(frame)) == []


@pytest.mark.parametrize("question, asked", [
    ("What is the average amount excluding returns?", True),
    ("Average amount of returned orders", True),
    ("What is the average amount by region?", False),
    ("How many orders are there?", False),
])
def test_asked_for_matches_the_question_words(question, asked):
    flt = _one("def main(orders):\n    orders = orders[orders['status'] != 'returned']\n    return orders.amount.mean()\n")
    assert asked_for(flt, question) is asked


def test_a_comparison_is_asked_for_by_its_column_and_a_comparison_word_or_its_number():
    flt = _one("def main(orders):\n    orders = orders[orders['amount'] > 50]\n    return orders.region.value_counts()\n")
    assert asked_for(flt, "Which region has the most orders with an amount over 50?")
    assert asked_for(flt, "Region counts of big orders (50+)")
    assert not asked_for(flt, "Which region has the most orders?")


def test_an_answer_that_names_the_exclusion_has_disclosed_it():
    flt = _one("def main(orders):\n    orders = orders[orders['status'] != 'returned']\n    return orders.amount.mean()\n")
    assert disclosed_in(flt, "Excluding returned orders, the average is 54.4.")
    assert not disclosed_in(flt, "The average order amount is 54.4.")


def test_scope_filters_on_unparseable_code_is_empty():
    assert scope_filters("def main(:\n", "q", "a") == []


def test_a_threshold_at_zero_is_not_a_flag():
    """rweb6 messy_xlsx: ``Quantity > 0`` was read as a 0/1 flag, and an answer
    naming "quantity" counted as having disclosed it."""
    flt = _one("def main(orders):\n    orders = orders[orders['amount'] > 0]\n    return orders.amount.sum()\n")
    assert flt.flags == [] and flt.excluded == "amount <= 0"
    assert not disclosed_in(flt, "The total amount sold is 4,950.")


def test_rows_that_are_only_counted_are_not_a_population():
    """A share ``len(hit) / len(df)``: the filter is the numerator."""
    code = ("import pandas as pd\ndef main(orders):\n    hit = orders[orders['status'] == 'returned']\n"
            "    return pd.DataFrame({'share': [len(hit) / len(orders)]})\n")
    assert detect_row_filters(code, _used()) == []


def test_a_filter_on_model_coefficients_is_not_a_row_filter():
    code = ("import pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
            "def main(df):\n    X = df[['amount']]\n    model = LogisticRegression().fit(X, df['region'])\n"
            "    coefs = pd.Series(model.coef_[0], index=X.columns)\n"
            "    return coefs[coefs.index.str.startswith('am')].sum()\n")
    assert detect_row_filters(code, _used()) == []
