"""How much the membership test over-strips, measured on a synthetic set.

This is NOT real analyst data -- none exists yet. It is thirty questions written
to resemble what analysts ask of a retail-orders dataset, against a schema with
deliberately ordinary column names. The numbers quoted in
docs/analytics/DESIGN.md come from this file, and the assertions fail if the
behaviour moves away from what the document says.
"""
from app.analytics.names import strip_names

SCHEMA = ["orders", "order_id", "order_date", "customer_id", "customer_name", "region", "status",
          "product", "category", "quantity", "unit_price", "discount", "total", "sales_rep",
          "ship_date", "channel"]

QUESTIONS = [
    "what is the average total by region",
    "show me the top 10 customers by revenue this year",
    "how many orders were cancelled last month",
    "plot total over time by month",
    "which category has the highest discount",
    "why did sales drop in March",
    "what is the status of the pipeline",
    "compare unit_price across category",
    "show the trend of quantity for the last 12 weeks",
    "which sales rep closed the most orders",
    "average order value per customer",
    "is there a correlation between discount and quantity",
    "break down total by channel and region",
    "how many customers ordered more than once",
    "show a histogram of unit_price",
    "what share of orders shipped late",
    "top 5 products by total in the west region",
    "count of orders by status",
    "what changed week over week",
    "forecast next quarter revenue",
    "which regions are growing fastest",
    "list customers with no orders in 90 days",
    "median days between order_date and ship_date",
    "can you make that a bar chart instead",
    "summarise the data for me",
    "are there any outliers in quantity",
    "group by customer_name and sum total",
    "what does the distribution look like",
    "filter to status shipped and redo it",
    "which product category is most profitable",
]


def measure():
    tokens = removed = questions_touched = 0
    kinds = {"exact": 0, "common_word": 0, "substring": 0}
    for question in QUESTIONS:
        _, hits = strip_names(question, SCHEMA)
        n = sum(hits.values())
        tokens += len(question.split())
        removed += n
        questions_touched += bool(n)
        for kind in kinds:
            kinds[kind] += hits[kind]
    return tokens, removed, questions_touched, kinds


def test_documented_overstrip_figures_still_hold():
    tokens, removed, touched, kinds = measure()
    share_removed = removed / tokens
    share_common = kinds["common_word"] / removed
    print(f"\ntokens={tokens} removed={removed} ({share_removed:.0%}) questions_touched={touched}/{len(QUESTIONS)} "
          f"kinds={kinds} common_word_share={share_common:.0%}")
    # Measured when written: 35 of 194 words (18%), 21 of 30 questions, 21 of 35
    # removals (60%) classified common_word.
    assert 0.15 <= share_removed <= 0.22          # DESIGN.md: "about one word in five"
    assert 19 <= touched <= 23                    # DESIGN.md: "seven questions in ten"
    assert 0.50 <= share_common <= 0.70           # DESIGN.md: "about 60% are plain English words"
    assert sum(kinds.values()) == removed


def test_a_question_with_no_schema_words_is_untouched():
    assert strip_names("what changed week over week", SCHEMA) == (
        "what changed week over week", {"exact": 0, "common_word": 0, "substring": 0})
