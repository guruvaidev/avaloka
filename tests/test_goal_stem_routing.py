"""The app reads "returns" / "return" as the column ``returned``.

README check (rc3 web) on ``orders``: "What drives returns?" came back "I've
held back my answer", and "What is the return rate by channel?" ran on the
sample because "return" did not match ``returned``, so the database pushdown
never recognised it. The goal->column matching of #542/#554 now folds simple
stems and plurals, whole words only, in the pushdown parser, in the step
narration, and in the intent routing that reuses driver_target (#546).
"""

import pytest
from langchain_core.messages import HumanMessage

from app.api import chat_voice
from app.core.db_pushdown import AggregateIntent, parse_aggregate_intent

ORDERS = ["order_id", "order_date", "region", "channel", "product", "units",
          "unit_price", "discount_pct", "returned"]


@pytest.mark.parametrize("question", ["What is the return rate by channel?",
                                      "return rate by channels",
                                      "What is the RETURNED rate by Channel?"])
def test_return_rate_by_channel_is_the_rate_of_returned_by_channel(question):
    assert parse_aggregate_intent(question, ORDERS) == AggregateIntent("rate", "returned", "channel")


@pytest.mark.parametrize("question", ["What is the turn rate by channel?", "turnover rate by channel"])
def test_turn_is_not_returned(question):
    assert parse_aggregate_intent(question, ORDERS) is None


def test_the_pushdown_still_declines_a_filtered_question():
    assert parse_aggregate_intent("return rate by channel where units > 3", ORDERS) is None


@pytest.mark.parametrize("goal", ["What drives returns?", "what drives returned"])
def test_what_drives_returns_routes_to_analysis(goal):
    from app.agents.avaloka_agent.intent_classifier import asks_what_drives, classify_intent
    assert asks_what_drives(goal)
    result = classify_intent([HumanMessage(content=goal)])
    assert result.intent == "statistical_analysis", result


def _orders_stats():
    def cat(values):
        return {"type": "categorical", "n_unique": len(values), "missing_ratio": 0.0,
                "top_values": [{"value": v, "count": c} for v, c in values.items()]}
    columns = {
        "order_id": {"type": "numeric", "n_unique": 2000, "missing_ratio": 0.0},
        "region": cat({"North": 500, "South": 500, "East": 500, "West": 500}),
        "channel": cat({"web": 900, "store": 700, "partner": 400}),
        "product": cat({"A": 700, "B": 700, "C": 600}),
        "units": {"type": "numeric", "n_unique": 20, "missing_ratio": 0.0, "mean": 3.1, "min": 1, "max": 20},
        "discount_pct": {"type": "numeric", "n_unique": 30, "missing_ratio": 0.0, "mean": 5, "min": 0, "max": 30},
        "returned": {"type": "numeric", "n_unique": 2, "missing_ratio": 0.0, "mean": 0.3, "min": 0, "max": 1},
    }
    return {"data_shape": {"rows": 2000, "columns": len(columns)}, "column_statistics": columns}


def test_the_narration_reads_returns_as_returned():
    final = {"avaloka_intent": "statistical_analysis",
             "execution_result": {"status": "success", "message": "done", "output_row_count": 3},
             "output_json": [{"channel": "partner", "returned_rate": 0.55},
                             {"channel": "web", "returned_rate": 0.25}]}
    prompts = {p["step"]: p for p in chat_voice.prompts_after_turn(
        final, question="What drives returns?", sample_statistics=_orders_stats())}
    assert "`returned`" in prompts["understanding"]["message"], prompts["understanding"]
    labels = [r["label"] for r in prompts["result"]["quick_replies"]]
    assert "Train a model on returned" in labels, labels


def test_the_narration_does_not_read_a_column_inside_another_word():
    # "average" is not ``age``; "turn" is not ``returned``.
    assert chat_voice._columns_in("average turn", ["age", "returned"]) == []
    assert chat_voice._columns_in("Average Age by Admission Type", ["Age", "Admission Type"]) == [
        "Admission Type", "Age"]


# ---------------------------------------------------------- one rule, both paths
# The DB pushdown parser (#559), the step narration and the CLI's
# driver_target all match through avaloka.goal_analysis.word_stems. These
# pin that they agree, so a second, different stem rule cannot creep back.
_PROFILE = {"columns": [{"name": c, "role": "boolean" if c == "returned" else "categorical",
                         "n_unique": 2 if c == "returned" else 4} for c in ORDERS]}


def _paths(question):
    from avaloka.goal_analysis import column_mentions, driver_target
    intent = parse_aggregate_intent(question, ORDERS)
    target = driver_target(question, _PROFILE)
    return {
        "pushdown": intent.measure if intent else None,
        "narration": "returned" if "returned" in chat_voice._columns_in(question, ORDERS) else None,
        "cli_mentions": "returned" if "returned" in {c for _s, _e, c in column_mentions(question, ORDERS)} else None,
        "cli_driver": (target or {}).get("column"),
    }


def test_return_rate_by_channel_is_returned_on_every_path():
    paths = _paths("What is the return rate by channel?")
    assert paths["pushdown"] == paths["narration"] == paths["cli_mentions"] == "returned", paths


def test_what_drives_returns_is_returned_on_every_path():
    paths = _paths("What drives returns?")
    assert paths["narration"] == paths["cli_mentions"] == paths["cli_driver"] == "returned", paths


@pytest.mark.parametrize("question", ["What is the turn rate by channel?", "What drives turn?"])
def test_turn_is_returned_on_no_path(question):
    assert set(_paths(question).values()) == {None}, _paths(question)


@pytest.mark.parametrize("col,word,same", [
    ("returned", "return", True), ("churned", "churn", True),
    ("used", "us", False), ("rated", "rate", False),       # #559: the verb is 4+ letters
    ("returned", "turn", False),
])
def test_the_ed_rule_is_559s(col, word, same):
    from avaloka.goal_analysis import word_stems
    assert bool(word_stems(col) & word_stems(word)) is same
    hit = parse_aggregate_intent(f"{word} rate by channel", [col, "channel"])
    assert (hit is not None and hit.measure == col) is same
