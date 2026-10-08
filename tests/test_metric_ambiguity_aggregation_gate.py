"""The metric-ambiguity stem-collision rule and the planner retry loop.

PR #420 narrowed ``_detect_metric_ambiguity`` from "ask whenever a prompt word is
shared by two qualified columns" to "ask only when the user AGGREGATES that word
and two or more NUMERIC columns share it". This file was added in #432 to pin
what that gate did, so the narrowing could be ruled on against facts.

The ruling: the aggregation-word condition is gone. With gross_revenue and
net_revenue present, "What was revenue last month?" silently picked one, and
that is worse than asking. The gate now asks exactly when two or more numeric
columns collide on a stem the user typed, whether or not an aggregation word
precedes it. A stem that directly follows per / for each / across is a
grouping, not a measure, and is skipped (the owner's second ruling); so is one
after "by", once a column word has been named before it. The cases here that asserted "no aggregation word, so no question"
were rewritten to assert that rule; the file keeps its name so its history
stays readable.

Kept from #420, and pinned below: only numeric columns are options, the
``AVALOKA_METRIC_CLARIFICATION`` off switch, the rule that a long analytical
reply replaces a pending question, and ``_invoke_planner_with_recovery``.

No LLM: the detector is deterministic, and the retry loop is driven by a
scripted fake installed over ``planner.llm``.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents import planner

REVENUE = {"uploaded_csv_columns": ["gross_revenue", "net_revenue", "month"]}


def _measure(prompt: str, state: dict = REVENUE):
    details = planner._detect_ambiguous_prompt_details(prompt, state)
    return details and details["pending_clarification"]["measure"]


# ---------------------------------------------------------------------------
# What the gate asks about
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", [
    "average revenue by region",
    "What was the total revenue last month?",
    "sum of the revenue",
    "total monthly revenue",
    "distribution of revenue",
    "highest revenue month",
    "Average Revenue per month for the last 3 months",
])
def test_an_aggregated_shared_measure_asks_which_column(prompt):
    details = planner._detect_ambiguous_prompt_details(prompt, REVENUE)

    assert details is not None, f"{prompt!r} aggregates an ambiguous measure and did not ask"
    pending = details["pending_clarification"]
    assert pending["type"] == "metric_choice"
    assert pending["measure"] == "revenue"
    assert pending["columns"] == ["gross_revenue", "net_revenue"]
    assert details["message"] == "Which revenue measure do you mean: gross revenue or net revenue?"


@pytest.mark.parametrize("prompt", [
    "average gross revenue by region",      # the qualifier is already named
    "show me avg net revenue",
    "What was net revenue last month?",
    "average of all revenue",
    "average units per revenue band",       # revenue is grouped on, not measured
    "average month",                        # not shared by two columns
    "How many rows are there?",             # names no shared stem at all
])
def test_an_already_qualified_or_unshared_measure_does_not_ask(prompt):
    assert _measure(prompt) is None


@pytest.mark.parametrize("prompt", [
    "What was revenue last month?",         # the prompt #420 answered by picking one
    "revenue by region",
    "Why did revenue fall?",
    "plot revenue over time",
    "average revenue per revenue band",     # measured as well as grouped on
])
def test_a_shared_stem_asks_without_an_aggregation_word(prompt):
    details = planner._detect_ambiguous_prompt_details(prompt, REVENUE)

    assert details is not None, f"{prompt!r} names a stem two columns share and did not ask"
    assert details["pending_clarification"]["measure"] == "revenue"
    assert details["pending_clarification"]["columns"] == ["gross_revenue", "net_revenue"]


CHANNEL_COLUMNS = ["channel_type", "channel_rank", "subscribers_total", "subscribers_30d"]
# Three views of the same four columns. In the first two BOTH channel columns
# are possible measures, so only the grouping rule can keep the question on
# subscribers. In the third the numeric-only rule would do it anyway, which is
# why that state alone proves nothing about grouping.
CHANNEL_STATES = {
    "untyped": {"uploaded_csv_columns": CHANNEL_COLUMNS},
    "all numeric": {"uploaded_csv_columns": CHANNEL_COLUMNS,
                    "schema": dict.fromkeys(CHANNEL_COLUMNS, "int64")},
    "channel_type is text": {"uploaded_csv_columns": CHANNEL_COLUMNS,
                             "schema": {**dict.fromkeys(CHANNEL_COLUMNS, "int64"),
                                        "channel_type": "object"}},
}


@pytest.mark.parametrize("typing", list(CHANNEL_STATES))
@pytest.mark.parametrize("prompt", [
    "average subscribers per channel",
    "subscribers per channel",
    "per channel, average subscribers",     # the grouping comes first
    "subscribers by channel",
    "show me subscribers grouped by channel",
    "for each channel show subscribers",    # via the older "each X" rule
    "subscribers across channels",
])
def test_a_stem_after_a_grouping_preposition_is_not_the_measure(prompt, typing):
    details = planner._detect_ambiguous_prompt_details(prompt, CHANNEL_STATES[typing])

    assert details is not None
    assert details["pending_clarification"]["measure"] == "subscribers"
    assert details["pending_clarification"]["columns"] == ["subscribers_total", "subscribers_30d"]


@pytest.mark.parametrize("prompt", [
    "count rows per channel",
    "uploads by channel",
    "show the top five for each channel",
    "compare uploads across channels",
])
def test_a_grouping_alone_never_asks(prompt):
    """Both channel columns are numeric here and nothing else collides."""
    state = {"uploaded_csv_columns": ["channel_type", "channel_rank", "uploads"]}
    assert _measure(prompt, state) is None


@pytest.mark.parametrize("typing", ["untyped", "all numeric"])
def test_known_limitation_a_grouping_named_first_without_a_preposition_is_asked_about(typing):
    """Pinned, not endorsed. The grouping rule keys on per / by / for each /
    across. Here "channel" follows "which", so nothing marks it as a grouping
    and prompt order decides: channel comes first and two numeric columns share
    it. Knowing that subscribers is the thing being measured needs a rule about
    what is aggregated, which this gate deliberately no longer has."""
    details = planner._detect_ambiguous_prompt_details(
        "which channel has the most subscribers", CHANNEL_STATES[typing],
    )

    assert details["pending_clarification"]["measure"] == "channel"
    assert details["pending_clarification"]["columns"] == ["channel_type", "channel_rank"]
    # The measure is reported as the user typed it.
    assert _measure("channels ranked on subscribers", CHANNEL_STATES[typing]) == "channels"
    # "by" with no column word before it is read as naming the thing itself,
    # which is right for "sort by revenue" and wrong for these two.
    assert _measure("by channel, average subscribers", CHANNEL_STATES[typing]) == "channel"
    assert _measure("how many videos by channel", CHANNEL_STATES[typing]) == "channel"


def test_the_numeric_only_rule_hides_that_limitation_when_one_channel_column_is_text():
    state = CHANNEL_STATES["channel_type is text"]
    assert _measure("which channel has the most subscribers", state) == "subscribers"
    assert not hasattr(planner, "_measure_is_aggregated")


@pytest.mark.parametrize("prompt", [
    "sort by revenue",
    "order by revenue",
    "rank by revenue",
    "sort the rows by revenue, highest first",
    "show revenue sorted by revenue",
])
def test_by_names_the_measure_when_no_column_word_comes_before_it(prompt):
    """"sort by revenue" over gross_revenue and net_revenue is the ambiguity the
    feature exists for; the grouping rule must not swallow it."""
    assert _measure(prompt) == "revenue"


def test_by_is_a_grouping_once_a_column_word_has_been_named():
    state = {"uploaded_csv_columns": ["units", "revenue_band_low", "revenue_band_high"]}
    assert _measure("units by revenue", state) is None
    assert _measure("sort by revenue", state) == "revenue"
    assert _measure("month by month revenue") == "revenue"


def test_known_limitation_per_also_introduces_a_denominator():
    """Pinned, not endorsed. In "revenue per subscriber" the word after "per"
    is divided by, not grouped on, and two columns could be the divisor. "per"
    skips unconditionally, so this does not ask."""
    state = {"uploaded_csv_columns": ["revenue", "subscribers_total", "subscribers_30d"]}
    assert _measure("average revenue per subscriber", state) is None
    assert _measure("average revenue and subscribers", state) == "subscribers"


# ---------------------------------------------------------------------------
# What counts as a collision
# ---------------------------------------------------------------------------


def _columns(prompt: str, columns: list, schema: dict | None = None):
    state = {"uploaded_csv_columns": columns}
    if schema is not None:
        state["schema"] = schema
    details = planner._detect_ambiguous_prompt_details(prompt, state)
    return details and details["pending_clarification"]["columns"]


@pytest.mark.parametrize("columns", [
    ["revenue", "month"],                   # the bare column
    ["net_revenue", "month"],               # one qualified column
    ["net_revenue", "revenue_band"],        # second match is text, see schema below
    ["net_revenue", "cost_total", "units_sold"],
])
def test_a_single_matching_column_never_asks(columns):
    schema = {"revenue_band": "object"}
    assert _columns("What was revenue last month?", columns, schema) is None
    assert _columns("total revenue", columns, schema) is None


def test_a_three_way_collision_offers_all_three():
    columns = ["gross_revenue", "net_revenue", "recurring_revenue", "month"]
    details = planner._detect_ambiguous_prompt_details(
        "What was revenue last month?", {"uploaded_csv_columns": columns},
    )

    assert details["pending_clarification"]["columns"] == columns[:3]
    assert details["message"] == (
        "Which revenue measure do you mean: gross revenue, net revenue or recurring revenue?"
    )
    pending = details["pending_clarification"]
    assert planner._resolve_pending_clarification(pending, "recurring").endswith(
        "Use the recurring_revenue column for revenue."
    )


def test_a_three_way_collision_drops_the_text_column():
    columns = ["gross_revenue", "net_revenue", "revenue_band"]
    schema = {"gross_revenue": "float64", "net_revenue": "float64", "revenue_band": "object"}
    assert _columns("What was revenue?", columns, schema) == ["gross_revenue", "net_revenue"]


@pytest.mark.parametrize("columns,prompt", [
    (["gross_revenue", "net_revenue"], "show revenue"),             # shared suffix
    (["revenue_2023", "revenue_2024"], "show revenue"),             # shared prefix
    (["total_revenue_usd", "total_revenue_eur"], "show revenue"),   # shared middle
    (["revenue_2023", "net_revenue"], "show revenue"),              # prefix against suffix
    (["GrossRevenue", "netRevenue"], "show revenue"),               # camelCase
    (["GROSS_REVENUE", "Net Revenue"], "show REVENUE"),             # case and spaces
    (["gross_revenues", "net_revenues"], "show revenue"),           # plural columns
    (["gross_revenue", "net_revenue"], "show revenues"),            # plural prompt
    (["entry_fees", "exit_fee"], "show fees"),                      # mixed
])
def test_a_stem_collides_wherever_it_sits_and_however_it_is_written(columns, prompt):
    assert _columns(prompt, columns) == columns


@pytest.mark.parametrize("columns,prompt", [
    (["prevenue_gross", "prevenue_net"], "show revenue"),       # a stem is a whole token
    (["gross_revenue", "net_income"], "show revenue"),          # nothing shared
    (["revenue", "net_revenue"], "show revenue"),               # the bare column was named
    (["revenues", "net_revenue"], "show revenue"),
    (["revenue", "gross_revenue", "net_revenue"], "show revenue"),
    (["gross_revenue", "revenue_gross"], "show revenue"),       # same qualifier, nothing to ask
])
def test_these_are_not_collisions(columns, prompt):
    assert _columns(prompt, columns) is None


@pytest.mark.parametrize("columns,prompt", [
    (["revenue_2023", "revenue_2024"], "show revenue 2024"),
    (["video_views_for_the_last_30_days", "subscribers_for_the_last_30_days"],
     "total views for the last 30 days"),                       # a word only one option has
    (["revenue_last_month", "revenue_last_year"], "What was revenue last month?"),
])
def test_a_word_only_one_option_has_picks_it(columns, prompt):
    assert _columns(prompt, columns) is None


def test_text_columns_sharing_a_word_are_not_options():
    state = {
        "uploaded_csv_columns": ["revenue_band", "revenue_tier"],
        "schema": {"revenue_band": "object", "revenue_tier": "object"},
    }
    assert _measure("average revenue", state) is None


def test_a_schema_given_as_a_json_string_is_read():
    state = {
        "uploaded_csv_columns": ["revenue_band", "revenue_tier"],
        "schema": '{"revenue_band": "object", "revenue_tier": "object"}',
    }
    assert _measure("average revenue", state) is None


def test_one_numeric_and_one_text_column_leave_nothing_to_choose():
    state = {
        "uploaded_csv_columns": ["revenue_usd", "revenue_band"],
        "schema": {"revenue_usd": "float64", "revenue_band": "object"},
    }
    assert _measure("average revenue", state) is None


@pytest.mark.parametrize("value", ["0", "false", "OFF", " no "])
def test_the_gate_can_be_switched_off(monkeypatch, value):
    monkeypatch.setenv("AVALOKA_METRIC_CLARIFICATION", value)
    assert _measure("average revenue by region") is None


def test_the_gate_is_on_by_default(monkeypatch):
    monkeypatch.delenv("AVALOKA_METRIC_CLARIFICATION", raising=False)
    assert _measure("average revenue by region") == "revenue"


def test_grammar_words_inside_column_names_are_never_the_measure():
    state = {"uploaded_csv_columns": ["views_for_the_last_30_days", "likes_for_the_last_30_days"]}
    assert _measure("total for the last 30", state) is None
    # "days" is a noun both columns carry, and the prompt names neither views
    # nor likes, so there really are two totals it could mean.
    assert _measure("total for the last 30 days", state) == "days"
    assert _measure("total views for the last 30 days", state) is None


# ---------------------------------------------------------------------------
# Answering the question
# ---------------------------------------------------------------------------


def _pending():
    return planner._detect_ambiguous_prompt_details(
        "average revenue by region", REVENUE,
    )["pending_clarification"]


@pytest.mark.parametrize("reply,column", [
    ("net", "net_revenue"),
    ("gross revenue please", "gross_revenue"),
])
def test_a_short_reply_picks_the_column(reply, column):
    resolved = planner._resolve_pending_clarification(_pending(), reply)
    assert resolved == f"average revenue by region. Use the {column} column for revenue."


@pytest.mark.parametrize("reply", [
    "show me the average net revenue by region for 2024",
    "average gross revenue by region and by month too",
])
def test_a_full_analytical_request_is_a_new_question_not_a_pick(reply):
    assert planner._resolve_pending_clarification(_pending(), reply) is None


# ---------------------------------------------------------------------------
# _invoke_planner_with_recovery
# ---------------------------------------------------------------------------


class _EmptyToolCall(Exception):
    pass


class _ContextLength(Exception):
    pass


class _ScriptedPlanner:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def scripted(monkeypatch):
    """Install a scripted planner LLM and classify its errors by type, so the
    loop is tested without depending on any provider's error wording."""
    def install(*outcomes):
        llm = _ScriptedPlanner(*outcomes)
        monkeypatch.setattr(planner, "llm", llm)
        return llm

    monkeypatch.setattr(planner, "_salvage_noparam_tool_call", lambda exc: None)
    monkeypatch.setattr(planner, "_is_empty_tool_call", lambda exc: isinstance(exc, _EmptyToolCall))
    monkeypatch.setattr(planner, "_is_context_length_error",
                        lambda exc: isinstance(exc, _ContextLength))
    return install


_MSGS = [
    SystemMessage(content="system"),
    HumanMessage(content="first question"),
    AIMessage(content="first answer"),
    HumanMessage(content="second question"),
]


def test_a_first_attempt_that_answers_is_returned_as_is(scripted):
    llm = scripted("answer")

    assert planner._invoke_planner_with_recovery(_MSGS, "auto", {"reasoning_effort": "low"}) == "answer"
    assert len(llm.calls) == 1
    assert llm.calls[0][1]["tool_choice"] == "auto"
    assert llm.calls[0][1]["reasoning_effort"] == "low"


def test_an_empty_tool_call_is_retried(scripted):
    llm = scripted(_EmptyToolCall(), _EmptyToolCall(), "answer")

    assert planner._invoke_planner_with_recovery(_MSGS, "auto", {}) == "answer"
    assert len(llm.calls) == 3


def test_the_loop_gives_up_after_three_attempts_and_raises_the_last_error(scripted):
    llm = scripted(_EmptyToolCall(), _EmptyToolCall(), _EmptyToolCall("third"), "never reached")

    with pytest.raises(_EmptyToolCall, match="third"):
        planner._invoke_planner_with_recovery(_MSGS, "auto", {})
    assert len(llm.calls) == planner._PLANNER_MAX_ATTEMPTS == 3


def test_high_effort_steps_down_after_an_empty_answer(scripted, monkeypatch):
    monkeypatch.setattr(planner, "PLANNER_SUPPORTS_REASONING", True)
    llm = scripted(_EmptyToolCall(), "answer")

    planner._invoke_planner_with_recovery(_MSGS, "auto", {"reasoning_effort": "high"})

    assert [kwargs["reasoning_effort"] for _, kwargs in llm.calls] == ["high", "medium"]


def test_the_callers_kwargs_are_not_mutated_by_the_step_down(scripted, monkeypatch):
    monkeypatch.setattr(planner, "PLANNER_SUPPORTS_REASONING", True)
    scripted(_EmptyToolCall(), "answer")
    kwargs = {"reasoning_effort": "high"}

    planner._invoke_planner_with_recovery(_MSGS, "auto", kwargs)

    assert kwargs == {"reasoning_effort": "high"}


def test_a_context_length_error_retries_with_system_last_ai_and_last_human(scripted):
    llm = scripted(_ContextLength(), "answer")

    assert planner._invoke_planner_with_recovery(_MSGS, "auto", {}) == "answer"

    retried = llm.calls[1][0]
    assert [type(m).__name__ for m in retried] == ["SystemMessage", "AIMessage", "HumanMessage"]
    assert [m.content for m in retried] == ["system", "first answer", "second question"]


def test_the_prompt_is_trimmed_only_once(scripted):
    llm = scripted(_ContextLength(), _ContextLength("still too long"), "never reached")

    with pytest.raises(_ContextLength, match="still too long"):
        planner._invoke_planner_with_recovery(_MSGS, "auto", {})
    assert len(llm.calls) == 2


def test_an_unrecognised_error_is_raised_at_once_for_the_outer_handler(scripted):
    llm = scripted(RuntimeError("provider 500"), "never reached")

    with pytest.raises(RuntimeError, match="provider 500"):
        planner._invoke_planner_with_recovery(_MSGS, "auto", {})
    assert len(llm.calls) == 1


def test_a_salvageable_no_parameter_tool_call_is_recovered_on_a_retry_too(scripted, monkeypatch):
    """The reason the loop exists: recovery used to run on the first error only."""
    class _BadArgs(Exception):
        pass

    monkeypatch.setattr(
        planner, "_salvage_noparam_tool_call",
        lambda exc: "show_schema" if isinstance(exc, _BadArgs) else None,
    )
    llm = scripted(_EmptyToolCall(), _BadArgs(), "never reached")

    response = planner._invoke_planner_with_recovery(_MSGS, "auto", {})

    assert len(llm.calls) == 2
    assert response.tool_calls[0]["name"] == "show_schema"
    assert response.tool_calls[0]["args"] == {}
