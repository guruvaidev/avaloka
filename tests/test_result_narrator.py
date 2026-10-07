"""The chat answer beside a result table: app/agents/result_narrator.py (PR #420).

PR #420 rewrote the narrator: a new system prompt (summarise, do not recite),
reasoning-effort and token tuning with a retry when a reasoning model returns
nothing, a second LLM call when the reply recites the table, conditional
data-quality rules, and a post-filter that deletes data-quality bullets the user
did not ask for. The module had no test file before the PR and none after it.

Plain tests pin that control flow. ``defect`` tests assert intended behaviour
against defects confirmed by running the merged code (73a67a48); they are
xfail(strict=True) and must be removed when the defect is.

No LLM: a scripted fake replaces ``narrator_llm``. That means nothing here says
anything about the QUALITY of a real model's reply -- only about what the
narrator sends, when it calls again, and what it does to the text it gets back.
"""

from __future__ import annotations

import base64
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents import result_narrator as rn


class _Reply:
    def __init__(self, content, finish_reason="stop"):
        self.content = content
        self.response_metadata = {"finish_reason": finish_reason}


class _ScriptedLLM:
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
def narrator(monkeypatch):
    def install(*outcomes, reasoning=True):
        llm = _ScriptedLLM(*outcomes)
        monkeypatch.setattr(rn, "narrator_llm", llm)
        monkeypatch.setattr(rn, "_IS_REASONING", reasoning)
        return llm
    return install


ROWS = [{"region": r, "sales": s} for r, s in
        (("East", 120), ("West", 90), ("North", 45), ("South", 30), ("Central", 12))]
RECITAL = "\n".join(f"- {r['region']}: {r['sales']} units, {i} orders, rank {i}, {i * 3} returns"
                    for i, r in enumerate(ROWS, start=1))
SUMMARY = "East leads on sales.\n- East (120) is ten times Central (12)."


def _system(call) -> str:
    return call[0][0].content


def _human(call) -> str:
    return call[0][1].content


# ---------------------------------------------------------------------------
# What is sent
# ---------------------------------------------------------------------------


def test_a_reasoning_model_is_called_with_a_budget_and_low_effort(narrator):
    llm = narrator(_Reply(SUMMARY))

    assert rn._narrate("total sales by region", ROWS) == SUMMARY
    assert llm.calls[0][1] == {"max_tokens": rn._NARRATOR_MAX_TOKENS, "reasoning_effort": "low"}


def test_a_non_reasoning_model_is_not_sent_a_reasoning_effort(narrator):
    llm = narrator(_Reply(SUMMARY), reasoning=False)

    rn._narrate("total sales by region", ROWS)

    assert llm.calls[0][1] == {"max_tokens": rn._NARRATOR_MAX_TOKENS}


def test_the_prompt_carries_the_question_the_true_row_count_and_the_columns(narrator):
    llm = narrator(_Reply(SUMMARY))
    rows = [{"k": i, "v": i * 2} for i in range(rn.MAX_ROWS + 40)]

    rn._narrate("show v by k", rows)

    human = _human(llm.calls[0])
    assert "User request:\nshow v by k" in human
    assert f"{len(rows):,} rows, first {rn.MAX_ROWS} shown" in human
    assert "columns: ['k', 'v']" in human
    sent = json.loads(human.split("):\n", 1)[1])
    assert len(sent) == rn.MAX_ROWS


def test_empty_cells_are_left_out_of_the_prompt():
    rows = [{"col": "fare", "mean": 12.5, "top": None, "mode": "nan", "n": 0, "flag": False},
            "not a row"]
    assert rn._compact_rows(rows) == [{"col": "fare", "mean": 12.5, "n": 0, "flag": False}]


# ---------------------------------------------------------------------------
# Data-quality rules are conditional
# ---------------------------------------------------------------------------


def test_an_ordinary_question_forbids_data_quality_talk(narrator):
    llm = narrator(_Reply(SUMMARY))
    rn._narrate("total sales by region", ROWS)
    assert rn._NO_DQ_RULE in _system(llm.calls[0])
    assert rn._DQ_RULES not in _system(llm.calls[0])


@pytest.mark.parametrize("question", [
    "are there missing values in sales?",
    "check data quality",
    "find duplicates and outliers",
])
def test_a_data_quality_question_gets_the_data_quality_rules(narrator, question):
    llm = narrator(_Reply(SUMMARY))
    rn._narrate(question, ROWS)
    assert rn._DQ_RULES in _system(llm.calls[0])
    assert rn._NO_DQ_RULE not in _system(llm.calls[0])


def test_a_result_that_reports_issues_gets_the_data_quality_rules(narrator):
    llm = narrator(_Reply(SUMMARY))
    rn._narrate("profile the table", [{"col": "a", "issues": ""}, {"col": "b", "issues": "3 nulls"}])
    assert rn._DQ_RULES in _system(llm.calls[0])


def test_an_issues_column_that_is_entirely_empty_does_not_count():
    rows = [{"col": "a", "issues": ""}, {"col": "b", "issues": None}, {"col": "c", "issues": "NaN"}]
    assert rn._wants_data_quality("profile the table", rows) is False


def test_unrequested_data_quality_bullets_are_removed_from_the_reply(narrator):
    narrator(_Reply("East leads.\n- East has 120.\n- No data-quality issues were reported."))
    assert rn._narrate("total sales by region", ROWS) == "East leads.\n- East has 120."


def test_requested_data_quality_bullets_are_kept(narrator):
    text = "Two columns have gaps.\n- sales has 3 missing values."
    narrator(_Reply(text))
    assert rn._narrate("which columns have missing values?", ROWS) == text


def test_stripping_never_leaves_an_empty_reply(narrator):
    text = "- No issues were found.\n- No missing values."
    narrator(_Reply(text))
    assert rn._narrate("total sales by region", ROWS) == text


# ---------------------------------------------------------------------------
# Retry when the reasoning budget ate the answer
# ---------------------------------------------------------------------------


def test_an_empty_answer_at_the_lowest_effort_is_retried_once_with_double_the_budget(narrator):
    llm = narrator(_Reply("", finish_reason="length"), _Reply(SUMMARY))

    assert rn._narrate("total sales by region", ROWS) == SUMMARY
    assert [kwargs["max_tokens"] for _, kwargs in llm.calls] == [
        rn._NARRATOR_MAX_TOKENS, rn._NARRATOR_MAX_TOKENS * 2,
    ]


def test_an_empty_answer_that_was_not_truncated_is_not_retried(narrator):
    llm = narrator(_Reply("", finish_reason="stop"))

    assert rn._narrate("total sales by region", ROWS) is None
    assert len(llm.calls) == 1


def test_rejected_tuning_kwargs_fall_back_to_a_plain_call(narrator):
    llm = narrator(TypeError("unexpected keyword 'reasoning_effort'"), _Reply(SUMMARY))

    assert rn._narrate("total sales by region", ROWS) == SUMMARY
    assert llm.calls[1][1] == {}


# ---------------------------------------------------------------------------
# The "you recited the table" second call
# ---------------------------------------------------------------------------


def test_a_reply_that_recites_the_table_is_sent_back_for_a_summary(narrator):
    llm = narrator(_Reply(RECITAL), _Reply(SUMMARY))

    assert rn._narrate("total sales by region", ROWS) == SUMMARY

    rewrite = llm.calls[1][0]
    assert isinstance(rewrite[-2], AIMessage) and rewrite[-2].content == RECITAL
    assert isinstance(rewrite[-1], HumanMessage) and rewrite[-1].content == rn._REWRITE_NOTE
    assert llm.calls[1][1]["reasoning_effort"] == "low"


def test_an_empty_rewrite_keeps_the_original_reply(narrator):
    narrator(_Reply(RECITAL), _Reply(""))
    assert rn._narrate("total sales by region", ROWS) == RECITAL


def test_a_tiny_result_is_never_sent_back_however_many_numbers_it_quotes(narrator):
    llm = narrator(_Reply(RECITAL))

    assert rn._narrate("total sales by region", ROWS[:3]) == RECITAL
    assert len(llm.calls) == 1


@pytest.mark.parametrize("text,rows,expected", [
    ("East leads with 120.", 5, False),
    ("\n".join(f"- point {i}" for i in range(7)), 5, True),      # 7 bullets > 6
    (" ".join(str(n) for n in range(100, 113)), 5, True),        # 13 numbers > 12
    (" ".join(str(n) for n in range(100, 113)), 3, False),       # tiny table
])
def test_too_detailed_thresholds(text, rows, expected):
    assert rn._too_detailed(text, rows) is expected


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


def _state(rows_csv: str, question: str = "total sales by region") -> dict:
    uri = "data:text/csv;base64," + base64.b64encode(rows_csv.encode()).decode()
    return {
        "messages": [HumanMessage(content=question)],
        "output_file_data": {"content": uri},
        "execution_result": {"status": "success"},
    }


_CSV = "region,sales\n" + "\n".join(f"{r['region']},{r['sales']}" for r in ROWS) + "\n"


def test_the_node_appends_the_narrated_answer(narrator):
    narrator(_Reply(SUMMARY))

    out = rn.result_narrator_node(_state(_CSV))

    assert isinstance(out["messages"][-1], AIMessage)
    assert out["messages"][-1].content == SUMMARY


def test_the_node_falls_back_to_a_table_description_without_a_model(monkeypatch):
    monkeypatch.setattr(rn, "narrator_llm", None)

    out = rn.result_narrator_node(_state(_CSV))

    assert out["messages"][-1].content == rn._fallback(rn._result_rows(_state(_CSV)))
    assert "region" in out["messages"][-1].content


def test_the_node_falls_back_when_the_model_raises_on_both_calls(narrator):
    narrator(RuntimeError("provider 500"), RuntimeError("provider 500"))

    out = rn.result_narrator_node(_state(_CSV))

    assert out["messages"][-1].content == rn._fallback(rn._result_rows(_state(_CSV)))


# ---------------------------------------------------------------------------
# Confirmed defects
# ---------------------------------------------------------------------------


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT PR420-N1: the optional 'rewrite as a summary' call is not "
        "guarded. If it raises (timeout, 429, 500) the exception leaves "
        "_narrate, result_narrator_node catches it, and the user gets the generic "
        "fallback text -- although a complete, accurate first answer was already "
        "in hand. An optional polish step must not cost the answer. Remove this "
        "xfail when fixed."
    ),
)
def test_a_failed_rewrite_keeps_the_answer_already_in_hand(narrator):
    narrator(_Reply(RECITAL), TimeoutError("read timed out"), TimeoutError("read timed out"))

    out = rn.result_narrator_node(_state(_CSV))

    assert out["messages"][-1].content == RECITAL


@pytest.mark.parametrize("label", ["None", "none", "null", "NULL"])
def test_a_category_literally_named_none_keeps_its_label(label):
    """Was DEFECT PR420-N2. _compact_rows dropped any cell whose text was
    'none' or 'null', and the prompt tells the model 'a cell left out of a row
    was empty'. A group whose LABEL is one of those words -- payment_method =
    'None' is a real category -- lost its label, so the model saw {'n': 5} and
    could not name the group. Those two words are now kept.

    The original reason also listed 'nan' and 'nat'. Those are deliberately
    still dropped (see the next test, and test_empty_cells_are_left_out_of_the_
    prompt): they are how pandas prints a null, and every cell here is a string,
    so a label spelled 'nan' cannot be told from a stringified null.
    """
    rows = [{"payment_method": label, "n": 5}, {"payment_method": "Card", "n": 9}]
    assert rn._compact_rows(rows)[0] == {"payment_method": label, "n": 5}


@pytest.mark.parametrize("cell", ["", "  ", "nan", "NaN", "NaT", "<NA>", None])
def test_cells_that_are_how_a_null_is_printed_are_still_left_out(cell):
    """The accepted residual of the N2 fix: a category literally named 'nan' is
    still dropped. Pinned so nobody has to rediscover it."""
    assert rn._compact_rows([{"payment_method": cell, "n": 5}]) == [{"n": 5}]


@pytest.mark.parametrize("cell", ["None", "null", "nan", ""])
def test_an_issues_cell_that_says_none_is_still_not_a_data_quality_finding(cell):
    """_wants_data_quality reads an `issues` column, where 'None' means there is
    no issue. Keeping 'None' as a category label must not turn it into one."""
    rows = [{"column": "fare", "issues": cell}]
    assert rn._wants_data_quality("average fare by borough", rows) is False


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT PR420-N3: _DQ_REQUEST_RE matches the bare words 'quality', "
        "'errors', 'issues', 'problems'. 'average product quality score by "
        "region' and 'count of error codes per service' are ordinary analytical "
        "questions about columns with those names, yet they switch the prompt to "
        "the data-quality rules, which instruct the model to say 'no dedicated "
        "data-quality checks were run' -- the unrequested filler the PR set out to "
        "remove. Remove this xfail when fixed."
    ),
)
@pytest.mark.parametrize("question", [
    "average product quality score by region",
    "count of error codes per service",
])
def test_a_column_named_quality_or_error_is_not_a_data_quality_request(question):
    assert rn._wants_data_quality(question, ROWS) is False


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT PR420-N4: _strip_unrequested_quality deletes any bullet matching "
        "_DQ_LINE_RE, which includes 'issues were' and 'missing values'. For a "
        "support-ticket table the finding '- Agent A: 40 issues were closed' IS "
        "the answer, and it is silently removed from the reply. The filter cannot "
        "tell a finding from filler. Remove this xfail when fixed."
    ),
)
def test_a_finding_that_happens_to_mention_issues_is_not_deleted():
    reply = "Agent A leads.\n- Agent A: 40 issues were closed\n- Agent B closed 12"
    assert rn._strip_unrequested_quality(reply) == reply


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT PR420-N5 (response time): _NUMBER_RE counts each part of an ISO "
        "date as a number, so '2024-03-01' is three. Four dates alone are 12 -- "
        "exactly the limit; with the two values the summary also quotes it is 14, "
        "over the 12-number limit, and triggers a second, full LLM call on a "
        "time-series answer -- extra latency from the PR that exists to cut it. "
        "Remove this xfail when fixed."
    ),
)
def test_dates_are_not_counted_as_recited_table_values():
    reply = (
        "Sales peaked on 2024-03-01 and bottomed on 2024-06-01.\n"
        "- 2023-01-01 opened at 5.\n"
        "- 2023-02-01 rose to 6."
    )
    assert rn._too_detailed(reply, 12) is False
