"""Log lines that stay readable, and say which model answered.

Two things drove this. A coder turn logged its whole message list at INFO, and
those messages carry a 100-row markdown preview of the result table, so one
line ran to tens of thousands of characters. And once an agent can fail over to
a second provider, the model that served a request is no longer a constant you
can read off the config -- it has to come from the response.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.log_utils import (describe_messages, describe_response, preview,
                                response_model)

# The shape that started this: a rendered table inside a chat message.
BIG_TABLE = (
    "Transformed data — 190 rows × 6 columns. Showing the first 100.\n\n"
    + "| Country | Youtuber | subscribers |\n|:--|:--|--:|\n"
    + "".join(f"| C{i} | Channel {i} | {i * 1000} |\n" for i in range(100))
)


# --------------------------------------------------------------------------- #
# preview
# --------------------------------------------------------------------------- #

def test_a_short_value_is_left_alone():
    assert preview("hello") == "hello"


def test_a_long_value_is_cut_and_says_so():
    out = preview(BIG_TABLE, limit=60)
    assert len(out) < 120
    assert "chars total" in out, "a truncated line must not look like a small one"
    assert f"{len(BIG_TABLE):,}" in out


def test_preview_keeps_the_head_which_identifies_the_payload():
    assert preview(BIG_TABLE, limit=40).startswith("Transformed data")


def test_newlines_do_not_break_the_line():
    assert "\n" not in preview("a\nb\nc" * 100, limit=20)


def test_none_is_readable():
    assert preview(None) == "None"


def test_non_strings_are_accepted():
    assert "1" in preview({"a": 1})


# --------------------------------------------------------------------------- #
# describe_messages
# --------------------------------------------------------------------------- #

def test_message_shape_is_reported_without_content():
    out = describe_messages([HumanMessage(content=BIG_TABLE),
                             AIMessage(content="ok")])
    assert "2 messages" in out
    assert "HumanMessage" in out and "AIMessage" in out
    # The whole point: none of the table reaches the log.
    assert "Channel 42" not in out
    assert len(out) < 200


def test_total_size_is_visible():
    out = describe_messages([HumanMessage(content="x" * 5000)])
    assert "5,000" in out


def test_empty_and_none_are_handled():
    assert describe_messages([]) == "0 messages"
    assert describe_messages(None) == "0 messages"


def test_non_string_content_does_not_raise():
    assert "1 messages" in describe_messages([SystemMessage(content=[{"a": 1}])])


# --------------------------------------------------------------------------- #
# Which model answered -- the reason this exists
# --------------------------------------------------------------------------- #

def _answer(meta=None, usage=None, content="hi", tool_calls=None):
    m = AIMessage(content=content, tool_calls=tool_calls or [])
    m.response_metadata = meta or {}
    if usage:
        m.usage_metadata = usage
    return m


def test_groq_reports_model_name():
    assert response_model(_answer({"model_name": "openai/gpt-oss-120b"})) == "openai/gpt-oss-120b"


def test_openai_spec_servers_report_model():
    """OpenRouter, vLLM and Ollama send `model`, not `model_name`. Reading only
    the primary provider's key would leave every failover line saying unknown."""
    assert response_model(_answer({"model": "qwen3.8:27b"})) == "qwen3.8:27b"


def test_a_response_without_metadata_is_distinguishable():
    """A stubbed or deterministic reply is worth telling apart from a real one."""
    assert response_model(_answer({})) is None
    assert "unknown-model" in describe_response(_answer({}))


def test_the_description_leads_with_the_model():
    out = describe_response(_answer({"model_name": "openai/gpt-oss-120b"}))
    assert out.startswith("model=openai/gpt-oss-120b")


def test_the_description_carries_effort_finish_and_cost():
    out = describe_response(_answer(
        {"model_name": "openai/gpt-oss-120b", "reasoning_effort": "low",
         "finish_reason": "tool_calls"},
        usage={"input_tokens": 9874, "output_tokens": 82},
    ))
    assert "effort=low" in out
    assert "finish=tool_calls" in out
    assert "9874" in out and "82" in out


def test_tool_calls_are_named_not_dumped():
    out = describe_response(_answer(
        {"model_name": "m"},
        tool_calls=[{"name": "respond_to_user", "args": {"response_text": BIG_TABLE},
                     "id": "1", "type": "tool_call"}],
    ))
    assert "tools=respond_to_user" in out
    assert "Channel 42" not in out, "tool arguments must not reach the log"


def test_a_long_answer_is_measured_not_printed():
    out = describe_response(_answer({"model_name": "m"}, content=BIG_TABLE))
    assert f"content={len(BIG_TABLE):,}c" in out
    assert "Channel 42" not in out


def test_the_whole_line_stays_short():
    out = describe_response(_answer(
        {"model_name": "openai/gpt-oss-120b", "reasoning_effort": "high",
         "finish_reason": "stop", "system_fingerprint": "fp_" + "x" * 200},
        usage={"input_tokens": 9874, "output_tokens": 82}, content=BIG_TABLE,
    ))
    assert len(out) < 200, f"log line is {len(out)} chars: {out}"


def test_no_response_is_not_an_error():
    assert describe_response(None) == "no response"


# --------------------------------------------------------------------------- #
# These lines reach a Windows console
# --------------------------------------------------------------------------- #

def test_output_is_ascii_only():
    """A non-ASCII character in a log line raises UnicodeEncodeError inside the
    logging handler on a cp1252 console -- the log line becomes a crash. Caught
    live: an arrow and an ellipsis did exactly that."""
    samples = [
        preview(BIG_TABLE, limit=40),
        describe_messages([HumanMessage(content=BIG_TABLE)]),
        describe_response(_answer(
            {"model_name": "openai/gpt-oss-120b", "reasoning_effort": "low"},
            usage={"input_tokens": 1, "output_tokens": 2})),
    ]
    for s in samples:
        s.encode("cp1252")   # raises if any character is unencodable
        assert s.isascii(), f"non-ASCII in log output: {s!r}"


def test_unicode_in_the_payload_never_reaches_the_line():
    """The payload itself may hold em-dashes and smart quotes -- those must be
    measured, not echoed."""
    out = describe_messages([HumanMessage(content="Transformed data — 190 rows × 6 columns")])
    out.encode("cp1252")
    assert out.isascii()
