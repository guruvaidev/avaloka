"""The product benchmark grades every assistant message of the turn.

The chat shows each assistant message a turn produces: a delegated turn says
what it is about to do, then what it did. Grading only the last graded the
result and never the narration before it.
"""
import pytest

pytest.importorskip("app.api.reply_segments")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from avaloka.benchmark.conversational_product import _reply_from_state  # noqa: E402


def test_the_turns_messages_are_all_graded_in_order():
    """Fails on the base: only the last message was graded."""
    final = {"messages": [HumanMessage(content="earlier"), AIMessage(content="old answer"),
                          HumanMessage(content="Compute distinct aisles."),
                          AIMessage(content="On it. Expect seconds to minutes."),
                          AIMessage(content="There are 134 distinct aisles.")]}
    assert _reply_from_state(final) == ("On it. Expect seconds to minutes.\n\n"
                                        "There are 134 distinct aisles.")


def test_a_turn_with_no_reply_is_empty():
    assert _reply_from_state({"messages": [HumanMessage(content="q")]}) == ""
