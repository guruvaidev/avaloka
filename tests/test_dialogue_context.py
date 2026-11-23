"""The recent turns the direct reply sees: bounded, own words, current message excluded."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.avaloka_agent import agent as agent_mod


def test_the_dialogue_is_bounded_and_ends_before_the_current_message():
    long = "x" * 5000
    msgs = []
    for i in range(10):
        msgs += [HumanMessage(content=f"q{i} {long}"), AIMessage(content=f"a{i}")]
    msgs.append(HumanMessage(content="current"))
    text = agent_mod._dialogue_before(msgs)
    lines = text.splitlines()
    assert len(lines) == agent_mod._DIALOGUE_MESSAGES
    assert lines[0].startswith("user: q7") and lines[-1] == "assistant: a9"
    assert "current" not in text
    assert all(len(line) <= agent_mod._DIALOGUE_CHARS + len("assistant: ") + 1 for line in lines)


def test_the_web_chats_appended_context_is_not_dialogue():
    text = agent_mod._dialogue_before([
        HumanMessage(content="What is the mean?\n\n[Analysis context] columns: a, b"),
        AIMessage(content="37.31"),
        HumanMessage(content="now")])
    assert "Analysis context" not in text and "user: What is the mean?" in text


def test_a_first_message_has_no_dialogue():
    assert agent_mod._dialogue_before([HumanMessage(content="hi")]) == "none"
