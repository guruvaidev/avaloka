"""1.6 release suite: Bugs-1.6 chatbot replies (BG48, BG49, BG50).

The three bugs are about the wording of a model's reply to "what can you
do?": a dense wall of text (BG48), ending at a dead end (BG49), and ignoring
the dataset the user has loaded (BG50). The reply text is LLM output and is
not deterministic, so these tests do NOT grade a reply. They certify the part
the code controls: what the direct-reply path actually sends to the model.
The reply model is replaced by a recorder, and each test asserts on the
messages it received. Live reply quality is a benchmark question.
"""
from __future__ import annotations

from typing import Any, List

import pytest
from langchain_core.messages import HumanMessage

import app.agents.avaloka_agent.agent as agent
from app.agents.avaloka_agent.intent_classifier import IntentResult

COLUMNS = ["loan_amount", "interest_rate", "state_code", "activity_year"]


class _RecordingLLM:
    def __init__(self):
        self.calls: List[List[Any]] = []

    def invoke(self, messages):
        self.calls.append(messages)

        class _R:
            content = "1. Train a baseline model on loan_amount. Which would you like?"
        return _R()


@pytest.fixture
def sent(monkeypatch):
    """Run the direct-reply path for a capabilities question with a dataset
    loaded, and return (system_prompt, user_payload) as sent to the model."""
    llm = _RecordingLLM()
    monkeypatch.setattr(agent, "_get_reply_llm", lambda: llm)
    monkeypatch.setattr(agent, "_reply_llm_error", None, raising=False)
    state = {"messages": [HumanMessage(content="What kind of models can you build?")]}
    redis_ctx = {"dataset_id": "ds-1", "schema_columns": COLUMNS, "input_data_type": "csv"}
    reply = agent._render_direct_reply(
        state, IntentResult(intent="exploration", confidence=0.9), redis_ctx, None, "exploring")
    assert reply.startswith("1. Train"), "the recorder's reply is returned unchanged"
    assert len(llm.calls) == 1
    system, human = llm.calls[0]
    return str(system.content), str(human.content)


def test_bg48_reply_instructions_require_short_scannable_lists(sent):
    """BG48: the model is told to keep replies short and to give options as a
    short numbered list, not a paragraph."""
    system, _ = sent
    assert "Keep replies short" in system
    assert "short numbered list" in system


def test_bg49_reply_instructions_require_an_invitation_to_act(sent):
    """BG49: the model is told never to end on a bare fact and to close any
    list of options by inviting the user to pick one."""
    system, _ = sent
    assert "Never end on a bare fact" in system
    assert "CLOSE BY INVITING THE USER TO PICK ONE" in system


def test_bg50_reply_is_grounded_in_the_loaded_columns(sent):
    """BG50: the loaded dataset's columns are in what the model receives, and
    it is told to ground its suggestions in them."""
    system, payload = sent
    for col in COLUMNS:
        assert col in payload, col
    assert "Ground every suggestion in the columns" in system
