"""The model step with a fake client replaying recorded replies.

No provider is called: the fake has the one method ``model_intent`` uses,
``complete(system, prompt, *, max_tokens)``, and returns the next recorded
string from ``fixtures/intent/<name>.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avaloka.outcomes.semantics import ColumnSemantics
from avaloka.recipes.intent import IntentRejected, REJECTIONS
from avaloka.recipes.intent_model import MAX_TOKENS, intent_prompt, model_intent, parse_reply
from avaloka.recipes.specs import SPECS

FIXTURES = Path(__file__).parent / "fixtures" / "intent"
SEM = {n: ColumnSemantics(name=n, role=r) for n, r in
       {"order_date": "time", "region": "dimension", "revenue": "measure", "cost": "measure"}.items()}
CANDIDATES = [SPECS["trend"], SPECS["segment"], SPECS["compare_periods"]]
QUESTION = "How has revenue moved month by month?"


class FakeClient:
    """Replays ``replies`` in order and records every call."""

    def __init__(self, name: str) -> None:
        self.replies = json.loads((FIXTURES / f"{name}.json").read_text())["replies"]
        self.calls: list[tuple[str, str, int]] = []

    def complete(self, system: str, prompt: str, *, max_tokens: int = 600) -> str | None:
        self.calls.append((system, prompt, max_tokens))
        if len(self.calls) > len(self.replies):
            raise AssertionError("called more times than there are recorded replies")
        return self.replies[len(self.calls) - 1]


def test_valid_reply_is_recognized_first_time():
    client = FakeClient("valid")
    intent = model_intent(client, QUESTION, SEM, CANDIDATES)
    assert intent is not None and intent.source == "model"
    assert intent.recipe == "trend"
    assert intent.params == {"measure": "revenue", "time": "order_date", "grain": "month"}
    assert len(client.calls) == 1 and client.calls[0][2] == MAX_TOKENS


def test_rejected_reply_is_retried_once_with_the_reason():
    client = FakeClient("retry")
    intent = model_intent(client, QUESTION, SEM, CANDIDATES)
    assert intent is not None and intent.recipe == "trend"
    assert len(client.calls) == 2
    retry_prompt = client.calls[1][1]
    assert "rejected: unknown_recipe" in retry_prompt and "'forecast'" in retry_prompt
    assert client.calls[0][0] == client.calls[1][0]          # the system prompt does not move
    assert client.calls[1][1].endswith("JSON:")


def test_two_rejections_return_none_and_stop():
    client = FakeClient("fails_twice")
    assert model_intent(client, QUESTION, SEM, CANDIDATES, max_attempts=2) is None
    assert len(client.calls) == 2                             # never a third attempt


def test_a_valid_id_outside_the_candidates_is_rejected():
    client = FakeClient("not_a_candidate")
    assert model_intent(client, QUESTION, SEM, CANDIDATES) is None
    assert "not_a_candidate" in client.calls[1][1]


def test_provider_failure_is_not_retried():
    class Dead:
        calls = 0

        def complete(self, system, prompt, *, max_tokens=600):
            self.calls += 1
            return None

    client = Dead()
    assert model_intent(client, QUESTION, SEM, CANDIDATES) is None and client.calls == 1


def test_prompt_lists_the_closed_vocabularies():
    system, user = intent_prompt(QUESTION, {n: c.role for n, c in SEM.items()}, CANDIDATES)
    for word in ("trend", "segment", "compare_periods", "causal_question", "ambiguous", "multi_question",
                 "day, week, month, quarter, year", "last_period", '"confidence"'):
        assert word in system
    assert "ab_test" not in system                            # only the candidates are offered
    assert "- revenue: measure" in user and QUESTION in user and user.endswith("JSON:")


def test_parse_reply_accepts_fenced_json_and_names_bad_json():
    assert parse_reply('```json\n{"a": 1}\n```')["a"] == 1
    assert parse_reply(' {"a": 1} ')["a"] == 1
    with pytest.raises(IntentRejected) as exc:
        parse_reply("Sure! Here is the JSON: {recipe: trend}")
    assert exc.value.reason == "not_json" and "not_json" in REJECTIONS
