"""Regression tests: query/inspection outputs must never be activated as the
new working dataset. Activating them makes each turn consume the previous
answer as its input (800 bytes -> 67 -> 18 in the reported session).
"""
import pytest

import app.agents.planner as planner_mod
from app.agents.planner import _classify_output_replaces_dataset


@pytest.fixture
def llm_always_true(monkeypatch):
    """LLM double that would activate everything — proves the deterministic
    guard short-circuits before the model is consulted."""

    class R:
        content = '{"activate_output_as_dataset": true}'

    monkeypatch.setattr(planner_mod, "llm", object())
    monkeypatch.setattr(planner_mod, "_routing_llm", lambda: (lambda _prompt: R()))


@pytest.mark.parametrize("prompt", [
    "give me 9 random rows",
    "give me rows where Age is 41",
    "give me al unique names",
    "show me the first 5 rows",
    "list the distinct countries",
    "what is the average price?",
    "how many rows are there?",
])
def test_query_requests_never_activate_output(llm_always_true, prompt):
    assert _classify_output_replaces_dataset(prompt, "1. plan", "cols") is False


def test_modification_requests_still_reach_the_classifier(llm_always_true):
    result = _classify_output_replaces_dataset(
        "remove duplicate rows and impute missing prices", "1. plan", "cols"
    )
    assert result is True


def test_no_llm_defaults_to_not_activating(monkeypatch):
    monkeypatch.setattr(planner_mod, "llm", None)
    assert _classify_output_replaces_dataset(
        "remove duplicate rows", "1. plan", "cols") is False
