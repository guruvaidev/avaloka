"""The app profiler's domain is always one of ``avaloka.domains.VOCABULARY``.

Names decide first; the model is consulted only for a generic or unsure read,
and whatever free text it returns is mapped onto the vocabulary or rejected
to ``generic``. The model is stubbed: no key, no network.
"""

from __future__ import annotations

import json

import pytest

from app.agents import profiling_agent as pa
from avaloka.domains import DOMAIN_IDS, GENERIC, VOCABULARY

HEALTHCARE = ["patient_id", "admission_type", "diagnosis_code", "length_of_stay", "total_charges"]
OPAQUE = ["a", "b", "c", "value_1", "value_2"]
ROWS = [{"a": 1, "b": "x", "c": 2.5}]


@pytest.fixture
def stub_model(monkeypatch):
    """Replace the model with a recorder; ``answers['domain']`` is what Call 1 returns."""
    calls: list[str] = []
    answers: dict = {"domain": None}

    def fake_call_json(system, user, label):
        calls.append(label)
        if label == "domain":
            return {"domain": answers["domain"]} if answers["domain"] is not None else None
        return {"column_explanations": {}, "predictive_values": {}}

    monkeypatch.setattr(pa, "profiling_llm", object())
    monkeypatch.setattr(pa, "_call_json", fake_call_json)
    monkeypatch.setattr(pa, "_PROFILE_PARALLEL", False)
    return calls, answers


@pytest.mark.parametrize("free_text", [
    "Media/YouTube Analytics",
    "Scientific/Something Entirely New",
    "Sports/Match Statistics",
    "",
    None,
    {"nested": "object"},
])
def test_free_text_outside_the_vocabulary_becomes_generic(stub_model, free_text):
    calls, answers = stub_model
    answers["domain"] = {"category": free_text, "confidence": 0.99, "reasoning": "made up"}

    result = pa.profile_quick(OPAQUE, ROWS)["quick_profiling_result"]
    domain = result["domain"]
    assert domain["category"] in DOMAIN_IDS
    assert domain["category"] == GENERIC
    assert domain["source"] in {"model", "model_rejected"}
    assert domain["hints"] == VOCABULARY[GENERIC].hints()
    assert domain["questions"] == list(VOCABULARY[GENERIC].questions)
    assert calls.count("domain") == 1


def test_free_text_with_a_known_alias_maps_into_the_vocabulary(stub_model):
    calls, answers = stub_model
    answers["domain"] = {"category": "FinTech/Payments", "confidence": 0.8, "reasoning": "cards"}

    domain = pa.profile_quick(OPAQUE, ROWS)["quick_profiling_result"]["domain"]
    assert domain["category"] == "finance"
    assert domain["label"] == VOCABULARY["finance"].label
    assert domain["source"] == "model"
    assert domain["model_said"] == "FinTech/Payments"
    assert domain["confidence"] == 0.8
    # Hints always come from the vocabulary, never from the model's prose.
    assert domain["hints"] == VOCABULARY["finance"].hints()
    assert domain["questions"] == list(VOCABULARY["finance"].questions)


def test_confident_names_never_consult_the_model(stub_model):
    calls, answers = stub_model
    answers["domain"] = {"category": "Retail/E-commerce", "confidence": 0.99, "reasoning": "wrong"}

    domain = pa.profile_quick(HEALTHCARE, [])["quick_profiling_result"]["domain"]
    assert domain["category"] == "healthcare"
    assert domain["source"] == "vocabulary"
    assert "domain" not in calls, "the model must not be asked when the names decide"
    assert {h["column"] for h in domain["matched_cues"]} <= set(HEALTHCARE)


def test_profile_full_marks_domain_incomplete_only_when_the_model_was_needed_and_failed(stub_model):
    calls, answers = stub_model
    answers["domain"] = None  # model never returns JSON
    stats = {"column_statistics": {}, "data_quality": {}}

    full = pa.profile_full(OPAQUE, ROWS, stats)["full_profiling_result"]
    assert full["domain"]["category"] == GENERIC
    assert "domain" in full["incomplete_sections"]

    full = pa.profile_full(HEALTHCARE, [], stats)["full_profiling_result"]
    assert full["domain"]["category"] == "healthcare"
    assert "domain" not in full["incomplete_sections"]


def test_the_model_prompt_lists_the_closed_vocabulary():
    assert all(d in pa._DOMAIN_CHOICES for d in DOMAIN_IDS)


def test_profile_domain_exposes_questions_for_chat_voice_and_no_sample_value(stub_model):
    """``profile["domain"]["questions"]`` is the field chat-voice reads; the
    rows passed to the profiler never surface in it."""
    rows = [{"patient_id": "MRN-SENTINEL-77", "total_charges": 4321.0}]
    domain = pa.profile_quick(HEALTHCARE, rows)["quick_profiling_result"]["domain"]
    assert domain["questions"] == list(VOCABULARY["healthcare"].questions)
    assert "MRN-SENTINEL-77" not in json.dumps(domain)


def test_one_cue_asks_the_model_and_never_stands_alone(stub_model):
    """A single cue (``Ticket`` on the Titanic list) is a tentative read: the
    model is asked, and if it never answers the domain is generic, not the
    one-cue guess."""
    calls, answers = stub_model
    titanic = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Ticket", "Fare"]
    answers["domain"] = None
    domain = pa.profile_quick(titanic, [])["quick_profiling_result"]["domain"]
    assert calls.count("domain") == 1
    assert domain["category"] == GENERIC
    assert domain["hints"] == VOCABULARY[GENERIC].hints()
