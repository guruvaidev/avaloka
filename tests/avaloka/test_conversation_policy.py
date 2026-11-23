"""Tests for expertise adaptation and cognitive-load controls."""

import importlib.util
from pathlib import Path
import sys


_POLICY_PATH = (
    Path(__file__).parents[2] / "app/agents/avaloka_agent/conversation_policy.py"
)
_SPEC = importlib.util.spec_from_file_location("avaloka_conversation_policy", _POLICY_PATH)
assert _SPEC and _SPEC.loader
_POLICY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _POLICY
_SPEC.loader.exec_module(_POLICY)
ExpertiseProfile = _POLICY.ExpertiseProfile
build_conversation_directive = _POLICY.build_conversation_directive
infer_expertise = _POLICY.infer_expertise
policy_for = _POLICY.policy_for


def test_explicit_novice_gets_guided_policy_without_reduced_rigor():
    profile = infer_expertise(["I'm new to data science. What should I do?"])
    policy = policy_for(profile)
    directive = build_conversation_directive(policy)

    assert profile.level == "novice"
    assert profile.explicit
    assert policy.max_options == 2
    assert policy.recommend_default
    assert "plain language" in directive
    assert "Do not patronize or lower statistical rigor" in directive
    assert "Ask at most one focused question" in directive


def test_explicit_expert_gets_compact_reproducible_details():
    profile = infer_expertise([
        "I'm a data scientist; skip the basics and include confidence interval diagnostics."
    ])
    policy = policy_for(profile)
    directive = build_conversation_directive(policy)

    assert profile.level == "expert"
    assert profile.explicit
    assert policy.expose_method_details
    assert "estimand" in directive
    assert "reproducible SQL/code/artifact" in directive


def test_implicit_expertise_requires_repeated_evidence():
    assert infer_expertise(["Show the confidence interval."]).level == "practitioner"
    profile = infer_expertise([
        "Use a calibration set and report exchangeability assumptions.",
        "Include the query plan and confidence interval diagnostics.",
    ])
    assert profile.level == "expert"
    assert not profile.explicit


def test_multi_turn_explicit_preference_is_sticky_but_can_be_overridden():
    novice = infer_expertise(["I'm a novice; explain it simply."])
    retained = infer_expertise(["Now analyze revenue by region."], prior=novice.as_dict())
    expert = infer_expertise(["Expert mode; skip the basics."], prior=retained)

    assert retained.level == "novice"
    assert retained.explicit
    assert expert.level == "expert"
    assert expert.explicit


def test_balanced_default_does_not_guess_from_short_or_polite_language():
    profile = infer_expertise(["Could you please analyze this dataset?"])
    assert profile == ExpertiseProfile(
        "practitioner", 0.5, ("insufficient evidence; use balanced detail",)
    )


def test_all_policies_bound_questions_and_progressively_disclose():
    for level in ("novice", "practitioner", "expert"):
        policy = policy_for(ExpertiseProfile(level, 1.0, ("test",), True))
        directive = build_conversation_directive(policy)
        assert policy.max_clarifying_questions == 1
        assert policy.progressive_disclosure
        assert "lead with the decision/result" in directive
        assert "do the mechanical work" in directive
        assert "Never hide uncertainty" in directive
