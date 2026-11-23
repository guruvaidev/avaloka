"""Expertise-adaptive conversation policy for Avaloka.

The policy reduces cognitive burden through progressive disclosure, a recommended
default, bounded choices, and one focused clarification only when the answer
would materially change. It never lowers analytical rigor for novice users.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ExpertiseProfile:
    level: str  # novice | practitioner | expert
    confidence: float
    evidence: tuple[str, ...]
    explicit: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationPolicy:
    expertise: ExpertiseProfile
    explanation_style: str
    max_options: int
    recommend_default: bool
    define_terms: bool
    expose_method_details: bool
    max_clarifying_questions: int
    progressive_disclosure: bool = True

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expertise"] = self.expertise.as_dict()
        return payload


_NOVICE_EXPLICIT = re.compile(
    r"\b(i(?:'m| am) (?:a )?(?:beginner|novice|new to (?:data|analytics|statistics))|"
    r"explain (?:it )?(?:simply|like i(?:'m| am) new)|i don'?t know (?:sql|statistics|data science)|"
    r"this is my first (?:analysis|dataset|experiment))\b",
    re.IGNORECASE,
)
_EXPERT_EXPLICIT = re.compile(
    r"\b(i(?:'m| am) (?:a |an )?(?:data scientist|statistician|ml engineer|analytics engineer|expert)|"
    r"expert mode|skip (?:the )?basics|assume (?:i know|expertise)|give me (?:the )?(?:sql|technical details))\b",
    re.IGNORECASE,
)
_NOVICE_SIGNALS = (
    re.compile(r"\bwhat does .+ mean\b", re.IGNORECASE),
    re.compile(r"\bcan you explain\b", re.IGNORECASE),
    re.compile(r"\bwhere do i start\b", re.IGNORECASE),
    re.compile(r"\bwhat should i do\b", re.IGNORECASE),
)
_EXPERT_SIGNALS = (
    re.compile(r"\b(?:window function|cte|cross[- ]validation|heteroskedastic|estimand|confounder|"
               r"cardinality|calibration set|exchangeability|bootstrap|chi[- ]square|fdr|cuped)\b", re.IGNORECASE),
    re.compile(r"\b(?:show|return|include) (?:the )?(?:sql|query plan|assumptions|diagnostics|confidence interval)\b", re.IGNORECASE),
)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message if isinstance(message, str) else "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def infer_expertise(
    messages: Iterable[Any],
    prior: ExpertiseProfile | dict[str, Any] | None = None,
) -> ExpertiseProfile:
    """Infer communication depth conservatively, honoring explicit user direction."""
    texts = [_message_text(message).strip() for message in messages]
    texts = [text for text in texts if text][-8:]
    latest = texts[-1] if texts else ""
    if _NOVICE_EXPLICIT.search(latest):
        return ExpertiseProfile("novice", 1.0, ("explicit novice preference",), True)
    if _EXPERT_EXPLICIT.search(latest):
        return ExpertiseProfile("expert", 1.0, ("explicit expert preference",), True)

    if isinstance(prior, dict) and prior.get("level") in {"novice", "practitioner", "expert"}:
        prior = ExpertiseProfile(
            str(prior["level"]), float(prior.get("confidence", 0.6)),
            tuple(prior.get("evidence") or ("prior turn",)), bool(prior.get("explicit", False)),
        )
    novice_score = expert_score = 0
    evidence: list[str] = []
    for text in texts:
        novice_hits = sum(bool(pattern.search(text)) for pattern in _NOVICE_SIGNALS)
        expert_hits = sum(bool(pattern.search(text)) for pattern in _EXPERT_SIGNALS)
        novice_score += novice_hits
        expert_score += expert_hits
        if novice_hits:
            evidence.append("asked for foundational guidance")
        if expert_hits:
            evidence.append("used or requested advanced analytical detail")

    if expert_score >= 2 and expert_score > novice_score:
        return ExpertiseProfile("expert", min(0.9, 0.55 + 0.1 * expert_score), tuple(dict.fromkeys(evidence)))
    if novice_score >= 2 and novice_score > expert_score:
        return ExpertiseProfile("novice", min(0.9, 0.55 + 0.1 * novice_score), tuple(dict.fromkeys(evidence)))
    if isinstance(prior, ExpertiseProfile) and prior.explicit:
        return prior
    if isinstance(prior, ExpertiseProfile) and prior.confidence >= 0.7:
        return ExpertiseProfile(prior.level, max(0.55, prior.confidence - 0.05), prior.evidence)
    return ExpertiseProfile("practitioner", 0.5, ("insufficient evidence; use balanced detail",))


def policy_for(profile: ExpertiseProfile) -> ConversationPolicy:
    if profile.level == "novice":
        return ConversationPolicy(
            profile, explanation_style="guided_plain_language", max_options=2,
            recommend_default=True, define_terms=True, expose_method_details=False,
            max_clarifying_questions=1,
        )
    if profile.level == "expert":
        return ConversationPolicy(
            profile, explanation_style="compressed_technical", max_options=3,
            recommend_default=False, define_terms=False, expose_method_details=True,
            max_clarifying_questions=1,
        )
    return ConversationPolicy(
        profile, explanation_style="balanced", max_options=3,
        recommend_default=True, define_terms=False, expose_method_details=True,
        max_clarifying_questions=1,
    )


def build_conversation_directive(policy: ConversationPolicy) -> str:
    """Return a prompt directive that keeps rigor fixed while changing presentation."""
    common = (
        "Reduce cognitive burden: lead with the decision/result, do the mechanical work, "
        "surface only decisions the user must make, and offer no more than "
        f"{policy.max_options} next actions. Ask at most one focused question, and only "
        "when different answers materially change the valid analysis. Never hide uncertainty, "
        "sampling, assumptions, data quality, cost, or failure."
    )
    if policy.expertise.level == "novice":
        return common + (
            " Use plain language, define at most one essential term inline, recommend a safe "
            "default with a short reason, and put formulas/code behind an optional deeper-details "
            "offer. Do not patronize or lower statistical rigor."
        )
    if policy.expertise.level == "expert":
        return common + (
            " Be compact and technical. State the estimand, grain, filters, method, assumptions, "
            "diagnostics, uncertainty, and execution tradeoffs directly. Skip elementary definitions "
            "unless asked and make reproducible SQL/code/artifact details easy to inspect."
        )
    return common + (
        " Use concise professional language, state the method and key assumptions, recommend a "
        "default when useful, and offer deeper implementation detail rather than front-loading it."
    )
