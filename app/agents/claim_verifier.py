"""Claim verification — does the evidence support what we told the user?

The existing Validator checks **code**: syntax, schema grounding, join
row-explosion, dtype misuse. It never asks whether the *answer* is right. For
an LLM-driven analysis product that is the gap that matters, because the
failure that destroys credibility is not a crash — it is a confident, wrong,
well-formatted answer.

Named ``ClaimVerifier`` rather than ``Verifier`` on purpose: a bare "Verifier"
reads as a sibling of the code Validator, and the whole point is that they
check different things.

| | Validator | ClaimVerifier |
| --- | --- | --- |
| Subject | the code | the claim |
| Asks | will this run against this schema? | is this conclusion supported? |
| Catches | bad column, unsafe join | overreach, causal language, unsupported numbers |

Five checks, each targeting a way an automated analysis misleads:

1. **Causal language from correlational evidence** — "drives", "causes",
   "because of" asserted from a correlation. The single most common overreach,
   and the most damaging in a business report.
2. **Numbers not present in the evidence** — a figure in the narrative that
   appears nowhere in the computed results, i.e. an invented statistic.
3. **Claims contradicted by the model's own metrics** — asserting a reliable
   prediction from a model that did not beat its baseline.
4. **Unhedged extrapolation** — "will", "guarantees", "always" beyond the data.
5. **Significance asserted without a test** — "significant" used where no test
   was run, which readers hear as statistical significance.

Deterministic and dependency-free. The checks are string- and evidence-level,
so they run without an LLM, cost nothing, and cannot themselves hallucinate.
An LLM reviewer can be layered on later; this floor should not depend on one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langchain_core.messages import AIMessage

from app.agents.contract import AgentSpec, Stage, agent

logger = logging.getLogger(__name__)


class Verdict(str, Enum):
    SUPPORTED = "supported"       # evidence backs the claim
    UNSUPPORTED = "unsupported"   # nothing in evidence backs it
    OVERREACH = "overreach"       # evidence exists but the claim exceeds it
    CONTRADICTED = "contradicted"  # evidence points the other way


#: Verbs asserting causation. Matched as whole words so "causal analysis" as a
#: section heading does not trip the check.
#:
#: Base forms matter as much as inflected ones: "late deliveries DRIVE churn"
#: and "delays CAUSE churn" are the phrasings an analyst actually writes, and
#: an earlier version listing only "drives"/"causes" missed both.
_CAUSAL_TERMS = (
    "cause", "causes", "caused", "causing",
    "drive", "drives", "driven by", "drove",
    "lead to", "leads to", "led to",
    "result in", "results in", "resulted in",
    "because of", "due to", "responsible for",
    "impact of", "effect of",
    "influence", "influences", "determine", "determines",
)

#: Noun uses of a causal verb that are not themselves causal claims.
#: "root cause analysis" names a method; it asserts nothing.
_CAUSAL_FALSE_POSITIVES = ("root cause", "causal analysis", "cause analysis")

#: Language promising the future or admitting no exception.
_ABSOLUTE_TERMS = (
    "will always", "always", "never", "guarantees", "guaranteed", "certainly",
    "definitely", "will increase", "will decrease", "ensures",
)

#: Words a reader hears as a statistical claim.
_SIGNIFICANCE_TERMS = ("significant", "significantly", "statistically")

#: Evidence keys that indicate a real test was performed.
_TEST_EVIDENCE_KEYS = ("p_value", "pvalue", "p_val", "confidence_interval",
                       "ci_low", "ci_high", "test_statistic", "significance_test")

#: Evidence that would license causal language.
_CAUSAL_EVIDENCE_KEYS = ("experiment", "randomized", "randomised", "ab_test",
                         "treatment", "control", "causal", "uplift",
                         "instrument", "did", "difference_in_differences")

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?%?")


@dataclass
class ClaimFinding:
    claim: str
    verdict: Verdict
    check: str
    detail: str
    evidence_ref: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim[:400],
            "verdict": self.verdict.value,
            "check": self.check,
            "detail": self.detail,
            "evidence_ref": self.evidence_ref,
        }


@dataclass
class VerificationReport:
    findings: List[ClaimFinding] = field(default_factory=list)
    claims_checked: int = 0

    @property
    def problems(self) -> List[ClaimFinding]:
        return [f for f in self.findings if f.verdict is not Verdict.SUPPORTED]

    @property
    def safe_to_present(self) -> bool:
        """False when any claim is contradicted or overreaches the evidence."""
        return not any(f.verdict in (Verdict.CONTRADICTED, Verdict.OVERREACH)
                       for f in self.findings)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "safe_to_present": self.safe_to_present,
            "claims_checked": self.claims_checked,
            "n_problems": len(self.problems),
            "findings": [f.as_dict() for f in self.findings],
        }

    def summary(self) -> str:
        if not self.problems:
            return f"All {self.claims_checked} claim(s) supported by the evidence."
        return (f"{len(self.problems)} of {self.claims_checked} claim(s) need attention: "
                + "; ".join(f.detail for f in self.problems[:3]))


# --------------------------------------------------------------------------- #
# Evidence helpers
# --------------------------------------------------------------------------- #

def _flatten(evidence: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested evidence to dotted keys for lookup and number extraction."""
    out: Dict[str, Any] = {}
    if isinstance(evidence, dict):
        for key, value in evidence.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(evidence, (list, tuple)):
        for i, value in enumerate(evidence):
            out.update(_flatten(value, f"{prefix}[{i}]"))
    else:
        out[prefix] = evidence
    return out


def _evidence_numbers(flat: Dict[str, Any]) -> List[float]:
    numbers: List[float] = []
    for value in flat.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numbers.append(float(value))
        elif isinstance(value, str):
            for match in _NUMBER.findall(value):
                try:
                    numbers.append(float(match.rstrip("%")))
                except ValueError:
                    continue
    return numbers


def _has_key_like(flat: Dict[str, Any], needles: Sequence[str]) -> Optional[str]:
    for key in flat:
        lowered = key.lower()
        if any(n in lowered for n in needles):
            return key
    return None


def _mentions(text: str, terms: Iterable[str]) -> Optional[str]:
    lowered = text.lower()
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return term
    return None


def split_claims(narrative: str) -> List[str]:
    """Split a narrative into sentence-level claims worth checking."""
    if not narrative:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", narrative)
    return [p.strip() for p in parts if len(p.strip()) > 15]


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def check_causal_overreach(claim: str, flat: Dict[str, Any]) -> Optional[ClaimFinding]:
    """Causal language is only licensed by experimental or causal evidence."""
    lowered = claim.lower()
    if any(phrase in lowered for phrase in _CAUSAL_FALSE_POSITIVES):
        return None
    term = _mentions(claim, _CAUSAL_TERMS)
    if not term:
        return None
    support = _has_key_like(flat, _CAUSAL_EVIDENCE_KEYS)
    if support:
        return None
    return ClaimFinding(
        claim=claim, verdict=Verdict.OVERREACH, check="causal_overreach",
        detail=(f"asserts causation ({term!r}) from evidence that is correlational; "
                "no experiment, treatment/control or causal estimate is present"),
    )


def check_invented_numbers(claim: str, flat: Dict[str, Any],
                           tolerance: float = 0.01) -> Optional[ClaimFinding]:
    """A figure in the narrative that appears nowhere in the evidence."""
    stated = []
    for match in _NUMBER.finditer(claim):
        raw = match.group(0)
        try:
            value = float(raw.rstrip("%"))
        except ValueError:
            continue
        is_percentage = raw.endswith("%")

        # Preserve the useful exemption for ordinary prose counts ("3
        # segments") and calendar years. Percentages are explicitly excluded:
        # "8.0%" is a statistic even though its float value is an integer.
        if not is_percentage and abs(value) < 10 and value.is_integer():
            continue
        if not is_percentage and 1900 <= value <= 2100 and value.is_integer():
            continue
        stated.append((value, raw))
    if not stated:
        return None

    # ``verify_claims`` supplies flattened evidence, while this check is also a
    # public helper used directly by tests and callers. Flatten defensively so
    # nested metrics are genuinely inspected instead of being mistaken for an
    # empty evidence set.
    known = _evidence_numbers(_flatten(flat))
    if not known:
        value, raw = stated[0]
        return ClaimFinding(
            claim=claim, verdict=Verdict.UNSUPPORTED, check="invented_number",
            detail=(f"cites {raw}, but no computed numeric evidence was provided "
                    "to support it"),
        )

    for value, raw in stated:
        if not any(abs(value - k) <= max(tolerance, abs(k) * tolerance) for k in known):
            return ClaimFinding(
                claim=claim, verdict=Verdict.UNSUPPORTED, check="invented_number",
                detail=(f"cites {raw}, which does not appear in the computed "
                        "evidence within tolerance"),
            )
    return None


def check_contradicted_by_metrics(claim: str, flat: Dict[str, Any]) -> Optional[ClaimFinding]:
    """Reliability asserted for a model that failed its own baseline."""
    beats = None
    for key, value in flat.items():
        if key.lower().endswith("beats_baseline"):
            beats = value
            break
    if beats is not False:
        return None
    if not _mentions(claim, ("accurate", "accurately", "reliable", "reliably",
                             "predicts", "strong", "good fit", "performs well")):
        return None
    return ClaimFinding(
        claim=claim, verdict=Verdict.CONTRADICTED, check="contradicted_by_metrics",
        detail=("claims the model is reliable, but evaluation reports it does not "
                "beat a trivial baseline"),
        evidence_ref="evaluation_report.beats_baseline",
    )


def check_absolute_language(claim: str, flat: Dict[str, Any]) -> Optional[ClaimFinding]:
    """Promises about the future that no finite sample licenses."""
    term = _mentions(claim, _ABSOLUTE_TERMS)
    if not term:
        return None
    return ClaimFinding(
        claim=claim, verdict=Verdict.OVERREACH, check="absolute_language",
        detail=(f"uses absolute/predictive language ({term!r}); a finite sample "
                "cannot support a claim admitting no exception"),
    )


def check_unsupported_significance(claim: str, flat: Dict[str, Any]) -> Optional[ClaimFinding]:
    """'Significant' read as statistical, with no test performed."""
    term = _mentions(claim, _SIGNIFICANCE_TERMS)
    if not term:
        return None
    if _has_key_like(flat, _TEST_EVIDENCE_KEYS):
        return None
    return ClaimFinding(
        claim=claim, verdict=Verdict.OVERREACH, check="unsupported_significance",
        detail=(f"uses {term!r} with no p-value, confidence interval or test "
                "statistic in the evidence; readers will hear a statistical claim"),
    )


CHECKS = (
    check_contradicted_by_metrics,
    check_causal_overreach,
    check_unsupported_significance,
    check_absolute_language,
    check_invented_numbers,
)


def verify_claims(narrative: str, evidence: Any) -> VerificationReport:
    """Check every claim in *narrative* against *evidence*."""
    flat = _flatten(evidence or {})
    report = VerificationReport()
    for claim in split_claims(narrative):
        report.claims_checked += 1
        for check in CHECKS:              # ordered: strongest verdict wins
            finding = check(claim, flat)
            if finding is not None:
                report.findings.append(finding)
                break
    if report.problems:
        logger.warning("[claim-verifier] %s", report.summary())
    return report


CLAIM_VERIFIER_SPEC = AgentSpec(
    name="claim_verifier",
    stage=Stage.VERIFY,
    reads=("messages",),
    writes=("verification_report", "verification_safe_to_present"),
    description="Checks narrative claims against the computed evidence.",
)


@agent(CLAIM_VERIFIER_SPEC)
def claim_verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node. Reports; the caller decides whether to soften or present."""
    narrative = state.get("analysis_narrative") or ""
    if not narrative:
        # The main ETL graph communicates its final narrative through the
        # LangChain message channel. Accept dict-shaped messages as well so the
        # node also works on API/checkpoint payloads.
        for message in reversed(state.get("messages") or []):
            if isinstance(message, AIMessage):
                narrative = message.content or ""
                break
            if isinstance(message, dict) and message.get("role") in {"assistant", "ai"}:
                narrative = message.get("content") or ""
                break

    execution_result = state.get("execution_result") or {}
    evidence = {
        "evaluation_report": state.get("evaluation_report"),
        "integrity_report": state.get("integrity_report"),
        "execution_output": (
            state.get("output_json")
            or state.get("execution_output_data")
            or execution_result.get("output_json")
            or execution_result.get("output_data")
        ),
        "training_metrics": state.get("training_metrics"),
    }
    # "No analysis ran" and "analysis ran and produced no numbers" are
    # different states, and only the second is evidence of anything.
    #
    # This node is the `end` target from the conversational route, so ordinary
    # chat turns reach it with every evidence key None. Verifying against an
    # empty set there marks true, benign sentences as invented -- "This dataset
    # has 12 columns" can never be supported, because the profile is not among
    # the keys collected below. Those false findings cost nothing in safety
    # (safe_to_present only trips on CONTRADICTED/OVERREACH) and everything in
    # signal: verification_report is what makes the evidence layer credible,
    # and a report that cries wolf on small talk stops being read.
    #
    # check_invented_numbers keeps its fail-closed behaviour for direct callers
    # who genuinely pass an empty evidence set; this guard is about not
    # ASKING it when no analysis was attempted.
    if not any(value is not None for value in evidence.values()):
        return {
            "verification_report": {
                "safe_to_present": True,
                "claims_checked": 0,
                "n_problems": 0,
                "findings": [],
                "skipped": "no analysis evidence was produced for this turn",
            },
            "verification_safe_to_present": True,
        }

    report = verify_claims(narrative, evidence)
    return {
        "verification_report": report.as_dict(),
        "verification_safe_to_present": report.safe_to_present,
    }
