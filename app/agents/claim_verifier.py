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

Six checks, each targeting a way an automated analysis misleads:

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
6. **A method the code never ran** — "the gradient-boosted model achieves…"
   over a script that fitted a linear regression. The numbers are real, so the
   other checks pass; what is wrong is the description of how they were made.

And a figure given to the wrong entity: one that is in the evidence, but is
another group's or the total (:class:`FigureAttribution`), or on a
what-predicts question is not the column's association (:func:`check_driver_figures`).

Deterministic and dependency-free. The checks are string- and evidence-level,
so they run without an LLM, cost nothing, and cannot themselves hallucinate.
An LLM reviewer can be layered on later; this floor should not depend on one.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from langchain_core.messages import AIMessage

from app.agents.analysis_outcome import (ANSWERED, FAILED, NO_RESULT, NOTHING_TO_VERIFY,
                                         PARTIAL, REFUSED, analysis_outcome, result_records)
from app.agents.contract import AgentSpec, Stage, agent
from app.agents import derived_categories
from app.agents.filter_disclosure import apply_filter_scope, apply_truncation_scope
from app.agents.user_text import last_user_text

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

#: A variable said to move an outcome's odds is a causal claim too: "Higher
#: glucose levels increase the likelihood of readmission" (rc14 nullxlsx) was
#: read off a table of glucose bins. "an increased risk" asserts nothing.
_CAUSAL_MOVES = re.compile(
    r"\b(?:increase|raise|boost|elevate|reduce|lower|decrease|cut|halve|double)s?\s+"
    r"(?:the\s+|a\s+|one'?s\s+|your\s+|their\s+|its\s+)?(?:likelihood|risk|chances?|odds|probability)\b",
    re.IGNORECASE)

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

#: Words that make a figure the difference of two others: "a gap of 18,756",
#: "about 9.3 percentage points", "1,310 units fewer than the top item".
_DIFFERENCE_WORDS = re.compile(
    r"\b(?:difference|differs?|gap|apart|margin|more than|less than|fewer than|higher than|lower than|"
    r"greater than|smaller than|ahead|behind|exceeds?|leads? by|percentage points?|pp|points? (?:higher|lower|"
    r"more|less))\b", re.IGNORECASE)
_POINTS = re.compile(r"\b(?:percentage points?|pp|points?)\b", re.IGNORECASE)

#: Evidence keys that indicate a real test was performed.
_TEST_EVIDENCE_KEYS = ("p_value", "pvalue", "p_val", "confidence_interval",
                       "ci_low", "ci_high", "test_statistic", "significance_test")

#: Evidence that would license causal language.
_CAUSAL_EVIDENCE_KEYS = ("experiment", "randomized", "randomised", "ab_test",
                         "treatment", "control", "causal", "uplift",
                         "instrument", "did", "difference_in_differences")

#: Evidence key holding the script that produced the result. It is read by
#: ``check_method_not_in_code`` only: a script is full of literals (a split
#: ratio, a tree count) that are not results, so it is kept out of the numeric
#: evidence the invented-number check trusts.
_CODE_KEY = "executed_code"

#: Model families a narrative can name, as (terms a reader sees, substrings the
#: script must contain for the name to be true). Markers are matched against
#: the lower-cased script with its comments and string literals removed, so
#: "gradientboosting" covers GradientBoostingRegressor and
#: HistGradientBoostingClassifier alike, "xgb" covers both the xgboost import
#: and XGBRegressor, and a comment saying "a RandomForest would be better" does
#: not count as one being fitted.
_METHOD_FAMILIES = {
    "gradient boosting": (
        ("gradient-boosted", "gradient boosted", "gradient boosting",
         "gradient-boosting", "boosted tree", "boosted trees", "boosted model",
         "boosting model", "gbm", "gbdt", "gbrt",
         "xgboost", "lightgbm", "catboost"),
        ("gradientboosting", "xgb", "lightgbm", "lgbm", "catboost"),
    ),
    "random forest": (
        ("random forest", "random-forest", "random forests"),
        ("randomforest",),
    ),
    "neural network": (
        ("neural network", "neural networks", "neural net", "neural-network",
         "deep learning", "multilayer perceptron", "multi-layer perceptron",
         "perceptron", "mlp"),
        ("torch", "keras", "tensorflow", "mlpregressor", "mlpclassifier"),
    ),
}

#: Words that, just before a model's name, say the sentence is not claiming it
#: produced the result: a negation ("no", "never", "unlike"), a comparison
#: ("outperforms", "compared with", "versus"), a suggestion ("could", "consider")
#: or a restated request ("you asked for"). Judged on the few words before the
#: mention, not the whole sentence: "the random forest reached 0.91, which
#: isn't bad" claims the forest ran, and the "isn't" three clauses later does
#: not change that. The cue may be followed by up to three words ("never
#: trained a neural network", "you asked for a gradient-boosted model").
_NOT_ASSERTED_BEFORE = re.compile(
    r"(?:\b(?:no|not|never|neither|nor|without|unlike|than|versus|vs\.?|against|over|"
    r"instead of|rather than|compared (?:with|to)|comparable (?:with|to)|relative to|"
    r"beats?|outperform(?:s|ed|ing)?|exceed(?:s|ed)?|within|close to|near|similar to|like|"
    r"as (?:good|accurate|well|fast) as|match(?:es|ed|ing)?|"
    r"could|would|should|might|may|consider(?:ing)?|try(?:ing)?|recommend(?:s|ed)?|suggest(?:s|ed)?|"
    r"asked for|requested|wanted|expected|instead)\b|n't)"
    r"(?:\s+\S+){0,3}\s*$",
    re.IGNORECASE)
#: Words right after the mention that say it did not run or is only an option.
_NOT_ASSERTED_AFTER = re.compile(
    r"^\s*(?:\w+\s+){0,3}?(?:timed out|failed|was not|wasn't|were not|weren't|never|could not|"
    r"couldn't|did not|didn't|is not|isn't|unavailable|unable|would|could|should|might|may|"
    r"instead|baseline|alternative|option|next step|later)\b", re.IGNORECASE)
#: How many words before a mention are read for a cue.
_CUE_WINDOW = 6

_COMMENT_OR_STRING = re.compile(
    r'#[^\n]*|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')


def _code_only(script: str) -> str:
    """The script with its comments and string literals blanked: a model named
    in a comment, a docstring or a print() message was not fitted."""
    return _COMMENT_OR_STRING.sub(" ", script).lower()


def _asserted_mention(claim: str, terms) -> Optional[str]:
    """The term of *terms* the claim asserts produced the result, else None.

    Longer terms first, and a mention judged not asserted silences the shorter
    terms inside it: "no gradient boosting model" is not re-read as a claim
    about a "boosting model".
    """
    lowered = claim.lower()
    silenced: List[Tuple[int, int]] = []
    for term in sorted(terms, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(term)}\b", lowered):
            if any(s <= m.start() < e or s < m.end() <= e for s, e in silenced):
                continue
            before = " ".join(lowered[:m.start()].split()[-_CUE_WINDOW:])
            after = lowered[m.end():]
            if _NOT_ASSERTED_BEFORE.search(before) or _NOT_ASSERTED_AFTER.match(after):
                silenced.append((m.start(), m.end()))
                continue
            return term
    return None

# _NUMBER = re.compile(r"-?\d+(?:\.\d+)?%?")
#
# A narrative writes large values compactly ("56.1B", "24 million") or with
# space-like thousands separators ("56 106 087 508", often U+202F). The old
# pattern read those as 56.1 / 24, or as four separate numbers (56, 106, ...),
# and flagged correct sentences: "cites 56, which does not appear in the
# computed evidence" for a 56,106,087,508 value in the result table.
#
# A figure may be in scientific notation: "5.1e-06", "5.1 × 10⁻⁶", "8.7×10^-9".
# Read as two numbers, "work_type_Never_worked (5.1 × 10⁻⁶)" was withheld for
# citing 5.1 (rc14, groq); the exponent is part of the figure. "× 10" with no
# exponent ("2 × 10 grid") is not one.
# gpt-oss writes "41.6\u202fM": a space before an upper-case scale letter is
# still the scale. Read as 41.6, every such figure was "not in the evidence";
# hidden while nothing else made the reply unsafe, but shown, and false, in
# the notice as soon as a superlative or threshold did. A plain space counts
# only after two digits or a decimal ("28 M", "41.6 M"): "Type 2 B" is a
# label, and lower case keeps the old rule ("5 m" may be metres or minutes).
_NUMBER = re.compile(
    r"(?<![\w.])-?"
    r"(?:\d{1,3}(?:[,\u00a0\u202f\u2009]\d{3})+|\d{1,3}(?: \d{3}){2,}|\d+)"
    r"(?:\.\d+)?"
    r"(?:[eE][-+]?\d{1,3}(?![\w.])|\s?[×xX·*]\s?10(?:\s?(?:\^|\*\*)\s?[-+\u2212]?\d{1,3}|[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,3}))?"
    r"(?:\s?%|\s?percent\b|\s?(?:thousand|million|billion|trillion)\b|(?:[kKmMbBtT]|bn|Bn|BN|mn|tn)(?![\w])"
    r"|(?:[\u00a0\u202f\u2009]|(?<=\d\d) |(?<=\.\d) )(?:[KMBT]|Bn|BN)(?![\w&]))?"
)
# gpt-oss writes "41.6\u202fM": a space before an upper-case scale letter is
# still the scale (1.6 PR #699). Read as 41.6, "17.9 M" matched no row, so
# Extremes could not find the measure of "the lowest average is for Autos &
# Vehicles at about 17.9 M" and checked nothing, and every such figure was
# "not in the evidence". A plain space counts only after two digits or a
# decimal ("28 M", "41.6 M"): "Type 2 B" is a label, "2019 M&A" a year, and
# lower case keeps the old rule ("5 m" may be metres or minutes).
#: The exponent of a figure in scientific notation, and the superscript digits it may be written in.
_EXPONENT = re.compile(r"(?:[eE]([-+]?\d{1,3})|\s?[×xX·*]\s?10\s?(?:\^|\*\*)?\s?([-+\u2212⁻⁺]?[\d⁰¹²³⁴⁵⁶⁷⁸⁹]+))$")
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺\u2212", "0123456789-+-")
_SCALES = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12,
}
_SCALE_SUFFIX = re.compile(r"^(.*\d)\s?([A-Za-z]+)$")


def _is_percent(raw: str) -> bool:
    return raw.rstrip().endswith("%") or raw.rstrip().lower().endswith("percent")


def _exponent(raw: str) -> tuple:
    """``(mantissa, exponent)`` of *raw*: "5.1 × 10⁻⁶" is ("5.1", -6), "38K" ("38K", 0)."""
    text = re.sub(r"\s?(?:%|percent)$", "", raw.strip(), flags=re.IGNORECASE)
    m = _EXPONENT.search(text)
    if not m:
        return text, 0
    return text[:m.start()], int((m.group(1) or m.group(2)).translate(_SUPERSCRIPT))


def _parse_number(raw: str) -> Optional[float]:
    text, exponent = _exponent(raw)
    if abs(exponent) > 300:
        return None
    scale = 10.0 ** exponent
    m = _SCALE_SUFFIX.match(text)
    if m and m.group(2).lower() in _SCALES:
        text, scale = m.group(1), _SCALES[m.group(2).lower()]
    try:
        return float(re.sub(r"[,\u00a0\u202f\u2009 ]", "", text)) * scale
    except ValueError:
        return None


@dataclass
class ClaimFinding:
    claim: str
    verdict: Verdict
    check: str
    detail: str
    evidence_ref: Optional[str] = None
    #: The claim as it may be shown instead, when it is true in substance and
    #: only mislabelled (:func:`relabelled`); :func:`presentable_reply` uses it.
    rewrite: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "claim": self.claim[:400],
            "verdict": self.verdict.value,
            "check": self.check,
            "detail": self.detail,
            "evidence_ref": self.evidence_ref,
        }
        if self.rewrite is not None:
            out["rewrite"] = self.rewrite
        return out


@dataclass
class VerificationReport:
    findings: List[ClaimFinding] = field(default_factory=list)
    #: The statements a check resolved: it found the evidence and judged
    #: the claim, true or not. Not every sentence (rc18d: "I checked 5 claims"
    #: of a reply two of whose claims no check had read).
    claims_checked: int = 0
    #: The statements of the reply (:func:`split_claims`), checked or not.
    claims_seen: Optional[int] = None
    #: Claims a check could not settle and the reply must not pass as checked:
    #: a superlative over a cut of the result (claim_extremes). Not counted.
    unconfirmed: List[str] = field(default_factory=list)
    #: Warnings the reply owes the reader whatever its claims say
    #: (:func:`baseline_caveat`); :func:`presentable_reply` adds them.
    caveats: List[str] = field(default_factory=list)

    @property
    def problems(self) -> List[ClaimFinding]:
        return [f for f in self.findings if f.verdict is not Verdict.SUPPORTED]

    @property
    def safe_to_present(self) -> bool:
        """False when any claim is contradicted or overreaches the evidence."""
        return not any(f.verdict in (Verdict.CONTRADICTED, Verdict.OVERREACH)
                       for f in self.findings)

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "safe_to_present": self.safe_to_present,
            "claims_checked": self.claims_checked,
            "claims_seen": self.claims_checked if self.claims_seen is None else self.claims_seen,
            "n_problems": len(self.problems),
            "findings": [f.as_dict() for f in self.findings],
        }
        if self.unconfirmed:
            out["unconfirmed_claims"] = [c[:400] for c in self.unconfirmed]
        if self.caveats:
            out["caveats"] = list(self.caveats)
        return out

    def summary(self) -> str:
        seen = self.claims_checked if self.claims_seen is None else self.claims_seen
        of = f" of {seen}" if seen > self.claims_checked else ""
        if not self.problems:
            # The count, no verdict (owner's ruling): logs read like the step card.
            return f"Checked {self.claims_checked}{of} claim(s) against the evidence; none flagged."
        return (f"{len(self.problems)} of {seen} claim(s) need attention ({self.claims_checked} checked): "
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
    for key, value in flat.items():
        if key == _CODE_KEY:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numbers.append(float(value))
        elif isinstance(value, str):
            for match in _NUMBER.findall(value):
                parsed = _parse_number(match)
                if parsed is not None:
                    numbers.append(parsed)
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


#: Where a sentence ends: after . ! or ? and a space, but not after an
#: abbreviation. rc15 split "($545.08 vs. $182.27)" at "vs.", withheld the
#: first half and SHOWED the second, "$182.27).", to the user.
_SENTENCE_END = (r"(?<=[.!?])(?<!\bvs\.)(?<!\bVs\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bapprox\.)(?<!\betc\.)"
                 r"(?<!\bcf\.)(?<!\bNo\.)(?<!\bno\.)(?<!\bFig\.)(?<!\bfig\.)(?<!\ba\.m\.)(?<!\bp\.m\.)"
                 r"(?<!\bA\.M\.)(?<!\bP\.M\.)\s+")


def split_claims(narrative: str) -> List[str]:
    """Split a narrative into sentence-level claims worth checking.

    Code is not a claim: fenced blocks and inline `identifiers` are removed first,
    so a script or a column name like `age_25%` is never read as a statistic.
    """
    if not narrative:
        return []
    text = re.sub(r"```.*?```", " ", narrative, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", " ", text)
    parts = re.split(_SENTENCE_END + r"|\n+", text)
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
    if not term and (moves := _CAUSAL_MOVES.search(claim)):
        term = moves.group(0).lower()
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


def question_numbers(question: Any) -> List[float]:
    """The numbers the user's question states, each as it may be restated:
    "within 30 days" -> 30; "over 5%" -> 5 and 0.05."""
    out: List[float] = []
    for match in _NUMBER.finditer(str(question or "")):
        value = _parse_number(match.group(0))
        if value is None:
            continue
        out.append(value)
        if _is_percent(match.group(0)):
            out.append(value / 100)
    return out


def _restates(value: float, restated: Sequence[float]) -> bool:
    """*value* is one the question stated: a restatement, not a claim."""
    return any(abs(value - q) <= 1e-9 * max(1.0, abs(q)) for q in restated)


def _minus_normalized(text: str) -> str:
    """Every dash before a figure is a minus.

    Typographic minus ("−2,008.49") is still a negative number. "–2,008.49"
    (en dash), "−$2,008.49" and "-$2,008.49" are negative numbers too, and so
    is "‑0.17" with a non-breaking hyphen (groq, rc6: "congestion_surcharge
    (‑0.17)" was read as 0.17 and the result's -0.170 did not match it).
    """
    # A dash with a space on each side is punctuation, not a sign: "Highest
    # sales: 22423 – $24,615.97" was read as -24,615.97 (rc15).
    text = re.sub(r"(?:(?<=\s)|^)[-\u2212\u2013\u2014\u2011\u2012](?=\s)", " ", text)
    text = re.sub(r"[-\u2212\u2013\u2011\u2012]\s?[$€£₹]", "-", text)
    return re.sub(r"[\u2212\u2013\u2011\u2012]", "-", text)


def stated_operands(narrative: str, flat: Dict[str, Any], tolerance: float = 0.01) -> List[float]:
    """The evidence values the narrative itself states, each as the evidence
    holds it (13.3% -> 0.1325): what a difference it quotes may be taken of."""
    known = _evidence_numbers(_flatten(flat))
    out: List[float] = []
    for match in _NUMBER.finditer(_minus_normalized(re.sub(r"`[^`\n]*`", " ", narrative or ""))):
        value = _parse_number(match.group(0))
        if value is None:
            continue
        for k in known:
            if abs(value - k) <= max(tolerance * 0.1, abs(k) * 0.001) or (
                    _is_percent(match.group(0)) and abs(value - k * 100) <= 0.051):
                out.append(k)
                break
    return list(dict.fromkeys(out))[:60]


def _a_stated_difference(value: float, raw: str, claim: str, operands: Sequence[float]) -> bool:
    """*value* is the gap between two figures the narrative states and the
    evidence holds: "Enterprise leads consumer by $485.54" (545.33 - 59.79),
    "about 9.3 percentage points" (0.1325 - 0.0397). Only a claim that says it
    is a difference qualifies, and only to the precision it is written in."""
    if len(operands) < 2 or not _DIFFERENCE_WORDS.search(claim):
        return False
    digits = re.sub(r"[^\d.]", "", raw.split("%")[0])
    decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
    tol = 0.5 * 10 ** -decimals + 0.0005 * abs(value) + 1e-9
    scales = (1.0, 100.0) if _POINTS.search(claim) or _is_percent(raw) else (1.0,)
    for i, a in enumerate(operands):
        for b in operands[i + 1:]:
            if any(abs(abs(value) - abs(a - b) * scale) <= tol for scale in scales):
                return True
    return False


def check_invented_numbers(claim: str, flat: Dict[str, Any],
                           tolerance: float = 0.01,
                           restated: Sequence[float] = (),
                           operands: Sequence[float] = ()) -> Optional[ClaimFinding]:
    """A figure in the narrative that appears nowhere in the evidence.

    A number the question itself stated ("within 30 days") is the reply
    restating it, not a figure the analysis produced: ``restated`` exempts it.
    Before, "within 30 days the rate was 0.11" was flagged on the 30 and,
    once flagged numbers were withheld, a correct answer was hidden.

    A difference the claim says it is ("a gap of $485.54") is supported when
    it is the gap between two figures the narrative states and the evidence
    holds (``operands``, :func:`stated_operands`).
    """
    from app.agents.claim_figures import bound_of, bound_supported, column_gaps

    stated = []
    normalized = _minus_normalized(claim)
    for match in _NUMBER.finditer(normalized):
        raw = match.group(0)
        value = _parse_number(raw)
        if value is None:
            continue
        is_percentage = _is_percent(raw)

        # Preserve the useful exemption for ordinary prose counts ("3
        # segments") and calendar years. Percentages are explicitly excluded:
        # "8.0%" is a statistic even though its float value is an integer.
        if not is_percentage and abs(value) < 10 and value.is_integer():
            continue
        if not is_percentage and 1900 <= value <= 2100 and value.is_integer():
            continue
        if _restates(value, restated):
            continue
        # "over 70% of the importance" is a bound, not a value: it holds when
        # what the claim names is over 70% (claim_figures.bound_supported).
        bound = bound_of(normalized, match.start())
        if bound is not None and bound_supported(normalized, flat, value, raw, bound):
            continue
        approx = bool(_APPROX.search(normalized[max(0, match.start() - 18):match.start()]))
        stated.append((value, raw, approx, bound))
    if not stated:
        return None

    # ``verify_claims`` supplies flattened evidence, while this check is also a
    # public helper used directly by tests and callers. Flatten defensively so
    # nested metrics are genuinely inspected instead of being mistaken for an
    # empty evidence set.
    known = _evidence_numbers(_flatten(flat))
    if not known:
        value, raw = stated[0][:2]
        return ClaimFinding(
            claim=claim, verdict=Verdict.UNSUPPORTED, check="invented_number",
            detail=(f"cites {raw}, but no computed numeric evidence was provided "
                    "to support it"),
        )

    def _matches(value: float, k: float) -> bool:
        return abs(value - k) <= max(tolerance, abs(k) * tolerance)

    for value, raw, approx, bound in stated:
        # A percentage may be stored as a fraction: "33.6%" vs 0.336 in the table.
        # Rounded to one decimal of a percent, so allow half a tenth of a point.
        if _is_percent(raw) and any(
            _matches(value, k * 100) or abs(value - k * 100) <= 0.051 for k in known
        ):
            continue
        # A figure is the evidence rounded to the precision it is written in:
        # "≈ 15 %" is 0.1538 (rc14 nullxlsx), which a 1% tolerance missed.
        if any(figure_matches(value, raw, k, approx) for k in known):
            continue
        if not any(_matches(value, k) for k in known):
            if _a_stated_difference(value, raw, claim, operands):
                continue
            # "the gap between enterprise and SMB is $362.81": a gap of two
            # values of one result column, whether or not the reply states them.
            if _DIFFERENCE_WORDS.search(claim) and any(
                    figure_matches(value, raw, g, approx) for g in column_gaps(_flatten(flat), not _is_percent(raw))):
                continue
            cited = f"{bound} {raw}" if bound else raw
            return ClaimFinding(
                claim=claim, verdict=Verdict.UNSUPPORTED, check="invented_number",
                detail=(f"cites {cited}, which does not appear in the computed "
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
    # Presenting the model as ready to use is the same claim in other words:
    # "the training was successful ... you can now run a prediction" got past a
    # list that only knew the vocabulary of accuracy.
    if not _mentions(claim, ("accurate", "accurately", "reliable", "reliably",
                             "predicts", "strong", "good fit", "performs well",
                             "successful", "successfully", "ready to use",
                             "run a prediction", "make predictions")):
        return None
    return ClaimFinding(
        claim=claim, verdict=Verdict.CONTRADICTED, check="contradicted_by_metrics",
        detail=("claims the model is reliable, but evaluation reports it does not "
                "beat a trivial baseline"),
        evidence_ref="evaluation_report.beats_baseline",
    )


_LOWEST_TERMS = ("lowest", "smallest", "minimum", "laggard")
_HIGHEST_TERMS = ("highest", "largest", "greatest", "maximum", "leader")
_ROW = re.compile(r"^(?:execution_output|output_json|output_data)\[\d+\]\.(.+)$")
#: "second-highest", "third largest", "next highest": a rank below the extreme.
_RANKED_EXTREME = re.compile(
    r"\b(?P<rank>second|2nd|third|3rd|fourth|4th|fifth|5th|next)[\s\-\u2010\u2011]+(?:most\s+)?"
    r"(?P<term>highest|largest|greatest|maximum|biggest|lowest|smallest|minimum)\b", re.IGNORECASE)


def check_ordering_contradicted(claim: str, flat: Dict[str, Any]) -> Optional[ClaimFinding]:
    """"Lowest at 68.12" when the result table holds a lower value.

    Each figure in the claim is located in a result column. For every
    superlative the claim makes, at least one located figure must be that
    column's extreme. Figures that match no column are left to the invented-
    number check.
    """
    # "Germany has the second-highest count at 256" says 256 is second, not
    # the max (gemma, rc6); the rank is checked as a rank.
    ranked = [m for m in _RANKED_EXTREME.finditer(claim) if m.group("rank").lower() != "next"]
    plain = _RANKED_EXTREME.sub(" ", claim)
    wants_low = _mentions(plain, _LOWEST_TERMS)
    wants_high = _mentions(plain, _HIGHEST_TERMS)
    if not (wants_low or wants_high or ranked):
        return None
    columns: Dict[str, List[float]] = {}
    for key, value in flat.items():
        m = _ROW.match(key)
        if m and isinstance(value, (int, float)) and not isinstance(value, bool):
            columns.setdefault(m.group(1), []).append(float(value))
    columns = {c: v for c, v in columns.items() if len(v) >= 2}
    if not columns:
        return None

    # Tolerance is the precision the claim states: "68.12" is 68.115-68.125.
    # A relative tolerance is too loose here -- 1% of 68 swallows the 0.39 gap
    # between the true minimum and the figure claimed to be it.
    def _close(value: float, raw: str, actual: float) -> bool:
        digits = re.sub(r"[^\d.]", "", raw)
        decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
        return abs(value - actual) <= 0.5 * 10 ** -decimals + 1e-9

    located: List[tuple] = []          # (value, raw, column)
    for match in _NUMBER.finditer(claim):
        raw = match.group(0)
        value = _parse_number(raw)
        if value is None or _is_percent(raw) or _SCALE_SUFFIX.match(raw.strip()):
            continue
        for col, vals in columns.items():
            if any(_close(value, raw, v) for v in vals):
                located.append((value, raw, col))
    if not located:
        return None
    for m in ranked:
        n = _ORDINAL_VALUES[m.group("rank").lower()]
        high = m.group("term").lower() not in ("lowest", "smallest", "minimum")
        expected = {col: sorted(set(columns[col]), reverse=high)[n - 1:n] for _v, _r, col in located}
        if any(expected[col] and _close(value, raw, expected[col][0]) for value, raw, col in located):
            continue
        value, raw, col = located[0]
        if not expected[col]:
            continue
        return ClaimFinding(
            claim=claim, verdict=Verdict.CONTRADICTED, check="ordering_contradicted",
            detail=(f"says {m.group(0)!r} at {raw}, but the {m.group(0).lower()} of {col!r} in the "
                    f"computed result is {expected[col][0]:g}"),
            evidence_ref=f"execution_output.{col}",
        )
    for term, pick in ((wants_low, min), (wants_high, max)):
        if not term:
            continue
        if not any(_close(value, raw, pick(columns[col])) for value, raw, col in located):
            value, raw, col = located[0]
            extreme = pick(columns[col])
            return ClaimFinding(
                claim=claim, verdict=Verdict.CONTRADICTED, check="ordering_contradicted",
                detail=(f"says {term!r} at {raw}, but the {pick.__name__} of {col!r} in the "
                        f"computed result is {extreme:g}"),
                evidence_ref=f"execution_output.{col}",
            )
    return None


def _fold(text: str) -> str:
    return re.sub(r"[\s_\-\u2010\u2011\u2013]+", " ", str(text).lower()).strip()


def _category_values(flat: Dict[str, Any], used: Optional["DataUsed"], words: Iterable[str]) -> List[str]:
    """The category values of the result and of the data that contain one of
    *words*: "never smoked", "Never_worked", folded."""
    import pandas as pd

    words = [w for w in words]
    values = {str(v) for k, v in flat.items() if _ROW.match(k) and isinstance(v, str)}
    for _label, frame in (used.frames if used else []):
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series) or series.nunique(dropna=True) > 1000:
                continue
            values.update(str(v) for v in series.dropna().unique())
    out = set()
    for value in values:
        folded = _fold(value)
        if folded and any(re.search(rf"\b{re.escape(w)}\b", folded) for w in words):
            out.add(folded)
    return sorted(out, key=len, reverse=True)


def check_absolute_language(claim: str, flat: Dict[str, Any],
                            used: Optional["DataUsed"] = None) -> Optional[ClaimFinding]:
    """Promises about the future that no finite sample licenses.

    A term inside a category value of the result or the data is that
    category's name, not a promise: "never smoked (0.048)" and "never-worked
    (0.0)" are smoking_status and work_type values (groq, rc6)."""
    folded = _fold(claim)
    terms = [t for t in _ABSOLUTE_TERMS if re.search(rf"\b{re.escape(t)}\b", folded)]
    if not terms:
        return None
    names = [(m.start(), m.end()) for v in _category_values(flat, used, terms)
             for m in re.finditer(rf"(?<!\w){re.escape(v)}(?!\w)", folded)]
    term = next((t for t in terms for m in re.finditer(rf"\b{re.escape(t)}\b", folded)
                 if not any(s <= m.start() and m.end() <= e for s, e in names)), None)
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


# --------------------------------------------------------------------------- #
# Recomputation on the data the run actually read
# --------------------------------------------------------------------------- #
# A figure can be in the result table and still be wrong: the result is what
# the code computed, not what the data says. On rc3 a two-sheet workbook was
# read as one sheet twice; "United Kingdom ... 38,024" was in the result, so
# every check above passed it, and the truth was 19,012. These checks go back
# to the frames the sandbox recorded reading (execution_result["reads"]) and
# recompute, the way the CLI's Validator reconciles a goal's figures (#542).

@dataclass
class DataUsed:
    """The frames a run read, and the workbook sheets among them."""
    frames: List[tuple] = field(default_factory=list)      # (label, DataFrame)
    sheets: List[str] = field(default_factory=list)
    #: The path each frame was read from, in the order of ``frames``.
    sources: List[str] = field(default_factory=list)


#: Files larger than this are not re-read to verify a claim.
RECOMPUTE_MAX_BYTES = 512 * 1024 * 1024

#: Words that make a figure a count or a total of a group.
_COUNT_WORDS = re.compile(
    r"\b(?:rows?|counts?|records?|entries|units|sold|totals?|sums?|summed|transactions|"
    r"orders|occurrences|number of|quantity|quantities|invoices|purchases|sales)\b", re.IGNORECASE)


def data_used(state: Mapping[str, Any]) -> DataUsed:
    """Load the frames the run read. Nothing is guessed: no recorded read, no frames."""
    import os

    import pandas as pd

    from app.agents.input_reader import read_input_frame
    from file_handler.null_policy import read_csv_best_effort

    result = state.get("execution_result") if isinstance(state.get("execution_result"), Mapping) else {}
    # Every frame the run used: main(df)'s primary as well as the datasets
    # the code read itself. Keeping only the code's reads dropped the primary
    # main(df) had used, and a correct two-sheet count (37,881) was withheld
    # as "contradicted" against one sheet's 18,869 (rc5). A read the code
    # never used is marked so (execution_agent._main_reads_its_frame).
    reads = [r for r in result.get("reads") or []
             if isinstance(r, Mapping) and r.get("path") and r.get("used") is not False]
    used = DataUsed()
    seen = set()
    for read in reads:
        key = (str(read["path"]), read.get("sheet_name"))
        if key in seen:
            continue
        seen.add(key)
        if read.get("sheet_name"):
            used.sheets.append(str(read["sheet_name"]))
        try:
            if os.path.isfile(key[0]) and os.path.getsize(key[0]) > RECOMPUTE_MAX_BYTES:
                continue
            frame = read_input_frame(key[0], read_csv_best_effort, sheet_name=key[1])
        except Exception:  # noqa: BLE001 - an unreadable frame verifies nothing
            logger.debug("[claim-verifier] could not re-read a frame the run used", exc_info=True)
            continue
        if isinstance(frame, pd.DataFrame):
            used.frames.append((read.get("alias") or read.get("sheet_name") or os.path.basename(key[0]), frame))
            used.sources.append(key[0])
    return used


def _figures(text: str, restated: Sequence[float] = ()) -> List[tuple]:
    """(position, value, raw) for each count-like figure: integers >= 10, not years or percents."""
    out = []
    for match in _NUMBER.finditer(text):
        raw = match.group(0)
        value = _parse_number(raw)
        if value is None or _is_percent(raw) or abs(value) < 10:
            continue
        if 1900 <= value <= 2100 and value.is_integer() and not re.search(r"[,\s]", raw):
            continue
        if _restates(value, restated):
            continue
        out.append((match.start(), value, raw))
    return out


def _labels_in(claim: str, frame: Any) -> List[tuple]:
    """(position, column, label) for each category value the claim names, longest first."""
    import pandas as pd

    lowered = claim.lower()
    hits: List[tuple] = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        if series.nunique(dropna=True) > 50_000:     # identifiers, free text: not groups
            continue
        for label in series.dropna().astype(str).str.strip().unique():
            if len(label) < 3 or label.replace(".", "").isdigit() or label.lower() in _GENERIC_VALUES:
                continue
            if len(label) == 3:
                # "SMB" names the segment 'smb'; a three-letter word in running
                # text ("the", "web") only when written as the data writes it.
                m = re.search(rf"(?<!\w)(?:{re.escape(label)}|{re.escape(label.upper())})(?!\w)", claim)
            else:
                m = re.search(rf"(?<!\w){re.escape(label.lower())}(?!\w)", lowered)
            if m:
                hits.append((m.start(), m.end(), str(column), label))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept: List[tuple] = []
    for hit in hits:                       # a label inside a longer one is not a second label
        if not any(k[0] <= hit[0] and hit[1] <= k[1] for k in kept):
            kept.append(hit)
    return kept


def _group_candidates(frames: Sequence[Any], column: str, label: str) -> Dict[str, float]:
    """Every count, total and mean of the group ``column == label``, per frame and over all frames."""
    import pandas as pd

    per_frame: List[Dict[str, float]] = []
    for frame in frames:
        if column not in frame.columns:
            continue
        rows = frame[frame[column].astype(str).str.strip() == label]
        numeric = rows.apply(pd.to_numeric, errors="coerce").select_dtypes("number")
        numeric = numeric[[c for c in numeric.columns if numeric[c].notna().any() and c != column]]
        valid = numeric[(numeric > 0).all(axis=1)] if len(numeric.columns) else numeric
        figures = {"rows": float(len(rows))}
        # "Self-employed: 44 missing BMI records" is a count of the group too.
        figures.update({f"missing {c}": float(rows[c].isna().sum()) for c in rows.columns if c != column})
        for c in numeric.columns:
            figures[f"sum of {c}"] = float(numeric[c].sum())
            figures[f"sum of positive {c}"] = float(numeric[c][numeric[c] > 0].sum())
            figures[f"sum of {c} where every figure is positive"] = float(valid[c].sum())
            figures[f"mean of {c}"] = float(numeric[c].mean())
        # Revenue is a quantity times a price: a group's "$24,615.97" is no
        # column's total, but the total of a product of two (rc14 messy_xlsx).
        # And the code that computes it usually keeps the sales first
        # (``Quantity > 0``): 85123A's 10,132.48 is the total of Quantity ×
        # UnitPrice over its rows with a positive Quantity, and with only the
        # unfiltered product here it was withheld as "no count, total or mean
        # equal to 10,132.48" (rc16 messy_xlsx).
        products = list(numeric.columns)[:12]
        for i, a in enumerate(products):
            for b in products[i + 1:]:
                product = numeric[a] * numeric[b]
                figures[f"sum of {a} × {b}"] = float(product.sum())
                figures[f"sum of {a} × {b} where {a} is positive"] = float(product[numeric[a] > 0].sum())
                figures[f"sum of {a} × {b} where {b} is positive"] = float(product[numeric[b] > 0].sum())
                figures[f"sum of {a} × {b} where both are positive"] = float(
                    product[(numeric[a] > 0) & (numeric[b] > 0)].sum())
        per_frame.append(figures)
    out: Dict[str, float] = {}
    for i, figures in enumerate(per_frame):
        for name, value in figures.items():
            out[f"{name} (frame {i + 1})" if len(per_frame) > 1 else name] = value
    if len(per_frame) > 1:
        for name in per_frame[0]:
            if not name.startswith("mean of "):
                out[name] = sum(f.get(name, 0.0) for f in per_frame)
    return out


def _close_figure(value: float, raw: str, candidate: float) -> bool:
    if _SCALE_SUFFIX.match(raw.strip()):                  # "38K", "1.2 million": rounded
        return abs(value - candidate) <= 0.05 * max(abs(candidate), 1.0)
    return abs(value - candidate) <= max(0.5, 0.001 * abs(candidate))


def check_group_figure(claim: str, flat: Dict[str, Any], used: Optional[DataUsed],
                       restated: Sequence[float] = ()) -> Optional[ClaimFinding]:
    """A count or total stated for a group must be recomputable from the data used."""
    if used is None or not used.frames or not _COUNT_WORDS.search(claim):
        return None
    figures = _figures(claim, restated)
    if not figures:
        return None
    import pandas as pd

    frames = [f for _, f in used.frames]
    labels = _labels_in(claim, frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True))
    # "WORLD WAR 2 GLIDERS": a figure inside a label is part of its name.
    figures = [f for f in figures if not any(s <= f[0] < e for s, e, _c, _l in labels)]
    for i, (start, _end, column, label) in enumerate(labels):
        nxt = labels[i + 1][0] if i + 1 < len(labels) else len(claim)
        after = [f for f in figures if start < f[0] < nxt]
        before = [f for f in figures if f[0] < start]
        if not (after or (len(labels) == 1 and before)):
            continue
        _pos, value, raw = (after or before[-1:])[0]
        candidates = _group_candidates(frames, column, label)
        if any(_close_figure(value, raw, c) for c in candidates.values()):
            continue
        n_rows = candidates.get("rows", 0.0)
        multiple = ""
        for name, c in candidates.items():
            if c > 0 and value / c >= 1.5 and abs(value / c - round(value / c)) < 1e-6:
                multiple = (f"; {raw} is exactly {round(value / c)} times its {name}, "
                            "as if the same rows were counted more than once")
                break
        where = (f"{len(frames)} frames" if len(frames) > 1
                 else "the join the code performed" if used.frames[0][0] == _JOINED else "the data")
        return ClaimFinding(
            claim=claim, verdict=Verdict.CONTRADICTED, check="group_figure_mismatch",
            detail=(f"says {raw} for {label!r}, but recomputed on {where} the run read it has "
                    f"{n_rows:,.0f} rows and no count, total or mean equal to {raw}{multiple}"),
            evidence_ref=f"data.{column}",
        )
    return None


# --------------------------------------------------------------------------- #
# Whose figure is it?
# --------------------------------------------------------------------------- #
# A figure can be in the evidence and still be wrong for the entity the
# sentence gives it to. On rc12 (work4) "Self‑employed: 0.079 stroke rate, 201
# missing BMI records" was verified: 201 is in the result, but it is the whole
# file's missing BMI (the code took df['bmi'].isna().sum() before grouping);
# Self-employed has 44. Each figure is attached to the entity the sentence
# names before it, and must be that entity's value: in the data the run read
# when the entity is one of its categories, else in the result row the entity
# keys. A figure that is only another entity's, or only the total, is withheld.

#: Says the figure is approximate: "≈ 0.132", "about 8%", "~0.13".
_APPROX = re.compile(r"(?:[≈~∼]|\b(?:about|around|roughly|approximately|approx\.?|nearly|almost|close to|some))"
                     r"\W{0,3}$", re.IGNORECASE)
#: Between an entity and a figure: the figure is someone else's ("819 of the
#: 5,110 patients", "0.079 against an overall 0.049", "than", "vs"), or a
#: bound rather than a value ("below 0.05").
_NOT_ITS_FIGURE = re.compile(
    r"\d\s*(?:of|out of)\b|\b(?:of the|in total|overall|whole|entire|across|dataset|population|average|baseline|"
    r"than|versus|vs|compared|against|relative|whereas|while|but|below|under|above|over|at least|at most|up to|"
    r"within|beyond|exceed\w*|less|more|fewer)\b|[/<>]", re.IGNORECASE)
#: Just after a figure: it is the whole file's ("201 in all").
_TOTAL_AFTER = re.compile(r"^\W*(?:\w+\W+)?(?:in total|in all|overall|altogether|combined|all told|across)\b",
                          re.IGNORECASE)
#: A figure is looked for among the other groups of a column with at most this many.
ATTRIBUTION_MAX_GROUPS = 50
#: Labels of a result row that make it the total, not a group.
_TOTAL_LABELS = frozenset({"total", "all", "overall", "grand total", "everyone", "all rows", "all patients"})
#: Hyphens, dashes, underscores and the spaces a reply writes between words.
_FOLD_CHARS = str.maketrans({c: " " for c in "-_‐‑‒–—−   "})
_CELL = re.compile(r"^(?:execution_output|output_json|output_data)\[(\d+)\]\.(.+)$")
#: After a figure, up to the entity it is of: "19,012 rows in 2010 Q4".
_FIGURE_OF_NEXT = re.compile(r"^\s*(?:[A-Za-z]+\s+){0,2}(?:in|for|during|at|from)\s+(?:the\s+)?$", re.IGNORECASE)
#: Two figures read as one range: "0.13–0.14", "7 to 9%".
_RANGE_JOIN = re.compile(r"^\s*(?:[-‐‑‒–—]|to)\s*$")


def _fold_chars(text: str) -> str:
    """*text* folded for matching with its length kept: lower case, and every
    hyphen, underscore or odd space a space. Positions in it are positions in
    *text*."""
    return "".join(ch.lower() if len(ch.lower()) == 1 else ch for ch in str(text).translate(_FOLD_CHARS))


def _as_number(value: Any) -> Optional[float]:
    """A numeric cell, also when the result was read back from CSV as text."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value == value else None
    if isinstance(value, str):
        try:
            out = float(value.strip().replace(",", ""))
        except ValueError:
            return None
        return out if out == out else None
    return None


def _decimals(raw: str) -> int:
    digits = re.sub(r"[^\d.]", "", _exponent(raw)[0])
    return len(digits.split(".", 1)[1]) if "." in digits else 0


def figure_tolerance(value: float, raw: str, approx: bool = False) -> float:
    """How far *value*, written as *raw*, may be from what it reports.

    A figure is the evidence rounded to the precision it is written in:
    "0.079" is 0.0785-0.0795, half a unit of its last digit; "201" is
    200.5-201.5. Said to be approximate ("≈ 0.132", "about 8%") it may be a
    whole unit of its last digit off. A scaled figure ("38K", "1.2 million")
    is a unit of its mantissa's last digit at that scale, and so is one in
    scientific notation ("5.1 × 10⁻⁶" is 5.05e-6 to 5.15e-6)."""
    mantissa, exponent = _exponent(raw)
    m = _SCALE_SUFFIX.match(mantissa)
    scale = 10.0 ** max(-300, min(300, exponent))
    if m and m.group(2).lower() in _SCALES and not _is_percent(raw):
        scale *= _SCALES[m.group(2).lower()]
    unit = 10.0 ** -_decimals(raw) * scale
    return (1.0 if approx else 0.5) * unit + 1e-9 * max(1.0, abs(value))


def figure_matches(value: float, raw: str, candidate: float, approx: bool = False,
                   high: Optional[float] = None) -> bool:
    """*value* (or the range *value*..*high*) reports *candidate*. A
    percentage reports a fraction ("8%" is 0.0794) or a percent."""
    tol = figure_tolerance(high if high is not None else value, raw, approx)
    low, top = (value, high) if high is not None else (value, value)
    targets = (candidate * 100, candidate) if _is_percent(raw) else (candidate,)
    return any(low - tol <= t <= top + tol for t in targets)


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:.4g}"


def attached_figures(claim: str, restated: Sequence[float] = ()) -> List[Dict[str, Any]]:
    """Each figure of *claim* as a pair can be made of it: position, value,
    the range's top when it is a range ("0.13–0.14"), and whether it is said
    to be approximate. Prose counts under 10 and years are not figures."""
    text = claim.translate(str.maketrans({c: "-" for c in "−–‑‒‐"}))
    found = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0)
        value = _parse_number(raw)
        if value is None:
            continue
        found.append({"start": m.start(), "end": m.end(), "value": value, "raw": raw.strip(), "high": None})
    out: List[Dict[str, Any]] = []
    for fig in found:
        prev = out[-1] if out else None
        if prev is not None and prev["high"] is None and _RANGE_JOIN.match(text[prev["end"]:fig["start"]]) \
                and fig["value"] >= prev["value"] > 0:
            prev.update(end=fig["end"], high=fig["value"], raw=fig["raw"])   # "7 to 9%": both are percents
            continue
        out.append(fig)
    figures = []
    for fig in out:
        value, raw = fig["value"], fig["raw"]
        if not _is_percent(raw) and fig["high"] is None:
            if abs(value) < 10 and value.is_integer():
                continue
            if 1900 <= value <= 2100 and value.is_integer():
                continue
        if _restates(value, restated):
            continue
        fig["approx"] = bool(_APPROX.search(claim[max(0, fig["start"] - 18):fig["start"]]))
        figures.append(fig)
    return figures


def _label_spans(claim: str, labels: Iterable[str]) -> List[tuple]:
    """``(start, end, label)`` for each label *claim* names: as written,
    then folded (case, spacing, hyphens, underscores and their unicode
    forms: "Self‑employed" with a non-breaking hyphen is "Self-employed").
    Longest first; a label inside a longer one is not a second label."""
    folded = _fold_chars(claim)
    hits: List[tuple] = []
    for label in labels:
        text = str(label).strip()
        if len(text) < 3:
            continue
        exact = 0
        if len(text) == 3:
            # "SMB" is the segment 'smb'; a three-letter word in running text
            # only when written as the data writes it (_labels_in).
            spans = [m.span() for m in re.finditer(
                rf"(?<!\w)(?:{re.escape(text)}|{re.escape(text.upper())})(?!\w)", claim)]
        else:
            spans = [m.span() for m in re.finditer(rf"(?<!\w){re.escape(text)}(?!\w)", claim)]
            if not spans:
                exact = 1
                words = _fold_chars(text).split()
                if not words:
                    continue
                pattern = r"\s+".join(re.escape(w) for w in words)
                spans = [m.span() for m in re.finditer(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", folded)]
        hits.extend((s, e, text, exact) for s, e in spans)
    # Of labels that differ only in case ("RECOVERED", "Recovered"), the one
    # written as the claim writes it, then the first by name: the order of a
    # set decided it before, differently from run to run (rc14 FP replay).
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0]), h[3], h[2]))
    kept: List[tuple] = []
    for hit in hits:
        if not any(k[0] <= hit[0] < k[1] or k[0] < hit[1] <= k[1] for k in kept):
            kept.append(hit[:3])
    return kept


def _pairs(claim: str, spans: Sequence[tuple], figures: Sequence[Dict[str, Any]]) -> List[tuple]:
    """``(span, figure, segment, first)``: each figure with the entity named
    last before it, unless what lies around it says the figure is not its
    own ("of the 5,110", "than", "201 in all"). *segment* is the text from
    the entity to the next one: what the sentence says about it. *first* is
    whether no other figure comes between: a later one may belong to
    something the sentence names that is not an entity ("heart disease
    (0.135) and glucose (0.132)"), so only the first may be another
    entity's.

    A list of entities followed by its figures is paired in order: in
    "hypertension and heart_disease (~0.028 and 0.025)" 0.028 is
    hypertension's, and a list it cannot pair gives its figures to no one
    (rc14: "0.028 is hypertension's importance, not heart_disease's")."""
    from app.agents.claim_figures import respective

    paired = respective(claim, spans, figures)
    out = []
    for k, fig in enumerate(figures):
        if any(s[0] <= fig["start"] < s[1] for s in spans):
            continue
        # "19,012 rows in 2010 Q4 and 18,869 rows in 2011 Q1": a figure
        # followed by "in <entity>" is that entity's, not the one before it (rc15).
        nxt_span = next((s for s in spans if s[0] >= fig["end"]), None)
        if nxt_span is not None and k not in paired and _FIGURE_OF_NEXT.match(claim[fig["end"]:nxt_span[0]]):
            out.append((nxt_span, fig, claim[fig["start"]:nxt_span[1]], True))
            continue
        before = [s for s in spans if s[1] <= fig["start"]]
        if not before:
            continue
        if k in paired and paired[k] is None:
            continue
        span = paired.get(k) or before[-1]
        between = claim[before[-1][1]:fig["start"]]
        # A paired figure's owner is settled by the pairing ("over SMB ($545.08 vs. $182.27)").
        if (k not in paired and _NOT_ITS_FIGURE.search(between)) or re.search(r"[;!?]\s|(?<!vs)\.\s", between) \
                or _TOTAL_AFTER.match(claim[fig["end"]:]):
            continue
        if k in paired:
            out.append((span, fig, claim[span[0]:span[1]] + claim[before[-1][1]:fig["end"]], True))
            continue
        nxt = min((s[0] for s in spans if s[0] >= span[1]), default=len(claim))
        first = not any(span[1] <= f["start"] < fig["start"] for f in figures)
        out.append((span, fig, claim[span[0]:max(nxt, fig["end"])], first))
    return out


class FigureAttribution:
    """The entities a reply may give figures to, read once per reply: the
    category values of the frames the run read, and the text cells of the
    result rows."""

    def __init__(self, flat: Mapping[str, Any], used: Optional["DataUsed"] = None):
        rows: Dict[int, Dict[str, Any]] = {}
        for key, value in flat.items():
            m = _CELL.match(key)
            if m:
                rows.setdefault(int(m.group(1)), {})[m.group(2)] = value
        self.rows = [rows[i] for i in sorted(rows)]
        #: label -> [(row index, column)] of the result rows it keys.
        self.row_keys: Dict[str, List[tuple]] = {}
        for i, row in enumerate(self.rows):
            for col, value in row.items():
                if isinstance(value, str) and _as_number(value) is None and value.strip():
                    label = value.strip()
                    if label.lower() not in _GENERIC_VALUES and label.lower() not in _GENERIC_LABELS:
                        self.row_keys.setdefault(label, []).append((i, col))
        self.frame = None
        #: label -> the data's category column it is a value of (unambiguous ones only).
        self.groups: Dict[str, str] = {}
        frames = [f for _l, f in (used.frames if used else [])]
        if frames:
            self._index_frames(frames)

    def _index_frames(self, frames: Sequence[Any]) -> None:
        import pandas as pd

        frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        owners: Dict[str, set] = {}
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                continue
            if series.nunique(dropna=True) > 1000:          # identifiers, free text: not groups
                continue
            for label in series.dropna().astype(str).str.strip().unique():
                if len(label) < 3 or label.replace(".", "").isdigit() or label.lower() in _GENERIC_VALUES \
                        or label.lower() in _GENERIC_LABELS:
                    continue
                owners.setdefault(label, set()).add(str(column))
        self.groups = {label: next(iter(cols)) for label, cols in owners.items() if len(cols) == 1}
        self.frame = frame if self.groups else None

    # -- the data ------------------------------------------------------------
    def _measures(self, segment: str) -> List[str]:
        """The data's columns *segment* names: what its figures are of."""
        from avaloka.goal_analysis import column_mentions

        columns = [str(c) for c in self.frame.columns]
        named = [c for _s, _e, c in column_mentions(segment, columns)] + _named_columns(segment, columns)
        return list(dict.fromkeys(named))

    def _stats(self, rows: Any, measures: Sequence[str], counts: bool) -> Dict[tuple, float]:
        import pandas as pd

        out: Dict[tuple, float] = {}
        if counts:
            out[("rows", "")] = float(len(rows))
        for column in measures:
            if column not in rows.columns:
                continue
            out[("missing", column)] = float(rows[column].isna().sum())
            numeric = pd.to_numeric(rows[column], errors="coerce")
            if numeric.notna().any():
                out[("mean", column)] = float(numeric.mean())
                out[("sum", column)] = float(numeric.sum())
        return out

    def _describe(self, stat: tuple, segment: str) -> str:
        kind, column = stat
        if kind == "rows":
            return "row count"
        m = re.search(r"[_\s]+".join(re.escape(p) for p in re.split(r"[_\s]+", column) if p), segment, re.IGNORECASE)
        name = m.group(0) if m else column
        if kind == "missing":
            return f"missing {name}"
        if kind == "mean":
            return f"{name} rate" if _binary(self.frame[column]) else f"mean {name}"
        return f"total {name}"

    def _by_data(self, label: str, fig: Dict[str, Any], segment: str, first: bool = True) -> Optional[tuple]:
        """``(True, None)`` when the data gives *label* this figure,
        ``(False, reason)`` when it gives it only to another group or the
        whole file, None when the data cannot say."""
        column = self.groups.get(label)
        if column is None or self.frame is None:
            return None
        frame = self.frame
        measures = [c for c in self._measures(segment) if c != column]
        counts = bool(_COUNT_WORDS.search(segment)) and not measures
        if not measures and not counts:
            return None
        keys = frame[column].astype(str).str.strip()
        own = self._stats(frame[keys == label], measures, counts)

        def hit(candidate: float) -> bool:
            return figure_matches(fig["value"], fig["raw"], candidate, fig["approx"], fig["high"])

        if any(hit(v) for v in own.values()):
            return True, None
        if _DIFFERENCE_WORDS.search(segment):
            return None                                  # "0.03 higher": a gap, not a value
        total = self._stats(frame, measures, counts)
        for stat, value in total.items():
            if hit(value) and stat in own:
                what = self._describe(stat, segment)
                scope = "overall" if stat[0] == "mean" else "total"
                what = what[len("total "):] if what.startswith("total ") else what
                return False, (f"{fig['raw']} is the {scope} {what}, not {label}'s ({_fmt(own[stat])})")
        others = keys.unique()
        if not first or len(others) > ATTRIBUTION_MAX_GROUPS:
            return None
        for other in others:
            if other == label or other.lower() in _GENERIC_VALUES:
                continue
            theirs = self._stats(frame[keys == other], measures, counts)
            for stat, value in theirs.items():
                if hit(value) and stat in own:
                    return False, (f"{fig['raw']} is {other}'s {self._describe(stat, segment)}, "
                                   f"not {label}'s ({_fmt(own[stat])})")
        return None

    # -- the result ----------------------------------------------------------
    def _by_table(self, label: str, fig: Dict[str, Any], segment: str) -> Optional[tuple]:
        keyed = self.row_keys.get(label)
        if not keyed:
            return None

        def hit(candidate: Optional[float]) -> bool:
            return candidate is not None and figure_matches(fig["value"], fig["raw"], candidate,
                                                            fig["approx"], fig["high"])

        own_rows = {i for i, _c in keyed}
        if any(hit(_as_number(v)) for i in own_rows for v in self.rows[i].values()):
            return True, None
        if _DIFFERENCE_WORDS.search(segment):
            return None
        key_columns = {c for _i, c in keyed}
        for j, row in enumerate(self.rows):
            if j in own_rows:
                continue
            for col, value in row.items():
                if not hit(_as_number(value)):
                    continue
                mine = next((_as_number(self.rows[i].get(col)) for i in sorted(own_rows)
                             if _as_number(self.rows[i].get(col)) is not None), None)
                if mine is None:
                    continue
                other = next((str(row[k]).strip() for k in key_columns if isinstance(row.get(k), str)), None)
                if other is None:
                    continue
                whose = f"the total {col}" if other.lower() in _TOTAL_LABELS else f"{other}'s {col}"
                return False, f"{fig['raw']} is {whose}, not {label}'s ({_fmt(mine)})"
        return None

    def check(self, claim: str, restated: Sequence[float] = ()) -> Optional[ClaimFinding]:
        """The first figure *claim* gives an entity that is not the entity's."""
        labels = set(self.row_keys) | set(self.groups)
        if not labels:
            return None
        figures = attached_figures(claim, restated)
        if not figures:
            return None
        spans = _label_spans(claim, labels)
        for (_start, _end, label), fig, segment, first in _pairs(claim, spans, figures):
            verdict = self._by_data(label, fig, segment, first)
            if verdict is None and first:
                verdict = self._by_table(label, fig, segment)
            if verdict is None or verdict[0]:
                continue
            column = self.groups.get(label)
            return ClaimFinding(
                claim=claim, verdict=Verdict.CONTRADICTED, check="figure_misattributed",
                detail=verdict[1], evidence_ref=f"data.{column}" if column else "execution_output")
        return None


def _binary(series: Any) -> bool:
    """A 0/1 column, whose mean is a rate."""
    import pandas as pd

    values = set(pd.to_numeric(series, errors="coerce").dropna().unique().tolist())
    return bool(values) and values <= {0.0, 1.0}


def check_figure_attribution(claim: str, flat: Dict[str, Any], used: Optional["DataUsed"] = None,
                             restated: Sequence[float] = ()) -> Optional[ClaimFinding]:
    """A figure the claim gives an entity must be that entity's (:class:`FigureAttribution`)."""
    return FigureAttribution(flat, used).check(claim, restated)


_READERS = frozenset({"read_csv", "read_parquet", "read_excel", "read_json", "read_table", "read_feather",
                      "read_csv_best_effort", "read_input_frame"})
#: Name words that say nothing about which dataset a variable holds.
_FRAME_WORDS = frozenset({"df", "data", "frame", "dataframe", "table", "tbl", "raw", "input", "src"})
_JOINED = "the join the code performed"
#: A join larger than this is not reproduced to check one figure.
JOIN_MAX_ROWS = 5_000_000


def _names(node: Any) -> Optional[str]:
    """The variable a merge operand is: ``orders``, ``orders[['a', 'b']]``, ``orders.copy()``."""
    for _ in range(6):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("copy", "dropna", "drop_duplicates", "reset_index", "astype"):
            node = node.func.value
        else:
            return None
    return None


def _keys(node: Any) -> Optional[List[str]]:
    value = _constant(node)
    if isinstance(value, str):
        return [value]
    if isinstance(node, (ast.List, ast.Tuple)):
        keys = [_constant(e) for e in node.elts]
        return keys if keys and all(isinstance(k, str) for k in keys) else None
    return None


def code_joins(code: Any) -> List[Dict[str, Any]]:
    """Every merge or join in the generated code: ``{"left", "right"}`` (the
    variable names, or None), ``"left_on"``/``"right_on"`` (column lists, None
    when the keys are not literal or are the index) and ``"how"``."""
    try:
        tree = ast.parse(str(code or ""))
    except (SyntaxError, ValueError):
        return []
    out: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("merge", "join")):
            continue
        owner = node.func.value
        args = list(node.args)
        if isinstance(owner, ast.Name) and owner.id in ("pd", "pandas"):
            if node.func.attr != "merge" or len(args) < 2:
                out.append({"left": None, "right": None, "left_on": None, "right_on": None, "how": None})
                continue
            left, right = args[0], args[1]
        else:
            if not args:
                continue
            left, right = owner, args[0]
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        how = _constant(kw["how"]) if "how" in kw else ("left" if node.func.attr == "join" else "inner")
        on = _keys(kw["on"]) if "on" in kw else None
        left_on = _keys(kw["left_on"]) if "left_on" in kw else on
        right_on = _keys(kw["right_on"]) if "right_on" in kw else on
        if node.func.attr == "join" or any(k in kw for k in ("left_index", "right_index")):
            left_on = right_on = None              # joined on an index: not reproduced
        out.append({"left": _names(left), "right": _names(right), "left_on": left_on, "right_on": right_on,
                    "how": how if isinstance(how, str) else None})
    return out


def _read_paths(code: Any) -> Dict[str, str]:
    """``{variable: path}`` for each ``v = pd.read_csv('path')`` in the code."""
    try:
        tree = ast.parse(str(code or ""))
    except (SyntaxError, ValueError):
        return {}
    out: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Call) and node.value.args:
            func = node.value.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            path = _constant(node.value.args[0])
            if name in _READERS and isinstance(path, str):
                out[node.targets[0].id] = path
    return out


def _frame_of(name: Optional[str], used: DataUsed, paths: Mapping[str, str]) -> Optional[int]:
    """Which frame of *used* the variable *name* holds: by the path it was
    read from, else by its name ("df_orders" holds the frame read as orders)."""
    if not name:
        return None
    path = paths.get(name)
    if path and path in used.sources:
        return used.sources.index(path)
    from avaloka.goal_analysis import word_stems

    words = {w for w in re.findall(r"[a-z]+", name.lower()) if w not in _FRAME_WORDS}
    if not words:
        return None
    hits = []
    for i, (label, _frame) in enumerate(used.frames):
        label_words = set(re.findall(r"[a-z]+", re.sub(r"\.\w+$", "", str(label).lower())))
        if any(word_stems(w) & word_stems(lw) for w in words for lw in label_words):
            hits.append(i)
    return hits[0] if len(hits) == 1 else None


def group_frames(used: Optional[DataUsed], code: Any) -> Optional[DataUsed]:
    """The frames a group figure is recomputed on.

    When the code joined two of the frames it read, the figure is one of that
    join, so the join is reproduced from the code's own keys and the figure is
    recomputed on it. Recomputed on the separate frames instead, the segment
    of the customers frame has no order amounts at all, and a correct
    "Enterprise $545.33 ... SMB $182.17" was withheld (rc6). A join that cannot
    be reproduced (index keys, computed keys, frames it cannot place, more than
    one join) checks nothing: None.
    """
    if used is None or len(used.frames) < 2:
        return used
    joins = code_joins(code)
    if not joins:
        return used
    if len(joins) != 1:
        return None
    join = joins[0]
    how, left_on, right_on = join["how"], join["left_on"], join["right_on"]
    if how not in ("inner", "left", "right", "outer") or not left_on or not right_on \
            or len(left_on) != len(right_on):
        return None
    paths = _read_paths(code)
    left, right = _frame_of(join["left"], used, paths), _frame_of(join["right"], used, paths)
    if len(used.frames) == 2 and how in ("inner", "outer") and (left is None) != (right is None):
        left, right = (left, 1 - left) if left is not None else (1 - right, right)
    elif len(used.frames) == 2 and how in ("inner", "outer") and left is None and right is None:
        left, right = 0, 1
    if left is None or right is None or left == right:
        return None
    import pandas as pd

    a, b = used.frames[left][1], used.frames[right][1]
    if not (set(left_on) <= set(map(str, a.columns)) and set(right_on) <= set(map(str, b.columns))):
        return None
    try:
        sizes = a.groupby(left_on).size().rename("a").to_frame().join(
            b.groupby(right_on).size().rename("b").rename_axis(left_on), how="inner")
        if float((sizes["a"] * sizes["b"]).sum()) + len(a) + len(b) > JOIN_MAX_ROWS:
            return None
        joined = pd.merge(a, b, left_on=left_on, right_on=right_on, how=how)
    except Exception:  # noqa: BLE001 - a join that cannot be reproduced checks nothing
        logger.debug("[claim-verifier] could not reproduce the code's join", exc_info=True)
        return None
    return DataUsed(frames=[(_JOINED, joined)])


_CONSTANT_TERMS = re.compile(
    r"\b(?:consistently|uniformly|constant|invariably|always|identical|the same|no variation|"
    r"does not vary|doesn't vary|remains)\b", re.IGNORECASE)
_ACROSS_TERMS = re.compile(r"\b(?:across|regardless|every|all|each|whether)\b", re.IGNORECASE)
_RATE_COLUMN = re.compile(r"rate|share|ratio|proportion|pct|percent|prob|mean|avg|average", re.IGNORECASE)
_RATE_WORDS = re.compile(r"\b(?:rate|share|proportion|percentage|probability|ratio)\b", re.IGNORECASE)


def _stem(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())[:6]


def check_constant_rate(claim: str, flat: Dict[str, Any], used: Optional[DataUsed],
                        restated: Sequence[float] = ()) -> Optional[ClaimFinding]:
    """"The rate is 1.0 across all combinations" must hold in the result and the data."""
    if not (_CONSTANT_TERMS.search(claim) and _ACROSS_TERMS.search(claim) and _RATE_WORDS.search(claim)):
        return None
    stated = None
    for match in _NUMBER.finditer(claim):
        value = _parse_number(match.group(0))
        if value is None:
            continue
        value = value / 100 if _is_percent(match.group(0)) else value
        if 0 <= value <= 1:
            stated, raw = value, match.group(0)
            break
    if stated is None:
        return None

    # 1) the result: every rate it holds must be the stated one.
    rates: Dict[str, List[float]] = {}
    for key, value in flat.items():
        m = _ROW.match(key)
        if not m or not _RATE_COLUMN.search(m.group(1)):
            continue
        number = _parse_number(str(value)) if isinstance(value, str) else value
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            rates.setdefault(m.group(1), []).append(float(number))
    for column, values in rates.items():
        off = [v for v in values if abs(v - stated) > 0.005]
        if off:
            return ClaimFinding(
                claim=claim, verdict=Verdict.CONTRADICTED, check="constant_rate_contradicted",
                detail=(f"says the rate is {raw} throughout, but {len(off)} of {len(values)} values of "
                        f"{column!r} in the result differ (from {min(values):g} to {max(values):g})"),
                evidence_ref=f"execution_output.{column}",
            )

    # 2) the data: a rate of 0 or 1 everywhere means every row has (or none
    # has) the outcome. If the outcome's column takes several values in the
    # data, the rates computed from it cannot all be that.
    if used is None or not used.frames or stated not in (0.0, 1.0):
        return None
    # The outcome is what the rate is of: "the readmission rate", or the
    # result's "readmission_rate" column. Its column in the data shares a stem.
    subjects = re.findall(r"([A-Za-z]{4,})[\s_-]+(?:rate|share|proportion|percentage|probability|ratio)\b",
                          claim + " " + " ".join(rates), flags=re.IGNORECASE)
    words = {_stem(w) for w in subjects} - {""}
    for _label, frame in used.frames:
        for column in frame.columns:
            if _stem(str(column)) not in words:
                continue
            shares = frame[column].dropna().astype(str).value_counts(normalize=True)
            if len(shares) < 2:
                continue
            shown = ", ".join(f"{k} {v:.1%}" for k, v in shares.head(3).items())
            return ClaimFinding(
                claim=claim, verdict=Verdict.CONTRADICTED, check="constant_rate_contradicted",
                detail=(f"says the rate is {raw} across every group, but {column!r} takes "
                        f"{len(shares)} values in the data ({shown}), so the rates computed from "
                        "it are not all " + raw),
                evidence_ref=f"data.{column}",
            )
    return None


def check_method_not_in_code(claim: str, flat: Dict[str, Any]) -> Optional[ClaimFinding]:
    """A model family named in the narrative that the executed script never used.

    "The gradient-boosted tree model achieves a test RMSE of 10.44" passes every
    other check when 10.44 really is in the result table. It is still false if
    the script that produced 10.44 fitted ``np.linalg.lstsq``: the reader is told
    which method earned the number, and that is the part they will repeat.

    Silent without a script to compare against, and silent when the words
    around the mention only suggest, compare or rule the method out.
    """
    code = flat.get(_CODE_KEY)
    if not isinstance(code, str) or not code.strip():
        return None
    script = _code_only(code)
    for family, (terms, markers) in _METHOD_FAMILIES.items():
        term = _asserted_mention(claim, terms)
        if term and not any(marker in script for marker in markers):
            return ClaimFinding(
                claim=claim, verdict=Verdict.CONTRADICTED, check="method_not_in_code",
                detail=(f"names a {family} model ({term!r}), but the executed code "
                        f"contains no {family} estimator"),
                evidence_ref=_CODE_KEY,
            )
    return None


_ONLY = re.compile(r"\b(?:only|just|solely|exclusively)\b", re.IGNORECASE)


def check_sheet_scope(claim: str, flat: Dict[str, Any], used: Optional[DataUsed],
                      restated: Sequence[float] = ()) -> Optional[ClaimFinding]:
    """"Only sheet X was analysed" when the code read more than one sheet."""
    if used is None or len(set(used.sheets)) < 2 or not _ONLY.search(claim):
        return None
    lowered = claim.lower()
    named = [s for s in dict.fromkeys(used.sheets) if s.lower() in lowered]
    if "sheet" not in lowered or len(named) != 1:
        return None
    others = [s for s in dict.fromkeys(used.sheets) if s not in named]
    return ClaimFinding(
        claim=claim, verdict=Verdict.CONTRADICTED, check="sheet_scope_contradicted",
        detail=(f"says only sheet {named[0]!r} was used, but the code read sheets "
                + ", ".join(repr(s) for s in dict.fromkeys(used.sheets))
                + f" ({', '.join(repr(s) for s in others)} as well)"),
    )


CHECKS = (
    check_contradicted_by_metrics,
    check_method_not_in_code,
    check_ordering_contradicted,
    check_causal_overreach,
    check_unsupported_significance,
    check_absolute_language,
    check_invented_numbers,
)

#: Checks that need the frames the run read; they run before CHECKS, since a
#: figure the data contradicts is the strongest finding a claim can get.
DATA_CHECKS = (check_sheet_scope, check_constant_rate, check_group_figure)


def _cites_figures(claim: str, restated: Sequence[float], labels: Iterable[str] = ()) -> bool:
    """*claim* states a figure of the result: not a prose count, a year, a
    number the question gave, or one inside a label ("paper chain kit 50's")."""
    spans = _label_spans(claim, labels) if labels else []
    return any(not any(s <= f["start"] < e for s, e, _l in spans) for f in attached_figures(claim, restated))


def verify_claims(narrative: str, evidence: Any, used: Optional[DataUsed] = None,
                  question: Optional[str] = None, code: Optional[str] = None,
                  datasets: Optional[DataUsed] = None) -> VerificationReport:
    """Check every claim in *narrative* against *evidence*, and against the
    frames the run read when *used* is given. Numbers *question* states are
    restatements, never findings. With *code* (the analysis that ran), the
    reply is also checked against the question (:func:`semantic_findings`)."""
    flat = _flatten(evidence or {})
    restated = question_numbers(question)
    operands = stated_operands(narrative, flat)
    report = VerificationReport()
    grouped: Any = _NO_VALUE
    attribution: Optional[FigureAttribution] = None
    extremes: Any = None
    claims = split_claims(narrative)
    report.claims_seen = len(claims)
    #: Indices of the claims a check resolved (D13.2): a finding, a figure
    #: found in the evidence, or a superlative or ratio judged true.
    resolved: set = set()
    prior: Optional[str] = None
    for n, claim in enumerate(claims):
        finding = None
        if extremes is not None and prior is not None and not any(f.claim == prior for f in report.findings):
            # "X sold the most. / It moved 3,648 pieces": the figure gives the
            # superlative before it its measure. The finding is X's sentence's.
            earlier = extremes.carried_superlative(claim, prior)
            if earlier is not None:
                report.findings.append(earlier)
                resolved.add(n - 1)
        if _NUMBER.search(claim):
            # Whose figure it is comes first: a figure that is another entity's
            # is in the evidence, and every check below would pass it (work4).
            if attribution is None:
                attribution = FigureAttribution(flat, used)
            finding = attribution.check(claim, restated)
        for data_check in () if finding is not None else DATA_CHECKS:
            frames = used
            if data_check is check_group_figure:
                if grouped is _NO_VALUE:          # the join is reproduced once, when a claim needs it
                    grouped = group_frames(used, code) if _COUNT_WORDS.search(claim) else _NO_VALUE
                frames = None if grouped is _NO_VALUE else grouped
            finding = data_check(claim, flat, frames, restated=restated)
            if finding is not None:
                break
        if extremes is None:
            from app.agents.claim_extremes import Extremes
            extremes = Extremes(flat, used, code)
        if finding is None:
            # A superlative, a ratio or a trend checked on all of what it is about (claim_extremes).
            finding = extremes.check(claim, prior)
        figures_found = False
        for check in () if finding is not None else CHECKS:   # ordered: strongest verdict wins
            if check is check_invented_numbers:
                # A multiple a check judged ("about 1.4 times the next") is no figure of the result.
                said = claim
                for start, end in extremes.judged_spans.get(claim, ()):
                    said = said[:start] + " " * (end - start) + said[end:]
                finding = check(said, flat, restated=restated, operands=operands)
                if finding is not None:
                    finding.claim = claim
                else:
                    figures_found = _cites_figures(said, restated, extremes.labels)
            elif check is check_absolute_language:
                finding = check(claim, flat, used=used)
            else:
                finding = check(claim, flat)
            if finding is not None:
                break
        if finding is not None:
            report.findings.append(finding)
        if finding is not None or figures_found:
            resolved.add(n)
        prior = claim
    _add_semantic_findings(report, narrative, flat, used, question, code, restated, datasets)
    # A check against the question flags a claim too; one it flags is one it checked.
    flagged = {f.claim[:400] for f in report.findings}
    resolved |= {n for n, claim in enumerate(claims) if claim[:400] in flagged}
    if extremes is not None:
        # A superlative or ratio judged true, the one its next claim completed among
        # them ("It moved 3,648 pieces"); and not one over a cut nothing confirmed,
        # whatever its figure: that one was not checked (D13.1).
        resolved |= {n for n, claim in enumerate(claims) if claim in extremes.resolved}
        report.unconfirmed = [c for c in claims if c in extremes.unconfirmed and c[:400] not in flagged]
        resolved -= {n for n, claim in enumerate(claims) if claim in report.unconfirmed}
        report.caveats += list(dict.fromkeys(extremes.unconfirmed[c] for c in report.unconfirmed))
    report.claims_checked = min(len(claims), len(resolved) + len(flagged - {c[:400] for c in claims}))
    if report.problems:
        logger.warning("[claim-verifier] %s", report.summary())
    return report


def _question(state: Mapping[str, Any]) -> str:
    """The user's last message: the question the reply answers, in the
    user's own words (app/agents/user_text). With the web chat's appended
    "[Analysis context]" block, every column of every dataset was "in the
    question" (#588)."""
    return last_user_text(state)


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

    # The analysis step decides what may be verified (app/agents/analysis_outcome).
    # A failed step left a failure message, not an answer: verifying its
    # sentences against the validator's dry run once produced "I checked 4
    # claims ... all of them hold" under "Expected 2 fields in line 4". A step
    # that succeeded with no rows left nothing a number could be checked
    # against. In both cases the report says so and verifies nothing.
    outcome = analysis_outcome(state)
    if outcome in NOTHING_TO_VERIFY:
        return _not_verified(outcome)

    # Only the executed result is evidence: output_json, or in the ETL graph the
    # CSV data URI on output_file_data (output_json is built later by the API).
    # The validator's dry run on the sample (execution_output_data) is not the
    # result, and a figure found only there is not one the analysis produced.
    # A turn with no execution step (training, a reused artifact) may still
    # carry result rows; those count, the dry run never does.
    execution_output = result_records(state)
    result_shape = None
    if isinstance(execution_output, list) and execution_output and isinstance(execution_output[0], dict):
        result_shape = {"rows": len(execution_output), "columns": len(execution_output[0])}

    evidence = {
        "evaluation_report": state.get("evaluation_report"),
        "integrity_report": state.get("integrity_report"),
        "execution_output": execution_output,
        "result_shape": result_shape,
        "training_metrics": state.get("training_metrics"),
        "training_result": _training_evidence(state),
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
        joined = _join_claims_report(state, narrative)
        if joined is not None:
            return joined
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

    try:
        used = data_used(state) if outcome in (ANSWERED, PARTIAL) else None
    except Exception:  # noqa: BLE001 - recomputation is extra evidence, never a crash
        logger.debug("[claim-verifier] could not load the frames the run read", exc_info=True)
        used = None
    # Attached after the guard above on purpose: ``generated_code`` survives in
    # state from an earlier turn, so on its own it is not evidence that this
    # turn ran anything. It is only offered when the result table came from the
    # script and no training agent contributed -- a model trained by the MTA is
    # described by training_metrics, not by the coder's script, and comparing
    # its narrative to that script would flag a true sentence.
    code = state.get("generated_code")
    if (isinstance(code, str) and code.strip() and execution_output
            and state.get("training_metrics") is None
            and state.get("evaluation_report") is None):
        evidence[_CODE_KEY] = code

    report = verify_claims(narrative, evidence, used, question=_question(state), code=_code(state),
                           datasets=_lazy_datasets(state, narrative, used))
    out = report.as_dict()
    out["analysis"] = outcome
    if outcome == ANSWERED:
        # The CLI's gate (#542): PASS only when the goal was answered AND its
        # figures reconcile. Here: a successful step with rows, and no cited
        # number missing from them or contradicted by them.
        reconciled = not any(f.check in _RECONCILIATION_CHECKS for f in report.problems)
        out["goal_answered"] = True
        out["claims_reconciled"] = reconciled
        out["verified"] = reconciled and report.safe_to_present
    elif outcome == PARTIAL:
        # Its figures are checked against the tables it did compute, but part
        # of the question has no figure at all: never "answered", never verified.
        out["goal_answered"] = False
        out["claims_reconciled"] = not any(f.check in _RECONCILIATION_CHECKS for f in report.problems)
        out["verified"] = False
    # A breakdown into categories the code assigned by keyword matching
    # (app/agents/derived_categories): its figures reconcile with the result
    # whatever the keywords missed, so it is judged on its method instead.
    derived = derived_categories.review(_code(state), used.frames if used else (), execution_output)
    if derived is not None:
        out["derived_categories"] = derived
        if derived["withheld"]:
            out["safe_to_present"] = out["verified"] = False
    if outcome in (ANSWERED, PARTIAL):
        # Rows the question did not ask to drop, which the answer does not
        # mention: a scope line, and "scoped" in place of "verified".
        out = apply_filter_scope(out, state, narrative, _code(state), used)
        # A cut the code made (the top 10 by revenue) is listed the same way,
        # in the code's words (rc18d).
        out = apply_truncation_scope(out, state, narrative, _code(state), execution_output)
        # Repairs the replan loop made to the data before the run that
        # produced these figures (app/agents/replan.py): one plain line each,
        # with its counts, whatever the narrator said about them.
        if state.get("replan_transforms"):
            from app.agents.replan_disclosure import apply_repair_disclosure

            out = apply_repair_disclosure(out, state)
    if report.unconfirmed and "verified" in out:
        # A superlative over a cut that no recompute confirmed: never verified.
        out["verified"] = False
    return {
        "verification_report": out,
        "verification_safe_to_present": out["safe_to_present"],
    }


#: The checks that compare a cited number with the result's values.
_RECONCILIATION_CHECKS = frozenset({
    "invented_number", "ordering_contradicted", "contradicted_by_metrics",
    "group_figure_mismatch", "constant_rate_contradicted", "sheet_scope_contradicted", "figure_misattributed",
})

#: The checks of a reply against the question (:func:`semantic_findings`).
_SEMANTIC_CHECKS = frozenset({"condition_mismatch", "driver_ranking_contradicted", "join_claim_contradicted",
                              "raw_coefficient_importance", "identifier_as_predictor",
                              "importance_not_association", "driver_figure_mismatch"})
#: A reply that answers another question does not reconcile either.
_RECONCILIATION_CHECKS = _RECONCILIATION_CHECKS | _SEMANTIC_CHECKS

#: Checks whose reason the reply gives for what it held back: a figure given to
#: the wrong entity says whose it is ("201 is the total missing BMI, not
#: Self-employed's (44)").
_EXPLAINED_CHECKS = _SEMANTIC_CHECKS | {"figure_misattributed"}

#: A superlative, a trend or a range checked on all of what it is about
#: (claim_extremes): each withholds its claim and says why.
_EXTREME_CHECKS = frozenset({"superlative_contradicted", "superlative_of_subset", "trend_contradicted",
                             "range_contradicted", "ratio_contradicted", "threshold_contradicted",
                             "threshold_of_subset"})
_RECONCILIATION_CHECKS = _RECONCILIATION_CHECKS | _EXTREME_CHECKS
_EXPLAINED_CHECKS = _EXPLAINED_CHECKS | _EXTREME_CHECKS

#: Checks whose finding is about a number: any verdict from them withholds it.
_NUMERIC_CHECKS = frozenset({"invented_number"}) | _RECONCILIATION_CHECKS

_NOT_VERIFIED_REASONS = {
    FAILED: "the analysis step failed, so there is no result to check claims against",
    NO_RESULT: "the analysis produced no result to check claims against",
    REFUSED: "the request was declined, so nothing ran and there is nothing to check",
}


def _not_verified(outcome: str) -> Dict[str, Any]:
    """The report for a turn whose analysis produced nothing to verify.

    Nothing is checked and nothing is called supported. ``safe_to_present`` is
    False because no figure in the reply is backed by a result; the reply that
    goes out is the failure itself, which the chat voice words plainly.
    """
    return {
        "verification_report": {
            "safe_to_present": False,
            "claims_checked": 0,
            "n_problems": 0,
            "findings": [],
            "analysis": outcome,
            "goal_answered": False,
            "claims_reconciled": None,
            "verified": False,
            "skipped": _NOT_VERIFIED_REASONS[outcome],
        },
        "verification_safe_to_present": False,
    }


# --------------------------------------------------------------------------- #
# Does the answer answer the question?
# --------------------------------------------------------------------------- #
# Every check above asks whether a figure matches what the code computed. On
# rc5 a figure matched and the answer was still wrong. Asked for the share of
# encounters readmitted within 30 days, the code counted readmitted == '>30';
# "approximately 0.35" was in the result, so "I checked 1 claim ... all of
# them hold" (the truth is 0.1116). The checks below compare the reply with
# the question and with the data: the condition the code filtered on, a
# ranking of drivers, a join said to match nothing, and a model metric quoted
# without its baseline. Each flags only a clear contradiction.

#: Words that put a number into a condition: "within 30 days" is ``< 30``.
#: Longer phrases first, so "no more than" is not read as "more than".
_CONDITION_WORDS = (
    ("no more than", "<="), ("not more than", "<="), ("at most", "<="), ("up to", "<="),
    ("no less than", ">="), ("not less than", ">="), ("at least", ">="),
    ("within", "<"), ("under", "<"), ("below", "<"), ("less than", "<"), ("fewer than", "<"),
    ("shorter than", "<"), ("before", "<"),
    ("more than", ">"), ("over", ">"), ("above", ">"), ("greater than", ">"),
    ("longer than", ">"), ("after", ">"), ("beyond", ">"), ("exceeding", ">"),
    ("<=", "<="), (">=", ">="), ("<", "<"), (">", ">"),
)
_CONDITION_OPS = {w: op for w, op in _CONDITION_WORDS}
_CONDITION = re.compile(
    r"(?<![\w<>])(?P<word>" + "|".join(re.escape(w) for w, _ in _CONDITION_WORDS) + r")"
    r"\s*(?P<num>\d+(?:\.\d+)?)(?![\d.]\d)(?P<unit>\s+[A-Za-z]+)?", re.IGNORECASE)
#: A category literal that is itself a comparison: readmitted's '<30' / '>30'.
_COMPARATOR_LITERAL = re.compile(r"^\s*([<>]=?)\s*(\d+(?:\.\d+)?)\s*$")
_AST_OPS = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
_FILTER_METHODS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
#: Module names whose attributes are functions, never columns.
_NOT_FRAMES = frozenset({"np", "pd", "numpy", "pandas", "math", "re", "os", "self", "datetime", "dt"})
#: Category values that are also ordinary words, so a question using the word
#: is not asking for the category.
_GENERIC_VALUES = frozenset({"other", "others", "unknown", "none", "nan", "null", "yes", "true", "false",
                             "all", "any", "total", "missing", "n/a", "na", "not", "and"})
_NO_VALUE = object()


def _direction(op: str) -> int:
    return -1 if op in ("<", "<=") else 1 if op in (">", ">=") else 0


def question_conditions(question: Any) -> List[tuple]:
    """``(op, number, phrase)`` for each numeric condition the question states.

    "readmitted within 30 days" -> ``("<", 30.0, "within 30 days")``. A
    question without such a phrase has no conditions, and nothing is checked.
    """
    out = []
    for match in _CONDITION.finditer(str(question or "")):
        op = _CONDITION_OPS[match.group("word").lower()]
        phrase = match.group(0).strip()
        out.append((op, float(match.group("num")), phrase))
    return out


def _constant(node: Any) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _constant(node.operand)
        if isinstance(value, (int, float)):
            return -value
    return _NO_VALUE


def _column_of(node: Any, bare: bool = False) -> Optional[str]:
    """The column an expression reads: ``df['a']``, ``df.a``,
    ``df['a'].astype(str).str.strip()``, ``pd.to_numeric(df['a'])``; in a
    ``.query`` string (``bare``), a plain name."""
    for _ in range(12):
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                return sl.value
            return None
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                    and func.value.id in _NOT_FRAMES:
                if not node.args:
                    return None
                node = node.args[0]
                continue
            if isinstance(func, ast.Attribute):
                node = func.value
                continue
            return None
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return None if node.value.id in _NOT_FRAMES else node.attr
            node = node.value
            continue
        if bare and isinstance(node, ast.Name):
            return node.id
        return None
    return None


def _compare_filters(node: ast.Compare, bare: bool) -> List[tuple]:
    out = []
    left = node.left
    for op, right in zip(node.ops, node.comparators):
        sym = _AST_OPS.get(type(op))
        if sym:
            column, value = _column_of(left, bare), _constant(right)
            if column is not None and value is not _NO_VALUE:
                out.append((column, sym, value))
            else:
                column, value = _column_of(right, bare), _constant(left)
                if column is not None and value is not _NO_VALUE:
                    out.append((column, _FLIP[sym], value))
        left = right
    return out


def code_filters(code: Any) -> List[tuple]:
    """``(column, op, value)`` for every comparison of a column with a literal
    in the generated code: ``df[df['readmitted'] == '>30']`` gives
    ``("readmitted", "==", ">30")``. ``op`` is a comparison, ``in`` (isin) or
    ``contains`` (``.str.contains`` / ``startswith``)."""
    try:
        tree = ast.parse(str(code or ""))
    except (SyntaxError, ValueError):
        return []
    out: List[tuple] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            out.extend(_compare_filters(node, bare=False))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            attr = node.func.attr
            if attr in _FILTER_METHODS:
                column, value = _column_of(node.func.value), _constant(node.args[0])
                if column is not None and value is not _NO_VALUE:
                    out.append((column, _FILTER_METHODS[attr], value))
            elif attr == "isin" and isinstance(node.args[0], (ast.List, ast.Tuple, ast.Set)):
                column = _column_of(node.func.value)
                values = tuple(v for v in map(_constant, node.args[0].elts) if v is not _NO_VALUE)
                if column is not None and values:
                    out.append((column, "in", values))
            elif attr in ("contains", "startswith", "fullmatch", "match"):
                column, value = _column_of(node.func.value), _constant(node.args[0])
                if column is not None and isinstance(value, str):
                    out.append((column, "contains", value))
            elif attr == "query":
                text = _constant(node.args[0])
                if isinstance(text, str):
                    try:
                        expr = ast.parse(text.replace("`", ""), mode="eval")
                    except (SyntaxError, ValueError):
                        continue
                    for sub in ast.walk(expr):
                        if isinstance(sub, ast.Compare):
                            out.extend(_compare_filters(sub, bare=True))
    return out


def _comparator_facts(filters: Sequence[tuple]) -> List[tuple]:
    """``(direction, number, shown)``: each filter that keeps rows above or
    below a number, as a numeric comparison or a literal like ``'>30'``."""
    facts = []
    for column, op, value in filters:
        if op in ("<", "<=", ">", ">=") and isinstance(value, (int, float)):
            facts.append((_direction(op), float(value), f"{column} {op} {value:g}"))
            continue
        literals = [value] if op in ("==", "contains") else list(value) if op == "in" else []
        parsed = [m for m in (_COMPARATOR_LITERAL.match(v) for v in literals if isinstance(v, str)) if m]
        directions = {_direction(m.group(1)) for m in parsed}
        if parsed and len(directions) == 1 and len({m.group(2) for m in parsed}) == 1:
            shown = " or ".join(repr(m.group(0).strip()) for m in parsed)
            facts.append((directions.pop(), float(parsed[0].group(2)), f"{column} {shown}"))
    return facts


def _frame_values(used: Optional["DataUsed"], column: str, limit: int = 200) -> List[str]:
    for _label, frame in (used.frames if used else []):
        if column in frame.columns:
            values = frame[column].dropna().astype(str).str.strip().unique()
            return [str(v) for v in values] if len(values) <= limit else []
    return []


def _has_figure(claim: str, restated: Sequence[float]) -> bool:
    for match in _NUMBER.finditer(claim):
        value = _parse_number(match.group(0))
        if value is None or _restates(value, restated):
            continue
        if not _is_percent(match.group(0)) and 1900 <= value <= 2100 and value.is_integer():
            continue
        return True
    return False


def condition_mismatch(question: Any, code: Any, used: Optional["DataUsed"] = None) -> Optional[str]:
    """Why the code's filter answers a different question, or None.

    Flags only a clear contradiction: the question states "within 30 days"
    (``< 30``) and the code kept ``> 30`` and nothing below 30; or the
    question names a category of a column the code filtered to another
    category, which the question never mentions. A question with no explicit
    condition is never flagged.
    """
    if not code or not str(question or "").strip():
        return None
    filters = code_filters(code)
    if not filters:
        return None
    facts = _comparator_facts(filters)
    for op, number, phrase in question_conditions(question):
        same = [f for f in facts if abs(f[1] - number) <= 1e-9 * max(1.0, abs(number))]
        if not same or any(f[0] == _direction(op) for f in same):
            continue
        return f"the analysis counted {same[0][2]}, but you asked about {phrase}"
    lowered = str(question).lower()

    def _asks(value: str) -> bool:
        return bool(re.search(rf"(?<!\w){re.escape(value.lower())}(?!\w)", lowered))

    for column, op, value in filters:
        if op not in ("==", "contains") or not isinstance(value, str) or _COMPARATOR_LITERAL.match(value):
            continue
        if not value.strip() or _asks(value.strip()):
            continue
        kept = {str(v).strip().lower() for c, o, v in filters if c == column and o in ("==", "contains", "in")
                for v in (v if isinstance(v, tuple) else (v,))}
        asked = [v for v in _frame_values(used, column)
                 if len(v) >= 3 and v.lower() not in _GENERIC_VALUES and v.lower() not in kept and _asks(v)]
        if asked:
            return f"the analysis counted {column} = {value!r}, but you asked about {asked[0]!r}"
    return None


# --- drivers ------------------------------------------------------------------
#: "the strongest driver", "Top predictor:", "the leading risk factor", "ranks
#: first". A singular one is about the column its own clause names; a plural
#: one ("the top factors are ...") is about every column the claim lists.
_STRONGEST = re.compile(
    r"(?<![\w-])(?:strongest|most important|most predictive|most influential|most powerful|most significant|"
    r"biggest|largest|highest|greatest|main|primary|principal|chief|dominant|foremost|key|top|top-ranked|"
    r"leading|single best|best|number[- ]one|#1)\s+(?:[\w-]+\s+){0,2}?(?P<noun>predictors?|drivers?|factors?|"
    r"features?|associations?|influences?|contributors?|determinants?|signals?|importance|impact)\b"
    r"|\b(?:most strongly|strongest|most closely)\s+(?:associated|correlated|linked|related)\b"
    r"|\b(?:ranks?|ranked|comes?|came) first\b|\btops? the (?:list|ranking)\b",
    re.IGNORECASE)
#: "X matters little", and the list forms of it: "**Lower-impact factors**:
#: smoking status, age, ...", "minor predictors include ...".
_WEAKEST = re.compile(
    r"\b(?:very low|low|lower|lowest|least|less|minor|minimal|negligible|little|weak|weaker|weakest|tiny|small|"
    r"smaller|limited|no real|no meaningful)[\s-]+(?:(?!risk\b)[\w-]+\s+)?(?:importance|important|impact|influence|"
    r"influential|association|effect|predictive|relationship|correlation|contribution|contributors?|role|"
    r"factors?|predictors?|drivers?|features?|signals?)\b"
    r"|\bmatters? (?:little|less|least)\b|\bbarely matters?\b|\bcontribut\w* (?:little|less|least|minimally)\b",
    re.IGNORECASE)
_ORDINAL_VALUES = {"second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}
_ORDINAL = r"(?:second|2nd|third|3rd|fourth|4th|fifth|5th)"
#: A rank other than the first: "**Second**: work_type_children (0.50)",
#: "the third strongest predictor", "ranks second". "next" has no rank.
_ORDINAL_DRIVER = re.compile(
    rf"^\W*(?P<head>{_ORDINAL})\b(?:[\s-]+(?:place|strongest|most important|biggest|largest|top|leading|ranked))?"
    r"[\s*_]*[:–—-]"
    rf"|\b(?P<inline>{_ORDINAL}|next)[\s-]+(?:(?:strongest|most important|most predictive|most influential|"
    r"biggest|largest|top|leading|best|ranked|ranking)[\s-]+)?(?:[\w-]+\s+)?(?:predictor|driver|factor|feature|"
    r"association|contributor|determinant)\b"
    rf"|\branks? (?P<ranks>{_ORDINAL})\b",
    re.IGNORECASE)
#: Where a clause ends: what a singular "strongest X" can be about.
_CLAUSE_BREAK = re.compile(
    r"[;()\[\]]|,|\s[-–—]\s|\b(?:followed|then|while|whereas|but|ahead of|than|compared|versus|vs)\b",
    re.IGNORECASE)
_LIST_BREAK = re.compile(
    r"[;()\[\]]|\s[-–—]\s|\b(?:followed|then|while|whereas|but|ahead of|than|compared|versus|vs)\b",
    re.IGNORECASE)
#: Where a plural "the strongest predictors are ..." stops naming them: "...
#: are the work-type and smoking variables, with age and glucose level also
#: contributing" names two, not four (groq, rc7). On the left an order word
#: is part of the list ("Age, then heart disease, are the strongest ...").
_PLURAL_LEFT = re.compile(
    r";|\s[-–—]\s|\b(?:while|whereas|but|with|also|plus|although|though)\b|[.!?](?=\s|$)", re.IGNORECASE)
_PLURAL_RIGHT = re.compile(
    _PLURAL_LEFT.pattern + r"|\b(?:followed|then|ahead of|than|compared|versus|vs|before|next)\b", re.IGNORECASE)
#: A list said in order: "age, then heart disease", "followed by".
_ORDER_WORD = re.compile(r"\b(?:then|followed by|ahead of|before|next)\b", re.IGNORECASE)
#: "Other key factors: ..." follows the top ones; it does not claim the top.
#: "Smoking is a key risk factor" names one of several, not the top one.
_NOT_THE_TOP = re.compile(r"\b(?:other|additional|secondary|remaining|further|also|an?)\W*$", re.IGNORECASE)
_MEASURE_SYMBOL = {"pearson_r": "r", "eta": "eta", "cramers_v": "Cramer's V"}
#: A named driver is wrong only when its association is well below the top's.
DRIVER_RATIO = 0.6
#: A list with no order word ("the strongest are a, b and c") may not be in
#: order at all: only a column at under half the strength of one it precedes
#: is out of place.
DRIVER_LIST_RATIO = 0.5
#: ...and the top association is itself worth naming.
DRIVER_MIN_STRENGTH = 0.1
#: A stated rank ("second") is wrong when the column ranks this many places
#: away in the data, and its association differs from that rank's by more than
#: DRIVER_RANK_RATIO. Near-ties (0.135, 0.132, 0.128) are never flagged.
DRIVER_RANK_GAP = 3
DRIVER_RANK_RATIO = 0.8
#: A ranking said by comparison, with no "strongest" in it: "X and Y ... far
#: exceeding Z", "X ranks above Y", "X is a stronger predictor than Y",
#: "well above", "ahead of", "more ... than" (rc11 p2: "Hypertension and
#: smoking_status are close in rank, both far exceeding the next group
#: (heart_disease ~0.27)"). The columns before it are put above those after.
_COMPARATIVE = re.compile(
    r"\b(?:exceed\w*|outrank\w*|outweigh\w*|outperform\w*|surpass\w*|ahead of|above"
    r"|(?:more|stronger|greater|higher|larger|bigger)(?:\s+[\w-]+){0,3}?\s+than)\b",
    re.IGNORECASE)
#: Only a sentence about ranking predictors makes a comparison a ranking:
#: "glucose is higher than 200 in heart-disease patients" is not one.
_RANKING_CONTEXT = re.compile(
    r"\b(?:rank\w*|predict\w*|drivers?|driv(?:es|ing)|factors?|associat\w*|correlat\w*|importan\w*|"
    r"influen\w*|strong\w*|tiers?|signals?|contribut\w*|features?|variables?|top)\b", re.IGNORECASE)
#: Where the two sides of a comparison stop.
_COMPARE_BREAK = re.compile(
    r";|\s[-–—]\s|\b(?:while|whereas|but|although|though|followed|then)\b|[.!?](?=\s|$)|:", re.IGNORECASE)
_TOP_N_COUNTS = {"two": 2, "three": 3, "four": 4, "five": 5, "2": 2, "3": 3, "4": 4, "5": 5}
#: A list that says it is complete: "Top three: a, b, c", "the three
#: strongest predictors are ...". One with only an order ("followed by")
#: is a selection, and leaving a column out of it is not a claim.
_TOP_N = re.compile(
    r"\btop[\s-]+(?P<n>two|three|four|five|[2-5])\b"
    r"|\bthe\s+(?P<m>two|three|four|five|[2-5])\s+(?:strongest|most important|most predictive|leading|"
    r"biggest|top|main|key)\b", re.IGNORECASE)


def computed_drivers(question: Any, used: Optional["DataUsed"]) -> Optional[Dict[str, Any]]:
    """The associations of every column with the outcome a "what predicts /
    drives X" question names, measured on the frames the run read (the
    measures #554's swarm merge ranks with). None when the question is not
    such a question or its outcome is not a column of the data."""
    if used is None or not used.frames or not str(question or "").strip():
        return None
    from avaloka.goal_analysis import MIN_OUTCOME_ROWS, driver_ranking, driver_target, is_identifier_column

    for _label, frame in used.frames:
        target = driver_target(str(question), {}, frame)
        column = (target or {}).get("column")
        if not column or column not in frame.columns:
            continue
        value = target.get("value")
        stats, ranking, _skipped = driver_ranking(frame, column, value)
        outcome = stats["outcome"]
        if outcome["binary"] and min(outcome["positives"], outcome["negatives"]) < MIN_OUTCOME_ROWS:
            return None
        drivers = [d for d in ranking if not d.get("leakage")]
        if len(drivers) < 2:
            return None
        # Row identifiers are never ranked (IDENTIFIER_RULE); a reply that
        # ranks one anyway is caught by name (rc11 p3: "the patient ID
        # variable (~0.189)").
        identifiers = [str(c) for c in frame.columns if str(c) != column
                       and is_identifier_column(str(c), frame[c])]
        # A category's labels name its column too: "living rurally" is
        # Residence_type, "female" is gender (rc10).
        values = {d["column"]: list(stats["columns"][d["column"]].get("values") or [])
                  for d in drivers if not stats["columns"][d["column"]].get("quantity")}
        return {"target": column if value is None else f"{column} = {value}", "column": column,
                "drivers": drivers, "values": values, "identifiers": identifiers}
    return None


def _named_columns(text: str, columns: Iterable[str]) -> List[str]:
    """Columns *text* names, as written or with spaces for underscores; a
    one-hot name (``ever_married_Yes``) names its column."""
    lowered = text.lower()
    out = []
    for name in columns:
        name = str(name)
        if len(name) < 3:
            continue
        pattern = re.escape(name.lower()).replace("_", "[_ ]").replace(r"\ ", "[_ ]")
        if re.search(rf"(?<![a-z0-9]){pattern}(?:_[\w-]+)?(?![a-z0-9])", lowered):
            out.append(name)
    return out


def _driver_text(claim: str) -> str:
    """The claim as the driver checks read it: no markdown emphasis, and a
    non-breaking hyphen ("Lower‑impact") is a hyphen."""
    return re.sub(r"[*`]", "", claim).replace("‑", "-").replace("‐", "-")


#: Words of a column name too common to name it alone (avg_glucose_level is
#: "glucose", never "level"; smoking_status is "smoking").
_GENERIC_NAME_WORDS = frozenset({
    "avg", "average", "mean", "ever", "type", "status", "level", "rate", "count", "num", "number", "total",
    "flag", "value", "score", "code", "category", "group", "class", "kind", "name", "date", "time", "index",
})
#: Labels too common to name their column ("Yes", "Unknown", "never smoked").
_GENERIC_LABELS = frozenset({"yes", "no", "true", "false", "unknown", "other", "none", "never", "formerly",
                             "missing", "null", "smoked"})


def _alike(word: str, want: str) -> bool:
    """*word* is an inflection of *want*: "rurally" of "Rural", "employment"
    of "Self-employed", "smoking" of "smoking"."""
    from avaloka.goal_analysis import word_stems

    if word_stems(word) & word_stems(want):
        return True
    n = 0
    for a, b in zip(word, want):
        if a != b:
            break
        n += 1
    return n >= 5 and n >= 0.75 * min(len(word), len(want))


def _loose_mentions(text: str, columns: Sequence[str],
                    values: Optional[Mapping[str, Sequence[str]]] = None) -> List[tuple]:
    """``(position, column)`` for a column *text* names by a word of its name
    no other column has ("being married" is ``ever_married``, "glucose level"
    ``avg_glucose_level``), or by one of its labels ("living rurally" is
    ``Residence_type``, "female" ``gender``). A reply names a driver the way
    a reader would, not the way the table spells it (rc10)."""
    from avaloka.goal_analysis import _ordered_tokens

    words = [(m.start(), m.group(0).lower()) for m in re.finditer(r"[A-Za-z]+", text)]
    keys: Dict[str, set] = {}
    for col in columns:
        for token in _ordered_tokens(col):
            if len(token) >= 4 and token not in _GENERIC_NAME_WORDS:
                keys.setdefault(token, set()).add(col)
        for label in (values or {}).get(col) or ():
            for token in re.findall(r"[A-Za-z]+", str(label).lower()):
                if len(token) >= 4 and token not in _GENERIC_LABELS and token not in _GENERIC_NAME_WORDS:
                    keys.setdefault(token, set()).add(col)
    out = []
    for pos, word in words:
        hits = {c for token, cols in keys.items() if len(cols) == 1 and _alike(word, token) for c in cols}
        if len(hits) == 1:
            out.append((pos, next(iter(hits))))
    return out


def _mentioned(text: str, columns: Iterable[str], exclude: str,
               values: Optional[Mapping[str, Sequence[str]]] = None) -> List[str]:
    """Columns *text* names, in the order it names them, through #561's
    shared stem rule (``column_mentions``): "work_type_children" and "work
    type" name ``work_type``, "heart disease" names ``heart_disease``; and
    by a distinctive word or label (:func:`_loose_mentions`)."""
    from avaloka.goal_analysis import column_mentions

    columns = [str(c) for c in columns]
    found = [(s, c) for s, _e, c in column_mentions(text, columns)]
    lowered = text.lower()
    found += [(max(lowered.find(c.lower()), lowered.find(c.lower().replace("_", " "))), c)
              for c in _named_columns(text, columns)]
    found += _loose_mentions(text, columns, values)
    found.sort(key=lambda sc: sc[0])
    return list(dict.fromkeys(c for _s, c in found if c != exclude))


def _clause(text: str, start: int, end: int, breaks: Any = _CLAUSE_BREAK) -> str:
    """The clause of *text* around ``[start, end)``."""
    lo = max((m.end() for m in breaks.finditer(text) if m.end() <= start), default=0)
    hi = min((m.start() for m in breaks.finditer(text) if m.start() >= end), default=len(text))
    return text[lo:hi]


def _after_header(text: str, end: int) -> Optional[int]:
    """Where a list header's items start: "Lower-impact factors: a, b" -> after the colon."""
    m = re.match(r"[^:\n]{0,30}?:", text[end:])
    return end + m.end() if m else None


def _ranking_text(drivers: Sequence[Mapping[str, Any]], n: int = 5) -> str:
    return ", ".join(f"{d['column']} ({_MEASURE_SYMBOL.get(d['measure'], d['measure'])}="
                     f"{abs(d['strength']):.3f})" for d in drivers[:n])


def _first_by(driver: Mapping[str, Any], target: str) -> str:
    """"age has the strongest association with stroke (r=0.245)": the
    ranking is univariate association and is called that, never importance."""
    measure = driver["measure"]
    return (f"{driver['column']} has the strongest association with {target} "
            f"({_MEASURE_SYMBOL.get(measure, measure)}={abs(driver['strength']):.3f})")


def _is_driver_claim(claim: str) -> bool:
    text = _driver_text(claim)
    return bool(_STRONGEST.search(text) or _WEAKEST.search(text) or _ORDINAL_DRIVER.search(text))


def _span(text: str, start: int, end: int, left: Any, right: Any) -> str:
    """*text* around ``[start, end)``, from the last *left* break before it to
    the first *right* break after it."""
    lo = max((m.end() for m in left.finditer(text) if m.end() <= start), default=0)
    hi = min((m.start() for m in right.finditer(text) if m.start() >= end), default=len(text))
    return text[lo:hi]


def check_driver_ranking(claim: str, ranked: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Why a claim's "strongest predictor(s)", "highest importance", "key
    drivers", "second" or "X matters little" contradicts the computed
    ranking (:func:`avaloka.goal_analysis.driver_ranking`), or None.

    Each phrase is checked against the columns it is about, not every column
    the claim names: in "Heart disease is the strongest driver of stroke, ...
    followed by older age", the strongest is heart disease, and naming age
    later does not make it right (groq, rc6); in "The strongest predictors of
    stroke are the work-type and smoking variables, with age and glucose level
    also contributing" the strongest are work type and smoking (groq, rc7).

    What passes: a claim that names the top column among its strongest, or
    one whose named columns are each within DRIVER_RATIO of it (near-ties);
    a list said in order passes when no column in it is put ahead of one
    that is clearly stronger."""
    if not ranked:
        return None
    drivers = ranked["drivers"]
    strength = {d["column"]: abs(d["strength"]) for d in drivers}
    rank = {d["column"]: i + 1 for i, d in enumerate(drivers)}
    top = drivers[0]
    if abs(top["strength"]) < DRIVER_MIN_STRENGTH:
        return None
    text = _driver_text(claim)
    ranking = _ranking_text(drivers)

    def mentioned(span: str) -> List[str]:
        return [c for c in _mentioned(span, strength, ranked["column"], ranked.get("values")) if c in strength]

    def out_of_order(ahead: str, behind: str, how: str) -> str:
        return (f"it puts {ahead} {how} {behind}, but {behind} ranks {rank[behind]} and {ahead} "
                f"{rank[ahead]} ({_MEASURE_SYMBOL.get(drivers[rank[behind] - 1]['measure'])}="
                f"{strength[behind]:.3f} against {strength[ahead]:.3f}); "
                f"{_first_by(top, ranked['target'])}; ranked by association: {ranking}")

    if not mentioned(text):
        return None

    # "the second strongest predictor" is a rank, never "the strongest".
    ordinals = list(_ORDINAL_DRIVER.finditer(text))
    plain = text
    for m in ordinals:
        plain = plain[:m.start()] + " " * (m.end() - m.start()) + plain[m.end():]

    for m in _STRONGEST.finditer(plain):
        if _NOT_THE_TOP.search(plain[:m.start()]):
            continue
        noun = (m.group("noun") or "").lower()
        plural = noun.endswith("s") or noun in {"importance", "impact"}
        header = _after_header(plain, m.end())
        if plural and header is not None:
            end = min((b.start() for b in _PLURAL_RIGHT.finditer(plain) if b.start() >= header), default=len(plain))
            scope = plain[header:end]
        elif plural:
            scope = _span(plain, m.start(), m.end(), _PLURAL_LEFT, _PLURAL_RIGHT)
        elif header is not None:
            scope = _clause(plain, header, header)
        else:
            scope = _clause(plain, m.start(), m.end())
        named = mentioned(scope) or mentioned(plain)
        if not named:
            continue
        if plural and header is None:
            # The order check reads the whole list, "age, then BMI, then ...".
            scope = _span(plain, m.start(), m.end(), _PLURAL_LEFT, _PLURAL_LEFT)
        elif plural:
            end = min((b.start() for b in _PLURAL_LEFT.finditer(plain) if b.start() >= header), default=len(plain))
            scope = plain[header:end]
        if top["column"] not in named:
            if max(strength[c] for c in named) >= DRIVER_RATIO * abs(top["strength"]):
                continue
            best = max(named, key=lambda c: strength[c])
            return (f"it ranks {best} among the strongest, but in the data {_first_by(top, ranked['target'])}; "
                    f"ranked by association: {ranking}")
        if not plural:
            # "Age is the strongest predictor, followed by X and Y" is an
            # ordered list too, from the top down (rc11 p3).
            chain = _span(plain, m.start(), m.end(), _PLURAL_LEFT, _PLURAL_LEFT)
            if _ORDER_WORD.search(chain):
                scope = chain
        # An ordered list: nothing in it ahead of a column clearly stronger.
        named = mentioned(scope) or named
        ordered = bool(_ORDER_WORD.search(scope))
        if ordered or (plural and len(named) >= 3):
            ratio = DRIVER_RATIO if ordered else DRIVER_LIST_RATIO
            for i, ahead in enumerate(named):
                behind = next((c for c in named[i + 1:] if strength[c] >= DRIVER_MIN_STRENGTH
                               and strength[ahead] < ratio * strength[c]), None)
                if behind:
                    return out_of_order(ahead, behind, "ahead of")

    # "X and Y ... far exceeding Z", "X ranks above Y": every column before
    # the comparison is put above every column after it, by the same rule.
    if _RANKING_CONTEXT.search(plain):
        for m in _COMPARATIVE.finditer(plain):
            left = plain[max((b.end() for b in _COMPARE_BREAK.finditer(plain) if b.end() <= m.start()),
                             default=0):m.start()]
            right = plain[m.end():min((b.start() for b in _COMPARE_BREAK.finditer(plain) if b.start() >= m.end()),
                                      default=len(plain))]
            above, below = mentioned(left), mentioned(right)
            for ahead in above:
                behind = next((c for c in below if c != ahead and strength[c] >= DRIVER_MIN_STRENGTH
                               and strength[ahead] < DRIVER_RATIO * strength[c]), None)
                if behind:
                    return out_of_order(ahead, behind, "above")

    # "A, followed by B" and "B follows A" are the same order, said either
    # way round: the same rule, for lists said without "strongest" (rc12 p3).
    if _RANKING_CONTEXT.search(plain):
        for m in _SUCCESSION.finditer(plain):
            left = plain[max((b.end() for b in _COMPARE_BREAK.finditer(plain) if b.end() <= m.start()),
                             default=0):m.start()]
            right = plain[m.end():min((b.start() for b in _COMPARE_BREAK.finditer(plain)
                                       if b.start() >= m.end() and b.start() > m.start()), default=len(plain))]
            above, below = (mentioned(left), mentioned(right)) if m.group("first") else (mentioned(right),
                                                                                       mentioned(left))
            for ahead in above:
                behind = next((c for c in below if c != ahead and strength[c] >= DRIVER_MIN_STRENGTH
                               and strength[ahead] < DRIVER_RATIO * strength[c]), None)
                if behind:
                    return out_of_order(ahead, behind, "ahead of")

    # "Top three: a, b, c" says nothing stronger is missing: a column left
    # out that is clearly stronger than one put in contradicts it. A list
    # with only an order ("followed by") is a selection, not a claim of
    # completeness.
    for m in _TOP_N.finditer(plain):
        n = _TOP_N_COUNTS[(m.group("n") or m.group("m")).lower()]
        header = _after_header(plain, m.end())
        if header is not None:
            end = min((b.start() for b in _PLURAL_LEFT.finditer(plain) if b.start() >= header),
                      default=len(plain))
            scope = plain[header:end]
        else:
            scope = _span(plain, m.start(), m.end(), _PLURAL_LEFT, _PLURAL_LEFT)
        named = mentioned(scope)
        if not named or len(named) > n:
            continue
        for kept in named:
            missing = next((d["column"] for d in drivers[:n] if d["column"] not in named
                            and strength[kept] < DRIVER_RATIO * strength[d["column"]]), None)
            if missing:
                return (f"it gives {', '.join(named)} as the top {n}, but leaves out {missing}, which ranks "
                        f"{rank[missing]} ({_MEASURE_SYMBOL.get(drivers[rank[missing] - 1]['measure'])}="
                        f"{strength[missing]:.3f} against {strength[kept]:.3f} for {kept}); "
                        f"{_first_by(top, ranked['target'])}; ranked by association: {ranking}")

    for m in ordinals:
        word = (m.group("head") or m.group("inline") or m.group("ranks") or "").lower()
        n = _ORDINAL_VALUES.get(word)
        if n is None or n > len(drivers):
            continue
        header = _after_header(text, m.end()) if m.group("head") is None else m.end()
        scope = _clause(text, header, header) if header is not None else _clause(text, m.start(), m.end())
        named = [c for c in mentioned(scope) if c in rank]
        if not named:
            continue
        column = named[0]
        at_n = abs(drivers[n - 1]["strength"])
        low, high = sorted((strength[column], at_n))
        if abs(rank[column] - n) >= DRIVER_RANK_GAP and (high == 0 or low / high < DRIVER_RANK_RATIO):
            return (f"it ranks {column} {word}, but in the data {column} ranks {rank[column]} in its association "
                    f"with {ranked['target']} ({strength[column]:.3f}, against {at_n:.3f} for "
                    f"{drivers[n - 1]['column']}); ranked by association: {ranking}")

    for m in _WEAKEST.finditer(plain):
        header = _after_header(plain, m.end())
        scope = plain[header:] if header is not None else _clause(plain, m.start(), m.end(), _LIST_BREAK)
        if top["column"] in mentioned(scope):
            return (f"it says {top['column']} matters little, but {top['column']} has the strongest association "
                    f"with {ranked['target']} in the data; ranked by association: {ranking}")
    return None


# --- figures given to columns ------------------------------------------------
# On a what-predicts question a figure beside a column is that column's
# strength, and the answer's measure of strength is its association with the
# outcome (#573). rc12 showed "work_type_Self‑employed (0.32) and
# heart_disease (0.27) are also strong" (raw coefficients; work type's eta is
# 0.098, heart disease's r 0.135), "Hypertension follows at ≈ 0.093" (r=0.128)
# and "residence type ... the last barely 0.003" (eta=0.015): each figure was
# in the result, none is the column's association.

#: A sentence about how strongly columns go with the outcome.
_STRENGTH_CONTEXT = re.compile(
    _RANKING_CONTEXT.pattern + r"|\b(?:follow\w*|next|highest|lowest|lower|higher|weaker|stronger|smaller|larger|"
    r"effects?|impact\w*|weights?|scores?|coefficients?|tier)\b", re.IGNORECASE)
#: Beside a figure: it is a rate, a mean, an amount in the column's units, not a strength.
_NOT_STRENGTH = re.compile(
    r"\b(?:rates?|share|proportion|percent\w*|probabilit\w*|odds|means?|average|median|years?|mg|times|per|"
    r"ratio|prevalence|incidence|cases?|patients|people|counts?|chance|likelihood)\b|%|\$", re.IGNORECASE)
#: A bound on a group of columns: "all other factors contribute less than 0.001".
_BOUND = re.compile(
    r"\b(?P<op>less than|lower than|smaller than|weaker than|below|under|at most|no more than|"
    r"more than|greater than|higher than|stronger than|above|over|exceed\w*|at least)\s*(?:[≈~]\s*)?"
    r"(?P<num>\d*\.\d+)(?![\d%])", re.IGNORECASE)
_UPPER_BOUND = re.compile(r"less|lower|smaller|weaker|below|under|at most|no more", re.IGNORECASE)
#: One figure for every column of the list before it: "each around 0.13–0.14".
_EACH = re.compile(r"\b(?:each|all|both|every one)\b", re.IGNORECASE)
#: "Next tier: a and b": below the top, so the top column is not in it.
_NEXT_TIER = re.compile(r"\b(?:next|second|middle)[\s-]+(?:tier|group|band|level)\b", re.IGNORECASE)
#: "A, followed by B" puts A first; "A follows B", "A trails B" puts B first.
_SUCCESSION = re.compile(
    r"\b(?P<first>followed by|trailed by|then)\b|\b(?P<second>follows|trails|comes after|behind)\b", re.IGNORECASE)
#: Importance #573 lets a reply report: permutation, or coefficients on scaled features.
_REPORTED_IMPORTANCE = frozenset({"permutation", "scaled_coefficients"})
_KEEP_LENGTH = str.maketrans({"*": " ", "`": " ", "‑": "-", "‐": "-"})


def _column_spans(text: str, ranked: Mapping[str, Any]) -> List[tuple]:
    """``(start, end, column)`` for each driver column *text* names, as
    :func:`_mentioned` reads them (#561's stems, the name with spaces, a
    one-hot name, a distinctive word or label)."""
    from avaloka.goal_analysis import column_mentions

    columns = [d["column"] for d in ranked["drivers"]]
    spans = [(s, e, c) for s, e, c in column_mentions(text, columns)]
    for name in _named_columns(text, columns):
        pattern = re.escape(name.lower()).replace("_", "[_ ]").replace(r"\ ", "[_ ]")
        spans += [(m.start(), m.end(), name) for m in
                  re.finditer(rf"(?<![a-z0-9]){pattern}(?:_[\w-]+)?(?![a-z0-9])", text.lower())]
    for pos, column in _loose_mentions(text, columns, ranked.get("values")):
        spans.append((pos, pos + re.match(r"[A-Za-z]+", text[pos:]).end(), column))
    return sorted(set(s for s in spans if s[2] != ranked["column"]))


def driver_figures(claim: str, ranked: Optional[Mapping[str, Any]],
                   restated: Sequence[float] = ()) -> List[Dict[str, Any]]:
    """Each figure *claim* gives a column as its strength, with whether it is
    that column's association: ``{"column", "figure", "matches"}``. A figure
    is a column's when the column is the last one named before it in the
    same sentence; "each" gives it to every column of the list."""
    if not ranked:
        return []
    text = claim.translate(_KEEP_LENGTH)
    if not _STRENGTH_CONTEXT.search(text):
        return []
    strength = {d["column"]: abs(d["strength"]) for d in ranked["drivers"]}
    spans = [s for s in _column_spans(text, ranked) if s[2] in strength]
    if not spans:
        return []
    from app.agents.claim_figures import respective

    bounds = {m.start("num") for m in _BOUND.finditer(text)}
    out: List[Dict[str, Any]] = []
    figures = attached_figures(text, restated)
    paired = respective(text, spans, figures)
    for k, fig in enumerate(figures):
        top = fig["high"] if fig["high"] is not None else fig["value"]
        if fig["start"] in bounds or _is_percent(fig["raw"]) or "." not in fig["raw"] or abs(top) > 1:
            continue
        before = [s for s in spans if s[1] <= fig["start"]]
        if not before:
            continue
        between = text[before[-1][1]:fig["start"]]
        after = " ".join(text[fig["end"]:].split()[:2])
        if len(between) > 80 or re.search(r"[;!?]|\.\s", between) or _NOT_STRENGTH.search(between) \
                or _NOT_STRENGTH.search(after):
            continue
        owners = [before[-1]]
        if k in paired:                    # "X and Y (a and b)": a is X's (claim_figures.respective)
            if paired[k] is None:
                continue
            owners = [paired[k]]
        elif _EACH.search(between):
            start = max((m.end() for m in re.finditer(r"[;:]", text[:before[-1][0]])), default=0)
            owners = [s for s in before if s[0] >= start]
        for _s, _e, column in owners:
            low = abs(fig["value"]) if fig["high"] is None else fig["value"]
            out.append({"column": column, "figure": fig,
                        "matches": figure_matches(low, fig["raw"], strength[column], fig["approx"], fig["high"])})
    return out


def _strength_text(driver: Mapping[str, Any]) -> str:
    return f"{_MEASURE_SYMBOL.get(driver['measure'], driver['measure'])}={abs(driver['strength']):.3f}"


def check_driver_figures(claim: str, ranked: Optional[Mapping[str, Any]], kind: Optional[str],
                         restated: Sequence[float] = ()) -> Optional[str]:
    """Why a figure *claim* gives a column is not that column's association
    with the outcome, a bound it puts on a group of columns does not hold, or
    a "next tier" holds the top column; or None.

    A figure may be another measure when the analysis fit a model whose
    importance can be reported (:data:`_REPORTED_IMPORTANCE`); its order is
    still checked by :func:`check_driver_ranking`."""
    if not ranked or abs(ranked["drivers"][0]["strength"]) < DRIVER_MIN_STRENGTH:
        return None
    drivers = {d["column"]: d for d in ranked["drivers"]}
    target, ranking = ranked["target"], _ranking_text(ranked["drivers"])
    if kind not in _REPORTED_IMPORTANCE:
        for pair in driver_figures(claim, ranked, restated):
            if pair["matches"]:
                continue
            column, fig = pair["column"], pair["figure"]
            said = claim[fig["start"]:fig["end"]].strip()
            whose = next((d for d in ranked["drivers"] if d["column"] != column
                          and figure_matches(abs(fig["value"]), fig["raw"], abs(d["strength"]), fig["approx"],
                                             fig["high"])), None)
            if whose is not None:
                return (f"{said} is {whose['column']}'s association with {target} ({_strength_text(whose)}), "
                        f"not {column}'s ({_strength_text(drivers[column])}); ranked by association: {ranking}")
            return (f"it gives {column} {said}, but {column}'s association with {target} is "
                    f"{_strength_text(drivers[column])}; ranked by association: {ranking}")
    text = claim.translate(_KEEP_LENGTH)
    if not _STRENGTH_CONTEXT.search(text):
        return None
    spans = [s for s in _column_spans(text, ranked) if s[2] in drivers]
    for m in _BOUND.finditer(text):
        bound = float(m.group("num"))
        if not 0 < bound <= 1:
            continue
        lo = max((b.end() for b in _COMPARE_BREAK.finditer(text) if b.end() <= m.start()), default=0)
        named = list(dict.fromkeys(c for s, _e, c in spans if lo <= s < m.start()))
        tol = figure_tolerance(bound, m.group("num"))
        upper = bool(_UPPER_BOUND.match(m.group("op")))
        for column in named:
            value = abs(drivers[column]["strength"])
            if (value > bound + tol) if upper else (value < bound - tol):
                return (f"it says {', '.join(named)} {m.group('op').lower()} {m.group('num')}, but {column}'s "
                        f"association with {target} is {_strength_text(drivers[column])}; "
                        f"ranked by association: {ranking}")
    top, second = ranked["drivers"][0], ranked["drivers"][1]
    if abs(second["strength"]) < DRIVER_RATIO * abs(top["strength"]):
        for m in _NEXT_TIER.finditer(text):
            header = _after_header(text, m.end())
            lo = header if header is not None else max(
                (b.end() for b in _PLURAL_LEFT.finditer(text) if b.end() <= m.start()), default=0)
            hi = min((b.start() for b in _PLURAL_LEFT.finditer(text) if b.start() >= max(lo, m.end())),
                     default=len(text))
            if any(c == top["column"] and lo <= s < hi for s, _e, c in spans):
                return (f"it puts {top['column']} in the {m.group(0).lower()}, but {top['column']} ranks 1 "
                        f"({_strength_text(top)}), clearly ahead of {second['column']} ({_strength_text(second)}); "
                        f"ranked by association: {ranking}")
    return None


def relabelled(claim: str, ranked: Optional[Mapping[str, Any]], restated: Sequence[float] = ()) -> Optional[str]:
    """*claim* with its "importance" called what it is, when every figure it
    gives a column is that column's association: "Age is the strongest
    predictor of stroke (importance ≈ 0.245)" is true of age's r=0.245, and
    is shown as "(association r ≈ 0.245)" rather than withheld."""
    pairs = driver_figures(claim, ranked, restated)
    if not pairs or not all(p["matches"] for p in pairs):
        return None
    drivers = {d["column"]: d for d in ranked["drivers"]}
    out = claim
    for pair in sorted({id(p["figure"]): p for p in pairs}.values(), key=lambda p: -p["figure"]["start"]):
        fig, driver = pair["figure"], drivers[pair["column"]]
        symbol = _MEASURE_SYMBOL.get(driver["measure"], driver["measure"])
        number = out[fig["start"]:fig["end"]]
        approx = re.search(r"(?:[≈~∼]|\b(?:about|around|roughly|approximately|approx\.?))[\s  ]*$",
                           out[:fig["start"]])
        start = approx.start() if approx else fig["start"]
        lead = ""
        if approx and out[start - 1:start].isspace():
            start -= len(out[:start]) - len(out[:start].rstrip())
            lead = " "
        out = f"{out[:start]}{lead}{symbol} {'≈' if approx else '='} {number}{out[fig['end']:]}"
    out = re.sub(r"\b(?:feature[\s-]+)?importances\b|\bweights\b", "associations", out, flags=re.IGNORECASE)
    return re.sub(r"\b(?:feature[\s-]+)?importance\b|\bweight\b", "association", out, flags=re.IGNORECASE)


# --- joins --------------------------------------------------------------------
_EMPTY_JOIN = re.compile(
    r"\bempty (?:result|join|table|set|output|frame)\b"
    r"|\bno (?:rows?|records?|keys?|values?|ids?|customers?|orders?|entries)?\s*"
    r"(?:match|matches|matched|matching|in common|overlap)\b"
    r"|\bno (?:matching|overlapping|common)\b"
    r"|\bnothing (?:would )?match(?:es)?\b|\bzero match(?:es)?\b"
    r"|\bmatch[_ ]?rate (?:of |is |= ?)?0(?:\.0+)?\s*%?(?![\d.]*\d)"
    r"|\b0\s?% match"
    r"|\b(?:do(?:es)?n['’]?t|do(?:es)? not|never) (?:match|overlap)\b"
    r"|\breturns? (?:nothing|no rows)\b|\b(?:zero|0) rows\b"
    # "Show a preview of the join on customer_id (will be empty in this sample)" (groq, rc6)
    r"|\b(?:will|would|should|is|are|be|was|were|comes? (?:back|out)|returns?|ends? up)\s+(?:be\s+)?empty\b",
    re.IGNORECASE)
#: A condition is not an assertion: "check whether the join is empty".
_HEDGED_JOIN = re.compile(r"\b(?:if|whether|unless|in case|could|might|may)\b", re.IGNORECASE)
_JOIN_WORDS = re.compile(r"\b(?:join|joined|joins|joining|merge|merged|match|matches|matching|link|linked|"
                         r"key|keys|overlap)\b", re.IGNORECASE)


def _key_values(series: Any) -> set:
    out = set()
    for value in series.dropna().unique():
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        text = str(value).strip().lower()
        if text:
            out.add(text)
    return out


def _asserts_empty_join(claim: str) -> bool:
    """*claim* says a join matches nothing, not only that it might."""
    if not _JOIN_WORDS.search(claim):
        return False
    for m in _EMPTY_JOIN.finditer(claim):
        lo = max((b.end() for b in _LIST_BREAK.finditer(claim) if b.end() <= m.start()), default=0)
        if not _HEDGED_JOIN.search(claim[lo:m.start()]):
            return True
    return False


def check_join_claim(claim: str, narrative: str, used: Optional["DataUsed"]) -> Optional[str]:
    """Why "a join on customer_id returns nothing" is false, or None: the key's
    values are counted in both frames, and any overlap contradicts it.

    The reason is about the join the claim itself names. rc16 withheld "match
    on the year of signup vs. the year extracted from order_date ... no
    overlapping rows" saying "a join on customer_id finds no match, but 2,000
    customer_id values appear in both": customer_id was taken from another
    sentence of the reply. A claim naming columns is checked on those columns
    (a year of one matched to the year of the other included), and the
    reply's other sentences name the key only for a claim that names none
    (its key written as code, which split_claims removes)."""
    if used is None or len(used.frames) < 2 or not _asserts_empty_join(claim):
        return None
    shared: Dict[str, List[tuple]] = {}
    for label, frame in used.frames:
        for column in frame.columns:
            shared.setdefault(str(column), []).append((label, frame))
    every = list(shared)
    shared = {c: fs for c, fs in shared.items() if len(fs) >= 2}
    # The columns of the join it says is empty: named before that is said
    # ("match on the year of signup vs. ... order_date, but ... no overlapping
    # rows"), not the figures it says cannot be computed after.
    said = _EMPTY_JOIN.search(claim)
    own = _join_columns(_driver_text(claim[:said.start()] if said else claim), every)
    keys = [c for c in own if c in shared]
    if own and not keys:
        return _year_join(claim, own, used)
    if not own:
        keys = _named_columns(narrative, shared) or (list(shared) if len(shared) == 1 else [])
    if len(keys) != 1:
        return None
    key = keys[0]
    (a_label, a), (b_label, b) = shared[key][:2]
    left, right = _key_values(a[key]), _key_values(b[key])
    both = left & right
    if not both:
        return None
    return (f"it says a join on {key} finds no match, but {len(both):,} {key} values appear in both "
            f"{a_label} ({len(left):,} distinct) and {b_label} ({len(right):,} distinct), "
            "so the join returns rows")


#: Words of a column name that do not name it alone ("the year", "the date").
_JOIN_GENERIC = frozenset({"year", "date", "time", "id", "key", "code", "num", "number", "value", "name"})


def _join_columns(text: str, columns: Sequence[str]) -> List[str]:
    """Columns *text* names as written (``order_date``, "order date"), or by
    every distinctive word of the name as a word of its own ("the year of
    signup" names ``signup_year``). Not by a word's stem: "the join ...
    returns no rows" does not name ``returned``."""
    out = list(_named_columns(text, columns))
    rest = text.lower()
    for column in out:                    # "order_date" names order_date, not also order_id
        rest = re.sub(re.escape(column.lower()).replace("_", "[_ ]"), " ", rest)
    words = set(re.findall(r"[a-z0-9]+", rest))
    for column in columns:
        tokens = [t for t in re.findall(r"[a-z0-9]+", str(column).lower()) if t not in _JOIN_GENERIC]
        if column not in out and tokens and all(len(t) >= 4 and t in words for t in tokens):
            out.append(column)
    return out


def _years(series: Any) -> Optional[Any]:
    """The years of a column: its values when they are years, else the year of its dates."""
    import pandas as pd

    numbers = pd.to_numeric(series, errors="coerce").dropna()
    if len(numbers) and numbers.between(1900, 2100).all() and (numbers % 1 == 0).all():
        return numbers.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return None
    dates = pd.to_datetime(series, errors="coerce", format="mixed")
    return dates.dt.year.dropna().astype(int) if dates.notna().mean() >= 0.8 else None


def _year_join(claim: str, columns: Sequence[str], used: "DataUsed") -> Optional[str]:
    """Why "matching the signup year to the order year finds nothing" is false:
    one column of each frame, compared by year, and a year in both."""
    if not re.search(r"\byears?\b", claim, re.IGNORECASE):
        return None
    sides = []
    for label, frame in used.frames:
        mine = [c for c in columns if c in map(str, frame.columns)]
        if len(mine) == 1:
            sides.append((label, frame, mine[0]))
    if len(sides) != 2 or sides[0][2] == sides[1][2]:
        return None
    years = [_years(frame[column]) for _label, frame, column in sides]
    if any(y is None for y in years):
        return None
    both = sorted(set(years[0]) & set(years[1]))
    if not both:
        return None
    (a_label, _a, a_col), (b_label, _b, b_col) = sides
    shown = ", ".join(str(y) for y in both[:5]) + (" ..." if len(both) > 5 else "")
    counts = "; ".join(f"{int((y == both[0]).sum()):,} of the {len(y):,} rows of {lab} have {col} in {both[0]}"
                       for y, (lab, _f, col) in zip(years, sides))
    return (f"it says matching the year of {a_col} to the year of {b_col} finds no match, but "
            f"{len(both):,} year{'s' if len(both) != 1 else ''} ({shown}) appear{'s' if len(both) == 1 else ''} "
            f"in both {a_label} and {b_label} ({counts})")


# --- a model metric without its baseline -------------------------------------
_METRIC_WORDS = re.compile(
    r"\b(?:accuracy|auc|roc|f1|precision|recall|r2|r²|r-squared|rmse|mae|mse|mape|log ?loss|"
    r"cross[- ]validation|cross[- ]validated|cv)\b", re.IGNORECASE)
_BASELINE_ADMITTED = re.compile(
    r"(?:\bnot\b|n['’]t\b|\bfails?\b|\bfailed\b|\bno better\b|\bworse\b|\bbelow\b|\bunderperform\w*|"
    r"\bbarely\b|\bmatches\b)\W+(?:[\w-]+\W+){0,8}?baseline", re.IGNORECASE)


def _metric_at(flat: Mapping[str, Any], prefix: str, metric: str) -> Optional[float]:
    for key in (f"{prefix}cv_mean", f"{prefix}metrics.{metric}.mean", f"{prefix}metrics.{metric}"):
        value = flat.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def baseline_caveat(narrative: str, flat: Mapping[str, Any]) -> Optional[str]:
    """The warning a reply owes when it quotes a model metric while the model
    did not beat its trivial baseline, on the holdout or in cross-validation.

    The holdout may beat the baseline while the folds do not (stroke: 0.963
    vs 0.957 on the holdout, 0.9477 vs 0.9497 over 5 folds), so every
    ``beats_baseline`` in the evidence counts, not only the first one.
    """
    failed = [k for k, v in flat.items() if k.lower().endswith("beats_baseline") and v is False]
    if not failed:
        return None
    if not any(_METRIC_WORDS.search(c) and _NUMBER.search(c) for c in split_claims(narrative)):
        return None
    if _BASELINE_ADMITTED.search(narrative or ""):
        return None
    key = sorted(failed, key=lambda k: "cross_validation" not in k)[0]
    prefix = key[: -len("beats_baseline")]
    metric = str(flat.get(f"{prefix}primary_metric") or "accuracy")
    score = _metric_at(flat, prefix, metric)
    baseline = flat.get(f"{prefix}baseline.{metric}")
    folds = flat.get(f"{prefix}n_folds")
    where = (f"across {folds}-fold cross-validation" if "cross_validation" in key and folds
             else "in cross-validation" if "cross_validation" in key else "on the holdout")
    detail = ""
    if score is not None and isinstance(baseline, (int, float)) and not isinstance(baseline, bool):
        detail = f": {where} its {metric} is {score:.4g}, against {float(baseline):.4g} for the trivial baseline"
    elif where:
        detail = f": it does not beat it {where}"
    return f"The model does not reliably beat the trivial baseline{detail}."


# --- raw coefficients as importance -------------------------------------------
#: What puts every feature on one scale before a coefficient is read: a
#: scaler, or dividing (or multiplying) by a standard deviation.
_SCALERS = frozenset({"StandardScaler", "RobustScaler", "MinMaxScaler", "MaxAbsScaler", "scale", "zscore",
                      "minmax_scale", "robust_scale", "maxabs_scale", "standardize", "standardise"})
#: A sentence presenting a figure as how much a column matters.
_IMPORTANCE_WORDS = re.compile(
    r"\b(?:importan\w*|impact\w*|effects?|influen\w*|contribut\w*|predictors?|drivers?|factors?|weights?|"
    r"coefficients?|risk|matters?)\b", re.IGNORECASE)
#: A figure that is already scale-free: "r=0.245", "a correlation of 0.13".
_SCALE_FREE_FIGURE = re.compile(r"\b(?:r|eta|V|rho|tau)\s*[=≈]|correlat\w*|Cram[eé]r", re.IGNORECASE)
#: A coefficient read as what it is, a per-unit change: "each extra year adds
#: 0.07 to the log-odds", "the odds double per 10 mg/dL".
_PER_UNIT = re.compile(r"\b(?:per|each|every|for every|one[- ]unit|unit increase|log[- ]odds|odds)\b",
                       re.IGNORECASE)
_DRIVER_QUESTION = re.compile(
    r"\b(?:predict\w*|driv\w*|importan\w*|influen\w*|factors?|determin\w*|explain\w*|affect\w*|"
    r"contribut\w*|matters?)\b", re.IGNORECASE)


def raw_coefficient_importance(code: Any) -> bool:
    """The analysis read a fitted model's ``coef_`` with no feature put on a
    common scale first. Such a coefficient is per unit of its column: a 0/1
    dummy's 1.27 against age's 0.07 per year says nothing about which matters
    more (rc10: ever_married_Yes 1.27, Residence_type_Rural 1.22, age 0.07,
    while age has r=0.245 with stroke and marriage eta=0.108)."""
    try:
        tree = ast.parse(str(code or ""))
    except (SyntaxError, ValueError):
        return False
    nodes = list(ast.walk(tree))
    if not any(isinstance(n, ast.Attribute) and n.attr == "coef_" for n in nodes):
        return False
    for n in nodes:
        name = n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else None
        if name in _SCALERS:
            return False
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.Mult)) and any(
                isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "std"
                for side in (n.left, n.right) for c in ast.walk(side)):
            return False
    return True


def _presents_coefficient(claim: str, ranked: Optional[Mapping[str, Any]]) -> bool:
    """*claim* quotes a figure as how much a column matters."""
    text = _driver_text(claim)
    if not _NUMBER.search(text) or _SCALE_FREE_FIGURE.search(text) or _PER_UNIT.search(text):
        return False
    if not (_is_driver_claim(text) or _IMPORTANCE_WORDS.search(text)):
        return False
    if ranked:
        columns = [d["column"] for d in ranked["drivers"]]
        return bool(_mentioned(text, columns, ranked["column"], ranked.get("values")))
    return True


# --- identifiers and importance ------------------------------------------------
#: A sentence that sets an identifier aside rather than ranking it.
_SET_ASIDE = re.compile(
    r"\b(?:exclud\w*|drop\w*|remov\w*|ignor\w*|left out|leaves? out|set aside|omit\w*|not (?:a |an )?"
    r"(?:predictor|driver|factor|feature)|identifiers?|row labels?|no (?:predictive|real) )", re.IGNORECASE)
#: How a reply names an identifier other than by its column name: "the
#: patient ID variable", "UUID".
_ID_WORDS = re.compile(r"(?<![\w-])(?:ids?|uuids?|guids?)(?![\w-])", re.IGNORECASE)
_ID_NAME_TOKENS = frozenset({"id", "ids", "uuid", "guid"})


def check_identifier_predictor(claim: str, ranked: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Why *claim* presents a row identifier as a predictor or driver, or
    None. The ranking leaves identifiers out (IDENTIFIER_RULE in
    :mod:`avaloka.goal_analysis`); a model fit on every column can still
    score one (rc11 p3: a random forest's impurity scores put ``id`` third,
    0.189, above heart disease), and a reply that ranks it is wrong whatever
    the figure."""
    if not ranked or not ranked.get("identifiers"):
        return None
    from avaloka.goal_analysis import _ordered_tokens

    text = _driver_text(claim)
    if not (_RANKING_CONTEXT.search(text) or _IMPORTANCE_TERM.search(text)) or _SET_ASIDE.search(text):
        return None
    for column in ranked["identifiers"]:
        named = bool(_named_columns(text, [column])) or re.search(
            rf"(?<![\w-]){re.escape(column)}(?![\w-])", text, re.IGNORECASE)
        if not named and set(_ordered_tokens(column)) & _ID_NAME_TOKENS:
            named = bool(_ID_WORDS.search(text))
        if named:
            return (f"{column} is a row identifier, not a predictor: its values label rows and say nothing "
                    f"about {ranked['target']}; {_first_by(ranked['drivers'][0], ranked['target'])}; "
                    f"ranked by association: {_ranking_text(ranked['drivers'])}")
    return None


#: A sentence that calls its figures importance or weight.
_IMPORTANCE_TERM = re.compile(r"\b(?:feature[\s-]+)?importances?\b|\bweights?\b", re.IGNORECASE)


def model_importance(code: Any) -> Optional[str]:
    """What kind of importance the analysis computed: ``"permutation"``
    (``sklearn.inspection.permutation_importance``, the measure the model
    card reports), ``"scaled_coefficients"`` (``coef_`` on features put on
    one scale), ``"impurity"`` (a tree model's ``feature_importances_``),
    ``"raw_coefficients"`` (:func:`raw_coefficient_importance`), or None
    when it fit no model."""
    try:
        tree = ast.parse(str(code or ""))
    except (SyntaxError, ValueError):
        return None
    names = {n.id if isinstance(n, ast.Name) else n.attr for n in ast.walk(tree)
             if isinstance(n, (ast.Name, ast.Attribute))}
    if "permutation_importance" in names:
        return "permutation"
    if "coef_" in names:
        return "raw_coefficients" if raw_coefficient_importance(code) else "scaled_coefficients"
    if "feature_importances_" in names:
        return "impurity"
    return None


def check_importance_wording(claim: str, ranked: Optional[Mapping[str, Any]], kind: Optional[str]) -> Optional[str]:
    """Why *claim* ranks columns by "importance" when the analysis measured
    none that can be reported as such, or None.

    On a what-predicts question the answer is the association ranking, and
    it is never called importance. A model's importance may be reported only
    when it is a permutation importance or a coefficient on scaled features
    (its order is still checked against the association ranking by
    :func:`check_driver_ranking`). A tree model's ``feature_importances_``
    is not one: it is impurity reduction on the rows the model was fit on,
    and it rises with a column's number of distinct values (rc11 p3: age
    0.199, glucose 0.194, the row id 0.189, heart disease 0.022). Raw
    coefficients have their own finding (:func:`raw_coefficient_detail`)."""
    if not ranked or kind in {"permutation", "scaled_coefficients", "raw_coefficients"}:
        return None
    text = _driver_text(claim)
    if not _IMPORTANCE_TERM.search(text) or _SCALE_FREE_FIGURE.search(text) or _PER_UNIT.search(text):
        return None
    columns = [d["column"] for d in ranked["drivers"]] + list(ranked.get("identifiers") or ())
    if not _mentioned(text, columns, ranked["column"], ranked.get("values")) \
            and not check_identifier_predictor(claim, ranked):
        return None
    if kind == "impurity":
        detail = ("its figures are a tree model's impurity scores (feature_importances_), measured on the rows "
                  "the model was fit on and inflated for columns with many distinct values, so they do not "
                  f"rank what predicts {ranked['target']}")
    else:
        detail = ("it ranks columns by importance, but the analysis fit no model whose importance was measured "
                  f"(a permutation importance); what was measured is each column's association with "
                  f"{ranked['target']}")
    return (f"{detail}; {_first_by(ranked['drivers'][0], ranked['target'])}; "
            f"ranked by association: {_ranking_text(ranked['drivers'])}")


def _may_rank(claim: str) -> bool:
    """*claim* might rank columns: worth computing the ranking for."""
    text = _driver_text(claim)
    return bool(_is_driver_claim(text) or _TOP_N.search(text) or _IMPORTANCE_TERM.search(text)
                or _NEXT_TIER.search(text)
                or (_RANKING_CONTEXT.search(text) and (_COMPARATIVE.search(text) or _SUCCESSION.search(text)
                                                       or _ID_WORDS.search(text))))


def raw_coefficient_detail(ranked: Optional[Mapping[str, Any]]) -> str:
    detail = ("its figures are raw model coefficients on unscaled features, which are per unit of each "
              "column (a 0/1 category against a measure in its own units) and do not rank what matters")
    if ranked:
        detail += (f"; {_first_by(ranked['drivers'][0], ranked['target'])}; "
                   f"ranked by association: {_ranking_text(ranked['drivers'])}")
    return detail


def semantic_findings(claims: Sequence[str], narrative: str, flat: Dict[str, Any],
                      used: Optional["DataUsed"] = None, question: Optional[str] = None,
                      code: Optional[str] = None, restated: Sequence[float] = (),
                      datasets: Optional["DataUsed"] = None) -> List[ClaimFinding]:
    """Findings for claims that do not answer the question asked."""
    findings: List[ClaimFinding] = []
    mismatch = condition_mismatch(question, code, used)
    if mismatch:
        figures = [c for c in claims if _has_figure(c, restated)] or list(claims)
        findings.extend(ClaimFinding(claim=c, verdict=Verdict.CONTRADICTED, check="condition_mismatch",
                                     detail=mismatch, evidence_ref="coder_definition.code")
                        for c in figures)
    flagged = {f.claim for f in findings}
    ranked = None
    raw_coef = raw_coefficient_importance(code)
    # A figure beside a column on a what-predicts question is its strength
    # (check_driver_figures), whether or not the sentence also ranks.
    strengths = bool(_DRIVER_QUESTION.search(str(question or ""))) and any(
        _NUMBER.search(c) and _STRENGTH_CONTEXT.search(c) for c in claims if c not in flagged)
    if raw_coef or strengths or any(_may_rank(c) for c in claims if c not in flagged):
        try:
            ranked = computed_drivers(question, used)
        except Exception:  # noqa: BLE001 - a ranking that cannot be computed contradicts nothing
            logger.debug("[claim-verifier] could not rank drivers", exc_info=True)
    # Coefficients on unscaled features are never shown as importance: on a
    # "what predicts" question (one whose outcome the data ranks, or that
    # asks it in words), every sentence quoting one as how much a column
    # matters is withheld, true or not, and the reason gives the ranking.
    raw_coef = raw_coef and (ranked is not None or bool(_DRIVER_QUESTION.search(str(question or ""))))
    kind = model_importance(code) if ranked is not None else None
    join_frames = used if used is not None and len(used.frames) >= 2 else datasets
    for claim in claims:
        if claim in flagged:
            continue
        detail = check_driver_ranking(claim, ranked)
        check = "driver_ranking_contradicted"
        coefficient = raw_coef and _presents_coefficient(claim, ranked)
        identifier = check_identifier_predictor(claim, ranked)
        wording = check_importance_wording(claim, ranked, kind)
        if detail or identifier:
            if not detail:
                detail, check = identifier, "identifier_as_predictor"
            elif identifier:
                detail += f"; {identifier.split('; ')[0]}"
            if coefficient:
                detail += f"; {raw_coefficient_detail(None)}"
            elif wording:
                detail += f"; {wording.split('; ')[0]}"
            invented = check_invented_numbers(claim, flat, restated=restated)
            if invented is not None:
                detail += f"; it also {invented.detail}"
        elif coefficient:
            detail, check = raw_coefficient_detail(ranked), "raw_coefficient_importance"
        elif wording:
            rewrite = relabelled(claim, ranked, restated)
            if rewrite is not None:
                # True in substance: its figures are each column's association,
                # only called importance. It is shown relabelled, not withheld.
                findings.append(ClaimFinding(
                    claim=claim, verdict=Verdict.SUPPORTED, check="importance_relabelled",
                    detail=(f"its figures are each column's association with {ranked['target']}, not a model's "
                            "importance; shown as association"), evidence_ref="data", rewrite=rewrite))
                continue
            detail, check = wording, "importance_not_association"
        elif (detail := check_driver_figures(claim, ranked, kind, restated)) is not None:
            check = "driver_figure_mismatch"
            if kind == "raw_coefficients":
                detail += f"; {raw_coefficient_detail(None)}"
        else:
            detail, check = check_join_claim(claim, narrative, join_frames), "join_claim_contradicted"
        if detail:
            findings.append(ClaimFinding(claim=claim, verdict=Verdict.CONTRADICTED, check=check,
                                         detail=detail, evidence_ref="data"))
    return findings


def _add_semantic_findings(report: VerificationReport, narrative: str, flat: Dict[str, Any],
                           used: Optional[DataUsed], question: Optional[str], code: Optional[str],
                           restated: Sequence[float], datasets: Optional[DataUsed]) -> None:
    """Merge :func:`semantic_findings` into *report*: a claim that answers
    another question gets that finding in place of a softer one."""
    claims = split_claims(narrative)
    for finding in semantic_findings(claims, narrative, flat, used, question, code, restated, datasets):
        if finding.verdict is Verdict.SUPPORTED and any(
                f.claim == finding.claim and f.check != "invented_number" for f in report.findings):
            continue                     # a relabel never clears another finding
        report.findings = [f for f in report.findings if f.claim != finding.claim]
        report.findings.append(finding)
    caveat = baseline_caveat(narrative, flat)
    if caveat:
        report.caveats.append(caveat)


def _join_claims_report(state: Mapping[str, Any], narrative: str) -> Optional[Dict[str, Any]]:
    """A turn with no analysis (a chat about how the tables relate) may still
    say a join matches nothing; that claim is checked against the datasets."""
    claims = [c for c in split_claims(narrative) if _asserts_empty_join(c)]
    if not claims:
        return None
    try:
        frames = datasets_used(state)
    except Exception:  # noqa: BLE001 - recomputation is extra evidence, never a crash
        logger.debug("[claim-verifier] could not load the datasets of the turn", exc_info=True)
        return None
    findings = semantic_findings(claims, narrative, {}, None, None, None, (), frames)
    if not findings:
        return None
    report = VerificationReport(findings=findings, claims_checked=len(claims),
                                claims_seen=len(split_claims(narrative)))
    logger.warning("[claim-verifier] %s", report.summary())
    return {"verification_report": report.as_dict(),
            "verification_safe_to_present": report.safe_to_present}


def _lazy_datasets(state: Mapping[str, Any], narrative: str, used: Optional[DataUsed]) -> Optional[DataUsed]:
    """The turn's datasets, read only when the reply says a join matches nothing
    and the run itself read fewer than two frames."""
    if used is not None and len(used.frames) >= 2:
        return None
    if not (_EMPTY_JOIN.search(narrative or "") and _JOIN_WORDS.search(narrative or "")):
        return None
    try:
        return datasets_used(state)
    except Exception:  # noqa: BLE001 - recomputation is extra evidence, never a crash
        logger.debug("[claim-verifier] could not load the datasets of the turn", exc_info=True)
        return None


def datasets_used(state: Mapping[str, Any]) -> DataUsed:
    """The frames of every dataset of the turn (``datasets_context``), for a
    claim about how they relate when no analysis ran."""
    import os

    import pandas as pd

    from app.agents.input_reader import read_input_frame
    from file_handler.null_policy import read_csv_best_effort

    used = DataUsed()
    seen = set()
    for meta in state.get("datasets_context") or state.get("multi_dataset_state") or []:
        if not isinstance(meta, Mapping):
            continue
        path = (meta.get("data_source_location") or meta.get("full_data_location")
                or meta.get("sample_data_location"))
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            if os.path.isfile(str(path)) and os.path.getsize(str(path)) > RECOMPUTE_MAX_BYTES:
                continue
            frame = read_input_frame(str(path), read_csv_best_effort, sheet_name=meta.get("sheet_name"))
        except Exception:  # noqa: BLE001 - an unreadable dataset verifies nothing
            logger.debug("[claim-verifier] could not read a dataset of the turn", exc_info=True)
            continue
        if isinstance(frame, pd.DataFrame):
            used.frames.append((meta.get("alias") or os.path.basename(str(meta.get("filename") or path)), frame))
    return used


def _training_evidence(state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """What the trainer returned: run id, losses, accuracy, F1, the model
    card's summary and importances. A training report quotes these verbatim,
    and before they were evidence eight such figures were withheld as
    invented (rc6). The reports it nests are evidence already, once."""
    result = state.get("training_result")
    out: Dict[str, Any] = {}
    if isinstance(result, Mapping):
        out = {k: v for k, v in result.items()
               if not (k in ("evaluation_report", "integrity_report") and state.get(k) is not None)}
    if state.get("mlflow_run_id"):
        out.setdefault("mlflow_run_id", state.get("mlflow_run_id"))
    return out or None


def _code(state: Mapping[str, Any]) -> str:
    """The code the analysis ran."""
    coder = state.get("coder_definition")
    return str(coder.get("code") or "") if isinstance(coder, Mapping) else ""

# --------------------------------------------------------------------------- #
# Enforcement: what the user is shown
# --------------------------------------------------------------------------- #

#: The verdicts that make ``safe_to_present`` False.
_UNSAFE_VERDICTS = frozenset({Verdict.CONTRADICTED.value, Verdict.OVERREACH.value})
_FENCE = re.compile(r"(```.*?```)", re.DOTALL)
#: split_claims's sentence boundary, captured so the reply keeps its layout.
_BOUNDARY = re.compile(r"(" + _SENTENCE_END + r"|\n+)")
WITHHELD_ANSWER = ("I've held back my answer: the numbers from the analysis don't support "
                   "what it said, so none of it is shown as a finding.")


def _withheld_note(n: int) -> str:
    return (f"I've held back {n} statement{'s' if n != 1 else ''} from this answer "
            f"because the numbers from the analysis don't support {'it' if n == 1 else 'them'}.")


def withheld_claims(report: Optional[Mapping[str, Any]]) -> set:
    """The claims of *report* that may not be shown as fact."""
    if not isinstance(report, Mapping):
        return set()
    unreconciled = report.get("claims_reconciled") is False
    return {str(f.get("claim"))[:400] for f in report.get("findings") or []
            if isinstance(f, Mapping) and f.get("claim") and (
                f.get("verdict") in _UNSAFE_VERDICTS
                or f.get("check") in _NUMERIC_CHECKS
                or (unreconciled and f.get("check") in _RECONCILIATION_CHECKS))}


def _said(claim: str) -> str:
    """*claim* as a reader reads it: no markdown, list marker, case or extra spacing."""
    text = re.sub(r"[*_`#>|]", "", str(claim))
    text = re.sub(r"^\s*(?:[-+•]|\d+[.)])\s+", "", text)
    return re.sub(r"\s+", " ", text).strip().rstrip(".!?:;, ").casefold()


def presentable_reply(text: Any, report: Optional[Mapping[str, Any]]) -> Any:
    """The reply as the user may see it, given the verifier's report.

    The node only reports; this is where its verdict is enforced. A claim the
    verifier found contradicted or overreaching (the findings that make
    ``safe_to_present`` False) is withheld from the reply and the reply says
    so. So is a claim citing a number the evidence does not hold (unsupported
    by ``invented_number``), though it leaves ``safe_to_present`` True. When a
    flagged claim cannot be found in the reply as written, the whole reply is
    withheld: an unsafe claim is never shown as fact.

    A report that is unsafe without findings is a failed or empty step
    (:func:`_not_verified`); its reply is the failure itself and is left alone.

    The text of :func:`presentable_segments`, without the commentary marks.
    """
    if not isinstance(text, str) or not text.strip() or not isinstance(report, Mapping):
        return text
    from app.api import reply_segments as rs

    return rs.text_of(presentable_segments(text, report, rs.PLAIN))


def presentable_segments(text: str, report: Optional[Mapping[str, Any]],
                         provenance: Any = "commentary") -> List[Dict[str, str]]:
    """:func:`presentable_reply` as reply segments (app/api/reply_segments.py).

    *provenance* is what wrote *text*: ``"plain"``, ``"commentary"`` or the
    segments the node that wrote it recorded. A sentence kept from *text*
    keeps the kind it was written with; everything this adds -- the withheld
    note, the "Why:" lines, the caveats (the scope line among them), the
    method line, the withheld-breakdown reply -- is the verifier's own text
    and plain. A relabelled claim (a rewrite) keeps its sentence's kind.
    """
    from app.api import reply_segments as rs

    if not isinstance(text, str):
        return []
    segs = rs.normalize([rs.segment(text, provenance)]) if isinstance(provenance, str) \
        else rs.normalize(provenance)
    if rs.text_of(segs) != text:
        segs = rs.commentary(text)          # provenance lost: never shown as plain
    if not text.strip() or not isinstance(report, Mapping):
        return segs
    held = derived_categories.withheld_breakdown(report)
    if held:
        return rs.plain(held)
    # A claim true in substance but mislabelled is shown as it should read.
    for f in report.get("findings") or []:
        if isinstance(f, Mapping) and f.get("rewrite") and f.get("claim") and f.get("verdict") not in _UNSAFE_VERDICTS:
            claim, rewrite = str(f["claim"]), str(f["rewrite"])
            if any(claim in s["text"] for s in segs):
                segs = [rs.segment(s["text"].replace(claim, rewrite), s["kind"]) for s in segs]
            elif claim in rs.text_of(segs):  # across segments: one kind for the lot
                kind = rs.COMMENTARY if any(s["kind"] == rs.COMMENTARY for s in segs) else rs.PLAIN
                segs = rs.normalize([rs.segment(rs.text_of(segs).replace(claim, rewrite), kind)])
    text = rs.text_of(segs)
    # No number the data does not support is shown as fact. A figure the
    # verifier could not find in the result (an unsupported number) is
    # withheld like a contradicted claim, whatever safe_to_present says:
    # "readmitted within 30 days ... approximately 0.35" was flagged
    # unsupported, which leaves safe_to_present True, and this returned the
    # reply unchanged on it (rc3, gemma). A figure that failed reconciliation
    # is withheld too. Unsupported prose without a number stays.
    unsafe = withheld_claims(report)
    if not unsafe:
        return _explained(segs, text, report)

    spans: List[tuple] = []
    at = 0
    for s in segs:
        spans.append((at, at + len(s["text"]), s["kind"]))
        at += len(s["text"])

    def kind_of(start: int, piece: str) -> str:
        end = start + len(piece)
        kinds = {k for a, b, k in spans if a < end and b > start}
        return rs.COMMENTARY if rs.COMMENTARY in kinds or not kinds else rs.PLAIN

    # A sentence the reply repeats is withheld wherever it stands: rc16 showed
    # the bullet a withheld sentence restated. Repeats are matched as the
    # reader reads them, without markdown, list markers, case or spacing.
    by_text = {_said(c): c for c in unsafe}

    def held(piece: str) -> set:
        return {by_text.get(_said(c), c[:400]) for c in split_claims(piece)} & unsafe

    found: set = set()
    n_held = 0
    kept: List[Any] = []
    at = 0
    for segment in _FENCE.split(text):
        if segment.startswith("```"):
            kept.append((segment, kind_of(at, segment)))   # code is never a claim (split_claims)
            at += len(segment)
            continue
        parts = _BOUNDARY.split(segment)
        gone = [bool(held(piece)) if i % 2 == 0 else False for i, piece in enumerate(parts)]
        # A piece too short to be a claim, on the same line as a withheld
        # sentence, is a fragment of it ("$182.27)." after "... ($545.08 vs."):
        # it goes with it, so nothing but whole sentences of the reply is shown.
        for i in range(0, len(parts), 2):
            if gone[i] or split_claims(parts[i]) or not parts[i].strip():
                continue
            if any(0 <= j < len(parts) and gone[j] and "\n" not in parts[(i + j) // 2] for j in (i - 2, i + 2)):
                gone[i] = True
        # Between two shown sentences goes the widest boundary of the gap a
        # withheld one leaves: the last sentence of a paragraph withheld, the
        # paragraph break after it stays, and the next paragraph is not run
        # into this one (rc16 related: "... key-to-key match). **Next actions**").
        pending: Optional[str] = None
        gap = False
        for i, piece in enumerate(parts):
            start, at = at, at + len(piece)
            if i % 2:                     # a boundary: the verifier's layout, plain
                if pending is None or not gap or piece.count("\n") > pending.count("\n"):
                    pending = piece
                continue
            if gone[i]:
                found |= held(piece)
                n_held += len(split_claims(piece)) or 0
                gap = True
                continue
            if kept and pending is not None:
                kept.append((pending, rs.PLAIN))
            kept.append((piece, kind_of(start, piece)))
            pending, gap = None, False
        if pending is not None and kept and not gap:
            kept.append((pending, rs.PLAIN))
    if found != unsafe:
        return _explained(rs.plain(WITHHELD_ANSWER), text, report)
    shown = rs.strip([rs.segment(p, k) for p, k in kept])
    if not split_claims(rs.text_of(shown)):
        return _explained(rs.plain(WITHHELD_ANSWER), text, report)
    return _explained(rs.join(shown, rs.plain(f"\n\n{_withheld_note(max(n_held, len(found)))}")), text, report)


def _explained(reply: List[Dict[str, str]], original: str, report: Mapping[str, Any]) -> List[Dict[str, str]]:
    """*reply* with the reasons a check against the question gave for what
    was held back, and the caveats the original reply did not already make.
    All of it is the verifier's own text: plain."""
    from app.api import reply_segments as rs

    reasons = list(dict.fromkeys(
        str(f.get("detail")) for f in report.get("findings") or []
        if isinstance(f, Mapping) and f.get("check") in _EXPLAINED_CHECKS and f.get("detail")))
    caveats = [str(c) for c in report.get("caveats") or [] if str(c) not in original]
    added = ""
    if reasons:
        added += "\n\n" + "\n".join(f"Why: {r[0].upper()}{r[1:]}." for r in reasons)
    if caveats:
        added += "\n\n" + "\n".join(caveats)
    text = rs.text_of(reply) + added
    method = derived_categories.with_method_line(text, report)
    return rs.join(reply, rs.plain(added), rs.plain(method[len(text):]))
