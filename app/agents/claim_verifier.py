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

Seven checks, each targeting a way an automated analysis misleads:

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
6. **A superlative naming the wrong group** — "the lowest is Autos & Vehicles"
   when another group in the result is lower.
7. **A threshold the result breaks** — "no category exceeds 30 million" when
   one does, or "all categories are above N" when one is not.

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

# _NUMBER = re.compile(r"-?\d+(?:\.\d+)?%?")
#
# A narrative writes large values compactly ("56.1B", "24 million") or with
# space-like thousands separators ("56 106 087 508", often U+202F). The old
# pattern read those as 56.1 / 24, or as four separate numbers (56, 106, ...),
# and flagged correct sentences: "cites 56, which does not appear in the
# computed evidence" for a 56,106,087,508 value in the result table.
#
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
    r"(?:\s?%|\s?percent\b|\s?(?:thousand|million|billion|trillion)\b|(?:[kKmMbBtT]|bn|Bn|BN|mn|tn)(?![\w])"
    r"|(?:[\u00a0\u202f\u2009]|(?<=\d\d) |(?<=\.\d) )(?:[KMBT]|Bn|BN)(?![\w&]))?"
)
_SCALES = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12,
}
_SCALE_SUFFIX = re.compile(r"^(.*\d)\s?([A-Za-z]+)$")


def _is_percent(raw: str) -> bool:
    return raw.rstrip().endswith("%") or raw.rstrip().lower().endswith("percent")


def _parse_number(raw: str) -> Optional[float]:
    text = re.sub(r"\s?(?:%|percent)$", "", raw.strip(), flags=re.IGNORECASE)
    scale = 1.0
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


def split_claims(narrative: str) -> List[str]:
    """Split a narrative into sentence-level claims worth checking.

    Code is not a claim: fenced blocks and inline `identifiers` are removed first,
    so a script or a column name like `age_25%` is never read as a statistic.
    """
    if not narrative:
        return []
    text = re.sub(r"```.*?```", " ", narrative, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", " ", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
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
    # Typographic minus ("−2,008.49") is still a negative number.
    # "–2,008.49" (en dash), "−$2,008.49" and "-$2,008.49" are negative numbers too.
    normalized = re.sub(r"[-\u2212\u2013]\s?[$€£₹]", "-", claim).replace("\u2212", "-").replace("\u2013", "-")
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

    def _matches(value: float, k: float) -> bool:
        return abs(value - k) <= max(tolerance, abs(k) * tolerance)

    for value, raw in stated:
        # A percentage may be stored as a fraction: "33.6%" vs 0.336 in the table.
        # Rounded to one decimal of a percent, so allow half a tenth of a point.
        if _is_percent(raw) and any(
            _matches(value, k * 100) or abs(value - k * 100) <= 0.051 for k in known
        ):
            continue
        if not any(_matches(value, k) for k in known):
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


def check_ordering_contradicted(claim: str, flat: Dict[str, Any]) -> Optional[ClaimFinding]:
    """"Lowest at 68.12" when the result table holds a lower value.

    Each figure in the claim is located in a result column. For every
    superlative the claim makes, at least one located figure must be that
    column's extreme. Figures that match no column are left to the invented-
    number check.
    """
    wants_low = _mentions(claim, _LOWEST_TERMS)
    wants_high = _mentions(claim, _HIGHEST_TERMS)
    if not (wants_low or wants_high):
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

# --------------------------------------------------------------------------- #
# Superlatives and thresholds over a grouped result
# --------------------------------------------------------------------------- #
#
# The 1.6 demo said "The lowest average is for Autos & Vehicles at about
# 17.9M" beside a table whose lowest was Travel & Events (12,500,000), and "No
# category exceeds 30 million" beside Trailers at 39,000,000. Neither was
# flagged. check_ordering_contradicted reads a claim's *figure*, never the
# group it names, and it skips every scaled figure ("17.9M"); nothing read a
# threshold at all, so the sentence fell to check_invented_numbers, whose
# UNSUPPORTED verdict does not raise the notice.
#
# These two checks read the group and the bound instead. They run only on a
# result with exactly one label column and one row per label, and only when
# the claim resolves to one direction and to columns that all agree; anything
# else is skipped, never guessed. The superlative lexicon (rank words, "at
# least", "top three", scoped phrases, null labels) follows 2.0's
# claim_extremes.

_ROW_CELL = re.compile(r"^(?:execution_output|output_json|output_data)\[(\d+)\]\.(.+)$")
_NORM_CHARS = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "*": " ",
})
#: Labels that are no group: the missing-value row is not a category.
_NULL_LABELS = frozenset({"", "nan", "none", "null", "missing", "n/a", "na", "<na>", "nat", "-"})
#: A row that sums the others is not one of them.
_TOTAL_LABELS = frozenset({"total", "grand total", "subtotal", "all", "overall", "sum"})
#: Measure words that name no column: "the highest average value".
_MEASURE_FILLER = frozenset({"the", "a", "an", "of", "per", "overall", "value", "values", "figure",
                             "level", "number", "amount"})
_MEASURE_SYNONYMS = {"average": "mean", "avg": "mean", "total": "sum"}
#: A column whose name gives its unit: "median_subscribers_million" holds 39.0 for 39,000,000.
_COLUMN_UNITS = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "mil": 1e6,
                 "billion": 1e9, "bn": 1e9}
_HIGH_TERMS = frozenset({"highest", "largest", "biggest", "greatest", "most", "top"})
_SUP = (r"(?P<sup>highest|largest|biggest|greatest|lowest|smallest|fewest"
        r"|(?<=the )(?:most|least|top|bottom))")
#: Not the extreme itself: "second highest", "one of the lowest", "not the top", "at least".
_SUP_RANKED = re.compile(r"(?:\b(?:second|third|fourth|fifth|sixth|2nd|3rd|4th|5th|6th|next|one of|among|"
                         r"near|nearly|almost|close to|not|n't|never)\s+(?:the\s+)?|"
                         r"\b(?:second|third|fourth|fifth|2nd|3rd|4th|5th|next)-)$")
#: A count after it is a list, not an extreme: "the top three", "lowest 5".
_SUP_COUNT = re.compile(r"^[\s-]*(?:\d|one|two|three|four|five|six|seven|eight|nine|ten)\b")
#: Says the claim is about a selection or an exception; the result cannot say.
_SCOPED = re.compile(
    r"\b(?:among|of these|of those|of the (?:top|selected|listed|shown|ones?)|in (?:this|the) "
    r"(?:list|selection|sample|subset)|in the top|within|listed|shown|selected|excluding|except|"
    r"other than|apart from|aside from|besides|outside|ignoring)\b")
_CLAUSE_END = re.compile(r",\s|;|:\s|\s-\s|\b(?:while|whereas|but|although|though)\b")
_CLAUSE_LEAD = re.compile(r"^\s*(?:(?:[-+\u2022]|\d+[.)])\s+)?(?:(?:and|so|thus|hence|also|yet|indeed|"
                          r"notably|overall|in fact|in short)\s+)*")
_THRESHOLD_HINT = re.compile(r"\b(?:no|none|all|every|each|not a single|not one)\b")
_SUPERLATIVE_HINT = re.compile(r"\b(?:highest|largest|biggest|greatest|lowest|smallest|fewest|most|least|"
                               r"top|bottom)\b")
_UP_CMP = (r"(?:exceeds?|exceeded|surpass(?:es|ed)?|tops|topped"
           r"|(?:goes|go|went|rises?|rose|climbs?|climbed)\s+(?:above|over|beyond)"
           r"|(?:(?:is|are|was|were|sits?|sat|stays?|stayed|remains?|remained)\s+)?"
           r"(?:above|over|more than|greater than|higher than|larger than|bigger than))")
_DOWN_CMP = (r"(?:(?:falls?|fell|dips?|dipped|drops?|dropped|goes|go|went|sinks?|sank)\s+(?:below|under|beneath)"
             r"|(?:(?:is|are|was|were|sits?|sat|stays?|stayed|remains?|remained)\s+)?"
             r"(?:below|under|beneath|less than|lower than|smaller than|fewer than))")
#: A word that starts the comparison, so it is never a measure word.
_NOT_VERB = (r"(?!(?:is|are|was|were|has|have|had|goes|go|went|falls?|fell|sits?|sat|stays?|stayed|remains?|"
             r"remained|dips?|dipped|drops?|dropped|rises?|rose|climbs?|climbed|sinks?|sank|exceeds?|surpass\w*|"
             r"tops?|above|over|below|under|beneath|more|less|fewer|greater|higher|lower|larger|smaller|"
             r"bigger)\b)")
_THRESHOLD = re.compile(
    r"^(?:(?P<neg>not a single|not one|none of the|none of them|none|no)|(?P<all>all of the|all the|all|every|each))"
    r"(?:\s+(?:\d+\s+)?(?P<noun>[a-z_]+)(?:'s?(?=\s))?)?"
    r"(?:\s+(?P<m1>(?:" + _NOT_VERB + r"[a-z_]+\s+){0,2}?" + _NOT_VERB + r"[a-z_]+)(?=\s+(?:exceed|surpass|top|go|went|rise|rose|climb|is|are|"
    r"was|were|sit|sat|stay|remain|above|over|more|greater|higher|larger|bigger|fall|fell|dip|drop|sink|sank|"
    r"below|under|beneath|less|lower|smaller|fewer|has|have|had)))?"
    r"(?:\s+(?:has|have|had)\s+(?:an?\s+|any\s+|the\s+)?(?P<m2>(?:[a-z_]+\s+){0,2}?[a-z_]+)?(?:\s+of)?)?"
    r"\s+(?:(?P<up>" + _UP_CMP + r")|(?P<down>" + _DOWN_CMP + r"))"
    r"\s+[$\u20ac\u00a3]?(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion|mn|bn|tn|k|m|b))?(?P<pct>\s?%|\s?percent)?(?![\w])"
    r"(?P<after>.*)$")
_AFTER_STOP = frozenset({"and", "so", "which", "because", "while", "but", "than", "as", "in", "per", "each",
                         "or", "for", "across", "on", "at", "with", "to", "the", "a", "an", "is", "are",
                         "was", "were", "this", "that", "it"})
_THRESHOLD_SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6,
                     "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "tn": 1e12}


def _norm(text: Any) -> str:
    """Casefolded, one kind of space, dash and quote; "&" read as "and"."""
    text = str(text).translate(_NORM_CHARS)
    text = re.sub(r"\s*&\s*", " and ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _stem(word: str) -> str:
    word = _MEASURE_SYNONYMS.get(word, word)
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _tokens(name: str) -> List[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [_stem(t) for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def _cell_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value.strip().replace(",", "")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _null_label(value: Any) -> bool:
    if value is None or (isinstance(value, float) and value != value):
        return True
    return isinstance(value, str) and _norm(value) in _NULL_LABELS


def _fmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    return f"{value:.4g}"


@dataclass
class _Grouped:
    """A result of one label column and one row per label."""
    label_column: str
    labels: List[str]                              # as the result writes them
    by_label: Dict[str, int]                       # normalised label -> row
    columns: Dict[str, List[Optional[float]]]      # numeric column -> value per row
    group_pattern: "re.Pattern[str]"

    def resolve(self, words: Sequence[str]) -> Optional[List[str]]:
        """The numeric columns *words* name; all of them for no words; None when
        the words name no column, or name the groups themselves ("the largest
        category" is a count the result may not hold)."""
        label_tokens = set(_tokens(self.label_column)) | {"group", "category", "segment"}
        wanted = [_stem(w) for w in words if w and w not in _MEASURE_FILLER]
        if any(w in label_tokens for w in wanted):
            return None
        if not wanted:
            return list(self.columns)
        hits = [c for c in self.columns if all(w in _tokens(c) for w in wanted)]
        return hits or None

    def unit(self, column: str) -> float:
        for token in _tokens(column):
            if token in _COLUMN_UNITS:
                return _COLUMN_UNITS[token]
        return 1.0


def _grouped_result(flat: Dict[str, Any]) -> Optional[_Grouped]:
    cells: Dict[int, Dict[str, Any]] = {}
    for key, value in flat.items():
        m = _ROW_CELL.match(key)
        if m:
            cells.setdefault(int(m.group(1)), {})[m.group(2)] = value
    if len(cells) < 2:
        return None
    rows = [cells[i] for i in sorted(cells)]
    names = list(dict.fromkeys(c for row in rows for c in row))
    numeric, text = [], []
    for name in names:
        present = [row.get(name) for row in rows if not _null_label(row.get(name))]
        if not present:
            continue
        (numeric if all(_cell_number(v) is not None for v in present) else text).append(name)
    if len(text) != 1 or not numeric:
        return None
    label_column = text[0]
    labels: List[str] = []
    by_label: Dict[str, int] = {}
    columns: Dict[str, List[Optional[float]]] = {c: [] for c in numeric}
    for row in rows:
        label = row.get(label_column)
        if _null_label(label) or _norm(label) in _TOTAL_LABELS:
            continue
        key = _norm(label)
        if key in by_label:
            return None                            # two rows per label: not one row per group
        by_label[key] = len(labels)
        labels.append(str(label))
        for c in numeric:
            columns[c].append(_cell_number(row.get(c)))
    names_in_text = sorted((k for k in by_label if len(k) >= 2), key=len, reverse=True)
    if len(labels) < 2 or not names_in_text:
        return None
    pattern = re.compile(r"(?<![\w])(?:" + "|".join(re.escape(k) for k in names_in_text) + r")(?![\w])")
    return _Grouped(label_column, labels, by_label, columns, pattern)


def _clauses(claim: str) -> List[str]:
    text = re.sub(r"^\s*(?:[-+\u2022]|\d+[.)])\s+", "", _norm(claim))
    return [_CLAUSE_LEAD.sub("", c).strip() for c in _CLAUSE_END.split(text) if c and c.strip()]


def _measure_words(text: Optional[str]) -> List[str]:
    return re.findall(r"[a-z_]+", (text or "").replace("-", " "))


def check_superlative_contradicted(claim: str, flat: Dict[str, Any]) -> Optional[ClaimFinding]:
    """"The lowest average is for Autos & Vehicles" when another group's average is lower.

    The group the superlative names must hold the maximum (or minimum) of the
    column the claim's words name. With no measure word every numeric column
    is a candidate, and they must agree; a claim about a group the result does
    not hold is not this check's to judge.
    """
    said_text = _norm(claim)
    if not _SUPERLATIVE_HINT.search(said_text) or _SCOPED.search(said_text):
        return None
    table = _grouped_result(flat)
    if table is None:
        return None
    group = table.group_pattern.pattern
    measure = r"(?P<measure>(?:[\s-]+[a-z_']+){0,4}?)"
    sup_first = re.compile(rf"(?:^|(?<=[\s(]))(?:the\s+)?{_SUP}{measure}\s+(?:is|was)\s+"
                           rf"(?:for\s+|in\s+|from\s+)?(?:the\s+)?(?P<group>{group})")
    group_first = re.compile(
        rf"^(?:the\s+)?(?P<group>{group})\s+(?:has|have|had|is|are|was|were|posts|records|holds|ranks|ranked|"
        rf"comes|came|sits)\s+(?:at\s+)?(?:the\s+)?{_SUP}{measure}"
        rf"(?=\s*$|\s*[.!?()]|\s+(?:at|with|of|on|in|by|for|across|overall)\b)")
    for clause in _clauses(claim):
        said = None
        for pattern in (sup_first, group_first):
            m = pattern.search(clause)
            if m is None:
                continue
            if _SUP_RANKED.search(clause[:m.start("sup")]) or _SUP_COUNT.match(clause[m.end("sup"):]):
                break
            # One group, named once: "the highest are Shows and Trailers" names two.
            others = [g for g in table.group_pattern.findall(clause) if g != m.group("group")]
            if others and pattern is sup_first:
                break
            said = m
            break
        if said is None:
            continue
        columns = table.resolve(_measure_words(said.group("measure")))
        if not columns:
            continue
        high = said.group("sup") in _HIGH_TERMS
        row = table.by_label[said.group("group")]
        verdicts = []
        for column in columns:
            values = table.columns[column]
            pool = [(v, i) for i, v in enumerate(values) if v is not None]
            if values[row] is None or len(pool) < 2:
                verdicts = []
                break
            extreme, at = (max if high else min)(pool, key=lambda p: p[0])
            margin = 1e-9 * max(1.0, abs(extreme))
            holds = values[row] >= extreme - margin if high else values[row] <= extreme + margin
            verdicts.append((holds, column, extreme, at))
        if not verdicts or any(v[0] for v in verdicts) != all(v[0] for v in verdicts):
            continue                                       # unresolved, or the columns disagree
        if verdicts[0][0]:
            continue
        _holds, column, extreme, at = verdicts[0]
        name, other = table.labels[row], table.labels[at]
        return ClaimFinding(
            claim=claim, verdict=Verdict.CONTRADICTED, check="superlative_contradicted",
            detail=(f"names {name} as the {said.group('sup')} by {column} "
                    f"({_fmt(table.columns[column][row])}), but {other} is "
                    f"{'higher' if high else 'lower'} at {_fmt(extreme)} in the computed result"),
            evidence_ref=f"execution_output.{column}",
        )
    return None


def check_threshold_contradicted(claim: str, flat: Dict[str, Any]) -> Optional[ClaimFinding]:
    """"No category exceeds 30 million" when a row of the result does.

    "No / none / not one X is above|below N" and "all / every / each X is
    above|below N" must hold for every group in the result. X must be the
    result's own groups ("category" for a ``category`` column, or a bare
    "none"); the bound is read at the precision it is stated, so "30 million"
    is broken only beyond 30.5 million.
    """
    if not _THRESHOLD_HINT.search(_norm(claim)) or _SCOPED.search(_norm(claim)):
        return None
    table = None
    for clause in _clauses(claim):
        m = _THRESHOLD.match(clause)
        if m is None or m.group("pct"):
            continue
        quantifier, noun = (m.group("neg") or m.group("all")), m.group("noun")
        bare = quantifier in ("none", "none of them")
        if table is None:
            table = _grouped_result(flat)
            if table is None:
                return None
        if bare:
            if noun is not None:
                continue
        elif noun is None or _stem(noun) not in set(_tokens(table.label_column)) | {"group"}:
            continue
        after: List[str] = []
        for word in _measure_words(m.group("after").split(",")[0].split(".")[0])[:3]:
            if word in _AFTER_STOP:
                break
            after.append(word)
        columns = table.resolve(_measure_words(m.group("m1")) + _measure_words(m.group("m2")) + after)
        if not columns:
            continue
        raw = m.group("num") + (f" {m.group('scale')}" if m.group("scale") else "")
        digits = m.group("num").replace(",", "")
        scale = _THRESHOLD_SCALES.get(m.group("scale") or "", 1.0)
        bound = float(digits) * scale
        tol = 0.5 * 10 ** -(len(digits.split(".", 1)[1]) if "." in digits else 0) * scale
        # The claim says every group is at most the bound (exceeds) or at least it (below);
        # "all X exceed N" says the reverse of "no X exceeds N".
        at_most = bool(m.group("up")) == bool(m.group("neg"))
        verdicts = []
        for column in columns:
            unit = table.unit(column)
            broken = [(v * unit, i) for i, v in enumerate(table.columns[column]) if v is not None
                      and (v * unit > bound + tol if at_most else v * unit < bound - tol)]
            verdicts.append((broken, column))
        if any(b for b, _c in verdicts) != all(b for b, _c in verdicts) or not verdicts[0][0]:
            continue                                       # holds, or the columns disagree
        broken, column = verdicts[0]
        worst, at = (max if at_most else min)(broken, key=lambda p: p[0])
        who = quantifier if bare else f"{'no' if m.group('neg') else 'every'} {noun}"
        side = "above" if m.group("up") else "below"
        more = f"; {len(broken)} groups break it" if len(broken) > 1 else ""
        return ClaimFinding(
            claim=claim, verdict=Verdict.CONTRADICTED, check="threshold_contradicted",
            detail=(f"says {who} is {side} {raw}, but {table.labels[at]} is at "
                    f"{_fmt(worst / table.unit(column))} by {column} in the computed result{more}"),
            evidence_ref=f"execution_output.{column}",
        )
    return None



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
    check_ordering_contradicted,
    check_superlative_contradicted,
    check_threshold_contradicted,
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
    # The executed result is the authoritative evidence. In the ETL graph it lives
    # in output_file_data (a CSV data URI); output_json is only built later by the
    # API layer, and execution_output_data is the validator's dry run.
    execution_output = state.get("output_json")
    if not execution_output:
        ofd = state.get("output_file_data")
        if isinstance(ofd, dict) and ofd.get("content"):
            try:
                from app.api.helpers import _datauri_csv_to_records
                execution_output = _datauri_csv_to_records(ofd["content"])
            except Exception:
                logger.debug("[claim-verifier] could not decode output_file_data", exc_info=True)
    execution_output = (
        execution_output
        or state.get("execution_output_data")
        or execution_result.get("output_json")
        or execution_result.get("output_data")
    )
    result_shape = None
    if isinstance(execution_output, list) and execution_output and isinstance(execution_output[0], dict):
        result_shape = {"rows": len(execution_output), "columns": len(execution_output[0])}

    evidence = {
        "evaluation_report": state.get("evaluation_report"),
        "integrity_report": state.get("integrity_report"),
        "execution_output": execution_output,
        "result_shape": result_shape,
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