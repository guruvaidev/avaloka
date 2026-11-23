"""Typed specialist planning, routing, cartography, and answer review."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class QuestionPlan:
    question: str
    output_shape: str
    numeric_result_required: bool
    return_only: bool
    statistical_intent: str
    prediction_intent: str
    transforms: tuple[str, ...]
    join_required: bool
    unstructured_required: bool
    temporal_required: bool
    ambiguities: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceMap:
    source_types: tuple[str, ...]
    schema_terms: tuple[str, ...]
    multi_source: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewVerdict:
    status: str  # pass | repair | cannot_verify
    findings: tuple[str, ...]
    covered_outputs: tuple[str, ...]


_STATISTICAL_TEST = re.compile(
    r"\b(chi[- ]?square|hypothesis test|p[- ]?value|statistically significant|confidence interval|association between)\b",
    re.IGNORECASE,
)
_EXPERIMENT = re.compile(r"\b(a/?b test|experiment|randomi[sz]ed|treatment|control group|minimum detectable effect|power analysis)\b", re.IGNORECASE)
_DESCRIPTIVE = re.compile(r"\b(average|mean|median|fraction|proportion|percentage|percent|rate|volatility|return|correlation|quantile|moving average|ema|standard deviation)\b", re.IGNORECASE)
# "return" is the financial noun; the imperative "Return only the ..." that
# DAB puts on nearly every question is not a request for a number.
_NUMERIC_QUESTION = re.compile(r"\b(how many|how much|count|average|mean|fraction|proportion|percentage|percent|statistic|revenue|volatility|returns?(?!\s+(?:only|just|exactly|the|a|an|all|one|each|every|its|their|them|it|as|in)\b))\b", re.IGNORECASE)
# The answer is a label -- a name, category, code, state, month -- not a number.
# A numeric word in such a question qualifies the label ("which state has the
# highest average revenue"); it is not the answer. On DataAgentBench rc7 the
# numeric gate fired on a stage label, BANT factors, a month and a state code
# (crmarenapro:1, :3, :4, :9) and discarded two of those answers.
_LABEL_NOUNS = (
    r"(?:names?|titles?|labels?|categor(?:y|ies)|types?|kinds?|genres?|stages?|"
    r"months?|days?|states?|countr(?:y|ies)|cit(?:y|ies)|regions?|"
    r"compan(?:y|ies)|products?|platforms?|stores?|abbreviations?|codes?|"
    r"ids?|identifiers?|factors?|reasons?)"
)
_LABEL_QUESTION = re.compile(
    r"\b(?:which|who|whom|whose)\b"
    r"|\bwhat\s+" + _LABEL_NOUNS + r"\b"
    r"|\bwhat\s+(?:should|would|is|are|was|were)\s+(?:the\s+)?(?:[\w-]+\s+){0,2}?" + _LABEL_NOUNS + r"\b"
    r"|\breturn\s+(?:only\s+)?(?:one or several of\s+)?(?:the\s+)?(?:[\w-]+\s+){0,3}?" + _LABEL_NOUNS + r"\b",
    re.IGNORECASE,
)
# An explicit request for a number keeps the requirement even in a label
# question ("which store ..., and how many units did it sell").
# "calculate"/"compute" are left out: DAB's appended policy text uses them to
# define a ranking metric ("compute handle time as ...") in label questions.
_NUMERIC_ASK = re.compile(
    r"\b(?:how many|how much"
    r"|what (?:is|was|are|were) (?:the|its|their) (?:[\w-]+\s+){0,2}?(?:number|count|average|mean|median|total|sum|fraction|proportion|percentage|percent|ratio|rate|share|statistic|revenue|volatility|ratings?)"
    r"|(?:include|report|and) (?:the|its|their) (?:[\w-]+\s+){0,2}?(?:number|count|average|total|sum|percentage|revenue|ratings?) of"
    r"|what (?:fraction|proportion|percentage|percent|share|ratio))\b",
    re.IGNORECASE,
)
_TEMPORAL = re.compile(r"\b(since|between|during|before|after|latest|earliest|year|month|quarter|date|daily|weekly|annual|20\d{2}|19\d{2})\b", re.IGNORECASE)
_UNSTRUCTURED = re.compile(r"\b(free[- ]?text|unstructured|transcript|description|readme|document|article text|semantic|knowledge article)\b", re.IGNORECASE)
_LIST_OUTPUT = re.compile(r"\b(list|top\s+\d+|which\s+\d+|return all|what are|which states|which books|which packages|which companies)\b", re.IGNORECASE)


def numeric_result_required(text: str) -> bool:
    """Whether the answer itself must contain a number.

    A numeric word is not enough when the question asks which/what/who for a
    label; then only an explicit numeric ask (how many, calculate, what is the
    average ...) keeps the requirement.
    """
    if not _NUMERIC_QUESTION.search(text):
        return False
    if _LABEL_QUESTION.search(text) and not _NUMERIC_ASK.search(text):
        return False
    return True


def build_question_plan(question: Any, database_description: str = "") -> QuestionPlan:
    text = question if isinstance(question, str) else json.dumps(question, sort_keys=True, default=str)
    lower = text.casefold()
    return_only = "return only" in lower
    output_shape = "list" if _LIST_OUTPUT.search(text) else "scalar"
    if any(marker in lower for marker in ("include their", "report the", "provide the name of each")):
        output_shape = "table"
    if not return_only and any(marker in lower for marker in ("explain", "why", "recommend", "summary")):
        output_shape = "narrative"

    if _EXPERIMENT.search(text):
        statistical_intent = "experimental"
    elif _STATISTICAL_TEST.search(text):
        statistical_intent = "inferential"
    elif _DESCRIPTIVE.search(text) or _NUMERIC_QUESTION.search(text):
        statistical_intent = "descriptive"
    else:
        statistical_intent = "none"

    prediction_intent = "none"
    if re.search(r"\b(prediction interval|prediction set|coverage|conformal)\b", text, re.IGNORECASE):
        prediction_intent = "interval"
    elif re.search(r"\b(predict|forecast|score)\b", text, re.IGNORECASE):
        prediction_intent = "point"

    transforms: list[str] = []
    for name, pattern in (
        ("log10", r"\blog10|log[- ]?transform"),
        ("ema", r"\bexponential moving average|\bema\b"),
        ("volatility", r"\bvolatility|intraday range"),
        ("period_return", r"\boverall returns?|period returns?"),
        ("ratio", r"\bfraction|proportion|percentage|percent|rate\b"),
        ("temporal", _TEMPORAL.pattern),
    ):
        if re.search(pattern, text, re.IGNORECASE):
            transforms.append(name)

    description = database_description.casefold()
    source_markers = sum(marker in description for marker in ("postgres", "mongodb", "sqlite", "duckdb", "mysql"))
    join_required = bool(re.search(r"\b(join|across (?:all |multiple )?(?:sources|databases|tables)|associated with|corresponding)\b", text, re.IGNORECASE)) or source_markers > 1
    unstructured = bool(_UNSTRUCTURED.search(text)) or "unstructured" in description
    if join_required:
        transforms.append("normalize_join_keys")
    if unstructured:
        transforms.append("bounded_text_extraction")
    if "mongodb" in description or "nested" in description or "json" in description:
        transforms.append("nested_projection")

    ambiguities: list[str] = []
    if re.search(r"\bsignificantly\b", text, re.IGNORECASE) and not re.search(r"\b(alpha|p[- ]?value|confidence|compared with|than)\b", text, re.IGNORECASE):
        ambiguities.append("significance criterion is not explicit; use a documented default or ask")
    return QuestionPlan(
        question=text.strip(), output_shape=output_shape,
        numeric_result_required=numeric_result_required(text),
        return_only=return_only, statistical_intent=statistical_intent,
        prediction_intent=prediction_intent,
        transforms=tuple(dict.fromkeys(transforms)), join_required=join_required,
        unstructured_required=unstructured, temporal_required=bool(_TEMPORAL.search(text)),
        ambiguities=tuple(ambiguities),
    )


def build_source_map(database_description: str) -> SourceMap:
    lower = database_description.casefold()
    source_types = tuple(
        name for name, markers in (
            ("postgresql", ("postgres", "postgresql")),
            ("mongodb", ("mongo", "mongodb")),
            ("sqlite", ("sqlite",)),
            ("duckdb", ("duckdb",)),
            ("mysql", ("mysql",)),
            ("files", ("csv", "parquet", "jsonl", "file")),
        ) if any(marker in lower for marker in markers)
    )
    terms = re.findall(r"(?im)^\s*(?:table|collection|database|source)\s*[:=-]\s*([A-Za-z0-9_.-]+)", database_description)
    return SourceMap(source_types, tuple(dict.fromkeys(terms))[:50], len(source_types) > 1)


def route_specialists(plan: QuestionPlan, source_map: SourceMap) -> tuple[str, ...]:
    route: list[str] = ["cartographer"]
    if plan.transforms or plan.join_required or plan.unstructured_required or source_map.multi_source:
        route.append("transformer")
    if plan.statistical_intent != "none":
        route.append("statistician")
    if plan.statistical_intent == "experimental":
        route.append("experimenter")
    if plan.prediction_intent == "interval":
        route.append("calibrator")
    route.append("skeptic")
    return tuple(route)


def specialist_context(plan: QuestionPlan, source_map: SourceMap, route: Iterable[str]) -> str:
    payload = {
        "question_plan": plan.as_dict(),
        "source_map": source_map.as_dict(),
        "specialists": list(route),
        "instructions": (
            "Treat this as an evidence contract. Inspect before selecting fields; "
            "verify join grain and denominators; use deterministic calculations; "
            "satisfy the exact output shape. Proposed specialist names do not grant "
            "new tools—use only the supplied read-only tools."
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def review_answer(
    plan: QuestionPlan,
    answer: str,
    *,
    evidence_refs: Iterable[str],
    successful_evidence: int,
) -> ReviewVerdict:
    """Independent, deterministic answer-shape/evidence gate."""
    text = answer.strip()
    findings: list[str] = []
    if not text:
        findings.append("answer is empty")
    if successful_evidence <= 0:
        findings.append("answer has no successful data evidence")
    if not tuple(evidence_refs) and successful_evidence > 0:
        findings.append("explicit evidence references were omitted; attach all retained evidence")
    if plan.numeric_result_required and not re.search(r"[-+]?\d", text):
        findings.append("the question requires a numeric result but the answer contains no number")
    if plan.return_only:
        if "\n\n" in text or re.search(r"(?i)\b(here is|based on|because|i found|the answer is)\b", text):
            findings.append("the user requested return-only output but the answer adds explanation")
    if plan.output_shape == "scalar" and plan.return_only and len(text.splitlines()) > 1:
        findings.append("the requested scalar answer spans multiple lines")
    unsupported = re.search(r"(?i)\b(i (?:cannot|could not|was unable)|not enough information)\b", text)
    if unsupported and successful_evidence:
        findings.append("the answer abandons the task despite retained evidence")
    status = "repair" if findings else "pass"
    return ReviewVerdict(status, tuple(findings), ("answer",) if text else ())
