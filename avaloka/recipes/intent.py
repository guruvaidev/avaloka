"""From a question to an :class:`Intent`, and from an Intent to a :class:`Bound` recipe.

Three steps, each closed:

* :func:`recognize` turns JSON — a model's reply or a test fixture — into an
  Intent, or raises :class:`IntentRejected` with a reason from
  :data:`REJECTIONS`. It is a recognizer, not a parser with defaults: an
  unknown key, recipe, parameter, column, grain or window, or a column of the
  wrong role, is a rejection the caller can retry on or count.
* :func:`heuristic_intent` is the deterministic first pass of the routing
  table (design §3.1): cue patterns choose the recipe, column mentions bind the
  parameters, and three flags say what the heuristics could not settle.
* :func:`bind` is the generic binder every recipe delegates to: preconditions
  from the profile, defaults for optional parameters, an :class:`Ask` when a
  required parameter is unbound and a few columns could fill it, a
  :class:`Refusal` when none can.

A causal question is never refused here (§3.2 rule 2, §9.1): the
``causal_question`` flag travels on the Intent and the report reads it to pin
the associational wording. Nothing in this module imports ``app.*``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from avaloka.domains import normalise
from avaloka.outcomes.semantics import ColumnSemantics

from .base import GRAINS, INTENT_FLAGS, PARAM_NAMES, REASONS, Ask, Bound, Intent, RecipeSpec, Refusal
from .registry import RECIPE_IDS
from .specs import DEFAULTS, ROLE_ACCEPTS, SPECS, WINDOWS

# ── recognizing ────────────────────────────────────────────────────────────

#: Why a payload is not an Intent. Closed, so the benchmark can count each.
REJECTIONS: tuple[str, ...] = (
    "not_an_object", "missing_key", "unknown_key", "unknown_recipe", "not_a_candidate",
    "unknown_parameter", "parameter_not_in_recipe", "bad_value_shape", "unknown_column",
    "role_mismatch", "unknown_grain", "unknown_window", "unknown_flag", "bad_confidence",
    "not_json",
)

PAYLOAD_KEYS: tuple[str, ...] = ("recipe", "params", "flags", "confidence")


class IntentRejected(ValueError):
    """A payload failed the closed-vocabulary check. ``reason`` is a REJECTIONS
    member; ``detail`` names the offending value for the retry prompt."""

    def __init__(self, reason: str, detail: str = "") -> None:
        if reason not in REJECTIONS:
            raise ValueError(f"unknown rejection reason {reason!r}")
        self.reason, self.detail = reason, detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def _check_column(name: Any, role: str, columns: Mapping[str, str]) -> str:
    if not isinstance(name, str):
        raise IntentRejected("bad_value_shape", f"a column name must be a string, got {type(name).__name__}")
    if name not in columns:
        raise IntentRejected("unknown_column", f"{name!r} is not a column; columns: {sorted(columns)}")
    accepted = ROLE_ACCEPTS.get(role, (role,))
    if columns[name] not in accepted:
        raise IntentRejected("role_mismatch", f"{name!r} has role {columns[name]!r}; {role!r} needs one of {accepted}")
    return name


def _check_param(spec: RecipeSpec, key: Any, value: Any, columns: Mapping[str, str]) -> tuple[str, Any]:
    if key not in PARAM_NAMES:
        raise IntentRejected("unknown_parameter", f"{key!r}; allowed: {PARAM_NAMES}")
    pspec = next((p for p in spec.params if p.name == key), None)
    if pspec is None:
        raise IntentRejected("parameter_not_in_recipe", f"{spec.id} takes {[p.name for p in spec.params]}, not {key!r}")
    if pspec.role == "grain":
        if value not in GRAINS:
            raise IntentRejected("unknown_grain", f"{value!r}; allowed: {GRAINS}")
        return key, value
    if pspec.role == "window":
        if value not in WINDOWS:
            raise IntentRejected("unknown_window", f"{value!r}; allowed: {WINDOWS}")
        return key, value
    if pspec.many:
        if not isinstance(value, (list, tuple)) or not value:
            raise IntentRejected("bad_value_shape", f"{key!r} takes a non-empty list of column names")
        return key, tuple(_check_column(v, pspec.role, columns) for v in value)
    if isinstance(value, (list, tuple)):
        raise IntentRejected("bad_value_shape", f"{key!r} takes one column name, not a list")
    return key, _check_column(value, pspec.role, columns)


def recognize(payload: Mapping[str, Any], *, question: str, columns: Mapping[str, str],
              source: str) -> Intent:
    """Recognize ``payload`` as an Intent or raise :class:`IntentRejected`.

    ``columns`` maps column name to role. Every key of :data:`PAYLOAD_KEYS` is
    required and nothing else is allowed, so a model that invents a field or
    drops one is told so rather than quietly completed.
    """
    if not isinstance(payload, Mapping):
        raise IntentRejected("not_an_object", f"got {type(payload).__name__}")
    missing = [k for k in PAYLOAD_KEYS if k not in payload]
    if missing:
        raise IntentRejected("missing_key", f"{missing}; required: {PAYLOAD_KEYS}")
    extra = [k for k in payload if k not in PAYLOAD_KEYS]
    if extra:
        raise IntentRejected("unknown_key", f"{extra}; allowed: {PAYLOAD_KEYS}")
    recipe = payload["recipe"]
    if recipe not in RECIPE_IDS:
        raise IntentRejected("unknown_recipe", f"{recipe!r}; allowed: {RECIPE_IDS}")
    spec = SPECS[recipe]
    raw_params = payload["params"]
    if not isinstance(raw_params, Mapping):
        raise IntentRejected("bad_value_shape", "params must be an object")
    params = dict(_check_param(spec, k, v, columns) for k, v in raw_params.items())
    flags = payload["flags"]
    if not isinstance(flags, (list, tuple)):
        raise IntentRejected("bad_value_shape", "flags must be a list")
    bad = [f for f in flags if f not in INTENT_FLAGS]
    if bad:
        raise IntentRejected("unknown_flag", f"{bad}; allowed: {INTENT_FLAGS}")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        raise IntentRejected("bad_confidence", f"{confidence!r}; a number between zero and one")
    return Intent(question=question, recipe=recipe, params=params, confidence=float(confidence),
                  source=source, flags=tuple(dict.fromkeys(flags)))


# ── the causal lexicon ─────────────────────────────────────────────────────
# Copied from app/agents/claim_verifier.py (_CAUSAL_TERMS, _CAUSAL_MOVES,
# _CAUSAL_FALSE_POSITIVES at check_causal_overreach :338): avaloka must not
# import app.* (S1.08). Keep the two in step by hand; test_heuristic_intent
# pins the shared words.
_CAUSAL_TERMS: tuple[str, ...] = (
    "cause", "causes", "caused", "causing",
    "drive", "drives", "driven by", "drove",
    "lead to", "leads to", "led to",
    "result in", "results in", "resulted in",
    "because of", "due to", "responsible for",
    "impact of", "effect of",
    "influence", "influences", "determine", "determines",
)
#: Question-side additions. The verifier reads an analyst's *claim* ("X drives
#: Y"); a user's *question* asks the same thing as a verb ("does X affect Y",
#: "what impact does X have") that never appears in a written claim. Design
#: §9.1 names this residual; these are the forms the golden set found.
_CAUSAL_QUESTION_TERMS: tuple[str, ...] = (
    "affect", "affects", "affected", "affecting", "impact", "impacts", "impacted",
    "effect on", "why is", "why are", "why did", "why does", "why do", "because", "what makes",
)
_CAUSAL_MOVES = re.compile(
    r"\b(?:increase|raise|boost|elevate|reduce|lower|decrease|cut|halve|double)s?\s+"
    r"(?:the\s+|a\s+|one'?s\s+|your\s+|their\s+|its\s+)?(?:likelihood|risk|chances?|odds|probability)\b",
    re.IGNORECASE)
_CAUSAL_FALSE_POSITIVES = ("root cause", "causal analysis", "cause analysis")


def causal_term(question: str) -> str | None:
    """The first causal word the question uses, or None."""
    lowered = question.lower()
    if any(p in lowered for p in _CAUSAL_FALSE_POSITIVES):
        return None
    for term in _CAUSAL_TERMS + _CAUSAL_QUESTION_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return term
    m = _CAUSAL_MOVES.search(question)
    return m.group(0).lower() if m else None


# ── recipe cues ────────────────────────────────────────────────────────────
# Families of patterns, one per recipe. The trend, relationship and cohort
# cues are those of avaloka.outcomes.recommend.GOAL_CUES, widened to the
# phrasings of the golden set. A question's recipe is the family with the most
# distinct patterns matched; a tie is the ``ambiguous`` flag, broken by the
# order below (the recipe with the stricter preconditions first, so a tie
# resolves towards the recipe whose binder can still refuse).
_TIME_WORD = r"(?:month|week|day|quarter|year|date|time|period)s?"
#: Patterns that count half: "by X" is how every grouped recipe names its
#: optional dimension, and a causal verb is a fallback cue for the drivers
#: recipe, not a stronger one than "spike" or "versus" beside it.
WEAK = 0.5
RECIPE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ab_test", (
        r"\ba/b\b", r"\ba-b\b", r"\bab test", r"\bsplit test", r"\bexperiment", r"\bvariants?\b",
        r"\bcontrol group\b", r"\bcontrol\b", r"\btreatment\b", r"\b(?:test|control) arms?\b",
        r"\bversion [ab]\b", r"\bgroup [ab]\b", r"\blift\b", r"\bholdout\b")),
    ("cohort_retention", (
        r"\bretention\b", r"\bretain", r"\bcohorts?\b", r"\bcome back\b", r"\bcame back\b",
        r"\bstill active\b", r"\bvintages?\b", r"\bmonths? on book\b", r"\bkeep (?:coming|using|buying)\b",
        r"\brepeat (?:customers|buyers|users|purchases)\b")),
    ("regress", (
        r"\bdrivers?\b", r"\bdriving\b", r"\bdrives?\b", r"\bexplain", r"\bpredict", r"\bregress",
        r"\bdepends? on\b", r"\bdependent on\b", r"\bfactors\b", r"\bhow much of\b", r"\bdetermine",
        r"\binfluenc", r"\bmodel\b", r"\bcontribut\w* (?:most )?to\b", r"\baccount for\b", r"\bvariance explained\b",
        # A causal question with no other cue is answered by the drivers recipe,
        # associationally worded (§3.2 rule 2); the flag carries the wording.
        r"\bcaus(?:e|es|ed|ing)\b", r"\baffect", r"\bimpact", r"\beffect of\b", r"\beffect on\b",
        r"\bbecause of\b", r"\bdue to\b", r"\bresponsible for\b", r"\blead(?:s)? to\b", r"\bled to\b",
        r"\bwhat makes\b")),
    ("correlate", (
        r"\bcorrelat", r"\brelationship\b", r"\brelated\b", r"\brelate\b", r"\bmove together\b",
        r"\bassociat", r"\bconnected\b", r"\blinked\b", r"\bgo together\b", r"\bco-?vary\b",
        r"\bgo up with\b", r"\bgoes up with\b", r"\btrack each other\b", r"\bin step\b")),
    ("anomalies", (
        r"\banomal", r"\boutlier", r"\bunusual\b", r"\bspikes?\b", r"\bspiked\b", r"\bdips?\b",
        r"\bweird\b", r"\bstrange\b", r"\bodd\b", r"\babnormal", r"\bout of the ordinary\b",
        r"\bstands? out\b", r"\bsuspicious\b", r"\birregular", r"\bglitch", r"\bjump(?:s|ed)?\b")),
    ("compare_periods", (
        r"\bcompar(?:e|ed|es|ing|ison)\b", r"\bversus\b", r"\bvs\.?\b", r"\bchanged? since\b",
        r"\bthan last\b", r"\bthan (?:the )?(?:previous|prior)\b", r"\byoy\b", r"\byear[- ]over[- ]year\b",
        r"\bmonth[- ]over[- ]month\b", r"\bquarter[- ]over[- ]quarter\b", r"\bwhat changed\b",
        r"\bvariance to\b", r"\bbefore and after\b", r"\bprior period\b", r"\bbudget\b",
        r"\bagainst (?:last|the previous|the prior)\b", r"\bytd\b", r"\byear[- ]to[- ]date\b",
        r"\bprevious (?:month|quarter|year|period)\b", r"\bq[1-4]\b", r"\bup or down\b",
        r"\bthis (?:month|quarter|year) (?:against|to|and) last\b", r"\bdelta\b", r"\bdifference between\b")),
    ("trend", (
        r"\bover time\b", r"\btrends?\b", r"\btrending\b", r"\bmonthly\b", r"\bweekly\b", r"\bdaily\b",
        r"\bquarterly\b", r"\byearly\b", r"\bannual\b", r"\bgrowth\b", r"\bgrowing\b", r"\bsince \d{4}\b",
        r"\bover the (?:last|past)\b", r"\bhow has\b", r"\bevolv", r"\bseasonal", r"\btrajectory\b",
        rf"\b(?:by|per|each|every) {_TIME_WORD}\b", r"\btime series\b", r"\bmoved\b", r"\bmoving\b",
        r"\bhistor", r"\bvolatil", r"\bmomentum\b")),
    ("segment", (
        r"\bsegment", rf"\bby (?!{_TIME_WORD}\b)[a-z_]", rf"\bper (?!{_TIME_WORD}\b)[a-z_]",
        r"\bacross\b", r"\btop\b", r"\bbottom\b", r"\bbest\b", r"\bworst\b", r"\bhighest\b", r"\blowest\b",
        r"\bbreakdown\b", r"\bbreak (?:it |this )?down\b", r"\bshare\b", r"\bmix\b", r"\bconcentrat",
        r"\bsplit\b", r"\bgroup(?:ed|ing)? by\b", r"\brank", r"\blargest\b", r"\bbiggest\b",
        r"\bwhich (?:group|groups|segment|segments)\b", r"\bmatter most\b", r"\bpareto\b",
        r"\bthe most\b", r"\bthe least\b")),
    ("profile", (
        r"\bprofile\b", r"\bdata quality\b", r"\bquality\b", r"\bwhat(?:'s| is) in\b", r"\boverview\b",
        r"\bsummar(?:y|ise|ize) (?:of )?(?:the |this |my )?(?:data|file|dataset|table|csv|spreadsheet)\b",
        r"\bdescribe (?:the |this |my )?(?:data|file|dataset|table|csv|spreadsheet)\b",
        r"\btrust", r"\bmissing values\b", r"\bduplicates?\b", r"\bwhat columns\b", r"\bschema\b",
        r"\bsanity check\b", r"\bhealth\b", r"\bclean\b", r"\bfirst look\b", r"\bwhat (?:do i|do we|did i) have\b")),
    ("describe", (
        r"\bdistribution\b", r"\baverage\b", r"\bmean\b", r"\bmedian\b", r"\btypical\b",
        r"\bdescribe\b(?! (?:the |this |my )?(?:data|file|dataset|table|csv|spreadsheet)\b)", r"\bspread\b",
        r"\brange\b", r"\bpercentile", r"\bquartile", r"\bhistogram\b", r"\bskew", r"\bvariab",
        r"\bstandard deviation\b", r"\bstats? (?:for|of|on)\b",
        r"\bsummar(?:y|ise|ize)\b(?! (?:of )?(?:the |this |my )?(?:data|file|dataset|table|csv|spreadsheet)\b)",
        r"\bhow (?:big|large|long|high|low|much) (?:is|are) (?:a |an |the )?typical\b", r"\bshape\b")),
)
_WEAK_CUES: frozenset[str] = frozenset({
    rf"\bby (?!{_TIME_WORD}\b)[a-z_]", rf"\bper (?!{_TIME_WORD}\b)[a-z_]",
    r"\bcaus(?:e|es|ed|ing)\b", r"\baffect", r"\bimpact", r"\beffect of\b", r"\beffect on\b",
    r"\bbecause of\b", r"\bdue to\b", r"\bresponsible for\b", r"\blead(?:s)? to\b", r"\bled to\b",
    r"\bwhat makes\b",
})
assert _WEAK_CUES <= {c for _, cues in RECIPE_CUES for c in cues}
_CUE_RE: dict[str, list[tuple[re.Pattern[str], float]]] = {
    r: [(re.compile(c, re.IGNORECASE), WEAK if c in _WEAK_CUES else 1.0) for c in cues]
    for r, cues in RECIPE_CUES}
_TIE_ORDER: tuple[str, ...] = tuple(r for r, _ in RECIPE_CUES)
assert set(_TIE_ORDER) == set(RECIPE_IDS)


def cue_scores(text: str) -> dict[str, float]:
    """Distinct cue patterns matched per recipe, weak ones at half weight;
    recipes with none are absent."""
    out: dict[str, float] = {}
    for recipe, pats in _CUE_RE.items():
        n = sum(w for p, w in pats if p.search(text))
        if n:
            out[recipe] = n
    return out


def _choose_recipe(scores: Mapping[str, float]) -> tuple[str | None, bool]:
    """(the recipe, whether another tied with it)."""
    if not scores:
        return None, False
    best = max(scores.values())
    tied = [r for r in _TIE_ORDER if scores.get(r) == best]
    return tied[0], len(tied) > 1


# ── column mentions ────────────────────────────────────────────────────────
# A column is mentioned when every distinctive token of its name appears in
# the question, stemmed (orders ~ order, customers ~ customer). Generic
# suffix tokens (id, date, ms, ...) need not appear: "customers" names
# customer_id and "latency" names latency_ms. The score is the number of
# tokens found, so "order date" outranks a column matched on "order" alone,
# while a name typed verbatim scores the same as its full token match.
_GENERIC_TOKENS = frozenset({
    "id", "ids", "key", "code", "num", "number", "no", "date", "dt", "time", "ts", "at", "name",
    "flag", "amount", "amt", "value", "count", "total", "col", "column", "field", "type",
    "ms", "sec", "secs", "seconds", "min", "mins", "minutes", "hours", "hrs", "days", "months", "years",
    "usd", "eur", "gbp", "pct", "cnt", "avg", "sum", "max", "mean",
})
#: Words a question uses for a measure column named otherwise. Applied only to
#: columns whose role is measure, so "sales" never names a dimension.
MEASURE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "revenue": ("sales", "turnover", "income", "receipts", "takings"),
    "sales": ("revenue", "turnover"),
    "amount": ("spend", "spent", "spending", "value", "total", "sum"),
    "price": ("cost", "charge", "pricing"),
    "cost": ("spend", "expense", "expenses", "spending"),
    "quantity": ("units", "volume", "qty", "sold"),
    "qty": ("quantity", "units", "volume"),
    "profit": ("margin", "earnings"),
    "margin": ("profit", "profitability"),
    "churn": ("attrition",), "churned": ("attrition",),
    "conversion": ("converted", "convert"), "converted": ("conversion", "conversions"),
    "latency": ("delay", "delays", "response", "slowness"),
    "duration": ("length", "time taken", "how long"),
    "satisfaction": ("csat", "happiness", "happy"),
    "visits": ("traffic", "visitors", "sessions"),
    "clicks": ("ctr", "clicked"),
    "age": ("old", "older", "younger"),
    "salary": ("pay", "wages", "wage", "compensation", "earn"),
}


_STEM_RULES: tuple[tuple[str, str], ...] = (
    ("ies", "y"), ("sses", "ss"), ("xes", "x"), ("ches", "ch"), ("shes", "sh"), ("zes", "z"), ("oes", "o"),
    ("ing", ""), ("ed", ""), ("s", ""),
)


def _stem(token: str) -> str:
    """Enough of a stemmer to pair a column name with its plural or verb form
    (services ~ service, churned ~ churn); never a dictionary."""
    for suffix, repl in _STEM_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + repl
    return token


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _position(words: Sequence[str], candidates: Sequence[str]) -> int | None:
    """Index of the first question word whose stem is one of ``candidates``."""
    for i, w in enumerate(words):
        if _stem(w) in candidates or w in candidates:
            return i
    return None


def _synonym_position(token: str, qwords: Sequence[str], qtext: str) -> int | None:
    """Where the question uses a synonym of a measure word; a multi-word
    synonym ("how long") is located as a phrase."""
    syn = MEASURE_SYNONYMS.get(token, ())
    pos = _position(qwords, tuple(_stem(x) for x in syn) + tuple(syn))
    if pos is not None:
        return pos
    for phrase in (x for x in syn if " " in x):
        at = qtext.find(phrase)
        if at >= 0:
            return len(_tokens(qtext[:at]))
    return None


def _mention_score(column: str, role: str, qwords: Sequence[str], qtext: str) -> tuple[int, int]:
    """(tokens of the column's name the question uses, where the first is);
    score zero when a distinctive token is missing."""
    ctoks = _tokens(normalise(column))
    if not ctoks:
        return 0, 0
    exact = re.search(rf"(?<!\w){re.escape(column.lower())}(?!\w)", qtext)
    if exact:
        return len(ctoks), len(_tokens(qtext[: exact.start()]))   # the exact name, as typed
    distinctive = [t for t in ctoks if t not in _GENERIC_TOKENS] or ctoks
    found, first = 0, len(qwords)
    for t in distinctive:
        pos = _position(qwords, (_stem(t),))
        if pos is None and role in ROLE_ACCEPTS["measure"]:
            pos = _synonym_position(t, qwords, qtext)
        if pos is None:
            return 0, 0
        found, first = found + 1, min(first, pos)
    found += sum(1 for t in ctoks if t in _GENERIC_TOKENS and _position(qwords, (_stem(t),)) is not None)
    return found, first


def mentioned_columns(question: str, sem: Mapping[str, ColumnSemantics]) -> list[tuple[str, int]]:
    """Columns the question names, best match first, then in the order the
    question names them: ``[(name, score), ...]``."""
    qtext = question.lower()
    qwords = _tokens(qtext)
    hits = [(n, *_mention_score(n, c.role, qwords, qtext)) for n, c in sem.items()]
    hits.sort(key=lambda h: (-h[1], h[2]))
    return [(n, s) for n, s, _ in hits if s > 0]


def _by_role(sem: Mapping[str, ColumnSemantics], role: str) -> list[str]:
    return [n for n, c in sem.items() if c.role in ROLE_ACCEPTS.get(role, (role,))]


_GRAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("day", r"\b(?:daily|days?)\b"), ("week", r"\b(?:weekly|weeks?)\b"),
    ("month", r"\b(?:monthly|months?)\b"), ("quarter", r"\b(?:quarterly|quarters?)\b"),
    ("year", r"\b(?:yearly|annual(?:ly)?|years?)\b"),
)
_NOT_A_GRAIN = re.compile(r"\b(?:last|past|this|previous|prior|next|a|one)\s+$")


def explicit_grain(question: str) -> str | None:
    """A grain the question names as a reporting step ("monthly", "by week"),
    not as a window ("last month"). Two different grains → None."""
    text = question.lower()
    found: list[str] = []
    for grain, pat in _GRAIN_WORDS:
        for m in re.finditer(pat, text):
            if _NOT_A_GRAIN.search(text[: m.start()]) and not re.match(r"\b(?:daily|weekly|monthly|quarterly|yearly|annual)", m.group(0)):
                continue
            found.append(grain)
            break
    found = list(dict.fromkeys(found))
    return found[0] if len(found) == 1 else None


_WINDOW_WORDS: tuple[tuple[str, str], ...] = (
    ("ytd", r"\b(?:ytd|year[- ]to[- ]date)\b"),
    ("last_year", r"\b(?:last|previous|prior) year\b|\byoy\b|\byear[- ]over[- ]year\b"),
    ("last_quarter", r"\b(?:last|previous|prior) quarter\b|\bquarter[- ]over[- ]quarter\b|\bqoq\b"),
    ("last_month", r"\b(?:last|previous|prior) month\b|\bmonth[- ]over[- ]month\b|\bmom\b"),
)


def explicit_window(question: str) -> str | None:
    for window, pat in _WINDOW_WORDS:
        if re.search(pat, question.lower()):
            return window
    return None


# ── the heuristic pass ─────────────────────────────────────────────────────
_CLAUSE_SPLIT = re.compile(r";\s*|\band also\b|\band then\b|\bthen\b|\?\s+(?=\w)", re.IGNORECASE)
#: "does X affect Y", "impact of X on Y": the outcome is the later measure.
_X_ACTS_ON_Y = re.compile(
    r"\b(?:affect|impact|influenc|determin|caus|driv|predict\w* \w+ of|effect of|lead|led|explain\w* \w+ of)", re.IGNORECASE)


def _clauses(question: str) -> list[str]:
    return [c for c in _CLAUSE_SPLIT.split(question) if c.strip()]


def _multi_question(clauses: Sequence[str]) -> bool:
    """Two clauses that each cue a recipe, and not the same one (§3.2 rule 3)."""
    if len(clauses) < 2:
        return False
    recipes = {_choose_recipe(cue_scores(c))[0] for c in clauses}
    recipes.discard(None)
    return len(recipes) >= 2


def _mask_time_columns(question: str, sem: Mapping[str, ColumnSemantics]) -> str:
    """The question with each time column's name replaced by "date", so
    "trend by order_date" cues a time step, not a grouping."""
    text = question
    for n, c in sem.items():
        if c.role == "time":
            text = re.sub(rf"(?<!\w){re.escape(n)}(?!\w)", "date", text, flags=re.IGNORECASE)
    return text


def _pick_outcome(question: str, fits: Sequence[tuple[str, int]]) -> str:
    """Of equally good measure mentions, the regression outcome: the later one
    when the question says one acts on the other, else the first named."""
    return fits[-1][0] if _X_ACTS_ON_Y.search(question) else fits[0][0]


def _bind_mentions(spec: RecipeSpec, question: str, sem: Mapping[str, ColumnSemantics]
                   ) -> tuple[dict[str, Any], bool, bool]:
    """(params, every required bound, a required param had two equal matches)."""
    mentions = mentioned_columns(question, sem)
    used: set[str] = set()
    params: dict[str, Any] = {}
    ambiguous = False
    for p in spec.params:
        if p.role == "grain":
            g = explicit_grain(question)
            if g:
                params[p.name] = g
            continue
        if p.role == "window":
            w = explicit_window(question)
            if w:
                params[p.name] = w
            continue
        fits = [(n, s) for n, s in mentions if n not in used and sem[n].role in ROLE_ACCEPTS[p.role]]
        if p.many:
            if fits:
                params[p.name] = tuple(n for n, _ in fits)
                used.update(params[p.name])
            continue
        if fits:
            if len(fits) > 1 and fits[0][1] == fits[1][1] and p.required:
                if p.name != "outcome":
                    ambiguous = True        # two columns fit equally; the binder will ask
                    continue
                fits = [(_pick_outcome(question, fits), fits[0][1])]
            params[p.name] = fits[0][0]
            used.add(fits[0][0])
        elif p.role in ("time", "id"):
            # Questions rarely name the date or the identifier column; when the
            # table has exactly one, there is nothing to choose.
            only = [t for t in _by_role(sem, p.role) if t not in used]
            if len(only) == 1:
                params[p.name] = only[0]
                used.add(only[0])
    complete = all(p.name in params for p in spec.params if p.required)
    return params, complete, ambiguous


def heuristic_intent(question: str, sem: Mapping[str, ColumnSemantics],
                     profile: Mapping[str, Any]) -> Intent:
    """The deterministic intent (design §3.1), with no model.

    Confidence: 1.0 when exactly one cue family matched and every required
    parameter bound uniquely; minus 0.2 when other families also matched,
    minus 0.3 for a tie (``ambiguous``), minus 0.3 for an unbound required
    parameter, minus 0.2 for ``multi_question``; 0.0 when no cue matched.
    ``profile`` is accepted for parity with :func:`bind` and unused here: the
    heuristics read only the question and the column roles.
    """
    del profile
    clauses = _clauses(question)
    multi = _multi_question(clauses)
    # A two-part question is routed out of the lane (§3.2 rule 3); the intent
    # records the first part, which is what the lane would answer alone.
    focus = clauses[0] if multi else question
    scores = cue_scores(_mask_time_columns(focus, sem))
    recipe, tie = _choose_recipe(scores)
    flags: list[str] = []
    confidence = 0.0
    params: dict[str, Any] = {}
    if recipe is not None:
        params, complete, equal_matches = _bind_mentions(SPECS[recipe], focus, sem)
        confidence = 1.0
        if len(scores) > 1:
            confidence -= 0.2
        if tie or equal_matches:
            flags.append("ambiguous")
            confidence -= 0.3
        if not complete:
            confidence -= 0.3
    if causal_term(question):
        flags.append("causal_question")
    if multi:
        flags.append("multi_question")
        confidence -= 0.2
    return Intent(question=question, recipe=recipe, params=params,
                  confidence=max(0.0, min(1.0, round(confidence, 2))), source="heuristic",
                  flags=tuple(flags))


# ── binding ────────────────────────────────────────────────────────────────
ASK_LIMIT = 6   # more candidate columns than this and the Ask names no options

#: Which refusal a failed precondition is, and the words for it (no digits).
_PRECONDITION_REFUSALS: dict[str, tuple[str, str]] = {
    "has_time": ("no_time_column", "This needs a date or time column, and none was found."),
    "has_measure": ("no_measure", "This needs a numeric column to measure, and none was found."),
    "has_two_measures": ("no_measure", "This needs at least two numeric columns, and fewer were found."),
    "has_dimension": ("unsupported_question", "This needs a column with a few categories to group by, and none was found."),
    "has_id": ("unsupported_question", "This needs an identifier column for the entities to follow, and none was found."),
    "has_arm": ("unsupported_question", "This needs an assignment column with a handful of arms, and none was found."),
    "min_rows_30": ("insufficient_rows", "There are too few rows for this analysis to say anything reliable."),
    "min_periods_4": ("insufficient_periods", "There are too few time periods to read a trend from."),
}
assert all(r in REASONS for r, _ in _PRECONDITION_REFUSALS.values())

_ASK_TEXT: dict[str, str] = {
    "measure": "Which column holds the figure to analyse?",
    "measures": "Which numeric columns should be compared with each other?",
    "dimension": "Which column should the groups come from?",
    "time": "Which column holds the dates?",
    "id": "Which column identifies the entity to follow over time?",
    "arm": "Which column says which arm each row was assigned to?",
    "outcome": "Which column is the outcome to measure?",
}
_NO_CANDIDATE_REASON: dict[str, str] = {
    "measure": "no_measure", "time": "no_time_column", "dimension": "unsupported_question",
    "id": "unsupported_question",
}


def _rows(profile: Mapping[str, Any]) -> int | None:
    """Row count from either profile shape the CLI writes; None when absent."""
    for holder in (profile, profile.get("dataset") or {}):
        n = holder.get("n_rows") if isinstance(holder, Mapping) else None
        if isinstance(n, int) and not isinstance(n, bool):
            return n
    return None


def _periods(profile: Mapping[str, Any]) -> int | None:
    n = profile.get("n_periods")
    return n if isinstance(n, int) and not isinstance(n, bool) else None


def _has_arm(sem: Mapping[str, ColumnSemantics]) -> bool:
    return any(c.role == "dimension" and (c.distinct == 0 or 2 <= c.distinct <= 4) for c in sem.values())


def precondition_holds(name: str, sem: Mapping[str, ColumnSemantics], profile: Mapping[str, Any]) -> bool:
    """Answer one PRECONDITIONS member from roles and the profile, never rows.
    A count the profile lacks (rows, periods) is taken as satisfied and left
    to the recipe's gate, which reads the rows and can refuse with the figure."""
    if name == "has_time":
        return bool(_by_role(sem, "time"))
    if name == "has_measure":
        return bool(_by_role(sem, "measure"))
    if name == "has_two_measures":
        return len(_by_role(sem, "measure")) >= 2
    if name == "has_dimension":
        return bool(_by_role(sem, "dimension"))
    if name == "has_id":
        return bool(_by_role(sem, "id"))
    if name == "has_arm":
        return _has_arm(sem)
    if name == "min_rows_30":
        n = _rows(profile)
        return n is None or n >= 30
    if name == "min_periods_4":
        n = _periods(profile)
        return n is None or n >= 4
    raise ValueError(f"unknown precondition {name!r}")


def _default_for(spec: RecipeSpec, name: str, params: Mapping[str, Any],
                 sem: Mapping[str, ColumnSemantics]) -> str | None:
    if name == "grain" and "time" in params:
        return sem[params["time"]].grain
    return DEFAULTS.get((spec.id, name))


def _candidates(role: str, sem: Mapping[str, ColumnSemantics], used: set[str]) -> list[str]:
    if role == "arm":
        return [n for n, c in sem.items() if n not in used and c.role == "dimension"
                and (c.distinct == 0 or 2 <= c.distinct <= 4)]
    return [n for n in _by_role(sem, role) if n not in used]


def bind(intent: Intent, spec: RecipeSpec, sem: Mapping[str, ColumnSemantics],
         profile: Mapping[str, Any]) -> Bound | Ask | Refusal:
    """Bind ``intent`` to ``spec``: preconditions, then one decision per parameter.

    A required column parameter the question did not name is bound when
    exactly one column of its role exists (there is nothing to ask), asked
    about with the candidates as options when a few exist, asked about with no
    options when many do, and refused when none do. A ``many`` parameter takes
    every candidate when a few exist. Optional column parameters stay unbound;
    optional grain and window take their defaults. The causal flag is not read
    here: the question binds normally and the report uses associational
    wording (§3.2 rule 2).
    """
    for name in spec.preconditions:
        if not precondition_holds(name, sem, profile):
            reason, text = _PRECONDITION_REFUSALS[name]
            return Refusal(reason, text)
    columns = {n: c.role for n, c in sem.items()}
    params: dict[str, Any] = {}
    used: set[str] = set()
    for p in spec.params:
        if p.name in intent.params:
            _, value = _check_param(spec, p.name, intent.params[p.name], columns)
            params[p.name] = value
            used.update(value if isinstance(value, tuple) else (value,))
            continue
        if p.role in ("grain", "window"):
            default = _default_for(spec, p.name, params, sem)
            if default is not None:
                params[p.name] = default
            continue
        if not p.required:
            continue
        role = "arm" if p.name == "arm" else p.role
        fits = _candidates(role, sem, used)
        if not fits:
            reason = _NO_CANDIDATE_REASON.get(p.role, "unsupported_question")
            return Refusal(reason, _PRECONDITION_REFUSALS["has_" + p.role][1]
                           if "has_" + p.role in _PRECONDITION_REFUSALS else _ASK_TEXT[p.name])
        if p.many:
            if len(fits) <= ASK_LIMIT:
                params[p.name] = tuple(fits)
                used.update(fits)
                continue
            return Ask(p.name, _ASK_TEXT[p.name])
        if len(fits) == 1:
            params[p.name] = fits[0]
            used.add(fits[0])
            continue
        return Ask(p.name, _ASK_TEXT[p.name], tuple(fits) if len(fits) <= ASK_LIMIT else ())
    return Bound(recipe_id=spec.id, version=spec.version, params=params)
