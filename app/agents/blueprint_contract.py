# blueprint_contract.py
"""Machine-checkable acceptance criteria for the coder's blueprint.

Why this module exists
----------------------
The blueprint step (``app/agents/coder.py::_generate_pseudocode``) used to emit
free-text numbered steps and nothing else. Nothing downstream could check them:
``coder_pseudocode`` was written into state, injected into the coder prompt, and
forwarded to the reporting layer, but ``app/agents/validator.py`` never read it
(grep for "pseudocode" in that file returned zero hits).

So the plan stated an intent that no component could verify. The two halves of
the pipeline compensated in the only way left to them, and both grew a rulebook
of accumulated production failures:

  * the blueprint prompt grew ~21 numbered special-case rules (YEAR FILTERING,
    SPARSE VS DENSE, AGGREGATION / CHART OUTPUT, ...), and
  * the validator grew ~20 bespoke AST visitors.

Rules only ever cover the failures that have already happened. This module
replaces the *mechanism*: the blueprint now also emits a **contract** -- a small,
closed vocabulary of postconditions on the result frame -- and the validator
checks the plan's own stated contract instead of guessing from a rulebook.

Why a structured schema rather than prose or executable assertions
-----------------------------------------------------------------
Three options were available for the criteria format.

1. *Natural language.* Rejected. The pipeline already has an LLM reading prose
   criteria (``logical_semantic_validator_node``) and it is the least reliable
   gate in the system -- in default mode it cannot even block (see
   ``validator.py``: non-fabrication verdicts return ``logical_semantic_error:
   False`` with feedback downgraded to advisory). Adding more prose adds tokens
   and variance, not reliability.

2. *Executable assertions written by the model.* Rejected. It maximises
   expressiveness but the assertions then need validating themselves: they can
   be vacuous (``assert True``), wrong, or unsafe, and they would run against
   user data. Trading one unverified artifact for another is not progress.

3. *A structured schema over a closed vocabulary, compiled by us into checks.*
   Chosen. Because the vocabulary is closed we can (a) reject a malformed or
   vacuous contract before trusting it, (b) compile it deterministically with
   no LLM in the loop, and (c) name the exact violated postcondition in the
   repair feedback. Crucially, it is the part that generalises: one
   ``row_relation: one_per_group`` predicate subsumes the whole four-part
   "AGGREGATION / CHART OUTPUT RULE" and keeps working for group-bys nobody
   enumerated in advance.

The vocabulary is deliberately small. Anything that cannot be expressed in it
is simply not asserted -- an unstated postcondition is better than a
special-case rule, because a rule that covers one production incident is the
bug this module exists to remove.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- closed vocabulary -----------------------------------------------------

RESULT_KINDS = {
    "aggregate",      # one row per group; strictly fewer rows than the input
    "row_transform",  # same rows, changed/added columns
    "subset",         # a filtered selection of input rows
    "reshape",        # pivot / unpivot: row count changes by construction
    "scalar",         # a single summary value, returned as a one-row frame
}

ROW_RELATIONS = {
    "one_per_group",     # requires group_by
    "same_as_input",
    "fewer_than_input",
    "at_most",           # requires n
    "exactly",           # requires n
    "unconstrained",
}

DTYPES = {"numeric", "string", "datetime", "boolean", "any"}
ROLES = {"key", "metric", "passthrough"}

MAX_COLUMNS = 64


class ContractError(ValueError):
    """The contract itself is malformed -- distinct from a contract violation."""


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    role: str = "metric"
    dtype: str = "any"
    min: Optional[float] = None
    max: Optional[float] = None

    @staticmethod
    def parse(raw: Any) -> "ColumnSpec":
        if isinstance(raw, str):
            return ColumnSpec(name=raw)
        if not isinstance(raw, dict):
            raise ContractError(f"column spec must be a string or object, got {type(raw).__name__}")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ContractError("column spec needs a non-empty 'name'")
        role = str(raw.get("role") or "metric").lower()
        dtype = str(raw.get("dtype") or "any").lower()
        if role not in ROLES:
            role = "metric"
        if dtype not in DTYPES:
            dtype = "any"

        def _num(key: str) -> Optional[float]:
            v = raw.get(key)
            if v is None or isinstance(v, bool):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        return ColumnSpec(name=name.strip(), role=role, dtype=dtype,
                          min=_num("min"), max=_num("max"))


@dataclass(frozen=True)
class RowRelation:
    type: str = "unconstrained"
    group_by: Tuple[str, ...] = ()
    n: Optional[int] = None

    @staticmethod
    def parse(raw: Any) -> "RowRelation":
        if raw is None:
            return RowRelation()
        if isinstance(raw, str):
            t = raw.lower().strip()
            return RowRelation(type=t if t in ROW_RELATIONS else "unconstrained")
        if not isinstance(raw, dict):
            return RowRelation()
        t = str(raw.get("type") or "unconstrained").lower().strip()
        if t not in ROW_RELATIONS:
            t = "unconstrained"
        gb = raw.get("group_by") or []
        if isinstance(gb, str):
            gb = [gb]
        group_by = tuple(str(c) for c in gb if isinstance(c, (str, int)))
        n = raw.get("n")
        try:
            n = int(n) if n is not None else None
        except (TypeError, ValueError):
            n = None
        # A group relation without a group is not a constraint.
        if t == "one_per_group" and not group_by:
            t = "fewer_than_input"
        if t in {"at_most", "exactly"} and n is None:
            t = "unconstrained"
        return RowRelation(type=t, group_by=group_by, n=n)


@dataclass(frozen=True)
class BlueprintContract:
    """The blueprint's stated postconditions on the result frame."""

    result_kind: str = "row_transform"
    source_columns: Tuple[str, ...] = ()
    result_columns: Tuple[ColumnSpec, ...] = ()
    row_relation: RowRelation = field(default_factory=RowRelation)
    unique_key: Tuple[str, ...] = ()
    not_null: Tuple[str, ...] = ()

    # ---- construction ----

    @staticmethod
    def parse(raw: Any) -> "BlueprintContract":
        if not isinstance(raw, dict):
            raise ContractError("contract must be a JSON object")

        kind = str(raw.get("result_kind") or "row_transform").lower().strip()
        if kind not in RESULT_KINDS:
            raise ContractError(f"unknown result_kind {kind!r}; expected one of {sorted(RESULT_KINDS)}")

        cols_raw = raw.get("result_columns") or []
        if not isinstance(cols_raw, list):
            raise ContractError("result_columns must be a list")
        if len(cols_raw) > MAX_COLUMNS:
            cols_raw = cols_raw[:MAX_COLUMNS]
        cols = tuple(ColumnSpec.parse(c) for c in cols_raw)

        src = raw.get("source_columns") or []
        if isinstance(src, str):
            src = [src]
        if not isinstance(src, list):
            raise ContractError("source_columns must be a list")

        def _strs(key: str) -> Tuple[str, ...]:
            v = raw.get(key) or []
            if isinstance(v, str):
                v = [v]
            if not isinstance(v, list):
                return ()
            return tuple(str(x) for x in v if isinstance(x, (str, int)))

        return BlueprintContract(
            result_kind=kind,
            source_columns=tuple(str(c) for c in src if isinstance(c, (str, int))),
            result_columns=cols,
            row_relation=RowRelation.parse(raw.get("row_relation")),
            unique_key=_strs("unique_key"),
            not_null=_strs("not_null"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_kind": self.result_kind,
            "source_columns": list(self.source_columns),
            "result_columns": [
                {k: v for k, v in
                 {"name": c.name, "role": c.role, "dtype": c.dtype,
                  "min": c.min, "max": c.max}.items() if v is not None}
                for c in self.result_columns
            ],
            "row_relation": {
                k: v for k, v in
                {"type": self.row_relation.type,
                 "group_by": list(self.row_relation.group_by) or None,
                 "n": self.row_relation.n}.items() if v is not None
            },
            "unique_key": list(self.unique_key),
            "not_null": list(self.not_null),
        }

    # ---- self-assessment ----

    def is_vacuous(self) -> bool:
        """True when the contract asserts nothing worth checking.

        A contract that constrains nothing is worse than no contract: it looks
        like a gate while passing everything. Such contracts are discarded
        rather than trusted.
        """
        has_shape = self.row_relation.type != "unconstrained"
        has_cols = bool(self.result_columns)
        has_keys = bool(self.unique_key) or bool(self.not_null)
        return not (has_shape or has_cols or has_keys)

    def unknown_source_columns(self, schema_columns: Any) -> List[str]:
        """source_columns that do not exist in the dataset schema.

        This is the cheapest and most valuable static check: it catches a
        blueprint built on a hallucinated column before a line of code is
        written, which is the failure the whole pipeline is worst at.
        """
        known = {_norm(c) for c in _as_columns(schema_columns)}
        if not known:
            return []
        return [c for c in self.source_columns if _norm(c) not in known]


# --- helpers ---------------------------------------------------------------

def _norm(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _as_columns(schema: Any) -> List[str]:
    if isinstance(schema, dict):
        return [str(k) for k in schema.keys()]
    if isinstance(schema, (list, tuple, set)):
        return [str(c) for c in schema]
    return []


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_contract_block(text: str) -> Optional[Dict[str, Any]]:
    """Pull the contract JSON out of an LLM response.

    The blueprint response is "numbered steps, then a fenced JSON object". Both
    shapes are tolerated: a fenced block, or a bare trailing object.
    """
    if not text:
        return None

    for match in reversed(_JSON_BLOCK.findall(text)):
        try:
            obj = json.loads(match)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj.get("contract") if isinstance(obj.get("contract"), dict) else obj

    # Bare trailing object: scan back from the last closing brace.
    end = text.rfind("}")
    while end != -1:
        start = text.rfind("{", 0, end)
        while start != -1:
            try:
                obj = json.loads(text[start:end + 1])
            except ValueError:
                start = text.rfind("{", 0, start)
                continue
            if isinstance(obj, dict):
                return obj.get("contract") if isinstance(obj.get("contract"), dict) else obj
            break
        break
    return None


def split_steps_and_contract(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Separate the prose steps from the contract block."""
    if not text:
        return "", None
    raw = extract_contract_block(text)
    steps = _JSON_BLOCK.sub("", text).strip()
    if not steps:
        steps = text.strip()
    return steps, raw


def parse_blueprint(text: str, *, schema: Any = None) -> Tuple[str, Optional[BlueprintContract], List[str]]:
    """Parse a blueprint response into (steps, contract, problems).

    ``problems`` describes why a contract was discarded; a discarded contract
    degrades to None so the pipeline behaves exactly as it did before rather
    than blocking on a contract it could not read.
    """
    steps, raw = split_steps_and_contract(text)
    problems: List[str] = []
    if raw is None:
        return steps, None, ["blueprint did not include a contract block"]
    try:
        contract = BlueprintContract.parse(raw)
    except ContractError as exc:
        return steps, None, [f"contract rejected: {exc}"]

    if contract.is_vacuous():
        return steps, None, ["contract rejected: asserts no checkable postcondition"]

    if schema is not None:
        unknown = contract.unknown_source_columns(schema)
        if unknown:
            problems.append(
                "contract references columns that are not in the schema: "
                + ", ".join(sorted(unknown))
            )
    return steps, contract, problems


# --- the checker -----------------------------------------------------------

@dataclass
class Violation:
    rule: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.rule}: {self.detail}"


_PLACEHOLDER_RE = re.compile(r"[<>*{}]|\b(each|every|any|one per|per)\b", re.IGNORECASE)


def is_placeholder_column(name: str) -> bool:
    """Is this declared column a label the plan cannot know in advance?

    A reshape names its output columns from the DATA -- pivoting Medical
    Condition produces one column per condition -- so a blueprint can only write
    something like "<any medical condition>" or "one column per condition".
    Demanding such a name literally rejects a correct pivot, which is a false
    rejection of exactly the kind this gate must not produce. The postconditions
    that do not depend on unknown labels (row counts, key uniqueness, the named
    index column) still apply.
    """
    return bool(_PLACEHOLDER_RE.search(str(name)))


# --- where an exact row count comes from ---------------------------------------
#
# The blueprint model writes ``row_relation: exactly n`` itself, and it can
# invent n: rc11 p1 ("what are the strongest predictors of stroke?") declared
# n=11 because its source_columns held 11 names -- the target ``stroke``
# included -- and the correct 10-predictor answer was HARD-rejected. A count
# is therefore enforced only when one of the sources below reproduces it.

COUNT_SOURCES = ("data", "question", "plan_columns")

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "fifty": 50, "hundred": 100,
}
_NUM = r"(\d{1,4}|" + "|".join(_NUMBER_WORDS) + r")"
_SUPERLATIVE = (r"(?:most|least|highest|lowest|largest|smallest|biggest|best|worst|"
                r"top|bottom|strongest|weakest|longest|shortest|oldest|newest|youngest|"
                r"cheapest|busiest|common|frequent|popular|expensive|important)")
_QUESTION_COUNT_RES = (
    # "top 5", "first ten", "bottom 3"
    re.compile(r"\b(?:top|bottom|first|last)\s+" + _NUM + r"\b", re.IGNORECASE),
    # "5 most common", "three highest", "10 strongest"
    re.compile(r"\b" + _NUM + r"\s+" + _SUPERLATIVE + r"\b", re.IGNORECASE),
)

def is_identifier_column(name: str, frame: Any = None) -> bool:
    """Is *name* a row identifier? #573's ``IDENTIFIER_RULE``
    (``avaloka.goal_analysis.is_identifier_column``) judged on the input
    frame's values; one rule for the ranking, the verifier and this contract.

    The rule reads values, so a column absent from the frame is not judged
    and is not treated as an identifier.
    """
    actual = _resolve(frame.columns, name) if frame is not None and len(frame) else None
    if actual is None:
        return False
    from avaloka.goal_analysis import is_identifier_column as _rule  # local: pandas-free import

    return bool(_rule(str(actual), frame[actual]))


def question_counts(question: str) -> List[int]:
    """Row counts the question itself states: "top 5", "three highest"."""
    found: List[int] = []
    for pattern in _QUESTION_COUNT_RES:
        for raw in pattern.findall(question or ""):
            n = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw.lower()]
            if n > 0 and n not in found:
                found.append(n)
    return found


def question_targets(contract: "BlueprintContract", question: str) -> List[str]:
    """source_columns the question names -- the outcome of "predictors of stroke"."""
    text = " " + re.sub(r"[^a-z0-9]+", " ", (question or "").lower()) + " "
    targets = []
    for col in contract.source_columns:
        spaced = re.sub(r"[^a-z0-9]+", " ", str(col).lower()).strip()
        if spaced and f" {spaced} " in text:
            targets.append(col)
    return targets


@dataclass(frozen=True)
class RowCountDecision:
    """Why an ``exactly n`` count is HARD or ADVISORY.

    ``source`` is the first of :data:`COUNT_SOURCES` whose derived count equals
    n, or ``"model"`` when none does. ``derived`` holds every count that could
    be derived, by source, so a trace shows what n was compared against.
    """

    n: int
    source: str
    derived: Dict[str, Any] = field(default_factory=dict)
    #: False when the input frame is not the rows the result was computed on,
    #: so no count was derived from it (see :func:`code_reads_own_data`).
    data_checked: bool = True
    #: What the ``data`` source was, in words: "100-row validation sample"
    #: when the input frame is the validator's sample (see :func:`data_label`).
    data_label: str = "data"

    @property
    def hard(self) -> bool:
        return self.source != "model"

    def _name(self, source: str) -> str:
        return self.data_label if source == "data" else source.replace("_", " ")

    def describe(self) -> str:
        if self.hard:
            return f"derived from the {self._name(self.source)}"
        if not self.data_checked:
            prefix = "the validation sample is not the result's only input, so no count was taken from it; "
            if not self.derived:
                return prefix + "nothing in the question or the plan's columns gives a count"
        else:
            prefix = ""
            if not self.derived:
                return f"nothing in the {self.data_label}, the question or the plan's columns gives a count"
        shown = ", ".join(f"{'the ' if k == 'data' else ''}{self._name(k)} gives {v}"
                          for k, v in self.derived.items())
        return f"{prefix}none of the derivable counts matches: {shown}"

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "source": self.source, "hard": self.hard, "derived": dict(self.derived),
                "data_checked": self.data_checked}


def exact_count_decision(contract: "BlueprintContract", source: Any,
                         question: str = "", *, observed: bool = True,
                         label: str = "data") -> RowCountDecision:
    """Decide whether the contract's ``exactly n`` is derivable.

    n is **derivable** -- and so enforced HARD -- when it equals a count
    computed deterministically, with no model in the loop, from one of:

    * ``data``: the number of distinct value combinations, in the input frame,
      of the plan's group-by columns (``row_relation.group_by``; failing that,
      its key columns -- ``unique_key`` or role ``key`` -- that exist in the
      input);
    * ``question``: a count the question states, "top 5" / "first ten" /
      "3 highest" (:func:`question_counts`);
    * ``plan_columns``: the number of the plan's ``source_columns`` once the
      target and the identifier columns are removed. The target is a source
      column the question names (:func:`question_targets`); when the question
      names none, the list's membership is unknown and nothing is derived.

    Any other n was invented by the blueprint model and is ADVISORY.

    ``observed=False`` says *source* is not the rows the result was computed
    on (the code read its own data): no ``data`` count is taken from it.
    """
    n = contract.row_relation.n
    derived: Dict[str, Any] = {}
    have_source = source is not None and hasattr(source, "columns") and len(source)

    keys = list(contract.row_relation.group_by)
    if not keys:
        keys = list(contract.unique_key) or [c.name for c in contract.result_columns if c.role == "key"]
    if have_source and keys and observed:
        resolved = [_resolve(source.columns, k) for k in keys]
        if all(r is not None for r in resolved):
            derived["data"] = int(source[list(resolved)].drop_duplicates().shape[0])

    counts = question_counts(question)
    if counts:
        derived["question"] = counts if len(counts) > 1 else counts[0]

    targets = question_targets(contract, question)
    if targets:
        frame = source if have_source else None
        kept = [c for c in contract.source_columns
                if c not in targets and not is_identifier_column(c, frame)]
        derived["plan_columns"] = len(kept)

    for name in COUNT_SOURCES:
        value = derived.get(name)
        if value == n or (isinstance(value, list) and n in value):
            return RowCountDecision(n=n, source=name, derived=derived, data_checked=observed,
                                    data_label=label)
    return RowCountDecision(n=n, source="model", derived=derived, data_checked=observed, data_label=label)


# --- where a min/max bound comes from ------------------------------------------
#
# The same failure as the invented exact count, on a column's range: rc16's
# taxi turn ("What is the average fare_amount by payment_type?") declared
# ``average_fare_amount`` min 0.0. Taxi fares include refunds and adjustments,
# so a payment type's mean was -1.52 and a correct result was HARD-rejected in
# every run. A bound is therefore enforced only when a deterministic fact
# guarantees that a correct result already satisfies it.

BOUND_SOURCES = ("data", "domain", "question")

#: Aggregations whose value lies within the source column's observed range.
_WITHIN_RANGE_AGGS = frozenset({"mean", "avg", "average", "median", "min", "minimum", "max",
                                "maximum", "mode", "first", "last", "typical"})
_SUM_AGGS = frozenset({"sum", "total"})
_COUNT_TOKENS = frozenset({"count", "counts", "cnt", "num", "number", "rows", "size",
                           "frequency", "freq", "nunique", "distinct", "unique"})
_SPREAD_TOKENS = frozenset({"std", "stdev", "stddev", "variance", "var", "iqr", "range"})
_FRACTION_TOKENS = frozenset({"rate", "ratio", "proportion", "share", "fraction",
                              "probability", "prob"})
_PERCENT_TOKENS = frozenset({"percent", "percentage", "pct"})
#: A rate *of change* can be negative or exceed 1; these tokens void the
#: fraction/percent domain facts.
_CHANGE_TOKENS = frozenset({"change", "growth", "delta", "diff", "difference", "lift",
                            "return", "yoy", "mom", "increase", "decrease", "z", "zscore"})
#: Words too generic to tie a question's qualifier to a column.
_GENERIC_TOKENS = frozenset({"amount", "value", "values", "total", "avg", "average", "mean",
                             "sum", "count", "number", "the", "of", "per", "by"})

_QNUM = r"(-?\d+(?:\.\d+)?)"
_QUESTION_BOUND_RES = (
    ("min", re.compile(r"\b(?:only\s+)?(?:positive|non-?negative|nonnegative)\b", re.IGNORECASE), 0.0),
    ("min", re.compile(r"(?:\bat\s+least|\bno\s+less\s+than|\bgreater\s+than|\bmore\s+than|\babove|"
                       r"\bover|\bexceeding|>=|>)\s*\$?" + _QNUM, re.IGNORECASE), None),
    ("max", re.compile(r"(?:\bat\s+most|\bno\s+more\s+than|\bless\s+than|\bunder|\bbelow|"
                       r"\bnot\s+exceeding|<=|<)\s*\$?" + _QNUM, re.IGNORECASE), None),
    ("between", re.compile(r"\bbetween\s+\$?" + _QNUM + r"\s+and\s+\$?" + _QNUM, re.IGNORECASE), None),
)


def _tokens(name: Any) -> set:
    return {t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(name))}


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _referenced_source_columns(name: str, contract: "BlueprintContract", source: Any) -> List[str]:
    """Source columns a result column is computed from, read off its name:
    ``average_fare_amount`` -> ``fare_amount``. Exact matches win."""
    candidates = list(contract.source_columns)
    if source is not None and hasattr(source, "columns"):
        candidates += [str(c) for c in source.columns if str(c) not in candidates]
    want = _norm(name)
    exact = [c for c in candidates if _norm(c) == want]
    if exact:
        return exact[:1]
    return [c for c in candidates if len(_norm(c)) >= 3 and _norm(c) in want]


def _data_interval(spec: ColumnSpec, contract: "BlueprintContract",
                   source: Any) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """The range a correct value must lie in, from the input's observed values
    under the aggregation the column's name states."""
    import pandas as pd

    if source is None or not hasattr(source, "columns") or not len(source):
        return None
    tokens = _tokens(spec.name)
    refs = _referenced_source_columns(spec.name, contract, source)
    if len(refs) != 1:
        return None
    actual = _resolve(source.columns, refs[0])
    if actual is None:
        return None
    values = pd.to_numeric(source[actual], errors="coerce").dropna()
    if not len(values):
        return None
    lo, hi = float(values.min()), float(values.max())
    same_column = _norm(refs[0]) == _norm(spec.name)
    if tokens & _WITHIN_RANGE_AGGS or (same_column and contract.result_kind == "subset"):
        return lo, hi
    if tokens & _SUM_AGGS:
        return (0.0 if lo >= 0 else None), (0.0 if hi <= 0 else None)
    return None


def _domain_interval(spec: ColumnSpec) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Range facts that hold for any data: a count or a spread is >= 0, a
    rate or share is in [0, 1], a percentage in [0, 100]."""
    tokens = _tokens(spec.name)
    if tokens & _CHANGE_TOKENS:
        return None
    if tokens & _PERCENT_TOKENS:
        return 0.0, 100.0
    if tokens & _FRACTION_TOKENS:
        return 0.0, 1.0
    if tokens & (_COUNT_TOKENS | _SPREAD_TOKENS) or (tokens and tokens <= {"n"}):
        return 0.0, None
    return None


def question_bounds(question: str, words: set) -> Tuple[Optional[float], Optional[float]]:
    """Bounds the question states about a column named by one of *words*:
    "only positive fares", "fares of at least 5", "between 10 and 20".

    A qualifier counts only when one of *words* (singularised) is within four
    words of it, so "trips over 5 miles" bounds distance, not the fare.
    """
    if not question or not words:
        return None, None
    lo: Optional[float] = None
    hi: Optional[float] = None
    for side, pattern, fixed in _QUESTION_BOUND_RES:
        for m in pattern.finditer(question):
            before = re.findall(r"[a-z0-9]+", question[:m.start()].lower())[-4:]
            after = re.findall(r"[a-z]+", question[m.end():].lower())[:4]
            if not {_singular(w) for w in before + after} & words:
                continue
            if side == "between":
                a, b = sorted((float(m.group(1)), float(m.group(2))))
                lo = a if lo is None else max(lo, a)
                hi = b if hi is None else min(hi, b)
            elif side == "min":
                v = fixed if fixed is not None else float(m.group(1))
                lo = v if lo is None else max(lo, v)
            else:
                v = float(m.group(1))
                hi = v if hi is None else min(hi, v)
    return lo, hi


def _column_words(spec: ColumnSpec, contract: "BlueprintContract", source: Any) -> set:
    words = set()
    for name in [spec.name] + _referenced_source_columns(spec.name, contract, source):
        words |= {_singular(t) for t in _tokens(name) if len(t) >= 3}
    return words - _GENERIC_TOKENS


@dataclass(frozen=True)
class BoundDecision:
    """Why one declared min/max bound is HARD or ADVISORY.

    ``source`` is the first of :data:`BOUND_SOURCES` whose derived range
    implies the bound, or ``"model"`` when none does. ``derived`` holds every
    range that could be derived, by source, as ``[lo, hi]`` with ``None`` for
    an unknown side.
    """

    column: str
    side: str  # "min" or "max"
    value: float
    source: str
    derived: Dict[str, Any] = field(default_factory=dict)
    #: False when the input frame is not the rows the result was computed on,
    #: so no range was derived from it (see :func:`code_reads_own_data`).
    data_checked: bool = True
    #: What the ``data`` source was, in words: "100-row validation sample"
    #: when the input frame is the validator's sample (see :func:`data_label`).
    #: "data gives [-27.5, 94.0]" was the sample's range; the full taxi file
    #: spans [-404.1, 1616.5] (rc18c).
    data_label: str = "data"

    @property
    def hard(self) -> bool:
        return self.source != "model"

    def describe(self) -> str:
        if self.hard:
            return f"derived from the {self.data_label if self.source == 'data' else self.source}"
        if not self.data_checked:
            prefix = "the validation sample is not the result's only input, so no range was taken from it; "
            if not self.derived:
                return prefix + "nothing in the column's domain or the question gives a range"
        else:
            prefix = ""
            if not self.derived:
                return f"nothing in the {self.data_label}, the column's domain or the question gives a range"
        shown = ", ".join(f"the {self.data_label} gives {v}" if k == "data" else f"{k} gives {v}"
                          for k, v in self.derived.items())
        return f"{prefix}no derivable range implies it: {shown}"

    def to_dict(self) -> Dict[str, Any]:
        return {"column": self.column, "side": self.side, "value": self.value,
                "source": self.source, "hard": self.hard, "derived": dict(self.derived),
                "data_checked": self.data_checked}


def bound_decision(spec: ColumnSpec, side: str, contract: "BlueprintContract", source: Any,
                   question: str = "", *, observed: bool = True, label: str = "data") -> BoundDecision:
    """Decide whether a declared ``min`` / ``max`` on *spec* is derivable.

    A bound is **derivable** -- and so enforced HARD -- when a range computed
    deterministically, with no model in the loop, *implies* it (a derived
    lower bound L backs a declared min b when L >= b; a derived upper bound H
    backs a declared max b when H <= b), so a correct result cannot violate
    it. The ranges come from:

    * ``data``: the observed [min, max] of the source column the result
      column's name refers to, under the aggregation the name states. A mean,
      median, min, max (and a subset's own column) lies within that range; a
      sum is >= 0 only when every value is, and <= 0 only when every value is.
      A mean of a column with negative values can be negative.
    * ``domain``: facts that hold for any data. A count or a spread
      (std/variance) is >= 0; a rate, ratio, share or proportion is in
      [0, 1]; a percentage in [0, 100]. A name that says change/growth/delta
      voids these.
    * ``question``: a bound the question states about that column ("only
      positive fares", "at least 5", "between 10 and 20";
      :func:`question_bounds`).

    Any other bound was invented by the blueprint model and is ADVISORY.

    ``observed=False`` says *source* is not the rows the result was computed
    on (the code read its own data): no ``data`` range is taken from it. A
    sample whose Quantity was all positive "derived" a min of 0 for a sum the
    full file, returns included, took to -366 (rc18b messy_xlsx).
    """
    value = float(spec.min if side == "min" else spec.max)
    derived: Dict[str, Any] = {}
    data = _data_interval(spec, contract, source) if observed else None
    if data is not None:
        derived["data"] = list(data)
    domain = _domain_interval(spec)
    if domain is not None:
        derived["domain"] = list(domain)
    q = question_bounds(question, _column_words(spec, contract, source))
    if q != (None, None):
        derived["question"] = list(q)

    for name in BOUND_SOURCES:
        rng = derived.get(name)
        if not rng:
            continue
        lo, hi = rng
        if side == "min" and lo is not None and lo >= value - 1e-9:
            return BoundDecision(spec.name, side, value, name, derived, observed, label)
        if side == "max" and hi is not None and hi <= value + 1e-9:
            return BoundDecision(spec.name, side, value, name, derived, observed, label)
    return BoundDecision(spec.name, side, value, "model", derived, observed, label)


def _resolve(result_columns: Any, wanted: str) -> Optional[str]:
    """Match a contract column name to an actual result column.

    Matching is normalised (case/space/underscore-insensitive) because the
    contract states intent, not spelling; holding generated code to an exact
    label would measure naming luck and produce exactly the kind of false
    rejection that trains people to ignore the validator.
    """
    cols = list(result_columns)
    norm = {_norm(c): c for c in cols}
    return norm.get(_norm(wanted))


@dataclass
class ContractReport:
    """What a contract check decided.

    ``violations`` are HARD: they send the code back for repair.
    ``advisories`` are postconditions the plan stated but could not back up --
    logged and recorded, never a rejection. ``row_count`` and ``bounds``
    record where an ``exactly n`` count and each min/max bound came from, so a
    trace shows why each was or was not enforced.
    """

    violations: List[Violation] = field(default_factory=list)
    advisories: List[Violation] = field(default_factory=list)
    row_count: Optional[RowCountDecision] = None
    #: One entry per declared min/max bound: where it came from (see
    #: :func:`bound_decision`).
    bounds: List[BoundDecision] = field(default_factory=list)


# --- whether the input frame is what the result was computed on ---------------
#
# The validator checks the result of a SAMPLE run: ``main(df)`` is called on the
# first rows of the dataset and ``source`` is that same sample. Generated code
# does not always use the frame it is given. The coder is told to read any other
# dataset inside ``main``, and a union of two tables or a join is written as
# ``pd.read_csv(<path>)`` of each file -- so the result comes from the full
# files while ``source`` is still a 100-row sample. Every comparison of the
# result with ``source`` then compares two different sets of rows: rc18b
# rejected a correct per-product summary of two full files (2,776 rows) because
# "the input holds 73 distinct group(s)" -- the 73 pairs of the sample.

#: Calls that load data: ``pd.read_csv``, ``read_csv_best_effort``,
#: ``pl.scan_parquet``, ``open``, ``pd.ExcelFile``.
_LOAD_PREFIXES = ("read_", "scan_")
_LOAD_NAMES = frozenset({"open", "ExcelFile", "loadtxt", "genfromtxt", "read_sql", "read_sql_query"})


def _is_main_guard(stmt: Any) -> bool:
    return isinstance(stmt, ast.If) and "__name__" in ast.unparse(stmt.test) and "__main__" in ast.unparse(stmt.test)


def code_reads_own_data(code: str) -> bool:
    """Whether *code*'s result can come from rows other than the frame its
    ``main`` is given: it loads data itself (outside the ``__main__`` block,
    which the sample run never executes), or ``main`` takes more than one
    frame. Unparseable code answers False (nothing is known)."""
    try:
        tree = ast.parse(str(code or ""))
    except (SyntaxError, ValueError):
        return False
    for stmt in tree.body:
        if _is_main_guard(stmt):
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "main":
            required = len(stmt.args.posonlyargs) + len(stmt.args.args) - len(stmt.args.defaults)
            if required > 1:
                return True
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if name in _LOAD_NAMES or name.startswith(_LOAD_PREFIXES):
                return True
    return False


def _key_text(value: Any) -> Optional[str]:
    """A group-key value compared across frames: ``85123`` read as int in one
    frame and as ``85123.0`` in another is one key; a missing value is none."""
    import math

    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)) or type(value).__module__ == "numpy":
        try:
            f = float(value)
        except (TypeError, ValueError):
            return str(value).strip()
        if math.isnan(f):
            return None
        return str(int(f)) if f.is_integer() else repr(f)
    text = str(value).strip()
    if text.lower() in ("nan", "nat", "none", "<na>", ""):
        return None
    try:
        return _key_text(float(text))
    except (TypeError, ValueError):
        return text


def _foreign_group_keys(result: Any, resolved: List[str], source: Any, src_resolved: List[str]) -> int:
    """How many of the result's group keys the source frame does not hold.

    A per-group aggregate of *source* can only hold keys *source* holds; any
    other key means the result was computed on other rows."""
    def keys(frame: Any, cols: List[str]) -> set:
        out = set()
        for row in frame[list(cols)].itertuples(index=False, name=None):
            texts = tuple(_key_text(v) for v in row)
            if None not in texts:
                out.add(texts)
        return out

    return len(keys(result, resolved) - keys(source, src_resolved))


def check_contract(contract: BlueprintContract, result: Any, source: Any,
                   *, question: str = "") -> List[Violation]:
    """Check an executed result frame against the blueprint's contract.

    ``result`` and ``source`` are pandas DataFrames. Returns the list of
    violated (HARD) postconditions -- empty means the code satisfied the plan
    it was written from. :func:`evaluate_contract` also returns the advisories.
    """
    return evaluate_contract(contract, result, source, question=question).violations


def evaluate_contract(contract: BlueprintContract, result: Any, source: Any,
                      *, question: str = "", source_is_input: bool = True,
                      source_is_sample: bool = False) -> ContractReport:
    """:func:`check_contract`, with advisories and the row-count and bound decisions.

    ``source_is_input=False`` says *source* is not the rows the result was
    computed on (:func:`code_reads_own_data`). Every check that compares the
    result with *source* -- its row count, its group count, a range or count
    derived from its values -- is then skipped and recorded as an advisory;
    checks of the result alone (columns, dtypes, duplicate group keys, a
    unique key, a scalar) still run.

    ``source_is_sample=True`` says *source* is the validator's sample, not the
    dataset: a range or count taken from it, and a value of the result
    computed on it, are then named as the sample's in every sentence.
    """
    import pandas as pd  # local: keeps this module importable without pandas

    report = ContractReport()
    violations = report.violations
    if result is None:
        violations.append(Violation("result", "the code did not return a DataFrame"))
        return report
    if isinstance(result, pd.Series):
        result = result.to_frame()
    if not isinstance(result, pd.DataFrame):
        violations.append(Violation("result", f"expected a DataFrame, got {type(result).__name__}"))
        return report

    n_res = len(result)
    n_src = len(source) if source is not None else None
    sampled = source_is_sample and source is not None
    label = f"{n_src}-row validation sample" if sampled else "data"
    # The result's own values are the sample run's only when it ran on the sample.
    has = "the result on the validation sample has" if sampled and source_is_input else "has"
    the_input = f"the {label}" if sampled else "the input"
    if source is not None and not source_is_input:
        # Recorded only where a comparison is skipped; a range or count not
        # taken from the sample says so in its own decision.
        if contract.row_relation.type in ("one_per_group", "same_as_input", "fewer_than_input"):
            report.advisories.append(Violation(
                "input_not_compared",
                f"data beyond the {n_src}-row validation sample reaches the result (the code loads "
                "data itself or takes more than one frame), so the result's row count was not "
                "compared with the sample's",
            ))
        n_src = None

    # --- declared columns must exist, with the declared kind of values ---
    for spec in contract.result_columns:
        if is_placeholder_column(spec.name):
            # The label is data-dependent; there is nothing to look up.
            continue
        actual = _resolve(result.columns, spec.name)
        if actual is None:
            violations.append(Violation(
                "missing_column",
                f"the plan promised a column {spec.name!r}; result has {list(result.columns)}",
            ))
            continue
        series = result[actual]
        if spec.dtype == "numeric" and not pd.api.types.is_numeric_dtype(series):
            violations.append(Violation(
                "wrong_dtype",
                f"{actual!r} was declared numeric but holds {series.dtype}",
            ))
            continue
        if spec.dtype == "datetime" and not pd.api.types.is_datetime64_any_dtype(series):
            violations.append(Violation(
                "wrong_dtype",
                f"{actual!r} was declared datetime but holds {series.dtype}",
            ))
            continue
        if spec.dtype == "boolean" and not (
            pd.api.types.is_bool_dtype(series)
            or set(series.dropna().unique()).issubset({0, 1, True, False})
        ):
            violations.append(Violation(
                "wrong_dtype",
                f"{actual!r} was declared boolean but holds {series.dtype}",
            ))
            continue
        if pd.api.types.is_numeric_dtype(series):
            s = series.dropna()
            if len(s):
                # Only a bound a deterministic fact backs is HARD; see
                # bound_decision for what "derivable" means.
                for side, declared in (("min", spec.min), ("max", spec.max)):
                    if declared is None:
                        continue
                    observed = float(s.min()) if side == "min" else float(s.max())
                    broken = (observed < declared - 1e-9) if side == "min" else (observed > declared + 1e-9)
                    decision = bound_decision(spec, side, contract, source, question,
                                              observed=source_is_input, label=label)
                    report.bounds.append(decision)
                    if not broken:
                        continue
                    if decision.hard:
                        violations.append(Violation(
                            "out_of_range",
                            f"{actual!r} declared {side} {declared} but {has} {observed} "
                            f"({decision.describe()})",
                        ))
                    else:
                        advisory = Violation(
                            "unverified_range",
                            f"{actual!r} declared {side} {declared}, a bound the plan did not "
                            f"derive ({decision.describe()}); {has} {observed}. Advisory only.")
                        report.advisories.append(advisory)
                        logger.warning("Blueprint contract advisory: %s", advisory.detail)

    # --- row-count relation: the deterministic form of the aggregation rule ---
    rel = contract.row_relation
    if rel.type == "one_per_group" and rel.group_by:
        resolved = [_resolve(result.columns, g) for g in rel.group_by]
        if any(r is None for r in resolved):
            missing = [g for g, r in zip(rel.group_by, resolved) if r is None]
            violations.append(Violation(
                "missing_group_key",
                f"the plan groups by {missing} but the result has no such column(s)",
            ))
        else:
            dupes = int(result.duplicated(subset=resolved).sum())
            if dupes:
                violations.append(Violation(
                    "duplicate_group_rows",
                    f"the plan promised one row per {list(rel.group_by)} "
                    f"but {dupes} duplicate key rows were returned",
                ))

            # How many groups the input actually contains. This is the honest
            # yardstick, and using it matters: validation runs against a sample,
            # and in a sample a high-cardinality key (a hospital, a customer, an
            # order id) is often near-unique. Comparing the result to the input
            # ROW COUNT then flags a perfectly correct aggregation as "you
            # returned the raw rows" -- a false rejection that trains the coder
            # to rewrite working code. Comparing to the DISTINCT GROUP COUNT is
            # correct at any sample size.
            #
            # The yardstick holds only when the result was computed on these
            # rows. When the code read its own data (source_is_input False) or
            # the result holds group keys the input does not, it was not, and
            # the comparison is skipped: "the input holds 73 distinct group(s)"
            # said of a 100-row sample, against a result built from two full
            # files, was false (rc18b). The duplicate-key check above is valid
            # at any size and still runs.
            src_groups: Optional[int] = None
            if source is not None and len(source) and source_is_input:
                src_resolved = [_resolve(source.columns, g) for g in rel.group_by]
                if all(r is not None for r in src_resolved):
                    foreign = _foreign_group_keys(result, resolved, source, src_resolved)
                    if foreign:
                        n_src = None          # the fallback below compares the same frames
                        report.advisories.append(Violation(
                            "input_not_compared",
                            f"the result holds {foreign} {list(rel.group_by)} key(s) that the "
                            f"{len(source)}-row validation sample does not (the code read other "
                            "rows or rewrote the keys), so its row count was not compared with "
                            "the sample's group count",
                        ))
                    else:
                        src_groups = int(source[list(src_resolved)].drop_duplicates().shape[0])

            if src_groups is not None:
                if n_res > src_groups:
                    violations.append(Violation(
                        "not_aggregated",
                        f"the plan promised one row per {list(rel.group_by)}; {the_input} "
                        f"holds {src_groups} distinct group(s) but {n_res} rows were "
                        "returned, so the result is not one row per group",
                    ))
            elif n_src is not None and n_res >= n_src:
                # The group columns are not visible in the input (they were
                # derived), so only the row count is left -- and it says little.
                # A result with MORE rows than the input cannot be its raw
                # rows: one row per token split out of 100 recipe names is 174
                # rows, and was refused as "the raw rows were returned"
                # (rc18c). Exactly the input's row count is the one count that
                # shows no aggregation; anything above it cannot be judged.
                if n_res == n_src:
                    carries = all(_resolve(result.columns, c) is not None for c in source.columns)
                    violations.append(Violation(
                        "not_aggregated",
                        f"the plan promised one row per {list(rel.group_by)}, a column {the_input} "
                        f"does not hold, but the result has exactly as many rows as {the_input} "
                        f"({n_res})" + (f" and all {len(source.columns)} of its columns" if carries else "")
                        + ": no aggregation shows in the row count",
                    ))
                else:
                    report.advisories.append(Violation(
                        "aggregation_not_judged",
                        f"the plan groups by {list(rel.group_by)}, which {the_input} does not hold, "
                        f"and the result has {n_res} rows from {n_src} input rows; from the row "
                        "count alone it cannot be told whether the rows were aggregated",
                    ))
    elif rel.type == "same_as_input" and n_src is not None and n_res != n_src:
        violations.append(Violation(
            "row_count_changed",
            f"the plan promised one result row per input row; got {n_res} vs {n_src}",
        ))
    elif rel.type == "fewer_than_input" and n_src is not None and n_res > n_src:
        # Strictly greater, not >=: on a small validation sample a correct
        # reduction can legitimately return exactly as many rows as it was given.
        violations.append(Violation(
            "not_reduced",
            f"the plan promised no more rows than the input; got {n_res} vs {n_src}",
        ))
    elif rel.type == "at_most" and rel.n is not None and n_res > rel.n:
        violations.append(Violation(
            "too_many_rows", f"the plan promised at most {rel.n} rows; got {n_res}"))
    elif rel.type == "exactly" and rel.n is not None:
        # Only a count the plan can back up is a HARD postcondition; see
        # exact_count_decision for what "derivable" means.
        decision = exact_count_decision(contract, source, question, observed=source_is_input,
                                        label=label)
        report.row_count = decision
        if n_res != rel.n:
            if decision.hard:
                violations.append(Violation(
                    "wrong_row_count",
                    f"the plan promised exactly {rel.n} rows ({decision.describe()}); got {n_res}"))
            else:
                advisory = Violation(
                    "unverified_row_count",
                    f"the plan stated exactly {rel.n} rows, a count it did not derive "
                    f"({decision.describe()}); got {n_res}. Advisory only.")
                report.advisories.append(advisory)
                logger.warning("Blueprint contract advisory: %s", advisory.detail)

    if contract.result_kind == "scalar" and n_res != 1:
        violations.append(Violation(
            "not_scalar",
            f"the plan described a single summary value but returned {n_res} rows",
        ))

    # --- key uniqueness ---
    if contract.unique_key:
        resolved = [_resolve(result.columns, k) for k in contract.unique_key]
        if all(r is not None for r in resolved):
            dupes = int(result.duplicated(subset=resolved).sum())
            if dupes:
                violations.append(Violation(
                    "key_not_unique",
                    f"{list(contract.unique_key)} was declared unique but {dupes} rows repeat it",
                ))

    # --- null policy ---
    for col in contract.not_null:
        actual = _resolve(result.columns, col)
        if actual is not None:
            n_null = int(result[actual].isna().sum())
            if n_null:
                violations.append(Violation(
                    "unexpected_nulls",
                    f"{actual!r} was declared non-null but has {n_null} missing values",
                ))

    return report


def format_violations(violations: List[Violation], contract: BlueprintContract) -> str:
    """Repair feedback for the coder.

    It names the postcondition the plan itself stated, so the coder is told what
    it promised rather than handed another rule to memorise.
    """
    lines = [
        "The code ran, but its result does not satisfy the acceptance criteria "
        "stated in your own blueprint for this task:",
        "",
    ]
    for v in violations:
        lines.append(f"  - {v.detail}")
    lines += [
        "",
        "Blueprint contract that was not met:",
        json.dumps(contract.to_dict(), indent=2),
        "",
        "Rewrite the code so the returned DataFrame satisfies these "
        "postconditions. Do not change the contract to fit the code.",
    ]
    return "\n".join(lines)
