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


def check_contract(contract: BlueprintContract, result: Any, source: Any) -> List[Violation]:
    """Check an executed result frame against the blueprint's contract.

    ``result`` and ``source`` are pandas DataFrames. Returns the list of
    violated postconditions -- empty means the code satisfied the plan it was
    written from.
    """
    import pandas as pd  # local: keeps this module importable without pandas

    violations: List[Violation] = []
    if result is None:
        return [Violation("result", "the code did not return a DataFrame")]
    if isinstance(result, pd.Series):
        result = result.to_frame()
    if not isinstance(result, pd.DataFrame):
        return [Violation("result", f"expected a DataFrame, got {type(result).__name__}")]

    n_res = len(result)
    n_src = len(source) if source is not None else None

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
                if spec.min is not None and float(s.min()) < spec.min - 1e-9:
                    violations.append(Violation(
                        "out_of_range",
                        f"{actual!r} declared min {spec.min} but has {float(s.min())}",
                    ))
                if spec.max is not None and float(s.max()) > spec.max + 1e-9:
                    violations.append(Violation(
                        "out_of_range",
                        f"{actual!r} declared max {spec.max} but has {float(s.max())}",
                    ))

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
            src_groups: Optional[int] = None
            if source is not None and len(source):
                src_resolved = [_resolve(source.columns, g) for g in rel.group_by]
                if all(r is not None for r in src_resolved):
                    src_groups = int(source[list(src_resolved)].drop_duplicates().shape[0])

            if src_groups is not None:
                if n_res > src_groups:
                    violations.append(Violation(
                        "not_aggregated",
                        f"the plan promised one row per {list(rel.group_by)}; the input "
                        f"holds {src_groups} distinct group(s) but {n_res} rows were "
                        "returned, so the result is not one row per group",
                    ))
            elif n_src is not None and n_res >= n_src:
                # The group columns are not visible in the input (they were
                # derived), so fall back to the row-count comparison.
                violations.append(Violation(
                    "not_aggregated",
                    f"the plan promised one row per {list(rel.group_by)} but the result "
                    f"has {n_res} rows against {n_src} input rows -- the raw rows were "
                    "returned instead of the aggregate",
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
    elif rel.type == "exactly" and rel.n is not None and n_res != rel.n:
        violations.append(Violation(
            "wrong_row_count", f"the plan promised exactly {rel.n} rows; got {n_res}"))

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

    return violations


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
