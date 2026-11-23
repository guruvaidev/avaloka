"""The recipe contract: closed vocabularies and the five-method protocol.

Every value a recipe exchanges with the rest of the system is drawn from a
closed list defined here or in ``avaloka.outcomes``. That is what makes the
intent parser a *recognizer* (an out-of-vocabulary value is a parse failure,
never a coerced default), lets ``recipe_id``, ``verdict`` and ``reason`` go to
telemetry, and lets a golden test assert a verdict exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, Union, runtime_checkable

from avaloka.outcomes import semantics as _sem
from avaloka.outcomes import spec as _spec

# ── closed vocabularies ────────────────────────────────────────────────────

#: Column roles a parameter may bind to (``avaloka.outcomes.semantics.ROLES``).
ROLES: tuple[str, ...] = tuple(_sem.ROLES)
#: Time grains (``avaloka.outcomes.semantics.GRAINS``).
GRAINS: tuple[str, ...] = tuple(_sem.GRAINS)
#: What kind of evidence a recipe produces; the Outcome spec's list, so the
#: report's wording rules (associational never reads as an effect) apply.
EVIDENCE_KINDS: tuple[str, ...] = tuple(_spec._EVIDENCE_KINDS)
#: Caveat kinds a quality gate may raise; the Outcome spec's list, so a gate's
#: outcome is already a renderable caveat.
CAVEAT_KINDS: tuple[str, ...] = tuple(_spec._CAVEAT_KINDS)

#: Parameter names the intent parser may fill. Nothing else is a parameter.
PARAM_NAMES: tuple[str, ...] = (
    "measure",      # one measure column
    "measures",     # two or more measure columns (correlate)
    "dimension",    # a grouping column
    "time",         # the time column
    "grain",        # one of GRAINS
    "window",       # a period label the recipe understands ("last_month", "ytd", "all")
    "arm",          # the assignment column of an experiment
    "outcome",      # the outcome column of an experiment or regression
    "id",           # the entity id column (cohorts)
)

#: Preconditions answered from the profile without reading rows.
PRECONDITIONS: tuple[str, ...] = (
    "has_time", "has_measure", "has_two_measures", "has_dimension", "has_id", "has_arm",
    "min_rows_30", "min_periods_4",
)

#: Quality gates that read rows. Each gate's outcome is a :class:`Gate`.
GATES: tuple[str, ...] = (
    "duplicates", "missing_measure", "date_parse", "constant", "fanout_join",
    "small_groups", "high_cardinality", "sample_ratio", "leakage", "period_gaps",
    "min_rows", "incomplete_period", "simpson", "collinearity", "heteroscedasticity", "flag_share",
    "min_cohorts", "power",
)

#: Why a run did not end in a report. Closed, so it can be stored and counted.
REASONS: tuple[str, ...] = (
    "insufficient_rows", "insufficient_periods", "no_time_column", "no_measure",
    "ambiguous_measure", "ambiguous_dimension", "causal_question", "leakage_suspected",
    "fanout_join", "sample_ratio_mismatch", "constant_column", "engine_timeout",
    "unsupported_question", "missing_dependency", "internal",
    "too_large",            # over the sampled lane's ceiling, or the engine ran out of memory
)

#: What the intent parser noticed about the question itself. A flag changes
#: how the run proceeds, never what it computes: ``causal_question`` pins the
#: associational wording (design §3.2 rule 2, §9.1), ``ambiguous`` sends the
#: question to the model for a tie-break, ``multi_question`` sends it out of
#: the recipe lane (§3.2 rule 3).
INTENT_FLAGS: tuple[str, ...] = ("causal_question", "ambiguous", "multi_question")

_VERSION = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class Verdict(str, Enum):
    """How a recipe run ended. Written by code, never by a model."""
    PASS = "pass"             # a report with findings
    QUALIFIED = "qualified"   # a report, with named caveats that change the reading
    ASK = "ask"               # one question back to the user; nothing computed
    REFUSE = "refuse"         # the recipe will not run on this data; reason in REASONS


# ── specs ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParamSpec:
    """One parameter of a recipe, bound to a column of a given role or a grain."""
    name: str
    role: str                       # a ROLES member, or "grain" / "window" for non-column params
    required: bool = True
    many: bool = False              # the parameter takes two or more columns
    choices: tuple[str, ...] = ()   # for grain/window: the allowed values

    def __post_init__(self) -> None:
        if self.name not in PARAM_NAMES:
            raise ValueError(f"unknown parameter {self.name!r}; allowed: {PARAM_NAMES}")
        if self.role not in ROLES + ("grain", "window"):
            raise ValueError(f"unknown role {self.role!r} for {self.name!r}")


@dataclass(frozen=True)
class RecipeSpec:
    """The identity and contract of a recipe. ``version`` is ``MAJOR.MINOR``:
    MAJOR changes when a reported number could change for the same input,
    MINOR when evidence or slots are added (design §2.3)."""
    id: str
    version: str
    title: str
    params: tuple[ParamSpec, ...]
    preconditions: tuple[str, ...]
    quality_gates: tuple[str, ...]
    evidence_kind: str
    explain_slots: tuple[str, ...] = ("headline",)

    def __post_init__(self) -> None:
        from .registry import RECIPE_IDS  # local: registry imports this module
        if self.id not in RECIPE_IDS:
            raise ValueError(f"unknown recipe id {self.id!r}; allowed: {RECIPE_IDS}")
        if not _VERSION.match(self.version):
            raise ValueError(f"version must be MAJOR.MINOR, got {self.version!r}")
        bad = [p for p in self.preconditions if p not in PRECONDITIONS]
        if bad:
            raise ValueError(f"unknown preconditions {bad}")
        bad = [g for g in self.quality_gates if g not in GATES]
        if bad:
            raise ValueError(f"unknown gates {bad}")
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence kind {self.evidence_kind!r}")
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names in {self.id}: {names}")

    @property
    def major(self) -> int:
        return int(self.version.split(".")[0])


# ── what flows between the steps ───────────────────────────────────────────

ParamValue = Union[str, tuple[str, ...]]


@dataclass(frozen=True)
class Intent:
    """The parsed question. ``params`` holds only PARAM_NAMES keys; a value is
    a column name (or a tuple of them for ``many`` params), a grain or a window.
    ``source`` says who produced it, so the benchmark can score each path."""
    question: str
    recipe: str | None
    params: Mapping[str, ParamValue] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "heuristic"       # "heuristic" | "model" | "fixture"
    flags: tuple[str, ...] = ()     # INTENT_FLAGS members

    def __post_init__(self) -> None:
        bad = [k for k in self.params if k not in PARAM_NAMES]
        if bad:
            raise ValueError(f"unknown intent parameters {bad}")
        if self.source not in ("heuristic", "model", "fixture"):
            raise ValueError(f"unknown intent source {self.source!r}")
        bad = [f for f in self.flags if f not in INTENT_FLAGS]
        if bad:
            raise ValueError(f"unknown intent flags {bad}")


@dataclass(frozen=True)
class Bound:
    """A recipe with every required parameter bound to a column, grain or window."""
    recipe_id: str
    version: str
    params: Mapping[str, ParamValue]


@dataclass(frozen=True)
class Ask:
    """One question back to the user. ``text`` is template text written by code."""
    parameter: str
    text: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class Refusal:
    """The recipe will not run. ``reason`` is a REASONS member; ``text`` is template text."""
    reason: str
    text: str

    def __post_init__(self) -> None:
        if self.reason not in REASONS:
            raise ValueError(f"unknown refusal reason {self.reason!r}")


@dataclass(frozen=True)
class Gate:
    """One quality gate's outcome. A qualified or failed gate carries the figure
    that decided it, so the report can show it as evidence, never as prose."""
    name: str
    status: str                       # "passed" | "qualified" | "failed"
    caveat_kind: str | None = None    # a CAVEAT_KINDS member when qualified
    reason: str | None = None         # a REASONS member when failed
    value: float | None = None        # the deciding figure
    unit: str | None = None           # an EVIDENCE_UNITS member

    def __post_init__(self) -> None:
        if self.name not in GATES:
            raise ValueError(f"unknown gate {self.name!r}")
        if self.status not in ("passed", "qualified", "failed"):
            raise ValueError(f"unknown gate status {self.status!r}")
        if self.status == "qualified" and self.caveat_kind not in CAVEAT_KINDS:
            raise ValueError(f"qualified gate {self.name} needs a caveat kind")
        if self.status == "failed" and self.reason not in REASONS:
            raise ValueError(f"failed gate {self.name} needs a reason")
        if self.unit is not None and self.unit not in _spec.EVIDENCE_UNITS:
            raise ValueError(f"unknown unit {self.unit!r}")


@dataclass(frozen=True)
class GateReport:
    gates: tuple[Gate, ...]

    @property
    def verdict(self) -> Verdict:
        if any(g.status == "failed" for g in self.gates):
            return Verdict.REFUSE
        if any(g.status == "qualified" for g in self.gates):
            return Verdict.QUALIFIED
        return Verdict.PASS

    @property
    def reason(self) -> str | None:
        for g in self.gates:
            if g.status == "failed":
                return g.reason
        return None

    @property
    def caveat_kinds(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(g.caveat_kind for g in self.gates if g.status == "qualified"))  # type: ignore[arg-type]


@dataclass(frozen=True)
class Evidence:
    """What ``compute`` produced: a JSON tree of leaves (``evidence`` in the
    Outcome spec; a leaf is a number or ``{"value", "unit"}``) and the data
    tables charts and notebooks are built from (``{"columns": [...], "rows": [[...]]}``)."""
    tree: Mapping[str, Any]
    tables: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetReport:
    """What a run spent against its :class:`~avaloka.recipes.budget.Budget`:
    the source's size, whether it was sampled, the engine caps in force, the
    elapsed wall clock and whether the clock ran out. The API stores it on
    the run row."""
    total_rows: int
    rows_analyzed: int
    total_bytes: int
    sampled: bool
    elapsed_s: float
    memory_limit: str
    threads: int
    timed_out: bool = False


@dataclass(frozen=True)
class RunResult:
    """How one run ended. Exactly one of ``report``/``ask``/``refusal`` is set
    for PASS-or-QUALIFIED / ASK / REFUSE. ``budget_report`` is set once the
    engine has run (never for an ``ask`` or a refusal at bind time)."""
    recipe_id: str
    version: str
    verdict: Verdict
    reason: str | None = None
    report: Mapping[str, Any] | None = None
    notebook: Mapping[str, Any] | None = None
    gates: GateReport | None = None
    evidence: Evidence | None = None
    ask: Ask | None = None
    refusal: Refusal | None = None
    budget_report: BudgetReport | None = None


# ── the protocol ───────────────────────────────────────────────────────────

@runtime_checkable
class Recipe(Protocol):
    """The five methods every recipe implements. ``con`` is a DuckDB connection
    with the dataset registered as the view ``t`` (``avaloka.recipes.engine``)."""
    spec: RecipeSpec

    def bind(self, intent: Intent, sem: Mapping[str, _sem.ColumnSemantics],
             profile: Mapping[str, Any]) -> Bound | Ask | Refusal: ...

    def check(self, con: Any, bound: Bound) -> GateReport: ...

    def compute(self, con: Any, bound: Bound) -> Evidence: ...

    def report(self, evidence: Evidence, bound: Bound, gates: GateReport) -> dict[str, Any]: ...

    def notebook(self, bound: Bound) -> dict[str, Any]: ...
