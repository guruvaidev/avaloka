"""Benchmark data model: tasks, expectations, and scorecards.

A :class:`BenchmarkTask` binds a dataset (synthetic or Kaggle) to a mission and a
set of :class:`Expectations` that encode *ground truth* — the lane the router
should pick, the columns that are planted leaks, the roles the profiler should
infer, the floor a model must clear over a naive baseline. Scoring turns each
expectation into a :class:`CheckResult`; a :class:`TaskResult` aggregates the
checks for one (task, source) run; a :class:`Scorecard` aggregates everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class DatasetSpec:
    """How to obtain a benchmark dataset as a local file (source-agnostic)."""

    name: str
    kind: str                                   # "synthetic" | "kaggle"
    target: Optional[str] = None
    # synthetic: a builder that writes a CSV to `dest` and returns the path.
    builder: Optional[Callable[[Path], Path]] = None
    # kaggle: dataset slug (owner/name) and optional preferred CSV inside it.
    kaggle_slug: Optional[str] = None
    preferred_file: Optional[str] = None
    notes: str = ""


@dataclass
class Expectations:
    """Ground-truth assertions a correct mission/coordination must satisfy."""

    # data-engineering
    expected_lane: Optional[str] = None                     # "online"|"sampled_online"|"batch"
    expected_roles: dict[str, str] = field(default_factory=dict)   # column -> role
    expect_quality_flags: dict[str, list[str]] = field(default_factory=dict)
    #   e.g. {"high_missing_columns": ["income"], "constant_columns": ["flag"]}
    min_quality: Optional[int] = None
    max_quality: Optional[int] = None
    # exact-convergence (batch / coordinator): columns whose stats must match single-pass
    exact_columns: list[str] = field(default_factory=list)

    # data-science
    must_detect_leakage: list[str] = field(default_factory=list)   # columns that MUST be flagged
    expected_verdict: Optional[str] = None                  # "pass"|"warn"|"fail"
    min_score: Optional[float] = None                       # absolute floor on the primary metric
    min_gain_over_baseline: Optional[float] = None          # margin over a naive baseline
    expected_max_level: Optional[int] = None                # validator deployment-level cap
    expected_fireflies: list[str] = field(default_factory=list)   # must appear in the ledger
    forbidden_fireflies: list[str] = field(default_factory=list)  # must NOT appear

    # planning: the split a correct plan must choose, and -- separately -- the
    # split the model must actually have been fitted with. Those were allowed to
    # disagree, so a temporal plan was reported over a random shuffle.
    expected_split: Optional[str] = None                    # e.g. "time_based_holdout"
    expected_task: Optional[str] = None                     # e.g. "regression"

    # the baseline. A score with nothing to compare it against is not evidence,
    # and the suite scored one it computed itself rather than asking Avaloka for
    # hers -- so a missing baseline was invisible here.
    require_baseline_reported: bool = False
    expect_beats_baseline: Optional[bool] = None            # True, or False for no-signal data

    # features that must survive into the model. Continuous measurements were
    # dropped as "identifier-like" and nothing noticed.
    must_use_features: list[str] = field(default_factory=list)

    # economics sanity (applied to every mission by default in scoring)
    require_positive_multiplier: bool = True


@dataclass
class BenchmarkTask:
    """A single benchmark unit: a dataset + a mission + expectations."""

    name: str
    family: str                                 # "data_engineering" | "data_science"
    kind: str                                   # "analyze" | "train" | "batch"
    goal: str
    dataset: DatasetSpec
    expectations: Expectations = field(default_factory=Expectations)
    target: Optional[str] = None
    metric: Optional[str] = None
    deployable: bool = False
    budget: Optional[float] = None
    # which source connectors to exercise this task through.
    source_schemes: tuple[str, ...] = ("file",)
    # multi-Avaloka coordination: if set, also run the fleet and check convergence.
    coordinate_workers: Optional[int] = None


@dataclass
class CheckResult:
    """One graded assertion."""

    name: str
    passed: bool
    detail: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        # Scorers may hand us numpy booleans (from np.isclose etc.); normalise to
        # native types so the scorecard is always JSON-serialisable.
        self.passed = bool(self.passed)
        self.weight = float(self.weight)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TaskResult:
    """All checks for one (task, source-scheme) run, plus provenance."""

    task: str
    family: str
    kind: str
    source_scheme: str
    checks: list[CheckResult]
    error: Optional[str] = None
    artifacts_dir: Optional[str] = None
    duration_s: float = 0.0

    @property
    def score(self) -> float:
        total = sum(c.weight for c in self.checks) or 1.0
        got = sum(c.weight for c in self.checks if c.passed)
        return got / total

    @property
    def passed(self) -> bool:
        return self.error is None and all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task, "family": self.family, "kind": self.kind,
            "source_scheme": self.source_scheme, "score": round(self.score, 4),
            "passed": self.passed, "error": self.error,
            "artifacts_dir": self.artifacts_dir, "duration_s": round(self.duration_s, 3),
            "checks": [c.as_dict() for c in self.checks],
        }


@dataclass
class Scorecard:
    """The full benchmark result."""

    results: list[TaskResult] = field(default_factory=list)

    def add(self, result: TaskResult) -> None:
        self.results.append(result)

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def by_family(self) -> dict[str, float]:
        fams: dict[str, list[float]] = {}
        for r in self.results:
            fams.setdefault(r.family, []).append(r.score)
        return {f: round(sum(v) / len(v), 4) for f, v in fams.items()}

    def failures(self) -> list[TaskResult]:
        return [r for r in self.results if not r.passed]

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 4),
            "n_runs": len(self.results),
            "n_passed": sum(1 for r in self.results if r.passed),
            "by_family": self.by_family(),
            "results": [r.as_dict() for r in self.results],
        }
