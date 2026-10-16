"""Benchmark data model: tasks, expectations, and scorecards.

A :class:`BenchmarkTask` binds a dataset (synthetic or Kaggle) to a mission and a
set of :class:`Expectations` that encode *ground truth* — the lane the router
should pick, the columns that are planted leaks, the roles the profiler should
infer, the floor a model must clear over a naive baseline. Scoring turns each
expectation into a :class:`CheckResult`; a :class:`TaskResult` aggregates the
checks for one (task, source) run; a :class:`Scorecard` aggregates everything.
"""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

#: Bumped whenever the scorecard's shape changes, so a consumer reading an old
#: file knows it is old rather than silently misreading a renamed field.
SCORECARD_SCHEMA_VERSION = 1


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> Optional[str]:
    """The commit this scorecard describes, or None when it cannot be known.

    Returns None rather than a placeholder: "unknown" in this field would be
    indistinguishable from a real value to anything parsing the file, and the
    whole point of recording it is that a number without a commit is not
    evidence. Honours CI's own variable first (a Bitbucket/GitHub runner has it
    without a .git directory), then asks git.
    """
    for var in ("BITBUCKET_COMMIT", "GITHUB_SHA", "CI_COMMIT_SHA"):
        value = os.environ.get(var)
        if value:
            return value.strip()
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@dataclass
class GroundTruth:
    """What the file *actually* contains, declared by the builder that wrote it.

    A spreadsheet is a canvas, so "did it read the file" is not a yes/no. The
    reader can open a workbook, return a frame, and be wrong about every column
    name, the header row, the row count and the types -- which is what produced
    insight text about ``Unnamed: 13``. Grading that needs the truth stated
    independently of the reader, so each builder declares it here and the
    comprehension scorers compare the reader's answer against it.

    ``header_row`` is 1-based, as a human counts rows in a spreadsheet, and is
    ``None`` for formats that have no such notion (json, parquet, avro).
    """

    columns: list[str] = field(default_factory=list)
    n_rows: Optional[int] = None
    header_row: Optional[int] = None
    #: The sheet a correct read must land on, for a workbook with more than one.
    #:
    #: Which sheet was chosen was not expressible here, so no check could grade
    #: it -- and sheet choice is a defect class of its own: both readers pick a
    #: sheet by a size heuristic (the CLI by non-empty cell count, the upload
    #: path by row count), so a longer reference or appendix sheet wins over the
    #: statement the question is about. That failure is silent and the resulting
    #: frame is clean, with no placeholder name to give it away.
    sheet_name: Optional[str] = None
    numeric_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    #: A figure planted in the data that a "meaningful answer" must reproduce,
    #: e.g. {"label": "Total revenue", "period": "2023", "value": 1234000.0}.
    #:
    #: A financial statement's answer is a *row-label* lookup rather than a
    #: column one -- "net income for 2023" names a row, not a field -- so the
    #: builders that have row labels carry them here under "row_labels" (or
    #: "line_items" for a statement). Keeping them in one place means the
    #: expected labels cannot drift from the expected figure they index.
    planted_answer: dict[str, Any] = field(default_factory=dict)
    #: Free-text note on the shape this file carries, shown in the HTML report.
    shape: str = ""


@dataclass
class DatasetSpec:
    """How to obtain a benchmark dataset as a local file (source-agnostic)."""

    name: str
    kind: str                                   # "synthetic" | "kaggle" | "edgar"
    target: Optional[str] = None
    # synthetic: a builder that writes the dataset to `dest` and returns the
    # path. It was CSV-only; `suffix` is what lets a builder write a workbook,
    # a JSON document or an Avro container instead, so the suite can carry the
    # eight declared formats it had no shape for.
    builder: Optional[Callable[[Path], Path]] = None
    # A spreadsheet builder needs openpyxl, which is in the `benchmark` extra
    # and not a runtime dependency; without it the task FAILS (the dataset
    # cannot be built), names the extra, and is never reported as a skip.
    suffix: str = ".csv"
    # kaggle: dataset slug (owner/name) and optional preferred CSV inside it.
    kaggle_slug: Optional[str] = None
    preferred_file: Optional[str] = None
    # Which of the API's declared upload extensions this dataset exercises.
    # The coverage check reads this, so a format cannot be declared supported
    # while no benchmark task carries a file of that format.
    declared_format: Optional[str] = None
    # What the file really contains, for the comprehension scorers.
    ground_truth: Optional[GroundTruth] = None
    # edgar: an SEC filing, fetched only when --edgar is passed.
    edgar_cik: Optional[str] = None
    edgar_accession: Optional[str] = None
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

    # goal-analysis ground truth (analyze missions only; goal_analysis.json).
    # A grouped answer can keep its total and still be wrong: categories lost at
    # ingestion merge into one bucket holding exactly their combined sum. So the
    # groups are compared one by one, and anything not planted is refused.
    expected_groups: dict[str, float] = field(default_factory=dict)   # label -> value
    forbid_unexpected_groups: bool = False
    expected_goal_status: Optional[str] = None              # "answered" | "limited"

    # profile fidelity (data_quality.json): exact label counts and exact gaps.
    expected_category_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    expected_missing: dict[str, int] = field(default_factory=dict)   # column -> missing cells

    # schema as the mission saw it (data_quality.json): the exact column names
    # in order, no pandas placeholder name ("Unnamed: 0"), the exact row count.
    expected_columns: list[str] = field(default_factory=list)
    forbid_unnamed_columns: bool = False
    expected_n_rows: Optional[int] = None

    # names of checks that are guards rather than assertions (see
    # CHECK_CATEGORIES). A name that matches no check the run produced is an
    # error: a guard that never ran reads exactly like one that holds.
    guard_checks: tuple[str, ...] = ()

    # columns of expected_roles whose role check is a guard rather than an
    # assertion (see CHECK_CATEGORIES): it holds today and is there to fail a
    # known wrong fix.
    guard_roles: tuple[str, ...] = ()

    # ── comprehension ────────────────────────────────────────────────────
    #
    # Whether the system understood the file it was given. Every field above
    # grades what Avaloka *did* with a frame; none of them grades whether the
    # frame is the table that was in the file. A workbook read off by two rows
    # produces a frame of placeholder names, and every check above can still
    # pass on it -- which is how insight text naming 'Unnamed: 13' was scored
    # 1.00. These are graded against DatasetSpec.ground_truth.

    #: No resolved column may match ``^Unnamed:\s*\d+$`` or ``^column_\d+$``.
    no_placeholder_column_names: bool = False
    #: The header row the reader reports must equal this (1-based).
    header_row_identified: Optional[int] = None
    #: The sheet the reader resolved must equal this. Left None for a
    #: single-sheet file, where there is no choice to get wrong.
    sheet_identified: Optional[str] = None
    #: Every one of these names must be present in the resolved frame, exactly.
    expected_columns_resolved: list[str] = field(default_factory=list)
    #: column -> "numeric" | "datetime" | "text".
    expected_dtypes: dict[str, str] = field(default_factory=dict)
    #: Exact row count of the table inside the file.
    expected_row_count: Optional[int] = None
    #: Every column name quoted in the generated insight/report text must exist
    #: in the frame. This is the user-visible auto-insights defect, and it is
    #: measured nowhere else.
    #:
    #: Note what this does *not* say. The reference set is the frame the mission
    #: actually built, so on a misread file the check asks only whether the
    #: prose is consistent with the misread frame -- which it usually is, the
    #: prose being generated from it. The placeholder branch is what gives this
    #: variant teeth. Use ``narration_columns_from_ground_truth`` for the
    #: stronger question.
    narration_names_only_real_columns: bool = False
    #: Every column name in the generated text must be a column the **file**
    #: has, per ``expected_columns_resolved``, plus
    #: ``narration_allowed_extra_columns``.
    #:
    #: This is the strong form, and it is a different question from the field
    #: above: not "is the prose consistent with what the reader built" but "is
    #: the prose about the table that was in the file". A workbook whose
    #: two-row header collapses to ``Sales`` and ``Sales.1`` produces a report
    #: naming ``'Sales'`` -- true of the frame, false of the file, and the
    #: frame-relative check cannot see it. Graded only when
    #: ``expected_columns_resolved`` is non-empty, since without ground truth
    #: there is nothing to be right about.
    narration_columns_from_ground_truth: bool = False
    #: Names the generated text may use that are not in the file: columns the
    #: mission legitimately engineers (``Revenue_log``, one-hot names) and any
    #: derived label a correct report would mention.
    #:
    #: Measured on this suite it is empty, and that is a finding rather than an
    #: oversight: no mission here writes an engineered column into
    #: ``transformed_dataset.parquet``, on the analyze path or the train path --
    #: encoding happens inside the sklearn pipeline, not in the artifact. The
    #: field exists so that a transform which *does* add a column can declare
    #: it rather than force the check to be loosened for everyone.
    narration_allowed_extra_columns: list[str] = field(default_factory=list)
    #: The reader must **refuse** this dataset, and its message must name each
    #: of these strings.
    #:
    #: For a document whose shape has no single correct reading -- a record
    #: holding a list of objects, a file with two top-level arrays -- refusing
    #: is the right answer and reading it is the defect. Graded at load time,
    #: because that is where the decision is made. An expectation of "refuse"
    #: that a reader satisfies by crashing with an unrelated error would be no
    #: better than the guess, so the message has to name the offending key.
    expected_refusal: list[str] = field(default_factory=list)

    #: A planted figure the mission's answer must reproduce, as
    #: {"label": ..., "value": ..., "tolerance": ...}.
    expected_answer: dict[str, Any] = field(default_factory=dict)

    # economics sanity (applied to every mission by default in scoring)
    require_positive_multiplier: bool = True

    def __post_init__(self) -> None:
        # A guard on a column nothing checks would never run, and a check that
        # never runs reads exactly like one that holds.
        unchecked = [c for c in self.guard_roles if c not in self.expected_roles]
        if unchecked:
            raise ValueError(f"guard_roles {unchecked} have no entry in expected_roles")


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


#: What a check claims. The two must never share a pass count:
#:
#: * ``assertion`` -- a property holds now. Red on a tree where the defect is
#:   live. The default, so a scorer that says nothing is making an assertion.
#: * ``guard`` -- a known wrong fix has not been applied. Green the day it is
#:   written, so a green guard is not evidence that anything was fixed.
#:
#: Both fail the run: a broken guard exits 1 exactly as a failed assertion does.
CHECK_CATEGORIES = ("assertion", "guard")


def _category_counts(checks: list["CheckResult"]) -> dict[str, dict[str, int]]:
    """``{category: {"passed": n, "total": n}}``, every category always present."""
    counts = {cat: {"passed": 0, "total": 0} for cat in CHECK_CATEGORIES}
    for check in checks:
        counts[check.category]["total"] += 1
        counts[check.category]["passed"] += int(check.passed)
    return counts


@dataclass
class CheckResult:
    """One graded check: an assertion, or a guard (see CHECK_CATEGORIES)."""

    name: str
    passed: bool
    detail: str = ""
    weight: float = 1.0
    category: str = "assertion"

    def __post_init__(self) -> None:
        # Scorers may hand us numpy booleans (from np.isclose etc.); normalise to
        # native types so the scorecard is always JSON-serialisable.
        self.passed = bool(self.passed)
        self.weight = float(self.weight)
        # A closed set: a typo must not invent a third category that no count
        # includes.
        if self.category not in CHECK_CATEGORIES:
            raise ValueError(
                f"check {self.name!r}: category {self.category!r} is not one of "
                f"{CHECK_CATEGORIES}")

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

    def by_category(self) -> dict[str, dict[str, int]]:
        return _category_counts(self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task, "family": self.family, "kind": self.kind,
            "source_scheme": self.source_scheme, "score": round(self.score, 4),
            "passed": self.passed, "error": self.error,
            "artifacts_dir": self.artifacts_dir, "duration_s": round(self.duration_s, 3),
            "by_category": self.by_category(),
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

    def by_category(self) -> dict[str, dict[str, int]]:
        return _category_counts([c for r in self.results for c in r.checks])

    def as_dict(self) -> dict[str, Any]:
        return {
            # Provenance first. Every number this project has published so far
            # was a bare float in a log with nothing tying it to the code that
            # produced it, which is a named finding in the paper audit: the
            # scorecard could not be reproduced because nobody could say what
            # it was a scorecard *of*.
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "commit": git_commit(),
            "generated_at": _utcnow_iso(),
            "overall_score": round(self.overall_score, 4),
            "n_runs": len(self.results),
            "n_passed": sum(1 for r in self.results if r.passed),
            "by_family": self.by_family(),
            "by_category": self.by_category(),
            "results": [r.as_dict() for r in self.results],
        }
