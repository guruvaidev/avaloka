"""Task, dataset, and result schema shared by every benchmark and track.

The design principle (see ``benchmarks.md`` §9) is *one task schema* for external
benchmarks (DAB, DSBench, ...) and Avaloka-specific tracks alike, so the runner,
metrics, and reporting are written once.

A benchmark is a list of :class:`BenchmarkTask`. Running a task under a
:class:`~benchmarks.core.config.RunConfig` for N trials produces N
:class:`TrialResult` objects, aggregated into a :class:`TaskResult`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Split(str, Enum):
    """Public train / public dev / private held-out test (benchmark design rule #1)."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class FailureKind(str, Enum):
    """Stable failure taxonomy used in journals, scorecards, and regressions."""

    PROVISIONING = "provisioning_error"
    SUT_START = "sut_start_error"
    PROVIDER = "provider_error"
    TOOL = "tool_error"
    BUDGET = "budget_exceeded"
    TIMEOUT = "timeout"
    ANSWER = "answer_error"
    EVALUATOR = "evaluator_error"
    POLICY = "policy_error"
    UNKNOWN = "unknown_error"


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset a task runs against.

    ``loader`` returns the in-memory data (e.g. a ``pandas.DataFrame``) or a handle
    (connection string, path) depending on the track. ``conditions`` are the
    Track-1 dataset-condition tags (``imbalanced``, ``rare_event``, ``concept_drift``,
    ``high_cardinality``, ...) used to slice results by data regime.
    """

    id: str
    description: str = ""
    loader: Optional[Callable[[], Any]] = None
    conditions: tuple[str, ...] = ()
    # Fingerprint guards against silent dataset drift / contamination between runs.
    fingerprint: Optional[str] = None

    def load(self) -> Any:
        if self.loader is None:
            raise ValueError(f"DatasetSpec {self.id!r} has no loader")
        return self.loader()

    @staticmethod
    def fingerprint_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]


# An evaluator scores a raw mission answer against the task's reference and returns
# a dict of *quality* metrics in [0, 1]-ish space (higher is better) plus any
# track-specific diagnostics. It must be pure and deterministic given its inputs.
Evaluator = Callable[["BenchmarkTask", Any], Dict[str, Any]]


@dataclass(frozen=True)
class BenchmarkTask:
    """One gradable unit of work.

    ``mission`` is the natural-language instruction handed to the system under test
    (Avaloka or a baseline). ``reference`` is the ground-truth answer or artifact the
    evaluator compares against — for Track 1 this is the *full-data* result.
    """

    id: str
    track: str
    mission: str
    dataset: DatasetSpec
    evaluator: Evaluator
    reference: Any = None
    split: Split = Split.DEV
    # Free-form knobs an evaluator or adapter may read (target column, tolerance,
    # rare-segment definition, expected clarifying questions, injected failure, ...).
    params: Dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def evaluate(self, answer: Any) -> Dict[str, Any]:
        return self.evaluator(self, answer)


@dataclass
class MissionOutcome:
    """What an adapter returns for a single mission execution.

    Captures the 4-tuple ``(Quality, Cost, Latency, Data Scanned)`` inputs plus the
    trajectory needed to keep runs reproducible and auditable (design rule #3).
    Quality is *not* here — it is computed by the task evaluator from ``answer``.
    """

    answer: Any
    # None means unknown. Zero is a real, explicitly reported zero-cost execution.
    cost_usd: Optional[float] = None
    latency_s: float = 0.0
    data_scanned_ratio: Optional[float] = None  # fraction of population read, [0, 1]
    tool_calls: int = 0
    tokens: int = 0            # total; kept as-is so existing readers do not move
    # Token split and price provenance. Cost is a *derived* figure — recording
    # only the dollar total makes it impossible to tell a real measurement from
    # a default, or to reprice a past run when the catalog moves. See
    # benchmarks.core.pricing for what each cost_source value means; anything
    # other than "catalog" must not be read as an invoice.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_source: str = "unknown"
    retries: int = 0
    recovered: Optional[bool] = None  # Track 5: did it recover from an injected failure?
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_kind: Optional[FailureKind] = None


@dataclass
class TrialResult:
    """One trial of one task under one config."""

    task_id: str
    track: str
    config_id: str
    trial_index: int
    quality: Dict[str, float]
    outcome: MissionOutcome

    @property
    def passed(self) -> bool:
        """A trial "passes" when its evaluator emits a ``pass`` signal >= 0.5.

        Evaluators that don't define pass/fail should set ``quality["pass"]``
        explicitly; absent it we fall back to a ``score`` threshold.
        """
        if self.outcome.error:
            return False
        if "pass" in self.quality:
            return self.quality["pass"] >= 0.5
        return self.quality.get("score", 0.0) >= 0.5


@dataclass
class TaskResult:
    """Aggregate of all trials of a task under a config (see metrics.aggregate)."""

    task_id: str
    track: str
    config_id: str
    trials: List[TrialResult]
    stats: Dict[str, Any] = field(default_factory=dict)


def classify_failure(error: Optional[str]) -> Optional[FailureKind]:
    """Best-effort compatibility classifier for adapters not yet emitting a kind."""
    if not error:
        return None
    text = error.lower()
    prefixes = {
        "provisioning": FailureKind.PROVISIONING,
        "sut_start": FailureKind.SUT_START,
        "adapter_error": FailureKind.SUT_START,
        "avaloka_runtime_unavailable": FailureKind.SUT_START,
        "provider": FailureKind.PROVIDER,
        "llm_": FailureKind.PROVIDER,
        "tool_": FailureKind.TOOL,
        "budget": FailureKind.BUDGET,
        "timeout": FailureKind.TIMEOUT,
        "evaluator": FailureKind.EVALUATOR,
        "answer": FailureKind.ANSWER,
        "policy": FailureKind.POLICY,
    }
    for prefix, kind in prefixes.items():
        if text.startswith(prefix) or f"{prefix}_error" in text:
            return kind
    if "timed out" in text or "timeoutexpired" in text:
        return FailureKind.TIMEOUT
    return FailureKind.UNKNOWN
