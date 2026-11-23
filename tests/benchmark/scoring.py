"""
Scoring + scorecard generation for the benchmark.

Running a task produces a ``TaskResult`` (the per-check outcomes + a 0..1 score).
A run over many tasks produces a ``Scorecard`` that can be rendered as JSON
(machine-readable, for regression gating) or Markdown (human-readable report).

Scoring rules
-------------
* Any failed CRITICAL check  -> task score 0.0 and ``passed = False``.
* Otherwise score = weighted fraction of passed checks, where
  CRITICAL=3, MAJOR=2, MINOR=1.
* Suite score = weight-averaged task scores.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .tasks import BenchmarkTask, CheckOutcome, Severity, TaskContext

_SEVERITY_WEIGHT = {Severity.CRITICAL: 3.0, Severity.MAJOR: 2.0, Severity.MINOR: 1.0}


@dataclass
class TaskResult:
    task_id: str
    stage: str
    dataset_key: str
    source_kind: str
    passed: bool
    score: float
    outcomes: List[CheckOutcome] = field(default_factory=list)
    duration_s: float = 0.0
    error: Optional[str] = None
    client: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["outcomes"] = [
            {**asdict(o), "severity": o.severity.value} for o in self.outcomes
        ]
        return d


def score_outcomes(outcomes: List[CheckOutcome]) -> tuple[bool, float]:
    if not outcomes:
        return False, 0.0
    critical_failed = any(
        (not o.passed) and o.severity == Severity.CRITICAL for o in outcomes
    )
    if critical_failed:
        return False, 0.0
    total = sum(_SEVERITY_WEIGHT[o.severity] for o in outcomes)
    earned = sum(_SEVERITY_WEIGHT[o.severity] for o in outcomes if o.passed)
    score = earned / total if total else 0.0
    return score >= 0.999, score


def run_checks(
    task: BenchmarkTask,
    result: Dict[str, Any],
    ctx: TaskContext,
    client_name: str,
    duration_s: float = 0.0,
) -> TaskResult:
    outcomes: List[CheckOutcome] = []
    for check in task.checks:
        try:
            outcomes.append(check(result, ctx))
        except Exception as e:  # a check that throws is a failed MAJOR check
            outcomes.append(
                CheckOutcome(
                    name=getattr(check, "__name__", "check"),
                    passed=False,
                    severity=Severity.MAJOR,
                    detail=f"check raised {type(e).__name__}: {e}",
                )
            )
    passed, score = score_outcomes(outcomes)
    return TaskResult(
        task_id=task.task_id,
        stage=task.stage.value,
        dataset_key=task.dataset_key,
        source_kind=ctx.source_kind,
        passed=passed,
        score=round(score, 4),
        outcomes=outcomes,
        duration_s=round(duration_s, 3),
        error=result.get("error"),
        client=client_name,
    )


@dataclass
class Scorecard:
    client: str
    results: List[TaskResult] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add(self, r: TaskResult) -> None:
        self.results.append(r)

    @property
    def suite_score(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.score for r in self.results) / len(self.results), 4)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(1 for r in self.results if r.passed) / len(self.results), 4)

    def by_stage(self) -> Dict[str, float]:
        agg: Dict[str, List[float]] = {}
        for r in self.results:
            agg.setdefault(r.stage, []).append(r.score)
        return {k: round(sum(v) / len(v), 4) for k, v in agg.items()}

    def to_json(self) -> str:
        return json.dumps(
            {
                "client": self.client,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "suite_score": self.suite_score,
                "pass_rate": self.pass_rate,
                "by_stage": self.by_stage(),
                "meta": self.meta,
                "results": [r.as_dict() for r in self.results],
            },
            indent=2,
        )

    def to_markdown(self) -> str:
        lines = [
            f"# Avaloka Benchmark Scorecard — `{self.client}`",
            "",
            f"- **Suite score:** {self.suite_score:.3f}",
            f"- **Pass rate:** {self.pass_rate:.1%} "
            f"({sum(1 for r in self.results if r.passed)}/{len(self.results)})",
            "",
            "## By stage",
            "",
            "| Stage | Score |",
            "| --- | --- |",
        ]
        for stage, sc in sorted(self.by_stage().items()):
            lines.append(f"| {stage} | {sc:.3f} |")
        lines += ["", "## Tasks", "", "| Task | Dataset | Source | Pass | Score | Notes |", "| --- | --- | --- | --- | --- | --- |"]
        for r in self.results:
            note = ""
            failed = [o for o in r.outcomes if not o.passed]
            if failed:
                note = "; ".join(f"{o.name} ({o.severity.value})" for o in failed[:3])
            elif r.error:
                note = r.error[:60]
            mark = "✅" if r.passed else "❌"
            lines.append(
                f"| {r.task_id} | {r.dataset_key} | {r.source_kind} | {mark} | "
                f"{r.score:.2f} | {note} |"
            )
        return "\n".join(lines)
