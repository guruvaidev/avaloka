"""Why did the loop stop?

Avaloka's coder/validator cycle is its central quality mechanism: the Coder
writes, three validators inspect, feedback returns, the Coder rewrites. It is
bounded by ``retry_count >= 2`` in ``app/agents/coder.py`` — an inline
constant, with nothing recording what happened.

So the questions you would actually want answered about loop quality cannot be
answered:

* How often does the loop converge versus exhaust its budget?
* Which validator raises the objection that ends up unfixable?
* Does the second attempt fix the first attempt's complaint, or a different one?
* Is the loop oscillating — fixing A, breaking B, fixing B, breaking A?

That last one matters most and is invisible today. An oscillating loop looks
identical to a converging one from the outside: both stop after two attempts.
One produced working code; the other produced a coin flip.

This records the shape of each loop so those become measurements. It is
deliberately passive — it observes, never intervenes — so switching it off
changes nothing about behaviour.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Retry ceiling. Mirrors the constant in coder.py; exported so the bound is
#: named and configurable rather than a literal buried in a conditional.
DEFAULT_MAX_RETRIES = int(os.getenv("AVALOKA_CODER_MAX_RETRIES", "2"))


class LoopOutcome(str, Enum):
    """How a loop ended. The distinction the current code does not record."""

    CONVERGED = "converged"          # validators satisfied
    EXHAUSTED = "exhausted"          # hit the retry ceiling still failing
    HARD_FAILED = "hard_failed"      # unrecoverable, abandoned early
    OSCILLATED = "oscillated"        # alternated between failures, never settled
    ABORTED = "aborted"              # cancelled or errored outside the loop


@dataclass
class Attempt:
    """One pass through the loop."""

    index: int
    failing_stage: Optional[str] = None      # syntax | static | logical | execution
    feedback: Optional[str] = None
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.failing_stage is None

    def signature(self) -> str:
        """What went wrong, normalised for comparison across attempts.

        Compared by stage and the leading text of the feedback rather than the
        whole message, because messages embed line numbers and identifiers that
        shift between attempts while the underlying objection is the same.
        """
        if self.passed:
            return "ok"
        head = (self.feedback or "").strip().lower()[:80]
        return f"{self.failing_stage}:{head}"


@dataclass
class LoopTrace:
    """The full shape of one coder/validator loop."""

    loop_id: str
    max_retries: int = DEFAULT_MAX_RETRIES
    attempts: List[Attempt] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)

    @property
    def outcome(self) -> LoopOutcome:
        if not self.attempts:
            return LoopOutcome.ABORTED
        if self.attempts[-1].passed:
            return LoopOutcome.CONVERGED
        if self._oscillating():
            return LoopOutcome.OSCILLATED
        if len(self.attempts) > self.max_retries:
            return LoopOutcome.EXHAUSTED
        return LoopOutcome.HARD_FAILED

    def _oscillating(self) -> bool:
        """True when a previously-seen failure signature comes back.

        Re-encountering an objection you already addressed means the fix for
        the later problem undid the fix for the earlier one. Three attempts are
        needed to see it: fail A, fail B, fail A again.
        """
        signatures = [a.signature() for a in self.attempts if not a.passed]
        if len(signatures) < 3:
            return False
        return len(set(signatures)) < len(signatures)

    @property
    def converged(self) -> bool:
        return self.outcome is LoopOutcome.CONVERGED

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)

    @property
    def duration_s(self) -> float:
        return sum(a.duration_s for a in self.attempts)

    def stages_hit(self) -> List[str]:
        """Distinct validator stages that objected, in order of first appearance."""
        seen: List[str] = []
        for a in self.attempts:
            if a.failing_stage and a.failing_stage not in seen:
                seen.append(a.failing_stage)
        return seen

    def blocking_stage(self) -> Optional[str]:
        """The stage that stopped the loop converging, or None if it converged.

        A converged loop has no blocking stage. Returning the last objection it
        *overcame* would be actively misleading in an aggregate: a validator
        that raises issues the Coder reliably fixes would rank alongside one
        that reliably defeats it, which is the opposite of what the number is
        for.
        """
        if self.converged:
            return None
        for a in reversed(self.attempts):
            if a.failing_stage:
                return a.failing_stage
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "outcome": self.outcome.value,
            "converged": self.converged,
            "attempts_used": self.attempts_used,
            "max_retries": self.max_retries,
            "duration_s": round(self.duration_s, 3),
            "stages_hit": self.stages_hit(),
            "blocking_stage": self.blocking_stage(),
            "attempts": [
                {"index": a.index, "failing_stage": a.failing_stage,
                 "passed": a.passed, "duration_s": round(a.duration_s, 3)}
                for a in self.attempts
            ],
        }

    def summary(self) -> str:
        text = (f"loop {self.loop_id}: {self.outcome.value} after "
                f"{self.attempts_used} attempt(s)")
        blocking = self.blocking_stage()
        if blocking and not self.converged:
            text += f", blocked at {blocking}"
        return text


def summarise(traces: Sequence[LoopTrace]) -> Dict[str, Any]:
    """Aggregate loop behaviour across a run. What a harness reports.

    Convergence *rate* is the number worth watching. A single loop failing is
    normal; a convergence rate that drops after a prompt or model change is a
    regression, and today nothing would notice.
    """
    if not traces:
        return {"n_loops": 0}
    outcomes: Dict[str, int] = {}
    for trace in traces:
        key = trace.outcome.value
        outcomes[key] = outcomes.get(key, 0) + 1
    converged = [t for t in traces if t.converged]
    blocking: Dict[str, int] = {}
    for trace in traces:
        stage = trace.blocking_stage()
        if stage and not trace.converged:
            blocking[stage] = blocking.get(stage, 0) + 1
    return {
        "n_loops": len(traces),
        "convergence_rate": round(len(converged) / len(traces), 4),
        "outcomes": outcomes,
        "mean_attempts": round(sum(t.attempts_used for t in traces) / len(traces), 2),
        "mean_attempts_when_converged": (
            round(sum(t.attempts_used for t in converged) / len(converged), 2)
            if converged else None
        ),
        "blocking_stages": dict(sorted(blocking.items(), key=lambda kv: -kv[1])),
    }


def trace_from_state(state: Dict[str, Any], loop_id: str = "coder") -> LoopTrace:
    """Best-effort reconstruction of a loop from the agent state.

    The graph does not yet thread a trace object through, so this reads what
    ``coder.py`` already writes (``retry_count``, ``code_validation_feedback``).
    It is a bridge, not the destination: threading a real trace gives per-attempt
    timings and stages this cannot recover.
    """
    trace = LoopTrace(loop_id=loop_id)
    retries = int(state.get("retry_count") or 0)
    feedback = state.get("code_validation_feedback")
    for i in range(max(retries, 1)):
        final = i == max(retries, 1) - 1
        trace.record(Attempt(
            index=i,
            failing_stage=None if (final and not feedback) else "unknown",
            feedback=feedback if not final else None,
        ))
    return trace
