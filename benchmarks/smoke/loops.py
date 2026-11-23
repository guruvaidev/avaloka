"""Do the retry loops terminate, and do they make progress?

Avaloka's coder/validator cycle is its quality mechanism. Two ways it fails
that a functional test will not catch:

* **Non-termination.** A loop with no ceiling, or a ceiling that is never
  compared, runs until something else times out. In a pilot that is a hung
  session, not an error message.
* **Oscillation.** Fix A, break B, fix B, break A. It terminates, so it looks
  fine — and it produces working code roughly half the time.

The second is why a bounded loop is not sufficient evidence of a healthy one.
These checks read the loop's *structure* statically (no LLM calls, so they run
in CI) and exercise the classifier's behaviour on synthetic traces.
"""

from __future__ import annotations

import inspect
import re
from typing import List

from benchmarks.smoke.result import Severity, SmokeResult

SURFACE = "loops"


def run(*, include_slow: bool = False) -> List[SmokeResult]:
    return [
        _check_retry_ceiling_exists(),
        _check_ceiling_is_configurable(),
        _check_feedback_is_threaded(),
        _check_oscillation_is_detectable(),
        _check_termination_on_synthetic_traces(),
    ]


def _coder_source() -> str:
    try:
        from app.agents import coder
        return inspect.getsource(coder)
    except Exception:  # noqa: BLE001 - import may need optional deps
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "app/agents/coder.py"
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def _check_retry_ceiling_exists() -> SmokeResult:
    """An unbounded retry loop is a hung session, not an error."""
    src = _coder_source()
    # Match a numeric literal OR a named constant. An earlier version only
    # matched digits, so naming the ceiling CODER_MAX_RETRIES -- an improvement --
    # made this check report the loop as unbounded. A structural check that
    # penalises the cleaner form is worse than no check.
    bounded = bool(re.search(r"retry_count\s*>=?\s*(\d+|[A-Z_][A-Z0-9_]*)", src))
    return SmokeResult(
        check="retry_ceiling_exists", surface=SURFACE, passed=bounded,
        severity=Severity.CRITICAL,
        detail=("the coder loop compares retry_count against a ceiling" if bounded
                else "no retry ceiling found; the loop may not terminate"),
    )


def _check_ceiling_is_configurable() -> SmokeResult:
    """A magic number cannot be tuned per deployment.

    Warning, not critical: an inline bound still terminates. But a pilot that
    needs a different budget should not require a code change.
    """
    src = _coder_source()
    configurable = "AVALOKA_CODER_MAX_RETRIES" in src or "MAX_RETRIES" in src
    return SmokeResult(
        check="retry_ceiling_configurable", surface=SURFACE, passed=configurable,
        severity=Severity.WARNING,
        detail=("the retry ceiling is named/configurable" if configurable
                else "the retry ceiling is an inline literal; tuning it needs a code change"),
    )


def _check_feedback_is_threaded() -> SmokeResult:
    """A retry without feedback is a re-roll, not a repair."""
    src = _coder_source()
    threaded = "code_validation_feedback" in src
    return SmokeResult(
        check="validator_feedback_threaded", surface=SURFACE, passed=threaded,
        severity=Severity.CRITICAL,
        detail=("validator feedback is passed back into the coder" if threaded
                else "no feedback channel; retries would repeat the same mistake"),
    )


def _check_oscillation_is_detectable() -> SmokeResult:
    from app.core.loop_telemetry import Attempt, LoopOutcome, LoopTrace

    t = LoopTrace(loop_id="smoke", max_retries=5)
    for i, stage in enumerate(("syntax", "static", "syntax")):
        t.record(Attempt(index=i, failing_stage=stage, feedback=f"{stage} objected"))
    detected = t.outcome is LoopOutcome.OSCILLATED
    return SmokeResult(
        check="oscillation_detected", surface=SURFACE, passed=detected,
        severity=Severity.CRITICAL,
        detail=("a loop that revisits a solved objection is classified as oscillating"
                if detected else "oscillation is not distinguished from exhaustion"),
        evidence={"outcome": t.outcome.value},
    )


def _check_termination_on_synthetic_traces() -> SmokeResult:
    """Every synthetic loop shape must classify to a terminal outcome."""
    from app.core.loop_telemetry import Attempt, LoopOutcome, LoopTrace

    shapes = {
        "immediate_pass": (None,),
        "repair": ("syntax", None),
        "exhausted": ("syntax", "static", "logical"),
        "oscillating": ("syntax", "static", "syntax"),
    }
    unresolved: List[str] = []
    for name, stages in shapes.items():
        t = LoopTrace(loop_id=name, max_retries=2)
        for i, stage in enumerate(stages):
            t.record(Attempt(index=i, failing_stage=stage,
                             feedback=None if stage is None else f"{stage} objected"))
        if t.outcome not in set(LoopOutcome):
            unresolved.append(name)
    return SmokeResult(
        check="loop_shapes_terminate", surface=SURFACE, passed=not unresolved,
        severity=Severity.CRITICAL,
        detail="every loop shape classifies to a terminal outcome" if not unresolved
               else f"unclassified: {unresolved}",
    )
