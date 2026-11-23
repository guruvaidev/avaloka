"""Shared result types for smoke checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class Severity(str, Enum):
    """How much a failed check should stop a pilot."""

    CRITICAL = "critical"   # do not run: results would be untrustworthy
    WARNING = "warning"     # runnable, but a known weakness
    INFO = "info"


@dataclass
class SmokeResult:
    check: str
    surface: str            # harness | prompts | loops | graph
    passed: bool
    severity: Severity = Severity.CRITICAL
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check, "surface": self.surface, "passed": self.passed,
            "severity": self.severity.value, "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass
class SmokeReport:
    results: List[SmokeResult] = field(default_factory=list)

    def extend(self, more: Iterable[SmokeResult]) -> None:
        self.results.extend(more)

    @property
    def failures(self) -> List[SmokeResult]:
        return [r for r in self.results if not r.passed]

    @property
    def blocking(self) -> List[SmokeResult]:
        return [r for r in self.failures if r.severity is Severity.CRITICAL]

    @property
    def safe_to_run(self) -> bool:
        """False when any CRITICAL check failed.

        Deliberately strict: the point of a pre-flight check is that a red
        result stops the flight, not that it is noted and overridden.
        """
        return not self.blocking

    def by_surface(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            bucket = out.setdefault(r.surface, {"passed": 0, "failed": 0})
            bucket["passed" if r.passed else "failed"] += 1
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "safe_to_run": self.safe_to_run,
            "n_checks": len(self.results),
            "n_failed": len(self.failures),
            "n_blocking": len(self.blocking),
            "by_surface": self.by_surface(),
            "results": [r.as_dict() for r in self.results],
        }

    def summary(self) -> str:
        if not self.failures:
            return f"All {len(self.results)} smoke checks passed."
        head = f"{len(self.failures)} of {len(self.results)} smoke checks failed"
        if self.blocking:
            head += f" ({len(self.blocking)} BLOCKING)"
        return head + ": " + "; ".join(r.check for r in self.failures[:4])
