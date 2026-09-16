"""The work-unit ledger.

Every firefly activation is recorded as a :class:`WorkUnit` — an economically
meaningful unit of work that maps to a human responsibility. The metaphor only
becomes credible when agents are associated with *measured* work, not
anthropomorphic demonstrations, so the ledger is the spine of Avaloka's
economics story.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class WorkUnit:
    """One unit of validated work performed by a firefly.

    Attributes mirror the economic identity in the spec: each firefly replaces
    or accelerates an expensive human responsibility and the unit therefore
    carries the manual effort it stands in for and the real cost it incurred.
    """

    firefly: str
    human_role: str
    output: str
    manual_minutes: float          # human effort this unit replaces / accelerates
    compute_cost_usd: float = 0.0  # CPU/GPU/storage actually consumed
    model_cost_usd: float = 0.0    # LLM / hosted-model spend (0 when fully local)
    duration_seconds: float = 0.0  # wall-clock time the firefly was active
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Ledger:
    """Append-only record of every firefly that activated during a mission."""

    def __init__(self) -> None:
        self._units: list[WorkUnit] = []

    def record(self, unit: WorkUnit) -> WorkUnit:
        self._units.append(unit)
        return unit

    @property
    def units(self) -> list[WorkUnit]:
        return list(self._units)

    # --- aggregates -------------------------------------------------------
    @property
    def manual_minutes(self) -> float:
        return sum(u.manual_minutes for u in self._units)

    @property
    def compute_cost_usd(self) -> float:
        return sum(u.compute_cost_usd for u in self._units)

    @property
    def model_cost_usd(self) -> float:
        return sum(u.model_cost_usd for u in self._units)

    @property
    def duration_seconds(self) -> float:
        return sum(u.duration_seconds for u in self._units)

    def as_list(self) -> list[dict[str, Any]]:
        return [u.as_dict() for u in self._units]


class activation:
    """Context manager that times a firefly activation.

    Usage::

        with activation() as clock:
            ... do work ...
        unit = WorkUnit(..., duration_seconds=clock.seconds)
    """

    def __init__(self) -> None:
        self.seconds = 0.0
        self._start = 0.0

    def __enter__(self) -> "activation":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start
