"""The Firefly base class.

A firefly is an *economic* concept, not an anthropomorphic demo: it activates
only when its expertise is needed and corresponds to an expensive human
responsibility. Each firefly, when activated, performs real work, writes its
artifacts and records a :class:`WorkUnit` capturing the manual effort it
replaced and the cost it incurred.
"""

from __future__ import annotations

import abc
from typing import Any

from avaloka.mission.context import MissionContext
from avaloka.mission.ledger import WorkUnit, activation

# A nominal local-compute price so even free/local missions attribute an honest,
# tiny cost to the work performed (CPU-seconds, not zero). Overridable per unit.
LOCAL_CPU_USD_PER_SECOND = 0.0006  # ~ commodity vCPU-hour amortised


class Firefly(abc.ABC):
    """Base class for all Avaloka fireflies.

    Subclasses implement :meth:`perform`, returning ``(output_label,
    manual_minutes, extra)``. The base class handles timing, compute-cost
    attribution and ledger bookkeeping so every firefly is measured uniformly.
    """

    #: short, stable identifier (e.g. "data_scout")
    name: str = "firefly"
    #: the expensive human responsibility this firefly replaces/accelerates
    human_role: str = "specialist"

    def __init__(self, ctx: MissionContext) -> None:
        self.ctx = ctx

    @abc.abstractmethod
    def perform(self) -> tuple[str, float, dict[str, Any]]:
        """Do the work. Return (output description, manual_minutes_replaced, extra)."""

    def manual_minutes_estimate(self) -> float:  # optional override
        return 0.0

    def activate(self) -> WorkUnit:
        """Run the firefly, time it, cost it and record it on the ledger."""
        with activation() as clock:
            output, manual_minutes, extra = self.perform()

        model_cost = float(extra.pop("model_cost_usd", 0.0)) if extra else 0.0
        # Compute cost scales with wall-clock unless the firefly reported its own.
        compute_cost = float(extra.pop("compute_cost_usd", clock.seconds * LOCAL_CPU_USD_PER_SECOND))
        notes = str(extra.pop("notes", "")) if extra else ""

        unit = WorkUnit(
            firefly=self.name,
            human_role=self.human_role,
            output=output,
            manual_minutes=manual_minutes,
            compute_cost_usd=round(compute_cost, 6),
            model_cost_usd=round(model_cost, 6),
            duration_seconds=round(clock.seconds, 4),
            notes=notes,
            metadata=extra or {},
        )
        self.ctx.ledger.record(unit)
        # Reflect spend into the mission budget (hard ceiling enforced here).
        self.ctx.budget.spend(unit.compute_cost_usd + unit.model_cost_usd, what=self.name)
        return unit
