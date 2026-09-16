"""Budget-aware intelligence.

A mission carries a financial limit and (optionally) a runtime limit. The
planner consults the budget to decide depth: whether to sample, how many
candidate models to train, when hyper-parameter search should stop and whether
an ensemble is worth its inference cost. The budget is a *hard* governance
boundary — when spend would exceed it the system refuses to proceed rather than
silently overrunning.
"""

from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """Raised when a planned action would exceed the mission's hard cost limit."""


@dataclass
class Budget:
    """Mission cost ceiling and live spend accounting.

    ``limit_usd`` is the hard ceiling the customer declared (``--budget`` /
    ``--max-cost``). ``None`` means *unconstrained* (still tracked, never
    blocked). All spend flows through :meth:`spend` so the remaining headroom
    is always known to the planner.
    """

    limit_usd: float | None = None
    max_runtime_seconds: float | None = None
    spent_usd: float = 0.0

    @property
    def unlimited(self) -> bool:
        return self.limit_usd is None

    def remaining(self) -> float:
        if self.limit_usd is None:
            return float("inf")
        return max(0.0, self.limit_usd - self.spent_usd)

    def can_afford(self, amount_usd: float) -> bool:
        if self.limit_usd is None:
            return True
        return self.spent_usd + amount_usd <= self.limit_usd + 1e-9

    def spend(self, amount_usd: float, what: str = "") -> float:
        """Record spend, refusing to cross the hard ceiling."""
        if not self.can_afford(amount_usd):
            raise BudgetExceeded(
                f"Action '{what or 'unspecified'}' costs ${amount_usd:,.2f} but only "
                f"${self.remaining():,.2f} of the ${self.limit_usd:,.2f} budget remains. "
                "Avaloka refuses to proceed rather than overrun the declared budget."
            )
        self.spent_usd += amount_usd
        return self.remaining()

    def fraction_used(self) -> float:
        if not self.limit_usd:
            return 0.0
        return min(1.0, self.spent_usd / self.limit_usd)
