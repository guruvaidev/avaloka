"""Mission core: the atomic Avaloka product is a *Data Mission*.

A customer does not buy chat messages, agent invocations or notebook cells.
They commission a completed mission: ``goal + budget -> governed deliverable
bundle``. This package holds the economic and accounting machinery that makes a
mission a measurable, priced outcome rather than an interaction.
"""

from avaloka.mission.budget import Budget, BudgetExceeded
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.mission.economics import EconomicReport, compute_economics
from avaloka.mission.ledger import Ledger, WorkUnit

__all__ = [
    "Budget",
    "BudgetExceeded",
    "MissionContext",
    "MissionKind",
    "EconomicReport",
    "compute_economics",
    "Ledger",
    "WorkUnit",
]
