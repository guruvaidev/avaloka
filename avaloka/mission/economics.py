"""The per-dollar value equation.

Avaloka does not claim to "replace a data scientist for $20". It claims to
compress a multi-person data-science workflow into a *governed mission* with
measurable time, cost and quality. Every mission therefore closes with an
economic report:

    Customer value   = human hours avoided x loaded hourly cost
                       + delay avoided + rework avoided + infra savings
    Avaloka cost     = mission charge + compute + storage + human review time
    Multiplier       = Customer value / Avaloka cost

The "estimated manual effort" is deliberately conservative and *calibratable*:
today it is derived from per-firefly heuristics; it should later be calibrated
from observed customer-workflow data rather than invented by an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from avaloka.mission.ledger import Ledger

# Pricing tiers map to the product architecture in the spec. ``mission_charge``
# is Avaloka's software/orchestration charge; compute and model cost are passed
# through separately so customers never confuse the software price with an
# inflated cloud bill.
TIER_MISSION_CHARGE = {
    "community": 0.0,     # free, local CLI — must produce a real success, not a teaser
    "professional": 29.0,
    "team": 79.0,
    "enterprise": 150.0,
}

DEFAULT_LOADED_HOURLY_RATE = 80.0  # loaded labour cost per the spec's illustration


@dataclass
class EconomicReport:
    mission_duration_seconds: float
    estimated_manual_effort_hours: float
    human_review_hours: float
    compute_cost_usd: float
    model_cost_usd: float
    mission_charge_usd: float
    total_mission_cost_usd: float       # Avaloka-side invoice (charge + compute + model)
    human_review_cost_usd: float        # review_hours x loaded rate
    total_customer_cost_usd: float      # mission cost + human review value
    estimated_labor_value_usd: float
    economic_multiplier: float
    loaded_hourly_rate_usd: float
    tier: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _duration(self) -> str:
        s = self.mission_duration_seconds
        return f"{s:,.1f} seconds" if s < 90 else f"{s / 60.0:,.1f} minutes"

    def rows(self) -> list[tuple[str, str]]:
        """Human-facing rows in the shape the spec prescribes (honest denominator).

        The multiplier is value created vs. *total customer cost* — which always
        includes the human review the customer still performs — so free/local
        missions report a defensible figure rather than a divide-by-zero blow-up.
        """
        return [
            ("Mission duration", self._duration()),
            ("Estimated manual effort", f"{self.estimated_manual_effort_hours:,.1f} hours"),
            ("Human review required", f"{self.human_review_hours:,.1f} hours"),
            ("Avaloka compute cost", f"${self.compute_cost_usd:,.2f}"),
            ("Avaloka model/API cost", f"${self.model_cost_usd:,.2f}"),
            ("Avaloka mission charge", f"${self.mission_charge_usd:,.2f}"),
            ("Total mission cost (Avaloka)", f"${self.total_mission_cost_usd:,.2f}"),
            ("Human review cost", f"${self.human_review_cost_usd:,.2f}"),
            ("Total customer cost", f"${self.total_customer_cost_usd:,.2f}"),
            ("Estimated labor value created", f"${self.estimated_labor_value_usd:,.0f}"),
            ("Economic multiplier", f"{self.economic_multiplier:,.1f}x"),
        ]


def compute_economics(
    ledger: Ledger,
    *,
    tier: str = "community",
    loaded_hourly_rate: float = DEFAULT_LOADED_HOURLY_RATE,
    human_review_hours: float | None = None,
    wall_clock_seconds: float | None = None,
) -> EconomicReport:
    """Turn the work-unit ledger into a defensible economic report.

    ``human_review_hours`` is the human time still required to sign off the
    mission. When not supplied it is estimated as a small fraction of the
    manual effort avoided (a mission that replaces a lot of work still needs
    proportionally more review).
    """
    manual_hours = ledger.manual_minutes / 60.0
    if human_review_hours is None:
        # Conservative: ~9% of avoided effort, floored so review is never free.
        human_review_hours = max(0.25, round(manual_hours * 0.09, 2))

    mission_charge = TIER_MISSION_CHARGE.get(tier, 0.0)
    compute = ledger.compute_cost_usd
    model = ledger.model_cost_usd
    total_mission_cost = mission_charge + compute + model

    review_cost = human_review_hours * loaded_hourly_rate
    # Total customer cost = the Avaloka invoice plus the human review the customer
    # still performs. This is always positive (review is never free), so the
    # multiplier stays defensible even for a free, local mission.
    total_customer_cost = total_mission_cost + review_cost

    labor_value = manual_hours * loaded_hourly_rate
    multiplier = labor_value / max(total_customer_cost, 0.01)

    duration = wall_clock_seconds if wall_clock_seconds is not None else ledger.duration_seconds

    return EconomicReport(
        mission_duration_seconds=duration,
        estimated_manual_effort_hours=round(manual_hours, 2),
        human_review_hours=round(human_review_hours, 2),
        compute_cost_usd=round(compute, 2),
        model_cost_usd=round(model, 2),
        mission_charge_usd=round(mission_charge, 2),
        total_mission_cost_usd=round(total_mission_cost, 2),
        human_review_cost_usd=round(review_cost, 2),
        total_customer_cost_usd=round(total_customer_cost, 2),
        estimated_labor_value_usd=round(labor_value, 2),
        economic_multiplier=round(multiplier, 1),
        loaded_hourly_rate_usd=loaded_hourly_rate,
        tier=tier,
    )
