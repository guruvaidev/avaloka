"""Plan gates for paid features, decided by app.core.editions.

Data transfer, model training and model inference spend compute that Avaloka
pays for on the hosted service, so editions.py lists them as COST gates. This
module is the one place the planner and the API ask "may this user do X?",
and it answers from the editions matrix, so the two cannot disagree:

The deployment comes from editions.declared_deployment():

* Self-hosted (AVALOKA_DEPLOYMENT=self_hosted) or undeclared (unset, or a
  value it does not recognise): always allowed. The operator pays their own
  bill; editions.py says a COST gate never applies.
* Hosted (AVALOKA_DEPLOYMENT=avaloka_hosted): allowed when the user's plan
  (free / professional / enterprise) has the capability in editions._HOSTED.
* No user on the request (internal callers, scheduled re-runs): allowed.
* Hosted, and the plan lookup failed (Supabase down): refused as
  *unavailable*, never as an upgrade. The API answers 503 "plan check
  unavailable" and chat says the check could not run; neither tells a user
  who may be paying to upgrade.
* AVALOKA_DISABLE_PLAN_GATES=1: allowed (operator kill switch).

Everything is read per call; nothing is cached here (plan_limits caches the
Supabase lookup).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from app.core import editions

logger = logging.getLogger(__name__)

#: Feature -> the editions.py capability that governs it.
#: A data transfer runs as a Ray job on the cluster Avaloka operates, so it is
#: the ray_distributed COST gate (Professional and Enterprise when hosted).
FEATURE_CAPABILITY = {
    "data_transfer": "ray_distributed",
    "model_training": "ml_training",
    "model_inference": "ml_inference",
    # Ray's own dashboard at /ray/: the same plans that run work on Ray.
    "ray_dashboard": "ray_distributed",
}

_FEATURE_NAMES = {
    "data_transfer": "Data transfer",
    "model_training": "Model training",
    "model_inference": "Model inference",
    "ray_dashboard": "The Ray dashboard",
}

_FEATURE_DETAIL = {
    "data_transfer": "move data between databases and cloud storage",
    "model_training": "train, schedule or re-run machine-learning models",
    "model_inference": "run predictions with your trained models",
    "ray_dashboard": "watch and stop your Ray jobs in the Ray dashboard",
}


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    feature: str
    plan: Optional[str]          # None when unknown
    deployment: str
    reason: str                  # short, for logs and tests
    #: True when the plan lookup failed on a hosted deployment: refused, but
    #: because the plan could not be checked, not because the plan lacks it.
    unavailable: bool = False

    def allowed_plans(self) -> list:
        """Hosted plans that include the feature, cheapest first,
        e.g. ['professional', 'enterprise']."""
        cap = FEATURE_CAPABILITY[self.feature]
        return [
            e for e in ("free", "professional", "enterprise")
            if getattr(editions.resolve_capabilities("avaloka_hosted", e), cap)
        ]


def gates_disabled() -> bool:
    return os.getenv("AVALOKA_DISABLE_PLAN_GATES", "").strip() == "1"


def check_feature(
    feature: str,
    *,
    user_id: Optional[str] = None,
    plan: Optional[str] = None,
) -> GateDecision:
    """May this user use ``feature`` on this deployment?

    ``plan`` is used when the caller already knows it; otherwise it is looked
    up from ``user_id`` with plan_limits.try_resolve_plan (None on failure).
    """
    if feature not in FEATURE_CAPABILITY:
        raise KeyError(f"unknown gated feature: {feature!r}")
    declared = editions.declared_deployment()
    deployment = "avaloka_hosted" if declared == "hosted" else "self_hosted"
    plan = (str(plan).strip().lower() or None) if plan else None

    if gates_disabled():
        return GateDecision(True, feature, plan, deployment, "gates disabled")
    if declared == "self_hosted":
        return GateDecision(True, feature, plan, deployment, "self-hosted: no cost gates")
    if declared == "undeclared":
        return GateDecision(True, feature, plan, deployment,
                            "deployment undeclared: treated as self-hosted")
    if plan is None and not user_id:
        # Internal callers, scheduled re-runs created by an allowed user, tests.
        return GateDecision(True, feature, None, deployment, "no user on the request")
    if plan is None:
        from app.core import plan_limits  # lazy: keeps Supabase out of import time
        plan = plan_limits.try_resolve_plan(user_id)
    if plan is None:
        logger.warning("[plan-gate] plan lookup failed for user=%s; refusing %s as unavailable",
                       user_id, feature)
        return GateDecision(False, feature, None, deployment, "plan check unavailable",
                            unavailable=True)

    caps = editions.resolve_capabilities(deployment, plan)
    allowed = bool(getattr(caps, FEATURE_CAPABILITY[feature]))
    return GateDecision(allowed, feature, plan, deployment,
                        "plan includes it" if allowed else "plan does not include it")


def upgrade_message(decision: GateDecision) -> str:
    """The chat / API message for a blocked feature."""
    plan = (decision.plan or "free").replace("_", " ").title()
    allowed = " and ".join(p.title() for p in decision.allowed_plans()) or "paid"
    return (
        f"{_FEATURE_NAMES[decision.feature]} is available on the {allowed} plans. "
        f"Your current plan is {plan}. Upgrade your plan to "
        f"{_FEATURE_DETAIL[decision.feature]}; analysis, charts and code keep "
        "working on your current plan."
    )


def unavailable_message(decision: GateDecision) -> str:
    """The chat / API message when the plan could not be checked."""
    return (
        f"Plan check unavailable: {_FEATURE_NAMES[decision.feature].lower()} could not be "
        "checked against your plan right now. Please try again in a minute; analysis, "
        "charts and code keep working."
    )
