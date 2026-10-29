"""Plan gates for paid features, decided by app.core.editions.

Data transfer, model training and model inference spend compute that Avaloka
pays for on the hosted service, so editions.py lists them as COST gates. This
module is the one place the planner and the API ask "may this user do X?",
and it answers from the editions matrix, so the two cannot disagree:

* Self-hosted (AVALOKA_DEPLOYMENT unset or "self_hosted"): always allowed. The
  operator pays their own bill; editions.py says a COST gate never applies.
* Avaloka-hosted: allowed when the user's plan (free / professional /
  enterprise) has the capability in editions._HOSTED.
* The plan cannot be known (no user on the request, Supabase down): allowed
  and logged. A lookup failure must never lock paying users out.
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
}

_FEATURE_NAMES = {
    "data_transfer": "Data transfer",
    "model_training": "Model training",
    "model_inference": "Model inference",
}

_FEATURE_DETAIL = {
    "data_transfer": "move data between databases and cloud storage",
    "model_training": "train, schedule or re-run machine-learning models",
    "model_inference": "run predictions with your trained models",
}


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    feature: str
    plan: Optional[str]          # None when unknown
    deployment: str
    reason: str                  # short, for logs and tests

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
    deployment = editions.deployment_from_env()
    plan = (str(plan).strip().lower() or None) if plan else None

    if gates_disabled():
        return GateDecision(True, feature, plan, deployment, "gates disabled")
    if deployment == "self_hosted":
        return GateDecision(True, feature, plan, deployment, "self-hosted: no cost gates")
    if plan is None and not user_id:
        # Internal callers, scheduled re-runs created by an allowed user, tests.
        return GateDecision(True, feature, None, deployment, "no user on the request")
    if plan is None:
        from app.core import plan_limits  # lazy: keeps Supabase out of import time
        plan = plan_limits.try_resolve_plan(user_id)
    if plan is None:
        logger.warning("[plan-gate] plan unknown for user=%s; allowing %s", user_id, feature)
        return GateDecision(True, feature, None, deployment, "plan unknown: fail open")

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