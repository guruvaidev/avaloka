"""Edition and capability resolution.

Two *independent* axes decide what a deployment may do. Collapsing them into a
single "tier" enum is what previously made the model self-contradictory — it
could not express that a self-hosted developer has a scheduler while a
free hosted user does not, even though both are "free".

    deployment   who operates it and pays for the compute
    edition      what was purchased

Every gate must declare *why* it exists. There are exactly three legitimate
reasons, enumerated by :class:`GateReason`:

    OPEN         not gated at all
    COST         Avaloka pays for the cycles, so hosted use is capped
    COMMERCIAL   the implementation is not published; only commercial
                 builds carry the code at all

A gate that is neither COST nor COMMERCIAL is a nag. Do not add one.

Two invariants follow, and both are enforced by tests:

1. A self-hosted deployment never reports a capability "locked" for a
   COMMERCIAL reason. In an open-source build that code is simply absent, so
   there is nothing to unlock and nothing to circumvent.
2. A COST gate never applies to a self-hosted deployment. The operator is
   paying their own bill; capping them protects nobody.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Dict, Literal

Deployment = Literal["self_hosted", "avaloka_hosted"]
Edition = Literal["oss", "free", "professional", "enterprise"]

DEPLOYMENTS: tuple[Deployment, ...] = ("self_hosted", "avaloka_hosted")
EDITIONS: tuple[Edition, ...] = ("oss", "free", "professional", "enterprise")


class GateReason(str, Enum):
    """Why a capability is gated. See module docstring."""

    OPEN = "open"
    COST = "cost"
    COMMERCIAL = "commercial"


@dataclass(frozen=True)
class Capabilities:
    """What a resolved deployment may actually do.

    Every field defaults to ``False`` so a new capability is denied until it is
    deliberately granted in a matrix below.
    """

    # Core analysis — the open core.
    data_analysis: bool = False
    export_generated_code: bool = False
    local_execution: bool = False
    file_connectors: bool = False

    # Execution and scale.
    ray_distributed: bool = False
    entire_dataset: bool = False

    # Scheduling and automation.
    scheduler: bool = False
    batch_jobs: bool = False
    scheduled_delivery: bool = False

    # Connectivity to the customer's own estate.
    database_connectors: bool = False
    cloud_connectors: bool = False
    cloud_provisioning: bool = False

    # Collaboration.
    team_collaboration: bool = False
    share_analysis: bool = False
    comment_on_analysis: bool = False
    notifications: bool = False

    # Advanced reasoning. Present on the develop-1.7 line only: develop-1.6
    # ships no swarm code at all, which is what lets it release as the
    # open-source edition while swarm is held for its own launch.
    swarm_intelligence: bool = False

    # Machine learning.
    ml_training: bool = False
    ml_inference: bool = False

    # Enterprise operations.
    sso_audit: bool = False

    def enabled(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self) if getattr(self, f.name))

    def disabled(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self) if not getattr(self, f.name))


#: Why each capability is gated. Drives explanations shown to users and keeps
#: the rationale next to the matrix rather than scattered through call sites.
GATE_REASON: Dict[str, GateReason] = {
    "data_analysis": GateReason.OPEN,
    "export_generated_code": GateReason.OPEN,
    "local_execution": GateReason.OPEN,
    "file_connectors": GateReason.OPEN,
    "ray_distributed": GateReason.COST,
    "entire_dataset": GateReason.COST,
    "scheduler": GateReason.COST,
    "batch_jobs": GateReason.COST,
    "scheduled_delivery": GateReason.COMMERCIAL,
    "database_connectors": GateReason.COMMERCIAL,
    "cloud_connectors": GateReason.COMMERCIAL,
    "cloud_provisioning": GateReason.COMMERCIAL,
    "team_collaboration": GateReason.COMMERCIAL,
    "share_analysis": GateReason.COMMERCIAL,
    "comment_on_analysis": GateReason.COMMERCIAL,
    "notifications": GateReason.COMMERCIAL,
    "swarm_intelligence": GateReason.COMMERCIAL,
    # COST, not COMMERCIAL: the MTA code ships here and a self-hosted operator
    # runs it on their own hardware. What a *hosted* plan may spend on training
    # compute is what the gate governs.
    "ml_training": GateReason.COST,
    "ml_inference": GateReason.COST,
    "sso_audit": GateReason.COMMERCIAL,
}

#: Everything the *published* code can do. A self-hosted operator gets all of
#: it: they bring their own Ray cluster, cloud credentials and LLM keys, they
#: pay their own bill, and they carry their own risk. Nothing here is withheld,
#: because withholding it would protect nothing.
OSS_CAPABILITIES = Capabilities(
    data_analysis=True,
    export_generated_code=True,
    local_execution=True,
    file_connectors=True,
    ray_distributed=True,
    entire_dataset=True,
    scheduler=True,
    batch_jobs=True,
    # Model training and inference run locally on the operator's own hardware.
    # What open source cannot do is schedule them onto a cloud cluster it did
    # not provision — that is cloud_provisioning + scheduled_delivery, below.
    ml_training=True,
    ml_inference=True,
)

#: Hosted entitlements. These are COST gates: Avaloka operates the compute, so
#: what a hosted user may spend is governed by what they bought.
_HOSTED: Dict[Edition, Capabilities] = {
    # A hosted OSS identity is not a thing you can buy; treat it as free.
    "oss": Capabilities(
        data_analysis=True,
        export_generated_code=True,
        local_execution=True,
        file_connectors=True,
    ),
    "free": Capabilities(
        data_analysis=True,
        export_generated_code=True,
        local_execution=True,
        file_connectors=True,
        # No scheduler and no batch jobs: Avaloka pays for every cycle a free
        # hosted user would schedule. This is the COST line, not a value line.
    ),
    "professional": Capabilities(
        data_analysis=True,
        export_generated_code=True,
        local_execution=True,
        file_connectors=True,
        ray_distributed=True,
        entire_dataset=True,
        scheduler=True,
        batch_jobs=True,
        scheduled_delivery=True,
        # The defining Professional capability: reach the customer's own cloud
        # and databases, on the customer's own infrastructure and billing.
        database_connectors=True,
        cloud_connectors=True,
        cloud_provisioning=True,
        swarm_intelligence=True,
        ml_training=True,
        ml_inference=True,
    ),
    "enterprise": Capabilities(
        data_analysis=True,
        export_generated_code=True,
        local_execution=True,
        file_connectors=True,
        ray_distributed=True,
        entire_dataset=True,
        scheduler=True,
        batch_jobs=True,
        scheduled_delivery=True,
        database_connectors=True,
        cloud_connectors=True,
        cloud_provisioning=True,
        # The defining Enterprise capabilities: a company working as teams.
        team_collaboration=True,
        share_analysis=True,
        comment_on_analysis=True,
        notifications=True,
        swarm_intelligence=True,
        ml_training=True,
        ml_inference=True,
        sso_audit=True,
    ),
}


def normalize_deployment(value: str | None) -> Deployment:
    """Unknown values resolve to ``self_hosted``.

    Defaulting to self-hosted is the safe direction: it never grants a hosted
    entitlement, and it never charges anyone for compute they did not ask for.
    """
    candidate = (value or "").strip().lower().replace("-", "_")
    return candidate if candidate in DEPLOYMENTS else "self_hosted"  # type: ignore[return-value]


def normalize_edition(value: str | None) -> Edition:
    """Unknown values resolve to the least-privileged edition."""
    candidate = (value or "").strip().lower()
    return candidate if candidate in EDITIONS else "oss"  # type: ignore[return-value]


def commercial_build() -> bool:
    """True when the commercial package is installed alongside this build.

    This is the load-bearing check for the COMMERCIAL gate, and it is
    deliberately not a flag: an open-source build cannot satisfy it by editing
    a boolean, because the code being gated is not on disk. Editing this
    function to return ``True`` in an OSS checkout grants nothing — the imports
    behind those capabilities still fail.
    """
    try:
        import avaloka_commercial  # noqa: F401  (import-only probe)
    except ImportError:
        return False
    return True


def resolve_capabilities(
    deployment: str | None,
    edition: str | None = None,
    *,
    has_commercial_build: bool | None = None,
) -> Capabilities:
    """Resolve the capability set for a deployment/edition pair.

    ``has_commercial_build`` is injectable so tests can exercise both build
    shapes; in production it is discovered via :func:`commercial_build`.
    """
    mode = normalize_deployment(deployment)
    plan = normalize_edition(edition)
    commercial = commercial_build() if has_commercial_build is None else has_commercial_build

    if mode == "self_hosted":
        # The operator pays their own bill: no COST gates apply. COMMERCIAL
        # capabilities stay off in an open-source build because that code is
        # not present; a commercial self-hosted build unlocks them by licence.
        if not commercial:
            return OSS_CAPABILITIES
        return _HOSTED[plan] if plan != "oss" else OSS_CAPABILITIES

    granted = _HOSTED[plan]
    if commercial:
        return granted
    # An open-source build has no commercial code, so strip anything that
    # depends on it rather than advertising a capability that cannot run.
    stripped = {
        f.name: False
        for f in fields(granted)
        if GATE_REASON[f.name] is GateReason.COMMERCIAL
    }
    return replace(granted, **stripped)


def explain(capability: str, capabilities: Capabilities) -> str:
    """A short, honest reason a capability is unavailable.

    Never invites an upgrade for something that is simply open, and never
    nags a self-hosted operator about code they were not shipped.
    """
    if capability not in GATE_REASON:
        raise KeyError(f"unknown capability: {capability!r}")
    if getattr(capabilities, capability):
        return f"{capability} is available"
    reason = GATE_REASON[capability]
    if reason is GateReason.COST:
        return (
            f"{capability} is not included in this hosted plan because Avaloka "
            "provides the compute. Self-host, or upgrade, to enable it."
        )
    if reason is GateReason.COMMERCIAL:
        return f"{capability} is available in Avaloka Professional and Enterprise."
    return f"{capability} is available"
