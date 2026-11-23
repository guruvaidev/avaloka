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

Separately from gates, one capability carries a declared PREVIEW LIMIT: the
swarm ships in every edition, but its fan-out is capped at 2 clones per
mission on the open-source and Free editions (16 Professional, unlimited
Enterprise). That is an owner decision, applied to self-hosted installs too,
and documented as its own category in docs/EDITIONS.md rather than filed as
COST or COMMERCIAL. The number lives in avaloka/swarm_limits.py; see
swarm_limits() below. Add no other limit without declaring it the same way.

The swarm *console* is a different thing from the swarm. The console is the
live /swarm page and the swarm/trace API routes behind it (missions, records,
clones, episodes, telemetry, the thread trace, agent metrics). The swarm runs
in every edition; the console is a Professional and Enterprise feature, gated
COMMERCIAL by owner decision (2026-10-18). Open-source and Free editions get
the same mission told back in the chat instead (avaloka.swarm_narration), so
nothing a clone found is withheld — only the live window onto it. See
swarm_console_gate() below; the API dependency that applies it lives in
app/api/swarm_gate.py.

Two invariants follow, and both are enforced by tests:

1. A self-hosted deployment never reports a capability "locked" for a
   COMMERCIAL reason. In an open-source build that code is simply absent, so
   there is nothing to unlock and nothing to circumvent.
2. A COST gate never applies to a self-hosted deployment. The operator is
   paying their own bill; capping them protects nobody.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Dict, Literal, Mapping, Optional

logger = logging.getLogger(__name__)

Deployment = Literal["self_hosted", "avaloka_hosted"]
#: What the deployment says it is. ``undeclared`` is neither value set: unset,
#: empty, or a word this build does not know.
DeploymentState = Literal["hosted", "self_hosted", "undeclared"]
DEPLOYMENT_ENV = "AVALOKA_DEPLOYMENT"
#: 1.6 name of the same three states; declared_deployment() returns it.
DeclaredDeployment = DeploymentState
Edition = Literal["oss", "free", "professional", "enterprise"]

DEPLOYMENTS: tuple[Deployment, ...] = ("self_hosted", "avaloka_hosted")
EDITIONS: tuple[Edition, ...] = ("oss", "free", "professional", "enterprise")

#: Whether a capability's code is on this line. "coming" means the plan is
#: entitled to it but it has not shipped, so nothing may present it as
#: available today (the Billing tab shows "Coming soon").
CapabilityStatus = Literal["shipped", "coming"]


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
    auto_insights: bool = False

    # Execution and scale.
    ray_distributed: bool = False
    entire_dataset: bool = False

    # Scheduling and automation.
    scheduler: bool = False
    batch_jobs: bool = False
    scheduled_delivery: bool = False

    # The reports workspace (/reports): saved reports, browsed and organised
    # across projects. Sharing a report by email or public link is not this
    # flag; every plan has that and nothing gates it.
    reports: bool = False

    # Connectivity to the customer's own estate.
    database_connectors: bool = False
    cloud_connectors: bool = False
    cloud_provisioning: bool = False

    # Collaboration. Owner ruling 2026-11-01 (NEW-BG-01, revised): teams and
    # collaborator invites are a paid offering from Professional up
    # (team_collaboration); the organisation's own structure -- departments,
    # roles, branches and users -- is Enterprise (org_structure).
    team_collaboration: bool = False
    org_structure: bool = False
    share_analysis: bool = False
    comment_on_analysis: bool = False
    notifications: bool = False

    # Swarm: a mission clones Avaloka into sub-Avalokas that work the data in
    # parallel. Open in every edition; how many clones a mission may dispatch
    # is a limit, not a gate. See swarm_limits() below.
    swarm_intelligence: bool = False

    # The dataset catalog (docs/design/1.7-DATA-PLATFORM.md part C). The
    # personal catalog -- every dataset this user's analyses touched, with
    # schema versions, reuse and delete -- is open in every edition.
    # Project and organisation visibility, stewards, access requests and the
    # audit trail are the shared catalog, which joins the Collaboration line
    # beside share_analysis and is Enterprise only (owner ruling 2026-11-01).
    dataset_catalog: bool = False
    shared_catalog: bool = False

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
    "auto_insights": GateReason.OPEN,
    "ray_distributed": GateReason.COST,
    "entire_dataset": GateReason.COST,
    "scheduler": GateReason.COST,
    "batch_jobs": GateReason.COST,
    "scheduled_delivery": GateReason.COMMERCIAL,
    # COST: the reports workspace ships in the published UI, and on the hosted
    # service Avaloka stores and serves every saved report.
    "reports": GateReason.COST,
    "database_connectors": GateReason.COMMERCIAL,
    "cloud_connectors": GateReason.COMMERCIAL,
    "cloud_provisioning": GateReason.COMMERCIAL,
    "team_collaboration": GateReason.COMMERCIAL,
    "org_structure": GateReason.COMMERCIAL,
    "share_analysis": GateReason.COMMERCIAL,
    "comment_on_analysis": GateReason.COMMERCIAL,
    "notifications": GateReason.COMMERCIAL,
    # OPEN: the swarm ships in the published tree. Its fan-out is limited per
    # edition by swarm_limits(), which states that limit rather than hiding it.
    "swarm_intelligence": GateReason.OPEN,
    "dataset_catalog": GateReason.OPEN,
    "shared_catalog": GateReason.COMMERCIAL,
    # COST, not COMMERCIAL: the MTA code ships here and a self-hosted operator
    # runs it on their own hardware. What a *hosted* plan may spend on training
    # compute is what the gate governs.
    "ml_training": GateReason.COST,
    "ml_inference": GateReason.COST,
    "sso_audit": GateReason.COMMERCIAL,
}

#: Shipped or coming, per capability. Every capability declares one, like its
#: gate reason. Entitlement (the matrices below) and status are separate: a
#: plan can be entitled to something that has not shipped yet.
CAPABILITY_STATUS: Dict[str, CapabilityStatus] = {
    "data_analysis": "shipped",
    "export_generated_code": "shipped",
    "local_execution": "shipped",
    "file_connectors": "shipped",
    "auto_insights": "shipped",
    # 1.7: the personal dataset catalog (#753) and the Enterprise shared catalog.
    "dataset_catalog": "shipped",
    "shared_catalog": "shipped",
    "ray_distributed": "shipped",
    "entire_dataset": "shipped",
    "scheduler": "shipped",
    "batch_jobs": "shipped",
    "scheduled_delivery": "shipped",
    "reports": "shipped",
    "database_connectors": "shipped",
    "cloud_connectors": "shipped",
    "cloud_provisioning": "shipped",
    "team_collaboration": "shipped",
    "org_structure": "shipped",
    "share_analysis": "shipped",
    "comment_on_analysis": "shipped",
    "notifications": "shipped",
    # The swarm shipped with 2.0 (avaloka coordinate; see Capabilities).
    "swarm_intelligence": "shipped",
    "ml_training": "shipped",
    "ml_inference": "shipped",
    # Owner ruling 2026-11-02: Enterprise is entitled to SSO and the audit log,
    # and neither has shipped on this line.
    "sso_audit": "coming",
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
    auto_insights=True,
    ray_distributed=True,
    entire_dataset=True,
    scheduler=True,
    batch_jobs=True,
    reports=True,
    # Model training and inference run locally on the operator's own hardware.
    # What open source cannot do is schedule them onto a cloud cluster it did
    # not provision — that is cloud_provisioning + scheduled_delivery, below.
    ml_training=True,
    ml_inference=True,
    swarm_intelligence=True,
    dataset_catalog=True,
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
        swarm_intelligence=True,
        dataset_catalog=True,
        auto_insights=True,
    ),
    "free": Capabilities(
        data_analysis=True,
        export_generated_code=True,
        local_execution=True,
        file_connectors=True,
        # Swarm at the preview limit (swarm_limits): two clones, locally.
        swarm_intelligence=True,
        dataset_catalog=True,
        auto_insights=True,
        # No scheduler and no batch jobs: Avaloka pays for every cycle a free
        # hosted user would schedule. This is the COST line, not a value line.
    ),
    "professional": Capabilities(
        data_analysis=True,
        export_generated_code=True,
        local_execution=True,
        file_connectors=True,
        auto_insights=True,
        ray_distributed=True,
        entire_dataset=True,
        scheduler=True,
        batch_jobs=True,
        scheduled_delivery=True,
        reports=True,
        # The defining Professional capability: reach the customer's own cloud
        # and databases, on the customer's own infrastructure and billing.
        database_connectors=True,
        cloud_connectors=True,
        cloud_provisioning=True,
        # Teams (members on Professional or higher) and collaborator invites,
        # including inviting Free users onto your own resources (referrals).
        team_collaboration=True,
        # Owner ruling 2026-11-01: sharing analyses, commenting on them and
        # notifications are Professional and Enterprise. Free shares reports
        # by email and public link only, which every plan has and no flag gates.
        share_analysis=True,
        comment_on_analysis=True,
        notifications=True,
        swarm_intelligence=True,
        dataset_catalog=True,
        ml_training=True,
        ml_inference=True,
    ),
    "enterprise": Capabilities(
        data_analysis=True,
        export_generated_code=True,
        local_execution=True,
        file_connectors=True,
        auto_insights=True,
        ray_distributed=True,
        entire_dataset=True,
        scheduler=True,
        batch_jobs=True,
        scheduled_delivery=True,
        reports=True,
        database_connectors=True,
        cloud_connectors=True,
        cloud_provisioning=True,
        # The defining Enterprise capabilities: a company working as an
        # organisation -- its structure, and teams of any of its members.
        team_collaboration=True,
        org_structure=True,
        share_analysis=True,
        comment_on_analysis=True,
        notifications=True,
        swarm_intelligence=True,
        dataset_catalog=True,
        shared_catalog=True,
        ml_training=True,
        ml_inference=True,
        sso_audit=True,
    ),
}


def _declared(value: Optional[str]) -> DeploymentState:
    candidate = (value or "").strip().lower().replace("-", "_")
    if candidate == "avaloka_hosted":
        return "hosted"
    if candidate == "self_hosted":
        return "self_hosted"
    return "undeclared"


_undeclared_warned = False
_undeclared_lock = threading.Lock()


def _warn_undeclared(value: Optional[str]) -> None:
    global _undeclared_warned
    with _undeclared_lock:
        if _undeclared_warned:
            return
        _undeclared_warned = True
    # Only whether the variable is set: its value is operator configuration, and
    # nothing about any user or their data belongs in this line.
    how = "not set" if not (value or "").strip() else "set to a value this build does not know"
    logger.warning(
        "[deployment] %s is %s, so this deployment is undeclared; set it to self_hosted or "
        "avaloka_hosted. Until it is, each check keeps its own default: billing and plan "
        "limits treat this deployment as hosted, while editions, local auth and the API "
        "docs treat it as self-hosted.",
        DEPLOYMENT_ENV, how,
    )


def deployment_state(env: Optional[Mapping[str, str]] = None) -> DeploymentState:
    """The ONE reading of ``AVALOKA_DEPLOYMENT``: hosted, self_hosted or undeclared.

    Every hosted/self-hosted decision reads this (billing in
    app/core/plan_limits.py, the API docs and whoami gate in app/api/server.py,
    local auth). ``undeclared`` is not a third kind of deployment: each caller
    keeps the default it had before the value could be declared, and the first
    undeclared reading in a process logs one warning saying so.
    """
    raw = (os.environ if env is None else env).get(DEPLOYMENT_ENV)
    state = _declared(raw)
    if state == "undeclared":
        _warn_undeclared(raw)
    return state


def normalize_deployment(value: str | None) -> Deployment:
    """Unknown values resolve to ``self_hosted``.

    Defaulting to self-hosted is the safe direction: it never grants a hosted
    entitlement, and it never charges anyone for compute they did not ask for.
    Parsed exactly as deployment_state() parses the variable.
    """
    return "avaloka_hosted" if _declared(value) == "hosted" else "self_hosted"


def deployment_from_env() -> Deployment:
    """This process's deployment, from ``AVALOKA_DEPLOYMENT``.

    Read on every call (not at import), so a test or an operator that sets the
    variable later is honoured. Unset or unknown (undeclared) means
    ``self_hosted``; a hosted service sets ``AVALOKA_DEPLOYMENT=avaloka_hosted``.
    """
    return "avaloka_hosted" if declared_deployment() == "hosted" else "self_hosted"


def declared_deployment() -> DeclaredDeployment:
    """1.6's name for :func:`deployment_state`: ``hosted`` (``avaloka_hosted``),
    ``self_hosted``, or ``undeclared`` when the variable is unset or unrecognised.

    The plan gates (plan_gate.check_feature), the billing switch
    (plan_limits._billing_disabled) and the plan-lookup fail mode read it. One
    reading, one warning per process, and the warning never repeats the value.
    """
    return deployment_state()


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
    # Before anything else: a capability that has not shipped is not available,
    # even to a plan entitled to it, so no branch below may say it is.
    if CAPABILITY_STATUS[capability] == "coming":
        return f"{capability} is coming to Avaloka {_sold_on(capability)}; it is not available yet."
    if getattr(capabilities, capability):
        return f"{capability} is available"
    reason = GATE_REASON[capability]
    if reason is GateReason.COST:
        return (
            f"{capability} is not included in this hosted plan because Avaloka "
            "provides the compute. Self-host, or upgrade, to enable it."
        )
    if reason is GateReason.COMMERCIAL:
        return f"{capability} is available in Avaloka {_sold_on(capability)}."
    return f"{capability} is available"


def _sold_on(capability: str) -> str:
    """The paid editions that grant ``capability``, e.g. "Professional and Enterprise"."""
    paid = [e.title() for e in ("professional", "enterprise") if getattr(_HOSTED[e], capability)]
    return " and ".join(paid) or "Enterprise"


def swarm_limits(edition: str | None = None, *, has_commercial_build: bool | None = None):
    """How wide a swarm may fan out for ``edition`` in this build.

    The table lives in :mod:`avaloka.swarm_limits` so the standalone CLI, which
    never imports ``app``, enforces the same numbers the API reports. Like
    :func:`resolve_capabilities`, it follows installed code: without the
    commercial package every edition gets the open-source preview.
    """
    from avaloka.swarm_limits import limits_for

    commercial = commercial_build() if has_commercial_build is None else has_commercial_build
    return limits_for(edition, has_commercial_build=commercial)


#: Editions whose callers may open the swarm console (the live /swarm page and
#: the swarm/trace routes). Everyone else gets the chat summary.
SWARM_CONSOLE_EDITIONS: tuple[Edition, ...] = ("professional", "enterprise")


def swarm_console_gate(edition: str | None) -> GateReason:
    """Why the swarm console is, or is not, open to ``edition``.

    OPEN on Professional and Enterprise, COMMERCIAL everywhere else — including
    an ``oss`` install, where the chat carries the mission summary instead.
    Unlike :func:`resolve_capabilities` this follows the edition alone, not the
    installed code: the console's code *is* in the published tree, and what the
    gate protects is the live view, not an import.
    """
    return GateReason.OPEN if normalize_edition(edition) in SWARM_CONSOLE_EDITIONS else GateReason.COMMERCIAL


def explain_swarm_console(edition: str | None) -> str:
    """The one line a gated caller reads. Never nags an edition that has it."""
    if swarm_console_gate(edition) is GateReason.OPEN:
        return "The live swarm console is available on this edition."
    return ("The live swarm console is available in Avaloka Professional and Enterprise. "
            "Your swarm still runs here, and Avaloka tells you what the team did in the chat.")
