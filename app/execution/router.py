"""Execution router — turns a mission + workload estimate into a concrete plan.

This is the single place that decides *how* a mission runs, so that the CLI,
web, REST and MCP interfaces never do. v1 fully supports the ``local`` target
and declares the ``ray`` / KubeRay batch path as a planned-but-stubbed branch
(it produces a plan and flags it ``not_implemented`` rather than pretending to
run). Approval gates from the vision (sampled→full, over-budget, write ops) are
surfaced as ``requires_approval``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.execution.estimator import WorkloadEstimate, estimate_workload, DatasetStats
from app.execution.environment import CloudEnvironment, resolve_cloud_target
from app.missions.schema import DataMission, ExecutionMode, FullDataEngine


@dataclass
class ExecutionPlan:
    mission_name: str
    mode: ExecutionMode
    target: str                 # "local" | "ray"
    cloud_target: str            # gcp | aws | azure | local (always resolved)
    cloud_source: str            # how cloud was decided: mission|source_uri|env|...
    cloud_knowledge: dict        # service catalog the agents reason with
    agent_context: str           # cloud-specific system-prompt fragment
    steps: List[str]
    requires_approval: bool
    approval_reason: Optional[str]
    estimate: WorkloadEstimate
    implemented: bool
    notes: List[str] = field(default_factory=list)


def _steps_for(mode: ExecutionMode, mission: DataMission) -> List[str]:
    goal = mission.spec.goal.type.value
    base = ["Inspect schema and data contract", "Profile distributions and missingness"]
    if mission.spec.validation.detect_leakage:
        base.append("Detect target leakage")
    if mode in (ExecutionMode.SAMPLED, ExecutionMode.HYBRID):
        base.append("Build representative sample")
    base.append(f"Run {goal} on " + ("sample" if mode is ExecutionMode.SAMPLED else "data"))
    if mode in (ExecutionMode.HYBRID, ExecutionMode.BATCH):
        base.append("Confirm result on full data (distributed)")
    if mission.spec.output.report:
        base.append("Generate report")
    if mission.spec.output.deployment_package:
        base.append("Package deployment artifacts")
    return base


def route(
    mission: DataMission,
    stats: Optional[DatasetStats] = None,
    estimate: Optional[WorkloadEstimate] = None,
    environment: Optional[CloudEnvironment] = None,
) -> ExecutionPlan:
    estimate = estimate or estimate_workload(mission, stats)
    mode = estimate.recommended_mode
    ex = mission.spec.execution

    # Environment-awareness: resolve the cloud this mission runs against once —
    # an explicit cloud_target wins, else the data URI's scheme, else the
    # detected runtime environment. The router, estimator and agents all consume
    # this single resolution rather than branching on platform strings.
    cloud = resolve_cloud_target(
        pinned=ex.cloud_target,
        source_uri=mission.spec.source.uri,
        env=environment,
    )

    # target engine: a full/hybrid/batch pass uses the configured full-data
    # engine; pure interactive/sampled work stays local.
    if mode in (ExecutionMode.BATCH, ExecutionMode.HYBRID):
        target = ex.full_data_engine.value
    else:
        target = FullDataEngine.LOCAL.value

    # approval gates (vision §16): a full distributed pass beyond budget, or any
    # batch that crosses the sampled→full boundary, needs explicit sign-off.
    requires_approval = False
    approval_reason = None
    if mode in (ExecutionMode.HYBRID, ExecutionMode.BATCH):
        requires_approval = True
        approval_reason = "Full distributed pass — confirm runtime/cost before launching."
    if ex.max_cost_usd is not None and estimate.full_cost_usd[1] > ex.max_cost_usd \
            and mode in (ExecutionMode.HYBRID, ExecutionMode.BATCH):
        approval_reason = (
            f"Estimated full-pass cost ${estimate.full_cost_usd[1]:.2f} exceeds "
            f"budget ${ex.max_cost_usd:.2f}."
        )

    notes: List[str] = []
    implemented = target == FullDataEngine.LOCAL.value
    if not implemented:
        notes.append(
            f"Target '{target}' (Ray/KubeRay) is declared but not yet wired in "
            "this build; plan is produced, execution is stubbed."
        )
    if cloud.is_cloud:
        notes.append(
            f"Environment-aware: reasoning in {cloud.provider.value.upper()} "
            f"primitives (registry {cloud.service_catalog.get('container_registry')}, "
            f"object store {cloud.service_catalog.get('object_store_name')})."
        )

    return ExecutionPlan(
        mission_name=mission.metadata.name,
        mode=mode,
        target=target,
        cloud_target=cloud.provider.value,
        cloud_source=cloud.source,
        cloud_knowledge=dict(cloud.service_catalog),
        agent_context=cloud.system_prompt_fragment(),
        steps=_steps_for(mode, mission),
        requires_approval=requires_approval,
        approval_reason=approval_reason,
        estimate=estimate,
        implemented=implemented,
        notes=notes,
    )
