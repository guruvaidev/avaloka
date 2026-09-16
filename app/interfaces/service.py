"""Shared mission service used by every interface.

The whole point of this module is that the CLI and the MCP server (and later
REST + web) call the *same* functions with the *same* intent dict, so they
cannot drift apart. Interfaces own only their surface parsing; everything from
``intent`` onward lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.execution.estimator import DatasetStats
from app.execution.router import ExecutionPlan, route
from app.missions.compiler import compile_mission
from app.missions.schema import DataMission


@dataclass
class PlannedMission:
    mission: DataMission
    plan: ExecutionPlan


def _stats_from_intent(intent: Dict[str, Any]) -> Optional[DatasetStats]:
    keys = ("rows", "columns", "bytes", "type_complexity", "cardinality_skew", "rare_class_fraction")
    if not any(intent.get(k) is not None for k in keys):
        return None
    return DatasetStats(
        rows=intent.get("rows"),
        columns=intent.get("columns"),
        bytes=intent.get("bytes"),
        type_complexity=float(intent.get("type_complexity", 0.0) or 0.0),
        cardinality_skew=float(intent.get("cardinality_skew", 0.0) or 0.0),
        rare_class_fraction=intent.get("rare_class_fraction"),
    )


def plan_mission(intent: Dict[str, Any]) -> PlannedMission:
    """Intent -> canonical mission -> execution plan. The shared entry point."""
    mission = compile_mission(intent)
    plan = route(mission, stats=_stats_from_intent(intent))
    return PlannedMission(mission=mission, plan=plan)


def planned_to_dict(planned: PlannedMission) -> Dict[str, Any]:
    """One JSON-able representation of a planned mission for every non-CLI surface.

    The MCP server and REST API both return this, so an agent calling
    ``avaloka.plan_mission`` and a script hitting the REST endpoint see the same
    shape — the cross-interface invariant applies to output as well as input.
    """
    plan, est = planned.plan, planned.plan.estimate
    return {
        "mission": planned.mission.canonical(),
        "plan": {
            "mission_name": plan.mission_name,
            "mode": plan.mode.value,
            "target": plan.target,
            "cloud_target": plan.cloud_target,
            "cloud_source": plan.cloud_source,
            "cloud_knowledge": plan.cloud_knowledge,
            "agent_context": plan.agent_context,
            "steps": list(plan.steps),
            "requires_approval": plan.requires_approval,
            "approval_reason": plan.approval_reason,
            "implemented": plan.implemented,
            "notes": list(plan.notes),
        },
        "estimate": {
            "recommended_mode": est.recommended_mode.value,
            "workload_score": est.workload_score,
            "full_cost_usd": list(est.full_cost_usd),
            "full_runtime_minutes": list(est.full_runtime_minutes),
            "sampled_cost_usd": list(est.sampled_cost_usd),
            "sampled_rows": est.sampled_rows,
            "full_pass_worthwhile": est.full_pass_worthwhile,
            "rationale": est.rationale,
            "notes": list(est.notes),
        },
    }
