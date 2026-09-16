"""Avaloka execution brain: workload estimation and execution routing.

The estimator scores a mission's workload and cost; the router turns that
score plus the mission's budget/latency constraints into a concrete plan.
All interfaces share this layer — none of them decides execution themselves.
"""

from app.execution.estimator import DatasetStats, WorkloadEstimate, estimate_workload
from app.execution.environment import (
    CloudEnvironment,
    CloudProvider,
    detect_environment,
    provider_for_uri,
    resolve_cloud_target,
)
from app.execution.router import ExecutionPlan, route

__all__ = [
    "DatasetStats",
    "WorkloadEstimate",
    "estimate_workload",
    "ExecutionPlan",
    "route",
    "CloudEnvironment",
    "CloudProvider",
    "detect_environment",
    "provider_for_uri",
    "resolve_cloud_target",
]
