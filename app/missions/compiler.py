"""Compile interface-agnostic *intent* into a canonical DataMission.

Every interface (CLI, conversation, REST, MCP) collects the same flat
``intent`` dict and calls :func:`compile_mission`. That shared call is what
guarantees the invariant ``cli_mission == mcp_mission``: identical intent in,
byte-identical DataMission out. Interfaces hold no analytical decision logic;
they only translate their surface (flags, tool args, NL slots) into this dict.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from app.missions.schema import (
    DataMission,
    Execution,
    ExecutionMode,
    FullDataEngine,
    Goal,
    GoalType,
    MissionMetadata,
    MissionSpec,
    Output,
    Sampling,
    SamplingStrategy,
    Source,
    Validation,
    ValidationLevel,
)


def _slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return text or "mission"


def _derive_name(intent: Dict[str, Any]) -> str:
    if intent.get("name"):
        return _slug(str(intent["name"]))
    # Derive a stable name from the goal or the source basename so that
    # equivalent intent yields an equivalent mission name across interfaces.
    if intent.get("goal_description"):
        return _slug(str(intent["goal_description"]))[:48]
    uri = str(intent.get("source_uri", "dataset"))
    base = uri.rstrip("/").split("/")[-1].split(".")[0]
    return _slug(base or "dataset")


def _enum(value: Optional[str], enum_cls, default):
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    return enum_cls(str(value).lower())


def compile_mission(intent: Dict[str, Any]) -> DataMission:
    """Build a DataMission from a flat intent dict.

    Required: ``source_uri``. Everything else has a defensible default so that
    a one-line ``avaloka analyze data.parquet`` and the equivalent MCP call
    both compile to a complete, runnable mission.
    """
    if not intent.get("source_uri"):
        raise ValueError("intent must include 'source_uri'")

    goal = Goal(
        type=_enum(intent.get("goal_type"), GoalType, GoalType.ANALYSIS),
        description=str(intent.get("goal_description", "")),
        target=intent.get("target"),
        primary_metric=intent.get("primary_metric"),
    )

    execution = Execution(
        mode=_enum(intent.get("mode"), ExecutionMode, ExecutionMode.AUTO),
        interactive_latency_seconds=intent.get("interactive_latency_seconds", 60),
        full_data_engine=_enum(
            intent.get("full_data_engine"), FullDataEngine, FullDataEngine.LOCAL
        ),
        kubernetes_context=intent.get("kubernetes_context"),
        cloud_target=intent.get("cloud_target"),
        max_cost_usd=intent.get("max_cost_usd"),
        max_runtime_minutes=intent.get("max_runtime_minutes"),
    )

    sampling = Sampling(
        strategy=_enum(intent.get("sampling_strategy"), SamplingStrategy, SamplingStrategy.AUTO),
        preserve=list(intent.get("preserve", []) or []),
        sample_size=intent.get("sample_size"),
        confidence=intent.get("confidence"),
        margin_error=intent.get("margin_error"),
    )

    validation = Validation(
        level=_enum(intent.get("validation_level"), ValidationLevel, ValidationLevel.RESEARCH),
        detect_leakage=bool(intent.get("detect_leakage", False)),
        subgroup_stability=bool(intent.get("subgroup_stability", False)),
        temporal_validation=bool(intent.get("temporal_validation", False)),
    )

    output = Output(
        report=bool(intent.get("report", True)),
        notebook=bool(intent.get("notebook", False)),
        model=bool(intent.get("model", goal.type == GoalType.PREDICTIVE_MODEL)),
        register_in_mlflow=bool(intent.get("register_in_mlflow", False)),
        deployment_package=bool(intent.get("deployment_package", False)),
        location=intent.get("output_location"),
    )

    spec = MissionSpec(
        source=Source(uri=str(intent["source_uri"]), format=intent.get("format")),
        goal=goal,
        execution=execution,
        sampling=sampling,
        validation=validation,
        output=output,
    )

    return DataMission(metadata=MissionMetadata(name=_derive_name(intent)), spec=spec)
