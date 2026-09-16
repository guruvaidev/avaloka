"""The canonical DataMission contract.

This is a *superset* schema: a single mission shape that covers both the
existing ETL code-generation work and the data-science / ML missions in the
product vision (analysis, model training, inference, deployment). Interfaces
never embed analytical decision logic — they translate intent into a
DataMission, and the shared engine decides how to execute it.

Modelled on the vision's `apiVersion: avaloka.ai/v1 / kind: DataMission` YAML,
typed with Pydantic v2.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


API_VERSION = "avaloka.ai/v1"
KIND = "DataMission"


class GoalType(str, Enum):
    """What the user is ultimately trying to produce.

    ``etl`` keeps the existing code-generation product working unchanged; the
    remaining values are the data-science / ML missions layered on top.
    """

    ETL = "etl"
    DATA_QUALITY = "data_quality"
    ANALYSIS = "analysis"
    PREDICTIVE_MODEL = "predictive_model"
    INFERENCE = "inference"


class ExecutionMode(str, Enum):
    """How the work is executed. ``AUTO`` defers the choice to the router."""

    AUTO = "auto"
    INTERACTIVE = "interactive"  # full analysis, fits latency/memory budget
    SAMPLED = "sampled"          # statistically defensible preview
    HYBRID = "hybrid"            # sampled preview, then approved full pass
    BATCH = "batch"             # distributed full scan / training


class FullDataEngine(str, Enum):
    """Where a full-data / batch pass runs when one is needed."""

    LOCAL = "local"
    RAY = "ray"


class SamplingStrategy(str, Enum):
    AUTO = "auto"
    NONE = "none"
    RANDOM = "random"
    STRATIFIED = "stratified"
    TEMPORAL = "temporal"


class ValidationLevel(str, Enum):
    NONE = "none"
    RESEARCH = "research"
    DEPLOYMENT = "deployment"


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str = Field(..., description="Dataset location, e.g. gs://… s3://… file://…")
    format: Optional[str] = Field(None, description="parquet, csv, json, delta, …")


class Goal(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: GoalType
    description: str = ""
    # Data-science / ML fields (ignored for plain ETL goals).
    target: Optional[str] = None
    primary_metric: Optional[str] = Field(None, alias="primaryMetric")


class Execution(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: ExecutionMode = ExecutionMode.AUTO
    interactive_latency_seconds: Optional[int] = Field(60, alias="interactiveLatencySeconds")
    full_data_engine: FullDataEngine = Field(FullDataEngine.LOCAL, alias="fullDataEngine")
    kubernetes_context: Optional[str] = Field(None, alias="kubernetesContext")
    # cloud_target lets a mission pin / record which cloud it ran against so
    # the router and agents can stay environment-aware (gcp|aws|azure|local).
    cloud_target: Optional[str] = Field(None, alias="cloudTarget")
    max_cost_usd: Optional[float] = Field(None, alias="maxCostUsd")
    max_runtime_minutes: Optional[int] = Field(None, alias="maxRuntimeMinutes")


class Sampling(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    strategy: SamplingStrategy = SamplingStrategy.AUTO
    preserve: List[str] = Field(default_factory=list)
    sample_size: Optional[int] = Field(None, alias="sampleSize")
    confidence: Optional[float] = None
    margin_error: Optional[float] = Field(None, alias="marginError")


class Validation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    level: ValidationLevel = ValidationLevel.RESEARCH
    detect_leakage: bool = Field(False, alias="detectLeakage")
    subgroup_stability: bool = Field(False, alias="subgroupStability")
    temporal_validation: bool = Field(False, alias="temporalValidation")


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    report: bool = True
    notebook: bool = False
    model: bool = False
    register_in_mlflow: bool = Field(False, alias="registerInMlflow")
    deployment_package: bool = Field(False, alias="deploymentPackage")
    # ETL output sink (kept so ETL missions are first-class in the superset).
    location: Optional[str] = None


class MissionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class MissionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    goal: Goal
    execution: Execution = Field(default_factory=Execution)
    sampling: Sampling = Field(default_factory=Sampling)
    validation: Validation = Field(default_factory=Validation)
    output: Output = Field(default_factory=Output)


class DataMission(BaseModel):
    """The one object every interface compiles to and the engine executes."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    api_version: str = Field(API_VERSION, alias="apiVersion")
    kind: str = KIND
    metadata: MissionMetadata
    spec: MissionSpec

    def canonical(self) -> dict:
        """Stable, alias-keyed dict used for the cross-interface invariant.

        Two interfaces that received equivalent intent must produce byte-equal
        ``canonical()`` output. Excludes nothing and uses field aliases so the
        result matches the vision's YAML keys.
        """
        return self.model_dump(by_alias=True, mode="json")
