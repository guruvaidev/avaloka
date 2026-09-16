from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from app.agents.mta_v2.schema import TrainingPlan, TrainingResult


# Local default so we don't depend on config.py
DEFAULT_ASSISTANT_ID = "avaloka"

class ChatResponse(BaseModel):
    messages: List[Dict[str, str]]
    planner_definition: Dict[str, Any] = {}
    ready_to_summarize: bool = False
    ready_to_code: bool = False
    coder_definition: Dict[str, Any] = {}
    task_info: Optional[Dict[str, Any]] = None
    output_file_data: Optional[Dict[str, Any]] = None
    output_json: Optional[List[Dict[str, Any]]] = None
    # Planner deliberation trace (ChatGPT-style "thinking") — present when
    # the planner model ran with reasoning_format="parsed".
    reasoning: Optional[str] = None
    planner_graph_path: Optional[str] = None
    planner_graph_status: Optional[str] = None
    planner_graph_display_url: Optional[str] = None
    visualization_config: Optional[Dict[str, Any]] = None
    visualization_status: Optional[str] = None
    # Model Training Agent fields
    training_scheduled: Optional[bool] = False                      # v1.2
    inference_scheduled: Optional[bool] = False                     # v1.4
    configure_inference_service_scheduled: Optional[bool] = False   # v1.4
    stop_inference_service_scheduled: Optional[bool] = False        # v1.4
    training_plan: Optional[TrainingPlan] = None                    # v1.2
    training_result: Optional[TrainingResult] = None                # v1.2
    training_task: Optional[Dict[str, Any]] = None
    training_metrics: Optional[Dict[str, Any]] = None
    model_artifacts: Optional[Dict[str, Any]] = None
    mlflow_run_id: Optional[str] = None
    training_completed: Optional[bool] = False
    ready_to_train: Optional[bool] = False
    training_status: Optional[str] = None
    dataset_size_bytes: Optional[int] = None

    # NEW: dropdown + multi-dataset support
    datasets: List[Dict[str, Any]] = Field(default_factory=list)
    active_dataset_ids: List[str] = Field(default_factory=list)
    visualization_configs: Dict[str, Any] = Field(default_factory=dict)
    visualization_statuses: Dict[str, str] = Field(default_factory=dict)
    analysis_fidelity: Optional[str] = None
    selected_sample_name: Optional[str] = None
    analysis_task_id: Optional[str] = None
    execution_context: Optional[Dict[str, Any]] = None
    
    # Context Memory diagnostics
    memory_hints: Optional[List[Dict[str, Any]]] = None
    prior_artifact_found: Optional[bool] = None
    session_logic_signature: Optional[str] = None
    memory_context_unavailable: Optional[bool] = None

    # Evidence layer (v1.6). Purely additive: every field is Optional with a
    # None default, so existing UI clients that do not know about them are
    # unaffected and no existing field changes shape.
    integrity_report: Optional[Dict[str, Any]] = None
    integrity_safe_to_train: Optional[bool] = None
    evaluation_report: Optional[Dict[str, Any]] = None
    evaluation_beats_baseline: Optional[bool] = None
    verification_report: Optional[Dict[str, Any]] = None
    verification_safe_to_present: Optional[bool] = None
    agent_errors: Optional[List[Dict[str, Any]]] = None


class UploadResponse(BaseModel):
    dataset_id: str
    session_id: str
    thread_id: str
    schema_: Union[Dict[str, Any], List[str]] = Field(..., alias="schema")
    samples: List[Dict[str, Any]]
    ddl_schema: str
    rows_sampled: int
    visualization_config: Optional[VisualizationConfig] = None
    visualization_status: Optional[str] = None
    # dataset size for fidelity / UI
    file_size_bytes: int = 0
    file_size_mb: float = 0.0
    # Sampler state (Phase A on upload; Phase B via /preview after polling)
    sample_status: Optional[str] = None
    quick_sample_rows: Optional[List[Dict[str, Any]]] = None
    sample_statistics: Optional[Dict[str, Any]] = None
    estimated_rows: Optional[int] = None
    # Profiler state (Phase A on upload; Phase B via /preview after polling)
    profiling_status: Optional[str] = None
    quick_profiling_result: Optional[Dict[str, Any]] = None
    profiling_result: Optional[Dict[str, Any]] = None
    # Portfolio: keyed by sample name, each value is a list of row-dicts
    portfolio_samples: Optional[Dict[str, List[Dict[str, Any]]]] = None
    available_samples: Optional[List[str]] = None
    background_task_id: Optional[str] = None
    analysis_fidelity: Optional[str] = None
    selected_sample_name: Optional[str] = None
    requires_fidelity_choice: Optional[bool] = None
    fidelity_prompt: Optional[str] = None
    estimated_runtime_hint: Optional[str] = None
    # Set when a single uploaded file expanded into several datasets (one per
    # Excel sheet). Top-level fields describe the primary (largest) sheet.
    datasets: Optional[List["UploadedDatasetOut"]] = None

    model_config = {"populate_by_name": True}

    @property
    def schema(self) -> Union[Dict[str, Any], List[str]]:
        """Property to access schema field (avoids shadowing BaseModel.schema)"""
        return self.schema_

class DatasetOption(BaseModel):
    dataset_id: str
    filename: Optional[str] = None
    alias: Optional[str] = None


class UploadedDatasetOut(BaseModel):
    dataset_id: str
    filename: str
    alias: str
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    visualization_config: Optional[Dict[str, Any]] = None
    visualization_status: Optional[str] = None
    # Sampler + profiler state (same as UploadResponse)
    sample_status: Optional[str] = None
    quick_sample_rows: Optional[List[Dict[str, Any]]] = None
    sample_statistics: Optional[Dict[str, Any]] = None
    profiling_status: Optional[str] = None
    quick_profiling_result: Optional[Dict[str, Any]] = None
    profiling_result: Optional[Dict[str, Any]] = None
    # Portfolio: keyed by sample name
    portfolio_samples: Optional[Dict[str, List[Dict[str, Any]]]] = None
    available_samples: Optional[List[str]] = None
    analysis_fidelity: Optional[str] = None
    selected_sample_name: Optional[str] = None
    # Excel-sheet provenance: set when this dataset came from one sheet of a
    # multi-sheet workbook upload.
    sheet_name: Optional[str] = None
    source_workbook: Optional[str] = None
    notes: Optional[List[str]] = None


class MultiUploadResponse(BaseModel):
    session_id: str
    thread_id: str
    datasets: List[UploadedDatasetOut] = Field(default_factory=list)


class ThreadCreateIn(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)

 
    
class ThreadOut(BaseModel):
    thread_id: str
    created_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # we keep a default but it's now local, not imported
    assistant_id: Optional[str] = DEFAULT_ASSISTANT_ID


class MessageCreateIn(BaseModel):
    role: str = Field("user", description="user | assistant | system")
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    stream: bool = Field(False, description="If true, respond with SSE")

    # allow selecting which datasets are active (order matters: first = primary)
    dataset_ids: Optional[List[str]] = None
    analysis_fidelity: Optional[str] = None
    selected_sample_name: Optional[str] = None


# ---- dataset models ----

class DatasetListItem(BaseModel):
    dataset_id: str
    created_at: str
    size_bytes: int
    columns: List[str] = Field(default_factory=list)
    # Multi-upload grouping + labelling: datasets uploaded together share
    # group_session_id, so the UI can collapse them into one "X + N more" card
    # and title each by filename instead of the raw dataset_id UUID.
    session_id: Optional[str] = None
    group_session_id: Optional[str] = None
    filename: Optional[str] = None
    alias: Optional[str] = None
    source_kind: Optional[str] = None


class DatasetPreviewResponse(BaseModel):
    dataset_id: str
    schema_: Union[List[str], Dict[str, str]] = Field(..., alias="schema")
    samples: List[Dict[str, Any]]
    ddl_schema: str
    rows_sampled: int
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    # Kept for backward compat
    profile_status: Optional[str] = None
    total_rows_estimated: Optional[int] = None
    sample_pct: Optional[float] = None
    profiling_result: Optional[Dict[str, Any]] = None
    # Sampler state
    sample_status: Optional[str] = None
    quick_sample_rows: Optional[List[Dict[str, Any]]] = None
    full_sample_rows: Optional[List[Dict[str, Any]]] = None
    sample_statistics: Optional[Dict[str, Any]] = None
    total_rows_exact: Optional[int] = None
    # Profiler state
    profiling_status: Optional[str] = None
    quick_profiling_result: Optional[Dict[str, Any]] = None
    full_profiling_result: Optional[Dict[str, Any]] = None
    # Portfolio: keyed by sample name
    portfolio_samples: Optional[Dict[str, List[Dict[str, Any]]]] = None
    available_samples: Optional[List[str]] = None
    background_task_id: Optional[str] = None
    analysis_fidelity: Optional[str] = None
    selected_sample_name: Optional[str] = None
    analysis_task_id: Optional[str] = None
    requires_fidelity_choice: Optional[bool] = None
    fidelity_prompt: Optional[str] = None
    estimated_runtime_hint: Optional[str] = None
    # carry the upload-time auto-viz so open-for-analysis renders the
    # same 4-chart config as a fresh upload instead of a 2-chart fallback.
    visualization_config: Optional[Dict[str, Any]] = None
    visualization_status: Optional[str] = None

    model_config = {"populate_by_name": True}

    @property
    def schema(self) -> Union[List[str], Dict[str, str]]:
        """Property to access schema field (avoids shadowing BaseModel.schema)"""
        return self.schema_


class ProfileStatusResponse(BaseModel):
    """Lightweight status response for polling the background refinement progress."""
    dataset_id: str
    status: str                                # overall: quick_sample | refining | full_profile | error
    sample_status: Optional[str] = None        # sampler-specific status
    profiling_status: Optional[str] = None     # profiler-specific status
    total_rows_estimated: Optional[int] = None
    total_rows_exact: Optional[int] = None
    sampled_rows: Optional[int] = None
    sample_pct: Optional[float] = None
    error: Optional[str] = None


DatasetPreviewResponse.model_rebuild()


# ---- register existing storage ----

class RegisterExistingIn(BaseModel):
    storage_uri: str  # ex: "s3://avaloka-s3-testing-bucket/test-input-data/"
    key: str          # ex: "07860d80-b0f0-4d20-a40d-e3f0baa80370.csv"
    connection_id: str        # Supabase id
    schema_json_: str | None = Field(None, alias="schema_json")
    
    model_config = {"populate_by_name": True}
    
    @property
    def schema_json(self) -> str | None:
        """Property to access schema_json field (avoids shadowing BaseModel.schema_json)"""
        return self.schema_json_


# ---- bucket listing ----

# class BucketListResponse(BaseModel):
#     backend: str
#     bucket: str
#     prefix: str
#     # [{ "key": "...", "size": "...", "updated": "..." }]
#     objects: List[Dict[str, str]]


# ---- database chat ----

class DatabaseConnectRequest(BaseModel):
    # Sent in the request body (not query params) so the api_key never lands in
    # access logs.
    customer_id: str
    api_key: str


class DatabaseChatRequest(BaseModel):
    role: str = Field("user", description="user | assistant | system")
    content: str
    dataset_id: Optional[str] = None
    database_id: Optional[str] = None
    customer_id: Optional[str] = None
    session_id: Optional[str] = Field(
        None, description="session_id from /api/database/connect; reuses stored credentials"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    stream: bool = Field(False, description="If true, respond with SSE")


class DatabaseChatResponse(BaseModel):
    messages: List[Dict[str, str]]
    query_result: Optional[Dict[str, Any]] = None
    database_info: Optional[Dict[str, Any]] = None
    planner_definition: Dict[str, Any] = {}
    ready_to_summarize: bool = False
    ready_to_code: bool = False
    coder_definition: Dict[str, Any] = {}
    output_file_data: Optional[Dict[str, Any]] = None
    output_json: Optional[List[Dict[str, Any]]] = None
    dataset_id: Optional[str] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    visualization_config: Optional[Dict[str, Any]] = None
    visualization_status: Optional[str] = None

class VisualizationConfig(BaseModel):
    """
    High-level schema for the visualization agent output.

    We keep inner parts flexible (Dict[str, Any]) to avoid
    breaking changes while still typing the top-level.
    """
    version: Optional[str] = None
    dataset: Optional[Dict[str, Any]] = None
    task: Optional[Dict[str, Any]] = None
    columns: Optional[List[Dict[str, Any]]] = None
    feature_ranking: Optional[Dict[str, Any]] = None
    charts: Optional[List[Dict[str, Any]]] = None
    model_diagnostics: Optional[Dict[str, Any]] = None
    bias_diagnostics: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None


# UploadResponse forward-references UploadedDatasetOut and VisualizationConfig
# (defined later in the file), so resolve it now that all classes exist.
UploadResponse.model_rebuild()


# ─────────────────────────────────────────────────────────────────────────────
# WBS 0.6 v3.0 — Asset Persistence retrieval models
# ─────────────────────────────────────────────────────────────────────────────


class AssetEntry(BaseModel):
    """Single versioned asset entry — one per prompt execution."""
    object_key: str
    prompt_ts:  str   # e.g. "20260407_143022" — timestamp of the prompt that generated this


class CodeAsset(BaseModel):
    """Most-recent generated .py script + full history of all versions."""
    available:    bool                    = False
    object_key:   Optional[str]           = None   # most recent
    signed_url:   Optional[str]           = None   # 15-min download URL for most recent
    history:      List[AssetEntry]        = Field(default_factory=list)  # all versions


class OutputAsset(BaseModel):
    """Most-recent execution output CSV + full history."""
    available:    bool                    = False
    object_key:   Optional[str]           = None
    signed_url:   Optional[str]           = None
    history:      List[AssetEntry]        = Field(default_factory=list)


class VizAsset(BaseModel):
    """Most-recent persisted visualization config JSON."""
    available:    bool                    = False
    object_key:   Optional[str]           = None
    signed_url:   Optional[str]           = None
    history:      List[AssetEntry]        = Field(default_factory=list)


class JobAsset(BaseModel):
    """Git job-definition reference (written by planner tool call)."""
    available:    bool                    = False
    git_branch:   Optional[str]           = None
    git_repo:     Optional[str]           = None


class PersistError(BaseModel):
    """Errors from the last background persist cycle, surfaced to the UI."""
    errors:       List[str]               = Field(default_factory=list)


class AssetResponse(BaseModel):
    """
    Full asset manifest for a session.
    Returned by GET /api/assets/{session_id}.
    """
    session_id:    str
    dataset_id:    Optional[str]          = None
    code_asset:    CodeAsset              = Field(default_factory=CodeAsset)
    output_asset:  OutputAsset            = Field(default_factory=OutputAsset)
    viz_asset:     VizAsset               = Field(default_factory=VizAsset)
    job_asset:     JobAsset               = Field(default_factory=JobAsset)
    persist_errors: List[str]             = Field(default_factory=list)



class InsightRewriteIn(BaseModel):
    session_id: Optional[str] = None
    insight_id: Optional[str] = None
    rewrite_type: str = "regenerate"
    instructions: Optional[str] = None

class InsightRewriteOut(BaseModel):
    analysis_id: str
    insight_id: str
    rewrite_type: str
    insights: Any
    version: int
    created_at: str


# app/api/schemas.py
class InsightFeedbackIn(BaseModel):
    insight_id: Optional[str] = None
    feedback_type: str  # 'positive' | 'negative'
    comment: Optional[str] = None

class InsightFeedbackOut(BaseModel):
    id: str
    analysis_id: str
    insight_id: Optional[str] = None
    feedback_type: str
    comment: Optional[str] = None
    created_at: str


class BucketFolder(BaseModel):
    name: str        # display label, e.g. "2024"
    prefix: str      # pass back as ?prefix= to navigate in, e.g. "sales/2024/"


class BucketListResponse(BaseModel):
    backend: str
    bucket: str
    prefix: str                      # the level currently being viewed
    folders: List[BucketFolder] = Field(default_factory=list)
    # [{ "key": "...", "size": "...", "updated": "..." }]
    objects: List[Dict[str, str]]
