from typing_extensions import TypedDict
from typing import Annotated, List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
import pandas as pd

from app.agents.mta_v2.schema import TrainingPlan, TrainingResult

import logging
import re

_logger = logging.getLogger(__name__)

# dta_database_registry entries are handles: a UI database connection (_source
# "ui", opened by customer id), a cloud_datasets row (_source "cloud_ui", opened
# by row id) and a connection typed into chat (_source "chat", registered in the
# same stores; app/core/typed_secrets.py). The store is read again when a
# transfer runs, so no credential belongs in the registry, and with it every
# checkpoint. An entry with no _source is a chat registration from a checkpoint
# written before chat connections were sealed; it is scrubbed the same way.
DTA_HANDLE_SOURCES = frozenset({"ui", "cloud_ui", "chat"})
DTA_SECRET_FIELDS = frozenset({
    "password", "access_key", "secret_key", "session_token", "gcp_service_account_json",
    "service_account_json", "az_access_key", "az_sas_token", "az_client_secret",
    "azure_connection_string", "connection_string", "account_key", "sas_token", "sas",
})
# Anything shaped like a credential (2.0's DTA_SECRET_FIELD_HINTS), so a field
# not listed above still cannot carry one.
DTA_SECRET_FIELD_HINTS = ("password", "passwd", "pwd", "secret", "key", "token",
                          "conn_str", "connection_string", "credential", "sas", "private")
_DSN_WITH_PASSWORD = re.compile(r"[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s@/]+@", re.IGNORECASE)


def _is_secret_field(name: Any) -> bool:
    n = str(name).lower()
    return n in DTA_SECRET_FIELDS or any(h in n for h in DTA_SECRET_FIELD_HINTS)


def scrub_registry_entry(entry: Any) -> Any:
    """An entry without its credential fields; the entry itself when it has none.

    Every entry is scrubbed, whatever its source. A secret-named field is
    dropped, and so is a value that embeds ``user:password@``. The dropped
    fields' names are logged, never their values.
    """
    if not isinstance(entry, dict):
        return entry
    dropped = sorted(str(k) for k, v in entry.items()
                     if _is_secret_field(k) or (isinstance(v, str) and _DSN_WITH_PASSWORD.search(v)))
    if not dropped:
        return entry
    _logger.warning("dta_database_registry: dropped credential field(s) %s from a %s entry",
                    dropped, entry.get("_source") or "chat (pre-sealing)")
    return {k: v for k, v in entry.items() if str(k) not in dropped}


def scrub_dta_registry(current: Any, update: Any) -> Optional[dict]:
    """Reducer for ``dta_database_registry``: the update replaces the registry, scrubbed.

    Defence in depth -- the planner writes no credential into a handle entry --
    and the same scrub runs when the planner reads the registry, for a
    checkpoint written before this rule.
    """
    if not isinstance(update, dict):
        return update
    return {alias: scrub_registry_entry(e) for alias, e in update.items()}


def filter_messages(x: list, y: list):
    if not x:
        return y
    if not y:
        return x
    filtered_y = [m for m in y if m not in x]
    if not filtered_y and x[-1] != y[-1]:
        filtered_y.append(y[-1])
    return x + filtered_y

# State type
class ETLState(TypedDict):
    user_id: str
    session_id: str
    thread_id: Optional[str]
    user_prompt: Optional[str]
    messages: Annotated[List[BaseMessage], filter_messages]
    planner_definition: dict
    plan: Optional[str]
    ready_to_summarize: bool
    ready_to_code: bool
    coder_definition: dict
    data_source_location: str
    output_location: str
    schema: dict
    infrastructure_request: dict
    infrastructure_provisioned: dict
    execution_result: dict
    input_data_type: str
    output_file_data: dict
    sample_data: str

    memory_hints: list
    prior_artifact_found: bool
    session_logic_signature: str
    memory_context_unavailable: bool

    # Planner deliberation trace (reasoning_format="parsed") surfaced to the
    # UI as a ChatGPT-style "thinking" panel. Must be declared here or
    # LangGraph drops the key from node returns.
    reasoning_trace: Optional[str]

    # Top-level conversational router. These channels must be declared or
    # LangGraph silently drops the node's routing/context updates.
    connection_event: Optional[dict]
    avaloka_mode: Optional[str]
    avaloka_intent: Optional[str]
    redis_context: Optional[dict]
    discovery_result: Optional[dict]
    delegate_to_planner: Optional[bool]
    avaloka_handoff_message: Optional[str]
    avaloka_recommendation: Optional[dict]
    avaloka_swarm: Optional[list[dict]]
    avaloka_swarm_message: Optional[str]
    default_infra_platform: Optional[str]

    active_mcp_servers: List[str]
    # Tenant-scoped MCP auth token; the MCP maps key->customer, so schema
    # hot-load must use the requesting tenant's key, not a global one.
    mcp_api_key: Optional[str]
    milvus_context_id: str

    # Data upload and preview fields
    deploy_on_k8s: bool
    uploaded_csv_preview: List[List[str]]
    uploaded_csv_columns: List[str]

    # Code generation fields
    generated_code: Optional[str]
    coder_pseudocode: Optional[str]
    coder_contract: Optional[dict]
    contract_error: Optional[bool]
    contract_violations: Optional[list]
    coder_raw_response: Optional[str]
    logical_review_feedback: Optional[str]
    logical_review_decided_by: Optional[str]
    logical_review_judged_correct: Optional[bool]
    logical_review_judged_fabricated: Optional[bool]
    logical_review_recheck_ran: Optional[bool]
    logical_review_recheck_confirmed: Optional[bool]
    logical_review_overruled: Optional[bool]
    multi_action_actions: Optional[List[Dict[str, Any]]]
    multi_action_missing: Optional[List[Dict[str, Any]]]
    multi_action_status: Optional[str]
    multi_action_notes: Optional[List[str]]
    structured_action_plan: Optional[Dict[str, Any]]
    output_tables: Optional[List[Dict[str, Any]]]

    # Execution and output fields
    # Using Any instead of pd.DataFrame to avoid Pydantic schema generation issues
    execution_output_data: Optional[Any]
    execution_output_preview: Optional[Any]
    output_json: Optional[List[Dict[str, Any]]]
    # Written by every execution node next to execution_result. Undeclared,
    # LangGraph dropped all four on the way out of the node (measured on
    # 0.6.11), so in the graph the writes never arrived and the reads of
    # execution_error in route_after_execution and the narrator never fired.
    # tests/execution_agent/test_execution_result_contract.py fails if a node
    # writes a key that is not declared here.
    execution_stdout: Optional[str]
    execution_stderr: Optional[str]
    execution_error: Optional[str]
    fresh_output_produced: Optional[bool]

    # Planner graph visualization fields
    planner_graph_path: Optional[str]
    planner_graph_base64: Optional[str]
    planner_graph_status: Optional[str]
    planner_graph_error: Optional[str]
    visualization_config: Optional[Dict[str, Any]]
    visualization_status: Optional[str]

    # Scheduler fields
    task_list: list[str]
    task_schedule: Optional[dict]
    task_info: Optional[dict]
    task_operation: Optional[dict]
    task_run_statistics: Optional[dict]

    # MTA-specific fields (ML Training)
    training_scheduled: bool                    # v1.2
    inference_scheduled: bool                   # v1.4
    configure_inference_service_scheduled: bool # v1.4
    stop_inference_service_scheduled: bool      # v1.4
    training_plan: Optional[TrainingPlan]       # v1.2
    training_result: Optional[TrainingResult]   # v1.2
    training_task: Optional[dict]           # Task definition for model training
    model_artifacts: Optional[dict]         # Paths to trained models (native + ONNX)
    mlflow_run_id: Optional[str]            # MLflow experiment run ID
    training_metrics: Optional[dict]        # Training and validation metrics
    model_registry: Optional[dict]          # Model type and configuration registry
    ray_config: Optional[dict]              # Ray Train distributed training configuration
    validation_params: Optional[dict]       # Cross-validation and evaluation parameters
    integrity_report: Optional[dict]        # Pre-training data-integrity evidence
    pii_report: Optional[dict]              # Column classifications from the preparation scan
    integrity_safe_to_train: Optional[bool] # Whether integrity checks found blockers
    evaluation_report: Optional[dict]       # Model performance versus a trivial baseline
    evaluation_beats_baseline: Optional[bool]
    # rlhf_feedback: Optional[dict]         # Reinforcement Learning from Human Feedback
    business_recommendations: Optional[dict]  # Business impact analysis and recommendations
    ready_to_train: Optional[bool]          # Flag to indicate readiness for training
    training_completed: Optional[bool]      # Flag to indicate training completion
    training_plan_reply_action: Optional[str]  # confirm | update | modify_dataset | cancel | unclear | none
    clear_training_state: Optional[bool]       # Clear pending training session fields
    pending_clarification: Optional[dict]       # Latest unresolved planner clarification, if any
    pending_suggestions: Optional[list]         # Numbered analysis ideas last offered, so "run 3" resolves

    # Evidence layer. ClaimVerifier is the final graph gate and publishes its
    # deterministic report through these API-facing channels.
    analysis_narrative: Optional[str]
    integrity_report: Optional[dict]
    integrity_safe_to_train: Optional[bool]
    evaluation_report: Optional[dict]
    evaluation_beats_baseline: Optional[bool]
    verification_report: Optional[dict]
    verification_safe_to_present: Optional[bool]
    agent_errors: Optional[list[dict]]

        # Execution mode configuration
    execution_mode: Optional[str]           # "local", "k8s-ray" - determines execution strategy
    enable_training: Optional[bool]         # Flag to enable/disable ML training
    skip_to_training: Optional[bool]

    # Plan gates (set by send_message / planner). Must be declared or LangGraph
    # drops them: user_plan is the requester's plan ("free" | "professional" |
    # "enterprise"); plan_upgrade_required names the paid feature a turn was
    # refused ("data_transfer" | "model_training"), else None.
    user_plan: Optional[str]
    plan_upgrade_required: Optional[str]

    # Planner-side data transfer state (chat-registered DBs, last transfer).
    # Entries from a store hold its handle and display fields, never a
    # credential; the reducer strips any from every write.
    dta_database_registry: Annotated[Optional[dict], scrub_dta_registry]
    dta_injection_script: Optional[str]
    dta_last_transfer: Optional[dict]
    is_dta_request: Optional[bool]
    # The database the user is currently analyzing (customer_id), surfaced from the
    # active DB-sample session's `db://{customer_id}/…` location so a DTA transfer can
    # use it as the IMPLICIT source without the user naming it in the prompt.
    active_db_customer_id: Optional[str]
    # The table the user analyzed (parsed from their query), used as the implicit
    # source table instead of the connection's auto-detected first table.
    active_db_source_table: Optional[str]

    # ── Multi-dataset ─────────────────────────
    multi_dataset_state: list[dict]
    datasets_context: Optional[list[dict]]
    active_dataset_id: Optional[str]
    active_dataset_ids: Optional[list[str]]

    # ── Ray / cloud execution fields ───────────────────
    ray_workers: Optional[int]
    ray_job_name: Optional[str]
    ray_job_namespace: Optional[str]
    ray_job_logs: Optional[str]
    benchmark_metrics: Optional[dict]

    # Cloud connection identifiers
    connection_id: Optional[str]
    cloud_connection_id: Optional[str]
    storage_connection_id: Optional[str]

    # Data source split (cloud vs local path)
    sample_data_location: Optional[str]
    full_data_location: Optional[str]
    data_source_location_cloud: Optional[str]
    data_source_location_local: Optional[str]
    full_data_source_location_cloud: Optional[str]
    active_data_source_location: Optional[str]
    active_data_source_location_cloud: Optional[str]
    active_data_source_location_local: Optional[str]
    activate_output_as_dataset: Optional[bool]
    data_source_was_modified: Optional[bool]
    latest_output_location: Optional[str]
    latest_output_location_local: Optional[str]
    latest_output_dataset_id: Optional[str]
    lineage_model_id: Optional[str]
    latest_output_columns: Optional[List[str]]
    latest_output_row_count: Optional[int]
    latest_output_created_at: Optional[str]
    latest_output_is_trainable: Optional[bool]
    dataset_size_bytes: Optional[int]
    file_size_bytes: Optional[int]
    # Byte-aware routing signals (measured at ingest by the samplers; written
    # once by send_message before the planner — downstream nodes read, never
    # recompute). estimated_full_bytes = estimated uncompressed in-memory
    # footprint of the WHOLE frame (total_rows * bytes_per_row), NOT the
    # on-disk/compressed source size in file_size_bytes.
    bytes_per_row: Optional[float]
    estimated_full_bytes: Optional[int]
    # True when the byte gate decided this request must run as a background
    # job (estimated_full_bytes >= ANALYSIS_MAX_INMEMORY_BYTES). Set once by
    # send_message pre-planner; routes plan_etl -> code_etl (summarizer
    # structurally bypassed) -> schedule_task. Only meaningful when
    # AVALOKA_BYTE_ROUTING is on.
    analysis_defer_required: Optional[bool]

    # Ray job configuration
    ray_namespace: Optional[str]
    ray_timeout_s: Optional[int]
    ray_secret_name: Optional[str]
    ray_execution_profile: Optional[str]
    ray_fallback_reason: Optional[str]     # why a turn meant for Ray ran locally (labelled in the reply)

    # Dataset identifier for execution nodes
    dataset_id: Optional[str]

    # --- Sampler Agent state ---
    # Set by sampling_agent_daft; written twice (Phase A quick, Phase B full)
    quick_sample_rows: Optional[List[Dict[str, Any]]]    # Phase A: immediate sample
    full_sample_rows: Optional[List[Dict[str, Any]]]     # Phase B: Full sample
    sample_statistics: Optional[Dict[str, Any]]          # column_statistics + data_quality; Phase B overwrites
    sample_status: Optional[str]                         # "quick_sample" | "refining" | "full_sample" | "error"

    # --- Profiling Agent state ---
    # Set by profiling_agent; depends on sample_statistics above
    quick_profiling_result: Optional[Dict[str, Any]]     # Phase A: domain + column semantics (LLM calls 1+2)
    full_profiling_result: Optional[Dict[str, Any]]      # Phase B: all insights (LLM calls 1+2+3)
    profiling_status: Optional[str]                      # "quick_profile" | "refining" | "full_profile" | "error"
    analysis_fidelity: Optional[str]                     # "quick_sample" | "portfolio_samples" | "entire_dataset"
    selected_sample_name: Optional[str]                  # selected sample from available_samples
