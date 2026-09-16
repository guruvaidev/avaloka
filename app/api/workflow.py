import csv
import io
import json
import logging
import os
import re
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.etl_state import ETLState
from app.core.analysis_limits import byte_routing_enabled

from app.agents.coder import coder_node
from app.agents.state import CodingAgentState
from app.agents.validator import syntactic_validator_node, static_semantic_validator_node, logical_semantic_validator_node, execute_code_node
import subprocess
from app.agents.summarizer import summarize_etl_job
from app.agents.planner import plan_etl_job, _detect_ambiguous_prompt_details
from app.agents.infra_agent import infra_agent_node
from app.agents.execution_agent import execution_agent_node_local, execution_agent_node, execution_agent_node_ray
from app.agents.planner_graph_agent import planner_graph_agent_node
from app.agents.visualization_agent import visualization_agent_node
from app.agents.scheduler import task_scheduler_node
from app.agents.avaloka_agent import avaloka_agent_node, route_avaloka
from app.agents.claim_verifier import claim_verifier_node

logger = logging.getLogger(__name__)

# Mirrors app.api.server.LARGE_DATASET_THRESHOLD_BYTES — duplicated (not
# imported) to avoid a circular import, since server.py imports build_graph
# from this module. Keep the two values in sync.
LARGE_DATASET_THRESHOLD_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB


# ---------------------------------------------------------------------------
# Helpers: cloud provider / infra platform inference
# ---------------------------------------------------------------------------

def _infer_cloud_provider(state: ETLState) -> str:
    uri = (state.get("data_source_location_cloud") or state.get("data_source_location") or "").lower()
    if uri.startswith("s3://"):
        return "aws"
    if uri.startswith(("gs://", "gcs://")):
        return "gcp"
    if uri.startswith(("az://", "azure://")):
        return "azure"
    return "unknown"


def _infer_infra_platform_for_agent(state: ETLState) -> str:
    provider = _infer_cloud_provider(state)
    supported = {"gcp", "aws"}
    if os.getenv("ENABLE_AZURE_INFRA", "0") == "1":
        supported.add("azure")
    if provider in supported:
        return provider
    default_platform = state.get("default_infra_platform") or os.getenv("DEFAULT_INFRA_PLATFORM", "gcp")
    if default_platform not in supported:
        default_platform = "gcp"
    logger.warning(
        "Could not infer supported infra platform (got %s). Defaulting to %s.",
        provider, default_platform
    )
    return default_platform


_INFERENCE_KEYWORDS_RE = re.compile(
    r"\b(?:predict|predictions?|inference|infer|run\s+the\s+model|use\s+the\s+model)\b",
    re.IGNORECASE,
)


def _looks_like_inference_request(state: ETLState) -> bool:
    """Deterministic inference detection: predict/inference phrasing in the
    latest user message, with a trained model available to run against."""
    has_model = bool(
        state.get("training_result")
        or state.get("model_artifacts")
        or state.get("mlflow_run_id")
        or state.get("training_completed")
    )
    if not has_model:
        return False
    text = state.get("user_prompt") or get_user_prompt(state.get("messages", []) or [])
    return bool(text and _INFERENCE_KEYWORDS_RE.search(str(text)))


def _has_embedded_json_payload(text: str) -> bool:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text or ""):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return True
    return False


def _looks_like_structured_inference_request(state: ETLState) -> bool:
    if not _looks_like_inference_request(state):
        return False
    text = state.get("user_prompt") or get_user_prompt(state.get("messages", []) or [])
    text = str(text).split("\n\n[Analysis context]", 1)[0]
    return _has_embedded_json_payload(text)


def _looks_like_direct_inference_request(state: ETLState) -> bool:
    if not _looks_like_inference_request(state):
        return False
    text = state.get("user_prompt") or get_user_prompt(state.get("messages", []) or [])
    text = str(text).split("\n\n[Analysis context]", 1)[0].lower()
    if re.search(r"\b(?:configure|deploy|start|stop)\b.{0,40}\binference\s+service\b", text):
        return False
    if _has_embedded_json_payload(text):
        return True
    direct_inference = re.search(
        r"\b(?:run|execute|perform|do)\s+(?:an?\s+)?inference\b",
        text,
    )
    return bool(
        direct_inference
        and re.search(r"\b(?:with|using|values?|row|input)\b", text)
    )


def _message_role(message) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "").lower()
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, AIMessage):
        return "ai"
    role = getattr(message, "type", None)
    if role:
        return str(role).lower()
    return ""


def _message_content(message) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list) and content:
            first = content[0]
            return first.get("text", "") if isinstance(first, dict) else str(first)
        return str(content or "")
    return str(getattr(message, "content", "") or "")


def _latest_user_index(messages: list) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if _message_role(messages[index]) in {"human", "user"}:
            return index
    return None


# ---------------------------------------------------------------------------
# route_planner_output — merged:
#   • k8s-ray fast-path (from current branch)
#   • full MTA/training-plan keyword detection (from develop-1.2)
#   • schedule_task routing preserved (Shawn's fix)
# ---------------------------------------------------------------------------

def route_planner_output(state: ETLState):
    # ── Scheduler: training task type must be checked first (develop-1.2) ──
    # if (state.get("task_schedule", {}) or {}).get("task_type") == "training":
    #     return "schedule_task"

    # ts = (state.get("task_schedule") or {})
    # if ts.get("task_type") in ("training", "execute") and not state.get("ready_to_summarize"):
    #     return "schedule_task"

    # execution_mode = state.get("execution_mode", "cloud")

    # # ── k8s-ray fast-path ──────────────────────────────────────────────────
    # if execution_mode == "k8s-ray":
    #     data_uri = (state.get("data_source_location_cloud") or "").strip()

    if state.get("skip_to_training", False):
        return "train_models"

    ts = (state.get("task_schedule") or {})
    # ✅ DIAGNOSTIC LOG — delete once confirmed working
    logger.info(
        "route_planner_output: task_schedule=%s execution_mode=%s ready_to_summarize=%s",
        ts,
        state.get("execution_mode"),
        state.get("ready_to_summarize"),
    )

    # ── Execution already completed → summarize, don't let fidelity policy
    #    re-decide execution_mode / inject a fresh task_schedule and hijack
    #    the route to schedule_task, discarding the result we already have ──
    if state.get("ready_to_summarize"):
        # The planner may mark an inference turn ready to summarize after
        # producing an ETL-shaped tool call.  Preserve the model intent before
        # honoring that stale planner flag so the request reaches the MTA.
        if _looks_like_inference_request(state):
            try:
                from app.agents.model_training_agent import model_training_agent_node  # noqa: F401
                logger.info(
                    "route_planner_output: inference request overrides ready_to_summarize -> train_models"
                )
                return "train_models"
            except ImportError:
                pass
        # ── D-01 structural bypass: on the byte-gated deferred path the
        #    summarizer must never run — merely instructing it to behave
        #    differently still yields a fluent answer assembled from plan
        #    metadata. Route straight to the coder (it consumes state["plan"],
        #    not the summarizer's planner_definition); code_etl then routes to
        #    schedule_task via the defer policy. Graph-level, not prompt-level.
        if byte_routing_enabled() and state.get("analysis_defer_required"):
            logger.info(
                "route_planner_output: defer_required -> code_etl (summarizer bypassed structurally)"
            )
            return "prepare_code"
        logger.info("route_planner_output: Routing to summarize_etl (ready_to_summarize=True, pre-fidelity)")
        return "summarize"

    execution_mode = state.get("execution_mode", "cloud")

    # ── Fidelity policy: normal prompting always uses the sampling approach.
    #    The scheduler must never be triggered automatically by fidelity —
    #    only a genuine, explicit task_operation request may schedule a task.
    #    "entire_dataset" therefore executes directly, same as the other
    #    fidelities, instead of auto-escalating to k8s-ray + schedule_task. ──
    # This is a conditional-edge router: LangGraph discards any state mutation
    # made here (only node returns persist), so execution_mode is kept in a
    # local variable and never written back to state. coding_subgraph_node
    # decides and persists execution_mode authoritatively.
    fidelity = state.get("analysis_fidelity")
    if fidelity == "quick_sample":
        execution_mode = "local"
    elif fidelity in ("portfolio_samples", "entire_dataset"):
        execution_mode = state.get("execution_mode", execution_mode)

    ts = (state.get("task_schedule") or {})

    # ── develop-1.2: training or explicit async Ray execute → Celery / scheduler ──
    if ts.get("task_type") == "training" or (
        ts.get("task_type") == "execute" and execution_mode == "k8s-ray"
    ):
        return "schedule_task"

    # ── k8s-ray fast-path ──────────────────────────────────────────────────
    # NOTE: k8s-ray execution submits to an existing Ray cluster. It must NOT
    # go through the Docker/k8s infra provisioning agent (that path is for deploy_on_k8s).
    if execution_mode == "k8s-ray":
        data_uri = (state.get("data_source_location_cloud") or "").strip()
        conn_id  = state.get("connection_id")

        # GUARD: no cloud URI or no connection_id → fall back to local
        if not data_uri or not conn_id:
            logger.warning(
                "route_planner_output: k8s-ray but data_uri=%s conn_id=%s -> falling back to local",
                data_uri, conn_id,
            )
            execution_mode = "local"
            # Fall through to normal routing below

        else:
            # Honor an explicit schedule request if one is genuinely present
            # (e.g. task_operation flow) — fidelity alone no longer sets this.
            if (state.get("task_schedule") or {}).get("task_type") == "execute":
                return "schedule_task"
            # Otherwise, go directly to Ray execution once code exists (no Celery schedule).
            if state.get("coder_definition", {}).get("code"):
                logger.info("route_planner_output: infra ready + code exists -> execute_on_ray")
                return "execute_on_ray"
            # No code yet → fall through so planner/coder can generate it, then route_after_code will execute_on_ray.

    if state.get("enable_training"):
        return "train_models"

    if _looks_like_direct_inference_request(state):
        try:
            from app.agents.model_training_agent import model_training_agent_node  # noqa: F401
            logger.info("route_planner_output: direct inference request -> train_models")
            return "train_models"
        except ImportError:
            pass

    # ── task_operation (scheduler) ─────────────────────────────────────────
    if state.get("task_operation"):
        logger.info("route_planner_output: Routing to schedule_task (task_operation detected)")
        return "task_operation"

    # ── ready_to_summarize / ready_to_code ────────────────────────────────
    if state.get("ready_to_summarize"):
        logger.info("route_planner_output: Routing to summarize_etl (ready_to_summarize=True)")
        return "summarize"
    elif state.get("ready_to_code"):
        logger.info("route_planner_output: Routing to prepare_code (ready_to_code=True)")
        return "prepare_code"

    # ── infra provisioning ─────────────────────────────────────────────────
    if (
        not state.get("infrastructure_provisioned")
        and state.get("infrastructure_request")
    ):
        return "provision_infra"

    # ── Deterministic inference routing: a predict/inference request with a
    #    trained model in session must reach the MTA even when the planner
    #    LLM didn't set enable_training ──────────────────────────────────────
    if _looks_like_inference_request(state):
        try:
            from app.agents.model_training_agent import model_training_agent_node  # noqa: F401
            logger.info("route_planner_output: inference request with trained model -> train_models")
            return "train_models"
        except ImportError:
            pass

    # ── already executed successfully ─────────────────────────────────────
    if state.get("execution_result", {}).get("status") == "success":
        return "end"

    return "continue_planning"


# ---------------------------------------------------------------------------
# route_after_summary / route_after_graph
# ---------------------------------------------------------------------------

def route_after_summary(state: ETLState):
    return "generate_graph"


def route_after_graph(state: ETLState):
    """Route to code_etl after planner graph generation."""
    return "code_etl"


# ---------------------------------------------------------------------------
# route_after_code — schedule_task preserved (Shawn's fix) + execute_on_ray
# ---------------------------------------------------------------------------

def route_after_code(state: ETLState):
    # Check if training is enabled and MTA is available
    enable_training = state.get("enable_training", False)
    mta_available = False
    try:
        from app.agents.model_training_agent import model_training_agent_node  # noqa: F401
        mta_available = True
    except ImportError:
        pass

    # NOTE: execution_mode / task_schedule for the current fidelity are decided
    # and returned by coding_subgraph_node (a node, so LangGraph persists it) —
    # this router only reads them; mutating state here would be discarded
    # before task_scheduler_node runs.

    if (mta_available and enable_training and
            state.get("coder_definition", {}).get("code") and
            not state.get("ready_to_train")):
        # Training is enabled, code is ready but no training initiated yet - go to MTA
        return "train_models"
    elif (mta_available and enable_training and
          state.get("ready_to_train") and
          not state.get("training_completed")):
        # Training is enabled, ready but not completed - stay in training
        return "train_models"
    elif state.get("task_schedule", {}) and state.get("coder_definition", {}).get("code"):
        # ── schedule_task must come BEFORE execution checks
        return "schedule_task"
    elif state.get("coder_definition", {}).get("code"):
        # Code is ready and training is disabled/done/unavailable - execute code
        mode = state.get("execution_mode")
        if mode is None and state.get("data_source_location_cloud") and state.get("connection_id"):
            mode = "k8s-ray"
        if mode == "k8s-ray":
            return "execute_on_ray"
        elif state.get("deploy_on_k8s"):
            return "execute_on_k8s"
        else:
            return "execute_locally"
    else:
        # No code to execute - end the workflow
        return "end"


# ---------------------------------------------------------------------------
# route_after_training — full keyword-aware logic from develop-1.2 + Ray-aware
# ---------------------------------------------------------------------------

def route_after_training(state: ETLState):
    task_schedule = state.get("task_schedule") or {}
    if task_schedule.get("task_type") in {"training", "execute", "start_inference", "stop_inference"} or \
        state.get("training_scheduled", False) or \
        state.get("inference_scheduled", False) or \
        state.get("configure_inference_service_scheduled", False) or \
        state.get("stop_inference_service_scheduled", False):
        return "schedule_task"
    return "end"


# ---------------------------------------------------------------------------
# route_after_execution
# ---------------------------------------------------------------------------

def route_after_execution(state: ETLState):
    """Route to visualization after execution completes."""
    execution_result = state.get("execution_result", {})
    if execution_result.get("status") in {"error", "failed"} or state.get("execution_error"):
        return "end"
    sample_data = state.get("uploaded_csv_preview", [])
    if execution_result or sample_data:
        return "visualize"
    return "end"


# ---------------------------------------------------------------------------
# Coding subgraph
# ---------------------------------------------------------------------------

def check_validation_status(state: CodingAgentState) -> str:
    """Routes back to coder if validation fails, otherwise ends."""
    has_error = (
        state.get("syntax_error")
        or state.get("static_semantic_error")
        or state.get("logical_semantic_error")
    )
    if not has_error:
        return "end"
    retry_count = state.get("retry_count") or 0
    if retry_count >= 3:
        return "end"
    if retry_count >= 2:
        # Hard (syntax/static) failures get the final pass so the coder can
        # emit its deterministic stub. Soft/logical failures now ALSO get one
        # final constrained attempt (the coder restricts itself to plain
        # pandas on schema columns when logical_semantic_error is set at this
        # depth) — previously they terminated here, so one flaky retry became
        # a permanent "failed validation after all retries" on well-formed asks.
        return "refine"
    return "refine"


def build_coding_graph(checkpointer=None):
    # The coding subgraph is invoked once per coding_subgraph_node call WITHOUT a
    # thread_id, so it must NOT inherit the top-level graph's checkpointer — a
    # checkpointer would make .invoke() require a thread_id and raise. Only an
    # explicitly-passed checkpointer is honored.
    effective_checkpointer = checkpointer

    workflow = StateGraph(CodingAgentState)
    workflow.add_node("coder", coder_node)
    workflow.add_node("validator_syntax", syntactic_validator_node)
    workflow.add_node("validator_static", static_semantic_validator_node)
    workflow.add_node("execute_code", execute_code_node)
    workflow.add_node("validator_logical", logical_semantic_validator_node)

    workflow.set_entry_point("coder")
    workflow.add_edge("coder", "validator_syntax")
    workflow.add_edge("validator_syntax", "validator_static")
    workflow.add_edge("validator_static", "execute_code")
    workflow.add_edge("execute_code", "validator_logical")

    workflow.add_conditional_edges(
        "validator_logical",
        check_validation_status,
        {"refine": "coder", "end": END},
    )
    if effective_checkpointer is not None:
        return workflow.compile(checkpointer=effective_checkpointer)
    return workflow.compile()


# Module-level globals
_graph_checkpointer = None
coding_graph = None
coder_node_is_mock = False
mocked_coder_node = None


def _preview_to_csv(preview) -> str:
    """Render an uploaded_csv_preview as CSV text (header + value rows).

    Handles both preview shapes: a list[dict] keyed by column name (the API
    path) and a list[list] with a header row followed by value rows (test/legacy
    path). Iterating a dict yields only its keys, so the list[dict] case must be
    expanded explicitly rather than joined row-by-row.
    """
    if isinstance(preview, str):
        try:
            preview = json.loads(preview)
        except (ValueError, TypeError):
            return ""
    if not isinstance(preview, list) or not preview:
        return ""

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")

    first = next((r for r in preview if isinstance(r, (dict, list, tuple))), None)
    if isinstance(first, dict):
        headers: list = []
        seen = set()
        for row in preview:
            if isinstance(row, dict):
                for key in row.keys():
                    if key not in seen:
                        seen.add(key)
                        headers.append(key)
        writer.writerow(headers)
        for row in preview:
            if isinstance(row, dict):
                writer.writerow([row.get(h, "") for h in headers])
    else:
        for row in preview:
            if isinstance(row, (list, tuple)):
                writer.writerow(list(row))

    return buf.getvalue().strip("\n")


def get_user_prompt(messages: list) -> str:
    for m in reversed(messages):
        if getattr(m, "type", None) == "human" or m.__class__.__name__ == "HumanMessage":
            return getattr(m, "content", "") or ""
        if isinstance(m, dict) and (m.get("type") == "human" or m.get("role") == "user"):
            c = m.get("content", "")
            return c if isinstance(c, str) else (c[0].get("text", "") if c else "")
    return ""


def coding_subgraph_node(state: ETLState) -> dict:
    global coding_graph, coder_node_is_mock, mocked_coder_node
    if coder_node_is_mock or hasattr(coder_node, "_mock_return_value"):
        target = (
            mocked_coder_node
            if coder_node_is_mock and mocked_coder_node is not None
            else coder_node
        )
        mock_result = target(state)
        if isinstance(mock_result, dict):
            return mock_result
        return dict(mock_result)

    try:
        requirements = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
    except Exception as e:
        logger.error("Failed to get requirements: %s", e)
        requirements = ""

    sample_data = _preview_to_csv(state.get("uploaded_csv_preview"))

    user_prompt = state.get("user_prompt") or get_user_prompt(state.get("messages", []))
    clarification_details = _detect_ambiguous_prompt_details(user_prompt)
    if clarification_details:
        clarification = clarification_details["message"]
        logger.info("coding_subgraph_node: asking clarification before code generation: %s", user_prompt[:120])
        messages = list(state.get("messages", []))
        if not any(getattr(msg, "content", None) == clarification for msg in messages):
            messages.append(AIMessage(content=clarification))
        return {
            "messages": messages,
            "ready_to_code": False,
            "ready_to_summarize": False,
            "coder_definition": {},
            "generated_code": "",
            "coder_raw_response": None,
            "execution_error": clarification,
            "pending_clarification": clarification_details.get("pending_clarification"),
        }

    plan_text = state.get("plan")
    # Plans persisted by older sessions/tests may be dicts; the coder and its
    # guards expect a string.
    if plan_text is not None and not isinstance(plan_text, str):
        plan_text = json.dumps(plan_text, default=str)
    if not plan_text:
        plan_text = (
            f"1. Load the data from "
            f"{state.get('data_source_location', 'the provided source')} into a pandas DataFrame.\n"
            f"2. Fulfill this request: \"{user_prompt}\" using pandas operations.\n"
            f"3. Save the final DataFrame to "
            f"{state.get('output_location', 'the requested output path')}."
        )

    initial_coding_state = {
        "user_prompt": user_prompt,
        "plan": plan_text,
        "data_source_location": state.get("data_source_location", ""),
        "output_location": state.get("output_location", ""),
        "schema": state.get("schema", {}),
        "input_data_type": state.get("input_data_type", ""),
        "sample_data": sample_data,
        "requirements": requirements,
        "messages": state.get("messages", []),
        "retry_count": 0,
        "coder_pseudocode": state.get("coder_pseudocode"),
        "uploaded_csv_preview": state.get("uploaded_csv_preview"),
        "analysis_fidelity": state.get("analysis_fidelity"),
        "selected_sample_name": state.get("selected_sample_name"),
        "multi_dataset_state": state.get("multi_dataset_state"),
        "datasets_context": state.get("datasets_context"),
        "memory_hints": state.get("memory_hints"),
        "session_logic_signature": state.get("session_logic_signature"),
    }

    if coding_graph is None:
        coding_graph = build_coding_graph()
    final_coding_state = coding_graph.invoke(initial_coding_state)

    generated_code = final_coding_state.get("generated_code", "")
    coder_message_text = final_coding_state.get("llm_raw_response")

    blocked_reason = final_coding_state.get("coder_blocked_reason")
    if blocked_reason:
        # A guard refusal is the answer itself: surface it as the final chat
        # reply instead of executing the placeholder stub (whose output would
        # bury the message under execution/visualization noise).
        logger.info("coding_subgraph_node: coder blocked the request: %s", blocked_reason[:120])
        messages = list(state.get("messages", []))
        if not any(getattr(msg, "content", None) == blocked_reason for msg in messages):
            messages.append(AIMessage(content=blocked_reason))
        return {
            "messages": messages,
            "coder_definition": {},
            "generated_code": "",
            "coder_raw_response": blocked_reason,
            "coder_pseudocode": final_coding_state.get("coder_pseudocode"),
            "execution_error": blocked_reason,
        }

    validation_failed = bool(
        final_coding_state.get("syntax_error")
        or final_coding_state.get("static_semantic_error")
        or final_coding_state.get("logical_semantic_error")
    )
    if validation_failed:
        feedback = final_coding_state.get("code_validation_feedback") or "unspecified validation error"
        refusal = (
            "Generated code failed validation after all retries and will not be executed. "
            f"Last validation feedback: {feedback}"
        )
        logger.warning("coding_subgraph_node: %s", refusal)
        messages = list(state.get("messages", []))
        if not any(getattr(msg, "content", None) == refusal for msg in messages):
            messages.append(AIMessage(content=refusal))
        return {
            "messages": messages,
            "coder_definition": {},
            "generated_code": "",
            "coder_raw_response": coder_message_text,
            "coder_pseudocode": final_coding_state.get("coder_pseudocode"),
            "logical_review_feedback": final_coding_state.get("logical_review_feedback"),
            "execution_error": refusal,
        }

    updates = {
        "coder_definition": {"code": generated_code},
        "coder_raw_response": coder_message_text,
        "generated_code": generated_code,
        "coder_pseudocode": final_coding_state.get("coder_pseudocode"),
        "execution_output_data": final_coding_state.get("execution_output_data"),
        "execution_output_preview": final_coding_state.get("execution_output_preview"),
        "execution_stdout": final_coding_state.get("execution_stdout"),
        "execution_stderr": final_coding_state.get("execution_stderr"),
        "execution_error": final_coding_state.get("execution_error"),
        "logical_review_feedback": final_coding_state.get("logical_review_feedback"),
    }

    # ── Fidelity-based execution policy ─────────────────────────────────────
    # This must be decided here (a node's return value, which LangGraph
    # persists via its reducers) rather than inside route_after_code (a
    # router, whose in-place state mutations are discarded before the next
    # node runs). Mental model: the scheduler is never triggered by fidelity
    # alone — only entire_dataset requests on a genuinely LARGE dataset (with
    # a cloud connection to actually run on) get offloaded to Ray/Celery.
    # Everything else (quick_sample, portfolio_samples, or entire_dataset on
    # a small/local dataset) executes directly in this request.
    fidelity = state.get("analysis_fidelity")
    is_large_dataset = (
        state.get("file_size_bytes") or state.get("dataset_size_bytes") or 0
    ) >= LARGE_DATASET_THRESHOLD_BYTES
    has_cloud = bool(state.get("data_source_location_cloud") and state.get("connection_id"))
    if byte_routing_enabled() and state.get("analysis_defer_required") is not None:
        # Byte routing decided this turn pre-planner (True = defer, False =
        # full-frame local). Respect that decision — do NOT fall through to the
        # legacy entire_dataset recompute below, which keys on ON-DISK source
        # size (>=1GB + cloud) and would re-inject k8s-ray + task_schedule for
        # a below-cap run the server already routed local.
        if not state.get("analysis_defer_required"):
            updates["execution_mode"] = "local"
            updates["task_schedule"] = None
        # Deferral: never execute inline — schedule the full-dataset job. The
        # no-cloud case is declined server-side before the graph runs; if it
        # somehow reaches here, refuse with empty coder_definition so
        # route_after_code ends the run instead of executing a >budget frame.
        elif has_cloud:
            updates["execution_mode"] = "k8s-ray"
            updates["task_schedule"] = {
                "task_type": "execute",
                "schedule_type": "relative",
                "second": 0,
                "max_runs": 1,
            }
        else:
            refusal = (
                "This dataset's estimated in-memory size exceeds the interactive "
                "analysis budget, and no cloud connection is available to run it "
                "as a background job. Register the dataset from cloud storage to "
                "analyze it in full."
            )
            logger.warning(
                "coding_subgraph_node: defer_required without cloud path — refusing inline execution"
            )
            return {
                "messages": list(state.get("messages", [])) + [AIMessage(content=refusal)],
                "coder_definition": {},
                "generated_code": "",
                "execution_error": refusal,
                "execution_mode": "local",
            }
    elif fidelity == "quick_sample":
        updates["execution_mode"] = "local"
    elif fidelity == "portfolio_samples":
        updates["execution_mode"] = state.get("execution_mode") or "local"
    elif fidelity == "entire_dataset":
        if is_large_dataset and has_cloud:
            updates["execution_mode"] = "k8s-ray"
            updates["task_schedule"] = {
                "task_type": "execute",
                "schedule_type": "relative",
                "second": 0,
                "max_runs": 1,
            }
        else:
            updates["execution_mode"] = state.get("execution_mode") or "local"

    messages = list(state.get("messages", []))
    primary_message = final_coding_state.get("primary_llm_response")
    if primary_message:
        logger.warning("Primary coder response: %s", primary_message)
    if coder_message_text:
        logger.warning("Coder raw response: %s", coder_message_text)

    def append_if_new(content, current):
        if not content:
            return current
        if any(getattr(msg, "content", None) == content for msg in current):
            return current
        return current + [AIMessage(content=content)]

    updated_messages = messages
    updated_messages = append_if_new(primary_message, updated_messages)
    updated_messages = append_if_new(coder_message_text, updated_messages)

    if len(updated_messages) != len(messages):
        updates["messages"] = updated_messages

    return updates


from app.services.memory_plane import retrieve_memory
from langchain_core.messages import HumanMessage

def _extract_user_prompt(messages: list) -> str:
    """
    Robust extraction of latest user prompt.
    Works with LangChain messages OR dict-shaped messages.
    """
    for m in reversed(messages):
        # Case 1: LangChain HumanMessage
        if isinstance(m, HumanMessage):
            return getattr(m, "content", "") or ""

        # Case 2: Message has .type attribute
        if getattr(m, "type", None) == "human":
            return getattr(m, "content", "") or ""

        # Case 3: Dict-based message
        if isinstance(m, dict) and (
            m.get("type") == "human" or m.get("role") == "user"
        ):
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list) and content:
                return content[0].get("text", "")

    return ""

def memory_injection_node(state: ETLState) -> dict:
    """
    Layered Memory Injection Node.
    Extracts user query, retrieves memory, and injects hints into ETLState.
    """

    print("\n===== MEMORY INJECTION NODE =====")

    # 1️⃣ Extract query safely
    query = state.get("user_prompt")

    if not query:
        query = _extract_user_prompt(state.get("messages", []))

    if not query:
        print("[memory_injection] No user prompt found. Skipping memory retrieval.")
        return {
            "memory_hints": [],
            "prior_artifact_found": False,
            "session_logic_signature": None,
            "memory_context_unavailable": False
        }

    print("User Prompt:", query)

    # 2️⃣ Retrieve memory
    try:
        memory_payload = retrieve_memory(query, state)

        if not isinstance(memory_payload, dict):
            memory_payload = {}

    except Exception as e:
        print("[memory_injection] Memory retrieval failed:", str(e))
        return {
            "memory_hints": [],
            "prior_artifact_found": False,
            "session_logic_signature": None,
            "memory_context_unavailable": True
        }

    print("Memory Payload:", memory_payload)

    import uuid
    import os
    
    active_mcp_servers = []
    if os.environ.get("MCP_SERVER_URL"):
        active_mcp_servers.append(os.environ.get("MCP_SERVER_URL"))
    elif os.environ.get("MCP_URL"):
        active_mcp_servers.append(os.environ.get("MCP_URL"))

    # 3️⃣ Ensure contract fields always exist
    return {
        "memory_hints": memory_payload.get("memory_hints", []),
        "prior_artifact_found": memory_payload.get("prior_artifact_found", False),
        "session_logic_signature": memory_payload.get("session_logic_signature"),
        "memory_context_unavailable": memory_payload.get("memory_context_unavailable", False),
        "milvus_context_id": str(uuid.uuid4()),
        "active_mcp_servers": active_mcp_servers,
    }

# ---------------------------------------------------------------------------
# build_graph — combined node set: Ray + MTA + scheduler + infra
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    global coding_graph, coder_node_is_mock, mocked_coder_node, _graph_checkpointer
    _graph_checkpointer = checkpointer

    coder_node_is_mock = hasattr(coder_node, "_mock_return_value")
    mocked_coder_node = coder_node if coder_node_is_mock else None
    coding_graph = build_coding_graph()

    graph_builder = StateGraph(ETLState)

    # ── Core nodes ──────────────────────────────────────────────────────────
    graph_builder.add_node("memory_injection", memory_injection_node)
    graph_builder.add_node("avaloka_agent", avaloka_agent_node)
    graph_builder.add_node("plan_etl", plan_etl_job)
    graph_builder.add_node("summarize_etl", summarize_etl_job)
    graph_builder.add_node("generate_planner_graph", planner_graph_agent_node)
    graph_builder.add_node("code_etl", coding_subgraph_node)
    graph_builder.add_node("provision_infra", infra_agent_node)
    graph_builder.add_node("schedule_task", task_scheduler_node)
    graph_builder.add_node("execute_on_ray", execution_agent_node_ray)
    graph_builder.add_node("execute_locally", execution_agent_node_local)
    graph_builder.add_node("execute_on_k8s", execution_agent_node)
    graph_builder.add_node("visualize", visualization_agent_node)
    graph_builder.add_node("verify_claims", claim_verifier_node)

    # ── Optional MTA node ───────────────────────────────────────────────────
    try:
        from app.agents.model_training_agent import model_training_agent_node

        def debug_model_training_agent_node(state: ETLState):
            return model_training_agent_node(state)

        graph_builder.add_node("train_models", debug_model_training_agent_node)
    except ImportError:
        pass

    # Profiling node: registered for future graph routing; currently called in-process by server.py
    try:
        from app.agents.profiling_agent import profiling_agent_node
        graph_builder.add_node("profile_data", profiling_agent_node)
    except ImportError:
        pass  # profiling agent not available, skip
    graph_builder.set_entry_point("memory_injection")
    graph_builder.add_edge("memory_injection", "avaloka_agent")
    graph_builder.add_conditional_edges(
        "avaloka_agent",
        route_avaloka,
        {"delegate_to_planner": "plan_etl", "end": "verify_claims"},
    )

    # ── plan_etl → conditional routing ─────────────────────────────────────
    route_planner_map = {
        "summarize":         "summarize_etl",
        "prepare_code":      "code_etl",
        "continue_planning": "verify_claims",
        "task_operation": "schedule_task",
        "provision_infra": "provision_infra",
        "execute_locally": "execute_locally",
        "execute_on_k8s": "execute_on_k8s",
        "execute_on_ray": "execute_on_ray",
        "schedule_task": "schedule_task",
        "end": "verify_claims"
    }
    try:
        from app.agents.model_training_agent import model_training_agent_node  # noqa: F401
        route_planner_map["train_models"] = "train_models"
    except ImportError:
        pass
    graph_builder.add_conditional_edges("plan_etl", route_planner_output, route_planner_map)

    # ── summarize_etl → conditional routing ────────────────────────────────
    route_after_summary_map = {"generate_graph": "generate_planner_graph"}
    try:
        from app.agents.model_training_agent import model_training_agent_node  # noqa: F401
        route_after_summary_map["train_models"] = "train_models"
    except ImportError:
        pass
    graph_builder.add_conditional_edges(
        "summarize_etl", route_after_summary, route_after_summary_map
    )

    # ── generate_planner_graph → code_etl ──────────────────────────────────
    graph_builder.add_conditional_edges(
        "generate_planner_graph", route_after_graph, {"code_etl": "code_etl"}
    )

    # ── provision_infra loops back to plan_etl ─────────────────────────────
    graph_builder.add_edge("provision_infra", "plan_etl")

    # ── code_etl → conditional routing ─────────────────────────────────────
    route_after_code_map = {
        "schedule_task":   "schedule_task",
        "execute_on_ray":  "execute_on_ray",
        "execute_on_k8s":  "execute_on_k8s",
        "execute_locally": "execute_locally",
        "end":             "verify_claims",
    }
    try:
        from app.agents.model_training_agent import model_training_agent_node  # noqa: F401
        route_after_code_map["train_models"] = "train_models"
    except ImportError:
        pass
    graph_builder.add_conditional_edges("code_etl", route_after_code, route_after_code_map)

    # ── train_models → conditional routing ─────────────────────────────────
    try:
        from app.agents.model_training_agent import model_training_agent_node  # noqa: F401
        route_after_training_map = {
            "execute_on_ray":  "execute_on_ray",
            "execute_on_k8s":  "execute_on_k8s",
            "execute_locally": "execute_locally",
            "schedule_task":   "schedule_task",
            "code_etl":        "code_etl",
            "end":             "verify_claims",
        }
        graph_builder.add_conditional_edges(
            "train_models", route_after_training, route_after_training_map
        )
    except ImportError:
        pass

    # ── execution nodes → visualize / end ──────────────────────────────────
    for exec_node in ("execute_on_k8s", "execute_locally", "execute_on_ray"):
        graph_builder.add_conditional_edges(
            exec_node,
            route_after_execution,
            {"visualize": "visualize", "end": "verify_claims"},
        )

    # ── terminal edges ─────────────────────────────────────────────────────
    graph_builder.add_edge("visualize", "verify_claims")
    graph_builder.add_edge("schedule_task", "verify_claims")
    graph_builder.add_edge("verify_claims", END)

    return graph_builder
