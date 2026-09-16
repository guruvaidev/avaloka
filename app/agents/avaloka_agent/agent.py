"""Top-level Avaloka conversational agent (LangGraph node).

Order of operations on each turn:
1. Load condensed Redis context for the bound thread.
2. Detect a fresh ``connection_event`` from state (set by /api/upload,
   /api/register-existing-storage, /api/database/connect).
3. Classify the user message into one intent (Groq, fallback heuristic).
4. Run discovery (openclaw if enabled, else native) when intent is
   ``onboarding`` or ``exploration``.
5. Reason about size → fidelity → execution mode.
6. Resolve infra preference.
7. Decide: delegate to ``plan_etl`` OR reply directly.
8. Return state updates that the rest of the graph already understands.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.agents.avaloka_agent.discovery_tools import discover as native_discover
from app.agents.avaloka_agent.execution_profile import decide_execution_profile
from app.agents.avaloka_agent.infra_preference import resolve_infra_preference
from app.agents.avaloka_agent.intent_classifier import IntentResult, classify_intent
from app.agents.avaloka_agent.openclaw_client import (
    OpenclawDiscoveryClient,
    OpenclawUnavailable,
)
from app.agents.avaloka_agent.prompts import (
    AVALOKA_SYSTEM_PROMPT,
    DIRECT_REPLY_PROMPT,
    HANDOFF_LINES,
)
from app.agents.avaloka_agent.session_context import (
    load_thread_context,
    render_for_prompt,
)
# Swarm is an optional commercial capability. Resolved through capabilities so
# an open-source build without the module degrades to no narration instead of
# failing to import. See app/agents/avaloka_agent/capabilities.py.
from app.agents.avaloka_agent.capabilities import announce as swarm_announce
from app.agents.avaloka_agent.capabilities import plan_swarm
from app.agents.suggestions import pick_suggestion, remember_offered_options
from app.core.inference import build_chat_model
from app.core.model_config import resolve as resolve_model
from app.graph.etl_state import ETLState

logger = logging.getLogger(__name__)


# Resolved centrally: this file previously hardcoded llama-3.3-70b-versatile,
# which Groq has since removed from the account (HTTP 404). Model choice now
# lives in app/core/model_config.py and is env-overridable.
_GROQ_MODEL = resolve_model("conversational")
_KEY_ENV = "GROQ_API_KEY_PLANNING_AGENT"

_reply_llm: Optional[Any] = None


#: Why the reply model is unavailable, in the operator's words. Set alongside
#: ``_reply_llm = None`` so the user can be told what to fix instead of being
#: handed canned text that looks like a considered answer.
_reply_llm_error: Optional[str] = None


def _diagnose_missing_model() -> Optional[str]:
    """Name the misconfiguration, or None when the configuration looks usable.

    The failure this exists for: ``INFERENCE_PROVIDER=groq`` with an EMPTY
    Groq key but a populated OpenRouter key. The operator supplied a valid
    key, the selected provider could not use it, and every turn silently fell
    back to canned text. Saying "no model configured" would have been wrong
    too -- a key WAS present, just not for the selected provider.
    """
    provider = (os.getenv("INFERENCE_PROVIDER") or "groq").strip().lower()
    keys = {
        "groq": ("GROQ_API_KEY", os.getenv(_KEY_ENV) or os.getenv("GROQ_API_KEY")),
        "openrouter": ("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY")),
        "openai": ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
    }
    if provider == "local":
        return None                      # a local server needs no key
    if provider not in keys:
        return (f"INFERENCE_PROVIDER is set to '{provider}', which this build does "
                f"not know. Known providers: {', '.join(sorted(keys))}, local.")
    var, value = keys[provider]
    if (value or "").strip():
        return None
    others = [name for name, (_v, val) in keys.items()
              if name != provider and (val or "").strip()]
    hint = (f" A key for {' and '.join(others)} IS set — either set {var} or "
            f"set INFERENCE_PROVIDER to {others[0]}." if others else
            f" Set {var}, or set INFERENCE_PROVIDER=local to use a local model.")
    return f"INFERENCE_PROVIDER is '{provider}' but {var} is empty.{hint}"


def _get_reply_llm() -> Optional[Any]:
    global _reply_llm, _reply_llm_error
    if _reply_llm is not None:
        return _reply_llm
    problem = _diagnose_missing_model()
    if problem:
        # Loud, once, with the variable named. A silent degrade here is the
        # difference between "the product is broken" and "my config is wrong",
        # and the operator is the only one who can tell them apart.
        if _reply_llm_error != problem:
            logger.error("avaloka_agent: no usable reply model — %s", problem)
        _reply_llm_error = problem
        return None
    api_key = os.environ.get(_KEY_ENV) or os.environ.get("GROQ_API_KEY")
    try:
        _reply_llm = build_chat_model(
            role="planning",
            agent="CONVERSATIONAL",
            tier="large",
            temperature=0.3,
            groq_model=_GROQ_MODEL,
            groq_api_key=api_key,
        )
        _reply_llm_error = None
    except Exception as exc:
        _reply_llm_error = (f"the configured model could not be constructed: "
                            f"{type(exc).__name__}: {exc}")
        logger.error("avaloka_agent: reply LLM init failed — %s", _reply_llm_error)
        _reply_llm = None
    return _reply_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_human_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            content = m.content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
                return ""
            return content or ""
    return ""


def _extract_thread_id(state: Dict[str, Any]) -> Optional[str]:
    # A session id is not a thread id: passing it to get_thread_session silently
    # loses Redis context and connection events. Interfaces must provide the real
    # thread id; planner metadata is retained as a compatibility fallback.
    return (
        state.get("thread_id")
        or (state.get("planner_definition") or {}).get("thread_id")
    )


async def _load_redis_context_async(state: Dict[str, Any]) -> Dict[str, Any]:
    thread_id = _extract_thread_id(state)
    if not thread_id:
        return {}
    try:
        return await load_thread_context(thread_id)
    except Exception as exc:
        logger.debug("avaloka_agent: redis context load failed: %s", exc)
        return {}


def _load_redis_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Sync wrapper around the async loader. Safe both inside and outside an
    existing event loop (LangGraph nodes run sync by default)."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # We're inside an event loop already (e.g. tests using pytest-asyncio).
        # Schedule and wait via run_coroutine_threadsafe is non-trivial;
        # safer to spin a fresh loop in a worker thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _load_redis_context_async(state)).result()
    try:
        return asyncio.run(_load_redis_context_async(state))
    except Exception as exc:
        logger.debug("avaloka_agent: sync redis loader failed: %s", exc)
        return {}


def _run_discovery(state: Dict[str, Any], redis_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Try openclaw first if enabled, then fall back to native discovery."""
    if OpenclawDiscoveryClient.is_enabled():
        try:
            client = OpenclawDiscoveryClient()
            if not client.is_available():
                raise OpenclawUnavailable("gateway healthcheck failed")
            conn_id = state.get("connection_id")
            cloud_uri = (
                state.get("data_source_location_cloud")
                or state.get("data_source_location")
            )
            if cloud_uri and str(cloud_uri).lower().startswith(
                ("s3://", "gs://", "gcs://", "az://", "azure://")
            ):
                result = client.discover_object_store(cloud_uri, conn_id or "")
            elif conn_id:
                result = client.discover_database(conn_id)
            elif redis_ctx.get("dataset_id"):
                result = client.sample_rows(redis_ctx["dataset_id"], n=50)
            else:
                result = {"source": "openclaw", "note": "no datasource bound"}
            result.setdefault("source", "openclaw")
            return result
        except OpenclawUnavailable as exc:
            logger.info("avaloka_agent: openclaw unavailable, falling back: %s", exc)
        except Exception as exc:
            logger.warning("avaloka_agent: openclaw call failed, falling back: %s", exc)
    return native_discover(redis_ctx, state)


# ---------------------------------------------------------------------------
# Reply rendering
# ---------------------------------------------------------------------------


def _describe_dataset(
    redis_ctx: Dict[str, Any],
    discovery: Optional[Dict[str, Any]],
) -> str:
    """A plain-text description of the bound dataset, with no LLM.

    Used by the deterministic fallback so an install with no model key can
    still answer "what is in this dataset?". Says so plainly when nothing is
    bound rather than inventing a shape.
    """
    cols = list(redis_ctx.get("schema_columns") or [])
    types = redis_ctx.get("schema_types") or {}
    rows = redis_ctx.get("row_count")
    if not cols and isinstance(discovery, dict):
        cols = list(discovery.get("columns") or [])
        types = discovery.get("dtypes") or discovery.get("types") or types
        rows = discovery.get("row_count", rows)

    if not cols:
        return ("I don't have a dataset bound to this conversation yet — "
                "upload a file or connect a datasource and I'll describe it.")

    head = f"This dataset has {len(cols)} columns"
    if rows:
        head += f" and {rows} rows"
    lines = [head + ":", ""]
    for c in cols[:50]:
        t = types.get(c) if isinstance(types, dict) else None
        lines.append(f"- {c}" + (f" ({t})" if t else ""))
    if len(cols) > 50:
        lines.append(f"- …and {len(cols) - 50} more")
    lines += ["", "Ask me to profile it, plot something, or train a baseline model."]
    return "\n".join(lines)


def _render_direct_reply(
    state: Dict[str, Any],
    intent: IntentResult,
    redis_ctx: Dict[str, Any],
    discovery: Optional[Dict[str, Any]],
    avaloka_mode: str,
) -> str:
    """Render a user-facing reply for non-delegating intents."""
    llm = _get_reply_llm()
    user_msg = _last_human_text(state.get("messages") or [])
    ctx_text = render_for_prompt(redis_ctx)

    # Deterministic fallback so the graph stays useful with no model key.
    #
    # There are two very different reasons to be here and they must not look
    # the same to the user:
    #   * no model is configured at all -- the deterministic answers below are
    #     the intended product, and they are genuinely useful;
    #   * a model IS configured and cannot be used -- the operator set a key,
    #     the selected provider could not use it, and answering in canned text
    #     hides a fixable misconfiguration behind something that reads like a
    #     considered reply. Say what is wrong and name the variable.
    if llm is None and _reply_llm_error:
        return (
            "I can't reach a language model, so I'm not able to answer this "
            "properly rather than guess.\n\n"
            f"**What's wrong:** {_reply_llm_error}\n\n"
            "Data profiling, schema questions and the evidence checks still "
            "work without a model — ask me what's in your dataset and I'll "
            "answer from the computed profile."
        )
    if llm is None:
        if avaloka_mode == "onboarding" and (redis_ctx.get("dataset_id") or discovery):
            cols = redis_ctx.get("schema_columns") or []
            return (
                f"Connected. I can see "
                f"{len(cols) if cols else 'your'} columns"
                + (f" ({', '.join(map(str, cols[:8]))}{'…' if len(cols) > 8 else ''})" if cols else "")
                + ". What would you like to do? "
                "Try: 'show feature importance', 'train a baseline model', "
                "or 'convert to parquet'."
            )
        if avaloka_mode == "status":
            return "I can list active tasks and trained models. Ask 'list tasks' or 'list models'."
        if intent.intent == "exploration":
            # Answer the question actually asked. Discovery has already run for
            # this intent and the schema is sitting in `discovery` / `redis_ctx`
            # -- without this branch the ladder fell through to the generic
            # "tell me what you're trying to figure out", i.e. it held the
            # schema and asked the user what they wanted instead of showing it.
            # That is the first question most people ask ("what is in this
            # dataset?"), and on an OSS install with no model key configured it
            # is the ONLY path, which is precisely when this fallback is meant
            # to keep the graph useful.
            return _describe_dataset(redis_ctx, discovery)
        if intent.intent == "chit_chat":
            return (
                "Hi — I'm Avaloka, your data scientist. I dig into data, engineer "
                "it, model it, and hand you an answer you can act on. Connect a "
                "datasource (CSV, Parquet, S3/GCS, or a database) and I'll take a "
                "look."
            )
        return "Tell me a little about what you're trying to figure out, and I'll take it from there."

    payload = (
        f"<avaloka_mode>{avaloka_mode}</avaloka_mode>\n"
        f"<intent>{intent.intent}</intent>\n"
        f"<session_context>\n{ctx_text}\n</session_context>\n"
        f"<discovery_result>{discovery or 'none'}</discovery_result>\n"
        f"<recommendation>{state.get('avaloka_recommendation') or 'none'}</recommendation>\n"
        f"<user_message>{user_msg}</user_message>"
    )

    try:
        result = llm.invoke([
            SystemMessage(content=AVALOKA_SYSTEM_PROMPT + "\n\n" + DIRECT_REPLY_PROMPT),
            HumanMessage(content=payload),
        ])
        out = getattr(result, "content", "") or ""
        return out if isinstance(out, str) else str(out)
    except Exception as exc:
        logger.warning("avaloka_agent: reply LLM failed: %s", exc)
        return "Got it — let me know what analysis you'd like to run."


# ---------------------------------------------------------------------------
# Pre-configuration of state for plan_etl when delegating
# ---------------------------------------------------------------------------


def _preconfigure_for_planner(
    state: Dict[str, Any],
    intent: IntentResult,
) -> Dict[str, Any]:
    """Set state fields the planner already understands so it picks the
    right downstream path without re-classifying."""
    updates: Dict[str, Any] = {}

    if intent.intent in ("ml_training", "ml_inference"):
        updates["enable_training"] = True

    if intent.intent == "infrastructure":
        updates["infrastructure_request"] = state.get("infrastructure_request") or {
            "type": state.get("default_infra_platform", "gcp"),
            "app_type": "python-docker",
            "use_ray": True,
        }

    if intent.intent == "schedule":
        ts = state.get("task_schedule") or {}
        if not ts.get("task_type"):
            updates["task_schedule"] = {
                "task_type": "execute",
                "schedule_type": "relative",
                "second": 0,
                "max_runs": 1,
            }

    if intent.intent in (
        "statistical_analysis",
        "visualization",
        "data_transfer",
    ):
        # Hint to the planner that code generation is the right path. The
        # planner's own LLM still produces the actual plan text.
        updates["ready_to_code"] = True

    return updates


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


def avaloka_agent_node(state: ETLState) -> Dict[str, Any]:
    """Top-level node. Returns a dict shallow-merged into ETLState."""
    messages = list(state.get("messages") or [])
    user_text = _last_human_text(messages)

    # A reply of "5" answers the numbered list offered last turn. Resolve it
    # before classifying, because the bare number carries no intent of its own:
    # classified as-is it reads as chit-chat, and the analysis the user asked
    # for by number never runs. The resolved text is what gets classified and
    # handed to the planner; the message the user actually typed is left in the
    # history untouched.
    picked = pick_suggestion(user_text, state)
    if picked is not None:
        logger.info("avaloka_agent: %r refers to a previously offered option.", user_text)
        user_text = picked
        messages = messages[:-1] + [HumanMessage(content=picked)]

    # 1. Redis-backed thread context (best-effort)
    redis_ctx = _load_redis_context(dict(state))

    # 2. Connection event detection: state takes precedence over redis
    connection_event = state.get("connection_event") or redis_ctx.get("connection_event")
    has_connection_event = bool(connection_event)

    # 3. Classify intent (uses Groq if available, heuristic otherwise)
    session_summary = render_for_prompt(redis_ctx)
    intent = classify_intent(
        messages,
        has_connection_event=has_connection_event,
        session_summary=session_summary,
    )
    logger.info(
        "avaloka_agent: intent=%s confidence=%.2f reasoning=%s",
        intent.intent, intent.confidence, intent.reasoning,
    )

    # 4. Discovery (only when relevant)
    discovery: Optional[Dict[str, Any]] = None
    if intent.intent in ("onboarding", "exploration") or has_connection_event:
        discovery = _run_discovery(dict(state), redis_ctx)

    # 5. Mode classification (drives which prompt rendering / routing branch)
    if has_connection_event and intent.intent in ("onboarding", "chit_chat", "clarification", "exploration"):
        avaloka_mode = "onboarding"
    elif intent.intent == "status":
        avaloka_mode = "status"
    elif intent.intent in ("exploration", "chit_chat", "clarification"):
        avaloka_mode = "exploring"
    else:
        avaloka_mode = "analyzing"

    # 6. Decide delegation. Data-touching intents and ambiguous
    #    ("clarification") messages both delegate to plan_etl, which has
    #    the richer toolset for asking follow-ups. Direct replies are
    #    reserved for clearly conversational intents.
    delegate = intent.should_delegate()

    # 7. Build the state update bundle
    updates: Dict[str, Any] = {
        "avaloka_mode": avaloka_mode,
        "avaloka_intent": intent.intent,
        "redis_context": redis_ctx or None,
        "discovery_result": discovery,
        "delegate_to_planner": bool(delegate),
    }

    if delegate:
        # Pre-configure execution profile + infra preference so plan_etl
        # routes correctly without re-doing the same reasoning.
        updates.update(decide_execution_profile(dict(state), intent.intent))
        updates.update(resolve_infra_preference(dict({**state, **updates}), redis_ctx))
        updates.update(_preconfigure_for_planner(dict({**state, **updates}), intent))

        handoff = HANDOFF_LINES.get(intent.intent, "On it…")
        rec = updates.get("avaloka_recommendation") or {}
        if rec.get("execution_mode") == "k8s-ray" and rec.get("eta_text"):
            handoff = f"{handoff} Running on the Ray cluster — ETA {rec['eta_text']}."
        # Surface the handoff via a dedicated field rather than the message
        # stream — the downstream planner / coder / executor own user-facing
        # AI messages, and we don't want to interfere with their content.
        updates["avaloka_handoff_message"] = handoff
        # The clone plan for this intent: the ordered set of focused copies
        # Avaloka is dispatching. Rides in state and is surfaced to the UI via
        # execution_context.swarm; it narrates the real pipeline, never routes.
        swarm_plan = plan_swarm(intent.intent)
        if swarm_plan:
            updates["avaloka_swarm"] = swarm_plan
            updates["avaloka_swarm_message"] = swarm_announce(intent.intent)
        # Clear connection_event so we only onboard once.
        if state.get("connection_event"):
            updates["connection_event"] = None
        if picked is not None:
            # Hand the planner the option text rather than the number, and
            # retire the offer now it has been taken.
            updates["messages"] = messages
            updates["pending_suggestions"] = None
        return updates

    # Direct reply path
    reply_text = _render_direct_reply(
        dict({**state, **updates}),
        intent,
        redis_ctx,
        discovery,
        avaloka_mode,
    )
    updates["messages"] = messages + [AIMessage(content=reply_text)]
    # The prompt tells this reply to end with a numbered list and to invite a
    # pick ("Say the number and I'll start"). Record what was offered, or that
    # nothing was, so the next turn can honour that invitation instead of
    # receiving a bare "5" with no record of what 5 meant.
    remember_offered_options(updates, reply_text)
    if avaloka_mode == "onboarding" and state.get("connection_event"):
        updates["connection_event"] = None
    return updates


def route_avaloka(state: ETLState) -> str:
    """LangGraph edge selector for the avaloka_agent node."""
    return "delegate_to_planner" if state.get("delegate_to_planner") else "end"
