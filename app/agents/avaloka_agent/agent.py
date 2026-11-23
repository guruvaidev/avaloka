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
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.agents.user_text import own_words
from app.api import reply_segments as _rs

from app.agents.avaloka_agent.discovery_tools import discover as native_discover
from app.agents.avaloka_agent.conversation_policy import (
    build_conversation_directive, infer_expertise, policy_for,
)
from app.agents.avaloka_agent.execution_profile import LOCAL_ETA_TEXT, decide_execution_profile
from app.agents.avaloka_agent.infra_preference import resolve_infra_preference
from app.agents.avaloka_agent.correction import (
    CorrectionFacts,
    correction_facts,
    phrasing_holds,
    reasoning_text,
)
from app.agents.avaloka_agent.intent_classifier import (
    IntentResult,
    asks_about_results,
    classify_intent,
)
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
    # The provider build_chat_model will actually use, not a second guess: with
    # INFERENCE_PROVIDER unset that is OpenRouter (MP-01, owner decision
    # 2026-11-01), or the provider whose key is present.
    from app.core.inference import resolve_provider

    provider = ((os.getenv("INFERENCE_PROVIDER") or "").strip().lower()
                or resolve_provider(role="planning", agent="CONVERSATIONAL",
                                    groq_api_key=os.getenv(_KEY_ENV) or None))
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


async def _load_redis_context_on_private_loop(state: Dict[str, Any]) -> Dict[str, Any]:
    """The loader as run by ``asyncio.run``: a loop that exists for one call.

    The shared cache hands each loop its own Redis client (app/core/cache.py),
    so this loop must not touch the API loop's sockets; it closes its own
    client before ``asyncio.run`` closes the loop under it.
    """
    try:
        return await _load_redis_context_async(state)
    finally:
        try:
            from app.services import session_service
            release = getattr(session_service.cache, "release_loop_client", None)
            if release is not None:
                await release()
        except Exception as exc:
            logger.debug("avaloka_agent: releasing loop redis client failed: %s", exc)


def _load_redis_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Sync wrapper around the async loader. Safe both inside and outside an
    existing event loop (LangGraph nodes run sync by default, on a worker
    thread, while the API's own loop keeps serving on the main thread)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        # We're inside an event loop already (e.g. tests using pytest-asyncio).
        # Schedule and wait via run_coroutine_threadsafe is non-trivial;
        # safer to spin a fresh loop in a worker thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _load_redis_context_on_private_loop(state)).result()
    try:
        return asyncio.run(_load_redis_context_on_private_loop(state))
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


#: How much of the conversation the direct reply sees before the current
#: message: the last six messages (three exchanges), each cut at 600 characters.
#: Without any of it, "that's wrong, it's 42.5" arrived with no record of what
#: had been said, so there was no claim to stand by. Three exchanges hold the
#: claim being disputed, the question it answered and the turn before; the cap
#: keeps a pasted table or a long earlier answer from swamping a small model's
#: context, which is the same unbounded growth the memory plane was cut back for.
_DIALOGUE_MESSAGES = 6
_DIALOGUE_CHARS = 600


def _dialogue_before(messages: List[BaseMessage]) -> str:
    """The recent turns before the current human message, as plain lines."""
    history = list(messages or [])
    for i in range(len(history) - 1, -1, -1):
        if isinstance(history[i], HumanMessage):
            history = history[:i]
            break
    lines = []
    for m in history[-_DIALOGUE_MESSAGES:]:
        if not isinstance(m, (HumanMessage, AIMessage)):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content)
        if isinstance(m, HumanMessage):
            content = own_words(content)
        content = " ".join(content.split())
        if len(content) > _DIALOGUE_CHARS:
            content = content[:_DIALOGUE_CHARS].rstrip() + "…"
        if content:
            lines.append(f"{'user' if isinstance(m, HumanMessage) else 'assistant'}: {content}")
    return "\n".join(lines) or "none"


def _render_direct_reply(
    state: Dict[str, Any],
    intent: IntentResult,
    redis_ctx: Dict[str, Any],
    discovery: Optional[Dict[str, Any]],
    avaloka_mode: str,
    conversation_directive: str,
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
        f"<conversation_so_far>\n{_dialogue_before(state.get('messages') or [])}\n</conversation_so_far>\n"
        f"<user_message>{user_msg}</user_message>\n"
        f"<communication_policy>{conversation_directive}</communication_policy>"
    )

    try:
        result = llm.invoke([
            SystemMessage(content=AVALOKA_SYSTEM_PROMPT + "\n\n" + DIRECT_REPLY_PROMPT + "\n\n" + conversation_directive),
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

    # An "infrastructure" intent no longer pre-sets infrastructure_request: that
    # sent any message the classifier filed there straight to provision_infra.
    # The planner asks for confirmation of an explicit request first
    # (app/agents/infra_confirmation.py).

    # A "schedule" intent is routed to the planner and nothing more. This used
    # to pre-set a hard-coded run-once task_schedule (relative, 0 seconds, one
    # run) without reading the sentence; the planner then saw a schedule
    # already present, skipped its LLM, and the router ended the turn with no
    # reply and no task. Every message containing "schedule", "every day",
    # "cron"... produced an empty turn. The planner's own schedule_task tool
    # is the only thing that can turn "every Monday 9am" into a cron.

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


def _results_in_hand(state: Dict[str, Any]) -> bool:
    """Whether the reply prompt will carry verified results to answer from.

    ``avaloka_recommendation`` is forwarded into the reply as
    ``<recommendation>``; its ``results`` are verified analysis results for the
    connected dataset. The execution profile written beside them (fidelity,
    mode, ETA) is not a result and does not count.
    """
    rec = state.get("avaloka_recommendation")
    return isinstance(rec, dict) and bool(rec.get("results"))


# ---------------------------------------------------------------------------
# New work with no data bound
# ---------------------------------------------------------------------------

#: Intents whose work runs on a dataset. Delegated with none bound, the
#: planner wrote code against no input, the run failed, and the user read "the
#: analysis code stopped with an error". Transfer, infrastructure, schedules,
#: status and inference name their own targets and are not stopped here. A
#: correction re-plans work on data, so it needs data too.
_DATA_INTENTS = frozenset({"statistical_analysis", "visualization", "ml_training",
                           "sampling_mode", "correction"})

#: State that says a data source is bound to this conversation: an upload, a
#: bucket path, a database connection, a dataset id.
_SOURCE_KEYS = (
    "data_source_location", "data_source_location_cloud", "data_source_location_local",
    "full_data_source_location_cloud", "full_data_location", "sample_data_location",
    "connection_id", "active_dataset_id", "active_dataset_ids", "datasets_context",
    "dataset_id", "uploaded_csv_columns", "db_credential_handles", "source_table",
    "dataset_size_bytes", "file_size_bytes",
)

#: A request that names where its data is ("from s3://…", "the orders table",
#: "load titanic from kaggle") is the planner's to fetch; it is not stopped.
_NAMES_A_SOURCE = re.compile(
    r"://|\bkaggle\b|\.(?:csv|tsv|parquet|xlsx?|json|avro)\b|"
    r"\b(?:database|table|bucket|s3|gcs|postgres\w*|mysql|snowflake|bigquery|sqlite)\b|"
    # A SQL statement names its own table: "SELECT ... FROM orders".
    r"\bselect\b[\s\S]*?\bfrom\s+[a-z_\"`\[]", re.I)

#: Column-like names the user wrote (snake_case or `quoted`), to say which data.
_COLUMN_NAME = re.compile(r"`([^`]{1,64})`|\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")

#: A prior artifact the request refers to ("the cleaned dataset", "the filtered
#: table", "those results"): with nothing connected it does not exist, and the
#: reply says so rather than leaving the reference hanging.
_PRIOR_ARTIFACT = re.compile(
    r"\b(?:the|that|this)\s+(?P<name>(?:[a-z][a-z\-]*\s+){1,2}?)"
    r"(?P<kind>dataset|data|table|frame|results?|output|sample)\b", re.I)
_ARTIFACT_FILLER = frozenset({"entire", "whole", "full", "same", "current", "new", "original",
                              "connected", "uploaded", "raw", "underlying"})


def _data_bound(state: Dict[str, Any], redis_ctx: Dict[str, Any]) -> bool:
    if any(state.get(k) for k in _SOURCE_KEYS):
        return True
    return bool(redis_ctx.get("dataset_id") or redis_ctx.get("schema_columns")
                or redis_ctx.get("connection_id"))


def _needs_data(intent: IntentResult, user_text: str, state: Dict[str, Any],
                redis_ctx: Dict[str, Any]) -> bool:
    """New work on a dataset, asked with none bound and none named."""
    new_work = intent.intent in _DATA_INTENTS or (
        intent.intent == "clarification" and not asks_about_results(user_text))
    return (new_work and not _data_bound(state, redis_ctx)
            and _NAMES_A_SOURCE.search(user_text or "") is None)


def _needs_data_reply(messages: List[BaseMessage]) -> str:
    """What is needed and how to provide it, in plain words. Written by code:
    every word of it is true whatever the request was, and nothing in it is a
    guess about data the conversation does not have."""
    said = " ".join(own_words(m.content) for m in messages
                    if isinstance(m, HumanMessage) and isinstance(m.content, str))
    columns: List[str] = []
    for quoted, snake in _COLUMN_NAME.findall(said):
        name = quoted or snake
        if name and name not in columns:
            columns.append(name)
    which = (f"a table with the columns {', '.join(columns[:-1])} and {columns[-1]}"
             if len(columns) > 1 else
             f"a table with the column {columns[0]}" if columns else
             "the data this request is about")
    current = next((own_words(m.content) for m in reversed(messages)
                    if isinstance(m, HumanMessage) and isinstance(m.content, str)), "")
    referred = ""
    for m in _PRIOR_ARTIFACT.finditer(current):
        words = [w for w in m.group("name").split() if w.lower() not in _ARTIFACT_FILLER]
        if words:
            # "the cleaned dataset": the earlier step it names never ran here.
            referred = (f" The {' '.join(words)} {m.group('kind')} you refer to does not exist "
                        "in this conversation: no data was connected, so no earlier step ran on any.")
            break
    return (
        "There is no dataset connected to this conversation yet, so I haven't run "
        f"anything.{referred} To do this I need {which}.\n\n"
        "You can provide it by uploading a file (CSV, Parquet, Excel or JSON), by "
        "giving a bucket path such as s3://… or gs://…, or by connecting a database. "
        "As soon as it is connected I'll say what I'm about to run and how long it "
        f"should take -- a job like this usually runs in {LOCAL_ETA_TEXT} -- then run it."
    )


def _correction_reasoning(state: Dict[str, Any], facts: CorrectionFacts) -> str:
    """The reasoning for a re-plan: the model's phrasing if it keeps to the
    facts code wrote, otherwise code's own words for the same facts."""
    llm = _get_reply_llm()
    if llm is None:
        return reasoning_text(facts)
    from app.agents.avaloka_agent.prompts import CORRECTION_REASONING_PROMPT
    payload = (
        f"<correction>\n<original>{facts.original}</original>\n"
        f"<corrected>{facts.corrected}</corrected>\n<subject>{facts.subject}</subject>\n"
        f"</correction>\n"
        f"<conversation_so_far>\n{_dialogue_before(state.get('messages') or [])}\n"
        f"</conversation_so_far>"
    )
    try:
        result = llm.invoke([SystemMessage(content=CORRECTION_REASONING_PROMPT),
                             HumanMessage(content=payload)])
        candidate = getattr(result, "content", "") or ""
    except Exception as exc:                          # noqa: BLE001 - code's words instead
        logger.warning("avaloka_agent: correction reasoning LLM failed: %s", exc)
        return reasoning_text(facts)
    known = [str(state.get("avaloka_recommendation") or "")] + [
        own_words(m.content) for m in (state.get("messages") or [])
        if isinstance(m, HumanMessage) and isinstance(m.content, str)]
    if isinstance(candidate, str) and phrasing_holds(candidate, facts, known):
        return candidate.strip()
    logger.info("avaloka_agent: correction phrasing did not keep to the facts; using code's words")
    return reasoning_text(facts)


def _results_ambiguity(state: Dict[str, Any], user_text: str) -> Optional[Dict[str, Any]]:
    """The clarifying question for an ambiguous ask about results in hand, or None.

    The planner's own measure check (``_detect_metric_ambiguity``) decides it,
    over the measures the verified results name: "What was revenue last
    quarter?" against gross_revenue and net_revenue is ambiguous for the same
    reason it is against a table with those two columns. When it fires the
    question is the check's own text -- "Which revenue measure do you mean:
    gross revenue or net revenue?" -- written by code, and no model is asked
    to write the reply, so neither candidate figure can reach it. Recorded on
    gpt-oss, a prompt rule alone was not enough: the model showed both figures
    and then asked which.
    """
    rec = state.get("avaloka_recommendation")
    results = rec.get("results") if isinstance(rec, dict) else None
    if not isinstance(results, dict):
        return None
    measures = {str(k): "float" for k, v in results.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if len(measures) < 2:
        return None
    from app.agents.planner import _detect_metric_ambiguity
    return _detect_metric_ambiguity(user_text, {"schema": measures,
                                                "avaloka_intent": state.get("avaloka_intent")})


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

    expertise = infer_expertise(
        messages,
        state.get("avaloka_expertise") or redis_ctx.get("avaloka_expertise"),
    )
    conversation_policy = policy_for(expertise)
    conversation_directive = build_conversation_directive(conversation_policy)
    # 2. Connection event detection: state takes precedence over redis
    connection_event = state.get("connection_event") or redis_ctx.get("connection_event")
    has_connection_event = bool(connection_event)

    # 3. Classify intent (uses Groq if available, heuristic otherwise)
    session_summary = render_for_prompt(redis_ctx)
    intent = classify_intent(
        messages,
        has_connection_event=has_connection_event,
        session_summary=session_summary,
        results_in_hand=_results_in_hand(state),
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
    elif intent.intent in ("exploration", "chit_chat", "clarification", "discussion"):
        avaloka_mode = "exploring"
    else:
        avaloka_mode = "analyzing"

    # 6. Decide delegation. Data-touching intents and ambiguous
    #    ("clarification") messages both delegate to plan_etl, which has
    #    the richer toolset for asking follow-ups. Direct replies are
    #    reserved for clearly conversational intents.
    pending = state.get("pending_clarification")
    answering_planner = isinstance(pending, dict) and pending.get("type") in {
        "filter_value", "metric_choice", "infrastructure_confirmation",
        # The replan loop's question (app/agents/replan_choice.py).
        "replan_measured_choice",
    }
    delegate = intent.should_delegate() or answering_planner
    needs_data = (delegate and not answering_planner
                  and _needs_data(intent, user_text, dict(state), redis_ctx))
    if needs_data:
        # Upstream of the planner: no code is planned, written or run with no
        # data to run it on. The reply says what is needed and how to give it.
        delegate = False
        avaloka_mode = "needs_data"

    # 7. Build the state update bundle
    updates: Dict[str, Any] = {
        "avaloka_mode": avaloka_mode,
        "avaloka_intent": intent.intent,
        "avaloka_intent_reasoning": (getattr(intent, "reasoning", "") or None),
        "avaloka_intent_confidence": getattr(intent, "confidence", None),
        # The sealed credential handle is for discovery only; it does not
        # travel in graph state.
        "redis_context": {k: v for k, v in redis_ctx.items() if k != "db_credential_handle"} or None,
        "discovery_result": discovery,
        "delegate_to_planner": bool(delegate),
        "avaloka_expertise": expertise.as_dict(),
        "avaloka_conversation_policy": conversation_policy.as_dict(),
        "conversation_guidance": conversation_directive,
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
        elif rec.get("eta_text"):
            # The profile's own estimate for a local run: a fact code wrote.
            handoff = f"{handoff} Expect {rec['eta_text']}."
        # The handoff is said in the chat: it is the line that tells the user
        # what is about to happen before a long job, and kept only in this
        # field it reached the trace and nobody else. It goes in as commentary
        # after the user's message; the planner reads the latest human turn by
        # searching back from the end, so the request it plans is unchanged.
        updates["avaloka_handoff_message"] = handoff
        updates["messages"] = messages + [_rs.commentary_message(content=handoff)]
        if intent.intent == "correction":
            # A correction re-plans, and the user is told why before it runs:
            # the facts are code's (correction_facts), the phrasing is checked.
            facts = correction_facts(user_text)
            if facts is not None:
                reasoning = _correction_reasoning(dict({**state, **updates}), facts)
                updates["correction"] = facts.as_dict()
                updates["avaloka_handoff_message"] = reasoning
                updates["messages"] = messages + [_rs.commentary_message(content=reasoning)]
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
            updates.setdefault("messages", messages)
            updates["pending_suggestions"] = None
        return updates

    if needs_data:
        said = []
        facts = correction_facts(user_text) if intent.intent == "correction" else None
        if facts is not None:
            # A correction is still re-planned in words, then told what it needs.
            said.append(_correction_reasoning(dict({**state, **updates}), facts))
            updates["correction"] = facts.as_dict()
        said.append(_needs_data_reply(messages))
        updates["messages"] = messages + [_rs.commentary_message(content=t) for t in said]
        return updates

    # Direct reply path
    ambiguity = _results_ambiguity(dict({**state, **updates}), user_text) \
        if intent.intent == "discussion" else None
    if ambiguity is not None:
        # Asked in code's words; the model does not write this turn at all.
        updates["messages"] = messages + [_rs.commentary_message(content=ambiguity["message"])]
        updates["pending_clarification"] = ambiguity["pending_clarification"]
        return updates
    reply_text = _render_direct_reply(
        dict({**state, **updates}),
        intent,
        redis_ctx,
        discovery,
        avaloka_mode,
        conversation_directive,
    )
    updates["messages"] = messages + [_rs.commentary_message(content=reply_text)]
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
