# app/services/memory_plane.py
#
# Environment variables consumed here:
#   MEMORY_DEFAULT_STYLE_HINT        – Fallback style hint when LLM/DB unavailable
#                                      (default: "Prefer clean pandas code.")
#   MEMORY_CIRCUIT_BREAKER_TIMEOUT   – Hard timeout (s) for the retrieval thread
#                                      (default: 20.0 -- a successful retrieval
#                                      measures 4-6s; 3.0 aborted every call)
#   MEMORY_MCP_FALLBACK_URL          – MCP server URL when active_mcp_servers is empty
#                                      Reads MCP_SERVER_URL first, then this var
#                                      (default: http://localhost:8080/sse)
#   MCP_API_KEY                      – Auth token sent to the MCP server
#   REDIS_SCHEMA_TTL                 – TTL (s) used when hot-loading schema into Redis
#                                      (default: 86400)
#   MILVUS_TOP_K                     – Episodic memory top-k for similarity search
#                                      (default: 5)

import os
import json
import logging
import threading
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from cachetools import TTLCache

from app.core.inference import build_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from app.graph.etl_state import ETLState
from app.services.memory_runtime import validate_memory_runtime_config
from app.core.model_config import resolve as resolve_model
from app.core.model_fallback import attach_fallback

logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, ".env"))

# --- Module-level tuneable constants (all overridable via env) ---
_DEFAULT_STYLE_HINT       = os.environ.get("MEMORY_DEFAULT_STYLE_HINT",       "Prefer clean pandas code.")
# 3.0s was shorter than a successful retrieval takes, so the breaker fired on
# EVERY call and the memory plane never returned anything. Measured against a
# healthy deployment (Chroma + Milvus both connected, embedding model warm):
# 5.8s cold, 4.4s warm. The loop was not broken -- it was being killed by its
# own timeout, and the only symptom was memory_hints=None, which looks
# identical to "nothing has been learned yet".
#
# 20s leaves headroom for a cold embedding load without letting a genuinely
# hung backend stall a turn. Lower it once the model is pre-warmed in the image.
_CIRCUIT_BREAKER_TIMEOUT  = float(os.environ.get("MEMORY_CIRCUIT_BREAKER_TIMEOUT", "20.0"))
# MCP_SERVER_URL is the canonical .env name; MEMORY_MCP_FALLBACK_URL is the override
_MCP_FALLBACK_URL         = (
    os.environ.get("MCP_SERVER_URL")
    or os.environ.get("MEMORY_MCP_FALLBACK_URL")
    or "http://localhost:8080/sse"
)
_REDIS_SCHEMA_TTL         = int(os.environ.get("REDIS_SCHEMA_TTL",            "86400"))
_MILVUS_TOP_K             = int(os.environ.get("MILVUS_TOP_K",                "5"))
# Durable context-memory TTL (Redis write-through of session hints/preferences).
# Defaults to the session lifetime (7 days) so a remembered preference lives at
# least as long as the session it belongs to.
_CTX_MEMORY_TTL           = int(os.environ.get("MEMORY_PREFERENCES_TTL_SECONDS", str(7 * 24 * 3600)))


def _k_ctxmem_hints(session_id: str) -> str:
    return f"ctxmem:{session_id}:hints"


def _k_ctxmem_prefs(session_id: str) -> str:
    return f"ctxmem:{session_id}:prefs"


def _ensure_sse_url(url: str) -> str:
    """
    Normalize an MCP URL to its SSE endpoint.

    MCP_SERVER_URL is the canonical BASE form elsewhere (server.py/settings.py
    append `/call_tool`), but the SSE transport used here needs the `/sse`
    endpoint. Appending it when missing lets operators set the canonical base
    form without memory-plane connecting to the wrong path. Idempotent for URLs
    that already end in `/sse`.
    """
    if not url:
        return url
    trimmed = url.rstrip("/")
    return trimmed if trimmed.endswith("/sse") else trimmed + "/sse"

_api_key = os.environ.get("GROQ_API_KEY_PLANNING_AGENT") or os.environ.get("GROQ_API_KEY_CODING_AGENT")

memory_llm = build_chat_model(
    role="planning",
    agent="MEMORY",
    tier="small",
    temperature=0.1,
    groq_model=resolve_model("memory"),
    groq_api_key=_api_key,
)
if memory_llm is None:
    logger.warning("Memory LLM disabled; missing API key.")

class MemoryOrchestrator:
    """
    Orchestrates memory storage and retrieval for the Avaloka ETL workflow
    across four persistent tiers: Redis (L1), ChromaDB (L2), Postgres (L3), Milvus (L4).
    """

    # Simple list of past queries for LLM context, since this doesn't strictly
    # belong in the heavy DBs unless using a formal ConversationBuffer DB.
    _past_queries_store: TTLCache = TTLCache(maxsize=10000, ttl=86400)
    _session_hints_store: TTLCache = TTLCache(maxsize=10000, ttl=86400)
    # Preferences the user EXPLICITLY asked to remember (store_user_preference
    # tool). Kept separately so the top-3 hint contract can reserve slots for
    # them — ordinary hint churn must never evict an explicit preference.
    _session_preferences_store: TTLCache = TTLCache(maxsize=10000, ttl=86400)
    # Guards the in-process stores: the planner thread writes preferences while
    # retrieve_memory's worker thread reads/appends hints.
    _store_lock = threading.Lock()

    def __init__(self, llm_client=None):
        self._explicit_llm = llm_client
        validate_memory_runtime_config()

        # Initialize Layer 1-4 Database Clients
        from app.services.db.redis_client import redis_client
        from app.services.db.chroma_client import chroma_client
        from app.services.db.postgres_client import postgres_client
        from app.services.db.milvus_client import milvus_client

        self.redis = redis_client
        self.chroma = chroma_client
        self.postgres = postgres_client
        self.milvus = milvus_client

    @property
    def llm(self):
        """Dynamically resolve LLM to support Pytest module patching"""
        if self._explicit_llm is not None:
             return self._explicit_llm
        import app.services.memory_plane as mp
        return mp.memory_llm

    @classmethod
    def clear_all_sessions(cls):
        """
        TEST-ONLY helper: reset all in-process stores AND the durable
        context-memory copies (every `ctxmem:*` key in Redis — required for
        test isolation now that preferences write through to Redis).

        WARNING: do not point REDIS_URL at a shared/production Redis while
        running the test suite — this wipes all users' remembered
        preferences there. Other live infra data (schema cache, Chroma,
        Postgres, Milvus) is not affected.
        """
        from app.services.db.redis_client import redis_client
        from app.services.db.chroma_client import chroma_client
        from app.services.db.postgres_client import postgres_client

        if hasattr(redis_client,    "_fallback_store"):       redis_client._fallback_store.clear()
        if hasattr(chroma_client,   "_fallback_namespaces"): chroma_client._fallback_namespaces.clear()
        if hasattr(postgres_client, "_mock_table"):          postgres_client._mock_table.clear()
        cls._past_queries_store.clear()
        cls._session_hints_store.clear()
        cls._session_preferences_store.clear()
        cls._hydration_failed.clear()
        # Also drop the durable write-through copies, otherwise hydration
        # would resurrect cleared sessions on the next retrieval.
        try:
            redis_client.delete_prefix("ctxmem:")
        except Exception:
            pass

    def _embed(self, text: str) -> Optional[List[float]]:
        """
        Embed text to the Milvus vector dimension so L4 inserts/searches run
        against real vectors, not the zero vector (which is degenerate under a
        COSINE index). Returns None on failure so the client keeps its existing
        zero-vector fallback rather than raising inside a memory operation.
        """
        try:
            from app.services.embedding_utils import embed_text
            return embed_text(text, target_dim=self.milvus._vector_dim)
        except Exception as e:
            logger.warning(f"Memory embedding failed ({e}); Milvus op will fall back to zero-vector.")
            return None

    def _extract_context_with_llm(self, query: str, past_queries: List[str]) -> Dict:
        """Uses the LLM to extract context and preferences from queries."""
        if not self.llm:
            return {"new_hints": [], "logic_signature": ""}

        system_prompt = """You are the Memory Context engine for an ETL application.
Analyze the user's current query and compare it to their past queries.
Extract any recurring preferences, business logic rules, or specific dataset contextual facts they seem to care about.
Return your findings strictly as a JSON object with two keys:
{"new_hints": ["string array of discovered facts"], "logic_signature": "a short sentence summarizing technical style preferences"}
If the query is too simple to extract anything meaningful, return {"new_hints": [], "logic_signature": ""}
"""
        human_prompt = f"Past Queries: {past_queries}\n\nCurrent Query: {query}\n\nExtract memory context as JSON."

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])

            content = response.content.strip()
            start_idx = content.find('{')
            end_idx = content.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                json_str = content[start_idx:end_idx+1]
                return json.loads(json_str)
            return {}

        except Exception as e:
            logger.error(f"Failed to dynamically extract memory context: {e}")
            return {}

    def _retrieve_memory_internal(self, query: str, session_id: str = "default", dataset_id: str = "unknown", state: Dict = None) -> Dict:
        """
        Internal implementation of Tiered Memory bounds for a session.
        """
        if state is None:
            state = {}
        user_id = state.get("user_id", "default")

        circuit_breaker_triggered = False

        # Restore durable session memory (hints/preferences) after a process
        # restart so remembered context survives redeploys.
        self._hydrate_session_memory(session_id)

        if session_id not in self._past_queries_store:
            self._past_queries_store[session_id] = []

        past_queries = self._past_queries_store[session_id].copy()
        self._past_queries_store[session_id].append(query)

        new_hints_this_context = []
        accumulated_hints = self._session_hints_store.get(session_id, []).copy()
        # Provenance buckets for the top-3 contract: every source except "llm"
        # is query-filtered at its origin, so the tier a hint came from is a
        # reliable relevance signal (see _compose_top_hints).
        tiered_hints: Dict[str, List[str]] = {"session": list(accumulated_hints)}

        def _add_hint(tier: str, hint: str) -> bool:
            if hint in accumulated_hints:
                return False
            accumulated_hints.append(hint)
            tiered_hints.setdefault(tier, []).append(hint)
            return True

        logic_sig = _DEFAULT_STYLE_HINT
        prior_artifact_found = False

        try:
            self.postgres.connect()
            self.redis.connect()
            self.chroma.connect()
            self.milvus.connect()

            # Layer 3: Artifact Context
            # Surfaces rich metadata from prior runs so the Planner can
            # actively reuse chart URLs / model metrics and skip redundant work.
            if self.postgres.check_artifact_exists(query, user_id):
                prior_artifact_found = True
                prior_meta = self.postgres.get_artifact(query, user_id)
                if prior_meta:
                    hint_parts = []

                    if prior_meta.get("status"):
                        hint_parts.append(f"status={prior_meta['status']}")

                    # Chart / visualisation URLs — reuse instead of regenerating
                    for url_field in ("chart_url", "chart_urls", "plot_url",
                                      "visualization_url", "dashboard_url"):
                        val = prior_meta.get(url_field)
                        if val:
                            if isinstance(val, list):
                                hint_parts.append(f"chart_urls={val[:3]}")
                            else:
                                hint_parts.append(f"chart_url={val}")
                            break

                    # Model / evaluation metrics — skip retraining if acceptable
                    for metric_field in ("model_metrics", "metrics", "eval_metrics",
                                        "accuracy", "f1_score", "rmse", "mape"):
                        val = prior_meta.get(metric_field)
                        if val:
                            hint_parts.append(f"metrics={val}")
                            break

                    # Output artifact path — reuse directly
                    for path_field in ("output_path", "output_file", "artifact_path",
                                       "file_data", "model_path"):
                        val = prior_meta.get(path_field)
                        if val:
                            hint_parts.append(f"output={val}")
                            break

                    # Session reference for traceability
                    if prior_meta.get("session"):
                        hint_parts.append(f"session={prior_meta['session']}")

                    if hint_parts:
                        _add_hint(
                            "artifact",
                            "[L3] Prior artifact — reuse to skip redundant work: "
                            + ", ".join(hint_parts)
                        )
                    else:
                        # Metadata exists but has no field we recognise — surface raw summary
                        _add_hint(
                            "artifact",
                            f"[L3] Prior artifact detected for this query "
                            f"({len(prior_meta)} metadata fields stored)."
                        )

            # Layer 1: Domain Context — semantic schema cache lookup
            # Uses search_schema_semantically so the query text drives cache hit selection
            schema_info = self.redis.search_schema_semantically(user_id, query)
            if schema_info:
                _add_hint("domain", f"Domain insight: {schema_info}")
            elif dataset_id and dataset_id not in ("unknown", "unknown_dataset"):
                # Cache miss — attempt dynamic MCP hot-load. Isolated in its
                # own try/except: a missing `mcp` package or an unreachable
                # MCP server must degrade to "no schema insight", NOT trip
                # the outer circuit breaker and discard every other memory
                # layer's hints (stored user preferences included).
                try:
                    from app.services.mcp_cache_loader import fetch_schema_sync
                    active_servers = state.get("active_mcp_servers", [])
                    mcp_url = _ensure_sse_url(
                        active_servers[0]
                        if active_servers
                        else (
                            os.environ.get("MCP_SERVER_URL")
                            or os.environ.get("MCP_URL")
                            or _MCP_FALLBACK_URL
                        )
                    )

                    # Prefer the tenant's own MCP key from state — the MCP maps
                    # key->customer, so using a single global MCP_API_KEY would
                    # return that one customer's schema for every tenant. Fall
                    # back to the global env key only for single-tenant setups.
                    api_key = state.get("mcp_api_key") or os.environ.get("MCP_API_KEY", "")

                    logger.info(f"Triggering MCP extraction for dataset: {dataset_id} via server: {mcp_url}")
                    schema_payload = fetch_schema_sync(mcp_url, api_key, dataset_id)

                    if schema_payload and "error" not in schema_payload:
                        # Persist into Layer 1 Redis for subsequent requests
                        self.redis.set_schema(user_id, dataset_id, schema_payload, ttl_seconds=_REDIS_SCHEMA_TTL)
                        _add_hint("domain", f"Domain insight (hot-loaded via MCP): {json.dumps(schema_payload)}")
                except Exception as mcp_exc:
                    logger.warning(f"MCP schema hot-load skipped (non-fatal): {mcp_exc}")

            user_sig = self.chroma.get_user_signature(session_id)
            if user_sig:
                logic_sig = user_sig

            recent_insights = self.milvus.search_similar_insights(
                session_id, query_vector=self._embed(query), top_k=_MILVUS_TOP_K
            )
            for record in recent_insights:
                _add_hint("similar", record["content"])

            # LLM Dynamic Extraction (Populates DBs for future queries)
            if self.llm:
                extracted_memory = self._extract_context_with_llm(query, past_queries)

                raw_new_hints = extracted_memory.get("new_hints", [])
                for hint in raw_new_hints:
                    new_hints_this_context.append(hint)
                    # Insert new episodic memory into Layer 4 (Milvus)
                    self.milvus.insert_insight(session_id, hint, "memory_llm", vector=self._embed(hint))
                    if _add_hint("llm", hint):
                        self._session_hints_store.setdefault(session_id, []).append(hint)

                new_logic_sig = extracted_memory.get("logic_signature")
                if new_logic_sig:
                    logic_sig = new_logic_sig
                    self.chroma.update_user_signature(session_id, logic_sig)

                if new_hints_this_context:
                    self._persist_session_memory(session_id)

            # --- Top-3 Summary Contract ---
            # Ensures the Planner does not receive token overflow.
            # Explicit user preferences hold reserved slots so ordinary hint
            # churn can never evict them; leftover slots fill by relevance
            # tier (artifact > domain > similar > llm > recent session hints).
            top_3_hints = self._compose_top_hints(session_id, tiered_hints)

        except Exception as e:
            logger.error(f"Memory Circuit Breaker Triggered: {e}")
            circuit_breaker_triggered = True
            top_3_hints = []
            accumulated_hints = []
            logic_sig = _DEFAULT_STYLE_HINT

        final_unavailable = True if not self.llm or circuit_breaker_triggered else False

        return {
            "new_hints_this_context": new_hints_this_context,
            "accumulated_memory_hints": accumulated_hints,
            "memory_hints": top_3_hints, # Enforcing Top-3 Contract for Planner Agent
            "prior_artifact_found": prior_artifact_found,
            "session_logic_signature": logic_sig,
            "memory_context_unavailable": final_unavailable
        }


    def store_user_preference(self, preference: str, session_id: str = "default") -> bool:
        """
        Persist an explicitly stated user preference (e.g. "The user's favorite
        column is 'reordered'") so future retrievals surface it as a hint.

        This is the direct write path behind the planner's `store_user_preference`
        tool — unlike `_extract_context_with_llm`, it requires no LLM and never
        raises. The preference goes into the session stores that retrieval
        serves from (with reserved top-3 slots, see `_compose_top_hints`); the
        Layer 4 Milvus write is best-effort in a background thread so a slow or
        down Milvus can never block the planner turn. Returns True when the
        session-store write succeeded — the tier retrieval actually reads.
        """
        preference = (preference or "").strip()
        if not preference:
            return False

        # Restore any previously persisted memory first so a fresh process
        # can't overwrite durable preferences with a partial list.
        self._hydrate_session_memory(session_id)

        stored = False
        is_new = False
        try:
            with self._store_lock:
                session_hints = self._session_hints_store.setdefault(session_id, [])
                if preference not in session_hints:
                    session_hints.append(preference)
                    is_new = True
                prefs = self._session_preferences_store.setdefault(session_id, [])
                if preference in prefs:
                    prefs.remove(preference)
                prefs.append(preference)  # newest last — wins a reserved slot
            stored = True
        except Exception as e:
            logger.error(f"Failed to store user preference in session store: {e}")

        if stored:
            # Write-through to Redis so the preference survives restarts.
            self._persist_session_memory(session_id)

        if is_new:
            threading.Thread(
                target=self._persist_preference_to_milvus,
                args=(session_id, preference),
                daemon=True,
            ).start()

        if stored:
            logger.info(f"🧠 AVALOKA CONTEXT MEMORY STORED (session={session_id}): {preference}")
        return stored

    def _persist_preference_to_milvus(self, session_id: str, preference: str) -> None:
        """Best-effort durable write for an explicit preference. Never raises."""
        try:
            self.milvus.insert_insight(session_id, preference, "user_explicit", vector=self._embed(preference))
        except Exception as e:
            logger.error(f"Failed to persist user preference to Milvus: {e}")

    # Sessions whose last durable read FAILED (as opposed to finding nothing).
    # While a session is in this set, _persist_session_memory refuses to
    # overwrite Redis until a read succeeds — a transient blip at restart
    # must never let an empty in-process view erase stored preferences.
    _hydration_failed: set = set()

    def _hydrate_session_memory(self, session_id: str) -> None:
        """
        Restore a session's hints/preferences from Redis into the in-process
        stores after a process restart. Each store is hydrated independently
        (their TTLCache entries expire at different times, so one may be
        present while the other is missing). In-process entries stay
        authoritative while they exist. Never raises.
        """
        need_hints = session_id not in self._session_hints_store
        need_prefs = session_id not in self._session_preferences_store
        if not (need_hints or need_prefs):
            return
        try:
            hints = self.redis.get_json(_k_ctxmem_hints(session_id), strict=True) if need_hints else None
            prefs = self.redis.get_json(_k_ctxmem_prefs(session_id), strict=True) if need_prefs else None
        except Exception as e:
            with self._store_lock:
                self._hydration_failed.add(session_id)
            logger.warning(f"Context-memory hydration FAILED for session {session_id} (will not overwrite durable copy): {e}")
            return
        restored = False
        with self._store_lock:
            self._hydration_failed.discard(session_id)
            if need_hints and isinstance(hints, list) and session_id not in self._session_hints_store:
                self._session_hints_store[session_id] = [h for h in hints if isinstance(h, str)]
                restored = True
            if need_prefs and isinstance(prefs, list) and session_id not in self._session_preferences_store:
                self._session_preferences_store[session_id] = [p for p in prefs if isinstance(p, str)]
                restored = True
        if restored:
            logger.info(f"🧠 AVALOKA CONTEXT MEMORY RESTORED from Redis (session={session_id})")

    def _persist_session_memory(self, session_id: str) -> None:
        """
        Write-through the session's hints/preferences to Redis. Never raises.

        Snapshot and writes happen under _store_lock so a concurrent
        store_user_preference can't interleave a newer preference between our
        snapshot and our write (last-writer-wins would drop it). If the last
        durable read for this session failed, retry it first — and skip the
        write entirely while Redis still can't be read.
        """
        try:
            if session_id in self._hydration_failed:
                self._hydrate_session_memory(session_id)
                if session_id in self._hydration_failed:
                    logger.warning(
                        f"Skipping context-memory write-through for session {session_id}: durable copy unreadable"
                    )
                    return
            with self._store_lock:
                hints = list(self._session_hints_store.get(session_id, []))
                prefs = list(self._session_preferences_store.get(session_id, []))
                self.redis.set_json(_k_ctxmem_hints(session_id), hints, ttl_seconds=_CTX_MEMORY_TTL)
                self.redis.set_json(_k_ctxmem_prefs(session_id), prefs, ttl_seconds=_CTX_MEMORY_TTL)
        except Exception as e:
            logger.warning(f"Context-memory write-through failed for session {session_id}: {e}")

    # Relevance order for the non-preference top-3 slots. "artifact" (L3 hit
    # for this exact query), "domain" (semantic schema match) and "similar"
    # (vector-ranked Milvus results) are query-filtered at their origin;
    # "llm" extractions are unranked and must never evict them. Session
    # history fills last, newest first.
    _HINT_TIER_ORDER = ("artifact", "domain", "similar", "llm")

    def _compose_top_hints(self, session_id: str, hints: Any, limit: int = 3) -> List[str]:
        """
        Top-3 contract with reserved slots: explicit user preferences (newest
        first, up to `limit`) always make the cut; remaining slots are filled
        with the most RELEVANT ordinary hints, not the most recently appended.

        `hints` is either a provenance dict ({tier: [hint, ...]}) — slots fill
        in `_HINT_TIER_ORDER`, leftover slots taking the most recent "session"
        hints — or a flat list, which is treated entirely as session history
        and degrades to the original `accumulated_hints[-limit:]` behavior.
        """
        tiered = hints if isinstance(hints, dict) else {"session": list(hints)}
        prefs = list(self._session_preferences_store.get(session_id, []))[-limit:]
        slots = limit - len(prefs)
        if slots <= 0:
            return prefs
        fill: List[str] = []
        for tier in self._HINT_TIER_ORDER:
            for hint in tiered.get(tier, ()):
                if hint not in prefs and hint not in fill:
                    fill.append(hint)
        fill = fill[:slots]
        remaining = slots - len(fill)
        if remaining > 0:
            session_pool = [h for h in tiered.get("session", ())
                            if h not in prefs and h not in fill]
            fill.extend(session_pool[-remaining:])
        return fill + prefs

    def retrieve_memory(self, query: str, session_id: str = "default", dataset_id: str = "unknown", state: Dict = None) -> Dict:
        """
        Wrap _retrieve_memory_internal in a hard timeout (MEMORY_CIRCUIT_BREAKER_TIMEOUT seconds).
        Returns a safe empty payload if exceeded so LangGraph nodes are never blocked.
        """
        _empty = {
            "new_hints_this_context":  [],
            "accumulated_memory_hints": [],
            "memory_hints":            [],
            "prior_artifact_found":    False,
            "session_logic_signature": _DEFAULT_STYLE_HINT,
            "memory_context_unavailable": True,
        }
        # NOTE: a daemon thread + Event, deliberately NOT a ThreadPoolExecutor.
        # A `with ThreadPoolExecutor(...)` block calls shutdown(wait=True) on
        # exit, which JOINS the worker — a hung lookup kept blocking the caller
        # for its full duration even after TimeoutError fired. And even with
        # shutdown(wait=False), executor threads are non-daemon and are joined
        # at interpreter exit, so one hung lookup made the whole process hang
        # on shutdown. A daemon thread is simply abandoned in both cases: the
        # caller is released at the timeout and process exit stays clean.
        result_box: Dict[str, Any] = {}
        done = threading.Event()

        def _worker() -> None:
            try:
                result_box["result"] = self._retrieve_memory_internal(query, session_id, dataset_id, state)
            except Exception as exc:  # noqa: BLE001 - breaker must never raise
                result_box["error"] = exc
            finally:
                done.set()

        threading.Thread(target=_worker, name="memory-retrieval", daemon=True).start()

        if not done.wait(timeout=_CIRCUIT_BREAKER_TIMEOUT):
            logger.error(
                f"Memory Circuit Breaker: retrieval exceeded {_CIRCUIT_BREAKER_TIMEOUT}s. "
                "Returning empty payload. Tune MEMORY_CIRCUIT_BREAKER_TIMEOUT if needed."
            )
            return _empty

        if "error" in result_box:
            logger.error(f"Memory Circuit Breaker (execution error): {result_box['error']}")
            return _empty

        result = result_box["result"]

        # Highly visible log for demoing to senior leadership
        hints = result.get("accumulated_memory_hints", [])
        if hints:
            logger.info("==================================================")
            logger.info(f"🧠 AVALOKA CONTEXT MEMORY INJECTED ({len(hints)} items):")
            for h in hints:
                logger.info(f"  -> {h}")
            logger.info("==================================================")

        return result


# Global orchestrator instance for Phase 3 backward compatibility in LangGraph
_orchestrator = MemoryOrchestrator(llm_client=None)

def retrieve_memory(query: str, state: ETLState) -> Dict:
    """
    Phase 3 entry point: Delegates to the Tiered MemoryOrchestrator.
    Maintains LangGraph node interface contract safely passing global State dicts inside.
    """
    session_id = state.get("session_id", "default")
    # Determine dataset identity if available for Layer 1 lookup
    data_source = state.get("data_source_location", "")
    dataset_id = os.path.basename(data_source) if data_source else "unknown_dataset"

    return _orchestrator.retrieve_memory(query, session_id, dataset_id, state)


def store_user_preference(preference: str, state: ETLState) -> bool:
    """
    Entry point for the planner's `store_user_preference` tool: persists an
    explicitly requested preference for the session found in state. Never raises.
    """
    session_id = (state or {}).get("session_id", "default")
    return _orchestrator.store_user_preference(preference, session_id)

