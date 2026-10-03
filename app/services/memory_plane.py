# app/services/memory_plane.py
#
# Environment variables consumed here:
#   MEMORY_DEFAULT_STYLE_HINT        – Fallback style hint when LLM/DB unavailable
#                                      (default: "Prefer clean pandas code.")
#   MEMORY_CIRCUIT_BREAKER_TIMEOUT   – Overall ceiling (s) for one retrieval (default: 8.0)
#   MEMORY_STAGE_TIMEOUT             – Per-store timeout (s); a slow store is skipped,
#                                      the others are kept (default: 3.0)
#   MEMORY_EMBED_TIMEOUT             – Timeout (s) for embedding the query (default: 4.0)
#   MEMORY_CONNECT_TIMEOUT           – Timeout (s) for opening the four connections (default: 5.0)
#   MEMORY_MCP_SCHEMA_TIMEOUT        – Timeout (s) for the MCP schema hot-load (default: 3.0)
#   MEMORY_WARMUP                    – "0" disables connect + embedding warm-up at import
#                                      (set it in tests; default: on)
#   MEMORY_MCP_FALLBACK_URL          – MCP server URL when active_mcp_servers is empty
#                                      Reads MCP_SERVER_URL first, then this var
#                                      (default: http://localhost:8080/sse)
#   MCP_API_KEY                      – Auth token sent to the MCP server
#   REDIS_SCHEMA_TTL                 – TTL (s) used when hot-loading schema into Redis
#                                      (default: 86400)
#   MILVUS_TOP_K                     – Episodic memory top-k for similarity search
#                                      (default: 5)
#
# Latency design (why a turn used to take 20s here):
#   * All four stores were re-connected on every request -> now once per process.
#   * Every stage shared one 20s breaker, so one hung store discarded everything
#     and the log never said which -> now each stage has its own timeout and is
#     skipped on its own; "[memory] retrieval Xs stages={...}" logs every turn.
#   * LLM extraction + Milvus/Chroma writes ran inside the request although they
#     only feed FUTURE turns -> now a background thread.
#   * The query was embedded twice -> once.
#   * MCP schema hot-load ran for CSV uploads (dataset_id = file name, which the
#     MCP server does not know) -> database sources only.





import os
import json
import logging
import threading
import time
from typing import Callable, Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
from cachetools import TTLCache

from app.core.inference import build_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from app.graph.etl_state import ETLState
from app.services.memory_runtime import validate_memory_runtime_config
from app.core.model_config import resolve as resolve_model
from app.core.model_fallback import attach_fallback  # noqa: F401

logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, ".env"))

# --- Module-level tuneable constants (all overridable via env) ---
_DEFAULT_STYLE_HINT       = os.environ.get("MEMORY_DEFAULT_STYLE_HINT", "Prefer clean pandas code.")
# Overall ceiling for one retrieval. Each stage also has its own timeout, so a
# slow store is skipped instead of discarding everything. With connections held
# open and the embedding model warmed at startup, a normal retrieval is < 1s.
_CIRCUIT_BREAKER_TIMEOUT  = float(os.environ.get("MEMORY_CIRCUIT_BREAKER_TIMEOUT", "8.0"))
_STAGE_TIMEOUT            = float(os.environ.get("MEMORY_STAGE_TIMEOUT", "3.0"))
_EMBED_TIMEOUT            = float(os.environ.get("MEMORY_EMBED_TIMEOUT", "4.0"))
_CONNECT_TIMEOUT          = float(os.environ.get("MEMORY_CONNECT_TIMEOUT", "5.0"))
_MCP_SCHEMA_TIMEOUT       = float(os.environ.get("MEMORY_MCP_SCHEMA_TIMEOUT", "3.0"))
# MCP_SERVER_URL is the canonical .env name; MEMORY_MCP_FALLBACK_URL is the override
_MCP_FALLBACK_URL         = (
    os.environ.get("MCP_SERVER_URL")
    or os.environ.get("MEMORY_MCP_FALLBACK_URL")
    or "http://localhost:8080/sse"
)
_REDIS_SCHEMA_TTL         = int(os.environ.get("REDIS_SCHEMA_TTL", "86400"))
_MILVUS_TOP_K             = int(os.environ.get("MILVUS_TOP_K", "5"))
# Durable context-memory TTL (Redis write-through of session hints/preferences).
# Defaults to the session lifetime (7 days) so a remembered preference lives at
# least as long as the session it belongs to.
_CTX_MEMORY_TTL           = int(os.environ.get("MEMORY_PREFERENCES_TTL_SECONDS", str(7 * 24 * 3600)))


def _memory_scope(user_id: str, session_id: str) -> str:
    """The key under which cross-session memory is stored and retrieved.

    A returning user gets a new session_id, so anything filed under the old one
    is unreachable no matter how durable the store underneath is. The user is
    the stable identity, and it is the one the user means by "remember this".

    Falls back to session_id when there is no user -- an anonymous session still
    accumulates context for as long as it lasts, which is the old behaviour.
    """
    user_id = (user_id or "").strip()
    if user_id and user_id != "default":
        return user_id
    return session_id


def _k_ctxmem_hints(session_id: str) -> str:
    return f"ctxmem:{session_id}:hints"


def _k_ctxmem_prefs(session_id: str) -> str:
    return f"ctxmem:{session_id}:prefs"


def _ensure_sse_url(url: str) -> str:
    """
    Normalize an MCP URL to its SSE endpoint.

    MCP_SERVER_URL is the canonical BASE form elsewhere (server.py/settings.py
    append `/call_tool`), but the SSE transport used here needs the `/sse`
    endpoint. Idempotent for URLs that already end in `/sse`.
    """
    if not url:
        return url
    trimmed = url.rstrip("/")
    return trimmed if trimmed.endswith("/sse") else trimmed + "/sse"


def _call_with_timeout(fn: Callable[[], Any], timeout: float) -> Tuple[bool, Any, Optional[BaseException]]:
    """Run fn() in a daemon thread. Returns (ok, value, error).

    ok=False with error=None means it timed out; the thread is abandoned
    (daemon, so it never blocks the request or process exit).
    """
    box: Dict[str, Any] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_worker, name="memory-stage", daemon=True).start()
    if not done.wait(timeout):
        return False, None, None
    if "error" in box:
        return False, None, box["error"]
    return True, box.get("value"), None


_DB_SOURCE_TYPES = {"database", "db", "sql", "postgres", "postgresql", "mysql",
                    "bigquery", "snowflake", "redshift", "mcp"}


def _is_db_source(state: Dict) -> bool:
    """Only database sources have a schema the MCP server can hot-load.

    For a CSV upload, dataset_id is the file name, which the MCP server does
    not know; the call just waits on the SSE connection.
    """
    if state.get("active_db_customer_id") or state.get("active_db_source_table"):
        return True
    return str(state.get("input_data_type") or "").lower() in _DB_SOURCE_TYPES


_api_key = (
    os.environ.get("GROQ_API_KEY_PLANNING_AGENT")
    or os.environ.get("GROQ_API_KEY_CODING_AGENT")
    or os.environ.get("GROQ_API_KEY")
)

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
    # retrieval and background-learning threads read/append hints.
    _store_lock = threading.Lock()

    # Sessions whose last durable read FAILED (as opposed to finding nothing).
    # While a session is in this set, _persist_session_memory refuses to
    # overwrite Redis until a read succeeds — a transient blip at restart
    # must never let an empty in-process view erase stored preferences.
    _hydration_failed: set = set()

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

        # Connect once per process, not once per request -- and each store on
        # its own. One store that hangs on connect (Milvus collection.load())
        # must not block the other three: before, warm-up held a single lock
        # while Milvus hung, so every request's "connect" stage waited 5s and
        # Redis/Chroma/Postgres were treated as down too.
        self._store_ready: Dict[str, bool] = {n: False for n in self._STORES}
        self._store_locks: Dict[str, threading.Lock] = {n: threading.Lock() for n in self._STORES}

    # ------------------------------------------------------------------
    # Connections and warm-up
    # ------------------------------------------------------------------

    _STORES = ("postgres", "redis", "chroma", "milvus")
    # Which store each retrieval stage reads, so a failing stage marks only
    # that store for reconnect.
    _STAGE_STORE = {"hydrate": "redis", "artifact": "postgres", "schema_cache": "redis",
                    "mcp_schema": "redis", "signature": "chroma", "milvus": "milvus"}

    def _connect_store(self, name: str) -> bool:
        """Connect one store if needed. Never waits on another thread that is
        already connecting it (e.g. a slow warm-up): returns False instead."""
        if self._store_ready.get(name):
            return True
        lock = self._store_locks[name]
        if not lock.acquire(blocking=False):
            return False
        try:
            if self._store_ready.get(name):
                return True
            start = time.monotonic()
            getattr(self, name).connect()
            self._store_ready[name] = True
            logger.info("[memory] connected %s in %.2fs", name, time.monotonic() - start)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("[memory] connecting %s failed: %s", name, exc)
            return False
        finally:
            lock.release()

    def _connect_all(self, timeout: float) -> Dict[str, bool]:
        """Connect every store in parallel, each bounded by `timeout`.
        A store still connecting when the time is up is simply not ready yet."""
        pending = [n for n in self._STORES if not self._store_ready.get(n)]
        threads = []
        for name in pending:
            t = threading.Thread(target=self._connect_store, args=(name,),
                                 name=f"memory-connect-{name}", daemon=True)
            t.start()
            threads.append(t)
        deadline = time.monotonic() + timeout
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        return dict(self._store_ready)

    def _warm_embedding(self) -> None:
        start = time.monotonic()
        try:
            self._embed("warm up")
            logger.info("[memory] embedding model ready in %.2fs", time.monotonic() - start)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[memory] embedding warm-up failed: %s", exc)

    def warm_up(self) -> None:
        """At startup: load the embedding model and open all four connections,
        each in its own thread, so the first request pays for neither and one
        slow store cannot hold up the rest. Never raises."""
        start = time.monotonic()
        emb = threading.Thread(target=self._warm_embedding, name="memory-warm-embed", daemon=True)
        emb.start()
        ready = self._connect_all(timeout=60.0)
        slow = [n for n, ok in ready.items() if not ok]
        if slow:
            logger.warning(
                "[memory] warm-up: %s not connected after %.0fs; retrieval will skip "
                "it until it connects", ", ".join(slow), time.monotonic() - start,
            )
        emb.join(60.0)
        logger.info("[memory] warm-up done in %.2fs (ready=%s)", time.monotonic() - start,
                    {n: ok for n, ok in self._store_ready.items()})

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

        if hasattr(redis_client, "_fallback_store"):
            redis_client._fallback_store.clear()
        if hasattr(chroma_client, "_fallback_namespaces"):
            chroma_client._fallback_namespaces.clear()
        if hasattr(postgres_client, "_mock_table"):
            postgres_client._mock_table.clear()
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
                HumanMessage(content=human_prompt),
            ])

            content = response.content.strip()
            start_idx = content.find('{')
            end_idx = content.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                return json.loads(content[start_idx:end_idx + 1])
            return {}

        except Exception as e:
            logger.error(f"Failed to dynamically extract memory context: {e}")
            return {}

    def _artifact_hint(self, query: str, user_id: str) -> Tuple[bool, Optional[str]]:
        """Layer 3: (prior_artifact_found, hint). Surfaces metadata from prior
        runs so the planner can reuse chart URLs / metrics / outputs."""
        if not self.postgres.check_artifact_exists(query, user_id):
            return False, None
        prior_meta = self.postgres.get_artifact(query, user_id)
        if not prior_meta:
            return True, None
        hint_parts = []
        if prior_meta.get("status"):
            hint_parts.append(f"status={prior_meta['status']}")
        # Chart / visualisation URLs — reuse instead of regenerating
        for url_field in ("chart_url", "chart_urls", "plot_url",
                          "visualization_url", "dashboard_url"):
            val = prior_meta.get(url_field)
            if val:
                hint_parts.append(f"chart_urls={val[:3]}" if isinstance(val, list) else f"chart_url={val}")
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
        if prior_meta.get("session"):
            hint_parts.append(f"session={prior_meta['session']}")
        if hint_parts:
            return True, ("[L3] Prior artifact — reuse to skip redundant work: "
                          + ", ".join(hint_parts))
        return True, (f"[L3] Prior artifact detected for this query "
                      f"({len(prior_meta)} metadata fields stored).")

    def _hot_load_schema(self, user_id: str, dataset_id: str, state: Dict) -> Optional[str]:
        """Layer 1 cache miss: fetch the schema from MCP (database sources only)."""
        from app.services.mcp_cache_loader import fetch_schema_sync
        active_servers = state.get("active_mcp_servers", [])
        mcp_url = _ensure_sse_url(
            active_servers[0] if active_servers else (
                os.environ.get("MCP_SERVER_URL") or os.environ.get("MCP_URL") or _MCP_FALLBACK_URL
            )
        )
        # Prefer the tenant's own MCP key — the MCP maps key -> customer, so a
        # single global key would return one customer's schema for every tenant.
        api_key = state.get("mcp_api_key") or os.environ.get("MCP_API_KEY", "")
        logger.info(f"Triggering MCP extraction for dataset: {dataset_id} via server: {mcp_url}")
        schema_payload = fetch_schema_sync(mcp_url, api_key, dataset_id)
        if schema_payload and "error" not in schema_payload:
            self.redis.set_schema(user_id, dataset_id, schema_payload, ttl_seconds=_REDIS_SCHEMA_TTL)
            return f"Domain insight (hot-loaded via MCP): {json.dumps(schema_payload)}"
        return None

    def _learn_in_background(self, query: str, past_queries: List[str],
                             session_id: str, memory_scope: str) -> None:
        """LLM extraction + Milvus/Chroma/Redis writes. These only store memory
        for FUTURE turns, so they run off the request path. Never raises."""
        start = time.monotonic()
        try:
            extracted = self._extract_context_with_llm(query, past_queries) or {}
            new_hints = [h for h in (extracted.get("new_hints") or [])
                         if isinstance(h, str) and h.strip()]
            added = False
            for hint in new_hints:
                with self._store_lock:
                    hints = self._session_hints_store.setdefault(session_id, [])
                    if hint not in hints:
                        hints.append(hint)
                        added = True
                if not self._store_ready.get("milvus"):
                    continue  # Milvus not connected: keep the hint in session memory only
                try:
                    self.milvus.insert_insight(memory_scope, hint, "memory_llm", vector=self._embed(hint))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[memory] Milvus insert failed: %s", exc)
            new_sig = extracted.get("logic_signature")
            if new_sig and self._store_ready.get("chroma"):
                try:
                    self.chroma.update_user_signature(memory_scope, new_sig)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[memory] signature update failed: %s", exc)
            if added:
                self._persist_session_memory(session_id)
            logger.info("[memory] background learning %.2fs: %d new hints",
                        time.monotonic() - start, len(new_hints))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[memory] background learning failed: %s", exc)

    # ------------------------------------------------------------------
    # Retrieval (read path)
    # ------------------------------------------------------------------

    def _retrieve_memory_internal(self, query: str, session_id: str = "default",
                                  dataset_id: str = "unknown", state: Dict = None) -> Dict:
        """
        Read path only. Each stage has its own timeout: a slow or failing store
        is skipped and logged, and the hints from the other stores are kept.
        Learning (LLM extraction + writes) happens in a background thread.
        """
        if state is None:
            state = {}
        user_id = state.get("user_id", "default")
        # Cross-session tiers (Chroma L2, Milvus L4) key off this rather than the
        # session, so a returning user reaches what an earlier session learned.
        memory_scope = _memory_scope(user_id, session_id)
        timings: Dict[str, float] = {}
        t_start = time.monotonic()

        def run(name: str, fn: Callable[[], Any], timeout: float = _STAGE_TIMEOUT,
                default: Any = None) -> Any:
            stage_start = time.monotonic()
            ok, value, err = _call_with_timeout(fn, timeout)
            timings[name] = round(time.monotonic() - stage_start, 2)
            if ok:
                return value
            if err is None:
                logger.warning("[memory] stage %s timed out after %.1fs; skipped", name, timeout)
            else:
                logger.warning("[memory] stage %s failed (%s); skipped", name, err)
            # Next request reconnects that store, in case its connection is the problem.
            store = self._STAGE_STORE.get(name)
            if store and err is not None:
                self._store_ready[store] = False
            return default

        # Restore durable session memory (hints/preferences) after a restart.
        if self._store_ready.get("redis"):
            run("hydrate", lambda: self._hydrate_session_memory(session_id))

        with self._store_lock:
            past_queries = list(self._past_queries_store.get(session_id, []))
            self._past_queries_store[session_id] = past_queries + [query]

        accumulated_hints = list(self._session_hints_store.get(session_id, []))
        # Provenance buckets for the top-3 contract (see _compose_top_hints).
        tiered_hints: Dict[str, List[str]] = {"session": list(accumulated_hints)}

        def _add_hint(tier: str, hint: Optional[str]) -> None:
            if hint and hint not in accumulated_hints:
                accumulated_hints.append(hint)
                tiered_hints.setdefault(tier, []).append(hint)

        logic_sig = _DEFAULT_STYLE_HINT

        ready = run("connect", lambda: self._connect_all(_CONNECT_TIMEOUT),
                    timeout=_CONNECT_TIMEOUT + 0.5, default={}) or {}

        def ok(store: str) -> bool:
            if ready.get(store):
                return True
            timings.setdefault(f"skip_{store}", 0.0)
            return False

        # Layer 3: prior artifacts for this exact query.
        found, artifact_hint = (run("artifact", lambda: self._artifact_hint(query, user_id),
                                    default=(False, None))
                                if ok("postgres") else (False, None))
        prior_artifact_found = bool(found)
        _add_hint("artifact", artifact_hint)

        # Layer 1: semantic schema cache; MCP hot-load only for database sources.
        schema_info = (run("schema_cache", lambda: self.redis.search_schema_semantically(user_id, query))
                       if ok("redis") else None)
        if schema_info:
            _add_hint("domain", f"Domain insight: {schema_info}")
        elif (dataset_id and dataset_id not in ("unknown", "unknown_dataset")
              and _is_db_source(state) and ok("redis")):
            _add_hint("domain", run("mcp_schema",
                                    lambda: self._hot_load_schema(user_id, dataset_id, state),
                                    timeout=_MCP_SCHEMA_TIMEOUT))

        # Layer 2: style signature. User-scoped first, legacy session-scoped
        # second, so signatures written before user scoping are not stranded.
        def _signature():
            sig = self.chroma.get_user_signature(memory_scope)
            if not sig and memory_scope != session_id:
                sig = self.chroma.get_user_signature(session_id)
            return sig

        user_sig = run("signature", _signature) if ok("chroma") else None
        if user_sig:
            logic_sig = user_sig

        # Layer 4: similar past insights. Embed the query ONCE. Legacy rows are
        # keyed by session, hence the second read.
        query_vec = (run("embed", lambda: self._embed(query), timeout=_EMBED_TIMEOUT)
                     if ok("milvus") else None)

        def _similar():
            rows = self.milvus.search_similar_insights(
                memory_scope, query_vector=query_vec, top_k=_MILVUS_TOP_K)
            if not rows and memory_scope != session_id:
                rows = self.milvus.search_similar_insights(
                    session_id, query_vector=query_vec, top_k=_MILVUS_TOP_K)
            return rows

        similar_rows = run("milvus", _similar, default=[]) if ok("milvus") else []
        for record in similar_rows or []:
            if isinstance(record, dict):
                _add_hint("similar", record.get("content"))

        # Top-3 contract: explicit preferences hold reserved slots; leftover
        # slots fill by relevance tier (artifact > domain > similar > session).
        top_3_hints = self._compose_top_hints(session_id, tiered_hints)

        # Learning for future turns runs after we return.
        if self.llm:
            threading.Thread(
                target=self._learn_in_background,
                args=(query, past_queries, session_id, memory_scope),
                name="memory-learn", daemon=True,
            ).start()

        logger.info("[memory] retrieval %.2fs stages=%s",
                    time.monotonic() - t_start, timings)

        return {
            # Hints learned from THIS query arrive next turn (background learning).
            "new_hints_this_context": [],
            "accumulated_memory_hints": accumulated_hints,
            "memory_hints": top_3_hints,  # Top-3 contract for the planner
            "prior_artifact_found": prior_artifact_found,
            "session_logic_signature": logic_sig,
            "memory_context_unavailable": not self.llm,
        }

    # ------------------------------------------------------------------
    # Explicit preferences (write path)
    # ------------------------------------------------------------------

    def store_user_preference(self, preference: str, session_id: str = "default",
                              scope_id: Optional[str] = None) -> bool:
        """
        Persist an explicitly stated user preference (e.g. "The user's favorite
        column is 'reordered'") so future retrievals surface it as a hint.

        Direct write path behind the planner's `store_user_preference` tool —
        requires no LLM and never raises. The preference goes into the session
        stores that retrieval serves from (with reserved top-3 slots); the
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
                args=(scope_id or session_id, preference),
                daemon=True,
            ).start()

        if stored:
            logger.info(f"🧠 AVALOKA CONTEXT MEMORY STORED (session={session_id}): {preference}")
        return stored

    def _persist_preference_to_milvus(self, scope_id: str, preference: str) -> None:
        """Best-effort durable write for an explicit preference. Never raises.

        scope_id is the memory scope (user where known, session otherwise): an
        explicit preference is something a user expects to still hold next
        time they log in.
        """
        try:
            self.milvus.insert_insight(scope_id, preference, "user_explicit", vector=self._embed(preference))
        except Exception as e:
            logger.error(f"Failed to persist user preference to Milvus: {e}")

    # ------------------------------------------------------------------
    # Durable session memory (Redis write-through)
    # ------------------------------------------------------------------

    def _hydrate_session_memory(self, session_id: str) -> None:
        """
        Restore a session's hints/preferences from Redis into the in-process
        stores after a process restart. Each store is hydrated independently.
        In-process entries stay authoritative while they exist. Never raises.
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
        store_user_preference can't interleave between snapshot and write.
        If the last durable read for this session failed, retry it first — and
        skip the write entirely while Redis still can't be read.
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
        hints — or a flat list, treated entirely as session history.
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

    # ------------------------------------------------------------------
    # Public entry with the overall circuit breaker
    # ------------------------------------------------------------------

    def retrieve_memory(self, query: str, session_id: str = "default",
                        dataset_id: str = "unknown", state: Dict = None) -> Dict:
        """
        Wrap _retrieve_memory_internal in a hard timeout (MEMORY_CIRCUIT_BREAKER_TIMEOUT).
        Returns a safe empty payload if exceeded so LangGraph nodes are never blocked.
        """
        _empty = {
            "new_hints_this_context": [],
            "accumulated_memory_hints": [],
            "memory_hints": [],
            "prior_artifact_found": False,
            "session_logic_signature": _DEFAULT_STYLE_HINT,
            "memory_context_unavailable": True,
        }
        # A daemon thread + Event, deliberately NOT a ThreadPoolExecutor: an
        # executor joins its worker on shutdown, so a hung lookup kept blocking
        # the caller (and process exit). A daemon thread is simply abandoned.
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
                "Returning empty payload. The '[memory] stage ...' warnings above show "
                "which store was slow."
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

# Open connections and load the embedding model in the background at import,
# so the first user request doesn't pay for them. MEMORY_WARMUP=0 disables
# this (e.g. in tests).
if os.environ.get("MEMORY_WARMUP", "1") != "0":
    threading.Thread(target=_orchestrator.warm_up, name="memory-warmup", daemon=True).start()


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
    user_id = (state or {}).get("user_id", "default")
    return _orchestrator.store_user_preference(
        preference, session_id, scope_id=_memory_scope(user_id, session_id)
    )