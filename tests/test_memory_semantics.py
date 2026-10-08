"""
tests/test_memory_semantics.py

Avaloka 1.2 Spec Compliance Tests — Memory Layer Semantics

Covers:
  - time_range_seconds temporal filter in Milvus search
  - notebook_id partition-scoped retrieval
  - cell_id / artifact_ref schema fields in insert
  - retrieve_historical_analysis Pydantic parse / tool schema contract
  - circuit breaker: 3-second timeout enforcement
  - embedding path: mock OPENAI_API_KEY activates OpenAIEmbeddings branch

All DB calls are mocked — no live infrastructure required.
Heavy transitive dependencies (groq, langchain_groq, pymilvus, celery…)
are injected as lightweight stubs before any app module is imported so the
suite runs in any venv — including ones that are missing those packages.
"""

import os
import sys
import time
import types
import pytest
import builtins
import importlib
from contextlib import nullcontext
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Venv-isolation: stub out packages that may be missing
# ---------------------------------------------------------------------------

# Modules this file actually fabricated. Anything absent from this set is the
# genuine installed package and must never be mutated below: pytest imports
# every test module during collection, so a MagicMock assigned onto the real
# langgraph.graph or langchain_core.messages is what every later test in the
# session sees, not just this one.
_FABRICATED: set[str] = set()


def _stub(name):
    """Register a throwaway module so `import <name>` never raises.

    Only stubs what is genuinely unimportable. These entries are never torn
    down, so stubbing a module that really exists leaves a MagicMock in
    sys.modules for the whole session and every later test module that imports
    it gets the mock instead of the real thing.
    """
    if name in sys.modules:
        return sys.modules[name]

    try:
        real = importlib.import_module(name)
    except Exception:
        real = None

    if real is not None:
        # The genuine package is installed: hand it back untouched and leave
        # sys.modules alone. Registering a mutable copy here is what leaked
        # mocks into the rest of the session -- the copy became the entry every
        # later importer resolved, so once the assignments below replaced
        # StateGraph or HumanMessage on it, every other test module saw the
        # mock instead of the real class.
        return real

    # Genuinely unimportable: fabricate a stub whose attribute access yields
    # mocks, so `from x.y import z` still resolves.
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda _n: MagicMock()
    sys.modules[name] = mod
    _FABRICATED.add(name)
    return mod

# Core stubs — must be registered before any app imports
for _pkg in [
    "groq",
    "langchain_groq",
    "langchain_openai",
    "pymilvus", "pymilvus.orm", "pymilvus.orm.collection",
    "celery", "celery.utils", "celery.utils.log", "celery.schedules", "celery.result",
    "redis", "redis.connection", "redis.client", "chromadb",
    "langchain_core", "langchain_core.messages", "langchain_core.prompts",
    "redbeat", "redbeat.schedulers",
    "langgraph", "langgraph.graph",
    # AWS / cloud
    "boto3", "botocore", "kubernetes",
    # App internals that drag in the above
    "app.core", "app.core.celery_app", "app.core.storage", "app.core.agent_llm",
    "app.api.cloud_connections",
    "app.agents.infra_agent", "app.agents.execution_agent",
    "app.agents.planner_graph_agent", "app.agents.scheduler",
    "app.infra", "app.infra.k8s_invoker", "app.infra.ray_job_runner",
    "app.infra.k8s_secrets",
    "graphviz",
]:
    _stub(_pkg)

# groq.APIError
if "groq" in _FABRICATED:
    sys.modules["groq"].APIError = type("APIError", (Exception,), {})

# langchain_groq.ChatGroq
if "langchain_groq" in _FABRICATED:
    sys.modules["langchain_groq"].ChatGroq = MagicMock

# langchain_core.messages named imports
if "langchain_core.messages" in _FABRICATED:
    _lcm = sys.modules["langchain_core.messages"]
    for _cls_name in ("AIMessage", "HumanMessage", "BaseMessage", "SystemMessage"):
        setattr(_lcm, _cls_name, MagicMock)

# langchain_core.prompts
if "langchain_core.prompts" in _FABRICATED:
    sys.modules["langchain_core.prompts"].ChatPromptTemplate = MagicMock

# pymilvus top-level attributes
if "pymilvus" in _FABRICATED:
    _pym = sys.modules["pymilvus"]
    for _attr in ("connections", "Collection", "CollectionSchema", "FieldSchema",
                  "DataType", "utility"):
        setattr(_pym, _attr, MagicMock())

# celery.utils.log
if "celery.utils.log" in _FABRICATED:
    sys.modules["celery.utils.log"].get_logger = MagicMock(return_value=MagicMock())
if "celery" in _FABRICATED:
    sys.modules["celery"].Celery = MagicMock
    sys.modules["celery"].shared_task = lambda *a, **kw: (lambda f: f)  # passthrough

# redbeat.schedulers — all names as mocks
if "redbeat.schedulers" in _FABRICATED:
    _rb = sys.modules["redbeat.schedulers"]
    for _n in ("RedBeatScheduler", "RedBeatSchedulerEntry", "RedBeatJSONEncoder",
               "RedBeatJSONDecoder", "get_redis", "ensure_conf"):
        setattr(_rb, _n, MagicMock)

# langgraph.graph — StateGraph and END
if "langgraph.graph" in _FABRICATED:
    _lg = sys.modules["langgraph.graph"]
    _lg.StateGraph = MagicMock
    _lg.END = "END"

# app.core.celery_app — expose celery_app and AvalokaScheduler
if "app.core.celery_app" in _FABRICATED:
    _ca = sys.modules["app.core.celery_app"]
    _ca.celery_app = MagicMock()
    _ca.AvalokaScheduler = MagicMock
    if "app.core" in _FABRICATED:
        sys.modules["app.core"].celery_app = _ca

# app.agents.infra_agent
if "app.agents.infra_agent" in _FABRICATED:
    # Never mock this onto the real module: tests/test_infra_integration.py
    # imports infra_agent_node and would silently exercise a MagicMock.
    sys.modules["app.agents.infra_agent"].infra_agent_node = MagicMock

# Wire sub-modules into parents
for _parent, _attr, _child in (
    ("langchain_core", "messages", "langchain_core.messages"),
    ("langchain_core", "prompts", "langchain_core.prompts"),
    ("redbeat", "schedulers", "redbeat.schedulers"),
    ("langgraph", "graph", "langgraph.graph"),
    ("app.infra", "k8s_invoker", "app.infra.k8s_invoker"),
    ("app.infra", "ray_job_runner", "app.infra.ray_job_runner"),
):
    if _parent in _FABRICATED:
        setattr(sys.modules[_parent], _attr, sys.modules[_child])
if "app.infra.ray_job_runner" in _FABRICATED:
    sys.modules["app.infra.ray_job_runner"].run_rayjob_from_yaml = MagicMock



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_collection():
    col = MagicMock()
    col.search.return_value = [[]]          # empty result set
    col.insert.return_value = None
    col.flush.return_value = None
    return col


# ---------------------------------------------------------------------------
# 1. Planner schema: HistoricalAnalysisParams + ToolCall literal
# ---------------------------------------------------------------------------

class TestPlannerToolSchema:
    def test_historical_analysis_params_valid(self):
        from app.agents.planner import HistoricalAnalysisParams
        p = HistoricalAnalysisParams(query="past YouTube ETL runs", time_range_seconds=600, notebook_id="nb-42")
        assert p.query == "past YouTube ETL runs"
        assert p.time_range_seconds == 600
        assert p.notebook_id == "nb-42"

    def test_historical_analysis_params_optional_fields(self):
        from app.agents.planner import HistoricalAnalysisParams
        p = HistoricalAnalysisParams(query="find previous runs")
        assert p.time_range_seconds is None
        assert p.notebook_id is None

    def test_toolcall_literal_includes_retrieve_historical(self):
        from app.agents.planner import ToolCall
        # Pydantic raises ValidationError if name is not in Literal — so this must not raise.
        tc = ToolCall(
            name="retrieve_historical_analysis",
            parameters={"query": "previous analysis"}
        )
        assert tc.name == "retrieve_historical_analysis"

    def test_tools_list_contains_retrieve_historical(self):
        from app.agents.planner import tools
        names = [t["function"]["name"] for t in tools]
        assert "retrieve_historical_analysis" in names, (
            "retrieve_historical_analysis missing from tools schema list sent to LLM"
        )

    def test_tool_name_to_param_mapping(self):
        from app.agents.planner import tool_name_to_param, HistoricalAnalysisParams
        assert tool_name_to_param.get("retrieve_historical_analysis") is HistoricalAnalysisParams


# ---------------------------------------------------------------------------
# 2. Milvus Client — schema fields & temporal / notebook filtering
# ---------------------------------------------------------------------------

class TestMilvusClientSchema:
    @pytest.fixture(autouse=True)
    def _patch_pymilvus(self):
        """Mock out pymilvus so MilvusClientImpl never talks to a live server."""
        with patch("app.services.db.milvus_client.connections") as mock_conn, \
             patch("app.services.db.milvus_client.Collection") as MockCol, \
             patch("app.services.db.milvus_client.utility") as mock_util:

            mock_util.has_collection.return_value = False
            mock_util.has_partition.return_value = True
            col_instance = _make_mock_collection()
            MockCol.return_value = col_instance
            self.col = col_instance
            yield

    def test_insert_includes_all_spec_fields(self):
        from app.services.db.milvus_client import MilvusClientImpl
        client = MilvusClientImpl()
        client._collection = self.col   # bypass connect()

        client.insert_insight(
            session_id="sess-1",
            content="ETL completed for YouTube stats",
            agent_role="coder",
            notebook_id="nb-yt",
            cell_id="cell-03",
            artifact_ref="s3://bucket/output.csv"
        )

        self.col.insert.assert_called_once()
        call_data = self.col.insert.call_args[0][0]   # positional list of arrays
        # data order: vector, timestamp, session_id, notebook_id, agent_role, cell_id, artifact_ref, content
        assert call_data[2] == ["sess-1"],        "session_id mismatch"
        assert call_data[3] == ["nb-yt"],         "notebook_id mismatch"
        assert call_data[4] == ["coder"],         "agent_role mismatch"
        assert call_data[5] == ["cell-03"],       "cell_id mismatch"
        assert call_data[6] == ["s3://bucket/output.csv"], "artifact_ref mismatch"

    def test_search_passes_time_range_expr(self):
        from app.services.db.milvus_client import MilvusClientImpl
        client = MilvusClientImpl()
        client._collection = self.col

        before = int(time.time())
        client.search_similar_insights(
            session_id="sess-2",
            time_range_seconds=600
        )
        after = int(time.time())

        call_kwargs = self.col.search.call_args[1]
        expr = call_kwargs.get("expr", "")
        # expr must contain a timestamp >= cutoff clause
        assert "timestamp >=" in expr, f"Temporal filter missing from expr: {expr!r}"
        # The cutoff must be within [before-600, after] range
        cutoff = int(expr.split("timestamp >= ")[1].split()[0])
        assert before - 600 <= cutoff <= after

    def test_search_passes_notebook_id_expr(self):
        from app.services.db.milvus_client import MilvusClientImpl
        client = MilvusClientImpl()
        client._collection = self.col

        client.search_similar_insights(session_id="sess-3", notebook_id="nb-special")

        call_kwargs = self.col.search.call_args[1]
        expr = call_kwargs.get("expr", "")
        assert "notebook_id == 'nb-special'" in expr, f"notebook_id filter missing from expr: {expr!r}"

    def test_search_combined_filters(self):
        from app.services.db.milvus_client import MilvusClientImpl
        client = MilvusClientImpl()
        client._collection = self.col

        client.search_similar_insights(
            session_id="sess-4",
            notebook_id="nb-combo",
            time_range_seconds=300
        )

        call_kwargs = self.col.search.call_args[1]
        expr = call_kwargs.get("expr", "")
        assert "session_id == 'sess-4'" in expr
        assert "notebook_id == 'nb-combo'" in expr
        assert "timestamp >=" in expr


# ---------------------------------------------------------------------------
# 3. Milvus Recorder — embedding path with OPENAI_API_KEY
# ---------------------------------------------------------------------------

class TestMilvusRecorderEmbedding:
    def _get_raw_fn(self):
        """Return _record_impl — the undecorated recorder logic."""
        import app.services.milvus_recorder as recorder_mod
        return recorder_mod._record_impl

    def _call_recorder(self, env_overrides, langchain_openai_mod=None, block_sentence_transformers=False):
        """Call the recorder's raw Python logic with a mocked milvus_client."""
        import app.services.milvus_recorder as recorder_mod
        import app.services.db.milvus_client as milvus_client_mod

        mock_client = MagicMock()
        mock_client._collection = MagicMock()   # truthy — bypass connect()
        mock_client._vector_dim  = 1536          # concrete so [0.0]*dim resolves

        extra_modules = {"langchain_openai": langchain_openai_mod} if langchain_openai_mod else {}
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "sentence_transformers" or name.startswith("sentence_transformers."):
                raise ImportError("sentence-transformers blocked for zero-vector fallback test")
            return real_import(name, *args, **kwargs)

        import_guard = (
            patch("builtins.__import__", side_effect=blocked_import)
            if block_sentence_transformers
            else nullcontext()
        )

        with patch.dict(os.environ, env_overrides, clear=False), \
             patch.dict(sys.modules, extra_modules), \
             patch.object(milvus_client_mod, "milvus_client", mock_client), \
             patch.object(recorder_mod, "milvus_client", mock_client), \
             import_guard:

            # Build and call a fresh, undecorated version of the function body
            # by grabbing the actual function object before the decorator wrapped it.
            raw = self._get_raw_fn()
            raw(
                session_id="test-sess",
                content="test content",
                agent_role="assistant",
                notebook_id="nb-test"
            )

        return mock_client

    def test_uses_openai_embeddings_when_key_set(self):
        fake_vector = [0.42] * 1536
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = fake_vector

        fake_lo = types.ModuleType("langchain_openai")
        fake_lo.OpenAIEmbeddings = MagicMock(return_value=mock_embedder)

        env = {k: v for k, v in os.environ.items()}
        env["OPENAI_API_KEY"] = "sk-test"

        mock_client = self._call_recorder(env, langchain_openai_mod=fake_lo)

        mock_client.insert_insight.assert_called_once()
        call_kwargs = mock_client.insert_insight.call_args[1]
        assert call_kwargs.get("vector") == fake_vector, "Expected real embedding vector"

    def test_falls_back_to_zeros_without_any_embedding_provider(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}

        mock_client = self._call_recorder(env, block_sentence_transformers=True)

        mock_client.insert_insight.assert_called_once()
        call_kwargs = mock_client.insert_insight.call_args[1]
        vec = call_kwargs.get("vector")
        # The zero-vector length comes from milvus_client._vector_dim (may be mocked),
        # but the values must all be 0.0.
        assert isinstance(vec, list), f"Expected a list, got {type(vec)}"
        assert all(v == 0.0 for v in vec), f"Expected all-zero vector, got: {vec[:5]}..."



# ---------------------------------------------------------------------------
# 4. Memory Plane — explicit 3-second circuit breaker
#
# The breaker is a daemon worker thread + Event (NOT a ThreadPoolExecutor):
# a `with ThreadPoolExecutor` block joins the hung worker on exit, so the
# caller stayed blocked for the full hang even after TimeoutError fired, and
# the non-daemon executor thread was joined again at interpreter exit,
# hanging process shutdown. These tests exercise the real threading behavior
# with a short patched timeout so the suite stays fast.
# ---------------------------------------------------------------------------

# Full payload contract consumed by app/api/workflow.py on the fallback path.
_BREAKER_EMPTY_PAYLOAD_KEYS = {
    "new_hints_this_context",
    "accumulated_memory_hints",
    "memory_hints",
    "prior_artifact_found",
    "session_logic_signature",
    "memory_context_unavailable",
}


class TestMemoryPlaneCircuitBreaker:
    # Patched breaker window for tests; hung lookups sleep much longer so a
    # regression back to join-the-worker behavior fails loudly on timing.
    FAST_TIMEOUT = 0.4
    HANG_SECONDS = 5.0

    @staticmethod
    def _orchestrator():
        from app.services.memory_plane import MemoryOrchestrator
        return MemoryOrchestrator(llm_client=None)

    def test_default_timeout_matches_documented_contract(self):
        """The default is written down in three places -- the code, the module
        header, and the Helm chart -- and they must say the same thing.

        It has drifted three times: to 105s while the header still said 3.0;
        the 3s contract itself proved wrong (a successful retrieval measured
        4-6s, so the breaker fired on every call and the memory plane never
        returned anything); and then the code default dropped to 8.0 while the
        chart went on setting 20.0, so the number a developer saw was not the
        number a deployment ran. No number is asserted here on purpose: what
        is pinned is that the three agree. Whether the number is big enough is
        tests/test_memory_staged_retrieval.py's question.
        """
        import re
        from pathlib import Path
        import app.services.memory_plane as mp

        root = Path(mp.__file__).resolve().parents[2]
        source = Path(mp.__file__).read_text()
        in_code = re.search(
            r'os\.environ\.get\("MEMORY_CIRCUIT_BREAKER_TIMEOUT",\s*"([0-9.]+)"\)', source)
        in_header = re.search(
            r"#\s+MEMORY_CIRCUIT_BREAKER_TIMEOUT\s.*\(default: ([0-9.]+)", source)
        in_chart = re.search(
            r"^\s*circuitBreakerTimeout:\s*([0-9.]+)\s*$",
            (root / "deploy/helm/avaloka/values.yaml").read_text(), re.M)
        assert in_code and in_header and in_chart, "a place the default is recorded has moved"

        assert float(in_header.group(1)) == float(in_code.group(1)), "module header disagrees with the code"
        assert float(in_chart.group(1)) == float(in_code.group(1)), "Helm chart disagrees with the code"
        expected = float(os.environ.get("MEMORY_CIRCUIT_BREAKER_TIMEOUT", in_code.group(1)))
        assert mp._CIRCUIT_BREAKER_TIMEOUT == expected

    def test_timeout_triggers_empty_hints(self):
        """A retrieval that outlives the breaker window yields the safe
        empty payload, flagged unavailable."""
        orch = self._orchestrator()

        def hang(*args, **kwargs):
            time.sleep(self.HANG_SECONDS)
            return {"accumulated_memory_hints": ["too late"]}

        with patch("app.services.memory_plane._CIRCUIT_BREAKER_TIMEOUT", self.FAST_TIMEOUT), \
             patch.object(orch, "_retrieve_memory_internal", side_effect=hang):
            result = orch.retrieve_memory("slow query", session_id="timeout-test")

        assert result["memory_hints"] == [], "Timed-out retrieval must yield no hints"
        assert result["accumulated_memory_hints"] == []
        assert result["memory_context_unavailable"] is True

    def test_caller_released_at_timeout_not_at_hang_end(self):
        """REGRESSION: the old `with ThreadPoolExecutor` breaker joined the
        hung worker on block exit, so the caller was released only when the
        hang ended (TimeoutError fired at 3s, caller unblocked at hang-end).
        The caller must be released at ~the timeout itself."""
        orch = self._orchestrator()

        with patch("app.services.memory_plane._CIRCUIT_BREAKER_TIMEOUT", self.FAST_TIMEOUT), \
             patch.object(orch, "_retrieve_memory_internal",
                          side_effect=lambda *a, **k: time.sleep(self.HANG_SECONDS)):
            start = time.monotonic()
            result = orch.retrieve_memory("q", session_id="release-test")
            elapsed = time.monotonic() - start

        assert result["memory_context_unavailable"] is True
        # Generous margin, but far below HANG_SECONDS: a join regression
        # would push elapsed to ~5s.
        assert elapsed < self.FAST_TIMEOUT + 1.0, (
            f"Caller blocked {elapsed:.2f}s — breaker is joining the hung worker again"
        )

    def test_abandoned_worker_is_daemon_thread(self):
        """REGRESSION: executor threads are non-daemon and get joined at
        interpreter exit, so one hung lookup froze process shutdown until
        SIGTERM. The abandoned worker must be a daemon thread."""
        import threading

        orch = self._orchestrator()
        release = threading.Event()

        with patch("app.services.memory_plane._CIRCUIT_BREAKER_TIMEOUT", self.FAST_TIMEOUT), \
             patch.object(orch, "_retrieve_memory_internal",
                          side_effect=lambda *a, **k: release.wait(self.HANG_SECONDS)):
            orch.retrieve_memory("q", session_id="daemon-test")
            workers = [t for t in threading.enumerate() if t.name == "memory-retrieval"]

        try:
            assert workers, "expected the abandoned memory-retrieval worker to still be alive"
            assert all(t.daemon for t in workers), (
                "memory-retrieval worker must be a daemon thread or process exit hangs"
            )
        finally:
            release.set()  # let the worker finish promptly instead of sleeping out

    def test_exception_returns_empty_hints(self):
        """A non-timeout exception from the internal retrieval also triggers
        the circuit breaker fallback path — and never propagates."""
        orch = self._orchestrator()

        with patch.object(orch, "_retrieve_memory_internal",
                          side_effect=RuntimeError("DB exploded")):
            result = orch.retrieve_memory("query", session_id="exc-test")

        assert result["memory_hints"] == []
        assert result["memory_context_unavailable"] is True

    def test_fast_retrieval_passes_through_untouched(self):
        """A retrieval that beats the breaker window is returned as-is."""
        orch = self._orchestrator()
        payload = {
            "new_hints_this_context": [],
            "accumulated_memory_hints": ["prefer pandas"],
            "memory_hints": ["prefer pandas"],
            "prior_artifact_found": True,
            "session_logic_signature": "sig",
            "memory_context_unavailable": False,
        }

        with patch.object(orch, "_retrieve_memory_internal", return_value=payload):
            result = orch.retrieve_memory("q", session_id="fast-test")

        assert result is payload

    def test_empty_payload_contract_keys(self):
        """The fallback payload must carry every key workflow.py consumes,
        so a breaker trip can never KeyError downstream."""
        orch = self._orchestrator()

        with patch("app.services.memory_plane._CIRCUIT_BREAKER_TIMEOUT", self.FAST_TIMEOUT), \
             patch.object(orch, "_retrieve_memory_internal",
                          side_effect=lambda *a, **k: time.sleep(self.HANG_SECONDS)):
            result = orch.retrieve_memory("q", session_id="contract-test")

        assert _BREAKER_EMPTY_PAYLOAD_KEYS.issubset(result.keys()), (
            f"missing keys: {_BREAKER_EMPTY_PAYLOAD_KEYS - set(result.keys())}"
        )

    def test_concurrent_hung_callers_all_released_independently(self):
        """Parallel LangGraph nodes may call retrieve_memory simultaneously;
        every caller must be released within its own breaker window even when
        all lookups hang (no shared lock / no serialization)."""
        import concurrent.futures

        orch = self._orchestrator()
        n_callers = 4

        with patch("app.services.memory_plane._CIRCUIT_BREAKER_TIMEOUT", self.FAST_TIMEOUT), \
             patch.object(orch, "_retrieve_memory_internal",
                          side_effect=lambda *a, **k: time.sleep(self.HANG_SECONDS)):
            start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_callers) as pool:
                results = [
                    f.result()
                    for f in [
                        pool.submit(orch.retrieve_memory, f"q{i}", f"s{i}")
                        for i in range(n_callers)
                    ]
                ]
            elapsed = time.monotonic() - start

        assert all(r["memory_context_unavailable"] for r in results)
        assert elapsed < self.FAST_TIMEOUT + 1.5, (
            f"{n_callers} concurrent callers took {elapsed:.2f}s — callers are being serialized"
        )


# ---------------------------------------------------------------------------
# 5. Workflow node: milvus_context_id & active_mcp_servers populated
# ---------------------------------------------------------------------------

class TestWorkflowStateFields:
    def test_memory_injection_node_populates_milvus_context_id(self):
        from app.api.workflow import memory_injection_node

        with patch("app.api.workflow.retrieve_memory") as mock_rm:
            mock_rm.return_value = {
                "memory_hints": [],
                "prior_artifact_found": False,
                "session_logic_signature": None,
                "memory_context_unavailable": False,
            }

            state = {
                "user_prompt": "analyse YouTube data",
                "messages": [],
                "session_id": "wf-sess",
            }
            result = memory_injection_node(state)

        assert "milvus_context_id" in result, "milvus_context_id not returned by memory_injection_node"
        assert result["milvus_context_id"], "milvus_context_id must be non-empty"
        assert "active_mcp_servers" in result

    def test_memory_injection_node_populates_mcp_from_env(self):
        from app.api.workflow import memory_injection_node

        mock_ret = {
            "memory_hints": [],
            "prior_artifact_found": False,
            "session_logic_signature": None,
            "memory_context_unavailable": False,
        }

        # --- Primary: MCP_SERVER_URL (canonical .env name) ---
        with patch("app.api.workflow.retrieve_memory", return_value=mock_ret), \
             patch.dict(os.environ, {"MCP_SERVER_URL": "http://custom-mcp:9090/sse"},
                        clear=False):
            state  = {"user_prompt": "test", "messages": [], "session_id": "wf-mcp"}
            result = memory_injection_node(state)

        assert "http://custom-mcp:9090/sse" in result["active_mcp_servers"], \
            "MCP_SERVER_URL must be picked up as the canonical env var"

        # --- Fallback alias: MCP_URL (legacy) ---
        env_no_server_url = {k: v for k, v in os.environ.items() if k != "MCP_SERVER_URL"}
        with patch("app.api.workflow.retrieve_memory", return_value=mock_ret), \
             patch.dict(os.environ, env_no_server_url, clear=True):
            os.environ["MCP_URL"] = "http://legacy-mcp:8080/sse"
            state  = {"user_prompt": "test", "messages": [], "session_id": "wf-mcp-legacy"}
            result = memory_injection_node(state)

        assert "http://legacy-mcp:8080/sse" in result["active_mcp_servers"], \
            "MCP_URL fallback alias must still be honoured"


# ---------------------------------------------------------------------------
# 6. Production runtime guardrails
# ---------------------------------------------------------------------------

class TestProductionMemoryRuntimeConfig:
    def test_dev_mode_reports_status_without_raising(self, monkeypatch):
        from app.services.memory_runtime import validate_memory_runtime_config

        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.delenv("MEMORY_STRICT_INFRA", raising=False)

        status = validate_memory_runtime_config()

        assert status["environment"] == "development"
        assert status["strict"] == "False"

    def test_production_requires_explicit_memory_infra(self, monkeypatch):
        from app.services.memory_runtime import MemoryConfigError, validate_memory_runtime_config

        for key in (
            "REDIS_URL",
            "CHROMA_STORAGE_PATH",
            "POSTGRES_URL",
            "MILVUS_HOST",
            "MILVUS_PORT",
            "MILVUS_COLLECTION",
            "OPENAI_API_KEY",
            "MEMORY_ALLOW_MEMORY_FALLBACKS",
        ):
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("APP_ENV", "production")

        with pytest.raises(MemoryConfigError) as exc:
            validate_memory_runtime_config()

        message = str(exc.value)
        assert "REDIS_URL" in message
        assert "POSTGRES_URL" in message
        assert "MILVUS_HOST" in message

    def test_strict_mode_blocks_dev_fallbacks(self, monkeypatch):
        from app.services.memory_runtime import MemoryConfigError, ensure_fallback_allowed

        monkeypatch.setenv("MEMORY_STRICT_INFRA", "true")

        with pytest.raises(MemoryConfigError):
            ensure_fallback_allowed("Layer 4 Milvus", "pymilvus is missing")
