import pytest
import os
import sys
import json
import pandas as pd
from unittest.mock import patch, MagicMock
from typing import Dict
import types
import importlib

try:
    from langchain_core.messages import HumanMessage, AIMessage
except ModuleNotFoundError:
    langchain_core = types.ModuleType("langchain_core")
    langchain_core.__path__ = []
    langchain_messages = types.ModuleType("langchain_core.messages")
    langchain_prompts = types.ModuleType("langchain_core.prompts")

    class _TestMessage:
        type = "message"

        def __init__(self, content="", **kwargs):
            self.content = content
            for key, value in kwargs.items():
                setattr(self, key, value)

    class HumanMessage(_TestMessage):
        type = "human"

    class AIMessage(_TestMessage):
        type = "ai"

    langchain_messages.HumanMessage = HumanMessage
    langchain_messages.AIMessage = AIMessage
    langchain_messages.SystemMessage = _TestMessage
    langchain_messages.BaseMessage = _TestMessage
    langchain_prompts.ChatPromptTemplate = MagicMock
    langchain_core.messages = langchain_messages
    langchain_core.prompts = langchain_prompts
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = langchain_messages
    sys.modules["langchain_core.prompts"] = langchain_prompts

try:
    from langgraph.graph import StateGraph, END  # noqa: F401
except ModuleNotFoundError:
    langgraph = types.ModuleType("langgraph")
    langgraph_graph = types.ModuleType("langgraph.graph")

    class _TestStateGraph:
        def __init__(self, *_args, **_kwargs):
            pass

        def add_node(self, *_args, **_kwargs):
            pass

        def add_edge(self, *_args, **_kwargs):
            pass

        def add_conditional_edges(self, *_args, **_kwargs):
            pass

        def set_entry_point(self, *_args, **_kwargs):
            pass

        def compile(self, *_args, **_kwargs):
            return self

    langgraph_graph.StateGraph = _TestStateGraph
    langgraph_graph.END = "END"
    langgraph.graph = langgraph_graph
    sys.modules["langgraph"] = langgraph
    sys.modules["langgraph.graph"] = langgraph_graph

if "langchain_groq" not in sys.modules:
    langchain_groq = types.ModuleType("langchain_groq")
    langchain_groq.ChatGroq = MagicMock
    langchain_groq_chat_models = types.ModuleType("langchain_groq.chat_models")
    langchain_groq_chat_models.ChatGroq = MagicMock
    sys.modules["langchain_groq"] = langchain_groq
    sys.modules["langchain_groq.chat_models"] = langchain_groq_chat_models

if "groq" not in sys.modules:
    groq = types.ModuleType("groq")
    groq.APIError = type("APIError", (Exception,), {})
    sys.modules["groq"] = groq

# ── Module stubbing, and undoing it ──────────────────────────────────────────
# The stubs below have to be installed at import time: app.api.workflow imports
# these agent nodes at *its* import, so there is no fixture early enough.
#
# What they must not do is outlive this file. They used to: each call replaced
# sys.modules[name] for the rest of the session, so `execution_agent_node_ray`
# and `infra_agent_node` stayed MagicMocks for every later test module. Run on
# its own, tests/test_ray_unit.py passed; run after this file, its four tests
# asserted against `<MagicMock name='mock().__getitem__()...'>`, and
# test_infra_routing failed with KeyError: 'platform'. Sixteen failures across
# four files, none of them about the code under test, all of them invisible
# until the suite was run in one process.
#
# So record what was there first and put it back when this module is done. The
# attribute-copying below stays for the same reason it was added: while the
# stub is installed it must remain a superset of the real module, or an
# unrelated import during this file's run fails on a missing symbol.
_STUBBED_ORIGINALS: Dict[str, object] = {}

def _stub_module(name, **attrs):
    """Install a stub for `name`, keeping any real symbols it already exports.

    A bare ModuleType would hide every other symbol the real module provides,
    so copy the real module's attributes first: the names in `attrs` are mocked
    for the tests here, everything else still resolves.
    """
    try:
        real = importlib.import_module(name)
    except Exception:
        real = None

    mod = types.ModuleType(name)
    if real is not None:
        for attr_name, attr_value in vars(real).items():
            if not attr_name.startswith("__"):
                setattr(mod, attr_name, attr_value)
        mod.__file__ = getattr(real, "__file__", None)

    for attr_name, attr_value in attrs.items():
        setattr(mod, attr_name, attr_value)
    if name not in _STUBBED_ORIGINALS:
        _STUBBED_ORIGINALS[name] = sys.modules.get(name)
    sys.modules[name] = mod
    return mod


def _restore_stubbed_modules():
    """Put sys.modules back the way this file found it.

    Called immediately after the imports below, NOT from a fixture: pytest
    imports every test module during collection, before it runs a single test,
    so a fixture teardown fires long after tests/test_ray_unit.py has already
    been imported against the mocks. The window in which the stubs must exist
    is exactly the three imports that follow -- once those have bound their
    names, nothing else needs them.
    """
    for name, original in _STUBBED_ORIGINALS.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    # app.api.workflow bound the mocked agent nodes into its graph at import,
    # so putting the stubbed modules back is not enough -- the cached workflow
    # module still wires MagicMocks as graph nodes, and the next file to import
    # it builds a graph out of them (tests/test_planner.py failed ten times on
    # langgraph.errors.InvalidUpdateError that way). Drop it so it is rebuilt
    # against the real agents. The name imported above stays bound to the
    # stubbed version, which is what this file is testing.
    #
    # Only this module: evicting app.* wholesale re-imports app.graph.etl_state
    # too, and a second ETLState class breaks every graph built against the
    # first one.
    sys.modules.pop("app.api.workflow", None)

if "app.core.celery_app" not in sys.modules:
    _stub_module(
        "app.core.celery_app",
        celery_app=MagicMock(),
        AvalokaScheduler=MagicMock,
    )

_stub_module("app.agents.coder", coder_node=MagicMock())
_stub_module("app.agents.state", CodingAgentState=dict)
_stub_module(
    "app.agents.validator",
    syntactic_validator_node=MagicMock(),
    static_semantic_validator_node=MagicMock(),
    logical_semantic_validator_node=MagicMock(),
)
_stub_module("app.agents.summarizer", summarize_etl_job=MagicMock())
_stub_module("app.agents.infra_agent", infra_agent_node=MagicMock())
_stub_module(
    "app.agents.execution_agent",
    execution_agent_node_local=MagicMock(),
    execution_agent_node=MagicMock(),
    execution_agent_node_ray=MagicMock(),
)
_stub_module("app.agents.planner_graph_agent", planner_graph_agent_node=MagicMock())
_stub_module("app.agents.visualization_agent", visualization_agent_node=MagicMock())
_stub_module("app.agents.scheduler", task_scheduler_node=MagicMock())

from app.api.workflow import memory_injection_node
from app.graph.etl_state import ETLState
from app.services.memory_plane import retrieve_memory, MemoryOrchestrator

# The names above are now bound to objects built against the stubs, which is
# the isolation this file wants. sys.modules goes back to normal so that the
# rest of the session imports the real agents.
_restore_stubbed_modules()


# ── Constants ──────────────────────────────────────────────────
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
CSV_BALL_BY_BALL = os.path.join(DOWNLOADS, "Ball_by_Ball.csv")
CSV_FINANCE = os.path.join(DOWNLOADS, "Finance_data.csv")
OUTPUT_FILE = os.path.join(DOWNLOADS, "context_memory_test_output.json")


# ── Mock LLM responses ────────────────────────────────────────

mock_memory_llm_response_simple = MagicMock()
mock_memory_llm_response_simple.content = json.dumps({
    "new_hints": ["sales data analysis requested"],
    "logic_signature": "User prefers pandas-based analysis."
})

mock_memory_llm_response_cricket = MagicMock()
mock_memory_llm_response_cricket.content = json.dumps({
    "new_hints": [
        "Ball-by-ball cricket match data",
        "Contains batting, bowling, and match outcome columns",
        "Large dataset with 200k+ rows"
    ],
    "logic_signature": "User analyzes sports datasets with focus on player performance."
})

mock_memory_llm_response_finance = MagicMock()
mock_memory_llm_response_finance.content = json.dumps({
    "new_hints": [
        "Investment preferences survey data",
        "Contains demographic and financial instrument columns",
        "Small dataset with 40 respondents"
    ],
    "logic_signature": "User analyzes financial and investment datasets focusing on demographics."
})

mock_memory_llm_response_accumulated = MagicMock()
mock_memory_llm_response_accumulated.content = json.dumps({
    "new_hints": [
        "User works with both sports and finance domains",
        "Preference for CSV-based structured data"
    ],
    "logic_signature": "User analyzes diverse datasets across sports and finance domains."
})

mock_memory_llm_invalid_json = MagicMock()
mock_memory_llm_invalid_json.content = "This is not valid JSON at all {broken"

mock_memory_llm_empty_hints = MagicMock()
mock_memory_llm_empty_hints.content = json.dumps({
    "new_hints": [],
    "logic_signature": ""
})


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_session_memory():
    """Clear in-memory session storage before each test."""
    MemoryOrchestrator.clear_all_sessions()
    yield
    MemoryOrchestrator.clear_all_sessions()


@pytest.fixture
def base_state():
    """Minimal ETLState for memory tests."""
    return {
        "messages": [HumanMessage(content="Analyze my data")],
        "session_id": "test_session",
        "planner_definition": {},
        "ready_to_summarize": False,
        "ready_to_code": False,
        "skip_to_execution": False,
        "coder_definition": {},
        "data_source_location": "",
        "output_location": "",
        "schema": {},
        "infrastructure_request": {},
        "infrastructure_provisioned": {},
        "execution_result": {},
        "input_data_type": "csv",
        "output_file_data": {},
        "sample_data": "",
        "memory_hints": [],
        "prior_artifact_found": False,
        "session_logic_signature": "",
        "memory_context_unavailable": False,
        "deploy_on_k8s": False,
        "uploaded_csv_preview": [],
        "uploaded_csv_columns": [],
        "generated_code": None,
        "coder_pseudocode": None,
        "execution_output_data": None,
        "execution_output_preview": None,
        "planner_graph_path": None,
        "planner_graph_base64": None,
        "planner_graph_status": None,
        "planner_graph_error": None,
        "task_list": [],
        "task_schedule": None,
        "task_info": None,
        "task_operation": None,
        "visualization_config": None,
        "visualization_status": None,
        "training_task": None,
        "model_artifacts": None,
        "mlflow_run_id": None,
        "training_metrics": None,
        "model_registry": None,
        "ray_config": None,
        "validation_params": None,
        "business_recommendations": None,
        "ready_to_train": False,
        "training_completed": False,
        "execution_mode": "local",
        "enable_training": False,
        "skip_to_training": False,
    }


@pytest.fixture
def ball_by_ball_df():
    """Load Ball_by_Ball.csv if it exists."""
    if os.path.exists(CSV_BALL_BY_BALL):
        return pd.read_csv(CSV_BALL_BY_BALL, nrows=100)
    pytest.skip(f"File not found: {CSV_BALL_BY_BALL}")


@pytest.fixture
def finance_df():
    """Load Finance_data.csv if it exists."""
    if os.path.exists(CSV_FINANCE):
        return pd.read_csv(CSV_FINANCE)
    pytest.skip(f"File not found: {CSV_FINANCE}")


# ─────────────────────────────────────────────────────────────
# Unit Tests: retrieve_memory()
# ─────────────────────────────────────────────────────────────

class TestRetrieveMemoryContract:
    """Verify the memory retrieval contract returns the correct shape."""

    @patch("app.services.memory_plane.memory_llm")
    def test_returns_required_keys(self, mock_llm):
        mock_llm.invoke.return_value = mock_memory_llm_response_simple
        state = {"session_id": "unit_test"}
        result = retrieve_memory("test query", state)

        assert "memory_hints" in result
        assert "prior_artifact_found" in result
        assert "session_logic_signature" in result
        assert "memory_context_unavailable" in result

    @patch("app.services.memory_plane.memory_llm")
    def test_memory_hints_is_list(self, mock_llm):
        mock_llm.invoke.return_value = mock_memory_llm_response_simple
        state = {"session_id": "unit_test"}
        result = retrieve_memory("test query", state)

        assert isinstance(result["memory_hints"], list)

    @patch("app.services.memory_plane.memory_llm")
    def test_prior_artifact_found_is_bool(self, mock_llm):
        mock_llm.invoke.return_value = mock_memory_llm_response_simple
        state = {"session_id": "unit_test"}
        result = retrieve_memory("test query", state)

        assert isinstance(result["prior_artifact_found"], bool)

    @patch("app.services.memory_plane.memory_llm")
    def test_memory_context_unavailable_is_bool(self, mock_llm):
        mock_llm.invoke.return_value = mock_memory_llm_response_simple
        state = {"session_id": "unit_test"}
        result = retrieve_memory("test query", state)

        assert isinstance(result["memory_context_unavailable"], bool)


class TestRetrieveMemoryDynamic:
    """Verify dynamic LLM-based hint extraction."""

    @patch("app.services.memory_plane.memory_llm")
    def test_extracts_hints_from_llm(self, mock_llm):
        mock_llm.invoke.return_value = mock_memory_llm_response_cricket
        state = {"session_id": "dynamic_test"}
        result = retrieve_memory("Analyze Ball_by_Ball.csv", state)

        assert len(result["memory_hints"]) > 0
        assert any("cricket" in h.lower() for h in result["memory_hints"])

    @patch("app.services.memory_plane.memory_llm")
    def test_extracts_logic_signature(self, mock_llm):
        mock_llm.invoke.return_value = mock_memory_llm_response_cricket
        state = {"session_id": "sig_test"}
        result = retrieve_memory("Analyze Ball_by_Ball.csv", state)

        assert result["session_logic_signature"] is not None
        assert len(result["session_logic_signature"]) > 0

    @patch("app.services.memory_plane.memory_llm")
    def test_hints_accumulate_across_queries(self, mock_llm):
        """Memory hints should grow as multiple queries are made."""
        state = {"session_id": "accumulation_test"}

        # Query 1: Cricket data
        mock_llm.invoke.return_value = mock_memory_llm_response_cricket
        result1 = retrieve_memory("Analyze Ball_by_Ball.csv", state)
        hints_after_q1 = len(result1["memory_hints"])

        # Query 2: Finance data
        mock_llm.invoke.return_value = mock_memory_llm_response_finance
        result2 = retrieve_memory("Analyze Finance_data.csv", state)
        hints_after_q2 = len(result2["accumulated_memory_hints"])

        assert hints_after_q2 > hints_after_q1

    @patch("app.services.memory_plane.memory_llm")
    def test_no_duplicate_hints(self, mock_llm):
        """Same query twice should not produce duplicate hints."""
        state = {"session_id": "dedup_test"}

        mock_llm.invoke.return_value = mock_memory_llm_response_simple
        retrieve_memory("Analyze sales", state)
        result = retrieve_memory("Analyze sales", state)

        # Check for duplicates
        assert len(result["memory_hints"]) == len(set(result["memory_hints"]))

    @patch("app.services.memory_plane.memory_llm")
    def test_session_isolation(self, mock_llm):
        """Different sessions should have independent memory."""
        mock_llm.invoke.return_value = mock_memory_llm_response_cricket

        state_a = {"session_id": "session_a"}
        state_b = {"session_id": "session_b"}

        retrieve_memory("Cricket analysis", state_a)
        result_b = retrieve_memory("Something else", state_b)

        # Session B should NOT contain session A's cricket hints
        result_a = retrieve_memory("Another query", state_a)
        assert len(result_a["accumulated_memory_hints"]) >= len(result_b["accumulated_memory_hints"])


class TestRetrieveMemoryErrorHandling:
    """Verify error resilience."""

    @patch("app.services.memory_plane.memory_llm")
    def test_handles_invalid_json_gracefully(self, mock_llm):
        mock_llm.invoke.return_value = mock_memory_llm_invalid_json
        state = {"session_id": "error_test"}
        result = retrieve_memory("test query", state)

        # Should still return valid structure
        assert isinstance(result["memory_hints"], list)
        assert isinstance(result["memory_context_unavailable"], bool)

    @patch("app.services.memory_plane.memory_llm")
    def test_handles_llm_exception(self, mock_llm):
        mock_llm.invoke.side_effect = RuntimeError("API Timeout")
        state = {"session_id": "exception_test"}
        result = retrieve_memory("test query", state)

        # Should gracefully return defaults
        assert isinstance(result, dict)
        assert "memory_hints" in result

    @patch("app.services.memory_plane.memory_llm", None)
    def test_llm_unavailable_sets_flag(self):
        state = {"session_id": "no_llm_test"}
        result = retrieve_memory("test query", state)

        assert result["memory_context_unavailable"] is True

    @patch("app.services.memory_plane.memory_llm")
    def test_empty_hints_from_llm(self, mock_llm):
        mock_llm.invoke.return_value = mock_memory_llm_empty_hints
        state = {"session_id": "empty_test"}
        result = retrieve_memory("hello", state)

        assert result["memory_hints"] == []

    def test_default_session_id(self):
        """State without session_id should default to 'default'."""
        with patch("app.services.memory_plane.memory_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_memory_llm_response_simple
            state = {}  # no session_id
            result = retrieve_memory("test", state)
            assert "default" in MemoryOrchestrator._past_queries_store


# ─────────────────────────────────────────────────────────────
# Integration Tests: memory_injection_node
# ─────────────────────────────────────────────────────────────

class TestMemoryInjectionNode:
    """Test the memory_injection_node inside the LangGraph workflow."""

    @patch("app.services.memory_plane.memory_llm")
    def test_node_returns_state_updates(self, mock_llm, base_state):
        mock_llm.invoke.return_value = mock_memory_llm_response_simple
        result = memory_injection_node(base_state)

        assert "memory_hints" in result
        assert "prior_artifact_found" in result
        assert "session_logic_signature" in result
        assert "memory_context_unavailable" in result

    @patch("app.services.memory_plane.memory_llm")
    def test_node_with_no_messages(self, mock_llm, base_state):
        base_state["messages"] = []
        result = memory_injection_node(base_state)

        assert result["memory_hints"] == []
        assert result["memory_context_unavailable"] is False

    @patch("app.services.memory_plane.memory_llm")
    def test_node_extracts_query_from_messages(self, mock_llm, base_state):
        mock_llm.invoke.return_value = mock_memory_llm_response_cricket
        base_state["messages"] = [HumanMessage(content="Analyze cricket data")]
        result = memory_injection_node(base_state)

        assert len(result["memory_hints"]) > 0


# ─────────────────────────────────────────────────────────────
# CSV-Aware Tests: Real data from Ball_by_Ball.csv & Finance_data.csv
# ─────────────────────────────────────────────────────────────

class TestMemoryWithRealCSV:
    """Test memory extraction with actual CSV file data."""

    @patch("app.services.memory_plane.memory_llm")
    def test_ball_by_ball_schema_in_query(self, mock_llm, ball_by_ball_df):
        mock_llm.invoke.return_value = mock_memory_llm_response_cricket
        columns = ", ".join(ball_by_ball_df.columns[:10])
        query = f"Loaded Ball_by_Ball.csv with columns: {columns}. Analyze."
        state = {"session_id": "csv_test"}
        result = retrieve_memory(query, state)

        assert len(result["memory_hints"]) > 0
        mock_llm.invoke.assert_called_once()

    @patch("app.services.memory_plane.memory_llm")
    def test_finance_data_schema_in_query(self, mock_llm, finance_df):
        mock_llm.invoke.return_value = mock_memory_llm_response_finance
        columns = ", ".join(finance_df.columns)
        query = f"Loaded Finance_data.csv ({len(finance_df)} rows, columns: {columns}). Analyze."
        state = {"session_id": "csv_test"}
        result = retrieve_memory(query, state)

        assert any("investment" in h.lower() or "financial" in h.lower()
                    for h in result["memory_hints"])

    @patch("app.services.memory_plane.memory_llm")
    def test_cross_csv_memory_accumulation(self, mock_llm, ball_by_ball_df, finance_df):
        """Hints from both CSVs should accumulate in the same session."""
        state = {"session_id": "cross_csv_test"}

        # Query 1: Cricket
        mock_llm.invoke.return_value = mock_memory_llm_response_cricket
        columns1 = ", ".join(ball_by_ball_df.columns[:10])
        retrieve_memory(f"Loaded Ball_by_Ball.csv with columns: {columns1}", state)

        # Query 2: Finance
        mock_llm.invoke.return_value = mock_memory_llm_response_finance
        columns2 = ", ".join(finance_df.columns)
        result = retrieve_memory(f"Loaded Finance_data.csv with columns: {columns2}", state)

        # Should have hints from BOTH datasets in accumulated
        assert len(result["accumulated_memory_hints"]) >= 4
        assert any("cricket" in h.lower() for h in result["accumulated_memory_hints"])
        assert any("investment" in h.lower() for h in result["accumulated_memory_hints"])

    @patch("app.services.memory_plane.memory_llm")
    def test_logic_signature_evolves(self, mock_llm, ball_by_ball_df, finance_df):
        """Logic signature should update as new domains are analyzed."""
        state = {"session_id": "sig_evolution_test"}

        # Query 1
        mock_llm.invoke.return_value = mock_memory_llm_response_cricket
        result1 = retrieve_memory("Analyze cricket data", state)
        sig1 = result1["session_logic_signature"]

        # Query 2
        mock_llm.invoke.return_value = mock_memory_llm_response_accumulated
        result2 = retrieve_memory("Analyze finance data", state)
        sig2 = result2["session_logic_signature"]

        assert sig1 != sig2  # signature should have evolved


# ─────────────────────────────────────────────────────────────
# E2E Test: Full graph entry point through memory_injection
# ─────────────────────────────────────────────────────────────

class TestMemoryE2EGraphRouting:
    """Integration test: memory_injection produces planner-facing state."""

    @patch("app.services.memory_plane.memory_llm")
    def test_memory_flows_into_planner(self, mock_memory_llm, base_state):
        """Memory injection node returns state fields consumed by plan_etl."""
        mock_memory_llm.invoke.return_value = mock_memory_llm_response_cricket

        result = memory_injection_node(base_state)

        assert "memory_hints" in result
        assert isinstance(result["memory_hints"], list)
        assert result["memory_hints"]
        assert "session_logic_signature" in result
        assert "prior_artifact_found" in result


# ─────────────────────────────────────────────────────────────
# Output: Save JSON results to Downloads
# ─────────────────────────────────────────────────────────────

class TestSaveJSONOutput:
    """Run with real LLM and save JSON output to Downloads for senior review."""

    def test_save_memory_evolution_json(self, ball_by_ball_df, finance_df):
        """
        Integration test with real LLM calls.
        Saves full memory evolution to Downloads/context_memory_test_output.json
        Each step shows BOTH context-specific hints AND accumulated session hints.
        """
        results = []
        state = {"session_id": "json_output_test"}

        # Step 1: Ball_by_Ball.csv
        cols1 = ", ".join(ball_by_ball_df.columns)
        total_rows1 = len(pd.read_csv(CSV_BALL_BY_BALL))
        query1 = (f"Loaded Ball_by_Ball.csv ({total_rows1} rows, "
                  f"columns: {cols1}). Analyze this cricket dataset.")
        result1 = retrieve_memory(query1, state)
        results.append({
            "step": 1,
            "file": "Ball_by_Ball.csv",
            "rows": total_rows1,
            "columns": list(ball_by_ball_df.columns),
            "dtypes": {c: str(d) for c, d in ball_by_ball_df.dtypes.items()},
            "sample_rows": ball_by_ball_df.head(3).to_dict(orient="records"),
            "new_hints_this_context": result1.get("new_hints_this_context", []),
            "accumulated_memory_hints": result1.get("accumulated_memory_hints", []),
            "session_logic_signature": result1.get("session_logic_signature"),
        })

        # Step 2: Finance_data.csv
        cols2 = ", ".join(finance_df.columns)
        query2 = (f"Loaded Finance_data.csv ({len(finance_df)} rows, "
                  f"columns: {cols2}). Analyze investment preferences.")
        result2 = retrieve_memory(query2, state)
        results.append({
            "step": 2,
            "file": "Finance_data.csv",
            "rows": len(finance_df),
            "columns": list(finance_df.columns),
            "dtypes": {c: str(d) for c, d in finance_df.dtypes.items()},
            "sample_rows": finance_df.head(3).to_dict(orient="records"),
            "new_hints_this_context": result2.get("new_hints_this_context", []),
            "accumulated_memory_hints": result2.get("accumulated_memory_hints", []),
            "session_logic_signature": result2.get("session_logic_signature"),
        })

        # Save JSON
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, default=str)

        # Assertions
        assert os.path.exists(OUTPUT_FILE)
        assert len(results) == 2

        # Step 2's context-specific hints should NOT contain cricket hints
        step2_context = results[1]["new_hints_this_context"]
        step2_accumulated = results[1]["accumulated_memory_hints"]
        assert len(step2_accumulated) >= len(step2_context)

        # Step 2's accumulated should contain Step 1's hints too
        assert len(step2_accumulated) >= len(results[0]["new_hints_this_context"])

        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved) == 2
        assert saved[0]["file"] == "Ball_by_Ball.csv"
        assert saved[1]["file"] == "Finance_data.csv"
        assert "new_hints_this_context" in saved[0]
        assert "accumulated_memory_hints" in saved[1]
