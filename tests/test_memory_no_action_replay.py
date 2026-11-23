"""
Regression tests: the memory plane never replays a data-moving action into an
analysis plan (app/services/memory_actions.py).

rc10: after a transfer was cancelled, memory surfaced "Transfer patients table
to e2ewhgroq as d1_e" and the coder's blueprint planned a to_sql write.
rc11/rc12: the session logic signature ("Use SQL to copy and rename the
table ...") went into the planner's csv_info, and about 1 plan in 4 for
"average age of patients with stroke" after a transfer session described a
transfer; the write guard refused it, so the user saw a false refusal.

Each session below runs through the real retrieval and background learning,
with a stand-in for the learner LLM that does what the real one did in the
rweb11 logs: given a transfer among the queries, it extracts the transfer as a
"preference" and a signature. The planner's csv_info and the coder's blueprint
prompt are then captured from the real planner and coder code.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

try:  # pragma: no cover
    import cachetools  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _ct = types.ModuleType("cachetools")

    class _TTLCache(dict):
        def __init__(self, maxsize=None, ttl=None):
            super().__init__()

    _ct.TTLCache = _TTLCache
    sys.modules["cachetools"] = _ct

from langchain_core.messages import AIMessage, HumanMessage

from app.services.memory_plane import MemoryOrchestrator

SESSION = "sess-no-action-replay"
TRANSFER = "Transfer patients table to e2ewhgroq as d1_e"
READ_QUESTION = "average age of patients with stroke"
PENDING_TRANSFER = {"type": "dta_transfer", "awaiting": "confirm"}

# What the learner LLM wrote in the rc11/rc12 sessions (rweb11 final-api.log).
LEARNED_TRANSFER_HINT = TRANSFER
LEARNED_TRANSFER_SIGS = (
    "Use SQL to copy and rename the table",
    "Prefers explicit table transfer with renaming, preserving exact data",
    "Perform direct table copy operations applying a '_copy' suffix to the new table name",
)
# Analysis memory that must still flow.
CHART_PREFERENCE = "The user prefers bar charts for category comparisons"
COLUMN_MEANING = "Column 'stroke' is 1 when the patient had a stroke, 0 otherwise"
ANALYSIS_SIG = "Prefers concise, data-driven answers with exact figures"

_WRITE_WORDS = ("transfer", "copy", "copies", "rename", "create table", "to_sql",
                "insert", "export", "e2ewhgroq", "d1_e", "_copy")


def _write_content(text):
    low = str(text or "").lower()
    return [w for w in _WRITE_WORDS if w in low]


def _fake_learner(query, past_queries):
    """The learner LLM as observed: a transfer anywhere in what it is shown
    becomes a learned preference and signature."""
    seen = " ".join([str(q) for q in past_queries] + [str(query)]).lower()
    if "transfer" in seen:
        return {"new_hints": [LEARNED_TRANSFER_HINT, COLUMN_MEANING],
                "logic_signature": " ".join(LEARNED_TRANSFER_SIGS)}
    return {"new_hints": [COLUMN_MEANING], "logic_signature": ANALYSIS_SIG}


class _SignatureStore:
    """Chroma's signature behaviour: appended per turn, served merged with a space."""

    def __init__(self):
        self.rows = {}
        self.reads = 0

    def connect(self):
        return True

    def update_user_signature(self, user_id, signature):
        self.rows.setdefault(user_id, []).append(signature)

    def get_user_signature(self, user_id):
        self.reads += 1
        rows = self.rows.get(user_id)
        return " ".join(rows[-10:]) if rows else None


def _orchestrator():
    mo = MemoryOrchestrator.__new__(MemoryOrchestrator)
    mo._explicit_llm = object()
    mo._init_runtime_state()
    mo.postgres = MagicMock()
    mo.postgres.check_artifact_exists.return_value = False
    mo.redis = MagicMock()
    mo.redis.search_schema_semantically.return_value = None
    mo.redis.get_json.return_value = None
    mo.chroma = _SignatureStore()
    mo.milvus = MagicMock()
    mo.milvus.search_similar_insights.return_value = []
    mo._embed = MagicMock(return_value=[0.0] * 8)
    mo._extract_context_with_llm = _fake_learner
    return mo


def _turn(mo, query, **state):
    state.setdefault("user_id", "u-1")
    result = mo._retrieve_memory_internal(query, session_id=SESSION,
                                          dataset_id="unknown_dataset", state=state)
    assert MemoryOrchestrator.wait_for_learning(timeout=300), "background learning never finished"
    return result


@pytest.fixture(autouse=True)
def _memory_isolation():
    MemoryOrchestrator.clear_all_sessions()
    yield
    MemoryOrchestrator.wait_for_learning(timeout=300)
    MemoryOrchestrator.clear_all_sessions()


# ── Capturing what the planner and coder are actually given ─────────────────


def _planner_csv_info(memory):
    """The csv_info the real planner hands its generate_code plan prompt."""
    from app.agents import planner
    from app.agents.planner import NoParams, PlannerOutput, ToolCall
    from app.graph.etl_state import ETLState

    captured = []

    class _PlanLLM:
        def invoke(self, *_a, **_k):
            return PlannerOutput(tool_call=ToolCall(name="generate_code", parameters=NoParams()))

        def __call__(self, prompt_value, *_a, **_k):
            captured.append(prompt_value.to_string())
            return AIMessage(content="1. Filter rows where stroke == 1.\n2. Report the mean of age.")

    state = ETLState(
        messages=[HumanMessage(content=READ_QUESTION)],
        user_prompt=READ_QUESTION,
        session_id=SESSION,
        memory_hints=memory.get("memory_hints") or [],
        session_logic_signature=memory.get("session_logic_signature"),
    )
    with patch.object(planner, "llm", _PlanLLM()):
        result = planner.plan_etl_job(state)
    assert captured, "the planner never built its generate_code plan prompt"
    assert result.get("plan"), "the plan was not accepted"
    return captured[-1]


def _coder_blueprint_prompt(memory):
    """The human prompt the real coder sends for its pseudocode blueprint."""
    from app.agents import coder

    captured = []

    class _CoderLLM:
        def invoke(self, messages, *_a, **_k):
            captured.append(messages[-1].content)
            return AIMessage(content="1. Filter stroke == 1.\n2. Mean of age.")

    state = {
        "plan": "1. Filter rows where stroke == 1.\n2. Report the mean of age.",
        "user_prompt": READ_QUESTION,
        "schema": {"age": "float64", "stroke": "int64"},
        "memory_hints": memory.get("memory_hints") or [],
        "session_logic_signature": memory.get("session_logic_signature"),
    }
    with patch.object(coder, "coder_llm", _CoderLLM()):
        coder._generate_pseudocode(state)
    assert captured, "the coder never built its blueprint prompt"
    return captured[-1]


def _memory_section(prompt):
    """The memory blocks of a prompt (the request itself is not memory)."""
    out = []
    for marker in ("[CONTEXT MEMORY", "[SESSION LOGIC SIGNATURE]"):
        at = prompt.find(marker)
        if at != -1:
            out.append(prompt[at:at + 1500])
    return "\n".join(out)


def _assert_no_write_replay(memory):
    assert not _write_content(" ".join(memory.get("memory_hints") or [])), memory["memory_hints"]
    assert not _write_content(memory.get("session_logic_signature")), memory["session_logic_signature"]
    planner_ctx = _memory_section(_planner_csv_info(memory))
    coder_ctx = _memory_section(_coder_blueprint_prompt(memory))
    assert not _write_content(planner_ctx), f"planner csv_info replays a write: {planner_ctx!r}"
    assert not _write_content(coder_ctx), f"coder blueprint replays a write: {coder_ctx!r}"
    return planner_ctx, coder_ctx


# ── Cancelled transfer, then the read question ───────────────────────────────


def test_cancelled_transfer_is_not_replayed_into_analysis_planning():
    mo = _orchestrator()
    _turn(mo, TRANSFER)
    _turn(mo, "no", pending_clarification=PENDING_TRANSFER)  # the cancel
    memory = _turn(mo, READ_QUESTION)
    assert mo.chroma.reads, "precondition: the signature store must be consulted"
    _assert_no_write_replay(memory)
    # Nothing was recorded as a suggestion either.
    stored = MemoryOrchestrator._session_hints_store.get(SESSION, [])
    assert not _write_content(" ".join(stored)), stored
    assert not _write_content(" ".join(mo.chroma.rows.get("u-1", []))), mo.chroma.rows


def test_completed_transfer_is_not_replayed_into_analysis_planning():
    mo = _orchestrator()
    _turn(mo, TRANSFER)
    _turn(mo, "yes", pending_clarification=PENDING_TRANSFER)  # the confirm
    memory = _turn(mo, READ_QUESTION, dta_last_transfer={"success": True, "dest_table": "d1_e"})
    assert mo.chroma.reads, "precondition: the signature store must be consulted"
    _assert_no_write_replay(memory)


def test_a_later_transfer_turn_is_not_handed_the_cancelled_one():
    """'Do it again' is never inferred: a new transfer turn gets no transfer
    from memory either."""
    mo = _orchestrator()
    _turn(mo, TRANSFER)
    _turn(mo, "no", pending_clarification=PENDING_TRANSFER)
    memory = _turn(mo, "transfer it to the warehouse")
    assert not _write_content(" ".join(memory["memory_hints"])), memory["memory_hints"]
    assert not _write_content(memory["session_logic_signature"]), memory["session_logic_signature"]


def test_write_memory_stored_before_the_fix_is_not_served():
    """Rows learned before this fix stay in Redis and Chroma for days; they are
    filtered on the way out. The session's own hints keep their analysis
    parts. A signature row keyed by the bare user predates the thread scope
    rule and names no thread, so none of it is served any more
    (tests/test_memory_thread_scope.py)."""
    mo = _orchestrator()
    MemoryOrchestrator._session_hints_store[SESSION] = [LEARNED_TRANSFER_HINT, COLUMN_MEANING]
    for sig in LEARNED_TRANSFER_SIGS + (ANALYSIS_SIG,):
        mo.chroma.update_user_signature("u-1", sig)
    memory = _turn(mo, READ_QUESTION)
    _assert_no_write_replay(memory)
    assert COLUMN_MEANING in memory["memory_hints"]
    assert ANALYSIS_SIG not in memory["session_logic_signature"]


def test_write_memory_already_on_state_never_reaches_the_prompts():
    """The planner and coder filter what is on the state, whatever put it there."""
    poisoned = {
        "memory_hints": [LEARNED_TRANSFER_HINT, CHART_PREFERENCE],
        "session_logic_signature": " ".join(LEARNED_TRANSFER_SIGS) + " " + ANALYSIS_SIG,
    }
    planner_ctx = _memory_section(_planner_csv_info(poisoned))
    coder_ctx = _memory_section(_coder_blueprint_prompt(poisoned))
    for ctx in (planner_ctx, coder_ctx):
        assert not _write_content(ctx), ctx
        assert CHART_PREFERENCE in ctx
        assert ANALYSIS_SIG in ctx


# ── Normal analysis memory still flows ───────────────────────────────────────


def test_analysis_preferences_and_hints_still_flow():
    mo = _orchestrator()
    mo.store_user_preference(CHART_PREFERENCE, session_id=SESSION)
    _turn(mo, "how many patients had a stroke?")
    memory = _turn(mo, READ_QUESTION)
    assert CHART_PREFERENCE in memory["memory_hints"]
    assert COLUMN_MEANING in memory["memory_hints"]
    assert ANALYSIS_SIG in memory["session_logic_signature"]
    planner_ctx = _memory_section(_planner_csv_info(memory))
    coder_ctx = _memory_section(_coder_blueprint_prompt(memory))
    for ctx in (planner_ctx, coder_ctx):
        assert CHART_PREFERENCE in ctx
        assert COLUMN_MEANING in ctx
        assert ANALYSIS_SIG in ctx


def test_dataset_facts_with_write_words_in_names_are_kept():
    """Column names are facts about the data, not actions."""
    from app.services.memory_actions import drop_write_actions
    facts = ["Column transfer_amount is in USD", "export_value is the FOB value",
             "Domain insight: transfers(id, amount, created_at)",
             "Use df.copy() before mutating a slice"]
    assert drop_write_actions(facts) == facts
