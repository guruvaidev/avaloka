"""
Regression tests: memory learned in one thread does not steer another thread's
question (app/services/memory_scope.py).

rc12 web check, clean stack, no transfers: every question ran in a fresh
thread, and the user-scoped Chroma signature carried what earlier threads had
asked for into the next one. The age3 payload (rweb12 final-api.log) was

    'session_logic_signature': 'They prefer concise numeric answers about
    missing data User prefers a concise, direct answer. Return the work_type
    with the highest average stroke rate'

for "average age of patients with stroke". age3 also computed top_work_type;
age4 failed its contract (plan promised 'average_age', result had work_type,
stroke_rate, missing_bmi_count); work4 showed "201 missing BMI records" (the
total) as Self-employed's.

Each thread below runs through the real retrieval and background learning,
with stores that keep rows under the key they are written with, and a stand-in
for the learner LLM that writes what the real one wrote in that run. The
planner's csv_info and the coder's blueprint prompt are captured from the real
planner and coder code.
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

USER = "u-rc12"

# The three questions of the rd_clean run, verbatim.
WORK_Q = "which work_type has the highest stroke rate?"
MISS_Q = "how many patients have missing bmi?"
AGE_Q = "average age of patients with stroke"

# What the learner LLM wrote for them (rweb12 final-api.log Memory Payload lines).
WORK_SIG = "User prefers a concise, direct answer. Return the work_type with the highest average stroke rate"
WORK_HINT = "Return the work_type with the highest stroke rate, optionally including the stroke rate value."
MISS_SIG = "They prefer concise numeric answers about missing data"
MISS_HINT = "User wants exact counts of missing BMI values in the patients table."
# General preferences and dataset facts, which must still cross threads.
STYLE_PREF = "User prefers a concise, direct answer."
UNITS_PREF = "User prefers metric units"
COLUMN_MEANING = "Column 'stroke' is 1 when the patient had a stroke, 0 otherwise"
PATIENTS_SCHEMA = "patients(id, gender, age, work_type, bmi, stroke)"
ADMISSIONS_SCHEMA = "admissions(id, ward, length_of_stay)"

# Thread-specific text: none of it may reach another thread's prompts. (The
# schema names work_type; that is a fact about the data, not an instruction.)
THREAD_SPECIFIC = ("Return the work_type", "highest", "stroke rate",
                   "missing data", "missing BMI", "concise numeric answer")
COLUMNS = ["id", "gender", "age", "hypertension", "heart_disease", "ever_married",
           "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke"]


def _fake_learner(query, past_queries):
    q = str(query).lower()
    if "work_type" in q:
        return {"new_hints": [WORK_HINT], "logic_signature": WORK_SIG}
    if "missing bmi" in q:
        return {"new_hints": [MISS_HINT, UNITS_PREF, COLUMN_MEANING], "logic_signature": MISS_SIG}
    return {"new_hints": [], "logic_signature": ""}


class _KeyedSignatures:
    """Chroma's signature store: appended under the key given, served merged."""

    def __init__(self):
        self.rows = {}

    def connect(self):
        return True

    def update_user_signature(self, user_id, signature):
        self.rows.setdefault(user_id, []).append(signature)

    def get_user_signature(self, user_id):
        rows = self.rows.get(user_id)
        return " ".join(rows[-10:]) if rows else None


class _KeyedInsights:
    """Milvus: rows filed under the session_id they are inserted with; a search
    returns that key's rows (the ranking is not what is under test)."""

    _vector_dim = 8

    def __init__(self):
        self.rows = {}

    def connect(self):
        return True

    def insert_insight(self, session_id, content, agent_role="assistant", vector=None, **_):
        self.rows.setdefault(session_id, []).append(content)

    def search_similar_insights(self, session_id, query_vector=None, top_k=None, **_):
        return [{"content": c} for c in self.rows.get(session_id, [])][: top_k or 5]


class _SchemaCache:
    """The L1 schema cache: one schema per dataset. Asked without a dataset it
    does what the semantic search does over a user's cache -- returns the best
    match of ALL their datasets, here the patients schema."""

    def __init__(self):
        self.schemas = {"patients": PATIENTS_SCHEMA, "admissions": ADMISSIONS_SCHEMA}

    def search_schema_semantically(self, user_id, query, dataset_id=None):
        if dataset_id:
            return self.schemas.get(dataset_id)
        return PATIENTS_SCHEMA


class _Artifacts:
    def __init__(self):
        self.rows = {}

    def connect(self):
        return True

    def record(self, query, meta):
        self.rows[query] = meta

    def check_artifact_exists(self, query, user_id="default"):
        return query in self.rows

    def get_artifact(self, query, user_id="default"):
        return self.rows.get(query)


def _orchestrator():
    mo = MemoryOrchestrator.__new__(MemoryOrchestrator)
    mo._explicit_llm = object()
    mo._init_runtime_state()
    mo.postgres = _Artifacts()
    cache = _SchemaCache()
    mo.redis = MagicMock()
    mo.redis.search_schema_semantically.side_effect = cache.search_schema_semantically
    mo.redis.get_json.return_value = None
    mo.chroma = _KeyedSignatures()
    mo.milvus = _KeyedInsights()
    mo._embed = MagicMock(return_value=[0.0] * 8)
    mo._extract_context_with_llm = _fake_learner
    return mo


def _thread(name, table="patients"):
    """A fresh thread as the rd_clean run made them: new session, new thread,
    new dataset id, the same database table."""
    return {"user_id": USER, "thread_id": f"thread-{name}", "session_id": f"sess-{name}",
            "dataset_id": f"ds-{name}", "active_db_customer_id": "e2eclinicgroq",
            "active_db_source_table": table, "uploaded_csv_columns": COLUMNS,
            "_cache_dataset": table}


def _turn(mo, query, thread):
    state = {k: v for k, v in thread.items() if not k.startswith("_")}
    result = mo._retrieve_memory_internal(query, session_id=thread["session_id"],
                                          dataset_id=thread["_cache_dataset"], state=state)
    assert MemoryOrchestrator.wait_for_learning(timeout=300), "background learning never finished"
    return result


@pytest.fixture(autouse=True)
def _memory_isolation():
    MemoryOrchestrator.clear_all_sessions()
    yield
    MemoryOrchestrator.wait_for_learning(timeout=300)
    MemoryOrchestrator.clear_all_sessions()


# ── Capturing what the planner and coder are actually given ─────────────────


def _planner_csv_info(memory, question=AGE_Q):
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
        messages=[HumanMessage(content=question)],
        user_prompt=question,
        session_id="sess-c",
        memory_hints=memory.get("memory_hints") or [],
        session_logic_signature=memory.get("session_logic_signature"),
    )
    with patch.object(planner, "llm", _PlanLLM()):
        result = planner.plan_etl_job(state)
    assert captured, "the planner never built its generate_code plan prompt"
    assert result.get("plan"), "the plan was not accepted"
    return captured[-1]


def _coder_blueprint_prompt(memory, question=AGE_Q):
    """The human prompt the real coder sends for its pseudocode blueprint."""
    from app.agents import coder

    captured = []

    class _CoderLLM:
        def invoke(self, messages, *_a, **_k):
            captured.append(messages[-1].content)
            return AIMessage(content="1. Filter stroke == 1.\n2. Mean of age.")

    state = {
        "plan": "1. Filter rows where stroke == 1.\n2. Report the mean of age.",
        "user_prompt": question,
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


def _leaks(text):
    low = str(text or "").lower()
    return [s for s in THREAD_SPECIFIC if s.lower() in low]


def _memory_text(memory):
    return " ".join(list(memory.get("memory_hints") or [])
                    + list(memory.get("accumulated_memory_hints") or [])
                    + [memory.get("session_logic_signature") or ""])


def _run_rd_clean_sequence(mo):
    """Thread A asks the work_type question, thread B the missing-bmi one
    (twice, as the run did), then a fresh thread C asks the age question."""
    _turn(mo, WORK_Q, _thread("a"))
    _turn(mo, WORK_Q, _thread("a"))
    _turn(mo, MISS_Q, _thread("b"))
    _turn(mo, MISS_Q, _thread("b"))
    return _turn(mo, AGE_Q, _thread("c"))


# ── The defect: age3/age4/work4 ──────────────────────────────────────────────


def test_a_fresh_thread_gets_no_instruction_learned_in_another_thread():
    mo = _orchestrator()
    memory = _run_rd_clean_sequence(mo)
    assert mo.chroma.rows, "precondition: the learner must have stored signatures"

    assert not _leaks(_memory_text(memory)), memory
    planner_ctx = _memory_section(_planner_csv_info(memory))
    coder_ctx = _memory_section(_coder_blueprint_prompt(memory))
    assert not _leaks(planner_ctx), f"planner csv_info carries thread A/B's question: {planner_ctx!r}"
    assert not _leaks(coder_ctx), f"coder blueprint carries thread A/B's question: {coder_ctx!r}"


def test_the_thread_that_learned_an_instruction_still_has_it():
    """Thread scope, not no scope: thread A's next turn sees what A taught."""
    mo = _orchestrator()
    _run_rd_clean_sequence(mo)
    memory = _turn(mo, WORK_Q, _thread("a"))
    assert "Return the work_type with the highest average stroke rate" in memory["session_logic_signature"]
    assert WORK_HINT in memory["accumulated_memory_hints"]


# ── What must still cross threads ────────────────────────────────────────────


def test_general_preferences_still_reach_a_fresh_thread():
    mo = _orchestrator()
    memory = _run_rd_clean_sequence(mo)
    assert STYLE_PREF in (memory["session_logic_signature"] or ""), memory
    assert UNITS_PREF in memory["memory_hints"], memory
    planner_ctx = _memory_section(_planner_csv_info(memory))
    coder_ctx = _memory_section(_coder_blueprint_prompt(memory))
    for ctx in (planner_ctx, coder_ctx):
        assert STYLE_PREF in ctx
        assert UNITS_PREF in ctx


def test_domain_facts_for_the_same_dataset_reach_a_fresh_thread():
    mo = _orchestrator()
    memory = _run_rd_clean_sequence(mo)
    assert COLUMN_MEANING in memory["accumulated_memory_hints"], memory
    assert f"Domain insight: {PATIENTS_SCHEMA}" in memory["accumulated_memory_hints"], memory


def test_domain_facts_from_another_dataset_do_not():
    mo = _orchestrator()
    _run_rd_clean_sequence(mo)
    memory = _turn(mo, "average length of stay by ward", _thread("d", table="admissions"))
    held = " ".join(memory["accumulated_memory_hints"])
    assert COLUMN_MEANING not in held, held
    assert PATIENTS_SCHEMA not in held, held
    assert f"Domain insight: {ADMISSIONS_SCHEMA}" in memory["accumulated_memory_hints"]
    # The user's preferences are not tied to a dataset.
    assert UNITS_PREF in memory["memory_hints"]


def test_an_explicit_preference_crosses_threads_but_a_task_instruction_does_not():
    mo = _orchestrator()
    a = _thread("a")
    mo.store_user_preference(UNITS_PREF, session_id=a["thread_id"], scope_id=USER)
    mo.store_user_preference("Return the work_type with the highest stroke rate",
                             session_id=a["thread_id"], scope_id=USER)
    import time
    deadline = time.monotonic() + 30
    while len(sum(mo.milvus.rows.values(), [])) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    memory = _turn(mo, AGE_Q, _thread("c"))
    assert UNITS_PREF in memory["memory_hints"]
    assert not _leaks(_memory_text(memory)), memory


def test_a_prior_artifact_from_another_thread_is_not_offered_for_reuse():
    mo = _orchestrator()
    mo.postgres.record(AGE_Q, {"output_path": "a_output.csv", "status": "success",
                               "session": "sess-a", "thread_id": "thread-a"})
    other = _turn(mo, AGE_Q, _thread("c"))
    assert not other["prior_artifact_found"]
    assert not any("a_output.csv" in h for h in other["accumulated_memory_hints"])
    own = _turn(mo, AGE_Q, _thread("a"))
    assert own["prior_artifact_found"]
    assert any("a_output.csv" in h for h in own["accumulated_memory_hints"])


# ── Rows stored before the fix ──────────────────────────────────────────────


def test_unscoped_rows_stored_before_the_fix_are_not_served_to_any_thread():
    """Pre-fix rows are keyed by the bare user (or session): they say nothing
    about the thread they came from, so no thread gets them."""
    mo = _orchestrator()
    for sig in (MISS_SIG, WORK_SIG):
        mo.chroma.update_user_signature(USER, sig)
    mo.chroma.update_user_signature("sess-c", WORK_SIG)
    mo.milvus.insert_insight(USER, WORK_HINT, "memory_llm")
    mo.milvus.insert_insight("sess-c", MISS_HINT, "memory_llm")
    memory = _turn(mo, AGE_Q, _thread("c"))
    assert not _leaks(_memory_text(memory)), memory
    assert not _leaks(_memory_section(_planner_csv_info(memory)))
    assert not _leaks(_memory_section(_coder_blueprint_prompt(memory)))


# ── The classifier ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("text, kind", [
    ("User prefers a concise, direct answer.", "preference"),
    ("User prefers metric units", "preference"),
    ("Prefer clean pandas code.", "preference"),
    ("The user prefers bar charts for category comparisons", "preference"),
    ("They prefer concise numeric answers about missing data", "task"),
    ("Return the work_type with the highest average stroke rate", "task"),
    ("Provide concise numeric answer", "task"),
    ("User wants exact counts of missing BMI values in the patients table.", "task"),
    ("They prefer concise aggregate queries on patient data", "task"),
    ("prefers stroke rate charts", "task"),
    ("Column 'stroke' is 1 when the patient had a stroke, 0 otherwise", "domain"),
    ("Domain insight: patients(id, age)", "domain"),
    ("bmi is measured in kg/m2", "domain"),
])
def test_classification_is_deterministic_and_conservative(text, kind):
    from app.services.memory_scope import classify
    assert classify(text, COLUMNS) == kind
    assert classify(text, COLUMNS) == classify(text, COLUMNS)
