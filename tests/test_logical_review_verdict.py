"""The logical reviewer records its verdict, and nothing it returns changes.

Two halves, driven by the same scripted scenarios (one per exit of
``validator.logical_semantic_validator_node``):

* ``test_pre_existing_keys_are_unchanged`` / ``..._llm_error_still_propagates``
  pin what the node returned BEFORE the verdict fields existed: the literal
  value of every pre-existing key, and how many times the reviewer is called.
  These import nothing new, so the same tests run and pass on the commit
  before this change. They are a characterisation of existing behaviour, and
  they are what fails if the instrumentation ever alters a result.
* everything else asserts the new fields, the closed vocabulary, the counters
  and the log line. New names are imported inside the tests so that the first
  half still collects on the commit before this change.
"""
import json
import logging
import threading
from types import SimpleNamespace

import pytest

import app.agents.validator as validator_mod

# A rationale carrying exactly what must never reach a verdict field: a column
# name, a line of code, a magic constant.
COLUMN = "patient_ssn_hash"
CODE = "df['patient_ssn_hash'].sum() / 0.4217"
RATIONALE = f"Column {COLUMN} is misused: `{CODE}` invents the constant 0.4217."
UNPARSABLE = "Reviewer response unparsable; skipping logical validation."

VERDICT_KEYS = (
    "logical_review_decided_by",
    "logical_review_judged_correct",
    "logical_review_judged_fabricated",
    "logical_review_recheck_ran",
    "logical_review_recheck_confirmed",
    "logical_review_overruled",
)

# The closed vocabulary, written out so a new member cannot appear unnoticed.
VOCABULARY = {
    "approved",
    "approved_on_recheck",
    "soft_flag_default_mode",
    "soft_flag_fabrication_unreadable",
    "blocked_strict_mode",
    "blocked_fabrication_confirmed",
    "blocked_fabrication_recheck_failed",
    "no_verdict_unparseable",
    "no_verdict_malformed_value",
    "no_verdict_llm_error",
    "skipped_no_reviewer",
    "skipped_upstream_error",
}

NO_REVIEWER = object()


def _reply(correct=None, fabricated=None, rationale=RATIONALE):
    body = {}
    if correct is not None:
        body["is_logically_correct"] = correct
    if fabricated is not None:
        body["is_fabricated"] = fabricated
    if rationale is not None:
        body["rationale"] = rationale
    return json.dumps(body)


class ScriptedReviewer:
    """Returns the scripted replies in order; an exception instance is raised."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return SimpleNamespace(content=reply)


def _verdict(decided_by, correct=None, fabricated=None, ran=False, confirmed=None, overruled=False):
    return {
        "logical_review_decided_by": decided_by,
        "logical_review_judged_correct": correct,
        "logical_review_judged_fabricated": fabricated,
        "logical_review_recheck_ran": ran,
        "logical_review_recheck_confirmed": confirmed,
        "logical_review_overruled": overruled,
    }


APPROVED = {"logical_semantic_error": False, "code_validation_feedback": "APPROVED"}
SOFT = {**APPROVED, "logical_review_feedback": RATIONALE}
BLOCKED = {"logical_semantic_error": True, "code_validation_feedback": RATIONALE}
LET_THROUGH_UNPARSED = {"logical_semantic_error": False, "code_validation_feedback": UNPARSABLE}

# name -> (strict env, state overrides, scripted replies, pre-existing keys the
#          node returned before this change, reviewer calls, verdict fields)
SCENARIOS = {
    # --- no review was due -------------------------------------------------
    "skipped_syntax_error": (
        None, {"syntax_error": True}, [], {}, 0, _verdict("skipped_upstream_error")),
    "skipped_static_semantic_error": (
        None, {"static_semantic_error": True}, [], {}, 0, _verdict("skipped_upstream_error")),
    # --- no verdict obtained, code let through -----------------------------
    "skipped_no_reviewer": (
        None, {}, NO_REVIEWER, APPROVED, 0, _verdict("skipped_no_reviewer")),
    "unparseable_prose": (
        None, {}, ["I think this looks fine overall."], LET_THROUGH_UNPARSED, 1,
        _verdict("no_verdict_unparseable")),
    "unparseable_empty_reply": (
        None, {}, [""], LET_THROUGH_UNPARSED, 1, _verdict("no_verdict_unparseable")),
    "unparseable_json_but_not_an_object": (
        None, {}, ["[1, 2]"], LET_THROUGH_UNPARSED, 1, _verdict("no_verdict_unparseable")),
    "unparseable_in_strict_mode": (
        "1", {}, ["not json"], LET_THROUGH_UNPARSED, 1, _verdict("no_verdict_unparseable")),
    # --- reviewer approved --------------------------------------------------
    "approved": (
        None, {}, [_reply(True, False)], APPROVED, 1,
        _verdict("approved", correct=True, fabricated=False)),
    "approved_in_strict_mode": (
        "1", {}, [_reply(True, False)], APPROVED, 1,
        _verdict("approved", correct=True, fabricated=False)),
    # --- reviewer said NOT correct; default mode reports APPROVED ----------
    "soft_flag_default_mode": (
        None, {}, [_reply(False, False)], SOFT, 1,
        _verdict("soft_flag_default_mode", correct=False, fabricated=False, overruled=True)),
    "soft_flag_strict_explicitly_off": (
        "0", {}, [_reply(False, False)], SOFT, 1,
        _verdict("soft_flag_default_mode", correct=False, fabricated=False, overruled=True)),
    "soft_flag_without_rationale": (
        None, {}, [_reply(False, False, rationale=None)],
        {**APPROVED, "logical_review_feedback": "No rationale provided."}, 1,
        _verdict("soft_flag_default_mode", correct=False, fabricated=False, overruled=True)),
    # An object with no verdict key is read as "not correct".
    "soft_flag_empty_object": (
        None, {}, ["{}"], {**APPROVED, "logical_review_feedback": "No rationale provided."}, 1,
        _verdict("soft_flag_default_mode", correct=False, fabricated=False, overruled=True)),
    "soft_flag_fabrication_not_confirmed": (
        None, {}, [_reply(False, True), _reply(False, False, rationale="second opinion")], SOFT, 2,
        _verdict("soft_flag_default_mode", correct=False, fabricated=True,
                 ran=True, confirmed=False, overruled=True)),
    "soft_flag_recheck_unparseable": (
        None, {}, [_reply(False, True), "no json here"], SOFT, 2,
        _verdict("soft_flag_default_mode", correct=False, fabricated=True,
                 ran=True, confirmed=False, overruled=True)),
    # --- recheck approved ---------------------------------------------------
    "approved_on_recheck": (
        None, {}, [_reply(False, True), _reply(True, False, rationale="fine")], APPROVED, 2,
        _verdict("approved_on_recheck", correct=False, fabricated=True, ran=True, confirmed=False)),
    # --- blocked ------------------------------------------------------------
    "blocked_strict_mode": (
        "1", {}, [_reply(False, False)], BLOCKED, 1,
        _verdict("blocked_strict_mode", correct=False, fabricated=False)),
    "blocked_strict_mode_fabricated_no_recheck": (
        "1", {}, [_reply(False, True)], BLOCKED, 1,
        _verdict("blocked_strict_mode", correct=False, fabricated=True)),
    "blocked_fabrication_confirmed": (
        None, {}, [_reply(False, True), _reply(False, True, rationale="still invented")], BLOCKED, 2,
        _verdict("blocked_fabrication_confirmed", correct=False, fabricated=True,
                 ran=True, confirmed=True)),
    "blocked_fabrication_recheck_raised": (
        None, {}, [_reply(False, True), RuntimeError("recheck transport down")], BLOCKED, 2,
        _verdict("blocked_fabrication_recheck_failed", correct=False, fabricated=True, ran=True)),
    "blocked_fabrication_recheck_not_an_object": (
        None, {}, [_reply(False, True), "[1]"], BLOCKED, 2,
        _verdict("blocked_fabrication_recheck_failed", correct=False, fabricated=True, ran=True)),
}


def _state(**overrides):
    state = {
        "generated_code": f"def main(df):\n    return {CODE}\n",
        "user_prompt": "Total the hashed identifiers",
        "plan": "sum one column",
        "schema": {COLUMN: "float64"},
        "syntax_error": False,
        "static_semantic_error": False,
        "execution_stdout": "",
        "execution_stderr": "",
        "execution_error": None,
        "execution_output_preview": [{COLUMN: 1.0}],
    }
    state.update(overrides)
    return state


def _run(monkeypatch, name):
    strict, overrides, replies, _pre, _calls, _fields = SCENARIOS[name]
    if strict is None:
        monkeypatch.delenv(validator_mod.STRICT_LOGIC_ENV, raising=False)
    else:
        monkeypatch.setenv(validator_mod.STRICT_LOGIC_ENV, strict)
    reviewer = None if replies is NO_REVIEWER else ScriptedReviewer(replies)
    monkeypatch.setattr(validator_mod, "validator_llm", reviewer)
    out = validator_mod.logical_semantic_validator_node(_state(**overrides))
    return out, reviewer


def _is_new_key(key):
    return key.startswith("logical_review_") and key != "logical_review_feedback"


# ---------------------------------------------------------------------------
# Behaviour unchanged. Runs, and passes, on the commit before this change.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_pre_existing_keys_are_unchanged(monkeypatch, name):
    _strict, _overrides, _replies, pre_existing, calls, _fields = SCENARIOS[name]
    out, reviewer = _run(monkeypatch, name)

    assert {k: v for k, v in out.items() if not _is_new_key(k)} == pre_existing
    assert (reviewer.calls if reviewer is not None else 0) == calls


@pytest.mark.parametrize("strict", [None, "1"])
def test_llm_error_still_propagates(monkeypatch, strict):
    """A failing reviewer call was never caught by the node, and still is not."""
    if strict is None:
        monkeypatch.delenv(validator_mod.STRICT_LOGIC_ENV, raising=False)
    else:
        monkeypatch.setenv(validator_mod.STRICT_LOGIC_ENV, strict)
    boom = RuntimeError("reviewer transport down")
    reviewer = ScriptedReviewer([boom])
    monkeypatch.setattr(validator_mod, "validator_llm", reviewer)

    with pytest.raises(RuntimeError) as excinfo:
        validator_mod.logical_semantic_validator_node(_state())
    assert excinfo.value is boom
    assert reviewer.calls == 1


def test_headline_pre_existing_half(monkeypatch):
    """Reviewer says not correct, not fabricated, default mode: still APPROVED."""
    out, _ = _run(monkeypatch, "soft_flag_default_mode")
    assert out["code_validation_feedback"] == "APPROVED"
    assert out["logical_semantic_error"] is False
    assert out["logical_review_feedback"] == RATIONALE


# ---------------------------------------------------------------------------
# The new fields. Fail on the commit before this change (the keys are absent).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_verdict_fields_per_path(monkeypatch, name):
    expected = SCENARIOS[name][5]
    out, _ = _run(monkeypatch, name)
    assert {k: out[k] for k in VERDICT_KEYS} == expected
    assert sorted(k for k in out if _is_new_key(k)) == sorted(VERDICT_KEYS)


def test_headline_verdict_is_recorded_as_overruled(monkeypatch):
    out, _ = _run(monkeypatch, "soft_flag_default_mode")
    # unchanged
    assert out["code_validation_feedback"] == "APPROVED"
    assert out["logical_semantic_error"] is False
    # recorded
    assert out["logical_review_judged_correct"] is False
    assert out["logical_review_judged_fabricated"] is False
    assert out["logical_review_overruled"] is True
    assert out["logical_review_decided_by"] == "soft_flag_default_mode"


def test_vocabulary_is_closed_and_every_member_is_reachable(monkeypatch):
    from app.agents.logical_review_verdict import (
        LogicalReviewDecision,
        record_logical_verdict,
    )

    assert {d.value for d in LogicalReviewDecision} == VOCABULARY
    seen = {SCENARIOS[name][5]["logical_review_decided_by"] for name in SCENARIOS}
    # llm error: test below. The two unreadable-value members are reached in
    # test_logical_review_verdict_types.py.
    assert seen | {"no_verdict_llm_error", "no_verdict_malformed_value",
                   "soft_flag_fabrication_unreadable"} == VOCABULARY

    for value in ("blocked", "APPROVED", RATIONALE, "", None, True):
        with pytest.raises(ValueError):
            record_logical_verdict(value)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_no_free_text_in_any_verdict_field(monkeypatch, caplog, name):
    """Booleans, None, and one closed-vocabulary string. Nothing from the model."""
    caplog.set_level(logging.INFO, logger="app.agents.logical_review_verdict")
    out, _ = _run(monkeypatch, name)

    for key in VERDICT_KEYS:
        value = out[key]
        if key == "logical_review_decided_by":
            assert type(value) is str and value in VOCABULARY  # plain str, not an Enum
        else:
            assert value is None or type(value) is bool, (key, value)
    rendered = json.dumps({k: out[k] for k in VERDICT_KEYS})  # checkpoint-serializable
    lines = [r.getMessage() for r in caplog.records
             if r.name == "app.agents.logical_review_verdict"]
    assert len(lines) == 1 and lines[0].startswith("logical_review_verdict decided_by=")
    for text in (rendered, lines[0]):
        for leak in (COLUMN, CODE, "0.4217", "invents", "second opinion", "main(df)"):
            assert leak not in text


def test_caller_cannot_smuggle_text_through_a_flag():
    from app.agents.logical_review_verdict import record_logical_verdict

    fields = record_logical_verdict(
        "soft_flag_default_mode",
        judged_correct="", judged_fabricated=RATIONALE,
        recheck_ran=CODE, recheck_confirmed=COLUMN,
    )
    assert fields["logical_review_judged_correct"] is False
    assert fields["logical_review_judged_fabricated"] is True
    assert fields["logical_review_recheck_ran"] is True
    assert fields["logical_review_recheck_confirmed"] is True


def test_llm_error_is_counted_and_logged_before_propagating(monkeypatch, caplog):
    from app.agents.logical_review_verdict import (
        logical_review_counts,
        reset_logical_review_counts,
    )

    reset_logical_review_counts()
    caplog.set_level(logging.INFO, logger="app.agents.logical_review_verdict")
    monkeypatch.delenv(validator_mod.STRICT_LOGIC_ENV, raising=False)
    monkeypatch.setattr(validator_mod, "validator_llm",
                        ScriptedReviewer([RuntimeError(f"timeout near {COLUMN}")]))

    with pytest.raises(RuntimeError):
        validator_mod.logical_semantic_validator_node(_state())

    counts = logical_review_counts()
    assert counts["by_decision"]["no_verdict_llm_error"] == 1
    assert counts["reviews"] == 1 and counts["no_verdict"] == 1
    lines = [r.getMessage() for r in caplog.records
             if r.name == "app.agents.logical_review_verdict"]
    assert len(lines) == 1
    assert "decided_by=no_verdict_llm_error" in lines[0]
    assert "judged_correct=None" in lines[0] and "overruled=False" in lines[0]
    assert COLUMN not in lines[0]


# ---------------------------------------------------------------------------
# Counters: the overrule rate needs a denominator.
# ---------------------------------------------------------------------------

MIX = (
    ["approved"] * 5
    + ["soft_flag_default_mode"] * 3
    + ["soft_flag_fabrication_not_confirmed"]
    + ["approved_on_recheck"]
    + ["blocked_fabrication_confirmed"] * 2
    + ["blocked_fabrication_recheck_raised"]
    + ["blocked_strict_mode"]
    + ["unparseable_prose"] * 2
    + ["skipped_no_reviewer"]
    + ["skipped_syntax_error"] * 4
)


def test_counters_and_overrule_rate_over_a_mix(monkeypatch, caplog):
    from app.agents.logical_review_verdict import (
        logical_review_counts,
        reset_logical_review_counts,
    )

    reset_logical_review_counts()
    caplog.set_level(logging.INFO, logger="app.agents.logical_review_verdict")
    for name in MIX:
        with monkeypatch.context() as m:
            _run(m, name)
    for _ in range(2):  # two failed reviewer calls
        with monkeypatch.context() as m:
            m.delenv(validator_mod.STRICT_LOGIC_ENV, raising=False)
            m.setattr(validator_mod, "validator_llm", ScriptedReviewer([RuntimeError("down")]))
            with pytest.raises(RuntimeError):
                validator_mod.logical_semantic_validator_node(_state())

    c = logical_review_counts()
    assert c["calls"] == len(MIX) + 2 == 23
    assert c["not_reviewed"] == 4  # upstream syntax error: no review was due
    assert c["reviews"] == 19

    # The four outcomes are distinct and partition the reviews.
    assert c["approved_by_reviewer"] == 6   # 5 first-pass + 1 on recheck
    assert c["overruled"] == 4              # said "not correct", reported APPROVED
    assert c["no_verdict"] == 5             # 2 unparseable + 2 llm error + 1 no reviewer
    assert c["blocked"] == 4
    assert (c["approved_by_reviewer"] + c["overruled"] + c["no_verdict"] + c["blocked"]
            == c["reviews"])
    assert c["calls"] == c["reviews"] + c["not_reviewed"]

    # A review with no verdict is a review, and neither correct nor incorrect.
    assert c["judged_correct"] == 5
    assert c["judged_not_correct"] == 9
    assert c["judged_correct"] + c["judged_not_correct"] + c["no_verdict"] == c["reviews"]
    assert c["judged_not_correct"] == c["overruled"] + c["blocked"] + 1  # + approved on recheck

    assert c["fabrication_flagged"] == 5
    assert c["fabrication_rechecked"] == 5
    assert c["fabrication_confirmed"] == 2

    # Each no-verdict cause is told apart.
    by = c["by_decision"]
    assert by["no_verdict_unparseable"] == 2
    assert by["no_verdict_llm_error"] == 2
    assert by["skipped_no_reviewer"] == 1
    assert by["skipped_upstream_error"] == 4
    assert set(by) == VOCABULARY and sum(by.values()) == c["calls"]

    assert c["overruled"] / c["reviews"] == pytest.approx(4 / 19)
    assert c["no_verdict"] / c["reviews"] == pytest.approx(5 / 19)

    # Integers only, and the last log line carries the same running totals.
    flat = {k: v for k, v in c.items() if k != "by_decision"}
    assert all(type(v) is int for v in list(flat.values()) + list(by.values()))
    lines = [r.getMessage() for r in caplog.records
             if r.name == "app.agents.logical_review_verdict"]
    assert len(lines) == c["calls"]  # one line per call, every path
    assert ("total_calls=23 total_reviews=19 total_approved_by_reviewer=6 total_overruled=4 "
            "total_no_verdict=5 total_blocked=4 total_not_reviewed=4") in lines[-1]


def test_counters_are_thread_safe():
    from app.agents.logical_review_verdict import (
        logical_review_counts,
        record_logical_verdict,
        reset_logical_review_counts,
    )

    reset_logical_review_counts()

    def work():
        for _ in range(500):
            record_logical_verdict("soft_flag_default_mode", judged_correct=False)
            record_logical_verdict("no_verdict_unparseable")

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    c = logical_review_counts()
    assert c["reviews"] == 8000 and c["overruled"] == 4000 and c["no_verdict"] == 4000
    assert c["judged_not_correct"] == 4000 and c["judged_correct"] == 0


# ---------------------------------------------------------------------------
# State: the keys are declared channels and travel up to the top-level graph.
# ---------------------------------------------------------------------------

def test_verdict_keys_are_declared_state_channels():
    from app.agents.logical_review_verdict import LOGICAL_REVIEW_VERDICT_KEYS
    from app.agents.state import CodingAgentState
    from app.graph.etl_state import ETLState

    assert tuple(LOGICAL_REVIEW_VERDICT_KEYS) == VERDICT_KEYS
    for key in VERDICT_KEYS:
        assert key in CodingAgentState.__annotations__
        assert key in ETLState.__annotations__


def test_verdict_survives_the_coding_graph_state(monkeypatch):
    """LangGraph drops any key that is not a declared channel of the subgraph."""
    from langgraph.graph import END, StateGraph

    from app.agents.state import CodingAgentState

    monkeypatch.delenv(validator_mod.STRICT_LOGIC_ENV, raising=False)
    monkeypatch.setattr(validator_mod, "validator_llm", ScriptedReviewer([_reply(False, False)]))
    g = StateGraph(CodingAgentState)
    g.add_node("validator_logical", validator_mod.logical_semantic_validator_node)
    g.set_entry_point("validator_logical")
    g.add_edge("validator_logical", END)
    final = g.compile().invoke(_state())

    assert {k: final.get(k) for k in VERDICT_KEYS} == SCENARIOS["soft_flag_default_mode"][5]
    assert final["code_validation_feedback"] == "APPROVED"
    assert final["logical_semantic_error"] is False


@pytest.mark.parametrize("failed", [False, True])
def test_coding_subgraph_node_passes_the_verdict_up(monkeypatch, failed):
    import app.api.workflow as wf

    fields = SCENARIOS["blocked_strict_mode" if failed else "soft_flag_default_mode"][5]
    result = {"generated_code": "x = 1", "llm_raw_response": None,
              "logical_semantic_error": failed, **fields}

    class FakeGraph:
        def invoke(self, initial):
            return dict(result)

    monkeypatch.setattr(wf, "build_coding_graph", lambda *a, **k: FakeGraph())
    monkeypatch.setattr(wf, "coding_graph", None, raising=False)
    monkeypatch.setattr(wf, "coder_node_is_mock", False, raising=False)
    monkeypatch.setattr(wf, "_detect_ambiguous_prompt_details", lambda prompt: None)

    updates = wf.coding_subgraph_node({"user_prompt": "clean data", "messages": [], "schema": {}})
    assert {k: updates[k] for k in VERDICT_KEYS} == fields


def test_verdict_is_not_in_the_api_response():
    from app.api.schemas import ChatResponse

    assert not [f for f in ChatResponse.model_fields if f.startswith("logical_review")]
