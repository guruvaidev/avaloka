"""The agent contract: declared writes, contained failures, wiring checks."""

from __future__ import annotations

import pytest

from app.agents.contract import (AGENT_ERRORS_KEY, AgentSpec, Stage,
                                 check_wiring, clear_registry, get,
                                 missing_reads, pipeline_order, register,
                                 registry, run_agent)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test gets a private registry; the real one is restored after."""
    import app.agents.contract as mod
    saved_specs, saved_impls = dict(mod._REGISTRY), dict(mod._IMPLS)
    clear_registry()
    yield
    clear_registry()
    mod._REGISTRY.update(saved_specs)
    mod._IMPLS.update(saved_impls)


def spec(name="a", stage=Stage.EVALUATE, reads=(), writes=("out",), required=False):
    return AgentSpec(name=name, stage=stage, reads=reads, writes=writes, required=required)


# --------------------------------------------------------------------------- #
# Spec validation — catch misdeclaration at definition time
# --------------------------------------------------------------------------- #

def test_an_agent_must_declare_writes():
    with pytest.raises(ValueError, match="no writes"):
        AgentSpec(name="noop", stage=Stage.EVALUATE, writes=())


def test_read_write_on_the_same_key_is_rejected():
    """Read-modify-write on one key makes agent ordering unanalysable."""
    with pytest.raises(ValueError, match="both reads and writes"):
        AgentSpec(name="rmw", stage=Stage.EVALUATE, reads=("x",), writes=("x",))


@pytest.mark.parametrize("bad", ["", "has space", "has-dash!", "sym#bol"])
def test_invalid_names_rejected(bad):
    with pytest.raises(ValueError):
        AgentSpec(name=bad, stage=Stage.EVALUATE, writes=("o",))


def test_duplicate_registration_is_an_error_not_a_silent_overwrite():
    register(spec(name="dup"), lambda s: {"out": 1})
    with pytest.raises(ValueError, match="already registered"):
        register(spec(name="dup"), lambda s: {"out": 2})


def test_unknown_agent_lookup_lists_known_names():
    register(spec(name="known"), lambda s: {"out": 1})
    with pytest.raises(KeyError, match="known"):
        get("missing")


def test_registry_copy_cannot_mutate_the_real_one():
    register(spec(name="a"), lambda s: {"out": 1})
    registry().clear()
    assert "a" in registry()


# --------------------------------------------------------------------------- #
# Declared writes are enforced
# --------------------------------------------------------------------------- #

def test_only_declared_keys_are_written():
    """A typo must not become a mystery field other agents later trust."""
    register(spec(name="leaky", writes=("out",)),
             lambda s: {"out": 1, "typoed_key": 99})
    updates = run_agent("leaky", {})
    assert updates == {"out": 1}
    assert "typoed_key" not in updates


def test_a_returned_non_mapping_is_a_recorded_failure():
    register(spec(name="wrong"), lambda s: ["not", "a", "mapping"])
    updates = run_agent("wrong", {})
    assert AGENT_ERRORS_KEY in updates
    assert "expected a mapping" in updates[AGENT_ERRORS_KEY][0]["error"]


def test_returning_none_is_treated_as_no_updates():
    register(spec(name="quiet"), lambda s: None)
    assert run_agent("quiet", {}) == {}


# --------------------------------------------------------------------------- #
# A crashing agent must never take the graph down
# --------------------------------------------------------------------------- #

def test_exceptions_become_structured_failures():
    def boom(state):
        raise RuntimeError("evaluator exploded")
    register(spec(name="boom"), boom)

    updates = run_agent("boom", {})
    assert "evaluator exploded" in updates[AGENT_ERRORS_KEY][0]["error"]
    assert updates[AGENT_ERRORS_KEY][0]["agent"] == "boom"
    assert "traceback" in updates[AGENT_ERRORS_KEY][0]


def test_a_failing_optional_agent_does_not_stop_the_pipeline():
    def boom(state):
        raise ValueError("nope")
    register(spec(name="opt"), boom)
    register(spec(name="next_one", writes=("other",)), lambda s: {"other": "ran"})

    run_agent("opt", {})
    assert run_agent("next_one", {}) == {"other": "ran"}


# --------------------------------------------------------------------------- #
# Missing inputs: skip when optional, fail when required
# --------------------------------------------------------------------------- #

def test_optional_agent_skips_quietly_when_inputs_are_absent():
    register(spec(name="opt", reads=("df",)), lambda s: {"out": 1})
    assert run_agent("opt", {}) == {}


def test_required_agent_records_a_failure_when_inputs_are_absent():
    register(spec(name="req", reads=("df",), required=True), lambda s: {"out": 1})
    updates = run_agent("req", {})
    assert "missing required state" in updates[AGENT_ERRORS_KEY][0]["error"]


def test_a_none_valued_key_counts_as_missing():
    s = spec(name="x", reads=("df",))
    assert missing_reads(s, {"df": None}) == ("df",)
    assert missing_reads(s, {"df": 123}) == ()


# --------------------------------------------------------------------------- #
# Ordering and wiring
# --------------------------------------------------------------------------- #

def test_pipeline_order_follows_the_lifecycle():
    register(spec(name="present", stage=Stage.PRESENT), lambda s: {"out": 1})
    register(spec(name="understand", stage=Stage.UNDERSTAND, writes=("a",)), lambda s: {"a": 1})
    register(spec(name="evaluate", stage=Stage.EVALUATE, writes=("b",)), lambda s: {"b": 1})
    assert pipeline_order(["present", "evaluate", "understand"]) == \
        ("understand", "evaluate", "present")


def test_wiring_check_catches_reading_a_key_produced_later():
    """The bug that passes on a warm cache and fails on a cold run."""
    register(spec(name="consumer", stage=Stage.EVALUATE,
                  reads=("features",), writes=("score",)), lambda s: {"score": 1})
    register(spec(name="producer", stage=Stage.PREPARE, writes=("features",)),
             lambda s: {"features": 1})

    problems = check_wiring(["consumer", "producer"])
    assert len(problems) == 1 and "produced later by producer" in problems[0]

    assert check_wiring(["producer", "consumer"]) == ()


def test_wiring_check_is_quiet_on_externally_supplied_keys():
    """A key nobody produces is an input to the graph, not an error."""
    register(spec(name="c", reads=("uploaded_df",), writes=("o",)), lambda s: {"o": 1})
    assert check_wiring(["c"]) == ()
