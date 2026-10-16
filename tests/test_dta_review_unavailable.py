"""A data-transfer logic review that cannot be built must not report APPROVED.

The reviewer and the coder were only built when a Groq key was present,
whatever the provider. OpenRouter is the default provider, so an install with
only an OpenRouter key had neither; and when the reviewer was None the review
returned ``APPROVED``. An absent gate reporting success is the worst state a
gate has, because nothing downstream can tell it from a pass.

Pinned here:

* the reviewer and the coder are built from the ACTIVE provider's credentials,
  and a Groq key is required on the Groq path only;
* with no reviewer the review blocks, names the setting to fix, and never says
  APPROVED;
* that block is terminal -- no repair attempts are spent regenerating code that
  nobody judged, because a coder cannot supply a key;
* the transfer is refused at pre-flight, before any work;
* running unreviewed is available only through the existing explicit opt-out.

No network: clients are constructed with dummy keys and never invoked.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

import app.agents.data_transfer_agent.daft_coder as dc
import app.agents.data_transfer_agent.daft_validator as dv
import app.agents.data_transfer_agent.data_transfer_agent as dta
from app.core import model_config as mc

_KEY_VARS = (
    "OPENROUTER_API_KEY", "INFERENCE_OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE",
    "GROQ_API_KEY", "GROQ_API_KEY_PLANNING_AGENT", "GROQ_API_KEY_CODING_AGENT",
    "DTA_CODER_MODEL", "DTA_VALIDATOR_MODEL",
)


@pytest.fixture
def no_credentials(monkeypatch):
    """No provider configured and no key of any kind: the suite's own state."""
    for name in list(os.environ):
        if name.startswith(("INFERENCE_", "AVALOKA_")) or name in _KEY_VARS:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dv, "validator_llm", None)
    return monkeypatch


def _state():
    return {
        "user_prompt": "sum sales into 'total'",
        "generated_code": "def transform_data(df):\n    return df",
        "execution_stdout": "ran",
        "execution_output_preview": [],
        "syntax_error": False,
        "static_semantic_error": False,
    }


# ---------------------------------------------------------------------------
# (a) the reviewer and the coder are built from the active provider
# ---------------------------------------------------------------------------

#: (env, client class or None, model). Model env vars are cleared by the fixture.
_CONFIGS = {
    "openrouter-key-only": ({"OPENROUTER_API_KEY": "k"}, "ChatOpenAI"),
    "groq-key-only": ({"GROQ_API_KEY": "k"}, "ChatGroq"),
    "groq-coding-key-only": ({"GROQ_API_KEY_CODING_AGENT": "k"}, "ChatGroq"),
    "both-keys": ({"OPENROUTER_API_KEY": "k", "GROQ_API_KEY": "k"}, "ChatOpenAI"),
    "no-keys": ({}, None),
    "explicit-groq-with-only-an-openrouter-key":
        ({"INFERENCE_PROVIDER": "groq", "OPENROUTER_API_KEY": "k"}, None),
    "explicit-openrouter-with-only-a-groq-key":
        ({"INFERENCE_PROVIDER": "openrouter", "GROQ_API_KEY": "k"}, None),
    "explicit-openrouter-with-its-key":
        ({"INFERENCE_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "k"}, "ChatOpenAI"),
}


@pytest.mark.parametrize("builder,table_key", [
    (dv._build_validator_llm, "dta_validator"),
    (dc._build_coder_llm, "dta_coder"),
], ids=["reviewer", "coder"])
@pytest.mark.parametrize("config", sorted(_CONFIGS))
def test_the_dta_model_is_built_from_the_active_providers_credentials(
    config, builder, table_key, no_credentials
):
    env, expected_class = _CONFIGS[config]
    for name, value in env.items():
        no_credentials.setenv(name, value)

    llm = builder()

    if expected_class is None:
        assert llm is None
        return
    assert llm is not None, f"{config}: not built, although the active provider has a key"
    assert type(llm).__name__ == expected_class
    assert llm.model_name == mc.DEFAULT_MODELS[table_key]


def test_with_only_an_openrouter_key_the_reviewer_is_on_the_large_model(no_credentials):
    """The default provider with its one key: the normal install."""
    from app.core import inference

    no_credentials.setenv("OPENROUTER_API_KEY", "k")
    llm = dv._build_validator_llm()

    assert llm is not None
    assert llm.model_name == inference._PROVIDER_MODEL_DEFAULTS[inference.OPENROUTER]["large"]


# ---------------------------------------------------------------------------
# (b) no reviewer: block, say why, never APPROVED
# ---------------------------------------------------------------------------

def test_no_reviewer_blocks_and_names_the_setting(no_credentials):
    out = dv.logical_semantic_validator_node(_state())

    assert out["logical_semantic_error"] is True
    assert out["code_validation_feedback"] != "APPROVED"
    assert out.get("code_validated") is not True
    assert out["logic_review_unavailable"] is True
    assert "could not run" in out["code_validation_feedback"]
    # No provider configured resolves to the default, so that is the key named.
    assert "OPENROUTER_API_KEY" in out["code_validation_feedback"]


def test_no_reviewer_on_explicit_groq_names_the_groq_key(no_credentials):
    no_credentials.setenv("INFERENCE_PROVIDER", "groq")
    no_credentials.setenv("OPENROUTER_API_KEY", "k")

    out = dv.logical_semantic_validator_node(_state())

    assert out["logical_semantic_error"] is True
    assert "GROQ_API_KEY_CODING_AGENT" in out["code_validation_feedback"]
    assert "'groq'" in out["code_validation_feedback"]


def test_a_blocked_review_gives_the_repair_loop_nothing_to_send_the_coder(no_credentials):
    """logical_review_feedback is what build_composite_feedback hands the coder."""
    out = dv.logical_semantic_validator_node(_state())

    assert not out.get("logical_review_feedback")


def test_advisory_opt_out_proceeds_unreviewed_and_says_so(no_credentials):
    """The explicit opt-out still proceeds, but it does not claim an approval."""
    no_credentials.setenv("AVALOKA_STRICT_LOGIC_VALIDATION", "0")

    out = dv.logical_semantic_validator_node(_state())

    assert out["logical_semantic_error"] is False
    assert out["code_validation_feedback"] != "APPROVED"
    assert out.get("code_validated") is not True
    assert "could not run" in out["logical_review_feedback"]

    # ...and the note reaches the user the way any advisory finding does.
    final_state = {"coder_definition": dict(out)}
    dta._surface_logic_review_warning(final_state)
    assert any("could not run" in w for w in final_state["warnings"])


# ---------------------------------------------------------------------------
# the block is terminal: no repair attempts are spent on a missing key
# ---------------------------------------------------------------------------

def _run_generation_with_stubbed_stages(monkeypatch):
    calls = {"coder": 0, "execution": 0}

    def _coder(_cdef):
        calls["coder"] += 1
        return {"generated_code": "def transform_data(df):\n    return df.select('a')"}

    def _execute(state):
        calls["execution"] += 1
        state["coder_definition"]["execution_error"] = None
        return state

    monkeypatch.setattr(dta, "sanitize_user_prompt_for_transformations", lambda prompt, llm: prompt)
    monkeypatch.setattr(dta, "daft_pseudocode_node", lambda cdef: {"coder_pseudocode": ""})
    monkeypatch.setattr(dta, "empty_source_guard_node", lambda cdef: {})
    monkeypatch.setattr(dta, "daft_retrieval_node", lambda cdef: {})
    monkeypatch.setattr(dta, "daft_coder_node", _coder)
    monkeypatch.setattr(dta, "syntactic_validator_node", lambda cdef: {"syntax_error": False})
    monkeypatch.setattr(dta, "static_semantic_validator_node",
                        lambda cdef: {"static_semantic_error": False})
    monkeypatch.setattr(dta, "execution_agent_node_local", _execute)

    state = dta.daft_code_generation_pipeline({"coder_definition": {"user_prompt": "keep a"}})
    return state, calls


def test_no_repair_attempts_are_spent_when_the_review_cannot_run(no_credentials):
    state, calls = _run_generation_with_stubbed_stages(no_credentials)

    assert calls == {"coder": 1, "execution": 1}, (
        f"the repair loop regenerated code for a missing reviewer: {calls}"
    )
    assert state["coder_definition"]["logical_semantic_error"] is True
    assert state["coder_definition"]["logic_review_unavailable"] is True


def test_a_real_wrong_verdict_still_goes_through_the_repair_loop(no_credentials):
    """The terminal path is for a missing reviewer only, not for a verdict."""
    class _Wrong:
        def invoke(self, _messages):
            class _R:
                content = '{"is_logically_correct": false, "rationale": "drops a column"}'
            return _R()

    no_credentials.setattr(dv, "validator_llm", _Wrong())
    state, calls = _run_generation_with_stubbed_stages(no_credentials)

    assert calls == {"coder": 5, "execution": 5}
    assert not state["coder_definition"].get("logic_review_unavailable")


# ---------------------------------------------------------------------------
# pre-flight: refused before any work, and a late key is picked up
# ---------------------------------------------------------------------------

_SRC = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p", "table": "src"}
_DST = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p", "table": "dst"}


def _transfer():
    return dta.data_transfer_pipeline(
        user_prompt="copy it", source_type="postgresql", destination_type="postgresql",
        source_credentials=_SRC, destination_credentials=_DST,
    )


@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
@patch("app.agents.data_transfer_agent.data_transfer_agent.probe_db_connection")
def test_a_transfer_is_refused_at_preflight_when_there_is_no_reviewer(
    mock_probe, mock_deduce, no_credentials
):
    with pytest.raises(RuntimeError) as excinfo:
        _transfer()

    message = str(excinfo.value)
    assert "logic-review LLM could not be initialised" in message
    assert "OPENROUTER_API_KEY" in message
    assert "AVALOKA_STRICT_LOGIC_VALIDATION=0" in message
    mock_probe.assert_not_called()
    mock_deduce.assert_not_called()


@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
@patch("app.agents.data_transfer_agent.data_transfer_agent.probe_db_connection")
def test_the_advisory_opt_out_lets_a_transfer_past_preflight_without_a_reviewer(
    mock_probe, mock_deduce, no_credentials
):
    no_credentials.setenv("AVALOKA_STRICT_LOGIC_VALIDATION", "0")
    mock_deduce.side_effect = Exception("stop after pre-flight")

    state, script = _transfer()

    assert script is None
    assert "Could not read source schema" in state["error_message"]


@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
@patch("app.agents.data_transfer_agent.data_transfer_agent.probe_db_connection")
def test_preflight_rebuilds_a_reviewer_whose_key_arrived_after_import(
    mock_probe, mock_deduce, no_credentials
):
    """Only the coder was rebuilt, so a late-loaded key gave real code a None reviewer."""
    no_credentials.setenv("OPENROUTER_API_KEY", "k")
    mock_deduce.side_effect = Exception("stop after pre-flight")

    _transfer()

    assert dv.validator_llm is not None
    assert dv.validator_llm.model_name == mc.DEFAULT_MODELS["dta_validator"]


def test_preflight_names_the_active_providers_setting_when_there_is_no_coder(no_credentials):
    no_credentials.setattr(dc, "coder_llm", None)

    with pytest.raises(RuntimeError) as excinfo:
        _transfer()

    assert "code-generation LLM could not be initialised" in str(excinfo.value)
    assert "OPENROUTER_API_KEY" in str(excinfo.value)


@patch("app.agents.data_transfer_agent.daft_coder.coder_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.daft_validator.validator_llm", new=MagicMock())
@patch("app.agents.data_transfer_agent.data_transfer_agent.daft_code_generation_pipeline")
@patch("app.agents.data_transfer_agent.data_transfer_agent._read_daft_df")
@patch("app.agents.data_transfer_agent.data_transfer_agent.deduce_schema")
@patch("app.agents.data_transfer_agent.data_transfer_agent.probe_db_connection")
def test_a_review_that_could_not_run_is_reported_as_configuration_not_as_a_bad_request(
    mock_probe, mock_deduce, mock_read, mock_codegen
):
    import pandas as pd

    reason = "The logic review could not run because no reviewer model is configured: X."
    mock_deduce.return_value = {"a": "Int64"}
    mock_read.return_value.to_pandas.return_value = pd.DataFrame({"a": [1]})
    mock_codegen.side_effect = lambda s: {
        **s,
        "coder_definition": {
            "generated_code": "def transform_data(df):\n    return df.select('a')",
            "logical_semantic_error": True,
            "logic_review_unavailable": True,
            "code_validation_feedback": reason,
        },
    }

    state, script = _transfer()

    assert script is None
    assert state["error_message"] == reason
    assert "rephrase" not in state["error_message"]
