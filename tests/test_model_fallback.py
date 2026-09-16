"""OpenRouter backup: when to fail over, and to what."""

from __future__ import annotations

import pytest

from app.core.model_config import DEFAULT_MODELS
from app.core.model_fallback import (DEGRADED_SUBSTITUTIONS,
                                     OPENROUTER_EQUIVALENTS, FallbackPlan,
                                     NoBackupConfigured, attach_fallback,
                                     backup_enabled, build_backup_chat_model,
                                     describe_backup, equivalent_for,
                                     BackupFailed, failover_exception_types,
                                     normalise_tools_for_openrouter,
                                     openrouter_key, plan_fallback,
                                     should_failover)


@pytest.fixture
def with_backup(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.delenv("AVALOKA_DISABLE_MODEL_FALLBACK", raising=False)


@pytest.fixture
def without_backup(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


# --------------------------------------------------------------------------- #
# Opt-in: absent config must change nothing
# --------------------------------------------------------------------------- #

def test_backup_is_off_when_no_key_is_set(without_backup):
    assert openrouter_key() is None
    assert backup_enabled() is False


def test_backup_is_on_once_a_key_exists(with_backup):
    assert backup_enabled() is True


def test_backup_can_be_switched_off_while_the_key_stays(with_backup, monkeypatch):
    """Turning it off should not require deleting credentials."""
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    assert backup_enabled() is False


def test_planning_without_a_backup_raises_rather_than_guessing(without_backup):
    with pytest.raises(NoBackupConfigured, match="OPENROUTER_API_KEY"):
        plan_fallback("openai/gpt-oss-120b")


def test_build_returns_none_without_a_key(without_backup):
    """Callers degrade to their existing LLM-disabled path."""
    assert build_backup_chat_model("openai/gpt-oss-120b") is None


# --------------------------------------------------------------------------- #
# Which failures deserve a second provider
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("status", [404, 429, 500, 502, 503, 504, 408])
def test_transient_and_missing_model_failures_fail_over(status):
    assert should_failover(status) is True


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_credential_and_malformed_request_failures_do_not(status):
    """OpenRouter would reject these identically; retrying only costs money."""
    assert should_failover(status) is False


def test_the_404_that_actually_happened_fails_over():
    """llama-3.3-70b-versatile returned 404 after Groq removed it."""
    assert should_failover(404) is True


@pytest.mark.parametrize("status", [200, 201, 301, 418])
def test_unknown_statuses_do_not_fail_over(status):
    """Defaulting to retry on an unrecognised code spends money for nothing."""
    assert should_failover(status) is False


@pytest.mark.parametrize("exc", [TimeoutError("t"), ConnectionError("c"), OSError("socket down")])
def test_transport_failures_fail_over(exc):
    assert should_failover(error=exc) is True


def test_a_value_error_is_not_a_transport_failure():
    assert should_failover(error=ValueError("bad arg")) is False


def test_no_status_and_no_error_means_no_failover():
    assert should_failover() is False


# --------------------------------------------------------------------------- #
# Model mapping — ids are not portable between providers
# --------------------------------------------------------------------------- #

def test_gpt_oss_maps_to_itself():
    """Verified live: the same id exists on both providers."""
    assert equivalent_for("openai/gpt-oss-120b") == "openai/gpt-oss-120b"
    assert equivalent_for("openai/gpt-oss-20b") == "openai/gpt-oss-20b"


def test_groq_compound_maps_to_a_substitute_not_itself():
    """OpenRouter: 'groq/compound is not a valid model ID' (verified live)."""
    backup = equivalent_for("groq/compound")
    assert backup is not None and backup != "groq/compound"


def test_every_configured_default_has_a_backup_mapping():
    """A model with no mapping cannot fail over, which defeats the point."""
    unmapped = {a: m for a, m in DEFAULT_MODELS.items()
                if equivalent_for(m) is None}
    assert not unmapped, f"no OpenRouter equivalent recorded for: {unmapped}"


def test_an_unmapped_model_raises_rather_than_silently_picking_one(with_backup):
    with pytest.raises(NoBackupConfigured, match="no OpenRouter equivalent"):
        plan_fallback("some/unknown-model-v9")


# --------------------------------------------------------------------------- #
# Capability downgrades must be visible
# --------------------------------------------------------------------------- #

def test_the_validator_substitution_is_not_marked_degraded(with_backup):
    """Compound -> qwen substitutes the model, not a capability the gate uses.

    Compound's agentic tool runtime IS lost, but the Validator never reaches
    for it: it prompts for a verdict and parses the reply with json.loads, with
    no bind_tools anywhere in app/agents/validator.py. Marking this degraded
    would fire a CAPABILITY DOWNGRADE warning on every validator failover for a
    capability the agent does not exercise, which trains operators to ignore
    the one warning that is supposed to be rare.
    """
    plan = plan_fallback("groq/compound")
    assert plan.backup_model == "qwen/qwen3.8-27b"
    assert plan.degraded is False


def test_an_identity_mapping_is_not_degraded(with_backup):
    assert plan_fallback("openai/gpt-oss-120b").degraded is False


def test_no_substitution_is_currently_declared_degraded():
    """Guards the claim the test above rests on.

    If a future substitution genuinely loses a capability its agent uses, it
    belongs in DEGRADED_SUBSTITUTIONS -- and this test failing is the prompt to
    say so deliberately rather than discovering it in an outage.
    """
    assert DEGRADED_SUBSTITUTIONS == set()


def test_the_degrade_is_logged_loudly(with_backup, caplog, monkeypatch):
    """The warning path is unused today but must still work when it is needed.

    DEGRADED_SUBSTITUTIONS is empty, so this drives plan_fallback through a
    patched entry instead of a real one. Deleting the test with the last real
    entry would leave the branch in plan_fallback uncovered until the day
    somebody adds a downgrade and needs the warning to fire.
    """
    import logging
    from app.core import model_fallback
    monkeypatch.setattr(model_fallback, "DEGRADED_SUBSTITUTIONS", {"groq/compound"})
    with caplog.at_level(logging.WARNING):
        plan_fallback("groq/compound", reason="404 from Groq")
    assert any("CAPABILITY DOWNGRADE" in r.message for r in caplog.records)


def test_a_clean_fallback_does_not_warn(with_backup, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        plan_fallback("openai/gpt-oss-120b")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_the_validator_failover_is_silent(with_backup, caplog):
    """The regression this replaces: a warning on every validator failover."""
    import logging
    with caplog.at_level(logging.WARNING):
        plan_fallback(DEFAULT_MODELS["validator"], reason="429 from Groq")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --------------------------------------------------------------------------- #
# Plan payload and diagnostics
# --------------------------------------------------------------------------- #

def test_plan_carries_the_reason(with_backup):
    plan = plan_fallback("openai/gpt-oss-20b", reason="429 rate limited")
    assert plan.reason == "429 rate limited"
    assert isinstance(plan, FallbackPlan)


def test_plan_serialises(with_backup):
    import json
    json.dumps(plan_fallback("openai/gpt-oss-120b").as_dict())


def test_describe_backup_reports_state(with_backup):
    described = describe_backup()
    assert described["enabled"] is True
    assert described["configured"] is True
    assert "groq/compound" in described["equivalents"]


def test_describe_backup_is_json_serialisable(without_backup):
    import json
    json.dumps(describe_backup())


# --------------------------------------------------------------------------- #
# Local fallback tier: profile-aware sizing, stated in the logs
# --------------------------------------------------------------------------- #

from app.core.model_fallback import (LOCAL_MODEL_CLUSTER, LOCAL_MODEL_LAPTOP,
                                     build_local_chat_model, deployment_profile,
                                     fallback_chain, local_fallback_model)


@pytest.fixture
def laptop(monkeypatch):
    monkeypatch.delenv("AVALOKA_DEPLOYMENT_PROFILE", raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)


@pytest.fixture
def cluster(monkeypatch):
    monkeypatch.delenv("AVALOKA_DEPLOYMENT_PROFILE", raising=False)
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")


def test_no_k8s_env_means_laptop(laptop):
    assert deployment_profile() == "laptop"


def test_kubelet_injected_env_means_cluster(cluster):
    """KUBERNETES_SERVICE_HOST is present in every pod; that is the signal."""
    assert deployment_profile() == "cluster"


def test_explicit_profile_beats_detection(cluster, monkeypatch):
    monkeypatch.setenv("AVALOKA_DEPLOYMENT_PROFILE", "laptop")
    assert deployment_profile() == "laptop"


def test_laptop_gets_the_small_model(laptop):
    """A laptop asked to serve 27B+ turns a fallback into a hang."""
    assert local_fallback_model() == LOCAL_MODEL_LAPTOP
    assert "4b" in LOCAL_MODEL_LAPTOP.lower()


def test_cluster_gets_the_large_model(cluster):
    assert local_fallback_model() == LOCAL_MODEL_CLUSTER
    assert any(s in LOCAL_MODEL_CLUSTER.lower() for s in ("27b", "31b", "26b"))


def test_the_size_decision_is_stated_in_the_logs(laptop, caplog):
    """The user requirement verbatim: indicated in the logs."""
    import logging
    with caplog.at_level(logging.INFO):
        local_fallback_model()
    joined = " ".join(r.message for r in caplog.records)
    assert "profile=laptop" in joined and LOCAL_MODEL_LAPTOP in joined


def test_cluster_log_names_model_and_alternate(cluster, caplog):
    import logging
    with caplog.at_level(logging.INFO):
        local_fallback_model()
    joined = " ".join(r.message for r in caplog.records)
    assert "profile=cluster" in joined and LOCAL_MODEL_CLUSTER in joined


def test_chain_orders_openrouter_before_local(with_backup, laptop):
    chain = fallback_chain("openai/gpt-oss-120b")
    assert [tier for tier, _, _ in chain] == ["openrouter", "local"]


def test_chain_without_openrouter_still_has_local(without_backup, laptop):
    """No key, no internet — the local tier must still exist."""
    chain = fallback_chain("openai/gpt-oss-120b")
    assert [tier for tier, _, _ in chain] == ["local"]


def test_local_tier_is_always_marked_degraded(with_backup, laptop):
    tier, _, degraded = fallback_chain("openai/gpt-oss-120b")[-1]
    assert tier == "local" and degraded is True


def test_local_chain_respects_profile(with_backup, cluster):
    _, model, _ = fallback_chain("openai/gpt-oss-120b")[-1]
    assert model == LOCAL_MODEL_CLUSTER


def test_env_overrides_the_laptop_model(laptop, monkeypatch):
    """Whatever Ollama actually serves wins over our default id."""
    import importlib

    import app.core.model_fallback as mod
    monkeypatch.setenv("AVALOKA_LOCAL_MODEL_LAPTOP", "gemma4:tiny-local")
    importlib.reload(mod)
    try:
        assert mod.local_fallback_model() == "gemma4:tiny-local"
    finally:
        monkeypatch.delenv("AVALOKA_LOCAL_MODEL_LAPTOP")
        importlib.reload(mod)


# --------------------------------------------------------------------------- #
# Attaching the backup: what a live agent actually gets
# --------------------------------------------------------------------------- #

class _Boom:
    """A primary that always fails with the given exception."""

    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def invoke(self, _input, _config=None, **_kwargs):
        self.calls += 1
        raise self.exc

    def with_fallbacks(self, fallbacks, *, exceptions_to_handle=(Exception,),
                       exception_key=None):
        """Stand in for LangChain's real wrapper: try primary, then fallback."""
        primary = self

        class _Wrapped:
            def invoke(self, value, config=None, **kwargs):
                try:
                    return primary.invoke(value, config, **kwargs)
                except exceptions_to_handle:
                    return fallbacks[0].invoke(value, config, **kwargs)

        return _Wrapped()


def _groq_error(cls_name, status):
    """Build a real Groq SDK exception — that is what LangChain re-raises."""
    import groq
    import httpx
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return getattr(groq, cls_name)("boom", response=response, body=None)


def test_no_key_means_the_primary_is_returned_untouched(without_backup):
    """The opt-in promise: unconfigured must be a no-op, not a wrapper."""
    primary = object()
    assert attach_fallback(primary, "openai/gpt-oss-120b") is primary


def test_a_none_primary_stays_none(with_backup):
    """Agents already treat a None llm as 'remote generation disabled'."""
    assert attach_fallback(None, "openai/gpt-oss-120b") is None


def test_an_unmapped_model_gets_no_fallback_rather_than_a_guess(with_backup):
    primary = object()
    assert attach_fallback(primary, "some/unknown-model-v9") is primary


def test_a_configured_backup_wraps_the_primary(with_backup):
    from langchain_groq import ChatGroq
    primary = ChatGroq(model="openai/gpt-oss-120b", api_key="gsk-test")
    wrapped = attach_fallback(primary, "openai/gpt-oss-120b", agent="planner")
    assert wrapped is not primary
    assert hasattr(wrapped, "invoke")


# --- which exceptions reach the backup ------------------------------------- #

def test_rate_limit_is_a_failover_exception():
    """The 429 that sent users the 'temporary problem' message."""
    import groq
    assert groq.RateLimitError in failover_exception_types()


@pytest.mark.parametrize("name", ["NotFoundError", "InternalServerError",
                                  "APIConnectionError"])
def test_transient_sdk_failures_are_failover_exceptions(name):
    import groq
    assert getattr(groq, name) in failover_exception_types()


@pytest.mark.parametrize("name", ["BadRequestError", "AuthenticationError",
                                  "PermissionDeniedError",
                                  "UnprocessableEntityError"])
def test_client_errors_are_not_failover_exceptions(name):
    """These would fail identically on OpenRouter."""
    import groq
    assert getattr(groq, name) not in failover_exception_types()


def test_timeouts_ride_in_on_the_connection_error_base():
    """408 has no dedicated entry because APITimeoutError subclasses it."""
    import groq
    assert issubclass(groq.APITimeoutError, groq.APIConnectionError)


@pytest.mark.parametrize("name,status", [("RateLimitError", 429),
                                         ("NotFoundError", 404),
                                         ("InternalServerError", 500)])
def test_should_failover_reads_the_status_off_an_sdk_exception(name, status):
    """should_failover(error=exc) must not answer False for a 429."""
    assert should_failover(error=_groq_error(name, status)) is True


@pytest.mark.parametrize("name,status", [("BadRequestError", 400),
                                         ("AuthenticationError", 401)])
def test_sdk_client_errors_still_do_not_fail_over(name, status):
    assert should_failover(error=_groq_error(name, status)) is False


# --- the Groq-shaped request the backup is handed -------------------------- #

PLANNER_SHAPED_TOOL = {
    "type": "function",
    "description": "Generate Python code based on user input or data source.",
    "function": {"name": "generate_code",
                 "parameters": {"type": "object", "properties": {}}},
}


def test_tool_descriptions_move_inside_the_function_object():
    """OpenRouter validates the OpenAI schema; the Planner's shape differs."""
    (fixed,) = normalise_tools_for_openrouter([PLANNER_SHAPED_TOOL])
    assert "description" not in fixed
    assert fixed["function"]["description"] == PLANNER_SHAPED_TOOL["description"]
    assert fixed["function"]["name"] == "generate_code"


def test_normalising_leaves_the_callers_tools_alone():
    """The Groq request must stay byte-identical — it is the same list object."""
    tools = [dict(PLANNER_SHAPED_TOOL)]
    normalise_tools_for_openrouter(tools)
    assert tools[0]["description"] == PLANNER_SHAPED_TOOL["description"]
    assert "description" not in tools[0]["function"]


def test_already_correct_tools_pass_through_unchanged():
    correct = {"type": "function",
               "function": {"name": "x", "description": "d", "parameters": {}}}
    assert normalise_tools_for_openrouter([correct]) == [correct]


@pytest.mark.parametrize("tools", [None, "not-a-list", []])
def test_normalising_tolerates_non_tool_payloads(tools):
    assert normalise_tools_for_openrouter(tools) == tools


def test_a_429_is_answered_by_the_backup(with_backup, monkeypatch):
    """End to end: the exact failure from the Planner log reaches OpenRouter."""
    seen = {}

    class _Backup:
        def invoke(self, _value, _config=None, **kwargs):
            seen.update(kwargs)
            return "answered by openrouter"

    monkeypatch.setattr("app.core.model_fallback.build_backup_chat_model",
                        lambda model, temperature=0.0: _Backup())
    primary = _Boom(_groq_error("RateLimitError", 429))
    wrapped = attach_fallback(primary, "openai/gpt-oss-120b", agent="planner")

    result = wrapped.invoke([("human", "hi")], tools=[PLANNER_SHAPED_TOOL],
                            tool_choice="required", reasoning_effort="high",
                            reasoning_format="parsed")

    assert result == "answered by openrouter"
    assert primary.calls == 1
    # Groq-shaped kwargs are translated on the way through, not forwarded raw.
    assert "description" not in seen["tools"][0]
    assert "reasoning_format" not in seen
    assert seen["tool_choice"] == "required"
    assert seen["reasoning_effort"] == "high"


def test_a_401_is_not_answered_by_the_backup(with_backup, monkeypatch):
    """A bad key must surface, not burn a second provider call to say the same."""
    monkeypatch.setattr("app.core.model_fallback.build_backup_chat_model",
                        lambda model, temperature=0.0: object())
    primary = _Boom(_groq_error("AuthenticationError", 401))
    wrapped = attach_fallback(primary, "openai/gpt-oss-120b")

    import groq
    with pytest.raises(groq.AuthenticationError):
        wrapped.invoke([("human", "hi")])


# --------------------------------------------------------------------------- #
# Through LangChain's real RunnableWithFallbacks
#
# The tests above use a hand-rolled `with_fallbacks` to keep the unit under
# test small. These use the genuine article, because the behaviour that matters
# most here — which exception survives when BOTH providers fail — lives in
# LangChain's implementation, not in ours.
# --------------------------------------------------------------------------- #

def _groq_primary_raising(exc):
    """A real ChatGroq whose generation step always fails with *exc*."""
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key="gsk-test")
    llm._generate = lambda *a, **k: (_ for _ in ()).throw(exc)
    return llm


class _RecordingBackup:
    def __init__(self, result="answered by openrouter", raises=None):
        self.result = result
        self.raises = raises
        self.seen = None

    def invoke(self, _value, _config=None, **kwargs):
        self.seen = kwargs
        if self.raises is not None:
            raise self.raises
        return self.result


def test_a_real_429_is_answered_by_the_backup(with_backup, monkeypatch):
    backup = _RecordingBackup()
    monkeypatch.setattr("app.core.model_fallback.build_backup_chat_model",
                        lambda model, temperature=0.0: backup)
    wrapped = attach_fallback(_groq_primary_raising(_groq_error("RateLimitError", 429)),
                              "openai/gpt-oss-120b", agent="planner")

    result = wrapped.invoke([("human", "hi")], tools=[PLANNER_SHAPED_TOOL],
                            tool_choice="required", reasoning_effort="high")

    assert result == "answered by openrouter"
    assert backup.seen["tool_choice"] == "required"
    assert "description" not in backup.seen["tools"][0]


def test_reasoning_effort_is_translated_for_a_reasoning_capable_backup(
        with_backup, monkeypatch):
    """qwen3.8-27b reasons, but under OpenRouter's field name, not gpt-oss's.

    Dropping the kwarg would silently discard the caller's intent; forwarding it
    verbatim would be an unknown body field, i.e. a 400 the primary never had.
    """
    backup = _RecordingBackup()
    monkeypatch.setattr("app.core.model_fallback.build_backup_chat_model",
                        lambda model, temperature=0.0: backup)
    wrapped = attach_fallback(_groq_primary_raising(_groq_error("RateLimitError", 429)),
                              "groq/compound", agent="validator")

    assert wrapped.invoke([("human", "hi")], reasoning_effort="high") ==         "answered by openrouter"
    assert backup.seen["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in backup.seen


def test_reasoning_effort_is_still_dropped_for_a_backup_that_cannot_reason(
        with_backup, monkeypatch):
    """compound-mini falls back to gpt-oss-20b, which keeps the gpt-oss spelling.

    The guard being tested is the OTHER branch: a backup in neither
    _REASONING_MODEL_PREFIXES nor _OPENROUTER_REASONING_MODELS must have the
    kwarg dropped rather than translated.
    """
    backup = _RecordingBackup()
    monkeypatch.setattr("app.core.model_fallback.build_backup_chat_model",
                        lambda model, temperature=0.0: backup)
    monkeypatch.setattr("app.core.model_fallback._OPENROUTER_REASONING_MODELS",
                        frozenset())
    wrapped = attach_fallback(_groq_primary_raising(_groq_error("RateLimitError", 429)),
                              "groq/compound", agent="validator")

    assert wrapped.invoke([("human", "hi")], reasoning_effort="high") ==         "answered by openrouter"
    assert "reasoning" not in backup.seen
    assert "reasoning_effort" not in backup.seen


def test_a_real_401_never_reaches_the_backup(with_backup, monkeypatch):
    backup = _RecordingBackup()
    monkeypatch.setattr("app.core.model_fallback.build_backup_chat_model",
                        lambda model, temperature=0.0: backup)
    wrapped = attach_fallback(_groq_primary_raising(_groq_error("AuthenticationError", 401)),
                              "openai/gpt-oss-120b")

    import groq
    with pytest.raises(groq.AuthenticationError):
        wrapped.invoke([("human", "hi")])
    assert backup.seen is None


def test_when_both_providers_fail_the_primary_error_is_the_one_raised(
        with_backup, monkeypatch):
    """A secondary OpenRouter 400 must not mask the Groq 429 that caused it.

    Callers' `except` blocks — and the operator reading the log — were both
    written against the primary's exception. BackupFailed subclasses OSError so
    with_fallbacks counts the backup as handled and re-raises the first error.
    """
    import groq
    backup = _RecordingBackup(raises=_groq_error("BadRequestError", 400))
    monkeypatch.setattr("app.core.model_fallback.build_backup_chat_model",
                        lambda model, temperature=0.0: backup)
    wrapped = attach_fallback(_groq_primary_raising(_groq_error("RateLimitError", 429)),
                              "openai/gpt-oss-120b")

    with pytest.raises(groq.RateLimitError):
        wrapped.invoke([("human", "hi")])


def test_backup_failed_is_a_failover_exception():
    """Load-bearing: this is what makes the primary's error survive."""
    assert issubclass(BackupFailed, OSError)
    assert should_failover(error=BackupFailed("nope")) is True


def test_a_bound_handle_still_inherits_the_fallback(with_backup, monkeypatch):
    """_routing_llm() calls .bind(reasoning_effort='low') on the shared handle."""
    backup = _RecordingBackup()
    monkeypatch.setattr("app.core.model_fallback.build_backup_chat_model",
                        lambda model, temperature=0.0: backup)
    wrapped = attach_fallback(_groq_primary_raising(_groq_error("RateLimitError", 429)),
                              "openai/gpt-oss-120b")

    assert wrapped.bind(reasoning_effort="low").invoke([("human", "hi")]) == \
        "answered by openrouter"
    assert backup.seen["reasoning_effort"] == "low"
