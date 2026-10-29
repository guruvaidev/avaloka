"""Plan gates for data transfer, model training and model inference.

The gate is decided by app/core/plan_gate.py from the app/core/editions.py
matrix (COST gates), so the planner, the API and editions.py always agree:

* self-hosted (AVALOKA_DEPLOYMENT unset): never gated;
* Avaloka-hosted: only plans whose capabilities include the feature;
* plan unknown (no user, Supabase lookup failed): allowed (fail open).

The lookup-failure tests use the REAL resolver in app/core/plan_limits.py with
only the Supabase query (_lookup_plan) replaced, so they certify what
production does.
"""
import asyncio

import pytest

from app.agents import planner
from app.core import editions, plan_gate, plan_limits


# ------------------------------------------------------------------ fixtures

@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("AVALOKA_DEPLOYMENT", "AVALOKA_DISABLE_PLAN_GATES", "AVALOKA_EDITION",
                 "AVALOKA_PLAN_LIMITS", "AVALOKA_DEFAULT_PLAN"):
        monkeypatch.delenv(name, raising=False)
    plan_limits.clear_plan_cache()
    yield
    plan_limits.clear_plan_cache()


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setenv("AVALOKA_DEPLOYMENT", "avaloka_hosted")


@pytest.fixture
def lookup(monkeypatch):
    """Replace only the Supabase query. ``lookup.table[user] = "free"`` etc.;
    an Exception value makes the query raise. ``lookup.calls`` logs queries."""
    class _Lookup:
        table = {}
        calls = []

    log = _Lookup()
    log.table, log.calls = {}, []

    def fake_lookup(user_id):
        log.calls.append(user_id)
        value = log.table.get(user_id, "free")
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(plan_limits, "_lookup_plan", fake_lookup)
    return log


def _state(plan=None, user_id="user-1", **extra):
    s = {"user_id": user_id, "messages": []}
    if plan is not None:
        s["user_plan"] = plan
    s.update(extra)
    return s


FEATURES = sorted(plan_gate.FEATURE_CAPABILITY)


# ------------------------------------------------------------- deployment

def test_deployment_defaults_to_self_hosted():
    assert editions.deployment_from_env() == "self_hosted"


def test_deployment_read_on_every_call(monkeypatch):
    monkeypatch.setenv("AVALOKA_DEPLOYMENT", "avaloka-hosted")
    assert editions.deployment_from_env() == "avaloka_hosted"
    monkeypatch.setenv("AVALOKA_DEPLOYMENT", "nonsense")
    assert editions.deployment_from_env() == "self_hosted"


# ------------------------------------------------------- self-hosted: open

@pytest.mark.parametrize("feature", FEATURES)
def test_self_hosted_never_gated_even_for_free(feature, lookup):
    lookup.table["user-1"] = "free"
    assert plan_gate.check_feature(feature, user_id="user-1", plan="free").allowed
    assert plan_gate.check_feature(feature, user_id="user-1").allowed
    assert lookup.calls == []          # no Supabase query at all


# ----------------------------------------------------------- hosted: by plan

@pytest.mark.parametrize("plan, allowed", [
    ("enterprise", True), ("professional", True), ("Enterprise", True),
    (" PROFESSIONAL ", True), ("free", False), ("starter", False), ("trial", False),
])
@pytest.mark.parametrize("feature", FEATURES)
def test_hosted_gate_by_plan(hosted, feature, plan, allowed):
    assert plan_gate.check_feature(feature, user_id="u", plan=plan).allowed is allowed


@pytest.mark.parametrize("feature", FEATURES)
@pytest.mark.parametrize("plan", ["free", "professional", "enterprise"])
def test_gate_agrees_with_editions_matrix(hosted, feature, plan):
    caps = editions.resolve_capabilities("avaloka_hosted", plan)
    expected = getattr(caps, plan_gate.FEATURE_CAPABILITY[feature])
    assert plan_gate.check_feature(feature, user_id="u", plan=plan).allowed is expected


@pytest.mark.parametrize("feature", FEATURES)
def test_gated_capabilities_are_cost_gates(feature):
    # editions.py: a COST gate never applies to a self-hosted deployment.
    cap = plan_gate.FEATURE_CAPABILITY[feature]
    assert editions.GATE_REASON[cap] is editions.GateReason.COST


def test_allowed_plans_cheapest_first(hosted):
    d = plan_gate.check_feature("data_transfer", user_id="u", plan="free")
    assert d.allowed_plans() == ["professional", "enterprise"]


def test_hosted_plan_resolved_from_user_id(hosted, lookup):
    lookup.table.update({"user-1": "enterprise", "user-2": "free"})
    assert plan_gate.check_feature("data_transfer", user_id="user-1").allowed
    assert not plan_gate.check_feature("data_transfer", user_id="user-2").allowed


def test_successful_lookup_is_cached(hosted, lookup):
    lookup.table["user-1"] = "enterprise"
    for _ in range(3):
        plan_gate.check_feature("model_training", user_id="user-1")
    assert lookup.calls == ["user-1"]


# ------------------------------------------------ fail open (real resolver)

def test_lookup_failure_fails_open_with_the_real_resolver(hosted, lookup):
    lookup.table["user-1"] = RuntimeError("supabase down")
    d = plan_gate.check_feature("data_transfer", user_id="user-1")
    assert d.allowed and d.plan is None


def test_lookup_failure_is_unknown_not_free(lookup):
    lookup.table["user-1"] = RuntimeError("supabase down")
    assert plan_limits.try_resolve_plan("user-1") is None
    # Upload limits keep their own policy: Free while the lookup is failing.
    assert plan_limits.resolve_plan("user-1") == "free"


def test_old_two_item_cache_entries_still_read(lookup, monkeypatch):
    # (expires_at, plan) entries, as older code or other tests write them.
    import time
    monkeypatch.setitem(plan_limits._CACHE, "user-9", (time.monotonic() + 60, "enterprise"))
    assert plan_limits.resolve_plan("user-9") == "enterprise"
    assert plan_limits.try_resolve_plan("user-9") == "enterprise"
    assert lookup.calls == []


def test_failed_lookup_is_retried_only_after_failure_ttl(hosted, lookup):
    lookup.table["user-1"] = RuntimeError("supabase down")
    plan_gate.check_feature("data_transfer", user_id="user-1")
    plan_gate.check_feature("data_transfer", user_id="user-1")
    assert lookup.calls == ["user-1"]


def test_no_user_is_allowed(hosted, lookup):
    # Internal callers, tests and scheduled re-runs carry no user.
    assert plan_gate.check_feature("data_transfer").allowed
    assert planner._premium_allowed({"messages": []})
    assert planner._premium_allowed(None)
    assert lookup.calls == []


def test_disable_switch(hosted, monkeypatch):
    monkeypatch.setenv("AVALOKA_DISABLE_PLAN_GATES", "1")
    assert plan_gate.check_feature("data_transfer", user_id="u", plan="free").allowed


def test_billing_disabled_install_is_not_gated(hosted, lookup, monkeypatch):
    monkeypatch.setenv("AVALOKA_EDITION", "oss")
    assert plan_gate.check_feature("model_training", user_id="user-1").allowed
    assert lookup.calls == []


# ------------------------------------------------------------ upgrade message

@pytest.mark.parametrize("feature, words", [
    ("data_transfer", "Data transfer is available on the Professional and Enterprise plans"),
    ("model_training", "Model training is available on the Professional and Enterprise plans"),
])
def test_upgrade_message_names_feature_and_plan(hosted, feature, words):
    msg = planner._upgrade_message(feature, _state("free"))
    assert words in msg
    assert "Your current plan is Free." in msg
    assert "analysis, charts and code keep working" in msg


def test_upgrade_message_title_cases_plan(hosted):
    msg = planner._upgrade_message("data_transfer", _state("starter_monthly"))
    assert "Your current plan is Starter Monthly." in msg


# ---------------------------------------------------------- data transfer gate

class _PastGate(Exception):
    """Raised by the first step after the gate, so no real lookup or job runs."""


@pytest.fixture
def stop_after_gate(monkeypatch):
    def first_step_after_gate(state):
        raise _PastGate()

    monkeypatch.setattr(planner, "_get_db_registry", first_step_after_gate)


def test_hosted_free_user_transfer_blocked_before_any_work(hosted, stop_after_gate):
    state = _state("free", pending_clarification={"kind": "dta", "awaiting": "destination"})
    msg = planner._handle_initiate_transfer(state, None)   # _PastGate here = gate failed
    assert msg.startswith("Data transfer is available on the Professional and Enterprise plans")
    assert state["plan_upgrade_required"] == "data_transfer"
    assert state["pending_clarification"] is None


def test_hosted_free_user_wizard_reply_blocked(hosted):
    state = _state("free", pending_clarification={"kind": "dta"})
    msg = planner._resolve_pending_dta(state, {"awaiting": "destination", "collected": {}}, "to my_bucket")
    assert "Data transfer is available" in msg
    assert state["plan_upgrade_required"] == "data_transfer"
    assert state["pending_clarification"] is None


@pytest.mark.parametrize("plan", ["enterprise", "professional"])
def test_hosted_paid_user_transfer_passes_the_gate(hosted, stop_after_gate, plan):
    state = _state(plan)
    with pytest.raises(_PastGate):
        planner._handle_initiate_transfer(state, None)
    assert "plan_upgrade_required" not in state


def test_self_hosted_free_user_transfer_passes_the_gate(stop_after_gate, lookup):
    # What the data-transfer wizard tests exercise: no deployment configured.
    state = _state("free")
    with pytest.raises(_PastGate):
        planner._handle_initiate_transfer(state, None)
    assert "plan_upgrade_required" not in state


def test_hosted_transfer_allowed_when_lookup_fails(hosted, stop_after_gate, lookup):
    lookup.table["user-1"] = RuntimeError("supabase down")
    state = _state()          # no user_plan on the state: resolved from user_id
    with pytest.raises(_PastGate):
        planner._handle_initiate_transfer(state, None)


# ---------------------------------------------------------- model training gate

def _ai_texts(out):
    return [m.content for m in out.get("messages") or [] if type(m).__name__ == "AIMessage"]


@pytest.mark.parametrize("flags", [
    {"enable_training": True},
    {"skip_to_training": True},
    {"task_schedule": {"task_type": "training", "cron": "0 3 * * *"}},
])
def test_hosted_free_user_training_blocked(hosted, flags):
    out = planner._apply_training_plan_gate(_state("free"), dict(_state("free"), **flags))
    assert out["plan_upgrade_required"] == "model_training"
    assert out["enable_training"] is False
    assert out["skip_to_training"] is False
    assert out["ready_to_code"] is False and out["ready_to_summarize"] is False
    assert not (out.get("task_schedule") or {}).get("task_type") == "training"
    assert any("Model training is available" in t for t in _ai_texts(out))


def test_stale_training_flag_on_analysis_turn_is_dropped_silently(hosted):
    out_in = dict(_state("free"), enable_training=True, ready_to_code=True)
    out = planner._apply_training_plan_gate(_state("free"), out_in)
    assert out["enable_training"] is False
    assert out["ready_to_code"] is True          # the analysis still runs
    assert "plan_upgrade_required" not in out
    assert _ai_texts(out) == []


def test_training_schedule_on_analysis_turn_is_still_blocked(hosted):
    out_in = dict(_state("free"), ready_to_code=True, task_schedule={"task_type": "training"})
    out = planner._apply_training_plan_gate(_state("free"), out_in)
    assert out["plan_upgrade_required"] == "model_training"
    assert out["task_schedule"] is None


def test_paid_user_training_untouched(hosted):
    out_in = dict(_state("professional"), enable_training=True)
    assert planner._apply_training_plan_gate(_state("professional"), out_in) is out_in


def test_self_hosted_free_user_training_untouched():
    out_in = dict(_state("free"), enable_training=True)
    assert planner._apply_training_plan_gate(_state("free"), out_in) is out_in


def test_non_training_turn_untouched(hosted):
    out_in = dict(_state("free"), ready_to_code=True)
    assert planner._apply_training_plan_gate(_state("free"), out_in) is out_in


def test_planner_node_applies_training_gate(hosted, monkeypatch):
    monkeypatch.setattr(planner, "_plan_etl_job_impl", lambda s: dict(s, enable_training=True))
    out = planner.plan_etl_job(_state("free"))
    assert out["plan_upgrade_required"] == "model_training"


# ------------------------------------------------------------ state and schema

def test_etl_state_declares_plan_keys():
    # LangGraph drops undeclared keys; without these the plan never reaches the planner.
    from app.graph.etl_state import ETLState
    keys = ETLState.__annotations__
    for k in ("user_plan", "plan_upgrade_required", "is_dta_request"):
        assert k in keys, k


def test_chat_response_has_plan_upgrade_required():
    from app.api.schemas import ChatResponse
    field = ChatResponse.model_fields["plan_upgrade_required"]
    assert field.default is None


# ----------------------------------------------------------------- server.py

@pytest.fixture
def server():
    from app.api import server as srv
    return srv


@pytest.mark.parametrize("plan, allowed", [
    ("enterprise", True), ("professional", True), ("Enterprise", True),
    ("free", False), ("", True), (None, True),    # empty/None = unknown -> fail open
])
def test_server_plan_allows_premium_hosted(server, hosted, plan, allowed):
    assert server._plan_allows_premium(plan) is allowed


def test_server_self_hosted_allows_free(server):
    assert server._plan_allows_premium("free") is True


def test_server_disable_switch(server, hosted, monkeypatch):
    monkeypatch.setenv("AVALOKA_DISABLE_PLAN_GATES", "1")
    assert server._plan_allows_premium("free") is True


def test_require_premium_plan_403_for_hosted_free(server, hosted, lookup):
    lookup.table["user-1"] = "free"
    with pytest.raises(server.HTTPException) as info:
        asyncio.run(server._require_premium_plan("user-1", "model_inference"))
    err = info.value
    assert err.status_code == 403
    assert err.detail["reason"] == "plan_upgrade_required"
    assert err.detail["feature"] == "model_inference"
    assert err.detail["plan"] == "free"
    assert err.detail["allowed_plans"] == ["professional", "enterprise"]
    assert err.detail["message"].startswith("Model inference is available")


def test_require_premium_plan_passes_for_paid(server, hosted, lookup):
    lookup.table["user-1"] = "enterprise"
    assert asyncio.run(server._require_premium_plan("user-1", "model_inference")) is None


def test_require_premium_plan_fails_open_like_the_planner(server, hosted, lookup):
    lookup.table["user-1"] = RuntimeError("supabase down")
    assert asyncio.run(server._require_premium_plan("user-1", "model_inference")) is None


def test_require_premium_plan_self_hosted_passes(server, lookup):
    lookup.table["user-1"] = "free"
    assert asyncio.run(server._require_premium_plan("user-1", "model_training")) is None