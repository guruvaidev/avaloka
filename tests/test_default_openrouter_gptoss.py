"""The default model is openai/gpt-oss-120b via OpenRouter, pinned (owner decision 2026-11-01).

What this file holds the code to, with no live call (the HTTP transport is mocked):

* the default provider and model, in code and in the Helm chart, agree;
* every OpenRouter request carries the provider pinning: ``only`` = the model
  family's allowlist, ``allow_fallbacks: false``, ``data_collection: "deny"``,
  ``zdr: true`` -- in the REQUEST BODY, not just on the client object;
* when the allowlisted providers cannot serve (HTTP 503), the request moves to the
  next tier, GPT-6 Luna, pinned to its own list; a 400 does not move;
* Luna is always sent reasoning effort "none" (its tool calling on Chat
  Completions needs it), even when the caller passes reasoning_effort;
* the planner survives gpt-oss Harmony phantom tool calls.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess

import httpx
import pytest
import yaml

from app.core import inference

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
GPTOSS_ONLY = ["deepinfra/bf16", "together", "crusoe", "parasail"]


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("INFERENCE_PROVIDER", "INFERENCE_PROVIDER_PLANNING", "INFERENCE_PROVIDER_CODING",
                 "INFERENCE_PROVIDER_VIZ", "GROQ_API_KEY", "GROQ_API_KEY_PLANNING_AGENT",
                 "GROQ_API_KEY_CODING_AGENT", "OPENROUTER_API_KEY", "INFERENCE_OPENROUTER_API_KEY",
                 "OPENAI_API_KEY", "INFERENCE_OPENROUTER_MODEL_LARGE", "INFERENCE_OPENROUTER_MODEL",
                 "AVALOKA_OPENROUTER_ONLY", "AVALOKA_OPENROUTER_ALLOW_FALLBACKS",
                 "AVALOKA_OPENROUTER_DATA_COLLECTION", "AVALOKA_OPENROUTER_ZDR",
                 "AVALOKA_OPENROUTER_FALLBACK_MODEL", "AVALOKA_OPENROUTER_FALLBACK_ONLY",
                 "AVALOKA_OPENROUTER_SORT", "AVALOKA_OPENROUTER_PROVIDERS",
                 "AVALOKA_DISABLE_MODEL_FALLBACK", "AVALOKA_PLANNER_PROVIDER", "AVALOKA_GROQ_FALLBACK_MODEL",
                 "AVALOKA_PLANNER_MODEL_OPENROUTER", "AVALOKA_PLANNER_OPENROUTER_ONLY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _build(**over):
    kw = dict(role="planning", agent="PLANNER", groq_model="openai/gpt-oss-120b",
              groq_api_key=None, tier="large", temperature=0.2)
    kw.update(over)
    return inference.build_chat_model(**kw)


class _Transport:
    """Stands in for the network: records each request body, answers per host and model.

    ``status_for`` maps "host|model", "host" or "model" to a status, or to
    (status, message); a status of None is a connection error (an outage).
    """

    def __init__(self, status_for=None):
        self.bodies: list = []
        self.hosts: list = []
        self.status_for = status_for or {}

    barrier = None
    barrier_models: set = set()

    def __call__(self, client, request, *args, **kwargs):
        body = json.loads(request.content or b"{}")
        if self.barrier is not None and body.get("model") in self.barrier_models:
            self.barrier_models = self.barrier_models - {body.get("model")}
            self.barrier.wait()
        self.bodies.append(body)
        self.hosts.append(request.url.host)
        model = body.get("model")
        rule = self.status_for.get(f"{request.url.host}|{model}",
                                   self.status_for.get(request.url.host,
                                                       self.status_for.get(model, 200)))
        status, message = rule if isinstance(rule, tuple) else (rule, "There is no available model "
                                                                "provider that meets your routing requirements")
        if status is None:
            raise httpx.ConnectError("connection refused", request=request)
        if status != 200:
            return httpx.Response(status, request=request,
                                  json={"error": {"message": message, "code": status}})
        return httpx.Response(200, request=request, json={
            "id": "gen-test", "object": "chat.completion", "created": 0, "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


@pytest.fixture
def transport(monkeypatch):
    t = _Transport()
    real_send = httpx.Client.send

    def send(self, request, **kw):
        # Only the model calls are ours. Anything else in the process (an embedding
        # model fetching its config, in the full suite) goes to the real transport,
        # or it would be recorded as if it were the agent's request.
        if request.url.path.endswith("/chat/completions"):
            return t(self, request, **kw)
        return real_send(self, request, **kw)

    monkeypatch.setattr(httpx.Client, "send", send)
    real_asend = httpx.AsyncClient.send

    async def asend(self, request, **kw):
        if request.url.path.endswith("/chat/completions"):
            # in a worker thread, so a barrier wait does not block the event loop
            return await asyncio.to_thread(t, self, request)
        return await real_asend(self, request, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)   # the SDK's retry back-off
    return t


# --------------------------------------------------------------------------- #
# the default
# --------------------------------------------------------------------------- #
def test_the_default_is_openrouter_running_gpt_oss_120b(clean_env):
    assert inference.resolve_provider(role="planning", agent="PLANNER") == "openrouter"
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    clean_env.setenv("GROQ_API_KEY", "k")
    assert inference.resolve_provider(role="coding", agent="CODER") == "openrouter"
    llm = _build()
    assert getattr(llm, "runnable", llm).model_name == "openai/gpt-oss-120b"


def test_the_helm_chart_ships_the_same_default_and_pinning():
    values = yaml.safe_load((ROOT / "deploy/helm/avaloka/values.yaml").read_text())["inference"]
    # "auto": the chart picks what its keys can serve (owner ruling 2026-11-02).
    assert values["provider"] == "auto"
    pin = values["openrouter"]
    # Blank: the chart defers to each model's list in code, so it cannot drift from
    # the code and choosing another model through the chart still works.
    assert pin["providersOnly"] == "" and pin["fallbackProvidersOnly"] == ""
    assert pin["allowFallbacks"] is False and pin["dataCollection"] == "deny" and pin["zdr"] is True
    assert pin["fallbackModel"] == "openai/gpt-6-luna"
    assert inference.openrouter_allowlist("openai/gpt-oss-120b") == GPTOSS_ONLY
    assert inference.openrouter_allowlist("openai/gpt-6-luna", fallback=True) == ["azure"]


def _chart_env(*args):
    """The environment the chart's ConfigMap gives the agents."""
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")
    out = subprocess.run([helm, "template", "t", str(ROOT / "deploy/helm/avaloka"), *args],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-600:]
    for doc in yaml.safe_load_all(out.stdout):
        if doc and doc.get("kind") == "ConfigMap" and "REDIS_URL" in (doc.get("data") or {}):
            return doc["data"]
    raise AssertionError("the chart rendered no backend ConfigMap")


@pytest.mark.parametrize("keys,expected", [
    (("secrets.groqPlanningKey=g", "secrets.groqCodingKey=g"), "groq"),
    (("secrets.openrouterKey=o",), "openrouter"),
    (("secrets.openrouterKey=o", "secrets.groqPlanningKey=g", "secrets.groqCodingKey=g"), "openrouter"),
    ((), "openrouter"),
], ids=["groq-only", "openrouter-only", "both", "none"])
def test_the_chart_picks_the_provider_its_keys_can_serve(keys, expected):
    """A Groq-only install keeps a model after helm upgrade (owner ruling 2026-11-02)."""
    args = [a for k in keys for a in ("--set-string", k)]
    assert _chart_env(*args).get("INFERENCE_PROVIDER") == expected


def test_an_explicit_chart_provider_is_kept():
    env = _chart_env("--set", "inference.provider=groq", "--set-string", "secrets.openrouterKey=o")
    assert env["INFERENCE_PROVIDER"] == "groq"


def test_with_an_existing_secret_the_backend_decides_from_the_keys_it_finds():
    env = _chart_env("--set", "existingSecret=my-secrets")
    assert "INFERENCE_PROVIDER" not in env, env.get("INFERENCE_PROVIDER")


def test_a_chart_install_can_run_luna_as_the_primary(clean_env, transport):
    """The chart's env must not force the gpt-oss hosts onto another model.

    An explicit AVALOKA_OPENROUTER_ONLY replaces the model's own list, so a chart that
    always emitted it would send Luna to hosts that do not serve it: every request refused.
    """
    env = _chart_env()
    assert "AVALOKA_OPENROUTER_ONLY" not in env, env.get("AVALOKA_OPENROUTER_ONLY")
    assert "AVALOKA_OPENROUTER_FALLBACK_ONLY" not in env
    for name, value in env.items():
        if name.startswith("AVALOKA_OPENROUTER"):
            clean_env.setenv(name, value)
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    clean_env.setenv("INFERENCE_OPENROUTER_MODEL_LARGE", "openai/gpt-6-luna")
    _build().invoke("hi")
    assert transport.bodies[0]["provider"]["only"] == ["azure"]
    assert transport.bodies[0]["provider"]["allow_fallbacks"] is False


def test_the_chart_passes_an_operator_allowlist_through():
    env = _chart_env("--set", "inference.openrouter.providersOnly=deepinfra/bf16")
    assert env["AVALOKA_OPENROUTER_ONLY"] == "deepinfra/bf16"


def test_configmap_renders_the_pinning_env():
    tpl = (ROOT / "deploy/helm/avaloka/templates/configmap.yaml").read_text()
    for key in ("AVALOKA_OPENROUTER_ONLY", "AVALOKA_OPENROUTER_ALLOW_FALLBACKS",
                "AVALOKA_OPENROUTER_DATA_COLLECTION", "AVALOKA_OPENROUTER_ZDR",
                "AVALOKA_OPENROUTER_FALLBACK_MODEL", "AVALOKA_OPENROUTER_FALLBACK_ONLY"):
        assert key in tpl, key


# --------------------------------------------------------------------------- #
# provider pinning reaches the request body
# --------------------------------------------------------------------------- #
def test_provider_pinning_is_in_the_request_body(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    _build().invoke("hi")
    assert transport.bodies, "no request was sent"
    body = transport.bodies[0]
    assert body["model"] == "openai/gpt-oss-120b"
    # sort "price": owner ruling 2026-11-02 (the cheapest listed provider first).
    assert body["provider"] == {"sort": "price", "only": GPTOSS_ONLY, "allow_fallbacks": False,
                                "data_collection": "deny", "zdr": True}


def test_the_small_tier_is_pinned_to_its_family_list(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    _build(tier="small").invoke("hi")
    body = transport.bodies[0]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["provider"]["only"] == GPTOSS_ONLY and body["provider"]["allow_fallbacks"] is False


def test_pinning_is_configurable(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    clean_env.setenv("AVALOKA_OPENROUTER_ONLY", "deepinfra/bf16")
    clean_env.setenv("AVALOKA_OPENROUTER_ALLOW_FALLBACKS", "true")
    _build().invoke("hi")
    assert transport.bodies[0]["provider"]["only"] == ["deepinfra/bf16"]
    assert transport.bodies[0]["provider"]["allow_fallbacks"] is True
    clean_env.setenv("AVALOKA_OPENROUTER_ONLY", "none")
    _build().invoke("hi")
    assert "only" not in transport.bodies[-1]["provider"]


# --------------------------------------------------------------------------- #
# the fallback tier
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [404, 503], ids=["no-listed-host-serves-it", "listed-hosts-down"])
def test_every_allowed_provider_down_moves_to_luna_pinned_to_azure(clean_env, transport, status):
    """404 is what OpenRouter returned live (2026-11-02), 503 is what it documents."""
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    transport.status_for["openai/gpt-oss-120b"] = status
    try:
        _build().invoke("hi")
    except Exception:  # noqa: BLE001 - the assertions below say what should have happened
        pass
    models = [b["model"] for b in transport.bodies]
    assert models and set(models[:-1]) == {"openai/gpt-oss-120b"}, models
    assert models[-1] == "openai/gpt-6-luna", f"no fallback to Luna after a {status}: {models}"
    luna = transport.bodies[-1]
    assert luna["provider"]["only"] == ["azure"] and luna["provider"]["allow_fallbacks"] is False
    assert luna["reasoning"] == {"effort": "none"}


def test_a_bad_request_does_not_move_to_the_fallback_tier(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    transport.status_for["openai/gpt-oss-120b"] = 400
    with pytest.raises(Exception):
        _build().invoke("hi")
    assert {b["model"] for b in transport.bodies} == {"openai/gpt-oss-120b"}


def test_the_fallback_tier_can_be_switched_off(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    clean_env.setenv("AVALOKA_OPENROUTER_FALLBACK_MODEL", "none")
    transport.status_for["openai/gpt-oss-120b"] = 503
    with pytest.raises(Exception):
        _build().invoke("hi")
    assert {b["model"] for b in transport.bodies} == {"openai/gpt-oss-120b"}


# --------------------------------------------------------------------------- #
# the chain: gpt-oss (OpenRouter) -> Luna (OpenRouter) -> gpt-oss (Groq), never silent
# (owner rulings 2026-11-02)
# --------------------------------------------------------------------------- #
OR, GROQ_HOST = "openrouter.ai", "api.groq.com"


def test_the_chain_is_luna_then_groq_last(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    clean_env.setenv("GROQ_API_KEY", "g")
    llm = _build()
    assert [(getattr(t, "provider", None), t.model) for t in getattr(llm, "fallbacks", [])] == [
        ("openrouter", "openai/gpt-6-luna"), ("groq", "openai/gpt-oss-120b")]


@pytest.mark.parametrize("status", [402, 404, 429, 503])
def test_each_cause_reaches_the_next_tier_and_says_which_model_answered(clean_env, transport, status):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    transport.status_for[f"{OR}|openai/gpt-oss-120b"] = status
    record = _record(_answer(_build()))
    assert record, f"a {status} was answered by another tier without saying so"
    assert record["configured_model"] == "openai/gpt-oss-120b"
    assert record["answered_model"] == "openai/gpt-6-luna" and record["status"] == status
    assert record["note"].endswith("; answered by openai/gpt-6-luna")


def test_openrouter_out_of_credit_falls_through_to_groq_last(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    clean_env.setenv("GROQ_API_KEY", "g")
    transport.status_for[OR] = 402
    record = _record(_answer(_build()))
    assert transport.hosts[-1] == GROQ_HOST and set(transport.hosts[:-1]) == {OR}
    assert record["provider"] == "groq" and record["answered_model"] == "openai/gpt-oss-120b"
    assert record["reason"] == "the openrouter account is out of credit (HTTP 402)"


def test_an_openrouter_outage_falls_through_to_groq_last(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    clean_env.setenv("GROQ_API_KEY", "g")
    transport.status_for[OR] = None
    record = _record(_answer(_build()))
    assert record["provider"] == "groq" and record["reason"] == "openrouter could not be reached"


def test_without_a_groq_key_the_chain_ends_in_a_true_error(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    transport.status_for[OR] = 402
    out = _answer(_build())
    assert isinstance(out, Exception), "answered without any tier able to"
    assert "out of credit (HTTP 402)" in str(out), f"not a true error: {out!r}"
    assert GROQ_HOST not in transport.hosts and len(transport.bodies) <= 6


def test_a_model_without_a_zero_retention_endpoint_is_named(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    clean_env.setenv("INFERENCE_OPENROUTER_MODEL_LARGE", "openai/gpt-4o")
    transport.status_for[f"{OR}|openai/gpt-4o"] = (
        404, "No allowed providers are available for the selected model.")
    assert _record(_answer(_build()))["note"] == (
        "no allowed zero-retention provider serves openai/gpt-4o; answered by openai/gpt-6-luna")


def test_the_configured_model_answering_carries_no_swap_record(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    assert _record(_answer(_build())) is None


# --------------------------------------------------------------------------- #
# every OpenRouter client is pinned, not only the factory's
# --------------------------------------------------------------------------- #
def test_the_groq_primarys_openrouter_backup_is_pinned(clean_env, transport):
    from app.core import model_fallback as mf

    clean_env.setenv("OPENROUTER_API_KEY", "k")
    mf.build_backup_chat_model("openai/gpt-oss-120b").invoke("hi")
    provider = transport.bodies[0]["provider"]
    assert provider["only"] == GPTOSS_ONLY and provider["allow_fallbacks"] is False
    assert provider["zdr"] is True and provider["data_collection"] == "deny"


def test_the_visualization_agents_openrouter_client_is_pinned(clean_env, transport):
    from app.agents import visualization_agent as viz

    clean_env.setenv("OPENROUTER_API_KEY", "k")
    clean_env.setattr(viz, "_VIZ_BACKEND", "openrouter")
    clean_env.setattr(viz, "_VIZ_MODEL", "openai/gpt-oss-120b")
    viz._build_viz_llm().invoke("hi")
    provider = transport.bodies[0]["provider"]
    assert provider["only"] == GPTOSS_ONLY and provider["allow_fallbacks"] is False


def test_the_visualization_agents_per_call_body_keeps_the_pinning(clean_env):
    """Its calls pass their own extra_body, which REPLACES the client's: it must carry the pins."""
    from app.agents import visualization_agent as viz

    clean_env.setattr(viz, "_VIZ_PROVIDER", "openrouter")
    clean_env.setattr(viz, "_VIZ_ACTIVE_MODEL", "openai/gpt-oss-120b")
    clean_env.setattr(viz, "_VIZ_IS_REASONING", True)
    clean_env.setattr(viz, "_VIZ_SORT_ACTIVE", True)
    clean_env.setattr(viz, "_VIZ_OPENROUTER_SORT", "throughput")
    provider = (viz._viz_request_kwargs("high").get("extra_body") or {}).get("provider") or {}
    assert provider.get("only") == GPTOSS_ONLY, provider
    assert provider.get("allow_fallbacks") is False and provider.get("zdr") is True
    assert provider.get("data_collection") == "deny"
    assert provider.get("sort") == "throughput"   # the agent's own, measured choice


# --------------------------------------------------------------------------- #
# Luna: reasoning off for tool calling
# --------------------------------------------------------------------------- #
def test_luna_is_sent_reasoning_none_even_when_the_caller_asks_for_effort(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    clean_env.setenv("INFERENCE_OPENROUTER_MODEL_LARGE", "openai/gpt-6-luna")
    llm = _build(reasoning_effort="medium")
    # the planner passes reasoning_effort at invoke time, chosen from its gpt-oss primary
    llm.invoke("hi", reasoning_effort="medium")
    body = transport.bodies[0]
    assert body["model"] == "openai/gpt-6-luna"
    assert body["reasoning"] == {"effort": "none"}
    assert "reasoning_effort" not in body
    assert body["provider"]["only"] == ["azure"]
    # A Luna primary has no Luna fallback behind it.
    assert not hasattr(llm, "fallbacks")


def test_gpt_oss_keeps_its_reasoning_effort(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    _build(reasoning_effort="medium").invoke("hi")
    assert transport.bodies[0]["reasoning"] == {"effort": "medium"}


# --------------------------------------------------------------------------- #
# the planner and gpt-oss Harmony phantom tool calls
# --------------------------------------------------------------------------- #
def test_harmony_framed_tool_names_resolve_to_the_real_tool():
    from app.agents import planner

    for raw in ("functions.respond_to_user", "respond_to_user<|channel|>commentary",
                "respond_to_user.commentary", "respond_to_user.json"):
        call = planner.parse_tool_call({"name": raw, "args": {"response_text": "hello"}})
        assert call.name == "respond_to_user", raw


class _FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def invoke(self, msgs, **kwargs):
        self.calls.append((msgs, kwargs))
        return self.reply


def _ai(name):
    from langchain_core.messages import AIMessage

    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": "c1"}])


def test_a_phantom_tool_call_is_re_prompted_once_with_the_real_tool_names():
    from app.agents import planner

    good = _ai("list_tasks")
    llm = _FakeLLM(good)
    out = planner._retry_phantom_tool_call_once(llm, ["m"], _ai("commentary"), tools=["t"])
    assert out is good
    assert len(llm.calls) == 1
    msgs, kwargs = llm.calls[0]
    assert "commentary" in msgs[-1].content and "respond_to_user" in msgs[-1].content
    assert kwargs == {"tools": ["t"]}


def test_a_real_tool_call_is_not_retried_and_a_second_phantom_is_not_looped():
    from app.agents import planner

    llm = _FakeLLM(_ai("json"))
    real = _ai("functions.list_tasks")
    assert planner._retry_phantom_tool_call_once(llm, ["m"], real) is real
    assert llm.calls == []
    out = planner._retry_phantom_tool_call_once(llm, ["m"], _ai("commentary"))
    assert len(llm.calls) == 1 and out.tool_calls[0]["name"] == "json"   # used as is, not retried again


# --------------------------------------------------------------------------- #
# the conversational agent diagnoses the provider build_chat_model will use
# --------------------------------------------------------------------------- #
def test_the_reply_model_diagnosis_follows_the_default(clean_env):
    from app.agents.avaloka_agent import agent

    clean_env.setenv("OPENROUTER_API_KEY", "k")
    assert agent._diagnose_missing_model() is None
    clean_env.delenv("OPENROUTER_API_KEY")
    clean_env.setenv("GROQ_API_KEY", "k")
    assert agent._diagnose_missing_model() is None        # a Groq-only install still works
    clean_env.delenv("GROQ_API_KEY")
    problem = agent._diagnose_missing_model()
    assert problem and "OPENROUTER_API_KEY" in problem


# --------------------------------------------------------------------------- #
# the record reaches the API, and the UI fixtures are the backend's own output
# --------------------------------------------------------------------------- #
UI_FIXTURES = ROOT / "ui" / "tests" / "fixtures"


def _answer(llm):
    """The reply, or the exception the chain raised (so a missing tier is an assertion)."""
    try:
        return llm.invoke("hi")
    except Exception as exc:  # noqa: BLE001
        return exc


def _record(out):
    meta = getattr(out, "response_metadata", None)
    assert isinstance(meta, dict), f"the chain did not answer: {out!r}"
    return meta.get("avaloka_answered_by")


def test_the_ui_fixture_for_a_zdr_swap_is_what_the_backend_records(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    clean_env.setenv("INFERENCE_OPENROUTER_MODEL_LARGE", "openai/gpt-4o")
    transport.status_for[f"{OR}|openai/gpt-4o"] = (
        404, "No allowed providers are available for the selected model.")
    expected = json.loads((UI_FIXTURES / "answered-by.json").read_text())["zdr_swap"]["answered_by"]
    assert _record(_answer(_build())) == expected


def test_the_ui_fixture_for_the_groq_last_tier_is_what_the_backend_records(clean_env, transport):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    clean_env.setenv("GROQ_API_KEY", "g")
    transport.status_for[OR] = 402
    expected = json.loads((UI_FIXTURES / "answered-by.json").read_text())["groq_last"]["answered_by"]
    assert _record(_answer(_build())) == expected


def test_the_model_screen_fixture_is_what_the_backend_reports(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    describe = getattr(inference, "describe_llm_setup", None)
    assert describe is not None, "no model screen data: app.core.inference.describe_llm_setup"
    assert describe() == json.loads((UI_FIXTURES / "llm-models.json").read_text())


def test_the_record_is_carried_from_the_planner_to_the_chat_response():
    """Static wiring: planner state -> ETLState -> ChatResponse (no model is called)."""
    from app.api.schemas import ChatResponse
    from app.graph.etl_state import ETLState

    assert "answered_by" in ETLState.__annotations__, "ETLState would drop answered_by"
    assert "answered_by" in ChatResponse.model_fields, "ChatResponse has no answered_by"
    planner_src = (ROOT / "app/agents/planner.py").read_text()
    assert 'state["answered_by"] = _answered_by(raw_response)' in planner_src
    server_src = (ROOT / "app/api/server.py").read_text()
    assert 'answered_by=final.get("answered_by")' in server_src
    assert '@app.get("/api/llm-models")' in server_src


# --------------------------------------------------------------------------- #
# the failure record: one per chain, isolated, never retained, never inherited
# (coordinator, 2026-11-03: the record must survive LangChain's per-runnable
# context copies and new asyncio tasks without a global registry)
# --------------------------------------------------------------------------- #
import threading  # noqa: E402

NO_ALLOWED = "No allowed providers are available for the selected model."


async def _aanswer(llm):
    try:
        return await llm.ainvoke("hi")
    except Exception as exc:  # noqa: BLE001
        return exc


def _failed(record):
    return [(f["model"], f["status"]) for f in (record or {}).get("tiers_failed", [])]


def test_concurrent_chains_keep_their_own_failure_record(clean_env, transport):
    """A thread chain and an asyncio chain fail their primaries for different reasons
    at the same moment (a barrier holds both primaries until both have arrived)."""
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    transport.status_for[f"{OR}|openai/gpt-oss-120b"] = (404, NO_ALLOWED)
    transport.status_for[f"{OR}|openai/gpt-4o"] = (404, "Model withdrawn.")
    transport.barrier = threading.Barrier(2, timeout=20)
    transport.barrier_models = {"openai/gpt-oss-120b", "openai/gpt-4o"}
    llm_thread = _build()
    clean_env.setenv("INFERENCE_OPENROUTER_MODEL_LARGE", "openai/gpt-4o")
    llm_task = _build()
    clean_env.delenv("INFERENCE_OPENROUTER_MODEL_LARGE")

    results = {}
    worker = threading.Thread(target=lambda: results.__setitem__("thread", _answer(llm_thread)))

    async def both():
        worker.start()
        return await _aanswer(llm_task)

    results["task"] = asyncio.run(both())
    worker.join(30)
    by_thread, by_task = _record(results["thread"]), _record(results["task"])
    assert by_thread["reason"] == "no allowed zero-retention provider serves openai/gpt-oss-120b"
    assert _failed(by_thread) == [("openai/gpt-oss-120b", 404)], by_thread
    assert by_task["reason"] == "openai/gpt-4o is not available from openrouter (HTTP 404)"
    assert _failed(by_task) == [("openai/gpt-4o", 404)], by_task


def test_no_failure_record_outlives_its_chain(clean_env, transport):
    """After every chain -- answered, swapped, or failed on every tier -- nothing is kept."""
    from app.core import model_tiers

    live = getattr(model_tiers, "live_chains", None)
    assert live is not None, "no way to see whether a chain left its record: model_tiers.live_chains"
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    llm = _build()
    for i in range(30):
        transport.status_for.clear()
        if i % 3 == 1:
            transport.status_for[f"{OR}|openai/gpt-oss-120b"] = 429   # swapped to Luna
        elif i % 3 == 2:
            transport.status_for[OR] = 402                            # every tier fails; it raises
        _answer(llm)
        assert live() == 0 and model_tiers.current() is None, f"chain {i} left its record behind"
    transport.status_for.clear()
    transport.status_for[OR] = 402
    for i in range(5):
        assert isinstance(asyncio.run(_aanswer(llm)), Exception)
        assert live() == 0 and model_tiers.current() is None, f"async chain {i} left its record behind"


def test_a_reused_thread_or_a_later_task_does_not_inherit_a_stale_reason(clean_env, transport):
    """The "stale reason" symptom: a chain that failed on every tier must not lend its
    failures to the next chain on the same thread, or to a later asyncio task."""
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    llm = _build()
    for run in (_answer, lambda llm_: asyncio.run(_aanswer(llm_))):
        transport.status_for.clear()
        transport.status_for[OR] = 402
        assert isinstance(run(llm), Exception)
        transport.status_for.clear()
        transport.status_for[f"{OR}|openai/gpt-oss-120b"] = (404, NO_ALLOWED)
        record = _record(run(llm))
        assert record["reason"] == "no allowed zero-retention provider serves openai/gpt-oss-120b"
        assert _failed(record) == [("openai/gpt-oss-120b", 404)], record
