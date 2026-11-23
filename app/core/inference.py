"""
Provider-agnostic chat-model factory for Avaloka agents.

Every agent LLM is built through :func:`build_chat_model`, which selects a
concrete LangChain chat model based on the configured *inference provider*.
This abstracts the agent layer away from any single vendor (Groq): the very
same agent code runs against a local OpenAI-spec model in Kubernetes
(vLLM / Ollama), Groq Cloud, OpenRouter, AWS Bedrock, GCP Vertex AI, or
Azure AI — selected purely by environment configuration.

    +-------------------+        INFERENCE_PROVIDER=...
    |   agent modules   |  ---->  build_chat_model()  ---->  BaseChatModel
    | (planner, coder,  |             |                      (ChatGroq /
    |  validator, ...)  |             |                       ChatOpenAI /
    +-------------------+             |                       ChatBedrock /
                                      |                       ChatVertexAI /
                                      v                       AzureChatOpenAI)
                          provider registry + env

Design guarantees
-----------------
* **OpenRouter is the default** (owner decision 2026-11-01): when
  ``INFERENCE_PROVIDER`` is unset, agents run ``openai/gpt-oss-120b`` through
  OpenRouter, pinned to a short list of upstream providers (see "OpenRouter
  provider pinning" below). ``INFERENCE_PROVIDER=groq`` still returns a
  ``ChatGroq`` configured exactly as before, including the gpt-oss reasoning
  kwargs.
* **"Disabled when unconfigured" is preserved.** If the selected provider
  has no credentials/endpoint, :func:`build_chat_model` returns ``None`` and
  the caller falls back to its deterministic path — exactly as it did when a
  ``GROQ_API_KEY`` was missing.
* **Optional SDKs are lazy.** ``langchain-aws`` (Bedrock) and
  ``langchain-google-vertexai`` (Vertex) are imported only when their
  provider is selected, so a Groq-only or local-only install never needs
  them.

Selecting a provider
--------------------
Global switch (default ``openrouter``)::

    INFERENCE_PROVIDER=local        # laptop / kind: OpenAI-spec model in-cluster
    INFERENCE_PROVIDER=groq         # Groq Cloud
    INFERENCE_PROVIDER=openrouter   # OpenRouter (default)
    INFERENCE_PROVIDER=bedrock      # AWS Bedrock
    INFERENCE_PROVIDER=vertex       # GCP Vertex AI
    INFERENCE_PROVIDER=azure        # Azure AI / Azure OpenAI

Per-role override (``PLANNING`` | ``CODING`` | ``VIZ``)::

    INFERENCE_PROVIDER_PLANNING=groq
    INFERENCE_PROVIDER_CODING=local

Per-agent override (highest precedence), e.g. the planner::

    AVALOKA_PLANNER_PROVIDER=local

OpenRouter routing (applies to EVERY agent on the OpenRouter path)::

    AVALOKA_OPENROUTER_SORT=throughput        # default; "price"/"latency", or "none"
    AVALOKA_<AGENT>_OPENROUTER_SORT=latency   # per-agent override
    AVALOKA_OPENROUTER_PROVIDERS=Groq,Cerebras  # optional preferred providers, in order

OpenRouter provider pinning (owner decision 2026-11-01; privacy: prompts carry
schemas and sample rows, so they go only to the providers listed)::

    AVALOKA_OPENROUTER_ONLY=deepinfra/bf16,together   # primary's allowlist; "none" = any
    AVALOKA_OPENROUTER_ALLOW_FALLBACKS=false          # default: never leave the list
    AVALOKA_OPENROUTER_DATA_COLLECTION=deny           # providers that train are excluded
    AVALOKA_OPENROUTER_ZDR=true                       # zero-data-retention endpoints only
    AVALOKA_OPENROUTER_FALLBACK_MODEL=openai/gpt-6-luna  # next tier; "none" disables
    AVALOKA_OPENROUTER_FALLBACK_ONLY=azure            # the fallback tier's allowlist
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Every chat model built anywhere in this process sends a sealed credential
# token as an opaque placeholder, never the token (app/core/typed_secrets.py).
from app.core.typed_secrets import install_model_scrub as _install_model_scrub  # noqa: E402

_install_model_scrub()

# --------------------------------------------------------------------------
# Provider names
# --------------------------------------------------------------------------
GROQ = "groq"
LOCAL = "local"          # OpenAI-spec server in-cluster (vLLM / Ollama)
OPENAI = "openai"        # OpenAI-spec server (api.openai.com or custom base)
OPENROUTER = "openrouter"
BEDROCK = "bedrock"      # AWS Bedrock
VERTEX = "vertex"        # GCP Vertex AI
AZURE = "azure"          # Azure AI / Azure OpenAI

# Normalise user-facing aliases to canonical provider names.
_PROVIDER_ALIASES = {
    # The empty string is what an unset INFERENCE_PROVIDER looks like by the
    # time it reaches here, so this entry -- not the `or` below -- is what
    # actually decides the default. It said GROQ, which is why setting the
    # fallback alone changed nothing. MP-01 (owner decision 2026-11-01):
    # OpenRouter, matching the Helm chart's inference.provider
    # (deploy/helm/avaloka/values.yaml). It was Groq from 2026-10-31.
    "": OPENROUTER,
    "groqcloud": GROQ,
    "groq-cloud": GROQ,
    "vllm": LOCAL,
    "ollama": LOCAL,
    "local-openai": LOCAL,
    "kind": LOCAL,
    "openai-compatible": OPENAI,
    "open-router": OPENROUTER,
    "router": OPENROUTER,
    "aws": BEDROCK,
    "aws-bedrock": BEDROCK,
    "amazon": BEDROCK,
    "gcp": VERTEX,
    "vertexai": VERTEX,
    "vertex-ai": VERTEX,
    "google": VERTEX,
    "gemini": VERTEX,
    "azure-openai": AZURE,
    "azureopenai": AZURE,
    "azure-ai": AZURE,
    "microsoft": AZURE,
}

# gpt-oss only: the low/medium/high reasoning_effort values are the gpt-oss
# value space. Other models (qwen3, deepseek-r1) use different reasoning
# controls and would 400 on these values, so reasoning kwargs are attached
# ONLY for Groq-hosted gpt-oss models.
REASONING_MODEL_PREFIXES = ("openai/gpt-oss",)


def supports_reasoning(model: str) -> bool:
    """True when *model* accepts gpt-oss style reasoning kwargs."""
    return (model or "").lower().startswith(REASONING_MODEL_PREFIXES)


def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first non-empty environment value among *names*."""
    for name in names:
        if not name:
            continue
        val = os.environ.get(name)
        if val:
            return val
    return default


def _default_provider_with_a_usable_key(groq_api_key: Optional[str] = None) -> str:
    """OpenRouter when it has a key (or nothing has one), else a provider that does.

    Only consulted when no provider was configured at all. OpenRouter is the
    default the Helm chart ships (deploy/helm/avaloka/values.yaml
    ``inference.provider``), so an unconfigured install and a chart install pick
    the same provider (MP-01, owner decision 2026-11-01: gpt-oss-120b via
    OpenRouter; it was Groq from 2026-10-31). The providers whose keys are a
    single env var are checked next, so an install holding only a Groq or
    OpenAI key keeps a working LLM on upgrade; anything needing cloud
    credentials must be selected explicitly.
    """
    if _env_first("OPENROUTER_API_KEY", "INFERENCE_OPENROUTER_API_KEY"):
        return OPENROUTER
    if groq_api_key or _env_first(
        "GROQ_API_KEY", "GROQ_API_KEY_PLANNING_AGENT", "GROQ_API_KEY_CODING_AGENT"
    ):
        logger.info(
            "No INFERENCE_PROVIDER set and no OpenRouter key; using Groq, which has one. "
            "Set INFERENCE_PROVIDER explicitly to pin this."
        )
        return GROQ
    if _env_first("OPENAI_API_KEY"):
        return OPENAI
    return OPENROUTER   # nothing usable: keep the documented default and let it report


def _canonical_provider(raw: Optional[str], groq_api_key: Optional[str] = None) -> str:
    """Resolve a provider name, defaulting to OpenRouter.

    The default was Groq, which meant a first run required a Groq account
    specifically. OpenRouter fronts many providers behind one key -- including
    Groq itself -- so one credential reaches whatever model a user already has
    access to, which is the better default for an open-source install. The
    owner's decision of 2026-11-01 keeps it: openai/gpt-oss-120b via OpenRouter
    (the cheapest per correct answer), pinned to a provider allowlist, with
    GPT-6 Luna as the optional higher tier; the Helm chart now ships the same.

    Groq remains fully supported and is still the fastest path: set
    INFERENCE_PROVIDER=groq, or AVALOKA_<AGENT>_PROVIDER=groq per agent.
    """
    key = (raw or "").strip().lower()
    if not key:
        # Nothing was configured, so this is the implicit default. Prefer
        # OpenRouter, but fall back to a provider whose key is present: selecting a provider
        # whose key is absent returns no LLM at all, and the only signal is a
        # log line. Anyone who had GROQ_API_KEY set and nothing else -- which is
        # what the older docs tell you to do -- would silently lose the LLM on
        # upgrade. An EXPLICIT setting is never second-guessed below, so
        # INFERENCE_PROVIDER=openrouter with no key still reports itself.
        return _default_provider_with_a_usable_key(groq_api_key)
    canonical = _PROVIDER_ALIASES.get(key, key or OPENROUTER)
    return canonical


def tool_choice_for_provider(provider: str, client_name: str = "") -> str:
    """The tool_choice value a provider actually accepts.

    Keyed on the PROVIDER, not the client class. ``local``, ``openrouter`` and
    ``openai`` all build the same ``ChatOpenAI``, so the class name cannot tell
    them apart -- and the two that matter need OPPOSITE values.

    Measured, not assumed:

      OpenRouter, openai/gpt-oss-120b
        required -> 0/3 tool calls, 3/3 finish_reason="error"  (19 tools)
        required -> 0/3 tool calls, 2/3 errors                 (3 tools, so
                                                                count is not it)
        auto     -> tool calls returned, 0/3 errors
      Ollama, gemma4:e4b
        required -> tool call returned
        auto     -> no tool call, prose answer instead

    Groq keeps "required": it works there, and the dispatch path relies on always
    receiving a tool call. Vertex spells it "any". An unknown provider, and a
    disabled LLM, get "required" so the name is always bound -- leaving it
    unbound raised NameError rather than reporting the disabled LLM, which broke
    57 tests.
    """
    if provider == VERTEX or client_name == "ChatVertexAI":
        return "any"
    if provider == OPENROUTER:
        return "auto"
    return "required"


#: The strongest tool_choice each provider ACCEPTS when a reply came back with
#: no tool call and the caller wants to insist. None means the provider has no
#: accepted forcing value (or none has been measured), so the caller keeps the
#: portable value from :func:`tool_choice_for_provider` and relies on the
#: prompt instead.
#:
#: Accepting a value and enforcing it are different things, and the table only
#: promises the first:
#:
#:   groq        "required"  accepted and enforced (dispatch relies on it)
#:   openai      "required"  accepted (OpenAI spec)
#:   local       "required"  accepted by Ollama, NOT enforced: gemma4:e4b on
#:                           Ollama answered "what predicts a stroke?" in prose
#:                           under "required" and under a named function
#:                           (measured 2026-10-17). An explicit instruction
#:                           that a tool call is mandatory, sent with it, got
#:                           a tool call 2/2.
#:   vertex      "any"       Vertex's spelling
#:   openrouter  None        "required" -> finish_reason="error", 0 tool calls
#:                           (see tool_choice_for_provider)
#:   bedrock/azure           not measured; None, never a guess
_FORCED_TOOL_CHOICE = {
    GROQ: "required",
    OPENAI: "required",
    LOCAL: "required",
    VERTEX: "any",
    OPENROUTER: None,
}


def forced_tool_choice_for_provider(provider: str, client_name: str = "") -> Optional[str]:
    """The tool_choice to retry with when a provider returned no tool call.

    Returns None when the provider rejects every forcing value or none has
    been measured for it; the caller then retries with the portable choice
    and an explicit instruction, which is the part that actually moved Ollama.
    """
    if client_name == "ChatVertexAI":
        return "any"
    return _FORCED_TOOL_CHOICE.get(provider)


def resolve_provider(*, role: str, agent: str, groq_api_key: Optional[str] = None) -> str:
    """
    Resolve the active provider for an agent, most specific wins:

    1. ``AVALOKA_<AGENT>_PROVIDER``            (per-agent)
    2. ``INFERENCE_PROVIDER_<ROLE>``           (per-role: PLANNING/CODING/VIZ)
    3. ``INFERENCE_PROVIDER``                  (global)
    4. OpenRouter if it has a key (or nothing has one), else a provider that does
    """
    raw = _env_first(
        f"AVALOKA_{agent.upper()}_PROVIDER",
        f"INFERENCE_PROVIDER_{role.upper()}",
        "INFERENCE_PROVIDER",
    )
    return _canonical_provider(raw, groq_api_key)


# --------------------------------------------------------------------------
# Per-provider model resolution (non-Groq providers)
#
# Each agent passes a Groq-native model id (e.g. "openai/gpt-oss-120b").
# That id is meaningless to Bedrock/Vertex/etc., so for non-Groq providers we
# resolve the model from environment, tiered as "large"|"small" so a single
# config maps every agent sensibly. Resolution order (most specific wins):
#
#   1. AVALOKA_<AGENT>_MODEL_<PROVIDER>     (this agent, this provider)
#   2. INFERENCE_<PROVIDER>_MODEL_<TIER>    (all agents of this tier)
#   3. INFERENCE_<PROVIDER>_MODEL           (all agents)
#   4. built-in default below
# --------------------------------------------------------------------------
#: The default locally served model: Qwen3-4B-Instruct-2507 at Q4_K_M, as the
#: Ollama library publishes it. Apache-2.0 on both the Hugging Face card and the
#: licence layer of this Ollama tag. It replaced qwen2.5:3b-instruct, whose
#: weights are under the Qwen RESEARCH licence (non-commercial) -- a commercial
#: edition must not default to them. This is the Instruct-2507 build, which has
#: no thinking mode, so tool calls arrive without a <think> preamble and nothing
#: has to send /no_think. (The bare ``qwen3:4b`` tag is the Thinking-2507 build;
#: do not swap to it without handling reasoning output.)
#: tests/test_local_model_licence.py holds every shipped local tag to a licence
#: allowlist; a new default must be added there with its licence, or CI fails.
DEFAULT_LOCAL_MODEL = "qwen3:4b-instruct-2507-q4_K_M"

_PROVIDER_MODEL_DEFAULTS = {
    # Laptop/kind default is the small model for BOTH tiers so a CPU-only
    # dev box never tries to serve a 14B model. Point _LARGE at the GPU-served
    # model (e.g. qwen2.5-14b/32b) when you run the vLLM GPU deployment.
    LOCAL: {"large": DEFAULT_LOCAL_MODEL, "small": DEFAULT_LOCAL_MODEL},
    OPENAI: {"large": "gpt-4o", "small": "gpt-4o-mini"},
    # Same models the Groq path uses, and the ones every measurement in
    # docs/test-reports was taken on -- openai/gpt-oss-120b routing accuracy
    # 54/54, the conversational suite, the blueprint harness. Changing the
    # default provider must not silently change the model underneath it, or
    # every published number stops describing the shipped configuration.
    OPENROUTER: {
        "large": "openai/gpt-oss-120b",
        "small": "openai/gpt-oss-20b",
    },
    BEDROCK: {
        "large": "meta.llama3-1-70b-instruct-v1:0",
        "small": "meta.llama3-1-8b-instruct-v1:0",
    },
    VERTEX: {"large": "gemini-2.5-flash", "small": "gemini-2.5-flash"},
    # Azure keys off *deployment* names, resolved separately below.
    AZURE: {"large": "gpt-4o", "small": "gpt-4o-mini"},
}


#: What each provider's builder below needs before it returns a client, in the
#: operator's words. Groq is absent on purpose: its key is per agent, so the
#: caller names it. Kept beside the builders so the two cannot drift apart.
_PROVIDER_REQUIREMENT = {
    OPENROUTER: "OPENROUTER_API_KEY",
    OPENAI: "OPENAI_API_KEY",
    LOCAL: "INFERENCE_LOCAL_BASE_URL",
    BEDROCK: "AWS_REGION and the langchain-aws package",
    VERTEX: "the langchain-google-vertexai package and Google credentials",
    AZURE: "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and a deployment name",
}


def provider_requirement(provider: str) -> str:
    """The setting to name when *provider* returned no client."""
    return _PROVIDER_REQUIREMENT.get(provider, f"credentials for provider {provider!r}")


def _resolve_model(provider: str, agent: str, tier: str) -> str:
    tier = "small" if str(tier).lower() == "small" else "large"
    default = _PROVIDER_MODEL_DEFAULTS.get(provider, {}).get(tier, "")
    return _env_first(
        f"AVALOKA_{agent.upper()}_MODEL_{provider.upper()}",
        f"INFERENCE_{provider.upper()}_MODEL_{tier.upper()}",
        f"INFERENCE_{provider.upper()}_MODEL",
        default=default,
    )


# --------------------------------------------------------------------------
# Concrete provider builders. Each returns a LangChain BaseChatModel or None
# when the provider is not configured (missing endpoint/credentials), which
# preserves each agent's "LLM disabled -> deterministic fallback" behavior.
# --------------------------------------------------------------------------
def _build_groq(*, model, temperature, api_key, reasoning_effort, reasoning_format):
    if not api_key:
        return None
    from langchain_groq import ChatGroq

    kwargs = dict(model=model, temperature=temperature, api_key=api_key)
    # Reasoning kwargs are gpt-oss only, and only when the caller asked for
    # them (reasoning_effort is None for non-reasoning agents). This keeps the
    # emitted ChatGroq byte-identical to the pre-abstraction construction.
    if reasoning_effort is not None and supports_reasoning(model):
        if reasoning_format is not None:
            kwargs["reasoning_format"] = reasoning_format
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(**kwargs)


# OpenRouter sends each request to one of several upstream providers. Its
# default choice is not the fastest: measured on gpt-oss locally, agents on the
# default route got ~55-70 output tokens/s (validator 780 tokens in 14.6 s),
# while the visualization agent, the only one that asked for
# sort=throughput, got ~1,000 tokens/s. Set the routing here, once, so every
# agent gets it.
_OPENROUTER_SORTS = ("throughput", "price", "latency")


#: Upstream providers OpenRouter may send each model family to (owner decision
#: 2026-11-01). Prompts carry table schemas and sample rows, so the default is a
#: short allowlist with ``allow_fallbacks: false``, ``data_collection: "deny"`` and
#: ``zdr: true`` -- egress stays inside the list, and a request no listed provider
#: can serve fails instead of going elsewhere. Measured 2026-11-02: HTTP 404 "No
#: allowed providers are available for the selected model" when no listed provider
#: serves the model; OpenRouter documents 503 for "no available model provider that
#: meets your routing requirements". Both move to the optional fallback tier
#: (AVALOKA_OPENROUTER_FALLBACK_MODEL).
#: Request-body fields per https://openrouter.ai/docs/features/provider-routing.
#: Chosen from OpenRouter's endpoint list and ZDR list (2026-11-01) and the
#: providers' own terms: US-based, no training on API data, no retention, on the
#: ZDR list, and tool calling supported. gpt-oss-20b (the small tier) is served
#: by DeepInfra and Parasail of these; Together and Crusoe serve 120b only.
#: GPT-6 Luna's ZDR endpoints are Azure's (OpenAI's own endpoint is not on
#: OpenRouter's ZDR list). See docs/privacy-and-egress.md for the evidence.
OPENROUTER_PROVIDER_ALLOWLISTS = (
    ("openai/gpt-oss-", ("deepinfra/bf16", "together", "crusoe", "parasail")),
    ("openai/gpt-6-luna", ("azure",)),
)

#: The optional higher-quality tier behind the default model (owner decision
#: 2026-11-01): GPT-6 Luna via OpenRouter. "none" (or AVALOKA_DISABLE_MODEL_FALLBACK=1)
#: switches it off; a primary that IS this model gets no fallback.
DEFAULT_OPENROUTER_FALLBACK_MODEL = "openai/gpt-6-luna"

#: Models whose function calling on Chat Completions needs reasoning switched
#: off (GPT-6 Luna supports tools there only with reasoning effort "none"; its
#: OpenRouter entry lists reasoning as optional with "none" a supported effort).
REASONING_OFF_MODEL_PREFIXES = ("openai/gpt-6-luna",)

_FALSE = ("0", "false", "no", "off")


def openrouter_allowlist(model: str, *, agent: str = "", fallback: bool = False) -> Optional[list]:
    """The providers OpenRouter may route *model* to, or None for its whole pool.

    An explicit setting wins (``AVALOKA_<AGENT>_OPENROUTER_ONLY`` /
    ``AVALOKA_OPENROUTER_ONLY`` for the primary, ``AVALOKA_OPENROUTER_FALLBACK_ONLY``
    for the fallback tier); "none" lifts the restriction. Otherwise the family
    default from :data:`OPENROUTER_PROVIDER_ALLOWLISTS`; a model with no family
    entry is not restricted.
    """
    names = (("AVALOKA_OPENROUTER_FALLBACK_ONLY",) if fallback else
             (f"AVALOKA_{agent.upper()}_OPENROUTER_ONLY" if agent else "",
              "AVALOKA_OPENROUTER_ONLY"))
    raw = _env_first(*names)
    if raw is not None:
        if raw.strip().lower() in ("none", "any", "off", "*"):
            return None
        return [p.strip() for p in raw.split(",") if p.strip()] or None
    for prefix, providers in OPENROUTER_PROVIDER_ALLOWLISTS:
        if (model or "").lower().startswith(prefix):
            return list(providers)
    return None


def openrouter_fallback_model() -> Optional[str]:
    """The model the OpenRouter primary falls back to, or None when switched off."""
    if (os.getenv("AVALOKA_DISABLE_MODEL_FALLBACK") or "").strip().lower() in {"1", "true", "yes"}:
        return None
    raw = _env_first("AVALOKA_OPENROUTER_FALLBACK_MODEL", default=DEFAULT_OPENROUTER_FALLBACK_MODEL)
    raw = (raw or "").strip()
    return None if raw.lower() in ("", "none", "off") else raw


def _openrouter_routing(agent: str, model: str = "", *, fallback: bool = False) -> Optional[dict]:
    """The OpenRouter ``provider`` preferences for *agent* and *model*, or None."""
    sort = (_env_first(
        f"AVALOKA_{agent.upper()}_OPENROUTER_SORT",
        "AVALOKA_OPENROUTER_SORT",
        # Owner ruling 2026-11-04: the fastest listed provider first. It was
        # "price" from 2026-11-02; a smoke sample showed price-sorted calls about
        # 4x slower and one of three chat turns without its tool call.
        # AVALOKA_OPENROUTER_SORT=price still selects the cheapest listed host.
        default="throughput",
    ) or "").strip().lower()
    order = [
        p.strip()
        for p in (_env_first(
            f"AVALOKA_{agent.upper()}_OPENROUTER_PROVIDERS",
            "AVALOKA_OPENROUTER_PROVIDERS",
            default="",
        ) or "").split(",")
        if p.strip()
    ]
    routing: dict = {}
    if sort in _OPENROUTER_SORTS:
        routing["sort"] = sort
    elif sort not in ("", "none", "off", "default"):
        logger.warning("Ignoring unknown OpenRouter sort %r for %s (use %s or none).",
                       sort, agent, "/".join(_OPENROUTER_SORTS))
    only = openrouter_allowlist(model, agent=agent, fallback=fallback)
    if only:
        routing["only"] = only
        # Decision 2026-11-01: never leave the allowlist. With fallbacks allowed,
        # OpenRouter may send the prompt to any provider when the listed ones are
        # busy, which is exactly the egress the list exists to prevent.
        allow = (_env_first("AVALOKA_OPENROUTER_ALLOW_FALLBACKS", default="false") or "")
        routing["allow_fallbacks"] = allow.strip().lower() not in _FALSE
    if order:
        routing["order"] = order
        if not only:
            # Preferred providers first, but never fail the request because they
            # are busy: fall back to the rest of the pool.
            routing["allow_fallbacks"] = True
    collection = (_env_first("AVALOKA_OPENROUTER_DATA_COLLECTION", default="deny") or "").strip().lower()
    if collection in ("allow", "deny"):
        routing["data_collection"] = collection
    zdr = (_env_first("AVALOKA_OPENROUTER_ZDR", default="true") or "").strip().lower()
    if zdr not in _FALSE:
        routing["zdr"] = True
    return routing or None


def _reasoning_off(model: str) -> bool:
    return (model or "").lower().startswith(REASONING_OFF_MODEL_PREFIXES)


def openrouter_extra_body(agent: str, model: str) -> dict:
    """The request-body additions for a client built outside the factory.

    The Groq primary's OpenRouter backup and the visualization agent's own
    OpenRouter client send prompts too, so they carry the same pinning.
    """
    extra: dict = {}
    routing = _openrouter_routing(agent, model)
    if routing:
        extra["provider"] = routing
    if _reasoning_off(model):
        extra["reasoning"] = {"effort": "none"}
    return extra


_OPENROUTER_CLASSES: dict = {}


def _openrouter_chat_class(*, reasoning_off: bool, primary: bool):
    """The ChatOpenAI subclass every OpenRouter client is built from.

    * It classifies its own failures (app/core/model_tiers.py): a 402, 404, 408, 429
      or 5xx, or a transport error, is raised as TierUnavailable, an OSError, which
      with_fallbacks hands to the next tier. 402 has no openai-SDK class of its own,
      so this cannot be done by listing SDK classes. A primary starts a fresh record.
    * With *reasoning_off* (GPT-6 Luna) it always sends reasoning effort "none" and
      never reasoning_effort: Luna calls functions on Chat Completions only with
      reasoning off, and the planner passes reasoning_effort at invoke time, which
      would override a constructor setting, so the payload is fixed where it is built.

    The class keeps the name ChatOpenAI: callers and tests identify the client by it.
    """
    key = (reasoning_off, primary)
    if key in _OPENROUTER_CLASSES:
        return _OPENROUTER_CLASSES[key]
    from langchain_openai import ChatOpenAI as _Base

    from app.core import model_tiers

    def _failed(self, exc):
        failure = model_tiers.classify(exc, model=self.model_name, provider=OPENROUTER)
        if failure is None:
            return None
        model_tiers.record(failure)
        return failure

    class ChatOpenAI(_Base):
        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            if reasoning_off:
                payload.pop("reasoning_effort", None)
                extra = dict(payload.get("extra_body") or {})
                extra["reasoning"] = {"effort": "none"}
                payload["extra_body"] = extra
            return payload

        def _generate(self, *args, **kwargs):
            if primary:
                model_tiers.begin()
            try:
                return super()._generate(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised, classified or as is
                failure = _failed(self, exc)
                if failure is None:
                    raise
                raise failure from exc

        async def _agenerate(self, *args, **kwargs):
            if primary:
                model_tiers.begin()
            try:
                return await super()._agenerate(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                failure = _failed(self, exc)
                if failure is None:
                    raise
                raise failure from exc

    _OPENROUTER_CLASSES[key] = ChatOpenAI
    return ChatOpenAI


def _build_openai_compatible(*, provider, model, temperature, agent: str = "",
                             reasoning_effort: Optional[str] = None,
                             fallback: bool = False):
    """Local vLLM/Ollama, hosted OpenAI, and OpenRouter — all OpenAI-spec."""
    from langchain_openai import ChatOpenAI

    if provider == LOCAL:
        base_url = _env_first(
            "INFERENCE_LOCAL_BASE_URL",
            "OPENAI_API_BASE",
            default="http://avaloka-local-llm:11434/v1",
        )
        # Local servers ignore the key but the OpenAI client requires a
        # non-empty string; default to a harmless placeholder.
        api_key = _env_first("INFERENCE_LOCAL_API_KEY", "OPENAI_API_KEY", default="sk-local")
    elif provider == OPENROUTER:
        base_url = _env_first(
            "INFERENCE_OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1"
        )
        api_key = _env_first("OPENROUTER_API_KEY", "INFERENCE_OPENROUTER_API_KEY")
        if not api_key:
            logger.warning("OpenRouter selected but OPENROUTER_API_KEY is not set — LLM disabled.")
            return None
    else:  # OPENAI
        base_url = _env_first("INFERENCE_OPENAI_BASE_URL", "OPENAI_API_BASE")
        api_key = _env_first("OPENAI_API_KEY", "INFERENCE_OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI provider selected but OPENAI_API_KEY is not set — LLM disabled.")
            return None

    if not model:
        logger.warning("%s selected but no model resolved — LLM disabled.", provider)
        return None

    kwargs = dict(model=model, temperature=temperature, api_key=api_key, base_url=base_url)
    if provider == OPENROUTER:
        extra_body: dict = {}
        routing = _openrouter_routing(agent, model, fallback=fallback)
        if routing:
            extra_body["provider"] = routing
        # Same reasoning effort the Groq path sends. Without it OpenRouter used
        # the model default, and gpt-oss spent most of its output on hidden
        # reasoning (validator: 648 of 780 tokens).
        if reasoning_effort is not None and supports_reasoning(model):
            extra_body["reasoning"] = {"effort": reasoning_effort}
        if _reasoning_off(model):
            extra_body["reasoning"] = {"effort": "none"}
        if extra_body:
            kwargs["extra_body"] = extra_body
        return _openrouter_chat_class(reasoning_off=_reasoning_off(model), primary=not fallback)(**kwargs)
    return ChatOpenAI(**kwargs)


def _build_bedrock(*, model, temperature):
    region = _env_first("INFERENCE_BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")
    if not region:
        logger.warning("Bedrock selected but no AWS region is set — LLM disabled.")
        return None
    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError:
        logger.warning(
            "Bedrock selected but langchain-aws is not installed — "
            "`pip install langchain-aws`. LLM disabled."
        )
        return None
    # Credentials come from the standard AWS chain (env, profile, IRSA).
    return ChatBedrockConverse(model=model, region_name=region, temperature=temperature)


def _build_vertex(*, model, temperature):
    project = _env_first("INFERENCE_VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT")
    location = _env_first("INFERENCE_VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION", default="us-central1")
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError:
        logger.warning(
            "Vertex selected but langchain-google-vertexai is not installed — "
            "`pip install langchain-google-vertexai`. LLM disabled."
        )
        return None
    # Credentials come from Application Default Credentials / Workload Identity.
    return ChatVertexAI(model=model, project=project, location=location, temperature=temperature)


def _build_azure(*, agent, tier, temperature):
    from langchain_openai import AzureChatOpenAI

    endpoint = _env_first("AZURE_OPENAI_ENDPOINT", "INFERENCE_AZURE_ENDPOINT")
    api_key = _env_first("AZURE_OPENAI_API_KEY", "INFERENCE_AZURE_API_KEY")
    api_version = _env_first("AZURE_OPENAI_API_VERSION", default="2024-10-21")
    tier_up = "SMALL" if str(tier).lower() == "small" else "LARGE"
    deployment = _env_first(
        f"AVALOKA_{agent.upper()}_AZURE_DEPLOYMENT",
        f"AZURE_OPENAI_DEPLOYMENT_{tier_up}",
        "AZURE_OPENAI_DEPLOYMENT",
    )
    if not (endpoint and api_key and deployment):
        logger.warning(
            "Azure selected but AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / "
            "deployment are not all set — LLM disabled."
        )
        return None
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        api_key=api_key,
        temperature=temperature,
    )


# --------------------------------------------------------------------------
# Public factory
# --------------------------------------------------------------------------
def _with_fallback(llm, *, model: str, temperature: float, agent: str,
                   provider: Optional[str] = None):
    """Attach the OpenRouter backup to whatever provider produced *llm*.

    Applied here rather than in each agent because this is the single place every
    agent's model is constructed. Before the provider abstraction existed, each
    agent wrapped its own ChatGroq; that pattern does not survive a factory, and
    re-adding it per agent would mean thirteen call sites drifting apart again.

    ``attach_fallback`` returns its argument unchanged when no backup is
    configured, so this is safe on every path including the ones that return
    None.
    """
    if llm is None:
        return None
    if provider == OPENROUTER:
        # The Groq-era backup IS OpenRouter, so it is never attached here: a 429
        # or an outage would be retried against the provider that just produced
        # it. The OpenRouter primary gets the next TIER instead (owner decision
        # 2026-11-01): gpt-oss-120b -> GPT-6 Luna, each pinned to its own
        # provider allowlist. With allow_fallbacks false, "no listed provider can
        # serve this" arrives as HTTP 404 or 503 and lands here; with the tier
        # switched off (AVALOKA_OPENROUTER_FALLBACK_MODEL=none) it reaches the caller.
        return _with_openrouter_tier(llm, model=model, temperature=temperature, agent=agent)
    try:
        from app.core.model_fallback import attach_fallback
    except Exception:  # noqa: BLE001
        return llm
    return attach_fallback(llm, model, temperature=temperature, agent=(agent or "").lower())


def _instrument(llm, *, agent: str, provider: str, model: str):
    """Record each call's provider, model, tokens, cost and latency on the
    agent-trace step that made it (app/trace). A no-op outside a graph node, and
    never fatal: a model that cannot take a callback is returned as built."""
    if llm is None:
        return None
    try:
        from app.trace.recorder import instrument
    except Exception:  # noqa: BLE001
        return llm
    return instrument(llm, agent=(agent or "").lower(), provider=provider, model=model)


#: The last tier, outside OpenRouter (owner ruling 2026-11-02): it answers when
#: OpenRouter itself cannot -- out of credit (402), an account-level 429, an outage.
DEFAULT_GROQ_LAST_TIER_MODEL = "openai/gpt-oss-120b"


def groq_last_tier_model() -> Optional[str]:
    """The Groq model at the end of the OpenRouter chain, or None when switched off."""
    if (os.getenv("AVALOKA_DISABLE_MODEL_FALLBACK") or "").strip().lower() in {"1", "true", "yes"}:
        return None
    raw = (_env_first("AVALOKA_GROQ_FALLBACK_MODEL", default=DEFAULT_GROQ_LAST_TIER_MODEL) or "").strip()
    return None if raw.lower() in ("", "none", "off") else raw


def _groq_tier_key() -> Optional[str]:
    return _env_first("GROQ_API_KEY_PLANNING_AGENT", "GROQ_API_KEY", "GROQ_API_KEY_CODING_AGENT")


def openrouter_chain(model: str) -> list:
    """The tiers behind an OpenRouter primary *model*, in order: (provider, model, armed).

    ``armed`` is False when the tier is configured but cannot be built (no Groq key):
    the model screen shows it that way rather than hiding it.
    """
    chain = []
    luna = openrouter_fallback_model()
    if luna and luna != model:
        chain.append((OPENROUTER, luna, bool(_env_first("OPENROUTER_API_KEY",
                                                        "INFERENCE_OPENROUTER_API_KEY"))))
    groq = groq_last_tier_model()
    if groq:
        chain.append((GROQ, groq, bool(_groq_tier_key())))
    return chain


def _instrument_tier(llm, *, agent: str, provider: str, model: str, configured_model: str):
    try:
        from app.trace.recorder import instrument
    except Exception:  # noqa: BLE001
        return llm
    return instrument(llm, agent=(agent or "").lower(), provider=provider, model=model,
                      configured_model=configured_model, primary_provider=OPENROUTER)


def _with_openrouter_tier(llm, *, model: str, temperature: float, agent: str):
    """*llm* with the fallback tiers behind it, or *llm* unchanged.

    gpt-oss-120b (OpenRouter, pinned) -> GPT-6 Luna (OpenRouter, pinned) -> gpt-oss-120b
    (Groq). Each tier that answers in place of *model* stamps the reply with which
    model answered and why (app/core/model_tiers.py), so the swap is never silent.
    With no tier armed, the primary's error -- a TierUnavailable naming the reason --
    reaches the caller.
    """
    try:
        from app.core.model_fallback import _TierCompat, openrouter_failover_exception_types
    except Exception:  # noqa: BLE001
        return llm
    tiers = []
    for provider, tier_model, armed in openrouter_chain(model):
        if not armed:
            continue
        if provider == OPENROUTER:
            backup = _build_openai_compatible(provider=OPENROUTER, model=tier_model,
                                              temperature=temperature, agent=agent, fallback=True)
        else:
            backup = _build_groq(model=tier_model, temperature=temperature, api_key=_groq_tier_key(),
                                 reasoning_effort=None, reasoning_format=None)
        if backup is not None:
            # On the agent trace like the primary: a failover shows as the failed
            # call, then the tier that answered and why (app/trace/recorder.py).
            backup = _instrument_tier(backup, agent=agent, provider=provider, model=tier_model,
                                      configured_model=model)
            tiers.append(_TierCompat(backup, tier_model, primary_model=model,
                                     primary_provider=OPENROUTER, provider=provider))
    if not tiers:
        return llm
    logger.info("[models] %s: fallback tiers armed (%s -> %s)", agent or model, model,
                " -> ".join(f"{t.provider}/{t.model}" for t in tiers))
    from app.core import model_tiers

    return model_tiers.chain(llm, tiers, openrouter_failover_exception_types())


def describe_llm_setup() -> dict:
    """What the model screen shows: each role's provider and model, and the chain behind it.

    No credentials: only whether a tier is armed (its key is present). The answering
    model of a given reply is reported with the reply itself (ChatResponse.answered_by).
    """
    roles = []
    for role, agent in (("planning", "PLANNER"), ("coding", "CODER"), ("viz", "VIZ")):
        provider = resolve_provider(role=role, agent=agent)
        if provider == GROQ:
            model = os.getenv(f"AVALOKA_{agent}_MODEL") or "openai/gpt-oss-120b"
        elif provider == AZURE:
            model = "azure-deployment"
        else:
            model = _resolve_model(provider, agent, "large")
        chain = [{"provider": provider, "model": model, "armed": True,
                  "providers_only": openrouter_allowlist(model, agent=agent) if provider == OPENROUTER else None}]
        if provider == OPENROUTER:
            for tier_provider, tier_model, armed in openrouter_chain(model):
                chain.append({"provider": tier_provider, "model": tier_model, "armed": armed,
                              "providers_only": (openrouter_allowlist(tier_model, fallback=True)
                                                 if tier_provider == OPENROUTER else None)})
        routing = None
        if provider == OPENROUTER:
            routing = {k: v for k, v in (_openrouter_routing(agent, model) or {}).items() if k != "only"}
        roles.append({"role": role, "agent": agent, "provider": provider, "model": model,
                      "chain": chain, "routing": routing})
    return {"roles": roles,
            "note": "Each reply states which model answered it when a fallback tier stood in."}


def build_chat_model(
    *,
    role: str,
    agent: str,
    groq_model: str,
    groq_api_key: Optional[str],
    tier: str = "large",
    temperature: float = 0.2,
    env_model_var: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    reasoning_format: Optional[str] = None,
):
    """
    Build an agent's chat model for the active inference provider.

    Parameters
    ----------
    role:
        Coarse role used for per-role provider selection: one of
        ``"planning"``, ``"coding"``, ``"viz"``.
    agent:
        Short uppercase id used in env-var names and logs (e.g. ``"PLANNER"``,
        ``"CODER"``, ``"MTA"``).
    groq_model:
        The Groq-native model id the agent used pre-abstraction. Used verbatim
        when the provider is Groq; ignored otherwise (see :func:`_resolve_model`).
    groq_api_key:
        The agent's Groq key. Only consulted when the provider is Groq.
    tier:
        ``"large"`` | ``"small"`` — maps to per-provider model tiers for the
        non-Groq providers so one config covers every agent.
    temperature:
        Sampling temperature (unchanged across providers).
    env_model_var:
        Legacy Groq model override env var (e.g. ``DTA_CODER_MODEL``); defaults
        to ``AVALOKA_<AGENT>_MODEL``. Groq path only.
    reasoning_effort / reasoning_format:
        gpt-oss reasoning controls. Attached only on the Groq path for a
        gpt-oss model; ``reasoning_effort=None`` means "non-reasoning agent".

    Returns
    -------
    A LangChain ``BaseChatModel`` (``.invoke`` / ``.bind_tools`` compatible),
    or ``None`` when the selected provider is not configured — in which case
    the caller keeps its existing deterministic fallback.
    """
    # The caller's Groq key counts as evidence when no provider is configured:
    # every agent passes one, and ignoring it meant a Groq-only deployment
    # silently resolved to OpenRouter and got no LLM at all.
    provider = resolve_provider(role=role, agent=agent, groq_api_key=groq_api_key)

    if provider == GROQ:
        env_var = env_model_var or f"AVALOKA_{agent.upper()}_MODEL"
        model = os.getenv(env_var, groq_model)
        llm = _build_groq(
            model=model,
            temperature=temperature,
            api_key=groq_api_key,
            reasoning_effort=reasoning_effort,
            reasoning_format=reasoning_format,
        )
        if llm is not None:
            logger.info(
                "%s LLM: groq/%s (reasoning=%s)",
                agent,
                model,
                reasoning_effort if (reasoning_effort and supports_reasoning(model)) else "off",
            )
        llm = _instrument(llm, agent=agent, provider=provider, model=model)
        return _with_fallback(llm, model=model, temperature=temperature, agent=agent, provider=provider)

    # ---- non-Groq providers -------------------------------------------------
    if provider == AZURE:
        llm = _build_azure(agent=agent, tier=tier, temperature=temperature)
        model_desc = "azure-deployment"
    else:
        model = _resolve_model(provider, agent, tier)
        model_desc = model
        if provider in (LOCAL, OPENAI, OPENROUTER):
            llm = _build_openai_compatible(
                provider=provider, model=model, temperature=temperature,
                agent=agent, reasoning_effort=reasoning_effort,
            )
        elif provider == BEDROCK:
            llm = _build_bedrock(model=model, temperature=temperature)
        elif provider == VERTEX:
            llm = _build_vertex(model=model, temperature=temperature)
        else:
            logger.warning("Unknown inference provider %r for %s — LLM disabled.", provider, agent)
            return None

    if llm is not None:
        if provider == OPENROUTER:
            routing = _openrouter_routing(agent, model_desc)
            effort = reasoning_effort if (reasoning_effort and supports_reasoning(model_desc)) else "default"
            logger.info("%s LLM: %s/%s (routing=%s, reasoning=%s)",
                        agent, provider, model_desc, routing or "openrouter-default", effort)
        else:
            logger.info("%s LLM: %s/%s", agent, provider, model_desc)
    # model_desc, not model: the Azure branch never assigns `model` because Azure
    # addresses a deployment name rather than a model id, and referencing it here
    # raised UnboundLocalError on that path alone.
    llm = _instrument(llm, agent=agent, provider=provider, model=model_desc)
    return _with_fallback(llm, model=model_desc, temperature=temperature, agent=agent, provider=provider)