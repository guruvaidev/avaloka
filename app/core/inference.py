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
* **Groq stays the default.** When ``INFERENCE_PROVIDER`` is unset or
  ``groq``, :func:`build_chat_model` returns a ``ChatGroq`` configured
  exactly as the agents configured it before this abstraction existed —
  including the gpt-oss reasoning kwargs — so the planner and every other
  agent behave byte-for-byte identically. Nothing about planner/agent logic
  changes; only the *backend* moves when you ask it to.
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
Global switch (default ``groq``)::

    INFERENCE_PROVIDER=local        # laptop / kind: OpenAI-spec model in-cluster
    INFERENCE_PROVIDER=groq         # Groq Cloud (default)
    INFERENCE_PROVIDER=openrouter   # OpenRouter
    INFERENCE_PROVIDER=bedrock      # AWS Bedrock
    INFERENCE_PROVIDER=vertex       # GCP Vertex AI
    INFERENCE_PROVIDER=azure        # Azure AI / Azure OpenAI

Per-role override (``PLANNING`` | ``CODING`` | ``VIZ``)::

    INFERENCE_PROVIDER_PLANNING=groq
    INFERENCE_PROVIDER_CODING=local

Per-agent override (highest precedence), e.g. the planner::

    AVALOKA_PLANNER_PROVIDER=local
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

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
    # fallback alone changed nothing.
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
    """OpenRouter when it has a key, else a provider that does.

    Only consulted when no provider was configured at all. Order is the
    preferred default first, then the providers whose keys are a single env var
    to check; anything needing cloud credentials is out of scope here and must
    be selected explicitly.
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
    access to, which is the better default for an open-source install.

    Groq remains fully supported and is still the fastest path: set
    INFERENCE_PROVIDER=groq, or AVALOKA_<AGENT>_PROVIDER=groq per agent.
    """
    key = (raw or "").strip().lower()
    if not key:
        # Nothing was configured, so this is the implicit default. Prefer
        # OpenRouter, but only if it can actually be used: selecting a provider
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
def resolve_provider(*, role: str, agent: str, groq_api_key: Optional[str] = None) -> str:
    """
    Resolve the active provider for an agent, most specific wins:

    1. ``AVALOKA_<AGENT>_PROVIDER``            (per-agent)
    2. ``INFERENCE_PROVIDER_<ROLE>``           (per-role: PLANNING/CODING/VIZ)
    3. ``INFERENCE_PROVIDER``                  (global)
    4. OpenRouter if it has a key, else a provider that does
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
_PROVIDER_MODEL_DEFAULTS = {
    # Laptop/kind default is the small model for BOTH tiers so a CPU-only
    # dev box never tries to serve a 14B model. Point _LARGE at the GPU-served
    # model (e.g. qwen2.5-14b/32b) when you run the vLLM GPU deployment.
    LOCAL: {"large": "qwen2.5:3b-instruct", "small": "qwen2.5:3b-instruct"},
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


def _build_openai_compatible(*, provider, model, temperature):
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
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key, base_url=base_url)


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
        # The backup IS OpenRouter. Wrapping an OpenRouter primary in an
        # OpenRouter fallback buys nothing and costs something: a 429 or an
        # outage is retried against the provider that just produced it, and the
        # wrapper changes the returned type, so callers that introspect the
        # model see RunnableWithFallbacks instead of the chat model. This could
        # not happen while Groq was the default primary.
        return llm
    try:
        from app.core.model_fallback import attach_fallback
    except Exception:  # noqa: BLE001
        return llm
    return attach_fallback(llm, model, temperature=temperature, agent=(agent or "").lower())


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
        return _with_fallback(llm, model=model, temperature=temperature, agent=agent, provider=provider)

    # ---- non-Groq providers -------------------------------------------------
    if provider == AZURE:
        llm = _build_azure(agent=agent, tier=tier, temperature=temperature)
        model_desc = "azure-deployment"
    else:
        model = _resolve_model(provider, agent, tier)
        model_desc = model
        if provider in (LOCAL, OPENAI, OPENROUTER):
            llm = _build_openai_compatible(provider=provider, model=model, temperature=temperature)
        elif provider == BEDROCK:
            llm = _build_bedrock(model=model, temperature=temperature)
        elif provider == VERTEX:
            llm = _build_vertex(model=model, temperature=temperature)
        else:
            logger.warning("Unknown inference provider %r for %s — LLM disabled.", provider, agent)
            return None

    if llm is not None:
        logger.info("%s LLM: %s/%s", agent, provider, model_desc)
    # model_desc, not model: the Azure branch never assigns `model` because Azure
    # addresses a deployment name rather than a model id, and referencing it here
    # raised UnboundLocalError on that path alone.
    return _with_fallback(llm, model=model_desc, temperature=temperature, agent=agent, provider=provider)
