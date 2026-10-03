"""One place that decides which model each agent uses.

Model choice was scattered across hardcoded literals: ``coder.py:27``,
``validator.py:15``, ``summarizer.py:26``, and so on. Changing the model an
agent uses meant editing that agent, which is why the Validator — the gate that
decides whether generated code is sound — quietly ran on an 8B model for
months.

Three things this fixes:

1. **Every agent's model is env-overridable**, so a model change is
   configuration rather than a code edit and review cycle.
2. **The defaults live together**, so "what runs where" is one table instead of
   a grep across nine files.
3. **A model the current backend cannot serve fails loudly and early**, with a
   message that says what to do, instead of a 404 from the provider halfway
   through an analysis.

Point 3 matters right now. Most agents build a ``ChatGroq`` directly, so for
them **only Groq-hosted models work today**. The exception is any agent in
``_ROUTABLE_AGENTS`` (currently ``visualization``): set
``AVALOKA_<AGENT>_BACKEND`` to ``openrouter`` or ``local`` to route it through
an OpenAI-compatible client instead. Moving every agent onto a non-Groq
provider still needs the provider-routing work in the custom inference stack.
Until that lands, :func:`check_model_supported` turns a Groq-only agent
configured with a foreign model into a clear error at startup.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class UnsupportedModelError(RuntimeError):
    """A configured model cannot be served by the active backend."""


# --------------------------------------------------------------------------- #
# Which models a Groq-only backend can actually serve
# --------------------------------------------------------------------------- #

#: Prefixes Groq hosts: open-weight models (Llama, gpt-oss, Qwen, etc.). Groq
#: does not host OpenAI's hosted GPT line, Anthropic, or Gemini. ``openai/gpt-oss-*``
#: IS on Groq — it is the open-weight release, not the hosted GPT API — which is
#: exactly the kind of confusion this list exists to settle. (Groq's Compound
#: agentic systems were decommissioned 2026-09-21 and are no longer listed here.)
GROQ_SERVEABLE_PREFIXES: Tuple[str, ...] = (
    "llama-",            # llama-3.1-*, llama-3.3-*
    "meta-llama/",       # meta-llama/llama-4-*
    "openai/gpt-oss",
    "qwen",
    "deepseek-r1",
    "mixtral-",
    "gemma",
    "whisper-",
)


_FOREIGN_MODEL_PROVIDERS: Dict[str, str] = {
    "gpt-5": "openai",
    "gpt-4": "openai",
    "o1": "openai",
    "o3": "openai",
    "claude-": "anthropic",
    "gemini-": "vertex",
    # Retired Groq models — kept here so a stale config fails loudly at startup
    # with an actionable message instead of a provider 404 mid-run.
    "compound": "groq (decommissioned 2026-09-21)",
    "groq/compound": "groq (decommissioned 2026-09-21)",
}


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

SUPPORTED_BACKENDS: Tuple[str, ...] = ("groq", "openrouter", "local")

#: Agents whose construction actually honours a non-Groq backend. Every other
#: agent still builds ChatGroq directly, so letting them "pass" a backend check
#: would be a lie: the request would silently go to Groq anyway.
_ROUTABLE_AGENTS = {"visualization"}


@dataclass(frozen=True)
class ModelChoice:
    """The model an agent will use, and where it came from."""

    agent: str
    model: str
    env_var: str
    is_default: bool

    def __str__(self) -> str:
        origin = "default" if self.is_default else f"env {self.env_var}"
        return f"{self.agent}={self.model} ({origin})"


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
#
# Verify these against your Groq console before relying on them. Groq
# deprecates and renames models, and a default that 404s is worse than one that
# is merely conservative. Every entry is overridable by the env var beside it,
# so a rename never requires a code change.

DEFAULT_MODELS: Dict[str, str] = {
    # Deliberation. gpt-oss-120b is open-weight and Groq-hosted, and supports
    # the reasoning-effort controls planner.py already sends.
    "planner":     "openai/gpt-oss-120b",

    # Code generation. gpt-oss-120b is the largest general model on this
    # account and supports reasoning effort, which suits pseudocode-then-code
    # authoring. The Coder's output is gated by the Validator, so raw
    # capability matters more than latency.
    "coder":       "openai/gpt-oss-120b",

    # Validation. The gate between a bad plan and an executed query, so it gets
    # the full 120b. Replaces llama-3.1-8b-instant (now 404 on this account); an
    # 8B model judging generated code was a false-confidence risk.
    "validator":   "openai/gpt-oss-120b",

    # Plan -> JSON contract. Every downstream agent trusts this output, so a
    # dropped constraint here propagates silently. 20b is sufficient for
    # schema-constrained extraction and leaves 120b capacity for the Coder.
    "summarizer":  "openai/gpt-oss-20b",

    "profiling":     "openai/gpt-oss-120b",

    # Auto Insights + the voice explainer: the most user-visible output in
    # Avaloka, so it gets the strongest model on the account, not the smallest.
    # Routable: AVALOKA_VISUALIZATION_BACKEND=openrouter|local to test others.
    "visualization": "openai/gpt-oss-120b",

    "dta_coder":     "openai/gpt-oss-120b",
    "dta_validator": "openai/gpt-oss-120b",
    "mta":           "openai/gpt-oss-120b",
    "mta_task_builder": "openai/gpt-oss-120b",
    "memory":           "openai/gpt-oss-20b",

    # Conversational agent. gpt-oss-120b supports the reasoning-effort controls
    # the agent already sends, which is the point of a conversational surface
    # that has to deliberate rather than pattern-match.
    "conversational":   "openai/gpt-oss-120b",
}

#: Legacy env vars that must keep working.
_LEGACY_ENV: Dict[str, str] = {
    "planner": "AVALOKA_PLANNER_MODEL",
    "dta_coder": "DTA_CODER_MODEL",
}


def env_var_for(agent: str) -> str:
    return _LEGACY_ENV.get(agent, f"AVALOKA_{agent.upper()}_MODEL")


def model_for(agent: str) -> ModelChoice:
    """Resolve an agent's model from env, falling back to the default table."""
    key = agent.lower()
    if key not in DEFAULT_MODELS:
        raise KeyError(f"unknown agent {agent!r}; known: {sorted(DEFAULT_MODELS)}")
    var = env_var_for(key)
    override = (os.getenv(var) or "").strip()
    return ModelChoice(
        agent=key,
        model=override or DEFAULT_MODELS[key],
        env_var=var,
        is_default=not override,
    )


def provider_for_model(model: str) -> Optional[str]:
    """The non-Groq provider a model needs, or None if Groq can serve it."""
    lowered = (model or "").strip().lower()
    if not lowered:
        return None
    # Check Groq first: "openai/gpt-oss" starts with "openai" but IS on Groq,
    # so an unqualified "openai" prefix match would be wrong.
    if lowered.startswith(GROQ_SERVEABLE_PREFIXES):
        return None
    for prefix, provider in _FOREIGN_MODEL_PROVIDERS.items():
        if lowered.startswith(prefix):
            return provider
    return None


def check_model_supported(agent: str, model: str, *, backend: str = "groq") -> None:
    """Raise when *model* cannot be served by *backend*.

    Called at agent construction so a misconfiguration surfaces at startup with
    an actionable message, rather than as a provider 404 midway through a run
    the user is waiting on.
    """
    if backend != "groq":
        return
    provider = provider_for_model(model)
    if provider is None:
        return
    hint = (
        f" Or set AVALOKA_{agent.upper()}_BACKEND=openrouter (or local) — "
        f"{agent} supports non-Groq routing."
        if agent.lower() in _ROUTABLE_AGENTS else ""
    )
    raise UnsupportedModelError(
        f"{agent}: model {model!r} is served by {provider!r}, not Groq, and this "
        f"agent builds a ChatGroq client directly.\n"
        f"Routing an agent to a non-Groq provider needs the provider-agnostic "
        f"inference stack (app/core/inference.build_chat_model). Until that is "
        f"merged, set {env_var_for(agent)} to a Groq-hosted model — see "
        f"GROQ_SERVEABLE_PREFIXES in app/core/model_config.py.{hint}"
    )


def backend_for(agent: str) -> str:
    """Provider for *agent*: env AVALOKA_<AGENT>_BACKEND, default groq."""
    key = agent.lower()
    var = f"AVALOKA_{key.upper()}_BACKEND"
    value = (os.getenv(var) or "groq").strip().lower()
    if value not in SUPPORTED_BACKENDS:
        raise UnsupportedModelError(
            f"{agent}: {var}={value!r} is not one of {SUPPORTED_BACKENDS}."
        )
    if value != "groq" and key not in _ROUTABLE_AGENTS:
        logger.warning("[models] %s=%s ignored: %s can only run on Groq today.", var, value, key)
        return "groq"
    return value


def resolve(agent: str, *, backend: Optional[str] = None) -> str:
    """Resolve and validate an agent's model. Returns the model id."""
    backend = backend or backend_for(agent)
    choice = model_for(agent)
    check_model_supported(agent, choice.model, backend=backend)
    logger.info("[models] %s backend=%s", choice, backend)
    return choice.model


def describe_all(*, backend: Optional[str] = None) -> Dict[str, Dict[str, object]]:
    """Every agent's resolved model and backend, and whether it can be served.

    Useful for a diagnostics endpoint and for support requests: one call
    answers "what is actually running where".
    """
    out: Dict[str, Dict[str, object]] = {}
    for agent in sorted(DEFAULT_MODELS):
        choice = model_for(agent)
        be = backend or backend_for(agent)
        needs = provider_for_model(choice.model)
        out[agent] = {
            "model": choice.model,
            "backend": be,
            "source": "default" if choice.is_default else choice.env_var,
            "serveable_by_backend": needs is None or be != "groq",
            "requires_provider": needs,
        }
    return out