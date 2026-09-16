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

Point 3 matters right now. Every agent builds a ``ChatGroq`` directly, so
**only Groq-hosted models work today**. Configuring an OpenAI model such as
``gpt-5`` is not a small change — it needs the provider-routing work in the
custom inference stack. Until that lands, :func:`check_model_supported` turns
that mistake into a clear error at startup rather than a cryptic failure later.
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

#: Prefixes Groq hosts. Groq serves open-weight models plus its own Compound
#: agentic systems; it does not host OpenAI's hosted GPT line, Anthropic, or
#: Gemini. ``openai/gpt-oss-*`` IS on Groq — it is the open-weight release, not
#: the hosted GPT API — which is exactly the kind of confusion this list exists
#: to settle.
GROQ_SERVEABLE_PREFIXES: Tuple[str, ...] = (
    "llama-",            # llama-3.1-*, llama-3.3-*
    "meta-llama/",       # meta-llama/llama-4-*
    "openai/gpt-oss",    # open-weight gpt-oss, NOT the hosted GPT API
    "compound",          # compound-beta, compound-beta-mini
    "groq/compound",     # newer Compound naming
    "qwen",
    "deepseek-r1",
    "mixtral-",
    "gemma",
    "whisper-",
)

#: Prefixes that are definitely NOT on Groq, with the provider that serves them.
#: Used to produce an actionable error rather than "model not found".
_FOREIGN_MODEL_PROVIDERS: Dict[str, str] = {
    "gpt-5": "openai",
    "gpt-4": "openai",
    "o1": "openai",
    "o3": "openai",
    "claude-": "anthropic",
    "gemini-": "vertex",
}


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

    # Validation. Compound is an agentic system rather than a bare model, which
    # suits a gate that has to reason about whether code matches an intent.
    # This replaces llama-3.1-8b-instant, which no longer exists on this Groq
    # account at all (verified: HTTP 404). Even when it did, an 8B model
    # deciding whether generated code is correct was a false-confidence risk,
    # and this gate is the last thing between a bad plan and an executed query.
    "validator":   "groq/compound",

    # Plan -> JSON contract. Every downstream agent trusts this output, so a
    # dropped constraint here propagates silently. 20b is sufficient for
    # schema-constrained extraction and leaves 120b capacity for the Coder.
    "summarizer":  "openai/gpt-oss-20b",

    "profiling":     "openai/gpt-oss-120b",
    "visualization": "openai/gpt-oss-20b",
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
    raise UnsupportedModelError(
        f"{agent}: model {model!r} is served by {provider!r}, not Groq, and every "
        f"agent currently builds a ChatGroq client directly.\n"
        f"Routing an agent to a non-Groq provider needs the provider-agnostic "
        f"inference stack (app/core/inference.build_chat_model). Until that is "
        f"merged, set {env_var_for(agent)} to a Groq-hosted model — see "
        f"GROQ_SERVEABLE_PREFIXES in app/core/model_config.py."
    )


def resolve(agent: str, *, backend: str = "groq") -> str:
    """Resolve and validate an agent's model. Returns the model id."""
    choice = model_for(agent)
    check_model_supported(agent, choice.model, backend=backend)
    logger.info("[models] %s", choice)
    return choice.model


def describe_all(*, backend: str = "groq") -> Dict[str, Dict[str, object]]:
    """Every agent's resolved model and whether the backend can serve it.

    Useful for a diagnostics endpoint and for support requests: one call
    answers "what is actually running where".
    """
    out: Dict[str, Dict[str, object]] = {}
    for agent in sorted(DEFAULT_MODELS):
        choice = model_for(agent)
        needs = provider_for_model(choice.model)
        out[agent] = {
            "model": choice.model,
            "source": "default" if choice.is_default else choice.env_var,
            "serveable_by_backend": needs is None or backend != "groq",
            "requires_provider": needs,
        }
    return out
