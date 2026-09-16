"""OpenRouter as an optional backup when the primary provider fails.

On 2026-08-17 Groq removed the entire Llama line from this project's account.
Six agents kept calling ``llama-3.3-70b-versatile`` and got HTTP 404 — not at
deploy time, but mid-analysis with a user waiting. Pinning verified model ids
fixes today's outage; it does nothing for the next deprecation, a rate limit,
or a regional outage.

This module adds a second place to ask. It is deliberately **opt-in**: without
``OPENROUTER_API_KEY`` set, behaviour is exactly as before.

Two things make this more than a try/except:

**Model ids are not portable.** ``openai/gpt-oss-120b`` and ``gpt-oss-20b``
exist under the *same id* on both providers, so those fail over unchanged. But
``groq/compound`` is Groq-proprietary — OpenRouter answers
``"groq/compound is not a valid model ID"``. The Validator therefore needs an
explicit substitute, not an identity mapping. :data:`OPENROUTER_EQUIVALENTS`
records that, verified by live probe.

**Not every failure should fail over.** A 404 or 5xx means "ask someone else".
A 401 means the key is wrong and OpenRouter will not fix it. A 400 means the
request is malformed and will be malformed there too. Retrying those wastes
money and latency to reach the same failure, so :func:`should_failover` is
explicit about which is which.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.runnables.base import Runnable

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

#: What OpenRouter should be asked when a Groq model is unavailable.
#:
#: Verified against the live OpenRouter catalogue on 2026-08-26. Identity
#: mappings are real, not assumed: gpt-oss-120b and gpt-oss-20b carry the same
#: id on both providers and returned HTTP 200 there.
#:
#: groq/compound is the interesting entry. It is an agentic system rather than
#: a bare model and has no OpenRouter counterpart, so the Validator falls back
#: to a strong general model instead. qwen/qwen3.8-27b was chosen for this gate:
#: the Validator parses the reply as raw JSON (no bind_tools), so instruction
#: following is the whole job, and its 1M context comfortably holds a review
#: prompt carrying the plan, the generated code and a data preview. It also
#: supports OpenAI-compatible tool-calling/structured-output requests should the
#: fallback path ever send them. It costs more per token than the alternatives
#: (0.425/2.55 per M vs 0.0875/0.35 for qwen3-235b-a22b-2507) -- an accepted
#: trade on a path that only runs when Groq is unavailable. The fallback still
#: differs from Compound's agentic runtime, but it is NOT classified as a
#: capability downgrade for this Validator.
OPENROUTER_EQUIVALENTS: Dict[str, str] = {
    "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b": "openai/gpt-oss-safeguard-20b",
    "groq/compound": "qwen/qwen3.8-27b",
    "groq/compound-mini": "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b": "qwen/qwen3.8-27b",
}

#: Substitutions that genuinely lose a capability required by the target
#: agent. The Validator's qwen/qwen3.8-27b fallback is intentionally
#: NOT listed here because the Validator currently consumes raw JSON and does
#: not depend on Compound's agentic tool runtime.
DEGRADED_SUBSTITUTIONS = set()

#: HTTP statuses worth asking a second provider about.
#:   404 model gone (exactly what happened)
#:   429 rate limited
#:   5xx provider trouble
FAILOVER_STATUSES = frozenset({404, 408, 429, 500, 502, 503, 504})

#: Statuses where a second provider would fail identically.
#:   400 malformed request · 401/403 credentials · 422 invalid payload
NO_FAILOVER_STATUSES = frozenset({400, 401, 403, 422})


class NoBackupConfigured(RuntimeError):
    """A fallback was requested but OpenRouter is not configured."""


class BackupFailed(OSError):
    """The backup provider failed too, so the primary's error should surface.

    Subclassing OSError is load-bearing rather than decorative: it puts this in
    :func:`failover_exception_types`, which makes LangChain's ``with_fallbacks``
    count the backup attempt as *handled* and re-raise the error from the
    PRIMARY runnable. Without it a secondary OpenRouter 400 would mask the Groq
    429 that caused the failover — the wrong error for the operator to read, and
    the wrong shape for callers whose ``except`` blocks were written against the
    Groq exception.
    """


@dataclass(frozen=True)
class FallbackPlan:
    """What to try instead, and what it costs in capability."""

    primary_model: str
    backup_model: str
    degraded: bool
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "primary_model": self.primary_model,
            "backup_model": self.backup_model,
            "degraded": self.degraded,
            "reason": self.reason,
        }


def openrouter_key() -> Optional[str]:
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    return key or None


def backup_enabled() -> bool:
    """True when a backup is both configured and not explicitly disabled."""
    if (os.getenv("AVALOKA_DISABLE_MODEL_FALLBACK") or "").strip().lower() in {"1", "true", "yes"}:
        return False
    return openrouter_key() is not None


def should_failover(status: Optional[int] = None, *, error: Optional[BaseException] = None) -> bool:
    """Whether this failure is worth retrying against the backup provider.

    Unknown statuses do **not** fail over. Defaulting to "try again elsewhere"
    on an unrecognised code spends money and latency to reach the same failure.
    """
    if status is not None:
        if status in NO_FAILOVER_STATUSES:
            return False
        return status in FAILOVER_STATUSES
    if error is not None:
        # An SDK exception already carries the status the caller would have had
        # to dig out by hand (groq.RateLimitError.status_code == 429). Reading
        # it here means one predicate governs both entry points, instead of
        # `should_failover(error=exc)` quietly answering False for a 429.
        exc_status = getattr(error, "status_code", None)
        if isinstance(exc_status, int):
            return should_failover(exc_status)
        # Transport-level failures (DNS, connection reset, timeout) are exactly
        # the case a second provider can answer.
        #
        # Checked by type, not by type *name*: TimeoutError, ConnectionError,
        # socket.error and ssl.SSLError are all OSError subclasses, so one
        # isinstance covers them. Matching on the name misses plain OSError
        # entirely, which is how a real "socket down" would arrive.
        return isinstance(error, OSError)
    return False


def equivalent_for(model: str) -> Optional[str]:
    """The OpenRouter model standing in for *model*, or None if unmapped."""
    return OPENROUTER_EQUIVALENTS.get(model)


def plan_fallback(model: str, *, reason: str = "primary provider unavailable",
                  quiet: bool = False) -> FallbackPlan:
    """Decide what to ask OpenRouter instead. Raises when no backup exists."""
    if not backup_enabled():
        raise NoBackupConfigured(
            "OpenRouter backup is not configured; set OPENROUTER_API_KEY to enable it "
            "(or unset AVALOKA_DISABLE_MODEL_FALLBACK if it is switched off)."
        )
    backup = equivalent_for(model)
    if backup is None:
        raise NoBackupConfigured(
            f"no OpenRouter equivalent recorded for {model!r}; add one to "
            "OPENROUTER_EQUIVALENTS in app/core/model_fallback.py"
        )
    degraded = model in DEGRADED_SUBSTITUTIONS
    plan = FallbackPlan(
        primary_model=model, backup_model=backup, degraded=degraded, reason=reason,
    )
    if degraded:
        logger.warning(
            "[models] falling back %s -> %s on OpenRouter (%s). This is a CAPABILITY "
            "DOWNGRADE: %s is an agentic system with tool use that the substitute "
            "does not provide.", model, backup, reason, model,
        )
    elif not quiet:
        # quiet=True is for callers that are only *resolving* the plan, not
        # acting on it. attach_fallback resolves one at startup to arm the
        # backup; logging "falling back" there would read, in the logs, as an
        # outage that never happened. The degraded warning above is not
        # silenced — knowing your backup is a downgrade is most useful before
        # you need it.
        logger.info("[models] falling back %s -> %s on OpenRouter (%s)",
                    model, backup, reason)
    return plan


def build_backup_chat_model(model: str, *, temperature: float = 0.0):
    """Construct a LangChain chat model pointed at OpenRouter.

    OpenRouter speaks the OpenAI wire protocol, so ChatOpenAI with a custom
    base_url is the whole integration. Returns None when unavailable rather
    than raising, so a caller can degrade to its existing "LLM disabled" path.
    """
    key = openrouter_key()
    if not key:
        return None
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        logger.warning("[models] langchain-openai is not installed; "
                       "OpenRouter backup unavailable")
        return None
    return ChatOpenAI(
        model=model, temperature=temperature,
        api_key=key, base_url=OPENROUTER_BASE_URL,
    )


# --------------------------------------------------------------------------- #
# Attaching the backup to a live agent
# --------------------------------------------------------------------------- #

#: Groq SDK exception classes whose HTTP status sits in FAILOVER_STATUSES.
#: Resolved by name at call time because the SDK is an optional import here and
#: because a name that disappears in an SDK upgrade should cost us that one
#: exception, not an ImportError at module load.
_FAILOVER_EXCEPTION_NAMES: Tuple[str, ...] = (
    "RateLimitError",       # 429 — the org-wide TPM ceiling
    "NotFoundError",        # 404 — model withdrawn from the account
    "InternalServerError",  # 5xx — provider trouble
    "APIConnectionError",   # transport; APITimeoutError (408) subclasses this
)

#: Invoke-time kwargs that mean something to Groq and nothing to OpenRouter.
#: Forwarded unchanged they become an unrecognised field in the payload, which
#: is a 400 — i.e. the fallback would fail for a reason the primary never had.
_GROQ_ONLY_INVOKE_KWARGS = frozenset({"reasoning_format", "service_tier"})

#: Kwargs only the gpt-oss family understands. Unlike the set above these are
#: dropped CONDITIONALLY: when the backup is the same gpt-oss model on another
#: host it must keep them, and when the backup is a different family — a local
#: Gemma or Qwen — it cannot accept them at all.
_REASONING_ONLY_INVOKE_KWARGS = frozenset({"reasoning_effort"})

#: Duplicated from app.core.agent_llm.REASONING_MODEL_PREFIXES rather than
#: imported: agent_llm imports attach_fallback from this module, so importing it
#: back would be a cycle. One tuple of one prefix is the cheaper of the two evils.
_REASONING_MODEL_PREFIXES = ("openai/gpt-oss",)

#: Backups that reason, but under OpenRouter's field name rather than gpt-oss's.
#: Verified against the live OpenRouter catalogue: these list "reasoning" in
#: supported_parameters. Membership here means reasoning_effort is TRANSLATED
#: rather than dropped -- see _adapt. A model absent from both this set and
#: _REASONING_MODEL_PREFIXES still has the kwarg dropped, which is the safe
#: default: an unrecognised body field is a 400 the primary never had.
_OPENROUTER_REASONING_MODELS = frozenset({"qwen/qwen3.8-27b"})


def _accepts_reasoning_kwargs(model: str) -> bool:
    """True when *model* understands gpt-oss reasoning kwargs."""
    return (model or "").lower().startswith(_REASONING_MODEL_PREFIXES)


def failover_exception_types() -> Tuple[type, ...]:
    """Exception classes worth retrying against the backup provider.

    The type-level mirror of :func:`should_failover`, shaped for LangChain's
    ``with_fallbacks(exceptions_to_handle=...)`` which filters by class rather
    than by predicate. Note what is deliberately absent: ``BadRequestError``,
    ``AuthenticationError``, ``PermissionDeniedError`` and
    ``UnprocessableEntityError`` are the NO_FAILOVER_STATUSES, and would fail
    identically on OpenRouter.
    """
    types: List[type] = [OSError]
    try:
        import groq
    except ImportError:  # pragma: no cover - groq is a hard dependency today
        return tuple(types)
    for name in _FAILOVER_EXCEPTION_NAMES:
        candidate = getattr(groq, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            types.append(candidate)
    return tuple(types)


def normalise_tools_for_openrouter(tools: Any) -> Any:
    """Move each tool's ``description`` inside its ``function`` object.

    The Planner declares tools with ``description`` as a sibling of
    ``function`` rather than a member of it. Groq accepts that shape, so it has
    never been worth changing — and changing it now would alter what the
    primary model sees, which is a behavioural change nobody asked for in a
    fallback patch.

    OpenRouter validates the payload against the OpenAI function schema, where
    ``description`` belongs to the function. Rewriting it here, on the backup
    path only, keeps the Groq request byte-identical while giving OpenRouter
    something it will accept — and incidentally lets the backup model actually
    read the tool descriptions.
    """
    if not isinstance(tools, list):
        return tools
    fixed = []
    for tool in tools:
        if not isinstance(tool, dict) or "description" not in tool:
            fixed.append(tool)
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            fixed.append(tool)
            continue
        moved = {k: v for k, v in tool.items() if k != "description"}
        moved["function"] = {"description": tool["description"], **fn}
        fixed.append(moved)
    return fixed


class _OpenRouterCompat(Runnable):
    """A backup chat model that accepts requests shaped for Groq.

    ``with_fallbacks`` hands the fallback the *same* kwargs the primary was
    called with, which is the right default and wrong here: those kwargs are a
    Groq payload. This adapter is the seam where they become an OpenRouter one.
    """

    def __init__(self, inner: Any, model: str) -> None:
        self.inner = inner
        self.model = model

    def _adapt(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        drop = set(_GROQ_ONLY_INVOKE_KWARGS)
        translate_reasoning = False
        if not _accepts_reasoning_kwargs(self.model):
            # The caller chose these by looking at its PRIMARY model. Whether the
            # backup can use them depends on the backup:
            #   * same gpt-oss id on another host -> keep, it speaks gpt-oss
            #   * a model that reasons under OpenRouter's own field -> translate
            #   * anything else -> drop, or the unknown field is a 400
            leaking = sorted(_REASONING_ONLY_INVOKE_KWARGS & kwargs.keys())
            translate_reasoning = bool(leaking) and self.model in _OPENROUTER_REASONING_MODELS
            if leaking and not translate_reasoning:
                logger.info(
                    "[models] dropping %s for backup %s: gpt-oss reasoning kwargs "
                    "the caller selected from its primary model", leaking, self.model,
                )
            drop |= _REASONING_ONLY_INVOKE_KWARGS
        adapted = {k: v for k, v in kwargs.items() if k not in drop}
        if translate_reasoning:
            # OpenRouter's unified spelling is reasoning={"effort": ...}; the
            # gpt-oss spelling reasoning_effort=... is not a field it knows.
            effort = kwargs.get("reasoning_effort")
            adapted["reasoning"] = {"effort": effort}
            logger.info(
                "[models] translating reasoning_effort=%r to OpenRouter "
                "reasoning={'effort': %r} for backup %s", effort, effort, self.model,
            )
        if "tools" in adapted:
            adapted["tools"] = normalise_tools_for_openrouter(adapted["tools"])
        return adapted

    def _failed(self, exc: BaseException) -> BackupFailed:
        logger.error("[models] OpenRouter backup %s failed too (%s: %s); "
                     "surfacing the primary provider's error instead",
                     self.model, type(exc).__name__, exc)
        return BackupFailed(f"OpenRouter backup {self.model} failed: {exc}")

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        logger.warning("[models] primary provider failed; answering from "
                       "OpenRouter backup %s", self.model)
        try:
            return self.inner.invoke(input, config, **self._adapt(kwargs))
        except BaseException as exc:
            raise self._failed(exc) from exc

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        logger.warning("[models] primary provider failed; answering from "
                       "OpenRouter backup %s (async)", self.model)
        try:
            return await self.inner.ainvoke(input, config, **self._adapt(kwargs))
        except BaseException as exc:
            raise self._failed(exc) from exc


def attach_fallback(primary: Any, model: str, *, temperature: float = 0.0,
                    agent: str = "") -> Any:
    """Return *primary* with an OpenRouter backup behind it.

    Wrapping the model rather than each call site means every invocation an
    agent makes — bare ``.invoke``, a ``prompt | llm`` chain, a ``.bind()``
    handle — inherits the fallback from one edit.

    Returns *primary* unchanged whenever a backup is unavailable for any
    reason, so this is safe to call unconditionally: with no
    ``OPENROUTER_API_KEY`` set, behaviour is exactly as before.

    When the backup fails too, ``with_fallbacks`` re-raises the *primary's*
    exception, so callers' existing error handling still sees the Groq error it
    was written against.
    """
    label = agent or model
    if primary is None or not backup_enabled():
        return primary
    try:
        plan = plan_fallback(model, reason=f"{label} startup", quiet=True)
    except NoBackupConfigured as exc:
        logger.warning("[models] %s has no OpenRouter backup (%s); "
                       "running without one", label, exc)
        return primary
    backup = build_backup_chat_model(plan.backup_model, temperature=temperature)
    if backup is None:
        logger.warning("[models] %s backup could not be constructed; "
                       "running without one", label)
        return primary
    logger.info("[models] %s: OpenRouter backup armed (%s -> %s)",
                label, plan.primary_model, plan.backup_model)
    return primary.with_fallbacks(
        [_OpenRouterCompat(backup, plan.backup_model)],
        exceptions_to_handle=failover_exception_types(),
    )


def describe_backup() -> Dict[str, object]:
    """Diagnostics: is a backup configured, and what would it substitute?"""
    return {
        "enabled": backup_enabled(),
        "configured": openrouter_key() is not None,
        "base_url": OPENROUTER_BASE_URL,
        "equivalents": dict(OPENROUTER_EQUIVALENTS),
        "degraded_substitutions": sorted(DEGRADED_SUBSTITUTIONS),
    }


# --------------------------------------------------------------------------- #
# Local fallback tier — profile-aware (laptop vs cluster)
# --------------------------------------------------------------------------- #
#
# The OpenRouter tier above still needs the internet and a paid key. The last
# resort is a locally served model over the OpenAI wire protocol — the vLLM
# (GPU) and Ollama (CPU) deployments shipped in deploy/inference/.
#
# One size does not fit both places. A cluster GPU can serve a 27–31B model; a
# laptop cannot, and asking it to is how a "fallback" becomes a hang. So the
# local tier is sized by deployment profile, the profile is detected rather
# than assumed, and the choice is stated in the logs — a silent size decision
# is exactly the kind of thing that gets debugged for an afternoon.
#
# Model ids verified against the live OpenRouter catalogue on 2026-08-22
# (google/gemma-4-31b-it, google/gemma-4-26b-a4b-it, qwen/qwen3.8-27b,
# google/gemma-3-4b-it). The ids your vLLM/Ollama actually serves may differ —
# every default below is env-overridable, and the serving manifest decides
# what exists.

#: In-cluster local endpoint (vLLM/Ollama are both OpenAI-spec).
LOCAL_BASE_URL = os.getenv("AVALOKA_LOCAL_LLM_URL", "http://avaloka-local-llm:8000/v1")

#: Cluster-profile local models: a GPU deployment can hold a real model.
#: gemma-4-31b is the strongest open-weight fit; qwen3.8-27b the alternate.
LOCAL_MODEL_CLUSTER = os.getenv("AVALOKA_LOCAL_MODEL_CLUSTER", "google/gemma-4-31b-it")
LOCAL_MODEL_CLUSTER_ALT = os.getenv("AVALOKA_LOCAL_MODEL_CLUSTER_ALT", "qwen/qwen3.8-27b")

#: Laptop-profile local model: small enough to serve on CPU/metal without
#: making the fallback slower than the outage it papers over.
LOCAL_MODEL_LAPTOP = os.getenv("AVALOKA_LOCAL_MODEL_LAPTOP", "google/gemma-3-4b-it")


def deployment_profile() -> str:
    """``laptop`` or ``cluster``, explicit env beating detection.

    Detection: KUBERNETES_SERVICE_HOST is injected into every pod by the
    kubelet, so its presence is the standard in-cluster signal. Anything else
    is treated as a laptop — the safe direction, since sending a 31B model to
    a laptop fails worse than sending a 4B model to a cluster.
    """
    explicit = (os.getenv("AVALOKA_DEPLOYMENT_PROFILE") or "").strip().lower()
    if explicit in {"laptop", "cluster"}:
        return explicit
    return "cluster" if os.getenv("KUBERNETES_SERVICE_HOST") else "laptop"


def local_fallback_model(*, profile: Optional[str] = None) -> str:
    """The local model this profile should fall back to, logged with why."""
    resolved = profile or deployment_profile()
    if resolved == "cluster":
        model = LOCAL_MODEL_CLUSTER
        logger.info(
            "[models] local fallback profile=cluster -> %s (alt: %s) at %s",
            model, LOCAL_MODEL_CLUSTER_ALT, LOCAL_BASE_URL)
    else:
        model = LOCAL_MODEL_LAPTOP
        logger.info(
            "[models] local fallback profile=laptop -> %s (small model chosen "
            "deliberately: a laptop serving a 27B+ model turns a fallback into "
            "a hang) at %s", model, LOCAL_BASE_URL)
    return model


def build_local_chat_model(*, temperature: float = 0.0,
                           profile: Optional[str] = None):
    """Chat model against the local vLLM/Ollama endpoint, or None if unusable."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        logger.warning("[models] langchain-openai missing; local fallback unavailable")
        return None
    return ChatOpenAI(model=local_fallback_model(profile=profile),
                      temperature=temperature,
                      api_key=os.getenv("AVALOKA_LOCAL_LLM_KEY", "not-needed"),
                      base_url=LOCAL_BASE_URL)


def fallback_chain(model: str) -> list:
    """Ordered fallback candidates for *model*: OpenRouter first, local last.

    Each entry is ``(tier, model_id, degraded)``. OpenRouter comes first when
    configured because it preserves model class; the local tier survives with
    no internet at all but is a capability step down by construction, and the
    laptop profile a further one — both marked degraded so the caller logs the
    downgrade rather than silently absorbing it.
    """
    chain = []
    if backup_enabled():
        backup = equivalent_for(model)
        if backup:
            chain.append(("openrouter", backup, model in DEGRADED_SUBSTITUTIONS))
    profile = deployment_profile()
    chain.append(("local", local_fallback_model(profile=profile), True))
    return chain
