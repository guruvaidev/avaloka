"""Which model answered, and why, when a model tier failed over.

Owner ruling 2026-11-02: a fallback is never silent. When the configured model
cannot answer and another tier does, the answer carries a record of it, which the
API and the UI show beside the reply:

    {"configured_model": "openai/gpt-4o", "answered_model": "openai/gpt-6-luna",
     "provider": "openrouter", "reason": "no allowed zero-retention provider serves
     openai/gpt-4o", "note": "... ; answered by openai/gpt-6-luna"}

The chain for the OpenRouter default (app/core/inference.py ``_with_openrouter_tier``):
gpt-oss-120b on OpenRouter -> GPT-6 Luna on OpenRouter -> gpt-oss-120b on Groq.

How a tier hands over: an OpenRouter client classifies its own error with
:func:`classify` and raises :class:`TierUnavailable`, an ``OSError``, so LangChain's
``with_fallbacks`` (which filters by exception CLASS) moves on for exactly the
statuses below. A 402 has no openai-SDK class of its own, so filtering by the SDK's
classes alone could not include it. 400/401/403/422 are not here: a malformed or
unauthorised request fails the same way on every tier.

Each failure is recorded in a context variable, and the tier that answers stamps the
result with the failures recorded for ITS primary model only, so a stale failure from
another call can never become this reply's reason.
"""
from __future__ import annotations

import contextvars
import logging
import threading
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Statuses after which the next tier is asked.
#:   402 OpenRouter credit exhausted
#:   404 no allowed provider serves the model (measured 2026-11-02), or model gone
#:   408 timeout · 429 rate limited (key or account) · 5xx provider trouble
FAILOVER_STATUSES = frozenset({402, 404, 408, 429, 500, 502, 503, 504})

#: Where the record lives on an AIMessage.
ANSWERED_BY_KEY = "avaloka_answered_by"

#: The failures of the chain now running. A chain (see :func:`chain`) binds a fresh
#: list here on entry and unbinds it in a ``finally``. LangChain runs each runnable
#: of a with_fallbacks chain inside a COPY of the context -- and, on the async path,
#: in a new asyncio task -- so a value REBOUND by the primary would never reach the
#: tier that answers. The copies do share the list object bound at entry, so every
#: tier appends to and reads the same record, while concurrent chains, each in its
#: own context, have their own. Outside a chain nothing is recorded, so nothing is
#: retained between requests.
_CHAIN: contextvars.ContextVar[Optional[List["TierUnavailable"]]] = contextvars.ContextVar(
    "avaloka_tier_chain", default=None)
_LIVE = 0
_LIVE_LOCK = threading.Lock()


def live_chains() -> int:
    """How many chains are running now; 0 when idle (no record outlives its chain)."""
    with _LIVE_LOCK:
        return _LIVE


def current() -> Optional[List["TierUnavailable"]]:
    """The record of the chain running in this context, or None outside a chain."""
    return _CHAIN.get()


def _enter():
    global _LIVE
    with _LIVE_LOCK:
        _LIVE += 1
    return _CHAIN.set([])


def _exit(token) -> None:
    global _LIVE
    _CHAIN.reset(token)
    with _LIVE_LOCK:
        _LIVE -= 1


def chain(primary: Any, tiers: list, exceptions: tuple) -> Any:
    """*primary* with *tiers* behind it, each call running with its own failure record.

    Equivalent to ``primary.with_fallbacks(tiers, exceptions_to_handle=exceptions)``;
    the class keeps the name RunnableWithFallbacks, which callers and tests check.
    """
    from langchain_core.runnables.fallbacks import RunnableWithFallbacks as _Base

    class RunnableWithFallbacks(_Base):
        def invoke(self, input, config=None, **kwargs):  # noqa: A002
            token = _enter()
            try:
                return super().invoke(input, config, **kwargs)
            finally:
                _exit(token)

        async def ainvoke(self, input, config=None, **kwargs):  # noqa: A002
            token = _enter()
            try:
                return await super().ainvoke(input, config, **kwargs)
            finally:
                _exit(token)

    # Build through primary.with_fallbacks so any override of it (and its
    # argument handling) is kept exactly as before; only LangChain's own
    # RunnableWithFallbacks is re-made as the record-binding subclass.
    built = primary.with_fallbacks(list(tiers), exceptions_to_handle=exceptions)
    if not isinstance(built, _Base):
        return built
    return RunnableWithFallbacks(runnable=built.runnable, fallbacks=list(built.fallbacks),
                                 exceptions_to_handle=built.exceptions_to_handle,
                                 exception_key=built.exception_key)


class TierUnavailable(OSError):
    """A model tier could not answer for a reason the next tier may not share."""

    def __init__(self, *, model: str, provider: str, status: Optional[int], detail: str = ""):
        self.model = model
        self.provider = provider
        self.status = status
        self.detail = (detail or "")[:300]
        self.reason = reason_for(status, self.detail, model=model, provider=provider)
        super().__init__(f"{model} via {provider} could not answer: {self.reason}")


def reason_for(status: Optional[int], detail: str, *, model: str, provider: str) -> str:
    """The user-facing reason, from the recorded status and message only."""
    low = (detail or "").lower()
    if status == 404 and "no allowed providers" in low:
        return f"no allowed zero-retention provider serves {model}"
    if status == 404:
        return f"{model} is not available from {provider} (HTTP 404)"
    if status == 402:
        return f"the {provider} account is out of credit (HTTP 402)"
    if status == 429:
        return f"{model} is rate-limited on {provider} (HTTP 429)"
    if status == 408:
        return f"{provider} timed out (HTTP 408)"
    if isinstance(status, int) and status >= 500:
        return f"{provider} is unavailable (HTTP {status})"
    return f"{provider} could not be reached"


def classify(exc: BaseException, *, model: str, provider: str) -> Optional[TierUnavailable]:
    """A :class:`TierUnavailable` for a failure the next tier may not share, else None."""
    if isinstance(exc, TierUnavailable):
        return exc
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status not in FAILOVER_STATUSES:
            return None
        body = getattr(exc, "body", None)
        detail = ""
        if isinstance(body, dict):
            err = body.get("error") if isinstance(body.get("error"), dict) else body
            detail = str(err.get("message") or "") if isinstance(err, dict) else ""
        return TierUnavailable(model=model, provider=provider, status=status,
                               detail=detail or str(exc))
    # Transport failures: the SDKs' connection and timeout errors, and plain OSError.
    if type(exc).__name__ in ("APIConnectionError", "APITimeoutError") or isinstance(exc, OSError):
        return TierUnavailable(model=model, provider=provider, status=None, detail=str(exc))
    return None


def begin() -> None:
    """A primary is starting: forget failures from an earlier attempt in this chain."""
    record_ = _CHAIN.get()
    if record_ is not None:
        record_.clear()


def record(failure: TierUnavailable) -> None:
    record_ = _CHAIN.get()
    if record_ is not None:
        record_.append(failure)
    logger.warning("[models] %s via %s could not answer: %s", failure.model, failure.provider,
                   failure.reason)


def failures_for(model: str, provider: str) -> Tuple[TierUnavailable, ...]:
    """The failures recorded in this chain, provided its primary was *model* on *provider*."""
    recorded = tuple(_CHAIN.get() or ())
    if recorded and recorded[0].model == model and recorded[0].provider == provider:
        return recorded
    return ()


def stamp(result: Any, *, configured_model: str, answered_model: str, provider: str,
          primary_provider: str = "openrouter") -> Any:
    """Attach the record to *result* (an AIMessage) and log it. Returns *result*."""
    failed = failures_for(configured_model, primary_provider)
    reason = failed[0].reason if failed else f"{configured_model} could not answer"
    record_ = {
        "configured_model": configured_model,
        "answered_model": answered_model,
        "provider": provider,
        "reason": reason,
        "status": failed[0].status if failed else None,
        "tiers_failed": [{"model": f.model, "provider": f.provider, "status": f.status,
                          "reason": f.reason} for f in failed],
    }
    record_["note"] = f"{reason}; answered by {answered_model}"
    meta = getattr(result, "response_metadata", None)
    if isinstance(meta, dict):
        meta[ANSWERED_BY_KEY] = record_
    logger.warning("[models] answered by a fallback tier: %s", record_["note"])
    begin()
    return result


def answered_by(response: Any) -> Optional[dict]:
    """The record on *response*, or None when the configured model answered."""
    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, dict) and isinstance(meta.get(ANSWERED_BY_KEY), dict):
        return dict(meta[ANSWERED_BY_KEY])
    return None
