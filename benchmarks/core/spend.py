"""Pre-run cost estimate and a hard spend ceiling.

The model matrix multiplies out: ``models x tasks x trials``. With
``anthropic/claude-opus-5`` at $25/1M completion tokens in the roster, a
carelessly-launched full sweep is real money, and the failure mode is the worst
kind — you find out from the invoice, after the fact, with nothing to roll back.

Two mechanisms, deliberately different in kind:

**Estimate (before)** — :func:`estimate_suite_cost` prices the sweep *before the
first call*, from the actual prompt text (so the input side is measured, not
guessed) and the configured ``max_tokens`` (so the output side is a true
ceiling, not a hope). It reports a band, and the top of that band is a number
the provider genuinely cannot exceed for that config.

**Ceiling (during)** — :class:`SpendLedger` accumulates observed spend as
trials complete and raises :class:`SpendCeilingExceeded` as soon as it crosses
the limit.

Two honest limits on that ceiling, because overstating it is worse than not
having it:

1. It charges *after* the call, so it is exceeded by at most one call's cost
   before it fires. Bounded by ``max_tokens``, so cents at this roster's prices.
2. **It cannot bound a row it cannot price.** An ``unpriced`` model (no catalog
   quote) or a ``usage_missing`` response (provider returned no token counts)
   charges $0.00 and is therefore invisible to both the estimate and the
   ceiling. Those calls are counted separately and reported, so the gap is
   labelled rather than hidden — but a sweep made entirely of unpriced rows is
   genuinely unbounded, and :meth:`SpendLedger.blind_calls` is how you find out.

Why both: an estimate alone is advisory and an unpriced model (see
:mod:`benchmarks.core.pricing`) estimates at zero, which is exactly the row that
would blow the budget unnoticed. A ceiling alone gives you no chance to decline
before spending anything.

Estimates use a documented, crude tokenizer proxy (4 chars/token). That is
deliberate: a real tokenizer differs per model and would make the estimate look
more precise than the ``max_tokens`` ceiling it is multiplied by. The estimate
is a decision aid with a stated band, not a quote.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import RunConfig
from .pricing import (ModelPrice, SOURCE_UNPRICED, SOURCE_USAGE_MISSING,
                      price_for)
from .task import BenchmarkTask

#: Characters per token. Crude on purpose — see the module docstring.
CHARS_PER_TOKEN = 4.0

#: Fraction of ``max_tokens`` a typical answer actually uses. Used only for the
#: low end of the band; the high end assumes the full allowance.
TYPICAL_COMPLETION_FRACTION = 0.35

#: Default ceiling when the caller does not set one. Deliberately equal to the
#: account's whole daily allowance, so the harness default can never plan to
#: spend more than a day's budget even before the live check below runs.
DEFAULT_MAX_SPEND_USD = 5.0

#: OpenRouter's key-metadata endpoint. Free — it is account metadata, not
#: inference — and it is the ONLY authority on what is actually left today.
OPENROUTER_KEY_URL = os.getenv("OPENROUTER_KEY_URL",
                               "https://openrouter.ai/api/v1/key")


@dataclass(frozen=True)
class RemainingBudget:
    """What the provider says is left. ``None`` fields mean "not reported"."""

    limit_usd: Optional[float]
    usage_usd: Optional[float]
    remaining_usd: Optional[float]
    reset_period: Optional[str]
    source: str                 # "live" | "unavailable"
    detail: str = ""

    @property
    def known(self) -> bool:
        return self.source == "live" and self.remaining_usd is not None


def fetch_remaining_budget(url: str = OPENROUTER_KEY_URL,
                           timeout: float = 15.0) -> RemainingBudget:
    """Ask OpenRouter what is left on the key today.

    Assuming a budget is how you exhaust one. The account this harness runs
    against is capped at $5/day, and once it is gone every remaining config
    fails — which, without the scorecard guards in
    :mod:`benchmarks.core.reporting`, publishes as "the frontier models score
    zero".

    Never raises and never logs the key: a budget check that can itself break
    the run is a worse problem than the one it solves. An unavailable answer is
    returned as ``source="unavailable"`` so the caller can decide whether to
    proceed, rather than silently proceeding on a guess.
    """
    import json
    import urllib.error
    import urllib.request

    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        return RemainingBudget(None, None, None, None, "unavailable",
                               "OPENROUTER_API_KEY is not set")
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}",
                      "User-Agent": "avaloka-bench/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        # Deliberately does not interpolate the key or the response body.
        return RemainingBudget(None, None, None, None, "unavailable",
                               f"{type(exc).__name__} querying the key endpoint")

    data = payload.get("data") or payload
    limit = data.get("limit")
    usage = data.get("usage")
    remaining = data.get("limit_remaining")
    if remaining is None and limit is not None and usage is not None:
        remaining = float(limit) - float(usage)
    return RemainingBudget(
        limit_usd=None if limit is None else float(limit),
        usage_usd=None if usage is None else float(usage),
        remaining_usd=None if remaining is None else float(remaining),
        reset_period=data.get("limit_reset"),
        source="live",
    )


class SpendCeilingExceeded(RuntimeError):
    """Observed spend crossed the configured ceiling; the run was stopped."""

    def __init__(self, spent_usd: float, ceiling_usd: float):
        self.spent_usd = spent_usd
        self.ceiling_usd = ceiling_usd
        super().__init__(
            f"spend ceiling exceeded: ${spent_usd:.4f} spent against a "
            f"${ceiling_usd:.2f} ceiling — run halted with partial results. "
            f"Raise it with --max-spend-usd if this was intended."
        )


def estimate_tokens(text: str) -> int:
    """Crude token count for a prompt we are about to send. Never negative."""
    return max(0, int(len(text or "") / CHARS_PER_TOKEN))


@dataclass
class ConfigEstimate:
    """Projected spend for one config over the whole task set."""

    config_id: str
    provider: str
    model: str
    price: ModelPrice
    n_tasks: int
    trials: int
    prompt_tokens: int          # summed across every call this config will make
    max_completion_tokens: int  # ditto, at the configured ceiling
    low_usd: float
    high_usd: float

    @property
    def calls(self) -> int:
        return self.n_tasks * self.trials

    @property
    def unpriced(self) -> bool:
        return self.price.source == SOURCE_UNPRICED

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "provider": self.provider,
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "low_usd": self.low_usd,
            "high_usd": self.high_usd,
            "cost_source": self.price.source,
        }


@dataclass
class SuiteEstimate:
    """The whole sweep, plus the honest caveats attached to it."""

    configs: List[ConfigEstimate] = field(default_factory=list)

    @property
    def low_usd(self) -> float:
        return sum(c.low_usd for c in self.configs)

    @property
    def high_usd(self) -> float:
        return sum(c.high_usd for c in self.configs)

    @property
    def calls(self) -> int:
        return sum(c.calls for c in self.configs)

    @property
    def unpriced_configs(self) -> List[str]:
        return [c.config_id for c in self.configs if c.unpriced]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_low_usd": self.low_usd,
            "total_high_usd": self.high_usd,
            "total_calls": self.calls,
            "unpriced_configs": self.unpriced_configs,
            "configs": [c.as_dict() for c in self.configs],
        }

    def render(self) -> str:
        """A table you read before saying yes."""
        lines = ["Estimated spend (before running):", ""]
        lines.append("| Config | Model | Calls | $ low | $ high | price source |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for c in sorted(self.configs, key=lambda x: -x.high_usd):
            lines.append(f"| {c.config_id} | `{c.model}` | {c.calls} | "
                         f"${c.low_usd:.4f} | ${c.high_usd:.4f} | {c.price.source} |")
        lines.append("")
        lines.append(f"**Total: ${self.low_usd:.4f} – ${self.high_usd:.4f} "
                     f"over {self.calls} calls.**")
        lines.append("")
        lines.append("The high figure assumes every answer uses its full "
                     "`max_tokens` allowance, so it is a ceiling the provider "
                     "cannot exceed for these configs — not a forecast.")
        if self.unpriced_configs:
            lines.append("")
            lines.append("> ⚠️  UNPRICED, and therefore **excluded from both "
                         "totals**: " + ", ".join(f"`{c}`" for c in self.unpriced_configs)
                         + ". No catalog quote exists for these models, so their "
                           "real spend is unknown — it is not zero.")
        return "\n".join(lines)


def _prompt_text(task: BenchmarkTask, config: RunConfig) -> str:
    """Reconstruct what the adapter will actually send, so the input side is measured.

    Imported lazily to keep this module usable when the adapter package is not
    loaded; falls back to mission+evidence if the template moves.
    """
    import json as _json

    evidence = _json.dumps(task.params.get("evidence", {}), sort_keys=True)
    try:
        from ..adapters.llm_direct import SYSTEM_PROMPT
        system = SYSTEM_PROMPT.format(evidence=evidence)
    except Exception:  # noqa: BLE001 - estimate must never be the thing that breaks
        system = evidence
    return f"{system}\n{task.mission}"


def estimate_suite_cost(tasks: Sequence[BenchmarkTask],
                        configs: Iterable[RunConfig]) -> SuiteEstimate:
    """Price the sweep before the first API call."""
    estimate = SuiteEstimate()
    tasks = list(tasks)
    for config in configs:
        provider = str(config.extra.get("provider", "groq"))
        model = str(config.extra.get("model") or config.llm)
        price = price_for(provider, str(config.extra.get("price_model") or model))
        max_tokens = int(config.extra.get("max_tokens", 400))

        prompt_tokens = sum(estimate_tokens(_prompt_text(t, config))
                            for t in tasks) * config.trials
        max_completion = max_tokens * len(tasks) * config.trials

        high = price.cost_usd(prompt_tokens, max_completion)
        low = price.cost_usd(
            prompt_tokens, int(max_completion * TYPICAL_COMPLETION_FRACTION))
        estimate.configs.append(ConfigEstimate(
            config_id=config.id, provider=provider, model=model, price=price,
            n_tasks=len(tasks), trials=config.trials,
            prompt_tokens=prompt_tokens, max_completion_tokens=max_completion,
            low_usd=low, high_usd=high,
        ))
    return estimate


@dataclass
class SpendLedger:
    """Running total with a hard stop.

    The runner charges every completed trial here. Crossing the ceiling raises
    rather than warns: a warning in a log nobody is watching is how you find out
    from the invoice.
    """

    ceiling_usd: Optional[float] = DEFAULT_MAX_SPEND_USD
    spent_usd: float = 0.0
    calls: int = 0
    unpriced_calls: int = 0
    usage_missing_calls: int = 0

    #: Sources whose $0.00 means "we could not price this", not "this was free".
    #: ``unknown`` is included: it is MissionOutcome's default, so an adapter
    #: that never set the field must not be counted as a priced zero.
    BLIND_SOURCES = (SOURCE_UNPRICED, SOURCE_USAGE_MISSING, "unknown", "")

    @property
    def blind_calls(self) -> int:
        """Calls the ceiling could not see. Non-zero means it is not a bound."""
        return self.unpriced_calls + self.usage_missing_calls

    def charge(self, cost_usd: float, *, cost_source: str = "") -> None:
        self.calls += 1
        if cost_source == SOURCE_UNPRICED or cost_source in ("unknown", ""):
            self.unpriced_calls += 1
        elif cost_source == SOURCE_USAGE_MISSING:
            self.usage_missing_calls += 1
        self.spent_usd += max(0.0, float(cost_usd or 0.0))
        if self.ceiling_usd is not None and self.spent_usd > self.ceiling_usd:
            raise SpendCeilingExceeded(self.spent_usd, self.ceiling_usd)

    @property
    def remaining_usd(self) -> Optional[float]:
        if self.ceiling_usd is None:
            return None
        return max(0.0, self.ceiling_usd - self.spent_usd)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "spent_usd": self.spent_usd,
            "ceiling_usd": self.ceiling_usd,
            "calls": self.calls,
            "unpriced_calls": self.unpriced_calls,
            "usage_missing_calls": self.usage_missing_calls,
            "blind_calls": self.blind_calls,
            "remaining_usd": self.remaining_usd,
        }
