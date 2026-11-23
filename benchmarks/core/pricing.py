"""Model prices, sourced from the live provider catalog rather than typed by hand.

The harness reports a Cost column. A cost column populated from numbers a human
typed into a dict is a fabricated figure with extra steps: model slugs churn,
prices move, and nothing tells you when the constant you copied last month
stopped being true. Worse, a mistyped slug does not fail loudly — the provider
returns HTTP 400/404 *mid-run*, after you have already paid for every config
that ran before it. This project has been bitten by exactly that (see
``app/core/model_fallback.py``: on 2026-08-17 six agents kept calling a model id
Groq had removed, and found out with a user waiting).

So prices come from a **snapshot of the provider catalog**
(:data:`SNAPSHOT_PATH`), refreshed by ``scripts/refresh_openrouter_prices.py``,
and every model id used by a track is validated against that snapshot by a test.
A slug that does not exist cannot reach a paid API call.

Three price provenances, and the scorecard is told which one it got, because
they are not equally trustworthy:

``catalog``
    The model was quoted by the provider we are actually calling. Trust it.
``openrouter_proxy``
    We are calling Groq, Groq publishes no machine-readable catalog, but
    OpenRouter lists the *same open-weight model id* and we use that quote as a
    stand-in. Right order of magnitude, wrong vendor — usable for ranking, not
    for an invoice.
``local_zero``
    Locally served (Ollama). Marginal API spend genuinely is zero. Electricity
    and the GPU are real but are not this harness's to measure, so we say zero
    and say why.
``free_tier``
    A ``:free`` slug the catalog genuinely quotes at $0.00. Real, but *not*
    comparable: free tiers are rate-limited and served differently, so a free
    twin of a paid model wins any quality-per-dollar ranking by dividing by
    zero. Priced, and quarantined.
``unpriced``
    No quote exists (e.g. ``groq/compound`` is Groq-proprietary and absent from
    OpenRouter). Cost stays ``0.0`` and is flagged ``unpriced`` — a hole in the
    data, explicitly labelled, never a guess.

The rule the whole module exists to enforce: **an unknown price is reported as
unknown, never as zero-that-looks-like-free and never as an invented number.**
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping, Optional

#: Live catalog endpoint. Only ``scripts/refresh_openrouter_prices.py`` and the
#: opt-in freshness test hit this; normal runs read the snapshot.
OPENROUTER_MODELS_URL = os.getenv(
    "OPENROUTER_MODELS_URL", "https://openrouter.ai/api/v1/models"
)

#: NOTE the directory name. ``benchmarks/data/`` would be the obvious home, but
#: the repo's .gitignore ignores ``data/`` globally (line 134) — putting the
#: snapshot there means it is silently never committed, and the harness then
#: cannot price anything on a fresh clone. ``catalog/`` is deliberately outside
#: that rule.
SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "catalog", "openrouter_models.json")

#: Provenance tags, ordered from most to least trustworthy.
SOURCE_CATALOG = "catalog"
SOURCE_PROXY = "openrouter_proxy"
SOURCE_LOCAL = "local_zero"
SOURCE_UNPRICED = "unpriced"
SOURCE_FREE_TIER = "free_tier"
#: Priced by the operator's own per-token rates (the DAB command's
#: AVALOKA_DAB_*_USD_PER_MTOK) times provider-reported usage. A real cost the
#: ceiling can see, but from rates someone typed in rather than the catalog.
SOURCE_OPERATOR_RATE = "operator_rate"
#: The provider returned no ``usage`` block, so token counts are unknown and the
#: $0.00 that falls out of them is meaningless. Distinct from a priced zero.
SOURCE_USAGE_MISSING = "usage_missing"

#: Providers whose marginal per-token API cost is genuinely zero.
LOCAL_PROVIDERS = frozenset({"ollama"})

#: Providers we can quote directly from the OpenRouter catalog.
CATALOG_PROVIDERS = frozenset({"openrouter"})


class PriceLookupError(RuntimeError):
    """A model id that a track asked for does not exist in the catalog."""


@dataclass(frozen=True)
class ModelPrice:
    """A per-token quote, normalised to $/1M tokens for human legibility."""

    model_id: str
    prompt_usd_per_1m: float
    completion_usd_per_1m: float
    source: str
    context_length: Optional[int] = None
    #: OpenRouter's tiered long-prompt pricing, cheapest threshold first. 59 of
    #: 422 catalog models have one. Ignoring it under-reports a long-context
    #: call by up to 6x — and a coder benchmark carrying repo context crosses
    #: the lowest threshold (32k prompt tokens) routinely.
    tiers: tuple = ()

    @property
    def is_priced(self) -> bool:
        """Whether this quote can be divided into a quality score.

        ``free_tier`` is deliberately excluded: it is a real catalog price, but
        $0.00 makes every efficiency ratio infinite.
        """
        return self.source in (SOURCE_CATALOG, SOURCE_PROXY)

    def rates_for(self, prompt_tokens: int) -> tuple:
        """($/1M prompt, $/1M completion) at this prompt length.

        The highest threshold at or below ``prompt_tokens`` wins, matching
        OpenRouter's own semantics.
        """
        prompt_rate, completion_rate = self.prompt_usd_per_1m, self.completion_usd_per_1m
        for threshold, tier_prompt, tier_completion in self.tiers:
            if prompt_tokens >= threshold:
                prompt_rate, completion_rate = tier_prompt, tier_completion
        return prompt_rate, completion_rate

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """USD for one call. Unpriced models cost 0.0 *and say so* via ``source``."""
        if self.source in (SOURCE_UNPRICED, SOURCE_LOCAL, SOURCE_USAGE_MISSING):
            return 0.0
        prompt_rate, completion_rate = self.rates_for(prompt_tokens)
        return ((prompt_tokens / 1_000_000.0) * prompt_rate
                + (completion_tokens / 1_000_000.0) * completion_rate)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_usd_per_1m": self.prompt_usd_per_1m,
            "completion_usd_per_1m": self.completion_usd_per_1m,
            "cost_source": self.source,
            "context_length": self.context_length,
            "tiers": [{"min_prompt_tokens": t, "prompt_usd_per_1m": p,
                       "completion_usd_per_1m": c} for t, p, c in self.tiers],
        }


def _unpriced(model_id: str, source: str = SOURCE_UNPRICED) -> ModelPrice:
    return ModelPrice(model_id=model_id, prompt_usd_per_1m=0.0,
                      completion_usd_per_1m=0.0, source=source)


# --------------------------------------------------------------------------- #
# Catalog loading
# --------------------------------------------------------------------------- #

def parse_catalog(payload: Mapping[str, Any]) -> Dict[str, ModelPrice]:
    """``{"data": [...]}`` -> ``{model_id: ModelPrice}``.

    The API quotes USD **per token** as strings (``"0.0000001"``); we normalise
    to $/1M once, here, so no caller ever has to remember the factor. Entries
    without a parseable prompt/completion pair are dropped rather than
    defaulted — a model we cannot price must not look free.
    """
    table: Dict[str, ModelPrice] = {}
    for entry in payload.get("data") or []:
        model_id = entry.get("id")
        if not model_id:
            continue
        pricing = entry.get("pricing") or {}
        try:
            prompt = float(pricing["prompt"]) * 1_000_000.0
            completion = float(pricing["completion"]) * 1_000_000.0
        except (KeyError, TypeError, ValueError):
            continue
        tiers = []
        for override in pricing.get("overrides") or []:
            try:
                tiers.append((int(override["min_prompt_tokens"]),
                              float(override["prompt"]) * 1_000_000.0,
                              float(override["completion"]) * 1_000_000.0))
            except (KeyError, TypeError, ValueError):
                continue        # a malformed tier is dropped, not guessed at
        # A catalog zero is real but not comparable — see SOURCE_FREE_TIER.
        source = SOURCE_FREE_TIER if (prompt == 0.0 and completion == 0.0) else SOURCE_CATALOG
        table[model_id] = ModelPrice(
            model_id=model_id,
            prompt_usd_per_1m=prompt,
            completion_usd_per_1m=completion,
            source=source,
            context_length=entry.get("context_length"),
            tiers=tuple(sorted(tiers)),
        )
    return table


def load_snapshot(path: str = SNAPSHOT_PATH) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)


@lru_cache(maxsize=4)
def catalog(path: str = SNAPSHOT_PATH) -> Dict[str, ModelPrice]:
    """The vendored price table. Cached: it is read once per process."""
    return parse_catalog(load_snapshot(path))


def fetch_catalog(url: str = OPENROUTER_MODELS_URL, timeout: float = 30.0) -> Dict[str, Any]:
    """Fetch the live catalog. ``urllib`` only — the harness stays dependency-light.

    Network-touching; used by the refresh script and the opt-in freshness test,
    never by a benchmark run.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "avaloka-bench/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #

def price_for(provider: str, model: str, *,
              table: Optional[Mapping[str, ModelPrice]] = None) -> ModelPrice:
    """Best available quote for ``(provider, model)``, always tagged with its source.

    Never raises for an unknown model — a run must not die because one row is
    unpriced. Use :func:`validate_models` at suite-build time when you want the
    loud failure instead.
    """
    prices = catalog() if table is None else table

    if provider in LOCAL_PROVIDERS:
        return _unpriced(model, SOURCE_LOCAL)

    hit = prices.get(model)
    if hit is not None:
        if provider in CATALOG_PROVIDERS:
            return hit
        # Same open-weight id served elsewhere (Groq, vLLM): usable as a proxy,
        # labelled so nobody mistakes it for that vendor's own price sheet.
        return ModelPrice(model_id=hit.model_id,
                          prompt_usd_per_1m=hit.prompt_usd_per_1m,
                          completion_usd_per_1m=hit.completion_usd_per_1m,
                          source=SOURCE_PROXY if hit.source == SOURCE_CATALOG else hit.source,
                          context_length=hit.context_length,
                          tiers=hit.tiers)

    return _unpriced(model)


def validate_models(pairs: Iterable[tuple[str, str]], *,
                    table: Optional[Mapping[str, ModelPrice]] = None) -> Dict[str, str]:
    """Assert every catalog-backed slug exists. Returns ``{model_id: source}``.

    This is the guard that stops a hand-typed slug reaching a paid API call.
    Local and known-proprietary models are exempt (they cannot be in the
    OpenRouter catalog by definition); everything routed via ``openrouter``
    must resolve or this raises.
    """
    prices = catalog() if table is None else table
    resolved: Dict[str, str] = {}
    missing: list[str] = []
    for provider, model in pairs:
        quote = price_for(provider, model, table=prices)
        if provider in CATALOG_PROVIDERS and quote.source not in (
                SOURCE_CATALOG, SOURCE_FREE_TIER):
            missing.append(f"{provider}:{model}")
        resolved[model] = quote.source
    if missing:
        raise PriceLookupError(
            "Model ids not present in the OpenRouter catalog snapshot: "
            + ", ".join(sorted(missing))
            + ". Either the slug is wrong or the snapshot is stale — run "
              "scripts/refresh_openrouter_prices.py."
        )
    return resolved


def batch_variant(model: str, *,
                  table: Optional[Mapping[str, ModelPrice]] = None) -> Optional[str]:
    """The ``:batch`` slug for a model when one exists, else ``None``.

    OpenRouter's batch tier is roughly half price. Whether the harness *should*
    use it is a measurement question, not a cost question — see
    :mod:`benchmarks.core.spend`.
    """
    if model.endswith(":batch"):
        return model
    prices = catalog() if table is None else table
    candidate = f"{model}:batch"
    return candidate if candidate in prices else None
