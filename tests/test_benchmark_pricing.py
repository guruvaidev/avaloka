"""Price sourcing, cost accounting, and the guard against hand-typed slugs.

The property under test throughout: a number in the Cost column is either
derived from a catalog quote or explicitly labelled as not being one. There is
no third option, and in particular "unknown" must never render as a plausible
zero.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.core import pricing
from benchmarks.core.pricing import (
    SOURCE_CATALOG, SOURCE_LOCAL, SOURCE_PROXY, SOURCE_UNPRICED, ModelPrice,
    PriceLookupError, batch_variant, catalog, parse_catalog, price_for,
    validate_models,
)


# --------------------------------------------------------------------------- #
# The vendored snapshot
# --------------------------------------------------------------------------- #

def test_snapshot_loads_and_is_a_real_catalog():
    table = catalog()
    assert len(table) > 300, "snapshot looks truncated"
    assert all(isinstance(v, ModelPrice) for v in table.values())


def test_snapshot_records_its_provenance():
    """A price table with no source and no date is indistinguishable from a guess."""
    snap = pricing.load_snapshot()
    assert snap["source_url"].startswith("https://openrouter.ai")
    assert snap["fetched_at"]
    assert snap["model_count"] == len(snap["data"])


def test_per_token_strings_are_normalised_to_per_million():
    """The API quotes $/token as a string; every caller must see $/1M floats."""
    table = parse_catalog({"data": [{
        "id": "vendor/model", "context_length": 128,
        "pricing": {"prompt": "0.0000005", "completion": "0.0000025"},
    }]})
    assert table["vendor/model"].prompt_usd_per_1m == pytest.approx(0.5)
    assert table["vendor/model"].completion_usd_per_1m == pytest.approx(2.5)


def test_unparseable_pricing_is_dropped_not_defaulted_to_zero():
    """A model we cannot price must be absent, never present-and-free."""
    table = parse_catalog({"data": [
        {"id": "a/no-pricing"},
        {"id": "b/null-price", "pricing": {"prompt": None, "completion": None}},
        {"id": "c/text", "pricing": {"prompt": "n/a", "completion": "n/a"}},
        {"id": "d/ok", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
    ]})
    assert set(table) == {"d/ok"}


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

def test_openrouter_model_is_a_direct_catalog_quote():
    price = price_for("openrouter", "anthropic/claude-opus-5")
    assert price.source == SOURCE_CATALOG
    assert price.prompt_usd_per_1m == pytest.approx(5.0)
    assert price.completion_usd_per_1m == pytest.approx(25.0)


def test_same_id_on_groq_is_flagged_as_a_proxy_not_a_quote():
    """gpt-oss-120b is served by both; the Groq row must not claim OR's price sheet."""
    groq = price_for("groq", "openai/gpt-oss-120b")
    router = price_for("openrouter", "openai/gpt-oss-120b")
    assert groq.source == SOURCE_PROXY
    assert router.source == SOURCE_CATALOG
    assert groq.prompt_usd_per_1m == router.prompt_usd_per_1m
    assert groq.is_priced and router.is_priced


def test_groq_proprietary_model_is_unpriced_not_free():
    """groq/compound is absent from OpenRouter (see model_fallback.py)."""
    price = price_for("groq", "groq/compound")
    assert price.source == SOURCE_UNPRICED
    assert not price.is_priced
    assert price.cost_usd(1_000_000, 1_000_000) == 0.0  # zero, but labelled


def test_local_serving_is_zero_and_says_why():
    price = price_for("ollama", "gemma4:31b")
    assert price.source == SOURCE_LOCAL
    assert price.cost_usd(10_000, 10_000) == 0.0


def test_unpriced_and_local_are_distinguishable():
    """Both cost 0.0; conflating them loses the 'we don't know' signal."""
    assert price_for("groq", "groq/compound").source != price_for("ollama", "x").source


# --------------------------------------------------------------------------- #
# Cost arithmetic
# --------------------------------------------------------------------------- #

def test_cost_is_prompt_and_completion_priced_separately():
    """Output tokens cost 5x input on opus-5; a blended rate would be wrong."""
    price = price_for("openrouter", "anthropic/claude-opus-5")
    # 1M in @ $5 + 1M out @ $25
    assert price.cost_usd(1_000_000, 1_000_000) == pytest.approx(30.0)
    assert price.cost_usd(1_000, 0) == pytest.approx(0.005)
    assert price.cost_usd(0, 1_000) == pytest.approx(0.025)


def test_zero_tokens_costs_zero():
    assert price_for("openrouter", "x-ai/grok-4.6").cost_usd(0, 0) == 0.0


# --------------------------------------------------------------------------- #
# The slug guard — the whole point of sourcing from a catalog
# --------------------------------------------------------------------------- #

def test_every_matrix_model_resolves():
    """A hand-typed slug must not be able to reach a paid API call."""
    from benchmarks.tracks.model_matrix.track import MATRIX
    resolved = validate_models((p, m) for _, p, m in MATRIX)
    # Assert on CONTENT, not a count: QA's F8 noted a hardcoded count silently
    # invalidated two tests in two days, and `>= 15` would pass with rows deleted.
    assert set(resolved) == {m for _, _, m in MATRIX}
    for _, provider, model in MATRIX:
        if provider == "openrouter":
            assert resolved[model] == SOURCE_CATALOG, model


def test_a_typo_in_an_openrouter_slug_fails_loudly():
    with pytest.raises(PriceLookupError) as exc:
        validate_models([("openrouter", "anthropic/claude-opus-5-typo")])
    assert "claude-opus-5-typo" in str(exc.value)
    assert "refresh_openrouter_prices" in str(exc.value)


def test_local_and_proprietary_slugs_are_exempt_from_the_guard():
    """They cannot be in OpenRouter's catalog by definition."""
    resolved = validate_models([("ollama", "gemma4:31b"), ("groq", "groq/compound")])
    assert resolved == {"gemma4:31b": SOURCE_LOCAL, "groq/compound": SOURCE_UNPRICED}


def test_suite_build_validates_before_any_credential_is_read(monkeypatch):
    """The guard runs at build() time, not at first call."""
    import benchmarks.tracks.model_matrix.track as track
    monkeypatch.setattr(track, "MATRIX",
                        [("m_bad", "openrouter", "not-a-real/model-xyz")])
    with pytest.raises(PriceLookupError):
        track.build()


# --------------------------------------------------------------------------- #
# Batch variants
# --------------------------------------------------------------------------- #

def test_batch_variant_found_when_it_exists_and_is_cheaper():
    slug = batch_variant("openai/gpt-5.6-sol")
    assert slug == "openai/gpt-5.6-sol:batch"
    std = price_for("openrouter", "openai/gpt-5.6-sol")
    batch = price_for("openrouter", slug)
    assert batch.completion_usd_per_1m < std.completion_usd_per_1m


def test_batch_variant_absent_returns_none_rather_than_a_guessed_slug():
    assert batch_variant("qwen/qwen3-coder-next") is None


def test_batch_variant_is_idempotent():
    assert batch_variant("openai/gpt-5.6-sol:batch") == "openai/gpt-5.6-sol:batch"


# --------------------------------------------------------------------------- #
# Live catalog — opt in; the default run stays offline
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_live_catalog_still_contains_every_matrix_model(monkeypatch):
    """Catches a deprecation before it becomes an HTTP 400 mid-sweep.

    Network-touching, so it is gated twice: the ``integration`` marker keeps it
    out of the default run, and the env var keeps it out of an ``-m integration``
    run that has no outbound network.
    """
    import os
    if os.getenv("AVALOKA_BENCH_LIVE_CATALOG") != "1":
        pytest.skip("set AVALOKA_BENCH_LIVE_CATALOG=1 to hit the live OpenRouter catalog")
    from benchmarks.tracks.model_matrix.track import MATRIX

    live = pricing.parse_catalog(pricing.fetch_catalog())
    missing = [m for _, p, m in MATRIX if p == "openrouter" and m not in live]
    assert not missing, f"deprecated upstream: {missing}"
