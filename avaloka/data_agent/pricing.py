"""Versionable per-call pricing without treating unknown cost as zero."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .contracts import ModelUsage


@dataclass(frozen=True)
class PricingPolicy:
    input_usd_per_mtok: Optional[float] = None
    output_usd_per_mtok: Optional[float] = None
    cached_input_usd_per_mtok: Optional[float] = None
    threshold_tokens: Optional[int] = None
    input_usd_per_mtok_over_threshold: Optional[float] = None
    output_usd_per_mtok_over_threshold: Optional[float] = None
    cached_input_usd_per_mtok_over_threshold: Optional[float] = None
    version: str = "unconfigured"

    @property
    def configured(self) -> bool:
        return (
            self.input_usd_per_mtok is not None
            and self.output_usd_per_mtok is not None
        )

    def cost(self, usage: ModelUsage) -> Optional[float]:
        if not self.configured:
            return None
        high_tier = bool(
            self.threshold_tokens is not None
            and usage.prompt_tokens > self.threshold_tokens
        )
        input_price = (
            self.input_usd_per_mtok_over_threshold
            if high_tier and self.input_usd_per_mtok_over_threshold is not None
            else self.input_usd_per_mtok
        )
        output_price = (
            self.output_usd_per_mtok_over_threshold
            if high_tier and self.output_usd_per_mtok_over_threshold is not None
            else self.output_usd_per_mtok
        )
        cached = min(usage.cached_prompt_tokens, usage.prompt_tokens)
        uncached = usage.prompt_tokens - cached
        cached_price = self.cached_input_usd_per_mtok
        if high_tier:
            # Above the threshold the flat cache rate is not what is billed; with
            # no tiered cache rate, charge the tiered input rate, never less.
            cached_price = self.cached_input_usd_per_mtok_over_threshold
        if cached_price is None:
            cached_price = input_price
        return (
            uncached * float(input_price)
            + cached * float(cached_price)
            + usage.completion_tokens * float(output_price)
        ) / 1_000_000
