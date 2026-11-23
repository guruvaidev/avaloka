"""Run configuration.

Model effects and agent effects are kept *separate* by expressing them as
configuration rather than code branches (benchmark design rule #4). To compare
"same Avaloka, different LLM" you vary ``llm`` and hold everything else fixed; to
compare "swarm vs single agent" you vary ``agents_enabled``; to compare "adaptive
vs fixed sample" you vary ``sampling_mode``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|credential|password|secret|"
    r"(?:^|[_-])token(?:$|[_-]))",
    re.IGNORECASE,
)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "<redacted>" if _SECRET_KEY.search(str(key))
            else _redact_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


def _safe_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
    """Return publishable config extras without ever persisting credentials."""
    return {
        key: "<redacted>" if _SECRET_KEY.search(key) else _redact_value(value)
        for key, value in sorted(extra.items())
        if not key.startswith("_")
    }


@dataclass(frozen=True)
class Budget:
    """Hard ceilings a run must respect; recorded on the scorecard."""

    max_cost_usd: Optional[float] = None
    max_latency_s: Optional[float] = None
    max_tokens: Optional[int] = None
    max_data_scanned_ratio: Optional[float] = None


@dataclass(frozen=True)
class RunConfig:
    """A single, fully-specified benchmark configuration.

    ``sut`` selects the adapter (``"avaloka"``, ``"fixed_sample"``,
    ``"full_data_reference"``, ``"single_agent"``, ...). The remaining fields are the
    axes we publish independently so nobody can attribute an Avaloka gain to the
    underlying foundation model.
    """

    id: str
    sut: str
    llm: str = "unset"
    # Track 4 ablation axis. ``None`` means "the full Avaloka swarm". Otherwise the
    # explicit set of enabled agents (e.g. {"planner","coder","validator"}).
    agents_enabled: Optional[frozenset[str]] = None
    # Track 1 axis. One of: quick_sample | portfolio_samples | entire_dataset |
    # random | stratified | reservoir | domain | adaptive | full.
    sampling_mode: str = "adaptive"
    # Track 3 axis. Router policy name, or a pinned model id.
    model_routing: str = "default"
    # Track 2 axis. Execution environment key (laptop_duckdb, postgres, ray_k8s, ...).
    environment: str = "local"
    trials: int = 5  # design rule #2: >= 5 trials/task
    internet_allowed: bool = False  # design rule #1: isolated networking by default
    budget: Budget = field(default_factory=Budget)
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def scorecard_config(self) -> Dict[str, object]:
        """The full config block published alongside every result (§7)."""
        return {
            "config_id": self.id,
            "sut": self.sut,
            "llm": self.llm,
            "agents_enabled": (
                sorted(self.agents_enabled) if self.agents_enabled is not None else "full_swarm"
            ),
            "sampling_mode": self.sampling_mode,
            "model_routing": self.model_routing,
            "environment": self.environment,
            "trials": self.trials,
            "internet_allowed": self.internet_allowed,
            "budget": {
                "max_cost_usd": self.budget.max_cost_usd,
                "max_latency_s": self.budget.max_latency_s,
                "max_tokens": self.budget.max_tokens,
                "max_data_scanned_ratio": self.budget.max_data_scanned_ratio,
            },
            "seed": self.seed,
            "extra": _safe_extra(self.extra),
        }

    def fingerprint(self) -> str:
        """Stable, secret-free identity used by checkpoint/resume."""
        execution_config = dict(self.scorecard_config())
        # Trial count controls repetition, not the behavior of an individual trial.
        execution_config.pop("trials", None)
        payload = json.dumps(execution_config, sort_keys=True, default=str,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
