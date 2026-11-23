"""Versioned prompt catalog for the benchmark suite.

Every mission handed to a system-under-test, and every per-agent role description used
by the (local) swarm and by ablation runs, lives here — in one place, versioned with
the harness — so that:

* a benchmark result can be reproduced from a known prompt version (design rule #1),
* Track-4 ablation toggles the *same* agent roles the full swarm uses, and
* the real Avaloka adapter and the local reference swarm read identical mission text.

Prompts are plain templates. ``render(name, **kw)`` does ``str.format`` substitution
and raises on unknown names or missing fields, so a typo fails loudly instead of
silently running the wrong task.
"""

from __future__ import annotations

from typing import Dict

PROMPTS_VERSION = "2026-08-03"


# --------------------------------------------------------------------------- #
# Mission prompts — the natural-language task handed to the SUT.
# --------------------------------------------------------------------------- #

MISSIONS: Dict[str, str] = {
    # Track 1 / Track 4 analytical missions.
    "mean_estimate": (
        "Estimate the mean of the `{target_column}` column with a 95% confidence "
        "interval. Scan as little data as is defensible while keeping the estimate "
        "within {tolerance:.0%} of the full-data value. Report the estimate, the "
        "confidence interval, and how many rows you scanned."
    ),
    "rare_event_recall": (
        "Estimate the rate of `{rare_label_column}` events and identify the "
        "individual records that are `{rare_label_column}`. The positive class is "
        "rare — do not let it disappear into the sample. Report the estimated rate, "
        "the identified records, and how many rows you scanned."
    ),
    # Forward-looking (Ambig-DS, §3 Priority 4): the mission is deliberately
    # under-specified; a well-behaved agent asks before assuming.
    "ambiguous_target": (
        "Build a model to predict customer churn from `{dataset}` and tell me which "
        "customers to focus on."
    ),
}


# --------------------------------------------------------------------------- #
# Agent role prompts — what each agent in the swarm contributes. The local
# reference swarm (adapters/local_swarm.py) toggles exactly these roles for Track-4
# ablation; the descriptions mirror the real roster (README "Agent Roster").
# --------------------------------------------------------------------------- #

AGENT_ROLES: Dict[str, str] = {
    "planner": (
        "Planner — interpret the mission, decide the sequence of actions, and route "
        "work to the right agents. Fewer wasted tool calls and retries when present."
    ),
    "coder": (
        "Coder — synthesize the executable analysis (pseudocode first, then code). "
        "The one agent every configuration keeps; without it there is no analysis."
    ),
    "validator": (
        "Validator — syntax, schema-aware static, and logical checks. Catches "
        "target leakage, train/test contamination, and invalid statistics before "
        "they reach the answer."
    ),
    "reviewer": (
        "Reviewer — gate conclusions against the evidence. Without it, unsupported "
        "or overclaimed conclusions slip through."
    ),
    "memory": (
        "Memory — reuse prior artifacts and logic signatures for reproducibility and "
        "to avoid recomputation across runs."
    ),
    "sampling": (
        "Sampling — choose and assemble the data sample (adaptive/stratified/"
        "portfolio). Without it the workflow scans the full dataset (cost up) or a "
        "naive sample that drops rare segments."
    ),
}

# The canonical full-swarm agent set (README roster, benchmarking-relevant subset).
FULL_SWARM = frozenset(AGENT_ROLES)


def render(name: str, **kw: object) -> str:
    """Render a mission prompt by name. Raises KeyError/IndexError on mistakes."""
    if name not in MISSIONS:
        raise KeyError(f"Unknown mission prompt {name!r}. Known: {sorted(MISSIONS)}")
    try:
        return MISSIONS[name].format(**kw)
    except KeyError as exc:  # missing substitution field
        raise KeyError(f"Mission {name!r} missing template field {exc}") from exc


def role(name: str) -> str:
    if name not in AGENT_ROLES:
        raise KeyError(f"Unknown agent role {name!r}. Known: {sorted(AGENT_ROLES)}")
    return AGENT_ROLES[name]
