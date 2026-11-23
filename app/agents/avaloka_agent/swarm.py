"""Avaloka's swarm — how she narrates cloning herself over the real agents.

Avaloka works by sending focused copies of herself at a job. Under the hood
those copies ARE the existing LangGraph agents (planner, coder, validator,
executor, MTA, visualizer). This module is the *naming + framing* layer: it
maps an intent to the ordered set of clones Avaloka will dispatch for it, so
the UI can show a compact "clones at work" feed in her voice — without any
agent needing to know it is being narrated.

The output is deliberately small and JSON-serialisable: it rides in
``ETLState['avaloka_swarm']`` and is surfaced to the UI via
``execution_context.swarm``. It carries no business logic and never changes
routing — the graph decides what actually runs; this only tells the story.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Each clone is a focused copy of Avaloka. ``focus`` is what that copy obsesses
# over, phrased in her first-person voice for direct display.
CLONE: Dict[str, Dict[str, str]] = {
    "scout":     {"name": "Ava-Scout",     "role": "Data analyst",
                  "focus": "reads the schema, the distributions and the cracks"},
    "sampler":   {"name": "Ava-Sampler",   "role": "Data scientist",
                  "focus": "carves out a sample that still tells the truth"},
    "engineer":  {"name": "Ava-Engineer",  "role": "Data engineer",
                  "focus": "shapes and cleans the data so the pipeline is reproducible"},
    "coder":     {"name": "Ava-Coder",     "role": "Analytics engineer",
                  "focus": "writes the code that answers the question"},
    "skeptic":   {"name": "Ava-Skeptic",   "role": "Model-risk specialist",
                  "focus": "tries to break the result before anyone else can"},
    "runner":    {"name": "Ava-Runner",    "role": "Execution engineer",
                  "focus": "runs the job and brings back exactly what it produced"},
    "scientist": {"name": "Ava-Scientist", "role": "ML scientist",
                  "focus": "trains the contenders and picks the one to trust"},
    "builder":   {"name": "Ava-Builder",   "role": "Infrastructure engineer",
                  "focus": "stands up the compute the job needs"},
    "scheduler": {"name": "Ava-Scheduler", "role": "Operations engineer",
                  "focus": "puts the job on a schedule and watches it"},
    "voice":     {"name": "Ava-Voice",     "role": "Analyst / consultant",
                  "focus": "writes it up the way you'd want to read it"},
}

# Ordered clone plan per intent. These mirror the real downstream path the
# graph takes for that intent, so the story never drifts from what runs.
_PLANS: Dict[str, List[str]] = {
    "statistical_analysis": ["scout", "sampler", "coder", "skeptic", "runner", "voice"],
    "visualization":        ["scout", "coder", "runner", "voice"],
    "data_transfer":        ["scout", "engineer", "coder", "runner", "voice"],
    "ml_training":          ["scout", "sampler", "engineer", "scientist", "skeptic", "voice"],
    "ml_inference":         ["scout", "runner", "voice"],
    "infrastructure":       ["builder", "voice"],
    "schedule":             ["scheduler", "voice"],
}


def plan_swarm(intent: str) -> List[Dict[str, Any]]:
    """Return the ordered clone plan for an intent (empty if not a work intent).

    Each entry is ``{clone, name, role, focus, status}`` with status seeded to
    ``"pending"`` — the UI (or a later progress emitter) flips these to
    ``"done"`` as the corresponding agent completes.
    """
    steps = _PLANS.get(intent, [])
    plan: List[Dict[str, Any]] = []
    for key in steps:
        meta = CLONE.get(key, {"name": f"Ava-{key}", "role": "specialist", "focus": ""})
        plan.append({
            "clone": key,
            "name": meta["name"],
            "role": meta["role"],
            "focus": meta["focus"],
            "status": "pending",
        })
    return plan


def announce(intent: str) -> str:
    """A one-line, first-person framing of the clones Avaloka is dispatching."""
    plan = plan_swarm(intent)
    if not plan:
        return ""
    names = [p["name"].replace("Ava-", "") for p in plan]
    if len(names) == 1:
        who = names[0]
    elif len(names) == 2:
        who = f"{names[0]} and {names[1]}"
    else:
        who = f"{', '.join(names[:-1])} and {names[-1]}"
    return f"Cloning myself into {len(plan)} focused copies — {who} — and I'll bring back one answer."
