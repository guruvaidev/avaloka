"""Can the agents actually reach a model, and is there one place to observe them?

Two checks the benchmark suite cannot make for itself, both learned the hard way.

**Reachability.** Groq removed the entire Llama line from this project's
account. Six agents kept naming ``llama-3.3-70b-versatile`` and got HTTP 404 —
not at deploy time, but mid-analysis. A benchmark run against a system whose
agents cannot call a model produces numbers that look like agent failures.
Before any DAB or DABstep run, confirm the models exist.

**A single chokepoint.** *Agent Lightning v1.0* (arXiv:2608.17528) makes the
case that the deploy-time harness, not the trainer, owns the interaction loop,
and the trainer observes only sequences of LLM request/response pairs. That
only works if such a sequence can be observed at one place.

``app/core/inference.build_chat_model()`` is exactly that place: every agent LLM
is constructed through it. Whether or not RL post-training is ever pursued, a
single chokepoint is what makes trajectory capture, replay, and per-agent cost
attribution possible at all. Its absence is not a bug today — it is a ceiling on
what can be measured later.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set

from benchmarks.smoke.result import Severity, SmokeResult

SURFACE = "models"
REPO = Path(__file__).resolve().parents[2]

#: Verified live against this project's Groq account on 2026-08-17 via
#: GET /openai/v1/models plus a one-token completion per entry.
KNOWN_LIVE: Set[str] = {
    "openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b",
    "groq/compound", "groq/compound-mini", "qwen/qwen3.6-27b", "allam-2-7b",
    "whisper-large-v3", "whisper-large-v3-turbo",
    "meta-llama/llama-prompt-guard-2-22m", "meta-llama/llama-prompt-guard-2-86m",
}

#: Removed from the account. Any reference is a guaranteed 404.
KNOWN_DEAD: Set[str] = {"llama-3.3-70b-versatile", "llama-3.1-8b-instant"}

AGENT_DIRS = ("app/agents", "app/core", "app/services")


def run(*, include_slow: bool = False) -> List[SmokeResult]:
    return [_check_no_dead_models(), _check_single_inference_chokepoint()]


def _sources() -> List[Path]:
    out: List[Path] = []
    for rel in AGENT_DIRS:
        base = REPO / rel
        if base.is_dir():
            out.extend(p for p in base.rglob("*.py") if "__pycache__" not in str(p))
    return out


def _check_no_dead_models() -> SmokeResult:
    """No agent may name a model the provider has removed."""
    hits: List[str] = []
    for path in _sources():
        body = path.read_text(encoding="utf-8", errors="replace")
        for dead in KNOWN_DEAD:
            if f'"{dead}"' in body or f"'{dead}'" in body:
                rel = path.relative_to(REPO)
                hits.append(f"{rel}: {dead}")
    return SmokeResult(
        check="no_dead_models_referenced", surface=SURFACE, passed=not hits,
        severity=Severity.CRITICAL,
        detail=("no agent references a removed model" if not hits else
                f"{len(hits)} reference(s) to models the provider returns 404 for — "
                "a benchmark run would measure agents that cannot call an LLM"),
        evidence={"hits": hits[:12], "dead": sorted(KNOWN_DEAD)},
    )


def _check_single_inference_chokepoint() -> SmokeResult:
    """One place where every agent LLM is built.

    Warning, not critical: agents constructing ChatGroq directly still work.
    But without a chokepoint there is nowhere to record request/response pairs,
    which caps what can ever be measured — trajectory replay, per-agent cost,
    and the harnessed-RL setup Agent Lightning describes all need it.
    """
    factory = REPO / "app/core/inference.py"
    if not factory.is_file():
        return SmokeResult(
            check="single_inference_chokepoint", surface=SURFACE, passed=False,
            severity=Severity.WARNING,
            detail=("app/core/inference.py absent; agents construct clients directly, "
                    "so LLM traffic cannot be observed in one place"),
        )
    direct = []
    for path in _sources():
        if path.name == "inference.py":
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        if "ChatGroq(" in body and "build_chat_model" not in body:
            direct.append(str(path.relative_to(REPO)))
    return SmokeResult(
        check="single_inference_chokepoint", surface=SURFACE, passed=not direct,
        severity=Severity.WARNING,
        detail=("every agent LLM is built through one factory, so traffic is observable"
                if not direct else
                f"{len(direct)} module(s) construct a client directly, bypassing the factory"),
        evidence={"bypassing": direct[:10]},
    )
