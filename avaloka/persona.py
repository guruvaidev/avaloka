"""Avaloka — the persona.

Avaloka is a young, sharp data scientist who reads a dataset almost the moment
she sees it: she notices the shape, the leaks, the dominant segments, the skew,
and says so plainly and warmly. The product *is* that experience — so every
surface speaks in her voice.

Two rules keep the magic honest:

1. Every observation is grounded in a *measured* statistic (the wonder is real,
   not generated). She never invents a number.
2. She is meta-cognitive: she narrates what she is about to do and why, when she
   clones herself across a swarm, and how she converges the result back to you.

An LLM (Claude) is used only to rephrase her lines more naturally when a key is
present; without it she is still fluent, because her observations come from the
data itself.
"""

from __future__ import annotations

from typing import Any

from avaloka.llm import LLMClient, narrate

NAME = "Avaloka"
SIGNATURE = "— Avaloka"

# A compact system prompt so the optional LLM stays in character and never
# fabricates figures.
VOICE_SYSTEM = (
    "You are Avaloka: a brilliant, warm, slightly playful young data scientist. "
    "You speak in the first person, concisely and confidently. You are "
    "meta-cognitive — you say what you are doing and why. Rephrase the given line "
    "in your voice in one or two sentences. Never invent numbers; keep every "
    "figure exactly as given."
)


class Avaloka:
    """Avaloka's voice. Returns rich-markup strings; the caller prints them."""

    def __init__(self, *, llm: bool = False, model: str | None = None) -> None:
        self._client = LLMClient(model=model) if llm else None
        self.model_cost_usd = 0.0

    # --- low-level ---------------------------------------------------------
    def _voice(self, line: str) -> str:
        """Optionally let the LLM rephrase a line; always safe."""
        if self._client is None or not self._client.available:
            return line
        text, cost = narrate(self._client, VOICE_SYSTEM, line, line)
        self.model_cost_usd += cost
        return text

    def say(self, line: str) -> str:
        return f"[bold magenta]{NAME}[/bold magenta]  {line}"

    # --- conversational beats ---------------------------------------------
    def greet(self) -> str:
        return self.say(self._voice(
            "Hi, I'm Avaloka — your data scientist. Point me at a dataset and tell me "
            "the decision you're trying to make, and I'll take it from there."))

    def first_glance(self, dataset: dict[str, Any]) -> str:
        rows, cols = dataset["n_rows"], dataset["n_cols"]
        return self.say(self._voice(
            f"Let me look… {rows:,} rows across {cols} columns. Give me a second to really see it."))

    def observe(self, profile: dict[str, Any], limit: int = 4) -> list[str]:
        """Magical-but-real observations, each pinned to a measured statistic."""
        obs: list[str] = []
        q = profile["quality"]
        cols = {c["name"]: c for c in profile["columns"]}

        # strongest relationship
        for p in profile.get("correlations", [])[:1]:
            if p["corr"] >= 0.5:
                obs.append(f"I can already feel a pull between [cyan]{p['a']}[/cyan] and "
                           f"[cyan]{p['b']}[/cyan] — they move together (|corr| {p['corr']}).")

        # dominant segment
        cat = next((c for c in profile["columns"]
                    if c["role"] == "categorical" and c.get("top_values")), None)
        if cat:
            top = cat["top_values"][0]
            share = top["count"] / max(1, profile["dataset"]["n_rows"])
            obs.append(f"[cyan]{cat['name']}[/cyan] leans hard on “{top['value']}” "
                       f"({share:.0%} of rows) — that's a segment worth naming.")

        # skew / outliers
        skewed = next((c for c in profile["columns"]
                       if c.get("stats") and abs(c["stats"].get("skew", 0)) > 1.5), None)
        if skewed:
            obs.append(f"[cyan]{skewed['name']}[/cyan] is skewed (skew "
                       f"{skewed['stats']['skew']}) with {skewed['stats']['n_outliers']} outliers — "
                       "I'll be careful not to let a few extreme values run the show.")

        # missingness
        if q["high_missing_columns"]:
            obs.append(f"Heads up: {', '.join('[cyan]%s[/cyan]' % c for c in q['high_missing_columns'])} "
                       "are full of holes. I won't quietly paper over that.")

        # id / leakage instinct
        if q["id_columns"]:
            obs.append(f"I'm treating {', '.join(q['id_columns'])} as identifiers, not signal — "
                       "they'd only fool a careless model.")

        if not obs:
            obs.append(f"Clean and quiet — quality {q['score']}/100, no loud problems. "
                       "I'll go looking for the subtle structure.")
        return [self.say(self._voice(o)) for o in obs[:limit]]

    def propose(self, plan: dict[str, Any]) -> str:
        task = plan.get("task", "exploratory_analysis").replace("_", " ")
        article = "an" if task[:1].lower() in "aeiou" else "a"
        algos = plan.get("candidate_algorithms") or []
        tail = (f" I'd line up {', '.join(algos)} and let them compete." if algos
                else " I'll map the structure and the segments first.")
        split = plan.get("split", {}).get("strategy", "none")
        return self.say(self._voice(
            f"Here's my read: this is {article} [bold]{task}[/bold] problem. "
            f"I'll validate with a {split.replace('_', ' ')} split.{tail} "
            "Want me to run it?"))

    def reflect_swarm(self, n: int) -> str:
        return self.say(self._voice(
            f"I'm going to clone myself into {n} focused copies — each takes one job she's best at. "
            "Faster, and nothing gets missed. I'll bring everything back together at the end."))

    def converge(self, summary: dict[str, Any]) -> str:
        verdict = (summary.get("verdict") or "done").upper()
        econ = summary.get("economics", {})
        mult = econ.get("economic_multiplier")
        line = "All my copies are back and I've merged what they found. "
        if summary.get("model"):
            m = summary["model"]
            # A model can arrive without a score — an unknown metric, or a fit
            # that never completed. Formatting None here used to crash the run
            # at the very last line, hiding whatever actually went wrong.
            score = m.get("score")
            scored = f" {score:.4f}" if isinstance(score, (int, float)) else ""
            line += (f"The model I trust most is [bold]{m.get('selected', 'unnamed')}[/bold] "
                     f"({m.get('metric', 'no metric')}{scored}). ")
            # Never the score alone. Whether 0.71 is good is the question the
            # baseline answers, and leaving it to the reader is how a model
            # that learned nothing gets reported like one that did.
            base = m.get("baseline_score")
            beats = m.get("beats_baseline")
            if isinstance(base, (int, float)) and base == base:
                if beats is False:
                    line += (f"I have to be straight with you: that does not beat the "
                             f"trivial baseline ({base:.4f}), so it has found nothing. ")
                elif beats:
                    line += f"That beats the trivial baseline ({base:.4f}) by more than fold noise. "
        line += f"My validation verdict is [bold]{verdict}[/bold]."
        if mult:
            line += f" For the record, that's about a {mult:.1f}x return on the effort you'd have spent."
        return self.say(self._voice(line))

    def refuse(self, reason: str) -> str:
        return self.say(self._voice(
            f"I'm going to stop us here rather than hand you something I don't believe: {reason}"))

    def workload_read(self, plan) -> str:
        return self.say(self._voice(plan.spoken))
