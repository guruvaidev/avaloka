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

An LLM (``--llm``: a local model, Groq, OpenRouter, OpenAI or Anthropic) is used
only to rephrase her lines more naturally; without it she is still fluent,
because her observations come from the data itself.
"""

from __future__ import annotations

from typing import Any

from avaloka.commentary import mark_commentary
from avaloka.llm import LLMClient, narrate
from avaloka.llm.guard import clean_rephrasing

NAME = "Avaloka"
SIGNATURE = "— Avaloka"

# A compact system prompt so the optional LLM stays in character and never
# fabricates figures. The output contract is spelled out because small models
# otherwise narrate the task: gemma4:e4b answered "I'm going to rephrase this to
# cleanly communicate that ..." and "To structure this request, I'm going to
# confirm ...". The old wording asked her to be meta-cognitive in the same
# breath as "rephrase the given line", and the model was meta-cognitive about
# the rephrasing. Her lines already say what she is doing with the data; the
# model's job is only the wording. avaloka.llm.guard is the backstop.
VOICE_SYSTEM = (
    "You are Avaloka: a brilliant, warm, slightly playful young data scientist, "
    "speaking in the first person, concisely and confidently.\n"
    "You will receive ONE line that Avaloka is about to say to the user. Reply with "
    "that line reworded in her voice, in one or two sentences, and NOTHING else.\n"
    "Rules:\n"
    "- Output only the reworded line itself. No preamble, no explanation, no quotes, "
    "no markdown, no options.\n"
    "- Never describe what you are doing to the text: do not say rephrase, reframe, "
    "restructure, this statement, this line, or similar.\n"
    "- Keep every number and every name exactly as given; never add a number.\n"
    "- Keep the meaning; add no new facts.\n"
    "Example. Line: Let me look… 1,200 rows across 9 columns. Give me a second to "
    "really see it.\n"
    "Reply: Ooh, 1,200 rows across 9 columns. Give me a second to really take it in."
)

#: The user turn: the line, marked so the model cannot mistake it for a request.
VOICE_PROMPT = "Line: {line}\nReply:"


#: A top value must hold at least this share of rows before it is called a
#: segment the data "leans hard on". Below it the line was being printed for
#: a 3% invoice number, which is noise dressed as an insight.
SEGMENT_MIN_SHARE = 1 / 3

#: ...and lead the second-commonest value by this factor. Gender at 50/50 was
#: being called a lean on "Male"; a lean is a value that clearly dominates.
SEGMENT_MIN_LEAD = 1.5

#: A column whose distinct values cover this fraction of rows is treated as
#: an identifier for observation purposes, the same line the leakage check
#: in :mod:`avaloka.stats` draws.
ID_LIKE_UNIQUE_PCT = 0.5

#: ...or whose distinct values outnumber max(this, sqrt(rows)): the average
#: value then appears fewer than sqrt(rows) times, which is a label (an invoice
#: number, a recipe title) rather than a segment, whatever its top share.
SEGMENT_MAX_CARDINALITY = 50


def _distinct(column: dict[str, Any], n_rows: int) -> float:
    n = column.get("n_unique")
    return float(n) if n is not None else float(column.get("unique_pct", 0.0)) * n_rows


class Avaloka:
    """Avaloka's voice. Returns rich-markup strings; the caller prints them."""

    def __init__(self, *, llm: bool = False, model: str | None = None,
                 provider: str | None = None, notify=None) -> None:
        self._client = (LLMClient(model=model, provider=provider, required=True, notify=notify)
                        if llm else None)
        self.model_cost_usd = 0.0

    # --- low-level ---------------------------------------------------------
    def _voice(self, line: str) -> str:
        """Optionally let the LLM rephrase a line; always safe."""
        if self._client is None or not self._client.available:
            return line
        text, cost = narrate(self._client, VOICE_SYSTEM, VOICE_PROMPT.format(line=line), line)
        self.model_cost_usd += cost
        spoken = clean_rephrasing(line, text)
        # The model's wording is commentary; the deterministic line it would
        # replace (and falls back to) is not.
        return line if spoken == line else mark_commentary(spoken)

    def say(self, line: str) -> str:
        return f"[bold magenta]{NAME}[/bold magenta]  {line}"

    # --- conversational beats ---------------------------------------------
    def greet(self) -> str:
        return self.say(self._voice(
            "Hi, I'm Avaloka — your data scientist. Point me at a dataset and tell me "
            "the decision you're trying to make, and I'll take it from there."))

    def first_glance(self, dataset: dict[str, Any]) -> str:
        rows, cols = dataset["n_rows"], dataset["n_cols"]
        # A planner's estimate is said as one, never as the count.
        shown = f"about {rows:,} rows (estimated)" if dataset.get("estimated") else f"{rows:,} rows"
        return self.say(self._voice(
            f"Let me look… {shown} across {cols} columns. Give me a second to really see it."))

    def observe(self, profile: dict[str, Any], limit: int = 4) -> list[str]:
        """Magical-but-real observations, each pinned to a measured statistic."""
        obs: list[str] = []
        q = profile["quality"]
        cols = {c["name"]: c for c in profile["columns"]}

        # Identifier-like columns carry no signal to observe: an invoice number
        # or a recipe title is "categorical" to the profiler, and two id
        # columns can correlate without there being a relationship to name.
        n_rows = max(1, profile["dataset"]["n_rows"])
        ids = set(q.get("id_columns") or [])
        ids |= {c["name"] for c in profile["columns"]
                if c.get("role") == "id" or c.get("unique_pct", 0.0) >= ID_LIKE_UNIQUE_PCT
                or _distinct(c, n_rows) > max(SEGMENT_MAX_CARDINALITY, n_rows ** 0.5)}

        # strongest relationship, between columns that are not identifiers
        for p in profile.get("correlations", []):
            if p["a"] in ids or p["b"] in ids:
                continue
            if p["corr"] >= 0.5:
                obs.append(f"I can already feel a pull between [cyan]{p['a']}[/cyan] and "
                           f"[cyan]{p['b']}[/cyan] — they move together (|corr| {p['corr']}).")
            break

        # dominant segment: the categorical column whose top value holds the
        # largest share of rows, and only when that share is a real lean. The
        # first categorical column's top value at 3% of rows is not a segment.
        best, best_share = None, 0.0        # the strongest genuine lean
        closest, closest_share = None, 0.0  # the strongest top value of any real categorical
        for c in profile["columns"]:
            if c.get("role") != "categorical" or not c.get("top_values") or c["name"] in ids:
                continue
            # A column with one value is a constant, not a segment: the
            # largest-share rule would otherwise always pick it at 100%.
            if _distinct(c, n_rows) <= 1 or len(c["top_values"]) < 2:
                continue
            tops = c["top_values"]
            share = tops[0]["count"] / n_rows
            if share > closest_share:
                closest, closest_share = c, share
            # A 50/50 split is not a lean either: the top value must clearly
            # lead the runner-up, not merely be the larger half.
            runner_up = tops[1]["count"] if len(tops) > 1 else 0
            if tops[0]["count"] < SEGMENT_MIN_LEAD * runner_up:
                continue
            if share > best_share:
                best, best_share = c, share
        if best is not None and best_share >= SEGMENT_MIN_SHARE:
            top = best["top_values"][0]
            obs.append(f"[cyan]{best['name']}[/cyan] leans hard on “{top['value']}” "
                       f"({best_share:.0%} of rows) — that's a segment worth naming.")
        elif closest is not None:
            # Say that nothing leans, rather than inventing a segment.
            top = closest["top_values"][0]
            obs.append(f"Nothing dominates a column — the closest is [cyan]{closest['name']}[/cyan] "
                       f"at {closest_share:.0%} “{top['value']}”; the segments are spread out.")

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
