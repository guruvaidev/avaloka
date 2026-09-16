"""``avaloka chat`` — the conversational experience.

This is Avaloka as a person you talk to. She greets you, reads whatever dataset
you point her at, says what she notices (grounded in real statistics), proposes a
plan, and — on your go-ahead — clones herself across the work and converges the
answer. It is a small, robust state machine: free-form when an LLM is available,
gently guided when it isn't, but always able to actually *run* the swarm.

Conversational verbs: a dataset path, ``analyze [goal]``, ``train <target>``,
``batch``, ``deploy``/``infer``, ``help``, ``quit``.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from avaloka import workload
from avaloka.io import load_dataset
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import (analyze_fireflies, run_analyze, run_train,
                              train_fireflies)
from avaloka.persona import Avaloka
from avaloka.stats import correlation_pairs, profile_column, quality_report
from avaloka.swarm import SwarmNarrator, make_progress

_DATA_SUFFIXES = (".csv", ".parquet", ".pq", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls")
_BYE = {"quit", "exit", "bye", "q", ":q"}

#: A reply that is entirely a reference to a numbered suggestion.
_PICK_RE = re.compile(
    r"^\s*(?:please\s+)?(?:run|do|go\s+with|start\s+with|pick|option|number|#)?"
    r"\s*#?\s*(\d{1,2})\s*(?:please|thanks)?\s*[.!]?\s*$",
    re.IGNORECASE,
)


def suggest_next_steps(profile: dict, limit: int = 5) -> list[str]:
    """Concrete next moves, named after columns this dataset actually has.

    Generic advice ("consider segmentation analysis") is advice about data
    science. Advice that names tenure_months is advice about *this* dataset,
    and only the second shows the schema was read.
    """
    cols = profile["columns"]
    by_role: dict[str, list[str]] = {}
    for col in cols:
        by_role.setdefault(col["role"], []).append(col["name"])

    numeric = by_role.get("numeric", [])
    categorical = by_role.get("categorical", [])
    boolean = by_role.get("boolean", [])
    datetimes = by_role.get("datetime", [])
    ideas: list[str] = []

    # A near-binary numeric column is almost always the outcome someone cares about.
    targets = [c["name"] for c in cols
               if c["role"] in {"numeric", "boolean"} and c.get("n_unique") == 2]
    for name in targets[:1]:
        ideas.append(f"train {name}")
        if categorical:
            ideas.append(f"analyze how {name} varies by {categorical[0]}")

    for pair in (profile.get("correlations") or [])[:1]:
        a, b = pair.get("a"), pair.get("b")
        if a and b:
            ideas.append(f"analyze the relationship between {a} and {b}")

    if datetimes and numeric:
        ideas.append(f"analyze how {numeric[0]} moves over {datetimes[0]}")
    if categorical and numeric:
        ideas.append(f"analyze {numeric[0]} broken down by {categorical[0]}")

    flagged = [i["column"] for i in profile["quality"].get("issues", []) if i.get("column")]
    if flagged:
        ideas.append(f"analyze the data-quality problems in {flagged[0]}")

    seen, unique = set(), []
    for idea in ideas:
        if idea.lower() not in seen:
            seen.add(idea.lower())
            unique.append(idea)
    return unique[:limit]


def render_suggestions(console: Console, ava: Avaloka, ideas: list[str]) -> None:
    """A numbered list, and an invitation to answer it with a number.

    Listing without inviting is a dead end: the reader is left guessing whether
    any of it can actually be run. The two are one move.
    """
    if not ideas:
        return
    console.print()
    console.print(ava.say("Here is what I would look at next:"))
    for n, idea in enumerate(ideas, 1):
        console.print(f"  [bold cyan]{n}.[/bold cyan] {idea}")
    console.print(ava.say("Say the number and I will start — or tell me something else entirely."))


def quick_profile(source: str, sample_rows: int = 100_000) -> dict:
    """A fast in-memory profile for Avaloka's first glance (samples very large files)."""
    handle = load_dataset(source)
    df = handle.frame
    if len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42)
    cols = [profile_column(df[c], len(df)) for c in df.columns]
    return {
        "dataset": {"source": handle.source, "format": handle.fmt, "sha256": handle.sha256,
                    "n_rows": handle.n_rows, "n_cols": handle.n_cols, "delimiter": handle.delimiter},
        "columns": cols,
        "quality": quality_report(df, cols),
        "correlations": correlation_pairs(df),
    }


def _greet_dataset(console: Console, ava: Avaloka, source: str) -> dict | None:
    try:
        wl = workload.route(source)
        profile = quick_profile(source)
    except Exception as exc:
        console.print(ava.say(f"I couldn't read that one — {exc}"))
        return None
    console.print(ava.first_glance(profile["dataset"]))
    console.print(ava.workload_read(wl))
    for line in ava.observe(profile):
        console.print("  " + line)
    # a light, plausible plan (no target known yet)
    console.print(ava.propose({"task": "exploratory_analysis", "split": {"strategy": "none"}}))
    ideas = suggest_next_steps(profile)
    render_suggestions(console, ava, ideas)
    if not ideas:
        console.print(ava.say("Tell me the [bold]decision[/bold] you're chasing, or say "
                              "`analyze`, or `train <column>` to predict something."))
    return {"source": source, "profile": profile, "workload": wl, "suggestions": ideas}


def _run_mission(console: Console, ava: Avaloka, ctx: MissionContext, fireflies, run_fn) -> None:
    narrator = SwarmNarrator(console.print, voice=ava)
    narrator.announce(len(fireflies))
    ctx.blackboard["workload"] = workload.route(ctx.data_source).as_dict()
    result = run_fn(ctx, make_progress(narrator))
    narrator.converge(ctx.ledger)
    console.print(ava.converge(result.summary))
    console.print(Panel(f"[bold green]Deliverables →[/bold green] {result.summary['output_dir']}",
                        border_style="green", expand=False))
    if result.summary.get("model"):
        console.print(ava.say(f"If you like it, I can serve it: `avaloka infer "
                              f"{result.summary['output_dir']} --target kubernetes --apply`."))
    console.print(ava.say("Ask me something else, say `suggest` for ideas, or `quit`."))


def run_chat(console: Console, data: str | None = None, llm: bool = False) -> None:
    ava = Avaloka(llm=llm)
    console.print(ava.greet())
    if llm and not ava._client.available:  # type: ignore[union-attr]
        console.print("[dim](No ANTHROPIC_API_KEY found — I'll still be myself, just without the polish.)[/dim]")
    # Escaped: rich reads a bare [goal] as a markup tag and prints nothing,
    # so the banner was advertising "analyze , train <target>".
    console.print("[dim]Commands: <dataset path>, a number, analyze \\[goal], "
                  "train <column>, columns, suggest, help, quit[/dim]\n")

    state: dict | None = None
    if data:
        state = _greet_dataset(console, ava, data)

    while True:
        try:
            raw = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not raw:
            continue
        low = raw.lower()

        if low in _BYE:
            console.print(ava.say("Anytime. Bring me your next dataset whenever you're ready."))
            break
        if low in {"help", "?"}:
            console.print(ava.say("Here is everything I understand:"))
            for cmd, what in (
                ("<path to a file>", "load a CSV, TSV, Parquet or Excel dataset"),
                ("a number", "run the suggestion I offered with that number"),
                ("analyze <goal>", "run an analysis toward a decision"),
                ("train <column>", "predict that column and validate the model"),
                ("columns", "show the schema I read, with roles and missingness"),
                ("suggest", "offer next steps again"),
                ("status", "what I currently have loaded"),
                ("help", "this list"),
                ("quit", "leave"),
            ):
                console.print(f"  [bold cyan]{cmd:<18}[/bold cyan] {what}")
            console.print(ava.say("Or just say what you are trying to find out, in your own words."))
            continue

        # A dataset path?
        token = raw.split()[0]
        if token.lower().endswith(_DATA_SUFFIXES) and Path(token).expanduser().exists():
            state = _greet_dataset(console, ava, token)
            continue

        if state is None:
            console.print(ava.say("First, point me at a dataset — a path to a CSV, TSV, "
                                  "Parquet or Excel file."))
            continue

        source = state["source"]

        if low in {"status", "what do you have", "where are we"}:
            ds = state["profile"]["dataset"]
            console.print(ava.say(
                f"Loaded [cyan]{Path(ds['source']).name}[/cyan] — {ds['n_rows']:,} rows x "
                f"{ds['n_cols']} columns, quality {state['profile']['quality']['score']}/100."))
            continue

        if low in {"columns", "schema", "cols"}:
            console.print(ava.say("Here is the schema as I read it:"))
            for col in state["profile"]["columns"]:
                miss = col.get("missing_pct", 0.0)
                flag = f"  [yellow]{miss:.0%} missing[/yellow]" if miss >= 0.05 else ""
                console.print(f"  [cyan]{col['name']:<24}[/cyan] {col['role']:<12}{flag}")
            continue

        if low in {"suggest", "suggestions", "ideas", "what next", "what now"}:
            state["suggestions"] = suggest_next_steps(state["profile"])
            render_suggestions(console, ava, state["suggestions"])
            continue

        # "3" answers the numbered list from the last turn. Without this the
        # invitation to "say the number" is a phrase that does nothing.
        pick = _PICK_RE.match(raw)
        pending = state.get("suggestions") or []
        if pick and pending:
            index = int(pick.group(1))
            if 1 <= index <= len(pending):
                chosen = pending[index - 1]
                console.print(ava.say(f"Taking [bold]{index}[/bold] — {chosen}."))
                raw, low = chosen, chosen.lower()
                # An offer is good for the turn that follows it.
                state["suggestions"] = []
            else:
                console.print(ava.say(
                    f"I only offered {len(pending)} option{'s' if len(pending) != 1 else ''}. "
                    f"Say a number between 1 and {len(pending)}, or tell me what you want."))
                continue

        if low.startswith("train"):
            parts = raw.split()
            if len(parts) < 2:
                console.print(ava.say("Which column should I predict? e.g. `train churned`."))
                continue
            target = parts[1]
            cols = [c["name"] for c in state["profile"]["columns"]]
            if target not in cols:
                near = difflib.get_close_matches(target, cols, n=3, cutoff=0.6)
                if near:
                    console.print(ava.say(
                        f"I don't see '{target}'. Did you mean [cyan]{near[0]}[/cyan]?"
                        + (f" (or {', '.join(near[1:])})" if len(near) > 1 else "")))
                else:
                    console.print(ava.say(
                        f"I don't see '{target}'. I have: {', '.join(cols)}."))
                continue
            console.print(ava.say(f"On it — predicting [cyan]{target}[/cyan]. Cloning the swarm."))
            ctx = MissionContext(kind=MissionKind.TRAIN, goal=f"Predict '{target}'",
                                 data_source=source, output_dir=Path("./avaloka-model").expanduser(),
                                 budget=Budget(), target=target, deployable=True)
            _run_mission(console, ava, ctx, train_fireflies(True), run_train)
            continue

        if low == "batch":
            console.print(ava.say("Use `avaloka batch <data>` from the shell for the Ray swarm — "
                                  "I'll partition it and converge the exact statistics."))
            continue

        if low.startswith(("deploy", "infer", "serve")):
            console.print(ava.say("Once you've trained a model, run `avaloka infer ./avaloka-model "
                                  "--target kubernetes --apply` and I'll stand up the REST API."))
            continue

        # Otherwise: treat it as the goal and analyse.
        goal = raw[len("analyze"):].strip() if low.startswith("analyze") else raw
        goal = goal or "Understand this dataset and surface what matters"
        console.print(ava.say(f"Good — my goal: [italic]{goal}[/italic]. Sending in the swarm."))
        ctx = MissionContext(kind=MissionKind.ANALYZE, goal=goal, data_source=source,
                             output_dir=Path("./avaloka-analysis").expanduser(), budget=Budget())
        _run_mission(console, ava, ctx, analyze_fireflies(), run_analyze)
