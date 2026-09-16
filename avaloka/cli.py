"""The Avaloka CLI — a scriptable acquisition surface, in Avaloka's voice.

    avaloka chat                                   # talk to Avaloka
    avaloka analyze data.csv --goal "..."          # full or sampled live analysis
    avaloka train   data.csv --target y --deployable
    avaloka batch   data.parquet --goal "..."      # full analysis as a Ray swarm
    avaloka deploy  ./model --target kubernetes --approve
    avaloka infer   ./model --target kubernetes --apply   # MLflow -> docker -> k8s REST

Avaloka is a young, sharp data scientist: she sizes the job, reads the data
almost at a glance, clones herself across the work and converges the answer back
to you. Local execution is free and keeps your data on your machine.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from avaloka import workload
from avaloka.deploy import LEVELS, plan_deployment
from avaloka.mission.budget import Budget, BudgetExceeded
from avaloka.mission.context import ExecutionMode, MissionContext, MissionKind
from avaloka.missions import (MissionResult, analyze_fireflies, run_analyze,
                              run_train, train_fireflies)
from avaloka.persona import Avaloka
from avaloka.swarm import SwarmNarrator, make_progress
from avaloka.version import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Avaloka — an AI data scientist for your terminal.\n\n"
        "Point it at a file and ask a question:\n\n"
        "  avaloka analyze sales.csv --goal \"why did revenue drop in Q3?\"\n"
        "  avaloka train   sales.csv --target churned\n"
        "  avaloka chat    sales.csv\n\n"
        "Reads CSV, TSV, Parquet and Excel. Runs locally by default — your data "
        "stays on your machine. Every run writes a bundle you can open, rerun "
        "and hand to someone else."
    ),
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Print the Avaloka version and exit.",
    ),
) -> None:
    """Avaloka CLI entry point."""


# --------------------------------------------------------------------------
# Failing usefully.
#
# Every traceback the CLI prints is a bug report aimed at the wrong person.
# The user mistyped a column name or pointed at a directory; they need the
# name of the problem and the next thing to try, not a pandas stack frame.
# --------------------------------------------------------------------------

#: Metrics the training mission knows how to optimise. Kept next to the check
#: that uses it so a new metric cannot be added without appearing in the error.
VALID_METRICS = ("roc_auc", "f1", "f1_macro", "accuracy", "r2", "rmse", "mae")


def _die(problem: str, *, fix: str | None = None, code: int = 1) -> "typer.Exit":
    console.print(f"[red]{problem}[/red]")
    if fix:
        console.print(f"[dim]{fix}[/dim]")
    return typer.Exit(code=code)


def _did_you_mean(value: str, options: list[str], n: int = 3) -> list[str]:
    """Closest spellings, for the overwhelmingly common typo case."""
    import difflib

    lowered = {o.lower(): o for o in options}
    hit = difflib.get_close_matches(value.lower(), list(lowered), n=n, cutoff=0.6)
    return [lowered[h] for h in hit]


def _columns_of(path: str) -> list[str]:
    try:
        from avaloka.io.loader import load_dataset

        return [str(c) for c in load_dataset(path).frame.columns]
    except Exception:
        return []


def _check_target(data: str, target: str) -> None:
    """Fail before the mission starts, not inside pandas five agents later."""
    columns = _columns_of(data)
    if not columns or target in columns:
        return
    suggestions = _did_you_mean(target, columns)
    fix = f"Did you mean: {', '.join(suggestions)}?" if suggestions else None
    shown = ", ".join(columns[:12]) + ("…" if len(columns) > 12 else "")
    raise _die(
        f"There is no column named {target!r} in this dataset.",
        fix=f"{fix + chr(10) if fix else ''}Columns available: {shown}",
    )


def _check_metric(metric: str | None) -> None:
    if metric is None or metric in VALID_METRICS:
        return
    suggestions = _did_you_mean(metric, list(VALID_METRICS))
    fix = f"Did you mean: {', '.join(suggestions)}?" if suggestions else None
    raise _die(
        f"{metric!r} is not a metric I know how to optimise.",
        fix=f"{fix + chr(10) if fix else ''}Choose one of: {', '.join(VALID_METRICS)}",
    )


def _load_or_die(data: str):
    """Resolve a dataset, turning every known failure into a usable sentence."""
    from avaloka.io import SourceError
    from avaloka.io.loader import UnusableDataset

    try:
        return _resolve(data)
    except (FileNotFoundError, SourceError) as exc:
        raise _die(
            f"I can't reach that dataset: {exc}",
            fix="Check the path, or pass a CSV, TSV, Parquet or Excel file.",
        )
    except UnusableDataset as exc:
        raise _die(f"I read {data}, but it isn't something I can analyse.", fix=str(exc))
    except IsADirectoryError:
        raise _die(
            f"{data} is a directory, not a dataset.",
            fix="Point me at a single CSV, TSV, Parquet or Excel file inside it.",
        )
    except PermissionError as exc:
        raise _die(f"I'm not allowed to read that file: {exc}")
    except UnicodeDecodeError:
        raise _die(
            f"{data} isn't text I can decode.",
            fix="It may be binary or in an unusual encoding — re-export it as UTF-8 CSV.",
        )
    except Exception as exc:
        # Parser errors carry the useful detail (line number, field counts) in
        # their message; the traceback around it adds nothing for the user.
        raise _die(
            f"I couldn't read {data}: {type(exc).__name__}: {exc}",
            fix="If the file has ragged rows or an unusual delimiter, fix the export "
                "and try again. Run with AVALOKA_TRACEBACK=1 to see the full trace.",
        )


def _prepare_output(output: Path) -> Path:
    """Fail on an unwritable destination before doing the work, not after."""
    out = output.expanduser()
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".avaloka-write-test"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise _die(
            f"I can't write results to {out}: {exc.strerror or exc}",
            fix="Pick a different --output directory.",
        )
    return out


def _mode(local: bool, execution: str | None) -> ExecutionMode:
    if local or not execution:
        return ExecutionMode.LOCAL
    try:
        return ExecutionMode(execution)
    except ValueError:
        return ExecutionMode.LOCAL


def _avaloka(quiet: bool, llm: bool) -> Avaloka | None:
    return None if quiet else Avaloka(llm=llm)


def _resolve(data: str):
    """Resolve any source URI (path, database, cloud object store) to a local file.

    Returns ``(local_path, workload_plan, resolved)`` so every command can size
    and analyse a dataset regardless of where it physically lives.
    """
    from avaloka.io import materialize
    resolved = materialize(data)
    local = str(resolved.local_path)
    return local, workload.route(local), resolved


def _plain_progress():
    def fn(event: str, message: str) -> None:
        if event == "start":
            console.print(f"  [dim]·[/dim] firefly [cyan]{message}[/cyan] activating…")
        else:
            console.print(f"  [green]✓[/green] {message}")
    return fn


def _print_economics(result: MissionResult) -> None:
    econ = result.economics
    table = Table(title="Mission economics", show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column(justify="right", style="bold")
    for label, value in econ.rows():
        table.add_row(label, value)
    console.print(Panel(table, border_style="green", expand=False))


def _print_fireflies(result: MissionResult) -> None:
    t = Table(title="Avaloka clones activated (measured work units)")
    t.add_column("Clone / firefly", style="cyan")
    t.add_column("Replaces")
    t.add_column("Manual min", justify="right")
    t.add_column("Cost $", justify="right")
    for u in result.context.ledger.units:
        t.add_row(u.firefly, u.human_role, f"{u.manual_minutes:.0f}",
                  f"{u.compute_cost_usd + u.model_cost_usd:.4f}")
    console.print(t)


def _print_answer(result: MissionResult) -> None:
    """What the mission concluded, in the terminal, before the accounting.

    The bundle always held these; the console did not, so a run ended on an
    economic multiplier and a bare verdict with no statement of what was found.
    """
    bb = result.context.blackboard
    findings = bb.get("findings") or []
    if findings:
        console.print()
        console.print("[bold]What I found[/bold]")
        for item in findings:
            console.print(f"  • {item}")

    limitations = bb.get("limitations") or []
    if limitations:
        console.print()
        console.print("[bold]What to be careful about[/bold]")
        for item in limitations:
            console.print(f"  [yellow]•[/yellow] {item}")


def _print_verdict(result: MissionResult) -> None:
    """The verdict plus the checks that produced it — never the word alone."""
    v = result.summary.get("verdict")
    if not v:
        return
    lvl = result.summary.get("max_safe_deployment_level")
    colour = {"pass": "green", "warn": "yellow", "fail": "red"}.get(v, "white")
    extra = f"; max safe deployment level [bold]{lvl}/4[/bold]" if lvl else ""
    console.print()
    console.print(f"Validation verdict: [{colour}]{v.upper()}[/{colour}]{extra}")

    checks = (result.context.blackboard.get("validation") or {}).get("checks") or []
    for check in checks:
        if check.get("status") in ("warn", "fail"):
            mark = "[red]✗[/red]" if check["status"] == "fail" else "[yellow]![/yellow]"
            console.print(f"  {mark} {check.get('name')}: {check.get('detail')}")


def _print_next_steps(result: MissionResult) -> None:
    out = result.summary["output_dir"]
    console.print()
    console.print(f"[bold green]Deliverables →[/bold green] {out}")
    console.print(f"  [dim]open {out}/executive_report.html[/dim]   the decision summary")
    console.print(f"  [dim]cat  {out}/README.md[/dim]               findings and limitations")
    console.print(f"  [dim]python {out}/analysis.py[/dim]           reproduce this run")


def _finish(result: MissionResult, ava: Avaloka | None) -> None:
    _print_answer(result)
    _print_verdict(result)
    console.print()
    _print_fireflies(result)
    console.print()
    _print_economics(result)
    _print_next_steps(result)
    if ava is not None:
        console.print(ava.converge(result.summary))


def _run_mission(ctx: MissionContext, fireflies: list[type], run_fn, ava: Avaloka | None,
                 wl: "workload.WorkloadPlan", quiet: bool) -> MissionResult:
    """Shared persona+swarm wrapper around a mission run."""
    if ava is not None:
        console.print(ava.first_glance({"n_rows": wl.n_rows, "n_cols": "several"})
                      if wl.n_rows else ava.greet())
        narrator = SwarmNarrator(console.print, voice=ava)
        narrator.announce(len(fireflies))
        on_progress = make_progress(narrator)
    else:
        narrator = None
        on_progress = _plain_progress()

    result = run_fn(ctx, on_progress)

    if narrator is not None:
        narrator.converge(ctx.ledger)
    # Real, magical-but-measured observations, surfaced after the swarm returns.
    profile = ctx.blackboard.get("profile")
    if ava is not None and profile:
        console.print()
        console.print(ava.say("Here's what jumped out at me:"))
        for line in ava.observe(profile):
            console.print("  " + line)
    _finish(result, ava)
    if wl.recommend_batch and ctx.kind == MissionKind.ANALYZE:
        console.print(Panel(
            f"[bold]Full-dataset option[/bold]\nThis was the sampled live read. For the exact "
            f"answer over all {wl.n_rows:,} rows, run it as a Ray swarm:\n"
            f"  [cyan]avaloka batch {ctx.data_source} --goal \"{ctx.goal}\"[/cyan]",
            border_style="magenta", expand=False))
    return result


# --------------------------------------------------------------------------
@app.command()
def analyze(
    data: str = typer.Argument(..., help="Path to a CSV or Parquet dataset."),
    goal: str = typer.Option(..., "--goal", "-g", help="The decision this analysis must inform."),
    budget: float | None = typer.Option(None, "--budget", "-b", help="Hard cost ceiling (USD)."),
    output: Path = typer.Option(Path("./avaloka-analysis"), "--output", "-o"),
    local: bool = typer.Option(True, "--local/--managed"),
    execution: str | None = typer.Option(None, "--execution", help="local|byoc|managed."),
    tier: str = typer.Option("community", "--tier"),
    hourly_rate: float = typer.Option(80.0, "--hourly-rate"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress Avaloka's voice (machine output only)."),
    llm: bool = typer.Option(False, "--llm", help="Let Avaloka phrase things with Claude (needs ANTHROPIC_API_KEY)."),
):
    """Turn a dataset and a question into a defensible analysis bundle."""
    output = _prepare_output(output)
    data, wl, _resolved = _load_or_die(data)

    ava = _avaloka(quiet, llm)
    if ava is not None:
        console.print(ava.greet())
        console.print(ava.workload_read(wl))
    else:
        console.print(Panel(f"[bold]Avaloka Analyze[/bold] · {goal}\n[dim]workload:[/dim] "
                            f"{wl.lane.value} ({wl.n_rows:,} rows)", border_style="cyan"))

    ctx = MissionContext(
        kind=MissionKind.ANALYZE, goal=goal, data_source=data, output_dir=output,
        budget=Budget(limit_usd=budget), execution_mode=_mode(local, execution), tier=tier,
        loaded_hourly_rate=hourly_rate)
    ctx.blackboard["workload"] = wl.as_dict()

    try:
        _run_mission(ctx, analyze_fireflies(), run_analyze, ava, wl, quiet)
    except BudgetExceeded as exc:
        console.print(f"[red]Budget exceeded:[/red] {exc}")
        raise typer.Exit(code=2)


@app.command()
def train(
    data: str = typer.Argument(..., help="Path to a CSV or Parquet dataset."),
    target: str = typer.Option(..., "--target", "-t", help="Target column to predict."),
    metric: str | None = typer.Option(None, "--metric", "-m", help="roc_auc|f1|f1_macro|accuracy|r2|rmse|mae."),
    goal: str | None = typer.Option(None, "--goal", "-g"),
    objective: str | None = typer.Option(None, "--objective"),
    max_cost: float | None = typer.Option(None, "--max-cost"),
    latency: float | None = typer.Option(None, "--deployment-latency-ms"),
    deployable: bool = typer.Option(False, "--deployable/--no-deployable"),
    output: Path = typer.Option(Path("./avaloka-model"), "--output", "-o"),
    local: bool = typer.Option(True, "--local/--managed"),
    execution: str | None = typer.Option(None, "--execution"),
    tier: str = typer.Option("community", "--tier"),
    hourly_rate: float = typer.Option(80.0, "--hourly-rate"),
    quiet: bool = typer.Option(False, "--quiet"),
    llm: bool = typer.Option(False, "--llm"),
):
    """Train, validate and optionally package a model to predict one column."""
    _check_metric(metric)
    output = _prepare_output(output)
    data, wl, _resolved = _load_or_die(data)
    _check_target(data, target)

    ava = _avaloka(quiet, llm)
    if ava is not None:
        console.print(ava.greet())
        console.print(ava.workload_read(wl))
    else:
        console.print(Panel(f"[bold]Avaloka Predict[/bold] · target={target}\n[dim]workload:[/dim] "
                            f"{wl.lane.value} ({wl.n_rows:,} rows)", border_style="cyan"))

    ctx = MissionContext(
        kind=MissionKind.TRAIN, goal=goal or f"Predict '{target}'", data_source=data,
        output_dir=output, budget=Budget(limit_usd=max_cost), target=target,
        metric=metric, objective=objective, deployable=deployable, deployment_latency_ms=latency,
        execution_mode=_mode(local, execution), tier=tier, loaded_hourly_rate=hourly_rate)
    ctx.blackboard["workload"] = wl.as_dict()

    try:
        result = _run_mission(ctx, train_fireflies(deployable), run_train, ava, wl, quiet)
    except BudgetExceeded as exc:
        console.print(f"[red]Budget exceeded:[/red] {exc}")
        raise typer.Exit(code=2)

    if deployable and result.summary.get("max_safe_deployment_level"):
        console.print(Panel(
            "[bold]Ship it[/bold]\nThe deployment package is ready. To stand up a REST API on "
            f"Kubernetes from the MLflow-packaged model:\n  [cyan]avaloka infer {output} "
            "--target kubernetes --apply[/cyan]", border_style="magenta", expand=False))


@app.command()
def chat(
    data: str | None = typer.Argument(None, help="Optional dataset to start from."),
    llm: bool = typer.Option(False, "--llm", help="Use Claude for free-form conversation (needs ANTHROPIC_API_KEY)."),
):
    """Ask questions about a dataset conversationally, one turn at a time."""
    from avaloka.chat import run_chat
    run_chat(console, data=data, llm=llm)


@app.command()
def batch(
    data: str = typer.Argument(..., help="Path to a (large) CSV or Parquet dataset."),
    goal: str = typer.Option("Full-dataset analysis", "--goal", "-g"),
    partitions: int = typer.Option(8, "--partitions", "-p", help="Number of swarm partitions."),
    target: str = typer.Option("local", "--target", help="local|kubernetes (KubeRay)."),
    image: str = typer.Option("avaloka-batch:latest", "--image", help="Ray worker image (k8s only)."),
    output: Path = typer.Option(Path("./avaloka-batch"), "--output", "-o"),
    apply: bool = typer.Option(False, "--apply", help="Actually submit the RayJob (kubernetes target)."),
    quiet: bool = typer.Option(False, "--quiet"),
    llm: bool = typer.Option(False, "--llm"),
):
    """Run the full (unsampled) analysis as an Apache Ray swarm; converge the result."""
    from avaloka import ray_batch
    from avaloka.serving import _have, _run

    ava = _avaloka(quiet, llm)
    from avaloka.io import SourceError
    try:
        data, wl, _resolved = _resolve(data)
    except (FileNotFoundError, SourceError) as exc:
        console.print(f"[red]I can't reach that dataset:[/red] {exc}")
        raise typer.Exit(code=1)
    if ava is not None:
        console.print(ava.greet())
        console.print(ava.say(f"Batch mode. I'll split {wl.n_rows:,} rows into {partitions} "
                              "partitions, send a clone to each, and converge the exact statistics."))

    out = output.expanduser(); out.mkdir(parents=True, exist_ok=True)
    res = ray_batch.run_ray_local(data, n_partitions=partitions)

    from avaloka.util import write_json
    write_json(out / "batch_profile.json", res.converged)
    artifacts = ray_batch.write_batch_artifacts(out, data, image=image, n_partitions=partitions)

    t = Table(title=f"Converged full-dataset profile ({res.engine}, {res.n_partitions} partitions)")
    t.add_column("Column", style="cyan"); t.add_column("Role"); t.add_column("Missing%", justify="right")
    t.add_column("Summary")
    for c in res.converged["columns"]:
        if c["role"] == "numeric":
            summ = f"μ={c['mean']:.3g} σ={c['std']:.3g} [{c['min']:.3g}, {c['max']:.3g}]"
        else:
            top = c.get("top_values", [])[:2]
            summ = ", ".join(f"{x['value']}({x['count']})" for x in top)
        t.add_row(c["name"], c["role"], f"{c['missing_pct'] * 100:.1f}", summ)
    console.print(t)
    console.print(f"[bold]Exact rows analysed:[/bold] {res.converged['n_rows']:,} "
                  "(no sampling — partial sums combined losslessly).")

    if target == "kubernetes":
        console.print("\n[bold]KubeRay submission:[/bold]")
        cmds = ray_batch.submit_commands(artifacts["manifest"])
        for c in cmds:
            console.print(f"  [dim]{c}[/dim]" if c.startswith("#") else f"  [dim]$[/dim] {c}")
        if apply:
            if _have("kubectl"):
                ok, outp = _run(["kubectl", "apply", "-f", str(artifacts["manifest"])])
                console.print(f"  kubectl apply: {'[green]ok[/green]' if ok else '[red]failed[/red]'} — {outp[-160:]}")
            else:
                console.print("  [yellow]kubectl not found — submit the manifest from a cluster context.[/yellow]")
    if ava is not None:
        console.print(ava.say("All partitions are back and merged. That's the whole dataset, "
                              "exactly — not an estimate."))
    console.print(f"\n[bold green]Batch artifacts →[/bold green] {out}")


@app.command()
def infer(
    package: str = typer.Argument(..., help="Path to a train-mission output directory (--deployable)."),
    target: str = typer.Option("kubernetes", "--target", help="kubernetes (REST via Deployment+Service)."),
    image: str | None = typer.Option(None, "--image", help="Override the container image tag."),
    mlflow_uri: str | None = typer.Option(None, "--mlflow-uri", help="MLflow tracking URI to register/resolve the model."),
    registry: str | None = typer.Option(None, "--registry", help="Container registry to push to (e.g. ghcr.io/acme)."),
    apply: bool = typer.Option(False, "--apply", help="Actually build the image and apply the manifest."),
    push: bool = typer.Option(False, "--push", help="docker push after build (needs --registry)."),
    quiet: bool = typer.Option(False, "--quiet"),
    llm: bool = typer.Option(False, "--llm"),
):
    """Stand up a REST inference API on Kubernetes from a trained MLflow model."""
    from avaloka import serving
    from avaloka.util import write_json

    ava = _avaloka(quiet, llm)
    try:
        plan = serving.prepare(package, target=target, image=image, mlflow_uri=mlflow_uri, registry=registry)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    if ava is not None:
        console.print(ava.say(f"Turning the model into a live REST API: MLflow → Docker "
                              f"({plan.image}) → Kubernetes → endpoint."))

    t = Table(title="Inference pipeline: MLflow → Docker → Kubernetes → REST")
    t.add_column("Stage", style="cyan"); t.add_column("Action"); t.add_column("Detail")
    for s in plan.steps:
        t.add_row(s["stage"], s["action"], s["detail"])
    console.print(t)
    console.print("\n[bold]Commands:[/bold]")
    for c in plan.commands:
        console.print(f"  [dim]$[/dim] {c}")

    if apply:
        console.print("\n[bold]Applying…[/bold]")
        plan = serving.apply(plan, push=push)
        for n in plan.notes:
            console.print(f"  [dim]·[/dim] {n}")
        if plan.endpoint:
            console.print(Panel(
                f"[bold green]REST API live[/bold green]  {plan.endpoint}\n"
                f"  health:  [cyan]curl {plan.endpoint}/health[/cyan]\n"
                f"  predict: [cyan]{serving.curl_example(plan.endpoint, plan.package_dir)}[/cyan]",
                border_style="green", expand=False))
        else:
            console.print("[yellow]Endpoint not resolved yet — see notes above "
                          "(LoadBalancer may still be provisioning).[/yellow]")
    else:
        console.print("\n[dim]This was a plan. Re-run with --apply to build and deploy.[/dim]")

    write_json(Path(plan.package_dir) / "deployment" / "inference_plan.json", plan.as_dict())
    if ava is not None and not apply:
        console.print(ava.say("Say the word (--apply) and I'll stand it up for real."))


@app.command()
def deploy(
    package: str = typer.Argument(..., help="Path to a train-mission output directory."),
    target: str = typer.Option("local", "--target", help="local|docker|kubernetes|ray."),
    level: int = typer.Option(2, "--level", help="Requested deployment level (1-4)."),
    approve: bool = typer.Option(False, "--approve", help="Human approval required for Level 4 (production)."),
    scale_to_zero: bool = typer.Option(False, "--scale-to-zero"),
    max_monthly_cost: float | None = typer.Option(None, "--max-monthly-cost"),
):
    """Check a trained model against the deployment gates and write a deploy plan."""
    try:
        plan = plan_deployment(package, target, requested_level=level, approved=approve,
                               scale_to_zero=scale_to_zero, max_monthly_cost=max_monthly_cost)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    from avaloka.util import write_json
    write_json(Path(plan.package_dir) / "deployment" / "deploy_plan.json", plan.as_dict())

    t = Table(title="Deployment-level gates")
    t.add_column("Level"); t.add_column("Name"); t.add_column("Status"); t.add_column("Missing")
    for g in plan.gates:
        status = "[green]pass[/green]" if g["passed"] else "[yellow]blocked[/yellow]"
        t.add_row(str(g["level"]), g["name"], status, ", ".join(g["missing_artifacts"]) or "—")
    console.print(t)
    console.print(f"\nValidator cap: [bold]Level {plan.validator_cap}/4[/bold]  "
                  f"({LEVELS[plan.validator_cap]})")
    console.print(f"Requested: Level {plan.requested_level}  →  "
                  f"Achieved: [bold]Level {plan.achieved_level}[/bold] ({LEVELS[plan.achieved_level]})")
    if plan.blocked_reason:
        console.print(Panel(plan.blocked_reason, title="Gate", border_style="yellow"))
    console.print("\n[bold]Run:[/bold]")
    for cmd in plan.commands:
        console.print(f"  [dim]$[/dim] {cmd}" if not cmd.startswith("#") else f"  [dim]{cmd}[/dim]")


@app.command()
def coordinate(
    data: str = typer.Argument(..., help="Dataset URI: a path, sqlite:///db#table, s3://…, gs://…, az://…"),
    goal: str = typer.Option(..., "--goal", "-g", help="The decision the fleet must inform."),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of sub-Avalokas in the fleet."),
    kind: str = typer.Option("analyze", "--kind", help="analyze|train."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target column (train)."),
    metric: str | None = typer.Option(None, "--metric", "-m"),
    deployable: bool = typer.Option(False, "--deployable/--no-deployable"),
    budget: float | None = typer.Option(None, "--budget", "-b", help="Per-sub-Avaloka cost ceiling (USD)."),
    output: Path = typer.Option(Path("./avaloka-fleet"), "--output", "-o"),
    quiet: bool = typer.Option(False, "--quiet"),
    llm: bool = typer.Option(False, "--llm"),
):
    """Multi-Avaloka — one main Avaloka fans a fleet of self-clones across the same
    dataset (each runs its own swarm on a shard) and converges one answer.

    The source may live anywhere: a local file, a database table/query or cloud
    object storage — it's resolved to a local file before the fleet is dispatched.
    """
    from avaloka.coordinator import MainAvaloka
    from avaloka.io import SourceError
    from avaloka.mission.context import MissionKind

    ava = _avaloka(quiet, llm)
    mk = MissionKind.TRAIN if kind == "train" else MissionKind.ANALYZE

    def on_progress(event: str, message: str) -> None:
        if quiet:
            return
        if event == "announce":
            console.print(ava.say(f"Cloning myself into {message} sub-Avalokas — each takes a shard "
                                  "of the same data and runs her own swarm.") if ava
                          else f"[magenta]Fleet[/magenta] dispatching {message} sub-Avalokas")
        elif event == "dispatch":
            console.print(f"  [magenta]✦[/magenta] {message}")
        elif event == "report":
            console.print(f"    [green]↩[/green] [dim]{message}[/dim]")
        elif event == "converge":
            console.print(ava.say(f"All copies are back. I merged them into one answer — {message}.")
                          if ava else f"[magenta]Fleet[/magenta] converged — {message}")

    if ava is not None:
        console.print(ava.greet())
    try:
        main = MainAvaloka()
        cr = main.coordinate(data, goal=goal, output_dir=output.expanduser(), workers=workers,
                             kind=mk, target=target, metric=metric, deployable=deployable,
                             budget=budget, on_progress=on_progress)
    except (FileNotFoundError, SourceError) as exc:
        console.print(f"[red]I can't reach that source:[/red] {exc}")
        raise typer.Exit(code=1)

    t = Table(title=f"Converged full-dataset profile ({cr.workers} sub-Avalokas, {cr.n_rows:,} rows)")
    t.add_column("Column", style="cyan"); t.add_column("Role"); t.add_column("Missing%", justify="right")
    t.add_column("Summary")
    for c in cr.converged_profile["columns"]:
        if c["role"] == "numeric":
            summ = f"μ={c['mean']:.3g} σ={c['std']:.3g} [{c['min']:.3g}, {c['max']:.3g}]"
        else:
            summ = ", ".join(f"{x['value']}({x['count']})" for x in c.get("top_values", [])[:2])
        t.add_row(c["name"], c["role"], f"{c['missing_pct'] * 100:.1f}", summ)
    console.print(t)

    s = Table(title="Fleet (each sub-Avaloka ran her own firefly swarm)")
    s.add_column("Sub-Avaloka", style="cyan"); s.add_column("Rows", justify="right")
    s.add_column("Verdict"); s.add_column("Manual min", justify="right")
    for sh in cr.shards:
        s.add_row(f"Ava-{sh.shard_id:02d}", f"{sh.n_rows:,}", str(sh.verdict or "—"),
                  f"{sh.manual_minutes:.0f}")
    console.print(s)

    colour = {"pass": "green", "warn": "yellow", "fail": "red"}.get(cr.verdict, "white")
    console.print(f"\nConverged verdict: [{colour}]{cr.verdict.upper()}[/{colour}]  "
                  f"(max safe deployment level [bold]{cr.max_safe_deployment_level}/4[/bold])")
    if cr.selected_model:
        m = cr.selected_model
        console.print(f"Selected model: [bold]{m['selected']}[/bold] "
                      f"({m['metric']} {m['score']:.4f}, from shard {m['from_shard']})")
    econ = cr.economics
    console.print(f"Fleet effort replaced: [bold]{econ.estimated_manual_effort_hours:.1f}h[/bold] · "
                  f"multiplier [bold]{econ.economic_multiplier:.1f}x[/bold]")
    console.print(f"\n[bold green]Fleet deliverables →[/bold green] {output}")


@app.command()
def benchmark(
    family: str | None = typer.Option(None, "--family", help="data_engineering|data_science."),
    kaggle: bool = typer.Option(False, "--kaggle", help="Include Kaggle-dataset tasks (if downloaded)."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Artifacts dir (default: temp)."),
    json_out: Path | None = typer.Option(None, "--json", help="Write the scorecard JSON here."),
):
    """Run the data-engineering + data-science benchmark against this CLI."""
    import json as _json
    import tempfile

    from avaloka.benchmark.runner import BenchmarkRunner
    from avaloka.benchmark.suite import all_tasks

    tasks = all_tasks(include_kaggle=kaggle)
    if family:
        tasks = [t for t in tasks if t.family == family]
    base = output.expanduser() if output else Path(tempfile.mkdtemp(prefix="avaloka-bench-"))
    console.print(f"Running [bold]{len(tasks)}[/bold] benchmark tasks → [dim]{base}[/dim]")
    card = BenchmarkRunner(base, verbose=False).run(tasks)

    t = Table(title="Avaloka benchmark scorecard")
    t.add_column("Task", style="cyan"); t.add_column("Source"); t.add_column("Score", justify="right")
    t.add_column("Result")
    for r in card.results:
        skipped = r.error and "unavailable" in r.error
        mark = "[green]PASS[/green]" if r.passed else ("[yellow]SKIP[/yellow]" if skipped else "[red]FAIL[/red]")
        t.add_row(r.task, r.source_scheme, f"{r.score:.2f}", mark)
    console.print(t)
    for fam, score in card.by_family().items():
        console.print(f"  [dim]{fam}[/dim] {score:.2f}")
    console.print(f"[bold]OVERALL[/bold] {card.overall_score:.2f}  "
                  f"({card.as_dict()['n_passed']}/{card.as_dict()['n_runs']} runs passed)")
    if json_out:
        json_out.expanduser().write_text(_json.dumps(card.as_dict(), indent=2))
        console.print(f"[green]Scorecard →[/green] {json_out}")
    hard = [r for r in card.failures() if not (r.error and "unavailable" in r.error)]
    if hard:
        raise typer.Exit(code=1)


@app.command()
def version():
    """Print the Avaloka version."""
    console.print(f"avaloka {__version__}")


if __name__ == "__main__":
    app()


# --------------------------------------------------------------------------


def main() -> None:
    """Entry point with a boundary that turns crashes into sentences.

    The dataset is loaded lazily inside the mission, so a malformed file
    surfaces several agents deep — well past the pre-flight checks. Without a
    boundary here the user sees a pandas stack frame and has to guess which of
    their inputs caused it. Set ``AVALOKA_TRACEBACK=1`` to get the trace back
    when you are the one debugging Avaloka rather than your data.
    """
    import os
    import sys

    try:
        app()
    except (typer.Exit, SystemExit):
        # click has already translated these into an exit status; re-raising
        # typer.Exit outside click's runtime just prints it as an exception.
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow] Nothing was left half-written.")
        sys.exit(130)
    except Exception as exc:
        if os.getenv("AVALOKA_TRACEBACK"):
            raise
        from avaloka.io.loader import UnusableDataset

        if isinstance(exc, UnusableDataset):
            console.print(f"[red]That dataset isn't something I can analyse.[/red]")
            console.print(f"[dim]{exc}[/dim]")
        else:
            name = type(exc).__name__
            console.print(f"[red]I hit a problem I couldn't recover from:[/red] {name}: {exc}")
            console.print("[dim]If the file has ragged rows, mixed delimiters or an unusual "
                          "encoding, fixing the export is usually the quickest path. "
                          "Run again with AVALOKA_TRACEBACK=1 for the full trace.[/dim]")
        sys.exit(1)
