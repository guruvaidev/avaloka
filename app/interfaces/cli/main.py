"""`avaloka` CLI.

A thin surface that parses flags into an intent dict and hands off to the
shared mission service. It deliberately contains no analytical logic — the same
intent through the MCP server (or the deployed API via ``--endpoint``) produces
the same mission and plan.

Run in-process:
    python -m app.interfaces.cli.main plan customers.parquet \
        --goal "Explain churn drivers" --rows 84000000 --max-cost 20

Drive a deployed backend (identical JSON):
    python -m app.interfaces.cli.main plan customers.parquet --goal "…" --json \
        --endpoint http://localhost:9000 --token "$AVALOKA_API_TOKEN"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.interfaces.service import plan_mission, planned_to_dict


def build_intent(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate parsed CLI args into the interface-agnostic intent dict."""
    intent: Dict[str, Any] = {
        "source_uri": args.source,
        "format": args.format,
        "goal_type": args.goal_type,
        "goal_description": args.goal,
        "target": args.target,
        "primary_metric": args.metric,
        "mode": args.mode,
        "full_data_engine": args.execution,
        "kubernetes_context": args.cluster_context,
        "cloud_target": args.cloud,
        "max_cost_usd": args.max_cost,
        "max_runtime_minutes": args.max_runtime,
        "sampling_strategy": args.sampling,
        "confidence": args.confidence,
        "margin_error": args.margin_error,
        "output_location": args.output,
        # optional inspection stats so a cold `plan` still estimates well.
        "rows": args.rows,
        "columns": args.columns,
        "type_complexity": args.type_complexity,
        "rare_class_fraction": args.rare_class_fraction,
    }
    if args.detect_leakage:
        intent["detect_leakage"] = True
    return {k: v for k, v in intent.items() if v is not None}


def plan_remote(intent: Dict[str, Any], endpoint: str, token: Optional[str]) -> Dict[str, Any]:
    """POST the intent to a deployed API's /api/missions/plan and return its JSON.

    The API returns ``planned_to_dict()`` — the same shape the local path produces —
    so local and remote are byte-identical for the same intent.
    """
    import requests

    url = endpoint.rstrip("/") + "/api/missions/plan"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.post(url, json=intent, headers=headers, timeout=30)
    except requests.RequestException as e:
        raise SystemExit(f"error: could not reach avaloka API at {endpoint}: {e}")
    if resp.status_code != 200:
        raise SystemExit(f"error: API {url} returned {resp.status_code}: {resp.text.strip()[:300]}")
    return resp.json()


def _money(band) -> str:
    return f"${band[0]:.2f}–${band[1]:.2f}"


def render(planned: Dict[str, Any]) -> str:
    """Render the planned_to_dict shape (works for both local and remote results)."""
    mission, plan, est = planned["mission"], planned["plan"], planned["estimate"]
    spec = mission.get("spec", {})
    goal = spec.get("goal", {})
    source = spec.get("source", {})
    name = mission.get("metadata", {}).get("name", plan.get("mission_name", ""))
    lines: List[str] = []
    lines.append(f"Mission: {name}  ({goal.get('type', '')})")
    lines.append(f"Source:  {source.get('uri', '')}")
    if goal.get("description"):
        lines.append(f"Goal:    {goal['description']}")
    lines.append("")
    lines.append("Mission plan")
    lines.append("─" * 56)
    for i, step in enumerate(plan.get("steps", []), 1):
        lines.append(f"  {i}. {step}")
    lines.append("")
    lines.append(f"Recommended mode:  {plan.get('mode')}  (target: {plan.get('target')}"
                 + (f", cloud: {plan['cloud_target']}" if plan.get("cloud_target") else "") + ")")
    lines.append(f"Rationale:         {est.get('rationale')}")
    lines.append(f"Sampled estimate:  {_money(est['sampled_cost_usd'])}"
                 + (f", ~{est['sampled_rows']:,} rows" if est.get("sampled_rows") else ""))
    frt = est["full_runtime_minutes"]
    lines.append(f"Full-data pass:    {_money(est['full_cost_usd'])}, "
                 f"{frt[0]:.0f}–{frt[1]:.0f} min"
                 f"  ({'worthwhile' if est.get('full_pass_worthwhile') else 'not yet worth it'})")
    if plan.get("requires_approval"):
        lines.append(f"Approval needed:   {plan.get('approval_reason')}")
    for note in list(est.get("notes", [])) + list(plan.get("notes", [])):
        lines.append(f"Note:              {note}")
    return "\n".join(lines)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", help="dataset URI (gs://, s3://, file path, …)")
    p.add_argument("--goal", help="natural-language objective")
    p.add_argument("--goal-type", dest="goal_type",
                   choices=["etl", "data_quality", "analysis", "predictive_model", "inference"])
    p.add_argument("--target", help="target column (for model goals)")
    p.add_argument("--metric", help="primary metric, e.g. pr_auc")
    p.add_argument("--mode", choices=["auto", "interactive", "sampled", "hybrid", "batch"])
    p.add_argument("--execution", choices=["local", "ray"], help="full-data engine")
    p.add_argument("--cluster-context", dest="cluster_context")
    p.add_argument("--cloud", choices=["gcp", "aws", "azure", "local"],
                   help="cloud the mission targets (environment-awareness)")
    p.add_argument("--max-cost", dest="max_cost", type=float)
    p.add_argument("--max-runtime", dest="max_runtime", type=int, help="minutes")
    p.add_argument("--sampling", choices=["auto", "none", "random", "stratified", "temporal"])
    p.add_argument("--confidence", type=float)
    p.add_argument("--margin-error", dest="margin_error", type=float)
    p.add_argument("--format")
    p.add_argument("--output", help="output location (ETL sink / results)")
    p.add_argument("--detect-leakage", dest="detect_leakage", action="store_true")
    # inspection stats (until the profiler agent supplies them automatically)
    p.add_argument("--rows", type=int)
    p.add_argument("--columns", type=int)
    p.add_argument("--type-complexity", dest="type_complexity", type=float,
                   help="0=simple numerics … 1=text/images")
    p.add_argument("--rare-class-fraction", dest="rare_class_fraction", type=float)
    p.add_argument("--json", action="store_true", help="print canonical mission+plan+estimate JSON")
    # Remote mode: drive a deployed backend instead of planning in-process.
    p.add_argument("--endpoint", default=os.getenv("AVALOKA_API_URL"),
                   help="deployed avaloka API base URL (env AVALOKA_API_URL). "
                        "When set, planning runs on the backend, not in-process.")
    p.add_argument("--token", default=os.getenv("AVALOKA_API_TOKEN"),
                   help="HS256 JWT bearer token for the API (env AVALOKA_API_TOKEN).")


def _add_viz(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", nargs="?", help="dataset CSV to visualize (sample-based)")
    p.add_argument("--target", help="target/label column (for supervised framing)")
    p.add_argument("--rows", type=int, default=500, help="how many sample rows to read")
    p.add_argument("--out", help="directory to write the rendered artifact into")
    p.add_argument("--open", dest="open_mode", choices=["browser", "link", "none"],
                   help="how to surface the result: browser (default on a TTY), link "
                        "(print a file:// URL — headless/CI default), or none")
    p.add_argument("--json", action="store_true",
                   help="print the visualization_config JSON instead of rendering")
    p.add_argument("--planner-graph", dest="planner_graph", metavar="THREAD_ID",
                   help="fetch the planner-graph PNG for a deployed thread (needs --endpoint)")
    p.add_argument("--endpoint", default=os.getenv("AVALOKA_API_URL"),
                   help="deployed avaloka API base URL (env AVALOKA_API_URL)")
    p.add_argument("--token", default=os.getenv("AVALOKA_API_TOKEN"),
                   help="HS256 JWT bearer token (env AVALOKA_API_TOKEN)")


def _load_sample_rows(source: str, limit: int) -> List[Dict[str, Any]]:
    """Read up to ``limit`` rows of a CSV into a list of dicts (the viz agent's input)."""
    import csv as _csv

    rows: List[Dict[str, Any]] = []
    with open(source, "r", encoding="utf-8", errors="ignore", newline="") as f:
        for i, row in enumerate(_csv.DictReader(f)):
            if i >= max(1, limit):
                break
            rows.append({k: _coerce(v) for k, v in row.items()})
    return rows


def _coerce(v: Any) -> Any:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def cmd_viz(args: argparse.Namespace) -> int:
    """Render a dataset's visualizations (or fetch the planner-graph PNG) and open them.

    Visualizations are JSON chart specs the terminal can't draw, so we render them to a
    self-contained HTML page and open it in the browser (or print a file:// link when
    headless). ``--planner-graph`` fetches the one real image the backend serves.
    """
    from app.interfaces import artifacts

    mode = artifacts.resolve_open_mode(getattr(args, "open_mode", None))

    if args.planner_graph:
        if not args.endpoint:
            raise SystemExit("error: --planner-graph requires --endpoint (the deployed API)")
        try:
            png = artifacts.fetch_planner_graph(
                args.endpoint, args.planner_graph, token=args.token, out_dir=args.out)
        except Exception as e:  # noqa: BLE001 - clean CLI error, not a stack trace
            raise SystemExit(f"error: could not fetch planner graph: {e}")
        print(f"planner graph: {artifacts.open_artifact(str(png), mode=mode)}")
        return 0

    if not args.source:
        raise SystemExit("error: a dataset CSV is required (or use --planner-graph <thread> --endpoint …)")
    rows = _load_sample_rows(args.source, args.rows)
    if not rows:
        raise SystemExit(f"error: no rows read from {args.source}")

    from app.agents.visualization_agent import build_visualization_config_from_sample
    cfg = build_visualization_config_from_sample(
        dataset_id=Path(args.source).stem, sample_rows=rows, target_column=args.target)

    if args.json:
        print(json.dumps(cfg, indent=2, sort_keys=True))
        return 0

    html = artifacts.visualization_to_html(cfg, rows, title=f"Avaloka — {Path(args.source).name}")
    path = artifacts.write_artifact(html, suffix=".html", out_dir=args.out)
    n = len(cfg.get("charts") or [])
    print(f"rendered {n} chart(s): {artifacts.open_artifact(str(path), mode=mode)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avaloka", description="Avaloka — AI data scientist")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_common(sub.add_parser("plan", help="show the mission plan + estimate without running"))
    _add_common(sub.add_parser("analyze", help="plan and (when wired) execute an analysis"))
    _add_viz(sub.add_parser("viz", help="render a dataset's visualizations and open them (browser/link)"))
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "viz":
        return cmd_viz(args)

    intent = build_intent(args)

    # Local and remote must return the identical planned_to_dict shape.
    if args.endpoint:
        result = plan_remote(intent, args.endpoint, args.token)
    else:
        result = planned_to_dict(plan_mission(intent))

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(render(result))
    if args.command == "analyze" and not result["plan"].get("implemented"):
        print("\n(analyze: local execution path is the next wire-up; "
              "distributed target is stubbed in this build.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
