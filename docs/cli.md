# Avaloka mission planner (CLI) & MCP

> **Not the `avaloka` command.** This document covers the *planning* interface,
> `python -m app.interfaces.cli.main`, which prices and routes a workload
> without executing it. If you want to actually analyse a file, train a model
> or chat about a dataset — `avaloka analyze`, `avaloka train`, `avaloka chat` —
> that is a different tool, documented in [README_CLI.md](../README_CLI.md).

Avaloka ships a mission-planning command-line interface. It compiles a
natural-language goal plus dataset characteristics into a **canonical mission**
and an **execution plan + cost/runtime estimate** — the same core the MCP server
and (in future) the REST surface use, so all front doors agree.

The CLI runs **in-process** (no backend required): it is the fastest way to see
how Avaloka would route and price a workload before committing compute.

## Invocation

```bash
python -m app.interfaces.cli.main <command> <source> [options]
```

There are two commands:

| Command | What it does |
| ------- | ------------ |
| `plan` | Show the mission plan + estimate **without** running anything. |
| `analyze` | Plan and (when wired) execute an analysis. Execution is being wired up; today it plans and reports the execution path. |

`<source>` is a dataset URI (a local path, `gs://…`, `s3://…`, etc.).

## Examples

Plan an analysis on a large dataset, capped at $20:

```bash
python -m app.interfaces.cli.main plan customers.parquet \
    --goal "Explain churn drivers" \
    --rows 84000000 --max-cost 20
```

Emit machine-readable JSON (the canonical `planned_to_dict` shape):

```bash
python -m app.interfaces.cli.main plan app/sample_data/sales_data.csv \
    --goal "Explain churn drivers" --rows 84000000 --json
```

Request distributed execution on a named cluster context:

```bash
python -m app.interfaces.cli.main plan customers.parquet \
    --goal "Segment customers" --execution ray --cluster-context kind-avaloka
```

## Options

Both commands share the same flags:

| Flag | Purpose |
| ---- | ------- |
| `--goal` | The analysis goal (natural language). |
| `--goal-type` | Category of goal (drives the mission template). |
| `--target`, `--metric` | Prediction target and success metric. |
| `--mode` | Requested analysis mode. |
| `--execution {local,ray}` | Preferred execution substrate. |
| `--cluster-context` | Kube/Ray context name (an **input to the cost model**, not a live connection). |
| `--cloud {gcp,aws,azure,local}` | Cloud used for cost estimation. |
| `--max-cost`, `--max-runtime` | Budget ceilings; exceeding them flags the plan as requiring approval. |
| `--sampling`, `--confidence`, `--margin-error` | Sampling policy inputs. |
| `--rows`, `--columns`, `--type-complexity`, `--rare-class-fraction` | Dataset-shape hints that drive the estimator. |
| `--detect-leakage` | Enable target-leakage checks in the plan. |
| `--format`, `--output` | Output format and destination. |
| `--json` | Print the canonical JSON representation. |

> **Note on `--cluster-context` / `--cloud`.** These are *cost-model inputs* —
> they change the recommended mode and the estimate, they do not open a
> connection to a backend.

### Local vs. remote

By default the CLI plans **in-process** — no backend required. To drive a
deployed Avaloka API instead, point it at an endpoint (the backend exposes
`POST /api/missions/plan`, which returns the same canonical shape):

```bash
python -m app.interfaces.cli.main plan customers.parquet --goal "Explain churn" \
    --endpoint http://localhost:9000 --token "$AVALOKA_API_TOKEN" --json
```

`--endpoint` (env `AVALOKA_API_URL`) and `--token` (env `AVALOKA_API_TOKEN`, the
bearer JWT the API expects) select remote mode. Local and remote return
byte-identical JSON for the same intent — the whole point of the shared core.

## Output shape (`--json`)

```json
{
  "mission": { "...": "canonical mission" },
  "plan": {
    "mission_name": "...",
    "mode": "...",
    "target": "...",
    "cloud_target": "...",
    "steps": ["..."],
    "requires_approval": false,
    "approval_reason": null,
    "implemented": true,
    "notes": ["..."]
  },
  "estimate": {
    "recommended_mode": "...",
    "workload_score": 0.0,
    "full_cost_usd": [0.0, 0.0],
    "full_runtime_minutes": [0.0, 0.0],
    "sampled_cost_usd": [0.0, 0.0],
    "sampled_rows": 0,
    "full_pass_worthwhile": true,
    "rationale": "...",
    "notes": ["..."]
  }
}
```

This is produced by `app/interfaces/service.py::planned_to_dict`.

## MCP server

The same core is exposed over the **Model Context Protocol** so agents (e.g. in
an IDE or another orchestrator) can call it as tools:

```bash
python -m app.interfaces.mcp.server        # stdio transport
```

Tools: `plan_mission`, `estimate_cost`, `describe_environment`, `inspect_intent`.
`plan_mission` returns the identical JSON to the CLI's `--json` for the same
intent — that cross-interface equality is a deliberate invariant.
