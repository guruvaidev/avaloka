# Avaloka mission planner (CLI) & MCP

> **Not the `avaloka` command.** This document covers the *planning* interface,
> `python -m app.interfaces.cli.main`, which prices and routes a workload
> without executing it. If you want to actually analyse a file, train a model
> or chat about a dataset, that is a different tool — the `avaloka` console
> script (`pyproject.toml:77` → `avaloka.cli:main`), whose commands are
> `analyze`, `train`, `chat`, `batch`, `infer`, `deploy`, `coordinate`,
> `benchmark` and `version`. It is documented in
> [README_CLI.md](../README_CLI.md).
>
> Confusingly, the planning interface's `argparse` `prog` is also `avaloka`
> (`app/interfaces/cli/main.py:241`), so its `--help` output says `avaloka`
> while you invoke it as `python -m app.interfaces.cli.main`.

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

There are three commands (`app/interfaces/cli/main.py:243-245`):

| Command | What it does |
| ------- | ------------ |
| `plan` | Show the mission plan + estimate **without** running anything. |
| `analyze` | Plan and (when wired) execute an analysis. Execution is being wired up; today it plans and reports the execution path. |
| `viz` | Read a sample of a dataset, pick charts for it, and render a self-contained HTML artifact. |

`<source>` is a dataset URI (a local path, `gs://…`, `s3://…`, etc.). For `viz`
it is optional, because `viz` can instead fetch a thread's planner graph from a
running API (see below).

## Examples

Plan an analysis on a large dataset, capped at $20:

```bash
python -m app.interfaces.cli.main plan customers.parquet \
    --goal "Explain churn drivers" \
    --rows 84000000 --max-cost 20
```

Emit machine-readable JSON (the canonical `planned_to_dict` shape):

```bash
python -m app.interfaces.cli.main plan app/sample_data/salaries.csv \
    --goal "Explain churn drivers" --rows 84000000 --json
```

Request distributed execution on a named cluster context:

```bash
python -m app.interfaces.cli.main plan customers.parquet \
    --goal "Segment customers" --execution ray --cluster-context kind-avaloka
```

## Options

`plan` and `analyze` share the same flags (`app/interfaces/cli/main.py:118`).
`viz` has its own set, listed separately below.

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

## `viz`

```bash
python -m app.interfaces.cli.main viz app/sample_data/salaries.csv --open link
```

Reads a sample of rows, asks the visualization agent for chart specs, and writes
a self-contained HTML artifact. It **works without an LLM key** — the agent
falls back to a heuristic chart policy and says so on stderr, then renders
anyway. Verified: the command above produced four charts with no provider key
set.

| Flag | Purpose |
| ---- | ------- |
| `--target` | Target/label column, for supervised framing. |
| `--rows` | How many sample rows to read (default `500`). |
| `--out` | Directory to write the rendered artifact into. |
| `--open {browser,link,none}` | How to surface the result. `browser` is the default on a TTY; `link` (print a `file://` URL) is the headless/CI default. |
| `--json` | Print the `visualization_config` JSON instead of rendering. |
| `--planner-graph THREAD_ID` | Fetch a deployed thread's planner-graph PNG instead of charting a file. Needs `--endpoint`. |
| `--endpoint`, `--token` | As above (`AVALOKA_API_URL` / `AVALOKA_API_TOKEN`). |

> **`viz` reads CSV only.** The sample loader is `csv.DictReader`
> (`app/interfaces/cli/main.py:171`), so unlike `plan` and `analyze` — which
> take a dataset URI and only ever reason about its *shape* — `viz` must be
> pointed at a real, readable CSV file. Parquet and the cloud URI schemes are
> not handled on this path.

## Output shape (`--json`)

```json
{
  "mission": { "...": "canonical mission" },
  "plan": {
    "mission_name": "...",
    "mode": "...",
    "target": "...",
    "cloud_target": "...",
    "cloud_source": "...",
    "cloud_knowledge": { "...": "service catalog Avaloka reasons with" },
    "agent_context": { "...": "..." },
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

This is produced by `app/interfaces/service.py::planned_to_dict`
(`app/interfaces/service.py:47`), which is the single definition of the shape —
read it there rather than trusting the sketch above, which omits nothing today
but is not generated from the code.

## MCP server

The same core is exposed over the **Model Context Protocol** so agents (e.g. in
an IDE or another orchestrator) can call it as tools:

```bash
python -m app.interfaces.mcp.server        # stdio transport
```

It needs the `mcp` package (`pip install mcp`); without it `build_server()`
raises with that instruction rather than failing obscurely
(`app/interfaces/mcp/server.py:187`). The pure tool functions in the module work
without it.

Five tools are registered, and the names a client actually sees are:

| Registered name | Purpose |
| --- | --- |
| `plan_mission_tool` | Compile intent into a mission and return the full plan + estimate. |
| `estimate_cost` | Just the workload/cost/runtime estimate and recommended mode. |
| `describe_environment` | The detected cloud (`gcp\|aws\|azure\|local`), how it was detected, and the service catalog. |
| `inspect_intent` | Echo the canonical mission an intent compiles to, without estimating or routing. |
| `visualize` | Chart specs (Vega-Lite-flavoured) for a small sample of rows, plus a self-contained HTML artifact. |

> **The first tool is `plan_mission_tool`, not `plan_mission`.** `TOOL_SPECS`
> declares the name `plan_mission` (`app/interfaces/mcp/server.py:52`), but the
> registration passes only `description=` to `@server.tool(...)`, so FastMCP
> derives the name from the decorated function — `plan_mission_tool`
> (`app/interfaces/mcp/server.py:198`). Verified by listing the tools off a
> built server. If you are writing a client, bind to `plan_mission_tool`; if you
> are fixing the code, pass an explicit `name=` so the two agree.

`plan_mission_tool` returns the identical JSON to the CLI's `--json` for the same
intent — that cross-interface equality is a deliberate invariant, and
`POST /api/missions/plan` ([api.md](api.md)) is the third surface that shares
it.
