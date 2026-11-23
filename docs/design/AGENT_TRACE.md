# Agent trace and the granular planner graph

**Status:** Built on the 1.7 line (`app/trace/`, `avaloka/swarm_trace.py`, `ui/src/components/agent-trace/`).
**Audience:** Engineering, Product.

One analysis turn runs about twenty graph steps: the lead agent reads the
intent, the planner chooses a route, the coding team writes and checks code,
an executor runs it, and the narrator and claim verifier finish. The agent trace
records each of those steps as it happens, and serves them back to the user:
what each agent decided and why, what it calculated and on which data, which
model it called and what that cost, and what it handed to the next agent.

## 1. Where the data comes from

Nothing in the agents was rewritten to produce a trace. Three hooks are added
where the graph and its models are built:

| Hook | Installed in | Records |
| --- | --- | --- |
| `traced_node` | `app/api/workflow.py`, every node of both graphs | timing, status and exception *class*, the state keys the node wrote, and what `app/trace/describe.py` reads out of them |
| `traced_router` | the same file, every conditional edge | the branch taken, a one-sentence reason, and the state signals the router read |
| `instrument` (a LangChain callback) | `app/core/inference.build_chat_model`, plus the OpenRouter backup in `model_fallback.attach_fallback` | provider, model, tokens in/out, provider-reported cost, latency, outcome |

A step being recorded is held in a context variable. A model call made deep
inside a node, or a node of the coding sub-graph that `code_etl` runs, attaches
to the right parent step without an argument being threaded through.

Steps travel on the graph state as `ETLState.agent_trace`. Its reducer starts a
new list for each turn, because the top-level graph is checkpointed per thread
and a plain append would carry every earlier turn forward. After the graph
returns, the server stores the turn in Redis as `thread:{id}:trace`, keeping the
last 25 turns for the session TTL. Deleting the thread history deletes it too.

Recording never changes what the graph does. Every hook swallows its own
failures, and a node that raises is noted and the exception re-raised unchanged.

### Cost

OpenRouter reports the cost of each call, and that figure is used as given
(`cost_source: "provider"`). For other providers a price is applied only if the
operator configured one in `AVALOKA_MODEL_PRICES` (`cost_source:
"price_table"`). There are no built-in price constants. An unknown cost is
`null`, and totals report `unpriced_calls` rather than presenting a partial sum
as the whole.

## 2. API

| Route | Returns |
| --- | --- |
| `GET /threads/{id}/trace[?turn=<id>\|latest]` | the trace below |
| `GET /threads/{id}/planner-graph?format=json[&turn=]` | the granular graph (section 3). Without `format=json` the route still serves the PNG |
| `GET /api/agent-metrics` | per-agent aggregates across the caller's traced threads |
| `POST /threads/{id}/messages` | now also carries `trace_turn_id` and `trace_url` |
| `POST /api/swarm/missions` | start a coordinated mission; returns `{mission_id, status: "running", grant, limits}` (section 6) |
| `GET /api/swarm/missions` | the caller's missions, newest first |
| `GET /api/swarm/missions/{id}[?records=1]` | the mission document: grant, clones with their steps, merge |
| `GET /api/swarm/missions/{id}/records?after=&limit=` | the live endpoint: records with `seq > after`, `cursor`, `done` |
| `GET /api/swarm/missions/{id}/clones/{clone_id}` | one clone's document and its records |
| `GET /api/swarm/missions/{id}/episodes` | the RL view, one episode per clone |
| `GET /api/swarm/missions/{id}/telemetry` | the metadata-only records telemetry would send, for review |
| `DELETE /api/swarm/missions/{id}` | forget the mission |

All of them are owner-only: 401 without a user, and 404 for a thread or
mission that is not the caller's, so another user cannot learn that it exists.
The mission routes and `/api/agent-metrics` are also the **swarm console**,
which is Professional and Enterprise (docs/EDITIONS.md): each carries
`Depends(require_swarm_console)` (`app/api/swarm_gate.py`) and answers
`403 {"code": "swarm_console_unavailable"}` on the open-source and Free
editions, where the chat tells the mission back instead
(`app/api/chat_swarm.py`). `/threads/{id}/trace` and the planner graph are not
console and stay open everywhere. `GET /api/capabilities` reports the gate
under `features.swarm_console`. The 401 comes first, so an anonymous caller
learns nothing about the edition.

```text
trace = {schema_version, thread_id, swarm_limits, totals, agents[], turns[]}
turn  = {turn_id, question, started_at, ended_at, latency_ms, data_scope,
         steps[], handoffs[], swarm{clones[], message}, totals}
step  = {kind: "agent", step_id, parent_id, turn_id, seq, node, agent, role, clone,
         started_at, ended_at, latency_ms, status, error_class,
         inputs{question, data_scope},
         decision{type, value, reason, ...},
         calculation{operation, code, operations[], data_scope, values, result, findings, charts},
         gates[], inference[], written[]}
route = {kind: "decision", step_id, seq, node: <router>, after: <node>,
         decision{type: "route", router, chosen, reason, signals}}
```

`calculation.operations` is read from the code's AST. It lists loads, casts,
filters, derived columns, group-by→aggregate (with the formula as written),
joins, sorts, samples, cleans and writes, in data-flow order, each with the
`columns` it touches (checked against the dataset's columns when the state
knows them). A script's entry point usually sits at the bottom, so the load
appears last in the text but first here. The coder's steps carry the
operations of each attempt; the executor's step (`execute_locally`,
`execute_on_ray`, `execute_on_k8s`) carries the operations of the code it
actually ran, and the dry run's step those of the attempt it tried.

A training step (`train_models`) carries `calculation.models[]` (name, type,
target, features, metrics) and `calculation.evaluation` (primary metric,
metric means, baseline, `beats_baseline`), read from `training_result` and
`evaluation_report`.

`handoffs[].passed` lists state keys, never their values.

Swarm clones appear as steps of the turn: each clone the lead agent narrated
is shown with the agent steps that actually did its work, and its status is
`done`, `failed` or `not_run`.

## 3. The planner graph

The graph is built from the trace, so it exists for every turn, not only the
ones that went through the summarize route (the PNG's only source).

Its spine is the data lane: what was done to the user's data, in data-flow
order, read from the code that actually ran. The agents, their decisions and
the coding team's checks are drawn beside it and the UI hides them by default
(a "Show agents and checks" toggle brings them back), so the first picture a
user sees is how their prompt was acted upon, not plan → code → test → evaluate.

| Node type | Lane | Shows |
| --- | --- | --- |
| `question` | agents | the user's question; a `data` edge joins it to the load |
| `agent` | agents, code | the agent, its subtitle (intent, plan, verdict), latency, tokens, cost, model |
| `decision` | agents | a router: the branch chosen, its reason and signals |
| `gate` | agents | an approval or clarification gate and its status |
| `data` | data | the load (dataset, rows, `columns[]`, the read formula) and the sample (name, fidelity, rows) |
| `operation` | data | one data operation: its label names the columns (`Filter on monthly_charges`, `Group by region → mean`), with `formula`, `columns[]` and `source` |
| `check` | code | each validator and the dry run, pass or fail, with the reason |
| `result` | data | the computed result, when it is small enough to show |
| `train` | data | one model trained: name, type, target and features (`columns[]`), its metrics |
| `evaluate` | data | the evaluation: the primary metric and the metric means, the baseline, `beats_baseline` |
| `evidence` | agents | the claim verifier: safe or unsafe, and how many claims were flagged |

Operation nodes come from, in order of trust (`source`): `executed`, the code
the executor step ran; `validated`, the last coder attempt none of whose checks
failed; `generated`, the coding team's final code. Never from plan text, and
never from an attempt that failed validation. A turn that re-runs existing code
without a coding step still shows its operations, read from the executor.

The edge types are `flow`, `route`, `handoff`, `data`, `check`, `retry`,
`evidence` and `gate`. A failing check joins the next coder attempt with a
`retry` edge that carries the reason.

Every node has a `lane` (`agents`, `code` or `data`), a `layer` (longest path
from the question) and an `order` within its layer and lane. The UI draws lanes
as side-by-side columns and layers as rows, top to bottom, closing up the rows
left empty when the agents are hidden. Nodes from a recorded step carry its
`step_id`, and clicking one opens that step.

## 4. Privacy

The trace is the user's own analysis. It is stored in their cluster, served
only to them, and may show their columns, formulas and values. The planner
graph carries column names and small values for the same reason, and only in
this local view.

Telemetry is a separate projection: `app/trace/telemetry.agent_events` maps
steps onto the `agent.started` / `agent.completed` records that
`telemetry_service/schema.py` accepts.

- It is an allowlist built from closed vocabularies: the agent name, the
  outcome, the exception class, timings, retry count, provider, token counts,
  and the model when it is a shipped default (otherwise `"other"`).
- It contains no field a column name, a file name, a value, the question or a
  code fragment could travel in.
- Its `session_id` is random per process, never the thread or user id.
- Nothing sends it yet. The writer and uploader are TELEMETRY.md §4.3–4.4, and
  are not built.

`tests/test_agent_trace_telemetry.py` validates the projection with the ingest
service's own `validate_batch`. It plants column names, a file name, values, a
question and an exception message, and checks that none of them appear.

## 5. Known limits

- Scheduled runs (Celery) and the `langgraph dev` deployment record steps on
  their graph state, but only the chat path persists them to Redis.
- `data_scope.dataset` is the upload's alias or file name. Database and cloud
  sources show the working file.
- The OpenRouter cost is whatever the provider reports in `usage.cost`. A
  provider that stops reporting it shows as unpriced.

## 6. Swarm trace: one record per clone, and the RL view

The chat trace above records one agent team working one question. A
coordinated mission (`avaloka coordinate`, `MainAvaloka.coordinate`) is a
different shape: the main Avaloka shards the dataset and dispatches a fleet of
clones, each of which runs its own firefly swarm on its shard, and then merges
what they bring back. `coordination.json` keeps the merged answer. The swarm
trace keeps *how each clone got there*, separately, so the logs and the
clones' interactions can be used for reinforcement learning on the 1.7 line.

### 6.1 The recorder

`avaloka/swarm_trace.py` is stdlib-only (the `avaloka` package never imports
`app.*`, S1.08) so the standalone CLI records exactly what the server serves.
`SwarmTraceRecorder` appends records under a lock with a gap-free, monotonic
`seq` starting at 1; an optional `sink` callable sees each record as it is
appended, which is how the server goes live and how the CLI streams
`swarm_trace.jsonl`. Every record is JSON-plain by construction (numpy
scalars, enums and NaN are converted at record time).

Every record is `{seq, ts, type, mission_id, clone_id?, step_id?, ...}`:

| Type | Carries |
| --- | --- |
| `swarm.requested` | `kind`, `goal`, `source` (provenance-safe URI), `requested_clones`, `requested_parallel`, `granted_clones`, `granted_parallel`, `limited`, `message`, `derived_from`, `limits` |
| `swarm.sharded` | `clones`, `total_rows`, `rows_per_clone[]`, `columns[]` |
| `clone.dispatched` | `clone_id` (`ava-00`), `name` (`Ava-00`), `shard_id`, `received: {rows, columns[], goal, kind, target, metric, shard_path}` |
| `clone.step.started` | `clone_id`, `step_id` (`ava-00/s01`), `step_no`, `firefly`, `human_role` |
| `clone.step.completed` | the same, plus `output`, `duration_ms`, `manual_minutes`, `compute_cost_usd`, `model_cost_usd`, `metadata`, `outcome: "ok"|"failed"` (`error_class`, `error` when failed) |
| `clone.returned` | `clone_id`, `status: "done"|"failed"`, `verdict`, `max_safe_deployment_level`, `model`, `sampling`, `duration_ms`, `cost_usd`, `steps`, `error_class?`, `error?` |
| `swarm.merged` | `rules` (`profile: "exact sufficient statistics"`, `verdict: "worst case"`, `model: "best honest score"`), `verdict`, `max_safe_deployment_level`, `selected_model` (with `from_shard`), `contributions: [{clone_id, rows, verdict, selected, cost_usd}]`, `n_rows`, `columns` |
| `swarm.finished` | `status: "done"|"failed"`, `duration_ms`, `economics`, `error_class?` |

`derived_from` (2.0) is how `requested_clones` was arrived at when nobody
typed it: the sizing sentence from `avaloka/swarm_sizing.py`, numbers only
("120,000 rows, 2.7 MiB, sampled_online lane: ceil(rows/25,000)=5,
ceil(bytes/64 MiB)=1, within [2, 64] -> 5; 3 row_groups, one clone per
partition up to that -> 3"). It is `null` when the caller asked for the
count explicitly (`--swarm N`, `workers` in the API). It names no column and
no file, and the telemetry projection (§4) does not carry it; it is for the
local trace and the narration. The same string is kept on the grant in
`coordination.json` (`execution.swarm.derived_from`).

The coordinator emits them in that order: `swarm.requested` right after the
edition grant, `swarm.sharded` after sharding, and per clone
`clone.dispatched` → `clone.step.started`/`clone.step.completed` for each
firefly → `clone.returned`. The step records come from the mission's own
`on_progress("start" | "done")` narration, and `"done"` reads the `WorkUnit`
the firefly just put on the ledger (`Firefly.activate` records before `_run`
narrates, so the unit is always there). A firefly that raises recorded no
unit; its step is closed as `failed` with the exception class, the clone
returns `failed`, `swarm.finished` is `failed`, and the exception propagates
exactly as it did before. A request refused before sharding (an unreachable
source, a missing target column) still leaves `swarm.requested` and a failed
`swarm.finished`.

### 6.2 The mission document

`recorder.mission()` derives one document from the records; `clones` and
`merge` are rebuilt from them every time, so there is one source of truth:

```text
mission = {schema_version: "1.0", mission_id, kind, goal, source, status,
           started_at, ended_at, limits, grant, sharding, clones[], merge,
           economics, error_class, records[]}
clone   = {clone_id, name, shard_id, status, rows, received, dispatched_at,
           returned_at, steps[], verdict, max_safe_deployment_level, model,
           sampling, cost_usd, duration_ms, error_class, error}
step    = {step_id, step_no, firefly, human_role, started_at, ended_at, status,
           output, duration_ms, manual_minutes, compute_cost_usd, model_cost_usd,
           metadata, error_class, error}
```

The CLI writes `<output>/swarm_trace.json` beside `coordination.json` (which
points at it: `trace: {path, records, mission_id}`) and appends
`<output>/swarm_trace.jsonl` live, one record per line, flushed, so another
process can tail it. Library callers get the file too: `coordinate` creates a
recorder when none is given.

### 6.3 The RL view

`episodes(mission)` is one episode per clone:

```text
episode = {episode_id: clone_id, mission_id, clone: {clone_id, name, shard_id},
           context: {rows, columns, goal, kind, target, metric},
           steps: [{t, step_id, action: {firefly, human_role},
                    observation: {output, metadata},
                    cost: {duration_ms, model_cost_usd, compute_cost_usd, manual_minutes},
                    outcome}],
           terminal: {status, verdict, max_safe_deployment_level, score,
                      selected_by_merge, cost_usd},
           reward: null}
```

`reward` is `null` on purpose. Assigning reward is the 1.7 learned layer's
decision, not the recorder's: whether a clone's work was good depends on what
the layer is optimising for, and a fixed formula in the recorder would bake
that choice into every trace ever written. The signals it will use are all on
`terminal`: `selected_by_merge` (the merge picked this clone's model),
`verdict`, `score` and the cost.

### 6.4 The server

`app/trace/swarm_service.py` starts a mission on a daemon thread of the API
process with a recorder whose `sink` is the in-process registry
(`app/trace/swarm_store.py`), so the mission is listed as `running` at once
and its records are readable while the clones work. `POST /api/swarm/missions`
takes `{source, goal, kind, workers?, max_parallel?, target?, metric?}` where
`source` is one of `{"sample": "churn" | "salaries" | "healthcare" | "youtube"
| "football" | "taxi"}` (the bundled files), `{"thread_id": id}` (that
thread's current dataset, the working copy the chat path reads, owner-checked)
or `{"path": p}`, which is refused with 400 unless
`AVALOKA_SWARM_ALLOW_PATHS=1`. `workers` defaults as the CLI does: absent, the
fleet is sized from the data (`avaloka/swarm_sizing.py`, Professional and
Enterprise; the open-source and Free editions get their preview of two) and
the response's `grant.derived_from` says how; the edition cap is applied by
`avaloka.swarm_limits` and announced in `grant`, never silently.

The chat uses the same service without the route: a message that asks for the
team (`app.api.chat_swarm.TRIGGER_RE`, or the "Send my team at it" quick reply)
starts a mission on the thread's working copy with `source={"thread_id"}`,
waits up to `AVALOKA_CHAT_SWARM_TIMEOUT_S` (60 s) for the clones, and answers
with `avaloka.swarm_narration.narrate(records, VERBOSE)` as one assistant
message plus an `assistant_prompts` entry of step `swarm`. Every mission that
runs on a thread's dataset is recorded on that thread's session
(`chat_swarm.remember`; `POST /api/swarm/missions` with `source.thread_id`
does it through `server._note_thread_mission`), and the next chat turn appends
its story as a `swarm` prompt whatever started it; an unfinished one is told so
far and stays pending.

**Polling, not SSE.** The UI polls `/records?after=<cursor>` every second.
`EventSource` cannot send headers and the UI authenticates with a Bearer
header on every request; records are append-only with a dense, monotonic
`seq`, so incremental polling loses nothing; and a two-clone laptop demo
produces tens of records, not thousands. `seq` starts at 1, so the first poll
omits `after` (or sends `0`); `cursor` is the last `seq` returned and `done`
is true once the mission has finished and nothing is left past the cursor.

**Storage.** The running recorder lives in memory; the registry holds at most
64 missions and evicts the oldest *finished* one. When a mission finishes its
document is written to Redis as `swarm:{mission_id}` with the session TTL and
indexed under `user:{uid}:swarms`, as `app/trace/store.py` does for the chat
trace (size-capped: a document over 1.5 MB is stored without its records and
marked `truncated`). It works with no Redis at all: the document then says
`persistence: "memory"` and that it is lost when the server restarts.

### 6.5 Privacy: the local trace and the telemetry projection

The full per-clone record, outputs and all, lives in the local store and is
served by the local API only, to its owner. Telemetry is a separate
projection, `app/trace/telemetry.swarm_events`, built the same way as
`agent_events` (section 4): each completed clone step becomes an
`agent.started` / `agent.completed` pair whose `agent` is the firefly name
(already in the ingest's `AGENTS`; an unknown name is dropped, never renamed),
and one `swarm.completed` record carries the fan-out and its outcome:
`clones_requested`, `clones_granted`, `parallel`, `limited`, `edition`,
`kind`, `outcome`, `verdict`, `duration_ms`, `row_count_bucket`. Nothing
else: no goal, no source, no column, no clone output, no path, no mission id,
no clone name. `GET /api/swarm/missions/{id}/telemetry` returns exactly those
records so a reviewer can read what would leave the machine; nothing sends
them (TELEMETRY.md §4.3–4.4 are not built).

`tests/test_swarm_trace_telemetry.py` runs a real two-clone mission on a frame
with a planted column name (`patient_hiv_status`), file name, goal and values,
validates the projection with the ingest's own `validate_batch`, and asserts
none of the planted strings appear. The ingest refuses `goal` by name, so a
client that started sending it would have the whole batch rejected.

### 6.6 Known limits

- A mission runs on a thread of the API process; it cannot be cancelled, and
  `DELETE` forgets it rather than stopping it.
- Clone steps are the CLI fireflies, which make no model calls today, so the
  `agent.completed` records they project carry no provider, model or token
  fields unless a step's metadata names them.
- Without Redis, the mission is gone when the server restarts; with it, the
  index and the document share the session TTL.
- `episodes()` describes a mission after the fact. Nothing trains on it yet;
  `reward` stays `null` until the learned layer assigns it.
