# Research

The research behind Avaloka: the question it asks, the mechanisms that answer
it, what has been measured, and what has not.

This page is self-contained. Read it and you should know what the work claims
without opening the PDF — and know exactly which of those claims the code in
this repository backs, because every one of them cites `file:line` here.

Avaloka 2.0.0, Python 3.11–3.12 (`VERSION`, `pyproject.toml:7`,
`pyproject.toml:16`).

---

## 1. The question

Natural-language analytics fails in production for a reason that is not prompt
quality. A planner that is right 95% of the time, wired to a substrate that
executes whatever it emits, is a system that silently ships a wrong number once
in twenty. The failure that destroys trust is not a crash — it is a confident,
well-formatted, wrong answer.

So the question is not "can a better model write better analysis code?" It is:

> How much of an agentic analytics system's reliability can be moved out of the
> model and into deterministic, machine-checkable gates — and what does that buy
> you?

Avaloka's bet is that the useful unit of engineering is not a smarter prompt but
a **checkable result**: every stage states, in a closed vocabulary, what its
output must satisfy, and something with no LLM in it checks that claim.

## 2. The mechanisms

Three of them carry the argument. All three are deterministic. All three exist
because the rule-based alternative had already been tried and was losing.

### 2.1 The blueprint contract — a mechanism instead of a rulebook

The coder writes a numbered blueprint before it writes Python. For a long time
nothing downstream could read it: `coder_pseudocode` was written to state,
injected into the coder prompt, forwarded to reporting — and the validator never
looked at it (`app/agents/blueprint_contract.py:5-11`). Both halves of the
pipeline compensated in the only way left to them, and both grew rulebooks: the
blueprint prompt accumulated roughly 21 numbered special-case rules, the
validator roughly 20 bespoke AST visitors (`blueprint_contract.py:15-19`; the
visitor count is checkable with
`grep -c 'class .*Visitor\|class .*Transformer' app/agents/validator.py`, which
returns 20 in a 2,562-line file).

The module states why that cannot win:

> Rules only ever cover the failures that have already happened.
> — `app/agents/blueprint_contract.py:21`

The fix replaces the mechanism rather than adding rules. The blueprint now also
emits a **contract** — a small closed vocabulary of postconditions on the result
frame — and `contract_validator_node` checks the plan's own stated contract
instead of guessing from a rulebook (`app/agents/validator.py:2267-2299`).
Because the contract is stated for that specific task, checking it adds no
general rule to memorise and costs no tokens (`validator.py:2279-2280`), and it
is silent by design when there is nothing trustworthy to check — no contract, no
successful execution, or an empty result all leave the pipeline untouched
(`validator.py:2282-2299`).

### 2.2 Claim verification — checking the answer, not the code

`app/agents/claim_verifier.py` (444 lines) asks a question the validator never
asks. The split is in the module header as a table
(`claim_verifier.py:14-17`): the Validator's subject is *the code*, and it asks
"will this run against this schema?"; the ClaimVerifier's subject is *the
claim*, and it asks "is this conclusion supported?"

Five checks, each aimed at a distinct way an automated analysis misleads
(`claim_verifier.py:19-32`):

1. causal language asserted from correlational evidence — "drives", "causes",
   "because of";
2. numbers in the narrative that appear nowhere in the computed evidence, i.e.
   an invented statistic;
3. claims contradicted by the model's own metrics — a reliable prediction
   asserted from a model that did not beat its baseline;
4. unhedged extrapolation — "will", "guarantees", "always";
5. significance asserted where no test was run.

The design constraint matters more than the list: the checks are string- and
evidence-level, so they "run without an LLM, cost nothing, and cannot themselves
hallucinate" (`claim_verifier.py:33-34`). An LLM reviewer can be layered on
top — the floor must not depend on one. It runs as the terminal node before
`END` (`app/api/workflow.py:1078`).

### 2.3 Pre-training integrity — six checks that gate training

`app/agents/integrity_agent.py` (328 lines) runs six pure
DataFrame-in / findings-out checks before any training job starts
(`integrity_agent.py:115-253`):

| Check | Function |
| --- | --- |
| target present among the features | `check_target_in_features:115` |
| target correlation above threshold | `check_target_correlation:127` |
| duplicate rows across splits | `check_duplicate_rows:152` |
| identifier-like features | `check_identifier_features:170` |
| near-constant features | `check_near_constant:192` |
| temporal leakage | `check_temporal_leakage:219` |

Findings carry a severity and a blocker makes `safe_to_train` false
(`integrity_agent.py:84-93`), published to state as `integrity_safe_to_train`
(`integrity_agent.py:304,327`). The thresholds are constants, not model
judgement: `LEAKAGE_CORRELATION_THRESHOLD = 0.98`
(`app/agents/mta_v2/training_docker_image/src/data_integrity.py:16`),
`IDENTIFIER_UNIQUENESS_RATIO = 0.95` and `NEAR_CONSTANT_RATIO = 0.01`
(`integrity_agent.py:45,48`).

**The limit is the interesting part.** One class of leakage this agent
structurally cannot catch is preprocessing fitted before the split. Imputation
values, one-hot vocabularies and scaler statistics computed on the full frame
leak test-set information into training, and the integrity agent inspects the
DataFrame — where "the rows are byte-identical either way; what leaks is a
statistic computed at the wrong moment" (`docs/architecture.md:306-310`). Both
training paths therefore fit preprocessing on the training split only
(`local_trainer.fit_feature_preprocessing`,
`ray_job.fit_ray_feature_preprocessing`), and because `ray_job.py` ships inside
a Docker image and cannot import `app.*`, the two are separate implementations
whose agreement is maintained by test rather than by shared code
(`docs/architecture.md:298-318`). Unit tests on both paths are the only defence
available, which is why they are required rather than optional.

### 2.4 Statistics as the first input (2.0)

The 2.0 line adds a layer the 1.6 paper only gestured at: before any model is
consulted, the runtime reads what the data physically *is* and lets those
numbers decide how much is touched, by how many workers, and in what order.
Everything below is deterministic.

**Size without a scan.** `avaloka/workload.py` counts a CSV exactly at or below
4 MB and otherwise extrapolates the mean line length of the first 4 MB over the
file's byte size, flagging the result as an estimate (`_count_csv_rows`,
`peek`); Parquet row counts come from the footer. The server does the same for
uploads and object-store prefixes — a listing for total bytes, a 64 KB head read
for bytes per row, their ratio for rows
(`app/agents/sampling_agent_daft.py::_estimate_dataset_size`). Catalogue
statistics are collected when a source is resolved
(`avaloka/source_statistics.py`, from `avaloka/io/sources.py::materialize`):
file size, format and Parquet row-group count; for a database table the indexes
and primary key, SQLite `page_count × page_size`, PostgreSQL
`pg_total_relation_size`, and native partition columns and child counts from
`pg_partitioned_table` / `pg_inherits`. A read-only query yields statistics
marked `statistics_partial`, because there is no relation to inspect.

**Lanes, and a sampler that obeys them.** Rows and bytes select a lane
(`workload.py::classify`): `online` at or below 100,000 rows and 50 MB —
analyse everything live; `sampled_online` at or below 5,000,000 rows and 1 GB —
sampled answer now, full pass offered as `avaloka batch`; `batch` beyond. The
lane is on the blackboard before the first firefly runs (`avaloka/cli.py:384`,
`avaloka/chat.py:146`) and `avaloka/fireflies/sampling_specialist.py:40-45`
obeys it rather than re-deciding: only `online` analyses the full frame; any
other lane draws 50,000 rows, stratified by the target when it has ≤ 50
classes, seed 42, reason recorded as an assumption. The server-side sampler is
the finer version of the same rule (`app/agents/sampling_async.py:30-95`): five
row tiers, a 50,000-row quick preview from `LARGE` up with the full profile
deferred to a background job, and under byte routing a 1,000-row probe of
in-memory bytes per row against a cap of 0.40 × the container memory limit —
labelled in `app/core/analysis_limits.py:25-31` an **unvalidated hypothesis**.

**The Workload Optimizer.** `avaloka/fireflies/workload_optimizer.py` (60
lines, no LLM) sits between the Sampling Specialist and the Data Engineer in
both rosters (`avaloka/missions.py:92,97`; `tests/avaloka/test_workload_optimizer.py`
asserts the order). From the Data Scout's profile (`n_rows`, `n_unique`,
`top_values`, `missing_pct`, `constant_columns` — `avaloka/stats.py:99-198`) and
the catalogue statistics it estimates a filter factor per predicate — equality:
top-value count / `n_rows`, else 1 / `n_unique`; `in`: list length / `n_unique`;
null tests: `missing_pct`; range: 1/3 — orders predicates by factor with
indexed columns first, drops constant columns from the projection, marks a
predicate for pushdown when the source is Parquet or the column is indexed or a
partition key, picks `partition_pruned_scan` → `index_filtered_scan` →
`projected_scan`, and multiplies the factors into an estimated result
cardinality. The plan is `workload_optimization.json`, folded into the planner's
plan and `mission.json`.

**What the plan is not wired to — read this before quoting "optimizer".** No
production code sets the `analysis_filters` the optimizer reads (only its unit
test does), and `avaloka/fireflies/data_engineer.py` reads `working_frame` and
`profile`, never the plan. In 2.0 the plan is an auditable recommendation
beside the executed scan, not the executed scan. The speedups in §5 are paired
benchmarks of the *strategy* the plan recommends, run by
`avaloka/benchmark/sqlite_efficiency.py` and `workload_efficiency.py`, not by
Avaloka's generated code; they bound what applying the plan would buy.

**The swarm.** `avaloka/coordinator.py::MainAvaloka` loads the dataset once,
splits it **by row position** (`np.array_split`, `:159`) into as many shards as
the edition grants (`avaloka/swarm_limits.py`: oss/free 2, professional 16,
enterprise unlimited, resolved from the installed build; a capped request is
announced, `coordinator.py:296-304`), and runs a full sub-mission per shard in
a `ThreadPoolExecutor` (`:371`) — each clone reloads only its own shard file.
Each clone is handed a lane from its shard's row count (`online` ≤ 100,000,
`:214-220`), so the same Sampling Specialist rule decides whether a shard is
sampled; this is the code behind "swarm, workload routing and adaptive sampling
go hand in hand". Convergence is exact where it can be: shard profiles merge as
sufficient statistics (counts, missing, min, max, mean/variance via sums —
`coordinator.py:10-25`, `:395-408`), verdicts worst-case, the training model by
best honest score. Every edition runs clones **locally**; there is no cluster
fan-out in this build. `avaloka/swarm_trace.py` records every clone step with a
gap-free `seq`; `reward` is null by design and no learned layer consumes it.

**Statistics in the conversation.** The first thing said about a dataset is
read off its profile (`avaloka/chat.py:122-141`): row/column counts, the lane in
the first person with a `~` when the count was estimated
(`workload.py:89-90`), observations each pinned to a statistic
(`avaloka/persona.py:70-100`), and next steps that name this dataset's columns
by profiled role (`chat.py:43`). Goal answers are deterministic and carry their
scope — "Sample of N working rows" vs "All N working rows"
(`avaloka/goal_analysis.py:119`) — and a goal naming a measure no column
supplies is marked unresolved with the candidates listed rather than answered
with a row count (`:32-59`). On the data-agent path a question is classified
without a model (`avaloka/data_agent/specialists.py::build_question_plan`) by
output shape, statistical intent (descriptive / inferential / experimental),
prediction intent, transforms, join and ambiguity, which routes cartographer →
transformer → statistician → experimenter → calibrator → skeptic
(`route_specialists`), and `review_answer` is a deterministic shape-and-evidence
gate. The executable specialists (`avaloka/fireflies/analytical_specialists.py`)
compute a typed, SHA-256-fingerprinted `StatisticalPlan`
(`avaloka/statistics/core.py:16-26,98`), an A/B analysis, or a split-conformal
interval; `missions.py:104-125` inserts them before the Validator only when a
caller has placed the typed plan on the blackboard — the CLI does not.

## 3. Where the gates actually sit

The coding subgraph is wired like this (`app/api/workflow.py:483-498`):

```
coder → validator_syntax → validator_static → execute_code
      → validator_contract → validator_logical
```

Two of the four validators run **after** the code has executed. That is
deliberate, and the reason is in the code: "The contract check runs on the
executed result, before the LLM reviewer: a deterministic verdict is cheaper and
more reliable than a judged one, and when it fires the reviewer has nothing to
add" (`workflow.py:494-496`).

What makes executing-before-checking safe is that validation execution runs on a
**100-row sample**, not the dataset — `DEFAULT_VALIDATION_SAMPLE_ROWS = 100`,
overridable via `AVALOKA_VALIDATION_SAMPLE_ROWS`
(`app/agents/validator.py:2107-2121`), with the frame built by
`_build_sample_dataframe` (`validator.py:2124`).

Two things that a short description usually gets wrong:

* **The LLM layer is advisory by default.** `logical_semantic_validator_node`
  "in the default configuration cannot block at all: any verdict other than
  confirmed fabrication is downgraded to `logical_semantic_error: False` with
  the rationale kept as advisory notes" (`validator.py:2270-2276`). The
  deterministic contract gate exists precisely because of that.
* **There are four validator layers, not three.** `docs/architecture.md:52` and
  `:70-71` still describe three; the graph wires four
  (`workflow.py:484-489`).

## 4. The system around the gates

**Fidelity → substrate routing.** Three execution targets are sibling graph
nodes — `execute_on_ray`, `execute_locally`, `execute_on_k8s`
(`app/api/workflow.py:1073-1075`) — and local is a *routed* destination, not a
fallback: `if fidelity == "quick_sample": execution_mode = "local"`
(`workflow.py:262-263`). Policy: `quick_sample` → local, `portfolio_samples` →
Daft/Ray, `entire_dataset` → k8s-Ray via Celery
(`docs/architecture.md:74-76`); implementations at
`app/agents/execution_agent.py:574` (k8s), `:708` (local), `:1021` (ssh). One
code path from a laptop to a cluster.

Ahead of execution, `app/execution/estimator.py` declares a workload model
`W = f(B, R, C, T, K, A, P, L, Q, $)` (`estimator.py:5-9`) and recommends a mode
before compute is committed. Read its header before quoting any figure from it:
the coefficients are "deliberately simple, documented heuristics calibrated
against the vision's worked examples … meant to be replaced by measured
coefficients later" (`estimator.py:10-14`). The genuinely statistical part is
sample sizing, `n ≈ (z/2E)²` at z=1.96, floored at 50,000
(`estimator.py:93-98`).

**Sampling.** Size-tiered adaptive sizing — under 1k rows take all; <100k →
1k/10k; <1M → 2k/20k; <10M → 10k/50k; else 20k/100k
(`app/agents/sampling_agent_daft.py:49-63`) — plus K-capped stratified sampling
with bias-correction weights (`sampling_agent_daft.py:1264`) and
quantile-stratified sampling (`:1325`). Database sampling is a plain uniform
draw, `ORDER BY RANDOM() LIMIT n` per dialect
(`app/agents/sampling_agent_v2.py:50-95`). There is no error-bounded or
latency-targeted approximate-answer path, and no reservoir sampler.

**Model routing as a single chokepoint.** `app/core/inference.py` (511 lines)
resolves seven providers — `groq`, `local` (vLLM/Ollama), `openai`,
`openrouter`, `bedrock`, `vertex`, `azure` (`inference.py:67-75`) — behind a
global `INFERENCE_PROVIDER` with per-role and per-agent overrides
(`inference.py:45-55`), and lazy provider imports so a Groq-only install never
pulls Bedrock or Vertex. One table decides *which model* for each of twelve
agents (`app/core/model_config.py:98-134`: planner, coder, validator,
summarizer, profiling, visualization, dta_coder, dta_validator, mta,
mta_task_builder, memory, conversational), and `app/core/model_fallback.py`
decides what happens when a provider fails — OpenRouter first, then a local tier
sized by deployment profile (`docs/architecture.md:273-280`). This was built in
response to an outage: Groq removed the entire Llama line and six agents 404'd
mid-analysis (`docs/architecture.md:279-280`; `model_config.py:70` records
`"compound": "groq (decommissioned 2026-09-21)"`).

**MCP in both directions.** Inbound, Avaloka speaks MCP to database estates; the
server exposes exactly three read-only tools — `query`, `list_tables`,
`describe_table` (`app/mcp_server/multi_tenant_mcp_server.py:9`), with
`$out`/`$merge` rejected on Mongo (`:20`) and a first-word allowlist on SQL
(`:759`); the client side is `MCPConnectionManager`
(`app/agents/sampling_agent_v2.py:99`). Outbound, **Avaloka is itself an MCP
server**: `app/interfaces/mcp/server.py` (228 lines) exposes mission-level tools
to external agents — `plan_mission`, `estimate_cost`, `describe_environment`,
`inspect_intent`, `visualize` (`server.py:93,107,123,136,143`, registered at
`:197-215`, stdio transport at `:221`).

**The graph.** Registered nodes (`app/api/workflow.py:1064-1094`):
`memory_injection`, `avaloka_agent`, `plan_etl`, `summarize_etl`,
`generate_planner_graph`, `code_etl`, `execute_actions`, `provision_infra`,
`schedule_task`, `execute_on_ray`, `execute_locally`, `execute_on_k8s`,
`visualize`, `narrate_result`, `verify_claims`, plus optional `train_models`
(`:1087`) and `profile_data` (`:1094`). Roster table at
`docs/architecture.md:44-59`. Agents declare `stage`/`reads`/`writes` through
`app/agents/contract.py`; undeclared writes are dropped and a crashing agent
becomes a structured failure (`docs/architecture.md:264-266`).

**Scheduling** is Celery + RedBeat on Redis (`app/agents/scheduler.py:10,16,24`;
`app/core/celery_app.py:13,44,71`), and scheduled runs replay serialized state
through the same validation path (`scheduler.py:18`,
`app/core/scheduled_run_context.py`).

**Memory and offline operation.** An episodic memory plane with similarity
search, per-session preference and style hints, and a circuit breaker with a
hard retrieval timeout (`app/services/memory_plane.py`,
`docs/architecture.md:366-380`, `docs/context-memory.md`) runs as the graph's
*first* node (`workflow.py:1064,1097`). The embedding model is baked into the
image and `memory.offlineEmbeddings` defaults to true, so with a local model
server the full stack runs with no internet at all
(`docs/architecture.md:382-386`).

---

## 5. What has been measured

Everything in this section is reproducible from this repository. Nothing in it
is an estimate.

### The deterministic regression suite — and the limit it states about itself

```bash
python -m avaloka.benchmark run                 # scorecard
python -m avaloka.benchmark run --json out.json # machine-readable
```

**17/17 passed, overall 1.00** (data_engineering 1.00, data_science 1.00) —
`docs/benchmarks.md:14-19`, per-task table at `:25-41`, task definitions at
`avaloka/benchmark/suite.py`.

That number must never travel without its limit: **the suite scores 17/17 with
every API key unset — it never calls an LLM** (`docs/benchmarks.md:61-73`),
reproduced with the exact `env -u` invocation at
`docs/test-reports/1.6-reliability-measurements.md:120-134`. It measures
Avaloka's own Python — ingestion parity across file, SQLite and object storage;
split logic; leakage checks — not the model behind it. A model matrix over it
would produce four identical 1.00s implying a discrimination the benchmark
cannot make (`1.6-reliability-measurements.md:135-137`). And a saturated score
is a floor, not a ceiling: tasks are added when a defect is found, so the suite
grows toward the things that have actually broken
(`docs/benchmarks.md:70-73`).

### Router stability measured as a rate, not an assertion

The reported failure: Avaloka offered seven numbered analyses, the user said
`run 3`, and the model-training agent answered "I can only help with
model-training and inference tasks." Rate at which a numbered pick was
misrouted, **12 trials per input**, where "before" is the same tree with only
the classifier-prompt fix reverted
(`docs/test-reports/1.6-reliability-measurements.md:11-24`):

| user says | before | after |
| --- | ---: | ---: |
| `run 3` | 11/12 | **0/12** |
| `run 5` | 10/12 | **0/12** |
| `run 1` | 12/12 | **0/12** |
| `do 2`, `option 3`, `go ahead with 4` | 0/12 | 0/12 |

Only the word *run* was ever affected, which is why the defect presented as
intermittent rather than broken. The methodological point is the transferable
one: a single-shot assertion would have passed most runs and been quarantined
as flaky, which is how this survived a suite that already had 18 tests for the
resolver (`:33-36`). `tests/conversational/test_router_stability.py` asserts on
a rate over N trials with zero tolerance.

### Cross-model routing accuracy

Nine cases × six trials = 54 routing decisions per model, run 2026-09-20 via
OpenRouter (`docs/test-reports/1.6-reliability-measurements.md:88-107`):

| model | correct | wall | cost | fails |
| --- | ---: | ---: | ---: | --- |
| `openai/gpt-oss-120b` | **54/54** | 22.0s | $0.0043 | — |
| `anthropic/claude-opus-4.6` | **54/54** | 15.7s | $0.0743 | — |
| `google/gemma-4-31b-it` | **54/54** | 15.3s | $0.0013 | — |
| `openai/gpt-4o-mini` | 48/54 | 10.8s | $0.0018 | `run the training plan` 6/6 |

Total spend **$0.0817**. Two findings: the hardened prompt is not tuned to one
model — three models of very different size and price all score perfectly, so
the fix generalises rather than papering over one model's quirk — and the
failure lands exactly where you would expect, on the one case whose correct
answer depends on reading "training plan" as a noun rather than reacting to the
verb "run" (`:109-118`).

### Suite-level results

| What | Result | Source |
| --- | --- | --- |
| `tests/conversational/` | 77 passed, including 11 live router-stability trials (132 classification calls, ~22s) | `1.6-reliability-measurements.md:39-40` |
| contract suite across 1.6 | 8 failed / 121 passed → 4 / 128 → **0 failed / 132 passed** | `:50-54` |
| spreadsheet ingest (9-sheet workbook) | datasets extracted 1 → **18**; columns named `Unnamed: N` 16-of-16 → **0** | `:62-67` |
| test files under `tests/` | 273 | `find tests -name '*.py' \| wc -l` |

---

### Paired scan efficiency (2.0)

What the strategy the Workload Optimizer recommends buys, measured by
`avaloka/benchmark/sqlite_efficiency.py`: baseline is `SELECT *` with the
filter and projection in Python; optimized is the indexed, projected SQL the
plan describes. In-memory SQLite, rows tiled to 100,000, medians, result sets
asserted identical. **VM steps** (SQLite virtual-machine steps) are the portable
metric; wall speedups are from the committing machine, which the result files do
not identify.

| dataset | returned | VM steps | cut | wall |
| --- | ---: | ---: | ---: | ---: |
| Kaggle Telco Customer Churn (7 repeats) | 159 | 2,300,006 → 1,123 | 99.95% | 2291× |
| Kaggle Titanic (7) | 112 | 1,400,006 → 794 | 99.94% | 1899× |
| Kaggle Iris (7) | 667 | 800,006 → 4,679 | 99.42% | 182× |
| `2021-2022 Football Team Stats.csv`, bundled (5) | 1,020 | 2,200,006 → 7,150 | 99.68% | 310× |
| `salaries.csv`, bundled (5) | 16,666 | 700,006 → 116,672 | 83.33% | 7.1× |

Sources: `docs/workload-optimizer-kaggle-results.json`,
`docs/workload-optimizer-benchmark-results.json` (committed 2026-08-15 with the
optimizer). Read narrowly: the baseline is the worst reasonable scan, so these
are ceilings; `salaries`, where the predicate keeps a sixth of the rows, is
closer to what a weakly selective filter buys. Neither benchmark runs the
optimizer's code path or Avaloka's generated script. The Parquet companion
(`workload_efficiency.py`, opt-in over downloaded Kaggle data) has no committed
result, so no number is reported for it.

## 6. What has *not* been measured

Stated as plainly as section 5, because a reader who takes section 5 for more
than it is has been misled by this page.

* **No external-benchmark result.** Avaloka has **not** been evaluated on
  DABstep, or on any third-party data-agent benchmark. The PDF cites DABstep as
  though it were the evaluation target; it is task-design inspiration and
  related work, nothing more. There is no DABstep harness in this repository.
* **No comparison against a non-Avaloka baseline.** Single-shot NL-to-SQL,
  and agentic verification without the contract gate, are named in the PDF as
  comparison arms. Neither was run.
* **No productivity measurement.** Time-to-trainable-dataset, manual feature
  steps eliminated and time-to-endpoint are proposed metrics with no recorded
  values.
* **The prompt-suite total is 245, not 281.** `docs/architecture.md:327` says
  281; the suite's own README says 86 + 96 + 44 + 19
  (`tests/prompt_suite/README.md:7-12`), and parsing the four suite JSONs gives
  exactly those. 245 is the number to use.
* **The per-1k-turn cost table and its multipliers**
  (`docs/model-routing-cost.md:51-58`) are derived from *static* prompt sizes
  and a price snapshot, with output tokens "typical, not measured per model".
  The doc says so itself at `:110-117`. Not a measurement.
* **"26s on Groq vs 167s on OpenRouter"** (`docs/model-routing-cost.md:100-101`)
  is a single unreplicated timing with no run artifact in the tree.
* **The coding-correctness figures** in `docs/model-routing-cost.md:62-72` cite
  a test directory that is not present on this branch, so they are not
  reproducible here.
* **"≈84M rows ⇒ ≈$14 / ≈17 min"** (`app/execution/estimator.py:11`) are
  calibration targets from worked examples, not measurements.
* **`ANALYSIS_MEM_FRACTION_DEFAULT = 0.40`**
  (`app/core/analysis_limits.py:31`) is labelled at `:25` "UNVALIDATED
  HYPOTHESIS — tune from shadow-mode data before trusting it."
* **The eighteen-namespace serving incident** (`docs/architecture.md:291-292`)
  is an incident narrative, not an instrumented measurement.

---

* **The optimizer's plan is not applied.** `workload_optimization.json` is
  recorded; nothing executes it (§2.4). Any end-to-end speedup attributed to
  the optimizer is unmeasured, and the scan table in §5 is a ceiling.
* **The swarm has never been measured.** Convergence is exact by construction
  and unit-tested (`tests/avaloka/test_coordinator.py`, 11 cases), but no
  wall-clock, cost or quality comparison of a sharded run against a single
  mission exists, and no run has used more than a thread pool on one machine.
* **The benchmark tracks run reference policies, not Avaloka.**
  `benchmarks/tracks/{adaptive_sampling,swarm_ablation,reliability}` run a
  stdlib reference swarm and reference sampling strategies over a seeded
  20,000-row synthetic dataset; the adapter that puts the real graph behind
  them (`benchmarks/adapters/avaloka_graph.py`) requires cloud credentials and
  no result is committed. `benchmarks/results/` is gitignored and empty.
* **No DataAgentBench run yet.** `benchmarks/external/dab/` can run the
  official 54-query suite (five trials each) and refuses to export incomplete
  output; the run is pending — see the model-card section below.
* **The per-clone trace trains nothing.** `reward` is null by design.
* **The 1.6 measurements have not been re-run on 2.0.** The 17/17 suite,
  the misroute rates and the routing-accuracy table are carried forward as
  1.6 numbers.

### The model-card comparison, pending

Frontier model cards compare systems row by row on the same benchmark, same
data, same budget, with provenance under every number. The paper adopts that
layout for the comparison it most needs and has not run: Avaloka 2.0 with every
gate on, Avaloka 1.0, and base models answering the same questions alone. The
skeleton is shaped to the run actually being made. Until it lands every open
cell reads exactly `TBD (bench-compare run <id>)`; no cell is filled from
memory, a vendor page or an unlogged trial, and a cell that cannot exist says
why.

| | Avaloka 2.0 (harness) | Avaloka 1.0 (harness) | Gemini one-shot | Claude one-shot | gpt-oss-120b one-shot | gemma4:e4b one-shot, local |
| --- | --- | --- | --- | --- | --- | --- |
| **Correctness** | | | | | | |
| DAB pass@1, 54 queries, 1 trial, 95% CI | TBD (bench-compare run <id>) | not run — capability absent ᵃ | 12-query subset only ᵇ | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) |
| DAB pass@1, 12-query one-per-dataset subset ᵇ | TBD (bench-compare run <id>) | not run — capability absent ᵃ | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) |
| DAB trial-to-trial std, 10 queries × 3 | TBD (bench-compare run <id>) ᶜ | not run | not run | not run | not run | not run |
| DS suite overall score, 11 runs, 2.0 grader | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) |
| Conversational integrity pass@1, 9 probes | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) |
| Track 1 adaptive sampling (reference adapters) | TBD (bench-compare run <id>) ᵈ | n/a ᵈ | n/a ᵈ | n/a ᵈ | n/a ᵈ | n/a ᵈ |
| **Efficiency (DAB)** | | | | | | |
| Cost per query (USD) | TBD (bench-compare run <id>) | not run — capability absent ᵃ | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | $0 (local) |
| Latency per query (s) | TBD (bench-compare run <id>) | not run — capability absent ᵃ | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) |
| Tool calls / retries | TBD (bench-compare run <id>) | not run — capability absent ᵃ | 0 by construction ᵉ | 0 by construction ᵉ | 0 by construction ᵉ | 0 by construction ᵉ |
| Data scanned (fraction) | not instrumented | not run — capability absent ᵃ | not instrumented | not instrumented | not instrumented | not instrumented |
| **Governance** | | | | | | |
| Planted leak columns flagged (of 2), DS suite | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) | TBD (bench-compare run <id>) ᶠ | TBD (bench-compare run <id>) ᶠ | TBD (bench-compare run <id>) ᶠ | TBD (bench-compare run <id>) ᶠ |
| Unsupported claims | not instrumented | not instrumented | not instrumented | not instrumented | not instrumented | not instrumented |
| Contract catches past the LLM reviewer | not instrumented | no contract | n/a | n/a | n/a | n/a |
| **Reliability** | | | | | | |
| Track 5 recovery under injected faults (reference adapter) | TBD (bench-compare run <id>) ᵈ | n/a ᵈ | n/a ᵈ | n/a ᵈ | n/a ᵈ | n/a ᵈ |

Planned identifiers, to be confirmed by the run's provenance: Avaloka 2.0 =
the `integration/2.0` tree; Avaloka 1.0 = the first public commit (`VERSION`
1.0.0; the repo has no tags); Gemini = `gemini-4-argon` via Google's
OpenAI-compatible endpoint; Claude = `claude-haiku-5.5` via OpenRouter (larger
Claude models are outside the budget); `openai/gpt-oss-120b` via Groq;
`gemma4:e4b` locally via Ollama.
ᵃ Avaloka 1.0 has no natural-language multi-database agent — its graph is
upload-centric — so the row is *not run, capability absent*, not zero.
ᵇ The Gemini harness cell is a 12-query, one-per-dataset subset for cost; the
same subset is scored for every column so the row is comparable.
ᶜ Avaloka 2.0 over gpt-oss-120b only.
ᵈ Tracks 1 and 5 run local reference adapters with no model axis; the 2.0 row
is a reference measurement and the other columns cannot differ by construction.
ᵉ One-shot has no tools by construction.
ᶠ Model-alone leakage is a harness-computed holdout metric over 3 trials, so ±
is real there; single-trial DAB cells carry an exact-binomial 95% interval
instead of a standard deviation.

**Methodology, fixed before the numbers.** *Tasks:* the official
DataAgentBench — 54 queries over 12 datasets on PostgreSQL, MongoDB, SQLite and
DuckDB, hints off, **one trial per query** (seed = trial index) through
`benchmarks/external/dab`; five trials would cost ~5× and are not run, so
pass@1 carries an exact-binomial 95% interval and best-of-N is not reported.
The deterministic DS suite (11 runs per column, every column graded by the 2.0
grader), nine deterministic conversational-integrity probes (faithful = 1.00 /
sycophant = 0.00 brackets), and for the 2.0 column only the Track 1 sampling
and Track 5 fault-injection reference adapters. *What "alone" means, exactly:*
the base model sees the question and the database description once, writes one
Python program, and it runs once against the same databases — no tools, no
feedback, no retry — scored by the same official validators
(`scripts/bench/oneshot_dab_agent.py`). The 2.0 column is the data-agent
mission: question plan → specialist routing → bounded plan/act/inspect over
`query_db`/`list_db`/`execute_python` → answer review. The gap between the two
therefore measures the **whole harness** — tool loop, planning and gates
together — not the validators alone; a tool loop without review, which would
isolate the gates, has not been built. *Metrics:* pass@1; cost from
provider-reported usage at stated public rates (Groq public rate, OpenRouter
catalog, OpenRouter proxy quote for Gemini, $0 local); latency, tool calls and
retries from the records; data scanned is not populated by the data-agent path
→ *not instrumented*. *Provenance per column, printed when the run lands:*
record file under `benchmarks/results/2.0/`, date, git SHA of the SUT tree and
whether it was dirty, exact model id + provider string, trials, seed,
machine/container (M1 Max; `avaloka-ci-deps` for CLI/DS cells, native arm64
venv for DAB), total spend from the ledger. *Rules:* a row that cannot be run
for a column prints the reason, never zero; governance counts for a base model
are what the evaluator counted; the contract-catch row — results the
deterministic contract rejected after the LLM reviewer passed them — is the
cleanest test of the paper's central claim and is *not instrumented* in any of
these runs, recorded here as the first instrumentation to add.

## 7. How to read the PDF against this code

[`../Avaloka-Research-Paper.pdf`](../Avaloka-Research-Paper.pdf) is the ICLR
2026 *Agents in the Wild* workshop paper, **rebuilt against the 2.0 code** (14
pages; the sections marked (2.0) above are in it). The version originally
submitted described a system that had drifted from the implementation, and this
build corrects it rather than reprinting it.

The table below is the audit that produced those corrections. It is kept
because the corrections are the interesting part: each row is a claim that was
checked against the tree, and the right-hand column is what the code actually
does. Rows marked **corrected** no longer appear in the PDF you can download
here; rows marked **open** are still true of it.

| The submitted version said | What the code does |
| --- | --- |
| **corrected.** "Optimizer-aware" — in the title, abstract, first contribution, and a whole section | There is no database query optimizer anywhere in Avaloka: no EXPLAIN generation, no plan parsing, no cardinality or cost estimates. The nearest thing is `validate_join_safety` (`app/agents/validator.py:148-167`), which blocks cross joins from DataFrame row-count metadata and blocks outright when counts are unknown (`:157-159`). What occupies that slot is 2,562 lines of schema-grounded Python AST static analysis over pandas/sklearn, plus the machine-checkable result contract of §2.1. |
| **corrected.** Validation gates run before execution | Two of four validators run after `execute_code` (`workflow.py:483-498`); the contract validator reads `execution_output_data` (`validator.py:2297`); the LLM layer cannot block by default (`validator.py:2270-2276`). The accurate description is *execute on a 100-row sample, then check the contract*. |
| **corrected.** "Spark-on-Kubernetes / Ray" | pyspark is removed — schema and DDL inference use PyArrow, no JVM (`requirements.txt:17-18`, `app/agents/sampling_agent.py:28`). Real dependencies are `ray[air,serve]==2.49.2` (`requirements.txt:99`) and Daft (`:100,141`). |
| **corrected.** All execution runs inside Kubernetes, "exclusively on Kubernetes" | Three execution substrates, local among them as a routed destination (§3, §4). |
| **corrected.** Cloud-agnostic deployment | Multi-cloud in the provisioner, single-cloud in the serving path: `app/agents/mta_v2/` reads `GCP_PROJECT_ID` directly (`docs/architecture.md:336-344`). Treat AWS/Azure inference as manual configuration. |
| **corrected.** The Scheduler deploys jobs as Kubernetes workloads | Celery + RedBeat on Redis (`app/agents/scheduler.py:10,16,24`). The state-replay claim beside it is correct. |
| **open.** MCP yields "schemas, statistics, provenance" and runs `EXPLAIN` | Three tools — `query`, `list_tables`, `describe_table`. Schemas yes; statistics, provenance and EXPLAIN are not exposed. And the outbound direction — Avaloka as an MCP server — is missing from the PDF entirely. |
| **open.** Sampling chosen to satisfy a latency or error target, BlinkDB-style | No answer-level approximation and no error bounds in the sampling path (§4). |
| **corrected.** An eight-agent roster | Seventeen registered graph nodes (§4). |
| **open.** An evaluation section, written in the future tense | The study it proposes was never run. Sections 5 and 6 above are the real state. |

The architecture figure was redrawn for the same reason. The original showed
one validator box for four nodes, validation strictly before execution, no
model-routing layer, no local execution path, sampling downstream of execution
rather than grounding the planner before it, and Azure AKS as the Ray control
plane — contradicting the GCP-bound serving path above. The current figure
(`fig/avaloka-architecture.png`) draws the real order, with execution sitting
*between* the validators.

---

## 8. Limitations

1. **The reliability evidence is internal.** Every number in section 5 comes
   from Avaloka's own suites. There is no external-benchmark result and no
   comparison against a non-Avaloka baseline.
2. **The headline 17/17 is a regression floor, not a capability score**, and it
   provably cannot discriminate between models
   (`docs/benchmarks.md:61-73`).
3. **The deterministic gates are necessarily incomplete.** The contract
   vocabulary is closed, so a failure mode with no vocabulary for it passes; the
   claim verifier is string- and evidence-level, so a subtler overreach passes;
   the integrity agent cannot see preprocessing-order leakage at all (§2.3).
4. **Cloud portability is partial** — provisioner multi-cloud, serving GCP-bound
   (`docs/architecture.md:336-344`).
5. **Serving and distributed training assume configured infrastructure**
   (Kubernetes, database credentials, a model endpoint). Offline operation is a
   first-class mode for the analysis path (`docs/architecture.md:382-386`), but
   it is not the same as a full offline cluster.
6. **Two agent systems live in this repository.** `app/` is the LangGraph system
   the paper describes. `avaloka/` is a separate swarm — `avaloka/fireflies/`
   (analysis_planner, data_scout, data_engineer, sampling_specialist,
   ml_engineer, model_scientist, validator, finops, reporter) with its own
   mission budget and ledger (`avaloka/mission/`) and a coordinator and swarm
   (`avaloka/{coordinator,swarm}.py`). The paper scopes to `app/`.

---

## 9. Files here

| Path | What it is |
| --- | --- |
| [`../Avaloka-Research-Paper.pdf`](../Avaloka-Research-Paper.pdf) | The ICLR 2026 *Agents in the Wild* workshop paper, rebuilt against the 2.0 code |
| `avaloka-paper.pdf` | The same paper, alongside this page |
| `fig/avaloka-architecture.png` | The architecture figure embedded in the paper |

**Why there are no LaTeX sources.** The paper builds against the ICLR 2026
template, and `iclr2026_conference.sty`, `iclr2026_conference.bst`,
`natbib.sty` (Patrick W Daly) and `fancyhdr.sty` are third-party packages under
their own licences — they cannot ship under Apache-2.0, and a `.tex` that will
not compile is worse than no `.tex`. So this tree carries the compiled papers
and this page. The template is available from the ICLR 2026 author kit if you
want to rebuild.

**A citation correction.** arXiv:2508.05002 is **"AgenticData: An Agentic Data
Analytics System for Heterogeneous Data"** (Sun, Li, Zhou, Ma, Xu, Li). It is
*not* "Agents in the Wild: Safety, Security, and Beyond" — that is an ICLR 2026
**workshop**, not a paper. The two are now cited separately in the sources. The
PDFs here predate the fix and should not be cited for it.
