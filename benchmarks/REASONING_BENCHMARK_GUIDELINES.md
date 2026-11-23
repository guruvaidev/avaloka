# PR 157 Reasoning Benchmark Guidelines and Implementation Scope

> **Status:** Active engineering guideline and implementation scope
> **Branch:** `benchmarks-develop-1.7`
> **PR 157 code reviewed:** `0f09884` (`feature/agentic-converstation-and-openclaw`)
> **Comparison base:** `e995ec0` (`develop-1.7`)
> **Primary benchmark:** Berkeley DataAgentBench (DAB), 54 tasks / 12 datasets
> **Purpose:** Turn the current DAB harness into an honest, reproducible evaluation of
> Avaloka itself, then improve Avaloka's data-agent capabilities against the failures the
> benchmark exposes.

This document is normative for Avaloka reasoning benchmarks. New benchmark adapters and
reasoning-agent changes must preserve its system-under-test, evidence, isolation, repeated-run,
budget, and reporting requirements. A score that violates those requirements is diagnostic
only and must not be published as an Avaloka result.

### Implementation progress

Change set 1 began on 2026-08-09. The repository now has an Avaloka-owned data reasoning
mission, `dab-command-v2`, explicit SUT identity, an evaluator failure boundary, tier-aware
pricing support, hard runner budgets, effective-config fingerprints, atomic reports, and an
fsynced per-trial resume journal. The upstream DAB `DataAgent` is no longer imported by the
bridge; only its sanctioned low-level database transports are reused.

Still required before a public result: stronger filesystem isolation from validators and prior
results, native connector contracts replacing DAB-specific transports, scan/data-movement
telemetry, pre-call cost reservation, semantic answer review, and a uniform five-trial run.


### Complex-data prerequisite

The first complex-data foundation is now implemented in
[`docs/complex-data-foundation.md`](../docs/complex-data-foundation.md). PDF/OCR images,
video, LeRobot and RLDS sources have bounded inspection contracts plus recursive schema and
format-specific validation. This is a prerequisite for multimodal DSBench/MLE-bench fixtures,
but it does not change the DAB Mongo score by itself: the DataQueryMission must consume the
new schema/evidence contract and the answer validator must still be made semantic before an
official rerun.

The executable-specialist design that follows this prerequisite—including feature/evidence
transformation, statistical analysis, experimentation, conformal prediction, routing, ablations,
and separate evaluation tracks—is defined in
[`docs/dab-agent-improvement-design.md`](../docs/dab-agent-improvement-design.md).

## 1. Executive decision

PR 157 supplies useful building blocks—missions, Fireflies, a coordinator, a budget object,
an execution ledger, validators, and product LangGraph routing—but it does not yet provide a
natural-language, multi-database data agent suitable for DAB.

At the review baseline, the executable DAB bridge was not testing Avaloka. It imported and
ran the upstream benchmark's `common_scaffold.DataAgent` directly. At that point, Avaloka
supplied the outer trial harness and telemetry only. Consequently, the existing results are useful for
diagnosing task classes, database provisioning, model cost, and runner reliability, but they
must not be described as Avaloka benchmark results.

The implementation will therefore proceed in this order:

1. Establish an auditable Avaloka system-under-test boundary.
2. Make the benchmark runner resumable, budget-enforcing, and evaluator-safe.
3. Add a shared Avaloka data-query mission with read-only heterogeneous database tools.
4. Add bounded planning, execution, recovery, and answer review.
5. Run MongoDB as the first capability gate, then all 12 DAB datasets under one pinned
   configuration.
6. Publish a score only after five trials per task complete under the same configuration.

## 2. What was reviewed

The review covered the PR 157 implementation and the benchmark work added after it:

- `avaloka/coordinator.py`
- `avaloka/missions.py`
- `avaloka/fireflies/`
- `avaloka/mission/budget.py`
- `avaloka/mission/ledger.py`
- `avaloka/io/`
- `avaloka/llm/`
- `app/api/workflow.py`
- `app/agents/planner.py`
- `app/agents/coder.py`
- `app/agents/validator.py`
- `app/agents/execution_agent.py`
- `app/agents/avaloka_agent/`
- `benchmarks/`
- `avaloka/benchmark/`
- `tests/benchmark/`
- `scripts/avaloka_dab_agent.py`

PR 157 is a large change—234 files, 23,108 additions, and 681 deletions relative to the
reviewed `develop-1.7` base—so this scope separates benchmark-critical work from broader
product refactoring.

## 3. Current findings

### 3.1 Capabilities PR 157 gives us

- `MainAvaloka` can shard a materialized table and converge deterministic profile results.
- Analyze and train missions have explicit specialist stages and artifact production.
- Firefly activations produce auditable work units through a ledger.
- The mission budget can reject known spend that would exceed a financial ceiling.
- The product graph already has planner, coder, validation, local execution, Ray/Kubernetes
  execution, visualization, and model-training paths.
- Product prompts and validators emphasize reproducibility, leakage prevention, clarification,
  and honest user communication.
- The benchmark package has a useful generic task/config/result abstraction, official DAB
  task discovery, subprocess isolation, score aggregation, and Markdown/JSON reporting.

These are suitable foundations. They should be reused rather than replaced.

### 3.2 Gaps that prevent a valid DAB claim

#### The review-baseline benchmark bypassed Avaloka

Before Change set 1, `scripts/avaloka_dab_agent.py` imported
`common_scaffold.DataAgent` from the DAB checkout and called `agent.run()`. The upstream agent performs the planning, model calls, tool selection,
iteration, and final-answer generation. Neither the PR 157 mission pipeline nor the product
LangGraph controls the run.

Change set 1 corrects this code boundary. No result should be represented as an Avaloka
score until the remaining isolation gates and a uniform run are complete.

#### PR 157 missions are table workflows, not heterogeneous query missions

`avaloka.missions` runs a fixed sequence over one materialized pandas dataset. It is designed
for profiling, cleanup, sampling, modeling, validation, and reporting. DAB requires iterative
natural-language investigation across PostgreSQL, MongoDB, SQLite, and DuckDB, sometimes with
cross-system joins and unstructured transformations.

`avaloka.io` materializes a source locally and generally assumes one table or query. That is
not an interactive, multi-database tool layer, and full materialization conflicts with the
eventual data-locality and data-scanned claims.

#### Existing swarm identity is not yet an ablatable execution swarm

`app/agents/avaloka_agent/swarm.py` explicitly states that it is a narration layer and does
not change routing. PR 157's `avaloka` Fireflies perform real work, but their fixed pipeline
does not implement the planner/tool/reviewer loop DAB needs. The new data-query mission must
make specialist contributions observable and removable so swarm ablations measure real
behavior.

#### Validation does not prove the final answer answers the question

The current validators cover generated code, data quality, leakage, and model stability.
DAB additionally needs answer-level review: question decomposition coverage, correct
denominators, time and geography semantics, join cardinality, unit consistency, evidence
links, and exact output serialization.

#### Budgets are declared but not enforced end to end

- `benchmarks.core.config.Budget` declares latency, token, cost, and scan ceilings.
- The DAB command adapter currently enforces only a subprocess timeout.
- PR 157's mission budget charges known work after an action; it cannot pre-empt a long model
  or tool loop.
- Tool-result size, row count, database scans, retries, and per-call deadlines are not bounded.

The uncapped diagnostic run demonstrated why hard enforcement is required: one failed task
ran 100 iterations for about 23 minutes and was estimated at roughly $22.43. A 25-iteration
rerun failed more cheaply at roughly $1.08.

#### Benchmark runs are not durable enough

The generic runner keeps trials in memory and writes the report after a task or suite
finishes. An interrupt can lose completed work. There is no stable resume key, no atomic
per-trial checkpoint, and evaluator failures are mixed with agent failures.

The official validator loader also depends on `DAB_ROOT` being present on `PYTHONPATH`, drops
the validator's reason string, and does not classify provisioning, evaluator, model, tool,
budget, and answer failures separately.

#### Telemetry is incomplete

- Cost calculation uses environment-supplied flat input/output token prices and does not
  account for provider price tiers or caching.
- The effective iteration cap and other environment overrides are absent from the published
  config.
- The DAB command adapter is marked `requires_cloud = False` even when it calls Gemini.
- Database scans, bytes read, bytes transferred, per-tool latency, retries, and invalid-query
  recovery are not consistently recorded.
- Credentials must remain environment-only; no benchmark token or API key may be written to
  the repository or result artifacts.

#### Benchmark code has three overlapping homes

The repository currently contains:

1. `benchmarks/` — the generic external/internal benchmark harness.
2. `avaloka/benchmark/` — PR 157's product mission and artifact benchmark.
3. `tests/benchmark/` — an older client-contract and synthetic benchmark suite.

Keeping three runners and three result models will create drift. `benchmarks/` should own
experiment execution and reporting. Product-level tests may remain in `avaloka/benchmark/`
temporarily, but should use the shared result/config contracts. `tests/benchmark/` should be
migrated to tests of the canonical harness and retired once parity is demonstrated.

## 4. Diagnostic baseline—not an Avaloka score

The current pilot reached all 54 DAB task definitions and proved that the local database
fixtures, MongoDB 8 restore compatibility, Gemini invocation, and official validators can be
made operational. It also exposed useful behavior:

- BookReview: 14/15 successful trials in the completed five-trial run.
- AG News: 3/20 successful trials, including two pre-restart process failures.
- MongoDB/Yelp: every selected task executed without a MongoDB availability error after the
  restore fix, but answer accuracy remained low (2/7 in the gate run).
- Long-loop datasets such as StockMarket and PATENTS need stronger convergence and cost
  controls.
- The combined mixed-configuration diagnostic was approximately 47% task-averaged Pass@1.

This percentage combines different run settings and measures the upstream DAB agent, so it is
not leaderboard-valid and is not an Avaloka baseline. The first valid baseline will be
created only after Workstream A below is complete.

## 5. Target architecture

The benchmark and product must call the same reusable execution core:

```text
DAB task / Avaloka product request
              |
              v
      Avaloka DataQueryMission
              |
    +---------+----------+
    | plan -> execute -> |
    | inspect -> recover |
    | verify -> finalize |
    +---------+----------+
              |
      read-only tool registry
    +-----+------+--------+--------+
    | SQL | Mongo| Python | artifact|
    +-----+------+--------+--------+
              |
      budget + ledger + trace
              |
      structured final answer
              |
      official DAB validator
```

The upstream DAB repository may supply fixture configuration and sanctioned low-level
database tool implementations. It must not supply the agent, planner, reviewer, prompts, or
termination policy being evaluated.

The shared core should live in a new `avaloka/data_agent/` package. The DAB command bridge
will invoke it directly, while the product LangGraph will invoke the same mission through a
thin graph node. This avoids a benchmark-only agent and avoids coupling a local benchmark to
Redis, Celery, Kubernetes, or the full web application.

Proposed package shape:

```text
avaloka/data_agent/
  mission.py          # public synchronous/async mission entrypoint
  state.py            # typed mission, observation, evidence, and answer state
  orchestrator.py     # bounded plan/act/inspect/recover/review state machine
  planner.py          # question decomposition and evidence plan
  reviewer.py         # answer/evidence and semantic validation
  answer.py           # stable final-answer contract and serialization
  providers/          # model provider interface and usage accounting
  tools/
    base.py           # read-only tool/result contracts
    catalog.py        # source discovery and schema catalog
    sql.py            # PostgreSQL, SQLite, DuckDB
    mongo.py          # MongoDB aggregation/find support
    python.py         # sandboxed transformation of bounded result artifacts
```

## 6. Implementation workstreams

### Workstream A — Correct the system-under-test boundary (P0)

#### Changes

- Rewrite `scripts/avaloka_dab_agent.py` to call `avaloka.data_agent.mission.run(...)`.
- Remove its import and construction of `common_scaffold.DataAgent`.
- Keep `benchmarks/adapters/dab_command.py` as a process-isolation boundary, but rename the
  registered SUT from the generic `dab_command` to an explicit `avaloka_dab` once migration
  is complete.
- Advance the command contract to `dab-command-v2` with:
  - task identity and question;
  - source descriptions and connection aliases;
  - pinned provider/model and generation settings;
  - hard cost, token, time, iteration, scan, and result-size budgets;
  - trace and artifact directory supplied by the harness;
  - no validator path, expected answer, or hidden grading data.
- Record `sut_name`, `sut_version`, Git commit, dirty-worktree flag, model, prompt version,
  tool-registry version, and effective policy in every trial.
- Add an anti-bypass test that makes importing `common_scaffold.DataAgent` fail while a
  benchmark task still completes through the Avaloka mission.

#### Acceptance

- No production code under `scripts/` or `avaloka/` imports the upstream `DataAgent`.
- A trace names Avaloka planner/executor/reviewer stages and their work units.
- The agent process cannot read `validate.py`, reference answers, or other query directories.
- The official validator runs only in the parent benchmark process after the agent exits.
- The answer and trajectory are reproducible from the recorded effective configuration.

### Workstream B — Make the runner trustworthy and resumable (P0)

#### Changes

- Add a per-trial append-only JSONL journal and atomic materialized report.
- Resume by `(suite version, task id, trial index, config hash, SUT commit, dataset
  fingerprint)`; reject incompatible resumes.
- Write a trial record immediately after execution and again after evaluation.
- Preserve the official validator reason in result artifacts.
- Put `DAB_ROOT` on the validator module's import path inside the loader rather than requiring
  callers to set `PYTHONPATH`.
- Classify failures as:
  - `provisioning_error`
  - `sut_start_error`
  - `provider_error`
  - `tool_error`
  - `budget_exceeded`
  - `timeout`
  - `answer_error`
  - `evaluator_error`
- Enforce every `RunConfig.Budget` field and mark exceeded trials distinctly.
- Add dataset/database/tag filters to the CLI, including a first-class MongoDB selector.
- Include `RunConfig.extra` and resolved environment overrides in the scorecard.
- Mark any adapter that can call a hosted model as cloud-requiring.
- Use provider-returned usage data and versioned, tier-aware pricing tables. Report cost as
  `unknown`, not zero, when pricing or usage is unavailable.
- Separate infrastructure success, execution success, and answer correctness in reports.

#### Acceptance

- A killed 54-task run resumes without rerunning completed compatible trials.
- Repeated reporting from the same journal is byte-stable aside from timestamps.
- A validator import error cannot be counted as an agent-answer failure.
- No completed trial is lost on Ctrl-C or process termination after its checkpoint is fsynced.
- Cost, latency, token, iteration, and scan limits produce explicit termination reasons.

### Workstream C — Add heterogeneous, data-local, read-only tools (P0/P1)

#### Changes

- Define a common source contract for:
  - connection health;
  - catalog/schema inspection;
  - query or aggregation execution;
  - explain/estimated scan;
  - bounded previews;
  - durable result handles.
- Implement adapters for PostgreSQL, SQLite, DuckDB, and MongoDB.
- Reuse sanctioned DAB tool transports where helpful, but wrap them behind Avaloka contracts
  and telemetry. Do not reuse the upstream agent loop.
- Apply read-only policy at both prompt and connector levels:
  - reject writes and DDL;
  - use least-privilege database users where fixtures permit;
  - enforce statement deadlines;
  - cap rows and bytes returned;
  - redact secrets from errors and traces.
- Represent large intermediate results as files or handles, not prompt text.
- Allow bounded Python transforms over selected artifacts for cross-database joins and
  unstructured extraction.
- Collect rows examined/returned, bytes read/transferred, query latency, error category, and
  retry count for every tool call.
- Restore each DAB database fixture once per isolated benchmark environment rather than once
  per trial. Verify fixture fingerprints before each run and restore only on drift.

#### Acceptance

- Connector contract tests pass for all four DAB database families.
- The MongoDB gate runs all seven Yelp tasks without provisioning or connector errors.
- Cross-source results can be joined through bounded artifact handles.
- An attempted write is blocked deterministically and logged as a policy event.
- Tool outputs cannot exceed configured row, byte, or deadline limits.

### Workstream D — Build a bounded Avaloka data-query mission (P1)

#### Changes

- Implement a typed state machine rather than an unbounded free-form loop:
  1. interpret the question;
  2. identify required facts and sources;
  3. inspect schemas;
  4. produce an executable evidence plan;
  5. execute the smallest next step;
  6. inspect result shape and quality;
  7. repair invalid queries or revise the plan;
  8. review evidence coverage;
  9. serialize the final answer.
- Make termination evidence-based. A task finishes only when required facts are supported or
  the mission returns a precise blocked/refusal result.
- Add progress detection: repeated equivalent calls, unchanged evidence, or repeated query
  errors must trigger repair, model escalation, or bounded termination.
- Compact context by retaining schemas, plans, evidence summaries, result handles, and error
  signatures rather than replaying complete tool output.
- Reserve budget before every model/tool call and reconcile actual usage afterward.
- Propagate cancellation and deadline information into database, Python, and model calls.
- Extend the PR 157 ledger to capture real model and tool spend per work unit.
- Preserve deterministic seeds, provider request identifiers, Gemini function-call IDs, tool
  call IDs, and artifact hashes in the trace.

#### Acceptance

- The loop cannot exceed hard iteration, token, cost, latency, or scan ceilings.
- Repeated-call detection stops the known 100-iteration failure mode.
- Invalid SQL/Mongo queries can be repaired without discarding successful prior work.
- The context size remains bounded when tool results are large.
- A mission has an auditable chain from each final claim to evidence and tool calls.

### Workstream E — Add semantic review and exact answers (P1)

#### Changes

- Add an Ava-Skeptic/reviewer stage that checks:
  - every part of the question was answered;
  - filters, dates, geography, units, and populations match the request;
  - denominators and aggregation levels are explicit;
  - joins did not duplicate or drop material populations;
  - numeric claims can be recomputed from retained evidence;
  - uncertainty and incomplete evidence are disclosed;
  - the final representation matches the requested list/table/scalar format.
- Require the reviewer to issue structured findings (`pass`, `repair`, or `cannot_verify`)
  rather than only a narrative critique.
- Permit one or more bounded repair turns when the reviewer finds a correctable issue.
- Add deterministic answer serialization after review so formatting changes do not corrupt a
  correct result.
- Make planner, executor, reviewer, and memory contributions individually toggleable for
  genuine swarm ablations.

#### Acceptance

- Unit tests cover denominator mistakes, join multiplication, temporal mismatch, missing
  subquestions, hallucinated columns, and wrong output shapes.
- A reviewer-disabled benchmark configuration changes real execution behavior, not only UI
  narration.
- Key numeric results are recomputed from evidence before finalization.

### Workstream F — Improve capabilities using DAB failure clusters (P1/P2)

The diagnostic runs define the first accuracy backlog:

| Failure cluster | DAB signal | Avaloka change |
| --- | --- | --- |
| Unstructured record extraction | Yelp/Mongo accuracy | Schema-aware nested-field inspection, text normalization, bounded local transforms, evidence-preserving extraction |
| Aggregation and denominator errors | AG News | Explicit population/denominator plan, group-level sanity checks, reviewer recomputation |
| Cross-database joins | DAB's strongest product alignment | Source aliases, typed join keys, cardinality checks, result-handle joins |
| Long non-convergent search | StockMarket/PATENTS | Progress detector, plan revision, evidence completeness, hard iteration and cost limits |
| Domain/dependency reasoning | DEPS/PATENTS | Better schema discovery, domain-term grounding from supplied descriptions, no web fallback in isolated runs |
| Exact output mismatch | Multiple validators | Typed answer schema and deterministic final serialization |
| Invalid query recovery | All database families | Error classification, targeted repair prompts, successful-evidence retention |

Each capability fix must add a small hermetic regression fixture that reproduces the behavior
without requiring a paid model. Paid DAB runs confirm generalization; they must not be the
only test.

### Workstream G — Consolidate benchmark ownership (P2)

#### Changes

- Declare `benchmarks/` the canonical experiment runner, result schema, CLI, and reporter.
- Keep product mission correctness tests under `tests/avaloka/`.
- Adapt useful `avaloka/benchmark/` datasets and scorers to the canonical task/result model;
  keep `python -m avaloka.benchmark` as a compatibility wrapper during migration.
- Move unique `tests/benchmark/` cases into `benchmarks/tests/` or `tests/avaloka/`, then
  remove its reference-client runner after parity tests pass.
- Clearly label deterministic reference adapters such as `local_swarm` as simulations; never
  mix their scores with real Avaloka runs.

#### Acceptance

- One CLI owns local tracks and external benchmarks.
- One versioned result schema supports trial journals, scorecards, and submissions.
- CI cannot accidentally report a reference adapter as the Avaloka SUT.

## 7. Delivery plan

### Change set 1 — Benchmark integrity

Scope: Workstreams A and B.

- Correct SUT identity and command protocol.
- Validator import/reason fix.
- Durable checkpoints and resume.
- Failure taxonomy, effective config, hard-budget enforcement, and truthful cost fields.
- No accuracy claims yet.

This change set is the gate for all subsequent benchmark work.

### Change set 2 — Data-query core and connectors

Scope: Workstream C plus the minimal state/mission contracts from Workstream D.

- Shared `avaloka.data_agent` entrypoint.
- Read-only PostgreSQL, SQLite, DuckDB, and MongoDB tools.
- Bounded result artifacts and full tool telemetry.
- Hermetic connector contract tests.

### Change set 3 — Planner, recovery, and budgeted execution

Scope: remainder of Workstream D.

- Evidence plan and bounded execution loop.
- Context compaction, progress detection, cancellation, retries, and ledger integration.
- Gemini provider support must preserve function-call/tool-call IDs and provider usage.

### Change set 4 — Reviewer and accuracy backlog

Scope: Workstreams E and F.

- Semantic reviewer and deterministic answer serializer.
- Regression fixtures for observed DAB failure clusters.
- Real ablation controls.

### Change set 5 — Promotion and consolidation

Scope: Workstream G and full-run automation.

- Canonical benchmark CLI and CI lanes.
- Pinned full DAB runs, scorecards, comparison reports, and submission validation.
- Retire duplicated runners only after parity is proven.

## 8. Test and benchmark gates

### Gate 0 — Offline CI on every change

- Unit tests for planner state transitions, answer serialization, budgets, pricing, failure
  taxonomy, checkpoint/resume, and contamination boundaries.
- Connector contract tests using small local fixtures.
- Fake-provider tests for tool-call IDs, retry behavior, token accounting, and cancellation.
- No hosted model and no external network.

### Gate 1 — MongoDB capability gate

- Run all seven Yelp queries with one pinned Gemini model and configuration.
- Start with one trial per task; move to three trials after infrastructure errors reach zero.
- Required operational result: 100% task execution coverage, zero provisioning/evaluator
  errors, complete usage/cost traces, and successful resume after a forced interruption.
- Accuracy is reported separately and becomes a regression floor after the first genuine
  Avaloka-native run. The upstream-agent 2/7 diagnostic is not that floor.

### Gate 2 — Cross-database smoke

- At least one representative task each for PostgreSQL, MongoDB, SQLite, and DuckDB.
- Include one cross-source join, one invalid-query recovery, and one unstructured transform.
- Enforce hard defaults for iteration, time, tokens, cost, scan, and result size.

### Gate 3 — Full 54-task baseline

- One trial per task under one pinned SUT, model, prompt, tool, database-fixture, and budget
  configuration.
- Zero validator import failures and zero unclassified errors.
- Report task success, infrastructure success, cost per successful task, latency, tokens,
  tool calls, retries, scans, bytes transferred, and termination reasons.
- Use this run to set realistic per-dataset regression floors; do not retroactively tune the
  evaluator or hidden assets.

### Gate 4 — Repeated release candidate

- Five trials for each of 54 tasks: 270 total trials.
- Same immutable configuration for every task unless the benchmark formally defines a task
  resource class in advance.
- Report mean, median, standard deviation, Pass@1, Best-of-N, and complete failure/cost
  distributions—not only the best run.
- Validate the submission bundle independently before any public claim.

Suggested initial safety defaults, to be calibrated from the first Avaloka-native run:

| Control | Initial default |
| --- | --- |
| Agent iterations | 25 hard maximum |
| Wall time | 10 minutes per task; explicit exception profile when justified |
| Provider cost | $0.50 soft warning, $2.00 hard stop per task |
| Tool call deadline | 120 seconds |
| Returned rows | 10,000 per call unless paged to an artifact |
| In-prompt tool result | 100 KB maximum; larger results become handles |
| Trials | 1 for development, 3 for capability gates, 5 for release results |

These are operational safeguards, not leaderboard rules. Any exception must be declared
before execution and recorded in the scorecard.

## 9. Definition of done

The benchmark-readiness effort is complete when all of the following are true:

- The DAB bridge executes Avaloka's own planner, tools, recovery policy, reviewer, and answer
  serializer.
- No upstream DAB agent or benchmark answer/validator is reachable by the SUT process.
- PostgreSQL, MongoDB, SQLite, and DuckDB are supported through read-only Avaloka contracts.
- The MongoDB gate and the full 54-task run have zero infrastructure/evaluator errors.
- Every trial is checkpointed, resumable, budget-enforced, and auditable.
- Cost is computed from provider usage with a versioned pricing source or marked unknown.
- The scorecard records effective code, model, prompt, tool, data, and budget configurations.
- Agent, provider, benchmark, and evaluator failures are reported separately.
- A five-trial, single-configuration run is complete before an external score is quoted.
- Accuracy improvements are protected by hermetic regression tests.
- Secret values do not appear in Git history, logs, traces, reports, or subprocess arguments.

## 10. Explicitly out of scope for this phase

- Public leaderboard submission before the Avaloka-native SUT and five-trial gates pass.
- Claiming that DAB proves adaptive sampling, cross-cloud portability, or statistical
  confidence; those require Avaloka-specific tracks.
- DSBench, DAComp, MLE-bench, Agents' Last Exam, and human-productivity studies until the DAB
  execution core and canonical harness are stable.
- Product UI polish unrelated to executing or auditing the benchmark mission.
- Benchmark-specific shortcuts, hard-coded task answers, validator introspection, or access to
  hidden grading assets.
- Storing source-control, Gemini, database, or other credentials in repository files.

## 11. Immediate next action

Implement Change set 1 first. Its pull request should contain the corrected Avaloka SUT
boundary, `dab-command-v2`, durable trial checkpoints, official-validator isolation, complete
failure taxonomy, and enforced/recorded budgets. Once that is green, create the minimal
`avaloka.data_agent` mission and MongoDB connector required for the seven-task Yelp gate.
