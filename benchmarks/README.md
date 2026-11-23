# Avaloka Benchmark Harness

Runnable test suite for Avaloka's benchmark tracks.

The [PR 157 reasoning benchmark guidelines](REASONING_BENCHMARK_GUIDELINES.md) are the
engineering policy for system-under-test integrity, evidence-based reasoning, contamination
isolation, cost controls, and release-quality DAB runs.

> **Status:** v0.2. The **core harness** (task schema, runner, metrics, reporting,
> scorecard) and **Tracks 1 (Adaptive Sampling)**, **4 (Swarm Ablation)**, and
> **5 (Reliability & Recovery)** are runnable end-to-end today with zero external
> dependencies. DAB now has an Avaloka-native pilot adapter; paid execution still
> requires the official fixtures, local database services, and a configured hosted model.
> Other external benchmarks and Tracks 2/3/6 remain scaffolded.
>
> New here? Read [`GUIDE.md`](GUIDE.md) — especially the **Local-first (no cloud)**
> section. Mission and agent-role prompts are versioned in [`prompts/`](prompts/).

## Quick start

```bash
# from repo root
python3 -m benchmarks.run list
python3 -m benchmarks.run run track1_adaptive_sampling
python3 -m benchmarks.run run track4_swarm_ablation --trials 5 --out benchmarks/results
pytest benchmarks/tests -q          # harness smoke tests (stdlib only)
```

Hosted-model/cloud SUTs are **skipped unless `--allow-cloud`**,
so default runs stay local. See [`GUIDE.md`](GUIDE.md) → Local-first.

`list` and Track 1 run on the Python standard library alone — no pandas, numpy, Groq
key, cloud creds, Ray, or server required. That is deliberate: the harness must be
testable in CI in isolation (design rule #1, "isolated execution").

## Architecture

```
benchmarks/
  README.md                # this file — architecture
  GUIDE.md                 # how to run / read / extend; local-first policy
  run.py                   # CLI: list | run [--allow-cloud]
  prompts/                 # versioned mission prompts + per-agent role prompts
  core/
    task.py                # BenchmarkTask, DatasetSpec, MissionOutcome, Trial/TaskResult
    config.py              # RunConfig — model/agent/sampling/env axes kept SEPARATE
    metrics.py             # aggregation (Pass@1, Best-of-N, mean/median/std) + primitives
    runner.py              # trial loop + hard budgets + aggregation
    checkpoint.py          # fsynced execution/evaluation journal + compatible resume
    registry.py            # suite name -> factory (lazy)
    reporting.py           # JSON record + markdown scorecard (+ full config block)
  adapters/
    base.py                # SystemUnderTest protocol (one method: run_mission)
    baselines.py           # Track-1 strategies: full/random/stratified/reservoir/domain/adaptive
    local_swarm.py         # Track-4 local reference swarm (cloud-free, ablatable)
    reliability.py         # Track-5 local fault-injection adapter (cloud-free)
    avaloka_graph.py       # `avaloka` (LangGraph) + `avaloka_sampler` — requires_cloud
  tracks/
    adaptive_sampling/     # Track 1 — flagship (datasets + tasks + evaluators + configs)
    swarm_ablation/        # Track 4 — agent ablation over Track-1 tasks (local)
    reliability/           # Track 5 — fault injection + recovery over Track-1 task (local)
  external/
    dab/                   # official task loader, isolated evaluator, submission
  tests/                   # stdlib smoke tests (forwarded into repo `tests/` for CI)
  results/                 # scorecard output (gitignored except .gitkeep)
```

### The one contract everything shares

A benchmark is a list of `BenchmarkTask`. Running a task under a `RunConfig` for N
trials yields N `TrialResult`s (each carrying a `MissionOutcome` = answer + the
`(Quality, Cost, Latency, Data Scanned)` 4-tuple inputs), aggregated into a
`TaskResult` with Pass@1, Best-of-N, and mean/median/std of every metric.

A **system under test** implements one method:

```python
class SystemUnderTest(Protocol):
    key: str
    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome: ...
```

Adapters resolve by `RunConfig.sut` string, so suites never import the Avaloka app —
`avaloka_graph.py` loads `app/api/workflow.py:build_graph` **lazily** inside
`run_mission`.

### Model effects vs agent effects (design rule #4)

These are configuration axes on `RunConfig`, never code branches:

| Axis | Field | Used by |
| --- | --- | --- |
| Foundation model | `llm`, `model_routing` | Track 3 |
| Agents enabled | `agents_enabled` | Track 4 ablation |
| Sampling strategy | `sampling_mode` | Track 1 |
| Execution environment | `environment` | Track 2 |

So "same Avaloka, different LLM" and "swarm vs single agent" are separate runs of the
same tasks — nobody can attribute an Avaloka gain to the foundation model.

## How to add a benchmark

1. **A new track:** create `tracks/<name>/track.py`, define tasks + evaluators, and
   decorate a factory with `@register_suite("<name>", kind="track", ...)`. Add the
   import to `core/registry.py:_ensure_loaded`.
2. **A new external benchmark:** create `external/<name>/adapter.py`, load its tasks
   into `BenchmarkTask`s, target the `avaloka` SUT, register the suite. Vendor its data
   under `external/<name>/data/` (gitignored) with a fingerprint manifest.
3. **A new system under test:** implement the `SystemUnderTest` protocol and decorate
   with `@register_adapter("<key>")`.

## Grounding & honesty notes

The suite is grounded in Avaloka's *actual* code (`app/agents/`, `app/api/workflow.py`,
`app/graph/etl_state.py`). Two things `benchmarks.md` aspires to that the code does
**not** implement yet — and which these benchmarks are partly designed to force:

- **Statistical stopping in the sampler.** `sampling_agent_daft.py` today does
  size-tiers (`calculate_adaptive_sample_sizes`) + executor selection (`should_use_ray`)
  + a multi-strategy portfolio. There is **no confidence-interval / margin-of-error
  convergence loop**. Track 1's `adaptive_sample` baseline is the *reference policy* the
  real agent should grow into, not a mirror of current behaviour.
- **Multi-provider LLM routing.** The product graph remains Groq-oriented. The new
  benchmark-facing `avaloka.data_agent` has a provider boundary, but automated cost-quality
  routing is not implemented; Track 3 remains forward-looking.
- **DAB connectors.** The Avaloka-native DAB pilot wraps the benchmark's sanctioned read-only
  PostgreSQL, MongoDB, SQLite, and DuckDB transports. These are not yet Avaloka-owned,
  production connector contracts (see `external/dab/README.md`).
