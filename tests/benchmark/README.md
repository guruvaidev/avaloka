# Avaloka Data Engineering & Data Science Benchmark

A capability benchmark **and** swarm-intelligence test harness for the new
Avaloka CLI. It drives the CLI through a stable Python entrypoint over real
datasets (Kaggle, with a deterministic synthetic fallback) and scores it across
the full data lifecycle the CLI is meant to automate:

```
sample → analyze features → build features → transform → train → serve (k8s)
```

It also contains **swarm tests** that model many Avaloka agents — each an
open-claw agent with planner/coder/etc. sub-agents — coordinating under one
*main* Avaloka on the same dataset.

---

## Why this design

The CLI doesn't exist yet, so the benchmark is built around a **contract** the
CLI must satisfy, plus a **reference client** that implements that contract
today by driving the repo's existing agents (`app/agents/sampling_agent.py`) and
deterministic numpy/pandas/PyTorch logic. This means:

* The benchmark **runs now** (offline, synthetic data) against the reference
  client — the same way the existing `tests/` fall back to tmp CSVs.
* The moment the CLI ships an `avaloka.benchmark_client()` factory, point the
  suite at it (`AVALOKA_BENCH_CLIENT=avaloka:benchmark_client`) and the *same*
  tasks + thresholds score the real thing.
* The reference client doubles as the **oracle**: ML tasks must meet/beat a
  baseline it computes, so the CLI has to produce genuinely useful models — not
  just well-formed JSON.

---

## The contract (`adapters.AvalokaClient`)

Every method takes a source **URI** and returns a plain `dict` with a fixed
shape (so scoring is implementation-agnostic):

| Method | Returns |
| --- | --- |
| `sample(uri, stratify_by, sample_size)` | `schema, rows, ddl_schema, table_name, error` |
| `analyze_features(uri, target)` | `features:[{name,dtype,role,missing_fraction,cardinality}], error` |
| `build_features(uri, target, recipe)` | `feature_names, n_features, matrix_path, error` |
| `transform(uri, operations)` | `output_path, rows, columns, error` |
| `train(uri, target, task_kind, heavy, time_budget_s)` | `metrics:{...}, model_path, model_kind, error` |
| `serve(model_ref, platform, replicas)` | `endpoint, predict_url, status, ready, error` |

To benchmark the real CLI, ship a module exposing a factory that returns an
object with these methods:

```python
# avaloka/__init__.py  (the CLI package, to be written)
def benchmark_client() -> "AvalokaClient":
    return AvalokaCliClient(...)   # wraps the open-claw planner/coder/etc.
```

```bash
AVALOKA_BENCH_CLIENT=avaloka:benchmark_client pytest tests/benchmark
```

---

## Datasets & sources

`registry.py` pins real Kaggle datasets (Titanic, Telco Churn, Housing,
NYC Taxi) each with a faithful **synthetic schema**. `datasets.materialize()`
resolves in order: local cache → real Kaggle (if `AVALOKA_BENCH_KAGGLE=1` +
creds) → deterministic synthetic generation (seeded, byte-stable).

`sources.py` lands the *same logical dataset* through six connectors —
`local`, `sqlite`, `postgres`, `gcs`, `s3`, `azure`. Real cloud/DB paths
activate only when their env switch + SDK + creds are present; otherwise
filesystem-backed fakes are used (sqlite for SQL, a local "bucket" for object
stores). The source suite asserts **schema parity** across all of them.

---

## Scoring

`scoring.py` turns per-check outcomes into a 0–1 task score:

* Any failed **CRITICAL** check ⇒ task score `0.0`.
* Otherwise score = severity-weighted fraction passed (CRITICAL=3, MAJOR=2,
  MINOR=1).
* Suite score = mean of task scores. `Scorecard` renders JSON (CI gate) and
  Markdown (report).

---

## The swarm model — many Avalokas, one coordinator

> *"Multiple data scientists working on the same dataset … multiple Avalokas
> (open-claw) coordinate work under one main Avaloka agent."*

```
                          ┌─────────────────────────┐
                          │      MainAvaloka         │   (coordinator)
                          │  plan · dispatch ·       │
                          │  consensus · aggregate   │
                          └───────────┬─────────────┘
            decompose goal into a dependency DAG of SubTasks
        ┌───────────────┬────────────┴────────────┬───────────────┐
        ▼               ▼                          ▼               ▼
 WorkerAvaloka-0  WorkerAvaloka-1            WorkerAvaloka-2   WorkerAvaloka-N
 (planner/coder   (planner/coder            (planner/coder    ...
  sub-agents)      sub-agents)               sub-agents)
        └───────────────┴───── shared dataset (one URI) ────────┴──────┘
```

Each `WorkerAvaloka` wraps an `AvalokaClient` (its sub-agents). The
`MainAvaloka` coordinator runs three coordination patterns the spec implies:

1. **Data-parallel feature analysis.** The feature columns are sharded into
   *disjoint, complete* partitions, one per worker — multiple data scientists
   genuinely profiling the same dataset at once. The coordinator **merges**
   the shard catalogs (dedup by column).
2. **Competitive/redundant training.** `n_model_candidates` workers each train a
   model; the coordinator selects a **consensus** winner (majority vote on model
   kind + best metric) and discards the rest. This is what makes the swarm
   robust to a bad seed or a crashed worker.
3. **Dependency-aware execution with fault tolerance.** Subtasks run in waves
   respecting a dependency DAG (`sample → analyze → build → transform → train →
   serve`). Workers retry; a worker that keeps failing a stage is routed around
   as long as *some* candidate succeeds — the swarm still ships a model and a
   k8s inference endpoint. Provenance (`sub_id → worker`) is recorded for every
   subtask.

The coordinator is deterministic so these properties are asserted as fast unit
tests (`test_swarm_bench.py`), independent of LLM nondeterminism:

* `test_plan_partitions_features_disjointly` — coverage + no overlap.
* `test_plan_covers_full_lifecycle` — all six stages present.
* `test_swarm_scenario` — end-to-end consensus model + ready k8s endpoint.
* `test_swarm_fault_tolerance` — survives a permanently-failing worker.
* `test_swarm_run_is_deterministic` — reproducible aggregation.

Scenarios live in `catalog/swarm_scenarios.py` (full-lifecycle, competitive
5-scientist race, regression-to-serving, fault-tolerant, and a heavy deep-model
race).

---

## Running

```bash
# offline, hermetic — reference client, synthetic data, light models
pytest tests/benchmark -q

# just the swarm coordination tests
pytest tests/benchmark/test_swarm_bench.py -q

# real Kaggle data
AVALOKA_BENCH_KAGGLE=1 KAGGLE_USERNAME=... KAGGLE_KEY=... pytest tests/benchmark

# heavy: real PyTorch deep-learning training (gated off by default)
AVALOKA_BENCH_HEAVY=1 pytest tests/benchmark -m heavy

# the real CLI under test
AVALOKA_BENCH_CLIENT=avaloka:benchmark_client pytest tests/benchmark

# generate a scorecard (JSON + Markdown), usable as a CI gate
python -m tests.benchmark.run_benchmark --include-swarm --out report
```

### Markers

| Marker | Meaning | Default |
| --- | --- | --- |
| `heavy` | real deep-learning training | skipped unless `AVALOKA_BENCH_HEAVY=1` |
| `cloud` | real GCS/S3/Azure/Postgres | skipped unless the source's env switch is set |
| `kaggle` | real Kaggle download | uses synthetic fallback unless enabled |
| `swarm` | multi-avaloka coordination | runs |
| `slow` | long-running | runs |

### Environment switches

| Var | Effect |
| --- | --- |
| `AVALOKA_BENCH_CLIENT` | `reference` (default) or `module:factory` for the real CLI |
| `AVALOKA_BENCH_KAGGLE=1` | use real Kaggle (needs `kagglehub` + creds) |
| `AVALOKA_BENCH_HEAVY=1` | enable real PyTorch DL tasks |
| `AVALOKA_BENCH_GCS/S3/AZURE=1` | use the real object store (needs SDK + creds) |
| `AVALOKA_BENCH_PG_DSN` | real Postgres DSN for the `postgres` source |
| `AVALOKA_BENCH_DATA_DIR` | dataset cache dir (default `~/.avaloka-bench`) |
| `AVALOKA_BENCH_ARTIFACTS` | generated model/transform artifacts (defaults under the data dir) |

---

## Files

| File | Role |
| --- | --- |
| `registry.py` | dataset registry (Kaggle refs + synthetic schemas) |
| `datasets.py` | downloader + deterministic synthetic generator |
| `sources.py` | local/sqlite/postgres/gcs/s3/azure connectors |
| `adapters.py` | `AvalokaClient` contract + reference client + resolver |
| `tasks.py` | task spec dataclasses + reusable checks |
| `scoring.py` | metrics, scorecard (JSON/Markdown) |
| `ml_train.py` | baseline oracle + heavy PyTorch training |
| `swarm.py` | `MainAvaloka` coordinator + `WorkerAvaloka` |
| `harness.py` | run-a-task glue (materialize → dispatch → score) |
| `catalog/` | concrete DE/DS tasks + swarm scenarios |
| `test_*.py` | pytest suites |
| `run_benchmark.py` | standalone runner / report generator |
