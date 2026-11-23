# Avaloka Benchmark Guide

How to run, read, and extend the benchmark suite. Pairs with
[`README.md`](README.md) (architecture). If you read one thing first, read the **Local-first** section.

Changes to a reasoning agent or external benchmark adapter must also follow the
[PR 157 reasoning benchmark guidelines](REASONING_BENCHMARK_GUIDELINES.md). In particular,
the system under test may use benchmark-provided database transports, but it may not delegate
reasoning to a benchmark reference agent or read validators and expected answers.

---

## Local-first (no cloud) — the default

Every benchmark in this suite runs **locally with the Python standard library** by
default. No Groq key, no cloud credentials, no Ray, no server. This is enforced, not
just encouraged:

- Track 1 (Adaptive Sampling) and Track 4 (Swarm Ablation) use **local reference
  adapters** (`baselines.py`, `local_swarm.py`) over synthetic, fingerprinted data.
- The runner **skips any cloud/Groq-driven SUT** (the real `avaloka` graph, the
  `avaloka_sampler`) unless you pass `--allow-cloud`:

  ```bash
  python -m benchmarks.run run track1_adaptive_sampling            # local only
  python -m benchmarks.run run some_suite --allow-cloud            # opt into cloud SUTs
  ```

  Without `--allow-cloud`, cloud configs print a `SKIP ...` line and are dropped, so a
  local run can never silently start billing Groq or spinning up Ray.

**Order of operations:** get a benchmark passing locally against the reference adapters
first; only then flip to `--allow-cloud` to measure the real Avaloka graph on the same
tasks. The reference adapters double as the offline regression baseline the real system
is compared against.

---

## Running

```bash
python -m benchmarks.run list                                   # discover suites
python -m benchmarks.run run track1_adaptive_sampling           # run a suite
python -m benchmarks.run run track4_swarm_ablation --trials 5   # override trials
python -m benchmarks.run run track1_adaptive_sampling \
    --config t1_adaptive --config t1_full                       # subset of configs
python -m benchmarks.run run track1_adaptive_sampling --out benchmarks/results
pytest benchmarks/tests -q                                      # smoke tests
```

`--out DIR` writes `<suite>.json` (full record, machine-readable) and `<suite>.md`
(the scorecard). Results are gitignored.

---

## Reading a scorecard

Every row is one `(config, task)` pair. Columns are the reportable surface from
`benchmarks.md` design rule #3 — you never see quality without its cost:

| Column | Meaning |
| --- | --- |
| Pass@1 | Fraction of the ≥5 trials that passed the task's pass bar |
| Best-of-N | Did *any* trial pass (1.0/0.0) |
| Quality (mean) | Task quality score across trials |
| Cost $ (mean) | Modelled cost (data scanned + tool calls) |
| Latency s (mean) | Wall-clock / modelled latency |
| Data scanned (mean) | Fraction of the population read, `[0,1]` |

Each config row is printed with its **full config block** (sut, llm, sampling mode,
environment, agents enabled, trials, internet allowed) so a number is never divorced
from how it was produced. The JSON record additionally carries mean/median/std/min/max
for every quality key and every 4-tuple field.

**Model effects vs agent effects are separate runs.** To attribute a change correctly:
vary exactly one axis (`llm`, `agents_enabled`, `sampling_mode`, `environment`) and
hold the rest fixed. That is why "swarm vs single agent" and "adaptive vs fixed sample"
are different `RunConfig`s over the *same* tasks, not code branches.

---

## Prompts

All mission text and per-agent role descriptions live in
[`prompts/__init__.py`](prompts/__init__.py), versioned by `PROMPTS_VERSION`. Tasks
render their mission from the catalog (`render("mean_estimate", ...)`) rather than
hard-coding strings, so:

- a result is reproducible from a known prompt version,
- Track-4 ablation toggles the *same* agent roles (`AGENT_ROLES`) the full swarm uses,
- the real `avaloka` adapter and the local reference swarm read identical mission text.

Add a new mission to `MISSIONS`, a new agent role to `AGENT_ROLES`; `render()` and
`role()` raise loudly on typos or missing template fields.

---

## The tracks today

| Suite | Status | Runs locally? |
| --- | --- | --- |
| `track1_adaptive_sampling` | Runnable | ✅ stdlib only |
| `track4_swarm_ablation` | Runnable | ✅ stdlib only (local reference swarm) |
| `track5_reliability` | Runnable | ✅ stdlib only (local fault injection) |
| `dab` (external) | Avaloka-native pilot | ❌ requires DAB services + hosted model |

Tracks 2, 3, 6 and the remaining external benchmarks are specified in
`benchmarks.md` and slot into the same harness — each is a new `tracks/<name>/track.py`
(or `external/<name>/adapter.py`) plus, where the real system isn't reachable locally,
a local reference adapter like `local_swarm.py`.

---

## Extending

### Add a track
1. `tracks/<name>/track.py`: build `BenchmarkTask`s (missions via the prompt catalog),
   write evaluators returning `{"score": ..., "pass": ...}` plus any diagnostics, and
   `@register_suite("<name>", kind="track", ...)` a factory returning `(tasks, configs)`.
2. Add `from ..tracks.<name> import track` to `core/registry.py:_ensure_loaded`.
3. Add a stdlib smoke test under `benchmarks/tests/` asserting the intended directional
   result (e.g. "removing X degrades Y").

### Add an external benchmark
1. `external/<name>/adapter.py`: load its tasks into `BenchmarkTask`s targeting the
   `avaloka` SUT; register the suite; document data provenance + fingerprints.
2. Vendor data under `external/<name>/data/` (gitignored) with a fetch script — never
   commit held-out answers (design rule #1).

### Add a system under test
Implement the `SystemUnderTest` protocol (`adapters/base.py`) — one `run_mission`
method returning a `MissionOutcome` — and `@register_adapter("<key>")`. Set
`requires_cloud = True` if it drives Groq/cloud so the local-first gate covers it.

---

## CI

The CI pipeline runs `pytest tests/`. `tests/test_benchmarks_smoke.py` forwards
the harness smoke tests into that path, so the suite is exercised on every
`feature/*`, `bugfix/*`, `hotfix/*`, and `release/*` build — with zero cloud access.
