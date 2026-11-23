# Avaloka CLI — Test Plan

| | |
|---|---|
| **Applies to** | the `avaloka` command (package `avaloka/`, version 1.0.0) on `main` |
| **Scope** | the nine CLI commands, the mission runner and fireflies, workload sizing, source resolution, deployment gates, the serving plan, the chat session, packaging and the process boundary |
| **Type** | test plan **with** test code. Case status: EXISTS (file::test) = already automated before this plan; NEW = implemented by this plan in `tests/avaloka/test_cli_plan_*.py`; GATED = implemented, skips unless its substrate is present |
| **Derived from** | *Avaloka on Kubernetes — Test Plan* (PR #229). Same five-layer, cheapest-first structure, same hermetic invariant, same "pin every finding with a test" rule, re-targeted from the Helm chart to the CLI |

---

## 1. Objective

Define the complete verification matrix for the open-source CLI and implement
the part of it that can run on a laptop with no key, no cluster and no network.
The pre-existing suite (`tests/avaloka/`, 77 cases) covers the happy path of
each mission and the error sentences the CLI prints for bad input. The additions
target the contracts between the documentation and the binary, the arithmetic
behind the economics table, every branch of the deployment gates, the chat
session (previously untested), the process boundary, and the serving sequence.
Eight defects found while writing the plan are tracked in §6; each is pinned by
a strict-xfail test so the suite reports exactly what is still broken.

## 2. Strategy

Five layers, cheapest first. Invariant: a defect is caught at the lowest layer
able to detect it. Nothing trains a model to prove a fact `plan_deployment` can
prove on an empty directory.

| Layer | What runs | Marker | Dependencies | Cadence |
|---|---|---|---|---|
| L0 — static contracts | pyproject, VERSION, README_CLI.md, the Typer app's registry; no mission | — | `pip install -e ".[dev]"` | every PR, blocking |
| L1 — unit | pure functions: workload, budget, economics, deploy gates, serving plan, suggestions | — | none | every PR, blocking |
| L2 — in-process CLI | every command through Typer's `CliRunner` on 400-row fixtures; missions run for real | — | none | every PR, blocking |
| L3 — process boundary | `python -m avaloka`, the console script, `main()`'s crash contract; local Ray batch; the benchmark | `slow` for Ray/benchmark | `ray` (requirements.txt) for `batch` | every PR; slow cases nightly |
| L4 — external substrate | `infer --apply`, `batch --target kubernetes --apply` | `cluster` | Docker + kubectl + kind, `AVALOKA_TEST_ALLOW_APPLY=1` | manual / pre-release |

**Hermetic invariant.** `pytest tests/avaloka -q` passes with no Docker, no
cluster, no API key and no network. `-m "not slow"` finishes in about a minute.
A missing dependency skips with a reason; it never fails or hangs.

**Safety rails** (themselves under test):

1. The suite never spends a key. An autouse fixture deletes `ANTHROPIC_API_KEY`
   for every test in `tests/avaloka/`, so `--llm` is exercised only for its
   degradation contract (S5.15).
2. The suite never shells out to docker or kubectl. Every apply-path test
   patches `serving._have` and `serving._run` and asserts on the argv it would
   have run (S5.09, S5.13, S8.01–S8.03).
3. The one live test (S8.05) refuses any kube context that is not `kind-*` or
   `minikube`, and is gated by both a marker and an environment flag.
4. Every bundle is written under `tmp_path`. Tests that exercise the default
   output directory `chdir` into `tmp_path` first.

## 3. Scope

**In:** `avaloka/cli.py` (analyze, train, chat, batch, deploy, infer,
coordinate, benchmark, version, `--version`), `avaloka/__main__.py`,
`missions.py`, `fireflies/*`, `workload.py`, `io/*`, `deploy.py`, `serving.py`,
`ray_batch.py`, `coordinator.py`, `chat.py`, `persona.py` and `llm/client.py`
(fallback behaviour only), `benchmark/*`, packaging (`pyproject.toml`,
`VERSION`, `examples/`), `README_CLI.md` as a contract.

**Out:** the platform in `app/` (its own suites); the mission-planner CLI
`python -m app.interfaces.cli.main` and the MCP server
(`tests/test_cli_experience.py`, `tests/k8s/test_t4_cli.py`); the *content* of
LLM narration; the behaviour of generated manifests once applied to a cluster
(L4 proves the sequence and the endpoint, not Kubernetes).

**Borderline, deliberately in:** `README_CLI.md` and `--help` text, because they
are the interface a new user meets first and they drift silently (S1, F7).

## 4. Environment matrix

| Layer | Setup |
|---|---|
| L0–L2 | `python -m venv .venv && pip install -e ".[dev]"` |
| L3 | the above plus `ray` (installed by `./scripts/install.sh` via requirements.txt) |
| L4 | Docker Desktop running, kubectl, a `kind` cluster as the current context, `AVALOKA_TEST_ALLOW_APPLY=1` |

Toggles: `AVALOKA_TEST_ALLOW_APPLY` (L4 gate), `AVALOKA_TRACEBACK` (asserted
both ways in S7), `ANTHROPIC_API_KEY` (removed by the suite).

Fixtures (`tests/avaloka/conftest.py`): `churn_csv` (400 rows, real signal),
`leaky_csv`, `titanic_like_csv` (nulls in text columns), `uncorrelated_csv` (no
signal, no quality issues), `trained_package` (one deployable train bundle per
session), `synthetic_package` (a hand-written deployable layout with a
well-formed manifest).

---

## 5. Suites

### S1 — Packaging and documentation contracts (L0)
`tests/avaloka/test_cli_plan_contracts.py`

| ID | Case | Expected | Status |
|---|---|---|---|
| S1.01 | Console script | `[project.scripts] avaloka = "avaloka.cli:main"`, callable | NEW |
| S1.02 | Version single source | `VERSION` == `avaloka.__version__` == `--version` == `version` | NEW |
| S1.03 | Documented commands exist | every `avaloka <word>` in README_CLI.md is a registered command | NEW |
| S1.04 | `--help` completeness | lists all nine commands; leads with a runnable example | NEW (example: EXISTS `test_cli_robustness::test_help_leads_with_a_runnable_example`) |
| S1.05 | Metric list parity | `train --metric` help lists exactly `VALID_METRICS` | NEW |
| S1.06 | Deployment level names | the four `LEVELS` names appear in README_CLI.md | NEW |
| S1.07 | Bundled sample | `examples/*.csv` in package-data; `churn.csv` loads with a `churned` column | NEW |
| S1.08 | Standalone import | `import avaloka.cli` in a fresh interpreter imports nothing from `app.*` | NEW |
| S1.09 | Dependency footprint | exactly the eight runtime deps; `anthropic` only in the `llm` extra | NEW |
| — | `--version` exists | | EXISTS `test_cli_robustness::test_version_flag_exists` |

### S2 — Workload sizing and sampling (L1)

| ID | Case | Expected | Status |
|---|---|---|---|
| S2.01 | Lane boundaries | ≤100k rows and ≤50 MB online; ≤5M and ≤1 GB sampled; beyond that batch; either axis promotes; edges inclusive | NEW (coarse version EXISTS `test_experience::test_classify_lanes`) |
| S2.02 | `recommend_batch` | false for online, true otherwise | NEW |
| S2.03 | `as_dict()` | JSON-serialisable; lane as string; estimated counts spoken with `~` | NEW |
| S2.04 | `route == classify(peek)` | | NEW |
| S2.05 | Sampling obeys the lane | 400 rows + lane `sampled_online` is sampled (seed 42, said so); lane `online` is not | NEW |
| S2.06 | Stratified sampling | target with ≤50 classes stratifies; every class survives | NEW |
| — | `peek` counts rows; `materialize` for local, file://, sqlite, memory, missing object-storage backend; `route` over any source | | EXISTS `test_experience`, `test_sources` |

### S3 — Budget, ledger and economics (L1)

| ID | Case | Expected | Status |
|---|---|---|---|
| S3.01 | Zero budget | `limit_usd=0` is a ceiling (not unlimited); the first non-zero spend raises naming the action and amounts | NEW |
| S3.02 | Boundary arithmetic | inclusive at the limit; `remaining`, `fraction_used`; `None` is infinite | NEW |
| S3.03 | Multiplier denominator | `labour_value / (invoice + review)`; review floored at 0.25 h; 60 min at $80/h ⇒ 4.0x exactly | NEW |
| S3.04 | Empty ledger | 0 h, 0 value, 0.0x, review still floored | NEW |
| — | Hard ceiling raises; unlimited never blocks; ledger sums; multiplier finite for a free local mission | | EXISTS `test_core` |

### S4 — Deployment gates and serving plan (L1, synthetic packages)

| ID | Case | Expected | Status |
|---|---|---|---|
| S4.01 | Missing package / bad target | `FileNotFoundError`; `ValueError` naming the valid targets | NEW |
| S4.02 | Cumulative gates | L1 files only ⇒ achieved 1; L2 gate lists its six missing files | NEW |
| S4.03 | No validation report | cap 2; staging refused with the reason | NEW |
| S4.04 | Cap bounds achievement | full artifacts, cap 2 ⇒ 2 | NEW |
| S4.05 | Production governance | cap 4 without `--approve` ⇒ 3 and `--approve` named; with it ⇒ 4; cap 3 + approve ⇒ "clean validation verdict" | NEW (unapproved case EXISTS `test_missions::test_deploy_blocks_production_without_approval`) |
| S4.06 | Manifest per target | local/docker → compose; kubernetes; ray; absent ⇒ `None` | NEW |
| S4.07 | Commands | target-specific; cost guard prepended; scale-to-zero noted | NEW |
| S4.08 | `serving.prepare` | names from the manifest; stages `mlflow, docker, kubernetes, endpoint`; exact commands | NEW |
| S4.09 | Registry and MLflow | image rewritten under the registry; `docker push` added; registration first | NEW (EXISTS partial `test_experience::test_serving_prepare_plan`) |
| S4.10 | MLflow needs the script | `--mlflow-uri` without `mlflow/register.py` is refused up front | NEW |
| S4.11 | `curl_example` | one line, `/predict`, the packaged example body | NEW |
| — | Non-deployable layout refused; registry image lands in the rendered manifest; apply registers MLflow then applies | | EXISTS `test_experience` |

### S5 — Commands end to end (L2)
`tests/avaloka/test_cli_plan_commands.py`

| ID | Case | Expected | Status |
|---|---|---|---|
| S5.01 | Default output dir | `./avaloka-analysis` | NEW |
| S5.02 | Bundle facts | mission.json kind/goal/verdict/level 1; workload.json lane; unlimited budget recorded | NEW |
| S5.03 | `--quiet` | no persona lines; verdict, swarm table, economics, deliverables remain | NEW |
| S5.04 | `--deployable` | package files only when asked; "Ship it" panel names `avaloka infer … --apply` | NEW |
| S5.05 | Baseline in the summary | mission.json carries score, baseline, strategy, `beats_baseline`; console names the best model | NEW |
| S5.06 | Budget exceeded | exit 2, "Budget exceeded", no traceback | NEW |
| S5.07 | `deploy` | gate table, validator cap, `--approve` hint, `deploy_plan.json` | NEW |
| S5.08 | `deploy` errors | missing package / bad target exit 1 as sentences | NEW |
| S5.09 | `infer` plan-only | pipeline table, commands, "This was a plan", `inference_plan.json`; **no** shell-out | NEW |
| S5.10 | `infer` on an analysis bundle | exit 1, missing files named, `--deployable` suggested | NEW |
| S5.11 | `coordinate` | two workers, 400 rows converged, `coordination.json` | NEW (library-level EXISTS `test_coordinator`) |
| S5.12 | `batch` local | exact row count, `batch_profile.json`, RayJob manifest | NEW, `slow` (library-level EXISTS `test_experience`) |
| S5.13 | `batch --target kubernetes` | prints kubectl commands; `--apply` with no kubectl says so and runs nothing | NEW, `slow` |
| S5.14 | `benchmark` | green, `OVERALL`, JSON scorecard | NEW, `slow` (runner EXISTS `test_benchmark`) |
| S5.15 | `--llm` without a key | exit 0; identical file set and verdict to the plain run; model cost 0 | NEW |
| S5.16 | sqlite URI | `sqlite:///db#table` analyses end to end | NEW |
| S5.17 | Excel input | `.xlsx` analyses end to end (skips without openpyxl) | NEW |
| — | Header-only, binary, directory, missing file, unwritable output, unknown target/metric, findings printed, WARN names its check | | EXISTS `test_cli_robustness` |
| — | All deliverables for analyze/train; leakage fails validation; budget scales candidates | | EXISTS `test_missions` |
| — | Swarm activates exactly the right fireflies | | EXISTS `test_swarm` |

### S6 — The chat session (L2)

Zero tests existed for `avaloka chat` before this plan.

| ID | Case | Expected | Status |
|---|---|---|---|
| S6.01 | `columns`, `quit` | every column named; farewell printed; exit 0 | NEW |
| S6.02 | `suggest`, `help` | `1. train churned`; every idea names a real column; all verbs listed | NEW |
| S6.03 | Numbered pick | `2` after `suggest` runs it ("Taking 2") and writes the bundle | NEW |
| S6.04 | Out-of-range pick | corrected, nothing runs | NEW |
| S6.05 | `train churnd` / `train` | did-you-mean; "Which column"; nothing trained | NEW |
| S6.06 | No dataset loaded | asks for a dataset first | NEW |
| S6.07 | `batch`, `deploy` in chat | redirected to the shell command; nothing written | NEW |
| S6.08 | Closed stdin | clean exit 0, no traceback | NEW |
| S6.09 | `suggest_next_steps` | grounded in column names, de-duplicated, bounded by `limit` | NEW |

### S7 — The process boundary (L3)

| ID | Case | Expected | Status |
|---|---|---|---|
| S7.01 | `python -m avaloka --version` | exit 0, version printed | NEW |
| S7.02 | Console script | `avaloka --help` from the interpreter's bin dir (skips if not installed) | NEW |
| S7.03 | Crash contract | unexpected exception ⇒ exit 1, cause named, `AVALOKA_TRACEBACK=1` hint, no stack frames | NEW |
| S7.04 | `AVALOKA_TRACEBACK=1` | the real exception propagates | NEW |
| S7.05 | `UnusableDataset` | "isn't something I can analyse" with the detail | NEW |
| S7.06 | Ctrl-C | exit 130, "Stopped." | NEW |
| S7.07 | Normal exits | typer's `SystemExit` codes pass through unchanged | NEW |

### S8 — External substrate (L4): hermetic halves and the gated live half

| ID | Case | Expected | Status |
|---|---|---|---|
| S8.01 | Apply sequence | `docker build`, `docker push`, `kubectl apply -f -` (piped), `kubectl rollout status deployment/<name>`, then endpoint resolution; `applied=True` | NEW (on `synthetic_package`, see F8) |
| S8.02 | First failure stops | failed build ⇒ no kubectl; note carries the error text | NEW |
| S8.03 | No tools | both named in notes; nothing run; no crash | NEW |
| S8.04 | Manifest render | every Deployment's first container image rewritten to the planned image | NEW |
| S8.05 | Live `infer --apply` | builds, applies to the current kind context, prints "REST API live" | GATED (`cluster`, `slow`, `AVALOKA_TEST_ALLOW_APPLY=1`) |

---

## 6. Findings (design-time)

Each is pinned by a test in `tests/avaloka/test_cli_plan_defects.py`, marked
`defect` and `xfail(strict=True)`. While the defect stands the test fails as
expected and the suite is green. When a fix lands the test passes, strict xfail
fails the suite, and the fix removes the marker. Each must be fixed or
explicitly accepted in `README_CLI.md` before the next CLI release.

| # | Finding | Pinned by | Disposition |
|---|---|---|---|
| F1 | `train` crashes on any dataset with nulls in a text column: the Data Engineer casts text to pandas `string`, whose missing value `pd.NA` reaches scikit-learn's imputer (`X != X`) and raises "boolean value of NA is ambiguous". Reproduced on Kaggle Titanic. Most real exports have this shape | `test_f1_*` | **fix before release** — cast `string` columns to `object` / `np.nan` in `modeling.build_preprocessor` or stop the cast in `data_engineer` |
| F2 | `analysis.py` embeds the dataset path exactly as typed; a relative path makes the script fail from inside the bundle, which is where the bundle README says to run it | `test_f2_*` | fix — resolve to an absolute path (or copy the data into the bundle) at generation time |
| F3 | `analysis.py` writes `transformed_dataset.parquet` into the current directory, not the bundle | `test_f3_*` | fix with F2 |
| F4 | `analyze`/`train --llm` with no `ANTHROPIC_API_KEY` degrades silently; only `chat` prints a notice | `test_f4_*` | fix — print the same notice the chat prints (matches the platform's "honest failure" rule) |
| F5 | With no correlation ≥ 0.5 and no quality issue, "Key findings" is a generic sentence naming no column and no number, against README_CLI.md's "every observation is pinned to a measured statistic" | `test_f5_*` | decide — surface the strongest measured association with its value, or say explicitly that nothing exceeded the threshold |
| F6 | In `--quiet` mode the console prints the holdout score without its baseline; the baseline sentence exists only in persona prose, and a passing `beats_baseline` check is not printed | `test_f6_*` | fix — print the baseline next to the score in `_print_answer`, independent of the persona |
| F7 | `README_CLI.md` does not document `coordinate`, `benchmark` or `version` | `test_f7_*` | document |
| F8 | `ml_engineer` writes `deployment/kubernetes.yaml` as a single YAML list (`write_yaml(path, [deployment, service])`), not two `---` documents. `serving.prepare` falls back to default names, `serving.apply` fails at "manifest render", and `kubectl apply -f` would reject it. `avaloka infer --apply` cannot work on any bundle Avaloka produced | `test_f8_*` (two tests) | **fix before release** — `yaml.safe_dump_all` in `_deployment_manifests` |

## 7. Traceability

| README_CLI.md promise | Suites |
|---|---|
| "Three commands cover most work" (analyze, train, chat) | S5.01–S5.06, S6 |
| "Reads CSV, TSV, Parquet and Excel", any source URI | S5.16, S5.17, EXISTS `test_sources`, `test_excel_loading` |
| "Every failure names the problem and the next thing to try" | S5.06, S5.08, S5.10, S7.03–S7.06, EXISTS `test_cli_robustness` |
| "A verdict never appears without the check that produced it" | EXISTS `test_a_warn_verdict_says_which_check_warned` |
| "A model score never appears without the baseline it has to beat" | S5.05, **F6** |
| Workload lanes online / sampled / batch | S2.01–S2.05, S5.02 |
| Deployment levels 1–4, never silently promoted | S4.02–S4.05, S5.07 |
| Economics with an honest denominator | S3.03, S3.04 |
| Local execution is free and offline; LLM optional | S1.08, S1.09, S5.15, **F4** |
| Fireflies activate only when needed | EXISTS `test_swarm` |
| `avaloka infer`: MLflow → Docker → Kubernetes → REST | S4.08–S4.11, S5.09, S8, **F8** |
| Reproducible bundle (`python analysis.py`) | **F2**, **F3** |

## 8. CI gating

| Trigger | Runs | Gate |
|---|---|---|
| Every PR | `pytest tests/avaloka -m "not slow"` (S1–S8 hermetic halves, all pinned findings) | blocking |
| Nightly | `pytest tests/avaloka` including `slow` (Ray batch, benchmark) | blocking; red ⇒ triage before merge |
| Pre-release | S8.05 on a kind cluster with `AVALOKA_TEST_ALLOW_APPLY=1` | manual |
| Any change under `avaloka/` or to `README_CLI.md` | S1 mandatory | blocking |

## 9. Exit criteria

* S1–S7 green; the only skips are openpyxl (S5.17) and the console script (S7.02) when absent.
* S8.01–S8.04 green; S8.05 green at least once on kind before a release that advertises `infer --apply`.
* F1 and F8 fixed (their tests flipped to plain passes). F2–F7 each fixed or accepted in `README_CLI.md`.
* `pytest tests/avaloka -m "not slow"` under 90 s on a laptop; no test spends a key or shells out to docker/kubectl.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Training time makes L2 slow | one session-scoped `trained_package`; deploy/infer tests only read it |
| Rich wraps console output at 80 columns under `CliRunner` | assert on whitespace-collapsed text for prose; on exact paths only where the CLI promises not to wrap |
| Strict xfails flip when a fixture drifts | F5's fixture is constructed to produce zero quality issues; F6 skips the FAIL-verdict case it is not about |
| A developer's exported key turns a degradation test into a billable call | autouse fixture removes the key for the whole suite |
| `ray` absent on a minimal install | batch cases carry `slow`; CI nightly installs requirements.txt |
