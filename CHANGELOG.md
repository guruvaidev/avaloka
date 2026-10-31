# Changelog

All notable changes to Avaloka are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

On `develop-1.6` since 1.6.0, not yet tagged. Every entry below is a change
that broke, or would have broken, a clean install or a first run.

### Security

- **The MCP server's onboarding and admin routes require authentication.**
  `/customers/*`, `/admin/customers`, `/connect` and `/connections` were open
  to anyone who could reach the MCP server. They now take the customer's API
  key or the API's `MCP_SERVICE_TOKEN`, and are scoped to the caller's own
  connections; a deleted connection's key is refused at once. **Set
  `MCP_SERVICE_TOKEN` and rotate stored database credentials**; see the
  advisory in [SECURITY.md](SECURITY.md)
  (`app/mcp_server/customer_dbs.py`, `app/mcp_server/multi_tenant_mcp_server.py`).
- **Connections registered before owners existed keep working for transfers.**
  After the change above, the API's service could no longer fetch credentials
  for a connection bound to no owner, so database-to-database transfers from
  or into it failed. The service may now use such a connection on a user's
  behalf for status, test-connection and credentials only, and the first
  connect binds it to that user. Tested against real Postgres databases
  (`tests/test_customer_dbs_dta_e2e.py`).

### Changed

- **The Celery worker has its own Kubernetes service account identity.** The
  chart can annotate it for cloud workload identity
  (`celery.serviceAccount.annotations`), and its Role may create and watch
  RayJobs for scheduled Ray analyses (`celery.rbac.rayJobs`, on by default).


- **Python floor raised to 3.11, ceiling pinned below 3.13.**
  `app/services/session_service.py` uses `async with asyncio.timeout(...)`,
  added in 3.11; on 3.10 it raised `AttributeError`, `save_session` swallowed
  it and returned `False` — a silent loss of session state. The upper bound
  exists because `pyarrow==16.1.0` publishes no wheel past cp312, and without
  a cap pip tries to build Arrow C++ from source. `pyproject.toml`, the CI
  matrix, `scripts/install.sh` and `scripts/doctor.py` now all state the same
  range.
- **Default inference provider is OpenRouter**, resolved key-aware: OpenRouter
  if its key is present, else Groq, else OpenAI, else OpenRouter so the choice
  still reports itself. One key now fronts many models, instead of a first run
  requiring a Groq account specifically. An explicit `INFERENCE_PROVIDER` is
  never second-guessed, and Groq remains fully supported
  (`app/core/inference.py`).
- **SQLAlchemy's upper cap removed and `psycopg[binary]` added.** SQLAlchemy
  2.1 changed which DBAPI a bare `postgresql://` URL selects — psycopg2 on
  2.0.x, psycopg v3 on 2.1.x — so the v3 driver must be installed for that URL
  to keep working (`requirements.txt`).

### Fixed

- **`avaloka train` works on a text-label target.** A classification target
  of words -- `healthcare_dataset.csv --target "Test Results"`, the example in
  `docs/INSTALL.md` -- crashed with "Cannot cast array data from dtype('O') to
  dtype('int64')". Labels were encoded only when the column's dtype was
  `object`, and the target arrived as pandas' string dtype. Any non-numeric
  label is now encoded (`avaloka/modeling.py`).
- **pandas is capped below 3** in `pyproject.toml`, `requirements.txt` and
  the five image/server requirements files. `>=2.0` let a clean install pick
  3.0.6, on which the hermetic gate fails 13 more tests (preparation agent,
  PII, null policy, local inference): pandas 3 reads text as its own `str`
  dtype and much of the code asks `dtype == object`.
  `tests/test_pandas_ceiling.py` keeps the cap in every file.
- **A null-like word in a spreadsheet stays a category in the CLI** (#479).
  The CLI's Excel reader read `None`, `NA`, `N/A` as missing in every column;
  it now follows the CSV rule from #474, and that rule also recognises
  pandas 3's `str` dtype (`avaloka/io/loader.py`,
  `file_handler/null_policy.py`).
- **The CLI no longer answers a measure it could not find with a row count**
  (#475). "Total revenue by region" over `gross_revenue` and `net_revenue`
  counted rows per region and reported `status="answered"`; it now reports
  `limited` and names the candidate columns. Count goals and goals that name
  the column are answered as before (`avaloka/goal_analysis.py`).

- **A model that does not beat its baseline is no longer announced as a success.**
  Training through the API reported "The training was successful" and offered
  predictions for a model below its trivial baseline (accuracy 0.627 vs 0.668),
  although the evaluation report said so. The report now states the comparison
  and says not to use the model; the claim verifier also rejects success wording
  for a below-baseline model (`app/agents/mta_v2/agent.py`,
  `app/agents/claim_verifier.py`).
- **The claim verifier catches a wrong "lowest" or "highest".** "The west region
  has the lowest at 68.12" passed beside a result whose minimum was 67.73. A new
  check compares each superlative with the result column the figure came from.
- **`tool_choice` is now chosen per provider**, not per client class. `local`,
  `openrouter` and `openai` all build the same `ChatOpenAI`, so the class name
  cannot tell them apart, and the two that matter need opposite values:
  OpenRouter rejects `required` (measured: 0/3 tool calls, 3/3
  `finish_reason="error"`), while Ollama returns prose instead of a tool call
  without it. `tool_choice_for_provider()` in `app/core/inference.py` keys the
  decision on the provider.
- **The planner and MTA broke when no API key was set at all** —
  `TOOL_CHOICE` was left unbound, so the code raised `NameError` instead of
  reporting a disabled LLM.
- **Training through the API works without MLflow configuration.** The trainer
  and the model endpoints defaulted to MLflow's file store, which the pinned
  MLflow refuses to write; every training turn failed and `GET /api/models`
  returned 500 unless `MLFLOW_TRACKING_URI` was set. Both now default to a
  local SQLite `mlflow.db` (`app/agents/mta_v2/tracking.py`), and the variable
  is documented in `docs/INSTALL.md`.
- **The API image runs the dependency set it pins.** An unpinned
  `pip install langchain-google-vertexai` after the requirements upgraded
  langchain-core to 1.x and pyarrow to 23 inside the published image. It is
  removed (Vertex is optional; see `requirements-cloud-llm.txt`), the build now
  runs `pip check`, and the arm64 image gets an arm64 `kubectl` instead of an
  amd64 binary it could not execute (`deploy/docker/Dockerfile.api`).
- **A cold clone could not install.** `requirements.txt` pinned plain
  `psycopg2`, which publishes no wheels, so pip built it from source and the
  install died.

### Added

- **A Ray jobs panel** in the UI, backed by a `ray_jobs` table (a new Supabase
  migration, also applied by the Helm chart). Scheduled analyses that are not
  Ray jobs run in an executor pod.
- **Auto Insights across several datasets or sheets** in one analysis.
- **The benchmark declares its known defects**, and its exit code follows
  them strictly, as `xfail(strict=True)` does: a declared run that fails
  does not fail the run; an undeclared failure does, as before; a declared
  run that passes fails it too, until its entry is removed. Before this, the
  eleven known-red runs made the pull-request Benchmark Scorecard step and
  the public CI `benchmark` job red on every pull request, so a real
  regression looked the same as the known ones. The scorecard's counts do
  not change; `--strict` restores exit 1 on any failure
  (`avaloka/benchmark/known_defects.py`).

- **A live CI tier**, because a hermetic gate cannot see provider behaviour:
  `.github/workflows/live-provider-weekly.yml` calls real hosted providers and
  `live-local-model-nightly.yml` exercises a local model server, both driving
  `tests/live/`. Neither runs on pull requests, so a provider outage cannot
  redden a PR.
- **The public tree is now generated**, not hand-maintained:
  `scripts/generate-oss.sh` builds it from `develop-1.6` per
  `oss/manifest.yaml`, which carries eight exclusion rules — several covering
  whole directories — plus an overlay and a set of assertions the generated
  tree must satisfy. Editing the public tree directly is not supported: the
  next generation overwrites it. Publishing is one squash commit on top of the
  public `main`; the intermediate `oss/1.6` branch and the generator's `--write`
  mode are retired, and the dry run compares against the published tree. See
  [`docs/oss-branch-model.md`](docs/oss-branch-model.md).

### Removed

- **Four documents retired from `docs/`**, all stale on both branches rather
  than merely internal: `OSS_LAUNCH_READINESS.md` (a launch plan for a date
  that has passed, naming internal project identifiers, with its go/no-go
  boxes unticked), `TEST_PLAN-1.5.2.md` and `TEST_REPORT-1.5.2.md` (58 KB of
  QA material for `feature/k8s-deploy-1.5.2`, three of whose four Day-0
  blockers are now false), and `TASK-conversational-friendliness.md` (an
  internal ticket measured against a harness that is not on this branch). Git
  history keeps all four. [`docs/DOCS_MAP.md`](docs/DOCS_MAP.md) is the new
  index of what remains.

### Known issues

- **Celery's soft time limit is above its hard limit** (1860 s and 1800 s in
  `app/core/celery_app.py`), so a long task is stopped at 30 minutes without
  the soft-limit warning that lets it clean up. A fix is in progress.

- **A multi-sheet workbook is read from its longest sheet**, in the upload
  path (`file_handler/excel_connector.py::_resolve_sheet`), whatever the
  question is about. On a 10-K-style workbook that is the balance sheet when
  the question is about the income statement; a workbook whose decoy sheet
  is longer than its answer sheet is read off the decoy. The frame parses
  cleanly, so nothing in the result says so. Workaround: upload the sheet on
  its own. The fix is to choose the sheet where the question is; the eight
  tests that grade it are strict xfails in
  `tests/test_spreadsheet_comprehension_upload_path.py`.
- **The CLI's spreadsheet reader handles fewer shapes than the upload
  path**: two-row and merged headers, footnote markers, subtotal and total
  rows, a second table below the first, `(1,234)` negatives and a
  "thousands" scale stated in a title are not understood. Ten benchmark
  tasks measure this and fail, as does the check that the conversational
  benchmark does not yet drive the product. The eleven are declared in
  `avaloka/benchmark/known_defects.py`: they still count as failures in the
  scorecard (39/50), but no longer fail the run. See
  [`docs/benchmarks.md`](docs/benchmarks.md).
- **MinIO's images are not currently pullable** from the registries the Helm
  chart points at, including by digest, so the chart's default in-cluster
  object store does not come up unaided. A migration to an alternative object
  store has been proposed but is **not merged**.

## [1.6.0] — 2026-09-16

The 1.6 development line. Numbered to match `VERSION`, the Helm chart and the
version `GET /version` serves — those had drifted to five different answers
(DEFECT E14.01), and a release bump now moves them together.

### Fixed

- **A forgeable auth stack shipped as the chart default** (DEFECT E9.19, P0).
  `values.yaml` enabled in-cluster Supabase with Supabase's *published* demo
  signing key, which `secret.yaml` adopts as the API's `SUPABASE_JWT_SECRET` —
  so anyone could mint a `service_role` token. Supabase is now off by default,
  credentials ship empty, and the chart refuses to render if you enable it
  without supplying your own.
- **One test file aborted the whole test suite.** A module-level `sys.exit(1)`
  in `tests/planner_graph_agent/` made pytest report INTERNALERROR and run
  nothing at all, which is why CI could not pass.
- **The API server no longer imports Daft to start.** Daft was imported at
  module scope for a single availability flag; on any interpreter whose wheel
  does not match the CPU, that import raises SIGILL rather than ImportError and
  kills the process.
- **Vendored loader copies had drifted 96 lines** — the inference image was
  missing the non-UTF-8 encoding fallback entirely, so a CSV that loaded in the
  agent failed in the deployed service.

### Changed

- Images publish to GHCR with content-addressed tags, so a deployment pulls what
  already exists instead of rebuilding it.
- `scripts/doctor.py` reports whether the environment will actually work,
  including an x86_64-under-Rosetta Python, which breaks native wheels in a way
  that looks like a hang.

## [0.2] — 2026

Avaloka as an agentic team of data scientists that takes any dataset from raw
data to model inference.

### Added

- **Multi-agent workflow** on LangGraph: Planner, Planner Graph, Summarizer,
  Sampling, Profiling, Coder, Validator, Execution, Data Transfer (DTA), Model
  Training (MTA v1 & v2), Visualization, and Scheduler agents coordinating
  through one shared `ETLState`.
- **Pseudocode-first code authoring** with a three-layer validation gate
  (syntax, schema-aware static analysis, logical LLM review) and deterministic
  fallbacks when the LLM is unavailable.
- **Multi-mode execution** — local, Kubernetes Job, or Ray cluster — chosen
  automatically from plan context and analysis fidelity.
- **Kubernetes deployment on any cluster** via a multi-cloud provider
  abstraction (local `kind`, GKE, EKS) behind a single `provision` / `connect`
  bootstrap, with a Helm chart, Ray/KubeRay integration, and a one-command
  `make up`.
- **Ray Serve inference-as-a-service** (`RayService`) and an MTA v2 managed
  inference path.
- **React.js web UI** (built with Lovable) in the `ui/` directory, backed by a
  local Supabase database, over the REST API.
- **Mission-planning CLI** (`python -m app.interfaces.cli.main`) and an **MCP
  server** sharing one core with the REST API, plus a multi-tenant MCP server
  for customer database connections.
- **Data connectors** for CSV, Excel, Parquet, Avro, Delta Lake, Apache Iceberg,
  JSON, and XML.
- **Sampling & profiling** (Daft-powered portfolio samples, semantic profiling)
  and **RAG-powered code retrieval**.
- **Asset persistence** to GCS/S3 with signed-URL retrieval; MLflow experiment
  tracking and model registry; Celery/RedBeat scheduling.
- **Open-source project setup**: Apache-2.0 license, documentation set
  (architecture, deployment, API, CLI, testing, version matrix), and
  contribution/security/conduct guides.

### Roadmap

*As stated at the time. Two of these have since moved — see Unreleased.*

- Azure AKS provider. **Done**: `app/infra/providers/azure_aks.py`, wired into
  `app/infra/providers/factory.py`.
- CLI remote mode (`--endpoint`) to drive a deployed backend. **Partly done**:
  the mission-planning CLI (`python -m app.interfaces.cli.main`) takes
  `--endpoint`, defaulting to `AVALOKA_API_URL`; the `avaloka` console script
  does not.
- Loading real trained models into the Ray Serve inference app by default.
  **Still open**: `app/serve/inference.py` reports its backend as
  `deterministic-stub` when no model is loaded, which keeps the platform
  exercisable without claiming a model is being served.
