# Changelog

All notable changes to Avaloka are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

On `develop-1.6` since 1.6.0, not yet tagged. Every entry below is a change
that broke, or would have broken, a clean install or a first run.

### Changed

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
- **A cold clone could not install.** `requirements.txt` pinned plain
  `psycopg2`, which publishes no wheels, so pip built it from source and the
  install died.

### Added

- **A live CI tier**, because a hermetic gate cannot see provider behaviour:
  `.github/workflows/live-provider-weekly.yml` calls real hosted providers and
  `live-local-model-nightly.yml` exercises a local model server, both driving
  `tests/live/`. Neither runs on pull requests, so a provider outage cannot
  redden a PR.
- **`oss/1.6` is now generated**, not hand-maintained:
  `scripts/generate-oss.sh` builds it from `develop-1.6` per
  `oss/manifest.yaml`, which carries eight exclusion rules — several covering
  whole directories — plus an overlay and a set of assertions the generated
  tree must satisfy. Editing the public tree directly is not supported: the
  next generation overwrites it. See
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
