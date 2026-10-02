# Changelog

All notable changes to Avaloka are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

- Azure AKS provider.
- CLI remote mode (`--endpoint`) to drive a deployed backend.
- Loading real trained models into the Ray Serve inference app by default.
