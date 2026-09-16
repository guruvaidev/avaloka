# Changelog

All notable changes to Avaloka are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-09-14 — first open-source release

The first public release of Avaloka. The engine is the same one used internally
on the 1.6 line; this is its own version line because it is the first release
anyone outside the team can run.

### Run it with no network at all

- The sentence-transformers embedding model (~88 MB) is **baked into the API
  image at build time**. Retrieval previously fetched it from huggingface.co on
  the first query, which was slow enough to abort the memory retrieval and
  impossible on an air-gapped host. Verified with `docker run --network none`.
- `docs/INSTALL.md` documents the fully offline path, including a `--network
  none` check you can run rather than trust.
- Nothing calls home: no telemetry, no licence call, no model download.

### Fixes that a first-run user would otherwise have hit immediately

- **MinIO is pulled from quay.io.** `minio/minio` and `minio/mc` are no longer
  pullable anonymously from Docker Hub — even `:latest` — so the pod never
  started and the first upload returned `HTTP 500`. The `mc` image matters by a
  second route: it runs the bucket-creation hook.
- **The API image can be built at all.** The Vertex extra pinned
  `google-cloud-storage<3` against a `>=3.4` requirement.
- **`.tsv` and `.xml` upload** instead of returning `HTTP 500` for formats the
  API advertises as supported.
- **A provider with no usable key says so**, naming the variable, instead of
  degrading silently to canned text. A valid key for the *wrong* provider used
  to look exactly like no key at all.
- **Small datasets are read whole.** A size threshold was defined and never
  used, so anything under 1 GB was sampled — including a 12-row file, which was
  then reported to the user as "computed on a 12-row sample".

### Conversation

- "What is in this dataset?" is answerable with no model configured. The
  deterministic fallback held the computed schema and asked the user what they
  wanted instead of showing it.
- "Where should I start?" is no longer parsed as a SQL `WHERE` clause.
- "train a model to predict churn" routes to training, not inference.
- The sampling caveat's own suggested phrase — "run this on the entire dataset"
  — now does something.

### Memory plane

- Milvus ships in the chart with etcd, reusing the deployed MinIO.
- Chroma has a persistent volume. It was an `emptyDir`, so every pod restart
  silently erased what the system had learned.
- The retrieval circuit breaker was 3.0s against a 4–6s retrieval, so it
  aborted every call. The loop now persists across restarts.
- It exists to carry context forward. We make no claim that it improves
  analyses, and we have not measured that.

### Security and honest defaults

- No hardcoded Supabase project token in the edge functions.
- Shipping code no longer reads a private bucket belonging to the maintainers.
- The chart still defaults to Supabase's **published** demo credentials so a
  local install works with no setup — but they are labelled as public, rotation
  is documented, and `supabase.requireOwnCredentials` makes the chart refuse to
  render while they are in place. **Change them before any shared deployment.**

### Evidence

- All 126 Master Test Plan cases were executed against a live Kubernetes
  deployment: 73 PASS / 2 FAIL / 45 BLOCKED / 5 MANUAL / 1 N/A. The report and
  the harness that produced it are in `docs/test-reports/`.
- The two failures are conversational probes whose scores move run to run; that
  instability is the finding, and neither number is quoted as a score.
- The 45 blocked cases each name the dependency they need (cloud accounts,
  additional database servers, a Ray cluster, a load generator).

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
