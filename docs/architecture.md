# Avaloka Architecture

Avaloka is an **agentic team of data scientists**. You describe what you want in
plain language; a group of specialized agents plans the work, samples and
profiles your data, writes and validates code, runs it locally or on a
distributed cluster, trains models, serves inference, and persists every
artifact — carrying a dataset all the way from **raw data to model inference**.

This document maps the pieces and how a request flows through them. For running
it on a cluster see [deployment.md](deployment.md); for the HTTP surface see
[api.md](api.md); for the command line see [cli.md](cli.md).

---

## 1. The mental model

```text
        You  →  React web UI (ui/)  |  CLI  |  REST API
                 │
                 ▼
        ┌──────────────────┐      the "team lead": converses, decides the
        │  Planner          │      next action, delegates to specialists
        └──────────────────┘
                 │  orchestrates
   ┌─────────────┼───────────────────────────────────────────────┐
   ▼             ▼              ▼            ▼           ▼         ▼
 Sampling &   Coder +        Execution    Data       Model      Visualization
 Profiling    Validator      (local/Ray/  Transfer   Training   + Scheduler
              (3-layer)      k8s)         Agent      Agent
                                          (DTA)      (MTA v1/v2)
                 │
                 ▼
        Storage, MLflow, Ray Serve inference
```

Every agent shares one state object — `ETLState` (`app/graph/etl_state.py`, 80+
fields) — and the whole thing is wired together as a **LangGraph** graph
(`app/api/workflow.py::build_graph`). The graph is what makes the "team"
coordinate: each node reads and writes the shared state, and routing functions
decide who acts next.

## 2. The agent roster

| Agent | Module | Role on the team |
| ----- | ------ | ---------------- |
| **Planner** | `app/agents/planner.py` | Talks to the user, runs iterative analysis, decides the next step (keep planning, provision infra, summarize, code, train, schedule, execute). |
| **Planner Graph** | `app/agents/planner_graph_agent.py` | Renders the plan as a Mermaid DAG so you can see the execution path before running. |
| **Summarizer** | `app/agents/summarizer.py` | Freezes the conversational plan into a structured JSON contract for downstream agents. |
| **Sampling** | `app/agents/sampling_agent*.py` | Quick samples and Daft-powered portfolio samples with per-column statistics. |
| **Profiling** | `app/agents/profiling_agent.py` | Semantic profiling: infers domain, ranks columns by predictive power, flags data-quality issues. |
| **Coder** | `app/agents/coder.py` | Writes a numbered pseudocode blueprint, then emits Python that adheres to it. Deterministic fallback when the LLM is offline. |
| **Validator** | `app/agents/validator.py` | Three layers — syntax, schema-aware static analysis, and a logical LLM review — gate code before execution; feedback loops back to the Coder. |
| **Execution** | `app/agents/execution_agent.py` | Runs the code locally, as a Kubernetes Job, or on a Ray cluster. |
| **Data Transfer (DTA)** | `app/agents/data_transfer_agent/` | Generates, validates, and runs Daft-based ETL for cross-format / cross-store transfers. |
| **Model Training v1 (MTA)** | `app/agents/model_training_agent.py`, `mta/` | PyTorch training, ONNX export, MLflow tracking. |
| **Model Training v2 (MTA v2)** | `app/agents/mta_v2/` | Distributed training on Ray, model registry, and inference-service deployment. |
| **Visualization** | `app/agents/visualization_agent.py` | Turns execution outputs into charts. |
| **Scheduler** | `app/agents/scheduler.py` | Periodic runs, status, results, cancellation via Celery/RedBeat. |
| **Infrastructure** | `app/agents/infra_agent.py` | Provisions Kubernetes when the plan calls for it. |

## 3. Request lifecycle (raw data → inference)

1. **Ingest.** A file lands via `POST /api/upload` (or a cloud URI / database
   connection). `file_handler/` decodes the format (CSV, Excel, Parquet, Avro,
   Delta, Iceberg, JSON, XML) and a schema + sample are attached to the state.
2. **Plan.** The Planner converses with you, calling Sampling/Profiling to ground
   itself in the actual data, and proposes a plan. The Planner Graph visualizes
   it.
3. **Summarize.** The plan becomes a JSON contract.
4. **Author + validate.** The Coder writes pseudocode → Python; the Validator's
   three layers approve it or bounce it back with feedback.
5. **Route execution.** Based on `analysis_fidelity` and the plan, the router
   (`app/execution/router.py`) picks:
   - `quick_sample` → **local**
   - `portfolio_samples` → **Daft / Ray**
   - `entire_dataset` → **k8s-Ray** (via Celery)
6. **Execute.** The Execution agent runs it and captures stdout/stderr and output
   artifacts.
7. **Train (optional).** If the plan is a modeling task, MTA v1 (local PyTorch)
   or MTA v2 (distributed Ray) trains, exports, and registers the model in
   MLflow.
8. **Serve (optional).** The trained model is deployed for inference — see
   §5 on the serving path.
9. **Persist + visualize.** Generated code, outputs, and viz configs are
   background-persisted to object storage; the Visualization agent renders
   charts.

The canonical graph is in the top-level [README](../README.md#end-to-end-workflow)
and `avaloka-flowchart.mermaid`.

## 4. Deployment & execution substrate

This is the deployment surface — the ability to stand Avaloka up as a backend on
**any Kubernetes cluster, local or cloud**, and run its compute on Ray.

```text
app/infra/
  cluster_bootstrap.py   driver:  provision (create cluster) | connect (attach to Ray)
  install_k8s.py         preflight tool checks + KubeRay operator install
  cloud_provisioner.py   cluster lifecycle dispatch
  providers/             ClusterProvider abstraction behind a factory
    factory.py             local | gcp | aws   (azure is on the roadmap)
    local_kind.py          kind (Kubernetes-in-Docker)
    gcp_gke.py             Google Kubernetes Engine
    aws_eks.py             Amazon EKS
    base.py                run_command + the provider interface
  ray_manager.py         Ray lifecycle: KubeRay operator, RayCluster, RayService, connect
  deploy_stack.py        build/side-load images + `helm upgrade --install`
```

The provider abstraction means one bootstrap path works everywhere: `provision`
creates or reuses a cluster, installs the **KubeRay** operator, applies the
**RayCluster**, deploys the app via Helm, and stands up **Ray Serve**;
`connect` attaches to an existing Ray cluster instead. The KubeRay operator and
the Ray images are aligned on a single Ray version (see the
[version matrix](versions.md)).

See [deployment.md](deployment.md) for the step-by-step.

## 5. Serving / inference

Two inference paths exist in the codebase:

- **Ray Serve (`app/serve/inference.py`)** — a `RayService`
  (`deploy/helm/ray/rayservice.yaml`) hosts the Serve app at
  `avaloka-inference-serve-svc:8000`. `GET /` reports readiness; `POST /` with
  `{"features": {...}}` returns a prediction. The chart wires this as the default
  inference backend (`config.inferenceBackend: rayserve`), and the API posts
  predictions to it via `RAY_SERVE_URL`. Loading a real trained model into the
  `InferenceService` (vs. the scaffold response) is on the roadmap — check
  `model_loaded` in the readiness response to see whether a model is live.
- **MTA v2 managed inference (`app/agents/mta_v2/`)** — trains on Ray/GKE and
  deploys a containerized inference service fronted by a GCP API Gateway, driven
  from the API's `/api/models/{run_id}/configure-inference-service` route. This
  path is GCP-specific and selected with `config.inferenceBackend: gateway`.

The direction is to serve managed inference on the cloud-neutral Ray Serve
platform by default, keeping the GCP gateway path as an opt-in.

## 6. Interfaces — one core, many front doors

`app/interfaces/service.py` is the shared core: `plan_mission(intent)` compiles a
mission and produces an execution plan, and `planned_to_dict()` is the single
JSON representation every non-CLI surface returns. Three front doors call it:

- **CLI** — `python -m app.interfaces.cli.main` (`plan` / `analyze`). See
  [cli.md](cli.md).
- **MCP server** — `python -m app.interfaces.mcp.server`, exposing
  `plan_mission`, `estimate_cost`, `describe_environment`, `inspect_intent` as
  Model Context Protocol tools.
- **REST** — the FastAPI app in `app/api/server.py`. See [api.md](api.md).

The design invariant: identical intent yields identical output across all
surfaces, so they cannot drift apart.

The **React web UI** (`ui/`, built with Lovable, backed by a local Supabase
database) is an end-user front door over the REST API — the same backend the CLI
and MCP server share.

## 7. Storage & services

| Concern | Implementation |
| ------- | -------------- |
| Web UI | A **React.js** app (built with [Lovable](https://lovable.dev/)) in the `ui/` directory, backed by a **local Supabase** database. It consumes the REST API (§6). |
| Object storage | `app/core/storage.py` — cloud-agnostic `IBlobStore` over GCS / S3 |
| Session/thread state | Redis (`app/core/cache.py`, with a circuit breaker) |
| Sampling-profile cache & UI database | Supabase — the sampling profile cache (`app/agents/sampling_persistence.py`) and the local database backing the React UI |
| Experiment/model tracking | MLflow (`app/agents/mta*/`) |
| Job registry | GitHub repository (via the persistence service) |
| Scheduling | Celery + RedBeat (`app/core/celery_app.py`) |
| Code retrieval (RAG) | Daft embeddings (`app/rag/`) seed the Coder with prior snippets |
| **Memory plane** | Four tiers, all deployed by the chart — see §7.1 |

### 7.1 The memory plane

`app/services/memory_plane.py` orchestrates four persistent tiers. They are not
interchangeable caches; each answers a different question.

| Tier | Store | Holds | Lifetime |
| --- | --- | --- | --- |
| **L1** | Redis | Session/thread working state | Session |
| **L2** | Chroma | Usage context (`layer2_usage_context`) | Persistent (PVC) |
| **L3** | Postgres | Structured session records | Persistent |
| **L4** | Milvus | Episodic memory, retrieved by vector similarity | Persistent |

`memory_injection_node` is the **entry point of the agent graph** — every turn
passes through it before the conversational agent sees the message. It returns
`memory_hints` plus `memory_context_unavailable`, and that second field is
deliberately always a boolean: an absent field reads as "no memory was needed",
which is indistinguishable from "memory could not answer". Only `True`/`False`
are honest states, and the distinction is guarded by
`tests/test_memory_plane.py`.

**Embedding is local.** Queries are embedded with sentence-transformers
(`all-MiniLM-L6-v2`), and the model is **baked into the API image at build time**
under `HF_HOME=/opt/hf`. Nothing is fetched at query time, which is what allows
the whole stack to run air-gapped. A retrieval takes roughly 4–6 s cold and
under a second warm; the circuit breaker (`MEMORY_CIRCUIT_BREAKER_TIMEOUT`) must
stay above that or it aborts every call.

**What the loop is for.** It carries context forward so later work has something
to build on. It is *not* claimed to make analyses measurably better — that has
not been measured. See `docs/test-reports/three-pillar-coverage.md`.

## 8. Configuration — why each dependency exists

Avaloka is a complex system because it spans the **entire** raw-data-to-inference
lifecycle: reasoning, state, storage, distributed compute, training, tracking,
serving, and observability. Each external dependency in `.env` serves one stage
of that lifecycle. This section explains **why** each is there and what happens
without it, so the configuration reads as a map of the system rather than an
opaque list. The full template is [`.env.example`](../.env.example); the exact
variable names live there.

A guiding principle (see §9): **required** dependencies are few, and most others
**degrade honestly** — the system keeps working with reduced capability and says
so, rather than failing outright.

| Lifecycle stage | Depends on | Env vars | Required? | Why it's needed / what breaks without it |
| --------------- | ---------- | -------- | --------- | ---------------------------------------- |
| **Reasoning** (the agents' "brains") | Groq Cloud LLMs | `GROQ_API_KEY_PLANNING_AGENT`, `GROQ_API_KEY_CODING_AGENT` | **Yes** | The Planner and Coder call Groq for fast inference. Without keys the agents fall back to deterministic stubs only — no real planning or code synthesis. |
| **Session & thread state** | Redis | `REDIS_URL` | **Yes (for the API)** | Uploads, threads, and inference store session state in Redis. Without it the API returns `503` on `/api/upload`, `/threads`, and inference. A circuit breaker tolerates brief hiccups. |
| **Scheduling / long jobs** | Redis (Celery broker) | `CELERY_REDIS_URL` | For scheduled/async work | The Scheduler and full-dataset (`entire_dataset`) runs dispatch through Celery/RedBeat. Without it, only synchronous execution paths run. |
| **API authentication** | JWT (HS256) | `JWT_SECRET` | **Yes (for the API)** | Every non-public route validates a bearer JWT. Without a shared secret, auth cannot be verified. |
| **Durable artifacts** | GCS / S3 blob store | `GCS_BUCKET`, `AVALOKA_ASSET_BUCKET`, `GOOGLE_APPLICATION_CREDENTIALS` | For persistence | Generated code, execution outputs, viz configs, and models are persisted here and retrieved via signed URLs. Without it, artifacts live only for the request. |
| **Distributed compute** | Ray / KubeRay | `RAY_ADDRESS`, `MTA_RAY_DASHBOARD_URL` (shared-cluster training), provider creds | For Ray/k8s execution | `portfolio_samples` and `entire_dataset` fidelity run on Ray; large jobs and MTA v2 training run on the general Ray cluster. Ray Serve remains isolated for inference. Local execution needs none of this. |
| **Model training & registry** | MLflow | `MLFLOW_*`, `DISABLE_MLFLOW` | Optional | MTA logs experiments and registers models in MLflow. Falls back to local file storage when unset. |
| **Managed inference** | GCP (GKE + API Gateway) or Ray Serve | `GCP_PROJECT_ID`, `INFERENCE_*`, `K8S_*` | For served models | Deploys trained models as an inference service. The cloud-neutral default is in-cluster Ray Serve; the GCP-gateway path needs these. |
| **Sampling cache** | Supabase | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | Optional | Caches full profiling results so repeated requests skip recomputation. Without it, profiles are recomputed each time. |
| **Job registry** | GitHub repo | `GITHUB_SYSTEM_TOKEN`, `GITHUB_JOB_REGISTRY_REPO` | Optional | Job definitions are pushed to a Git registry for reproducibility/audit. Skipped when unset. |
| **Customer data connections** | MCP server | `MCP_SERVER_URL` | For DB chat / DTA | The multi-tenant MCP server brokers customer database credentials for the Data Transfer Agent and database chat. |
| **Graph runtime** | LangGraph | `LANGGRAPH_API_URL` | For the graph server | Points the API at the LangGraph runtime that executes the agent graph. |
| **Observability** | LangSmith | `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Optional | Traces agent runs for debugging. Purely additive. |
| **Data discovery (optional sidecar)** | openclaw | `OPENCLAW_*` | Optional | An optional discovery sidecar; off by default (`OPENCLAW_ENABLED=0`) — the agent uses native Python discovery when disabled. |
| **Infra defaults** | — | `DEFAULT_INFRA_PLATFORM`, `DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING` | Optional | Tunes routing decisions (which cloud, when to escalate to Ray training). |

**The short version:** to run the agents you need **Groq keys**; to run the API
you also need **Redis** and a **`JWT_SECRET`**. Everything else unlocks a specific
capability — cloud storage, distributed training, managed inference, caching,
tracking, database connections — and is optional until you need that stage.

## 9. Design principles

- **One shared state, many specialists.** Agents coordinate through `ETLState`,
  not through direct calls to each other.
- **Lazy heavy imports.** Deployment/serving modules import Ray/torch/daft inside
  functions so they can be inspected and unit-tested without the full stack.
- **Structured shell-outs.** All infra commands go through `run_command` and
  return a uniform outcome dict.
- **Degrade honestly.** When an LLM or service is unavailable, agents fall back
  to explicit safe stubs rather than pretending to succeed.
- **Cloud-neutral where it counts.** Storage, execution, and (increasingly)
  serving are abstracted so the same workload runs on kind, GKE, EKS, or AKS.

---

## 10. Design changes — August 2026 (the 1.6 PR wave)

The sections above describe the planner/coder/validator core. Since then the
system gained three layers, each with a one-line rationale and the PR that
carried it:

### The evidence layer (#271)
`contract.py` (agents declare stage/reads/writes; undeclared writes are
dropped, a crashing agent becomes a structured failure), `integrity_agent.py`
(leakage screened **before** training), `evaluation_agent.py`
(cross-validation with the splitter the data requires, a mandatory trivial
baseline, CI-gated wins), `claim_verifier.py` (the narrative checked against
the evidence, deterministically). The Validator checks the *code*; this layer
checks the *claim*.

### Model routing and fallback (#272, #286, #287)
One table decides *which model* (`model_config.py`, env-overridable,
live-verified); one factory decides *which provider*; `model_fallback.py`
decides *what happens when the provider fails* — OpenRouter first, then a
local vLLM/Ollama tier **sized by deployment profile** (cluster: 31B-class;
laptop: 4B, stated in the logs). `discover_models.py` finds new model
families from live catalogues with zero code changes and verifies tags before
suggesting pulls. Born of a real outage: Groq removed the entire Llama line
and six agents 404'd mid-analysis.

### Conversational agent with optional swarm (#273/#282, from #157)
`avaloka_agent/` routes conversation before planning; swarm is resolved by
**import probe** (`capabilities.py`) — an OSS build without the module
degrades to no narration, and no environment variable can claim a capability
whose code is absent. That property is what the edition split
(`docs/EDITIONS.md`, `docs/INSTALL.md`) relies on.

### Inference serving lifecycle (#265)
The legacy gateway minted an `inference-service-{uuid}` namespace plus a GCP API
Gateway per model and reclaimed neither: **eighteen accumulated between
2026-04-22 and 2026-08-14**, each holding a pod, ~1 vCPU and its own
LoadBalancer. Serving now runs in **one cluster**, scales to **zero when idle**,
and puts workers on **Spot**. The rule this encodes: an idle inference endpoint
is pure burn, so idleness must be a state the system can reach on its own rather
than a cleanup someone remembers to run.

### Training integrity (#292)
Preprocessing is fitted on the **training split only** — imputation values,
one-hot vocabularies and scaler statistics — and merely *applied* to validation.
Fitting before the split is textbook target leakage: the model sees test-set
statistics and reports a score it cannot reproduce in production. Both training
paths implement it (`local_trainer.fit_feature_preprocessing`,
`ray_job.fit_ray_feature_preprocessing`), and both carry an explicit
unknown-category channel so a value unseen in training is represented rather
than silently dropped.

This is a class of leakage the evidence layer **cannot** catch: `integrity_agent`
inspects the DataFrame, and preprocessing-order leakage leaves no trace there —
the rows are byte-identical either way; what leaks is a statistic computed at the
wrong moment. Unit tests on both paths are the only defence, which is why they
are required rather than optional.

The two paths are separate implementations by necessity (`ray_job.py` ships
inside a Docker image and cannot import `app.*`), so their agreement on
preprocessing, split rule and metric keys is maintained deliberately and should
be asserted by test rather than assumed.

### Fallback coverage and verification (#281, #290)
Fallback is not per-agent opt-in: **all twelve agents** resolve their LLM through
`model_fallback.py`, each Groq primary paired with an OpenRouter equivalent via
`with_fallbacks()`. Eight call `attach_fallback` directly; `dta_coder`,
`dta_validator`, `mta_v2` and the MTA classifier arrive through a single wrap in
`agent_llm.py`. A capability that covers most agents is a capability you cannot
reason about during an outage, so the coverage is the point.

Verification runs at two altitudes. `tests/prompt_suite/` drives **281 prompts**
through the real API end to end — upload → chat → training → inference →
transfer → schedule — across four suites (`v4` functionality F1–F13, `v2`
evidence-based, `v1` legacy Kaggle, and `heavy`, which is opt-in and k8s-only
because its entire-dataset and ~1,900-column items materialise 200 MB–1.9 GB and
will OOM a bare API). Above it, the 1.7 benchmark harness scores behaviour
deterministically. The first answers *"does every capability still work?"*; the
second answers *"is the answer any good?"*.

### Cloud portability (status)
`app/infra/providers/` resolves `local | gcp | aws | azure` through one factory
(`get_provider`), and `cloud_provisioner`, `cluster_bootstrap`, `deploy_stack`
and `ray_manager` all route through it; `tests/k8s/test_t5_multicloud.py` covers
it. **`app/agents/mta_v2/` does not.** Model serving and training still assume
GCP and read `GCP_PROJECT_ID` directly, so Avaloka is multi-cloud in the
provisioner and single-cloud in the serving path. Routing MTA through the same
factory is the remaining work; until then, treat AWS/Azure inference as
manual configuration.

### Measurement (1.7 line: #277, #283, #284)
A benchmark harness with plugin-discovered suites, a
(Quality, Cost, Latency, Data-Scanned) 4-tuple, resumable journals, and a
pre-flight smoke suite over five surfaces (harness/prompts/loops/graph/
models). Two field incidents shaped it: a runner type bug that scored its own
crash as 0.0 quality, and an upstream 429 that would have read as "the 4B
model fails everything" — both caught because harness faults are tagged
distinctly from task failures.

Current diagram: `docs/images/avaloka-architecture-2026-08.png`
(source: `avaloka-architecture-2026-08.mermaid` at repo root).


## 11. Design changes — September 2026 (deployment hardening)

This wave came from **driving a live Kubernetes deployment** rather than from
reading code. Every item below shipped and was invisible to the unit suite,
because each is a failure of what a user encounters rather than of what a
function returns.

### The memory plane became real

Previously the code existed and nothing ran it. Four faults were stacked behind
a single symptom — `memory_hints: None` — and each hid the next:

1. Milvus was never deployed; the chart expected one to exist elsewhere.
2. Chroma had **no persistent volume**, so every pod restart erased tier L2.
3. The Milvus MinIO credentials referenced key names the secret does not use,
   with `optional: true`, so they were silently unset and Milvus fell back to
   default credentials.
4. `MEMORY_CIRCUIT_BREAKER_TIMEOUT` was **3.0 s** against a 4–6 s retrieval, so
   the breaker aborted every call.

The chart now deploys Milvus with etcd (reusing the deployed MinIO), gives
Chroma a PVC, and ships a breaker above the real retrieval time.

### Offline operation is a first-class mode

The embedding model is baked into the image, `memory.offlineEmbeddings` defaults
to true, and `docs/INSTALL.md` documents a `--network none` verification. With a
local model server, the full stack runs with no internet at all.

### Routing corrections in the conversational layer

* `exploration` was answered from chat while holding the computed schema, so
  "what is in this dataset?" returned a request for clarification.
* `"Where should I start?"` was parsed as a SQL `WHERE` clause.
* `"train a model to predict churn"` matched `predict` before `train a model`
  on declaration order, and routed to inference.
* Sampling-mode switches never reached the planner that honours them — so the
  phrase the agent itself suggests did nothing.

Keyword matching is now longest-match rather than first-rule-wins, so
specificity decides rather than the order rules happen to be written in.

### Size-aware fidelity

`SMALL_THRESHOLD` was defined and never used, so everything under 1 GB was
sampled — including a 12-row file, which was then reported to the user as a
sample. Inputs at or below the small threshold are now read whole, locally.

### Honest failure

A provider configured with an unusable key degraded silently to canned text.
`_diagnose_missing_model()` now names the missing variable and says when a key
for a *different* provider is present. The deterministic fallback remains for
the genuinely-no-model case, which is a supported way to run.
