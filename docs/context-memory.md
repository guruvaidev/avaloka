# Avaloka Context Memory Implementation Summary

> **What this document is.** An implementation summary of the context-memory
> layer: which modules exist, what each is responsible for, and which
> environment variables they read. It is a developer's map of the subsystem, not
> a user guide — for what memory does *for you* as a user, read
> [USER_GUIDE.md](USER_GUIDE.md) §8 and
> [TECHNICAL_USER_GUIDE.md](TECHNICAL_USER_GUIDE.md) §6.2. The code it describes
> is merged; file paths and defaults below were re-verified against the tree.

## Purpose

This document summarizes the context-memory work implemented for Avaloka. The goal was to add an agentic memory layer that gives the Planner and Coder agents access to dataset context, user-specific usage patterns, team artifacts, and historical analysis results without relying only on prompt history.

The implementation follows the Avaloka Agentic Memory design specification and focuses on four memory layers:

- Layer 1: Domain Context
- Layer 2: Usage Context
- Layer 3: Artifact and Inference Context
- Layer 4: Time-Series Analysis Memory

## High-Level Flow

The runtime flow is:

```text
User request
  -> workflow.memory_injection_node
  -> app.services.memory_plane.retrieve_memory
  -> MemoryOrchestrator checks Redis, Chroma, Postgres, and Milvus layers
  -> memory_hints and session_logic_signature are added to ETLState
  -> Planner receives memory-aware state
  -> Execution outputs are recorded back into memory
```

The planner receives only the top memory hints needed for the current step, while longer accumulated context remains available through the memory layer.

## Files Added

### app/services/embedding_utils.py

Shared embedding utility for the memory stack.

Responsibilities:

- Uses OpenAI embeddings when `OPENAI_API_KEY` is configured.
- Falls back to local `sentence-transformers` when available.
- Falls back to zero vectors only as a final degraded mode.
- Pads or trims vectors to a consistent dimension.

### app/services/mcp_cache_loader.py

MCP schema loader for Layer 1 domain context.

Responsibilities:

- Calls the MCP `list_tables` tool.
- Retrieves dataset schema metadata on cache miss.
- Allows Redis schema cache to be hot-loaded from MCP.

### app/services/milvus_recorder.py

Celery task for Layer 4 time-series analysis memory.

Responsibilities:

- Records execution or analysis output in Milvus.
- Embeds textual results before insertion.
- Stores results under session/notebook context.
- Uses lazy runtime behavior to avoid blocking execution on memory failures.

### tests/test_memory_semantics.py

Spec-oriented tests for memory semantics.

Coverage includes:

- Planner tool schema for `retrieve_historical_analysis`.
- Milvus schema metadata fields.
- Temporal filtering with `time_range_seconds`.
- Notebook scoped retrieval.
- Embedding path behavior.
- Circuit breaker timeout behavior.
- MCP env variable selection.

## Major Files Updated

### app/graph/etl_state.py

Added memory-related state fields:

- `memory_hints`
- `active_mcp_servers`
- `session_logic_signature`
- `prior_artifact_found`
- `milvus_context_id`
- `memory_context_unavailable`

These fields let the LangGraph workflow pass memory context between the memory plane, planner, execution, and UI layers.

### app/api/workflow.py

Added `memory_injection_node` before the planner.

Responsibilities:

- Extracts the latest user query.
- Calls the memory plane.
- Adds memory outputs into `ETLState`.
- Ensures `milvus_context_id` and `active_mcp_servers` are populated.
- Supports `MCP_SERVER_URL` as the canonical MCP env variable, with `MCP_URL` as fallback.

### app/services/memory_plane.py

Implemented the central memory orchestrator.

Responsibilities:

- Coordinates all four memory layers.
- Enforces a hard retrieval timeout, **20 seconds by default**
  (`MEMORY_CIRCUIT_BREAKER_TIMEOUT`, `app/services/memory_plane.py:49`).
- Returns safe fallback payloads when memory is unavailable.
- Maintains top-3 planner hint contract.
- Supports in-process fallback memory for local/dev and missing infrastructure.
- Adds prior artifact metadata into planner hints when available.

Returned fields include:

- `memory_hints`
- `accumulated_memory_hints`
- `session_logic_signature`
- `prior_artifact_found`
- `memory_context_unavailable`

> **The timeout was 3.0s and that was a bug.** A successful retrieval measures
> 4.4s warm and 5.8s cold against a healthy deployment, so a 3-second breaker
> fired on *every* call and the memory plane never returned anything. The only
> symptom was `memory_hints=None`, which is indistinguishable from "nothing has
> been learned yet" — which is why it survived so long. The default is now 20s,
> leaving headroom for a cold embedding load without letting a hung backend
> stall a turn. The reasoning is recorded at
> `app/services/memory_plane.py:39-49`. If you lower it, pre-warm the embedding
> model in the image first.

### app/services/db/redis_client.py

Implemented Layer 1 Domain Context.

Responsibilities:

- Stores dataset schema by `dataset_id`.
- Stores schema embeddings when embedding providers are available.
- Performs vector similarity lookup when possible.
- Falls back to keyword overlap when embeddings are unavailable.
- Falls back to an in-process store when Redis or the Redis package is unavailable.

### app/services/db/chroma_client.py

Implemented Layer 2 Usage Context.

Responsibilities:

- Stores user/session logic signatures.
- Uses persistent ChromaDB when available.
- Supports configurable retention through `CHROMA_RETENTION_DAYS`.
- Falls back to an in-process namespace store in local/dev.

### app/services/db/postgres_client.py

Implemented Layer 3 Artifact and Inference Context.

Responsibilities:

- Stores artifact metadata keyed by deterministic query hash.
- Checks whether a prior artifact exists.
- Retrieves artifact metadata for reuse.
- Uses Postgres when `POSTGRES_URL` is configured.
- Falls back to local SQLite (`artifacts/layer3_artifacts.db`) when Postgres is
  not configured. `strict_memory_infra_enabled()` turns that fallback into an
  error, because SQLite here is dev-only
  (`app/services/db/postgres_client.py:33`).

> **Which Postgres driver `POSTGRES_URL` selects is version-dependent.** The URL
> goes straight into `create_engine` (`app/services/db/postgres_client.py:50`),
> and SQLAlchemy 2.1 changed which DBAPI a bare `postgresql://` URL picks —
> `psycopg2` on 2.0.x, `psycopg` (v3) on 2.1.x. `requirements.txt` installs both
> (`psycopg2-binary>=2.9.0` and `psycopg[binary]>=3.1`) precisely so the URL
> resolves either way, and `sqlalchemy>=2.0` is deliberately uncapped. If you
> want to stop depending on the version, name the driver:
> `postgresql+psycopg://…` or `postgresql+psycopg2://…`.

### app/services/db/milvus_client.py

Implemented Layer 4 Time-Series Analysis Memory.

Responsibilities:

- Creates a Milvus collection for analysis memory.
- Stores metadata fields:
  - `timestamp`
  - `session_id`
  - `notebook_id`
  - `agent_role`
  - `cell_id`
  - `artifact_ref`
  - `content`
- Supports search filters for session, notebook, and time range.
- Falls back gracefully when Milvus is unavailable.

### app/agents/planner.py

Updated planner schema and prompt behavior.

Changes:

- Added `HistoricalAnalysisParams`.
- Added `retrieve_historical_analysis` to the planner tool contract.
- Added planner guidance for historical analysis retrieval.
- Planner can consume `memory_hints` and `session_logic_signature`.

### app/agents/execution_agent.py

Added memory recording after successful execution.

Responsibilities:

- Sends successful execution summaries to Layer 4 memory.
- Records artifact metadata in Layer 3 storage.
- Uses `milvus_context_id` as notebook/session context.
- Lazily imports the Milvus recorder task to avoid a circular import with `celery_app`.

### app/api/server.py and app/api/streamlit_app.py

Added memory visibility and state propagation support.

Purpose:

- Surface memory fields in runtime state.
- Preserve memory context in API/UI flows.
- Help inspect memory behavior during development.

### docker-compose.yml

Added Milvus-related local infrastructure.

Purpose:

- Provides local Milvus stack support.
- Includes Attu for Milvus visualization.

### requirements.txt

Added memory-related dependencies. The spellings and pins as they stand:

| Requirement | Line |
| --- | --- |
| `chromadb` (unpinned) | `requirements.txt:145` |
| `pymilvus>=2.3.0` | `requirements.txt:169` |
| `langchain-openai>=0.3,<1.0` | `requirements.txt:174` |
| `sentence_transformers>=3.2.1` — note the underscore | `requirements.txt:140` |
| `sqlalchemy>=2.0`, uncapped on purpose | `requirements.txt:150` |

## CI

`bitbucket-pipelines.yml:52` still carries the branch-specific step for
`feature/context-memory`, which runs only the two memory test files so the
feature branch is not gated on unrelated infra/e2e tests. Other branches take
the normal path.

That step is **historical** — the work is merged, so nothing is pushed to that
branch any more. The repository also has GitHub Actions workflows
(`.github/workflows/ci.yml`, `.github/workflows/images.yml`, and the live tiers
`.github/workflows/live-provider-weekly.yml` and `.github/workflows/live-local-model-nightly.yml`); the memory tests
run there as part of the normal suite rather than as a special case.

## Tests

```bash
pytest tests/test_memory_semantics.py tests/test_memory_integration.py -q
```

Re-run during this documentation refresh: **42 passed, 5 skipped**
(`tests/test_memory_semantics.py` 24 passed; `tests/test_memory_integration.py` 18 passed,
5 skipped). The five skips are the cases that need live Redis, Chroma or Milvus.

Earlier revisions of this page recorded `15 passed` / `33 passed` as fixed
figures; the suites have grown since. Pass counts in prose go stale the next
time anyone adds a test — run the command rather than trusting a number here.

## Local Prompt Evaluation

The memory layer was exercised with a YouTube analytics prompt sequence over a
Global YouTube Statistics dataset. The equivalent file is in the repository at
`app/sample_data/Global_YouTube_Statistics.csv`.

Observed behavior:

- Dataset schema was loaded as Layer 1 domain context.
- Analysis logic accumulated across prompts.
- The system remembered concepts such as growth momentum, country opportunity, underserved niches, earnings efficiency, channel age, top creators, and view efficiency.
- Planner-facing hints were limited to top 3 as designed.
- Local fallback behavior worked even without live Redis, Chroma, or Milvus services.

## Current Runtime Behavior

The system can run in two modes:

### Local/dev fallback mode

When external services are missing:

- Redis falls back to in-process schema memory.
- Chroma falls back to in-process usage signatures.
- Postgres falls back to SQLite.
- Milvus returns empty retrieval results when unavailable.
- Session memory still accumulates through in-process fallback state.

### Infrastructure-backed mode

When services and env vars are configured:

- Redis stores and retrieves schema context.
- Chroma stores user usage signatures.
- Postgres stores artifact metadata.
- Milvus stores embedded time-series analysis memory.
- Embedding quality improves with OpenAI or sentence-transformers.

## Important Config Variables

Memory-related variables supported by code:

```text
MCP_SERVER_URL
MCP_URL
MCP_API_KEY
REDIS_URL
REDIS_SCHEMA_TTL
REDIS_VECTOR_SIM_THRESHOLD
CHROMA_STORAGE_PATH
CHROMA_COLLECTION
CHROMA_SIGNATURE_CAP
CHROMA_RETENTION_DAYS
POSTGRES_URL
MILVUS_HOST
MILVUS_PORT
MILVUS_COLLECTION
MILVUS_PARTITION
MILVUS_TOP_K
MILVUS_VECTOR_DIM
MILVUS_DEFAULT_NOTEBOOK
MILVUS_DEFAULT_CELL
OPENAI_API_KEY
OPENAI_EMBEDDING_MODEL
SENTENCE_TRANSFORMER_MODEL
EMBEDDING_TARGET_DIM
MEMORY_CIRCUIT_BREAKER_TIMEOUT
MEMORY_DEFAULT_STYLE_HINT
MEMORY_MCP_FALLBACK_URL
MEMORY_PREFERENCES_TTL_SECONDS
MEMORY_STRICT_INFRA
MEMORY_ALLOW_MEMORY_FALLBACKS
```

Defaults worth knowing, all from `app/services/memory_plane.py:38-61`:

| Variable | Default |
| --- | --- |
| `MEMORY_CIRCUIT_BREAKER_TIMEOUT` | `20.0` seconds |
| `MEMORY_DEFAULT_STYLE_HINT` | `"Prefer clean pandas code."` |
| `MEMORY_MCP_FALLBACK_URL` | `http://localhost:8080/sse` — used only when `MCP_SERVER_URL` is unset |
| `MEMORY_PREFERENCES_TTL_SECONDS` | `604800` (7 days), matching the session lifetime so a remembered preference outlives the session it was stated in |
| `REDIS_SCHEMA_TTL` | `86400` |
| `MILVUS_TOP_K` | `5` |

`MCP_SERVER_URL` is the canonical name; `MCP_URL` is read as a fallback in
`app/api/workflow.py:1034-1037`, and `MEMORY_MCP_FALLBACK_URL` is the
memory-plane-specific override.

## Open Questions For Production

These were recorded when the layer was built and have not been closed out in
this document. Treat them as an unresolved checklist, not as known gaps that
someone is actively tracking:

1. Which defaults are acceptable only for local/dev, and which configs must be explicit in staging/production?
2. ~~Should fallback behavior be enabled in production by default, or controlled by environment flags?~~ **Answered in code.** `strict_memory_infra_enabled()` (`app/services/memory_runtime.py:41`) rejects the dev-only fallbacks whenever the environment is a production one, unless `MEMORY_ALLOW_MEMORY_FALLBACKS=true` is set deliberately; `MEMORY_STRICT_INFRA=true` forces strict mode anywhere. So: off by default in production, opt-in.
3. What failure modes must be guaranteed for Redis, Chroma, Postgres, and Milvus?
4. Which Layer 1 and Layer 4 paths should move from best-effort behavior to mandatory infrastructure behavior?
5. What logs, metrics, traces, or alerts are required around writes, retrievals, fallbacks, and cache misses?
6. Is current artifact reuse sufficient, or do we need stronger versioning, lineage, deduplication, and audit logs?
7. What additional tests are required: integration, chaos/failure, load, migration, or end-to-end workflow tests?
8. What deployment checklist is required: env validation, startup checks, health checks, migrations, backup strategy, and rollback plan?
9. What security controls are needed for tenant isolation, sensitive data handling, memory access, and audit logs?

## Summary

The project now has an agentic context-memory foundation integrated into Avaloka's workflow, planner, execution, and test layers. It supports multi-layer retrieval, fallback-safe operation, memory injection into planner state, historical-analysis tooling, artifact reuse hints, and execution-result recording.

The implementation is suitable for local/dev and feature validation. Production readiness depends on final decisions around explicit infrastructure configuration, fallback policy, observability, security, and stronger end-to-end operational tests.
