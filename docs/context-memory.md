# Avaloka Context Memory Implementation Summary

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
- Enforces a 3-second circuit breaker.
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
- Falls back to local SQLite when Postgres is not configured.

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

Added memory-related dependencies:

- `chromadb`
- `pymilvus`
- `langchain-openai`
- `sentence-transformers`
- `sqlalchemy`

## Pipeline Update

Updated `bitbucket-pipelines.yml` with a branch-specific test path for:

```text
feature/context-memory
```

This branch now runs:

```bash
pytest tests/test_memory_semantics.py tests/test_memory_integration.py -q
```

The goal is to validate context-memory code without forcing unrelated full-repo infra/e2e tests in this feature branch.

The normal pipeline behavior remains for other feature branches.

## Testing Completed

Local memory test results:

```text
pytest tests/test_memory_semantics.py -q
15 passed
```

```text
pytest tests/test_memory_integration.py -q
18 passed, 5 skipped
```

```text
pytest tests/test_memory_semantics.py tests/test_memory_integration.py -q
33 passed, 5 skipped
```

Syntax checks also passed for memory-related files.

## Local Prompt Evaluation

The memory layer was tested with a YouTube analytics prompt sequence using:

```text
C:\Users\narla\Downloads\Global YouTube Statistics (1).csv
```

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
```

## Open Questions For Production

These are the main items to confirm before calling the implementation production-ready:

1. Which defaults are acceptable only for local/dev, and which configs must be explicit in staging/production?
2. Should fallback behavior be enabled in production by default, or controlled by environment flags?
3. What failure modes must be guaranteed for Redis, Chroma, Postgres, and Milvus?
4. Which Layer 1 and Layer 4 paths should move from best-effort behavior to mandatory infrastructure behavior?
5. What logs, metrics, traces, or alerts are required around writes, retrievals, fallbacks, and cache misses?
6. Is current artifact reuse sufficient, or do we need stronger versioning, lineage, deduplication, and audit logs?
7. What additional tests are required: integration, chaos/failure, load, migration, or end-to-end workflow tests?
8. What deployment checklist is required: env validation, startup checks, health checks, migrations, backup strategy, and rollback plan?
9. What security controls are needed for tenant isolation, sensitive data handling, memory access, and audit logs?

## Current Branch

The implementation is on:

```text
feature/context-memory
```

The cleaned branch tip contains one context-memory commit authored by Pramodd.

## Summary

The project now has an agentic context-memory foundation integrated into Avaloka's workflow, planner, execution, and test layers. It supports multi-layer retrieval, fallback-safe operation, memory injection into planner state, historical-analysis tooling, artifact reuse hints, and execution-result recording.

The implementation is suitable for local/dev and feature validation. Production readiness depends on final decisions around explicit infrastructure configuration, fallback policy, observability, security, and stronger end-to-end operational tests.
