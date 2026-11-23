# Avaloka Agent Capability Flow

> **Downloadable source map** | **As of commit:** `5062aacc2866` | **Date:** 2026-08-15
>
> This map distinguishes running code from conditional integrations, swarm narration, and designs.
> A named persona is not assumed to be an executable agent.

## Status legend

| Marker | Status | Meaning |
| --- | --- | --- |
| **E** | Executable | Checked-in behavior is callable or is an active graph node. |
| **C** | Conditional | Code exists but requires optional packages, credentials, services, or infrastructure. |
| **N** | Narration only | A team/swarm label describes work but is not an independent graph node/operator. |
| **P** | Proposed | Design only; it must not be presented or routed as implemented. |

```mermaid
flowchart LR
    E["E: executable"]:::exec
    C["C: conditional integration"]:::cond
    N["N: narration-only label"]:::narr
    P["P: proposed"]:::prop
    classDef exec fill:#d8f3dc,stroke:#2d6a4f
    classDef cond fill:#fff3bf,stroke:#b08900
    classDef narr fill:#e9ecef,stroke:#6c757d,stroke-dasharray: 5 5
    classDef prop fill:#f8d7da,stroke:#a61e4d,stroke-dasharray: 7 4
```

## Capability overview in natural language

Avaloka accepts natural-language intent through REST, the React UI, CLI, or MCP. Its front-door agent
classifies intent, discovers data, restores bounded session context, and either answers directly or
delegates. The planner can sample and profile data, clarify the request, resolve database/cloud
connections, provision infrastructure, schedule work, or freeze the conversation into a contract.

For analysis, the coder emits pseudocode and Python. Syntax, schema-aware static/safety, complex-loader,
and logical checks gate execution; logical rejection can return to the coder. Approved code runs
locally, on Ray, or as a Kubernetes job. Results can be rendered, persisted, and visualized.

Separate paths cover heterogeneous read-only database questions, Daft data transfer, local/distributed
model training, model tracking/export, inference deployment, memory, scheduling, and benchmarks. PDF,
image/OCR, video-frame, LeRobot, and RLDS support is bounded and conditional—not general multimodal
reasoning or robotics training.

The first DAB tranche now includes **Feature Transformer, Statistician, Experimenter, Conformal
Prediction, deterministic Cartographer routing, and an answer Skeptic**. Live schema introspection,
semantic document extraction, and full evidence recomputation remain proposed. Existing helpers
remain useful primitives for those future contracts.

## End-to-end intent and routing

```mermaid
flowchart TD
    U["Natural-language intent"]:::input
    SURF["UI / REST / CLI / MCP"]:::exec
    LEAD["Avaloka front-door\nclassify + discover"]:::exec
    MEM["Session and memory injection"]:::cond
    DIRECT["Direct reply"]:::exec
    PLAN["Planner"]:::exec
    SAMPLE["Sampling portfolio"]:::exec
    PROFILE["Semantic profiling"]:::cond
    INFRA["Infrastructure agent"]:::cond
    SUM["Structured summarizer"]:::exec
    DAG["Planner graph renderer"]:::cond
    CODE["Coder"]:::cond
    VS["Syntax validator"]:::exec
    VT["Static/schema/safety validator"]:::exec
    ROUTE{"Execution route"}:::decision
    LOCAL["Local"]:::exec
    RAY["Ray"]:::cond
    K8S["Kubernetes"]:::cond
    VL["Logical validator"]:::cond
    FIX{"Refine?"}:::decision
    VIZ["Render / visualize / persist"]:::cond
    SCHED["Celery/RedBeat schedule"]:::cond

    U --> SURF --> LEAD
    LEAD -->|simple intent| DIRECT
    LEAD -->|data mission| MEM --> PLAN
    PLAN -->|inspect| SAMPLE --> PLAN
    PLAN -->|profile| PROFILE --> PLAN
    PLAN -->|provision| INFRA --> PLAN
    PLAN -->|defer| SCHED
    PLAN -->|ready| SUM --> DAG --> CODE --> VS --> VT --> ROUTE
    ROUTE --> LOCAL & RAY & K8S
    LOCAL --> VL
    RAY --> VL
    K8S --> VL
    VL --> FIX
    FIX -->|bounded repair| CODE
    FIX -->|accept| VIZ

    classDef input fill:#d0ebff,stroke:#1971c2
    classDef exec fill:#d8f3dc,stroke:#2d6a4f
    classDef cond fill:#fff3bf,stroke:#b08900
    classDef decision fill:#e7f5ff,stroke:#1c7ed6
```

Evidence: [`app/api/workflow.py`](../app/api/workflow.py),
[`app/graph/etl_state.py`](../app/graph/etl_state.py),
[`app/agents/avaloka_agent/agent.py`](../app/agents/avaloka_agent/agent.py),
[`app/agents/planner.py`](../app/agents/planner.py), and
[`app/api/ui_contract.py`](../app/api/ui_contract.py).

## Analytical and data-query mission

```mermaid
flowchart TD
    Q["Question + sanctioned sources"]:::input
    DISC["Discovery: schemas, samples, connections"]:::exec
    TAB["Tabular/object loaders"]:::exec
    CX["PDF, image/OCR, sampled video"]:::cond
    ROB["LeRobot v3 / RLDS directories"]:::cond
    DB["Postgres / Mongo / SQLite / DuckDB"]:::cond
    REC["Recursive and format schema validators"]:::exec
    QP["Typed question/evidence planning"]:::exec
    LOOP["DataQueryMission / coder tool loop"]:::cond
    TOOLS["Sanctioned read-only tools"]:::cond
    CHECK["Code and answer checks"]:::exec
    ANS["Answer + evidence + telemetry"]:::exec
    DAB["Official DAB evaluator"]:::cond
    FT["Feature Transformer"]:::exec
    STAT["Ava-Statistician"]:::exec
    SKEP["Executable Ava-Skeptic"]:::exec

    Q --> DISC
    DISC --> TAB & CX & ROB & DB
    TAB --> REC
    CX --> REC
    ROB --> REC
    REC --> QP
    DB --> QP
    QP --> LOOP --> TOOLS --> LOOP
    LOOP --> CHECK --> ANS --> DAB
    QP -. future typed transforms .-> FT
    QP -. future estimand contract .-> STAT
    CHECK -. future independent review .-> SKEP

    classDef input fill:#d0ebff,stroke:#1971c2
    classDef exec fill:#d8f3dc,stroke:#2d6a4f
    classDef cond fill:#fff3bf,stroke:#b08900
    classDef partial fill:#e9ecef,stroke:#6c757d
    classDef prop fill:#f8d7da,stroke:#a61e4d,stroke-dasharray: 7 4
```

Evidence: [`avaloka/io/loader.py`](../avaloka/io/loader.py),
[`avaloka/io/complex.py`](../avaloka/io/complex.py),
[`avaloka/io/schema.py`](../avaloka/io/schema.py),
[`avaloka/data_agent/mission.py`](../avaloka/data_agent/mission.py), and
[`benchmarks/external/dab/README.md`](../benchmarks/external/dab/README.md).

| Complex capability | Status | Boundary |
| --- | --- | --- |
| PDF | C | Embedded text plus bounded OCR fallback; optional PDF/OCR dependencies required. |
| JPEG/PNG | C | Metadata and optional OCR; no general visual question answering. |
| Video | C | Metadata and bounded uniform frames, optionally OCR; no audio/shot/VLM semantics. |
| LeRobot v3 | C | Directory, Parquet, and video-manifest inspection; no policy training. |
| RLDS | C | TFDS directory decoding and episode/step invariants; TensorFlow/TFDS required. |
| Recursive schema | E | Nested, nullable, union, and tensor inference plus drift reporting. |

See [`docs/complex-data-foundation.md`](complex-data-foundation.md).

## Train, register, and inference flow

```mermaid
flowchart LR
    I["Modeling intent + prepared data"]:::input
    R{"Training route"}:::decision
    V1["MTA v1: local PyTorch"]:::cond
    V2["MTA v2: local or Ray"]:::cond
    PRE["Generic numeric/categorical preprocessing"]:::exec
    TRAIN["Train + evaluate"]:::cond
    MLF["MLflow tracking / registry"]:::cond
    ONNX["ONNX export / validation"]:::cond
    ART["Model artifact"]:::cond
    SERVE{"Serving mode"}:::decision
    RS["Ray Serve"]:::cond
    GCP["GCP managed service / gateway"]:::cond
    LI["Local/temporary inference"]:::cond
    CP["Conformal Prediction agent"]:::cond

    I --> R --> V1 & V2
    V1 --> PRE --> TRAIN
    V2 --> PRE
    TRAIN --> MLF --> ART
    TRAIN --> ONNX --> ART
    ART --> SERVE --> RS & GCP & LI
    TRAIN -. future calibration/coverage .-> CP

    classDef input fill:#d0ebff,stroke:#1971c2
    classDef exec fill:#d8f3dc,stroke:#2d6a4f
    classDef cond fill:#fff3bf,stroke:#b08900
    classDef decision fill:#e7f5ff,stroke:#1c7ed6
    classDef prop fill:#f8d7da,stroke:#a61e4d,stroke-dasharray: 7 4
```

Evidence: [`app/agents/model_training_agent.py`](../app/agents/model_training_agent.py),
[`app/agents/mta/`](../app/agents/mta/), [`app/agents/mta_v2/`](../app/agents/mta_v2/), and
[`app/serve/inference.py`](../app/serve/inference.py). A serving scaffold is not a live model; check
runtime readiness/model-loaded state.

## Data-transfer flow

```mermaid
flowchart TD
    ASK["Natural-language transfer request"]:::input
    RESOLVE["Resolve aliases, tables, formats, credentials"]:::exec
    SPEC["Typed transfer state"]:::exec
    PSEUDO["Daft ETL pseudocode"]:::cond
    CODE["Daft code generation"]:::cond
    SCHEMA["Schema deduction/comparison"]:::exec
    VALID["Code, join, type, destination validation"]:::exec
    REPAIR{"Repair?"}:::decision
    RUN["Execute transfer"]:::cond
    OUT["Rows, duration, result/redacted error"]:::exec

    ASK --> RESOLVE --> SPEC --> PSEUDO --> CODE --> SCHEMA --> VALID --> REPAIR
    REPAIR -->|bounded feedback| CODE
    REPAIR -->|accept| RUN --> OUT

    classDef input fill:#d0ebff,stroke:#1971c2
    classDef exec fill:#d8f3dc,stroke:#2d6a4f
    classDef cond fill:#fff3bf,stroke:#b08900
    classDef decision fill:#e7f5ff,stroke:#1c7ed6
```

Evidence: [`app/agents/data_transfer_agent/data_transfer_agent.py`](../app/agents/data_transfer_agent/data_transfer_agent.py)
and [`app/agents/data_transfer_agent/dta_state.py`](../app/agents/data_transfer_agent/dta_state.py).
Database/object-store connectors and MCP access remain environment-dependent.

## Infrastructure and scheduling flow

```mermaid
flowchart TD
    P["Execution profile: size, fidelity, preference"]:::exec
    R{"Environment router"}:::exec
    L["Local process"]:::exec
    RC["Existing Ray cluster"]:::cond
    KJ["RayJob / Kubernetes Job"]:::cond
    PROV["Provision or connect"]:::cond
    KIND["kind"]:::cond
    GKE["GKE"]:::cond
    EKS["EKS"]:::cond
    KR["KubeRay + RayCluster/RayService"]:::cond
    X["Bounded execution and artifacts"]:::exec
    S["Celery + RedBeat"]:::cond
    OPS["Status / list / result / cancel"]:::cond

    P --> R --> L & RC & KJ
    KJ --> PROV --> KIND & GKE & EKS
    KIND --> KR
    GKE --> KR
    EKS --> KR
    L --> X
    RC --> X
    KR --> X
    P -->|scheduled| S --> OPS

    classDef exec fill:#d8f3dc,stroke:#2d6a4f
    classDef cond fill:#fff3bf,stroke:#b08900
```

Evidence: [`app/execution/router.py`](../app/execution/router.py),
[`app/agents/execution_agent.py`](../app/agents/execution_agent.py),
[`app/infra/`](../app/infra/), and [`app/agents/scheduler.py`](../app/agents/scheduler.py).
These code paths do not imply a healthy cluster, worker, or valid credentials.

## Memory, persistence, and visualization

```mermaid
flowchart LR
    CTX["Thread/session context"]:::cond
    REDIS["Redis sessions/cache"]:::cond
    MEM["Memory orchestrator/preferences"]:::cond
    GRAPH["Agent graph"]:::exec
    RESULT["Execution result"]:::exec
    RENDER["Result renderer"]:::exec
    VIZ["Visualization artifact"]:::cond
    BLOB["GCS/S3 + signed URLs"]:::cond
    GIT["Git job registry"]:::cond
    MLF["MLflow history"]:::cond

    CTX --> REDIS --> MEM --> GRAPH --> RESULT --> RENDER --> VIZ --> BLOB
    RESULT --> BLOB
    GRAPH --> GIT
    RESULT --> MLF

    classDef exec fill:#d8f3dc,stroke:#2d6a4f
    classDef cond fill:#fff3bf,stroke:#b08900
```

Evidence: [`app/services/session_service.py`](../app/services/session_service.py),
[`app/services/memory_plane.py`](../app/services/memory_plane.py),
[`app/services/persistence_service.py`](../app/services/persistence_service.py),
[`app/agents/result_renderer.py`](../app/agents/result_renderer.py), and
[`app/agents/visualization_agent.py`](../app/agents/visualization_agent.py).

## Executable agents versus team labels

| Capability | Status | Current truth |
| --- | --- | --- |
| Front-door agent | E | Classifies, discovers, answers directly, or delegates. |
| Planner | E/C | Structured routing and fallbacks; rich reasoning requires an LLM. |
| Sampling / profiling | E/C | Local behavior exists; Daft/Ray and LLM semantics are optional. No confidence-convergence loop. |
| Coder / validator | E/C | Deterministic guards exist; generation and logical review may require an LLM. |
| Execution | E/C | Local is executable; Ray/Kubernetes/SSH/stores depend on environment. |
| Data Transfer Agent | C | Daft generation, validation, execution; needs connectors/runtime. |
| MTA v1/v2 | C | Training components exist; packages, Ray, MLflow, or cloud may be required. |
| Visualization / scheduler / infrastructure | C | Executable integrations with external dependencies. |
| Firefly/swarm roster labels | N or E by module | `avaloka/fireflies/` has callable classes; `app/agents/avaloka_agent/swarm.py` narrates a team plan. A label alone is not a graph node. |
| Feature Transformer | E | Reviewed feature recipes, structured parsing, nested projection, verified joins, and audited transforms. |
| Ava-Statistician | E | Typed deterministic descriptive/inferential operators with diagnostics and reproducibility hashes. |
| Ava-Experimenter | E | Conditional mission specialist with causal refusal, SRM, effect intervals, power, and sample-size primitives. |
| Conformal Prediction | E/C | Split-conformal engines are executable; mission activation requires calibration inputs. |
| Cartographer / executable Skeptic | E | Deterministic source mapping/routing and bounded evidence/output review; live schema introspection remains future work. |

See [`docs/dab-agent-improvement-design.md`](dab-agent-improvement-design.md) for the proposed design.

## Benchmark flow and status
