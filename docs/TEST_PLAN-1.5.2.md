# Avaloka 1.5.2 — Master End-to-End Test Plan

| | |
|---|---|
| **Applies to** | `ui-k8s-deploy-1.5.2` (merge of `feature/k8s-deploy-1.5.2` + PR #237) → `develop-1.5` |
| **Scope** | Full system E2E: platform, API, agents, data plane, inference, CLI/MCP, frontend, security, performance, lifecycle |
| **Executable form** | `tests/e2e/` (L5 API pack) and `tests/contract/` (static gate) — see [tests/e2e/README.md](../tests/e2e/README.md) |
| **Date** | 2026-08-03 |
| **Status** | Every case verified against the source; 20 defects confirmed and pinned |
| **Owner** | QA / System Test |

---

## 1. Objective

Verify that the complete Avaloka system — deployed on Kubernetes as shipped in
this branch — works end to end for a real user: sign in, upload data, converse
with the agent pipeline, get validated generated code executed, train models, run
inference through the in-cluster Ray Serve service, and see results in the React
portal.

**Headline item under test:** Ray Serve inference-as-a-service
(`INFERENCE_BACKEND=rayserve` default, RayService CR, MTA v2 managed inference),
plus multi-mode execution and the containerised React UI over the in-cluster
backend.

**A caveat that shapes the whole E7 suite:** the headline feature cannot serve a
per-run model today. `configure-inference-service` never patches `MODEL_URI` into
the RayService — it only annotates the CR — and the URI it would annotate resolves
from `ModelConfig.model_artifacts`, a field that does not exist. See §4, D-07.

## 2. System under test

| Layer | Components | Key sources |
|---|---|---|
| Frontend | React/Vite SPA in `ui/`, containerised behind nginx, runtime config via `window._env_`; Supabase auth (cloud or in-cluster) | `ui/src/services/backendApi.ts` (32 methods), `ui/src/services/assistantApi.ts`, `ui/deploy/nginx.conf.template`, `ui/docker-entrypoint.d/` |
| API | FastAPI — **36 routes / 32 OpenAPI paths, all in `server.py`**; HS256 JWT via `SUPABASE_JWT_SECRET` | `app/api/server.py`. `app/api/cloud_connections.py` declares **zero routes** — it is the credential helper layer (encryption, redaction, tenant scoping) |
| Agent pipeline | LangGraph ETL graph — see §5 E5.03 for the real traversal | `app/api/workflow.py`, `app/graph/etl_state.py`, `app/agents/*` |
| Mission core | One planner behind 3 doors: REST `/api/missions/plan`, CLI (incl. remote `--endpoint`), MCP — all converge on `app/interfaces/service.py:plan_mission` | `app/missions/{schema,compiler}.py`, `app/interfaces/*` |
| Data plane | Redis (L1 sessions + chat history), Chroma (L2 logic signatures), Postgres (L3 artifacts; **falls back to SQLite** when `POSTGRES_URL` is unset), Milvus (L4 episodic), GCS/local BlobStore | `app/services/db/*`, `app/core/storage.py` |
| Inference | MTA v2 `InferenceServiceManager` → Ray Serve (default) or legacy GCP gateway; ONNX with honest fallback | `app/agents/mta_v2/inference_service_manager.py`, `app/serve/inference.py` |
| File handling | **Upload gate and connector layer accept different sets** — see E3.01 | `app/api/server.py:299`, `file_handler/handler.py:36-52` |
| Platform | Helm chart `deploy/helm/avaloka/`, kind/GKE/EKS/AKS, KubeRay, RayCluster + RayService, webui + Supabase (ui branch) | `deploy/` |

Two legacy charts (`deploy/avaloka/`, `deploy/avaloka-router/`) still sit beside
the real one, and `deploy/README.md` documents the legacy path **first** (E14.06).

## 3. The two Day-0 questions, answered

### 3.1 Nothing executes background tasks in-cluster — D-01, P0

Celery is used (`app/core/celery_app.py`; scheduler, milvus_recorder,
execution_agent), but:

- no chart template runs a `celery worker` or `beat` — the chart deploys only
  api, langgraph, redis, chroma, postgres, ui (+ webui, supabase on the ui branch);
- there is no eager mode anywhere (zero hits for `task_always_eager`);
- the chart never sets `CELERY_REDIS_URL`, which `celery_app.py:23` reads for the
  broker.

So `.delay()` enqueues fail and are swallowed, and RedBeat schedule entries are
never fired. `/api/datasets/*/background-task-status`, Milvus episodic writes and
scheduled entire-dataset executions are **vacuous in every k8s deployment**. A
user can create a schedule, receive a 200, and have nothing ever run.

This blocks E5.08 (Celery leg), E6.04, E10.10 — they are not "runnable but
failing", they are testing something that cannot happen.

### 3.2 There is no MLflow backend — D-02, P0

`deploy/` contains zero MLflow references and sets no `MLFLOW_*` env.
`mlflow_manager.py:69-75` falls back `MLFLOW_TRACKING_URI` →
`MLFLOW_BACKEND_STORE_URI` → `"mlruns"`: a pod-local file store on a pod with no
volume. `ray_trainer.py:205-207` forwards the unset variables into Ray jobs, so a
Ray-trained model lands in the Ray pod's local store and is **invisible** to the
API's `GET /api/models`. Models also vanish on API-pod restart.

**Decision required before Wave 2:** add an MLflow backend (shared store URI +
artifact root) to the chart, or re-scope E7.01/E7.06 to the LocalTrainer leg and
document the limitation. Until then E7.01 as written is unrunnable.

### 3.3 A third blocker on the same footing — D-03, P0

The Helm chart deploys **no MCP server and no onboarding service**, and does not
set `MCP_SERVER_URL` on the API pod (default `http://localhost:8080`,
`app/core/settings.py:31`). Port conventions disagree across the repo (8080 /
8081 / 8082 / 8010). So `/api/database/connect` will always fail in-cluster: the
whole of E8 silently depends on services run by hand outside the cluster. Decide
and document how E8's dependencies are brought up, or E8 is untestable.

## 4. Defect register

Confirmed by reading the code. Each is pinned by a test that asserts the
*intended* behaviour under `xfail(strict=True)`, so the day it is fixed the test
fails and tells you to delete the marker. Run `pytest tests/e2e tests/contract -rxX`
for the live list.

| ID | Pri | Defect | Evidence | Pinned by |
|---|---|---|---|---|
| D-04 | **P0** | `POST /threads` never calls `_resolve_user_id` — anonymous callers create LangGraph threads and bind them to a caller-chosen session | `server.py:2028-2049` | `test_e2_auth.py::test_e2_05a_*` |
| D-05 | **P0** | `.env` is git-tracked on both branches carrying a live Supabase **service_role** JWT and the **JWT signing secret** — anyone with repo read access can mint valid tokens for every route | `git ls-files .env` | `test_c1_repo_hygiene.py::test_e11_05_env_file_is_not_tracked` |
| D-06 | **P0** | GCP service-account key `avalokagcpbucketserviceaccount.json` is neither committed nor **ignored** — no `.gitignore` or `.dockerignore` rule matches it | repo root | `test_c1_repo_hygiene.py::test_e11_05_credential_files_are_ignored` |
| D-07 | **P0** | Per-run model URI never resolves: `_resolve_model_uri` reads `cfg.model_artifacts`, which `ModelConfig` does not define — configure annotates nothing and Serve keeps the stub | `inference_service_manager.py:53-65`, `model.py:33-60` | `test_e7_inference.py::test_e7_02_*` |
| D-08 | **P0** | Unknown `INFERENCE_BACKEND` values silently route to the **legacy GCP gateway** (which reads a SA JSON from disk) instead of erroring | `inference_service_manager.py:36-49` | `test_e7_inference.py::test_e7_04_*` |
| D-09 | **P0** | `_to_vector` orders dict features **alphabetically** while training uses `feature_cols` order — silently wrong predictions, no error | `serve/inference.py:43-50`, `local_trainer.py:374` | `test_e7_inference.py::test_e7_05_*` |
| D-10 | **P0** | Unauthenticated `GET /customers/{id}/credentials` returns a tenant's **plaintext DB password**; `POST /admin/customers` lets anyone overwrite any tenant | `mcp_server/customer_dbs.py:684-732`, `multi_tenant_mcp_server.py:628-652` | E8/E11 manual pass (§6) |
| D-11 | **P0** | `app/mcp_server/customers.json` is tracked and contains an API key + passworded connection string; it is auto-loaded at MCP startup | tracked file | `test_c1_repo_hygiene.py::test_e11_05_mcp_customers_registry_not_tracked` |
| D-12 | **P0** | No NetworkPolicy anywhere — Ray workers running LLM-generated code reach redis (no auth), postgres, chroma and the internet unrestricted | `deploy/` | `test_c2_chart_invariants.py` (E11.10) |
| D-13 | **P0** | CI cannot fail: the only test step is `pytest … \|\| true`; no PR pipeline, no marker filter, no nightly/cloud tier, no image gate | `bitbucket-pipelines.yml:32` | `test_c3_release_coherence.py` (E14.07) |
| D-14 | **P0** | 17 unmarked test files reach real infrastructure (GKE, docker build, boto3, live services), so the hermetic filter does not deselect them — §5 entry criterion is unenforceable | `tests/` | `test_c3_release_coherence.py::test_hermetic_marker_audit` |
| D-15 | **P0** | nginx allowlist omits `/tasks/*/result/*`, `/datasets/*/preview`, `/buckets/list` — the SPA gets `index.html` instead of JSON; task results, preview and bucket browsing are broken in the shipped container | `ui/deploy/nginx.conf.template` | `test_c4_ui_wiring.py::test_e9_14a_*` |
| D-16 | **P0** | MCP env injection is dead: `10-env-config.sh` never writes `MCP_API_BASE`/`MCP_TOOL_BASE`, so the SPA always falls back to hardcoded ngrok hosts and the Helm values are inert | `ui/docker-entrypoint.d/10-env-config.sh` | `test_c4_ui_wiring.py::test_e9_15a_*` |
| D-17 | **P0** | The merged 37-test vitest suite is unrunnable — a later commit removed the `test` script and all vitest/@testing-library/jsdom devDependencies | `ui/package.json` | `test_c4_ui_wiring.py::test_e9_17_*` |
| D-18 | **P0** | Chart embeds 9 Supabase migrations; the UI needs 67. The migrate Job also runs `psql` with `ON_ERROR_STOP=0` and `\|\| echo … ignored`, so it reports success when every migration fails | `deploy/helm/.../supabase.yaml` | `test_c4_ui_wiring.py::test_e9_18*` |
| D-19 | **P0** | Chart defaults ship the **public demo** Supabase JWT secret and keys; `secret.yaml` adopts them as the API's `SUPABASE_JWT_SECRET` when `supabase.enabled=true` — one `--set` makes every token forgeable | `values.yaml`, `secret.yaml` (ui branch) | `test_c4_ui_wiring.py::test_e9_19a_*` |
| D-20 | **P0** | 0-byte upload has no empty-file check; it reaches sampling and surfaces as a 500 via the catch-all, not a clean 4xx | `server.py:1417-1438` | `test_e3_ingestion.py::test_e3_03a_*` |
| D-21 | **P1** | Wildcard CORS **with credentials** (`allow_origins=["*"]`, `allow_credentials=True`) — Starlette echoes any origin, so every origin is credential-allowed | `config.py:45-51`, `server.py:708-715` | `test_e2_hygiene.py::test_e2_11_*` |
| D-22 | **P1** | Full Celery tracebacks are returned to clients on the success path (`/tasks/{id}/result/{index}` and the SSE stream) | `server.py:4587,4591,2151,2153` | `test_e2_hygiene.py::test_e2_10b_*` |
| D-23 | **P1** | `.tsv`/`.orc` pass the upload gate but have no connector — upload 200s, then sampling raises `TypeError` | `server.py:299` vs `handler.py:36-52` | `test_e3_ingestion.py::test_e3_01c_*` |

Also pinned, lower severity: version incoherence across five surfaces (E14.01),
CHANGELOG roadmap drift (E14.02), legacy chart still documented first (E14.06),
`:latest` image tags everywhere (E14.10), shared over-privileged ServiceAccount
across api/langgraph/ui pods (E11.12), `APP_VERSION` never injected by the chart.

## 5. Verified behaviour worth stating explicitly

Cases where the implementation differs from what the naming, or the obvious
reading, suggests. Each was confirmed by reading the source. Testing any of
these against the assumed behaviour would produce a false result.

| Case | Actual behaviour |
|---|---|
| E2.03 / E14.01 | `/version` returns **`"1.2"`** — the hardcoded default at `server.py:296`; `APP_VERSION` is set nowhere in `deploy/` or `.env`. The version story is **five-way**: branch 1.5.2 / VERSION 0.2.0 / CHANGELOG [0.2] / Chart 0.1.0+0.2.0 / API 1.2 |
| E2.05 | `cloud_connections.py` declares **zero routes**. Sweep universe is the 36 `server.py` routes — and it must be *generated*, not hand-listed |
| E2.07 | With no secret and the flag unset the server **refuses to start** (`RuntimeError` in the lifespan, `server.py:275-287`) — crash-loop on k8s. Request-rejection is the flag-*set* behaviour |
| E2.06 | A token from a foreign issuer is rejected **only** if signed with a different secret. `verify_aud=False` and **no issuer check** — a token from any project sharing the secret is accepted |
| E2.10 | Error bodies are clean, but full Celery tracebacks are returned deliberately on the **success** path (D-22) |
| E2.11 | CORS ships `allow_origins=["*"]` **with** `allow_credentials=True` (D-21), so this is a defect-fix case, not a verification |
| E2.13 | Only `/threads` has paging (limit clamped 1..200). `/tasks` and `/datasets` have **none** |
| E3.01 | Delta and Iceberg are **415** at the API. The upload gate is `{csv,tsv,json,xml,parquet,avro,orc,xls,xlsx}`; Delta/Iceberg live only in the connector layer. Split into an API matrix and a connector matrix |
| E3.02 | Multi-file upload is **all-or-nothing**: any failure rolls back every file and returns a single 500, with no per-file reporting |
| E3.09 | There is one route, `GET /buckets/list`; browsing is via the `prefix` param, not a separate contents route. And `connection_id` is optional for **GCS**, so a user can enumerate buckets readable by the platform's own service account |
| E5.03 | The real traversal: `provision_infra` branches from `plan_etl` and **loops back to it**; `schedule_task` is **terminal**; execution sits **between** validation layers 2 and 3 (coder → syntax → static → execute → logical); the graph also has `train_models` and `profile_data` nodes |
| E5.05 | The "k8s" execution leg posts to a **GCP-provisioned** service — not runnable on kind. The Ray leg **silently falls back to local** when `connection_id` is missing, so the case can pass without touching Ray. Assert `execution_result.mode` |
| E6.05 | The top-3 hint relevance bug is **fixed and merged** (`b678bff`), with `tests/test_memory_top3_relevance.py` (17 tests). Exit criterion already satisfied |
| E6.07 | The memory circuit breaker is **merged app code** (`memory_plane.py:560-616`), covered by `tests/test_memory_semantics.py`. It is a per-call 3 s timeout, not a stateful open/half-open breaker |
| E6.09 | After `414c79a`, chat history and preferences **write through to Redis** and survive an API-pod restart. The loss surface is a **Redis**-pod restart. Split into two cases |
| E7.02 | `configure-inference-service` only annotates (and today annotates nothing — D-07); stop blanks `serveConfigV2` best-effort, **never checks the return code**, and logs success unconditionally |
| E7.07 / E12.07 | Autoscaling assertions are **vacuous**: the inference RayService is a fixed 1-replica Serve deployment with no autoscaler; `maxReplicas=4` belongs to the separate compute RayCluster |
| E9 (all) | The wiring is browser → NodePort 30090 → nginx → API service, same-origin. No port-forward in the user path |
| E9.13 | The UI **never consumes SSE** — it polls. Re-point or drop |
| E10.01 | Uploads, threads and inference do return 503, but `GET /datasets` returns **200 with `[]`** — a user cannot tell "no datasets" from "storage is down" |
| E10.07 | On GROQ failure the chat path converts planner exceptions to a **200 apology**; no FAILED task. Worse, a missing coder key makes the coder emit and **execute stub code** — a silent fake success |
| E12.06 | The image-size budget is not automated anywhere: CI builds no image and has no gate (D-13) |

**Status of existing automation** — coverage that already exists and should not
be rebuilt:
E4.03 and E4.07 → AUTO-partial (`test_t4_cli.py::test_t4d_7_cli_equals_mcp`,
`tests/test_mcp_server_integration.py`); E6.05 → AUTO; E6.07 → AUTO-partial;
E7.03 → AUTO-partial for the **stub leg only** (`test_t3_kuberay_execution.py`
asserts the key exists, not that a real model loads — the real-ONNX leg is new);
E7.04 → NEW-auto entirely (no test covers backend dispatch); E3.08 → cite
`tests/test_sampler_stats_regression.py`, since `tests/sampler/*.py` is never
collected (filenames match no `python_files` pattern); The suite holds **137** test files.

## 6. Suites and how each is executed

| Suite | Layer | How it runs now |
|---|---|---|
| **E1** Platform gate | L0–L4 | `tests/k8s/` T1 (clusterless) · T2/T3 (kind) · T5 (cloud) · **T6 (UI + Supabase, ui branch)**. Re-anchor from the absent K8S plan onto these tiers |
| **E2** API contract & auth | L5 | **Automated** — `tests/e2e/test_e2_contract.py`, `test_e2_auth.py`, `test_e2_hygiene.py` |
| **E3** Ingestion | L5 | **Automated** — `tests/e2e/test_e3_ingestion.py` |
| **E4** Mission core & parity | L5 | `tests/k8s/test_t4_cli.py` covers CLI↔REST↔MCP; new work is the live-REST leg + the full enum matrix, including `FullDataEngine{local,ray}` |
| **E5** Agent pipeline | L6 | Golden missions GM-1..GM-6 against the deployed stack. Note T4's "13/13" is the whole T4 tier and asserts **code presence only** — no task polling, no assets |
| **E6** Memory & context | L5/L6 | E6.05/E6.07 already automated; new: durable history across pod restart, preference recall, the adaptive planner knobs (§8) |
| **E7** Training & inference | L6 + kuberay | **Partly automated** — `tests/e2e/test_e7_inference.py` pins dispatch, `_to_vector`, serve honesty and the CR shape clusterlessly. The real-ONNX serve leg needs a cluster |
| **E8** SQL / MCP | L5 | **Blocked on D-03.** Also reword: `/api/v1/database/query` executes `content` as **raw SQL** — there is no NL→SQL in that route |
| **E9** Frontend | L7 | **Partly automated** — `tests/contract/test_c4_ui_wiring.py` for wiring; T6 for deployment; J1–J7 manual until the vitest harness is restored (D-17) and Playwright lands |
| **E10** Resilience | L5 | Manual/scripted against E-DEGRADED; correct the contracts per §5 |
| **E11** Security | cross | **Partly automated** — `tests/contract/test_c1_repo_hygiene.py` (hygiene + egress inventory), `test_c2_chart_invariants.py` (NetworkPolicy, RBAC, demo secrets) |
| **E12** Performance | L5/L6 | Baselines; replace the nonexistent `profile_nyc_taxi.py` driver with a committed one |
| **E13** Lifecycle | L5 | Manual; re-anchor from the absent RUNBOOK onto `deploy/README.md` Part B + `deploy/Makefile` |
| **E14** Release coherence | — | **Automated** — `tests/contract/test_c3_release_coherence.py` |

### The auth sweep is generated, not listed

The API has no route-level `Depends`; each route calls `_resolve_user_id` by
hand. A route added without that call fails open — which is not hypothetical
(D-04). So `route_inventory()` derives the sweep from the running app (or
`/openapi.json` when deployed): 32 routes × 5 anonymous probe kinds, plus the
cross-tenant "never 200" invariant over every `{id}` route. A route added
tomorrow is swept tomorrow with no plan edit.

Known fail-open routes live in one `KNOWN_FAIL_OPEN` set, each covered by a
dedicated strict-xfail, so the sweep stays green and can gate merges while the
defect is open.

### The dual-credential model

`/tasks*` routes need **both** a bearer token and an `X-Avaloka-Session` header.
A valid token with no session returns **400 "No session found"**, not 401 — the
sweep must not misread those, and the adversarial leg (user A's token + user B's
session id) is its own case.

## 6A. Test data — real datasets and the operations run against them

Synthetic fixtures are narrower and cleaner than anything a customer sends, so the
suite also carries a real-data leg backed by the project's own GCS bucket.

### 6A.1 Datasets

Source: `gs://avaloka-test-user-filestore/user-upload/inputs/`. Only **top-level
source inputs** are used. Both prefixes also contain `code-registry/` and
`execution-outputs/` objects — artifacts written by earlier pipeline runs — which
are deliberately excluded: reading them back would assert that the system
reproduces its own output, which proves nothing.

| Dataset | Object | Size | Shape | Why it is useful |
|---|---|---|---|---|
| **ieee-fraud** | `train_transaction.csv` | 683 MB | 394 cols, `isFraud` label | Widest real schema available; labelled target; extreme null density |
| | `train_identity.csv` | 26.5 MB | 41 cols | Joinable companion on `TransactionID`; fast enough for full-read assertions |
| | `test_transaction.csv` / `test_identity.csv` | 613 / 25.8 MB | — | Unlabelled holdout; reserved for inference-path work |
| **walmart** (M5) | `sell_prices.csv` | 203 MB | long/narrow | Opposite shape to ieee-fraud: many rows, few columns |
| | `sales_train_evaluation.csv` | 122 MB | wide time series | Date-column-per-day layout, stresses schema inference |
| | `calendar.csv` | 0.1 MB | small dimension | Join key and date parsing |

**One dataset per run.** `AVALOKA_TEST_DATASET` selects it (`ieee-fraud` default,
`walmart` alternative); the suite never reads both in one execution. Cost and
runtime stay bounded, and a failure is attributable to one dataset.

Excluded as not useful: `sample_submission.csv` (Kaggle scoring template, no real
records), and all `execution-outputs/` and `code-registry/` objects.

### 6A.2 How the data is used

Large objects are **never fully downloaded**. Tests issue HTTP range reads for the
leading bytes, which is enough to recover the header and thousands of complete
rows. Schema inference over the 683 MB object therefore reads under 0.02 % of it.
A truncated final row is discarded so no malformed record enters an assertion.
Slices are written to `tmp_path` and deleted with the test; nothing is cached in
the repo and no data is written back to the bucket.

### 6A.3 Operations exercised, and what each proves

| # | Operation | Assertion | Plan case |
|---|---|---|---|
| 1 | Schema inference from a range read | 41 columns in `train_identity`, 394 in `train_transaction`, key columns present | E3.08 |
| 2 | Wide-schema inference without full download | Schema recovered from < 0.02 % of a 683 MB object | E3.08 |
| 3 | **Bounded sampling** | `sample_data_from_source` emits ≤ `DEFAULT_SAMPLE_MAX_ROWS` (1000) from an input with more rows | **E11.09** |
| 4 | DDL generation | Generated DDL names every real column; none dropped | E3.08 |
| 5 | Groupby aggregation vs pandas ground truth | Group counts match exactly; read path deterministic | E3.08 / E5.07 |
| 6 | **Analytical query** (below) | Domain invariants hold | E3.08 |
| 7 | Source-input guard | No `execution-outputs/` or `code-registry/` path is read | E11.09 |

Operation 3 is the one that matters most for security review. The sample is what
gets embedded in LLM prompts, so the cap is a **data-egress control**, not a
performance tweak. It cannot be tested with synthetic fixtures — they are smaller
than the cap, so the bound never engages. Only real data exercises it.

### 6A.4 The analytical query

The query is the one a fraud analyst would actually ask:

```sql
-- Fraud rate per product category
SELECT ProductCD, COUNT(*) AS txns, SUM(isFraud) AS frauds,
       ROUND(100.0 * SUM(isFraud) / COUNT(*), 2) AS fraud_rate_pct
FROM train_transaction GROUP BY ProductCD ORDER BY txns DESC;

-- Amount profile by fraud class
SELECT isFraud, COUNT(*), AVG(TransactionAmt), MEDIAN(TransactionAmt)
FROM train_transaction GROUP BY isFraud;
```

Observed on a 7,532-row slice:

| ProductCD | txns | frauds | fraud rate |
|---|---|---|---|
| W | 5,891 | 92 | 1.56 % |
| C | 671 | 61 | **9.09 %** |
| H | 581 | 4 | 0.69 % |
| R | 230 | 16 | 6.96 % |
| S | 159 | 9 | 5.66 % |

Mean amount: **137.54** legitimate vs **152.27** fraudulent. Highest-null columns:
`D7` 97.7 %, `D13` 97.0 %, `dist2` 96.2 %.

Assertions pin **invariants, not memorised numbers**, because the slice size varies
with the range read: `ProductCD ⊆ {W,C,H,R,S}`; every rate in [0,1]; groupby drops
no rows; overall fraud rate in (0, 20 %) — a wider bound would not catch a
misparsed label column; mean amounts positive; at least one column above 90 % null.
That last one is the characteristic synthetic fixtures cannot supply at all, so
null-heavy handling is otherwise entirely untested.

### 6A.5 Golden missions — the questions a user actually asks

§6A.4 verifies the *data*. This verifies the *product loop*: a natural-language
question goes in, the agent pipeline generates code, and the answer is checked
against ground truth. Implemented in `tests/e2e/test_e5_golden_missions.py`.

| ID | The user types | Must reference | Ground truth |
|---|---|---|---|
| **GM-1** | "What is the fraud rate for each product category?" | `ProductCD`, `isFraud` | fraud rate per category |
| **GM-2** | "Compare the average transaction amount between fraudulent and legitimate transactions" | `TransactionAmt`, `isFraud` | mean amount per class |
| **GM-3** | "Which card network has the highest fraud rate?" | `card4`, `isFraud` | argmax of rate |
| **GM-4** | "Which columns have more than 90% missing values?" | `isnull`/`isna` | null-rate column set |

The set deliberately spans four different operations — aggregation, comparison,
ranking, data quality — and a test asserts that span, so four near-identical
groupbys cannot pass for coverage.

**What is asserted, and what is not.** Nothing matches generated text. LLM output
is non-deterministic, so a test that pins wording fails on a model change while
still passing if the answer is wrong. The invariants are: the generated program
imports a dataframe library, and it references the columns the question is about.
GM-1 and GM-2 additionally carry a pandas ground truth, so a plausible-but-wrong
aggregate cannot pass. Agents may vary the code; they may not vary the answer.

Example — what the coder produced for GM-1, having located `ProductCD` and
`isFraud` unaided among 394 columns:

```python
import pandas as pd

def main(df):
    grouped_df = df.groupby('ProductCD')
    total_transactions = grouped_df.size().reset_index(name='total_transactions')
    fraudulent_transactions = grouped_df['isFraud'].sum().reset_index(name='fraudulent_transactions')
    merged_df = pd.merge(total_transactions, fraudulent_transactions, on='ProductCD')
    merged_df['fraud_rate'] = merged_df['fraudulent_transactions'] / merged_df['total_transactions']
    return merged_df[['ProductCD', 'fraud_rate']].set_index('ProductCD')
```

Marked `integration` (real Groq spend), deselected from the hermetic gate.
**Result: 7 passed in 52 s**; skips cleanly with a reason when keys are absent.

**Known limitation.** `test_e5_07` recomputes ground truth rather than executing
the agent's code and diffing the result. Executing generated code requires the
validation gate (E5.04) and security guard (E11.02) in the loop. Until that lands,
these cases prove the agent *writes relevant code*, not that it *returns the right
number* — the remaining gap in E5.07.

### 6A.6 Running it

```bash
export GOOGLE_APPLICATION_CREDENTIALS=<service-account.json>
pytest tests/e2e/test_e3_real_dataset_gcs.py -m cloud -q                    # ieee-fraud
AVALOKA_TEST_DATASET=walmart pytest tests/e2e/test_e3_real_dataset_gcs.py -m cloud -q
```

Marked `cloud`, so the hermetic PR gate deselects all 7 — verified: `tests/e2e`
collects 392 of 399 under the hermetic filter. Result: **ieee-fraud 7 passed
(1 m 17 s)**; **walmart 6 passed, 1 skipped** (the fraud-specific query correctly
skips on a schema without an `isFraud` column).

---

## 7. Missing artifacts — disposition required

| Artifact | Referenced by | Action |
|---|---|---|
| `K8S_TEST_PLAN.md` (S1–S11, F1–F9) | E1.01–E1.04, §5 entry criteria, many "platform half" notes | Commit it, or restate E1 against `tests/k8s/` T1–T6. **E1.04 is unreviewable until then** |
| `RUNBOOK-k8s-deploy-1.5.2.md` | E13.01, E13.03, §4, §5 | Commit it, or repoint at `deploy/README.md` Part B (note: `make` lives in `deploy/`, there is no root Makefile) |
| `DEMO_RUNBOOK.md` | E14.05 | Create or drop the case |
| `test_bomb_logic.py` | E3.04 | Never existed — E3.04 is NEW-auto from scratch |
| `profile_nyc_taxi.py` | §4, E12.02 | Never existed — commit a real large-file driver |
| `demo_top3_hints.py`, `demo_circuit_breaker.py` | E6.05, E6.07 | Never existed — cite the real suites instead |
| `demo_mcp_tarpit.py` | E8.05 | Never existed — re-anchor on the real limits (422/404/400, `max_rows`, `query_timeout`); there is no tarpit and no request-size or rate limit |
| `tests/test_timestamp_planner_bug.py` | — | **Uncommitted** in a working tree, in no branch. Likely pins the planner/scheduler timezone defect (§8). Commit it before it is lost |
| `ui/docker-entrypoint.d/*.sh` edits | — | **Uncommitted**; likely the D-16 fix. Commit it — `test_e9_15a` will confirm |

## 8. Cases requiring new coverage

The conversational-reasoning upgrade (`414c79a`, merged into both branches) has no
coverage anywhere in the suite:

- **E6.10 (P0)** durable chat history across an API-pod restart — seed a thread,
  delete the pod, assert history returns; variant: kill Redis *during* restart and
  assert the durable copy is not clobbered by the empty in-process view.
- **E6.11 (P1)** `store_user_preference` recall across sessions and its reserved
  top-3 slot under LLM-noise flooding.
- **E5.14 (P1)** the adaptive planner: default model changed to
  `openai/gpt-oss-120b` with per-request reasoning effort. Verify the rollback
  knob (`AVALOKA_PLANNER_MODEL=llama-3.3-70b-versatile` must suppress reasoning
  kwargs) and **baseline the E5.11 GROQ budget** against the current model.
- **E5.15 (P2)** `reasoning_trace` exposure — cap, isolation from other users'
  task results, and whether `REASONING_FORMAT=hidden` suppresses it end to end.
- **E2.05b (P0)** `GET /threads/{id}/messages` changed from open to
  authenticated+ownership. That is a silent breaking change for any client that
  fetched history without a token — fold into the sweep and the UI contract check.

Further gaps: `GET /threads/{id}/code` appears in no case;
`X-Storage-URI` lets a client choose the write destination; scheduling is
timezone-naive (`server.py:4712` mixes a naive `datetime.now()` into the SSE
countdown; absolute schedules are silently UTC with no user timezone captured) —
run one leg with a non-UTC `TZ`; there is **no observability** (no `/metrics`, no
tracing), so E12's regression gates have no in-cluster measurement source and
E13.05's "root cause in 10 minutes" rests on logs alone; there is no
delete-my-data cascade (sampled **rows** persist in cloud Supabase after a dataset
is deleted — a DPA problem, and an egress path the E11.09 inventory must name);
and every chart image is `:latest`, so the exit review certifies an unpinned
artifact.

## 9. Execution order

0. **Day 0 — gate.** Rotate the leaked Supabase credentials and purge `.env`
   (D-05); ignore the SA key (D-06); commit the uncommitted work in §7. Decide
   D-01 (Celery), D-02 (MLflow), D-03 (MCP services) and E14.01 (version story).
   Land the CI gate (D-13) and marker hygiene (D-14) — until then no "green"
   claim means anything.
1. **Wave 1 (Days 1–2).** `pytest tests/e2e tests/contract` in CI on every PR.
   E2/E3/E4 deployed legs on a fresh E-LOCAL stack.
2. **Wave 2 (Days 2–4).** E5 golden missions, E6 (incl. the §8 additions), E7
   (needs the D-02 decision), E8 (needs D-03).
3. **Wave 3 (Days 4–5).** E10 degradation, E11 security pass, E12 baselines.
4. **Wave 4 (Days 5–6).** E9 — restore the vitest harness (D-17), fix the nginx
   allowlist (D-15) and MCP injection (D-16), then the manual J1–J7 pass.
5. **Pre-release.** One cloud leg, soak, E14 sweep, exit review.

## 10. Exit criteria

- All P0 cases pass or carry a signed waiver with an issue link and a release note.
- ≥ 90 % of P1 pass; every P1 failure triaged with an owner.
- **No `xfail` in `tests/e2e` or `tests/contract` is still marked P0** — each is
  either fixed (marker deleted) or explicitly waived.
- No secret material in the repo, images, logs or API responses — D-05, D-06,
  D-11 closed and the leaked credentials **rotated**, not just deleted.
- CI runs the hermetic pack on every PR and can go red (D-13).
- Version story unified across all five surfaces (E14.01).
- Shipped images pinned by digest or an immutable tag.
- D-01, D-02, D-03 answered with evidence, tested, or formally re-scoped.
- Frontend J1–J5 manual pass green; vitest harness restored and running in CI.
- Performance baselines recorded and committed as the regression reference.

## 11. Deliverables

1. This plan (reviewed and signed off).
2. **Shipped:** `tests/e2e/` (L5 API pack, dual-mode) and
   `tests/contract/` (static gate) — 411 passing assertions, 49 defect pins,
   no cluster required. See [tests/e2e/README.md](../tests/e2e/README.md).
3. Extended T4 golden missions, degradation pack, deployed-mode security sweep.
4. Baseline metrics file committed for regression gating.
5. Manual-pass checklists (E9, E13.01) with recorded results.
6. Defect log — §4 is the seed; every row needs a tracker issue.
