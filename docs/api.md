# Avaloka HTTP API

The FastAPI backend is the programmatic front door to the Avaloka agent team.
It handles dataset uploads, conversational threads (where the actual
plan → code → execute → train flow happens), task/asset retrieval, database
chat, and model management.

- **App object:** `app.api.server:app` (`app/api/server.py:1129`, FastAPI title
  `Avaloka UI API`)
- **Default port:** `9000` (the container `CMD` is
  `uvicorn app.api.server:app --host 0.0.0.0 --port 9000` —
  `deploy/docker/Dockerfile.api:104`)
- **Interactive docs:** `GET /docs` (Swagger UI), `GET /openapi.json` — the app
  does not override `docs_url`, so both are served
- **Auth:** Bearer JWT. The `sub` claim is the user id. Two verification paths,
  chosen by the token's own `alg` header (`app/api/server.py:1066`).

```bash
uvicorn app.api.server:app --host 0.0.0.0 --port 9000 --reload
```

## Authentication

Every non-public route expects an `Authorization: Bearer <jwt>` header.
`_resolve_user_id` (`app/api/server.py:1054`) picks the verification path from
the token's `alg` header:

| `alg` | Verified against | Configured by |
| --- | --- | --- |
| `ES256` / `RS256` | the Supabase JWKS at `{SUPABASE_URL}/auth/v1/.well-known/jwks.json` | `SUPABASE_URL` (`app/api/server.py:324`) |
| `HS256` (anything else) | the shared secret, symmetric | `SUPABASE_JWT_SECRET` (`app/api/server.py:322`) |

> **The secret's env var is `SUPABASE_JWT_SECRET`, not `JWT_SECRET`.** The
> module-level Python name is `JWT_SECRET`, which is where the wrong name in
> earlier revisions of this page came from. `app/api/server.py:322` reads
> `os.getenv("SUPABASE_JWT_SECRET", "")` and labels it "used only for legacy
> HS256 tokens".

An empty `SUPABASE_JWT_SECRET` **fails closed twice over**, because an empty
HS256 key would verify any token an attacker signs with the empty string:

- The server **refuses to start**. `_validate_auth_config()` raises at startup
  (`app/api/server.py:336`). `AVALOKA_ALLOW_INSECURE_AUTH=1` is the explicit
  opt-out (`app/api/server.py:333`) — it starts the server with authentication
  *disabled*, meaning every request is rejected as unauthenticated, not allowed.
- Even then, `_resolve_user_id` returns `None` without attempting verification
  (`app/api/server.py:1059`).

An `ES256`/`RS256` token with `SUPABASE_URL` unset is also rejected — there is
no JWKS client to resolve the key with (`app/api/server.py:1068`).

The user id is `payload["sub"]`, falling back to `payload["email"]` when `sub`
is absent (`app/api/server.py:1093`). Prefer `sub` and make it a UUID: it is
stored as the owner of sessions and datasets, and the Postgres persistence layer
rejects a non-UUID with `invalid input syntax for type uuid`. Tokens issued by
Supabase Auth already satisfy this. Requests without a valid token get `401`;
accessing another user's resource returns `403` (threads) or `404` (datasets).

```python
import jwt, os, time, uuid
secret = os.environ["SUPABASE_JWT_SECRET"]
user_id = str(uuid.uuid4())   # or the Supabase auth user's id
token = jwt.encode({"sub": user_id, "iat": int(time.time())}, secret, algorithm="HS256")
headers = {"Authorization": f"Bearer {token}"}
```

## Core flow

The typical lifecycle is **upload → open a thread → send messages → collect
assets**.

### 1. Upload a dataset

```text
POST /api/upload            multipart: file=(name, bytes, "text/csv")
```

Returns an `UploadResponse` for one file, or a `MultiUploadResponse` when
several are sent in the same request (the response model is a union —
`app/api/server.py:1399`). `UploadResponse` carries `dataset_id`, `thread_id`,
`session_id`, `schema`, and sample rows.

```bash
curl -XPOST http://localhost:9000/api/upload \
  -H "Authorization: Bearer $JWT" \
  -F "file=@app/sample_data/salaries.csv;type=text/csv"
```

**Limits** (`app/api/server.py:75-77`), each overridable by env var:

| Limit | Default | Env var |
| --- | --- | --- |
| Files per request | 10 | `AVALOKA_MAX_UPLOAD_FILES` |
| Bytes per file | 100 MB | `AVALOKA_MAX_UPLOAD_FILE_BYTES` |
| Bytes per request | 200 MB | `AVALOKA_MAX_UPLOAD_TOTAL_BYTES` |

For data that should not move, register it where it lives instead of uploading
it: `POST /api/register-existing-folder` (a directory,
`app/api/server.py:2647`) or `POST /api/register-existing-storage` (an
object-store prefix, `app/api/server.py:2057`). Both return an
`UploadResponse`.

### 2. Talk to the agent team

```text
POST   /threads                            create a thread
GET    /threads                            list your threads
DELETE /threads/{thread_id}                delete one (204)
POST   /threads/{thread_id}/messages       send a message → ChatResponse
GET    /threads/{thread_id}/messages       history
GET    /threads/{thread_id}/code           the generated code (coder_definition.code)
GET    /threads/{thread_id}/pending-turn   whether a turn is still running
GET    /threads/{thread_id}/planner-graph  the planned execution graph
```

Message body (`MessageCreateIn`):

```json
{
  "role": "user",
  "content": "Read the sales data, filter for sales in the 'USA', and save the result.",
  "metadata": { "dataset_id": "<dataset_id>" },
  "stream": false,
  "dataset_ids": null,
  "analysis_fidelity": null
}
```

The `ChatResponse` carries the full state of the turn — among many fields:
`messages`, `planner_definition`, `ready_to_code`, `coder_definition` (with the
generated `code`), `task_info`, `output_file_data`, `visualization_config`, and
the training block (`training_task`, `training_status`, `training_metrics`,
`model_artifacts`, `mlflow_run_id`, `ready_to_train`, `training_completed`).

### 3. Tasks & assets

```text
GET    /tasks                              list tasks
GET    /tasks/{id}/status                  poll a Celery task
GET    /tasks/{id}/info                    task metadata
GET    /tasks/{id}/runs                    run history for a scheduled task
GET    /tasks/{id}/result/{index}          a result artifact
DELETE /tasks/{id}                         cancel / remove a task

GET    /api/assets/{session_id}            signed URLs for persisted code/output
GET    /api/assets/{session_id}/code       just the code asset
GET    /api/assets/{session_id}/output     just the output asset
GET    /api/assets/{session_id}/job        the job record

GET    /datasets                           list datasets
GET    /datasets/{id}                      one dataset
GET    /datasets/{id}/preview              preview rows (`limit` query param)
DELETE /datasets/{id}                      delete one (204)
GET    /api/datasets/{id}/background-task-status   profiling / sampling progress
```

### 4. Analysis records and insights

An analysis is addressable on its own, separately from the thread that produced
it:

```text
GET  /analysis/{analysis_id}/code              the code that produced it
GET  /analysis/{analysis_id}/versions          version history
POST /analysis/{analysis_id}/restore           restore an earlier version
POST /analysis/{analysis_id}/refresh           re-run it
POST /analysis/{analysis_id}/save-and-execute  save an edit and run it
POST /analysis/{analysis_id}/feedback          → InsightFeedbackOut
POST /analysis/{analysis_id}/insights/explain  → InsightExplainOut
```

## Model training & inference

```text
GET    /api/models                                        list models
GET    /api/models/{run_id}                               model details
DELETE /api/models/{run_id}                               delete a model
POST   /api/models/{run_id}/inference                     run inference
POST   /api/models/{run_id}/configure-inference-service   deploy an inference service
POST   /api/models/{run_id}/stop-inference-service        tear it down
```

`configure-inference-service` → `inference` → `stop-inference-service` is the
managed-serving lifecycle. The managed path can deploy a containerized service
behind a GCP API Gateway; the direction is to serve via the in-cluster Ray Serve
platform so it is cloud-neutral.

The served endpoint **scales to zero when idle** — the default is two hours
without a request, `INFERENCE_IDLE_TIMEOUT_S`
(`app/agents/mta_v2/inference_autoscale.py:58`), with a `warm_replicas` floor of
1 brought back on the first request after a scale-down.

## Database chat (MCP-backed)

```text
POST /api/database/connect              { "customer_id": "...", "api_key": "..." }  → session_id
POST /api/v1/database/query             DatabaseChatRequest → DatabaseChatResponse
POST /api/database/tables-to-analysis   pull tables into a thread → MultiUploadResponse
GET  /buckets/list                      list cloud-storage objects (s3|gcs|azure)
```

Credentials are exchanged once via `/api/database/connect` and thereafter
referenced by `session_id`, so raw credentials do not travel in every query.

## Connections and integrations

```text
GET    /api/integrations                        provider-card state (never returns a token)
POST   /api/integrations/github/connect         store a GitHub connection
PATCH  /api/integrations/github                 update it
DELETE /api/integrations/github                 remove it (204)
POST   /api/mcp-connections/{id}/encrypt        encrypt stored credentials
POST   /api/mcp-connections/{id}/decrypt        decrypt them for use
```

`GET /api/integrations` returns only whether a token is stored plus the
non-secret config, never the token itself (`app/api/server.py:10637`).

## Mission planning

```text
POST /api/missions/plan     intent dict → planned_to_dict()
```

The same canonical `{mission, plan, estimate}` shape the planner CLI prints with
`--json` and the MCP server returns, so `python -m app.interfaces.cli.main plan
--endpoint …` is byte-identical to its local mode for the same intent
(`app/api/server.py:1220`). It requires auth like any other route, and a
malformed intent comes back as `400`, not `500`. See [cli.md](cli.md).

## Health & metadata

```text
GET /health        { status, graph_ready, langgraph_url, assistant_id,
                     upstream_reachable, upstream_timeout_s,
                     redis_connected, redis_mode }
GET /version       { "version": "<APP_VERSION>" }
GET /debug/whoami  { "user_id": "<sub or email, or null>" }
```

`/health` returns `200` even when Redis or the upstream LangGraph server is
unavailable — it reports `redis_connected: false` / `upstream_reachable: false`
instead (`app/api/server.py:1181`). **Use the response body, not just the status
code, as your readiness signal.**

`/health` and `/version` take no auth. `/debug/whoami` takes none either, but
returns `null` for an absent or invalid token — it is the cheapest way to check
that your JWT and `SUPABASE_JWT_SECRET` agree.

## Related surfaces

- **React web UI** — a React app in the `ui/` directory, backed by Supabase. It
  talks to this HTTP API, which the `deploy/helm/avaloka` chart deploys as its
  primary workload (port `9000`). See [deployment.md](deployment.md).
- **Streamlit UI** — a *different*, optional surface: `app/api/streamlit_app.py`,
  enabled with `--set ui.enabled=true`, which renders a second
  Deployment+Service on port `8501`
  (`deploy/helm/avaloka/values.yaml:493-505`). It is **off by default** and is
  not the React app above. Its `ui.image.repository` currently points at
  `ghcr.io/guruvaidev/avaloka-ray:main`, which is the Ray image name rather than
  a UI image — confirm the tag you intend is actually pullable before enabling
  it.
- **LangGraph dev server** — `langgraph.json` maps the `avaloka` graph to
  `app/api/langgraph_app.py:graph`; `langgraph dev` serves it on the LangGraph
  CLI's default `http://127.0.0.1:2024`.
- **MCP servers** — two unrelated sets. `app/mcp_server/customer_dbs.py` and
  `app/mcp_server/multi_tenant_mcp_server.py` are the customer-database
  registry; `app/interfaces/mcp/server.py` is the mission-planning tool server
  documented in [cli.md](cli.md).

`app/api/server.py` declares 53 routes at the time of writing. Rather than
trust a count in a document, read them from the source of truth:

```bash
grep -c '@app\.\(get\|post\|put\|patch\|delete\)' app/api/server.py
```

or run the server and open `GET /docs`.
