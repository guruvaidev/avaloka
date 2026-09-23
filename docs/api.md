# Avaloka HTTP API

The FastAPI backend is the programmatic front door to the Avaloka agent team.
It handles dataset uploads, conversational threads (where the actual
plan → code → execute → train flow happens), task/asset retrieval, database
chat, and model management.

- **App object:** `app.api.server:app`
- **Default port:** `9000` (the container `CMD` is
  `uvicorn app.api.server:app --host 0.0.0.0 --port 9000`)
- **Interactive docs:** `GET /docs` (Swagger UI), `GET /openapi.json`
- **Auth:** Bearer JWT (HS256). The `sub` claim is the user id; the signing
  secret is `JWT_SECRET`.

```bash
uvicorn app.api.server:app --host 0.0.0.0 --port 9000 --reload
```

## Authentication

Every non-public route expects an `Authorization: Bearer <jwt>` header. The token
is HS256-signed with `JWT_SECRET` and carries `{"sub": "<user-id>"}`. Requests
without a valid token get `401`; accessing another user's resource returns `403`
(threads) or `404` (datasets).

```python
import jwt, time
token = jwt.encode({"sub": "user-1", "iat": int(time.time())}, JWT_SECRET, algorithm="HS256")
headers = {"Authorization": f"Bearer {token}"}
```

## Core flow

The typical lifecycle is **upload → open a thread → send messages → collect
assets**.

### 1. Upload a dataset

```text
POST /api/upload            multipart: file=(name, bytes, "text/csv")
```

Returns an `UploadResponse` with `dataset_id`, `thread_id`, `session_id`,
`schema`, and sample rows.

```bash
curl -XPOST http://localhost:9000/api/upload \
  -H "Authorization: Bearer $JWT" \
  -F "file=@app/sample_data/sales_data.csv;type=text/csv"
```

### 2. Talk to the agent team

```text
POST /threads                          create a thread
POST /threads/{thread_id}/messages     send a message → ChatResponse
GET  /threads/{thread_id}/messages     history
GET  /threads/{thread_id}/code         the generated code
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
GET    /tasks                          list tasks
GET    /tasks/{id}/status              poll a Celery task
GET    /tasks/{id}/result/{index}      a result artifact
GET    /api/assets/{session_id}        signed URLs for persisted code/output
GET    /datasets                       list datasets
GET    /datasets/{id}/preview          preview rows
```

## Model training & inference

```text
GET    /api/models                                        list models
GET    /api/models/{run_id}                               model details
POST   /api/models/{run_id}/inference                     run inference
POST   /api/models/{run_id}/configure-inference-service   deploy an inference service
POST   /api/models/{run_id}/stop-inference-service        tear it down
```

`configure-inference-service` → `inference` → `stop-inference-service` is the
managed-serving lifecycle. The managed path can deploy a
containerized service behind a GCP API Gateway; the direction is to serve via
the in-cluster Ray Serve platform so it is cloud-neutral.

## Database chat (MCP-backed)

```text
POST /api/database/connect         { "customer_id": "...", "api_key": "..." }  → session_id
POST /api/v1/database/query        DatabaseChatRequest → DatabaseChatResponse
GET  /buckets/list                 list cloud-storage objects (s3|gcs|azure)
```

Credentials are exchanged once via `/api/database/connect` and thereafter
referenced by `session_id`, so raw credentials do not travel in every query.

## Health & metadata

```text
GET /health        { status, redis_connected, graph_ready, ... }
GET /version
GET /debug/whoami
```

`/health` returns `200` even when Redis is unavailable (it reports
`redis_connected: false`), so use the response body — not just the status code —
as your readiness signal.

## Related surfaces

- **React web UI** — a React.js app (built with Lovable) in the `ui/` directory,
  backed by a local Supabase database. It talks to this HTTP API, which the
  `deploy/helm/avaloka` chart deploys as its primary workload (port `9000`). An
  optional in-cluster web UI can be enabled with `--set ui.enabled=true`. See
  [deployment.md](deployment.md).
- **LangGraph dev server** — `langgraph.json` maps the `avaloka` graph; served by
  `langgraph dev` on `http://127.0.0.1:2024`.
- **MCP servers** — `app/mcp_server/customer_dbs.py` (customer DB registry) and
  `app/mcp_server/multi_tenant_mcp_server.py`.

For the full route list, run the server and open `GET /docs`.
