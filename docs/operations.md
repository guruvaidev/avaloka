# Operating Avaloka

What you touch once Avaloka is running and you want it to keep running:
experiment tracking, where artefacts go, serving a trained model, and the
failures that are worth recognising on sight.

**Scope.** The open-source edition targets **a laptop or a small Kubernetes
cluster you run yourself**, and everything here is written for that. It is what
we test. Larger deployments — provisioned and autoscaled clusters, managed
tracking databases, scheduled cloud workloads, multi-tenant isolation — are
what the commercial editions are for: see [avaloka.ai](https://avaloka.ai) or
email **[support@avaloka.ai](mailto:support@avaloka.ai)**.

You run the open-source edition at your own risk. Support is community and
best-effort.

**Related:** [Install](INSTALL.md) · [Kubernetes deployment](deployment.md) ·
[Architecture](architecture.md) · [Testing](testing.md) · [API](api.md) ·
[Editions](EDITIONS.md)

---

## Experiment tracking (MLflow)

A local MLflow server is enough for everything the open-source edition does,
and it is the default.

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 --port 5000
```

Point Avaloka at it:

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

The UI is then at `http://localhost:5000` — runs, metrics, parameters and
registered models.

MLflow is pinned at `3.14.0` (`requirements.txt`). To keep experiment history
on a shared Postgres instead of SQLite, set
`MLFLOW_BACKEND_STORE_URI` to a Postgres URL and
`MLFLOW_DEFAULT_ARTIFACT_ROOT` to a path or bucket both the server and the
workers can reach. That is as far as this guide goes; running tracking against
a managed cloud database with the availability and backup story that implies is
a commercial deployment concern.

---

## Where artefacts go

After every run Avaloka persists generated code, execution outputs and
visualization configs in the background.

| Setting | What it does |
| --- | --- |
| `GCS_BUCKET` (or `GCS_BUCKET_NAME`) | The bucket `app/core/settings.py:10` actually reads. |
| `AVALOKA_SIGNED_URL_EXPIRY_MINUTES` | Lifetime of the signed URLs the UI uses to fetch artefacts. Default **15** (`app/services/persistence_service.py:43`). |
| `GITHUB_SYSTEM_TOKEN`, `GITHUB_JOB_REGISTRY_REPO` | Optional: push job definitions to a registry repo. The repo defaults to `avaloka/avaloka-job-registry` (`persistence_service.py:40`). |
| `AVALOKA_JOB_REGISTRY_WRITE_JSON` | Set to `false` to stop writing the job-definition JSON (`persistence_service.py:41`). |

> `AVALOKA_ASSET_BUCKET` appears in `.env.example:129` and earlier revisions of
> this page, but **no code reads it** — a repo-wide search finds it only in
> `.env.example` and documentation. Setting it has no effect. The destination is
> resolved through `IBlobStore` / `_store_from_connection_uri`, with
> `GCS_BUCKET` as the setting for the GCS path.

> A local MinIO bucket used to be the natural local-first destination. The
> chart's MinIO images cannot be pulled right now
> ([deployment.md §2](deployment.md#2-quick-start-local-kind)), so on a local
> cluster there is no working in-cluster bucket until you mirror those images or
> point at a store you run.

Retrieval is through the API rather than the store directly:

```bash
curl localhost:9000/api/assets/<session_id>        # signed URLs for this session
curl localhost:9000/api/assets/<session_id>/code   # generated code
curl localhost:9000/api/assets/<session_id>/output # execution outputs
curl localhost:9000/api/assets/<session_id>/job    # job definition
```

Note the `/api` prefix (`app/api/server.py:8783-8924`); `/assets/<id>` without
it is not a route.

Persistence is **non-blocking** — it runs as a background task and never fails
the conversation turn. It retries with exponential backoff (3 attempts,
0s → 2s → 4s); a final failure surfaces in session state as `persist_errors`
rather than being swallowed.

---

## Serving a trained model

MTA v1 trains locally with PyTorch, exports ONNX, and records to MLflow. MTA v2
trains distributed on Ray and can stand up a containerized inference service.

**Both are open-source capabilities.** Train, register and serve models on
your laptop or your own Ray cluster, with full MLflow tracking and the model
registry. Set `INFERENCE_SERVICE_DOCKER_IMAGE` to the image to serve and deploy
it to the cluster your `kubeconfig` points at.

What is commercial is *scheduling that work onto a cloud cluster Avaloka
provisioned* — the managed path where Avaloka creates the cluster, stands up a
cloud API gateway, and handles client API keys and gateway URLs for you. The
open-source distribution refuses to provision managed cloud infrastructure, via
the `cloud_provisioning` gate in `get_provider()`; see
[deployment.md → A cluster you already run](deployment.md#4-a-cluster-you-already-run)
for where that gate lives and the one case where it is absent, and
[EDITIONS.md](EDITIONS.md) for the capability split.

---

## Troubleshooting

**MLflow database migration errors.** A schema left behind by a different
MLflow version. Drop the version table and let it re-migrate:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('mlflow.db')
conn.execute('DROP TABLE IF EXISTS alembic_version')
conn.commit()
"
```

**MLflow artifacts are not saving.**

- `MLFLOW_DEFAULT_ARTIFACT_ROOT` must be set in the server's environment, not
  only the client's
- The path or bucket must be writable by the MLflow server process
- The MLflow UI shows the resolved artifact URI per run — check it matches what
  you set

**Asset persistence failures.**

- Confirm `GCS_BUCKET` (or the connection URI for your store) is set and
  reachable — not `AVALOKA_ASSET_BUCKET`, which nothing reads
- Check `persist_errors` in session state; failures are reported there rather
  than raised
- Remember writes are backgrounded — an artefact may appear a moment after the
  answer does

**The inference service will not start.**

- `INFERENCE_SERVICE_DOCKER_IMAGE` must be pullable from the cluster, not just
  from your laptop. On `kind`, side-load it: `kind load docker-image <image>`
- `InferenceServiceManager.stop_inference_service(mlflow_run_id)` cleans up a
  stale deployment (`app/agents/mta_v2/inference_service_manager.py:89`); the
  API exposes it too (`app/api/server.py:8599`)

**Leftover files after a training run.**

```bash
find . -name "*.pth" -o -name "*.onnx" -o -name "temp*"
```

Tests clean up after themselves; interrupted runs may not.

---

## When you have outgrown this

Autoscaling node pools, provisioned clusters, scheduled cloud workloads, cost
controls, multi-tenant isolation and an SLA are not open-source capabilities,
and this guide will not get you there. That is a deliberate boundary, not a
gap we forgot: [avaloka.ai](https://avaloka.ai), or
**[support@avaloka.ai](mailto:support@avaloka.ai)**.
