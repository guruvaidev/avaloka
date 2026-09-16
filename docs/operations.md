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

To keep experiment history on a shared Postgres instead of SQLite, set
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
| `AVALOKA_ASSET_BUCKET` | Destination for persisted artefacts. A local MinIO bucket is the default local-first choice. |
| `AVALOKA_SIGNED_URL_EXPIRY_MINUTES` | Lifetime of the signed URLs the UI uses to fetch artefacts |
| `GITHUB_SYSTEM_TOKEN`, `GITHUB_JOB_REGISTRY_REPO` | Optional: push job definitions to a registry repo |

Retrieval is through the API rather than the store directly:

```bash
curl localhost:9000/assets/<session_id>     # signed URLs for this session's artefacts
```

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
cloud API gateway, and handles client API keys and gateway URLs for you. An open-source build refuses to provision managed cloud
infrastructure; see
[README → What the open-source edition is sized for](../README.md#what-the-open-source-edition-is-sized-for).

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

- Confirm `AVALOKA_ASSET_BUCKET` is set and reachable
- Check `persist_errors` in session state; failures are reported there rather
  than raised
- Remember writes are backgrounded — an artefact may appear a moment after the
  answer does

**The inference service will not start.**

- `INFERENCE_SERVICE_DOCKER_IMAGE` must be pullable from the cluster, not just
  from your laptop. On `kind`, side-load it: `kind load docker-image <image>`
- `InferenceServiceManager.stop_inference_service(run_id)` cleans up a stale
  deployment

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
