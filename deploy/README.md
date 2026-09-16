# Avaloka Deployment

This directory hosts two complementary deployment paths:

1. **Local Docker Desktop + Helm** — the existing app-only deployment via
   `deploy/avaloka/run.sh` (Streamlit frontend + backend + Redis). See
   [Part A](#part-a--avaloka-deployment-local--docker-desktop--helm).
2. **Kubernetes + Ray/KubeRay** — the newer provider abstraction that can
   connect to an existing Ray cluster or provision a fresh local `kind` one,
   install KubeRay, deploy avaloka, and stand up Ray Serve for
   inference-as-a-service. Provisioning managed cloud clusters is commercial.
   See [Part B](#part-b--deploying-avaloka-on-kubernetes-rraykuberay).

---

# Part A — Avaloka Deployment (Local – Docker Desktop + Helm)

## 1. Prerequisites

- Docker Desktop (with Kubernetes enabled)
- Helm v3+
- Valid GCP service account key JSON file
- Access to `/etc/hosts` file for hostname mapping

---

## 2. Setup Environment

Update `/deploy/avaloka/charts/avaloka-backend/docker/docker.env` and update specific variables

```aiexclude
GROQ_API_KEY_PLANNING_AGENT=
GROQ_API_KEY_CODING_AGENT=
:
:
GOOGLE_CLOUD_PROJECT=
GCS_BUCKET_NAME=
GCS_BUCKET=
GCS_PREFIX=
```
The rest of the default variables should work.

Also, update the `GCP_KEY_PATH` in `/deploy/avaloka/run.sh`. Make sure to use the correct location for windows operating systems:

```aiexclude
# Update to use appropriate location for windows
GCP_KEY_PATH="Users/$USER/.config/gcloud/application_default_credentials.json"
```

---

## 3. Add Local Hostnames

Edit `/etc/hosts` and add:
```
127.0.0.1       avaloka-langgraph.local
127.0.0.1       avaloka-api.local
127.0.0.1       avaloka avaloka.local
```

---

## 4. Build Dependencies & Install Chart

```bash
./deploy/avaloka/run.sh  
```

Verify deployment:
```bash
kubectl get pods -n avaloka
```

You should see:
```
avaloka-frontend
avaloka-backend
avaloka-redis-master
```

---

## 5. Access the Application

| Service       | URL                                                                      |
|---------------|--------------------------------------------------------------------------|
| Frontend      | [http://localhost](http://localhost)                            |
| Backend API   | [http://avaloka-api.local/health](http://avaloka-api.local/health)    |
| LangGraph | [http://avaloka-langgraph.local/docs](http://avaloka-langgraph.local/docs) |

---

## 6. Uninstall

To remove all components:
```bash
./deploy/avaloka/stop.sh  
```

---

# Part B — Deploying Avaloka on Kubernetes (Ray/KubeRay)

This path contains everything needed to run avaloka as a deployable Kubernetes
implementation. It can **connect to an existing Ray-on-Kubernetes cluster** or
**provision a fresh cluster** (local `kind`, or cloud GKE/EKS), install **KubeRay**,
deploy the avaloka app, and stand up **Ray Serve** for inference-as-a-service.

## Layout

```
deploy/
  Makefile                 # up / connect / status / down
  clusters/kind-cluster.yaml
  docker/Dockerfile.avaloka   # Streamlit + LangGraph control plane
  docker/Dockerfile.ray       # Ray runtime + avaloka compute deps + app code
  helm/avaloka/               # Helm chart for the avaloka app
  helm/ray/raycluster.yaml    # KubeRay RayCluster (distributed compute)
  helm/ray/rayservice.yaml    # KubeRay RayService (Ray Serve inference)
```

The orchestration logic lives in `app/infra/`:
`cluster_bootstrap.py` (driver) → `install_k8s.py` (preflight + KubeRay) →
`cloud_provisioner.py` / `providers/` (cluster lifecycle) → `ray_manager.py` (Ray) →
`deploy_stack.py` (images + Helm).

## Prerequisites

- `docker`, `kubectl`, `helm` (all paths)
- Local: `kind` (`brew install kind`)
- GCP: `gcloud` authenticated, `GCP_PROJECT_ID` set
- AWS: `aws` authenticated, `AWS_REGION` / `AWS_EKS_CLUSTER_ROLE_ARN` / `AWS_SUBNET_IDS` / `AWS_SECURITY_GROUP_IDS` set
- `GROQ_API_KEY_PLANNING_AGENT` and `GROQ_API_KEY_CODING_AGENT` exported (injected into the avaloka Secret)

## Quick start (local)

```bash
export GROQ_API_KEY_PLANNING_AGENT=...   # required by the agents
export GROQ_API_KEY_CODING_AGENT=...
export OPENROUTER_API_KEY=...            # optional: planner backup on Groq 429/404/5xx
cd deploy
make up PROVIDER=local
```

This builds the images, creates a kind cluster, installs the KubeRay operator,
applies the RayCluster, deploys avaloka, and deploys the Ray Serve app. When it
finishes:

- Avaloka UI: <http://localhost:8501>
- Ray dashboard: <http://localhost:8265>
- Inference: `kubectl port-forward svc/avaloka-inference-serve-svc 8000:8000`

Tear down with `make down PROVIDER=local`.

## Connect to an existing Ray cluster

```bash
cd deploy
make connect RAY_ADDRESS=ray://<head-host>:10001
```

Skips cluster/operator creation, validates Ray connectivity, and deploys avaloka
configured to use that Ray address.

## Cloud clusters, load balancing and scale

**Open source targets a laptop or a small Kubernetes cluster you run
yourself.** That is what the quick start above sets up and what the test suite
exercises.

Avaloka **provisioning** a managed cloud cluster is a Professional and
Enterprise capability: `make up PROVIDER=gcp|aws|azure` and
`get_provider("gcp"|"aws"|"azure")` refuse in an open-source build rather than
creating infrastructure that bills your cloud account.

What open source *does* support is deploying into a cluster **you** provisioned,
wherever it runs. `connect` attaches to any cluster your `kubeconfig` reaches,
and the chart, Ray/KubeRay, distributed execution and the scheduler all work
against it.

The cloud overlays under `deploy/helm/avaloka/values/` — `values-gke.yaml`,
`values-eks.yaml`, `values-aks.yaml` — remain in the tree as **worked examples**
of fronting the release with a provider L7 load balancer through the Gateway
API. They ship with placeholder hostnames (`test.example.com`,
`app.example.com`); substitute your own. Each lists its prerequisites at the top
of the file: the controller to install, the GatewayClass to create, and where
the certificate comes from.

Production concerns these examples deliberately do **not** cover — autoscaling
node pools, multi-instance release topologies, WAF and security policy, DNS
cutover, cost controls, multi-tenant isolation, and an SLA — are what the
commercial editions provide. See [avaloka.ai](https://avaloka.ai), or email
**[support@avaloka.ai](mailto:support@avaloka.ai)**.

> You run the open-source edition at your own risk. Support is community and
> best-effort.

## Object storage and MLflow

The chart deploys **MinIO** and an internal **MLflow tracking server** by default.
MLflow stores run/model metadata in PostgreSQL and proxies model artifacts to
`s3://avaloka/mlflow-artifacts` in MinIO. API and Ray clients use the internal
`http://avaloka-mlflow:5000` endpoint rather than a raw PostgreSQL URI.

Cloud overlays may continue to use GCS/S3/Azure for application datasets; that
backend wins for dataset storage while MinIO remains MLflow's artifact store.
Set `mlflow.backendStoreUri` to managed PostgreSQL for production. When
`existingSecret` is used, that Secret must contain `MLFLOW_BACKEND_STORE_URI`.

### Why object storage rather than a shared volume

Ray schedules workers across nodes, and every worker has to read the dataset the
API wrote. The obvious answer — one ReadWriteMany PVC mounted everywhere — does
not work here: neither kind nor Docker Desktop ships an RWX storage class. Their
default provisioner is node-local hostPath, so a second worker either fails to
mount the volume or mounts a different, empty one. Object storage sidesteps the
problem entirely: every worker reaches the same `s3://` URI over the network, no
matter which node it lands on.

### Why MinIO rather than Ceph

| | MinIO | Ceph (via Rook) |
|---|---|---|
| Ray / PyArrow / fsspec | Native — they already speak S3 | Needs RGW, Ceph's S3 gateway |
| Laptop (kind, Docker Desktop) | One container | Needs 3+ nodes, raw block devices, GBs for mon/mgr/osd |
| Scale-out | Add replicas + disks, same API | Scales further, far more to operate |
| Cost to adopt | `minio.enabled: true` | A second distributed system to run |

Ray, PyArrow and fsspec all talk S3 already, and Ceph's own S3 story *is* RGW —
so choosing Ceph means paying Ceph's operational cost for MinIO's interface. The
deciding factor is the laptop: Rook-Ceph cannot run on kind, which would leave
local and datacenter installs with different storage shapes and different bugs.
MinIO is the same manifest in both, from one container on a laptop to distributed
erasure coding across a datacenter rack.

Ceph is the better answer if you need RWX POSIX semantics or block storage for
other workloads. This chart intentionally requires MinIO while its managed
MLflow deployment is enabled.

### How it is wired

Enabling MinIO makes the chart set `STORAGE_BACKEND=s3` and publish the endpoint
and credentials to everything that reads `s3://` — the API, LangGraph, and the
Ray head and workers (via `envFrom` on `deploy/helm/ray/raycluster.yaml`):

| Variable | Value |
|----------|-------|
| `S3_ENDPOINT_URL` / `AWS_ENDPOINT_URL` | `http://avaloka-minio:9000` |
| `S3_BUCKET` | `avaloka` (created by a post-install hook) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | MinIO's root credentials |

Both endpoint names are set on purpose: the app reads `S3_ENDPOINT_URL`, botocore
understands `AWS_ENDPOINT_URL` natively, and PyArrow reads neither — it needs
`endpoint_override`, which `app/agents/mta_v2/loader/url.py` now passes. Without
that a Ray worker resolves the bucket against real AWS and fails with
`NoSuchBucket`, which looks like a missing dataset rather than a misconfiguration.

**A configured cloud backend always wins.** Enabling MinIO on a GKE/EKS/AKS
install deploys it but does not redirect the app's storage — `config.storageBackend`
of `gcs`/`s3`/`azure` takes precedence.

### Using it

The console is on <http://localhost:30092> for kind (`avaloka` / the password in
`minio.auth.secretKey`). To reach the S3 API from your laptop:

```bash
kubectl port-forward svc/avaloka-minio 9000:9000
```

```bash
mc alias set local http://localhost:9000 avaloka avaloka-minio-dev && mc ls local/avaloka
```

The default credentials are dev values for a cluster-internal Service. Override
`minio.auth.accessKey` / `minio.auth.secretKey` for anything beyond a laptop —
the on-prem overlay ships `CHANGE_ME` to force the decision.

MinIO's volume is the one piece of in-cluster state that is deliberately durable
(a PVC, not `emptyDir`): losing it means losing uploaded datasets and trained
models, not just a cache.

## Optional data stack

Postgres/Kafka/Milvus/Neo4j/OpenSearch are opt-in (disabled by default). A minimal
postgres example is wired up:

```bash
make up PROVIDER=local DATA_STACK=postgres
```

Values live in `app/infra/manifests/helm-values/<component>-values.yaml`.

## Inference as a service

`app/serve/inference.py` defines the Ray Serve app referenced by `helm/ray/rayservice.yaml`.
It is a **scaffold** today (echoes features, no model). A follow-up replaces
`InferenceService.predict` with a model produced by the training agents.
