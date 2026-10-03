# Avaloka Deployment

This directory hosts two complementary deployment paths:

1. **Local Docker Desktop + Helm** — the existing app-only deployment via
   `deploy/avaloka/run.sh` (Streamlit frontend + backend + Redis). See
   [Part A](#part-a--avaloka-deployment-local--docker-desktop--helm).
2. **Kubernetes + Ray/KubeRay** — the newer, multi-cloud provider abstraction that
   can connect to an existing Ray cluster or provision a fresh one (local `kind`,
   GKE, or EKS), install KubeRay, deploy avaloka, and stand up Ray Serve for
   inference-as-a-service. See [Part B](#part-b--deploying-avaloka-on-kubernetes-raykuberay).

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

## Cloud (GKE / EKS)

```bash
make up PROVIDER=gcp     # or PROVIDER=aws
```

Reuses the same flow. Cloud image delivery to a registry is environment-specific —
push `avaloka:latest` / `avaloka-ray:latest` to your registry and set the chart
`image.repository` / the RayCluster image accordingly.

The cloud overlays put the release behind a managed **Application Load Balancer**
rather than exposing a raw Service IP — see the next section.

## Enterprise load balancing (Application Load Balancer)

Out of the box the chart exposes NodePorts, which is right for `kind` and wrong for
anything public. The cloud overlays instead front the release with the provider's
**L7 Application Load Balancer**, expressed once in the **Gateway API** so the same
two resources drive all three clouds:

| Cloud | GatewayClass                     | What it provisions                    | Certificates                                  |
|-------|----------------------------------|---------------------------------------|-----------------------------------------------|
| GCP   | `gke-l7-global-external-managed` | Global external Application LB        | Certificate Manager map (`networking.gke.io/certmap`) |
| AWS   | your ALB class (`alb`)           | Application Load Balancer             | ACM ARNs via `LoadBalancerConfiguration`      |
| Azure | `azure-alb-external`             | Application Gateway for Containers    | `kubernetes.io/tls` Secret                    |

What that buys over `service.type: LoadBalancer`: one static anycast IP instead of
one public IP per Service, managed TLS for named hostnames, per-backend health
checks and timeouts, and a place to attach a WAF.

### Two GKE instances

`test.avaloka.ai` and `app.avaloka.ai` are **separate releases with separate public
IPs**, so test can be redeployed, broken and rolled back without touching
production:

| | test | app (production) |
|---|---|---|
| Values | `values-gke.yaml` | `values-gke.yaml` **+** `values-gke-app.yaml` |
| Namespace | `avaloka` | `avaloka-app` |
| Static IP | `avaloka-lb-ip` | `avaloka-app-lb-ip` |
| Cert map | `avaloka-cert-map` | `avaloka-app-cert-map` |
| Replicas | 1 | 2 |

```bash
helm upgrade --install avaloka-app deploy/helm/avaloka --namespace avaloka-app --create-namespace -f deploy/helm/avaloka/values/values-gke.yaml -f deploy/helm/avaloka/values/values-gke-app.yaml
```

**The separate namespace is required, not stylistic.** `avaloka.fullname` derives
every resource name from the *chart* name, not the release name, so two releases
in one namespace would both try to own a Service called `avaloka` and the second
install fails. Set `nameOverride` instead if you genuinely need them co-located.

### HTTPS only

Both GKE instances set `gateway.httpEnabled: false`, so no port-80 listener is
created and `http://` is refused rather than redirected. Google-managed
certificates are unaffected — Certificate Manager validates and renews over 443.

The trade-off is real: a user typing a bare hostname gets a connection error
instead of being bounced to HTTPS. Set `httpEnabled: true` (the chart default) to
serve `:80` purely as a 301.

The backing Services drop to `ClusterIP` in every cloud overlay. That is deliberate:
a `LoadBalancer` Service alongside the Gateway would publish a second address that
reaches pods directly, bypassing the load balancer's TLS, Cloud Armor policy and
access logs — while looking perfectly healthy.

### Routing

Each instance is an independent load balancer on its own IP; the routing inside
them is identical.

```
  test.avaloka.ai              app.avaloka.ai
  (ns avaloka)                 (ns avaloka-app)
        │                            │
        ▼                            ▼
  ┌───────────────┐            ┌───────────────┐
  │  Global ext.  │            │  Global ext.  │   :443 only — no :80 listener
  │  App LB       │            │  App LB       │   TLS via Certificate Manager
  │ avaloka-lb-ip │            │ …-app-lb-ip   │
  └───────┬───────┘            └───────┬───────┘
          └──────────┬─────────────────┘
                     │  (same routing in both)
     /api /threads /docs │ /health /version /openapi.json
              ┌──────────┴──────────┐
              ▼                     ▼
       avaloka :9000         avaloka-webui :80   ← everything else (/)
      (API, 1h timeout)      (React SPA on nginx)
```

API paths are matched at the load balancer and sent straight to the API Service.
That skips the webui nginx hop, and — the reason it matters — gives the API its own
backend, so agent requests can carry an hour-long timeout without loosening the one
guarding the static SPA. GCP's default backend timeout is 30s, which returns 504 to
the user while the pod is still working. Any path not matched still falls through to
the SPA, whose nginx proxies the same routes, so a missed prefix degrades rather
than breaks.

### GCP — first deploy

`deploy/gcp-lb-prereqs.sh` creates the resources that must outlive any single
release: a reserved global IP and a Certificate Manager map with a Google-managed
certificate per hostname. It is idempotent, so re-run it when adding a host.

```bash
./deploy/gcp-lb-prereqs.sh --project MY_PROJECT --hosts test.avaloka.ai,app.avaloka.ai
```

Add `--dry-run` to print the `gcloud` calls without executing them. The equivalents
by hand:

```bash
gcloud compute addresses create avaloka-lb-ip --global --ip-version=IPV4 --network-tier=PREMIUM
gcloud certificate-manager maps create avaloka-cert-map
gcloud certificate-manager certificates create avaloka-cert-app-avaloka-ai --domains=app.avaloka.ai
gcloud certificate-manager maps entries create avaloka-entry-app-avaloka-ai \
  --map=avaloka-cert-map --certificates=avaloka-cert-app-avaloka-ai --hostname=app.avaloka.ai
```

Then point DNS at the reserved IP — **one A record per hostname**:

```
test.avaloka.ai   A   <reserved IP>
app.avaloka.ai    A   <reserved IP>
```

Google-managed certificates stay `PENDING` until those records resolve: DNS *is* the
ownership check, so nothing serves HTTPS before propagation. Install once DNS is in:

```bash
helm upgrade --install avaloka deploy/helm/avaloka \
  -f deploy/helm/avaloka/values/values-gke.yaml \
  --set image.repository=us-central1-docker.pkg.dev/MY_PROJECT/avaloka/avaloka-api
```

Watch it come up — the first provision takes a few minutes:

```bash
kubectl get gateway avaloka-gateway -n <namespace> -w
kubectl describe gateway avaloka-gateway -n <namespace>         # Programmed / Accepted conditions
gcloud certificate-manager certificates list      # managed.state should reach ACTIVE
```

### Migrating avaloka.ai off Lovable

`avaloka.ai` currently resolves to Lovable's IP and is untouched by any of this.
`app.avaloka.ai` and `test.avaloka.ai` are new records pointing at the load
balancer, so the two run side by side with no downtime. When the k8s deployment is
proven on `app.avaloka.ai`, cut the apex over by adding `avaloka.ai` (and `www`) to
`gateway.hosts`, re-running the prereq script to mint their certificates, and only
then repointing the apex A record. Rollback is a DNS change.

### AWS / Azure

Both overlays are wired the same way and list their prerequisites at the top of the
file — the controller to install, the GatewayClass to create, and where the
certificate comes from:

- `deploy/helm/avaloka/values/values-eks.yaml` — AWS Load Balancer Controller
  ≥ v2.14 with Gateway API support. ACM ARNs go in `gateway.aws.certificateArns`;
  the controller does **not** read `listeners[].tls.certificateRefs`, which is why
  the overlay uses `tls.mode: external`.
- `deploy/helm/avaloka/values/values-aks.yaml` — ALB Controller for Application
  Gateway for Containers. The certificate must be a real in-cluster Secret; Key
  Vault via the Secrets Store CSI driver is not supported for this listener.

### Tuning

All under `gateway:` in `deploy/helm/avaloka/values.yaml`:

| Value | Purpose |
|-------|---------|
| `gateway.hosts` | Hostnames served. Each needs an A/CNAME record and a certificate. |
| `gateway.routeApiDirect` | Route API paths to the API Service instead of through nginx. |
| `gateway.extraRoutes` | Extra path → Service rules, e.g. exposing the MCP API. |
| `gateway.gcp.api.timeoutSec` | API backend timeout (default 3600s, vs GCP's 30s). |
| `gateway.gcp.securityPolicy` | Cloud Armor policy — WAF, rate limiting, IP allowlists. |
| `gateway.gcp.sslPolicy` | Minimum TLS version / cipher profile. |
| `gateway.routeTimeout` | Vendor-neutral per-route timeout, used on AWS/Azure. |

`gateway.gcp.securityPolicy` and `sslPolicy` ship empty so a first install succeeds
without them. Set both before treating the deployment as production-ready.

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
