# Deploying Avaloka on Kubernetes

Avaloka can be stood up as a backend on **any Kubernetes cluster your
`kubeconfig` reaches** and run its distributed compute on **Ray/KubeRay**. One
bootstrap path works across all of them, thanks to the provider abstraction in
`app/infra/providers/`.

The open-source edition **provisions** a local `kind` cluster and **connects**
to anything else. Provisioning a managed cloud cluster (GKE, EKS, AKS) is a
commercial capability; provider modules for all three exist
(`app/infra/providers/{gcp_gke,aws_eks,azure_aks}.py`) and the open-source
overlay refuses them at `get_provider()` — see
[Cloud clusters, load balancing and scale](#4-a-cluster-you-already-run) for
exactly where that gate lives and where it does not.

Ray, the Ray images, and the KubeRay operator are aligned on a single Ray
version — see the [version matrix](versions.md).

> **Two deployment paths exist in this repo.** This document covers **Part B**:
> the Kubernetes + Ray/KubeRay path driven by `app/infra/cluster_bootstrap.py`.
> There is also an older Docker-Desktop + Helm path (**Part A**,
> `deploy/avaloka/run.sh`) documented in [`deploy/README.md`](../deploy/README.md).

---

## 1. Prerequisites

| Target | Tools required |
| ------ | -------------- |
| All | `docker`, `kubectl`, `helm` v3+ |
| Local | `kind` (`brew install kind`) |
| GCP (GKE) | `gcloud` authenticated, `GCP_PROJECT_ID` set |
| AWS (EKS) | `aws` authenticated, `AWS_REGION` / `AWS_EKS_CLUSTER_ROLE_ARN` / `AWS_SUBNET_IDS` / `AWS_SECURITY_GROUP_IDS` set |
| Azure (AKS) | `az` authenticated, `AZURE_RESOURCE_GROUP` set |

Export the agent LLM keys before deploying — they are injected into the avaloka
Secret at install time:

```bash
export GROQ_API_KEY_PLANNING_AGENT=...
export GROQ_API_KEY_CODING_AGENT=...
export OPENROUTER_API_KEY=...        # optional backup provider
```

Those two Groq keys are what a chart deployment needs, because the chart pins
the provider: `inference.provider` defaults to `"groq"` and
`deploy/helm/avaloka/templates/configmap.yaml:31` always emits
`INFERENCE_PROVIDER`. A *local*
install behaves differently — with `INFERENCE_PROVIDER` unset,
`app/core/inference.py` resolves the default to OpenRouter, preferring whichever
provider actually has a key. To get that behaviour in-cluster, set
`--set inference.provider=openrouter` alongside the key.

`OPENROUTER_API_KEY` is otherwise optional and does a second, narrower job.
Groq's token-per-minute limit is enforced per organisation, so both Groq keys
share one budget; when the planner exhausts it the provider returns 429. Setting
this key lets the planner retry such failures — plus 404 and 5xx — on
OpenRouter. It reaches the pod through `secrets.openrouterKey`, and the Secret
key is only emitted when a value is supplied.

## 2. Quick start (local kind)

```bash
cd deploy
make up PROVIDER=local
```

This one command:

1. Preflights your tools.
2. Builds four images — `avaloka-api:latest`, `avaloka-ray:latest`,
   `avaloka-ui:latest` and `avaloka-functions:latest`
   (`app/infra/deploy_stack.py:35-39`) — and side-loads them into the kind
   nodes.
3. Creates the kind cluster from `deploy/clusters/kind-cluster.yaml` (control
   plane + one worker, with NodePort mappings to the host).
4. Installs the **KubeRay operator**.
5. Applies the **RayCluster** (`deploy/helm/ray/raycluster.yaml`).
6. `helm upgrade --install`s the avaloka app.
7. Deploys the **Ray Serve** inference app (`deploy/helm/ray/rayservice.yaml`).

When it finishes:

| Surface | Access |
| ------- | ------ |
| Avaloka API | <http://localhost:9010> — kind maps NodePort 30085 to host port **9010**, not 9000 (`deploy/clusters/kind-cluster.yaml:15-16`); docs at `/docs`, health at `/health` |
| Avaloka Web UI (React/SSR) | <http://localhost:30090> — `webui.enabled`, **on by default**, NodePort 30090 |
| Streamlit UI (opt-in) | `--set ui.enabled=true`; NodePort 30086 → port 8501. Off by default. A different thing from the React UI above. |
| Local Supabase gateway | <http://localhost:30091> (kong) |
| Object store (S3 API) | `kubectl port-forward svc/avaloka-seaweedfs 8333:8333` — SeaweedFS has no console and no host port is mapped |
| Ray dashboard | <http://localhost:8265> (kind NodePort 30265) |
| Inference | `kubectl port-forward svc/avaloka-inference-serve-svc 8000:8000` |

> **Port 9000 vs 9010.** The container and the Service both use 9000, so
> `kubectl port-forward svc/avaloka 9000:9000` gives you 9000. The kind
> *NodePort* path is the one that lands on 9010. Both are correct; they are
> different routes.

> **Object storage is SeaweedFS.** `seaweedfs.enabled` is `true` by default; it
> serves S3 inside the cluster as `avaloka-seaweedfs:8333` and holds uploads and
> MLflow artifacts. It replaced MinIO, whose images are no longer pullable —
> anonymous requests to `quay.io/minio/minio`, `quay.io/minio/mc` and the Docker
> Hub equivalents all return `401`. `minio.enabled` is now `false` and exists
> for one release as a rollback path; the chart refuses to render with both
> stores on. SeaweedFS and its bucket hook are pulled from Docker Hub by digest
> (`chrislusf/seaweedfs`, `amazon/aws-cli`) — mirror them for anything you
> depend on.

The chart deploys the **avaloka API** (FastAPI/uvicorn on `:9000`) as the primary
workload, along with a small in-cluster Redis it depends on. The React app in
`ui/` ships as the `webui` component — a Node SSR server on container port 3000,
`webui.enabled: true` by default — with a local Supabase database; see the root
README. The separate `ui` block is an **optional Streamlit** surface on 8501 and
is off by default, so `ui.enabled=true` does not turn on the React UI.

Tear down:

```bash
make down PROVIDER=local
```

## 3. Connect to an existing Ray cluster

If you already run Ray on Kubernetes, skip cluster/operator creation and just
deploy the app pointed at it:

```bash
cd deploy
make connect RAY_ADDRESS=ray://<head-host>:10001
```

## Images: pull, or build

Avaloka's own images are published to **GHCR** by
`.github/workflows/images.yml`, which runs on pushes to `main`, `develop-1.6`
and `oss/**`, and on `v*` tags — but only when something that actually goes
into an image changes (`app/`, `ui/`, `requirements.txt`, `pyproject.toml`,
`deploy/docker/**`). All four images are currently published and pull
anonymously:

| Image | What it runs |
| --- | --- |
| `ghcr.io/guruvaidev/avaloka-api` | the FastAPI backend (`app.api.server:app`) |
| `ghcr.io/guruvaidev/avaloka-ui` | the React UI |
| `ghcr.io/guruvaidev/avaloka-ray` | the Ray worker/head image |
| `ghcr.io/guruvaidev/avaloka-functions` | the Supabase edge functions |

Built for `linux/amd64` and `linux/arm64`, so Apple Silicon works without
emulation, and published with build provenance you can verify:

```bash
gh attestation verify oci://ghcr.io/guruvaidev/avaloka-api:main \
  --owner guruvaidev
```

**Which tags exist.** Verified against the registry: every one of the four
repositories serves `main` and `latest`, plus the content tags described below
and a `sha-<short>` tag per build. There are **no semver tags** —
`images.yml` would publish them on a `v*` git tag, but no such tag has been
pushed, so `:1.0.0` and `:1.6.0` do not resolve. There is also no
`develop-1.6` tag today, despite that branch being a build trigger. Use `main`,
or a content tag, and check before you pin:

```bash
docker manifest inspect ghcr.io/guruvaidev/avaloka-api:main
```

The chart defaults to these, so `helm install` needs no Docker and no build.

> **Why this matters.** The chart previously used bare names —
> `avaloka-api:latest` — which resolve to Docker Hub, where they do not exist.
> That works on `kind`, because `make images` side-loads them onto the node and
> `pullPolicy: IfNotPresent` then never pulls; it fails on every cluster that
> cannot side-load, with `ImagePullBackOff`. kind is exactly the environment in
> which the problem is invisible, so it survived to the release branch.
> `tests/contract/test_c2_chart_invariants.py` now fails if a bare Avaloka
> image name comes back.

### Avoiding a rebuild you do not need

Each image has two costs: the dependency layer — the virtualenv plus the baked
embedding model, minutes and hundreds of megabytes — and `COPY . /app`, which
is fast. The Dockerfile copies `requirements.txt` *before* the application
source precisely so a source-only change reuses the expensive layer.

`scripts/image_tag.py` makes that visible. It hashes the files that actually
build an image, so the tag stays the same across a hundred source-only commits:

```bash
python scripts/image_tag.py --all
#  api        deps-1383e62f295c   src-faa89e102e30
#  functions  deps-8736c714a0f5   src-239a9fa00214
```

The `deps-` tag changes only when `requirements.txt` or the Dockerfile changes.
The `src-` tag changes with the application source. CI publishes both, and
**skips the build entirely when the `src-` tag already exists** — so a branch
that changes nothing in an image spends no time rebuilding it.

From a checkout:

```bash
make images-status      # which images would a deploy have to build?
make images-pull        # pull the ones that already exist (and kind-load them)
```

`make images-status` prints `up to date` or `NEEDS BUILD` per image, which is
the question you actually want answered before a deployment.

The build cache is pushed to `ghcr.io/guruvaidev/avaloka-<image>:buildcache` as
well as GitHub's Actions cache. The Actions cache is scoped to one repository
and evicted after a week; the registry cache persists, is shared across
branches, and a local `docker build` can use it too — so `develop-1.6` reuses
`main`'s layers rather than rebuilding them.

### Which tag does the chart use?

`main`, for every image, on every branch:

| Chart value | Default | Exists on GHCR? |
| --- | --- | --- |
| `image.repository` / `.tag` (api) | `ghcr.io/guruvaidev/avaloka-api` / `main` | yes |
| `webui.image.repository` / `.tag` | `ghcr.io/guruvaidev/avaloka-ui` / `main` | yes |
| `ui.image.repository` / `.tag` (Streamlit) | `ghcr.io/guruvaidev/avaloka-ray` / `main` | yes |
| `supabase.functions.image` | `ghcr.io/guruvaidev/avaloka-functions:main` | yes |

> **The Ray cluster image is not one of these.** There is no `ray.image` value
> — `ray:` in `values.yaml` carries only `connectExisting`, `address` and
> `inClusterAddress`. The RayCluster and RayService CRs hard-code the bare name
> `avaloka-ray:latest`
> (`deploy/helm/ray/raycluster.yaml:32,80`, `deploy/helm/ray/rayservice.yaml:42,73`),
> which resolves to Docker Hub, where it does not exist. That is invisible on
> `kind`, because `make up` side-loads the image and `IfNotPresent` then never
> pulls — and it is `ImagePullBackOff` on any cluster that cannot side-load.
> **Edit those two manifests to a registry path before deploying Ray anywhere
> else.** This is the same defect class the warning above describes, still
> present in the Ray manifests.

> **This is where a release broke before.** The chart once pinned
> `develop-1.6`, a tag that has never existed on GHCR, so `helm install` hit
> `ImagePullBackOff` on every service. Before changing an image tag in
> `values.yaml`, resolve it against the registry — a tag that reads plausibly
> and 404s costs every user of that chart a debugging session.
> `tests/contract/test_c2_chart_invariants.py` guards the related failure (a
> bare image name resolving to Docker Hub) but it cannot tell you whether a
> registry tag exists.

`main` is a mutable tag: the same reference means different bits over time. For
a reproducible deployment, pin the content tag instead:

```bash
helm upgrade --install avaloka deploy/helm/avaloka \
  --set image.tag=$(python scripts/image_tag.py api --source)
```

### Building them yourself

Still the faster loop when you are changing code, and the only option offline:

```bash
make images PROVIDER=local          # builds and side-loads into kind
helm upgrade --install avaloka deploy/helm/avaloka \
  --set image.repository=avaloka-api \
  --set webui.image.repository=avaloka-ui
```

The builder stage of `deploy/docker/Dockerfile.api` carries a Rust toolchain.
Nothing should need it — pip installs with `--prefer-binary`, which picks the
newest version of a package that actually ships a wheel rather than the newest
version outright — but it is there so that a dependency without a wheel for
`linux/arm64` still builds rather than failing the release. The runtime stage
copies only the virtualenv, so none of the toolchain reaches the shipped image.

---

## 4. A cluster you already run

```bash
make up PROVIDER=local          # provisions kind
# or point kubectl at your own cluster and install the chart directly
```

**`PROVIDER=gcp|aws|azure` is a commercial capability and the open-source
distribution refuses it.** `get_provider()` raises
`CommercialCapabilityRequired` for `gcp`, `aws` and `azure` unless
`resolve_capabilities(...)` reports `cloud_provisioning`.

Be precise about where that gate lives: it is in
`oss/overlay/app/infra/providers/factory.py`, the overlay
`scripts/generate-oss.sh` applies when it builds the public tree from
`develop-1.6` per `oss/manifest.yaml`. The copy of `factory.py` on
`develop-1.6` has **no** such check — it resolves all four providers
unconditionally. So the refusal is real in the open-source release you
installed, and absent in an internal development checkout. If you are working
from `develop-1.6` directly, nothing stops `make up PROVIDER=gcp` from creating
a billable cluster; check your `PROVIDER` rather than relying on the gate.

What open source *does* support is deploying into a cluster you provisioned
yourself, wherever it runs: the chart, Ray/KubeRay, distributed execution and
the scheduler all work against any cluster your `kubeconfig` reaches.

On a cluster with a cloud load-balancer controller, set
`--set service.type=LoadBalancer`; the chart default is `NodePort`
(`values.yaml`, `service.type`), which suits kind. For images, the chart
defaults already point at GHCR and need no change. What *does* need changing is
the RayCluster/RayService image: push `avaloka-ray` to a registry your cluster
can reach and edit `deploy/helm/ray/raycluster.yaml` and
`deploy/helm/ray/rayservice.yaml`, which currently name it bare.

> **Scope.** The open-source edition is sized for a laptop or a small
> self-managed cluster, and that is what is tested. Provisioned and autoscaled
> clusters, scheduled cloud workloads and multi-tenant operation are
> Professional and Enterprise capabilities — [avaloka.ai](https://avaloka.ai)
> or **[support@avaloka.ai](mailto:support@avaloka.ai)**. You run the
> open-source edition at your own risk.

For repeatable, throwaway cloud testing, prefer an **ephemeral cluster**
pattern — create → test → unconditionally destroy, with a labelled-cluster
reaper to catch orphans — over hand-created cloud clusters, because a leaked
managed cluster bills continuously. See [`testing.md`](testing.md).

## 5. Under the hood

The orchestration is plain Python; the Makefile is a thin wrapper over
`python -m app.infra.cluster_bootstrap`:

```text
cluster_bootstrap.py   --mode provision|connect  --provider local|gcp|aws
        │
        ├─ install_k8s.check_tools_for(provider)     preflight
        ├─ deploy_stack.build_images()               docker build + kind load
        ├─ cloud_provisioner.provision(provider)     provider factory → create/kubectl
        ├─ install_k8s.install_kuberay()             KubeRay operator (Helm)
        ├─ ray_manager.apply_ray_cluster()           RayCluster CR
        ├─ deploy_stack.deploy_avaloka()             helm upgrade --install avaloka
        └─ ray_manager.deploy_ray_serve()            RayService CR
```

Direct invocation (equivalent to `make up PROVIDER=local`):

```bash
python -m app.infra.cluster_bootstrap --mode provision --provider local --namespace default
```

Useful flags: `--service-type ClusterIP|NodePort|LoadBalancer`, `--skip-serve`,
`--data-stack postgres kafka milvus neo4j opensearch`.

## 6. The Helm chart

`deploy/helm/avaloka/` is the chart `cluster_bootstrap` installs. Key values
(`deploy/helm/avaloka/values.yaml`):

| Value | Meaning |
| ----- | ------- |
| `image.repository` / `image.tag` / `image.pullPolicy` | The API image (`avaloka-api`). `IfNotPresent` for side-loaded kind images; `Always` + a registry path for cloud. |
| `service.type` / `port` / `nodePort` | API on `:9000`; default `NodePort` 30085 (reached on host port 9010 under kind). Set `LoadBalancer` explicitly for cloud. |
| `redis.enabled` / `redis.externalUrl` | Ship a small in-cluster Redis (default) or point at an external one — the API needs Redis for sessions/uploads/inference. |
| `webui.enabled` | The React/SSR web UI (`avaloka-ui`), Node on container port 3000, NodePort 30090. **On by default.** |
| `ui.enabled` | Opt-in **Streamlit** UI as a second Deployment/Service on `:8501` (NodePort 30086). Off by default. Not the React UI. |
| `seaweedfs.enabled` | In-cluster S3 object store (SeaweedFS). **On by default.** |
| `minio.enabled` | The store SeaweedFS replaced, kept for one release as a rollback path. Off by default; the chart refuses to render with both enabled. |
| `config.inferenceBackend` / `config.rayServeUrl` | `rayserve` (default) posts predictions to the in-cluster Ray Serve service; `gateway` uses the MTA v2 managed path. |
| `ray.connectExisting` / `ray.address` | Attach to an external Ray cluster vs. use the in-cluster one. |
| `ray.inClusterAddress` | `ray://avaloka-raycluster-head-svc:10001`. |
| `secrets.groqPlanningKey` / `groqCodingKey` / `jwtSecret` / `existingSecret` | Planner and coder LLM keys + JWT secret, injected at install; or reference a pre-created Secret. `secrets.groqApiKey` is a third, separate key read only by `app/agents/visualization_agent.py`. |
| `inference.provider` | `groq` by default; `local`, `openai`, `openrouter`, `bedrock`, `vertex` or `azure`. Always emitted as `INFERENCE_PROVIDER`, so it overrides the key-aware default in `app/core/inference.py`. |
| `secrets.openrouterKey` | Optional OpenRouter key arming the planner's backup provider on 429/404/5xx. Omitted from the Secret when empty. |
| `rbac.create` / `serviceAccount` | In-cluster RBAC so the app can submit Ray jobs and patch RayServices. |
| `resources`, `nodeSelector`/`tolerations`/`affinity` | Standard scheduling/scaling controls. |

Per-cloud overlays live beside the chart in `deploy/helm/avaloka/values/`
(`values-gke.yaml`, `values-gke-app.yaml`, `values-eks.yaml`,
`values-aks.yaml`, `values-minikube.yaml`, `values-onprem.yaml`) — that is the
directory `tests/contract/test_c2_chart_invariants.py` asserts against. A
second, older copy of five of them also exists under `deploy/avaloka/values/`;
prefer the chart-adjacent set.

Validate a chart change without a cluster:

```bash
helm lint deploy/helm/avaloka
helm template deploy/helm/avaloka
```

## 7. Ray & KubeRay

- **RayCluster** (`deploy/helm/ray/raycluster.yaml`): `avaloka-raycluster`, head
  service `avaloka-raycluster-head-svc` (client `10001`, dashboard `8265`), one
  worker autoscaling 1→4, in-tree autoscaling enabled.
- **RayService** (`deploy/helm/ray/rayservice.yaml`): `avaloka-inference`,
  Serve import path `app.serve.inference:app`, service
  `avaloka-inference-serve-svc:8000`.
- **Operator version** is `1.4.2`, the default of `KUBERAY_OPERATOR_VERSION` in
  `app/infra/ray_manager.py:34`; override it with that environment variable.

Because the app and the Ray cluster communicate over the Ray Client protocol
(`ray://…:10001`), **the Ray version must match on both ends** — the app image
(`requirements.txt`), the Ray image (`deploy/docker/Dockerfile.ray`), and both CR
`rayVersion` fields. The exact pinned version and the compatible operator release
are in the [version matrix](versions.md).

## 8. Optional data stack

Postgres / Kafka / Milvus / Neo4j / OpenSearch are opt-in and disabled by
default:

```bash
make up PROVIDER=local DATA_STACK=postgres
```

Values live in `app/infra/manifests/helm-values/<component>-values.yaml`.

## 8b. Storage: what survives a node going down

Every store that holds state Avaloka is expected to remember is a **StatefulSet
with a `volumeClaimTemplate`** — Redis, Chroma and Postgres. The claim belongs to
the workload, so the volume follows the pod when Kubernetes reschedules it
rather than being stranded on a node that went away. SeaweedFS and the MCP registry
keep standalone PVCs (moving them would orphan existing data); they are durable
either way.

### Replicas are not the answer

Postgres, Redis and Chroma are single-writer. Setting `replicas: 3` on these
gives you three independent empty stores, not one highly available one — each
replica gets its own volume and neither knows about the others. Real database HA
needs replication the database itself understands:

| Store | What HA actually requires |
| --- | --- |
| Postgres | streaming replication — [CloudNativePG](https://cloudnative-pg.io/) or Patroni |
| Redis | Sentinel or Redis Cluster |
| Chroma | not available in open-source Chroma (single-node) |
| Milvus | the Milvus operator in cluster mode |

The chart deliberately does not run any of these. When you need them, point the
chart at a managed service — `postgres.enabled=false` + `postgres.externalUrl`,
`redis.enabled=false` + `redis.externalUrl`.

### Node failure is the StorageClass's job

For a self-managed cluster, survival of a lost node comes from **replicated block
storage underneath the PVC**. Set it once, chart-wide:

```yaml
storage:
  className: longhorn
```

```bash
helm upgrade --install avaloka deploy/helm/avaloka --set storage.className=longhorn
```

Every claim in the chart — Redis, Postgres, Chroma, SeaweedFS, MCP, Milvus, Supabase
— resolves through that one value. Any component can still override it with its
own `persistence.storageClass`.

**Which one to use:**

| StorageClass | When it fits |
| --- | --- |
| **[Longhorn](https://longhorn.io/)** — *recommended default* | On-prem and bare metal. CNCF project, synchronous 3-way replication per volume, installs with one Helm command, rebuilds replicas automatically when a node dies. Needs `open-iscsi` on each node. The right answer for a small self-managed cluster. |
| [Rook-Ceph](https://rook.io/) (`rook-ceph-block`) | You also want object and shared-filesystem storage from the same pool, and have ≥5 nodes and someone willing to operate Ceph. More capable, considerably more to run. |
| [OpenEBS Mayastor](https://openebs.io/) | NVMe-class latency matters. Fastest of the three; newest, and the most particular about hardware (hugepages, dedicated devices). |
| Cloud CSI (`gp3`, `premium-rwo`, `standard-rwo`) | Managed Kubernetes. Already replicated **within a zone** — which survives a node, not a zone. Use a regional class if you need zone failure tolerance. |

**Sizing** defaults, all overridable per component:

| Component | Default | Holds |
| --- | --- | --- |
| `postgres.persistence.size` | 20Gi | layer-3 artifact store, keyed by user |
| `redis.persistence.size` | 8Gi | layer-1 schema cache, session hints and preferences |
| `chroma.persistence.size` | see values.yaml | layer-2 user signature |
| `seaweedfs.persistence.size` | 20Gi | uploaded datasets and trained models |

### Redis durability

`redis.appendOnly` is on by default. A volume alone is not durability: with RDB
snapshots only (the `redis:7` default of `save 3600 1 300 100 60 10000`), an
unclean shutdown discards every write since the last snapshot — up to an hour of
cached schema and remembered preferences. `appendfsync everysec` bounds that loss
to one second.

### Leaving it unset

With no `storage.className`, claims fall through to the cluster's default
StorageClass, so `kind` and Docker Desktop work with no configuration. Be clear
about what that gives you: a single-node default class (kind's `standard`,
`local-path`) writes to that one node's disk. It survives a pod restart. It does
not survive losing the node.

### Upgrading an existing install

Redis, Chroma and Postgres change `kind` from `Deployment` to `StatefulSet`, and
Helm cannot mutate a workload's kind in place. Delete the old workloads first;
their data was on an `emptyDir` or no volume at all, so there is nothing to
preserve:

```bash
kubectl delete deployment avaloka-redis avaloka-postgres avaloka-chroma --ignore-not-found
helm upgrade --install avaloka deploy/helm/avaloka --set storage.className=longhorn
```

The object store and MCP keep their existing claims and are untouched by the
upgrade. An `avaloka-minio` claim from an earlier install is kept too, but its
contents are not copied into SeaweedFS.

---

## 9. Verifying a deployment

```bash
make status                      # pods, services, RayClusters, RayServices

# API health. Two routes, two ports:
#   kind NodePort 30085 is mapped to host 9010 by deploy/clusters/kind-cluster.yaml
curl http://localhost:9010/health
#   or forward the Service explicitly
kubectl port-forward svc/avaloka 9000:9000 &
curl http://localhost:9000/health

# Ray dashboard / version (should match the version matrix)
kubectl port-forward svc/avaloka-raycluster-head-svc 8265:8265 &
curl http://localhost:8265/api/version

# Inference (Ray Serve)
kubectl port-forward svc/avaloka-inference-serve-svc 8000:8000 &
curl http://localhost:8000/                 # readiness
curl -XPOST http://localhost:8000/ -d '{"features":{"a":1}}'
```

`/health` returns `graph_ready`, `redis_connected` and `langgraph_url`
(`app/api/server.py:1198-1203`) — check it before anything else.

The [testing guide](testing.md) formalizes this into tiers (deploy-logic →
cluster deployment → KubeRay execution → agent workloads → multi-cloud matrix).
Use it as the source of truth for what "a working deployment" must prove.

## 10. Safety

- **Never run destructive operations against a cloud kube-context by accident.**
  Deployment scripts and cluster tests should refuse to `helm uninstall` /
  `kubectl delete` against a non-`kind`/`minikube` context without an explicit
  opt-in (`AVALOKA_TEST_ALLOW_CLOUD=1`). Check your current context with
  `kubectl config current-context` before running `make down`.
- **Keep secrets out of git.** Provide model-provider keys, cloud credentials,
  and `SUPABASE_JWT_SECRET` via environment/Secrets, never in committed values
  files. The variable is `SUPABASE_JWT_SECRET`; `JWT_SECRET` is only the Python
  name it is bound to (`app/api/server.py:322`) and setting it does nothing.
- **Don't expose the Ray dashboard publicly** — it has no authentication.
