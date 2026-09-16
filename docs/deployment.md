# Deploying Avaloka on Kubernetes

Avaloka can be stood up as a backend on **any Kubernetes cluster your
`kubeconfig` reaches** and run its distributed compute on **Ray/KubeRay**. One
bootstrap path works across all of them, thanks to the provider abstraction in
`app/infra/providers/`.

The open-source edition **provisions** a local `kind` cluster and **connects**
to anything else. Provisioning a managed cloud cluster (GKE, EKS, AKS) is a
commercial capability and is refused in an open-source build — see
[Cloud clusters, load balancing and scale](#4-a-cluster-you-already-run).

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
| Azure (AKS) | `az` authenticated *(provider is on the roadmap)* |

Export the agent LLM keys before deploying — they are injected into the avaloka
Secret at install time:

```bash
export GROQ_API_KEY_PLANNING_AGENT=...
export GROQ_API_KEY_CODING_AGENT=...
export OPENROUTER_API_KEY=...        # optional backup provider
```

`OPENROUTER_API_KEY` is optional. Groq's token-per-minute limit is enforced per
organisation, so both Groq keys share one budget; when the planner exhausts it
the provider returns 429. Setting this key lets the planner retry such failures
— plus 404 and 5xx — on OpenRouter. It reaches the pod through
`secrets.openrouterKey` in the chart, and the Secret key is only emitted when a
value is supplied.

## 2. Quick start (local kind)

```bash
cd deploy
make up PROVIDER=local
```

This one command:

1. Preflights your tools.
2. Builds the `avaloka:latest` and `avaloka-ray:latest` images and side-loads
   them into the kind nodes.
3. Creates the kind cluster from `deploy/clusters/kind-cluster.yaml` (control
   plane + one worker, with NodePort mappings to the host).
4. Installs the **KubeRay operator**.
5. Applies the **RayCluster** (`deploy/helm/ray/raycluster.yaml`).
6. `helm upgrade --install`s the avaloka app.
7. Deploys the **Ray Serve** inference app (`deploy/helm/ray/rayservice.yaml`).

When it finishes:

| Surface | Access |
| ------- | ------ |
| Avaloka API | <http://localhost:9000> (kind NodePort 30085); docs at `/docs`, health at `/health` |
| Avaloka Web UI (opt-in) | enable with `ui.enabled=true`; NodePort 30086 → <http://localhost:8501> |
| Ray dashboard | <http://localhost:8265> (kind NodePort 30265) |
| Inference | `kubectl port-forward svc/avaloka-inference-serve-svc 8000:8000` |

The chart deploys the **avaloka API** (FastAPI/uvicorn on `:9000`) as the primary
workload, along with a small in-cluster Redis it depends on. The web UI (the
React app in `ui/`) is opt-in (`--set ui.enabled=true`). The primary web UI is a
React.js app (built with Lovable) served from `ui/` with a local Supabase
database — see the root README.

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
`.github/workflows/images.yml` on every tag and on `main`:

| Image | What it runs |
| --- | --- |
| `ghcr.io/guruvaidev/avaloka-api` | the FastAPI backend (`app.api.server:app`) |
| `ghcr.io/guruvaidev/avaloka-ui` | the React UI |
| `ghcr.io/guruvaidev/avaloka-ray` | the Ray worker/head image |
| `ghcr.io/guruvaidev/avaloka-functions` | the Supabase edge functions |

Built for `linux/amd64` and `linux/arm64`, so Apple Silicon works without
emulation, and published with build provenance you can verify:

```bash
gh attestation verify oci://ghcr.io/guruvaidev/avaloka-api:1.0.0 \
  --owner guruvaidev
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

| Branch | Chart default | Why |
| --- | --- | --- |
| `oss/1.6` (release) | `1.0.0` | a released version, pinned |
| `develop-1.6` | `develop-1.6` | the branch tag, rebuilt when the image changes |

For a reproducible deployment, pin the content tag instead:

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

**`PROVIDER=gcp|aws|azure` is a commercial capability and an open-source build
refuses it** — `get_provider()` raises rather than creating a billable managed
cluster. What open source *does* support is deploying into a cluster you
provisioned yourself, wherever it runs: the chart, Ray/KubeRay, distributed
execution and the scheduler all work against any cluster your `kubeconfig`
reaches.

On a cluster with a cloud load-balancer controller the avaloka `Service`
defaults to `LoadBalancer`. Image delivery is environment-specific: push
`avaloka:latest` and `avaloka-ray:latest` to your registry and set the chart
`image.repository` and the RayCluster image accordingly.

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
| `service.type` / `port` / `nodePort` | API on `:9000`; `NodePort` 30085 locally, `LoadBalancer` on cloud. |
| `redis.enabled` / `redis.externalUrl` | Ship a small in-cluster Redis (default) or point at an external one — the API needs Redis for sessions/uploads/inference. |
| `ui.enabled` | Opt-in web UI as a second Deployment/Service on `:8501` (NodePort 30086). Off by default. |
| `config.inferenceBackend` / `config.rayServeUrl` | `rayserve` (default) posts predictions to the in-cluster Ray Serve service; `gateway` uses the MTA v2 managed path. |
| `ray.connectExisting` / `ray.address` | Attach to an external Ray cluster vs. use the in-cluster one. |
| `ray.inClusterAddress` | `ray://avaloka-raycluster-head-svc:10001`. |
| `secrets.groqPlanningKey` / `groqCodingKey` / `jwtSecret` / `existingSecret` | LLM keys + JWT secret, injected at install; or reference a pre-created Secret. |
| `secrets.openrouterKey` | Optional OpenRouter key arming the planner's backup provider on 429/404/5xx. Omitted from the Secret when empty. |
| `rbac.create` / `serviceAccount` | In-cluster RBAC so the app can submit Ray jobs and patch RayServices. |
| `resources`, `nodeSelector`/`tolerations`/`affinity` | Standard scheduling/scaling controls. |

Per-cloud overlays live in `deploy/avaloka/values/` (`values-gke.yaml`,
`values-eks.yaml`, `values-aks.yaml`, `values-minikube.yaml`,
`values-onprem.yaml`).

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
- **Operator version** is set by `KUBERAY_OPERATOR_VERSION` in
  `app/infra/ray_manager.py`.

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

## 9. Verifying a deployment

```bash
make status                      # pods, services, RayClusters, RayServices

# API health (chart primary workload, NodePort 30085 -> localhost:9000)
curl http://localhost:9000/health

# Ray dashboard / version (should match the version matrix)
kubectl port-forward svc/avaloka-raycluster-head-svc 8265:8265 &
curl http://localhost:8265/api/version

# Inference (Ray Serve)
kubectl port-forward svc/avaloka-inference-serve-svc 8000:8000 &
curl http://localhost:8000/                 # readiness
curl -XPOST http://localhost:8000/ -d '{"features":{"a":1}}'
```

The [testing guide](testing.md) formalizes this into tiers (deploy-logic →
cluster deployment → KubeRay execution → agent workloads → multi-cloud matrix).
Use it as the source of truth for what "a working deployment" must prove.

## 10. Safety

- **Never run destructive operations against a cloud kube-context by accident.**
  Deployment scripts and cluster tests should refuse to `helm uninstall` /
  `kubectl delete` against a non-`kind`/`minikube` context without an explicit
  opt-in (`AVALOKA_TEST_ALLOW_CLOUD=1`). Check your current context with
  `kubectl config current-context` before running `make down`.
- **Keep secrets out of git.** Provide `GROQ_*`, cloud credentials, and
  `JWT_SECRET` via environment/Secrets, never in committed values files.
- **Don't expose the Ray dashboard publicly** — it has no authentication.
