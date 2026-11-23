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
7. Deploys the **Ray Serve** inference app (`deploy/helm/ray/rayservice.yaml`) **only when
   `INFERENCE_BACKEND=rayserve`**; by default models are served by the chart's
   model-server Deployment (see §7b), which costs nothing when idle.

When it finishes:

| Surface | Access |
| ------- | ------ |
| Avaloka API | <http://localhost:9010> — kind maps NodePort 30085 to host port **9010**, not 9000 (`deploy/clusters/kind-cluster.yaml:15-16`); docs at `/docs`, health at `/health` |
| Avaloka Web UI (React/SSR) | <http://localhost:30090> — `webui.enabled`, **on by default**, NodePort 30090 |
| Streamlit UI (opt-in) | `--set ui.enabled=true`; NodePort 30086 → port 8501. Off by default. A different thing from the React UI above. |
| Local Supabase gateway | <http://localhost:30091> (kong) |
| Object store (S3 API) | `kubectl port-forward svc/avaloka-seaweedfs 8333:8333` — SeaweedFS has no console and no host port is mapped |
| Ray dashboard | `kubectl port-forward svc/avaloka-raycluster-head-svc 8265:8265`, then sign in with the token (see [Ray and MLflow authentication](#ray-and-mlflow-authentication)). There is no NodePort any more; kind's host port 8265 mapping is unused. |
| MLflow | `kubectl port-forward svc/avaloka-mlflow 5000:5000`, then sign in as `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` from `avaloka-secrets` |
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
`.github/workflows/images.yml`, which runs on pushes to `main` and
`develop-1.6`, and on `v*` tags — but only when something that actually goes
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
and a `sha-<short>` tag per build. From 2.0, every build also carries the
release number read from the `VERSION` file of the branch it was built from —
`:2.0.0` and `:2.0` for the 2.0 line, `:1.6.0` and `:1.6` for a branch whose
`VERSION` still says 1.6.0 — so the tag is decided by the tree, not by a
branch-name table. Those tags move with the branch until a `v*` git tag is
pushed. Check before you pin:

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
> (`deploy/helm/ray/raycluster.yaml:32,92`, `deploy/helm/ray/rayservice.yaml:42,78`),
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
| `config.deployment` | `AVALOKA_DEPLOYMENT`: who operates the install. `self_hosted` (default) or `avaloka_hosted`; anything else fails the render. See [Who operates the install](#who-operates-the-install-avaloka_deployment). |
| `config.inferenceBackend` / `config.rayServeUrl` | `rayserve` (default) posts predictions to the in-cluster Ray Serve service; `gateway` uses the MTA v2 managed path. |
| `ray.connectExisting` / `ray.address` | Attach to an external Ray cluster vs. use the in-cluster one. |
| `ray.inClusterAddress` | `ray://avaloka-raycluster-head-svc:10001`. |
| `secrets.groqPlanningKey` / `groqCodingKey` / `jwtSecret` / `existingSecret` | Planner and coder LLM keys + JWT secret, injected at install; or reference a pre-created Secret. `secrets.groqApiKey` is a third, separate key read only by `app/agents/visualization_agent.py`. |
| `inference.provider` | `groq` by default; `local`, `openai`, `openrouter`, `bedrock`, `vertex` or `azure`. Always emitted as `INFERENCE_PROVIDER`, so it overrides the key-aware default in `app/core/inference.py`. |
| `secrets.openrouterKey` | Optional OpenRouter key arming the planner's backup provider on 429/404/5xx. Omitted from the Secret when empty. |
| `secrets.mcpServiceToken` | `MCP_SERVICE_TOKEN`, the shared API <-> MCP service identity the planner presents to the onboarding routes. The API and the MCP pod read it from the same Secret. Empty generates one per release, kept across upgrades; with `existingSecret`, that Secret must carry it. See Safety. |
| `secrets.langgraphServiceToken` | `LANGGRAPH_SERVICE_TOKEN`, the shared API <-> LangGraph server secret. Generated and kept across upgrades when empty. Reaches the api and langgraph pods through a `secretKeyRef` (so with `existingSecret`, that Secret must carry it). The LangGraph server answers `401` to every call without it except `GET /ok`; it is internal-only, with no Ingress route and a NetworkPolicy that admits only the api pods. |
| `admin.emails` | Administrator addresses (list). **Empty by default, which means no admin.** See [Administrators](#administrators) below. |
| `rbac.create` / `serviceAccount` | In-cluster RBAC so the app can submit Ray jobs and patch RayServices. |
| `resources`, `nodeSelector`/`tolerations`/`affinity` | Standard scheduling/scaling controls. |

### Who operates the install (`AVALOKA_DEPLOYMENT`)

One variable says whether an install is your own or one of the Avaloka-operated
hosted sites. It is read in one place, `deployment_state()` in
`app/core/editions.py`, by every check that differs between the two.

| Value | Meaning |
| ----- | ------- |
| `self_hosted` | Your own install. No billing or plan caps; local auth may be enabled; the API docs and `/debug/whoami` are public. **The default** in the Helm chart (`config.deployment`), the quickstart and the scheduler compose file. |
| `avaloka_hosted` | An Avaloka-operated hosted site. Billing and plan caps apply; local auth is refused; `/openapi.json`, `/docs`, `/docs/oauth2-redirect`, `/redoc` and `/debug/whoami` need a login. |
| unset, or any other word | *Undeclared*. Each check keeps the default it had before the variable could be declared: billing and plan caps treat the install as hosted, while editions, local auth and the API docs treat it as self-hosted. The server logs one warning, naming only whether the variable is set. Declare it. |

`-` and `_` are interchangeable and case is ignored (`avaloka-hosted` works). The
chart refuses any value but the two above, so a typo cannot render an undeclared
install. `/api/capabilities` is public on both.

A service that bills its users by plan sets the hosted value in its own
values overlay:

```yaml
config:
  deployment: avaloka_hosted
```

Because the chart default is `self_hosted`, an install that bills plans and
upgrades without such an overlay must pass `--set config.deployment=avaloka_hosted`
on every `helm upgrade`; otherwise it renders `self_hosted` and premium features
are open to every plan.

`values-gke.yaml` is also a hosted service's overlay, so it declares
`avaloka_hosted` (and `values-gke-app.yaml` repeats it). `make up` (`app/infra/deploy_stack.py`) always passes
`--set-string config.deployment=self_hosted`, for every provider, so an install it
makes is self-hosted whatever its overlay says. Installing `values-gke.yaml` by
hand on your own cluster, add `--set config.deployment=self_hosted`.

Outside Helm, set `AVALOKA_DEPLOYMENT` in `.env` or the compose environment;
`deploy/compose/docker-compose.scheduler.yml` defaults it to `self_hosted`.

### Administrators

No address is an administrator by default, and nothing falls back to a built-in
address or to "the first user who signs up". Name them at install or upgrade:

```bash
helm upgrade --install avaloka deploy/helm/avaloka \
  --set 'admin.emails={ops@example.com,lead@example.com}'
# or, with make up / cluster_bootstrap:
export ADMIN_EMAILS=ops@example.com,lead@example.com
```

The chart passes the list, as the comma-separated `ADMIN_EMAILS` environment
variable, to two places:

- **The web UI's server functions.** Listed addresses may use the support desk
  (`/support-queries`). Case and surrounding whitespace are ignored. The browser
  never sees the list; it asks the server.
- **The `supabase-migrate` hook** (in-cluster Supabase only). After the
  migrations it grants the `admin` role in `public.user_roles` to each listed
  address that already has a confirmed account. It only adds: removing an
  address from the list does not revoke its grant, and admins granted any other
  way are left alone. An address with no account yet is granted on the first
  deploy after the account exists.

**Upgrading from 1.6.x:** set `admin.emails` (or `ADMIN_EMAILS`) on the upgrade.
Until you do, no account can open the support desk. Existing `user_roles`
grants are not touched.

**Admins require email confirmation.** With the in-cluster Supabase
(`supabase.enabled: true`), the chart **refuses to render** a non-empty
`admin.emails` while `supabase.auth.autoconfirm` is `true` (the chart default).
With autoconfirm on, gotrue confirms a new account without the owner clicking a
link, so anyone could sign up with a listed address that has no account yet and
be granted admin. Either turn autoconfirm off and configure SMTP so confirmation
mail is delivered:

```bash
helm upgrade --install avaloka deploy/helm/avaloka \
  --set supabase.auth.autoconfirm=false \
  --set-string supabase.auth.smtp.host=smtp.example.com \
  --set 'admin.emails={ops@example.com}'
```

or leave `admin.emails` empty. As a second line of defence the migrate hook
skips the admin grant, with a warning in its log, unless autoconfirm is
explicitly `false`. With a hosted Supabase project (`supabase.enabled: false`)
the chart cannot see the project's setting: make sure "Confirm email" is
enabled in the project's auth settings before listing admins.

### Supabase migrations

The `supabase-migrate` hook applies each file in
`files/supabase-migrations/` once and records it in
`avaloka_meta.schema_migrations` (a schema PostgREST does not expose). It used
to re-run every file on every deploy. On the first upgrade to a chart with
tracking, an install whose schema already has the objects of the newest
pre-tracking migrations is backfilled from `files/db-migrations.baseline`
without re-running anything; any other database runs the baseline files once
the old way. Migrations added later run inside a transaction and fail the hook
when they fail. `deploy/helm/avaloka/files/db-migrate.sh` documents the
cases.

#### Duplicate migration version 20261101120000

Two shipped migrations share the version prefix `20261101120000`:
`20261101120000_plan_enforced_org_features.sql` and
`20261101120000_support_admin_from_roles.sql`. **They are kept as they are.**

- **The chart hook is unaffected.** It records each file under its full name,
  so it applies both, once each.
- **Renaming either file would break the chart.** The hook would treat the
  renamed file as new and run it again.
- **The Supabase CLI keys migrations by the numeric prefix alone**, so it can
  record only one of the two. `tests/test_migration_versions_unique.py` fails
  on any new shared prefix.

The plan-enforcement file is superseded by
`20261101140000_plan_rules_pro_collab.sql`, which does not depend on it. On a
Supabase-CLI-managed project (`supabase db push`), do this before pushing
`20261101140000`:

1. Run `supabase migration list` and note whether `20261101120000` is recorded.
2. Apply the support-admin file by hand. It is idempotent, so it is safe even if it
   already ran:
   `psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f supabase/migrations/20261101120000_support_admin_from_roles.sql`
3. **Do not run `20261101120000_plan_enforced_org_features.sql` by hand.**
   `20261101140000` replaces everything it does. Running it after
   `20261101140000` would put back its Enterprise-only team and invite
   policies.
4. If `20261101120000` is not recorded, record it:
   `supabase migration repair --status applied 20261101120000`.
5. Run `supabase db push`. It applies `20261101140000` and anything later.

#### Old copies of three renumbered migrations, removed

`20261101100000_fix_bg03_analysis_visibility.sql`,
`20261101110000_fix_bg01_org_membership.sql` and
`20261101120000_bg05_bg06_roles_departments.sql` were renumbered to
`20261103130000`, `20261103140000` and `20261103150000`. The old copies had
also been shipped beside them, and the chart hook ran both. The old BG-05/06
file creates its unique indexes before
`20261103145000_dedupe_seed_departments_roles.sql` runs, so on an install with
duplicate department names in an organisation the upgrade stopped there. The
old copies are no longer shipped.

- **Chart installs: nothing to do.** An install that recorded the old files
  keeps those records and runs nothing again. Any other install runs only the
  renumbered files, in order.
- **Supabase-CLI-managed projects.** `20261101100000` and `20261101110000`
  appear in `supabase migration list` as recorded on the project with no local
  file. Mark them reverted, which removes the record only:
  `supabase migration repair --status reverted 20261101100000 20261101110000`.
  `20261101120000` is still a local version and needs nothing.
- **Prevention.** `tests/test_migration_versions_unique.py` fails when one
  migration ships under two versions, or when two files carry the same SQL.

#### Organisations with duplicate department or role names

An upgrade no longer stops on an organisation that holds two departments with the
same name (ignoring case) but different codes, or two roles with the same name and
department but different permissions.
`20261103145000_dedupe_seed_departments_roles.sql` removes exact duplicates, warns
about the rest and keeps them. `20261117110000_org_name_keys_per_org.sql` then
creates the unique name indexes (`departments_org_name_key`,
`config_roles_org_role_dept_key`) for every organisation except those, and warns
how many it left out.

**Such an organisation keeps its duplicate departments (or roles) until an admin
merges them, and the unique index then applies to it.** To find them:

```sql
SELECT organization_id, lower(department), count(*) FROM public.departments
 GROUP BY 1, 2 HAVING count(*) > 1;
SELECT organization_id, lower(role), lower(department), count(*) FROM public.config_roles
 GROUP BY 1, 2, 3 HAVING count(*) > 1;
```

To merge, keep one row of each name. Point the roles (`config_roles.department_id`)
and users (`app_users.department_id`, `app_users.role_id`) at the kept rows before
deleting the others: deleting a department also deletes the roles linked to it,
and clears the link on its users. Then re-apply the index file. It is idempotent:
it rebuilds the index from the rows as they are now, and it becomes the full index
once no organisation still holds a duplicate.

```sh
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 --single-transaction \
  -f deploy/helm/avaloka/files/supabase-migrations/20261117110000_org_name_keys_per_org.sql
```

`20261103145000` was edited in place: it used to raise an exception on such
duplicates. No install needs a manual step for that:

- **Chart installs.** The migrate hook records each file in the same transaction
  that applies it. An install where the old file raised never recorded it, so it
  runs the edited file. An install that recorded the old file had no such
  duplicates, so the edit would change nothing there.
- **Supabase-CLI-managed projects** (`supabase db push`, `supabase migration up`)
  track migrations by version, not content. The CLI applies a file's statements
  and its history insert in one batch, which runs as one transaction. A project
  where the old file raised therefore did not record `20261103145000`, and its
  next push applies the edited file. A project where the old file succeeded keeps
  that record, and the edit never reaches it. That is harmless: it had no such
  duplicates. No `supabase migration repair` is needed for this file. Use the
  same re-apply command after a merge, with
  `supabase/migrations/20261117110000_org_name_keys_per_org.sql` from the UI
  project.

#### A migration older than the newest one applied (`--include-all`)

`supabase db push` applies migrations in version order and refuses a local
migration whose version is older than the newest one already recorded on the
project. It reports local migration files "to be inserted before the last
migration on remote database". This happens when a change's migration was
written before a newer one merged.

- **Recovery on a Supabase-CLI-managed project.** Run `supabase migration list`
  to see which local files are unrecorded, confirm the older file is meant to be
  applied out of order, then run `supabase db push --include-all`, which applies
  every unrecorded local migration, older ones included.
- **The chart hook is unaffected.** It records each file by full name and runs
  every unrecorded file, whatever its version.
- **Prevention.** `tests/contract/test_migration_versions.py` and
  `tests/test_migration_versions_unique.py` fail a change whose new migration sorts before the newest migration already merged: against
  the destination branch in pull-request pipelines, and always against
  `MERGED_MIGRATIONS` in that file, which each migration change appends to.
  Give a new migration a version later than every merged one.
- **Merged out of version order on 1.6.** `20261103145000_dedupe_seed_departments_roles.sql`
  merged after `20261103160000_plan_descriptions_true.sql`, and the `20261106*`
  team migrations merged after `20261108100000_bg01_cleanup_selfserve_orgs.sql`.
  They keep their names, for the same reason as the duplicate above. A CLI-managed
  project that recorded the newer files first needs `--include-all` once to
  apply them.

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
  `avaloka-inference-serve-svc:8000`. **Opt-in** (`INFERENCE_BACKEND=rayserve`):
  it keeps a Ray head pod running, so the default serving path is the
  model-server Deployment below. Its Serve deployment autoscales 0→4 and its
  worker group scales to zero when idle.
- **Training is a RayJob on the shared cluster, submit-only.** The worker applies a
  `RayJob` with `clusterSelector: {ray.io/cluster: avaloka-raycluster}` (no
  `rayClusterSpec`, `shutdownAfterJobFinishes: false`, TTL 600 s, a GPU request
  only when the plan asked for one), after `ensure_shared_cluster()` has re-applied
  the cluster if needed and waited for it to be ready (`RAY_ENSURE_TIMEOUT_S`,
  600). Nothing waits in the worker: `poll_training_job` follows the CR on a
  15→60 s countdown chain, reads the result from the submitter's logs, registers
  the model, lists its endpoint ready-on-demand, closes the run record and rings
  the bell. `resume-stale-training-polls` (every 300 s) re-arms a chain the broker
  lost. `POST /api/ml-jobs/{id}/cancel` (or the chat's cancel with the job id)
  deletes the RayJob and marks the job cancelled. `MTA_RAY_SUBMISSION_MODE=direct`
  restores the dashboard-API path that waits in-process.
- **Per-job RayClusters** (the legacy `ray-trainer-<id>` path) are off:
  training uses the shared cluster through its in-cluster address
  (`MTA_RAY_DASHBOARD_URL`). `AVALOKA_ALLOW_LEGACY_RAY_CLUSTER=1` re-enables
  them with a ClusterIP head Service only; nothing creates a public
  LoadBalancer any more. The shared cluster is deleted after
  `RAY_IDLE_TEARDOWN_S` (2 h) without a job and re-applied on the next one.
- **Operator version** is `1.6.2`, the default of `KUBERAY_OPERATOR_VERSION` in
  `app/infra/ray_manager.py`; override it with that environment variable, but not
  below 1.6: that is the first operator that can read a token-authenticated
  dashboard with a token from a Secret we name (`authOptions.secretName`), and
  without it RayJobs never finish and the RayService never turns ready. Helm never
  upgrades CRDs, so `install_kuberay_operator()` (and `k8s_invoker`'s install path)
  first applies the chart's CRDs server-side (`helm show crds … | kubectl apply
  --server-side`).

### Ray and MLflow authentication

Both used to accept anyone who could reach them. Now:

- **Ray** runs with token authentication (`RAY_AUTH_MODE=token`, Ray 2.52+). One
  token, generated by the chart on first install and kept on every upgrade
  (`ray.auth`, `secrets.rayAuthToken`), is shared by the app's Ray clients
  (`RAY_AUTH_TOKEN` in `avaloka-secrets`, `RAY_AUTH_MODE` in `avaloka-config`), the
  RayCluster and RayService (every container, the autoscaler included, reads
  `auth_token` from `avaloka-ray-auth` by `secretKeyRef`; `authOptions` hands the
  operator the same Secret), and KubeRay-mode RayJobs (the job's own
  `<rayjob>-env` Secret carries it). Ray's client and `JobSubmissionClient` send
  it themselves; the app's plain HTTP calls to a dashboard add it
  (`app/infra/ray_auth.py`).
- **MLflow** runs `mlflow server --app-name basic-auth` (`mlflow.auth`). Its admin
  user is `MLFLOW_TRACKING_USERNAME` with password `MLFLOW_TRACKING_PASSWORD`, both
  in `avaloka-secrets`, the password generated and kept like the token. Every
  MLflow client gets both from that Secret and the mlflow client sends them. The
  user store is SQLite on the pod's emptyDir unless `mlflow.auth.databaseUri` is
  set, so the admin is re-created from the Secret on every start.
- **NetworkPolicies** admit only the clients: `api`, `langgraph`, `celery`,
  `celery-redbeat` and the Ray pods to MLflow; those four, other Ray pods and the
  KubeRay operator to the Ray pods. Neither has a NodePort or LoadBalancer.

From a workstation, `kubectl port-forward` (unaffected by the NetworkPolicies),
then:

```bash
# MLflow: the UI asks for the user and password; a script exports them.
export MLFLOW_TRACKING_URI=http://localhost:5000 MLFLOW_TRACKING_USERNAME=avaloka
export MLFLOW_TRACKING_PASSWORD="$(kubectl get secret avaloka-secrets -o jsonpath='{.data.MLFLOW_TRACKING_PASSWORD}' | base64 -d)"
# Ray: paste the token into the dashboard's login page, or for the CLI:
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN="$(kubectl get secret avaloka-ray-auth -o jsonpath='{.data.auth_token}' | base64 -d)"
ray job submit --address http://localhost:8265 -- python -c 'print(1)'
```

Rotating either: change the value in the Secret(s) (for Ray, `RAY_AUTH_TOKEN` in
`avaloka-secrets` and `auth_token` in `avaloka-ray-auth` together), then restart
the MLflow pod, the Ray clusters (re-apply `deploy/helm/ray/*.yaml`) and the app
pods.

Because the app and the Ray cluster communicate over the Ray Client protocol
(`ray://…:10001`), **the Ray version must match on both ends** — the app image
(`requirements.txt`), the Ray image (`deploy/docker/Dockerfile.ray`), and both CR
`rayVersion` fields. The exact pinned version and the compatible operator release
are in the [version matrix](versions.md).

## 7b. Model serving: one Deployment, zero when idle

Trained models are served by **one** Deployment, `<release>-model-server`
(`deploy/helm/avaloka/templates/model-server.yaml`, `app/serve/model_server.py`),
reachable only through the API: `POST /api/models/{run_id}/predict` does the
owner check and proxies to `http://<release>-model-server:8000/models/{run_id}/predict`
(ClusterIP; the NetworkPolicy admits the api, langgraph and celery pods only).
Models load from the MLflow registry on first use into an LRU bounded by
memory (`modelServer.memoryBudgetFraction` of the container limit) and count
(`modelServer.maxModels`).

**Lifecycle and cost.** The Deployment is installed with `replicas: 0`. After a
successful training the model is registered and its endpoint is listed
**ready-on-demand** -- nothing is warmed. The first prediction wakes the server
(the API answers `503` with `Retry-After` while the pod starts; the panel
retries). Under load an HPA scales 1→`modelServer.maxReplicas` on CPU. After
`modelServer.idleTimeoutS` (7200 s) without a prediction the Celery reaper
(`reap-idle-model-server`, every 300 s) scales it back to zero. Idle cost: no pods.
Where KEDA is installed, `modelServer.scaleToZero: keda` renders a ScaledObject
instead and the reaper stands down; KEDA is never a dependency.

**GPU on request.** `modelServer.gpu.enabled: true` adds a second zero-replica
Deployment with the provider's `nodeSelector`/`tolerations` given as values
(no cloud code); only registry versions tagged `avaloka.requires_gpu=true` are
deployed to it, under the same 2 h rule.

**RBAC.** The app and celery service accounts get `get`/`patch` on
`deployments/scale`, scoped by `resourceNames` to the two model-server
Deployments and nothing else.

**Endpoint state** lives in Postgres (`ml_endpoints`, `ml_jobs`, `ml_models`;
migration `20261117130000`), not in Redis or an MLflow artifact:
`GET /api/models/{run_id}/endpoint` and `GET /api/ml-jobs/{id}` are what the
panel polls.

**Upgrading an install that ran the RayService.** Installs from before 2026-11 run
the `avaloka-inference` RayService (a Ray head pod 24/7) with `RAY_SERVE_URL`
pointing at it. On upgrade with the default backend the bootstrap deletes that
RayService (`ray_manager.remove_ray_serve`, `kubectl delete rayservice
avaloka-inference --ignore-not-found`) and the hourly orphan reaper removes a
lingering one; nothing else is needed. Every model served before is still
listed from MLflow and is served by the model server on its first prediction
(loaded by `run_id`, `runs:/<run_id>/model_files/model.onnx`, or by its registry
version once registered). `helm upgrade --reuse-values` from an older values
file renders: every `modelServer.*` read is nil-guarded, so the Deployment is
simply absent until the new values are applied. Keep the RayService only by
setting `config.inferenceBackend: rayserve`.

Verifying on a cluster (not runnable on a developer laptop without one):

```bash
kubectl get deploy <release>-model-server                       # READY 0/0 after install
kubectl get hpa,networkpolicy -l app.kubernetes.io/component=model-server
# train a model in the chat, then:
curl -H "Authorization: Bearer $TOKEN" $API/api/models/<run_id>/endpoint     # status: ready
curl -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"feature": 1.0}' $API/api/models/<run_id>/predict                   # 503 + Retry-After once, then 200
kubectl get deploy <release>-model-server                       # READY 1/1
# after 2 h without predictions:
kubectl get deploy <release>-model-server                       # READY 0/0 again
kubectl get rayclusters,rayjobs -A                              # only avaloka-raycluster, and only while jobs run
kubectl get svc -A | grep LoadBalancer                          # nothing of ours
```

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

### Celery task results

A finished task result is written to Postgres (`public.celery_task_results`,
service role only) and cached in Redis; a reader that misses Redis loads it
from Postgres (`app/core/celery_results.py`). What is stored is a summary —
status, output location, run statistics, sample names — never the run's
messages, its arguments or its output file. A daily task deletes rows past
retention in bounded batches, finished tasks only.

| Value | Default | Env | Meaning |
| --- | --- | --- | --- |
| `celery.resultExpires` | `604800` | `CELERY_RESULT_EXPIRES` | seconds a result stays in Redis; `0` never expires |
| `celery.resultRetentionDays` | `365` | `CELERY_RESULT_RETENTION_DAYS` | days a result is kept in Postgres. 1–365; up to **1095 on the Enterprise edition**. Outside 1–1095 fails the render; above 365 without Enterprise fails the purge with a configuration error (never clamped) |

Both are read nil-safe, so `helm upgrade --reuse-values` of an older install
gets the defaults. Upgrading an install whose Redis already holds results with
no expiry: run `scripts/ops/backfill_celery_results.py` (dry run by default) to
copy them to Postgres and give each a TTL once its row is verified.

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

# Ray dashboard / version (should match the version matrix). Token auth is on:
# read the token into a variable, never echo it.
kubectl port-forward svc/avaloka-raycluster-head-svc 8265:8265 &
RAY_AUTH_TOKEN="$(kubectl get secret avaloka-ray-auth -o jsonpath='{.data.auth_token}' | base64 -d)"
curl -H "Authorization: Bearer $RAY_AUTH_TOKEN" http://localhost:8265/api/version

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
- **Generated secrets** (`DB_ENCRYPTION_KEY`, `MCP_SERVICE_TOKEN`,
  `AVALOKA_PII_KEY`) are created per install and kept: the chart generates
  each into the release Secret and keeps it across upgrades (a published value,
  such as the old `avaloka-local-dev-encryption-key` default, is replaced, and
  connections saved under it must be registered again: see INSTALL.md,
  "Upgrade consequence");
  `scripts/quickstart.sh` and `scripts/install.sh` write them to the env file.
  See [INSTALL.md](INSTALL.md), Configuration.
- **`MCP_SERVICE_TOKEN`** (`secrets.mcpServiceToken`) is generated per
  install: `scripts/quickstart.sh` writes one to `.env.quickstart`, and the
  chart generates one per release and keeps it across upgrades (pin your own
  with `openssl rand -hex 32`; with `existingSecret`, that Secret must carry
  it). Every `/customers` route on the MCP server requires an identity: the
  customer's api_key, a signed user assertion from the API, or this token
  plus the end user's id. `/customers/{id}/credentials` never returns a
  password to anyone, only a handle sealed under `DB_ENCRYPTION_KEY`, and only
  to the API: a signed user assertion for a database that user owns, or this
  token. A customer's api_key gets 403. The token is the one identity that
  reaches a customer registered before owners were recorded, so DB-to-DB
  transfers from such a legacy database need it; unset, that path fails
  closed. `/admin/*` never accepts it: those routes take only the onboarding
  service's signed MAC.
- **Don't expose the Ray dashboard publicly** — it has no authentication.
