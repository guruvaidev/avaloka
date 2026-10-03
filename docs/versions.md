# Version & Compatibility Matrix

This is the single source of truth for the versions Avaloka is built and tested
against. Prose elsewhere in the documentation links here rather than repeating
pinned numbers, so there is one place to update when a version changes.

> The authoritative pins live in `requirements.txt`, `deploy/docker/Dockerfile.*`,
> `deploy/helm/ray/*.yaml`, and `app/infra/ray_manager.py`. If this table and
> those files disagree, the files win — please open a fix.

## Core runtime

| Component | Version | Where it's pinned |
| --------- | ------- | ----------------- |
| Python | 3.11 – 3.12 | `pyproject.toml` (`requires-python = ">=3.11,<3.13"`); `scripts/install.sh` (`MIN_PY_MINOR=11`); `deploy/docker/Dockerfile.api` (`python:3.11-slim`); CI matrix is `["3.11", "3.12"]` |
| Graphviz (`dot`) | any recent | **system package, required.** The `graphviz>=0.20.1` entry in `requirements.txt` is only a binding. `.github/workflows/ci.yml` installs it with `apt-get install -y graphviz`. Not checked by `scripts/doctor.py`. |

## Distributed compute & serving

| Component | Version | Where it's pinned |
| --------- | ------- | ----------------- |
| Ray / Ray Serve (app client) | 2.49.2 | `requirements.txt` (`ray[air,serve]==2.49.2`) |
| Ray (cluster image) | 2.49.2 | `deploy/docker/Dockerfile.ray` (`rayproject/ray:2.49.2-py311`) |
| RayCluster CR `rayVersion` | 2.49.2 | `deploy/helm/ray/raycluster.yaml` |
| RayService CR `rayVersion` | 2.49.2 | `deploy/helm/ray/rayservice.yaml` |
| KubeRay operator | set by `KUBERAY_OPERATOR_VERSION` | `app/infra/ray_manager.py` |
| Daft (getdaft) | `>=0.5.0,<0.6` | `requirements.txt` |

> **Why 2.49.2.** The matrix previously named 2.53.0, a Ray release that does
> not exist — the newest published is 2.49.2. Every consumer agreed with every
> other, so the guard test passed, and `pip install -r requirements.txt` failed
> on every platform with "No matching distribution found for ray==2.53.0". A
> matrix that is internally consistent and externally wrong is the failure mode
> worth watching for here.


> **Java is no longer a prerequisite.** PySpark was removed — schema and DDL
> inference now use PyArrow, Ray Data's engine, so there is no JVM in the
> dependency set (`requirements.txt`: *"pyspark removed: schema/DDL inference
> now uses PyArrow (Ray Data's engine); no JVM."*). Earlier revisions of this
> matrix listed Java 11–17 and Spark 4.0.0+; both rows are gone because neither
> is installed.

> **Why two Postgres drivers.** SQLAlchemy is uncapped, and SQLAlchemy 2.1
> changed which DBAPI a bare `postgresql://` URL selects: psycopg2 on 2.0.x,
> psycopg v3 on 2.1.x. Installing only one makes the install fatal on whichever
> SQLAlchemy pip happens to resolve, so both are listed. Plain `psycopg2` is
> deliberately *not* listed — it has no wheels, so pip would build it from
> source and fail on a clean machine.

> **Ray version must match on both ends.** The app talks to the cluster over the
> Ray Client protocol (`ray://…:10001`), which requires the app-side Ray and the
> cluster-side Ray to be the same major/minor. Keep the app pin, the Ray image
> tag, and both CR `rayVersion` fields aligned. The KubeRay operator must be a
> release that supports the pinned Ray version.

## Agent framework & LLMs

| Component | Version | Where it's pinned |
| --------- | ------- | ----------------- |
| LangChain | 0.3.x | `requirements.txt` |
| LangGraph | 0.6.x | `requirements.txt` |
| LangGraph CLI | 0.4.x | `requirements.txt` |
| MCP SDK | `>=1.18,<2` | `requirements.txt` |
| langchain-openai | `>=0.3,<1.0` | `requirements.txt` (OpenRouter and local models use this client) |
| LLM provider (local / CLI default) | OpenRouter, resolved key-aware | `app/core/inference.py` — unset `INFERENCE_PROVIDER` resolves to OpenRouter, falling back to Groq when only a Groq key is present |
| LLM provider (Helm default) | Groq Cloud | `deploy/helm/avaloka/values.yaml` (`inference.provider: "groq"`), always emitted via `templates/configmap.yaml` |

## Web / API / platform

| Component | Version | Where it's pinned |
| --------- | ------- | ----------------- |
| FastAPI | ~0.116 | `requirements.txt` |
| Uvicorn | ~0.35 | `requirements.txt` |
| Pydantic | 2.7+ | `requirements.txt` |
| Web UI (React, built with Lovable) | see `ui/package.json` | `ui/` |
| Supabase (local, for the UI) | see `ui/` | `ui/` |
| Redis (client) | `>=5.0.4` | `requirements.txt` |
| SQLAlchemy | `>=2.0`, **no upper cap** | `requirements.txt` |
| Postgres driver | `psycopg[binary]>=3.1` **and** `psycopg2-binary>=2.9.0` | `requirements.txt` — both are installed deliberately; see the note below |
| Helm | v3+ | deployment prerequisite |
| Kubernetes | 1.28+ | deployment prerequisite. The kind config pins no node image (`deploy/clusters/kind-cluster.yaml`), so a local cluster gets whatever Kubernetes version your `kind` binary defaults to. |

## ML training & tracking

| Component | Version | Where it's pinned |
| --------- | ------- | ----------------- |
| PyTorch | `>=2.0.0` (no upper cap) | `requirements.txt` |
| torchvision | `>=0.15.0` | `requirements.txt` |
| numpy | `>=1.24.0`; capped `<2` only on macOS x86_64 | `requirements.txt` — older torch wheels were built against numpy 1.x |
| MLflow | `==3.14.0` | `requirements.txt` |
| scikit-learn | `>=1.3.0` | `requirements.txt` |

## Container images

| Image | Published tags that exist on GHCR | Chart default |
| --- | --- | --- |
| `ghcr.io/guruvaidev/avaloka-api` | `main`, `latest`, `sha-<short>`, `src-*`, `deps-*`, `buildcache` | `main` |
| `ghcr.io/guruvaidev/avaloka-ui` | same set | `main` |
| `ghcr.io/guruvaidev/avaloka-ray` | same set | `main` (used by `ui.image`; the Ray CRs name `avaloka-ray:latest` bare) |
| `ghcr.io/guruvaidev/avaloka-functions` | same set | `main` |

All four pull anonymously. **No semver tag exists** — `images.yml` would
publish one on a `v*` git tag, but none has been pushed, so `:1.0.0` / `:1.6.0`
do not resolve. Neither does `:develop-1.6`. Verify before pinning:
`docker manifest inspect ghcr.io/guruvaidev/avaloka-api:<tag>`.

Chart version is `1.6.0` / appVersion `1.6.0` (`deploy/helm/avaloka/Chart.yaml`),
which is independent of the image tags above.

### Third-party images that do not pull

| Image | Status |
| --- | --- |
| `quay.io/minio/minio`, `quay.io/minio/mc` | **Unpullable.** Anonymous manifest requests return `401` at every tag tried, including the pinned releases and `:latest`. Docker Hub returns `401` too. `quay.io/coreos/etcd:v3.5.16` returns `200`, so this is specific to MinIO. `minio.enabled` is `true` by default, so a default `helm install` leaves that pod in `ImagePullBackOff`. |

A SeaweedFS replacement exists on branch `feat/seaweedfs-object-storage` and is
**not merged**; nothing in this tree references SeaweedFS.

## Cloud targets

| Target | Provider module | Status |
| ------ | --------------- | ------ |
| Local (kind) | `app/infra/providers/local_kind.py` | Supported |
| GCP (GKE) | `app/infra/providers/gcp_gke.py` | Supported |
| AWS (EKS) | `app/infra/providers/aws_eks.py` | Supported |
| Azure (AKS) | `app/infra/providers/azure_aks.py` | Module present and resolved by `factory.py`; not exercised by CI |

To bump a version, update the pinning file(s) in the "Where it's pinned" column,
this table, and run the deploy-logic tests that assert cross-file consistency
(see [`testing.md`](testing.md)).
