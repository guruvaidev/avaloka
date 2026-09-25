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
| Python | 3.11+ | `deploy/docker/Dockerfile.avaloka` (`python:3.11-slim`); env markers in `requirements.txt` |
| Java (for PySpark) | 11–17 | runtime prerequisite |

## Distributed compute & serving

| Component | Version | Where it's pinned |
| --------- | ------- | ----------------- |
| Ray / Ray Serve (app client) | 2.58.0 | `requirements.txt` (`ray[air,serve]==2.58.0`) |
| Ray (cluster image) | 2.58.0 | `deploy/docker/Dockerfile.ray` (`rayproject/ray:2.58.0-py311`) |
| RayCluster CR `rayVersion` | 2.58.0 | `deploy/helm/ray/raycluster.yaml` |
| RayService CR `rayVersion` | 2.58.0 | `deploy/helm/ray/rayservice.yaml` |
| KubeRay operator | set by `KUBERAY_OPERATOR_VERSION` | `app/infra/ray_manager.py` |
| Apache Spark (PySpark) | 4.0.0+ | `requirements.txt` |
| Daft (getdaft) | 0.5–0.8 range | `requirements.txt` |

> **Why 2.58.0.** Every consumer must agree: the pip client, the cluster
> image, both CR `rayVersion` fields, and the two agent Dockerfiles. Ray refuses
> a client/cluster version mismatch at connect time, and nothing in the hermetic
> CI selection can catch it because CI never starts a cluster --
> `tests/k8s/test_t1_deploy_logic.py` has the guard, and `tests/k8s/` is marked
> integration. Dependabot bumped `requirements.txt` alone in PR #4 and split the
> matrix; this realigns the other six references upward rather than reverting,
> since 2.58.0 is published on PyPI and `rayproject/ray:2.58.0-py311` exists.


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
| MCP SDK | 1.9.4 | `requirements.txt` |
| LLM provider | Groq Cloud | runtime (API keys) |

## Web / API / platform

| Component | Version | Where it's pinned |
| --------- | ------- | ----------------- |
| FastAPI | ~0.116 | `requirements.txt` |
| Uvicorn | ~0.35 | `requirements.txt` |
| Pydantic | 2.7+ | `requirements.txt` |
| Web UI (React, built with Lovable) | see `ui/package.json` | `ui/` |
| Supabase (local, for the UI) | see `ui/` | `ui/` |
| Redis (client) | 5.0.4+ | `requirements.txt` |
| Helm | v3+ | deployment prerequisite |
| Kubernetes | 1.28+ (tested on kind 1.36) | deployment prerequisite |

## ML training & tracking

| Component | Version | Where it's pinned |
| --------- | ------- | ----------------- |
| PyTorch | 2.11.0 | `requirements.txt` |
| torchvision | 0.15.0+ | `requirements.txt` |
| MLflow | see `requirements.txt` | `requirements.txt` |
| scikit-learn | see `requirements.txt` | `requirements.txt` |

## Cloud targets

| Target | Provider module | Status |
| ------ | --------------- | ------ |
| Local (kind) | `app/infra/providers/local_kind.py` | Supported |
| GCP (GKE) | `app/infra/providers/gcp_gke.py` | Supported |
| AWS (EKS) | `app/infra/providers/aws_eks.py` | Supported |
| Azure (AKS) | — | Roadmap |

To bump a version, update the pinning file(s) in the "Where it's pinned" column,
this table, and run the deploy-logic tests that assert cross-file consistency
(see [`testing.md`](testing.md)).
