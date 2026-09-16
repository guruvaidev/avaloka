# Avaloka on Kubernetes — integration tests

Test tiers T1–T5 for the avaloka API-on-Kubernetes deployment. Tiers are additive
and gated by pytest markers + environment so the default CI run stays fast and
never touches a cloud cluster.

## Tiers

| File | Marker(s) | Needs | Runtime |
|---|---|---|---|
| `test_t1_deploy_logic.py` | *(none)* | helm (for lint/template tests) | < 30 s |
| `test_t4_cli.py` | *(none)* | — (remote-parity test needs `AVALOKA_API_URL`) | < 5 s |
| `test_t2_cluster_deployment.py` | `cluster`, `slow` | kind cluster + built images | 10–25 min |
| `test_t3_kuberay_execution.py` | `cluster`, `kuberay`, `slow` | KubeRay + RayCluster/RayService | 10–20 min |
| `test_t4_agent_workloads.py` | `integration` (+ `cluster` for DTA seam) | GROQ keys; deployed API for k8s variants | varies |
| `test_t5_multicloud.py` | `cloud`, `slow` | cloud creds + ephemeral clusters | 25–45 min/leg |

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `AVALOKA_TEST_KUBE_CONTEXT` | `kind-avaloka` | kube-context the cluster tiers use |
| `AVALOKA_TEST_NAMESPACE` | `avaloka-test` | namespace (never `default`) |
| `AVALOKA_TEST_ALLOW_CLOUD` | *(unset)* | **must be `1`** to allow any destructive op against a non-kind context, and to run T5 |
| `AVALOKA_TEST_REBUILD` | *(unset)* | force image rebuild in T2 setup |
| `AVALOKA_API_URL` / `AVALOKA_API_TOKEN` | *(unset)* | deployed API base URL + HS256 bearer for remote-parity / deployed-agent tests |
| `GROQ_API_KEY_PLANNING_AGENT` / `_CODING_AGENT` | *(unset)* | real agents; unset → T4 agent tests skip |
| `SUPABASE_JWT_SECRET` | *(unset)* | the secret the API validates JWTs against (auth tests) |
| `INFERENCE_BACKEND` | `rayserve` | `rayserve` (in-cluster RayService) or `gateway` (legacy GCP) |
| `GCP_PROJECT_ID`, `AWS_REGION`, `AZURE_RESOURCE_GROUP` | — | per-cloud creds for T5 legs |

## Running

```bash
# Default CI — fast, no cluster, no cloud:
pytest tests/ -m "not cluster and not cloud"

# T1 deploy logic (no cluster):
pytest tests/k8s/test_t1_deploy_logic.py -v

# Build + side-load images, then the cluster tiers on kind:
python -c "from app.infra import deploy_stack; deploy_stack.build_images(load_into_kind=True)"
export AVALOKA_TEST_KUBE_CONTEXT=kind-avaloka
pytest tests/k8s/test_t2_cluster_deployment.py -v -m cluster
pytest tests/k8s/test_t3_kuberay_execution.py  -v -m kuberay

# Agent workloads (needs GROQ keys; deployed variants need AVALOKA_API_URL):
pytest tests/k8s/test_t4_agent_workloads.py tests/k8s/test_t4_cli.py -v -m integration

# Multi-cloud (billable, opt-in):
AVALOKA_TEST_ALLOW_CLOUD=1 pytest tests/k8s/test_t5_multicloud.py -v -m cloud
```

## Teardown / orphan reaping

- T2/T3 tear down their namespace + releases in module finalizers (even on failure),
  **kind-context only** unless `AVALOKA_TEST_ALLOW_CLOUD=1`.
- Ephemeral cloud clusters (T5) are destroyed in a `finally` **and** an `atexit`
  fallback. Reap orphans (labelled `purpose=avaloka-integration-test`):

  ```bash
  python -m tests.k8s.helpers.ephemeral --reap-orphans --provider gcp
  ```

  The reaper only ever deletes clusters carrying that label/tag.

## Safety rails

- Your kubeconfig may carry contexts for live clusters. The cluster tiers refuse
  to run any destructive step against a non-kind context unless
  `AVALOKA_TEST_ALLOW_CLOUD=1` — the guard assumes the worst about what else
  your kubeconfig can reach, which is the right assumption.
- All cluster tests use namespace `avaloka-test`, never `default`.

## T6 — Web UI + local Supabase (the complete application)

`test_t6_ui_deployment.py` proves the full app deploys on kind:
- T6.1 (no cluster): chart renders `webui` + the Supabase stack; UI build artifacts exist
- T6.1b (no cluster): the chart's migration bundle matches `ui/supabase/migrations`
- T6.2 (`cluster`): the UI pod serves the SPA and `/env-config.js` (runtime config)
- T6.3 (`cluster`): the UI's nginx reverse-proxies `/health` to the in-cluster API
- T6.4 (`cluster`): local Supabase (db/auth/rest/kong) is up, gotrue `/auth/v1/health`
  answers via kong, and the migrations Job succeeded

Helm's `.Files.Glob` cannot read outside the chart directory, so the chart ships its
own copy of the migrations. Nothing keeps it in sync — re-copy after Lovable adds any
migration, or T6.1b fails:

```bash
cp ui/supabase/migrations/*.sql deploy/helm/avaloka/files/supabase-migrations/
```

Deploy the complete app on kind:
```bash
# build + load the UI image (backend/ray images as in T2/T3)
docker build -f ui/Dockerfile -t avaloka-ui:latest ui
kind load docker-image avaloka-ui:latest --name avaloka
helm upgrade --install avaloka deploy/helm/avaloka -n avaloka-test \
  --set webui.enabled=true --set supabase.enabled=true --wait
# UI at http://localhost:30090 ; Supabase gateway at http://localhost:30091
pytest tests/k8s/test_t6_ui_deployment.py -v -m cluster
```
