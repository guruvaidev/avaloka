# Testing Avaloka

Avaloka uses `pytest`. The suite mixes fast, hermetic unit tests with
integration tests that need real external services (a Kubernetes cluster, cloud
credentials, an LLM key, a database). Markers keep the default run hermetic so
you can iterate quickly and opt into the heavier tiers deliberately.

## Markers

Registered in `pytest.ini`:

| Marker | Meaning |
| ------ | ------- |
| `integration` | Exercises real agents / LLMs. |
| `cluster` | Requires a live Kubernetes cluster. |
| `kuberay` | Requires the KubeRay CRDs installed in the cluster. |
| `cloud` | Requires real cloud credentials. |
| `slow` | Multi-minute runtime. |

## Running the tiers

```bash
# 1. Fast, hermetic default — no cluster, no cloud, no real LLM
pytest -m "not cluster and not cloud and not integration"

# 2. A specific area
pytest tests/test_coder_integration.py -v      # coder (needs GROQ keys)
pytest tests/infra/ -v                          # deployment logic

# 3. Cluster tiers (a local kind cluster must be up)
export AVALOKA_TEST_KUBE_CONTEXT=kind-avaloka
pytest -m cluster -v
pytest -m kuberay -v

# 4. Multi-cloud (opt-in; needs credentials)
AVALOKA_TEST_ALLOW_CLOUD=1 pytest -m cloud -v
```

The test tiers, from cheapest to most expensive:

1. **Deploy-logic** — mocks the shell layer; asserts on rendered Helm manifests
   and command vectors. No cluster. Runs in seconds.
2. **Cluster deployment** — stands the backend up on a live cluster and checks
   health, RBAC, and config wiring.
3. **KubeRay execution** — installs the operator, brings up a RayCluster,
   submits a job, and exercises Ray Serve.
4. **Agent workloads** — coder/planner, Data Transfer, and Model Training flows
   with real prompts.
5. **Multi-cloud matrix** — the cluster + KubeRay + workload tiers parameterized
   over providers (local, GKE, EKS; AKS is on the roadmap), using ephemeral
   clusters that are created and then unconditionally destroyed.

## Skips, not failures

Tests that need an external service must **skip with a specific reason** when
that service is absent — never hang or fail spuriously. Follow the patterns
already used across the suite:

```python
if daft_coder.coder_llm is None:
    pytest.skip("GROQ_API_KEY_CODING_AGENT is required for integration tests.")

daft = pytest.importorskip("daft")

@pytest.mark.skipif(not os.getenv("TEST_PG_DSN"), reason="set TEST_PG_DSN for the live-Postgres test")
```

## Environment variables

| Variable | Purpose |
| -------- | ------- |
| `AVALOKA_TEST_KUBE_CONTEXT` | Kube context for cluster tests (default `kind-avaloka`). |
| `AVALOKA_TEST_ALLOW_CLOUD` | Set to `1` to permit destructive operations against non-local contexts. |
| `AVALOKA_TEST_REBUILD` | Set to `1` to force an image rebuild instead of reusing local images. |
| `GROQ_API_KEY_PLANNING_AGENT` / `GROQ_API_KEY_CODING_AGENT` | LLM keys for agent tests. |

## Safety rails

Cluster tests can create and delete real infrastructure. Non-negotiable:

- **Never** run destructive operations (`helm uninstall`, `kubectl delete`)
  against a non-`kind`/`minikube` context without `AVALOKA_TEST_ALLOW_CLOUD=1`.
  Check your context first: `kubectl config current-context`.
- Cluster tests use a dedicated namespace (`avaloka-test`), never `default`.
- Ephemeral cloud clusters are labelled at creation, destroyed in a `finally`
  block plus an `atexit` fallback, and never deleted unless the harness created
  them — a leaked managed cluster bills continuously.
- Never commit credentials or `.env` files (see [SECURITY.md](../SECURITY.md)).

## Writing tests

- New behavior comes with tests; bug fixes come with a regression test that
  fails before the fix.
- Keep the hermetic default green — gate anything needing a cluster, cloud, LLM,
  or database behind the appropriate marker and a skip guard.
- Reuse the real fixtures and prompts already in `tests/` rather than inventing
  new ones (e.g. `app/sample_data/sales_data.csv` for the coder flow).

---

## Suite reference

The marker-based tiers above are what you want day to day. This section lists
the suites by path, for when you know which area you are changing and want to
run only that.

#### Comprehensive Test Suite

The project includes a comprehensive test suite covering all components with both success and failure scenarios.

#### Model Training Agent (MTA) Tests

```bash
# MTA v1 tests
pytest tests/test_mta_agent.py tests/test_mta_training.py -v

# MTA v1 – specific categories
pytest tests/test_mta_training.py -k "failures" -v         # Validation failure tests
pytest tests/test_mta_training.py::TestPyTorchTraining -v  # PyTorch training tests
pytest tests/test_mta_training.py::TestONNXExport -v       # ONNX export tests

# MTA v2 tests (Ray + GCP inference)
pytest tests/test_mta_v2_unit.py tests/test_mta_v2_training.py tests/test_mta_v2_integration.py -v

# Run with coverage
pytest tests/test_mta_agent.py tests/test_mta_training.py --cov=app.agents.mta -v
```

#### Test Coverage

- ✅ **PyTorch Training** – Model creation, training pipeline, model saving
- ✅ **ONNX Export** – Model export, validation, failure scenarios
- ✅ **MTA v1 Integration** – Agent capabilities, task creation, execution
- ✅ **MTA v2 / Ray** – Ray distributed training, GCP inference, API gateway
- ✅ **Configuration Management** – Config creation, validation failures
- ✅ **Validation Failures** – Comprehensive error handling tests
- ✅ **Resource Cleanup** – Automatic cleanup of temporary files
- ✅ **Asset Persistence** – Cloud write retries, signed URL generation, GitHub registry
- ✅ **File Connectors** – CSV, Excel, Parquet, Avro, Delta, Iceberg, JSON, XML

#### Other Component Tests

```bash
# Sampling agent tests
pytest tests/sampler/sampler_unit_tests.py -v

# Profiling agent
pytest tests/test_profiling_agent_standalone.py -v

# Daft / DTA
pytest tests/test_daft.py tests/test_daft_coder_pipeline.py tests/test_dta_e2e.py -v

# Ray execution
pytest tests/test_ray_unit.py tests/test_ray_integration.py -v

# MCP server
pytest tests/test_mcp_server_integration.py -v

# File connectors
pytest tests/test_connectors.py -v

# Asset persistence
pytest tests/test_wbs06_asset_persistence.py -v

# All unit tests
pytest tests/ -v
```

#### Integration Tests

```bash
# Sampling integration tests
pytest tests/sampler/sampler_integration_tests.py -v

# Infrastructure tests
python -m unittest tests/infra/test_k8s_deployment.py
```

#### End-to-End Tests

#### GCP Tests

**Prerequisites:**

- `gcloud` CLI installed and authenticated
- Docker installed and running
- GCP permissions for GKE cluster creation

**Setup:**

```bash
export GCP_PROJECT_ID="your-gcp-project-id"
export GCP_CLUSTER_NAME="test-gke-cluster"  # Optional
export GCP_ZONE="us-central1-c"             # Optional
gcloud components install gke-gcloud-auth-plugin
```

**Run Test:**

```bash
python -m unittest tests/test_e2e_gke_groupby_sum.py
```

#### AWS Tests

**Prerequisites:**

- `aws` CLI installed and authenticated
- AWS permissions for EKS cluster creation

**Setup:**

```bash
export AWS_REGION="your-aws-region"
export AWS_EKS_CLUSTER_ROLE_ARN="arn:aws:iam::account:role/eks-role"
export AWS_SUBNET_IDS="subnet-xxxxxxxx,subnet-yyyyyyyy"
export AWS_SECURITY_GROUP_IDS="sg-zzzzzzzz"
```

**Run Test:**

```bash
python -m unittest tests/infra/test_eks_deployment.py
```

### Scheduler / Celery

- `pytest tests/test_coder_integration.py`
- `pytest tests/test_scheduler_integration.py` (requires Celery worker + Redis)

To run the scheduler tests end-to-end:

1. Export valid Groq keys so the real planner and coder can run:

   ```bash
   export GROQ_API_KEY_PLANNING_AGENT="your-planning-key"
   export GROQ_API_KEY_CODING_AGENT="your-coding-key"
   ```

2. Start Redis in Docker:

   ```bash
   docker compose -f docker-compose.scheduler.yml up redis -d
   ```

3. In another terminal, start the Celery worker:

   ```bash
   docker compose -f docker-compose.scheduler.yml up celery-worker
   ```

4. In another terminal, start the RedBeat scheduler:

   ```bash
   docker compose -f docker-compose.scheduler.yml up celery-redbeat-worker
   ```

5. Run `pytest tests/test_scheduler_integration.py`. The tests use sample datasets from `app/sample_data` and will schedule a real periodic task before asserting the scheduler node was invoked.

### Preparing Kaggle datasets

End-to-end Kaggle tests require the datasets to be present under `app/sample_data`. After installing and configuring the Kaggle CLI (`~/.kaggle/kaggle.json`), download the datasets once with:

```bash
python tests/download_kaggle_datasets.py            # download everything
python tests/download_kaggle_datasets.py --filter airline  # download a subset
```

The script is idempotent and skips datasets that already exist locally.

### Running Tests

#### Fast Feedback

- `pytest tests/test_coder_integration.py`
- `pytest tests/test_planner_new.py`

#### Infrastructure Suites

- **GCP** – `python -m unittest tests/test_e2e_gke_groupby_sum.py`
- **AWS** – `python -m unittest tests/infra/test_eks_deployment.py`

#### Kaggle Workflow

1. Download datasets (see above).
2. Run `pytest -s tests/test_e2e_kaggle.py` to see per-dataset progress logs.
