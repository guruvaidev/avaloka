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
| `kaggle` | Validates the optional downloaded Kaggle CSVs. |
| `defect` | Asserts intended behaviour against a confirmed defect; always paired with `xfail(strict=True)`. |
| `single` | Prompt suites — short conversational prompts (`Tier.SINGLE`). |
| `multi` | Prompt suites — fully specified multi-clause prompts (`Tier.MULTI`). |
| `transfer` | Prompt suites — transfer prompts routed to the DTA (`Tier.TRANSFER`). |
| `operations` | Prompt suites — transformation families with a result-table check (`Tier.OPERATION`). |

## Running the tiers

```bash
# 1. Fast, hermetic default — no cluster, no cloud, no real LLM
pytest -m "not cluster and not cloud and not integration"

# 1b. What `./scripts/ci.sh --fast` actually runs: the above, minus the
#     multi-minute and download-dependent tests. This is the one to copy.
pytest -m "not cluster and not cloud and not integration and not slow and not kaggle"

# 2. A specific area
pytest tests/test_coder_integration.py -v      # coder (marked `integration`; needs a model key)
pytest tests/infra/ -v                          # deployment logic

# 3. Cluster tiers (a local kind cluster must be up)
export AVALOKA_TEST_KUBE_CONTEXT=kind-avaloka
pytest -m cluster -v
pytest -m kuberay -v

# 4. Multi-cloud (opt-in; needs credentials)
AVALOKA_TEST_ALLOW_CLOUD=1 pytest -m cloud -v
```

Rather than assembling marker expressions by hand, prefer the gate script,
which is what CI runs:

```bash
./scripts/ci.sh --list       # the five stages
./scripts/ci.sh --fast       # stages 1-4, no benchmark
./scripts/ci.sh              # the full gate
```

Its exit code is the number of failed stages, and a stage that is skipped says
so rather than passing silently. Note that the data-science stage is skipped on
`oss/1.6`, where `tests/datascience/` is deliberately absent.

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
   over providers, using ephemeral clusters that are created and then
   unconditionally destroyed. Provider modules exist for GKE, EKS and AKS
   (`app/infra/providers/`); only local kind is exercised in CI.
6. **Live provider** — the only tier that spends real credit and the only one
   that can see how a provider actually behaves. See below.

## The live tier

The hermetic gate runs with no API key at all. That is deliberate — it is what
a contributor or a fork PR can run — but it cannot observe a provider's
behaviour, and that blind spot has cost real breakage: `tool_choice` was
`"required"` for every provider, OpenRouter answers that with
`finish_reason="error"` and no tool call, and every planner turn returned a
generic failure. Nothing hermetic caught it. The fix is
`tool_choice_for_provider()` in `app/core/inference.py`, which is keyed on the
provider because OpenRouter needs `auto` while Ollama needs `required`.

Two scheduled workflows cover it:

| Workflow | Schedule | What it exercises |
| --- | --- | --- |
| `.github/workflows/live-provider-weekly.yml` | Mondays 06:00 UTC, plus `workflow_dispatch` | A hosted provider — OpenRouter, with `INFERENCE_PROVIDER=openrouter` and the `OPENROUTER_API_KEY` secret. Weekly because it costs credit and the drift it catches is a provider changing behaviour. |
| `.github/workflows/live-local-model-nightly.yml` | 03:30 UTC nightly, plus `workflow_dispatch` | A local Ollama model (default `gemma4:e4b`; the model must support tools) — no hosted provider, no spend. |

Both run `python -m pytest tests/live/`. Run it yourself with a key set:

```bash
OPENROUTER_API_KEY=sk-or-… INFERENCE_PROVIDER=openrouter pytest tests/live/ -v
```

`tests/live/test_live_provider_smoke.py` is marked `integration`, so the
hermetic default excludes it, and it skips itself when none of
`OPENROUTER_API_KEY`, `INFERENCE_LOCAL_BASE_URL` or a Groq key is set. The
weekly workflow deliberately skips the whole *job* when the key is absent
rather than letting the suite skip every test and report a green tick that
measured nothing — worth copying if you wire this into your own CI.

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
| `OPENROUTER_API_KEY` | Arms the live provider tier (`tests/live/`). |
| `INFERENCE_PROVIDER` | Pins the provider. Leave unset for the key-aware default in `app/core/inference.py`. |
| `INFERENCE_LOCAL_BASE_URL` | Points the live tier at a local OpenAI-compatible server (Ollama, vLLM). |
| `TEST_PG_DSN` | Enables the live-Postgres test. |

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
  new ones. `app/sample_data/` holds the shipped CSVs — `salaries.csv` is the
  smallest and is what the docs use as the worked example.

---

## Suite reference

The marker-based tiers above are what you want day to day. This section lists
the suites by path, for when you know which area you are changing and want to
run only that.

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

#### Other Component Tests

```bash
# Sampling agent tests
pytest tests/sampler/sampler_unit_tests.py -v

# Profiling agent
pytest tests/test_profiling_agent_standalone.py -v

# Daft / DTA  (there is no tests/test_daft.py or tests/test_dta_e2e.py;
# the DTA suite is ~30 files named tests/test_dta_*.py)
pytest tests/test_daft_coder_pipeline.py tests/test_daft_retriever.py -v
pytest tests/test_dta_end_to_end.py -v
pytest tests/ -k "dta" -v          # the whole DTA suite

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

#### GCP Tests

> **These create a real GKE cluster and bill your project.** They are not
> gated by the edition check — that gate is in the open-source overlay's
> `factory.py`, not on a development checkout. Confirm your context and your
> intent before running them, and confirm the cluster is gone afterwards.

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
   docker compose -f deploy/compose/docker-compose.scheduler.yml up redis -d
   ```

3. In another terminal, start the Celery worker:

   ```bash
   docker compose -f deploy/compose/docker-compose.scheduler.yml up celery-worker
   ```

4. In another terminal, start the RedBeat scheduler:

   ```bash
   docker compose -f deploy/compose/docker-compose.scheduler.yml up celery-redbeat-worker
   ```

5. Run `pytest tests/test_scheduler_integration.py`. The tests use sample datasets from `app/sample_data` and will schedule a real periodic task before asserting the scheduler node was invoked.

### Preparing Kaggle datasets

End-to-end Kaggle tests require the datasets to be present under `app/sample_data`. After installing and configuring the Kaggle CLI (`~/.kaggle/kaggle.json`), download the datasets once with:

```bash
python tests/download_kaggle_datasets.py            # download everything
python tests/download_kaggle_datasets.py --filter airline  # download a subset
```

The script is idempotent and skips datasets that already exist locally.

### Kaggle workflow

1. Download the datasets (see above).
2. Run `pytest -s tests/test_e2e_kaggle.py` to see per-dataset progress logs.
   These carry the `kaggle` marker, so `./scripts/ci.sh --fast` excludes them.
