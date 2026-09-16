# Contributing to Avaloka

Thanks for your interest in improving Avaloka — an agentic team of data
scientists that takes any dataset from raw data to model inference. This guide
explains how to set up your environment, the conventions we follow, and how to
get a change merged.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Project layout](#project-layout)
- [Coding conventions](#coding-conventions)
- [Testing](#testing)
- [Commit and PR conventions](#commit-and-pr-conventions)
- [Licensing of contributions](#licensing-of-contributions)

## Ways to contribute

- **Report bugs** and request features via the issue tracker.
- **Improve documentation** — the `docs/` tree, this file, and inline docstrings.
- **Add a data connector** under `file_handler/` (see the existing CSV/Parquet
  connectors for the interface).
- **Add or harden an agent** under `app/agents/`.
- **Extend the deployment surface** — a new cloud provider under
  `app/infra/providers/`, or Helm chart improvements under `deploy/`.
- **Add tests** — see [`docs/testing.md`](docs/testing.md) for the testing tiers
  and conventions.

## Development setup

### Prerequisites

- **Python 3.11+** (the container images use `python:3.11-slim`; the app and the
  Ray cluster must run matching Ray versions — see [`docs/versions.md`](docs/versions.md)).
- **Java 11–17** on `PATH` for PySpark (`JAVA_HOME` set).
- For deployment work: `docker`, `kubectl`, `helm` v3+, and `kind` for a local
  cluster. The cloud provider code is present but gated: provisioning managed
  clusters is a commercial capability, so `gcloud` / `aws` / `eksctl` / `az` are
  only needed if you are working on those paths against a commercial build.

### Set up the environment

```bash
git clone <repository-url>
cd avaloka-dev
python -m venv agents-env
source agents-env/bin/activate          # Windows: agents-env\Scripts\activate
pip install -r requirements.txt
```

Copy the environment template and fill in your own keys — **never commit real
secrets**:

```bash
cp .env.example .env   # then edit; .env is .gitignore'd
```

At minimum you need `GROQ_API_KEY_PLANNING_AGENT` and `GROQ_API_KEY_CODING_AGENT`
for the agents to run. See the README's *Configuration* section for the full
list.

### Run it locally

```bash
# FastAPI backend (the HTTP API)
uvicorn app.api.server:app --host 0.0.0.0 --port 9000 --reload

# Streamlit UI
streamlit run app/api/streamlit_app.py

# Mission-planning CLI (in-process, no backend required)
python -m app.interfaces.cli.main plan app/sample_data/sales_data.csv \
    --goal "Explain churn drivers" --rows 84000000 --json
```

## Project layout

A short orientation — see [`docs/architecture.md`](docs/architecture.md) for the
full map.

| Area | Path | What lives here |
| ---- | ---- | --------------- |
| Agents | `app/agents/` | Planner, Coder, Validator, Execution, Sampling, Profiling, DTA, MTA v1/v2, Visualization, Scheduler |
| HTTP API | `app/api/` | FastAPI server (`server:app`), LangGraph workflow, schemas |
| Graph state | `app/graph/etl_state.py` | The `ETLState` TypedDict shared across agents |
| Execution routing | `app/execution/` | Router, estimator, environment for local/Ray/k8s selection |
| Missions | `app/missions/` | Mission compiler + schema (the canonical intent) |
| Interfaces | `app/interfaces/` | Mission-planning CLI + MCP server + shared `service.py` |
| Serving | `app/serve/inference.py` | Ray Serve inference app |
| Infra | `app/infra/` | `cluster_bootstrap`, `ray_manager`, `deploy_stack`, `providers/` |
| Deployment | `deploy/` | Helm charts, Dockerfiles, kind config, per-cloud values, Makefile |
| Connectors | `file_handler/` | CSV, Excel, Parquet, Avro, Delta, Iceberg, JSON, XML |
| Tests | `tests/` | Unit + integration; see [`docs/testing.md`](docs/testing.md) |

## Coding conventions

- **Style**: follow the surrounding code. Match its naming, docstring style, and
  comment density rather than importing a different house style into one file.
- **Imports of heavy/optional deps** (ray, torch, daft, pyspark, mcp) should be
  **lazy** — imported inside the function that needs them, not at module load —
  so that importing a module for inspection or unit testing does not require the
  whole ML stack. `app/infra/ray_manager.py` and `app/serve/inference.py` are
  the reference patterns.
- **Shell-outs** go through `app.infra.providers.base.run_command`, which returns
  a structured `{status, step_name, message, details}` outcome. Don't call
  `subprocess` directly in deployment code.
- **Cross-interface invariant**: the CLI, MCP server, and REST surface must call
  the *same* functions in `app/interfaces/service.py` and return the *same* JSON
  shape (`planned_to_dict`). If you touch one surface, keep the others in sync —
  there is a test for this.
- **Determinism/fallbacks**: agents should degrade to safe, explicit stubs when
  an LLM or external service is unavailable, rather than crashing. Make missing
  functionality visible, don't silently fake success.
- **License headers**: new source files should carry the short Apache header (see
  [Licensing of contributions](#licensing-of-contributions)).

## Testing

Avaloka uses `pytest`. The suite mixes fast unit tests with cluster- and
cloud-dependent integration tests, gated by markers so the default run stays
hermetic.

```bash
# Fast, hermetic default (no cluster, no cloud, no real LLM)
pytest -m "not cluster and not cloud and not integration"

# A specific area
pytest tests/test_coder_integration.py -v      # coder (needs GROQ keys)
pytest tests/infra/ -v                          # deployment logic

# Cluster / cloud tiers (see docs/testing.md for the full matrix)
export AVALOKA_TEST_KUBE_CONTEXT=kind-avaloka
pytest -m cluster -v
```

Markers (registered in `pytest.ini`): `cloud`, `integration`, `cluster`,
`kuberay`, `slow`. Tests that need an external service (a live cluster, cloud
credentials, an LLM key, a database) must **skip with a specific reason** when
that service is absent — never hang or fail spuriously. Follow the skip patterns
already used across the suite (`pytest.skip(...)`, `pytest.importorskip(...)`,
`@pytest.mark.skipif(...)`).

**Safety when running cluster tests**: never target a non-`kind`/`minikube`
kube-context for destructive operations without `AVALOKA_TEST_ALLOW_CLOUD=1`.
Cluster tests use the `avaloka-test` namespace, never `default`. See the safety
rails in [`docs/testing.md`](docs/testing.md).

New behavior should come with tests. Bug fixes should come with a regression
test that fails before the fix.

## Commit and PR conventions

- **Branch** off `main` (or the active development branch); open a pull request
  rather than committing directly to `main`.
- **Conventional-commit style** subject lines are used across the history, e.g.
  `feat(k8s): ...`, `fix(mta): ...`, `docs(readme): ...`.
- **Keep concerns separate.** Don't bury a chart rewrite inside a "add tests"
  commit — reviewers should be able to review development and tests
  independently.
- **PR descriptions** should state: what changed, which files carry the real
  regression risk (vs. purely additive files), how you verified it, and the
  **actual** test result (how many passed/skipped and why). Do not describe a
  test run as successful unless it was run and passed.
- **Do not commit** `.env`, credentials, service-account JSON, or generated
  artifacts.

## Licensing of contributions

Avaloka is licensed under the [Apache License 2.0](LICENSE). By submitting a
contribution, you agree that your contribution is licensed under the same terms,
per Section 5 of the license ("Submission of Contributions").

New source files should carry the standard header:

```python
# Copyright 2026 Avaloka.ai Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

If you modify an existing file materially, add a note that you changed it (the
license's redistribution clause 4b) — a line in the PR description is enough.


## Sign your work

Every commit needs a `Signed-off-by` line certifying the
[Developer Certificate of Origin](DCO) — that you wrote the change, or have the
right to contribute it under this project's licence:

```bash
git commit -s -m "your message"
```

which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

Use your real name and a working email. There is **no CLA** — a DCO sign-off is
all we ask, deliberately: it is a statement you can make in one flag rather than
a document you have to get signed.

Forgot on the last commit: `git commit --amend -s`. On several:
`git rebase --signoff HEAD~<n>`.

