# Avaloka

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](docs/versions.md)
[![Ray](https://img.shields.io/badge/runs%20on-Ray%20%2F%20KubeRay-028CF0.svg)](https://www.ray.io/)

**A team of specialist AI agents that takes a dataset from raw rows to a served
model — and checks its own work before it reports a result.**

In 2007 Jim Gray described a fourth paradigm of science, after the empirical,
the theoretical and the computational: discovery by exploring data at a scale
no one person can hold in their head. His argument was not really about data.
It was about instruments. Scientists were spending their time on the plumbing
of capturing, cleaning, moving and analysing data instead of on their questions.

Two decades on, the plumbing still stands between a curious person and an
answer. Avaloka is an attempt at the instrument. You ask a question in plain
language. A team of agents — planner, profiler, coder, validator, executor,
trainer, visualizer — samples the data, plans the analysis, writes and validates
the Python, runs it on your laptop or a Ray cluster, trains and serves a model
if you need one, and records every artifact along the way.

An instrument is only as good as its calibration, so three of those agents
exist to check the work of the others. Avaloka refuses to train on leaked data,
makes every model beat a trivial baseline, and checks the prose it writes
against the numbers it computed. Where the evidence is thin, it
[says so transparently](docs/test-reports/three-pillar-coverage.md), and we
constantly and consistently improve the data science agents.

## What makes Avaloka different

- **Yours, wherever it runs.** The same Helm chart runs on GKE, EKS, AKS,
  on-premises Kubernetes or your laptop. There is no lock-in and no cloud
  markup, and Avaloka operates no service that receives your data. Choose a
  local model and nothing leaves your environment.
- **Bring your own model, or several.** Planning, coding and visualization can
  each use a different model and provider, with an automatic fallback. Swap or
  upgrade models without changing how you work.
  [More below.](#bring-your-own-model)
- **Insight the moment data lands.** Upload a file, and Auto Insights profiles
  it, picks the right charts and the most informative columns, and suggests
  next steps before you ask a question.
- **A team of specialists.** Separate agents plan, write code, move data, train
  models, visualize and verify, instead of one model doing everything.
- **Checked before it is shown.** Generated code is validated before it runs,
  a trained model must beat a trivial baseline before it is called a success,
  and the numbers in each answer are checked against the results that were
  actually computed.
- **Every result has a trail.** Analyses live in a lineage graph and a layered
  memory, not in loose notebooks. Derived datasets and models trace back to
  their source data and the analysis that produced them.
- **Built for teams.** Share projects, analyses and reports with view, comment
  or edit access, and build on each other's work.
- **From question to production.** Turn an analysis into a scheduled job in one
  step. Large sources run on autoscaling Ray clusters in your own cloud, and
  Avaloka tells you up front when a job will be large or slow, before the bill
  arrives.
- **Connects to what you already have.** PostgreSQL, MySQL, MariaDB, SQL
  Server, Oracle, SQLite, MongoDB, S3, Google Cloud Storage and Azure Blob.
- **A laptop is enough.** The CLI runs a statistics-grounded analysis on your
  own machine, fully offline.

🚀 **[Quickstart](#quickstart)** · 📦 **[Install Guide](docs/INSTALL.md)** · 📖 **[Docs](docs/DOCS_MAP.md)** · 📊 **[Benchmarks](docs/benchmarks.md)** · ✍️ **[Blog](https://avaloka.ai/blog)**

---

## Quickstart

**Two commands, no API key, no cloud account.** You need Python 3.11 or 3.12 on
macOS or Linux.

```bash
git clone https://github.com/guruvaidev/avaloka.git && cd avaloka
./scripts/install.sh && source .venv/bin/activate     # ~5–10 minutes; prints "Everything checks out"
```

Ask your first question. The repo ships sample datasets, so this works on a
fresh clone:

```bash
avaloka analyze app/sample_data/healthcare_dataset.csv \
        --goal "average billing amount by medical condition"
```

Avaloka profiles the data, writes and validates the analysis, runs it, and
prints a verdict. The answer is in the terminal, and a folder under
`avaloka-analysis/` holds everything you need to open, rerun or share it:

| File | What it is |
| --- | --- |
| `executive_report.html` | the decision summary — open it in a browser |
| `README.md` | findings, assumptions and limitations |
| `analysis.py` · `analysis.ipynb` | code that reproduces the run exactly |
| `goal_analysis.json` | the full answer as data — every group, to the dollar |
| `lineage.json` · `data_quality.json` | where every number came from |

Then try the other commands:

```bash
avaloka train app/sample_data/yellow_tripdata_2015-01_dataset_50k.csv --target payment_type   # ~2 min
avaloka chat  app/sample_data/healthcare_dataset.csv     # numbered next steps; answer with a number
avaloka coordinate app/sample_data/healthcare_dataset.csv --workers 2 \
        --goal "average billing amount by medical condition"   # two copies of Avaloka, one answer
```

`train` fits several candidate models, checks the winner beats a trivial
baseline, and writes a validated, deployable model to `avaloka-model/`.
`coordinate` is the swarm: Avaloka clones herself, each copy works half of the
data with its own agents, and the main Avaloka merges what they found into one
profile and one verdict (`avaloka-fleet/coordination.json`). Your own files
work the same way — CSV, TSV, Parquet, Excel, JSON, XML, ORC and Avro.

> **Your data stays on your machine.** Everything above runs locally, with no
> model and no key. Add a model only when you want conversation and richer
> narration, and pick any provider, or a local one: see
> [Bring your own model](#bring-your-own-model).

**Want the full platform** — the web UI, the API, Ray and Kubernetes? Follow the
[Install Guide](docs/INSTALL.md); it also covers
[running completely offline](docs/INSTALL.md#running-avaloka-completely-offline).
If something does not work, run `python scripts/doctor.py` first, then see
[what goes wrong on a first run](docs/INSTALL.md#first-run-what-to-expect-and-what-goes-wrong).
The full CLI reference is [README_CLI.md](README_CLI.md).

---

## Bring your own model

Avaloka is model-agnostic. It is built around the work: the data, the plan,
the checks. No single model sits at its centre, so you choose who does the
thinking and can change your mind later.

- **Platform (web UI and API).** Groq, OpenRouter (which reaches Claude, GPT,
  Gemini, Llama, Qwen and many more), any OpenAI-compatible endpoint, AWS
  Bedrock, Google Vertex AI (Gemini), Azure OpenAI, or a local vLLM or Ollama
  model. Set `INFERENCE_PROVIDER`. To give planning, coding and visualization
  different models, use `INFERENCE_PROVIDER_PLANNING`, `_CODING` and `_VIZ`. To
  change a single agent, use `AVALOKA_<AGENT>_PROVIDER`. With
  `OPENROUTER_API_KEY` set, a failing provider falls back to OpenRouter
  automatically.
- **Your data stays where you put it.** With `INFERENCE_PROVIDER=local`, every
  prompt goes to a model on your own hardware, and nothing leaves your
  environment.
- **CLI.** The quickstart above needs no model at all. In 1.6, `--llm` adds
  conversational phrasing through Claude (`ANTHROPIC_API_KEY`). In 2.0 the CLI
  takes the same choice of providers as the platform.

## Why a model alone does not solve this

A frontier model will read your spreadsheet and give you a good answer about
that spreadsheet. The facts you actually want are rarely in one spreadsheet:
they are spread across a database, a warehouse and an object store, and the
question is the join between them.

But the harder problem is not reach. The failure that destroys credibility is
not a crash. It is a fluent, well-formatted, wrong answer.

We know the shape of that failure because we shipped it. Given a dataset of
pure noise, an earlier version of Avaloka's training path returned verdict
`pass` and `max_safe_deployment_level 4`: it certified a model that had learned
nothing as production-ready. Our own benchmark scored 1.00 before *and* after
the fix, because no task in it carried the shape that triggers the bug.

No model choice fixes that. A model has no privileged access to whether its own
answer is true. What fixes it is a harness that computes its own baseline, and
a benchmark that can fail.

> *A benchmark that cannot fail is a statement of intent, not a measurement.*

[How the harness works](docs/architecture.md) · [What the benchmark covers](docs/benchmarks.md)

---

## The team

Twelve roles, one shared state object. The **Planner** is the team lead: it
talks to you and decides the next step. Everything else is a specialist it
delegates to.

| | Agent | What it owns |
| --- | --- | --- |
| **Lead** | Planner | Converses, plans, routes — keep planning, provision, code, train, schedule, execute |
| **Look** | Sampling · Profiling | Samples and per-column stats; infers the domain, ranks predictive power, flags quality |
| **Write** | Coder | Numbered pseudocode first, then Python that follows it |
| **Check** | Validator | Three layers — syntax, schema-aware static, logical — before anything runs |
| **Run** | Execution | Locally, as a Kubernetes Job, or on a Ray cluster |
| **Move** | Data Transfer | Daft ETL across CSV, Parquet, Avro, Delta, Iceberg, JSON, XML, Excel |
| **Train** | Model Training | PyTorch, scikit-learn, ONNX and MLflow; distributed on Ray, with a model registry |
| **Show** | Visualization | Charts and a report from the results |
| **Judge** | Integrity · Evaluation · Claim Verifier | The evidence layer, below |

The full roster — Planner Graph, Summarizer, Scheduler, Infrastructure, the
conversational layer — is in [Architecture §2](docs/architecture.md).

### The evidence layer

Three checks stand between a result and you:

- **Integrity** refuses to train on leaked targets or duplicated splits.
- **Evaluation** makes every model beat a trivial baseline, measured on the
  **confidence interval, not the mean**, so a majority-class classifier cannot
  pass on floating-point noise.
- **Claim Verifier** checks the written summary against the computed numbers.
  It is **deterministic, with no LLM involved**, which is why the check cannot
  itself hallucinate.

The Validator asks *will this code run*. The Claim Verifier asks *is what we
just told you true*. They are different questions, and they need different
machinery.

---

## What's new in 1.6

1.6 is about **reading real-world data correctly, and being honest when it
can't.** The full list is in the [CHANGELOG](CHANGELOG.md); the stories behind
it are on the [blog](https://avaloka.ai/blog).

- **Spreadsheets as people actually make them.** Avaloka finds the real header
  under title rows and notes, joins two-row headers, and does not count
  subtotal or total rows as data. It tells you what it assumed.
- **More formats, fewer surprises.** JSON, XML, ORC and Avro join CSV, TSV,
  Parquet and Excel. A file with no single sensible reading is refused with a
  clear message instead of being quietly flattened. XML with a DTD is rejected.
- **No confident wrong answers.** If your question names a measure Avaloka
  cannot find, it reports `limited` and lists the candidate columns, instead of
  answering with a row count. `None`, `NA` and `N/A` in a text column stay
  categories, not missing values.
- **Stricter self-checks.** A validator verdict of `"false"` is never read as
  approval, a review that could not be built is never reported as approved, a
  model that does not beat its baseline is never announced as a success, and a
  wrong "lowest" or "highest" in the summary is caught against the numbers.
  Kubernetes and local execution now record results identically.
- **A benchmark that can fail.** New tasks grade whether the data was
  *understood*, not just whether the pipeline ran. Known defects are declared
  openly, and any new failure fails CI.
- **In the product:** notifications, attaching datasets to an analysis thread,
  fixed chat timeouts, and clearer narration and charts.
- **A safer install:** pandas is pinned below 3 until the codebase supports its
  new string type.
- **Ray jobs you can watch.** A Ray jobs panel lists every job with its status,
  and scheduled analyses that are not Ray jobs run in their own executor pod.
- **Auto Insights across sheets.** A workbook with several sheets, or several
  datasets in one analysis, gets one set of insights instead of a flood of
  bullet points and unwanted charts.
- **Database connections are authenticated.** The MCP server's onboarding and
  admin routes now require a key or the API's service token, and only the
  owner's customers are visible. **Self-hosters must set `MCP_SERVICE_TOKEN`**
  (the same value on the API and the MCP server) and should rotate the
  database connection strings stored there. See the advisory in
  [SECURITY.md](SECURITY.md).

## Coming in 2.0

The 1.7 development line ships publicly as **2.0**. In development now — not
in this release. Follow the [blog](https://avaloka.ai/blog)
for progress.

- **The swarm, in every edition.** A mission clones Avaloka into sub-Avalokas
  that each work a shard of the same data and converge on one answer; the
  open-source edition runs two clones per mission, sized for a laptop. Each
  clone's steps, outputs and hand-offs are recorded separately and served by a
  local API, so the UI can show the copies at work, and the record can later
  train the agents themselves.
- **A planner graph of what actually ran.** Not a generic plan → code → test
  loop: the loads, filters, group-bys, joins, aggregates, training and
  evaluation steps that were executed on your prompt, with the columns and
  metrics involved.
- **Nested and semi-structured data.** JSON Lines and NDJSON, nested fields
  flattened into stable dotted columns, and JSON-encoded columns expanded
  automatically.
- **MongoDB as a source.** Read-only, with bounded filter, projection and limit,
  and credentials kept out of the lineage record.
- **Evidence-driven reasoning.** The agent inspects the schema before querying,
  checks denominators, join cardinality, units, time windows and aggregation
  grain, and will not accept a final answer without evidence from a data tool.
- **Hard budgets.** Every mission is bounded in iterations, time, cost and
  tokens, so a run that is not converging stops cleanly.
- **External benchmarks.** A DataAgentBench runner, plus tracks for adaptive
  sampling, swarm ablation and recovery under injected faults.
- **Answers that fit the person asking.** Avaloka infers how expert you are and
  pitches the answer accordingly: the strategy for a CIO, the detail for an
  analyst.
- **A statistics specialist and a workload optimizer.** A dedicated statistical
  agent, and a planner that sizes a job before it runs.
- **Fully offline and air-gapped.** Local models end to end, with every
  third-party call switched off.

---

## What is proven, and where it is thin

We would rather publish the gaps than claim three pillars and evidence one.
Everything in this table is something you can run yourself.

| Claim | Evidence | Run it |
| --- | --- | --- |
| **It installs cleanly** | No compiler and no API key needed; checked on a bare `python:3.12-slim` image | `python scripts/doctor.py` |
| **The code is tested** | The hermetic test gate runs on every pull request, on Python 3.11 and 3.12, with no network and no services | `./scripts/ci.sh --fast` |
| **Answers are graded** | Deterministic benchmark, no LLM: **39/50** pass. The 11 that fail are declared [known defects](avaloka/benchmark/known_defects.py), and any *new* failure fails CI | `python -m avaloka.benchmark run` |
| **The evidence layer works** | 16 cases — leakage, duplicate splits, identifier-like features, the mandatory baseline, invented numbers, the scorer's own resistance to gaming — executed against a live deployment, mostly passing | [coverage report](docs/test-reports/three-pillar-coverage.md) |
| **Lineage is real, but thinly tested** | 2 cases pass: ancestry recorded across derivations, and PII located *through* lineage rather than by re-scanning. The store exists; the coverage is 2 cases, not 20 | [coverage report](docs/test-reports/three-pillar-coverage.md) |
| **Memory persists** | Recall survives a pod restart (`3.4s` cold, `0.6s` warm). It carries context forward; we do not yet claim it makes analyses measurably better | [coverage report](docs/test-reports/three-pillar-coverage.md) |

The honest summary: Avaloka today is a **strong harness, a real lineage graph,
and a working learning loop whose value is still ahead of it**. The
[test reports](docs/test-reports/) hold the per-case evidence for each row,
including which cases are blocked and on what.

**The benchmark number is a floor, not a leaderboard.** It scored 1.00 before
*and* after five real defects were fixed, because no task carried the shape
that triggers them. What it does **not** measure is in
[docs/benchmarks.md](docs/benchmarks.md); no external-benchmark harness is
committed to this tree yet.

**The gate** is `pytest -m "not cluster and not cloud and not integration"`:
unit and contract tests, no API key, no services, so a contributor or a fork
pull request can actually run it. End-to-end, integration and Kubernetes tests
need Redis, a server and cloud credentials and sit outside it
([Testing](docs/testing.md)). Two scheduled workflows call real providers and a
local model server on their own cadence, so a provider outage cannot redden a
pull request.

**Known limitations in 1.6:**

- In a multi-sheet workbook, the upload path reads the longest sheet, not
  necessarily the one your question is about. Upload that sheet on its own for now.
- The CLI's spreadsheet reader handles fewer layouts than the upload path.

Both are tracked in the [CHANGELOG](CHANGELOG.md).

---

## Run it your way

| | |
| --- | --- |
| **On your laptop** | The [Quickstart](#quickstart) above. |
| **On your own cluster, fully offline** | [Install Guide → Running completely offline](docs/INSTALL.md#running-avaloka-completely-offline): a local model server, the embedding model baked into the image, and storage, memory and auth deployed by the Helm chart. Nothing calls home, and a `--network none` check lets you prove it rather than trust it. |
| **Let us run it** | [avaloka.ai](https://avaloka.ai) hosts the Professional and Enterprise editions with auto-scaling compute — nothing to stand up. Enterprise adds team workspaces, the cloud scheduler and connections to your own clusters. |

The open-source edition is not a limited trial of the hosted ones. It is the
same engine on your own hardware, and it stays useful with no account, no key
and no network.

---

## Built on proven open source

Avaloka is an orchestration layer, not a reimplementation. Compute, storage,
vector search, experiment tracking and the database are done by projects that
were load-bearing in production long before Avaloka existed. What Avaloka adds
is the agent team, the evidence layer and the conversation.

| Layer | Built on |
| --- | --- |
| Distributed compute | Ray + KubeRay, Arrow, Daft |
| Storage & formats | PostgreSQL, S3-compatible object storage, Parquet, Delta, Iceberg, Avro |
| Vector & memory | Chroma, Milvus, Redis |
| ML & tracking | PyTorch, scikit-learn, ONNX, MLflow |
| Agent runtime | LangGraph / LangChain, Model Context Protocol |
| Platform | Kubernetes, Helm, FastAPI, Pydantic, Celery, Supabase |
| Data engine | pandas, NumPy, SQLAlchemy, SQLite |

None of it is hidden behind our abstraction: you can verify any number Avaloka
reports by running the layer underneath it yourself.

> **Before you deploy:** the chart's default object store is SeaweedFS, pulled
> from a single Docker Hub repository and pinned by digest. It replaced MinIO
> after MinIO's images stopped being pullable, and a digest does not protect
> against that happening again. Mirror it into a registry you control, and read
> [Deployment](docs/deployment.md) first.

Full per-project attribution and licences are in [NOTICE](NOTICE). Our thanks
to all of those communities, and to [Groq](https://groq.com/) for the inference
speed that makes the planner feel like a conversation rather than a batch job.

---

## Editions

| | **Open Source** | **Free** | **Professional** | **Enterprise** |
| --- | --- | --- | --- | --- |
| Runs on | your infrastructure | Avaloka cloud | your cloud, your billing | your cloud, your billing |
| Analysis, codegen, visualization | ● | ● | ● | ● |
| Export the generated code | ● | ● | ● | ● |
| Ray, scheduler, batch jobs | **●** | ○ | ● | ● |
| Model training, registry, inference serving | **●** | ○ | ● | ● |
| Cloud / database connectors | ○ | ○ | ● | ● |
| Provisioning & scheduling onto a cloud cluster | ○ | ○ | ● | ● |
| Teams, sharing, comments, notifications | ○ | ○ | ○ | ● |

The open-source edition is sized for a laptop or a small self-managed cluster.
That is the target, it is what gets tested, and it is what the capability
matrix grants.

| | |
| --- | --- |
| **Tested on** | macOS and Linux laptops; a single-node `kind` cluster; small self-managed clusters |
| **Not tested on** | large multi-node clusters, autoscaling node pools, multi-tenant or production workloads |
| **Support** | community, best-effort. You run it at your own risk. |

If you need scale — provisioned and autoscaled clusters, larger distributed
jobs, scheduled cloud workloads, SLAs — that is what the commercial editions
are for. See [avaloka.ai](https://avaloka.ai), or email
**[support@avaloka.ai](mailto:support@avaloka.ai)**. Full detail, including how
each boundary is enforced in code: [docs/EDITIONS.md](docs/EDITIONS.md).

---

## Learn more

| | |
| --- | --- |
| ✍️ **[Blog](https://avaloka.ai/blog)** | Release notes, design stories and what we are building next |
| 📦 **[Install Guide](docs/INSTALL.md)** | Laptop, Kubernetes, offline and air-gapped installs |
| 💻 [CLI Guide](README_CLI.md) | Every `avaloka` command, and what each error means |
| 🧰 [Developer Guide & SDK](docs/guide/index.html) | Build on Avaloka from Python; every example comes from a real run |
| 🚶 [User Guide](docs/USER_GUIDE.md) | Sign in, upload, ask, connect a database, train a model |
| 🗺️ [Architecture](docs/architecture.md) | The agent team, the request lifecycle, and why each dependency exists |
| ☸️ [Deployment](docs/deployment.md) | Helm, Ray/KubeRay, storage classes |
| 📊 [Benchmarks](docs/benchmarks.md) · [Reliability](docs/test-reports/1.6-reliability-measurements.md) | What was measured, and what it does not cover |
| 💰 [Model Routing & Cost](docs/model-routing-cost.md) | What each agent costs per turn, and which model is worth upgrading |
| 📄 [Research](docs/research/) | The paper behind the system |

Every document, and who it is for: [docs/DOCS_MAP.md](docs/DOCS_MAP.md).

---

## Contributing

We would love your help. Setup, conventions, test tiers and the PR workflow are
in [CONTRIBUTING.md](CONTRIBUTING.md). Contributions are accepted under the
[Developer Certificate of Origin](https://developercertificate.org/) — sign
your commits with `git commit -s`. No CLA.

Found a security issue? Please report it privately, per [SECURITY.md](SECURITY.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Third-party attribution is in
[NOTICE](NOTICE).
