# Avaloka

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](docs/versions.md)
[![Ray](https://img.shields.io/badge/runs%20on-Ray%20%2F%20KubeRay-028CF0.svg)](https://www.ray.io/)

**A team of specialist AI agents that takes a dataset from raw rows to a served
model — and checks its own work before it reports a result.**

In 2007 Jim Gray described a fourth paradigm of science, after the empirical,
the theoretical and the computational: discovery by exploring data at a scale
no one person can hold in their head. His argument was not really about data.
It was about instruments — that scientists were spending their time on the
plumbing of capturing, cleaning, moving and analysing data instead of on the
questions.

Two decades on, the plumbing is still there, and it still stands between a
curious person and an answer. Avaloka is an attempt at the instrument. You
state the question in plain language and a team of agents — planner, profiler,
coder, validator, executor, trainer, visualizer — samples the data, plans the
analysis, writes and validates the Python, runs it locally or on a Ray cluster,
trains a model, serves it, and records every artifact along the way.

An instrument is only as good as its calibration, so three of those agents
exist to check the other six. Avaloka refuses to train on leaked data, makes
every model beat a trivial baseline, and checks the prose it writes against the
numbers it computed. Where that evidence is thin,
[we say where](docs/test-reports/three-pillar-coverage.md).

📖 **[Full documentation](docs/index.html)** · 🚀 **[Install Guide](docs/INSTALL.md)** · 📊 **[Benchmarks](docs/benchmarks.md)**

---

## Why a model alone does not solve this

A frontier model will read your spreadsheet and give you a good answer about
that spreadsheet. The facts you actually want are rarely in one spreadsheet —
they are across an operational database, a warehouse and an object store, and
the question is the join between them.

But the harder problem is not reach. It is that the failure mode that destroys
credibility is not a crash. It is a fluent, well-formatted, wrong answer.

We know the shape of that failure because we shipped it. Given a dataset of
pure noise, an earlier version of Avaloka's training path returned verdict
`pass` and `max_safe_deployment_level 4` — it certified a model that had
learned nothing as production-ready. Our own benchmark scored 1.00 before *and*
after the fix, because no task in it carried the shape that triggers the bug.

No model choice fixes that, because a model has no privileged access to whether
its own answer is true. What fixes it is a harness that computes its own
baseline, and a benchmark that can fail.

> *A benchmark that cannot fail is a statement of intent, not a measurement.*

[How the harness works](docs/architecture.md) · [what the benchmark covers](docs/benchmarks.md)

---

## The team

Nine roles, one shared state object. The **Planner** is the team lead — it
talks to you and decides the next step; everything else is a specialist it
delegates to.

| | Agent | What it owns |
| --- | --- | --- |
| **Lead** | Planner | Converses, plans, routes — keep planning, provision, code, train, schedule, execute |
| **Look** | Sampling · Profiling | Samples and per-column stats; infers domain, ranks predictive power, flags quality |
| **Write** | Coder | Numbered pseudocode first, then Python that adheres to it |
| **Check** | Validator | Three layers — syntax, schema-aware static, logical — before anything runs |
| **Run** | Execution | Locally, as a Kubernetes Job, or on a Ray cluster |
| **Move** | Data Transfer | Daft ETL across CSV, Parquet, Avro, Delta, Iceberg, JSON, XML, Excel |
| **Train** | Model Training | PyTorch + ONNX + MLflow; distributed on Ray, with a model registry |
| **Show** | Visualization | Charts from execution output |
| **Judge** | Integrity · Evaluation · Claim Verifier | The evidence layer, below |

The full roster — Planner Graph, Summarizer, Scheduler, Infrastructure, the
conversational layer — is in [Architecture §2](docs/architecture.md).

### The evidence layer

Three checks stand between a result and you:

* **Integrity** — refuses to train on leaked targets or duplicate splits.
* **Evaluation** — every model must beat a trivial baseline, measured on the
  **confidence interval, not the mean**, so a majority-class classifier cannot
  sneak through on floating-point noise.
* **Claim Verifier** — checks the prose against the computed numbers. It is
  **deterministic, with no LLM involved**, which is why the check cannot itself
  hallucinate.

The Validator asks *will this code run*. The Claim Verifier asks *is what we
just told you true*. They are different questions and they need different
machinery.

---

## Quickstart

Five minutes, on your own machine, no cloud account.

```bash
git clone https://github.com/guruvaidev/avaloka.git && cd avaloka
./scripts/install.sh                 # creates .venv, installs deps, prints an edition report
```

Give it one model key — any provider — and start the API:

```bash
export GROQ_API_KEY="<your key>"     # or OPENAI_API_KEY / OPENROUTER_API_KEY / a local Ollama
uvicorn app.api.server:app --port 9000
curl localhost:9000/health
```

Then upload something and ask a question:

```bash
curl -F "file=@sales.csv" localhost:9000/api/upload
```

> **Port 9000, not 8000.** The container image serves on 9000 and the Helm
> service maps `8000 → 9000`. Uvicorn's own default is 8000, which is the single
> most common reason a fresh install "cannot reach the API" — pass `--port 9000`
> and every path lines up.

Prefer a cluster, a UI, or no internet at all? That is all in the
**[Install Guide](docs/INSTALL.md)** — including
[running Avaloka completely offline](docs/INSTALL.md#running-avaloka-completely-offline)
with a `--network none` check so you can prove it rather than trust it.

**Something not working?** The install guide's
[First run: what to expect, and what goes wrong](docs/INSTALL.md#first-run-what-to-expect-and-what-goes-wrong)
section lists every failure we hit bringing this stack up on a clean cluster.
Each one is silent — the symptom never names the cause.

---

## What is proven, and where it is thin

We would rather publish the gaps than claim three pillars and evidence one.

| Pillar | What is measured | Standing |
| --- | --- | --- |
| **Harness** | 16 executed test cases covering leakage, duplicate-split detection, identifier-like features, the mandatory baseline, invented-number detection, and the scorer's own resistance to gaming | **Strongest.** Most of the evidence layer is verified end to end against a live deployment |
| **Knowledge graph** | 2 executed cases: ancestry recorded across derivations, and PII located *through* lineage rather than by re-scanning | **Real but thinly covered.** The store exists and the cases pass; the coverage is 2 cases, not 20 |
| **Learning loop** | Deployed and verified recalling across a pod restart (`3.4s` cold with hints already present, `0.6s` warm) | **Runs.** It carries context forward so later work has something to build on. We do not claim it makes analyses measurably better; that is not what it is there for yet |

The honest summary is that Avaloka today is a **strong harness, a real graph,
and a working learning loop whose value is still ahead of it**. The
[test reports](docs/test-reports/) contain the per-case evidence for each row,
including which cases are blocked and on what.

**Benchmarks.** `python -m avaloka.benchmark run` is a deterministic regression
harness — no LLM involved — over the ingestion and data-science paths. It
currently passes 17/17. That number is a floor, not a leaderboard result: it
scored 1.00 before *and* after five real defects were fixed, because no task
carried the shape that triggers them. Full scorecard and an account of what it
does **not** measure: [docs/benchmarks.md](docs/benchmarks.md).

**Task design** is aligned with the multi-step data-agent framing established
by DABstep (Adyen / Hugging Face), as described in the
[research paper](docs/research/). We have not run that suite end to end and
claim no score on it — alignment of design is not a score.

**Tests.** `pytest -m "not cluster and not cloud and not integration"` is the
gate that must stay green: unit and contract tests, no cluster and no network.
The wider suite (end-to-end, integration, k8s) needs Redis, a running server
and cloud credentials. **77 tests across it currently fail** — down from 395,
by fixing causes rather than deleting tests. Each remaining one needs its own
diagnosis. See [Testing](docs/testing.md).

---

## Built on proven open source

Avaloka is an orchestration layer, not a reimplementation. Distributed
execution, storage, vector search, experiment tracking and the database are
done by projects that were load-bearing in production long before Avaloka
existed. What Avaloka adds is the agent team, the evidence layer and the
conversation. What it deliberately does *not* add is a homegrown substitute for
any of these.

| Layer | We use — rather than our own |
| --- | --- |
| Distributed compute | Ray + KubeRay, Spark, Arrow, Daft — not our own scheduler |
| Storage & formats | PostgreSQL, MinIO (S3), Parquet, Delta, Iceberg, Avro — not a bespoke artifact format |
| Vector & memory | Chroma, Milvus, Redis — not our own index |
| ML & tracking | PyTorch, scikit-learn, ONNX, MLflow — not our own AutoML |
| Agent runtime | LangGraph / LangChain, Model Context Protocol — not a proprietary agent protocol |
| Platform | Kubernetes, Helm, FastAPI, Pydantic, Celery, Supabase — not our own web, job or identity stack |
| Data engine | pandas, NumPy, DuckDB, SQLAlchemy — not our own dataframe |

Each of these is independently maintained and audited by a community far larger
than ours, and none of it is hidden behind our abstraction — you can verify any
number Avaloka reports by running the layer underneath it yourself.

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

## Documentation

Everything is indexed at **[docs/index.html](docs/index.html)**. The paths people
ask for most often:

| | |
| --- | --- |
| 📦 **[Install Guide](docs/INSTALL.md)** | Start here to run it |
| 🚶 [User Guide](docs/USER_GUIDE.md) | Sign in, upload, prompt, connect a database, train a model |
| 💻 [CLI Guide](README_CLI.md) | The `avaloka` command, from a terminal |
| 🗺️ [Architecture](docs/architecture.md) | The agent team, the request lifecycle, why each dependency exists |
| ☸️ [Deployment](docs/deployment.md) | Kubernetes, Ray/KubeRay, Helm, storage classes |
| 📊 [Benchmarks](docs/benchmarks.md) · [Reliability](docs/test-reports/1.6-reliability-measurements.md) | What was measured, and what it does not cover |
| 📄 [Research](docs/research/) | The paper behind the system |

---

## Contributing

Development setup, coding conventions, testing tiers and the PR workflow are in
[CONTRIBUTING.md](CONTRIBUTING.md). Contributions are accepted under the
[Developer Certificate of Origin](https://developercertificate.org/) — sign
commits with `git commit -s`. No CLA.

Security vulnerabilities: please report privately, per [SECURITY.md](SECURITY.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Third-party attribution is in
[NOTICE](NOTICE).
