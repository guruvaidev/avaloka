# Avaloka CLI

> A CLI-first, agentic data-science execution platform that converts datasets and
> business objectives into validated reports, reproducible models and deployable
> inference services — within explicit time and cost budgets.

**Avaloka is a person.** She's a young, sharp data scientist who sizes the job
the moment she sees it, reads your data almost at a glance, clones herself across
the work, and converges a single answer she'll stand behind. That experience —
not another analytics dashboard — is the product.

Avaloka's atomic deliverable is not a chat message or a notebook cell. It is a
**Data Mission**: give her data, an intended decision and an operating budget,
and her swarm of self-clones delivers the analysis, tested code, validated model
and deployment package — with every assumption, cost and limitation exposed.

```bash
avaloka chat                       # talk to her
avaloka chat customers.csv         # ...starting from a dataset
```

---

## Start here — five minutes

```bash
pip install -e .                                  # from this repo
avaloka analyze sales.csv --goal "why did revenue drop in Q3?"
```

That is the whole loop. Avaloka sizes the file, profiles it, plans an analysis,
writes and runs the code, validates the result, and leaves a bundle you can
open, rerun and hand to someone else. Nothing leaves your machine unless you
configure a hosted model provider.

Three commands cover most work:

| | |
| --- | --- |
| `avaloka analyze <file> --goal "..."` | answer a question about a dataset |
| `avaloka train <file> --target <column>` | predict a column, and validate the model |
| `avaloka chat <file>` | work it out conversationally, one turn at a time |

Reads **CSV, TSV, Parquet and Excel** (`.xlsx`/`.xlsm`/`.xls`), including
spreadsheets whose header sits below a title block.

---

## `avaloka chat` — the interactive session

The fastest way in when you do not yet know what you are looking for. Avaloka
reads the file, says what she notices, and **offers numbered next steps you can
answer with a number**:

```
$ avaloka chat customers.csv

Avaloka  500 rows across 7 columns.
Avaloka  region leans hard on "east" (28% of rows) - that's a segment worth naming.
Avaloka  I'm treating customer_id as identifiers, not signal.

Avaloka  Here is what I would look at next:
  1. train churned
  2. analyze how churned varies by region
  3. analyze the relationship between tenure_months and churned
  4. analyze monthly_spend broken down by region
Avaloka  Say the number and I will start - or tell me something else entirely.

you > 2
Avaloka  Taking 2 - analyze how churned varies by region.
```

Every suggestion names columns this dataset actually has, and every one is a
command the session can run. Inside a session:

| Type this | And she will |
| --- | --- |
| a path to a file | load that dataset |
| **a number** | run the suggestion with that number |
| `analyze <goal>` | run an analysis toward a decision |
| `train <column>` | predict that column and validate the model |
| `columns` | show the schema she read, with roles and missingness |
| `suggest` | offer next steps again |
| `status` | say what is currently loaded |
| `help` | list all of this |
| `quit` | leave |

Or just say what you are trying to find out, in your own words. A mistyped
column name gets a suggestion rather than a list: `train churnd` answers
*"Did you mean churned?"*

---

## When something goes wrong

Every failure names the problem and the next thing to try, rather than printing
a stack trace:

| What you did | What you get |
| --- | --- |
| `--target churnd` | *"There is no column named 'churnd'. Did you mean: churned?"* plus the column list |
| `--metric auc` | *"'auc' is not a metric I know how to optimise. Did you mean: roc_auc?"* |
| pointed at a directory | *"... is a directory, not a dataset."* |
| a file with headers and no rows | *"... has column headers but no rows. There is nothing to analyse."* |
| a binary or compressed file | *"... does not look like CSV text — its column names are unreadable."* |
| an unwritable `--output` | caught **before** the work starts, not after it |

Set `AVALOKA_TRACEBACK=1` to get the full trace when you are debugging Avaloka
itself rather than your data.

---

## Reading a result

A run reports what it found before it reports what it cost:

```
What I found
  • 'support_tickets' and 'last_login_days' are strongly associated (|corr|=0.98).
  • 'last_login_days' and 'churned' are strongly associated (|corr|=0.94).

What to be careful about
  • Only 120 rows; statistical conclusions are low-confidence.

Validation verdict: WARN; max safe deployment level 2/4
  ! data_sufficiency: Too few rows for stable EDA.

Deliverables -> ./avaloka-analysis
  open ./avaloka-analysis/executive_report.html   the decision summary
  cat  ./avaloka-analysis/README.md               findings and limitations
  python ./avaloka-analysis/analysis.py           reproduce this run
```

A verdict never appears without the check that produced it, and a model score
never appears without the baseline it has to beat:

```
The model I trust most is random_forest (roc_auc 0.7913).
That beats the trivial baseline (0.5000) by more than fold noise.
```

If it does not beat the baseline, Avaloka says so plainly and the validation
fails — a model that beats nothing is not a weak result, it is an absent one.

---

## Workload-based routing (online · sampled · batch)

Avaloka sizes every job *before* she touches it and picks the honest lane —
cheaply, by reading file bytes + a fast row estimate, so she can tell you the
plan before doing heavy work:

| Lane | When | What she does |
|---|---|---|
| **online** | small (≤100k rows, ≤50 MB) | analyses the **whole** dataset live, now |
| **sampled_online** | large (≤5M rows, ≤1 GB) | a statistically honest **sampled** read now **+** offers the full batch |
| **batch** | big-data (>5M rows / >1 GB) | recommends the full pass as a **Ray swarm on Kubernetes** |

The chosen lane is announced in her voice, written to `workload.json`, and drives
sampling automatically. Run the full unsampled pass any time with `avaloka batch`.

Three product promises:

1. **Outcome, not interaction** — a report, notebook, model or service, not a chat reply.
2. **Validation, not blind automation** — Avaloka exposes uncertainty and refuses unsafe deployment.
3. **Economics, not agent theatre** — every mission reports time, cost, quality and estimated work avoided.

## Install

```bash
pip install -e .          # from this repo
avaloka --version
```

Local execution is **free** and keeps your data on your machine. An LLM is used
only to enrich narrative prose (never to invent numbers) and is entirely
optional — install the extra and set `ANTHROPIC_API_KEY` to enable it:

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...   # only used in --managed / --execution byoc modes
```

## The swarm — Avaloka's self-clones

When a mission runs, Avaloka **clones herself**: each clone is *her*, narrowed to
the one job she's best at, working in parallel where the data allows. She
dispatches them, they return, and she **converges** everything into one answer:

```
Avaloka  Cloning myself into 6 focused copies — each takes the part she's best at.
  ✦ Ava-Scout sets off — she reads the schema, the distributions and the cracks.
  ✦ Ava-Sampler sets off — she carves out a sample that still tells the truth.
  ✦ Ava-Skeptic sets off — she tries to break the result before anyone else can.
Avaloka  All 6 copies are back. I'm merging what they found into one answer.
Avaloka  Here's what jumped out at me:
  contract leans hard on “month-to-month” (56% of rows) — a segment worth naming.
```

Every observation she makes is pinned to a **measured statistic** — the magic is
real, never generated. Add `--llm` (with `ANTHROPIC_API_KEY`) to let her phrase
things with Claude; `--quiet` drops to machine-only output for scripting.

Under the hood each clone is a **firefly**: an *economic* unit, not an
anthropomorphic demo. Each maps to an expensive human responsibility and
**activates only when its expertise is needed** — an analysis mission never wakes
the Model Scientist or ML Engineer.

| Firefly | Replaces | Economic output |
|---|---|---|
| Data Scout | Data analyst | Schema, distributions, quality report |
| Sampling Specialist | Data scientist | Statistically defensible working sample |
| Data Engineer | Data engineer | Transformations + reproducible pipeline |
| Analysis Planner | Senior data scientist | Hypotheses, task plan, resource plan |
| Model Scientist | ML scientist | Baselines, experiments, selected model |
| Validator / Skeptic | Model-risk specialist | Leakage, stability, validity checks |
| ML Engineer | ML platform engineer | Container, API, deployment manifest |
| FinOps | Infrastructure engineer | Cost & compute optimisation |
| Reporter | Analyst / consultant | Executive & technical reports |

Every activation is recorded as a measured **work unit** (manual minutes
replaced, compute/model cost, duration) — that ledger is what makes the
economics defensible.

## `avaloka analyze` — a dataset and a question, in, a defensible bundle out

```bash
avaloka analyze customers.csv \
  --goal "Understand churn drivers and identify actionable segments" \
  --budget 5 \
  --output ./churn-analysis
```

Produces a complete analysis bundle:

```
churn-analysis/
├── executive_report.html      ├── data_quality.json
├── technical_report.html      ├── assumptions.yaml
├── analysis.ipynb             ├── validation_report.json
├── analysis.py                ├── planner_graph.json
├── transformed_dataset.parquet├── lineage.json
├── README.md                  └── environment.lock
```

## `avaloka train` — predict a column, and find out whether you should trust it

```bash
avaloka train customers.csv \
  --target churned \
  --metric roc_auc \
  --max-cost 25 \
  --deployment-latency-ms 100 \
  --deployable \
  --output ./churn-model
```

Adds a validated model and a full deployment package: `model/`, `model_card.md`,
`evaluation_report.html`, `feature_contract.yaml`, `inference_schema.json`,
`requirements.lock`, `Dockerfile`, `service.py`, `tests/`,
`monitoring_config.yaml`, `deployment/{kubernetes,rayservice,local-compose}.yaml`
and `mlflow/`.

The planner is **budget-aware**: the cost ceiling decides how many candidate
models are trained, and FinOps emits an Economical / Balanced / Maximum-quality
options table with serving-cost estimates and a recommendation
(`cost_quality.json`).

## `avaloka batch` — the full analysis as a Ray swarm

For data too large to analyse live, Avaloka fans herself across **Apache Ray**:
each clone profiles one partition and she converges their partial statistics into
one **exact, full-dataset** profile (a real map-reduce — partial sums combine
losslessly, so there is *no* sampling error in the batch lane).

```bash
avaloka batch transactions.parquet --goal "Profile transactions" --partitions 8
avaloka batch transactions.parquet --target kubernetes --apply        # KubeRay RayJob
```

It runs a real local Ray job for proof and small clusters, and writes a KubeRay
`rayjob.yaml` (autoscaling workers, `minReplicas: 0`) plus a standalone
`ray_batch_driver.py` to submit to a cluster.

## `avaloka infer` — REST API on Kubernetes from the MLflow model

Turns the validated, packaged model into a live REST endpoint:

```
MLflow model  →  Docker image  →  Kubernetes Deployment+Service  →  REST API
```

```bash
avaloka infer ./churn-model --target kubernetes                       # plan (prints commands)
avaloka infer ./churn-model --target kubernetes --mlflow-uri http://mlflow:5000 \
  --registry ghcr.io/acme --apply --push                              # build + deploy + endpoint
```

`--apply` actually builds the image, applies the manifest, and resolves the
service IP/port (reusing the existing cluster helpers in `app/infra`), then prints
the live `curl` for `/health` and `/predict`. Without `--apply` it's a safe plan.

## `avaloka deploy` — check a model against the deployment gates

```bash
avaloka deploy ./churn-model --target kubernetes --level 3
avaloka deploy ./churn-model --target kubernetes --level 4 --approve   # production
```

Deployment is **gated** by four levels and Avaloka never silently promotes a
model to production:

| Level | Name | Requires |
|---|---|---|
| 1 | Research artifact | model file + model card |
| 2 | Deployment package | API, container, schema, tests, lock |
| 3 | Staging-ready | manifest, monitoring, registry descriptor |
| 4 | Production candidate | **human approval** + clean validation |

The Validator caps the level a package may claim; Level 4 additionally requires
`--approve`.

## Economics

Every mission closes with a defensible economic summary:

```
Mission duration               1.3 seconds
Estimated manual effort        23.1 hours
Human review required          2.1 hours
Total mission cost (Avaloka)   $29.00
Total customer cost            $195.40
Estimated labor value created  $1,848
Economic multiplier            9.5x
```

The multiplier is value created vs. **total customer cost** (the Avaloka invoice
*plus* the human review the customer still performs) — an honest denominator, so
a free local mission never reports an implausible figure. "Estimated manual
effort" is a conservative heuristic, to be calibrated from observed customer
workflows rather than invented by an LLM.

## Execution modes (software economics ≠ compute economics)

- `--local` (default): free, open, data stays put, you supply the compute.
- `--execution byoc`: bring-your-own-compute; Avaloka is the control plane, you pay the cloud.
- `--managed`: Avaloka-managed with hard cost limits and prepaid credits.

Invoices separate orchestration/validation from compute/serving so the software
price is never confused with an inflated cloud bill.

## Supported (initial market)

- **Data:** CSV, Parquet (PostgreSQL / object storage are the next connectors).
- **Problems:** profiling, EDA, binary/multiclass classification, regression.
- **Deployment:** local REST, Docker, Kubernetes, Ray Serve.

## Develop / test

```bash
pip install -e ".[dev]"
pytest tests/avaloka -q
```
