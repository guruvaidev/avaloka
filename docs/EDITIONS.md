# Avaloka Editions

Avaloka ships in one open-source edition and three commercial ones. This page
says exactly what each includes, and — more usefully — *why* each boundary
exists, so you can tell at a glance whether a limit applies to you.

Run `./scripts/install.sh --check` at any time to see what your own install
resolves to.

---

## The two questions that decide everything

Most "what do I get?" confusion comes from collapsing two independent questions
into one. Avaloka keeps them separate:

| Question | Values | Decides |
| --- | --- | --- |
| **Where does it run?** | self-hosted · Avaloka-hosted | who pays for the compute and carries the risk |
| **What did you buy?** | OSS · Free · Professional · Enterprise | which capabilities are licensed |

This is why the honest answer to *"does the free version have a scheduler?"* is
**it depends on where it runs** — and that is not evasion:

- **Self-hosted open source: yes.** It is your cluster and your bill. We do not
  cap what you do with your own hardware.
- **Free on Avaloka's cloud: no.** We pay for every scheduled cycle, so
  scheduling is reserved for paid plans.

Same code. Opposite answer. The difference is who is paying.

---

## Why a capability might be unavailable

There are exactly three reasons, and every gate in the codebase declares which
one applies (see `GATE_REASON` in `app/core/editions.py`):

| Reason | Meaning | Can I work around it? |
| --- | --- | --- |
| **OPEN** | Not gated at all | Nothing to work around |
| **COST** | Avaloka provides the compute, so hosted plans are capped | **Yes — self-host.** Then it is your bill and the cap disappears |
| **COMMERCIAL** | The implementation is not in the open-source distribution | No. The code is not on disk |

We do not ship gates for any other reason. If you find a capability disabled in
a self-hosted open-source install for a COST reason, that is a bug — please
file it.

---

## Edition summary

| | **Open Source** | **Free** | **Professional** | **Enterprise** |
| --- | --- | --- | --- | --- |
| Licence | Apache 2.0 | Commercial | Commercial | Commercial |
| Runs on | your infrastructure | Avaloka cloud | your cloud, your billing | your cloud, your billing |
| Compute paid by | you | Avaloka | you | you |
| Support | community | none | business hours, SLA | 24×7, SLA, indemnity |
| Best for | developers, students, evaluation | trying Avaloka | data teams at scale | companies working as teams |

---

## Capabilities

● included ○ not included

### Analysis — open in every edition

| | OSS | Free | Pro | Ent |
| --- | :-: | :-: | :-: | :-: |
| Conversational data analysis | ● | ● | ● | ● |
| Planner / Coder / Validator orchestration | ● | ● | ● | ● |
| Sampling and profiling | ● | ● | ● | ● |
| Visualization | ● | ● | ● | ● |
| File connectors — CSV, Parquet, Avro, Delta, Iceberg, JSON, XML, Excel | ● | ● | ● | ● |
| Local execution | ● | ● | ● | ● |
| **Export the generated code** | ● | ● | ● | ● |

Generated code is yours in every edition, including Free. It is your work
product; you take it and run it at your own discretion and your own risk, as
you would any code you commissioned.

### Execution and scheduling — COST-gated

| | OSS | Free | Pro | Ent |
| --- | :-: | :-: | :-: | :-: |
| Ray distributed execution | ● | ○ | ● | ● |
| Entire-dataset / large-scale runs | ● | ○ | ● | ● |
| Scheduler | **●** | **○** | ● | ● |
| Batch jobs | **●** | **○** | ● | ● |

Open source gets all of it. You bring your own Ray cluster, your own cloud
credentials and your own LLM keys, and you run it at your own risk.

### Connectivity — the Professional line

| | OSS | Free | Pro | Ent |
| --- | :-: | :-: | :-: | :-: |
| Database connectors | ○ | ○ | ● | ● |
| Cloud object storage — S3, GCS, Azure | ○ | ○ | ● | ● |
| Automated cloud provisioning — GKE, EKS | ○ | ○ | ● | ● |
| Scheduled delivery of results | ○ | ○ | ● | ● |
| Model training, registry, inference serving | **●** | ○ | ● | ● |
| Scheduling training onto a cloud cluster | ○ | ○ | ● | ● |

Professional is the edition that reaches into **your** cloud and **your**
databases, on **your** infrastructure and billing — including petabyte-scale
work that never leaves your environment.

### Collaboration — the Enterprise line

| | OSS | Free | Pro | Ent |
| --- | :-: | :-: | :-: | :-: |
| Organizations and teams | ○ | ○ | ○ | ● |
| Invite and manage members | ○ | ○ | ○ | ● |
| Share analysis | ○ | ○ | ○ | ● |
| Comment on analysis and insights | ○ | ○ | ○ | ● |
| Notifications and team delivery | ○ | ○ | ○ | ● |
| Roles, departments, permissions | ○ | ○ | ○ | ● |
| SSO and audit log | ○ | ○ | ○ | ● |

---

## How the boundary is enforced

Worth being direct about, because it is the first thing an engineer asks.

**Commercial capabilities are not hidden behind a flag you could flip.** They
are gated on whether the private `avaloka-commercial` distribution is
installed. In an open-source install that package is absent, so the code behind
those capabilities is not on disk. Setting `AVALOKA_EDITION=enterprise` in an
open-source checkout grants nothing, and we have a test that proves it.

For paid deployments running inside your own VPC with no route back to us,
entitlement is resolved from an **Ed25519-signed licence token, verified
offline**. Nothing phones home. The verifier is open source — publishing it is
safe, because issuing a licence needs a private key that is not in this
repository.

If a licence is missing, expired, or invalid, Avaloka **degrades to the
open-source capability set rather than refusing to start.** Your analysis keeps
working; the paid capabilities switch off.

---

## Upgrading

Installing a commercial edition adds one package to an otherwise identical
install:

```bash
./scripts/install.sh --edition professional --license-key "$AVALOKA_LICENSE_KEY"
```

See [INSTALL.md](INSTALL.md). Licence keys come from the Avaloka team.

## Questions

- **Can I use the open-source edition commercially?** Yes. Apache 2.0. See
  [LICENSE](../LICENSE). The "Avaloka" name and logo are trademarks and are not
  licensed with the code — Apache 2.0 §6 reserves them.
- **Can I self-host Professional or Enterprise?** Yes, that is the normal
  deployment. Offline licence verification exists for exactly this.
- **Will you open-source more over time?** Capabilities move outward, not
  inward. Anything published stays published.

---

## Where it runs, and what you licensed

Two independent questions decide what you get, and collapsing them is the usual
source of confusion:

| Question | Values | Decides |
| --- | --- | --- |
| **Where does it run?** | self-hosted · Avaloka-hosted | who pays for the compute |
| **What did you buy?** | OSS · Free · Professional · Enterprise | which capabilities are licensed |

This is why *"does the free version have a scheduler?"* has no single answer.
**Self-hosted open source: yes** — it is your cluster and your bill.
**Free on Avaloka's cloud: no** — we pay for every scheduled cycle. Same code,
opposite answer; the difference is who is paying.

A capability is unavailable for exactly one of three reasons, and every gate
declares which (`GATE_REASON` in `app/core/editions.py`):

| Reason | Meaning | Workaround |
| --- | --- | --- |
| `OPEN` | Not gated | Nothing to work around |
| `COST` | We provide the compute, so hosted plans are capped | **Self-host** — then it is your bill |
| `COMMERCIAL` | The implementation is not in the OSS distribution | None; the code is not on disk |

A capability disabled in a self-hosted OSS install for a `COST` reason is a bug.
The boundary is enforced by **import probe**, not configuration — no environment
variable can claim a capability whose module is absent — which is what makes the
split honest rather than advisory. See [docs/EDITIONS.md](EDITIONS.md).

```bash
./scripts/install.sh                 # open source
./scripts/install.sh --check         # what does my install resolve to?
./scripts/install.sh --edition professional --license-key "$KEY"
```

#### What the open-source edition is sized for

**A laptop, or a small Kubernetes cluster you run yourself.** That is the
target, it is what gets tested, and it is what the capability matrix grants.
Everything below follows from it.

| | |
| --- | --- |
| **Tested on** | macOS and Linux laptops; a single-node `kind` cluster; small self-managed clusters |
| **Not tested on** | large multi-node clusters, autoscaling node pools, multi-tenant or production workloads |
| **Support** | community, best-effort. You run it at your own risk. |

If you need scale — provisioned and autoscaled clusters, larger distributed
jobs, scheduled cloud workloads, SLAs — that is what the commercial editions
are for. See [avaloka.ai](https://avaloka.ai), or email
**[support@avaloka.ai](mailto:support@avaloka.ai)**.

**Included, and fully functional**

- The whole analysis path — profile, transform, analyse, visualise
- **Distributed Ray execution on a cluster you already run**, and the local
  scheduler (Celery + Redis via `docker-compose.scheduler.yml`). These are not
  capped: it is your cluster and your bill, so capping them would protect
  nothing. They are *sized* for a small cluster, not *limited* to one.
- Model training and inference locally
- The generated code, always. Whatever Avaloka writes, you can export and run in
  your own environment. There is no black box.
- File connectors: CSV, TSV, JSON, XML, Excel, Parquet, Avro, Delta, Iceberg
- Any model provider, including local models

**Not included**

- **Cloud infrastructure provisioning.** Avaloka will not create a GKE, EKS or
  AKS cluster for you. `get_provider("gcp"|"aws"|"azure")` refuses in an
  open-source build and tells you where to go. This is a genuine limitation,
  not a switch: standing up and paying for cloud infrastructure on your behalf
  is what the commercial editions do.
- **Scheduling work onto a cloud cluster.** The scheduler runs locally and
  schedules local and own-cluster work. What it cannot do is dispatch a job to
  a managed cluster it did not provision.
- **Scheduling a training run onto a cloud cluster.** MTA itself is yours —
  train and serve models locally or on your own Ray cluster, with MLflow
  tracking and the model registry. What is commercial is dispatching that work
  to managed cloud infrastructure Avaloka provisioned for you.
- **Scale operations** — autoscaling policy, node-pool management, cost
  controls, multi-tenant isolation.
- **Team features** — shared analyses, comments, notifications, SSO and audit.

> **What "connect" still does.** Avaloka can attach to any cluster your
> `kubeconfig` already reaches, including a cloud one you provisioned yourself,
> and run distributed Ray and scheduled jobs on it. The commercial line is
> Avaloka *creating and managing* that infrastructure for you — not whether
> your cluster happens to sit in a cloud.

##### With a commercial edition

Point Avaloka at any cloud and let it run there: provisioned and autoscaled
clusters, larger distributed execution, scheduled jobs on that cluster, and
delivery of the results. Teams share analyses, comment on them, and work from
the same connections and schedules.

The Enterprise UI adds the configuration surfaces for that — cloud Kubernetes
connections and scheduler configuration — which the open-source edition has no
use for, because it has no cloud cluster to point them at.

See [docs/EDITIONS.md](EDITIONS.md) for the capability-by-capability
breakdown, and run `./scripts/install.sh --check` to see what your install
resolves to.

---
