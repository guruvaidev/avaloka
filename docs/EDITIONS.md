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

## Why a capability might be unavailable, or limited

A capability is unavailable for exactly three reasons, and every gate in the
codebase declares which one applies (see `GATE_REASON` in
`app/core/editions.py`). One capability is available everywhere but *limited*
for a fourth, declared reason:

| Reason | Meaning | Can I work around it? |
| --- | --- | --- |
| **OPEN** | Not gated at all | Nothing to work around |
| **COST** | Avaloka provides the compute, so hosted plans are capped | **Yes — self-host.** Then it is your bill and the cap disappears |
| **COMMERCIAL** | The implementation is not in the open-source distribution | No. The code is not on disk |
| **PREVIEW LIMIT** | The capability ships and works, but how far it goes is capped as a preview of the paid editions | The number is in open code. Professional and Enterprise raise or remove it |

There is one preview limit today, and only one: the **swarm**, capped at 2
clones per mission on the open-source and Free editions, 16 on Professional,
and unlimited on Enterprise (see [Swarm](#swarm--open-with-a-preview-limit)).
It applies to self-hosted installs as well, which no COST gate does. That is
deliberate, and it is why it has its own name rather than being filed under
COST or COMMERCIAL.

We do not ship gates or limits for any other reason. If you find a capability
disabled in a self-hosted open-source install for a COST reason, or limited
for any reason other than the one above, that is a bug — please file it.

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
| Auto Insights | ● | ● | ● | ● |
| File connectors — CSV, Parquet, Avro, Delta, Iceberg, JSON, XML, Excel | ● | ● | ● | ● |
| Local execution | ● | ● | ● | ● |
| **Export the generated code** | ● | ● | ● | ● |

Generated code is yours in every edition, including Free. It is your work
product; you take it and run it at your own discretion and your own risk, as
you would any code you commissioned.

### Swarm — open, with a preview limit

| | OSS | Free | Pro | Ent |
| --- | :-: | :-: | :-: | :-: |
| Swarm: a mission clones Avaloka into parallel sub-Avalokas | ● | ● | ● | ● |
| Clones per mission | 2 | 2 | 16 | unlimited |
| Clone count sized from the data when none is asked for | — | — | ● | ● |
| Where the clones run | local | local | local | local |

Every edition gets the swarm. Its fan-out is the one **PREVIEW LIMIT**: the
open-source and Free editions get two clones per mission, run in parallel on
your machine. That is enough to watch a fleet converge on one answer, not
enough to replace a team. The number is two so that the swarm runs smoothly on
a laptop: each clone is a full sub-Avaloka running its own firefly swarm on a
shard of the data, and two of them is the headroom a laptop has. Professional
raises it to 16; Enterprise removes it.

The limit is never applied silently. Ask for more and Avaloka announces what it
did before dispatching anything:

> Swarm limited to 2 clones on the free open-source edition: you asked for 8, so 2 will run.

The CLI prints that line even with `--quiet`, `coordination.json` records it
under `execution.swarm`, and `GET /api/capabilities` reports the limit that
applies to the caller under `features.swarm`.

When no count is asked for, Professional and Enterprise size the fleet from
the data (see [Adaptive clone count](#adaptive-clone-count--professional-and-enterprise)
below); the open-source and Free editions run their preview of two whatever
the data says, and are told why in the same sentence.

The number is in open code: `avaloka/swarm_limits.py`, enforced in
`MainAvaloka.coordinate` (`avaloka/coordinator.py`). It is not a COST gate,
because on a self-hosted install the compute is yours, and it is not a
COMMERCIAL gate, because the swarm code ships in the open-source tree. Under
Apache 2.0 you may read it and change it; we say so rather than pretend the
number cannot be edited.

No edition fans clones out across a cluster yet. The coordinator runs them as
threads on one machine, and the capability report says `cluster_fan_out: false`
for every edition rather than advertise what the code does not do.

Like the rest of the matrix, the limit follows installed code. Without the
commercial package, `AVALOKA_EDITION=enterprise` still gets the preview.
`GET /api/capabilities` reports the limits that apply to the caller under
`features.swarm`.

### The swarm console — Professional and Enterprise

| | OSS | Free | Pro | Ent |
| --- | :-: | :-: | :-: | :-: |
| The swarm, told back in the chat after a mission | ● | ● | ● | ● |
| The live `/swarm` page and the swarm/trace API routes | – | – | ● | ● |

The swarm is everyone's; the *console* is not. The console is the live `/swarm`
page and the API routes behind it — `/api/swarm/missions…` (records, clones,
episodes, per-clone telemetry) and the operator's `/api/agent-metrics` — and it
is a Professional and Enterprise feature by owner decision (2026-10-18), gated
COMMERCIAL. The thread trace (`/threads/{id}/trace`) and the planner graph in
every format are **not** console: they show a user how their own prompt was
acted upon, and stay open in every edition. On the open-source and Free editions those routes answer
`403 {"code": "swarm_console_unavailable"}`, `GET /api/capabilities` says so under
`features.swarm_console`, and the UI hides the page rather than nag.

Nothing a clone found is withheld. Ask Avaloka in the chat to *swarm this* (or
press the quick reply) and she runs the same two-clone mission through the same
service, then tells you what the team did — team size, shards, each clone's
return, the merge, the verdict — from the mission's own trace records
(`avaloka/swarm_narration.py`). The same story is appended to the next chat
turn whenever a mission ran on the thread's dataset by any path (the live page,
the planner), keyed on the mission ids recorded on the thread's session. Paid
editions get that message too, with a link to replay it live.

The gate is `app.core.editions.swarm_console_gate()`, applied by one FastAPI
dependency, `require_swarm_console` (`app/api/swarm_gate.py`), that every
console route carries in its signature; `tests/test_swarm_console_gate.py`
enumerates the routes from the app so a new one cannot ship open.

### Adaptive clone count — Professional and Enterprise

`avaloka coordinate` without `--workers` (and `POST /api/swarm/missions`
without `workers`) no longer runs a flat default of four. The fleet is sized
from what is known about the data before anything is sharded
(`avaloka/swarm_sizing.py`, called from `MainAvaloka.coordinate`):

```text
by_rows  = ceil(rows  / 25,000)
by_bytes = ceil(bytes / 64 MiB)
by_size  = clamp(max(by_rows, by_bytes), 2, 64)
clones   = max(2, min(partitions, by_size))   if the source is partitioned
                                              (Parquet row groups, native
                                              table partitions)
         = by_size                             otherwise
```

One clone per partition or row group, up to the size-based count. Indexes
shape the scan inside a clone and do not change the count. The floor is two
because a fleet of one is a mission, not a swarm; the ceiling is a sanity
limit for one machine's thread pool, not an edition limit. The edition cap is
applied *after* this, by the same `grant` as an explicit request, so:

- **Open source and Free** run exactly their preview of two. When the data
  called for more, the announced line says so, in the data's terms rather
  than the user's: *"Swarm limited to 2 clones on the free open-source
  edition: the data sized the fleet at 5, so 2 will run."*
- **Professional** runs the derived count up to 16; **Enterprise** runs it
  as derived.
- `--workers N` (`--swarm N`) remains an explicit override in every edition,
  capped as before.

The reason is recorded, numbers only, as `derived_from` on the
`swarm.requested` trace record and under `execution.swarm.derived_from` in
`coordination.json`, and the CLI narrates it before dispatch ("I sized the
fleet from the data — 120,000 rows, 2.7 MiB, sampled_online lane:
ceil(rows/25,000)=5, ..."). It never names a column or a file, and the
telemetry projection does not carry it.

### Execution and scheduling — COST-gated

| | OSS | Free | Pro | Ent |
| --- | :-: | :-: | :-: | :-: |
| Ray distributed execution | ● | ○ | ● | ● |
| Entire-dataset / large-scale runs | ● | ○ | ● | ● |
| Scheduler | **●** | **○** | ● | ● |
| Batch jobs | **●** | **○** | ● | ● |
| Reports workspace (saved reports, organised across projects) | **●** | **○** | ● | ● |

Open source gets all of it. You bring your own Ray cluster, your own cloud
credentials and your own LLM keys, and you run it at your own risk.

This release does not fine-tune models from user feedback; if a later release
adds optional per-install fine-tuning, it will run on a GPU that you provide
and pay for, and will not be offered on Avaloka-hosted Free.

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
work.

### Collaboration — Professional teams, the Enterprise organization

| | OSS | Free | Pro | Ent |
| --- | :-: | :-: | :-: | :-: |
| Teams (on Pro, members need Professional or higher) | ○ | ○ | ● | ● |
| Invite collaborators onto your work, including Free users | ○ | ○ | ● | ● |
| Organization members (add and manage users) | ○ | ○ | ○ | ● |
| Share reports by email and public link | ● | ● | ● | ● |
| Share analysis (the `share_analysis` flag, beyond email and links) | ○ | ○ | ● | ● |
| Comment on analysis and insights | ○ | ○ | ● | ● |
| Notifications and team delivery | ○ | ○ | ● | ● |
| Roles, departments, branches, permissions | ○ | ○ | ○ | ● |
| SSO and audit log | ○ | ○ | ○ | ● |
| Dataset catalog, personal (the `dataset_catalog` flag: every dataset your analyses touched, schema versions, reuse, delete) | ● | ● | ● | ● |
| Shared dataset catalog (the `shared_catalog` flag: project and organization visibility, stewards, access requests, certification, audit) | ○ | ○ | ○ | ● |

Teams and collaborator invites resolve to `team_collaboration`, which starts at
Professional (owner ruling, 2026-11-01). The organization's own structure (its
members, roles, departments and branches) resolves to `org_structure`, which is
Enterprise only. `share_analysis`, `comment_on_analysis` and `notifications`
are separate flags, each on Professional and Enterprise (owner ruling,
2026-11-01). Every plan can share a report by emailing the results or by
public link; on Free that is the only sharing there is. The flags are the
contract; the row labels describe what they buy you.

The dataset catalog follows the same line. Every edition has a *personal*
catalog (`dataset_catalog`): the datasets your own analyses touched, with
their schema history, visible to you alone -- row-level security in the
hosted database has exactly one predicate, the owner. Project and
organization visibility, stewards, access requests, certification and the
audit trail are the *shared* catalog (`shared_catalog`), Enterprise only
(owner ruling, 2026-11-01; `docs/design/1.7-DATA-PLATFORM.md` §C.4).

---

## Sharing, collaboration and teams by edition

These are the product rules for the hosted and commercial plans.

| | Free | Professional | Enterprise |
| --- | :-: | :-: | :-: |
| Share reports (email and public link) | ● | ● | ● |
| Invite collaborators | ○ | ● (including Free users) | ● |
| Form teams | ○ | ● | ● |
| Share analyses, comment on them, notifications | ○ | ● | ● |
| Team members' plan | — | any plan (owner ruling, #788) | members of the Enterprise organisation |
| Departments, roles, branches and app users | ○ | ○ | ● |

- **Free** can share reports by email and by public link, and nothing else:
  no collaborator invites, no teams, no analysis sharing, comments or
  notifications.
- **Professional** can invite collaborators, including people on the Free
  plan, and can form teams. A team member may be on any plan (Free included);
  creating a team stays Professional and above. Professional also shares
  analyses, comments on them, and gets notifications.
- **Enterprise** includes everything above, plus departments, roles, branches
  and app users.

Avaloka support (support@avaloka.ai) can help organisations add people to teams, form teams, and change plans.

---

## What the open-source edition is sized for

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
  scheduler (Celery + Redis via `deploy/compose/docker-compose.scheduler.yml`). These are not
  capped: it is your cluster and your bill, so capping them would protect
  nothing. They are *sized* for a small cluster, not *limited* to one.
- Model training and inference locally
- The generated code, always. Whatever Avaloka writes, you can export and run in
  your own environment. There is no black box.
- File connectors: CSV, TSV, JSON, XML, Excel, Parquet, Avro, Delta, Iceberg
- Any model provider, including local models

**Not included**

- **Cloud infrastructure provisioning.** Avaloka will not create a GKE, EKS or
  AKS cluster for you. This is a genuine limitation, not a switch: standing up
  and paying for cloud infrastructure on your behalf is what the commercial
  editions do.

  In the published open-source tree,
  `get_provider("gcp"|"aws"|"azure")` raises `CommercialCapabilityRequired`
  with a message pointing you at `local` (kind) or your own already-provisioned
  cluster. **That gate lives in the OSS overlay**
  (`oss/overlay/app/infra/providers/factory.py:45-53`), which
  `scripts/generate-oss.sh` lays over `app/infra/providers/factory.py` when the
  public tree is generated. The commercial `develop-1.6` tree you may be
  reading this in carries the *ungated* factory — `app/infra/providers/factory.py`
  has no edition check in it at all — because that is where the enterprise
  capability lives. If you are auditing the boundary, read the overlay file, not
  the one in `app/`.
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

### With a commercial edition

Point Avaloka at any cloud and let it run there: provisioned and autoscaled
clusters, larger distributed execution, scheduled jobs on that cluster, and
delivery of the results. Teams share analyses, comment on them, and work from
the same connections and schedules.

The Enterprise UI adds the configuration surfaces for that — cloud Kubernetes
connections and scheduler configuration — which the open-source edition has no
use for, because it has no cloud cluster to point them at.

Run `./scripts/install.sh --check` to see what your own install resolves to;
the capability-by-capability breakdown is the matrix above.

---

## How the boundary is enforced

Worth being direct about, because it is the first thing an engineer asks.

**Commercial capabilities are not hidden behind a flag you could flip.** They
are gated on whether the private `avaloka-commercial` distribution is
installed — `commercial_build()` is an import probe, not a boolean
(`app/core/editions.py:231`). In an open-source install that package is absent,
so the code behind those capabilities is not on disk. Setting
`AVALOKA_EDITION=enterprise` in an open-source checkout grants nothing, and
`tests/test_editions.py` holds that to it.

**What the matrix is, and is not.** `app/core/editions.py` is the *declaration*
of the boundary: the capability fields, the per-capability `GATE_REASON`, and
`resolve_capabilities()`. It is not a decorator sprinkled through the codebase.
Today `resolve_capabilities()` has two callers outside tests, and the swarm
limit (which is a limit, not a capability flag) is enforced at a third site:

| Caller | What it gates |
| --- | --- |
| `oss/overlay/app/infra/providers/factory.py:47` | `cloud_provisioning` — refuses GKE/EKS/AKS in a public build |
| `scripts/check_edition.py:67` | nothing; it *reports* what your install resolves to |
| `avaloka/coordinator.py` (`MainAvaloka.coordinate`) | the swarm's clones per mission, from `avaloka/swarm_limits.py` |
| `app/api/swarm_gate.py` (`require_swarm_console`, on every swarm/trace route) | the swarm console — Professional and Enterprise, from `editions.swarm_console_gate()` |

So `cloud_provisioning` and the swarm console are the capabilities enforced in
code at a call site today, and the swarm's clone limit the one limit. The rest of the matrix is enforced by absence — the commercial code is
simply not in the distribution — which is the stronger mechanism, but it does
mean the matrix is a statement about what ships, not a runtime permission check
you can grep for. If you need a capability gated at a call site, add the
`resolve_capabilities()` check there; the declaration already exists.

The Billing tab's plan comparison reads the same declaration.
`scripts/gen_edition_matrix.py` writes `_HOSTED` to
`ui/src/lib/edition-matrix.gen.ts`, and
`tests/contract/test_edition_matrix_generated.py` fails if that committed file
differs from what the script produces. After changing a flag, run the script and
give any new capability a label in `ui/src/lib/edition-comparison.ts`; without
one, `tsc` and the unit test both fail. The table shows what a plan includes,
which is not a statement that each capability is enforced at a call site.

Each capability also declares a status in `CAPABILITY_STATUS`: `shipped`, or
`coming` when a plan is entitled to it but this line does not carry the code yet.
A `coming` capability is shown as "Coming soon" and never as included, and no
plan gate lets it through. SSO and the audit log are `coming` on 1.6.

Two UI gates read the matrix directly, so they cannot disagree with the table:
the Reports workspace (its routes and its rail entry) reads `reports`, and the
Auto Insights view reads `auto_insights`.

For paid deployments running inside your own VPC with no route back to us,
entitlement is resolved from an **Ed25519-signed licence token, verified
offline**. Nothing phones home. The verifier is open source — publishing it is
safe, because issuing a licence needs a private key that is not in this
repository.

If a licence is missing, expired, or invalid, Avaloka **degrades to the
open-source capability set rather than refusing to start.** Your analysis keeps
working; the paid capabilities switch off. `scripts/check_edition.py:63` is the
reference implementation: a rejected token is reported as rejected, `edition`
falls back to `oss`, and the report still prints.

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
