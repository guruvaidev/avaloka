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
