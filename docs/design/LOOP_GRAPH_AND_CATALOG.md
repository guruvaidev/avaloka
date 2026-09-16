# Compiled Experience, Loop & Graph Engineering, and the Local Knowledge Graph

**Status:** Proposed · **Extends:** *Agent Experience as a Local Wiki* · **Companion to:** *Anonymous Usage Telemetry*

---

## 1. The idea in one paragraph

WikiSkill compiles agent experience into **skills** — better procedures, injected into a system prompt. Avaloka can compile the same experience into two further artifacts it already knows how to *verify*: the **shape of its retry loops** and the **structure of its agent graph**. That is a genuine extension of the paper rather than an application of it, and the reason Avaloka can attempt it is that `benchmarks/smoke/loops.py` and `benchmarks/smoke/graph.py` already exist to say whether a proposed change made those artifacts better or worse.

---

## 2. Loop and graph as compiled artifacts

### 2.1 What Avaloka already treats as engineering objects

`benchmarks/smoke/loops.py` checks the coder/validator cycle for two failures a functional test will not catch:

- **Non-termination** — a ceiling that does not exist, or exists and is never compared. In a pilot that is a hung session rather than an error.
- **Oscillation** — fix A, break B, fix B, break A. It *terminates*, so it looks healthy, and it produces working code roughly half the time.

`benchmarks/smoke/graph.py` parses `app/api/workflow.py` and asserts that every edge points at a node that was added, and that no node is unreachable — *"dead code that looks alive"*.

Both run statically, with no model calls. That is what makes them usable as an acceptance gate on a machine-proposed change.

### 2.2 The three-artifact loop

```
        episodes ──► wiki patterns ──► Proposer ──►  ┌── SKILL.md      (procedure)
           ▲                                         ├── loop policy   (ceilings, give-up rules)
           │                                         └── graph edits   (routing, skips)
           │                                                    │
           └──── accepted only if ALL THREE gates pass ◄────────┘
                 · benchmark score strictly improves
                 · loops.py: terminates, no oscillation
                 · graph.py: no orphans, no dangling edges
```

WikiSkill's gate is a benchmark score. Avaloka's is a score **plus two structural properties**, which matters because the failure a benchmark is worst at catching is exactly the one `loops.py` was written for: an oscillating loop scores acceptably on average while behaving badly half the time.

### 2.3 What each artifact learns

| Artifact | Learned from | Example lesson |
| --- | --- | --- |
| **Skill** (`SKILL.md`) | Recurring failure modes | "Estimate the working set from dtypes before submitting a training job." |
| **Loop policy** | Loop-shape telemetry | "The validator converges within 3 attempts on schema errors and never on type-inference disputes; give up at 3 and ask, rather than at 8." |
| **Graph edit** | Node-level outcomes | "`profile_data` → `preparation` → `coder` beats routing straight to `coder` on frames with any currency column." |

The middle row is the one the paper cannot express. A loop budget is not a prompt; it is a control parameter, and it is learnable from exactly the telemetry `loops.py` was built to read.

### 2.4 Karpathy's framing, concretely

Two of his points map directly onto this and are worth stating as design constraints rather than as inspiration.

**The eval is the product.** A self-improving loop is only as good as the gate that accepts its proposals. Avaloka's gate is unusually strong — deterministic scoring, cost accounting, plus static loop and graph checks — and every hour spent strengthening it is worth more than an hour spent on the proposer.

**Keep the autonomy slider under the operator's hand.** Proposals are generated automatically, applied never. Same policy already settled for OOM recovery and autoscaling: propose, price, and wait. A system that rewrites its own graph unattended is not a feature anyone asked for.

---

## 3. Anomaly detection

The Medium article's fourth use case, and the one Avaloka is closest to shipping — because the substrate exists and nothing reads it.

### 3.1 What already exists

`app/agents/scheduler.py` and `app/core/task_metadata.py` retain **every run of every scheduled task**. A weekly "fraud rate by ProductCD" job has a history. Nobody compares this week to the last twenty.

### 3.2 Design

Detect on three levels, in increasing order of value:

| Level | Watches | Catches |
| --- | --- | --- |
| **Pipeline health** | Duration, rows in/out, failure rate | A job that suddenly takes 4× as long |
| **Schema** | Column set, dtypes, null rates between runs | The article's opening scenario — `user_id` renamed to `userId` |
| **Result** | The metric the analysis produces | Fraud rate moving outside its own historical band |

Method: a seasonal band per series (median plus a robust scale over a trailing window), not a mean and a standard deviation — one bad run would otherwise widen the band enough to hide the next one. Minimum history before any alert fires; a series with four observations has no distribution.

**The result level is the differentiator.** Most tools watch pipelines. Watching the *number the pipeline produces* is what a data scientist actually cares about, and Avaloka is one of the few systems that knows what the number means because it wrote the analysis that produced it.

### 3.3 Honest limits

An anomaly is a question, not a finding. "Fraud rate rose 40%" may be a data bug, a real event, or a definition change upstream — and Avaloka cannot tell which. The output should be *"this moved more than it usually does; here is the history and here is what changed in the inputs"*, delivered through the same propose-don't-act policy as everything else. An alerting system that cries wolf is switched off within a month, so the bar for firing must be high and the explanation must be specific.

---

## 4. Knowledge graph for the AI data catalog

### 4.1 The problem, in the article's words

Asked where a number comes from, *"the answer is often a Slack thread, a stale Confluence page, or ask Sarah, she built that pipeline two years ago."*

Avaloka has an unusual advantage here: it is not crawling someone else's warehouse and guessing at relationships. **It generated the analyses, so it knows the edges** — which dataset came from which, on which key, feeding which model.

### 4.2 A caution about infrastructure, applied to ourselves

The telemetry review rejected the stats-agent design partly because it required a customer to run a dedicated Redis cluster. Proposing Neo4j without the same scrutiny would be inconsistent, so:

| Deployment | Store | Why |
| --- | --- | --- |
| Laptop, kind, OSS | **Embedded** — SQLite with recursive CTEs, or an in-process graph | Zero infrastructure. Lineage on a single install is thousands of nodes, not millions. |
| Cluster, Professional, Enterprise | **Neo4j** | Cypher, real traversal performance, existing operational familiarity, multi-user |

One interface, two backends, selected by deployment profile — the same pattern already used for the local model tier. **Neo4j should be an upgrade, never a prerequisite.** The catalog must work for someone who ran `install.sh` ten minutes ago, or it will not exist for the users who most need it.

### 4.3 Model

```
(Dataset)-[:DERIVED_FROM {analysis_id, at}]->(Dataset)
(Dataset)-[:HAS_COLUMN]->(Column {name_hash, dtype, role})
(Dataset)-[:LOADED_FROM]->(Connection {kind})
(Analysis)-[:PRODUCED]->(Dataset)
(Model)-[:TRAINED_ON]->(Dataset)
(Model)-[:USES_FEATURE]->(Column)
(Dataset)-[:JOINS_ON]->(Column)
(Column)-[:CLASSIFIED_AS]->(PIIKind)
```

The `PII` edge is worth noting: the classifier from PR #300 already produces it, and putting it in the graph makes **"which models were trained on data derived from a column holding personal data"** a single query. That is a compliance answer nobody can currently produce at all.

### 4.4 What it unlocks

| Question | Needs |
| --- | --- |
| "What breaks if this column is renamed?" | Reverse traversal from `Column` — **the article's opening scenario** |
| "Where did this number come from?" | Path from result back to source |
| "Which models used personal data?" | `TRAINED_ON` ∘ `HAS_COLUMN` ∘ `CLASSIFIED_AS` |
| "Has anyone analysed this before?" | Similar-dataset lookup |
| "What is safe to delete?" | Nodes with no downstream consumers |

Lineage is the prerequisite for schema-drift remediation, so this is not a parallel feature to anomaly detection — it is the thing that makes the schema level of §3 actionable rather than merely informative.

### 4.5 Privacy

The same rule as the wiki, and it bites harder here: **column names are data.** `patient_hiv_status` is a diagnosis. The graph stores a salted hash of each column name plus its dtype and role, with the plaintext resolvable only locally, using the keyed scheme already built in PR #300. The graph is local, never transmitted, and excluded from telemetry by allowlist and by test.

---

## 5. How the pieces relate

```
     Knowledge Graph            Wiki                Compiled artifacts
   (structural facts)     (behavioural prose)     (skills · loops · graph)
   what exists, and       what went wrong,        what to do differently
   what depends on what   and what worked
          │                      │                          ▲
          └──────────┬───────────┘                          │
                     ▼                                      │
                episodes ────────────────────────────────────┘
                                    gated by
                       benchmarks + loops.py + graph.py
```

Three memories at three altitudes: **structure** (graph), **experience** (wiki), **procedure** (skills, loop policy, graph edits). All local. Only compiled procedure is ever a candidate for sharing, and only by explicit contribution.

---

## 6. Sequence

| # | Build | Depends on | Why here |
| --- | --- | --- | --- |
| 1 | Lineage capture into the embedded graph | — | Everything else reads it; it is also useful alone |
| 2 | Catalog queries + "where did this come from?" in chat | 1 | First user-visible value |
| 3 | Anomaly detection over scheduled-run history | — | Independent; substrate already exists |
| 4 | Schema-drift detection + proposed remediation | 1, 3 | The article's opening scenario, finally addressable |
| 5 | Local wiki + pattern maintainer | — | Phases 1–3 of the WikiSkill design |
| 6 | Skill proposer, benchmark-gated | 5 | WikiSkill proper |
| 7 | **Loop-policy and graph-edit proposals** | 6 | The extension; needs the gate from 6 to be trusted first |
| 8 | Neo4j backend for cluster deployments | 1 | Optional upgrade, never a prerequisite |

Items 1–4 are ordinary data engineering with immediate value and no learning loop. Items 5–7 are the research bet. **Sequencing matters: the learning loop should not ship before the evaluation gate has been exercised on work whose outcome we can already judge.**

---

## 7. Risks

| Risk | Control |
| --- | --- |
| Self-modification without oversight | Propose only; three gates; operator applies |
| Oscillating loop scores well on average | `loops.py` checks structure, not just outcome — the reason it exists |
| Graph edit creates an unreachable node | `graph.py` orphan check as an acceptance criterion |
| Neo4j becomes a de facto requirement | Embedded backend is the default and stays tested in CI |
| Column names leak via the catalog | Salted hashes; plaintext local only; excluded from telemetry by test |
| Anomaly alerts become noise | High firing bar, minimum history, explanation with every alert |
