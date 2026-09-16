# Agent Experience as a Local Wiki — Applying WikiSkill to Avaloka

**Status:** Proposed · **Companion to:** *Avaloka Anonymous Usage Telemetry — Design*
**Source:** Tang, Rashtchian, Ferng, Tomkins, Juan, Vu — *WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution* (arXiv:2608.27454)

---

## 1. Short answer

Yes — and a **local** wiki is the better version of the idea, not a compromise forced by privacy.

WikiSkill's wiki is model-written prose about what went wrong and what worked. On a customer install that prose is *about their data and their failures*, which makes it the one artifact in this system that must never be transmitted. Keeping it local is not a limitation we accept; it is what makes the feature shippable in an open-source product at all.

That gives a clean three-way split, and each channel has a different privacy posture:

| Channel | Contents | Leaves the customer |
| --- | --- | --- |
| **Telemetry** | Counters, durations, error classes | Yes — allowlisted, anonymous |
| **Wiki** | Prose patterns about this install's failures | **Never** |
| **Skills** | Procedural markdown, no customer specifics | Optionally, by explicit contribution |

Avaloka already has three of the four components this needs.

---

## 2. What WikiSkill actually proposes

Three layers, deliberately separated:

1. **Raw execution experience** — traces of what the agent did and how it turned out.
2. **A persistent wiki** — a directory of markdown pattern pages, each documenting a failure mode or a successful strategy with an actionable workaround. Maintained by incremental, patch-based edits (append, replace, insert a span) rather than rewrites, alongside an evolution log (`logs.md`) and a skill-impact tracker (`skill-impact.md`) recording proposal diffs, validation scores and acceptance outcomes.
3. **Executable skills** — `SKILL.md` (the procedure) plus `PURPOSE.md` (which wiki patterns motivated it). A Skill Proposer reads the wiki index and the impact tracker, retrieves specific pattern pages on demand, and proposes either a new skill or a patch to one existing skill.

Reported results: evolved skills outperform the same agents without retention, **transfer across models**, and let a smaller model with skills match a substantially larger model without them.

### 2.1 The finding that shapes the design

The paper's ablation shows that giving the **inference agent** access to the wiki *degrades* final skill quality. Only compiled skills are injected into the system prompt at run time.

That single result is why this is not a memory feature. **The wiki is a compiler input, not a runtime lookup.** Avaloka already has Context Memory — `app/services/memory_plane.py`, per-thread, consulted while answering. Bolting the wiki onto that path would reproduce the configuration the authors measured as worse. The wiki runs offline, between sessions; skills are the only thing the running agent ever sees.

Confusing the two is the main way this could be implemented badly.

---

## 3. What Avaloka already has

| WikiSkill component | Avaloka today | Gap |
| --- | --- | --- |
| Raw execution experience | Telemetry JSONL (companion design); Ledger records | Traces are not retained locally in a structured form |
| Persistent wiki | — | **Missing. This is the build.** |
| Executable skills | `deploy/openclaw/skills/avaloka-discovery/SKILL.md` — already YAML-front-mattered with `name`, `description`, `agent.model`, `tools` | Only one skill; no `PURPOSE.md`; nothing writes them |
| Validation gate | `benchmarks/run.py`, five tracks, deterministic scoring, cost accounting | Ready as-is |
| Named skill targets | 11 fireflies: `data_scout`, `sampling_specialist`, `data_engineer`, `analysis_planner`, `ml_engineer`, `model_scientist`, `validator`, `finops`, `reporter`, … | Each is a natural skill boundary |

The `SKILL.md` format is already the artifact WikiSkill compiles to. That is an unusual amount of alignment for a paper published this month, and it means the work is mostly plumbing rather than invention.

The *Design Doc for Stats Agent* also already proposed the correct acceptance rule — no prompt change merges unless the DAB suite strictly improves. That rule is WikiSkill's validation step, written before we read the paper.

---

## 4. Proposed design

### 4.1 Layout

```
~/.avaloka/experience/            # local only, never transmitted
├── wiki/
│   ├── index.md                  # generated table of contents
│   ├── logs.md                   # what each evolution round found
│   ├── skill-impact.md           # proposals, scores, accept/reject
│   └── patterns/
│       ├── coder-schema-drift-on-renamed-columns.md
│       ├── mta-oom-on-wide-object-frames.md
│       └── dta-json-not-readable-from-bucket.md
└── traces/                       # raw episodes, rotated, capped
```

Skills stay in the repo under `deploy/*/skills/<name>/`, because a skill is code: reviewed, diffed, versioned, shipped.

### 4.2 Pattern page

```markdown
---
id: mta-oom-on-wide-object-frames
agent: model_training_agent
kind: failure_mode
occurrences: 7
first_seen: 2026-08-14
last_seen: 2026-08-27
confidence: 0.71
status: active          # active | superseded | retired
---

## What happens
Training is OOM-killed on frames with many text columns, at row counts that
succeed when the same columns are numeric.

## Why
Object columns cost roughly 69 bytes per row against 8 for a float, so the
working set is far larger than the row count suggests.

## What works
Estimate the working set from dtypes before submitting, and shard rather than
raising a single worker past the ceiling.

## Evidence
7 episodes; 6 recovered after sharding, 1 needed a batch reduction.
```

Structured header, prose body. The header is what makes pruning and retrieval possible; the prose is what makes it useful to a model.

### 4.3 The loop

```
episodes ──► Wiki Maintainer ──► patterns/*.md ──► Skill Proposer ──► SKILL.md patch
   ▲                                                                        │
   │                                                                        ▼
   └──────────────────── accepted only if benchmarks improve ◄────── benchmarks/run.py
```

Runs offline — nightly, or on demand via `avaloka experience evolve`. Never during a user's analysis.

- **Wiki Maintainer** reads recent episodes and applies patch-based edits: a recurrence increments `occurrences` and updates `last_seen`; a new phenomenon creates a page. Rewriting a whole page is not permitted, because it destroys the evidence trail that makes a pattern auditable.
- **Skill Proposer** targets **one** skill per round, emitting a diff plus a `PURPOSE.md` naming the patterns that motivated it.
- **Gate**: the proposal is accepted only if the relevant benchmark track **strictly improves** and no track regresses. Rejections are recorded in `skill-impact.md` with their scores — a rejected proposal is evidence too, and re-proposing it next round without that record is how the loop wastes money.

### 4.4 Two problems the paper leaves open

The authors state plainly that WikiSkill "lacks an automated mechanism to prune the wiki". On a customer install running for a year, that is not a footnote.

**Growth and staleness.** Each pattern carries `occurrences`, `last_seen` and `status`. A pattern not seen for 90 days moves to `superseded`; one whose motivating behaviour no longer reproduces moves to `retired`. Retired pages are kept but excluded from the Proposer's index, so history survives without paying context for it. A hard cap on active patterns per agent forces consolidation rather than accumulation.

**Wrong lessons.** A pattern learned from one unusual dataset can be locally true and generally false. Three controls: a pattern needs **two independent episodes** before it can motivate a skill change; the benchmark gate must clear it; and every accepted skill records the pattern ids in `PURPOSE.md`, so a regression traced to a skill can be traced further back to the pattern and that pattern retired. Without the third, a bad lesson is unfindable once compiled.

### 4.5 Relationship to telemetry

Complementary, and the boundary is firm:

- Telemetry answers *"is Avaloka working for people?"* — counters, home, anonymous.
- The wiki answers *"is this install getting better at its own work?"* — prose, local, private.

**The wiki is never an input to telemetry.** It is model-written free text, and the telemetry design's own rule is that free text is never safe to transmit. What *may* travel, with explicit consent, is a compiled skill — reviewable markdown with no customer specifics, contributed the way a pull request is.

That yields a genuinely attractive property for an open-source product: **every install learns privately, and the community optionally benefits from the lessons without anybody's data moving.**

---

## 5. The experiment worth running

The paper's headline claims are that skills transfer across models and that a small model with skills matches a large one without. Both are directly testable on infrastructure that already exists.

`track7_model_matrix` holds the task, the prompt and the deterministic scoring fixed and varies only the model, and now carries per-call cost accounting. Adding a skills-on/skills-off dimension turns it into exactly the paper's ablation:

| | no skills | evolved skills |
| --- | --- | --- |
| `gemma3:4b` (local, free) | | |
| `openai/gpt-oss-120b` ($0.037/$0.17) | | |
| `anthropic/claude-opus-5` ($5/$25) | | |

If a 3.3 GB local model with evolved skills reaches a hosted 120B's score, that is both a strong product claim and a publishable result — and the measured full-sweep cost is **$0.19–$0.49**, so it is affordable within the existing daily budget.

We already have a local-model data point that makes this plausible: on the conversational integrity probes, `gemma3:4b` matched hosted `gpt-oss-120b` at Pass@1 0.67 with no skills at all.

---

## 6. Phases

| Phase | Work | Notes |
| --- | --- | --- |
| 1 | Local trace capture and `~/.avaloka/experience/` layout | Reuses the telemetry writer; strictly local sink |
| 2 | Wiki Maintainer — pattern pages, patch-based edits, `logs.md` | Deterministic where possible; a model only for prose |
| 3 | `avaloka experience show` / `prune` / `export` | Inspection first, as with telemetry |
| 4 | Skill Proposer + benchmark gate + `skill-impact.md` | Nothing merges without a benchmark improvement |
| 5 | Skills-on/off dimension in `track7_model_matrix` | The transfer experiment |
| 6 | Optional community skill contribution | Explicit, reviewable, opt-in |

Phases 1–3 are useful on their own: a searchable local record of what has gone wrong on this install, which is a support feature before it is a learning one.

---

## 7. Risks

| Risk | Control |
| --- | --- |
| Wiki treated as runtime memory | Architecturally separate from `memory_plane`; the inference agent never reads it — this is the paper's own ablation |
| Wiki transmitted by accident | Excluded at the telemetry allowlist; a test asserts no path from `experience/` to the uploader |
| Skills drift from measured reality | Benchmark gate; `PURPOSE.md` traceability; scheduled re-validation |
| Unbounded growth | Status lifecycle, per-agent caps, `prune` |
| Learning loop spends money unattended | Offline, scheduled, budget-capped through the existing spend ceiling |
| A skill helps one model and harms another | The matrix dimension in phase 5 measures exactly this before shipping a skill as a default |
