# Paper 3 — The system-and-lessons paper

**Target: MLSys 2027 Industry Track — due October 30, 2026.**
Conference Feb 2027, Montréal.

**Fallback:** KDD 2027 Cycle 2, Applied Data Science track (~Feb 2027), which
is built for deployed industrial systems.

## Why this venue fits an MVP

The MLSys Industry Track explicitly states that submissions **need not present
new ideas**, but should present **design methodology and detailed benchmarks
for systems that were actually built**. That is precisely the paper we can
write today without inventing anything: the system exists, it deploys, and it
has been measured.

It is also the one paper where the failures are the contribution rather than an
embarrassment to be minimised.

## The contribution

The architecture, and what it cost to make it work:

* a **LangGraph state machine** as the agent substrate — why a graph rather
  than a loop, and what that buys in inspectability
* an **MCP layer** for tool and database access, giving one interface across
  PostgreSQL, MongoDB, DuckDB, SQLite and object stores
* a **Kubernetes/Ray substrate** with size-driven routing between local and
  distributed execution
* an **evidence layer** — integrity checks, a mandatory baseline, a claim
  verifier — treated as part of the system rather than as evaluation
* **provider-agnostic model routing** with failover across Groq, OpenRouter,
  local Ollama, Bedrock, Vertex and Azure

## The lessons section — where the real value is

Every item below is a defect found by **driving a live deployment** rather than
by reading code or running unit tests. That framing is the argument: the unit
suite was green for all of them.

* **The friendliest question was the worst handled.** "Where should I start?"
  was parsed as a SQL `WHERE` clause and answered "What should i start should I
  filter by?"
* **Being classified was worse than not being classified.** The intent that
  literally means "look at my data" was the one intent that never reached the
  data; four of five equivalent phrasings answered correctly *because* they
  matched no keyword and fell through to a delegating path.
* **The product documented a phrase that did nothing.** The sampling caveat
  told users to type "run this on the entire dataset"; that phrase was routed
  to a non-delegating intent and changed nothing.
* **The commonest ML request routed to the wrong agent.** "Train a model to
  predict churn" matched `predict` before `train a model` on declaration order,
  and asked the system to run a model that did not exist yet.
* **Silent degradation on a valid key.** With `INFERENCE_PROVIDER=groq` and an
  empty Groq key but a populated OpenRouter key, the agent fell back to canned
  deterministic text with no warning. Measured effect: identical table dumps
  for four semantically different questions, and zero clarifying questions.
  With the provider correctly configured, the same four prompts produced two
  grounded answers and two clarifying questions.
* **Benchmarks that measure the wrong surface.** One conversational config
  graded the delegation handoff string — literally `"On it…"` — and scored
  0.00, indistinguishable from a deliberately sycophantic reference system.
* **Five consecutive provisioning failures that each produced a plausible
  0.00** on an external benchmark: a venv on the wrong Python, two missing
  dependencies, a `git lfs pull` that exits 0 having fetched nothing, and a
  container bind mount that showed an empty directory while the host path was
  full. None was a capability result. All four engines' scores would have been
  published as zeros.

The generalisable claim: **for agentic data systems, the failure modes that
matter are failures of what the user reads, and they are invisible to unit
tests.** Every defect above needed a real deployment and a real conversation to
surface. That is a methodology argument, and it is the paper's spine.

## What exists

* A deploying system: 19/20 pods on a local Kubernetes cluster, with the full
  upload → ask → answer path working end to end.
* **Conversational integrity**: mean quality 0.82 across six probes, bracketed
  by scripted faithful (1.00) and sycophantic (0.00) reference systems.
* **DAB**: a partial multi-engine run with official validators.
* **Tracks 1, 4, 5, 7**: adaptive sampling, swarm ablation, reliability under
  injected faults, and a model matrix spanning two and a half orders of
  magnitude of price.

## What it still needs

1. **Production numbers, or an honest statement that these are lab numbers.**
   MLSys Industry reviewers will look for deployment scale, and inventing it is
   worse than conceding it.
2. **A completed DAB run** — full 54 queries × 5 trials, or a clearly bounded
   partial with the bounds stated.
3. **Latency and cost distributions**, not means. Our observed DAB latencies
   ranged 15–159s; a mean hides exactly the tail a practitioner plans for.
4. **A before/after for the fixes**, so the lessons section is evidence rather
   than anecdote.

## Risk

The honest one: this paper is strongest if we do **not** oversell. An MVP
described as an MVP, with a rigorous failure analysis, is a credible industry
submission. The same system described as a production platform invites the one
question we cannot currently answer well — how it behaves at real scale, under
real concurrency, with real users.
