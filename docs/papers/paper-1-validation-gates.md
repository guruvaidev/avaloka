# Paper 1 — Optimizer-aware validation gates

**Target: EDBT 2027, Cycle 3 — abstract and paper due October 7, 2026.**
Research track (12p) or Experiments & Analysis. Conference Apr 6–9, 2027, Lille.

**Fallbacks in order:** PVLDB Vol. 20 (rolling), then SIGMOD 2027 Round 4.
One at a time — see [`submission-plan.md`](submission-plan.md).

## The contribution

An LLM agent that writes queries against a production database is
indistinguishable, from the database's point of view, from an unusually
prolific junior analyst. The interesting claim is that **the query optimizer
already knows things about a generated query that no amount of prompt
engineering can tell you**, and that those signals can be promoted to
first-class safety gates before execution:

* plan shape — a cartesian product where the analyst meant a join
* cardinality estimates — a predicate that will scan far more than intended
* cost estimates — the difference between a query that returns in seconds and
  one that will not return this afternoon
* static-semantic checks against the live catalog rather than the prompt's
  recollection of it

This reframes validation from "does the SQL parse" to "does the plan the
optimizer produced match what the user actually asked for". It is a database
contribution, which is why it goes to a database venue.

## Why EDBT is the right first venue

* Cycle 3 accepts **Experiments & Analysis** and **Vision** papers alongside
  full research papers, so the work does not have to be dressed as something it
  is not.
* The deadline is reachable without inventing results.
* It is an audience that will argue with the mechanism rather than be impressed
  by the demo — which is the review we want on this one.

## What exists

* The validator agent with optimizer-aware plan validation, in the shipping
  system rather than in a prototype.
* The evidence layer it sits inside: integrity checks for leakage and
  train/test contamination, a mandatory baseline, and a claim verifier that
  checks prose against computed evidence.
* Multi-engine reach — PostgreSQL, MongoDB, DuckDB, SQLite — which matters
  because a gate that only works on one engine is a trick, not a contribution.

## What it still needs — be honest about this

This is the gap between a submission and an accepted paper:

1. **An ablation.** Gates on versus gates off, over a corpus of generated
   queries, measuring caught-vs-missed and the false-positive rate. Without
   this there is no evidence the gates earn their latency.
2. **A cost model for the gate itself.** A validation step that doubles
   planning time needs to justify it.
3. **A failure taxonomy.** Which classes of generated-query error the optimizer
   catches, and — more interesting for reviewers — which it provably cannot.
4. **A baseline comparison** against parse-only validation and against
   execute-and-see-what-happens.

DAB is the natural corpus for (1) and (4): 54 queries over four engines with
official validators, so the gate's verdict can be scored against a
ground-truth-graded outcome rather than against our own judgement.

## Risks

* **"This is just EXPLAIN."** The answer has to be the ablation, not rhetoric:
  the contribution is the gating policy and its measured effect, not the
  observation that optimizers produce plans.
* **Overlap with Paper 2.** Sampling-first execution also reduces cost. Keep the
  contributions clean: this paper is about *correctness gates before
  execution*; Paper 2 is about *bounded exploration during it*. If the
  experiments start looking like the same table, they are one paper.
