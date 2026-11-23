# Paper 2 — Sampling-first agentic execution

**Target: PVLDB Vol. 20 (rolling monthly, ~Oct 1 for the next cycle) as a short
or vision paper**, or an ICDE 2027 round (Nov 11, 2026).

Not EDBT — that slot belongs to Paper 1, and two papers from the same system at
the same venue in the same cycle invites the "same contribution twice" reading.

## The contribution

An agent exploring an unfamiliar dataset runs many more queries than a human
would, most of them to orient rather than to answer. Running each at full
fidelity is the dominant cost of agentic analysis, and it is mostly wasted:
the answer to "what does this column look like" does not change between a
sample and the full table.

The claim is that **exploration should be sampled by default and full-fidelity
by exception, with the boundary decided by measured dataset size and stated to
the user** — BlinkDB-style bounded exploration, applied to agent behaviour
rather than to interactive querying.

The parts that make it a paper rather than an optimization:

* a **size-aware execution profile** that chooses fidelity and execution
  substrate (local vs. distributed) from measured bytes, not from a guess
* **sampling that announces itself** — every sampled answer says it is sampled,
  with a documented phrase that escalates to the full dataset
* **error bounds**, so a sampled answer is a claim with a confidence rather
  than a smaller number presented as the truth

That last point is where the honesty requirement and the cost requirement meet,
and it is the part reviewers will care about.

## What exists

* `decide_execution_profile`: size thresholds driving fidelity and
  `local` vs. `k8s-ray` routing, emitting a user-visible recommendation with
  size, ETA and a cost note.
* Adaptive sample-size tiers, and a sampling portfolio (stratified, random,
  reservoir, domain-specific) rather than a single strategy.
* **Track 1 (adaptive sampling)** already measures quality retained vs. data
  scanned vs. cost across strategies against full-data truth — the core table
  this paper needs.
* A live transcript where the system volunteers its own sampling caveat
  unprompted.

## What it still needs

1. **The defaults are currently wrong at the small end and we should say so.**
   `SMALL_THRESHOLD` is defined and not used in the fidelity decision, so
   anything under 1 GB is sampled — including a 12-row file. A paper arguing
   for principled fidelity selection cannot ship with a threshold that samples
   twelve rows. Fix first, then measure.
2. **Error bounds on sampled answers**, not just quality-retained ratios.
3. **A cost model with real prices** rather than ratios, so "cheaper" is a
   number a practitioner can act on.
4. **An escalation study**: how often does a user, shown a sampled answer,
   ask for the full run? That is the honest measure of whether sampling-first
   is acceptable in practice rather than merely cheaper.

## Risks

* **"Sampling is not new."** Correct, and the paper must concede it in the
  first paragraph. What is new is sampling as an *agent execution policy* with
  a disclosure contract attached, not sampling as a query technique.
* **Overlap with Paper 1** — see the note at the end of that file.
