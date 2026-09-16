# Three-pillar coverage: harness, learning loop, knowledge graph

The claim is that the model is the smallest part, and that the harness, the
learning loop and the knowledge graph decide enterprise outcomes. This file
records **how well the 1.6 Master Test Plan actually tests that claim**, having
executed all 126 cases against a live deployment.

Short answer: **the harness is well covered and largely proven, the knowledge
graph is real but thinly covered, and the learning loop now runs and persists —
deliberately scoped as foundation rather than as a measured improvement.**

> **Scope note.** The learning loop is in the product to carry context forward,
> so that memory-aware behaviour has somewhere to live and future work has
> something to build on. Demonstrating that it makes analyses measurably better
> is explicitly **not** a launch requirement. The case that would show it
> (`LL-05`) is kept below as a future extension, not as a blocker — and until
> someone runs it, we make no claim either way.

## What the existing plan covers

| Pillar | Cases | Result |
| --- | --- | --- |
| Harness / verification | 16 | mostly PASS |
| Knowledge graph / lineage | 2 | both PASS |
| Learning loop / memory | 1 | BLOCKED |

### Harness — covered

`DQ-01` leakage blocks training · `DQ-02` wired into the MTA path · `DQ-03`
duplicate rows across splits · `DQ-04` identifier-like feature flagged ·
`DQ-05` mandatory baseline · `DQ-07` invented number caught · `DQ-10`–`DQ-14`
PII detection, keyed hashing, join preservation · `CV-07` scorer not gameable ·
`CV-08` reply surface graded on a real question · `CV-09` a missing key is not
a 0.00 score.

That last group matters more than it looks: it tests the *evaluation system's*
honesty, not the product's. `CV-09` is the reason a crashed run cannot be
mistaken for a bad analysis.

### Knowledge graph — real, thin

`AG-22` ancestry recorded across derivations, `AG-23` PII found *through*
lineage. Both pass. `app/core/lineage.py` provides `NodeKind`, `EdgeKind`,
`Node`, `Edge` and a SQLite-backed `LineageStore`, so this is a graph rather
than a log.

Two cases is not coverage of a pillar. Nothing tests multi-hop traversal, graph
correctness after a failed transform, cross-dataset joins, or what the graph
says when a derivation is deleted.

### Learning loop — deployed, and scoped as foundation

One plan case, `CV-11` *"stated preference persists"*, was BLOCKED when the
plan was executed because no vector backend was deployed. That is now fixed.

`app/services/memory_plane.py` is substantial — 19 functions, episodic memory
with similarity search (`MILVUS_TOP_K`), preference and style-hint storage per
session, a circuit breaker with a hard retrieval timeout. The code is real.

Measured against the running deployment **after** the chart added Milvus, gave
Chroma a PVC, and the retrieval circuit breaker was raised above the time a
retrieval actually takes:

```
call 1:  3.4s  memory_context_unavailable=False  hints=2   <- survived a restart
call 2:  0.6s  memory_context_unavailable=False  hints=2   <- warm
```

The loop stores, persists across pod restarts, and retrieves. What it does
**not** have is evidence that it improves the analyses themselves — and by
design we are not claiming that. Treat the row as *working foundation*, not as
a quality result.

## Cases this plan is missing

Proposed additions, each written so it can only pass if the pillar is real.

### Learning loop — `LL-01` … `LL-06`

| ID | Feature | Expected |
| --- | --- | --- |
| `LL-01` | Vector backend deployed and reachable | The memory backend answers a health probe in the default install |
| `LL-02` | Episodic recall across sessions | A query similar to one asked in an earlier session retrieves that episode |
| `LL-03` | Stated preference persists | "Always use matplotlib, not seaborn" is honoured in a later turn and a later session |
| `LL-04` | Style hint reaches generated code | A stated code preference is visible in the code the coder emits |
| `LL-05` | *(future extension, not a launch blocker)* Second analysis is cheaper or better | The same analysis repeated on a similar dataset costs fewer tokens, fewer retries, or scores higher. Worth having when someone wants to make a quality claim; nothing today depends on it |
| `LL-06` | Degrades honestly when memory is down | With the backend stopped, the run completes and `memory_context_unavailable` is **true** — not silently absent |

`LL-01` through `LL-04` and `LL-06` are the ones that matter now: they prove the
loop is present, persistent and honest about being unavailable, which is what
"context for future extensions" requires. `LL-05` is what you would add the day
you want to claim the loop improves outcomes — a different and later question.

`LL-06` is the lesson from DEF-005: a subsystem that is down must say so rather
than vanish. It is already satisfied — `memory_context_unavailable` is now
always a bool, guarded by `tests/test_memory_plane.py`.

### Knowledge graph — `KG-01` … `KG-06`

| ID | Feature | Expected |
| --- | --- | --- |
| `KG-01` | Multi-hop traversal | Ancestry resolves across ≥3 derivations, not just parent→child |
| `KG-02` | Graph survives a failed transform | A failed step leaves no orphan edge claiming a derivation that never happened |
| `KG-03` | Cross-dataset join recorded | A join records both parents, not just the left side |
| `KG-04` | Column-level lineage | Ancestry resolves to columns, not only to datasets |
| `KG-05` | Deletion is represented | Removing a derived artifact does not silently break its ancestors' edges |
| `KG-06` | Graph answers a question the profile cannot | "Which downstream artifacts contain data derived from this PII column?" returns a computed answer |

`KG-06` is the pillar's real test: if the graph only restates what profiling
already knows, it is a log with extra steps.

### Harness — `HN-01` … `HN-03`

The harness is the best-covered pillar, but three gaps are worth closing
because each has already bitten us:

| ID | Feature | Expected | Why |
| --- | --- | --- | --- |
| `HN-01` | The grader targets the right surface | An adapter that grades a handoff string instead of an answer fails loudly | A config scored 0.00 by grading `"On it…"`, indistinguishable from a sycophantic reference |
| `HN-02` | A provisioning failure is never a score | Unfetched data, an empty mount or a missing dependency yields NOT MEASURED, never 0.00 | Five consecutive provisioning failures each produced a plausible 0.00 |
| `HN-03` | A model that cannot use the protocol is excluded, not scored | A model that emits tool calls as prose is reported as protocol-incompatible | `gemini-2.5-flash` scored 0/3 on DAB for protocol reasons, not analytical ones |

## What would strengthen the three-pillar claim

1. ~~Deploy the vector backend~~ — **done.** Milvus ships in the chart, Chroma
   persists, and the circuit breaker no longer aborts every retrieval.
2. Add `KG-06`. A graph that cannot answer a question profiling cannot answer
   is not earning its keep. This is the most valuable remaining item.
3. Add `HN-01`–`HN-03`, so the evaluation system is itself under test. Each
   comes from a failure already seen in practice.
4. `LL-05`, if and when a quality claim about the loop is wanted. Explicitly a
   future extension.

The README now describes the loop as **running, with its value still ahead of
it**, and this file is the reason. That is a deliberate scope decision, not an
oversight: the loop exists to carry context, and it demonstrably does.
