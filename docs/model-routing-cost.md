# Choosing models, and what each one costs

Six of Avaloka's LLM-backed agents run on an analysis turn, and they are not
interchangeable: their prompts differ by a factor of twenty-five, they fire at
different rates, and — measured — they do not contribute equally to a wrong
answer. Putting every agent on the same model, cheap or expensive, wastes money
in one direction or accuracy in the other. (Other agents also hold models —
the result narrator, the conversational agent's intent classifier, the memory
plane — but they are not on the per-turn analysis path and are not priced here.)

This page is the arithmetic behind the default configuration. Prices are
OpenRouter list, fetched 2026-09-28; re-check before quoting them. No figure on
this page is checked by CI.

## What each agent costs to run

Prompt sizes are measured from the shipped prompts. Calls-per-turn is read off
the code path — the coder is two calls because `_generate_pseudocode` runs
before code generation.

| agent | prompt tokens | output | calls/turn |
| --- | ---: | ---: | ---: |
| **planner** | **11,377** | ~400 | 1 |
| **coder** | 6,847 | ~900 | **2** |
| validator | 2,909 | ~300 | 1 |
| visualization | 1,660 | ~300 | 1 |
| profiling | 1,024 | ~250 | 1 |
| summarizer | 447 | ~350 | 1 |

The planner carries the largest prompt in the system and fires on every turn,
including turns that never reach the coder. It is the most expensive agent to
upgrade and — see below — the least rewarding.

## What the models cost

These model ids appear **only in this document** — none is a configured default
anywhere in the tree, and `scripts/ops/verify_models.py` does not validate them.
Confirm a slug against the provider's catalogue before setting it.

| model | $/Mtok in | $/Mtok out |
| --- | ---: | ---: |
| `openai/gpt-oss-120b` | 0.03 | 0.14 |
| `openai/gpt-5.6-luna` | 0.20 | 1.20 |
| `openai/gpt-5-mini` | 0.25 | 2.00 |
| `anthropic/claude-sonnet-5` | 2.00 | 10.00 |
| `anthropic/claude-opus-5.5` | 4.00 | 20.00 |
| `anthropic/claude-opus-5` | 5.00 | 25.00 |
| `openai/gpt-5.5` | 5.00 | 30.00 |

Two things people get wrong. **`gpt-5.5` is not cheaper than `opus-5.5`** — it
is the same on input and 50% more on output. And **`opus-5.5` is cheaper than
`opus-5`** while being the newer model, so there is no reason to pin `opus-5`.

## Configurations, priced

Per 1,000 analysis turns:

| configuration | $/1k turns | vs default |
| --- | ---: | ---: |
| all agents on `gpt-oss-120b` *(default)* | **1.41** | 1× |
| coder → `gpt-5.6-luna` | **5.65** | 4× |
| coder → `claude-sonnet-5` | 46.13 | 33× |
| planner + validator → `opus-5.5` | 72.03 | 51× |
| coder → `opus-5.5` | 91.52 | 65× |
| all agents on `opus-5.5` | 192.44 | 136× |

## Where the accuracy actually is

**Provenance, stated plainly: the harness that produced this table is not in
the repository.** It was run locally on 2026-09-28 as `tests/analytics_eval/`
and never committed — `git log --all -S analytics_eval` finds only this
document. So the numbers below are a record of a measurement, not something you
can reproduce from a clone, and they should be treated as the weakest evidence
on this page. They are kept because they are the only measurement that
attributes the coding gap, and because the conclusion they support (spend on the
coder) is cheap to act on and cheap to reverse.

12 complex analytics tasks, three arms:

| arm | coding correctness |
| --- | ---: |
| Avaloka on `gpt-oss-120b` | 82% |
| **same pipeline, coder swapped to a frontier model** | **100%** |
| bare frontier model, no Avaloka | 100% |

The second and third arms are identical, which means Avaloka's blueprint,
validator and executor cost **zero accuracy**. Failure attribution over 72 runs:
blueprint 0, code generation 0, validation 0, execution 5, wrong answer 8.

**The whole coding gap is the coder model.** Upgrading the planner or the
validator buys nothing measurable and costs the most, because the planner's
prompt is the largest and runs every turn.

## The recommendation

Spend on the coder, nowhere else:

```bash
INFERENCE_PROVIDER=openrouter                 # the default
AVALOKA_CODER_PROVIDER=openrouter
AVALOKA_CODER_MODEL_OPENROUTER=openai/gpt-5.6-luna
```

That is 4× the default cost rather than 65×, and it targets the only agent the
measurement implicates. Whether it recovers all 18 points or some of them is
**not yet measured** — the 100% arm used `claude-opus-5`.

Before treating any figure in the accuracy section as settled, the analytics
harness needs to be committed and re-run against the candidate model. Until
then the reproducible model comparison in this repo is the routing-accuracy
table in
[test-reports/1.6-reliability-measurements.md](test-reports/1.6-reliability-measurements.md),
which runs against code that ships.

For the lowest-latency path regardless of model, Groq still wins:

```bash
INFERENCE_PROVIDER=groq
```

`tests/conversational/` runs in 26s on Groq and 167s on OpenRouter — roughly
6×. On the planner, which is the surface a user is waiting on, that is felt.
(That suite is in the tree, so this one is reproducible: see
[test-reports/1.6-reliability-measurements.md](test-reports/1.6-reliability-measurements.md).)
A reasonable split is Groq for the planner and OpenRouter for the coder, since
the planner is latency-sensitive and the coder is accuracy-sensitive:

```bash
INFERENCE_PROVIDER=openrouter
AVALOKA_PLANNER_PROVIDER=groq
```

## Caveats

* Prompt sizes are the *static* prompt. A real turn adds schema, sample rows
  and conversation history, so absolute costs run higher than the table. The
  *ratios* between agents hold, and those drive the decision.
* Output-token estimates are typical, not measured per model. A reasoning model
  that emits long chains will cost more than the table suggests.
* Prices move. The table is a snapshot, not a contract.
* The agent-name component of `AVALOKA_<AGENT>_PROVIDER` and
  `AVALOKA_<AGENT>_MODEL_<PROVIDER>` must match the `agent=` literal at the
  model's construction site — `PLANNER`, `CODER`, `VALIDATOR`, `SUMMARIZER`,
  `PROFILING`, `VIZ`, `MTA`. A misspelled agent name is silently ignored, not
  rejected: resolution falls through to the role and global settings
  (`resolve_provider`, `app/core/inference.py:206`).
