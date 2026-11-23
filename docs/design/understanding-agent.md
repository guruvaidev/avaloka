# Understanding agent — what it is in 2.0

Status: **built** (2.0, `feat/2.0-domain-vocabulary`). Branch history: this note
replaced a proposal for a table-locating fallback that was never implemented;
that proposal is summarised in §6 and parked.

In 2.0 "the understanding agent" is **not** an agent. It is two things that
already run without a model call:

1. **The domain vocabulary** — `avaloka/domains.py`, a closed set of domains,
   each with a hint set, and a deterministic classifier over column names.
2. **Chat-voice's grounded questions** — the step-by-step prompts and quick
   replies (`app/api/chat_voice.py`, PR #525) that read the profile and ask the
   user one concrete question per step.

The vocabulary supplies the question; chat-voice asks it. A separate agent
with its own model calls is next release (§7).

---

## 1. The vocabulary

`avaloka.domains.VOCABULARY` holds sixteen domains:

healthcare, finance (banking), personal_finance, stock_market, retail
(e-commerce), telecom (churn), genomics, robotics, astronomy, manufacturing
(IoT), hr (people), education, logistics, marketing, public_sector, generic.

Each `Domain` carries:

| Field | What it is | Example (healthcare) |
| --- | --- | --- |
| `cues` | regexes matched against a normalised column name | `\bpatient`, `\badmi(ssion\|t)`, `\breadmi(ssion\|t)` |
| `aliases` | fragments a model's free text may use for this domain | `health`, `clinical`, `claims` |
| `units` | units the analysis must state before comparing | mg/dL or mmol/L for labs; days for length of stay |
| `targets` | typical outcome columns | readmission, length_of_stay, total_charges |
| `join_keys` | how rows in this domain join | patient_id, encounter_id |
| `sanity_checks` | 2–4 checks the analysis should answer or flag | "Billing per condition should be compared on the same admission type" |
| `questions` | 2–3 friendly first questions for the user | "Is each row a patient or a visit?" |

The text is published code: it names no customer, dataset, value or
organisation. The CLI package is standard-library only (S1.08), and so is this
module.

## 2. `classify(columns, sample_rows=None)`

Deterministic, no model.

- Column names are normalised (`PatientId` → `patient id`, `adj_close` →
  `adj close`) and matched against every domain's cues.
- A domain scores the number of **distinct** cues hit, so `score_1 .. score_9`
  counts once.
- The highest score wins. Confidence is `min(0.95, 0.45 + 0.12 × hits)`: one cue
  gives 0.57, two give 0.69, five give 0.95. A tie halves it.
- A read below `CONFIDENT` (0.6: one cue, or a tie) names no domain. It is
  `generic` with confidence **0.0**, the leading candidate is kept under
  `tentative`, and `reason` says why. One `Ticket` column does not make the
  Titanic passenger list a telecom dataset. No hit at all is `generic` with
  `tentative: null` and the reason "no column name matched a domain cue".
- `sample_rows` is optional and only feeds `type_check`: the names of matched
  columns grouped by whether their values are numeric or text. No value is
  copied anywhere; `hints` and `questions` are identical with or without rows.
  `tests/avaloka/test_domains.py::test_names_decide_and_sample_values_never_leak`
  plants a sentinel value and asserts it never appears in the result.

Result shape:

```json
{
  "domain": "telecom", "label": "Telecom / churn", "confidence": 0.95,
  "reason": "6 distinct cue(s) matched column names (Churn, Contract, ...)",
  "matched_cues": [{"column": "Churn", "cue": "\\bchurn"}, ...],
  "candidates": [{"domain": "telecom", "score": 6}],
  "hints": {"units": [...], "targets": [...], "join_keys": [...], "sanity_checks": [...]},
  "questions": ["Is each row a customer or a billing line?", ...],
  "type_check": {"numeric": ["tenure", ...], "text": ["Contract", ...]}
}
```

`coerce(text)` maps a free-text domain name onto the vocabulary (exact id or
label, then the earliest alias at a word start) and returns `generic` for
anything else: `"FinTech/Payments"` → `finance`, `"Media/YouTube Analytics"` →
`generic`, and `hr` is not found inside "Throughput".

## 3. Where it runs

**CLI** — `avaloka/fireflies/data_scout.py` adds `profile["domain"]` (the
`classify` result) to `data_quality.json`; the Reporter writes a `## Domain`
section into the README with the label, the confidence, the matched column
names, the first sanity check and the first question, and the narrative says
"the columns read as … data". `avaloka chat` says the first question after its
first glance when the domain is not generic.

**App** — `app/agents/profiling_agent.py::_resolve_domain` runs `classify`
first. The model (`_call_domain`) is consulted only when the deterministic
read is `generic` (no cue, one cue, or a tie). Its prompt lists the vocabulary; whatever it answers is passed through
`coerce`, so `domain["category"]` is always one of `DOMAIN_IDS`. `hints` and
`questions` always come from the vocabulary, never from the model's prose.
`domain["source"]` records who decided: `vocabulary`, `model`, or
`model_rejected` (the model answered outside the set and `generic` stands).
`incomplete_sections` lists `domain` only when the model was needed and never
answered.

The field both consumers read, in `quick_profiling_result` /
`full_profiling_result` (app) and `data_quality.json` (CLI):

```
profile["domain"]["category"]   # app: vocabulary id       (CLI: profile["domain"]["domain"])
profile["domain"]["label"]
profile["domain"]["confidence"]
profile["domain"]["hints"]["sanity_checks"]
profile["domain"]["questions"]  # the grounded first questions
```

## 4. The chat-voice hook

Chat-voice (`app/api/chat_voice.py`, on `feat/2.0-swarm-gate-and-chat-voice`,
PR #525) already reads `profiling_result["domain"]["category"]` in
`read_statistics` and treats `"unknown"` as no domain. Two one-line changes
on that branch finish the wiring once both are on `integration/2.0`:

1. In `read_statistics`, treat `"generic"` like `"unknown"`, and prefer
   `domain["label"]` over `category` for the sentence "this reads like …
   data" (the id is `stock_market`; the label is "Stock market").
2. In `prompts_after_upload`, when the domain is not generic, use
   `(prof.get("domain") or {}).get("questions", [None])[0]` as the step's
   `question` in place of the generic "what do you want to know" prompt, with
   the vocabulary's `targets` as quick replies.

This branch does not touch chat-voice: its builders are not on
`integration/2.0` yet, and the field is documented here so that either branch
can land first. `app/agents/visualization_agent.py` already treats `generic`
like `Unknown` (no domain line in the chart prompt).

## 5. Privacy: the domain is schema, and schema stays out of telemetry

The domain, its label, the matched columns, the reason, the hints and the
questions are all derived from column names, so for the standing rule they
are customer schema. They live in the profile (`data_quality.json`,
`quick_profiling_result` / `full_profiling_result`), the README, the local
understanding log (`understanding.jsonl`, the Data Scout's `decided.domain`)
and the model prompts (allowed: scope ruled 2026-10-10, the rule covers
telemetry only). They never enter a telemetry payload: every egress
projection (`avaloka.understanding_trace.egress_event`,
`app.trace.telemetry.agent_events` / `swarm_events`, the analytics export) is
a closed key set with no field a domain could travel in, and the ingest
service refuses unknown keys.

`tests/test_domain_vocabulary_telemetry_privacy.py` plants column names the
vocabulary reads as healthcare, in a distinctively named file, through a real
two-clone CLI mission and the traced app `profile_data` node, checks the local
layer carries the domain, and checks that no name, no file name and nothing
the vocabulary derived (id, label, hints, questions) is in any payload, the
ingest POST body or its store. A routed domain turns it red.

## 6. What was parked

The earlier proposal in this file described a fallback that locates the table
inside a document when a direct reader returns the wrong frame (stacked
headers, merged cells, footnotes, a decoy sheet). Its honest headline was that
most of those twelve defects are reader bugs or format holes, and only the
question-aware choice among several candidate tables needs a model. That work
belongs with the readers (`avaloka/io`, `file_handler/`) and is tracked in the
data layer, not here.

## 7. Next release

An understanding agent as its own agent, with its own model calls:

- domain relations in the lineage graph (needs this vocabulary plus a relation
  extractor);
- domain-specific analysis logic beyond the sanity checks (for example, a
  healthcare profile that stratifies by admission type on its own);
- a learned classifier, if the deterministic one's `generic` rate on real
  uploads proves high — measured first, from `domain["source"]` counts in the
  local trace (never in telemetry, §5).

Nothing in §7 is started. The vocabulary is the 2.0 value.
