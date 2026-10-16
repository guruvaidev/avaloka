# Understanding agent — design

Status: proposal. Nothing in this document is built.
Branch: `fix/readers-for-declared-formats`.
Scope: a fallback that locates the table inside a file when a direct reader has
not. It reads no files, returns no values, and is optional in every
configuration.

Claims are marked **[verified]** when checked by reading code on this branch and
**[unverified]** otherwise. Every forward-looking number in this document is
unverified by construction; the point of the marking is that none of them should
be quoted as a measurement.

---

## 1. Problem

Avaloka has two independent file readers and they disagree about what is in a
file. **[verified]**

| | CLI | Upload |
|---|---|---|
| Entry | `avaloka/io/loader.py::load_dataset` → `_read_excel` (~line 109) | `file_handler/handler.py::FileHandler` → `excel_connector.py::ExcelConnector` → `table_shape.py` |
| Sheet ranked by | most non-empty cells (`notna().sum().sum()`) | most rows (`len(pd.read_excel(...))`) |
| Reports its choice as | `frame.attrs["sheet_name"]`, `frame.attrs["header_row"]` (1-based) | `.selected_sheet`, `.shape_report["header_row"]` |
| Multi-table aware | no | yes, but only via `load_all_tables()` |

Neither asks what the question was. Both report a single frame. The failure
mode that matters is not the exception — it is the clean frame built from the
wrong cells, because every downstream check can pass on it.

### Measured defects this agent is the fallback for

Supplied as verified measurements in the brief; not re-derived here.

| # | Defect | Raises? |
|---|---|---|
| D1 | XML through the CLI: 1,682 rows, one column named `<?xml version='1.0' encoding='utf-8'?>` | no |
| D2 | Avro through the CLI: 196 rows, column names from the binary schema header | no |
| D3 | JSON through the CLI: `UnusableDataset` | yes |
| D4 | ORC through the CLI: `ValueError` | yes |
| D5 | Two-row stacked header collapses to `Sales`, `Sales.1`, `Cost`, `Cost.1`; the years vanish and the report quotes those names | no |
| D6 | Merged header cells → `Unnamed: 2`, `Unnamed: 4` (CLI) / `column_3`, `column_5` (upload) | no |
| D7 | `Region \| 2021 \| 2022 \| 2023` → `column_1..column_4` on upload, because `table_shape._looks_like_a_label("2021")` is False so `detect_header_row` returns None. CLI gets this right | no |
| D8 | Footnote markers stay glued to labels (`Aberdeen*`, `Bath (a)`); the footnote block is read as data rows | no |
| D9 | A second table below the first contaminates it (31 rows read, table has 20). `table_shape.extract_tables` splits it correctly and is never called | no |
| D10 | `(1,234)` stays text on the CLI path; upload converts it | no |
| D11 | Synthetic SEC `Financial_Report.xlsx`: upload resolves `'Consolidated Balance Sheets'` for an income-statement question, ranking 19 rows over 17. CLI resolves it correctly (58 vs 55 non-empty cells) | no |
| D12 | A decoy sheet that is longer *and* denser than the real one defeats both readers | no |

Ten of twelve raise nothing. That is the whole argument for the broad trigger in
§2 and for a return shape that can say "unsure" in §4.

### The honest headline

Working through D1–D12 (§8), the split is:

- **four format holes** (D1–D4) the agent cannot touch;
- **three reader bugs with a correct implementation already on the other path**
  (D7, D9, D10) that must not be routed through a fallback;
- **four shape problems** (D5, D6, D8, D11) closed by a deterministic,
  non-model tier;
- **one** (D12) that needs a model and a question, and cannot be closed without
  one.

So the agent's unique contribution — the part no reader fix delivers — is
**question-aware selection among several candidate tables**, plus the discipline
of returning "unsure" rather than a frame. Everything else on the list is
reader work. The design is sized to that.

---

## 2. Trigger — the one decision deferred to the owner

The owner's ruling is **readers first; the agent helps when direct reading
fails.** This design reads "fails" as:

> the read returned a frame that is probably not the table that was in the file,
> *or* raised.

### If the owner narrows it to exceptions only

Then the trigger fires on **D3 and D4 only** — the two defects that are already
loud and that need a reader branch, not an agent. It would never fire on:

D1 (XML as one column), D2 (Avro binary names), D5 (collapsed stacked header),
D6 (merged-cell placeholders), D7 (numeric header labels), D8 (footnote
contamination), D9 (second-table overread), D10 (parenthesised negatives),
D11 (wrong sheet on the 10-K), D12 (denser decoy).

Nine of the twelve, including every silent one. An exception-only trigger scopes
the agent to precisely the cases where it adds the least. **This is the decision
the owner may still narrow; the consequence is the paragraph above.**

Everything below assumes the broad reading.

---

## 3. Trigger conditions

Evaluated against a `ReadReceipt` — what the reader returned and what it claims
about its own work. Signals are the ones that already exist; none is invented.

### Class A — sufficient alone

| id | Signal | Source | Why this threshold |
|---|---|---|---|
| A1 | the read raised | `UnusableDataset`, `ValueError`, any reader exception | no frame exists; nothing to corroborate |
| A2 | frame width == 1 and rows >= 2, for a format declared tabular | `DatasetHandle.n_cols`, `.n_rows` | a one-column "table" with many rows is a parse artefact, not a shape. Binary, so no ratio. Catches D1 |
| A3 | placeholder-name ratio >= 0.5 | `comprehension.PLACEHOLDER_NAME_PATTERNS` (`^Unnamed:\s*\d+$`, `^column_\d+$`) over resolved names | reuses `table_shape.needs_reshaping`'s existing 0.5 rather than picking a second number. Catches D6 |
| A4 | `shape_report["synthesized_names"] is True` | `ExcelConnector.shape_report` | the reader's own admission that `detect_header_row` returned None. Catches D7 |
| A5 | a resolved name is not a name | `loader._printable_ratio` < 0.7, or length > 64, or contains `<`, `?`, newline | catches D1 and D2 even when width > 1 |

A2 and A5 both fire on D1. That is fine; the trigger set is a union, not a
partition.

### Class B — requires corroboration

Fires when **two or more** B signals hold, or **one** B signal holds *and* a
question is present whose anchors appear in the raw grid but not in the resolved
frame.

| id | Signal | Threshold and reason |
|---|---|---|
| B1 | sheet choice is unjustified | more than one candidate sheet **and** either (a) the two ranking statistics disagree about the winner, or (b) top/runner-up ratio < 1.25 on the chosen statistic. Clause (a) is free — both statistics are computable from one `header=None` read — and is decisive by construction: when "most cells" and "most rows" pick different sheets, neither heuristic justifies either. D11 is 19 vs 17 rows (1.12) and 58 vs 55 cells (1.05): thin on both, and the readers disagree |
| B2 | a row-block boundary lies inside the frame | a wholly-blank row strictly inside the frame's row span, or `table_shape.extract_tables(raw)` returns more than one table for the chosen sheet. Exact, deterministic, no threshold. Catches D9 |
| B3 | a text column that is numeric | >= 90% of non-null values parse as a number after stripping `$ , % ( )` while dtype is object. Catches D10. 90% rather than 100% tolerates a footnote row inside the span |
| B4 | duplicate-suffix names | at least one name matching `^(.*)\.\d+$` whose stem also appears — pandas' collapsed-duplicate signature. Corroborated by: the row above the detected header has >= 2 non-blank cells. Catches D5. Corroboration required because legitimate files do repeat labels |
| B5 | footnote contamination | a label matching `[*†‡]$` or `\((?:[a-z]\|\d{1,2})\)$`, or a trailing run of rows where column 0 is the only populated column. Catches D8. Corroboration required because `(a)` can be part of a real name |
| B6 | row count implausible | the frame's rows plus `dropped_leading_rows + dropped_empty_rows` does not account for the sheet's filled rows, by more than 10% | the 10% is slack for trailing blanks; **[unverified]** against a real workbook |

### Explicit non-triggers

- low quality score, high null ratio, skew — these grade the data, not the read;
- width 1 with 1 row (a scalar export) — genuinely that shape;
- a single-sheet file with a located header and no placeholder names;
- CSV/TSV/Parquet reads yielding >= 2 real names: nothing to adjudicate;
- **name disagreement on a self-describing format** (Parquet, Delta, Iceberg,
  and Avro read by a correct reader). The schema is in the file. A disagreement
  there is a reader bug, and firing an agent on it hides the bug.

### Firing policy, in order

1. Any Class A → fire.
2. Two or more Class B → fire.
3. One Class B plus a question-anchor mismatch → fire.
4. Otherwise → do not fire. Record `fired: false` with the signals evaluated, so
   a non-fire is as inspectable as a fire.

---

## 4. Interface

### Why a cell grid and not a file path

1. **It stops the agent becoming a third reader.** The two existing readers
   already disagree; a path-taking agent would add a third opinion about how to
   open a file. A grid is the input both readers already have, so the agent
   adjudicates instead of re-reading.
2. **Dependencies.** A path forces its holder to carry openpyxl / lxml /
   fastavro. The CLI's eight are fixed **[verified: `pyproject.toml` and
   `tests/avaloka/test_cli_dependency_isolation.py`]**. A grid is
   `list[list[str | float | None]]` — stdlib.
3. **Egress control.** A grid is sliceable and cappable: head and tail rows,
   truncated cell text, dropped columns, a bounded sheet count. A path cannot be
   truncated, and §7 turns on being able to bound exactly what leaves.
4. **Determinism and caching.** A grid is a value; hash it. A path's bytes can
   change under the cache.
5. **Testability.** A grid literal is a fixture. A path needs a workbook written
   by openpyxl, which is an extra.

### Request

```
UnderstandingRequest
  file_sha256: str            # DatasetHandle.sha256 (see caveat in §6)
  declared_format: str        # "excel" | "csv" | "xml" | ...
  sheets: list[SheetGrid]
      name: str
      index: int
      n_rows_filled: int      # the sheet's own dimension, not the frame's
      n_cols_filled: int
      cells: list[list[CellValue]]              # raw, header=None, pre-reshape
      merged_ranges: list[tuple[int,int,int,int]]   # when the format carries them
      truncated: bool
  receipt: ReadReceipt        # chosen_sheet, header_row, resolved_columns,
                              # n_rows, n_cols, shape_report, exception_type
  triggers: list[str]         # the ids from §3 that fired
  question: str | None        # the mission goal, verbatim
  budget: {deadline_ms, max_cells, max_model_calls, max_usd}
```

Grid caps (defaults): <= 12 sheets ranked by filled cells, <= 200 rows x 100
columns per sheet, head 100 + tail 20 when taller, cell text truncated to 200
characters. The caps live in the request so the agent can set `truncated` and
the caller knows the verdict was formed on a slice.

### Response

```
UnderstandingVerdict
  status: "resolved" | "unsure" | "refused" | "timeout" | "unavailable"
  tier: "T0" | "T1" | "T2"
  candidates: list[TableCandidate]
      id: str                 # stable: f"{sheet}:{r0}-{r1}:{c0}-{c1}"
      sheet: str
      row_span: (int, int)    # 0-based half-open, into the grid as sent
      col_span: (int, int)
      header_rows: list[int]  # more than one for a stacked header
      column_names: list[str]
      synthesized: list[bool] # per column, True when no grid cell backs the name
      orientation: "rows" | "columns"
      role: "primary" | "secondary" | "footnotes" | "notes"
      name: str               # human label
      confidence: float
      evidence: list[str]     # each item cites grid coordinates
  selected: str | None        # candidate id; only when a question was supplied
  selection_evidence: list[str]
  rationale: str              # <= 600 chars
  warnings: list[str]
  model_id: str | None
  cost_usd: float
  latency_ms: int
```

**The verdict carries coordinates and header text. It never carries data
values.** The caller re-slices its own grid by the spans. A model therefore
cannot introduce a number into the pipeline — it can only point at cells the
caller already has. This is the single most important property of the shape.

### Validation, before the caller may adopt anything

1. Every span is within the grid's bounds as sent.
2. Every `header_rows` index lies inside `row_span`.
3. For each column, `column_names[i]` equals the grid cell at
   (header row, column) after whitespace normalisation — or, for a stacked
   header, the ordered join of that column's cells across `header_rows` — or
   `synthesized[i]` is True.
4. `selected`, if set, names a candidate in `candidates`.

Any violation downgrades `status` to `unsure` and the verdict is not adopted.
This is what makes "the agent hallucinated a header" a mechanical check rather
than a matter of trust (§9, §10).

### What the caller does

| status | Caller behaviour |
|---|---|
| `resolved`, confidence >= 0.75, validation clean | adopt the span; record `applied: true` and the candidate id in the run manifest; if the adopted sheet differs from the reader's, say so in the report header |
| `resolved`, 0.5 <= conf < 0.75 | adopt only if T0 independently produced the same span; otherwise treat as `unsure` |
| `unsure`, or conf < 0.5 | keep the reader's frame; surface the candidates to the user as a choice; mark the run `understanding: advisory` |
| `refused` | keep the reader's frame; record the reason; do not retry on another provider |
| `timeout` | keep the reader's frame; never adopt a partial verdict |
| `unavailable` (no provider; hermetic run) | keep the reader's frame; `understanding: skipped` — not a failure |

A fallback that silently returns a guess is the defect class being fixed, so:
adoption is always recorded and attributable, and no non-`resolved` status ever
changes the frame.

---

## 5. Several tables in one document

Multi-frame output is the default, not a mode. `candidates` is a list; a
single-table file yields a list of one.

**Identity.** `sheet:rowspan:colspan` — deterministic, dedup-able, and
re-derivable by the caller. Note it embeds the sheet name, which is customer
data; see §9 for what that means for telemetry.

**Naming**, in precedence order: (1) a title cell immediately above the
candidate's header rows in the leftmost populated column; (2) the sheet name,
when the candidate is the only table on its sheet; (3) `"<sheet> table <n>"`.

**Selection by question, without inventing an answer** — three steps,
deterministic first:

1. **Lexical anchoring (T0, no model).** Tokenise the question, drop stopwords,
   keep 4-digit years and other numerals plus content words. For each candidate,
   count anchors present in its header cells and its first-column labels.
   `comprehension._label_words` and `_label_anchor` already do this job on the
   grading side **[verified]** and should be the same code. A candidate
   containing every anchor when no other does is selected at confidence 0.9 with
   no model call. **D11 resolves here**: "net income" and "2023" appear in the
   income statement's row labels and not in the balance sheet's.
2. **Model adjudication (T1/T2)** only when step 1 ties or finds nothing. The
   model sees candidate headers and first-column labels — not the value cells —
   and returns a candidate id plus evidence coordinates. It may answer `none`.
3. **Ambiguity is an outcome.** Two candidates still standing → return both with
   `selected: null`. A question with two plausible tables should score as
   "asked", not "guessed".

The agent returns a span; the existing deterministic pipeline computes the
figure from the adopted cells. "What was net income in 2023" is answered by
Avaloka reading the right cells, which keeps `score_expected_answer` a real
check on the pipeline rather than a check on a model's recall.

---

## 6. What blocks, what runs behind, what is cached

| Tier | Blocks? | Budget | Notes |
|---|---|---|---|
| T0 deterministic | yes | p50 50 ms, p99 250 ms on a 200x100 grid; hard bail at 1 s | pure pandas/python over a grid already in memory. **[unverified]** |
| T1 local model | yes | 8 s deadline (`AVALOKA_UNDERSTANDING_TIMEOUT_S`) | reached only when T0 is inconclusive |
| T2 hosted model | yes | same deadline | reached only when no local model answers |

The 8 s figure is a starting point, not a finding: it is chosen to sit under the
noise floor of a mission that already takes tens of seconds while leaving room
for a 3B model's time-to-first-token on CPU at a ~4k-token prompt.
**[unverified — measure before building T1.]**

**Nothing is async in v1, deliberately.** A verdict landing after the planner has
run would mean re-planning: lane, split and target are derived from the frame.
Re-planning on a late verdict is a separate design and is out of scope. The one
background pass worth having is a **warm cache**: on upload, run T0 over all
sheets and store the verdict by sha, so the blocking path is usually a cache hit.

**Cost.** <= 1 model call per cache key, <= 2 if the first fails validation.
Prompt bounded by the grid caps. Hard cap: the agent may not spend more than 2%
of `Budget.limit_usd` and must not run at all when `limit_usd` is 0. Cost is
attributed through the existing ledger mechanism. Absolute per-mission cost is
**[unverified]**.

**Cache key**: `file_sha256 + grid_caps_version + prompt_version + model_id +
question_anchor_set`. Keying on the anchor set rather than the raw question
means "net income 2023" and "what was the net income in 2023?" share an entry.

**Caveat on `sha256`.** `DatasetHandle.sha256` exists and is already computed,
but `loader._sha256` hashes only the **first 64 MiB** (`limit=64*1024*1024`)
**[verified]**. Two large workbooks differing only past that boundary collide.
Either add file size and mtime to the cache key, or state that >64 MiB workbooks
are out of scope. This design assumes the former.

**Cache location**: `~/.cache/avaloka/understanding/<key>.json` on the CLI; the
run's artifact directory on the upload path. The stored verdict contains sheet
names and header text, so it is customer data: local only, never uploaded.

---

## 7. Hermeticity, the deterministic tier, and provider policy

### Optional in every configuration

The subsystem is off unless `AVALOKA_UNDERSTANDING=1` (or `live=True` in the
benchmark). With every key unset, T0 runs — pure python, no network, no new
dependency — and T1/T2 report `unavailable`. The suite runs and scores, and the
comprehension family still produces its numbers.

The SKIP convention is already established: a `TaskResult` whose `error`
contains `"unavailable"` is how `avaloka/benchmark/__main__.py` and
`avaloka/cli.py` separate "could not run" from "ran and scored zero"
**[verified, documented in `avaloka/benchmark/conversational.py`]**. Use it;
do not invent a second convention.

### Provider policy — reuse, do not re-implement

`avaloka/benchmark/conversational.py` already implements the owner's ruling
exactly **[verified]**:

- `LOCAL_MODEL_PREFERENCE = ("qwen2.5:3b-instruct", "gemma3:270m", "gemma3:1b",
  "gemma3:4b", "gemma4:e4b")` — the owner's first choice, then Gemma from
  smallest upward;
- `_local_model()` reads the server's own tag list, so nothing is assumed
  installed;
- `GROQ_MODEL = "openai/gpt-oss-120b"` at `GROQ_ENDPOINT` as the hosted
  fallback only;
- `_refuse_anthropic()` rejects an Anthropic-looking model override rather than
  honouring it;
- transport is stdlib `urllib`, imported inside the live helpers — **which is
  how a model tier is added without touching the eight dependencies.**

These should be lifted into a shared `avaloka/providers.py` and used by both
callers. That is a refactor of existing code, not a new dependency.

**`avaloka/llm/client.py` must not be used.** It is Anthropic-only, keyed on
`ANTHROPIC_API_KEY`, with `llm = ["anthropic>=0.40"]` as its extra
**[verified]**. Routing the understanding agent through it would violate the
standing ruling on the first call.

**Licensing.** A Gemma model is reached only through a local Ollama server the
operator installed. No weights are vendored, redistributed or linked, and
nothing is added to `requirements*.txt`, so `NOTICE`'s per-dependency table
gains no row. If a Gemma derivative were ever bundled into an image or made a
default download, the Gemma Terms of Use analysis must be done at that point.
**This document has not done it and must not be read as having done it.**

### The deterministic tier — what T0 can and cannot do

**Can**, with no model call:

- **split a sheet into its real tables** — `table_shape.extract_tables(raw,
  min_rows=2)` already does this correctly and the single-frame readers simply
  never call it **[verified]**. D9 is a missing call site, not a comprehension
  problem;
- enumerate candidate sheets with **both** ranking statistics and detect
  disagreement (B1);
- detect a stacked header: two consecutive rows where the upper has fewer
  non-blank cells and its blanks sit under a merged range, or where the lower
  row's names collide pairwise;
- **join stacked header rows** into `Sales 2021` style names — mechanical, and
  it closes D5 outright;
- forward-fill a header across merged blanks — closes D6;
- accept a candidate header row whose non-label cells are all 4-digit integers
  in a plausible year range when the row below is numeric — closes D7;
- strip footnote markers into a sibling field and cut the footnote block where
  column 0 is the only populated column — closes most of D8;
- parse `(1,234)` as `-1234` — `table_shape._as_numeric` already does it on the
  upload path **[verified]**; closes D10;
- lexical question-to-candidate anchoring (§5 step 1) — closes D11.

**Cannot**:

- choose between two tables both plausible for a question with no lexical
  overlap ("how did we do last year" over unnamed sheets);
- read a prose caveat that governs a table ("the table below excludes
  discontinued operations");
- distinguish a real column named `2021` from a stray year in a note row when
  the layout is genuinely ambiguous;
- recognise a transposed table reliably — detectable in principle, but the
  heuristic is weak and being wrong is total;
- pick the right sheet when the decoy is longer **and** denser (D12).

**T0 is most of the value and none of the risk.** The model tier exists for D12
and for the residual of D8, and only when a question is present.

---

## 8. Which reds this closes, and which are reader bugs

| # | Agent closes it? | Where it should actually be fixed |
|---|---|---|
| D1 XML as one column | **No.** Triggers (A2, A5); honest verdict is `refused` — a grid cannot be produced from XML by a reader with no XML branch | **Reader.** `load_dataset` needs an XML branch or must raise. `file_handler/xml_connector.py` already exists. A plausible frame from a file it cannot parse is the bug |
| D2 Avro binary names | **No**, same reason | **Reader.** `file_handler/avro_connector.py` exists; the CLI needs a branch or a refusal |
| D3 JSON `UnusableDataset` | **No** | **Reader.** Loud and roughly correct today; needs a JSON branch (`file_handler/json_connector.py` exists) |
| D4 ORC `ValueError` | **No** | **Reader.** pyarrow is already one of the eight and reads ORC |
| D5 stacked header → `Sales.1` | **Yes, at T0** — mechanical header joining, no model | The primitive belongs in both readers; the agent is where it is implemented, the call site is the only question |
| D6 merged cells → `Unnamed: 2` | **Yes, at T0** | Reader, same primitive |
| D7 `2021` header → `column_1..4` | **Yes, at T0** | **Reader bug, squarely.** A one-line predicate (`_looks_like_a_label` rejects ints) makes `detect_header_row` return None. The CLI already gets this right, which proves it is a bug and not a hard problem. Do not route it through a fallback |
| D8 footnotes | **Partly, at T0** — marker stripping and block cutting are mechanical; deciding whether `(a)` is a marker is the residual the model may advise on | Reader for the mechanical part |
| D9 second table contaminates the first | **No agent needed** | **Call site.** `extract_tables` already splits it correctly. The clearest "do not paper over" item on the list |
| D10 `(1,234)` stays text on the CLI | **No agent needed** | **Reader parity.** The CLI needs `reinfer_dtypes`, which the upload path already has |
| D11 wrong sheet on `Financial_Report.xlsx` | **Yes** — and this is the case that needs the agent's *shape*, though T0's lexical anchoring resolves the measured instance without a model | Readers should also stop ranking on a single statistic, and should report their runner-up as well as their choice — neither does today |
| D12 denser decoy | **Only with a question, and only at T1/T2.** With no question: `unsure`, both candidates returned | Nothing in the reader can fix it; size heuristics are exhausted by construction |

Three of these (D7, D9, D10) have a working implementation on the other reader
and the fix is to call it. Shipping an agent that masks them would make the
codebase worse while the scorecard looked better. §10 contains the control that
makes masking visible.

---

## 9. Ownership

**Recommendation: a shared resolver that takes a cell grid, living at
`avaloka/understanding/`, with the grid *producer* on each side of the fence.**

The resolver is pure: grid in, spans out. Its dependency set is stdlib plus
pandas, which is already one of the eight. The part worth reusing is
`table_shape.py`, which is also pandas-only — but it lives in `file_handler/`,
and `tests/avaloka/test_cli_dependency_isolation.py::test_importing_the_cli_does_not_import_the_application`
asserts that importing `avaloka.cli` leaves **no `app.*` and no
`file_handler.*`** in `sys.modules` **[verified — note the brief mentions only
`app.*`]**. So the shape primitives must move rather than be imported. Moving
is right; mirroring them would create exactly the third reader this design
exists to avoid. Concretely: move `table_shape.py` to
`avaloka/understanding/shape.py` and leave `file_handler/table_shape.py` as a
re-export shim.

Against the three questions asked:

- **Does it keep S1.08 intact?** Yes, in the form the test enforces. The arrow
  points `app/` → `avaloka/`, and that direction is already in use:
  `app/analytics/schema.py:48` imports `avaloka.telemetry.version`
  **[verified]**. `avaloka.cli` imports nothing from `app` or `file_handler`.
- **Does it keep the eight at eight?** Yes. T0 needs pandas only. T1/T2 use
  stdlib `urllib`, exactly as `conversational.py` already does. The `.xlsx`
  grid producer needs openpyxl, which stays in the `benchmark` extra and is
  imported lazily with a message. **Finding:** `loader._read_excel` already
  calls `pd.read_excel`, which requires openpyxl — so the CLI's xlsx path
  already depends on an undeclared extra today. The dependency test does not
  catch it, because it checks the declared list and module-level imports, not
  what a runtime code path needs. Pre-existing, worth its own bug.
- **Does it avoid a third reader?** Yes, because the resolver never opens a
  file. Each reader keeps its own I/O and gains one call. Both already have the
  grid in hand: `loader._read_excel` reads `sheet_name=None, header=None` into
  `book`, and `ExcelConnector._load_df` already holds `raw` **[verified]**.

### What I would not do

- **Not put it in `app/`.** The upload path would get it and the CLI would not,
  and four measured defects (D1, D2, D5 on the CLI side, D10) live on the CLI.
  A fix reaching one of two readers leaves the disagreement in place, which is
  the root problem.
- **Not route it through `avaloka/llm/client.py`.** Anthropic-only; against the
  standing ruling.
- **Not add a dependency for it** — not openpyxl, not an HTTP client, not a
  model SDK.
- **Not make it a LangGraph node in `app/agents/`.** It must run with no graph,
  no checkpointer and no app import, or the CLI cannot use it.
- **Not let it return values.** Spans only.
- **Not build a fourth table IR.** A candidate is spans plus header text over
  the caller's own grid. It is not a semantic vocabulary and must not grow into
  one; if a semantic layer is ever wanted here, the existing
  `app/agents/blueprint_contract.py` is the thing to promote.

---

## 10. How it would be graded

Ground truth already exists per dataset (`DatasetSpec.ground_truth` →
`GroundTruth`), and the comprehension scorers already read it **[verified]**.

### New `Expectations` fields

| Field | Type | Purpose |
|---|---|---|
| `understanding_expected_to_fire` | `Optional[bool]` | three-valued. True: the reader misreads this file, so the trigger must fire. **False: the reader reads it correctly, so the trigger must not fire** — the negative control. None: not graded |
| `understanding_expected_candidate` | `Optional[str]` | the sheet plus row span a correct verdict must select |
| `understanding_max_tier` | `Optional[str]` | the most expensive tier allowed. A defect T0 should close cannot be scored green via a model call |

`understanding_max_tier` is the field that keeps the deterministic tier honest;
without it, every D5–D11 row above can quietly become a Groq call.

### New scorers in `avaloka/benchmark/comprehension.py`

Reading a new artifact `understanding.json` written beside `data_quality.json`.

| Scorer | Weight | Asserts |
|---|---|---|
| `score_understanding_trigger` | 2.0 | fired == `understanding_expected_to_fire`, **both directions**. A false positive on a clean file is a failing check, not a note |
| `score_understanding_selection` | 2.0 | the adopted candidate equals the expectation |
| `score_understanding_tier` | 1.0 | tier <= `understanding_max_tier` |
| `score_understanding_no_invented_names` | 2.0 | every adopted name is backed by a grid cell (the §4 invariant, graded rather than trusted) |
| `score_understanding_did_not_break_a_good_read` | 3.0 | for any task with `understanding_expected_to_fire = False`, the comprehension score with the agent enabled is not lower than without |

The negative control must appear on at least one clean CSV task and one clean
single-sheet workbook task, or it is not a control.

### Telling "the agent fixed it" from "the reader never needed help"

Run each comprehension task in **two arms** — reader-only
(`AVALOKA_UNDERSTANDING=0`) and reader+agent — and report both scores plus the
delta.

| reader-only | agent arm | `fired` | Reading |
|---|---|---|---|
| red | green | true | **the agent fixed it**; `understanding.json` must show `applied: true` and name the candidate |
| green | green | false | the reader never needed help. Correct — this is the control |
| green | green | true | **fired unnecessarily.** Scored as a `score_understanding_trigger` failure even though the end score is fine: it is latency, cost and egress for nothing |
| green | red | true | **the agent broke a working read.** The worst outcome, and the thing that would make the subsystem net-negative — hence weight 3.0 |

The two-arm run also makes a papered-over reader bug permanently visible: D7's
reader-only arm stays red forever until `_looks_like_a_label` is fixed, no
matter how green the agent arm is.

### Reproducibility

The arms are comparable only if the agent arm is deterministic. So: pin
`temperature: 0.0` (as `conversational.py` already does **[verified]**), pin
and record the model id, carry the subject in `TaskResult.source_scheme` (the
field already exists for exactly this "which subject produced this score"
purpose **[verified]**), and never average a T1 arm with a T0 arm under one
number. A hermetic scorecard reports T0 only; a `--live` scorecard reports
both, labelled.

---

## 11. Risks and failure modes

| Risk | Mechanism | Mitigation |
|---|---|---|
| Hallucinated header | the model emits a plausible name not in the file | coordinates-only return; every name must be a grid cell or an ordered join of header-row cells; violation → `unsure`. Graded by `score_understanding_no_invented_names` |
| Hallucinated figure | the model answers the question instead of locating the table | the response schema has no value field. A model that answers is unparseable, not wrong |
| Non-determinism breaks the benchmark | sampling, model churn, provider A/B | temperature 0, pinned model id recorded per run, hermetic default is T0-only, arms labelled by tier. **Residual:** a provider silently changing weights behind a model id. Unmitigable; detect it by keeping a golden verdict per fixture sha and failing the suite when it moves |
| Cost on a large workbook | 40 sheets x 50k cells | grid caps (§4), one model call per cache key, the 2%-of-budget cap, and T0 first — which usually terminates. A workbook beyond the caps is answered `unsure` with candidates, never silently sampled |
| Latency on the blocking path | a 3B model on CPU | 8 s deadline, T0 first, warm cache at upload. **[unverified — measure]** |
| **Customer cells leave the machine** | a model call inherently sends cell contents somewhere: T1 to localhost, T2 off-box | **T2 is opt-in per install and off by default.** T2 is the only tier that sends customer cells off the machine; T1 on localhost is not egress. The control is topological — keep the model local — exactly as the standing constraint says. `redact.scrub` must **not** be cited as the control here: a redacted header-detection prompt cannot detect headers, so redaction is the wrong tool for this lane |
| Telemetry leaks customer data | an eager implementation logs the chosen sheet name and columns | **Allowlist, not redaction.** May leave: the trigger ids that fired (closed vocabulary), tier, status, confidence bucketed to 0.1, latency, cost, candidate count, grid dimensions, `applied`. **May not leave: sheet names, column names, cell values, file names, the question, the rationale text, or the file sha** (a file identifier that joins across installs). Candidate ids and rationales are local-artifact-only |
| The agent becomes a third reader | scope creep — "just let it take a path for xlsx" | the request type has no path field, and that absence should be a test |
| A trigger masks a reader bug | D7 gets "fixed" by the agent and the predicate bug stays | `understanding_max_tier` plus the reader-only arm, which stays red |

---

## 12. Open questions for the owner

1. **The trigger definition.** Broad (§2) or exception-only. Narrowing it loses
   nine of twelve measured defects, including every silent one, and leaves the
   agent firing on exactly the two that need a reader branch instead. This is
   the decision this document flags as still the owner's.
2. **T2 (Groq) default.** Off by default is the recommendation, on privacy
   grounds. The owner may want it on for the hosted product, where cells already
   traverse the cluster.
3. **Does adopting a different sheet need interactive confirmation on the CLI?**
   It changes the answer, and a silent change of sheet is a different kind of
   surprise from a silent misread.
4. **Moving `table_shape.py` into `avaloka/`.** The right home for a pure shape
   library and the only way the CLI can use it, but it touches the upload path's
   import graph and its test files. Needs a call on sequencing against the
   branch reality (contract on 1.6, benchmark on 1.7, telemetry on neither).
5. **Two-arm benchmark cost.** Doubling the comprehension family's runtime buys
   attribution. Limit the duplicate arm to the negative controls, or pay for all
   of them?

---

## 13. Verification status

**Verified by reading code on this branch:** both readers' sheet-ranking
statistics and the stale `_read_excel` docstring that contradicts its own code;
`frame.attrs` keys; `shape_report` keys including `synthesized_names`;
`needs_reshaping`'s 0.5 threshold; `_looks_like_a_label` rejecting ints and
`detect_header_row`'s 0.7 label ratio; `extract_tables` existing and not being
called by either single-frame reader; the comprehension scorers and the
`Expectations` / `GroundTruth` / `CheckResult` / `TaskResult` shapes; the eight
declared dependencies and the test that pins them, including that it forbids
`file_handler.*` as well as `app.*`; `conversational.py`'s provider policy,
local preference order, Groq model and endpoint, `_refuse_anthropic`, stdlib
`urllib` transport, hermetic-by-default and the `"unavailable"` SKIP convention;
`avaloka/llm/client.py` being Anthropic-only; `loader._sha256`'s 64 MiB limit;
`app/analytics/schema.py:48` importing `avaloka.telemetry.version`;
`avaloka/serving.py:263` lazily importing `app.infra`.

**Not verified — did not run anything:** every latency and cost figure; that the
grid caps are sufficient for a real 10-K; that `extract_tables` resolves the
specific 31-vs-20 fixture; B6's 10% slack against a real workbook; the licence
analysis for any bundled Gemma; whether moving `table_shape.py` breaks the
upload path's tests; and — the load-bearing one — **whether a 3B local model can
select among candidate tables at usable accuracy.** The model tier should not be
built until that is measured, because if it cannot, the design reduces to T0
plus an honest `unsure`, which is still worth shipping.
