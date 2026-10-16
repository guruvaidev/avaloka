# The Avaloka Benchmark

A defensible answer to one question: **does the Avaloka CLI actually analyse and
train correctly, from any source?** It exercises the whole stack — source
connectors, workload routing, the firefly swarm, validation, economics and the
multi-Avaloka coordinator — against tasks with **known ground truth**, and
*scores the on-disk deliverables* rather than trusting the narration.

```bash
python -m avaloka.benchmark run                    # synthetic tasks, all sources
python -m avaloka.benchmark run --kaggle           # + downloaded Kaggle datasets
python -m avaloka.benchmark run --family data_science --json scorecard.json
python -m avaloka.benchmark run --html scorecard.html   # something to look at
python -m avaloka.benchmark serve --port 8765       # read it in a browser
python -m avaloka.benchmark run --edgar            # + one real SEC filing
avaloka benchmark                                  # same thing, via the CLI
avaloka benchmark serve --port 8765
```

Hermetic by default: with every provider key unset, `run` calls no network and
no model. `--kaggle` and `--edgar` are the only flags that reach out; `--live`
is the only one that talks to a model.

The spreadsheet and format datasets need readers the CLI itself does not ship
(the CLI's runtime dependencies are eight and contractually fixed), so:

```bash
pip install -e '.[benchmark]'     # openpyxl, fastavro, lxml
```

## Looking at the results

| Surface | What it gives you |
|---|---|
| terminal | pass/fail per run, failed checks inline |
| `--json` | the scorecard, with `schema_version`, the **commit SHA** and `generated_at` |
| `--html` | one self-contained page: failures first, every check with its detail, and resolved column names beside the ground-truth ones |
| `serve` | the same page over loopback-only HTTP, so it can be read in a browser |

The commit and schema version are in the JSON because a number nobody can tie
to a commit is not a result — which is a named finding in the paper audit.

## Families, grounded in ground truth

### Data engineering
| Task | What it proves |
|---|---|
| `de_ingestion_parity_churn` | The *same* profile comes out of a file, a SQLite table and object storage. |
| `de_data_quality_dirty` | Constant / high-missing / high-cardinality columns are detected. |
| `de_workload_routing_sampled` | A >100k-row dataset is routed to the `sampled_online` lane. |
| `de_batch_exact_convergence` | The Ray map-reduce converges to the **exact** full-dataset profile (mean/min/max), no sampling error. |

### Data science
| Task | What it proves |
|---|---|
| `ds_classification_churn` | The trained model **beats a naive baseline** (measured here, independently). |
| `ds_regression_housing` | Regression clearly beats the mean predictor; packaging clone stays dormant. |
| `ds_leakage_detection` | Planted leakage is caught, verdict is `fail`, deployment is capped at Level 1. |
| `ds_analysis_gating_and_swarm` | Analysis is a research artifact (Level 1); the multi-Avaloka fleet **converges exactly**. |
| `ds_temporal_split_on_time_ordered_data` | A date stored as text is read as a date, the plan chooses a temporal split, and the split is **actually performed** — every training row precedes every test row. |
| `ds_constant_column_does_not_break_training` | A single-valued column does not crash the mission. |
| `ds_continuous_features_reach_the_model` | All-distinct continuous measurements are **not** dropped as identifiers. |
| `ds_no_signal_is_reported_as_no_signal` | Data with no relationship yields `beats_baseline=False`, verdict `fail`, Level 1. |

> **Why those four exist.** The suite scored **1.00 before and after** five real
> defects were fixed, because no task carried the shape that triggers them — no
> datetime column, no constant column in a training mission, no all-continuous
> feature set, no dataset without signal. On the unfixed tree they score 0.27,
> 0.00, 0.00 and 0.18, taking the overall from 1.00 to 0.79. A benchmark that
> cannot fail is a statement of intent, not a measurement.
>
> One result is worth quoting alone: given pure noise, the old code returned
> verdict `pass` and `max_safe_deployment_level 4` — it certified a model that
> had learned nothing as production-ready. Asking Avaloka for *her* baseline,
> rather than computing one in the harness and comparing, is what catches that.

### Format comprehension

Did it understand the file it was given? Every other family grades what Avaloka
*did* with a frame; none of them asks whether the frame is the table that was in
the file. A workbook read two rows off produces placeholder column names, and
every other check still passes on it — which is how insight text naming
`Unnamed: 13` was scored 1.00.

The API accepts nine upload extensions. One of them had a benchmark task: every
`DatasetSpec` wrote a CSV, and the three source schemes re-read that same CSV.
`DatasetSpec.suffix` is what fixed that.

| Task group | What it covers |
|---|---|
| `fmt_reads_*` | one file per declared format: tsv, json, xml, parquet, orc, avro |
| `nested_*` | the nesting rule, clause by clause (see below) |
| `xlsx_*` | the workbook shapes that break readers — a title row above the header, merged header cells, a two-row stacked header, data on sheet 2, "in thousands" stated only in prose, parenthesised negatives, footnote markers glued to labels, a second table below the first, and period columns (`2021 2022 2023`) as the header |
| `xlsx_sec_financial_report` | a synthetic SEC `Financial_Report.xlsx`: multiple statements, period columns, thousands scaling, parenthesised negatives, and a **planted figure** a meaningful answer has to reproduce |
| `edgar_10k_financial_report` | the same shape against a real filing (`--edgar`, opt-in) |

The comprehension checks are graded against ground truth each builder declares:
`no_placeholder_column_names`, `header_row_identified`,
`expected_columns_resolved`, `expected_dtypes`, `expected_row_count`,
`narration_names_only_real_columns` and `expected_answer`.

`tests/avaloka/test_format_coverage.py` fails when a declared format has no task
and no declared deferral, in **both** directions — so a format cannot be offered
in the upload dialog and never read by the suite again. A deferral lives in
`datasets_formats.DEFERRED_FORMATS` with its reason, where review can see it.

#### The nesting rule

JSON and XML are trees; a dataframe is a rectangle. A reader has to decide what
to do with nesting, and the dishonest option is to decide silently. So the rule
is stated in `avaloka/io/loader.py` and **graded** by the `nested_*` tasks:

| Shape | Rule |
|---|---|
| array of flat objects | a table, one column per key |
| scalar nesting | flattens to dotted names — `{"geo":{"city":…}}` → `geo.city` |
| a record holding a list of objects | **refused, naming the key** |
| two top-level arrays | **refused, naming both** |

Refusing is the point. Exploding a one-to-many multiplies the row count,
keeping the first element drops data, and stringifying it makes the numbers
unusable — none of which is visible downstream. `Expectations.expected_refusal`
grades both that the reader refused *and* that its message named the offending
key, because a refusal that says only "could not read file" leaves the user
nowhere to go.

#### Installing the readers

pandas and pyarrow already decode JSON, Parquet and ORC, and XML goes through
the standard library's `etree` parser when `lxml` is absent — so three of the
four formats that were unreadable cost the CLI no dependency. `.xlsx` and
`.avro` do need one:

```bash
pip install 'avaloka[formats]'      # openpyxl, fastavro, lxml
```

The CLI's eight runtime dependencies are unchanged, and
`tests/avaloka/test_cli_dependency_isolation.py` fails if they grow or if a
reader is needed but declared in no extra.

A 10-K as filed is PDF or HTML, and Avaloka reads neither. What SEC also
publishes per filing is `Financial_Report.xlsx` and XBRL, both in the declared
set; those are what the EDGAR task fetches. Nothing here implies PDF support.

### Conversational scorers

The family is `conversational_scorers`, and the name is load-bearing. **It
scores the probe scorers and, optionally, a raw model. It does not measure
Avaloka's conversational behaviour, and no path on this branch does.**

`tests/conversational/probes.py` was pytest-only, so it produced a "passed" and
no number. It is now a scored family: one task per dimension (interactivity,
responsiveness, competence, integrity), plus a negative control, plus a
declared gap. Deterministic — no model grades a model. The junk responder runs
as a live negative control and reddens the family if junk starts scoring well,
and a `question_sent` check per probe pins that the question actually reached
the responder; a probe graded on an empty question yields a number of nothing,
which is how a recorded 0.80 came to mean nothing.

Every task records its subject in `source_scheme` — `ref:brilliant`,
`ref:junk`, `live:gemma3:4b` — so the scorecard says what each number is a
number *of*.

`--live` drives a real provider under the standing provider ruling: a local
model first, then Groq, **never Anthropic**. It reaches that model over HTTP; it
does not enter Avaloka's agent graph either.

#### The declared gap

`conv_avaloka_behaviour_is_unmeasured` is a standing **failure**, not a note:

```
avaloka_conversational_behaviour_is_measured: FAIL
  avaloka.benchmark.conversational_product does not exist. no path on this
  branch drives the conversational probes through Avaloka's own agent graph…
```

It is there because prose does not survive. Someone reading
`conversational 1.00` off a scorecard in six months will quote it as a product
number, which is exactly what happened to the 0.80. So the suite reports the
gap beside its score — the same device as `DEFERRED_FORMATS` in
`datasets_formats`, where a limitation lives in code and review can see it.

The check grades a **lookup, not a constant**: add
`avaloka/benchmark/conversational_product.py` with a `product_responder()`
factory that drives Avaloka and returns the user-visible reply, and the check
passes on its own with nobody editing it. A gap assertion that has to be
remembered and deleted by hand goes stale and then lies.

Synthetic tasks need no network and carry ground truth by construction, so the
benchmark always runs. Kaggle tasks reuse the repo's existing download
convention (`tests/download_kaggle_datasets.py` → `app/sample_data/<slug__>`) and
skip gracefully when a dataset isn't present.

## Source-agnostic by design (`avaloka.io.sources`)

Every task is re-published through each requested connector and resolved back to
a local file with honest provenance, so *analyse from anywhere* is measured, not
assumed:

| URI | Source |
|---|---|
| `data.csv` · `file:///data.csv` | local file |
| `sqlite:///app.db#customers` · `postgresql://…/db#query=SELECT …` | database (SQLAlchemy) |
| `s3://bucket/key.parquet` · `gs://…` · `az://…` · `memory://…` | object storage (fsspec) |

```bash
avaloka analyze "sqlite:///app.db#customers" --goal "Understand churn"
avaloka train   "s3://bucket/customers.parquet" --target churned --metric roc_auc
```

## Multi-Avaloka coordination (`avaloka.coordinator`)

The open-claw **main Avaloka** shards the dataset and dispatches a fleet of
**sub-Avalokas** — each a full clone running its own firefly swarm on its shard —
then converges one answer:

```bash
avaloka coordinate customers.csv --goal "Understand churn" --workers 6
avaloka coordinate "sqlite:///app.db#customers" --kind train --target churned --metric roc_auc
```

Two convergence guarantees keep the fleet honest:

* **Profiling converges exactly** — sub-Avalokas emit sufficient statistics; the
  main Avaloka combines them losslessly (the same map-reduce as `avaloka batch`),
  so the fleet profile equals a single pass.
* **Decisions converge conservatively** — verdicts merge worst-case (any shard
  that fails fails the fleet); the selected model is the best-scoring shard model.

## Scoring

`TaskResult.score` is the weighted fraction of checks passed; `Scorecard`
aggregates by family and overall. `python -m avaloka.benchmark run` exits non-zero
if any non-skipped run fails, so CI can gate on it.

## Tests

```bash
pytest tests/avaloka -q                       # unit + integration (Kaggle deselected)
pytest tests/avaloka -m kaggle -v             # opt-in Kaggle e2e (needs downloaded data)
```

* `test_sources.py` — connectors (file / database / object storage / routing).
* `test_swarm.py` — firefly activation, ordering, narration, convergence.
* `test_coordinator.py` — multi-Avaloka exact + conservative convergence.
* `test_benchmark.py` — the harness and scorers themselves.
* `test_kaggle_benchmark.py` — end-to-end over real Kaggle datasets (opt-in).
