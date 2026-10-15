# Benchmarks

Avaloka ships a benchmark you can run yourself. It is not a leaderboard entry
and it is not a marketing number — it is the regression harness the project
uses to know whether a change broke the data-science paths.

```bash
python -m avaloka.benchmark run                      # print the scorecard
python -m avaloka.benchmark run --json out.json      # machine-readable
python -m avaloka.benchmark run --family data_science
python -m avaloka.benchmark run --kaggle             # + downloaded Kaggle datasets
avaloka benchmark                                    # the same suite via the CLI
```

The two entry points take slightly different flags. The module form is declared
in `avaloka/benchmark/__main__.py`: `--out` sets the artifacts directory (a temp
dir by default) and `--quiet` suppresses per-task progress. `avaloka benchmark`
is declared in `avaloka/cli.py`: the artifacts directory is `--output`/`-o`, and
there is no `--quiet`.

The process **exits 1 if any non-skip run failed**, so a CI job that runs the
command fails with it; `.github/workflows/ci.yml` defines a `benchmark` job that
runs the full suite. Two cases do not change the exit code: a run whose error
text contains the word `unavailable` is reported as a skip (that is how a Kaggle
dataset that is not on disk is recognised), and a selection that matches no task
runs nothing and exits 0. `avaloka/benchmark/README.md` documents each task.

## Current scorecard

**19/19 passed — overall 1.00; the command exits 0.** Measured on `develop-1.6`
with no API keys set. The 14 tasks in `avaloka/benchmark/suite.py` expand to 19
runs because several are run once per source scheme and the fleet check adds one
more. Scores are reproducible; the **times are not** — they come from one run on
one machine and are included only to show the order of magnitude.

The scorecard counts two kinds of check apart. An **assertion** says a property
holds now. A **guard** says a known wrong fix has not been applied; it is green
the day it is written, so a guard that holds is not evidence that anything was
fixed. Either kind failing fails the run.

| Family | Score |
| --- | ---: |
| Data Engineering | 1.00 |
| Data Science | 1.00 |

| Kind of check | Result |
| --- | ---: |
| Assertions (a property holds now) | 136/136 passed |
| Guards (a known wrong fix has not been applied) | 4/4 held |

## Every task

| Task | Kind | Source | Result | Score | Time |
| --- | --- | --- | --- | ---: | ---: |
| `de_ingestion_parity_churn` | analyze | file | pass | 1.00 | 0.1s |
| `de_ingestion_parity_churn` | analyze | sqlite | pass | 1.00 | 0.2s |
| `de_ingestion_parity_churn` | analyze | object_storage | pass | 1.00 | 0.1s |
| `de_data_quality_dirty` | analyze | file | pass | 1.00 | 0.1s |
| `de_data_quality_dirty` | analyze | sqlite | pass | 1.00 | 0.1s |
| `de_workload_routing_sampled` | analyze | file | pass | 1.00 | 0.3s |
| `de_batch_exact_convergence` | batch | file | pass | 1.00 | 9.6s |
| `de_null_words_stay_categories` | analyze | file | pass | 1.00 | 0.1s |
| `de_spreadsheet_header_is_found` | analyze | file | pass | 1.00 | 0.1s |
| `ds_classification_churn` | train | file | pass | 1.00 | 8.0s |
| `ds_classification_churn` | train | sqlite | pass | 1.00 | 6.3s |
| `ds_regression_housing` | train | file | pass | 1.00 | 5.2s |
| `ds_leakage_detection` | train | file | pass | 1.00 | 4.3s |
| `ds_temporal_split_on_time_ordered_data` | train | file | pass | 1.00 | 3.4s |
| `ds_constant_column_does_not_break_training` | train | file | pass | 1.00 | 5.5s |
| `ds_continuous_features_reach_the_model` | train | file | pass | 1.00 | 3.5s |
| `ds_no_signal_is_reported_as_no_signal` | train | file | pass | 1.00 | 4.7s |
| `ds_analysis_gating_and_swarm` | analyze | file | pass | 1.00 | 0.1s |
| `ds_analysis_gating_and_swarm` | analyze+coordinate | fleet | pass | 1.00 | 0.2s |

Every task name above is a `name=` literal in `avaloka/benchmark/suite.py`. The
source column is that task's `source_schemes`, with two exceptions set by the
runner: the batch task always reads a file, and `fleet` is the coordinated run.

## What the tasks check

The **data-engineering** tasks cover ingestion, data-quality flags, workload
routing and exact batch convergence. One of them, `de_ingestion_parity_churn`,
reads the same dataset from three sources: a file, a SQLite table, and an
in-process `memory://` object store standing in for object storage. Each source
run is scored on its own against the same expectations (the routing lane and the
column roles). **The runs are not compared with each other**, so the task does
not establish that the three sources return identical data; see
[Known gaps in the checks](#known-gaps-in-the-checks).

`de_null_words_stay_categories` plants a 55-row payments file in which three of
the eight payment methods are the words `None`, `NA` and `N/A`, and asks for
total revenue by payment method. It reads the answer off two artifacts: the
grouped result in `goal_analysis.json` (every planted group with its exact sum,
and no group that was not planted) and the column profile in
`data_quality.json` (eight labels with their exact counts, no missing cells).
**It never compares the total**, because the total is 5500.0 whether or not the
categories survive: the lost ones merge into a single `nan` group that holds
exactly their combined sum. Its one guard pins the `amount` column, which writes
its gaps as the literal `NA`, to the role `numeric`. Reading the file with
`keep_default_na=False` would keep the three labels and turn `amount` into
text; with that change applied the guard is the check that fails
(`role[amount]: inferred 'categorical', expected 'numeric'`).

This task failed when it was added, because the CSV reader did turn those three
words into missing values: the grouped answer had 6 groups with a `nan` group of
1600.0, and the profile reported 16 missing cells. The reader now keeps a
null-like word as a label in a text column and as a gap in a numeric, boolean or
date column.

`de_spreadsheet_header_is_found` plants a spreadsheet whose table does not start
in cell A1: a title in row 1, a blank row, the header in row 3, everything
indented one column, and the pane frozen under the header. It checks that the
column names the mission saw are the real header (`region`, `units`, `revenue`),
that none is a placeholder such as `Unnamed: 0`, and that there are 12 rows.
The reader already does this, so all three checks are **guards**. With the
header search disabled they fail: the columns become
`['Quarterly sales report', 'Unnamed: 2', 'Unnamed: 3']` and the row count 14.

This task writes its file with `openpyxl`, which is in the package's `benchmark`
extra and not among its runtime dependencies. Without it the task **fails** with
`dataset could not be built: ModuleNotFoundError ... install the benchmark
extra: pip install "avaloka[benchmark]"`, the command exits 1, and the other
tasks still run. It is never reported as a skip.

The **data-science** tasks include four that were added as regression tests
after defects were found and fixed, alongside the leakage task:

| Task | What it is there to catch |
| --- | --- |
| `ds_leakage_detection` | target leakage passing silently into training |
| `ds_temporal_split_on_time_ordered_data` | a random split on time-ordered data (see the gap below) |
| `ds_constant_column_does_not_break_training` | one constant column crashing every training run |
| `ds_continuous_features_reach_the_model` | continuous features dropped as identifier-like |
| `ds_no_signal_is_reported_as_no_signal` | a confident model reported on pure noise |

## Known gaps in the checks

Found by breaking the code on purpose and running the suite. A check that still
passes when the thing it names is broken is listed here rather than left to be
discovered.

- **The temporal-split check does not look at what the mission did.** With the
  temporal split disabled in the modelling code, so that every model is fitted
  on a random shuffle, `ds_temporal_split_on_time_ordered_data` still scores
  1.00. The check performs the split itself and confirms the ordering; it does
  not inspect the split the model was fitted with.
- **Ingestion parity is not enforced across sources.** With the SQLite reader
  returning half the rows, `de_ingestion_parity_churn` still scores 1.00 on all
  three sources (750 rows from SQLite against 1,500 from the other two). The
  family failed in that experiment only because a different task lost a
  data-quality flag.
- **`transformation_reproducible` checks that the transformed dataset exists and
  is not empty.** It does not run anything twice or compare outputs.
- **`de_batch_exact_convergence` accepts a single-process engine** as well as
  local Ray, so a pass does not show that Ray ran.
- **`de_null_words_stay_categories` reads a CSV file only.** The same words in
  an Excel workbook are still read as missing values (measured: 16 of 55 cells),
  and no task covers it. A SQLite table and a Parquet file keep them; no task
  checks that either.
- **A grouped answer is checked up to ten groups and a profile up to ten
  labels**, because that is how many the analyzer and the profile write.

The suite does fail on other deliberate breaks: emptying the validator's
leakage findings, or raising inside the model step, both exit 1.

## What this benchmark does not measure

Stated plainly, because a perfect score invites the wrong reading.

**It does not compare language models.** The suite produces the scorecard above
with every API key unset — it never calls an LLM. It exercises Avaloka's own Python: the
ingestion paths, the split logic, the leakage checks. Running it against a
different model would produce identical scores and tell you nothing. The
key-unset run is reproduced verbatim in
[test-reports/1.6-reliability-measurements.md](test-reports/1.6-reliability-measurements.md),
which is also where the reason not to publish a model matrix over it is set out.

**A saturated score is a floor, not a ceiling.** Every task passing means
nothing regressed, not that the system is finished. Tasks are added when a
defect is found, so the suite grows toward the things that have actually
broken.

For a comparison that *does* separate models — where the model genuinely
decides the outcome — see the routing-accuracy table in
[test-reports/1.6-reliability-measurements.md](test-reports/1.6-reliability-measurements.md).

