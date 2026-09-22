# Benchmarks

Avaloka ships a benchmark you can run yourself. It is not a leaderboard entry
and it is not a marketing number — it is the regression harness the project
uses to know whether a change broke the data-science paths.

```bash
python -m avaloka.benchmark run                 # print the scorecard
python -m avaloka.benchmark run --json out.json # machine-readable
```

## Current scorecard

**17/17 passed — overall 1.00**

| Family | Score |
| --- | ---: |
| Data Engineering | 1.00 |
| Data Science | 1.00 |

## Every task

| Task | Kind | Source | Score | Time |
| --- | --- | --- | ---: | ---: |
| `de_ingestion_parity_churn` | analyze | file | 1.00 | 0.2s |
| `de_ingestion_parity_churn` | analyze | sqlite | 1.00 | 0.1s |
| `de_ingestion_parity_churn` | analyze | object_storage | 1.00 | 0.1s |
| `de_data_quality_dirty` | analyze | file | 1.00 | 0.1s |
| `de_data_quality_dirty` | analyze | sqlite | 1.00 | 0.1s |
| `de_workload_routing_sampled` | analyze | file | 1.00 | 0.2s |
| `de_batch_exact_convergence` | batch | file | 1.00 | 6.9s |
| `ds_classification_churn` | train | file | 1.00 | 3.2s |
| `ds_classification_churn` | train | sqlite | 1.00 | 3.4s |
| `ds_regression_housing` | train | file | 1.00 | 2.4s |
| `ds_leakage_detection` | train | file | 1.00 | 2.9s |
| `ds_temporal_split_on_time_ordered_data` | train | file | 1.00 | 2.1s |
| `ds_constant_column_does_not_break_training` | train | file | 1.00 | 2.6s |
| `ds_continuous_features_reach_the_model` | train | file | 1.00 | 2.1s |
| `ds_no_signal_is_reported_as_no_signal` | train | file | 1.00 | 2.9s |
| `ds_analysis_gating_and_swarm` | analyze | file | 1.00 | 0.0s |
| `ds_analysis_gating_and_swarm` | analyze+coordinate | fleet | 1.00 | 0.2s |

## What the tasks check

The **data-engineering** tasks read the same dataset through every supported
source — a file, a SQLite table, object storage — and require identical
results. Ingestion that silently differs by source is the kind of bug that
surfaces months later as a number nobody can reproduce.

The **data-science** tasks are regression tests for defects that were found and
fixed. That is what makes them worth keeping:

| Task | The defect it guards against |
| --- | --- |
| `ds_leakage_detection` | target leakage passing silently into training |
| `ds_temporal_split_on_time_ordered_data` | a random split on time-ordered data |
| `ds_constant_column_does_not_break_training` | one constant column crashing every training run |
| `ds_continuous_features_reach_the_model` | continuous features dropped as identifier-like |
| `ds_no_signal_is_reported_as_no_signal` | a confident model reported on pure noise |

## What this benchmark does not measure

Stated plainly, because a perfect score invites the wrong reading.

**It does not compare language models.** The suite scores 17/17 with every API
key unset — it never calls an LLM. It exercises Avaloka's own Python: the
ingestion paths, the split logic, the leakage checks. Running it against a
different model would produce identical scores and tell you nothing.

**A saturated score is a floor, not a ceiling.** Every task passing means
nothing regressed, not that the system is finished. Tasks are added when a
defect is found, so the suite grows toward the things that have actually
broken.

For a comparison that *does* separate models — where the model genuinely
decides the outcome — see the routing-accuracy table in
[test-reports/1.6-reliability-measurements.md](test-reports/1.6-reliability-measurements.md).

