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
avaloka benchmark                                  # same thing, via the CLI
```

## Two families, grounded in ground truth

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
