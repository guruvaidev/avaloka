# Swarm data-science prompts

These copy/paste prompts are designed around decisions, not algorithms. Give
Avaloka the dataset first, then describe the decision you need to make. Her
firefly swarm profiles and prepares the data, tests the analysis or model, and
converges one answer with assumptions, validation, economics, and reproducible
artifacts.

## Start a conversation

```bash
avaloka chat customers.csv
```

Then paste one of these prompts:

```text
Analyze which customer segments are most likely to churn, quantify the strongest
drivers, check whether missing or imbalanced data could mislead us, and recommend
three retention actions supported by the data.
```

```text
Compare revenue, order frequency, and product mix across regions. Find unusual
segments, distinguish meaningful differences from noise, and produce an executive
brief with the assumptions and limitations a sales leader should know.
```

```text
Investigate the operational factors associated with late deliveries. Test the
most plausible explanations, identify where the evidence is weak, and rank the
interventions by expected impact and confidence.
```

For a predictive mission, name the target column explicitly:

```text
train churned
```

The conversational command starts a deployable training mission. For precise
control over its goal, metric, budget, and output, use the equivalent command:

```bash
avaloka train customers.csv \
  --target churned \
  --goal "Predict churn early enough for the retention team to intervene; prioritize reliable ranking, expose leakage and segment-level failure modes" \
  --metric roc_auc \
  --max-cost 25 \
  --deployable \
  --output ./churn-model
```

## Use the swarm of swarms

`avaloka coordinate` creates several sub-Avalokas over independent shards. Each
sub-Avaloka runs her own specialist firefly swarm, and the main Avaloka combines
profiles exactly and validation verdicts conservatively.

Sub-Avalokas execute concurrently by default. Use `--max-parallel` to cap local
CPU/memory pressure independently of the number of shards. The resulting
`coordination.json` records source materialization, per-shard sampling decisions,
effective parallelism, elapsed time, validation verdicts, and the collect step.

Use it when the result must reflect the whole dataset or when independent model
attempts should compete before one answer is selected:

```bash
avaloka coordinate customers.csv \
  --kind analyze \
  --workers 4 \
  --max-parallel 4 \
  --goal "Build a full-population churn risk brief: reconcile segment patterns across shards, preserve minority behavior, and surface any validation warning instead of averaging it away" \
  --output ./churn-fleet-analysis
```

```bash
avaloka coordinate customers.csv \
  --kind train \
  --target churned \
  --metric roc_auc \
  --workers 4 \
  --deployable \
  --budget 10 \
  --goal "Select a robust churn model across independent sub-Avalokas; reject leakage, retain the worst-case validation verdict, and package the winning model for staging" \
  --output ./churn-fleet-model
```

## Full-dataset batch analysis

For a dataset too large for the interactive lane, ask every partition the same
question and converge exact sufficient statistics:

```bash
avaloka batch transactions.parquet \
  --partitions 8 \
  --goal "Profile the complete transaction population, find quality failures and seasonal shifts, and identify which questions require deeper causal analysis" \
  --output ./transaction-swarm
```

Add `--target kubernetes --apply` to submit the generated KubeRay job after
reviewing its manifest.

## Kaggle: Titanic, transfer through model selection

```bash
mkdir -p data/titanic
kaggle datasets download -d yasserh/titanic-dataset -p data/titanic --unzip
```

Parallel analysis and collection:

```bash
avaloka coordinate data/titanic/Titanic-Dataset.csv \
  --kind analyze --workers 4 --max-parallel 4 \
  --goal "Explain survival patterns across passenger class, sex, age, fare and family structure; quantify missingness, preserve shard-level warnings, and converge one evidence-ranked answer" \
  --output ./titanic-swarm-analysis
```

Competitive model training and conservative selection:

```bash
avaloka coordinate data/titanic/Titanic-Dataset.csv \
  --kind train --target Survived --metric roc_auc \
  --workers 4 --max-parallel 4 --deployable --budget 10 \
  --goal "Build a survival classifier, reject identifier leakage, compare errors across sex and passenger class, retain the worst shard validation verdict, and package the strongest validated candidate" \
  --output ./titanic-swarm-model
```

The opt-in real-data test is:

```bash
pytest tests/avaloka/test_kaggle_benchmark.py -m kaggle -v
```

### What each stage means

- **Transfer** resolves a file, SQL table/query, or object-store URI into a
  provenance-tracked local dataset. It is connector materialization, not the
  application graph's legacy Data Transfer Agent.
- **Sampling** is decided independently inside each shard and recorded. Small
  shards use all rows; large shards use deterministic random or target-stratified
  samples for their firefly missions.
- **Analysis** runs concurrently across sub-Avalokas. Full-shard sufficient
  statistics are collected separately so numeric population summaries do not
  inherit sampling error.
- **Training** runs one real Avaloka training mission per shard and selects the
  best validated candidate conservatively. This is the Avaloka firefly model
  engine; it does not claim to invoke `app/agents/mta_v2`.

## End-to-end decision patterns

### Retention

```text
Decision: which customers should receive a retention offer this month?
Outcome: a ranked churn model plus segment-level drivers and validation limits.
Guardrails: no leakage from post-churn fields; report minority-segment recall;
keep total experimentation cost below $25.
```

### Demand forecasting

```text
Decision: how much inventory should each region hold for the next planning cycle?
Outcome: quantify seasonality and regional differences, train a demand model, and
show error by region rather than only an overall score.
Guardrails: use time-aware validation, flag sparse products, and do not recommend
production deployment if recent-period performance is unstable.
```

### Fraud or anomaly triage

```text
Decision: which events should investigators review first?
Outcome: rank suspicious events and explain the strongest measurable signals.
Guardrails: preserve class imbalance, report precision at the review-capacity
threshold, test leakage, and surface disagreement between sub-Avalokas.
```

### Operations root-cause analysis

```text
Decision: which operational change is most likely to reduce late deliveries?
Outcome: a validated analysis of delay drivers, affected segments, and practical
interventions ordered by evidence strength.
Guardrails: separate association from causation, identify missing confounders,
and include assumptions and confidence in the executive report.
```

## A strong prompt contract

The most useful prompts contain four things:

1. **Decision** — who will act and what they must decide.
2. **Outcome** — analysis, prediction, deployment package, or full-data profile.
3. **Success measure** — business metric and, for models, an evaluation metric.
4. **Guardrails** — budget, latency, leakage, fairness, time split, or deployment
   constraints.

Avaloka turns that contract into artifacts rather than treating it as permission
to invent an answer. Check `validation_report.json`, `assumptions.yaml`, the
ledger-backed economics, and the executive report before acting on a result.
