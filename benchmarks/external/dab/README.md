# Running official DataAgentBench with Avaloka

This integration runs Avaloka against the upstream UC Berkeley DataAgentBench (DAB)
without copying its questions, gold answers, validators, or large databases into this
repository. DAB currently contains 54 queries over 12 datasets and PostgreSQL,
MongoDB, SQLite, and DuckDB. An official run is five trials per query (270 trials).

## 1. Provision upstream DAB

Requirements: Git LFS, Docker, PostgreSQL 17.5+, MongoDB, and the dependencies in
DAB's `environment.yaml`. The upstream PATENTS database alone is approximately 5 GB.

```bash
bash scripts/setup_dab.sh
export DAB_ROOT="$PWD/.benchmarks/DataAgentBench"
python -m benchmarks.external.dab.preflight
```

Follow upstream `README.md` to load PostgreSQL/MongoDB and build its `python-data:3.12`
image. `setup_dab.sh` downloads files and verifies upstream fingerprints; it does not
silently install or start system databases.

Set `DAB_USE_HINTS=1` to use `db_description_withhint.txt`. Report that choice with
leaderboard results. The default is no hints.

## 2. Configure the Avaloka-native DAB command

The checked-in `scripts/avaloka_dab_agent.py` implements `dab-command-v2`. It receives a
mission, sanctioned source configuration, an explicit model, and hard resource limits. It
runs `avaloka.data_agent.DataQueryMission`; it does not import DAB's reference `DataAgent` or
run validators. Validation remains in the parent harness after execution.

```bash
export AVALOKA_DAB_AGENT_CMD="$PWD/.benchmarks/dab-venv/bin/python $PWD/scripts/avaloka_dab_agent.py"
export AVALOKA_DAB_MODEL="gemini-2.5-flash"
```

The default configuration has a $2 hard cost ceiling. Paid runs therefore require a versioned
pricing configuration; unknown cost is never represented as zero:

```bash
export AVALOKA_DAB_PRICING_VERSION="provider-pricing-date"
export AVALOKA_DAB_INPUT_USD_PER_MTOK="<current input price>"
export AVALOKA_DAB_OUTPUT_USD_PER_MTOK="<current output price>"
# Tiered models: for an openrouter/ model the agent takes the long-prompt tier from
# benchmarks/catalog/openrouter_models.json (Haiku 5.5: $0.50/$2.50 from 100k prompt tokens).
# Setting the threshold yourself replaces it; for other providers, set it when the model is tiered:
# export AVALOKA_DAB_PRICING_THRESHOLD_TOKENS=99999   # the tier applies ABOVE this count
# export AVALOKA_DAB_INPUT_USD_PER_MTOK_OVER_THRESHOLD="<input price above it>"
# export AVALOKA_DAB_OUTPUT_USD_PER_MTOK_OVER_THRESHOLD="<output price above it>"
```

Keep provider keys in the process environment only. Do not place them in commands, `.env`
files committed to Git, reports, or traces. The child process must not read `validate.py`,
prior results, or external answer sources.

## 3. Run and export

```bash
python3 -m benchmarks.run run dab --allow-cloud --out benchmarks/results/dab-run
python3 -m benchmarks.external.dab.submission \
  benchmarks/results/dab-run/dab.json \
  --output benchmarks/results/dab-run/avaloka-dab-submission.json \
  --traces benchmarks/results/dab-run/traces
```

Run the three-query `bookreview` pilot first:

```bash
python3 -m benchmarks.run run dab --allow-cloud --dataset bookreview --trials 1 \
  --out benchmarks/results/dab-bookreview
```

The suite uses the official per-query `validate.py` graders and writes Avaloka's
quality/cost/latency/data-scanned scorecard. The exporter refuses incomplete output:
it requires exactly 270 unique `(dataset, query, run)` rows and writes a trace for each.

Run the seven-query MongoDB/Yelp gate before the full suite:

```bash
python3 -m benchmarks.run run dab --allow-cloud --dataset yelp --trials 1 \
  --out benchmarks/results/dab-yelp
```

`--out` also creates an fsynced trial journal. Repeating a compatible command resumes completed
trials; `--no-resume` deliberately starts fresh. Use `--database-type mongo` to include every
DAB dataset that uses MongoDB, currently not only Yelp.

Do not describe a partial run as an official DAB result. Retain full trajectories for
contamination review and export only a uniform five-trial, 270-run result.
