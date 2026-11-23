## Quickstart

**Two commands, no API key, no cloud account.** You need Python 3.11 or 3.12 on
macOS or Linux.

```bash
git clone https://github.com/guruvaidev/avaloka.git && cd avaloka
./scripts/install.sh && source .venv/bin/activate     # ~5–10 minutes; prints "Everything checks out"
```

Ask your first question. The repo ships sample datasets, so this works on a
fresh clone:

```bash
avaloka analyze app/sample_data/healthcare_dataset.csv \
        --goal "average billing amount by medical condition"
```

Avaloka profiles the data, writes and validates the analysis, runs it, and
prints a verdict. The answer is in the terminal, and a folder under
`avaloka-analysis/` holds everything you need to open, rerun or share it:

| File | What it is |
| --- | --- |
| `executive_report.html` | the decision summary — open it in a browser |
| `README.md` | findings, assumptions and limitations |
| `analysis.py` · `analysis.ipynb` | code that reproduces the run exactly |
| `goal_analysis.json` | the full answer as data — every group, to the dollar |
| `lineage.json` · `data_quality.json` | where every number came from |

Then try the other commands:

```bash
avaloka train app/sample_data/yellow_tripdata_2015-01_dataset_50k.csv --target payment_type   # ~2 min
avaloka chat  app/sample_data/healthcare_dataset.csv     # numbered next steps; answer with a number
avaloka coordinate app/sample_data/healthcare_dataset.csv --swarm 2 --debug \
        --goal "average billing amount by medical condition"   # two clones, one answer, and the team's log
```

`train` fits several candidate models, checks the winner beats a trivial
baseline, and writes a validated, deployable model to `avaloka-model/`.
`coordinate` is the swarm: Avaloka clones herself, each copy works half of the
data with its own agents, and the main Avaloka merges what they found into one
profile and one verdict (`avaloka-fleet/coordination.json`). `--debug` tells
you, in plain language, every step the team took — each sentence comes from a
record in `avaloka-fleet/swarm_trace.json`. Above your edition's cap the run
proceeds at the cap and says so once. Your own files work the same way — CSV,
TSV, Parquet, Excel, JSON, XML, ORC and Avro. Every command, with worked
examples: [the CLI guide](docs/cli-user-guide.html).

> **Your data stays on your machine.** Everything above runs locally, with no
> model and no key. Add a model only when you want the conversational planner
> and richer narration, and pick any provider, or a local one: see
> [Bring your own model](#bring-your-own-model).

**Want the full platform** — the web UI, the API, Ray and Kubernetes? Follow the
[Install Guide](docs/INSTALL.md); it also covers
[running completely offline](docs/INSTALL.md#running-avaloka-completely-offline).
This is Avaloka 2.0.0: `avaloka version` prints `avaloka 2.0.0`, the API's
`GET /version` returns `2.0.0`, and the container images are published as
`ghcr.io/guruvaidev/avaloka-{api,ui,ray,functions}:2.0` by the first build of
the 2.0 line (`:main` until then, which is also the Helm default). Release
notes: [docs/releases/2.0.md](docs/releases/2.0.md).
If something does not work, run `python scripts/doctor.py` first, then see
[what goes wrong on a first run](docs/INSTALL.md#first-run-what-to-expect-and-what-goes-wrong).
The full CLI reference is [README_CLI.md](README_CLI.md).

---
