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
avaloka coordinate app/sample_data/healthcare_dataset.csv --workers 2 \
        --goal "average billing amount by medical condition"   # two copies of Avaloka, one answer
```

`train` fits several candidate models, checks the winner beats a trivial
baseline, and writes a validated, deployable model to `avaloka-model/`.
`coordinate` is the swarm: Avaloka clones herself, each copy works half of the
data with its own agents, and the main Avaloka merges what they found into one
profile and one verdict (`avaloka-fleet/coordination.json`). Your own files
work the same way — CSV, TSV, Parquet, Excel, JSON, XML, ORC and Avro.

> **Your data stays on your machine.** Everything above runs locally. Add a
> model key only when you want the conversational planner and richer narration:
> `OPENROUTER_API_KEY`, `GROQ_API_KEY` or `OPENAI_API_KEY` — or a local Ollama
> with `INFERENCE_PROVIDER=local`.

**Want the full platform** — the web UI, the API, Ray and Kubernetes? Follow the
[Install Guide](docs/INSTALL.md); it also covers
[running completely offline](docs/INSTALL.md#running-avaloka-completely-offline).
If something does not work, run `python scripts/doctor.py` first, then see
[what goes wrong on a first run](docs/INSTALL.md#first-run-what-to-expect-and-what-goes-wrong).
The full CLI reference is [README_CLI.md](README_CLI.md).

---
