# Raw records for `docs/benchmarks/2.0-vs-1.0.md`

Every file here is written by a harness, not by hand, and every number in the
report is rendered from these files by `scripts/bench/summarize_2_0_vs_1_0.py`.
Each record carries the model id, the git commit and dirtiness of the tree that
produced it, timestamps, the effective configuration, and the observed cost with
its provenance (`catalog`, `openrouter_proxy`, `operator_rate`, `local_zero`,
`unpriced`). Trial journals (`*.trials.jsonl`, full trajectories) are not
committed; the DAB run directories under `$DAB_ROOT/query_*/logs/{avaloka,oneshot}/`
keep every model call and tool call.

| Folder | Contents |
| --- | --- |
| `dab/` | `benchmarks.run run dab` records. `2.0-*` = the Avaloka 2.0 harness (`scripts/avaloka_dab_agent.py`); `oneshot-*` = the model alone (`scripts/bench/oneshot_dab_agent.py`); `*-subset` = the 12-query one-per-dataset subset; `noise-*` = 3 trials on bookreview+yelp. `.md` files are the harness's own scorecards. |
| `ds/` | `2.0.json` and `1.0-graded-by-2.0-suite.json` = the CLI data-science suite (`avaloka.benchmark`), the 2.0 grader in both cases; `model-alone-*.json` = `scripts/bench/ds_model_alone.py`. |
| `dab/pilot/` | the two Groq pilots on the 3 `bookreview` queries (rehearsal tree `4542d264`, dirty); no table uses them. |
| `dab/logs/` | the harness's console log for every DAB invocation: `<record>.log` is the first pass, `<record>.retry.log` the retry/resume pass. They are the only source for money billed on trials the retry replaced, and for the Argon spend-ceiling abort. |
| `conv/` | `track6_conversational` records: `2.0-reply-surface.json` (with the scripted faithful/sycophant brackets), `1.0-reply-surface.json` (`bench20_conv_avaloka_v1`), `models.json` (`bench20_conv_models`); each with its harness `.md` scorecard. |

The folder is deliberately un-ignored in `benchmarks/results/.gitignore`;
everything else under `benchmarks/results/` stays local.
