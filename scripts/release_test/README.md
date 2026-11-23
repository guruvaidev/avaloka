# Release dataset harness

Runs a corpus of real datasets through the two public paths — the CLI fleet
(`avaloka coordinate` + `avaloka analyze`) and the container API (upload →
swarm mission → chat turn → trace) — and records what each run actually did:
exit status, wall time, peak RSS, clones dispatched/returned from the swarm
trace, whether the validator firefly ran, the insight text and its class
(sensible / vacuous / nonsense, with the reason), and crash tails.

The datasets are **not** in the repo. Write a manifest like
`corpus.example.json` pointing at local copies (one directory per download;
treat them as untrusted) and run inside the arm64 CI image:

```bash
scripts/release_test/api_stack.sh up gemma 9200 /path/to/checkout /path/to/state local gemma4:e4b
docker run --rm --platform linux/arm64 --network host -v /path/to/checkout:/w:ro -v /corpus:/corpus:ro \
  -v /results:/results -w /w -e PYTHONPATH=/w -e SUPABASE_JWT_SECRET="$(cat /path/to/state/jwt_secret)" \
  avaloka-ci-deps:latest python scripts/release_test/run_datasets.py \
  --corpus /corpus/corpus.json --out /results/gemma --paths cli,api --api http://127.0.0.1:9200 \
  --model-label local/gemma4:e4b
scripts/release_test/api_stack.sh down gemma
```

`results.jsonl` is the evidence; `checked: false` records verified nothing and
are never a pass. The classifier recomputes the CLI's numeric claims from the
data; model prose is only checked structurally and is reported as
"sensible (unverified)".
