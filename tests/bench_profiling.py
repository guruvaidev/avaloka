"""
Benchmark profiling-style LLM calls: OpenRouter vs Groq.

Splits a dataset's columns into groups (like the profiling agent does), sends one
call per group, and records per-call latency, HTTP status, tokens (incl. reasoning
tokens) and which upstream provider OpenRouter used. Runs at several concurrency
levels so you can see whether parallelism causes rate limiting.

Usage:
  pip install httpx pandas
  export OPENROUTER_API_KEY=...   GROQ_API_KEY=...
  python bench_profiling.py data.csv \
      --openrouter-model google/gemma-3-4b-it \
      --groq-model llama-3.1-8b-instant \
      --group-size 5 --concurrency 1 3 10

Optional: --prompt-file prompt.txt  (use your real profiling prompt; {columns} and
{sample} are filled in). Results are printed and saved to bench_results.csv.
"""
import argparse
import asyncio
import os
import statistics
import time

import httpx
import pandas as pd

ENDPOINTS = {
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
}

DEFAULT_PROMPT = (
    "You are a data profiling assistant. For each column below, give its likely "
    "semantic type, notable data quality issues, and a one-line description. "
    "Reply as compact JSON.\n\nColumns: {columns}\n\nSample rows:\n{sample}"
)


def build_groups(path, group_size, sample_rows):
    df = pd.read_csv(path, nrows=max(sample_rows, 1000))
    cols = list(df.columns)
    groups = [cols[i:i + group_size] for i in range(0, len(cols), group_size)]
    return [(g, df[g].head(sample_rows).to_csv(index=False)) for g in groups]


async def one_call(client, provider, model, prompt, sem, timeout):
    url, key_env = ENDPOINTS[provider]
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    headers = {"Authorization": f"Bearer {os.environ[key_env]}"}
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=body, headers=headers, timeout=timeout)
            dt = time.perf_counter() - t0
            row = {"status": r.status_code, "seconds": round(dt, 2),
                   "retry_after": r.headers.get("retry-after")}
            if r.status_code == 200:
                data = r.json()
                usage = data.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                row.update({
                    "upstream_provider": data.get("provider", provider),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": details.get("reasoning_tokens"),
                })
            else:
                row["error"] = r.text[:200]
            return row
        except Exception as e:  # timeouts, connection errors
            return {"status": "EXC", "seconds": round(time.perf_counter() - t0, 2),
                    "error": f"{type(e).__name__}: {e}"[:200]}


async def run(provider, model, groups, template, concurrency, timeout):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        tasks = [one_call(client, provider, model,
                          template.format(columns=", ".join(cols), sample=sample),
                          sem, timeout)
                 for cols, sample in groups]
        rows = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    for i, row in enumerate(rows):
        row.update({"provider": provider, "model": model, "concurrency": concurrency,
                    "group": i, "wall_seconds": round(wall, 2)})
    return rows, wall


def summarize(rows, wall):
    secs = [r["seconds"] for r in rows if r["status"] == 200]
    fails = [r for r in rows if r["status"] != 200]
    r429 = sum(1 for r in rows if r["status"] == 429)
    reasoning = [r.get("reasoning_tokens") or 0 for r in rows if r["status"] == 200]
    providers = sorted({str(r.get("upstream_provider")) for r in rows if r["status"] == 200})
    p = rows[0]
    line = (f"{p['provider']:<10} c={p['concurrency']:<3} wall={wall:6.1f}s  "
            f"calls={len(rows)} ok={len(secs)} fail={len(fails)} 429s={r429}")
    if secs:
        line += (f"  per-call median={statistics.median(secs):.1f}s max={max(secs):.1f}s"
                 f"  reasoning_tok(avg)={sum(reasoning) / len(reasoning):.0f}"
                 f"  upstream={','.join(providers)}")
    print(line)
    for f in fails[:3]:
        print(f"    fail: status={f['status']} {f.get('error', '')}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--openrouter-model")
    ap.add_argument("--groq-model")
    ap.add_argument("--group-size", type=int, default=5)
    ap.add_argument("--sample-rows", type=int, default=20)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 3, 10])
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--prompt-file")
    a = ap.parse_args()

    template = open(a.prompt_file).read() if a.prompt_file else DEFAULT_PROMPT
    groups = build_groups(a.dataset, a.group_size, a.sample_rows)
    print(f"{a.dataset}: {len(groups)} column groups\n")

    targets = [(p, m) for p, m in (("openrouter", a.openrouter_model), ("groq", a.groq_model)) if m]
    all_rows = []
    for provider, model in targets:
        print(f"== {provider} / {model}")
        for c in a.concurrency:
            rows, wall = await run(provider, model, groups, template, c, a.timeout)
            summarize(rows, wall)
            all_rows += rows
        print()
    pd.DataFrame(all_rows).to_csv("bench_results.csv", index=False)
    print("Saved bench_results.csv")


if __name__ == "__main__":
    asyncio.run(main())