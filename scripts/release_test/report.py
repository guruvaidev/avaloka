"""Build the dataset x path x model matrix (Markdown) from one or more results.jsonl files.

    python scripts/release_test/report.py results/*/results.jsonl > matrix.md
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def cell(rec: dict) -> str:
    cls = rec.get("insight_class", "?")
    if rec["path"] == "cli":
        co, an, sw = rec.get("coordinate", {}), rec.get("analyze", {}), rec.get("swarm", {})
        parts = [f"coordinate exit={co.get('exit')} {co.get('wall_s')}s {co.get('peak_rss_mb')}MB",
                 f"clones {sw.get('clones_dispatched')}/{sw.get('clones_returned')} validator={'yes' if sw.get('validator_ran') else 'NO'}",
                 f"analyze exit={an.get('exit')} {an.get('wall_s')}s", f"**{cls}**"]
    else:
        up, sm, ch = rec.get("upload", {}), rec.get("swarm_mission", {}), rec.get("chat", {})
        sw = sm.get("swarm", {})
        parts = [f"upload {up.get('status')} {up.get('wall_s', '')}s",
                 f"swarm clones {sw.get('clones_dispatched')}/{sw.get('clones_returned')} validator={'yes' if sw.get('validator_ran') else 'NO'}" if sw else f"swarm {sm.get('status')}",
                 f"chat {ch.get('status')} {ch.get('wall_s', '')}s{' deferred' if ch.get('deferred') else ''}", f"**{cls}**"]
        if rec.get("telemetry_leak"):
            parts.append(f"TELEMETRY LEAK {rec['telemetry_leak']}")
    if not rec.get("checked"):
        parts.append("_(unchecked: not a pass)_")
    return "<br>".join(str(p) for p in parts)


def main(files: list[str]) -> None:
    recs = [json.loads(l) for f in files for l in Path(f).read_text().splitlines() if l.strip()]
    cols = sorted({(r["path"], r.get("model", "")) for r in recs}, key=lambda c: (c[0] != "cli", c[1]))
    names = []
    for r in recs:
        if r["dataset"] not in names:
            names.append(r["dataset"])
    by = {(r["dataset"], r["path"], r.get("model", "")): r for r in recs}
    print("| dataset | format | " + " | ".join(f"{p} · {m}" for p, m in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for n in names:
        fmt = next((r.get("format") for r in recs if r["dataset"] == n), "")
        row = [cell(by[(n, p, m)]) if (n, p, m) in by else "—" for p, m in cols]
        print(f"| {n} | {fmt} | " + " | ".join(row) + " |")
    print()
    counts = Counter((r["path"], r.get("model", ""), r.get("insight_class")) for r in recs)
    print("| path · model | class | n |"); print("|---|---|---:|")
    for (p, m, c), k in sorted(counts.items()):
        print(f"| {p} · {m} | {c} | {k} |")
    unchecked = sum(1 for r in recs if not r.get("checked"))
    print(f"\n{len(recs)} records, {unchecked} unchecked (verified nothing; not passes).")
    print("\n### Insights\n")
    for r in recs:
        print(f"- **{r['dataset']}** [{r['path']} · {r.get('model','')}] → {r.get('insight_class')}: {r.get('insight_reason','')}")
        if r.get("insight"):
            print("  > " + str(r["insight"]).replace("\n", "\n  > ")[:1200])


if __name__ == "__main__":
    main(sys.argv[1:])
