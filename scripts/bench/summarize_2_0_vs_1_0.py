#!/usr/bin/env python3
"""Render the 2.0-vs-1.0 tables from the raw records in benchmarks/results/2.0/.

Every number in docs/benchmarks/2.0-vs-1.0.md comes out of this script, so the
report can be regenerated from the records and nothing in it is typed by hand.

    python scripts/bench/summarize_2_0_vs_1_0.py [benchmarks/results/2.0]
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/results/2.0")


def _load(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text()) if path.is_file() else None


def _mean(values: Iterable[float]) -> Optional[float]:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.fmean(values) if values else None


def _fmt(v: Optional[float], digits: int = 2) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


# ---------------------------------------------------------------- harness records
def harness_cells(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{config_id: {task_id: stats}} for a benchmarks.run record."""
    return {cid: body["tasks"] for cid, body in record["results"].items()}


INFRA_PREFIXES = ("provider_error", "sut_start_error", "provisioning_error")


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a proportion; the honest error bar for one trial per query."""
    if not n:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _is_infra(trial: Dict[str, Any]) -> bool:
    return str((trial.get("outcome") or {}).get("error") or "").startswith(INFRA_PREFIXES)


def dab_summary(record: Dict[str, Any], tasks_filter: Optional[set] = None) -> Dict[str, Any]:
    (cid, tasks), = harness_cells(record).items()
    rows = [(tid, t) for tid, t in tasks.items() if not tasks_filter or tid in tasks_filter]
    passes = [t["stats"]["pass_at_1"] for _, t in rows]
    # A query is an infrastructure failure when EVERY trial of it ended in a
    # provider/start-up error (after the once-only retry); it then has no answer
    # from the system under test and leaves the "answered" denominator.
    infra = [tid for tid, t in rows if t.get("trials") and all(_is_infra(tr) for tr in t["trials"])]
    # A query is all-errored when every trial ended with ANY error (infra, agent
    # give-up, timeout, budget, a program that crashed); it scores 0.00 without
    # an answer having been graded. Provider errors are counted per trial and
    # split by the provider's own error code where it gave one.
    errors = [str((tr.get("outcome") or {}).get("error") or "") for _, t in rows for tr in t.get("trials", [])]
    all_errored = [tid for tid, t in rows if t.get("trials")
                   and all((tr.get("outcome") or {}).get("error") for tr in t["trials"])]
    provider_codes: Dict[str, int] = {}
    for e in errors:
        if e.startswith("provider_error"):
            m = re.search(r"'code': '([a-z_]+)'", e)
            code = m.group(1) if m else "other"
            provider_codes[code] = provider_codes.get(code, 0) + 1
    record_cost = sum((tr.get("outcome") or {}).get("cost_usd") or 0.0 for _, t in rows for tr in t.get("trials", []))
    answered = [(tid, t) for tid, t in rows if tid not in set(infra)]
    cost = [t["stats"]["tuple"]["cost_usd"]["mean"] for _, t in rows]
    lat = [t["stats"]["tuple"]["latency_s"]["mean"] for _, t in rows]
    tool_calls = [tr["outcome"].get("tool_calls") for _, t in rows for tr in t.get("trials", [])]
    retries = [tr["outcome"].get("retries") for _, t in rows for tr in t.get("trials", [])]
    spend = (record.get("meta") or {}).get("spend") or {}
    cfg = record["results"][cid]["config"]
    k = int(round(sum(passes)))
    # The official leaderboard figure is the mean over DATASETS of each dataset's
    # mean per-query pass rate (upstream README, "Methodology"), not the
    # per-query mean; both are reported, labelled.
    by_ds: Dict[str, List[float]] = {}
    for tid, t in rows:
        by_ds.setdefault(tid.split(":")[1], []).append(t["stats"]["pass_at_1"])
    macro = _mean(_mean(v) for v in by_ds.values()) if by_ds else None
    return {
        "pass_at_1_dataset_macro": macro,
        "config": cid, "llm": cfg.get("llm"), "sut": cfg.get("sut"), "trials": cfg.get("trials"),
        "n_queries": len(rows), "pass_at_1": _mean(passes), "passed": k, "ci95": wilson(k, len(rows)),
        "infra": len(infra), "infra_ids": infra,
        "all_errored": len(all_errored), "provider_codes": provider_codes,
        "n_trials": sum(len(t.get("trials", [])) for _, t in rows),
        "passed_trials": int(round(sum(tr["quality"]["pass"] for _, t in rows for tr in t.get("trials", [])))),
        "record_cost_usd": record_cost,
        "pass_at_1_answered": _mean(t["stats"]["pass_at_1"] for _, t in answered),
        "cost_per_query": _mean(cost), "latency_s": _mean(lat),
        "tool_calls": _mean(tool_calls), "retries": _mean(retries),
        "spend_usd": spend.get("spent_usd"), "commit": (record.get("meta") or {}).get("git_commit"),
        "per_task": {tid: t["stats"]["pass_at_1"] for tid, t in rows},
    }


def per_dataset(summary: Dict[str, Any]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for tid, p in summary["per_task"].items():
        out.setdefault(tid.split(":")[1], []).append(p)
    return out


# ---------------------------------------------------------------- DS records
def ds_cli_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    rows = record["results"]
    by_task: Dict[str, List[float]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["score"] if r["passed"] or r["score"] else 0.0)
    return {"overall": record.get("overall_score"), "n_runs": record.get("n_runs"),
            "n_passed": record.get("n_passed"), "commit": record.get("commit"),
            "by_task": {k: _mean(v) for k, v in by_task.items()},
            "passed_by_task": {r["task"] + "/" + r.get("source_scheme", "") : r["passed"] for r in rows}}


def ds_model_alone_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    s = record["summary"]
    return {"model": record["model"], "provider": record["provider"], "mean_score": s["mean_score"],
            "pass_rate": s["pass_rate"], "by_task": s["by_task"], "spend_usd": s["spend_usd"],
            "trials": record["trials"],
            "measured": {r["task"]: r["measured"]["model"] for r in record["results"] if r["trial"] == 0}}


# ---------------------------------------------------------------- render
def main() -> int:
    out: List[str] = []
    P = out.append

    # ---- DAB
    P("### DAB (official DataAgentBench; pass@1 over single trials, official validators)\n")
    P("Cost columns are the sum of `cost_usd` over the trials kept in each record (provider-reported usage x the rate shown in §2). "
      "They exclude trials that were replaced by the once-only retry; money actually billed is in the spend table below. "
      "The harness header `Observed spend` is not used: it sees only the last invocation of a resumed run.\n")
    P("| System | Model | Queries | Pass@1 (per query) | 95% CI | Pass@1 (dataset macro, official) | Passed | All-errored | Infra fail (after retry) | Provider errors by code | Pass@1 (answered) | $/query | s/query | Tool calls | Cost in record $ | Record |")
    P("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    dab_dir = ROOT / "dab"
    subset_ids: Optional[set] = None
    summaries: Dict[str, Dict[str, Any]] = {}
    for path in sorted(dab_dir.glob("*.json")) if dab_dir.is_dir() else []:
        rec = _load(path)
        if not rec or "results" not in rec:
            continue
        s = dab_summary(rec)
        summaries[path.stem] = s
        system = "2.0 harness" if s["sut"] == "avaloka_dab" and "oneshot" not in path.stem else "model alone (one shot)"
        if path.stem.startswith("noise"):
            system = "2.0 harness, 3 trials (noise)"
        lo, hi = s["ci95"]
        codes = ", ".join(f"`{c}` {n}" for c, n in sorted(s["provider_codes"].items())) or "0"
        # A multi-trial record reports trials passed, so "Passed" never reads as queries.
        passed = s["passed"] if s["n_trials"] == s["n_queries"] else f"{s['passed_trials']}/{s['n_trials']} trials"
        P(f"| {system} | `{s['llm']}` | {s['n_queries']} | {_fmt(s['pass_at_1'])} | {_fmt(lo)}–{_fmt(hi)} | "
          f"{_fmt(s['pass_at_1_dataset_macro'])} | {passed} | {s['all_errored']} | {s['infra']} | {codes} | {_fmt(s['pass_at_1_answered'])} | {_fmt(s['cost_per_query'], 4)} | "
          f"{_fmt(s['latency_s'], 1)} | {_fmt(s['tool_calls'], 1)} | {_fmt(s['record_cost_usd'], 2)} | `{path.name}` |")
        if "subset" in path.stem and subset_ids is None:
            subset_ids = set(s["per_task"])
    if subset_ids:
        P("\nSame 12-query subset (one query per dataset), every system that has those queries:\n")
        P("| Record | Pass@1 on subset | Passed/12 |")
        P("| --- | ---: | ---: |")
        for name, s in summaries.items():
            sub = [p for t, p in s["per_task"].items() if t in subset_ids]
            if len(sub) == len(subset_ids):
                P(f"| `{name}` | {_fmt(_mean(sub))} | {int(round(sum(sub)))} |")
    full = {n: s for n, s in summaries.items() if s["n_queries"] == 54}
    if full:
        P("\nPer dataset (pass@1 mean), full 54-query runs:\n")
        names = list(full)
        P("| Dataset | " + " | ".join(f"`{n}`" for n in names) + " |")
        P("| --- |" + " ---: |" * len(names))
        datasets = sorted({d for s in full.values() for d in per_dataset(s)})
        for d in datasets:
            P(f"| {d} | " + " | ".join(_fmt(_mean(per_dataset(s).get(d, []))) for s in full.values()) + " |")

    # ---- spend actually billed, from the committed run logs
    logs = ROOT / "dab" / "logs"
    if logs.is_dir():
        P("\nSpend billed per DAB invocation (last `Observed spend` line of each committed log under `dab/logs/`; "
          "a `.retry.log` is the second invocation of the same record, which resumes kept trials and re-runs the rest):\n")
        P("| Log | Observed spend $ | Calls |")
        P("| --- | ---: | ---: |")
        total = 0.0
        for path in sorted(logs.glob("*.log")):
            hits = re.findall(r"- Observed spend: \*\*\$([0-9.]+)\*\* over (\d+) calls", path.read_text())
            if hits:
                usd, calls = hits[-1]
                total += float(usd)
                P(f"| `{path.name}` | {float(usd):.4f} | {calls} |")
        P(f"| **total, DAB** | **{total:.2f}** | |")

    # ---- DS
    P("\n### Data-science suite (synthetic ground truth, 2.0 grader)\n")
    ds_dir = ROOT / "ds"
    cli = {p.stem: ds_cli_summary(_load(p)) for p in sorted(ds_dir.glob("*.json"))
           if ds_dir.is_dir() and "model-alone" not in p.stem and _load(p) and "results" in _load(p)}
    ma = {p.stem: ds_model_alone_summary(_load(p)) for p in sorted(ds_dir.glob("model-alone-*.json"))
          if ds_dir.is_dir()}
    cols = list(cli) + list(ma)
    tasks = sorted({t for s in cli.values() for t in s["by_task"]} | {t for s in ma.values() for t in s["by_task"]})
    P("| Task | " + " | ".join(f"`{c}`" for c in cols) + " |")
    P("| --- |" + " ---: |" * len(cols))
    for t in tasks:
        P(f"| {t} | " + " | ".join(_fmt(cli[c]["by_task"].get(t)) if c in cli else _fmt(ma[c]["by_task"].get(t))
                                  for c in cols) + " |")
    P("| **mean** | " + " | ".join(_fmt(cli[c]["overall"]) if c in cli else _fmt(ma[c]["mean_score"]) for c in cols) + " |")

    # ---- conversational
    P("\n### Conversational integrity (deterministic scoring; scripted faithful / sycophant brackets)\n")
    conv_dir = ROOT / "conv"
    conv_cells: Dict[str, Dict[str, Any]] = {}
    for path in sorted(conv_dir.glob("*.json")) if conv_dir.is_dir() else []:
        rec = _load(path)
        if not rec or "results" not in rec:
            continue
        for cid, tasks_ in harness_cells(rec).items():
            conv_cells[cid] = {tid: t["stats"] for tid, t in tasks_.items()}
    probes = sorted({p for c in conv_cells.values() for p in c})
    cids = list(conv_cells)
    P("| Probe | " + " | ".join(f"`{c}`" for c in cids) + " |")
    P("| --- |" + " ---: |" * len(cids))
    for p in probes:
        P(f"| {p} | " + " | ".join(_fmt(conv_cells[c].get(p, {}).get("pass_at_1")) for c in cids) + " |")
    P("| **mean pass@1** | " + " | ".join(_fmt(_mean(s.get("pass_at_1") for s in conv_cells[c].values())) for c in cids) + " |")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
