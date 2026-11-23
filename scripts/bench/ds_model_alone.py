#!/usr/bin/env python3
"""Model-alone baseline for the CLI data-science suite.

Companion to ``docs/benchmarks/2.0-vs-1.0.md`` §3.2. The Avaloka CLI (1.0 and
2.0) runs ``avaloka.benchmark``'s data-science tasks with a deterministic
pipeline of fireflies and a validator; this script hands the SAME tasks — the
same seeded dataset builders, the same goals, targets and metrics, and the
same ``Expectations`` — to a bare foundation model, once, with no feedback.

The model returns one JSON report (task, split, column roles, leakage,
verdict, deployment ceiling, its own baseline) plus one training program. The
program runs once; the harness computes the holdout metric itself on a
deterministic 75/25 cut (the same cut ``avaloka.benchmark.scoring._baseline_score``
uses) and compares it with that same naive baseline. Nothing the model says
about its own quality is graded: ``beats_baseline`` here is measured.

Leakage is graded exactly as the 2.0 scorer grades the CLI: the leaking
columns must be named and the verdict/ceiling must refuse certification. It is
NOT graded by holdout AUC: on ``leaky`` the legitimate features alone reach
AUC 0.97 and the 2.0 CLI itself trains to 1.0 there, so a threshold cannot
separate honest from leaking models and would penalise the CLI's own answer.
``--rescore`` recomputes records written before this rule and keeps the
dropped check in them, marked ``dropped``.

Checks mirror ``avaloka/benchmark/scoring.py`` by name and weight where a bare
model can be held to them. Three 2.0 checks have no model-alone equivalent and
are omitted: ``economics_sane`` and ``firefly_*`` (harness structure) and
``temporal_split_performed`` (needs the fitted pipeline; the report's chosen
split is still graded as ``plan_split``). The report says so.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from avaloka.benchmark.scoring import _baseline_score  # noqa: E402
from avaloka.benchmark.suite import data_science_tasks  # noqa: E402
from benchmarks.adapters.llm_direct import PROVIDERS, _answer_text, _key_for  # noqa: E402
from benchmarks.core.pricing import price_for  # noqa: E402
from benchmarks.core.reporting import repository_metadata  # noqa: E402
from app.executor.env import scrubbed_env  # noqa: E402

PROMPT_VERSION = "ds-model-alone-v1"
HOLDOUT_CUT = 0.75          # identical to avaloka.benchmark.scoring._baseline_score
AUC_MARGIN = 0.02           # "materially better" when the task sets no margin
RMSE_REL_MARGIN = 0.02
DROPPED_CHECKS = ("leak_not_exploited",)   # see module docstring; --rescore removes them

SYSTEM_PROMPT = """You are a data scientist given one dataset and one goal. You get exactly one attempt and no feedback.
Return ONE JSON object and nothing else (no prose, no code fences) with exactly these keys:
- "task": "classification" or "regression"
- "split_strategy": "random_holdout", "stratified_holdout" or "time_based_holdout"
- "column_roles": an object mapping EVERY column to one of "id", "numeric", "categorical", "datetime", "constant", "target"
- "quality_flags": {"constant_columns": [...], "id_columns": [...], "high_missing_columns": [...]}
- "leakage_columns": columns that leak the target and must not be used as features
- "features_used": the columns your program actually trains on
- "baseline_strategy": the naive baseline you compare against (e.g. "majority_class", "prior", "mean")
- "baseline_score": your estimate of that baseline's score on the stated metric (a number)
- "expected_beats_baseline": true if you expect your model to beat the baseline materially; false if the data carries no usable signal
- "verdict": "pass", "warn" or "fail" — "fail" when the model must not be certified (target leakage, no signal, invalid validation)
- "max_safe_deployment_level": an integer 1-4 (1 = research artifact only, never serve; 2 = internal or shadow use; 3 = supervised deployment; 4 = production)
- "program": Python 3 source, as a JSON string, that reads ./train.csv (features and target) and ./test.csv (the same features, no target), fits a model, and writes ./predictions.csv with one column "prediction" aligned to the rows of test.csv — the probability that the target equals 1 for classification, the predicted value for regression. Only pandas, numpy and scikit-learn are available; read no other file; print nothing else.
Be honest in the report: if the data has no signal, say so and refuse to certify; if a column leaks the target, exclude it and say so."""


def _profile(frame: pd.DataFrame, target: str) -> str:
    lines = []
    for col in frame.columns:
        s = frame[col]
        lines.append(f"- {col}: dtype={s.dtype}, n_unique={s.nunique(dropna=True)}, "
                     f"missing={int(s.isna().sum())}" + ("  <- target" if col == target else ""))
    return "\n".join(lines)


def _user_message(task, frame: pd.DataFrame, train: pd.DataFrame, target: str) -> str:
    sample = train.head(25).to_csv(index=False)
    return (
        f"GOAL: {task.goal}\n"
        f"TARGET COLUMN: {target}\nMETRIC: {task.metric}\n"
        f"DEPLOYABLE: {'yes, the user wants to deploy this model' if task.deployable else 'no'}\n"
        f"ROWS: {len(train)} in train.csv, {len(frame) - len(train)} in test.csv\n"
        f"COLUMNS:\n{_profile(train, target)}\n\n"
        f"FIRST 25 ROWS OF train.csv:\n{sample}"
    )


_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S)


def _parse_report(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _call_model(provider: str, model: str, system: str, user: str, *,
                max_tokens: int, timeout: float) -> Dict[str, Any]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.2,
    }).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "avaloka-bench/1.0"}
    key = _key_for(provider)
    if provider != "ollama" and not key:
        raise RuntimeError(f"no key for {provider}; set {'/'.join(PROVIDERS[provider]['key_env'])}")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(PROVIDERS[provider]["url"], data=body, headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"provider_error: HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}")
    latency = time.perf_counter() - started
    choice = payload["choices"][0]
    return {"text": _answer_text(choice["message"].get("content")),
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage") or {}, "latency_s": latency,
            "generation_id": payload.get("id")}


def program_env(home: str) -> Dict[str, str]:
    """The model-written program's environment: the executor's allowlist, a private HOME.

    It ran with ``{**os.environ}``, so every provider key this script needs was
    readable by the code the model wrote. ``scrubbed_env`` is the same
    allowlist ``app.executor`` applies to product scripts.
    """
    return scrubbed_env(extra={"HOME": home, "PYTHONDONTWRITEBYTECODE": "1"})


def _run_program(program: str, train: pd.DataFrame, test_features: pd.DataFrame,
                 timeout: int) -> tuple[Optional[np.ndarray], str]:
    with tempfile.TemporaryDirectory(prefix="ds-model-alone-") as td:
        work = Path(td)
        train.to_csv(work / "train.csv", index=False)
        test_features.to_csv(work / "test.csv", index=False)
        (work / "program.py").write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", "program.py"], cwd=work, capture_output=True,
                                  text=True, timeout=timeout, check=False,
                                  env=program_env(td))
        except subprocess.TimeoutExpired:
            return None, f"timeout: program exceeded {timeout}s"
        if proc.returncode:
            return None, f"execution_error: exit {proc.returncode}: {proc.stderr[-600:]}"
        pred_path = work / "predictions.csv"
        if not pred_path.is_file():
            return None, "execution_error: no predictions.csv written"
        preds = pd.read_csv(pred_path)
        if "prediction" not in preds.columns or len(preds) != len(test_features):
            return None, (f"answer_error: predictions.csv has columns {list(preds.columns)} and "
                          f"{len(preds)} rows; expected 'prediction' x {len(test_features)}")
        values = pd.to_numeric(preds["prediction"], errors="coerce").to_numpy()
        if np.isnan(values).any():
            return None, "answer_error: non-numeric or missing predictions"
        return values, proc.stderr[-400:]


def _measure(metric: str, y_true: pd.Series, pred: np.ndarray) -> float:
    from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
    if metric == "roc_auc":
        return float(roc_auc_score(y_true.astype(int), pred))
    if metric == "r2":
        return float(r2_score(y_true, pred))
    return float(math.sqrt(mean_squared_error(y_true, pred)))  # rmse


def _check(name: str, passed: bool, detail: str, weight: float = 1.0) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail, "weight": float(weight)}


def grade(task, exp, report: Dict[str, Any], measured: Optional[float], baseline: float,
          error: Optional[str]) -> List[Dict[str, Any]]:
    metric = task.metric
    lower_better = metric in {"rmse", "mae"}
    checks: List[Dict[str, Any]] = []
    roles = report.get("column_roles") or {}
    flags = report.get("quality_flags") or {}
    leaks = set(report.get("leakage_columns") or [])
    used = set(report.get("features_used") or [])

    if measured is None:
        checks.append(_check("model_present", False, f"no gradable predictions: {error}", 2.0))
    else:
        checks.append(_check("model_present", True, f"{metric}={measured:.4f} on the harness holdout", 2.0))
        if exp.min_score is not None:
            ok = measured <= exp.min_score if lower_better else measured >= exp.min_score
            checks.append(_check("model_min_score", ok, f"{metric}={measured:.4f} vs floor {exp.min_score}", 2.0))
        gain = (baseline - measured) if lower_better else (measured - baseline)
        if exp.min_gain_over_baseline is not None:
            checks.append(_check("beats_baseline", gain >= exp.min_gain_over_baseline,
                                 f"{metric}: model={measured:.4f}, baseline={baseline:.4f}, "
                                 f"gain={gain:+.4f} >= {exp.min_gain_over_baseline} (measured)", 2.0))
        if exp.expect_beats_baseline is not None:
            margin = (RMSE_REL_MARGIN * abs(baseline)) if lower_better else AUC_MARGIN
            beats = gain >= margin
            checks.append(_check("beats_baseline", beats is exp.expect_beats_baseline,
                                 f"measured {metric}: model={measured:.4f}, baseline={baseline:.4f}, "
                                 f"gain={gain:+.4f}, margin={margin:.4f}; beats={beats}, "
                                 f"expected {exp.expect_beats_baseline}", 3.0))
    for col in exp.must_detect_leakage:
        checks.append(_check(f"leakage[{col}]", col in leaks,
                             f"'{col}' {'flagged' if col in leaks else 'NOT flagged'}; leaks={sorted(leaks)}", 2.0))
    if exp.expected_verdict:
        got = report.get("verdict")
        checks.append(_check("verdict", got == exp.expected_verdict,
                             f"verdict '{got}', expected '{exp.expected_verdict}'", 2.0))
    if exp.expected_max_level is not None:
        got = report.get("max_safe_deployment_level")
        checks.append(_check("max_safe_deployment_level", got == exp.expected_max_level,
                             f"ceiling {got}, expected {exp.expected_max_level}", 2.0))
    if exp.expected_task:
        got = report.get("task")
        checks.append(_check("plan_task", got == exp.expected_task, f"task '{got}', expected '{exp.expected_task}'"))
    if exp.expected_split:
        got = report.get("split_strategy")
        checks.append(_check("plan_split", got == exp.expected_split,
                             f"split '{got}', expected '{exp.expected_split}' (reported, not verified)", 2.0))
    if exp.require_baseline_reported:
        base = report.get("baseline_score")
        checks.append(_check("baseline_reported", isinstance(base, (int, float)) and not isinstance(base, bool)
                             and base == base, f"baseline_score={base!r} ({report.get('baseline_strategy')})", 2.0))
    for col, want in exp.expected_roles.items():
        got = roles.get(col)
        checks.append(_check(f"role[{col}]", got == want, f"inferred '{got}', expected '{want}'"))
    for key, cols in exp.expect_quality_flags.items():
        flagged = set(flags.get(key) or [])
        for col in cols:
            checks.append(_check(f"quality[{key}:{col}]", col in flagged,
                                 f"'{col}' {'in' if col in flagged else 'MISSING from'} {key}={sorted(flagged)}"))
    for name in exp.must_use_features:
        checks.append(_check(f"feature_used[{name}]", name in used,
                             f"{name} {'kept' if name in used else 'DROPPED'} (model saw: {sorted(used)})", 2.0))
    return checks


def score(checks: List[Dict[str, Any]]) -> tuple[bool, float]:
    total = sum(c["weight"] for c in checks)
    earned = sum(c["weight"] for c in checks if c["passed"])
    return all(c["passed"] for c in checks), (earned / total if total else 0.0)


def rescore(path: Path) -> None:
    record = json.loads(path.read_text())
    for row in record["results"]:
        kept = [c for c in row["checks"] if c["name"] not in DROPPED_CHECKS]
        for c in row["checks"]:
            if c["name"] in DROPPED_CHECKS:
                c["dropped"] = True
        row["passed"], s = score(kept)
        row["score"] = round(s, 4)
    rows = record["results"]
    by_task: Dict[str, List[float]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["score"])
    record["summary"].update({
        "mean_score": round(sum(r["score"] for r in rows) / len(rows), 4) if rows else None,
        "pass_rate": round(sum(1 for r in rows if r["passed"]) / len(rows), 4) if rows else None,
        "by_task": {k: round(sum(v) / len(v), 4) for k, v in by_task.items()},
    })
    record.setdefault("rescored", []).append({"at": datetime.now(timezone.utc).isoformat(),
                                              "dropped_checks": list(DROPPED_CHECKS)})
    path.write_text(json.dumps(record, indent=2, default=str))
    print(f"rescored {path}: mean {record['summary']['mean_score']} pass-rate {record['summary']['pass_rate']}")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--rescore":
        for p in sys.argv[2:]:
            rescore(Path(p))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--price-model", default=None, help="catalog id for the cost column when it differs")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--task", action="append", help="only these task names (repeatable)")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--timeout", type=int, default=180, help="program execution timeout (s)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = [t for t in data_science_tasks() if t.kind == "train"]
    if args.task:
        tasks = [t for t in tasks if t.name in set(args.task)]
    price = price_for(args.provider, args.price_model or args.model)
    rows: List[Dict[str, Any]] = []
    spent = 0.0
    with tempfile.TemporaryDirectory(prefix="ds-data-") as td:
        for task in tasks:
            dest = Path(td) / f"{task.dataset.name}{task.dataset.suffix}"
            task.dataset.builder(dest)
            frame = pd.read_csv(dest)
            target = task.target or task.dataset.target
            metric = task.metric
            kind = "classification" if metric in {"roc_auc", "accuracy", "f1", "f1_macro"} else "regression"
            cut = int(len(frame) * HOLDOUT_CUT)
            train, test = frame.iloc[:cut], frame.iloc[cut:]
            baseline = _baseline_score(frame, target, kind, metric)
            user = _user_message(task, frame, train, target)
            for trial in range(args.trials):
                started = time.perf_counter()
                report: Dict[str, Any] = {}
                measured = None
                error = None
                usage: Dict[str, Any] = {}
                try:
                    call = _call_model(args.provider, args.model, SYSTEM_PROMPT, user,
                                       max_tokens=args.max_tokens, timeout=600)
                    usage = call["usage"]
                    if call["finish_reason"] == "length":
                        error = f"truncation_error: finish_reason='length' at max_tokens={args.max_tokens}"
                    else:
                        report = _parse_report(call["text"])
                        program = report.get("program") or ""
                        if not isinstance(program, str) or not program.strip():
                            error = "answer_error: report carries no program"
                        else:
                            preds, note = _run_program(program, train, test.drop(columns=[target]), args.timeout)
                            if preds is None:
                                error = note
                            else:
                                measured = _measure(metric, test[target], preds)
                except json.JSONDecodeError as exc:
                    error = f"answer_error: report is not JSON: {exc}"
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}" if "provider_error" not in str(exc) else str(exc)
                checks = grade(task, task.expectations, report, measured, baseline, error)
                passed, s = score(checks)
                pt, ct = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
                cost = price.cost_usd(pt, ct)
                spent += cost
                row = {
                    "task": task.name, "family": task.family, "trial": trial, "metric": metric,
                    "passed": passed, "score": round(s, 4), "checks": checks,
                    "measured": {"model": measured, "baseline": baseline,
                                 "holdout": f"last {100 - int(HOLDOUT_CUT * 100)}% of rows"},
                    "report": {k: v for k, v in report.items() if k != "program"},
                    "program_chars": len(report.get("program") or "") if isinstance(report.get("program"), str) else 0,
                    "error": error, "prompt_tokens": pt, "completion_tokens": ct,
                    "cost_usd": cost, "cost_source": price.source if usage else "usage_missing",
                    "latency_s": round(time.perf_counter() - started, 2),
                }
                rows.append(row)
                print(f"[{'PASS' if passed else 'FAIL'}] {task.name:45s} trial {trial} score={s:.2f} "
                      f"{metric}={measured if measured is None else round(measured, 4)} base={baseline:.4f} "
                      f"${cost:.4f} {('ERR ' + error[:80]) if error else ''}", flush=True)

    by_task: Dict[str, List[float]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["score"])
    record = {
        "schema_version": "ds-model-alone-1",
        "prompt_version": PROMPT_VERSION,
        "system": "model_alone",
        "provider": args.provider, "model": args.model,
        "price": price.as_dict() if hasattr(price, "as_dict") else str(price),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_metadata(REPO),
        "trials": args.trials,
        "omitted_checks": ["economics_sane", "firefly_active[*]", "firefly_dormant[*]",
                           "temporal_split_performed"],
        "summary": {
            "mean_score": round(sum(r["score"] for r in rows) / len(rows), 4) if rows else None,
            "pass_rate": round(sum(1 for r in rows if r["passed"]) / len(rows), 4) if rows else None,
            "by_task": {k: round(sum(v) / len(v), 4) for k, v in by_task.items()},
            "spend_usd": round(spent, 4),
        },
        "results": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str))
    print(f"\nmean score {record['summary']['mean_score']}  pass-rate {record['summary']['pass_rate']}  "
          f"spend ${spent:.4f}\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
