"""Release dataset harness: run every dataset in a corpus manifest through the
CLI fleet and/or the container API, and record what actually happened.

    python scripts/release_test/run_datasets.py --corpus corpus.json --out results \
        --paths cli,api --api http://127.0.0.1:9000 --model-label local/gemma4:e4b

Each (dataset, path) produces one JSON line in <out>/results.jsonl with the exit
status, wall time, peak RSS of the process tree (CLI), clones dispatched and
returned (from the swarm trace, not from the CLI's own summary), whether the
validator firefly ran, the final insight text and its class, and the tail of any
crash log. Nothing here is a pass/fail oracle: the matrix is built from these
records and read by a person. A record with ``checked: false`` is one where the
harness could not verify anything (upload refused, timeout) and must not be
counted as a pass.

Datasets are untrusted input: this script only reads them, through the same
loaders the product uses, and never executes anything from their directory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import classify_cli, classify_llm  # noqa: E402

DEFAULT_GOAL = "What stands out in this data, and what should we decide because of it?"
ANALYZE_OBS_START = "Here's what jumped out at me:"
ANALYZE_OBS_END = ("What I found", "Mission economics", "Deliverables")


# ----------------------------------------------------------------------------- data
def load_frame(path: str, max_rows: int = 300_000) -> pd.DataFrame:
    """A frame to check claims against: the product's own loader, so the check
    sees the same columns the narrative saw (and fails the same way)."""
    from avaloka.io import load_dataset  # the CLI's loader
    frame = load_dataset(path).frame
    # A random sample, not head(): the first rows of a sorted export are not the file.
    return frame.sample(max_rows, random_state=0) if len(frame) > max_rows else frame


# ----------------------------------------------------------------------------- process watch
class PeakRSS(threading.Thread):
    """Poll a process tree's RSS every 0.5 s; cheap enough and honest enough."""

    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.pid, self.peak, self._halt = pid, 0, threading.Event()

    def run(self) -> None:
        import psutil
        try:
            root = psutil.Process(self.pid)
        except psutil.Error:
            return
        while not self._halt.is_set():
            total = 0
            try:
                for p in [root, *root.children(recursive=True)]:
                    try:
                        total += p.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                break
            self.peak = max(self.peak, total)
            self._halt.wait(0.5)

    def stop(self) -> int:
        self._halt.set()
        self.join(timeout=2)
        return self.peak


def run_cmd(cmd: list[str], timeout: int, cwd: str | None = None) -> dict[str, Any]:
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8", "COLUMNS": "200"})
    watch = PeakRSS(proc.pid); watch.start()
    try:
        out, err = proc.communicate(timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill(); out, err = proc.communicate(); code = "timeout"
    peak = watch.stop()
    return {"exit": code, "wall_s": round(time.monotonic() - t0, 1), "peak_rss_mb": round(peak / 2**20),
            "stdout": out, "stderr": err}


# ----------------------------------------------------------------------------- swarm trace
def trace_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = [r.get("type") for r in records]
    completed = [r for r in records if r.get("type") == "clone.step.completed"]
    validator = [r for r in completed if r.get("firefly") == "validator"]
    return {
        "records": len(records),
        "clones_dispatched": kinds.count("clone.dispatched"),
        "clones_returned": kinds.count("clone.returned"),
        "steps_completed": len(completed),
        "fireflies": sorted({r.get("firefly") for r in completed if r.get("firefly")}),
        "validator_ran": bool(validator),
        "validator_outcomes": sorted({str(r.get("outcome")) for r in validator}),
        "clone_verdicts": sorted({str(r.get("verdict")) for r in records if r.get("type") == "clone.returned"}),
        "finished": "swarm.finished" in kinds,
    }


# ----------------------------------------------------------------------------- CLI path
def run_cli(ds: dict[str, Any], out_dir: Path, workers: int, timeout: int) -> dict[str, Any]:
    goal = ds.get("goal") or DEFAULT_GOAL
    fleet = out_dir / "fleet"; fleet.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"dataset": ds["name"], "path": "cli", "model": "none (CLI narrative is deterministic)",
                           "format": ds.get("format"), "checked": False}
    r = run_cmd([sys.executable, "-m", "avaloka", "coordinate", ds["path"], "--workers", str(workers),
                 "--goal", goal, "--output", str(fleet), "--quiet"], timeout)
    rec["coordinate"] = {k: r[k] for k in ("exit", "wall_s", "peak_rss_mb")}
    (out_dir / "coordinate.stdout").write_text(r["stdout"]); (out_dir / "coordinate.stderr").write_text(r["stderr"])
    if r["exit"] != 0:
        rec["coordinate"]["error_tail"] = (r["stderr"] or r["stdout"])[-1500:]
    trace_file = fleet / "swarm_trace.jsonl"
    if trace_file.exists():
        records = [json.loads(l) for l in trace_file.read_text().splitlines() if l.strip()]
        rec["swarm"] = trace_summary(records)
        rec["checked"] = True
    coord = fleet / "coordination.json"
    if coord.exists():
        c = json.loads(coord.read_text())
        rec["coordinate"]["verdict"] = c.get("verdict"); rec["coordinate"]["n_rows"] = c.get("n_rows")
        rec["coordinate"]["swarm_clones"] = (c.get("execution") or {}).get("swarm", {}).get("clones")
        id_as_cat = [col["name"] for col in c.get("converged_profile", {}).get("columns", [])
                     if col.get("role") == "categorical" and col.get("top_values")
                     and col["top_values"][0].get("count", 0) <= 1]
        rec["coordinate"]["id_like_categoricals"] = id_as_cat[:5]
    # The persona narrative (where "X leans hard on 'Y' (0% of rows)" comes from) is
    # produced by `analyze`; coordinate prints profiles, not observations.
    an = out_dir / "analyze"
    a = run_cmd([sys.executable, "-m", "avaloka", "analyze", ds["path"], "--goal", goal, "--output", str(an)], timeout)
    rec["analyze"] = {k: a[k] for k in ("exit", "wall_s", "peak_rss_mb")}
    (out_dir / "analyze.stdout").write_text(a["stdout"]); (out_dir / "analyze.stderr").write_text(a["stderr"])
    if a["exit"] != 0:
        rec["analyze"]["error_tail"] = (a["stderr"] or a["stdout"])[-1500:]
    obs = extract_observations(a["stdout"])
    rec["insight"] = "\n".join(obs)
    rec["goal_answered"] = "does not map to a supported goal-specific calculation" not in a["stdout"]
    if obs:
        try:
            frame = load_frame(ds["path"])
            rec["insight_class"], rec["insight_reason"] = classify_cli(obs, frame)
            rec["checked"] = True
        except Exception as exc:  # the loader failing is itself a finding
            rec["insight_class"], rec["insight_reason"] = "unchecked", f"loader failed on recheck: {exc!r}"[:300]
    elif a["exit"] == 0:
        rec["insight_class"], rec["insight_reason"] = "vacuous", "analyze succeeded but printed no observation"
    else:
        rec["insight_class"], rec["insight_reason"] = "crash", (a["stderr"] or a["stdout"])[-300:]
    return rec


def extract_observations(stdout: str) -> list[str]:
    text = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
    # Rich wraps lines at the console width; join continuation lines first.
    lines, obs, on = text.splitlines(), [], False
    for line in lines:
        s = line.strip()
        if ANALYZE_OBS_START in s:
            on = True; continue
        if on and any(s.startswith(e) for e in ANALYZE_OBS_END):
            break
        if on and s:
            if s.startswith("Avaloka"):
                obs.append(s.replace("Avaloka", "", 1).strip())
            elif obs:
                obs[-1] += " " + s
    return obs


# ----------------------------------------------------------------------------- API path
class Api:
    def __init__(self, base: str, secret: str, user: str = "release-test"):
        import jwt, httpx
        self.base = base.rstrip("/")
        token = jwt.encode({"sub": user, "iat": int(time.time())}, secret, algorithm="HS256")
        self.h = {"Authorization": f"Bearer {token}"}
        self.c = httpx.Client(base_url=self.base, headers=self.h, timeout=httpx.Timeout(600.0, connect=10.0))

    def upload(self, path: str) -> dict[str, Any]:
        with open(path, "rb") as f:
            r = self.c.post("/api/upload", files={"file": (Path(path).name, f)})
        return {"status": r.status_code, "body": _json(r)}

    def swarm_mission(self, thread_id: str, goal: str, workers: int, timeout: int) -> dict[str, Any]:
        r = self.c.post("/api/swarm/missions", json={"source": {"thread_id": thread_id}, "goal": goal, "workers": workers})
        if r.status_code != 200:
            return {"status": r.status_code, "body": _json(r)}
        mid = r.json().get("mission_id") or r.json().get("id")
        t0 = time.monotonic(); recs: list[dict[str, Any]] = []
        while time.monotonic() - t0 < timeout:
            rr = self.c.get(f"/api/swarm/missions/{mid}/records", params={"after": 0, "limit": 5000})
            body = _json(rr)
            recs = body.get("records", body) if isinstance(body, dict) else body
            if isinstance(recs, list) and any(x.get("type") in ("swarm.finished",) for x in recs if isinstance(x, dict)):
                break
            time.sleep(1.0)
        out = {"status": 200, "mission_id": mid, "wall_s": round(time.monotonic() - t0, 1)}
        if isinstance(recs, list):
            out["swarm"] = trace_summary([x for x in recs if isinstance(x, dict)])
            out["timed_out"] = not out["swarm"]["finished"]
        tel = self.c.get(f"/api/swarm/missions/{mid}/telemetry")
        out["telemetry_status"] = tel.status_code
        out["telemetry_text"] = tel.text[:200_000]
        return out

    def chat(self, thread_id: str, dataset_id: str, content: str, timeout: int) -> dict[str, Any]:
        t0 = time.monotonic()
        r = self.c.post(f"/threads/{thread_id}/messages", json={"content": content, "metadata": {"dataset_id": dataset_id}})
        body = _json(r)
        out: dict[str, Any] = {"status": r.status_code, "deferred": False}
        if r.status_code != 200:
            out["body"] = body; out["wall_s"] = round(time.monotonic() - t0, 1); return out
        deferred = body.get("analysis_task_id") if isinstance(body, dict) else None
        if deferred:
            out["deferred"] = True
            while time.monotonic() - t0 < timeout:
                p = self.c.get(f"/threads/{thread_id}/pending-turn", params={"deferred_id": deferred})
                pb = _json(p)
                if isinstance(pb, dict) and pb.get("status") in ("done", "error", "superseded", "none"):
                    out["pending_status"] = pb.get("status")
                    body = pb.get("result") or pb.get("response") or pb
                    break
                time.sleep(2.0)
            else:
                out["pending_status"] = "timeout"
        out["wall_s"] = round(time.monotonic() - t0, 1)
        out["body"] = body
        out["text"] = last_assistant_text(body)
        return out

    def trace(self, thread_id: str) -> dict[str, Any]:
        r = self.c.get(f"/threads/{thread_id}/trace")
        return {"status": r.status_code, "body": _json(r)}


def _json(r) -> Any:
    try:
        return r.json()
    except Exception:
        return {"_text": r.text[:2000]}


def last_assistant_text(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    msgs = body.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") in ("assistant", "ai", None) and m.get("content"):
            return str(m["content"])
    for key in ("output", "summary", "content", "message"):
        if isinstance(body.get(key), str):
            return body[key]
    return ""


def run_api(ds: dict[str, Any], out_dir: Path, api: Api, model_label: str, workers: int, timeout: int,
            canary: list[str]) -> dict[str, Any]:
    goal = ds.get("goal") or DEFAULT_GOAL
    rec: dict[str, Any] = {"dataset": ds["name"], "path": "api", "model": model_label, "format": ds.get("format"),
                           "checked": False}
    p = Path(ds["path"])
    if p.is_dir():
        rec["upload"] = {"status": "skipped", "reason": "API upload takes files, not directories"}
        rec["insight_class"], rec["insight_reason"] = "not run", "directory source has no API upload path"
        return rec
    t0 = time.monotonic()
    up = api.upload(ds["path"])
    rec["upload"] = {"status": up["status"], "wall_s": round(time.monotonic() - t0, 1)}
    body = up["body"]
    if up["status"] != 200 or not isinstance(body, dict):
        rec["upload"]["error"] = json.dumps(body)[:800]
        rec["insight_class"], rec["insight_reason"] = "not run", f"upload refused: {json.dumps(body)[:200]}"
        return rec
    thread_id = body.get("thread_id")
    dataset_id = body.get("dataset_id") or ((body.get("datasets") or [{}])[0].get("dataset_id"))
    schema = body.get("schema")
    columns = list(schema.keys()) if isinstance(schema, dict) else list(schema or [])
    rec["upload"].update({"thread_id": thread_id, "rows_sampled": body.get("rows_sampled"),
                          "n_columns": len(columns), "file_size_mb": body.get("file_size_mb")})
    # (1) coordinated swarm on the uploaded dataset -- deterministic, traced per clone
    sw = api.swarm_mission(thread_id, goal, workers, timeout)
    rec["swarm_mission"] = {k: v for k, v in sw.items() if k not in ("telemetry_text",)}
    if "swarm" in sw:
        rec["checked"] = True
        leaks = [c for c in canary if c and c in (sw.get("telemetry_text") or "")]
        rec["telemetry_leak"] = leaks  # planted column/file names found in the telemetry projection
        (out_dir / "swarm_telemetry.json").write_text(sw.get("telemetry_text") or "")
    # (2) the chat turn -- this is where the model (gemma / Groq) is actually used
    ch = api.chat(thread_id, dataset_id, goal, timeout)
    rec["chat"] = {k: v for k, v in ch.items() if k != "body"}
    (out_dir / "chat_response.json").write_text(json.dumps(ch.get("body"), indent=1, default=str)[:400_000])
    rec["insight"] = ch.get("text", "")
    if ch["status"] == 200 and ch.get("pending_status") not in ("timeout", "error"):
        try:
            frame = load_frame(ds["path"])
        except Exception:
            frame = None
        rec["insight_class"], rec["insight_reason"] = classify_llm(rec["insight"], columns, frame)
        rec["checked"] = True
    else:
        rec["insight_class"] = "crash" if ch["status"] >= 500 or ch.get("pending_status") == "error" else "not run"
        rec["insight_reason"] = f"chat status {ch['status']} pending={ch.get('pending_status')}: {json.dumps(ch.get('body'))[:300]}"
    tr = api.trace(thread_id)
    if tr["status"] == 200 and isinstance(tr["body"], dict):
        agents = tr["body"].get("agents") or {}
        if isinstance(agents, dict):
            names = sorted(agents.keys())
        else:
            names = sorted({str(a.get("agent") or a.get("name") or a.get("id")) for a in agents if isinstance(a, dict)})
        rec["trace"] = {"agents": names, "turns": len(tr["body"].get("turns") or [])}
        (out_dir / "thread_trace.json").write_text(json.dumps(tr["body"], indent=1, default=str)[:400_000])
    else:
        rec["trace"] = {"status": tr["status"]}
    return rec


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="JSON list of {name, path, format, domain, notes, goal}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--paths", default="cli", help="comma list of cli,api")
    ap.add_argument("--api", default="http://127.0.0.1:9000")
    ap.add_argument("--model-label", default=os.environ.get("RELEASE_TEST_MODEL_LABEL", "unknown"))
    ap.add_argument("--jwt-secret", default=os.environ.get("SUPABASE_JWT_SECRET", ""))
    ap.add_argument("--only", default="", help="comma list of dataset names")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1500, help="seconds per command / per API step")
    ap.add_argument("--canary", default="zq_telemetry_canary_7731", help="comma list of planted strings to grep in telemetry")
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    only = {s for s in args.only.split(",") if s}
    paths = [p for p in args.paths.split(",") if p]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    api = Api(args.api, args.jwt_secret) if "api" in paths else None
    canary = [c for c in args.canary.split(",") if c]
    for ds in corpus:
        if only and ds["name"] not in only:
            continue
        for path in paths:
            label = args.model_label if path == "api" else "cli"
            d = out / ds["name"] / re.sub(r"[^A-Za-z0-9_.-]", "_", f"{path}-{label}")
            d.mkdir(parents=True, exist_ok=True)
            print(f"== {ds['name']} [{path}] {label}", flush=True)
            try:
                rec = run_cli(ds, d, args.workers, args.timeout) if path == "cli" else \
                    run_api(ds, d, api, args.model_label, args.workers, args.timeout,
                            canary + [Path(ds["path"]).name])
            except Exception as exc:  # the harness must not die on one dataset
                rec = {"dataset": ds["name"], "path": path, "model": label, "checked": False,
                       "insight_class": "harness-error", "insight_reason": repr(exc)[:500]}
            rec.update({"domain": ds.get("domain"), "notes": ds.get("notes"), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            with results.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            summary = {k: rec.get(k) for k in ("insight_class", "insight_reason", "checked")}
            print("   ", json.dumps(summary, default=str)[:400], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
