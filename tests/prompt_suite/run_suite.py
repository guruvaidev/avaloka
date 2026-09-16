#!/usr/bin/env python3
"""Avaloka test-suite runner — v1 / v2 / v4, batched, live terminal output.

Endpoints/credentials come from env vars (see README): AVALOKA_API_URL,
AVALOKA_AUTH_URL, AVALOKA_ANON_KEY, AVALOKA_TEST_EMAIL/PASSWORD, AVALOKA_CONNS.

  python3 run_suite.py --suite v4 --list                 # show batches
  python3 run_suite.py --suite v4 --batch F2             # run one batch
  python3 run_suite.py --suite v4 --batch F2 --only GQ-4 # single item
  python3 run_suite.py --suite v4 --batch F3 --from TR-5 # resume mid-batch
  python3 run_suite.py --suite v2 --batch ieee-fraud

Every item prints PROMPT and OUTPUT live; results also append to
results/<suite>_<ts>.jsonl. Dataset registrations/uploads and threads are
cached in state.json so repeated runs don't re-register (--reset clears).

MTA rule: model *usage* is never prompted — IN-* items call the Inference-tab
API endpoints directly. DTA: uses the two-GCS-connection arc + optional
in-cluster Postgres arc.
"""
import argparse, json, os, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import urllib.request, urllib.error

BASE = Path(__file__).resolve().parent
API = os.environ.get("AVALOKA_API_URL", "http://localhost:9000")
KONG = os.environ.get("AVALOKA_AUTH_URL", "http://localhost:30091")
ANON = os.environ.get("AVALOKA_ANON_KEY", "")
EMAIL = os.environ.get("AVALOKA_TEST_EMAIL", "")
PASSWORD = os.environ.get("AVALOKA_TEST_PASSWORD", "")
DATA_DIR = Path.home() / "v3assets"           # housing.csv / train.csv live here
CONNS = Path(os.environ.get("AVALOKA_CONNS", str(Path.home() / "v3assets" / "conns.json")))
STATE_F = BASE / "state.json"
BUCKET_PREFIX = "gs://avaloka-test-user-filestore/user-upload/inputs"

C = {"b": "\033[1m", "dim": "\033[2m", "g": "\033[32m", "r": "\033[31m",
     "y": "\033[33m", "c": "\033[36m", "x": "\033[0m"}

def http(method, url, body=None, headers=None, timeout=1800, form=None):
    h = dict(headers or {})
    if form is not None:
        boundary = uuid.uuid4().hex
        parts = b""
        for k, v in form.get("fields", {}).items():
            parts += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
        for fname, fpath in form.get("files", []):
            parts += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
                      f"filename=\"{fname}\"\r\nContent-Type: text/csv\r\n\r\n").encode()
            parts += Path(fpath).read_bytes() + b"\r\n"
        parts += f"--{boundary}--\r\n".encode()
        h["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        data = parts
    elif body is not None:
        h.setdefault("Content-Type", "application/json")
        data = json.dumps(body).encode()
    else:
        data = None
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode(errors="replace")
            code = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace"); code = e.code
    except Exception as e:
        return 0, {"error": str(e)}, time.time() - t0
    try: parsed = json.loads(raw)
    except Exception: parsed = {"text": raw[:2000]}
    return code, parsed, time.time() - t0

# INFRA-GUARD
HEAVY_IDS = {"AN-11","AN-12","AN-13","AN-14","FS-3","FS-4","FS-5","TR-8","TR-9","VZ-3","VZ-6","VZ-7","MP-10","DT-7"}
BIG_FILES = ("yellow_tripdata", "sell_prices", "train_transaction")
def is_heavy(item):
    if item["id"] in HEAVY_IDS: return True
    f = (item.get("file") or "") + " " + (item.get("dataset") or "")
    pr = (item.get("prompt") or "").lower()
    return any(b in f for b in BIG_FILES) and any(w in pr for w in ("entire", "whole file", "exact", "all rows", "not the sample"))
def api_healthy():
    import urllib.request
    try:
        return urllib.request.urlopen(API + "/health", timeout=6).status == 200
    except Exception:
        return False
def recover_api():
    for _ in range(48):
        time.sleep(5)
        if api_healthy(): return True
    return False

class Runner:
    def __init__(self, reset=False):
        self.state = {} if reset or not STATE_F.exists() else json.loads(STATE_F.read_text())
        self.conns = json.loads(CONNS.read_text()) if CONNS.exists() else {}
        self.token = None
        self.results_f = None
        self.model_runs = self.state.setdefault("model_runs", {})   # MX-id -> run_id
        self.last_session_id = self.state.get("last_session_id")    # for thread-less ops

    def save(self): STATE_F.write_text(json.dumps(self.state, indent=1))

    def _note_session(self, sid):
        if sid:
            self.last_session_id = sid
            self.state["last_session_id"] = sid

    def auth_sess(self, sid=None):
        """Auth headers + X-Avaloka-Session so thread-less ops (transfer, db-query,
        task-list) can prove session ownership. Falls back to the last session seen."""
        h = {"Authorization": f"Bearer {self.token}"}
        sid = sid or self.last_session_id
        if sid:
            h["X-Avaloka-Session"] = sid
        return h

    def login(self):
        code, r, _ = http("POST", f"{KONG}/auth/v1/token?grant_type=password",
                          {"email": EMAIL, "password": PASSWORD}, {"apikey": ANON}, 30)
        assert code == 200 and r.get("access_token"), f"login failed: {code} {r}"
        self.token = r["access_token"]

    def auth(self): return {"Authorization": f"Bearer {self.token}"}

    # ---- dataset management -------------------------------------------------
    def dataset_for(self, spec):
        """spec: 'upload:housing.csv' | 'register:instacart-mba/aisles.csv'"""
        ds = self.state.setdefault("datasets", {})
        if spec in ds:
            self._note_session(ds[spec].get("session_id")); return ds[spec]
        op, _, rest = spec.partition(":")
        if op == "upload":
            code, r, secs = http("POST", f"{API}/api/upload", headers=self.auth(),
                                 form={"files": [(rest, str(DATA_DIR / rest))]}, timeout=600)
        else:
            conn, _, fname = rest.partition("/")
            cid = self.conns.get(conn)
            code, r, secs = http("POST", f"{API}/api/register-existing-storage", {
                "connection_id": cid, "storage_uri": f"{BUCKET_PREFIX}/{conn}/",
                "key": fname}, self.auth(), timeout=900)
        if code != 200:
            raise RuntimeError(f"dataset setup {spec} failed: http={code} {str(r)[:300]}")
        info = r.get("datasets", [r])[0] if isinstance(r.get("datasets"), list) else r
        entry = {"dataset_id": info.get("dataset_id"), "thread_id": r.get("thread_id"),
                 "session_id": r.get("session_id"), "secs": round(secs, 1)}
        ds[spec] = entry; self._note_session(entry.get("session_id")); self.save()
        return entry

    def thread_for(self, key, dataset_spec):
        th = self.state.setdefault("threads", {})
        if key in th: return th[key]
        d = self.dataset_for(dataset_spec) if dataset_spec else {}
        tid = d.get("thread_id")
        if not tid:
            code, r, _ = http("POST", f"{API}/threads", {"metadata": {}}, self.auth(), 60)
            tid = r.get("thread_id")
        th[key] = {"thread_id": tid, "dataset_id": d.get("dataset_id"),
                   "session_id": d.get("session_id")}
        self.save(); return th[key]

    # ---- executors ----------------------------------------------------------
    def run_chat(self, item, suite):
        tkey = item.get("thread")
        if not tkey and item.get("file"):
            # v1/v2 style: stateful arc per batch+file — share one thread, and
            # register the file from the batch's connection on first use.
            conn = item.get("_batch", "")
            if conn == "tier-c-confirmations": conn, item["file"] = "walmart", item.get("file") or "calendar.csv"
            # doc rows sometimes list two files ("aisles.csv / departments.csv") — anchor on the first
            fname = item["file"].split("/")[-1] if "/" in item["file"] and " " not in item["file"] else item["file"].split(" / ")[0].strip()
            stem = Path(fname).stem
            tkey = f"{conn}-{stem}"
            suite.setdefault("thread_datasets", {})[tkey] = f"register:{conn}/{fname}"
        tkey = tkey or f"fresh:{item['id'].lower()}"
        fresh = tkey.startswith("fresh:")
        key = tkey.split(":", 1)[1] if fresh else tkey
        if fresh and key in self.state.get("threads", {}):
            del self.state["threads"][key]
        dspec = suite.get("thread_datasets", {}).get(key)
        t = self.thread_for(key, dspec)
        body = {"role": "user", "content": item["prompt"]}
        if t.get("dataset_id"):
            body["metadata"] = {"dataset_id": t["dataset_id"]}
            body["dataset_ids"] = [t["dataset_id"]]
        code, r, secs = http("POST", f"{API}/threads/{t['thread_id']}/messages",
                             body, self.auth_sess(t.get("session_id")), timeout=1800)
        msgs = r.get("messages") or []
        reply = ""
        for m in reversed(msgs):
            if m.get("role") in ("assistant", "ai"):
                reply = m.get("content") or ""; break
        meta = {k: r.get(k) for k in ("analysis_fidelity", "ready_to_code", "selected_sample_name") if r.get(k) is not None}
        # capture MTA run ids when a training turn reports one
        if item["id"].startswith("MX") and code == 200:
            txt = json.dumps(r)
            import re as _re
            m = _re.search(r"run[_ ]?id[\"':\s]+([a-f0-9]{16,32})", txt, _re.I)
            if m: self.model_runs[item["id"]] = m.group(1); self.save()
        return code, reply or json.dumps(r)[:1500], secs, meta

    def _resolve_run_id(self, model_of=""):
        rid = self.model_runs.get(model_of, "")
        if rid: return rid
        code, r, s = http("GET", f"{API}/api/models", headers=self.auth(), timeout=60)
        models = r if isinstance(r, list) else r.get("models", [])
        return (models[0].get("run_id") or models[0].get("id")) if models else ""

    def run_api(self, item):
        op = item["op"]
        if op == "buckets_list":
            cid = self.conns.get(item["conn"])
            u = (f"{API}/buckets/list?backend=gcs&bucket=avaloka-test-user-filestore"
                 f"&prefix=user-upload/inputs/{item['conn']}/&connection_id={cid}")
            return http("GET", u, headers=self.auth(), timeout=60)
        if op == "tasks_list":
            return http("GET", f"{API}/tasks", headers=self.auth_sess(), timeout=60)
        if op == "models_get":
            code, r, s = http("GET", f"{API}/api/models", headers=self.auth(), timeout=60)
            return code, r, s
        if op == "model_delete_throwaway":
            code, r, s = http("GET", f"{API}/api/models", headers=self.auth(), timeout=60)
            models = r if isinstance(r, list) else r.get("models", [])
            if not models: return 0, {"skip": "no models to delete"}, s
            rid = models[-1].get("run_id") or models[-1].get("id")
            return http("DELETE", f"{API}/api/models/{rid}", headers=self.auth(), timeout=60)
        if op in ("batch_inference", "batch_inference_missing_col"):
            rid = self._resolve_run_id(item.get("model_of", ""))
            if not rid: return 0, {"skip": "no trained model available"}, 0
            cc, cr, cs = http("POST", f"{API}/api/models/{rid}/configure-inference-service",
                              {}, self.auth(), timeout=180)
            if cc != 200:
                return cc, {"inference_service_unavailable": cr}, cs
            rows = [{"median_income": 3.2, "housing_median_age": 25, "total_rooms": 2000,
                     "population": 900} for _ in range(item.get("rows", 5))]
            if op == "batch_inference_missing_col":
                rows = [{"median_income": 3.2}]
            return http("POST", f"{API}/api/models/{rid}/inference",
                        rows, self.auth(), timeout=300)
        if op == "configure_inference_service":
            rid = self.model_runs.get(item.get("model_of", ""), "")
            if not rid: return 0, {"skip": "no trained model"}, 0
            return http("POST", f"{API}/api/models/{rid}/configure-inference-service",
                        {}, self.auth(), timeout=120)
        if op == "stop_inference_service":
            code, r, s = http("GET", f"{API}/api/models", headers=self.auth(), timeout=60)
            models = r if isinstance(r, list) else r.get("models", [])
            rid = (models[0].get("run_id") or models[0].get("id")) if models else ""
            if not rid: return 0, {"skip": "no model"}, 0
            return http("POST", f"{API}/api/models/{rid}/stop-inference-service",
                        {}, self.auth(), timeout=60)
        return 0, {"skip": f"unknown api op {op}"}, 0

    def run_ingest(self, item):
        if item["op"] == "upload":
            return self._wrap_dataset(f"upload:{item['file']}")
        if item["op"] == "register":
            return self._wrap_dataset(f"register:{item['conn']}/{item['file']}")
        if item["op"] == "register_negative":
            conn = item["conn"]; cid = self.conns.get(conn)
            u = (f"{API}/buckets/list?backend=gcs&bucket=avaloka-test-user-filestore"
                 f"&prefix=user-upload/inputs/{conn}/&connection_id={cid}")
            code, r, _ = http("GET", u, headers=self.auth(), timeout=60)
            objs = r.get("objects") or r.get("items") or []
            key = Path(objs[0].get("key", objs[0].get("name", ""))).name if objs else "unknown.csv"
            return http("POST", f"{API}/api/register-existing-storage", {
                "connection_id": cid, "storage_uri": f"{BUCKET_PREFIX}/{conn}/",
                "key": key}, self.auth(), timeout=300)
        return 0, {"skip": "unknown ingest op"}, 0

    def _wrap_dataset(self, spec):
        t0 = time.time()
        try:
            e = self.dataset_for(spec)
            return 200, e, time.time() - t0
        except RuntimeError as ex:
            return 500, {"error": str(ex)}, time.time() - t0

    # ---- main loop ----------------------------------------------------------
    def run_items(self, suite, items, log_path):
        self.results_f = open(log_path, "a")
        passed = failed = skipped = 0
        infra_streak = 0
        for it in items:
            if getattr(self, "skip_heavy", False) and is_heavy(it):
                print(f"{C['y']}SKIP-HEAVY » {it['id']} (known resource-bomb item){C['x']}")
                skipped += 1; self._log(log_path, it["id"], "SKIP-HEAVY", 0, 0, "excluded on bare-metal", {}); continue
            if not api_healthy():
                print(f"{C['y']}[infra] API down — attempting recovery...{C['x']}")
                if not recover_api():
                    print(f"{C['r']}[infra] recovery failed — aborting run (no fake FAILs){C['x']}"); break
                self.login()
            iid = it["id"]
            print(f"\n{C['b']}{C['c']}━━━ {iid} ━━━{C['x']}  {C['dim']}{it.get('dataset','')}{C['x']}")
            shown = it.get("prompt") or it.get("raw") or json.dumps({k: v for k, v in it.items() if k in ('op','conn','file')})
            print(f"{C['b']}PROMPT »{C['x']} {shown}")
            if it.get("expect"): print(f"{C['dim']}expect: {it['expect']}{C['x']}")
            kind = it.get("kind", "chat")
            if kind == "chat":
                try:
                    code, out, secs, meta = self.run_chat(it, suite)
                except RuntimeError as ex:
                    code, out, secs, meta = 500, {"error": str(ex)}, 0, {}
            elif kind == "api":
                code, out, secs = self.run_api(it); meta = {}
            elif kind == "ingest":
                code, out, secs = self.run_ingest(it); meta = {}
            else:
                print(f"{C['y']}SKIP » manual/k8s item — run by hand{C['x']}")
                skipped += 1
                self._log(log_path, iid, "SKIP", 0, 0, "manual", {})
                continue
            txt = out if isinstance(out, str) else json.dumps(out, default=str)
            ok = code == 200 and "skip" not in (out if isinstance(out, dict) else {})
            if isinstance(out, dict) and out.get("skip"):
                verdict, colour = "SKIP", C["y"]; skipped += 1
            elif ok:
                verdict, colour = "OK", C["g"]; passed += 1
            elif code == 0:
                verdict, colour = "INFRA", C["y"]; skipped += 1; infra_streak += 1
                if infra_streak >= 3:
                    print(f"{C['r']}[infra] 3 consecutive connection-level failures - aborting{C['x']}")
                    self._log(log_path, iid, verdict, code, secs, txt[:4000], meta)
                    break
            else:
                verdict, colour = "FAIL", C["r"]; failed += 1; infra_streak = 0
            print(f"{colour}{C['b']}OUTPUT [{verdict} http={code} {secs:.1f}s]{C['x']}"
                  + (f" {C['dim']}{meta}{C['x']}" if meta else ""))
            print(txt[:1600] + (f"{C['dim']} …[+{len(txt)-1600} chars]{C['x']}" if len(txt) > 1600 else ""))
            self._log(log_path, iid, verdict, code, secs, txt[:4000], meta)
        print(f"\n{C['b']}══ done: {C['g']}{passed} ok{C['x']}{C['b']} · "
              f"{C['r']}{failed} fail{C['x']}{C['b']} · {C['y']}{skipped} skip{C['x']}")

    def _log(self, path, iid, verdict, code, secs, out, meta):
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "id": iid, "verdict": verdict,
               "http": code, "secs": round(secs, 1), "output": out, "meta": meta}
        self.results_f.write(json.dumps(rec) + "\n"); self.results_f.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["v1", "v2", "v4", "heavy"])
    ap.add_argument("--batch"); ap.add_argument("--only"); ap.add_argument("--from", dest="frm")
    ap.add_argument("--skip-heavy", action="store_true", dest="skip_heavy")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    suite = json.loads((BASE / "suites" / f"{a.suite}_suite.json").read_text())
    if a.list:
        for b in suite["batches"]:
            print(f"{b['name']:26s} {b.get('title', b.get('description',''))[:60]:60s} {len(b['items'])} items")
        return
    batches = [b for b in suite["batches"] if not a.batch or b["name"].lower() == a.batch.lower()]
    if not batches: sys.exit(f"no batch named {a.batch}")
    items = []
    for b in batches:
        for i in b["items"]:
            i["_batch"] = b["name"]; items.append(i)
    if a.only: items = [i for i in items if i["id"] == a.only]
    if a.frm:
        idx = next((n for n, i in enumerate(items) if i["id"] == a.frm), 0)
        items = items[idx:]
    if not items: sys.exit("nothing to run")
    r = Runner(reset=a.reset); r.skip_heavy = a.skip_heavy; r.login()
    (BASE / "results").mkdir(exist_ok=True)
    log = BASE / "results" / f"{a.suite}_{datetime.now():%m%d_%H%M}.jsonl"
    print(f"{C['dim']}suite={a.suite} items={len(items)} log={log}{C['x']}")
    r.run_items(suite, items, log)

if __name__ == "__main__":
    main()
