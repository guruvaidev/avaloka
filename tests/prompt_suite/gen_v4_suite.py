#!/usr/bin/env python3
"""Generate v4_suite.json from TEST_PLAN_v4.md (the authoritative local copy).

Chat prompts run via /threads/{id}/messages. Special IDs (API actions, ingest
steps, MTA-inference-tab rule, k8s actions) are tagged so the runner dispatches
them to the right executor instead of prompting.
"""
import json, re, sys
from pathlib import Path

PLAN = Path(__file__).resolve().parents[1] / "avaloka-test-results" / "TEST_PLAN_v4.md"
OUT = Path(__file__).resolve().parent / "suites" / "v4_suite.json"

# ---- per-ID overrides -------------------------------------------------------
# kind: chat | ingest | api | k8s | skip
# MTA RULE (from Mehebub): registered models are NOT used via prompt — the
# Inference tab is the door. IN-1/IN-2 therefore run as batch-inference API
# calls (the tab's backend), never as chat.
KIND = {
    "ING-1": ("ingest", {"op": "upload", "file": "housing.csv"}),
    "ING-2": ("ingest", {"op": "upload", "file": "train.csv"}),
    "ING-3": ("ingest", {"op": "register", "conn": "instacart-mba", "file": "aisles.csv"}),
    "ING-4": ("ingest", {"op": "register", "conn": "instacart-mba", "file": "orders.csv"}),
    "ING-5": ("ingest", {"op": "register", "conn": "nyc-taxi", "file": "yellow_tripdata_2015-01.csv"}),
    "ING-7": ("api", {"op": "buckets_list", "conn": "instacart-mba"}),
    "ING-8": ("ingest", {"op": "register_negative", "conn": "talkingdata-adtracking-fraud"}),
    "MX-6":  ("api", {"op": "models_get"}),
    "MX-8":  ("api", {"op": "model_delete_throwaway"}),
    "IN-1":  ("api", {"op": "batch_inference", "model_of": "MX-1", "rows": 20, "note": "MTA rule: inference-tab API, not prompt"}),
    "IN-2":  ("api", {"op": "batch_inference", "model_of": "MX-2", "rows": 20, "note": "MTA rule: inference-tab API, not prompt"}),
    "IN-3":  ("api", {"op": "batch_inference", "model_of": "MX-1", "rows": 5}),
    "IN-4":  ("api", {"op": "batch_inference_missing_col", "model_of": "MX-1"}),
    "IN-5":  ("api", {"op": "configure_inference_service", "model_of": "MX-1", "note": "MTA rule: serving via inference-tab API; expect clear Ray-off message"}),
    "IN-6":  ("api", {"op": "stop_inference_service"}),
    "SC-3":  ("api", {"op": "tasks_list"}),
    "FS-6":  ("k8s", {"op": "restart_api_then_reask", "thread": "housing-main"}),
    "ME-2":  ("chat", {"prompt_override": "Average median_house_value by ocean_proximity, highest first, with row counts."}),
}

# thread routing: which conversation each chat item joins ("fresh:<key>" starts one)
THREAD = {
    "GQ-1": "housing-main", "GQ-2": "housing-main", "TR-1": "housing-main", "TR-2": "housing-main",
    "AN-1": "housing-main", "AN-2": "housing-main", "AN-3": "housing-main", "VZ-1": "housing-main",
    "VZ-5": "housing-main", "MX-7": "housing-main", "PG-1": "housing-main", "PG-2": "housing-main",
    "ME-1": "housing-main", "ME-2": "housing-main", "ME-3": "housing-main", "ME-4": "housing-main",
    "GQ-3": "titanic-main", "TR-3": "titanic-main", "TR-4": "titanic-main", "TR-5": "titanic-main",
    "TR-6": "titanic-main", "AN-4": "titanic-main", "AN-5": "titanic-main", "VZ-2": "titanic-main",
    "GQ-4": "aisles-main", "FS-1": "aisles-main", "FS-2": "aisles-main", "DT-1": "aisles-main",
    "GQ-5": "departments-main", "DT-4": "departments-main", "DT-3": "departments-main",
    "DT-6": "departments-main", "DT-7": "sellprices-main",
    "GQ-6": "calendar-main", "TR-10": "calendar-main", "AN-15": "calendar-main",
    "GQ-7": "identity-main", "AN-10": "identity-main",
    "GQ-8": "housing-main", "GQ-9": "fraud-main", "AN-8": "fraud-main", "AN-9": "fraud-main",
    "FS-3": "fraud-main", "SC-1": "fraud-main",
    "AN-6": "orders-main", "AN-7": "orders-main", "VZ-4": "orders-main",
    "AN-11": "sellprices-main", "AN-12": "sellprices-main", "VZ-6": "sellprices-main", "FS-4": "sellprices-main",
    "TR-7": "products-main", "TR-11": "products-main",
    "TR-8": "taxi-main", "TR-9": "taxi-main", "AN-13": "taxi-main", "AN-14": "taxi-main",
    "VZ-3": "taxi-main", "VZ-7": "taxi-main", "FS-5": "taxi-main",
    "MP-1": "fresh:housing-mta", "MX-1": "housing-mta", "PG-3": "housing-mta",
    "MP-2": "fresh:titanic-mta", "MP-3": "titanic-mta", "MX-2": "titanic-mta",
    "MP-4": "fresh:titanic-mta2", "MP-5": "titanic-mta2",
    "MP-6": "fresh:housing-mta2", "MP-7": "fresh:housing-mta3", "MP-8": "fresh:titanic-mta3",
    "MP-9": "fresh:orders-mta", "MX-4": "orders-mta",
    "MX-3": "fresh:titanic-mta4", "MP-10": "fresh:sellprices-mta",
    "MX-5": "housing-mta", "IN-5_chat_disabled": "",
    "SC-2": "fraud-main", "SC-4": "fraud-main", "SC-5": "fraud-main", "ING-6": "housing-main",
    "ING-9": "housing-main",
}

# dataset each thread needs active (upload name or conn/file)
THREAD_DATASET = {
    "housing-main": "upload:housing.csv", "housing-mta": "upload:housing.csv",
    "housing-mta2": "upload:housing.csv", "housing-mta3": "upload:housing.csv",
    "titanic-main": "upload:train.csv", "titanic-mta": "upload:train.csv",
    "titanic-mta2": "upload:train.csv", "titanic-mta3": "upload:train.csv", "titanic-mta4": "upload:train.csv",
    "aisles-main": "register:instacart-mba/aisles.csv", "departments-main": "register:instacart-mba/departments.csv",
    "products-main": "register:instacart-mba/products.csv", "orders-main": "register:instacart-mba/orders.csv",
    "orders-mta": "register:instacart-mba/orders.csv", "calendar-main": "register:walmart/calendar.csv",
    "sellprices-main": "register:walmart/sell_prices.csv", "sellprices-mta": "register:walmart/sell_prices.csv",
    "fraud-main": "register:ieee-fraud/train_transaction.csv", "identity-main": "register:ieee-fraud/train_identity.csv",
    "taxi-main": "register:nyc-taxi/yellow_tripdata_2015-01.csv",
}

def main():
    text = PLAN.read_text()
    groups, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^## (F\d+) · (.+?) — ", line)
        if m:
            cur = {"name": m.group(1), "title": m.group(2).strip(), "items": []}
            groups.append(cur); continue
        if cur and line.startswith("|") and not line.startswith("|---") and not line.startswith("| ID"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 4 or not re.match(r"^[A-Z]{2,3}-\d+$", cells[0]):
                continue
            iid, praw, ds, expect = cells[0], cells[1], cells[2], cells[3]
            kind, extra = KIND.get(iid, (None, {}))
            if kind is None:
                kind, extra = "chat", {}
            item = {"id": iid, "kind": kind, "dataset": ds, "expect": expect, "raw": praw}
            if kind == "chat":
                pm = re.search(r'"(.+)"', praw)
                item["prompt"] = extra.get("prompt_override") or (pm.group(1) if pm else praw)
                item["thread"] = THREAD.get(iid, "fresh:" + iid.lower())
            else:
                item.update(extra)
            cur["items"].append(item)
    out = {"suite": "v4", "source": str(PLAN), "thread_datasets": THREAD_DATASET, "batches": groups}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    n = sum(len(g["items"]) for g in groups)
    print(f"wrote {OUT} — {len(groups)} groups, {n} items")
    assert n == 100, f"expected 100 items, got {n}"

if __name__ == "__main__":
    main()
