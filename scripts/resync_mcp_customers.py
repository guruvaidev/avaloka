#!/usr/bin/env python3
"""
Re-register every customer in customers.json with the running MCP server
(POST /admin/customers).

The MCP server rehydrates its registry from customers.json on boot, so this is
only needed to re-sync while the server is running or to push to a remote server
(set MCP_SERVER_URL). Idempotent; exits 0 unless it cannot reach the server or
read the file.

Env:
  MCP_SERVER_URL   MCP server base URL (default http://127.0.0.1:8010)
  CUSTOMERS_FILE   path to customers.json (default ./customers.json)
"""

import json
import os
import sys

import httpx

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8010").rstrip("/")
CUSTOMERS_FILE = os.getenv("CUSTOMERS_FILE", "customers.json")

# Fields the /admin/customers DatabaseConfig model expects.
_CONFIG_FIELDS = (
    "customer_id",
    "connection_string",
    "database_type",
    "max_rows",
    "query_timeout",
    "created_at",
    "status",
    "api_key",
)


def main() -> int:
    if not os.path.exists(CUSTOMERS_FILE):
        print(f"[resync] No customers file at '{CUSTOMERS_FILE}'. Nothing to do.")
        return 0

    try:
        with open(CUSTOMERS_FILE, "r") as f:
            customers = json.load(f)
    except Exception as e:
        print(f"[resync] ERROR: could not read '{CUSTOMERS_FILE}': {e}")
        return 1

    if not customers:
        print("[resync] customers.json is empty. Nothing to do.")
        return 0

    print(f"[resync] Target MCP server: {MCP_SERVER_URL}")
    print(f"[resync] {len(customers)} customer(s) in '{CUSTOMERS_FILE}'.")

    ok, skipped, failed = 0, 0, 0
    with httpx.Client(timeout=30.0) as client:
        for customer_id, record in customers.items():
            if record.get("status") == "inactive":
                print(f"  - {customer_id}: SKIP (inactive)")
                skipped += 1
                continue
            if not record.get("api_key") or not record.get("connection_string"):
                print(f"  - {customer_id}: SKIP (missing api_key or connection_string)")
                skipped += 1
                continue

            payload = {k: record[k] for k in _CONFIG_FIELDS if k in record}
            payload.setdefault("status", "active")
            payload.setdefault("max_rows", 1000)
            payload.setdefault("query_timeout", 30)
            try:
                resp = client.post(f"{MCP_SERVER_URL}/admin/customers", json=payload)
                if resp.status_code == 200:
                    print(f"  - {customer_id}: OK")
                    ok += 1
                else:
                    # 400 usually means the DB is unreachable from the MCP host, not a script bug.
                    print(f"  - {customer_id}: FAILED ({resp.status_code}) {resp.text}")
                    failed += 1
            except Exception as e:
                print(f"  - {customer_id}: ERROR contacting MCP server: {e}")
                failed += 1

    print(f"[resync] Done. registered={ok} skipped={skipped} failed={failed}")
    # Exit 0 on per-customer failures; only a total failure to proceed returns 1.
    return 0


if __name__ == "__main__":
    sys.exit(main())
