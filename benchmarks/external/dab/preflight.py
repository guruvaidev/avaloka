"""Cheap structural checks before spending model budget on DAB."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .adapter import load_dab_tasks

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an official DAB checkout")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    tasks = load_dab_tasks()
    root = Path(os.environ["DAB_ROOT"]).resolve()
    checks = {"root": str(root), "queries": len(tasks),
              "postgres_client": bool(shutil.which("psql")),
              "mongo_shell": bool(shutil.which("mongosh")),
              "docker": bool(shutil.which("docker")),
              "agent_command": bool(os.environ.get("AVALOKA_DAB_AGENT_CMD")),
              "use_hints": os.environ.get("DAB_USE_HINTS", "0")}
    if args.json:
        print(json.dumps(checks, indent=2))
    else:
        for key, value in checks.items():
            print(f"{key}: {value}")
    return 0 if checks["queries"] == 54 else 2

if __name__ == "__main__":
    raise SystemExit(main())
