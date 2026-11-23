#!/usr/bin/env python3
"""Drop infrastructure-failed trials from a trial journal so a resume re-runs them once.

Report policy (docs/benchmarks/2.0-vs-1.0.md §4): a provider or start-up error
(Groq's ``tool_use_failed`` 400, a dropped connection, a missing fixture) is not
a wrong answer by either side. Each such trial is retried exactly once by
removing its journal lines and re-running the same command with the same
``--out`` (the journal resumes everything else). What still fails after the
retry is reported as an infrastructure failure in its own column.

    python scripts/bench/retry_infra_failures.py <run>/dab.trials.jsonl

The original journal is kept beside it as ``*.before-retry.jsonl``; a second
invocation on the same journal refuses to drop anything that was already
retried once, so the policy cannot silently become "retry until it passes".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

INFRA_PREFIXES = ("provider_error", "sut_start_error", "provisioning_error")


def main(path_str: str) -> int:
    path = Path(path_str)
    backup = path.with_suffix(".before-retry.jsonl")
    already = set()
    if backup.is_file():
        for line in backup.read_text().splitlines():
            if line.strip():
                already.add(json.loads(line)["key"])
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    infra_keys = set()
    for ev in lines:
        err = str((ev.get("outcome") or {}).get("error") or "")
        if err.startswith(INFRA_PREFIXES):
            infra_keys.add(ev["key"])
    retry = {k for k in infra_keys if k not in already}
    kept = [ev for ev in lines if ev["key"] not in retry]
    if not backup.is_file():
        backup.write_text(path.read_text())
    else:
        with backup.open("a") as fh:
            for ev in lines:
                if ev["key"] in retry:
                    fh.write(json.dumps(ev) + "\n")
    path.write_text("".join(json.dumps(ev) + "\n" for ev in kept))
    tasks = sorted({ev["task_id"] for ev in lines if ev["key"] in retry})
    print(f"{path}: {len(infra_keys)} infrastructure-failed trial(s); retrying {len(retry)} "
          f"(already retried once: {len(infra_keys - retry)}): {tasks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
