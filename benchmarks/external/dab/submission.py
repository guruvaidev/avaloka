"""Export an Avaloka benchmark record in official DAB leaderboard format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

def export(record_path: Path, output_path: Path, traces_dir: Path) -> int:
    record = json.loads(record_path.read_text())
    if record.get("suite") != "dab":
        raise ValueError(f"expected suite=dab, got {record.get('suite')!r}")
    rows: list[dict[str, Any]] = []
    traces_dir.mkdir(parents=True, exist_ok=True)
    for config_id, block in record.get("results", {}).items():
        for task_id, task in block.get("tasks", {}).items():
            prefix, dataset, query = task_id.split(":", 2)
            if prefix != "dab":
                continue
            for trial in task.get("trials", []):
                run = int(trial["trial_index"])
                outcome = trial["outcome"]
                answer = outcome.get("answer")
                if not isinstance(answer, str):
                    answer = json.dumps(answer, default=str)
                rows.append({"dataset": dataset, "query": query, "run": run,
                             "answer": answer})
                trace = {"dataset": dataset, "query": query, "run": run,
                         "config": config_id, "answer": answer,
                         "trajectory": outcome.get("trajectory", []),
                         "artifacts": outcome.get("artifacts", {}),
                         "error": outcome.get("error")}
                trace_path = traces_dir / f"{dataset}_query{query}_run{run}.json"
                trace_path.write_text(json.dumps(trace, indent=2, default=str) + "\n")
    rows.sort(key=lambda r: (r["dataset"], int(r["query"]), r["run"]))
    keys = {(r["dataset"], r["query"], r["run"]) for r in rows}
    if len(rows) != 270 or len(keys) != 270:
        raise ValueError(f"official submission requires 270 unique trials; found {len(keys)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2) + "\n")
    return len(rows)

def main() -> int:
    parser = argparse.ArgumentParser(description="Export official DAB submission + traces")
    parser.add_argument("record", type=Path, help="benchmarks.run DAB JSON record")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    args = parser.parse_args()
    count = export(args.record, args.output, args.traces)
    print(f"Wrote {count} DAB trials to {args.output}; traces in {args.traces}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
