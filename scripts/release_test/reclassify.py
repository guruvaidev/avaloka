"""Re-score the CLI insights in a results.jsonl with the current classifier.

    python scripts/release_test/reclassify.py results/cli/results.jsonl > results/cli/results.reclassified.jsonl

Only CLI records are re-scored (their claims recompute from the data); API
records are copied through unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import classify_cli, classify_llm  # noqa: E402
from run_datasets import load_frame  # noqa: E402


def main(path: str, corpus: str) -> None:
    by_name = {d["name"]: d for d in json.loads(Path(corpus).read_text())}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        obs = [l for l in str(rec.get("insight", "")).splitlines() if l.strip()]
        if rec.get("path") == "api" and rec.get("insight") and rec["dataset"] in by_name and rec.get("checked"):
            try:
                frame = load_frame(by_name[rec["dataset"]]["path"])
                rec["insight_class"], rec["insight_reason"] = classify_llm(rec["insight"], list(frame.columns), frame)
                rec["reclassified"] = True
            except Exception as exc:
                rec["insight_reason"] += f" (reclassify: loader failed {exc!r})"[:200]
        elif rec.get("path") == "cli" and obs and rec["dataset"] in by_name:
            try:
                frame = load_frame(by_name[rec["dataset"]]["path"])
                rec["insight_class"], rec["insight_reason"] = classify_cli(obs, frame)
                rec["reclassified"] = True
            except Exception as exc:
                rec["insight_class"], rec["insight_reason"] = "unchecked", f"loader failed on recheck: {exc!r}"[:300]
        print(json.dumps(rec, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
