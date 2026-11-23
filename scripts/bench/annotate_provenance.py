#!/usr/bin/env python3
"""Assert provenance on a record whose runner could not observe git.

    python scripts/bench/annotate_provenance.py <record.json> <commit> <branch> <dirty: true|false> [note]

Only fills fields that are null; never overwrites an observed value. Writes
``git_source = "asserted_by_operator"`` and the note, so a reader can tell an
asserted SHA from an observed one. Used for the two model-alone records
(gemini-4-argon, gemma4:e4b) that were already running inside a git-less
container when the fallback in ``benchmarks.core.reporting`` was added.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str]) -> int:
    path, commit, branch, dirty = Path(argv[1]), argv[2], argv[3], argv[4].lower() in ("1", "true", "yes")
    note = argv[5] if len(argv) > 5 else ""
    record = json.loads(path.read_text())
    meta = record.get("repository") if "repository" in record else record.setdefault("meta", {})
    if meta is None:
        meta = record["repository"] = {}
    if meta.get("git_commit"):
        print(f"{path}: observed commit {meta['git_commit']} kept; nothing asserted")
        return 0
    meta.update({"git_commit": commit, "git_branch": branch, "git_dirty": dirty,
                 "git_source": "asserted_by_operator",
                 "git_asserted_at": datetime.now(timezone.utc).isoformat(), "git_note": note})
    path.write_text(json.dumps(record, indent=2, default=str))
    print(f"{path}: asserted {commit} ({branch}, dirty={dirty})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
