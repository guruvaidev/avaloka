"""How long one lineage event takes to write, batched and not.

    AVALOKA_LINEAGE_DB_URL=postgresql://... python scripts/bench_lineage_writes.py [columns] [runs]

"Per-operation" is the write pattern the store had before it was moved to
Postgres: one transaction for every node and every edge. "Batched" is what
record_dataset does now. Both run on the same pooled engine, so the difference
is round trips and commits, not connection setup.

Writes to tenant ``bench`` and leaves the rows behind; point it at a scratch
database.
"""

from __future__ import annotations

import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.lineage import Edge, EdgeKind, LineageStore, Node, NodeKind


def per_operation(store: LineageStore, dataset_id: str, columns: list) -> None:
    store.add_node(Node(dataset_id, NodeKind.DATASET, "bench.csv"))
    for name, dtype in columns:
        cid = store.column_id(dataset_id, name)
        store.add_node(Node(cid, NodeKind.COLUMN, "", {"dtype": dtype}), plaintext_name=name)
        store.add_edge(Edge(dataset_id, EdgeKind.HAS_COLUMN, cid))


def batched(store: LineageStore, dataset_id: str, columns: list) -> None:
    store.record_dataset(dataset_id, label="bench.csv", columns=columns)


def main() -> int:
    n_columns = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    columns = [(f"column_{i}", "float64") for i in range(n_columns)]
    store = LineageStore("bench")
    if not store.available:
        print("no lineage database configured or reachable", file=sys.stderr)
        return 1
    store.column_id("warm", "up")   # mint the salt and open the pool outside the timing

    for label, write in (("per-operation", per_operation), ("batched", batched)):
        timings = []
        for _ in range(runs):
            dataset_id = f"ds:bench:{uuid.uuid4().hex}"
            started = time.perf_counter()
            write(store, dataset_id, columns)
            timings.append((time.perf_counter() - started) * 1000)
        timings.sort()
        print(f"{label:>14}: median {statistics.median(timings):8.1f} ms   "
              f"p95 {timings[int(len(timings) * 0.95) - 1]:8.1f} ms   "
              f"({n_columns} columns, {runs} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
