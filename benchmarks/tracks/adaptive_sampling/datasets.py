"""Synthetic datasets for Track 1, with deterministic full-data reference answers.

Real Track 1 will use fingerprinted public/dev datasets plus a private held-out set
(design rule #1). These synthetic generators exist so the harness, metrics, and
scorecard are exercised end-to-end in CI with zero external data — and so the dataset
*conditions* from benchmarks.md §4 (imbalance, rare events, long tails, high
cardinality) are represented explicitly and reproducibly.

Everything is stdlib + seeded ``random`` so results are byte-stable across machines.
"""

from __future__ import annotations

import random
import statistics
from functools import lru_cache
from typing import Any, Dict, List

from ...core.task import DatasetSpec


def _make_transactions(n: int, fraud_rate: float, seed: int) -> List[Dict[str, Any]]:
    """A transactions table with a rare fraud class, long-tailed amount, and
    high-cardinality merchant / geographic region columns."""
    rng = random.Random(seed)
    regions = [f"region_{i}" for i in range(12)]
    region_weights = [0.28, 0.2, 0.14, 0.1, 0.08, 0.06, 0.05, 0.035, 0.02,
                      0.015, 0.01, 0.005]  # geographic subpopulation skew
    records: List[Dict[str, Any]] = []
    for i in range(n):
        is_fraud = rng.random() < fraud_rate
        # Long-tailed amount: lognormal-ish; fraud skews to larger amounts.
        base = rng.lognormvariate(3.2, 0.9)
        amount = base * (3.5 if is_fraud else 1.0)
        records.append({
            "txn_id": i,
            "amount": round(amount, 2),
            "region": rng.choices(regions, weights=region_weights, k=1)[0],
            "merchant_id": f"m_{rng.randint(0, 5000)}",   # high cardinality
            "is_fraud": is_fraud,
        })
    return records


@lru_cache(maxsize=None)
def _dataset(n: int, fraud_rate: float, seed: int) -> Dict[str, Any]:
    records = _make_transactions(n, fraud_rate, seed)
    amounts = [r["amount"] for r in records]
    fraud_ids = [r["txn_id"] for r in records if r["is_fraud"]]
    region_counts: Dict[str, int] = {}
    for r in records:
        region_counts[r["region"]] = region_counts.get(r["region"], 0) + 1
    reference = {
        "n": n,
        "mean_amount": statistics.fmean(amounts),
        "fraud_rate": len(fraud_ids) / n,
        "fraud_ids": fraud_ids,
        "region_distribution": {k: v / n for k, v in region_counts.items()},
    }
    return {"records": records, "reference": reference}


def imbalanced_transactions(n: int = 20_000, fraud_rate: float = 0.01,
                            seed: int = 7) -> DatasetSpec:
    """Highly imbalanced (~1% fraud), long-tailed amount, high-cardinality merchant."""
    return DatasetSpec(
        id=f"txns_imbalanced_n{n}_f{fraud_rate}",
        description="Synthetic transactions: rare fraud, long-tailed amount, "
                    "geographic subpopulation skew, high-cardinality merchant.",
        loader=lambda: _dataset(n, fraud_rate, seed),
        conditions=("imbalanced", "rare_event", "long_tailed", "high_cardinality",
                    "geographic_subpopulations"),
        fingerprint=f"synthetic:{n}:{fraud_rate}:{seed}",
    )


def reference_for(spec: DatasetSpec) -> Dict[str, Any]:
    """Convenience: full-data ground truth for a synthetic spec."""
    return spec.load()["reference"]
