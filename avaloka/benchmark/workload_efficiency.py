"""Paired local-runtime benchmarks for statistics-driven scan optimization."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ScanMeasurement:
    mode: str
    wall_seconds: float
    cpu_seconds: float
    rows_returned: int
    columns_returned: int


@dataclass(frozen=True)
class EfficiencyResult:
    dataset: str
    source_bytes: int
    rows_total: int
    filter_column: str
    filter_value: Any
    projected_columns: list[str]
    baseline: ScanMeasurement
    optimized: ScanMeasurement
    wall_speedup: float
    cpu_speedup: float
    wall_reduction_pct: float
    cpu_reduction_pct: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _measure(callable_, repeats: int) -> tuple[float, float, pd.DataFrame]:
    wall, cpu, result = [], [], None
    callable_()  # warm file-system and Parquet metadata caches equally
    for _ in range(max(1, repeats)):
        wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
        result = callable_()
        cpu.append((time.process_time_ns() - cpu_start) / 1e9)
        wall.append((time.perf_counter_ns() - wall_start) / 1e9)
    return statistics.median(wall), statistics.median(cpu), result


def benchmark_parquet_scan(
    source: str | Path,
    *,
    filter_column: str,
    filter_value: Any,
    projected_columns: list[str],
    repeats: int = 5,
) -> EfficiencyResult:
    """Compare equivalent full-scan and predicate/projection-pushdown workloads.

    ``process_time_ns`` is the portable local-runtime CPU-cycle proxy. Hardware
    cycle counters are intentionally not used because they require privileged,
    platform-specific tooling and make CI results incomparable.
    """
    path = Path(source)

    def baseline() -> pd.DataFrame:
        frame = pd.read_parquet(path)
        return frame.loc[frame[filter_column] == filter_value, projected_columns]

    def optimized() -> pd.DataFrame:
        return pd.read_parquet(
            path,
            columns=projected_columns,
            filters=[(filter_column, "=", filter_value)],
        )

    base_wall, base_cpu, base_result = _measure(baseline, repeats)
    opt_wall, opt_cpu, opt_result = _measure(optimized, repeats)
    pd.testing.assert_frame_equal(
        base_result.reset_index(drop=True), opt_result.reset_index(drop=True),
        check_dtype=True,
    )
    wall_speedup = base_wall / opt_wall if opt_wall else float("inf")
    cpu_speedup = base_cpu / opt_cpu if opt_cpu else float("inf")
    return EfficiencyResult(
        dataset=path.name, source_bytes=path.stat().st_size,
        rows_total=int(pd.read_parquet(path, columns=[filter_column]).shape[0]),
        filter_column=filter_column, filter_value=filter_value,
        projected_columns=projected_columns,
        baseline=ScanMeasurement("full_scan_then_filter", base_wall, base_cpu,
                                 len(base_result), base_result.shape[1]),
        optimized=ScanMeasurement("predicate_projection_pushdown", opt_wall, opt_cpu,
                                  len(opt_result), opt_result.shape[1]),
        wall_speedup=wall_speedup, cpu_speedup=cpu_speedup,
        wall_reduction_pct=(1.0 - opt_wall / base_wall) * 100 if base_wall else 0.0,
        cpu_reduction_pct=(1.0 - opt_cpu / base_cpu) * 100 if base_cpu else 0.0,
    )


def choose_selective_filter(frame: pd.DataFrame) -> tuple[str, Any]:
    """Choose a deterministic, non-null categorical predicate for a dataset."""
    candidates = []
    for column in frame.columns:
        counts = frame[column].value_counts(dropna=True)
        if 2 <= len(counts) <= max(100, len(frame) // 10):
            candidates.append((int(counts.iloc[-1]), str(column), counts.index[-1]))
    if not candidates:
        raise ValueError("dataset has no suitable selective categorical predicate")
    _, column, value = min(candidates, key=lambda item: (item[0], item[1]))
    return column, value
