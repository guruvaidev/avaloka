"""Portable local CPU/runtime comparison for indexed workload optimization."""
from __future__ import annotations

import csv
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SQLiteEfficiencyResult:
    dataset: str
    rows_total: int
    rows_returned: int
    baseline_wall_seconds: float
    optimized_wall_seconds: float
    baseline_cpu_seconds: float
    optimized_cpu_seconds: float
    baseline_vm_steps: int
    optimized_vm_steps: int
    wall_speedup: float
    cpu_speedup: float
    vm_step_reduction_pct: float

    def as_dict(self):
        return asdict(self)


def load_csv_rows(source: str | Path):
    path = Path(source)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        reader = csv.reader(handle, dialect)
        columns = next(reader)
        rows = [row for row in reader if len(row) == len(columns)]
    return columns, rows


def choose_filter(columns, rows):
    candidates = []
    for index, column in enumerate(columns):
        counts = {}
        for row in rows:
            counts[row[index]] = counts.get(row[index], 0) + 1
        if 2 <= len(counts) <= max(100, len(rows) // 10):
            value, count = min(counts.items(), key=lambda item: (item[1], item[0]))
            candidates.append((count, column, value))
    if not candidates:
        raise ValueError("dataset has no suitable selective column")
    _, column, value = min(candidates)
    return column, value


def benchmark_sqlite_rows(dataset, columns, rows, filter_column, filter_value,
                          projected_columns, repeats=5, minimum_rows=100_000):
    """Compare full scan/Python filter with an indexed projected SQL query.

    SQLite VM steps are a deterministic CPU-work proxy alongside process CPU
    time. They avoid privileged, hardware-specific performance counters.
    """
    if rows and len(rows) < minimum_rows:
        rows = (rows * (minimum_rows // len(rows) + 1))[:minimum_rows]
    connection = sqlite3.connect(":memory:")
    quoted = lambda value: '"' + value.replace('"', '""') + '"'
    connection.execute("CREATE TABLE data(" + ",".join(f"{quoted(c)} TEXT" for c in columns) + ")")
    connection.executemany("INSERT INTO data VALUES(" + ",".join("?" for _ in columns) + ")", rows)
    connection.execute(f"CREATE INDEX workload_filter_idx ON data({quoted(filter_column)})")
    connection.commit()
    filter_index = columns.index(filter_column)
    projection_indexes = [columns.index(c) for c in projected_columns]

    def baseline():
        return [tuple(row[i] for i in projection_indexes) for row in connection.execute("SELECT * FROM data")
                if row[filter_index] == filter_value]

    select = ",".join(quoted(c) for c in projected_columns)
    def optimized():
        return list(connection.execute(
            f"SELECT {select} FROM data WHERE {quoted(filter_column)} = ?", (filter_value,)))

    def measure(callable_):
        walls, cpus, steps, result = [], [], [], None
        callable_()
        for _ in range(repeats):
            count = 0
            def progress():
                nonlocal count
                count += 1
                return 0
            connection.set_progress_handler(progress, 1)
            wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
            result = callable_()
            cpus.append((time.process_time_ns() - cpu_start) / 1e9)
            walls.append((time.perf_counter_ns() - wall_start) / 1e9)
            steps.append(count)
        connection.set_progress_handler(None, 0)
        return statistics.median(walls), statistics.median(cpus), int(statistics.median(steps)), result

    base_wall, base_cpu, base_steps, base_result = measure(baseline)
    opt_wall, opt_cpu, opt_steps, opt_result = measure(optimized)
    connection.close()
    assert base_result == opt_result
    return SQLiteEfficiencyResult(
        dataset=dataset, rows_total=len(rows), rows_returned=len(base_result),
        baseline_wall_seconds=base_wall, optimized_wall_seconds=opt_wall,
        baseline_cpu_seconds=base_cpu, optimized_cpu_seconds=opt_cpu,
        baseline_vm_steps=base_steps, optimized_vm_steps=opt_steps,
        wall_speedup=base_wall / opt_wall, cpu_speedup=base_cpu / opt_cpu,
        vm_step_reduction_pct=(1 - opt_steps / base_steps) * 100,
    )
