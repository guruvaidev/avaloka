"""
Standalone Test – Portfolio Sampler

Runs sample_with_profiling against a local or cloud file and writes timing,
a sample CSV, a profiling JSON, portfolio CSVs, and a full run log to the
output directory.

Usage:
    # Native Daft (no Ray)
    python tests/test_new_sampler_standalone.py --csv-path path/to/file.csv --no-ray

    # RayRunner (start Ray first: ray start --head --port=8502)
    python tests/test_new_sampler_standalone.py --csv-path gs://bucket/file.csv

    # Remote Ray cluster
    python tests/test_new_sampler_standalone.py \\
        --csv-path gs://bucket/file.csv \\
        --ray-address ray://<VM_IP>:10001

Output per run:
    sampler_test_output/
      run_20260224_120000/
        run_log.txt          ← full timestamped console output
        sample_output.csv    ← display sample rows
        profiling_results.json
        portfolio_samples/   ← individual stratified CSV files
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.sampling_agent_daft import (
    sample_with_profiling,
    get_ray_cluster_info,
    init_ray_cluster,
    _is_daft_on_ray_runner,
    RAY_CPUS,
    RAY_MEMORY_GB,
    RAY_ADDRESS,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


class TeeLogger:
    """Writes print() output to both the console and a log file at the same time."""

    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._fh = open(log_path, "w", buffering=1)
        self._console = sys.stdout

    def write(self, msg: str):
        ts_msg = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}" if msg.strip() else msg
        self._console.write(ts_msg)
        self._fh.write(ts_msg)

    def flush(self):
        self._console.flush()
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._console
        self._fh.close()


# ── Main test function ────────────────────────────────────────────────────────

def test_new_sampler(
    csv_path: str,
    output_dir: str = "sampler_test_output",
    use_ray: bool = False,
) -> dict | None:
    """Run the portfolio sampler and write a detailed run report to output_dir."""
    W = 80

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    log_path = run_dir / "run_log.txt"

    with TeeLogger(log_path) as log:
        print(f"\n{'='*W}")
        print(f"  PORTFOLIO SAMPLER TEST")
        print(f"{'='*W}")
        print(f"  File    : {csv_path}")
        print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Log     : {log_path}")

        # Machine resource info
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"\n  ── Machine resources ──")
        print(f"  CPUs visible to Python : {cpu_count}")
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"  RAM total / available  : {mem.total / 1e9:.1f} GB / {mem.available / 1e9:.1f} GB")
        except ImportError:
            pass
        print(f"  AVALOKA_RAY_CPUS env   : {os.getenv('AVALOKA_RAY_CPUS', f'(not set — will use all {cpu_count})')}")
        print(f"  AVALOKA_RAY_MEMORY_GB  : {os.getenv('AVALOKA_RAY_MEMORY_GB', '(not set — auto)')}")

        # Executor banner: which Daft runner is active for this run
        if use_ray:
            print(f"\n  ── Executor: Daft RayRunner ✅ ──")
        else:
            print(f"\n  ── Executor: Daft NativeRunner (--no-ray) ──")

        is_cloud_path = any(csv_path.startswith(p) for p in ["gs://", "s3://", "http://", "https://"])

        if is_cloud_path:
            print(f"\n  ☁️  Cloud file — size determined during sampling")
            file_size_mb = 0.0
        else:
            file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
            print(f"\n  📁 File size: {file_size_mb:.2f} MB")
            print(f"  📊 Counting rows...", end="", flush=True)
            try:
                with open(csv_path, "rb") as f:
                    num_rows = sum(1 for _ in f) - 1
                print(f" {num_rows:,} rows")
            except Exception as e:
                print(f" failed: {e}")

        print(f"\n{'─'*W}")
        print(f"  RUNNING SAMPLER")
        print(f"{'─'*W}\n")

        start_time = time.time()

        try:
            result = sample_with_profiling(
                path=csv_path,
                source_type="csv",
                sample_size=1000,
                use_ray=use_ray,
            )

            elapsed = time.time() - start_time

            if result.get("error"):
                print(f"\n❌ FAILED: {result['error']}")
                print(f"⏱️  Time elapsed: {fmt_time(elapsed)}")
                return None

            sample_rows = result.get("rows", [])
            schema = result.get("schema", {})
            profiling = result.get("profiling_result", {})
            portfolio_meta = profiling.get("portfolio_metadata", {})
            data_quality = profiling.get("data_quality", {})

            print(f"\n✅ SUCCESS!")
            print(f"⏱️  Total time: {fmt_time(elapsed)}")
            print(f"\n{'─'*W}")
            print(f"  RESULTS")
            print(f"{'─'*W}")
            print(f"  Sample size: {len(sample_rows)} rows")
            print(f"  Columns: {len(schema)}")
            print(f"  Schema: {', '.join(list(schema)[:5])}{'...' if len(schema) > 5 else ''}")
            print(f"\n  Portfolio samples: {portfolio_meta.get('num_samples', 0)}")
            print(f"  Sizing tier: {portfolio_meta.get('sizing_tier', 'N/A')}")
            print(f"  Portfolio sample size: {portfolio_meta.get('portfolio_sample_size', 'N/A'):,}")
            print(f"  Display sample size: {portfolio_meta.get('display_sample_size', 'N/A'):,}")
            for sample_name in portfolio_meta.get("sample_names", []):
                print(f"    - {sample_name}")

            # Ray cluster info
            print(f"\n{'─'*W}")
            print(f"  RAY CLUSTER INFO")
            print(f"{'─'*W}")
            ray_was_used = portfolio_meta.get("ray_used", False)
            ray_info_from_result = portfolio_meta.get("ray_info", {})

            if ray_was_used:
                print(f"  🟢 Ray was used for distributed execution")
                ray_info = ray_info_from_result or get_ray_cluster_info()
                if ray_info.get("initialized"):
                    print(f"  CPUs  : {ray_info.get('total_cpus', 'N/A')} total, {ray_info.get('available_cpus', 'N/A')} available")
                    print(f"  Memory: {ray_info.get('total_memory_gb', 0):.2f} GB total, {ray_info.get('available_memory_gb', 0):.2f} GB available")
                    print(f"  Nodes : {ray_info.get('num_nodes', 0)}")
            else:
                print(f"  🔴 Ray was not used (Daft native executor)")

            # Data quality summary
            total_rows = data_quality.get("total_rows", "N/A")
            total_cols = data_quality.get("total_columns", "N/A")
            rows_str = f"{total_rows:,}" if isinstance(total_rows, (int, float)) else str(total_rows)
            print(f"\n  Data shape   : {rows_str} rows × {total_cols} columns")
            completeness = data_quality.get("overall_completeness", 0)
            if isinstance(completeness, (int, float)):
                print(f"  Completeness : {completeness * 100:.2f}%")
            high_missing = data_quality.get("columns_with_high_missing", [])
            if high_missing:
                print(f"  ⚠️  Columns with >30% missing: {', '.join(high_missing)}")

            # Save outputs
            results_file = run_dir / "profiling_results.json"
            with open(results_file, "w") as f:
                json.dump({
                    "metadata": {
                        "csv_path": str(csv_path),
                        "file_size_mb": file_size_mb,
                        "elapsed_seconds": elapsed,
                        "use_ray": use_ray,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "profiling_result": profiling,
                }, f, indent=2)

            portfolio_samples = result.get("portfolio_samples", {})
            portfolio_dir = None
            if portfolio_samples:
                portfolio_dir = run_dir / "portfolio_samples"
                portfolio_dir.mkdir(exist_ok=True)
                print(f"\n  💾 Saving {len(portfolio_samples)} portfolio samples to {portfolio_dir}...")
                for name, rows in portfolio_samples.items():
                    if rows:
                        pdf = pd.DataFrame(rows)
                        out_file = portfolio_dir / f"{name}.csv"
                        pdf.to_csv(out_file, index=False)
                        print(f"    - {name}: {len(pdf):,} rows → {out_file.name}")

            print(f"\n{'─'*W}")
            print(f"  OUTPUT FILES")
            print(f"{'─'*W}")
            print(f"  Log            : {log_path}")
            print(f"  Profiling JSON : {results_file}")
            if portfolio_dir:
                print(f"  Portfolio      : {portfolio_dir} ({len(portfolio_samples)} files)")
            print(f"\n{'='*W}\n")

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ EXCEPTION: {e}")
            print(f"⏱️  Time elapsed: {fmt_time(elapsed)}")
            import traceback
            traceback.print_exc()
            return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Run the portfolio sampler and print a detailed report")
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV file (local or cloud URI: gs://, s3://)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sampler_test_output",
        help="Directory to write run output to",
    )
    parser.add_argument(
        "--no-ray",
        action="store_true",
        help="Disable Ray — use Daft native executor only",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Remote Ray cluster address, e.g. ray://<IP>:10001",
    )

    args = parser.parse_args()

    is_cloud_path = any(args.csv_path.startswith(p) for p in ["gs://", "s3://", "http://", "https://"])
    if not is_cloud_path and not os.path.exists(args.csv_path):
        print(f"❌ Error: File not found: {args.csv_path}")
        sys.exit(1)

    if args.ray_address:
        # Write the address into the environment so init_ray_cluster picks it up
        os.environ["AVALOKA_RAY_ADDRESS"] = args.ray_address

    # Pre-initialize Daft runner ONCE before test_new_sampler runs.
    # Daft only allows setting its runner once per process, so this must happen
    # before any Daft operations, including inside test_new_sampler.
    if args.no_ray:
        use_ray = False
    else:
        ray_ok = init_ray_cluster()
        use_ray = ray_ok and _is_daft_on_ray_runner()
        if not use_ray:
            print("⚠️  Ray unavailable — falling back to Daft NativeRunner")

    result = test_new_sampler(args.csv_path, args.output_dir, use_ray=use_ray)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
