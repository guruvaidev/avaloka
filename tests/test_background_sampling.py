"""
Standalone Test – Background Sampling

Simulates what the server does when a file is uploaded:
  1. sample_quick() returns a quick preview immediately.
  2. sample_with_profiling() runs in a background thread for large datasets.

Usage:
    python tests/test_background_sampling.py --path data/myfile.csv
    python tests/test_background_sampling.py --path gs://my-bucket/data/events.csv
    python tests/test_background_sampling.py --path gs://bucket/events.csv --no-ray
    # Connect to a remote Ray cluster on GCP (start with: ray start --head --port=8502)
    python tests/test_background_sampling.py \\
        --path gs://my-bucket/data/events.csv \\
        --ray-address ray://<VM_IP>:10001
    # Force the background job even for a small file
    python tests/test_background_sampling.py --path data/small.csv --force-background

Output per run:
    background_sampling_output/
      run_20260220_003215_myfile/
        run_log.txt        ← timestamped console output
        run_summary.json   ← structured summary of both jobs
        quick_sample.csv   ← rows from the quick preview
        full_sample.csv    ← rows from the full background profile
        portfolio_samples/ ← individual stratified portfolio CSVs
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.sampling_async import sample_quick
from app.agents.sampling_agent_daft import (
    sample_with_profiling,
    init_ray_cluster,
    _is_daft_on_ray_runner,
)

import pandas as pd


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def fmt_rows(n: Optional[int]) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def _is_cloud(path: str) -> bool:
    return any(path.startswith(p) for p in ("gs://", "s3://", "az://", "http://", "https://"))


def _source_type(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".") or "csv"


def _local_file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception:
        return 0.0


def _save_sample_csv(rows: list, dest: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(dest, index=False)


def _save_portfolio(portfolio_samples: dict, dest_dir: Path) -> None:
    """Write each portfolio sample to its own CSV file."""
    if not portfolio_samples:
        return
    dest_dir.mkdir(exist_ok=True)
    for name, rows in portfolio_samples.items():
        if rows:
            pd.DataFrame(rows).to_csv(dest_dir / f"{name}.csv", index=False)


class TeeLogger:
    """Writes print() output to both the console and a log file simultaneously."""

    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._fh = open(log_path, "w", buffering=1)
        self._console = sys.stdout
        self._lines: list[str] = []

    def write(self, msg: str):
        ts_msg = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}" if msg.strip() else msg
        self._console.write(ts_msg)
        self._fh.write(ts_msg)
        if msg.strip():
            self._lines.append(msg.rstrip())

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

def run_background_sampling_test(
    path: str,
    output_dir: str = "background_sampling_output",
    force_background: bool = False,
    no_ray: bool = False,
    ray_address: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full two-speed sampling flow and write results to a timestamped folder.

    Returns the run summary dict (also written as run_summary.json).
    """
    if ray_address:
        # Write the address into the environment so init_ray_cluster() picks it up
        os.environ["AVALOKA_RAY_ADDRESS"] = ray_address
        print(f"  ☁️  RAY_ADDRESS set → {ray_address}")

    source_type = _source_type(path)
    is_cloud = _is_cloud(path)
    file_name = Path(path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_{Path(file_name).stem[:30]}"

    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run_log.txt"
    summary: Dict[str, Any] = {
        "run_id":       run_name,
        "path":         path,
        "is_cloud":     is_cloud,
        "source_type":  source_type,
        "uploaded_at":  datetime.now(timezone.utc).isoformat(),
        "quick":        {},
        "full":         {},
        "timeline":     [],
    }

    with TeeLogger(log_path) as log:

        W = 80
        def hdr(title: str):
            print(f"\n{'═'*W}")
            print(f"  {title}")
            print(f"{'═'*W}")

        def rule(title: str = ""):
            print(f"\n{'─'*W}" + (f"\n  {title}" if title else ""))

        def event(tag: str, msg: str, ts: float):
            summary["timeline"].append({
                "tag": tag,
                "message": msg,
                "elapsed_s": round(ts, 3),
                "wall_clock": datetime.now(timezone.utc).isoformat(),
            })
            print(f"  ⏱  +{fmt_time(ts):>8}  │  {msg}")

        hdr("BACKGROUND SAMPLING TEST")
        print(f"  File       : {path}")
        print(f"  Cloud      : {'Yes' if is_cloud else 'No'}")
        if not is_cloud:
            fsz = _local_file_size_mb(path)
            print(f"  File size  : {fsz:.2f} MB")
            summary["file_size_mb"] = fsz
        print(f"  Output dir : {run_dir}")
        print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Show what hardware we're running on — useful for diagnosing performance on the GCP VM
        import multiprocessing, os as _os
        cpu_count = multiprocessing.cpu_count()
        print(f"\n  ── Machine resources ──")
        print(f"  CPUs visible to Python : {cpu_count}")
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"  RAM total / available  : {mem.total / 1e9:.1f} GB / {mem.available / 1e9:.1f} GB")
        except ImportError:
            pass
        print(f"  AVALOKA_RAY_CPUS env   : {_os.getenv('AVALOKA_RAY_CPUS', f'(not set — will use all {cpu_count})')}")
        print(f"  AVALOKA_RAY_MEMORY_GB  : {_os.getenv('AVALOKA_RAY_MEMORY_GB', '(not set — auto)')}")

        # Decide on executor and pre-initialize Ray ONCE before any Daft work.
        # Daft only allows setting its runner once per process, so we must do this
        # before quick preview runs — otherwise quick preview may lock the runner
        # to NativeRunner and the background job inherits that silently.
        _use_ray: bool
        if no_ray:
            _use_ray = False
            print(f"\n  ── Executor: Daft NativeRunner (--no-ray) ──")
        else:
            ray_ok = init_ray_cluster()
            if ray_ok and _is_daft_on_ray_runner():
                _use_ray = True
                print(f"\n  ── Executor: Daft RayRunner ✅ ──")
            else:
                _use_ray = False
                print(f"\n  ── Executor: Daft NativeRunner (Ray unavailable) ──")

        global_start = time.perf_counter()

        # Quick preview
        rule("Quick Preview (sample_quick)")
        print("  Running the quick preview — this is what the user sees immediately after upload.\n")

        phase_a_start = time.perf_counter()
        phase_a_result: Dict[str, Any] = {}
        phase_a_error: Optional[str] = None

        try:
            phase_a_result = sample_quick(
                path=path,
                source_type=source_type,
                use_ray=_use_ray,
            )
            phase_a_elapsed = time.perf_counter() - phase_a_start
        except Exception as exc:
            phase_a_elapsed = time.perf_counter() - phase_a_start
            phase_a_error = str(exc)
            import traceback; traceback.print_exc()

        if phase_a_error:
            print(f"\n  ❌ Quick preview FAILED: {phase_a_error}")
        else:
            tier         = phase_a_result.get("tier", "?")
            est_rows     = phase_a_result.get("estimated_rows")
            profile_stat = phase_a_result.get("profile_status", "?")
            needs_bg     = phase_a_result.get("should_run_background", False)
            accuracy     = phase_a_result.get("accuracy", "?")
            sample_pct   = phase_a_result.get("sample_pct")
            a_rows       = phase_a_result.get("rows") or []
            _a_schema_raw = phase_a_result.get("schema") or []
            # schema can be dict {col: type} or list — normalise to list
            a_schema = list(_a_schema_raw.keys()) if isinstance(_a_schema_raw, dict) else list(_a_schema_raw)

            print(f"\n  ✅ Quick preview done in {fmt_time(phase_a_elapsed)}")
            print(f"\n  Dataset profile:")
            print(f"    Tier              : {tier}")
            print(f"    Estimated rows    : {fmt_rows(est_rows)}")
            print(f"    Profile status    : {profile_stat}")
            print(f"    Accuracy estimate : {accuracy}")
            print(f"    Sample %          : {f'{sample_pct:.2f}%' if isinstance(sample_pct, float) else 'N/A'}")
            print(f"    Background needed : {'YES' if needs_bg else 'NO — full profile already done'}")
            print(f"\n  Sample returned to UI:")
            print(f"    Rows              : {len(a_rows)}")
            print(f"    Columns           : {len(a_schema)}")
            _schema_preview = ', '.join(str(c) for c in a_schema[:6])
            print(f"    Schema preview    : {_schema_preview}{'...' if len(a_schema) > 6 else ''}")

            event("quick_complete",
                  f"Quick preview done – tier={tier} est_rows={fmt_rows(est_rows)} sample_rows={len(a_rows)}",
                  time.perf_counter() - global_start)

            summary["quick"] = {
                "elapsed_s":        round(phase_a_elapsed, 3),
                "tier":             tier,
                "estimated_rows":   est_rows,
                "profile_status":   profile_stat,
                "accuracy":         accuracy,
                "sample_pct":       sample_pct,
                "rows_returned":    len(a_rows),
                "columns":          len(a_schema),
                "needs_background": needs_bg,
            }

            _save_sample_csv(a_rows, run_dir / "quick_sample.csv")
            print(f"\n  💾 Quick sample saved → {run_dir / 'quick_sample.csv'}")

        run_bg = (force_background or phase_a_result.get("should_run_background", False))

        if not run_bg:
            rule("Background Job – Skipped")
            print(f"  Dataset is small enough that the full profile already ran above.")
            print(f"  Use --force-background to run the background job anyway.")
            summary["full"]["skipped"] = True
        else:
            rule("Background Job – Full Portfolio")
            print("  This mirrors what the server does in a FastAPI BackgroundTask after upload.\n")

            phase_b_result: Dict[str, Any] = {}
            phase_b_error: Optional[str] = None
            phase_b_elapsed = 0.0
            phase_b_done = threading.Event()

            def _phase_b_worker():
                nonlocal phase_b_result, phase_b_error, phase_b_elapsed
                try:
                    t0 = time.perf_counter()
                    phase_b_result = sample_with_profiling(
                        path=path,
                        source_type=source_type,
                        use_ray=False if no_ray else (True if ray_address else None),
                    )
                    phase_b_elapsed = time.perf_counter() - t0
                except Exception as exc:
                    phase_b_elapsed = time.perf_counter() - (phase_b_start_wall)
                    phase_b_error = str(exc)
                    import traceback; traceback.print_exc()
                finally:
                    phase_b_done.set()

            phase_b_launch_offset = time.perf_counter() - global_start
            phase_b_start_wall = time.perf_counter()
            bg_thread = threading.Thread(target=_phase_b_worker, daemon=True)
            bg_thread.start()

            print(f"  🚀 Background thread started at +{fmt_time(phase_b_launch_offset)} after upload")
            event("phase_b_started",
                  "Phase B background thread launched",
                  phase_b_launch_offset)

            # Poll progress every 5 s
            poll_interval = 5.0
            spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
            spin_i = 0
            while not phase_b_done.wait(timeout=poll_interval):
                elapsed_so_far = time.perf_counter() - phase_b_start_wall
                sys.stdout._console.write(
                    f"\r  {spinner[spin_i % len(spinner)]}  Phase B running… "
                    f"+{fmt_time(elapsed_so_far)} elapsed   "
                )
                sys.stdout._console.flush()
                spin_i += 1

            sys.stdout._console.write("\r" + " " * 70 + "\r")
            sys.stdout._console.flush()

            bg_thread.join()
            total_elapsed = time.perf_counter() - global_start

            if phase_b_error:
                print(f"\n  ❌ Background job FAILED: {phase_b_error}")
                summary["full"]["error"] = phase_b_error
            else:
                pr       = phase_b_result.get("profiling_result", {})
                p_meta   = pr.get("portfolio_metadata", {})
                dq       = pr.get("data_quality", {})
                b_rows   = phase_b_result.get("rows") or []
                _b_schema_raw = phase_b_result.get("schema") or []
                b_schema = list(_b_schema_raw.keys()) if isinstance(_b_schema_raw, dict) else list(_b_schema_raw)
                exact_rows = dq.get("total_rows") or p_meta.get("total_rows")

                print(f"\n  ✅ Background job complete in {fmt_time(phase_b_elapsed)}")
                print(f"\n  Full portfolio results:")
                print(f"    Tier              : {p_meta.get('sizing_tier', 'N/A')}")
                print(f"    Exact row count   : {fmt_rows(exact_rows)}")
                print(f"    Portfolio samples : {p_meta.get('num_samples', 0)}")
                print(f"    Portfolio size    : {fmt_rows(p_meta.get('portfolio_sample_size'))}")
                print(f"    Display rows      : {len(b_rows)}")
                print(f"    Columns           : {len(b_schema)}")
                print(f"    Completeness      : {dq.get('overall_completeness', 0)*100:.2f}%")
                print(f"    Ray used          : {'Yes' if p_meta.get('ray_used') else 'No'}")

                for sname in p_meta.get("sample_names", []):
                    print(f"      - {sname}")

                high_missing = dq.get("columns_with_high_missing", [])
                if high_missing:
                    print(f"\n  ⚠️  Columns with >30% missing: {', '.join(high_missing)}")

                event("full_complete",
                      f"Full profile done – exact_rows={fmt_rows(exact_rows)} "
                      f"portfolio_samples={p_meta.get('num_samples', 0)} "
                      f"display_rows={len(b_rows)}",
                      total_elapsed)

                summary["full"] = {
                    "launch_offset_s":       round(phase_b_launch_offset, 3),
                    "elapsed_s":             round(phase_b_elapsed, 3),
                    "total_elapsed_s":       round(total_elapsed, 3),
                    "tier":                  p_meta.get("sizing_tier"),
                    "exact_rows":            exact_rows,
                    "num_portfolio_samples": p_meta.get("num_samples"),
                    "portfolio_sample_size": p_meta.get("portfolio_sample_size"),
                    "display_rows":          len(b_rows),
                    "columns":               len(b_schema),
                    "completeness_pct":      dq.get("overall_completeness", 0) * 100,
                    "ray_used":              p_meta.get("ray_used", False),
                    "sample_names":          p_meta.get("sample_names", []),
                }

                port_dir = run_dir / "portfolio_samples"
                _save_portfolio(phase_b_result.get("portfolio_samples", {}), port_dir)
                print(f"\n  💾 Portfolio samples    → {port_dir}/")
                if port_dir.exists():
                    for pf in sorted(port_dir.glob("*.csv")):
                        try:
                            nrows = sum(1 for _ in open(pf)) - 1  # subtract header
                            print(f"       📄 {pf.name:<45} {nrows:>7,} rows")
                        except Exception:
                            print(f"       📄 {pf.name}")

        hdr("TIMELINE SUMMARY")
        total = time.perf_counter() - global_start
        pa    = summary["quick"]
        pb    = summary["full"]

        print(f"  Started          : {summary['uploaded_at']}")
        print(f"  Estimated rows   : {fmt_rows(pa.get('estimated_rows'))}")
        if not is_cloud:
            print(f"  File size        : {summary.get('file_size_mb', 0):.2f} MB")
        print(f"  Tier             : {pa.get('tier', 'N/A')}")
        print(f"  Initial sample   : {fmt_rows(pa.get('rows_returned'))} rows")
        print()
        print(f"  ⚡ Quick preview : +0.00s → +{fmt_time(pa.get('elapsed_s', 0))}")
        if not pb.get("skipped"):
            print(f"  🔄 Background started : +{fmt_time(pb.get('launch_offset_s', 0))}")
            print(f"  ✅ Background done    : +{fmt_time(pb.get('total_elapsed_s', 0))}")
            print(f"  ⏱  Background took    : {fmt_time(pb.get('elapsed_s', 0))}")
            exact = pb.get("exact_rows")
            if exact and pa.get("estimated_rows"):
                accuracy_pct = abs(exact - pa["estimated_rows"]) / max(exact, 1) * 100
                print(f"  📐 Estimate accuracy  : within {accuracy_pct:.1f}% of actual ({fmt_rows(exact)} rows)")
        print(f"\n  🏁 Total wall time : {fmt_time(total)}")
        print(f"\n  Output folder  : {run_dir}")
        print(f"  Log file       : {log_path}")

        summary["total_elapsed_s"] = round(total, 3)
        summary_path = run_dir / "run_summary.json"
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"  Summary JSON   : {summary_path}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run the two-speed sampling flow and write results to a timestamped folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_background_sampling.py --path data/events.csv
  python tests/test_background_sampling.py --path gs://bucket/events.csv --no-ray
  # Connect to a remote Ray cluster on GCP (start with: ray start --head --port=8502 on the VM)
  python tests/test_background_sampling.py \\
      --path gs://avaloka-test-user-filestore/2019-Oct.csv \\
      --ray-address ray://<VM_EXTERNAL_IP>:10001
""",
    )
    parser.add_argument(
        "--path", "-p",
        required=True,
        help="Local file path OR cloud URI (gs://, s3://)",
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        metavar="ADDRESS",
        help="Remote Ray cluster address, e.g. ray://<VM_IP>:10001. Forces use_ray=True and overrides --no-ray.",
    )
    parser.add_argument(
        "--no-ray",
        action="store_true",
        help="Use Daft native executor only (no Ray). Ignored if --ray-address is given.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="background_sampling_output",
        help="Root directory for timestamped run folders (default: background_sampling_output)",
    )
    parser.add_argument(
        "--force-background",
        action="store_true",
        help="Always run the background full profile, even for small files",
    )

    args = parser.parse_args()

    if not _is_cloud(args.path) and not os.path.exists(args.path):
        print(f"❌  File not found: {args.path}")
        sys.exit(1)

    result = run_background_sampling_test(
        path=args.path,
        output_dir=args.output_dir,
        force_background=args.force_background,
        no_ray=args.no_ray,
        ray_address=args.ray_address,
    )

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
