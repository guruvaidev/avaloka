"""
Comprehensive Statistical Comparison: Old Sampler vs New Portfolio-Based Sampler

This script performs an unbiased statistical analysis comparing the accuracy
of the old sampling_agent.py vs the new sampling_agent_daft.py.

Usage:
    python tests/compare_samplers.py --csv-path "path/to/file.csv"
    
    Or run all tests:
    pytest tests/compare_samplers.py -v -s
"""

import os
import sys
import argparse
import json
import csv as csv_lib
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats as scipy_stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.sampling_agent import sample_data_from_source as old_sampler
from app.agents.sampling_agent_daft import sample_with_profiling as new_sampler


class SamplerComparison:
    """Statistical comparison between old and new samplers."""
    
    def __init__(self, csv_path: str, output_dir: str = "sampler_comparison_output"):
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Load original data
        print(f"\n{'='*80}")
        print(f"Loading original CSV: {csv_path}")
        print(f"{'='*80}")
        # Try multiple encodings like the samplers do
        try:
            self.original_df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.original_df = pd.read_csv(csv_path, encoding='iso-8859-1')
            except Exception:
                self.original_df = pd.read_csv(csv_path, encoding='cp1252')
        
        print(f"Original data shape: {self.original_df.shape}")
        print(f"Columns: {list(self.original_df.columns)}")
        
        self.results = {}
        
    def save_portfolio_samples(self, portfolio_samples: Dict[str, pd.DataFrame]):
        """Save portfolio samples to CSV files for inspection."""
        portfolio_dir = self.run_dir / "portfolio_samples"
        portfolio_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Saving {len(portfolio_samples)} portfolio samples to: {portfolio_dir}")
        print(f"{'='*80}")
        
        for name, df in portfolio_samples.items():
            file_path = portfolio_dir / f"{name}.csv"
            df.to_csv(file_path, index=False)
            print(f"  ✓ {name}: {len(df)} rows → {file_path.name}")
        
        return portfolio_dir
    
    def extract_portfolio_from_new_sampler(self) -> Dict[str, pd.DataFrame]:
        """Extract individual portfolio samples from new sampler for inspection."""
        from app.agents.sampling_agent_daft import (
            create_base_sample,
            create_sample_portfolio,
        )
        
        print(f"\n{'='*80}")
        print("Extracting Portfolio Samples from New Sampler")
        print(f"{'='*80}")
        
        # Get base sample and portfolio
        base_sample, full_df, total_rows, temp_file, _bytes_per_row = create_base_sample(self.csv_path, "csv")
        print(f"Base sample created: {base_sample.count_rows():,} rows")
        
        portfolio = create_sample_portfolio(
            base_sample,
            full_df=full_df,
            max_samples=5,
            sample_size_per=10_000
        )
        
        print(f"Portfolio created with {len(portfolio)} samples:")
        
        # Convert Daft DataFrames to Pandas for saving/analysis
        portfolio_dfs = {}
        for name, daft_df in portfolio.items():
            collected = daft_df.collect()
            pd_df = collected.to_pandas()
            portfolio_dfs[name] = pd_df
            print(f"  - {name}: {len(pd_df):,} rows")
        
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass
        
        return portfolio_dfs
    
    def run_samplers(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run both samplers and return their outputs as DataFrames."""
        print(f"\n{'='*80}")
        print("Running Old Sampler")
        print(f"{'='*80}")
        
        old_result = old_sampler(
            path=self.csv_path,
            source_type="csv",
            stratify_by=None,
            sample_size=1.0
        )
        
        if old_result.get("error"):
            raise RuntimeError(f"Old sampler failed: {old_result['error']}")
        
        old_sample_df = pd.DataFrame(old_result["rows"])
        print(f"Old sampler output: {len(old_sample_df)} rows")
        
        print(f"\n{'='*80}")
        print("Running New Portfolio-Based Sampler")
        print(f"{'='*80}")
        
        new_result = new_sampler(
            path=self.csv_path,
            source_type="csv",
            sample_size=1000
        )
        
        if new_result.get("error"):
            raise RuntimeError(f"New sampler failed: {new_result['error']}")
        
        new_sample_df = pd.DataFrame(new_result["rows"])
        print(f"New sampler output: {len(new_sample_df)} rows")
        
        # Save samples for inspection
        old_sample_df.to_csv(self.run_dir / "old_sampler_output.csv", index=False)
        new_sample_df.to_csv(self.run_dir / "new_sampler_output.csv", index=False)
        print(f"\n✓ Samples saved to: {self.run_dir}")
        
        return old_sample_df, new_sample_df
    
    def compute_numeric_statistics(
        self,
        original: pd.DataFrame,
        old_sample: pd.DataFrame,
        new_sample: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for numeric columns."""
        print(f"\n{'='*80}")
        print("Computing Numeric Column Statistics")
        print(f"{'='*80}")
        
        numeric_cols = original.select_dtypes(include=[np.number]).columns
        stats = {}
        
        for col in numeric_cols:
            if col not in old_sample.columns or col not in new_sample.columns:
                continue
            
            # Original statistics
            orig_values = original[col].dropna()
            if len(orig_values) == 0:
                continue
            
            orig_mean = orig_values.mean()
            orig_std = orig_values.std()
            orig_median = orig_values.median()
            orig_min = orig_values.min()
            orig_max = orig_values.max()
            
            # Old sampler statistics
            old_values = pd.to_numeric(old_sample[col], errors='coerce').dropna()
            old_mean = old_values.mean() if len(old_values) > 0 else np.nan
            old_std = old_values.std() if len(old_values) > 0 else np.nan
            old_median = old_values.median() if len(old_values) > 0 else np.nan
            
            # New sampler statistics
            new_values = pd.to_numeric(new_sample[col], errors='coerce').dropna()
            new_mean = new_values.mean() if len(new_values) > 0 else np.nan
            new_std = new_values.std() if len(new_values) > 0 else np.nan
            new_median = new_values.median() if len(new_values) > 0 else np.nan
            
            # Z-scores (how many standard deviations away from true mean)
            old_z_score = abs((old_mean - orig_mean) / orig_std) if orig_std > 0 else 0
            new_z_score = abs((new_mean - orig_mean) / orig_std) if orig_std > 0 else 0
            
            # Percentage errors
            old_mean_error = abs((old_mean - orig_mean) / orig_mean * 100) if orig_mean != 0 else 0
            new_mean_error = abs((new_mean - orig_mean) / orig_mean * 100) if orig_mean != 0 else 0
            
            old_std_error = abs((old_std - orig_std) / orig_std * 100) if orig_std != 0 else 0
            new_std_error = abs((new_std - orig_std) / orig_std * 100) if orig_std != 0 else 0
            
            stats[col] = {
                "original": {
                    "mean": float(orig_mean),
                    "std": float(orig_std),
                    "median": float(orig_median),
                    "min": float(orig_min),
                    "max": float(orig_max),
                    "count": int(len(orig_values))
                },
                "old_sampler": {
                    "mean": float(old_mean) if not np.isnan(old_mean) else None,
                    "std": float(old_std) if not np.isnan(old_std) else None,
                    "median": float(old_median) if not np.isnan(old_median) else None,
                    "count": int(len(old_values)),
                    "mean_error_pct": float(old_mean_error),
                    "std_error_pct": float(old_std_error),
                    "z_score": float(old_z_score)
                },
                "new_sampler": {
                    "mean": float(new_mean) if not np.isnan(new_mean) else None,
                    "std": float(new_std) if not np.isnan(new_std) else None,
                    "median": float(new_median) if not np.isnan(new_median) else None,
                    "count": int(len(new_values)),
                    "mean_error_pct": float(new_mean_error),
                    "std_error_pct": float(new_std_error),
                    "z_score": float(new_z_score)
                }
            }
            
            print(f"\n{col}:")
            print(f"  Original    → Mean: {orig_mean:,.2f}, Std: {orig_std:,.2f}")
            print(f"  Old Sampler → Mean: {old_mean:,.2f} (±{old_mean_error:.2f}%), Z-score: {old_z_score:.3f}")
            print(f"  New Sampler → Mean: {new_mean:,.2f} (±{new_mean_error:.2f}%), Z-score: {new_z_score:.3f}")
        
        return stats
    
    def compute_categorical_statistics(
        self,
        original: pd.DataFrame,
        old_sample: pd.DataFrame,
        new_sample: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for categorical columns."""
        print(f"\n{'='*80}")
        print("Computing Categorical Column Statistics")
        print(f"{'='*80}")
        
        categorical_cols = original.select_dtypes(include=['object']).columns
        stats = {}
        
        for col in categorical_cols:
            if col not in old_sample.columns or col not in new_sample.columns:
                continue
            
            # Original unique values
            orig_unique = original[col].nunique()
            orig_value_counts = original[col].value_counts()
            orig_top_5 = orig_value_counts.head(5).to_dict()
            
            # Old sampler unique values
            old_unique = old_sample[col].nunique()
            old_value_counts = old_sample[col].value_counts()
            
            # New sampler unique values
            new_unique = new_sample[col].nunique()
            new_value_counts = new_sample[col].value_counts()
            
            # Coverage: what % of original unique values are captured
            old_coverage = (old_unique / orig_unique * 100) if orig_unique > 0 else 0
            new_coverage = (new_unique / orig_unique * 100) if orig_unique > 0 else 0
            
            # Distribution similarity (Chi-square test if possible)
            try:
                # Align value counts
                all_values = set(orig_value_counts.index) | set(old_value_counts.index)
                orig_freq = [orig_value_counts.get(v, 0) for v in all_values]
                old_freq = [old_value_counts.get(v, 0) for v in all_values]
                
                # Normalize to proportions
                orig_prop = np.array(orig_freq) / sum(orig_freq)
                old_prop = np.array(old_freq) / sum(old_freq) if sum(old_freq) > 0 else np.zeros(len(old_freq))
                
                # KL divergence (measure of distribution difference)
                old_kl_div = scipy_stats.entropy(old_prop + 1e-10, orig_prop + 1e-10)
            except Exception:
                old_kl_div = None
            
            try:
                all_values = set(orig_value_counts.index) | set(new_value_counts.index)
                orig_freq = [orig_value_counts.get(v, 0) for v in all_values]
                new_freq = [new_value_counts.get(v, 0) for v in all_values]
                
                orig_prop = np.array(orig_freq) / sum(orig_freq)
                new_prop = np.array(new_freq) / sum(new_freq) if sum(new_freq) > 0 else np.zeros(len(new_freq))
                
                new_kl_div = scipy_stats.entropy(new_prop + 1e-10, orig_prop + 1e-10)
            except Exception:
                new_kl_div = None
            
            stats[col] = {
                "original": {
                    "unique_count": int(orig_unique),
                    "top_5_values": {str(k): int(v) for k, v in orig_top_5.items()},
                    "total_count": int(len(original[col]))
                },
                "old_sampler": {
                    "unique_count": int(old_unique),
                    "coverage_pct": float(old_coverage),
                    "kl_divergence": float(old_kl_div) if old_kl_div is not None else None,
                    "total_count": int(len(old_sample[col]))
                },
                "new_sampler": {
                    "unique_count": int(new_unique),
                    "coverage_pct": float(new_coverage),
                    "kl_divergence": float(new_kl_div) if new_kl_div is not None else None,
                    "total_count": int(len(new_sample[col]))
                }
            }
            
            print(f"\n{col}:")
            print(f"  Original    → Unique: {orig_unique:,}")
            old_kl_str = f"{old_kl_div:.4f}" if old_kl_div is not None else "N/A"
            new_kl_str = f"{new_kl_div:.4f}" if new_kl_div is not None else "N/A"
            print(f"  Old Sampler → Unique: {old_unique:,} ({old_coverage:.1f}% coverage), KL-div: {old_kl_str}")
            print(f"  New Sampler → Unique: {new_unique:,} ({new_coverage:.1f}% coverage), KL-div: {new_kl_str}")
        
        return stats
    
    def compute_overall_score(self, numeric_stats: Dict, categorical_stats: Dict) -> Dict[str, float]:
        """Compute overall accuracy scores for both samplers."""
        print(f"\n{'='*80}")
        print("Computing Overall Accuracy Scores")
        print(f"{'='*80}")
        
        # Numeric accuracy (lower error = better)
        old_numeric_errors = []
        new_numeric_errors = []
        
        for col, stats in numeric_stats.items():
            old_numeric_errors.append(stats["old_sampler"]["mean_error_pct"])
            old_numeric_errors.append(stats["old_sampler"]["std_error_pct"])
            new_numeric_errors.append(stats["new_sampler"]["mean_error_pct"])
            new_numeric_errors.append(stats["new_sampler"]["std_error_pct"])
        
        old_avg_numeric_error = np.mean(old_numeric_errors) if old_numeric_errors else 0
        new_avg_numeric_error = np.mean(new_numeric_errors) if new_numeric_errors else 0
        
        # Categorical coverage (higher = better)
        old_coverages = [stats["old_sampler"]["coverage_pct"] for stats in categorical_stats.values()]
        new_coverages = [stats["new_sampler"]["coverage_pct"] for stats in categorical_stats.values()]
        
        old_avg_coverage = np.mean(old_coverages) if old_coverages else 0
        new_avg_coverage = np.mean(new_coverages) if new_coverages else 0
        
        # KL divergence (lower = better distribution match)
        old_kl_divs = [stats["old_sampler"]["kl_divergence"] for stats in categorical_stats.values() 
                       if stats["old_sampler"]["kl_divergence"] is not None]
        new_kl_divs = [stats["new_sampler"]["kl_divergence"] for stats in categorical_stats.values() 
                       if stats["new_sampler"]["kl_divergence"] is not None]
        
        old_avg_kl = np.mean(old_kl_divs) if old_kl_divs else 0
        new_avg_kl = np.mean(new_kl_divs) if new_kl_divs else 0
        
        # Overall scores (0-100, higher = better)
        # Numeric: 100 - average error percentage
        old_numeric_score = max(0, 100 - old_avg_numeric_error)
        new_numeric_score = max(0, 100 - new_avg_numeric_error)
        
        # Categorical: average coverage
        old_categorical_score = old_avg_coverage
        new_categorical_score = new_avg_coverage
        
        # Distribution: 100 - (KL divergence * 10), capped at 0
        old_distribution_score = max(0, 100 - (old_avg_kl * 10))
        new_distribution_score = max(0, 100 - (new_avg_kl * 10))
        
        # Combined score (weighted average)
        old_overall = (old_numeric_score * 0.4 + old_categorical_score * 0.3 + old_distribution_score * 0.3)
        new_overall = (new_numeric_score * 0.4 + new_categorical_score * 0.3 + new_distribution_score * 0.3)
        
        scores = {
            "old_sampler": {
                "numeric_accuracy": float(old_numeric_score),
                "categorical_coverage": float(old_categorical_score),
                "distribution_similarity": float(old_distribution_score),
                "overall_score": float(old_overall)
            },
            "new_sampler": {
                "numeric_accuracy": float(new_numeric_score),
                "categorical_coverage": float(new_categorical_score),
                "distribution_similarity": float(new_distribution_score),
                "overall_score": float(new_overall)
            }
        }
        
        print(f"\nOld Sampler Scores:")
        print(f"  Numeric Accuracy: {old_numeric_score:.2f}/100")
        print(f"  Categorical Coverage: {old_categorical_score:.2f}/100")
        print(f"  Distribution Similarity: {old_distribution_score:.2f}/100")
        print(f"  OVERALL SCORE: {old_overall:.2f}/100")
        
        print(f"\nNew Sampler Scores:")
        print(f"  Numeric Accuracy: {new_numeric_score:.2f}/100")
        print(f"  Categorical Coverage: {new_categorical_score:.2f}/100")
        print(f"  Distribution Similarity: {new_distribution_score:.2f}/100")
        print(f"  OVERALL SCORE: {new_overall:.2f}/100")
        
        return scores
    
    def generate_conclusion(self, scores: Dict) -> str:
        """Generate conclusion about which sampler is better."""
        old_score = scores["old_sampler"]["overall_score"]
        new_score = scores["new_sampler"]["overall_score"]
        
        diff = new_score - old_score
        diff_pct = (diff / old_score * 100) if old_score > 0 else 0
        
        if abs(diff) < 2:
            winner = "TIE"
            conclusion = f"Both samplers perform nearly identically (difference: {abs(diff):.2f} points)."
        elif new_score > old_score:
            winner = "NEW SAMPLER"
            conclusion = f"The NEW portfolio-based sampler is {abs(diff_pct):.1f}% more accurate ({diff:.2f} points higher)."
        else:
            winner = "OLD SAMPLER"
            conclusion = f"The OLD sampler is {abs(diff_pct):.1f}% more accurate ({abs(diff):.2f} points higher)."
        
        return f"\n{'='*80}\nCONCLUSION: {winner}\n{'='*80}\n{conclusion}\n{'='*80}\n"
    
    def run_full_comparison(self):
        """Run complete comparison analysis."""
        print(f"\n{'#'*80}")
        print(f"# SAMPLER COMPARISON TEST")
        print(f"# File: {self.csv_path}")
        print(f"# Output: {self.run_dir}")
        print(f"{'#'*80}")
        
        # Extract and save portfolio samples
        portfolio_samples = self.extract_portfolio_from_new_sampler()
        portfolio_dir = self.save_portfolio_samples(portfolio_samples)
        
        # Run both samplers
        old_sample, new_sample = self.run_samplers()
        
        # Compute statistics
        numeric_stats = self.compute_numeric_statistics(
            self.original_df, old_sample, new_sample
        )
        
        categorical_stats = self.compute_categorical_statistics(
            self.original_df, old_sample, new_sample
        )
        
        # Compute scores
        scores = self.compute_overall_score(numeric_stats, categorical_stats)
        
        # Generate conclusion
        conclusion = self.generate_conclusion(scores)
        print(conclusion)
        
        # Save results
        results = {
            "metadata": {
                "csv_path": str(self.csv_path),
                "output_dir": str(self.run_dir),
                "portfolio_dir": str(portfolio_dir),
                "timestamp": datetime.now().isoformat(),
                "original_shape": list(self.original_df.shape),
                "old_sample_size": len(old_sample),
                "new_sample_size": len(new_sample)
            },
            "numeric_statistics": numeric_stats,
            "categorical_statistics": categorical_stats,
            "scores": scores,
            "conclusion": conclusion
        }
        
        results_file = self.run_dir / "comparison_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Full results saved to: {results_file}")
        print(f"✓ Portfolio samples saved to: {portfolio_dir}")
        print(f"✓ Old sampler output: {self.run_dir / 'old_sampler_output.csv'}")
        print(f"✓ New sampler output: {self.run_dir / 'new_sampler_output.csv'}")
        
        return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Compare old vs new sampler statistically")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="Global YouTube Statistics.csv",
        help="Path to CSV file to test"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sampler_comparison_output",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found: {args.csv_path}")
        sys.exit(1)
    
    comparison = SamplerComparison(args.csv_path, args.output_dir)
    results = comparison.run_full_comparison()
    
    return results


# Pytest test
def test_sampler_comparison_youtube():
    """Test sampler comparison with Global YouTube Statistics CSV."""
    csv_path = "Global YouTube Statistics.csv"
    
    if not os.path.exists(csv_path):
        import pytest
        pytest.skip(f"Test file not found: {csv_path}")
    
    comparison = SamplerComparison(csv_path, "sampler_comparison_output")
    results = comparison.run_full_comparison()
    
    # Assertions
    assert results is not None
    assert "scores" in results
    assert "conclusion" in results
    
    # Both samplers should have reasonable scores
    assert results["scores"]["old_sampler"]["overall_score"] > 0
    assert results["scores"]["new_sampler"]["overall_score"] > 0
    
    print("\n" + results["conclusion"])


if __name__ == "__main__":
    main()
