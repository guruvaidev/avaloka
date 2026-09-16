"""
Standalone test for Data Profiling Agent

Tests the profiling agent with output from sampling_agent_daft
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.sampling_agent_daft import sample_with_profiling
from app.agents.profiling_agent import profile_dataset


def test_profiling_agent(csv_path: str, *, use_ray: bool = True):
    """Test profiling agent end-to-end with sampling agent."""
    
    print("="*80)
    print("DATA PROFILING AGENT - STANDALONE TEST")
    print("="*80)
    print(f"File: {csv_path}\n")
    
    # Step 1: Run sampling agent
    print("Step 1: Running sampling agent (portfolio-based sampling)...")
    print("-"*80)
    
    sampling_result = sample_with_profiling(csv_path, "csv", use_ray=use_ray)
    
    if sampling_result.get("error"):
        print(f"❌ Sampling failed: {sampling_result['error']}")
        return
    
    profiling_result = sampling_result.get("profiling_result", {})
    sample_rows = sampling_result.get("rows", [])
    schema = sampling_result.get("schema", [])
    
    print(f"✓ Sampling completed:")
    print(f"  - Total rows: {profiling_result.get('data_shape', {}).get('rows', 0):,}")
    print(f"  - Columns: {len(schema)}")
    print(f"  - Display sample: {len(sample_rows)} rows")
    print(f"  - Portfolio samples: {profiling_result.get('portfolio_metadata', {}).get('num_samples', 0)}")
    print()
    
    # Step 2: Run profiling agent
    print("Step 2: Running profiling agent (semantic analysis with LLM)...")
    print("-"*80)
    
    enhanced_profile = profile_dataset(
        profiling_result=profiling_result,
        sample_rows=sample_rows,
        schema=schema,
        dataset_name=os.path.basename(csv_path),
        use_llm=True
    )
    
    semantic = enhanced_profile.get("semantic_analysis", {})
    status = semantic.get("status", "unknown")
    
    print(f"✓ Profiling completed: status={status}\n")
    
    # Step 3: Display results
    print("="*80)
    print("PROFILING RESULTS")
    print("="*80)
    print()
    
    if status == "completed":
        # Domain classification
        domain = semantic.get("domain", {})
        print("📊 DOMAIN CLASSIFICATION")
        print(f"  Category: {domain.get('category', 'unknown')}")
        print(f"  Confidence: {domain.get('confidence', 0):.0%}")
        print(f"  Reasoning: {domain.get('reasoning', 'N/A')}")
        print()
        
        # Column explanations (show all)
        column_explanations = semantic.get("column_explanations", {})
        print(f"📝 COLUMN EXPLANATIONS (all {len(column_explanations)} columns)")
        for i, (col, explanation) in enumerate(column_explanations.items(), 1):
            print(f"  {i}. {col}:")
            print(f"     {explanation}")
        print()
        
        # Data quality insights
        quality = semantic.get("data_quality_insights", {})
        print("✅ DATA QUALITY INSIGHTS")
        
        strengths = quality.get("strengths", [])
        if strengths:
            print("  Strengths:")
            for strength in strengths:
                print(f"    • {strength}")
        
        concerns = quality.get("concerns", [])
        if concerns:
            print("  Concerns:")
            for concern in concerns:
                print(f"    ⚠️  {concern}")
        
        recommendations = quality.get("recommendations", [])
        if recommendations:
            print("  Recommendations:")
            for rec in recommendations:
                print(f"    → {rec}")
        print()
        
        # Predictive power
        pred_power = semantic.get("predictive_power", {})
        high_value = pred_power.get("high_value_columns", [])
        low_value = pred_power.get("low_value_columns", [])
        
        print("🎯 PREDICTIVE POWER")
        if high_value:
            print("  High-value columns:")
            for item in high_value:
                print(f"    ✓ {item['column']}: {item['reason']}")
        
        if low_value:
            print("  Low-value columns:")
            for item in low_value:
                print(f"    ✗ {item['column']}: {item['reason']}")
        print()
        
        # Analysis suggestions
        suggestions = semantic.get("analysis_suggestions", [])
        if suggestions:
            print("💡 ANALYSIS SUGGESTIONS")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        print()
        
        # Quick insights (NEW)
        quick_insights = semantic.get("quick_insights", {})
        if quick_insights:
            print("⚡ QUICK INSIGHTS")
            
            readiness_score = quick_insights.get("data_readiness_score", 0)
            readiness_reasoning = quick_insights.get("readiness_reasoning", "")
            print(f"  Data Readiness: {readiness_score:.0f}/100")
            if readiness_reasoning:
                print(f"  → {readiness_reasoning}")
            print()
            
            key_relationships = quick_insights.get("key_relationships", [])
            if key_relationships:
                print("  Key Relationships:")
                for rel in key_relationships:
                    cols = rel.get("columns", [])
                    desc = rel.get("relationship", "")
                    print(f"    • {' ↔ '.join(cols)}: {desc}")
                print()
            
            red_flags = quick_insights.get("red_flags", [])
            if red_flags:
                print("  🚩 Red Flags:")
                for flag in red_flags:
                    print(f"    ⚠️  {flag}")
                print()
            
            quick_wins = quick_insights.get("quick_wins", [])
            if quick_wins:
                print("  ✨ Quick Wins:")
                for win in quick_wins:
                    print(f"    ✓ {win}")
            print()
    
    else:
        print(f"⚠️  Semantic analysis {status}")
        if "reason" in semantic:
            print(f"   Reason: {semantic['reason']}")
        print()
        print("Statistical profile is still available:")
        print(f"  - {len(enhanced_profile.get('column_statistics', {}))} columns analyzed")
        print(f"  - Data completeness: {enhanced_profile.get('data_quality', {}).get('overall_completeness', 0):.1%}")
        print()
    
    # Step 4: Save output
    output_dir = project_root / "profiling_test_output"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"profile_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(enhanced_profile, f, indent=2, default=str)
    
    print("="*80)
    print(f"✓ Full profile saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test data profiling agent")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="Global YouTube Statistics.csv",
        help="Path to CSV file to profile"
    )
    parser.add_argument(
        "--no-ray",
        action="store_true",
        help="Disable Ray usage in the sampling agent",
    )
    
    args = parser.parse_args()
    
    csv_path = args.csv_path
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(project_root, csv_path)
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    
    test_profiling_agent(csv_path, use_ray=(not args.no_ray))
