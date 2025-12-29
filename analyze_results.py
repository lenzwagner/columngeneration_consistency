#!/usr/bin/env python3
"""
Analyze results from CG experiments.
Calculates mean, min, max, median, std, and spread for key metrics.
"""

import pandas as pd
import sys
from pathlib import Path


def analyze_results(filepath: str):
    """Load results and compute statistics for key columns."""
    
    # Read Excel file
    df = pd.read_excel(filepath)
    print(f"Loaded {len(df)} rows from {filepath}\n")
    
    # Columns to analyze
    columns = [
        'undercover_naive', 'undercover_behavior',
        'cons_behavior', 'cons_naive',
        'perf_behavior', 'perf_naive',
        'understaffing_naive', 'understaffing_behavior'
    ]
    
    # Check which columns exist
    existing = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    
    if missing:
        print(f"Warning: Missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}\n")
    
    if not existing:
        print("No valid columns found!")
        return None
    
    # Compute statistics
    stats = df[existing].agg(['mean', 'min', 'max', 'median', 'std'])
    stats.loc['spread'] = stats.loc['max'] - stats.loc['min']
    
    # Print results
    print("=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    print(stats.round(4).to_string())
    print("=" * 80)
    
    # Print comparison summary
    print("\n📊 COMPARISON SUMMARY (Behavior vs Naive):")
    print("-" * 50)
    
    comparisons = [
        ('undercover', 'Undercoverage'),
        ('cons', 'Consistency'),
        ('perf', 'Perf Loss'),
        ('understaffing', 'Understaffing')
    ]
    
    for prefix, label in comparisons:
        naive_col = f"{prefix}_naive"
        behav_col = f"{prefix}_behavior"
        if naive_col in existing and behav_col in existing:
            naive_mean = df[naive_col].mean()
            behav_mean = df[behav_col].mean()
            diff = naive_mean - behav_mean
            pct = (diff / naive_mean) * 100 if naive_mean != 0 else 0
            better = "Behavior ✓" if behav_mean < naive_mean else "Naive ✓"
            print(f"  {label:15s}: Naive={naive_mean:8.2f}  Behavior={behav_mean:8.2f}  Δ={diff:+8.2f} ({pct:+5.1f}%)  [{better}]")
    
    print("-" * 50)
    
    return stats


def main():
    # Default to most recent DD file
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Find most recent DD_ file
        results_dir = Path("results")
        dd_files = sorted(results_dir.glob("DD_*.xlsx"), reverse=True)
        if dd_files:
            filepath = str(dd_files[0])
            print(f"Using most recent file: {filepath}\n")
        else:
            print("No DD_*.xlsx files found in results/")
            print("Usage: python analyze_results.py <path_to_excel>")
            sys.exit(1)
    
    analyze_results(filepath)


if __name__ == "__main__":
    main()
