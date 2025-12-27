"""
Benchmark: Bidirectional DP vs Forward-Only DP

Compares performance across different planning horizons to determine
which approach is faster.
"""

import time
import pandas as pd
import numpy as np
from subproblem_factory import create_subproblem

def create_test_data(n_days):
    """Create test data for given horizon."""
    T = list(range(1, n_days + 1))
    K = [1, 2, 3]
    I = list(range(1, 21))
    
    max_len = max(len(I), len(T), len(K))
    data = pd.DataFrame({
        'I': I + [np.nan] * (max_len - len(I)),
        'T': T + [np.nan] * (max_len - len(T)),
        'K': K + [np.nan] * (max_len - len(K))
    })
    
    # Random duals
    np.random.seed(42)
    duals_i = -5.0
    duals_ts = {(t, k): 0.3 + 0.4 * np.random.random() for t in T for k in K}
    
    return data, duals_i, duals_ts, T

def benchmark_single(n_days, n_runs=5):
    """Benchmark both methods for a given horizon."""
    data, duals_i, duals_ts, T = create_test_data(n_days)
    
    # Warm-up run (JIT compilation)
    sp = create_subproblem('labeling', duals_i, duals_ts, data, 1, 1, 0.05, 2, 5, 4)
    sp.buildModel()
    sp.solveModelOpt(60)
    
    sp = create_subproblem('labeling_bidir', duals_i, duals_ts, data, 1, 1, 0.05, 2, 5, 4)
    sp.buildModel()
    sp.solveModelOpt(60)
    
    # Benchmark forward-only
    fwd_times = []
    fwd_obj = None
    for _ in range(n_runs):
        sp = create_subproblem('labeling', duals_i, duals_ts, data, 1, 1, 0.05, 2, 5, 4)
        sp.buildModel()
        t0 = time.perf_counter()
        sp.solveModelOpt(60)
        fwd_times.append(time.perf_counter() - t0)
        fwd_obj = sp.model.objval
    
    # Benchmark bidirectional
    bidir_times = []
    bidir_obj = None
    for _ in range(n_runs):
        sp = create_subproblem('labeling_bidir', duals_i, duals_ts, data, 1, 1, 0.05, 2, 5, 4)
        sp.buildModel()
        t0 = time.perf_counter()
        sp.solveModelOpt(60)
        bidir_times.append(time.perf_counter() - t0)
        bidir_obj = sp.model.objval
    
    fwd_avg = np.mean(fwd_times) * 1000
    bidir_avg = np.mean(bidir_times) * 1000
    speedup = fwd_avg / bidir_avg if bidir_avg > 0 else 0
    match = abs(fwd_obj - bidir_obj) < 1e-6
    
    return {
        'T': n_days,
        'fwd_ms': fwd_avg,
        'bidir_ms': bidir_avg,
        'speedup': speedup,
        'match': match,
        'objval': fwd_obj
    }

if __name__ == '__main__':
    print("=" * 70)
    print("Benchmark: Bidirectional DP vs Forward-Only DP")
    print("=" * 70)
    print()
    
    horizons = [7, 10, 14, 16, 20, 25, 28, 30]
    results = []
    
    print(f"{'T':>4} | {'Forward (ms)':>12} | {'Bidir (ms)':>12} | {'Speedup':>8} | {'Match':>5}")
    print("-" * 55)
    
    for T in horizons:
        try:
            result = benchmark_single(T, n_runs=5)
            results.append(result)
            
            winner = "bidir" if result['speedup'] > 1 else "fwd"
            print(f"{result['T']:>4} | {result['fwd_ms']:>12.2f} | {result['bidir_ms']:>12.2f} | {result['speedup']:>7.2f}x | {str(result['match']):>5} ({winner})")
        except Exception as e:
            print(f"{T:>4} | ERROR: {e}")
    
    print("-" * 55)
    
    # Summary
    bidir_wins = sum(1 for r in results if r['speedup'] > 1)
    fwd_wins = len(results) - bidir_wins
    
    print(f"\nSummary: Bidir wins {bidir_wins}/{len(results)}, Forward wins {fwd_wins}/{len(results)}")
    
    if all(r['match'] for r in results):
        print("All objective values match ✓")
    else:
        print("WARNING: Some objective values do not match!")
