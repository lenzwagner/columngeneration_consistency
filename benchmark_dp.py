"""
Benchmark script to compare DP solvers:
- Original DP (subproblem_dp.py)
- Optimized DP (subproblem_dp_optimized.py)
- Numba DP (subproblem_dp_optimized.py - SubproblemDPNumba)
- BiDir DP (subproblem_dp_optimized.py - SubproblemDPBidir)
- MIP (subproblem.py)
"""

import pandas as pd
import numpy as np
import time
from subproblem import Subproblem
from subproblem_dp import SubproblemDP
from subproblem_dp_optimized import SubproblemDPOptimized, NUMBA_AVAILABLE
import gurobipy as gu

if NUMBA_AVAILABLE:
    from subproblem_dp_optimized import SubproblemDPNumba, SubproblemDPBidir


def run_benchmark(n_tests: int = 20, n_days_list: list = None):
    """Run benchmark comparing all solvers."""
    
    if n_days_list is None:
        n_days_list = [7, 10, 14, 21, 28]
    
    K = [1, 2, 3]
    
    print("=" * 80)
    print("DP Solver Benchmark")
    print("=" * 80)
    print(f"Tests per instance size: {n_tests}")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print()
    
    results = []
    
    for n_days in n_days_list:
        T = list(range(1, n_days + 1))
        
        mip_times = []
        dp_times = []
        dp_opt_times = []
        dp_numba_times = []
        
        all_match = True
        
        for seed in range(n_tests):
            np.random.seed(seed * 100 + n_days)
            
            eps = np.random.uniform(0.05, 0.15)
            chi = np.random.randint(2, 5)
            duals_i = np.random.uniform(-5, 10)
            duals_ts = {(t, k): np.random.uniform(0, 10) for t in T for k in K}
            
            data = pd.DataFrame({
                'I': [1] + [np.nan] * (max(1, len(T), len(K)) - 1),
                'T': T + [np.nan] * (max(1, len(T), len(K)) - len(T)),
                'K': K + [np.nan] * (max(1, len(T), len(K)) - len(K))
            })
            
            # MIP
            mip = Subproblem(duals_i=duals_i, duals_ts=duals_ts, df=data, i=1, 
                            iteration=0, eps=eps, Min_WD_i={1: 2}, Max_WD_i={1: 5}, chi=chi)
            mip.model.setParam('OutputFlag', 0)
            mip.buildModel()
            t0 = time.perf_counter()
            mip.solveModelOpt(timeLimit=60)
            mip_times.append(time.perf_counter() - t0)
            mip_obj = mip.model.ObjVal if mip.getStatus() == gu.GRB.OPTIMAL else float('inf')
            
            # Original DP
            dp = SubproblemDP(duals_i=duals_i, duals_ts=duals_ts, df=data, i=1,
                             iteration=0, eps=eps, Min_WD_i=2, Max_WD_i=5, chi=chi)
            dp.buildModel()
            t0 = time.perf_counter()
            dp.solveModelOpt(timeLimit=60)
            dp_times.append(time.perf_counter() - t0)
            dp_obj = dp.objval if dp.getStatus() == gu.GRB.OPTIMAL else float('inf')
            
            # Optimized DP
            dp_opt = SubproblemDPOptimized(duals_i=duals_i, duals_ts=duals_ts, df=data, i=1,
                                           iteration=0, eps=eps, Min_WD_i=2, Max_WD_i=5, chi=chi)
            dp_opt.buildModel()
            t0 = time.perf_counter()
            dp_opt.solveModelOpt(timeLimit=60)
            dp_opt_times.append(time.perf_counter() - t0)
            dp_opt_obj = dp_opt.objval if dp_opt.getStatus() == gu.GRB.OPTIMAL else float('inf')
            
            # Numba DP (if available)
            dp_numba_obj = float('inf')
            if NUMBA_AVAILABLE:
                dp_numba = SubproblemDPNumba(duals_i=duals_i, duals_ts=duals_ts, df=data, i=1,
                                             iteration=0, eps=eps, Min_WD_i=2, Max_WD_i=5, chi=chi)
                dp_numba.buildModel()
                t0 = time.perf_counter()
                dp_numba.solveModelOpt(timeLimit=60)
                dp_numba_times.append(time.perf_counter() - t0)
                dp_numba_obj = dp_numba.objval if dp_numba.getStatus() == gu.GRB.OPTIMAL else float('inf')
            
            # Check correctness
            if abs(dp_opt_obj - mip_obj) > 1e-4 or abs(dp_opt_obj - dp_obj) > 1e-4:
                all_match = False
                print(f"  Mismatch at T={n_days}, seed={seed}: MIP={mip_obj:.4f}, DP={dp_obj:.4f}, DP_OPT={dp_opt_obj:.4f}")
            if NUMBA_AVAILABLE and (abs(dp_numba_obj - mip_obj) > 1e-4 or abs(dp_numba_obj - dp_obj) > 1e-4):
                all_match = False
                print(f"  Numba Mismatch at T={n_days}, seed={seed}: MIP={mip_obj:.4f}, DP={dp_obj:.4f}, DP_NUMBA={dp_numba_obj:.4f}")
        
        # Compute averages
        mip_avg = np.mean(mip_times) * 1000
        dp_avg = np.mean(dp_times) * 1000
        dp_opt_avg = np.mean(dp_opt_times) * 1000
        
        speedup_vs_dp = dp_avg / dp_opt_avg if dp_opt_avg > 0 else float('inf')
        speedup_vs_mip = mip_avg / dp_opt_avg if dp_opt_avg > 0 else float('inf')
        
        status = "✓" if all_match else "✗"
        
        if NUMBA_AVAILABLE and dp_numba_times:
            dp_numba_avg = np.mean(dp_numba_times) * 1000
            print(f"T={n_days:2d}: MIP={mip_avg:7.1f}ms | DP={dp_avg:8.1f}ms | OPT={dp_opt_avg:7.1f}ms | NUMBA={dp_numba_avg:7.1f}ms | {status}")
        else:
            print(f"T={n_days:2d}: MIP={mip_avg:7.1f}ms | DP={dp_avg:8.1f}ms | DP_OPT={dp_opt_avg:7.1f}ms | "
                  f"Speedup vs DP: {speedup_vs_dp:5.1f}x | vs MIP: {speedup_vs_mip:4.1f}x | {status}")
        
        results.append({
            'n_days': n_days,
            'mip_avg_ms': mip_avg,
            'dp_avg_ms': dp_avg,
            'dp_opt_avg_ms': dp_opt_avg,
            'speedup_vs_dp': speedup_vs_dp,
            'speedup_vs_mip': speedup_vs_mip,
            'correct': all_match
        })
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    total_speedup_dp = sum(r['dp_avg_ms'] for r in results) / sum(r['dp_opt_avg_ms'] for r in results)
    print(f"Overall speedup vs original DP: {total_speedup_dp:.1f}x")
    
    return results


if __name__ == "__main__":
    run_benchmark(n_tests=10, n_days_list=[7, 8, 9, 10, 12, 14, 16, 18, 20])
