"""
Test script comparing MIP vs DP Numba solver.

Fast comparison using only:
- MIP (Gurobi)
- DP Numba (JIT-compiled, fastest)
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from subproblem import Subproblem
from subproblem_dp_optimized import NUMBA_AVAILABLE
import gurobipy as gu

if NUMBA_AVAILABLE:
    from subproblem_dp_optimized import SubproblemDPNumba


def compare_mip_vs_numba(
    duals_i: float,
    duals_ts: dict,
    T: list,
    K: list,
    eps: float,
    chi: int,
) -> dict:
    """
    Compare MIP and DP Numba solvers.
    """
    data = pd.DataFrame({
        'I': [1] + [np.nan] * (max(1, len(T), len(K)) - 1),
        'T': T + [np.nan] * (max(1, len(T), len(K)) - len(T)),
        'K': K + [np.nan] * (max(1, len(T), len(K)) - len(K))
    })

    result = {
        'mip_obj': None, 'mip_time': 0,
        'dp_numba_obj': None, 'dp_numba_time': 0,
        'mip_schedule': [],
        'numba_schedule': [],
        'mip_pool_size': 0,
        'numba_pool_size': 0,
        'match': False,
    }
    
    n_days = len(T)

    # MIP with solution pool
    mip = Subproblem(duals_i=duals_i, duals_ts=duals_ts, df=data, i=1,
                     iteration=0, eps=eps, Min_WD_i={1: 2}, Max_WD_i={1: 5}, chi=chi)
    mip.model.setParam('OutputFlag', 0)
    # Enable solution pool
    mip.model.setParam('PoolSearchMode', 2)  # Find multiple optimal solutions
    mip.model.setParam('PoolSolutions', 100)  # Store up to 100 solutions
    mip.model.setParam('PoolGap', 0.0)  # Only optimal solutions
    mip.buildModel()
    t0 = time.perf_counter()
    mip.solveModelOpt(timeLimit=60)
    result['mip_time'] = time.perf_counter() - t0
    
    if mip.getStatus() == gu.GRB.OPTIMAL:
        result['mip_obj'] = mip.model.ObjVal
        # Count solutions in pool with same optimal value
        n_solutions = mip.model.SolCount
        opt_val = mip.model.ObjVal
        pool_count = 0
        for sol_idx in range(n_solutions):
            mip.model.setParam('SolutionNumber', sol_idx)
            if abs(mip.model.PoolObjVal - opt_val) < 1e-6:
                pool_count += 1
        result['mip_pool_size'] = pool_count
        
        # Extract first schedule (after timing)
        mip.model.setParam('SolutionNumber', 0)
        mip_x = mip.getOptX()
        mip_sched = []
        for day in T:
            shift_worked = 0
            for k in K:
                if mip_x.get((day, k), 0) > 0.5:
                    shift_worked = k
                    break
            mip_sched.append(shift_worked)
        result['mip_schedule'] = mip_sched
        
        # Extract 2 more schedules from pool if available
        result['mip_pool_schedules'] = []
        for sol_idx in range(1, min(3, n_solutions)):
            mip.model.setParam('SolutionNumber', sol_idx)
            if abs(mip.model.PoolObjVal - opt_val) < 1e-6:
                pool_sched = []
                for day in T:
                    shift_worked = 0
                    for k in K:
                        # Access Xn attribute directly from stored variable
                        if mip.x[day, k].Xn > 0.5:
                            shift_worked = k
                            break
                    pool_sched.append(shift_worked)
                result['mip_pool_schedules'].append(pool_sched)

    # DP Numba
    if NUMBA_AVAILABLE:
        dp_numba = SubproblemDPNumba(duals_i=duals_i, duals_ts=duals_ts, df=data, i=1,
                                      iteration=0, eps=eps, Min_WD_i=2, Max_WD_i=5, chi=chi)
        dp_numba.buildModel()
        t0 = time.perf_counter()
        dp_numba.solveModelOpt(timeLimit=60)
        result['dp_numba_time'] = time.perf_counter() - t0
        
        if dp_numba.getStatus() == gu.GRB.OPTIMAL:
            result['dp_numba_obj'] = dp_numba.objval
            result['numba_pool_size'] = dp_numba.n_optimal
            # Extract first schedule (after timing)
            if dp_numba.best_path is not None:
                numba_sched = []
                for d in range(n_days):
                    shift = int(dp_numba.best_path[d + 1])
                    numba_sched.append(max(0, shift))
                result['numba_schedule'] = numba_sched

    # Check match
    tol = 1e-4
    if result['mip_obj'] is not None and result['dp_numba_obj'] is not None:
        result['match'] = abs(result['mip_obj'] - result['dp_numba_obj']) < tol

    return result


def run_random_tests(n_tests: int = 200, seed_start: int = 0):
    """
    Run n_tests random tests comparing MIP vs DP Numba.
    """

    print(f"Running {n_tests} random tests (MIP vs DP Numba)...")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("=" * 60)

    passed = 0
    failed = 0
    failed_cases = []

    mip_times = []
    dp_numba_times = []

    for i in range(n_tests):
        seed = seed_start + i
        np.random.seed(seed)

        # Fixed T=20, varying duals
        n_days = 28
        T = list(range(1, n_days + 1))
        K = [1, 2, 3]

        eps = 0.05
        chi = 4
        duals_i = np.random.randint(-22, 5)
        duals_ts = {(t, k): np.random.randint(0, 19) for t in T for k in K}

        result = compare_mip_vs_numba(
            duals_i=duals_i, duals_ts=duals_ts, T=T, K=K, eps=eps, chi=chi
        )

        mip_times.append(result['mip_time'])
        dp_numba_times.append(result['dp_numba_time'])

        status = "✓" if result['match'] else "✗"
        if result['match']:
            passed += 1
        else:
            failed += 1
            failed_cases.append({'seed': seed, 'mip_obj': result['mip_obj'], 'dp_numba_obj': result['dp_numba_obj']})
        
        # Print lines per instance
        print(f"[{i+1:3d}] seed={seed:4d} | MIP={result['mip_obj']:10.4f} ({result['mip_time']*1000:.1f}ms, pool={result['mip_pool_size']}) | NUMBA={result['dp_numba_obj']:10.4f} ({result['dp_numba_time']*1000:.1f}ms, pool={result['numba_pool_size']}) | {status}")
        print(f"      MIP:   {result['mip_schedule']}")
        sched_match = " ←" if result['mip_schedule'] == result['numba_schedule'] else ""
        print(f"      NUMBA: {result['numba_schedule']}{sched_match}")
        # Print 2 more pool solutions if available
        pool_scheds = result.get('mip_pool_schedules', [])
        if pool_scheds:
            for idx, ps in enumerate(pool_scheds):
                print(f"      Pool{idx+2}: {ps}")

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{n_tests} passed, {failed}/{n_tests} failed")
    print("=" * 60)

    print(f"\nTiming (avg ms):")
    print(f"  MIP:       {np.mean(mip_times)*1000:8.2f}ms")
    print(f"  DP_NUMBA:  {np.mean(dp_numba_times)*1000:8.2f}ms  (Speedup: {np.mean(mip_times)/np.mean(dp_numba_times):.1f}x)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mip_vs_numba_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': n_tests,
                'passed': passed,
                'failed': failed,
                'timing_ms': {
                    'mip_avg': np.mean(mip_times) * 1000,
                    'dp_numba_avg': np.mean(dp_numba_times) * 1000,
                }
            },
            'failed_cases': failed_cases
        }, f, indent=2)

    print(f"\nResults saved to: {filename}")

    if failed_cases:
        print("\nFailed cases:")
        for case in failed_cases[:10]:
            print(f"  Seed {case['seed']}: MIP={case['mip_obj']:.4f}, NUMBA={case['dp_numba_obj']}")
    else:
        print("\n✓ All tests passed!")
    
    return passed == n_tests


if __name__ == "__main__":
    import sys
    
    n_tests = 200
    if len(sys.argv) > 1:
        n_tests = int(sys.argv[1])
    
    success = run_random_tests(n_tests=n_tests)
    sys.exit(0 if success else 1)
