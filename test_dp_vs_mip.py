"""
Test script to compare DP and MIP subproblem solvers with fictional duals.
Verifies that both solvers produce identical optimal solutions.
"""

import pandas as pd
import numpy as np
from subproblem import Subproblem
from subproblem_dp import SubproblemDP
import gurobipy as gu

def compare_dp_vs_mip(
    duals_i: float,
    duals_ts: dict,
    T: list,
    K: list,
    eps: float = 0.1,
    chi: int = 3,
    Min_WD_i: int = 2,
    Max_WD_i: int = 5,
    verbose: bool = False
) -> dict:
    """
    Compare DP and MIP solvers with identical inputs.
    
    Returns dict with:
    - 'dp_obj': DP objective value
    - 'mip_obj': MIP objective value
    - 'obj_match': True if objectives match within tolerance
    - 'schedule_match': True if schedules are identical
    - 'dp_schedule': DP schedule as dict
    - 'mip_schedule': MIP schedule as dict
    """
    # Create DataFrame expected by solvers
    data = pd.DataFrame({
        'I': [1] + [np.nan] * (max(1, len(T), len(K)) - 1),
        'T': T + [np.nan] * (max(1, len(T), len(K)) - len(T)),
        'K': K + [np.nan] * (max(1, len(T), len(K)) - len(K))
    })
    
    # Create and solve MIP
    mip = Subproblem(
        duals_i=duals_i,
        duals_ts=duals_ts,
        df=data,
        i=1,
        iteration=0,
        eps=eps,
        Min_WD_i={1: Min_WD_i},
        Max_WD_i={1: Max_WD_i},
        chi=chi
    )
    mip.buildModel()
    mip.solveModelOpt(timeLimit=60)
    
    # Create and solve DP
    dp = SubproblemDP(
        duals_i=duals_i,
        duals_ts=duals_ts,
        df=data,
        i=1,
        iteration=0,
        eps=eps,
        Min_WD_i=Min_WD_i,
        Max_WD_i=Max_WD_i,
        chi=chi
    )
    dp.buildModel()  # No-op for DP
    dp.solveModelOpt(timeLimit=60)
    
    # Extract results
    mip_status = mip.getStatus()
    dp_status = dp.getStatus()
    
    result = {
        'dp_obj': None,
        'mip_obj': None,
        'obj_match': False,
        'schedule_match': False,
        'dp_schedule': {},
        'mip_schedule': {},
        'mip_status': mip_status,
        'dp_status': dp_status,
    }
    
    if mip_status == gu.GRB.OPTIMAL and dp_status == gu.GRB.OPTIMAL:
        mip_obj = mip.model.ObjVal
        dp_obj = dp.objval
        
        result['mip_obj'] = mip_obj
        result['dp_obj'] = dp_obj
        
        # Compare objectives
        result['obj_match'] = abs(mip_obj - dp_obj) < 1e-5
        
        # Compare schedules
        mip_x = mip.getOptX()
        dp_x = dp.getOptX()
        
        result['mip_schedule'] = {k: v for k, v in mip_x.items() if abs(v) > 0.5}
        result['dp_schedule'] = {k: v for k, v in dp_x.items() if abs(v) > 0.5}
        
        # Check schedule match
        mip_set = set(result['mip_schedule'].keys())
        dp_set = set(result['dp_schedule'].keys())
        result['schedule_match'] = mip_set == dp_set
        
        if verbose:
            print(f"  MIP obj: {mip_obj:.6f}")
            print(f"  DP obj:  {dp_obj:.6f}")
            print(f"  MIP schedule: {sorted(mip_set)}")
            print(f"  DP schedule:  {sorted(dp_set)}")
    else:
        if verbose:
            print(f"  MIP status: {mip_status}")
            print(f"  DP status: {dp_status}")
    
    return result


def run_test_scenarios():
    """Run multiple test scenarios comparing DP vs MIP."""
    
    T = list(range(1, 8))  # 7 days
    K = [1, 2, 3]          # 3 shifts
    
    # Initialize duals_ts with zeros
    duals_ts_zero = {(t, k): 0.0 for t in T for k in K}
    
    scenarios = []
    
    # Scenario 1: Zero duals (baseline)
    scenarios.append({
        'name': 'Zero duals (baseline)',
        'duals_i': 0.0,
        'duals_ts': duals_ts_zero.copy()
    })
    
    # Scenario 2: High lambda dual
    scenarios.append({
        'name': 'High lambda dual only',
        'duals_i': 10.0,
        'duals_ts': duals_ts_zero.copy()
    })
    
    # Scenario 3: Single high demand dual
    duals_single = duals_ts_zero.copy()
    duals_single[(3, 1)] = 5.0
    scenarios.append({
        'name': 'Single high demand dual (day 3, shift 1)',
        'duals_i': 0.0,
        'duals_ts': duals_single
    })
    
    # Scenario 4: High duals on early days (tests start-of-horizon constraint)
    duals_early = duals_ts_zero.copy()
    duals_early[(1, 1)] = 10.0
    duals_early[(2, 1)] = 10.0
    scenarios.append({
        'name': 'High duals on days 1-2 (start-of-horizon test)',
        'duals_i': 0.0,
        'duals_ts': duals_early
    })
    
    # Scenario 5: Uniform duals
    duals_uniform = {(t, k): 2.0 for t in T for k in K}
    scenarios.append({
        'name': 'Uniform duals (2.0)',
        'duals_i': 1.0,
        'duals_ts': duals_uniform
    })
    
    # Scenario 6: Gradient duals (increasing over days)
    duals_gradient = {(t, k): float(t) for t in T for k in K}
    scenarios.append({
        'name': 'Gradient duals (day number)',
        'duals_i': 0.5,
        'duals_ts': duals_gradient
    })
    
    # Scenario 7: Random duals
    np.random.seed(42)
    duals_random = {(t, k): np.random.uniform(0, 5) for t in T for k in K}
    scenarios.append({
        'name': 'Random duals (seed=42)',
        'duals_i': 2.0,
        'duals_ts': duals_random
    })
    
    # Scenario 8: High duals mid-week
    duals_mid = duals_ts_zero.copy()
    for k in K:
        duals_mid[(4, k)] = 8.0
        duals_mid[(5, k)] = 8.0
    scenarios.append({
        'name': 'High duals mid-week (days 4-5)',
        'duals_i': 1.0,
        'duals_ts': duals_mid
    })
    
    # Scenario 9: Alternating shift preferences
    duals_alt = duals_ts_zero.copy()
    for t in T:
        duals_alt[(t, (t % 3) + 1)] = 5.0
    scenarios.append({
        'name': 'Alternating shift preferences',
        'duals_i': 0.5,
        'duals_ts': duals_alt
    })
    
    # Scenario 10: Edge case - negative lambda dual
    scenarios.append({
        'name': 'Negative lambda dual',
        'duals_i': -5.0,
        'duals_ts': duals_uniform
    })
    
    # Run tests
    print("=" * 60)
    print("DP vs MIP Comparison Test")
    print("=" * 60)
    
    passed = 0
    failed = 0
    failed_scenarios = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {scenario['name']}")
        
        result = compare_dp_vs_mip(
            duals_i=scenario['duals_i'],
            duals_ts=scenario['duals_ts'],
            T=T,
            K=K,
            eps=0.1,
            chi=3,
            verbose=True
        )
        
        if result['obj_match']:
            if result['schedule_match']:
                print(f"  ✓ PASS - Objectives match (same schedule)")
            else:
                print(f"  ✓ PASS - Objectives match (different schedule - multiple optima)")
            passed += 1
        else:
            if not result['obj_match']:
                diff = abs(result['dp_obj'] - result['mip_obj']) if result['dp_obj'] and result['mip_obj'] else float('inf')
                print(f"  ✗ FAIL - Objective mismatch: DP={result['dp_obj']:.6f}, MIP={result['mip_obj']:.6f}, diff={diff:.6f}")
            if not result['schedule_match']:
                print(f"  ✗ FAIL - Schedule mismatch")
                dp_only = set(result['dp_schedule'].keys()) - set(result['mip_schedule'].keys())
                mip_only = set(result['mip_schedule'].keys()) - set(result['dp_schedule'].keys())
                if dp_only:
                    print(f"    DP only: {sorted(dp_only)}")
                if mip_only:
                    print(f"    MIP only: {sorted(mip_only)}")
            failed += 1
            failed_scenarios.append(scenario['name'])
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{len(scenarios)} passed, {failed}/{len(scenarios)} failed")
    print("=" * 60)
    
    if failed_scenarios:
        print("\nFailed scenarios:")
        for name in failed_scenarios:
            print(f"  - {name}")
    
    return passed == len(scenarios)


def run_extended_random_tests(n_tests: int = 20):
    """Run extended random tests with various parameters."""
    
    print("\n" + "=" * 60)
    print("Extended Random Tests")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for seed in range(n_tests):
        np.random.seed(seed)
        
        # Random parameters
        n_days = np.random.randint(5, 10)
        T = list(range(1, n_days + 1))
        K = [1, 2, 3]
        
        eps = np.random.uniform(0.05, 0.15)
        chi = np.random.randint(2, 5)
        
        duals_i = np.random.uniform(-2, 5)
        duals_ts = {(t, k): np.random.uniform(0, 8) for t in T for k in K}
        
        result = compare_dp_vs_mip(
            duals_i=duals_i,
            duals_ts=duals_ts,
            T=T,
            K=K,
            eps=eps,
            chi=chi,
            verbose=False
        )
        
        if result['obj_match'] and result['schedule_match']:
            print(f"  [{seed+1:2d}/{n_tests}] ✓ PASS (days={n_days}, eps={eps:.2f}, chi={chi})")
            passed += 1
        else:
            print(f"  [{seed+1:2d}/{n_tests}] ✗ FAIL (days={n_days}, eps={eps:.2f}, chi={chi})")
            if not result['obj_match']:
                print(f"          Obj diff: DP={result['dp_obj']:.4f}, MIP={result['mip_obj']:.4f}")
            failed += 1
    
    print(f"\nExtended tests: {passed}/{n_tests} passed")
    return passed == n_tests


if __name__ == "__main__":
    # Run fixed scenarios
    scenario_pass = run_test_scenarios()
    
    # Run extended random tests
    random_pass = run_extended_random_tests(20)
    
    # Final verdict
    print("\n" + "=" * 60)
    if scenario_pass and random_pass:
        print("✓ ALL TESTS PASSED - DP and MIP are aligned!")
    else:
        print("✗ SOME TESTS FAILED - DP and MIP have discrepancies")
    print("=" * 60)
