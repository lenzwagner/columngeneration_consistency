"""
Test loop comparing Numba Labeling vs Gurobi MIP solvers.
Uses shared start solution for fair comparison.
"""

from Utils.setup import Min_WD_i, Max_WD_i
from cg_behavior import column_generation_behavior
from Utils.demand import demand_dict_fifty
from Utils.compactsolver import Problem
import pandas as pd
import numpy as np
import time
import random

print("=" * 70)
print("Testing Gurobi SP vs Labeling SP with SHARED start solution")
print("=" * 70)

# Parameters
epsilon = 0.05
chi = 4
len_I = 6
prob = 1.0

# Data
T = list(range(1, 8))  # 7 days
I = list(range(1, len_I + 1))
K = [1, 2, 3]

data = pd.DataFrame({
    'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
    'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
    'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
})

random.seed(42)
demand_dict = demand_dict_fifty(len(T), prob, len_I, middle_shift=2)
eps = epsilon

time_cg_init = 2
max_itr = 300
output_len = 98
threshold = 6e-5
time_cg = 7200

print(f"\nInstance: I={len_I}, T={len(T)}, eps={eps}, chi={chi}")
print(f"Demand: {demand_dict}")

# Generate start solution ONCE
print("\n[0] Generating shared start solution...")
problem_start = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i, chi)
problem_start.buildLinModel()
problem_start.model.Params.MIPFocus = 1
problem_start.model.Params.Heuristics = 1
problem_start.model.Params.RINS = 10
problem_start.model.Params.TimeLimit = time_cg_init
problem_start.model.update()
problem_start.model.optimize()

# Extract start values
start_values = {
    'perf': {(t, s): problem_start.perf[1, t, s].x for t in T for s in K},
    'p': {t: problem_start.p[1, t].x for t in T},
    'x': {(t, s): problem_start.x[1, t, s].x for t in T for s in K},
    'c': {t: problem_start.sc[1, t].x for t in T},
    'r': {t: problem_start.r[1, t].x for t in T},
    'eup': {t: problem_start.e[1, t].x for t in T},
    'elow': {t: problem_start.b[1, t].x for t in T}
}
print("Start solution generated and shared.")
print("-" * 70)

# Test with Gurobi MIP SP
print("\n[1] Running CG with Gurobi SP...")
t0 = time.perf_counter()
result_mip = column_generation_behavior(
    data, demand_dict, eps, Min_WD_i, Max_WD_i,
    time_cg_init, max_itr, output_len, chi,
    threshold, time_cg, I, T, K, prob,
    sp_solver='mip',
    start_values=start_values,
    save_lp=True
)
mip_time = time.perf_counter() - t0
mip_obj = result_mip[8]
mip_itr = result_mip[10]
print(f"   MIP Result: obj={mip_obj:.4f}, iterations={mip_itr}, time={mip_time:.2f}s")

# Test with Labeling SP
print("\n[2] Running CG with Labeling SP...")
t0 = time.perf_counter()
result_lab = column_generation_behavior(
    data, demand_dict, eps, Min_WD_i, Max_WD_i,
    time_cg_init, max_itr, output_len, chi,
    threshold, time_cg, I, T, K, prob,
    sp_solver='labeling',
    start_values=start_values,
    save_lp=True
)
lab_time = time.perf_counter() - t0
lab_obj = result_lab[8]
lab_itr = result_lab[10]
print(f"   Labeling Result: obj={lab_obj:.4f}, iterations={lab_itr}, time={lab_time:.2f}s")

# Optional: Direct MIP comparison (without CG)
run_direct_mip = True  # Set to True to enable
direct_mip_obj = None
if run_direct_mip:
    print("\n[3] Running Direct MIP (without CG)...")
    from Utils.compactsolver import Problem
    t0 = time.perf_counter()
    problem_direct = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i, chi)
    problem_direct.buildLinModel()
    problem_direct.model.Params.TimeLimit = 3600
    problem_direct.model.Params.OutputFlag = 1
    problem_direct.model.update()
    problem_direct.model.optimize()
    direct_mip_time = time.perf_counter() - t0
    direct_mip_obj = problem_direct.model.objval
    print(f"   Direct MIP Result: obj={direct_mip_obj:.4f}, time={direct_mip_time:.2f}s")

# Compare
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
match = abs(mip_obj - lab_obj) < 1e-3
print(f"  CG+MIP SP:     obj={mip_obj:.4f}, iterations={mip_itr}, time={mip_time:.2f}s")
print(f"  CG+Labeling:   obj={lab_obj:.4f}, iterations={lab_itr}, time={lab_time:.2f}s")
if direct_mip_obj is not None:
    print(f"  Direct MIP:    obj={direct_mip_obj:.4f}")
print(f"  Match: {'✓' if match else '✗'}")
print("=" * 70)
