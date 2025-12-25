"""
Debug: Compare CG with Gurobi SP vs CG with Labeling SP
Step by step comparison.
"""

from Utils.setup import Min_WD_i, Max_WD_i
from masterproblem import MasterProblem
from subproblem_factory import create_subproblem
from Utils.compactsolver import Problem
from Utils.demand import demand_dict_fifty
import pandas as pd
import numpy as np
import random

print("=" * 70)
print("Debug: CG with Gurobi SP vs Labeling SP - Step by Step")
print("=" * 70)

# Parameters
epsilon = 0.05
chi = 4
len_I = 10
max_itr = 5  # Just 5 iterations for debugging

T = list(range(1, 8))
I = list(range(1, len_I + 1))
K = [1, 2, 3]

data = pd.DataFrame({
    'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
    'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
    'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
})

random.seed(42)
demand_dict = demand_dict_fifty(len(T), 1.0, len_I, middle_shift=2)

# Get common starting solution
print("\n[1] Getting initial solution...")
problem_start = Problem(data, demand_dict, epsilon, Min_WD_i, Max_WD_i, chi)
problem_start.buildLinModel()
problem_start.model.Params.TimeLimit = 60
problem_start.model.Params.OutputFlag = 0
problem_start.model.update()
problem_start.model.optimize()
start_values_perf = {(t, s): problem_start.perf[1, t, s].x for t in T for s in K}
print("Initial solution obtained.")

# Create two identical master problems
print("\n[2] Creating Master Problems...")
master_mip = MasterProblem(data, demand_dict, max_itr, 0, 0, 98, start_values_perf)
master_mip.buildModel()
master_mip.setStartSolution()
master_mip.updateModel()

master_lab = MasterProblem(data, demand_dict, max_itr, 0, 0, 98, start_values_perf)
master_lab.buildModel()
master_lab.setStartSolution()
master_lab.updateModel()

print("\n" + "=" * 70)
print("Starting CG iterations...")
print("=" * 70)

for itr in range(1, max_itr + 1):
    print(f"\n--- Iteration {itr} ---")
    
    # Solve both masters
    master_mip.model.setParam('OutputFlag', 0)
    master_mip.solveRelaxModel()
    master_lab.model.setParam('OutputFlag', 0)
    master_lab.solveRelaxModel()
    
    # Get duals
    duals_i_mip = master_mip.getDuals_i()
    duals_ts_mip = master_mip.getDuals_ts()
    duals_i_lab = master_lab.getDuals_i()
    duals_ts_lab = master_lab.getDuals_ts()
    
    print(f"MIP Master Obj: {master_mip.model.objval:.6f}")
    print(f"LAB Master Obj: {master_lab.model.objval:.6f}")
    
    # Solve SPs
    sp_mip = create_subproblem('mip', duals_i_mip, duals_ts_mip, data, 1, itr, epsilon, Min_WD_i, Max_WD_i, chi)
    sp_mip.buildModel()
    sp_mip.model.setParam('OutputFlag', 0)
    sp_mip.solveModelOpt(60)
    
    sp_lab = create_subproblem('labeling', duals_i_lab, duals_ts_lab, data, 1, itr, epsilon, Min_WD_i, Max_WD_i, chi)
    sp_lab.buildModel()
    sp_lab.solveModelOpt(60)
    
    rc_mip = sp_mip.model.objval
    rc_lab = sp_lab.objval
    
    sched_mip = sp_mip.getNewSchedule()
    sched_lab = sp_lab.getNewSchedule()
    
    x_mip = sorted([(k[0], k[1]) for k, v in sched_mip.items() if v > 0.5])
    x_lab = sorted([(k[0], k[1]) for k, v in sched_lab.items() if v > 0.5])
    
    print(f"MIP SP RC: {rc_mip:.6f}, Schedule: {x_mip}")
    print(f"LAB SP RC: {rc_lab:.6f}, Schedule: {x_lab}")
    
    # Compare getOptP
    p_mip = sp_mip.getOptP()
    p_lab = sp_lab.getOptP()
    print(f"MIP getOptP: {p_mip}")
    print(f"LAB getOptP: {p_lab}")
    
    # Add columns if improving
    threshold = 6e-5
    if rc_mip < -threshold:
        master_mip.addColumn(itr, sched_mip)
        master_mip.addLambda(itr)
        master_mip.updateModel()
    
    if rc_lab < -threshold:
        master_lab.addColumn(itr, sched_lab)
        master_lab.addLambda(itr)
        master_lab.updateModel()

print("\n" + "=" * 70)
print("After 5 iterations:")
print("=" * 70)
master_mip.model.setParam('OutputFlag', 0)
master_mip.solveRelaxModel()
master_lab.model.setParam('OutputFlag', 0)
master_lab.solveRelaxModel()
print(f"MIP Master LP: {master_mip.model.objval:.6f}")
print(f"LAB Master LP: {master_lab.model.objval:.6f}")
